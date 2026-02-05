> [!TIP]
> 
> The blog is written by human. Illustrations are drawn with Claude-4.5-Opus - the model is asked to generate txt figure according to the illustration drawn by human.

# Overview

Kimi-K2-Thinking is the first open source model showing that 4-bit quantization works well for post-training on very large MoE model. According to Kimi's [blog](https://moonshotai.github.io/Kimi-K2/thinking.html) and LMSys's [blog](https://lmsys.org/blog/2026-01-26-int4-qat/), INT4 QAT can match the BF16 baseline in performance and improve the RL rollout throughput significantly. The [QeRL paper](https://arxiv.org/pdf/2510.11696) also verifies this on other 4-bit dtype. For now, this is the SOTA efficient RL technique verified on very large model.

In this blog, we go throught the implementation of INT4 QAT in [Slime](https://github.com/THUDM/slime) which uses [Megatron](https://github.com/NVIDIA/Megatron-LM) as the training engine and [SGLang](https://github.com/sgl-project/sglang) as the rollout engine. In this blog, we will look into the fake quantization logic in Megatron and Marlin Kernel for efficient low-bit matrix multiplication in SGLang. We will also see how Slime transfers the QAT training weights into INT4 inference weights.

In one sentence, we will go through the whole pipeline and kernel-level optimization details of INT4 QAT.


# 1. Megatron INT4 Fake Quantization

What is QAT (quantization aware training)? In one sentence, QAT tries to simulate the low-bit inference behavior of the model in forward pass during training while keeping the weights in high-bit (16-bit) so weight update is in high precision.

High-precision weight update ensures the model to be trained with precise gradient signal. Fake quantization ensures low training-inference mismatch. Furthermore, after training, the model can achieve good performance since it has been fully adapted to low-bit inference during training.

Set `OPEN_TRAINING_INT4_FAKE_QAT_FLAG` to "1" in Slime to open the INT4 QAT feature.

When `OPEN_TRAINING_INT4_FAKE_QAT_FLAG` is set to "1", the entry point is the `_get_weight_tensors()` method in `TEGroupedLinear` class (in megatron/core/extensions/transformer_engine.py).

`TEGroupedLinear._get_weight_tensors()` to get the weight tensors of all experts --> `fake_int4_quantization_ste()` --> `_FakeInt4QuantizationSTE.apply()` to actually apply fake INT4 quantization to the weight tensors.

Note that INT4 QAT is only applied on MoE weights. That is, `up_proj`, `down_proj` and `gate_proj`.


<details>
<summary>_FakeInt4QuantizationSTE Code</summary>

```python
class _FakeInt4QuantizationSTE(torch.autograd.Function):
    """
    Straight-Through Estimator for INT4 quantization.

    Forward: Simulates INT4 quantization (quantize → dequantize)
    Backward: Passes gradients unchanged (identity function)
    """

    @staticmethod
    def forward(ctx, x, group_size):
        # m = out_features (output dimension of the linear layer)
        # n = in_features (input dimension of the linear layer) 
        m, n = x.shape
        block_size_m, block_size_n = 1, group_size

        # Step 1: Pad to align with group boundaries
        m_padded = ceil_div(m, block_size_m) * block_size_m
        n_padded = ceil_div(n, block_size_n) * block_size_n

        x_padded = torch.zeros(
            (m_padded, n_padded),
            dtype=x.dtype, device=x.device
        )
        x_padded[:m, :n] = x

        # Step 2: Reshape for group-wise operations
        # Shape: [m, n/group_size, group_size]
        x_view = x_padded.view(
            m_padded // block_size_m,
            block_size_m,
            n_padded // block_size_n,
            block_size_n
        )

        # Step 3: Compute per-group scale (symmetric quantization)
        # Find max absolute value in each group
        x_max = x_view.abs().float().amax(dim=(1, 3), keepdim=True)
        q_max = 7  # INT4 symmetric range: [-7, 7]
        x_scale = x_max / q_max
        x_scale = x_scale.clamp(min=1e-5)  # Avoid division by zero

        # Step 4: Quantize (round to nearest integer)
        x_div = x_view / x_scale
        x_round = torch.round(x_div)
        x_q_clamped = x_round.clamp(-q_max, q_max)

        # Step 5: Dequantize (restore to original scale)
        x_dequant_view = x_q_clamped * x_scale

        # Step 6: Remove padding and return
        x_dequant_full = x_dequant_view.view_as(x_padded)
        x_out = x_dequant_full[:m, :n].contiguous().to(x.dtype)

        return x_out

    @staticmethod
    def backward(ctx, grad_output):
        # STE: Gradients pass through unchanged
        return grad_output, None
```

</details>

This is per-channel, group-wise symmetric INT4 quantization on weights. `m` and `n` are out_features and in_features, respectively. The grouping logic of the whole weight tensor is:
```
        ←—————— n (in_features) ——————→

       +———————————+———————————+———————+
     ↑ | group 0   | group 1   |  ...  | ← row 0: each group has its own scale
     | +———————————+———————————+———————+
   m | | group 0   | group 1   |  ...  | ← row 1: each group has its own scale
 (out) |+——————————+———————————+———————+
     | |    ...    |    ...    |  ...  |
     ↓ +———————————+———————————+———————+

       |←group_size→|
```

What is the rationale of quantizing weights within groups rather than the whole matrix? See the picture below. Outliers increase the quantization error by enlarging the scale (so a larger range of different float value will be rounded to the same integer).Therefore, when there are outliers, we don't want them to have a global impact. Grouping makes sure that outliers only have local impact.

The group size indicates trade-off. Smaller group size → (1) smaller quantization error but (2) higher computation overhead for quantization and dequantization, and (3) memory overhead to store more scales.

![grouping rationale: avoid outlier](fake-int4-qat-tutorial/quantization_illustration.png).



# 2. Slime: synchronize weights between Megatron (training engine) and SGLang (rollout engine)

In RL post-training, the training and rollout engine are typically not on the same GPU. Therefore, after the training engine has updated the weights, we need to synchronize the updated weights to the rollout engine. Slime does this for us. The illustration of how slime does this:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        FULL WEIGHT SYNC PIPELINE                                │
│                                                                                 │
│  MEGATRON (Trainer)                             SGLANG (Rollout)                │
│  ┌─────────────────┐                           ┌─────────────────┐              │
│  │ FP16/BF16       │                           │ INT4/FP8        │              │
│  │ Megatron Format │                           │ HF Format       │              │
│  │ TP-sharded      │                           │ Full weights    │              │
│  └────────┬────────┘                           └────────▲────────┘              │
│           │                                             │                       │
│           ▼                                             │                       │
│  ┌─────────────────┐                                    │                       │
│  │ all_gather(TP)  │  Gather shards                     │                       │
│  └────────┬────────┘                                    │                       │
│           │                                             │                       │
│           ▼                                             │                       │
│  ┌─────────────────┐                                    │                       │
│  │ convert_to_hf() │                                    │                       │
│  │                 │                                    │                       │
│  │  1. remove_pad  │  Remove vocab padding              │                       │
│  │  2. name_convert│  Megatron→HF names                 │                       │
│  │  3. reshape     │  Split QKV, GLU, etc.              │                       │
│  │  4. quantize    │  FP16→INT4 (GPTQ format)           │                       │
│  │                 │                                    │                       │
│  └────────┬────────┘                                    │                       │
│           │                                             │                       │
│           ▼                                             │                       │
│  ┌─────────────────┐      NCCL Broadcast       ┌────────┴────────┐              │
│  │ HF-format       │  ════════════════════════►│ Receive weights │              │
│  │ GPTQ INT4       │                           │ (GPTQ format)   │              │
│  └─────────────────┘                           └────────┬────────┘              │
│                                                         │                       │
│                                                         ▼                       │
│                                                ┌─────────────────┐              │
│                                                │post_process     │              │
│                                                │  GPTQ → Marlin  │              │
│                                                │  (for inference)│              │
│                                                └─────────────────┘              │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

The entry point of weight sync locates in train.py. When `update_weights()` is called, the rollout engine will get a copy of weights from the training engine. Here is what things will look like:

```python
# 1. Before training loop
# Initial sync of loaded weights to ensure actor is ready
actor_model.update_weights()

# 2. Training loop
for rollout_id in range(num_rollouts):
    # Generate experience data from the environment
    rollout_data = rollout_manager.generate()
    
    # Perform the training step (asynchronous update)
    actor_model.async_train(rollout_id, rollout_data)
    
    # ... other logic (logging, validation, etc.) ...
    
    # Update weights after each training step to keep the actor in sync
    actor_model.update_weights()
```

## 2.1 UpdateWeightFromDistributed

Under the hood, `UpdateWeightFromDistributed.update_weights()` manages the end-to-end weight sync process. Let's trace through its `update_weights()` method:

<details>
<summary> UpdateWeightFromDistributed.update_weights() (slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py)</summary>

```python
@torch.no_grad()
def update_weights(self) -> None:
    """
    Main weight sync entry point.

    Flow: Pause → Pre-process → Sync Non-Expert → Sync Expert → Post-process → Resume
    """
    self.weight_version += 1

    # ========== Step 1: Pause SGLang and prepare for weight update ==========
    if dist.get_rank() == 0:
        ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
        ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])

        # INT4 pre-process: Convert Marlin format back to GPTQ for loading
        if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
            post_process_weights(
                restore_weights_before_load=True,    # Marlin → GPTQ
                post_process_quantization=False,
                rollout_engines=self.rollout_engines,
            )
    dist.barrier(group=get_gloo_group())

    # ========== Step 2: Sync non-expert parameters ==========
    buffer_size = 0
    converted_named_tensors = []
    pbar = tqdm(desc=f"[{self._group_name}] Update weights", total=0) if self._is_pp_src_rank else None

    for name, param in named_params_and_buffers(self.args, self.model):
        if ".experts." in name:
            continue  # Skip expert params for now
        buffer_size = self._update_weight_from_distributed(
            name, param, converted_named_tensors, buffer_size, pbar=pbar
        )

    if converted_named_tensors:
        self._update_bucket_weights_from_distributed(converted_named_tensors, pbar=pbar)

    dist.barrier(group=get_gloo_group())

    # ========== Step 3: Sync expert parameters (MoE) ==========
    buffer_size = 0
    named_tensors = []
    for name, param in named_params_and_buffers(self.args, self.model):
        if ".experts." not in name:
            continue  # Only expert params
        buffer_size = self._update_expert_weight_from_distributed(
            name, param, named_tensors, buffer_size, pbar=pbar
        )

    if named_tensors:
        self._update_expert_bucket_weights_from_distributed(named_tensors, pbar=pbar)

    dist.barrier(group=get_gloo_group())

    # ========== Step 4: Post-process and resume ==========
    if dist.get_rank() == 0:
        # INT4 post-process: Convert GPTQ to Marlin for fast inference
        if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
            post_process_weights(
                restore_weights_before_load=False,
                post_process_quantization=True,    # GPTQ → Marlin
                rollout_engines=self.rollout_engines,
            )
        ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
    dist.barrier(group=get_gloo_group())
```
</details>


### 2.1.1 Special Topic: GPTQ Format v.s Marlin Format and Their Rationale

> [!TIP]
> Marlin Format: An Efficient GPU Runtime Format Enabling Fast Memory Access

<details>
<summary>Unfold to see what is marline format and the rationale behind marlin format</summary>

LLM inference (especially the decode phase) is **memory-bandwidth bound** — the GPU spends most of its time waiting for weights to arrive from HBM (High Bandwidth Memory), not doing actual computation. Marlin format is designed to maximize memory access efficiency.

**Tiled Matrix Multiplication: How GPUs Compute C = A × B**

To understand why Marlin format matters, we first need to understand how GPUs perform matrix multiplication using tiles:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│   ═══════════════════════════════════════════════════════════════════════════   │
│   GPU TILING STRATEGY (16×16 tiles, accumulate along K):                        │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                 │
│   To compute ONE 16×16 output tile C[i:i+16, j:j+16]:                           │
│                                                                                 │
│        A [M × K]                    B [K × N]                                   │
│       ←────────── K ──────────→    ←────────── N ──────────→                    │
│      ┌──┬──┬──┬──┬──┬──┬──┬──┐    ┌──┬──┬──┬──┬──┬──┬──┬──┐                     │
│    ↑ │▓▓│░░│  │  │  │  │  │  │  ↑ │▓▓│  │  │  │  │  │  │  │                     │
│i → 16│▓▓│░░│  │  │  │  │  │  │  │ │▓▓│  │  │  │  │  │  │  │                     │
│    ↓ │▓▓│░░│  │  │  │  │  │  │  │ │▓▓│  │  │  │  │  │  │  │                     │
│      ├──┼──┼──┼──┼──┼──┼──┼──┤  K │░░│  │  │  │  │  │  │  │                     │
│      │  │  │  │  │  │  │  │  │  │ │░░│  │  │  │  │  │  │  │                     │
│      │  │  │  │  │  │  │  │  │  │ │░░│  │  │  │  │  │  │  │                     │
│      │  │  │  │  │  │  │  │  │  │ │  │  │  │  │  │  │  │  │                     │
│      │  │  │  │  │  │  │  │  │  ↓ │  │  │  │  │  │  │  │  │                     │
│      └──┴──┴──┴──┴──┴──┴──┴──┘    └──┴──┴──┴──┴──┴──┴──┴──┘                     │
│       ↑16↑16                       ↑16                                          │
│       k=0 k=1 ...                  j                                            │
│                                                                                 │
│   Step 1: Load A[i:i+16, 0:16] (▓▓) and B[0:16, j:j+16] (▓▓)                    │
│           C_tile += A_tile × B_tile                                             │
│                                                                                 │
│   Step 2: Load A[i:i+16, 16:32] (░░) and B[16:32, j:j+16] (░░)                  │
│           C_tile += A_tile × B_tile                                             │
│                                                                                 │
│   ... repeat for all K/16 tiles along K dimension ...                           │
│                                                                                 │
│   Final: C[i:i+16, j:j+16] = Σ(k) A_tile[k] × B_tile[k]                         │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**The Core Problem: GPU Memory Transactions**

When the GPU reads from HBM, it reads in **128-byte cache lines** — this is the minimum read unit. For INT4 weights, one cache line contains 256 weight values (128 bytes × 2 values/byte).

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    Memory Efficiency: Naive vs Marlin                           │
│                                                                                 │
│   To load ONE 16×16 tile from B for the tiled matmul above:                     │
│                                                                                 │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   NAIVE ROW-MAJOR LAYOUT (how weights are typically stored):                    │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                 │
│   B matrix [K × N] stored row by row in memory:                                 │
│                                                                                 │
│   Memory: [Row0: b0,0 b0,1 ... b0,N-1][Row1: b1,0 b1,1 ... b1,N-1][...]         │
│                                                                                 │
│   To load tile B[0:16, j:j+16], we need elements from 16 different rows:        │
│                                                                                 │
│   Row 0:  [...|b0,j  b0,j+1 ... b0,j+15|...]  ← Need 16 elements                │
│   Row 1:  [...|b1,j  b1,j+1 ... b1,j+15|...]  ← Need 16 elements                │
│   ...                                                                           │
│   Row 15: [...|b15,j b15,j+1...b15,j+15|...]  ← Need 16 elements                │
│                                                                                 │
│   Each cache line (128 bytes) loads 256 INT4 values, but we only need 16:       │
│                                                                                 │
│   Cache line for Row 0: [████░░░░░░░░░░░░░░░░░░░░░░░░░░░░]                      │
│                          ▲▲▲▲                                                   │
│                          └─── 16 values needed (6.25%)                          │
│                               240 values wasted (93.75%)                        │
│                                                                                 │
│   Total: 16 cache line loads × 6.25% utilization = massive waste!               │
│                                                                                 │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   MARLIN TILED LAYOUT (weights reorganized by 16×16 tiles):                     │ 
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                 │
│   B matrix stored tile by tile in memory:                                       │
│                                                                                 │
│   Memory: [Tile(0,0)][Tile(0,1)]...[Tile(0,N/16-1)][Tile(1,0)]...               │
│                                                                                 │
│   Each tile contains 16×16 = 256 INT4 values = 128 bytes = ONE cache line!      │
│                                                                                 │
│   Tile(k,j) in memory:                                                          │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │ b_k*16,j*16   b_k*16,j*16+1   ... b_k*16,j*16+15                        │   │
│   │ b_k*16+1,j*16 b_k*16+1,j*16+1 ... b_k*16+1,j*16+15                      │   │
│   │ ...                                                                     │   │
│   │ b_k*16+15,j*16 ...                b_k*16+15,j*16+15                     │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                     = 256 INT4 values = 128 bytes                               │
│                                                                                 │
│   To load tile B[k*16:(k+1)*16, j*16:(j+1)*16]:                                 │
│                                                                                 │
│   Cache line: [████████████████████████████████] (256/256 used = 100%)          │
│                                                                                 │
│   ONE cache line load gets the ENTIRE tile we need!                             │
│                                                                                 │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   COMPARISON:                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                 │
│   For K/16 tiles along K dimension:                                             │
│                                                                                 │
│   Row-Major:  K cache line loads, 6.25% utilization each                        │
│   Marlin:     K/16 cache line loads, 100% utilization each                      │
│                                                                                 │
│   SPEEDUP: 16x fewer memory transactions for the same computation!              │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Key Insight**: First, note that the so called memory layout means how matrix is stored (order of elements) in HBM. By storing weights in the exact order that Tensor Cores consume them (16×16 tiles), every byte loaded from memory is immediately useful. This is why Marlin format is essential for fast INT4 inference — same weights, but reorganized for optimal GPU memory access patterns.

</details>

> [!TIP]
> GPTQ Format: Packing 8 INT4 Values into 1 INT32

<details>
<summary>Unfold to see the what is GPTQ format and the rationale behind GPTQ format</summary>

Since INT4 (4-bit) is not a native data type that CPUs/GPUs can address directly, GPTQ packs 8 consecutive INT4 values into a single INT32 (8 × 4 bits = 32 bits). This is the standard storage format for INT4 quantized weights.

**Shape Transformation**: `[M, N]` (INT4 logical) → `[M, N//8]` (INT32 packed)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    INT4 → GPTQ Format Packing                                   │
│                                                                                 │
│   Original INT4 Matrix: [M=4, N=16] (each cell is 4 bits)                       │
│                                                                                 │
│          ←───────────────────── N = 16 ─────────────────────→                   │
│          col0 col1 col2 col3 col4 col5 col6 col7 │ col8 col9 ... col15          │
│         ┌────┬────┬────┬────┬────┬────┬────┬────┼────┬────┬────┬────┐           │
│   row0  │ 3  │ 7  │ 2  │ 15 │ 1  │ 8  │ 4  │ 11 │ 5  │ 9  │ 0  │ 12 │ ...       │
│         ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤           │
│   row1  │ 6  │ 14 │ 5  │ 3  │ 9  │ 2  │ 7  │ 10 │ 1  │ 4  │ 8  │ 13 │ ...       │
│         ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤           │
│   row2  │ 0  │ 11 │ 8  │ 6  │ 12 │ 5  │ 3  │ 9  │ 7  │ 2  │ 14 │ 1  │ ...       │
│         ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤           │
│   row3  │ 4  │ 1  │ 13 │ 7  │ 2  │ 10 │ 6  │ 15 │ 3  │ 8  │ 11 │ 5  │ ...       │
│         └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘           │
│          └─────────────── 8 values ─────────────┘                               │
│                        pack into 1 INT32                                        │
│                                                                                 │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   PACKING FORMULA (8 INT4 → 1 INT32):                                           │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                 │
│   For row0, columns 0-7: [3, 7, 2, 15, 1, 8, 4, 11]                             │
│                                                                                 │
│   32-bit integer layout:                                                        │
│   ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐                             │
│   │ w7  │ w6  │ w5  │ w4  │ w3  │ w2  │ w1  │ w0  │                             │
│   │ =11 │ =4  │ =8  │ =1  │ =15 │ =2  │ =7  │ =3  │                             │
│   ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤                             │
│   │bits │bits │bits │bits │bits │bits │bits │bits │                             │
│   │28-31│24-27│20-23│16-19│12-15│ 8-11│ 4-7 │ 0-3 │                             │
│   └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘                             │
│                                                                                 │
│   packed_int32 = (11 << 28) | (4 << 24) | (8 << 20) | (1 << 16) |               │
│                  (15 << 12) | (2 << 8)  | (7 << 4)  | 3                         │
│                                                                                 │
│   Binary:  1011 0100 1000 0001 1111 0010 0111 0011                              │
│            ──── ──── ──── ──── ──── ──── ──── ────                              │
│             11   4    8    1   15    2    7    3                                │
│                                                                                 │
│   Hex:     0xB481F273                                                           │
│                                                                                 │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   GPTQ PACKED MATRIX: [M=4, N//8=2] (each cell is INT32)                        │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                 │
│          ←── N//8 = 2 ──→                                                       │
│          col0        col1                                                       │
│         ┌───────────┬───────────┐                                               │
│   row0  │0xB481F273 │ 0x...     │  ← packs [3,7,2,15,1,8,4,11], [5,9,0,12,...]  │
│         ├───────────┼───────────┤                                               │
│   row1  │0xA7293E56 │ 0x...     │  ← packs [6,14,5,3,9,2,7,10], [1,4,8,13,...]  │
│         ├───────────┼───────────┤                                               │
│   row2  │0x9356C8B0 │ 0x...     │  ← packs [0,11,8,6,12,5,3,9], [7,2,14,1,...]  │
│         ├───────────┼───────────┤                                               │
│   row3  │0xF6A27D14 │ 0x...     │  ← packs [4,1,13,7,2,10,6,15],[3,8,11,5,...]  │
│         └───────────┴───────────┘                                               │
└─────────────────────────────────────────────────────────────────────────────────┘
```

</details>





### 2.1.2 Step 1: Pause SGLang and prepare for weight update

Here, expert weights (gate_proj, up_proj and down_proj) of the rollout engine is in marlin format for efficient inference. The tensor shape of marlin format is different from the GPTQ format that slime produces.

Therefore, we need to reshape the expert weight tensors on sglang so that the placeholder on sglang is of the correct shape.

In `update_weights()`, `restore_weights_before_load` is set to `True` and `post_process_quantization` is set to `False`. This means we want to "restore" the weights of the rollout engine to their original shape.

<details>
<summary>Code called under the hood for weights reshape on rollout engine</summary>

```python
def restore_weights_before_loading(self, layer: torch.nn.Module):
    """Forcibly resize parameters back to their original shapes (e.g., GPTQ format) before loading weights."""
    if not hasattr(layer, "_original_shapes"):
        return

    for name, orig_shape in layer._original_shapes.items():
        param = getattr(layer, name, None)

        if param is not None and param.shape != orig_shape:
            param.resize_(orig_shape)

    layer.is_marlin_converted = False
```

</details>

### 2.1.3 Step 2 and 3: Sync non-expert and expert parameters

> [!TIP]
> Expert and non-expert weights are sync'ed in different ways. Here we'll introduce why.

The method separates non-expert and expert parameters because:
- **Non-expert params**: Use Tensor Parallelism (TP) - need `all_gather` across TP group
- **Expert params**: Use Expert Parallelism (EP) - first need `all_gather` across TP group then need `all_gather` across EP group

<details>
<summary>Unfold to see a concrete example of the difference between non-expert param sync and expert param sync</summary>

#### Example Configuration

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         Example Configuration                                   │
│                                                                                 │
│   Model: Qwen3-MoE (or similar)                                                 │
│   - Hidden size: 4096                                                           │
│   - Intermediate size: 2048 (per expert)                                        │
│   - Number of experts: 64                                                       │
│                                                                                 │
│   Parallelism Config (8 GPUs):                                                  │
│   - TP (Tensor Parallel) = 2                                                    │
│   - EP (Expert Parallel) = 4                                                    │
│   - PP (Pipeline Parallel) = 1                                                  │
│                                                                                 │
│   GPU Layout:                                                                   │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │  GPU 0   │  GPU 1   │  GPU 2   │  GPU 3   │  GPU 4   │  GPU 5   │ ...   │   │
│   │  TP=0    │  TP=1    │  TP=0    │  TP=1    │  TP=0    │  TP=1    │       │   │
│   │  EP=0    │  EP=0    │  EP=1    │  EP=1    │  EP=2    │  EP=2    │       │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│   TP groups: {GPU0, GPU1}, {GPU2, GPU3}, {GPU4, GPU5}, {GPU6, GPU7}             │
│   EP groups: {GPU0, GPU2, GPU4, GPU6}, {GPU1, GPU3, GPU5, GPU7}                 │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### Case 1: Non-Expert Weight Sync (e.g., `attention.q_proj.weight`)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│              Non-Expert Sync: attention.q_proj.weight [4096, 4096]              │
│                                                                                 │
│   Original full weight: [4096, 4096]                                            │
│   TP sharding: split along output dimension (dim=0)                             │
│                                                                                 │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   STEP 1: Each GPU holds its TP shard                                           │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                 │
│   GPU 0 (TP=0):  q_proj.weight [2048, 4096]  ← rows 0~2047                      │
│   GPU 1 (TP=1):  q_proj.weight [2048, 4096]  ← rows 2048~4095                   │
│                                                                                 │
│   (GPU 2-7 have identical shards as GPU 0-1, duplicated across EP)              │
│                                                                                 │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   STEP 2: all_gather over TP group (ALL ranks receive full data)                │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                 │
│   all_gather({GPU0, GPU1}) →                                                    │
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │  Both GPU 0 and GPU 1 receive:                                          │   │
│   │  ┌────────────────────┐                                                 │   │
│   │  │ [2048, 4096] TP=0  │  from GPU 0                                     │   │
│   │  ├────────────────────┤                                                 │   │
│   │  │ [2048, 4096] TP=1  │  from GPU 1                                     │   │
│   │  └────────────────────┘                                                 │   │
│   │  concat(dim=0) → [4096, 4096]  ✓ Full weight on BOTH GPUs               │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   STEP 3: Only PP source rank uses the data                                     │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                 │
│   GPU 0 (PP source): convert_to_hf() → broadcast to SGLang  ✓ USED              │
│   GPU 1-7: return early, discard gathered data              ✗ DISCARDED         │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### Case 2: Expert Weight Sync (e.g., `experts.*.linear_fc1.weight`)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│           Expert Sync: 64 experts, each expert's linear_fc1 [4096, 4096]        │
│                                                                                 │
│   Total experts: 64                                                             │
│   EP = 4 → each EP rank holds 64/4 = 16 experts                                 │
│   TP = 2 → each expert's weight is TP-sharded across 2 GPUs                     │
│                                                                                 │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   INITIAL STATE: Expert distribution across GPUs                                │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                 │
│   ┌─────────────┬─────────────┬─────────────┬─────────────┬─────────────────┐   │
│   │   GPU 0     │   GPU 1     │   GPU 2     │   GPU 3     │    ...          │   │
│   │   TP=0      │   TP=1      │   TP=0      │   TP=1      │                 │   │
│   │   EP=0      │   EP=0      │   EP=1      │   EP=1      │                 │   │
│   ├─────────────┼─────────────┼─────────────┼─────────────┼─────────────────┤   │
│   │ Expert 0-15 │ Expert 0-15 │ Expert 16-31│ Expert 16-31│                 │   │
│   │ [2048,4096] │ [2048,4096] │ [2048,4096] │ [2048,4096] │                 │   │
│   │ (TP shard)  │ (TP shard)  │ (TP shard)  │ (TP shard)  │                 │   │
│   └─────────────┴─────────────┴─────────────┴─────────────┴─────────────────┘   │
│                                                                                 │
│   GPU 0 holds: experts[0:16].linear_fc1.weight[:2048, :]  (first half rows)     │
│   GPU 1 holds: experts[0:16].linear_fc1.weight[2048:, :]  (second half rows)    │
│   GPU 2 holds: experts[16:32].linear_fc1.weight[:2048, :] (first half rows)     │
│   ...                                                                           │
│                                                                                 │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   STEP 1: all_gather over TP group (reconstruct each expert's full weight)      │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                 │
│   all_gather({GPU0, GPU1}) for experts 0-15:                                    │
│                                                                                 │
│   GPU 0 & GPU 1 both have: experts[0:16] with full [4096, 4096] each  ✓         │
│   GPU 2 & GPU 3 both have: experts[16:32] with full [4096, 4096] each ✓         │
│   ...                                                                           │
│                                                                                 │
│   But each EP group only has 16 experts, not all 64!                            │
│                                                                                 │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   STEP 2: all_gather over EP group (collect all experts from all EP ranks)      │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                 │
│   all_gather({GPU0, GPU2, GPU4, GPU6}) across EP group:                         │
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │  After EP all_gather, ALL 4 GPUs (0,2,4,6) receive:                     │   │
│   │                                                                         │   │
│   │  From GPU 0 (EP=0): experts[0:16]   → 16 experts × [4096, 4096]         │   │
│   │  From GPU 2 (EP=1): experts[16:32]  → 16 experts × [4096, 4096]         │   │
│   │  From GPU 4 (EP=2): experts[32:48]  → 16 experts × [4096, 4096]         │   │
│   │  From GPU 6 (EP=3): experts[48:64]  → 16 experts × [4096, 4096]         │   │
│   │                                                                         │   │
│   │  Total: 64 experts × [4096, 4096] on ALL GPUs in EP group               │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   STEP 3: Only PP source rank uses the data                                     │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                 │
│   GPU 0 (PP source): convert_to_hf() → broadcast to SGLang  ✓ USED              │
│   GPU 1-7: return early, discard gathered data              ✗ DISCARDED         │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### Key Difference Summary

| Aspect | Non-Expert | Expert (MoE) |
|--------|-----------|--------------|
| **Parallelism** | TP only | TP + EP |
| **Initial state** | 1/TP of weight | 1/TP of 1/EP experts |
| **Gather steps** | 1 (TP all_gather) | 2 (TP all_gather → EP all_gather) |

</details>


### 2.1.4 Step 4: Post-process and resume

In this step, `post_process_weights()` is called with `post_process_quantization=True`, which triggers the GPTQ→Marlin format conversion.

> TODO: explain the repack kernel here.

# 3. SGLang: marlin kernel for efficient low-bit matmul

> TODO: explain the marlin kernel here.

# 4. Future Direction

```
    Weights
       ▲
       │
       │
       │
       │
       │
 MoE  ─│      ●
weights│   (current)
       │
       │
       └──────┬───────────────────────────→ Quantization Paradigm
            W4A16                         

       ● = Current: INT4 QAT on MoE weights (W4A16)
```

More datapoints on this graph may well be explored.
