> [!TIP]
> From zhihu (Kimi's infra engineer), to support INT4 QAT the infra team need to implement:
> Step 1. QAT logic in the training engine.
> Step 2. INT4 inference pipeline (out there, marlin kernel).
> Step 3. Transfer the QAT training weights into INT4 inference weights.
> Step 4. RL rollout (internel codebase).

# 1. Megatron INT4 Fake Quantization

When `OPEN_TRAINING_INT4_FAKE_QAT_FLAG` is set to "1", the entry point is the `_get_weight_tensors()` method in `TEGroupedLinear` class (in megatron/core/extensions/transformer_engine.py).

`TEGroupedLinear._get_weight_tensors()` to get the weight tensors of all experts --> `fake_int4_quantization_ste()` --> `_FakeInt4QuantizationSTE.apply()` to actually apply fake INT4 quantization to the weight tensors. 


<details>
<summary>`_FakeInt4QuantizationSTE` Code</summary>

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

# 2. Slime: synchronize weights between Megatron (training engine) and Sglang (rollout engine)

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

In RL post-training, the training and rollout engine are typically not on the same GPU. Therefore, after the training engine has updated the weights, we need to synchronize the updated weights to the rollout engine. Slime does this for us.

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

Here, expert weights (gated_proj, up_proj and down_proj) of the rollout engine is in marlin format for efficient inference. The tensor shape of marlin format is different from the GPTQ format that slime produces.

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

<details>
<summary>Unfold to see a deep-dive into the GPTQ→Marlin Repack Kernel</summary>

#### Overview: Three Key Insights of Marlin Format

The Marlin kernel achieves fast inference by optimizing for three things:

1. **Each warp thread gets weights from a single cache line** - No scattered memory accesses
2. **INT4→FP16 unpacking with minimal instructions** - Special interleave pattern enables fast conversion
3. **Data flows directly into Tensor Core registers without shuffle** - Weights are pre-arranged for mma.m16n8k16 instruction

Let's see how the `gptq_marlin_repack_kernel` implements each of these.

#### Kernel Constants and Tile Dimensions

```cpp
// From marlin.cuh
constexpr int tile_size = 16;        // Tensor Core tile dimension (16×16)
constexpr int tile_k_size = 16;      // Rows per tile (along K dimension)
constexpr int tile_n_size = 64;      // Columns per tile (along N dimension)
constexpr int repack_stages = 8;     // Pipeline stages for latency hiding
constexpr int repack_threads = 256;  // Threads per block

// INT4: 8 values per INT32
constexpr int pack_factor = 32 / 4 = 8;
```

**Why tile_n_size = 64?**
- A warp has 32 threads
- Each thread handles 2 elements along N (for 2 Tensor Core fragments)
- 4 warps × 32 threads × 2 elements = 256, but we process 16 columns per warp
- 4 warps × 16 columns = 64 columns per tile

#### Thread-to-Data Mapping: How Each Thread Knows What to Read

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     THREAD COORDINATE CALCULATION                                │
│                                                                                  │
│   For 256 threads (4 warps × 32 threads):                                        │
│                                                                                  │
│   warp_id = threadIdx.x / 32;    // 0, 1, 2, or 3                                │
│   th_id   = threadIdx.x % 32;    // 0~31 within each warp                        │
│                                                                                  │
│   Within each warp (32 threads):                                                 │
│   tc_col = th_id / 4;            // 0~7: which of 8 column groups                │
│   tc_row = (th_id % 4) * 2;      // 0, 2, 4, or 6: starting row                  │
│                                                                                  │
│   Example for warp 0:                                                            │
│   ┌────────┬────────┬─────────┬────────────────────────────────────────────────┐ │
│   │ th_id  │ tc_col │ tc_row  │ Handles rows (via tc_offsets)                  │ │
│   ├────────┼────────┼─────────┼────────────────────────────────────────────────┤ │
│   │   0    │   0    │   0     │ 0, 1, 8, 9                                     │ │
│   │   1    │   0    │   2     │ 2, 3, 10, 11                                   │ │
│   │   2    │   0    │   4     │ 4, 5, 12, 13                                   │ │
│   │   3    │   0    │   6     │ 6, 7, 14, 15                                   │ │
│   │   4    │   1    │   0     │ 0, 1, 8, 9                                     │ │
│   │   ...  │  ...   │  ...    │ ...                                            │ │
│   │  31    │   7    │   6     │ 6, 7, 14, 15                                   │ │
│   └────────┴────────┴─────────┴────────────────────────────────────────────────┘ │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### The tc_offsets Pattern: Matching Tensor Core mma.m16n8k16 Layout

```cpp
constexpr int tc_offsets[4] = {0, 1, 8, 9};
```

**Why {0, 1, 8, 9}?**

This pattern comes from how NVIDIA's Tensor Core `mma.m16n8k16` instruction expects its input:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│              TENSOR CORE mma.m16n8k16 INPUT LAYOUT                              │
│                                                                                  │
│   The 16×16 K-tile is divided into two 8×16 fragments:                          │
│                                                                                  │
│   K-tile (16 rows × N cols):                                                     │
│   ┌─────────────────────┐                                                        │
│   │    Fragment 0       │  rows 0-7                                              │
│   │  (rows 0,1,2,3,...) │                                                        │
│   ├─────────────────────┤                                                        │
│   │    Fragment 1       │  rows 8-15                                             │
│   │  (rows 8,9,10,11,..)│                                                        │
│   └─────────────────────┘                                                        │
│                                                                                  │
│   Each thread with tc_row=0 reads rows {0, 1, 8, 9}:                             │
│   - From Fragment 0: rows 0, 1  (consecutive pair)                               │
│   - From Fragment 1: rows 8, 9  (consecutive pair)                               │
│                                                                                  │
│   Thread mapping for 4 threads (tc_row = 0, 2, 4, 6):                            │
│   ┌─────────────────────────────────────────────────────────┐                    │
│   │ Thread (tc_row=0) → reads rows 0, 1, 8, 9               │                    │
│   │ Thread (tc_row=2) → reads rows 2, 3, 10, 11             │                    │
│   │ Thread (tc_row=4) → reads rows 4, 5, 12, 13             │                    │
│   │ Thread (tc_row=6) → reads rows 6, 7, 14, 15             │                    │
│   └─────────────────────────────────────────────────────────┘                    │
│                                                                                  │
│   All 16 rows are covered by exactly 4 threads!                                  │
│   Each thread reads 4 values (2 from each fragment).                             │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### Reading Values from Shared Memory

```cpp
// Each thread reads 8 INT4 values (4 for b1, 4 for b2)
uint32_t vals[8];

for (int i = 0; i < 4; i++) {
    int k_idx = tc_row + tc_offsets[i];  // e.g., 0, 1, 8, 9 for tc_row=0

    // Read from two column groups (b1 and b2, offset by 8)
    uint32_t b1_val = sh_stage_int_ptr[k_idx * sh_stride + cur_n];
    uint32_t b2_val = sh_stage_int_ptr[k_idx * sh_stride + cur_n + 8];

    // Extract INT4 value using bit operations
    vals[i]     = (b1_val >> (cur_pos * 4)) & 0xF;
    vals[4 + i] = (b2_val >> (cur_pos * 4)) & 0xF;
}
```

**Memory Access Pattern**:
- `sh_stride = 64` (tile_n_size)
- Each row of shared memory holds 64 INT4 values (16 INT32s)
- Threads in the same warp access consecutive memory addresses → coalesced!

#### pack_idx Interleave for Fast INT4→FP16 Conversion

```cpp
constexpr int pack_idx[8] = {0, 2, 4, 6, 1, 3, 5, 7};

uint32_t res = 0;
for (int i = 0; i < 8; i++) {
    res |= vals[pack_idx[i]] << (i * 4);
}
```

**Why this specific ordering?**

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│              pack_idx INTERLEAVE PATTERN EXPLANATION                             │
│                                                                                  │
│   Input vals[8]: [v0, v1, v2, v3, v4, v5, v6, v7]                                │
│                                                                                  │
│   With pack_idx = {0, 2, 4, 6, 1, 3, 5, 7}:                                      │
│                                                                                  │
│   Output bits:                                                                   │
│   ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐                              │
│   │ v7  │ v5  │ v3  │ v1  │ v6  │ v4  │ v2  │ v0  │                              │
│   │bits │bits │bits │bits │bits │bits │bits │bits │                              │
│   │28-31│24-27│20-23│16-19│12-15│8-11 │4-7  │0-3  │                              │
│   └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘                              │
│                                                                                  │
│   This groups:                                                                   │
│   - Lower 16 bits: v0, v2, v4, v6 (even-indexed, from Fragment 0 rows 0,1)       │
│   - Upper 16 bits: v1, v3, v5, v7 (odd-indexed, from Fragment 1 rows 8,9)        │
│                                                                                  │
│   ═══════════════════════════════════════════════════════════════════════════    │
│   WHY THIS ENABLES FAST INT4→FP16 UNPACKING:                                     │
│   ═══════════════════════════════════════════════════════════════════════════    │
│                                                                                  │
│   During matrix multiply, the Marlin kernel uses this sequence:                  │
│                                                                                  │
│   // Extract lower 4 FP16 values (from bits 0-15)                                │
│   frag_b0 = __byte_perm(packed, 0x00000F0F, 0x5040);  // 1 instruction!          │
│   frag_b0 = __hfma2(frag_b0, scale, zero);            // Convert to FP16         │
│                                                                                  │
│   // Extract upper 4 FP16 values (from bits 16-31)                               │
│   frag_b1 = __byte_perm(packed, 0x00000F0F, 0x7060);  // 1 instruction!          │
│   frag_b1 = __hfma2(frag_b1, scale, zero);            // Convert to FP16         │
│                                                                                  │
│   The interleaving allows __byte_perm to extract 4 values at once,               │
│   rather than requiring separate shift-and-mask for each value.                  │
│                                                                                  │
│   Without interleaving: 8 shifts + 8 masks = 16 instructions                     │
│   With interleaving:    2 __byte_perm = 2 instructions (8x faster!)              │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### Output Memory Layout: Tile-Contiguous Storage

```cpp
constexpr int tile_size = tile_k_size * tile_n_size / pack_factor;
//                      = 16 * 64 / 8 = 128 INT32s per tile

int out_offset = (k_tile_id * n_tiles + n_tile_id) * tile_size;
out_ptr[out_offset + th_id * 4 + warp_id] = res;
```

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    MARLIN OUTPUT MEMORY LAYOUT                                   │
│                                                                                  │
│   Each 16×64 tile is stored contiguously as 128 INT32 values:                    │
│                                                                                  │
│   Memory: [Tile(0,0): 128 INT32][Tile(0,1): 128 INT32]...[Tile(K/16-1,N/64-1)]   │
│                                                                                  │
│   Within each tile (128 INT32 = 1024 INT4 = 16×64 weights):                      │
│   ┌─────────────────────────────────────────────────────────────────────────┐    │
│   │ Thread 0 writes: offset 0, 4, 8, 12  (one from each warp)               │    │
│   │ Thread 1 writes: offset 1, 5, 9, 13                                     │    │
│   │ ...                                                                     │    │
│   │ Thread 31 writes: offset 31, 35, 39, 43                                 │    │
│   └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│   This layout ensures:                                                           │
│   1. All weights for one tile are in consecutive 128-byte cache lines           │
│   2. Weights are pre-ordered for Tensor Core fragment loading                    │
│   3. Each warp loads its fragment from a predictable offset                      │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### Pipeline Stages: Hiding Memory Latency

```cpp
constexpr int repack_stages = 8;

// Start filling pipeline
for (int pipe = 0; pipe < repack_stages - 1; pipe++) {
    fetch_to_shared(pipe, k_tile_id, n_tile_id + pipe);
}
wait_for_stage();

// Main loop: process one tile while fetching the next
while (n_tile_id < n_tiles) {
    for (int pipe = 0; pipe < repack_stages; pipe++) {
        // Fetch next tile (async, non-blocking)
        fetch_to_shared((pipe + 7) % 8, k_tile_id, n_tile_id + pipe + 7);

        // Process current tile (compute-bound)
        repack_tile(pipe, k_tile_id, n_tile_id + pipe);

        // Wait for fetch to complete
        wait_for_stage();
    }
    n_tile_id += repack_stages;
}
```

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    DOUBLE BUFFERING WITH 8 STAGES                                │
│                                                                                  │
│   Time →                                                                         │
│   ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐                            │
│   │ F0 │ F1 │ F2 │ F3 │ F4 │ F5 │ F6 │ P0 │ P1 │ P2 │  ...                       │
│   └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘                            │
│   ▲                             ▲    ▲                                           │
│   │                             │    └─ Process tile 0                           │
│   │                             └─ Fetch tile 6 (last of initial batch)          │
│   └─ Start fetching tile 0                                                       │
│                                                                                  │
│   F = Fetch (async memory load via cp.async)                                     │
│   P = Process (compute: extract, interleave, write)                              │
│                                                                                  │
│   By the time we need to process tile 0, it's already in shared memory!          │
│   Memory latency (~300 cycles) is completely hidden by computation.              │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### Summary: How the Three Insights are Implemented

| Insight | Implementation |
|---------|---------------|
| **1. Single cache line per thread** | Tile-contiguous layout (128 bytes = 16×64 INT4 = one tile). Coalesced shared memory access via `sh_stage_int_ptr[k_idx * 64 + cur_n]` |
| **2. Fast INT4→FP16 unpacking** | `pack_idx = {0,2,4,6,1,3,5,7}` interleaving enables `__byte_perm` to extract 4 values in 1 instruction |
| **3. Direct Tensor Core register flow** | `tc_offsets = {0,1,8,9}` matches mma.m16n8k16 fragment layout. No shuffle needed at runtime |

</details> 
