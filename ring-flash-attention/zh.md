[English](en.md) | [中文](zh.md)

# Ring Flash Attention（上下文并行的一个实例）

本文的目标是使用 flash attention API 实现 ring attention，即实现一个支持上下文并行（context parallelism）的 flash attention。本文基于优秀的开源项目 [ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention)。该项目作者也写了一篇[博客](https://zhuanlan.zhihu.com/p/683714620)，但我会在此基础上大幅扩展，提供更多数学推导、代码解析和图示说明。

## 为什么需要 ring (flash) attention

1. Ring attention（后文会详细解释）支持在极长上下文（例如 100 万 token）上进行点积注意力计算而不会 OOM。
2. 我们希望在 ring attention 的实现中使用高效的 flash attention API。只需要在 flash attn 外面包一层通信逻辑即可！

## Ring attention 前向传播

实现可以用一句话概括：每一步，(1) 执行通信 (2) 调用 flash attn API 进行本地计算 (3) 用 online softmax 更新本地结果。

见图 1 的逻辑流程。假设我们有四个处理器（GPU）。初始时，序列（具有很长的序列长度）被分成四个部分（与 GPU 数量相同）。每个 GPU 负责计算其本地子序列的输出。KV 在 GPU 之间传递和接收。接收到新的 KV 后，每个 GPU 会让本地子序列的 query 去关注新的 K 并相应地更新输出。最终，每个本地子序列会看到**整个**序列中它需要的所有 KV 来计算其本地输出。

![Ring attention 前向传播逻辑](ring-flash-attn-forward.png)

现在我们有两个关键问题要回答：(1) 如何用其他子序列的 KV 块来更新本地子序列的输出？(2) 如何使用 flash attention 的实现？

### 回答问题 1

问题一的答案展示在代码中。现在让我们推导其数学原理。注意"lse"代表"log-sum-exp"。

> Online softmax 代码

```python
@torch.jit.script
def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

    # new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
    # new_out = torch.exp(lse - new_lse) * out
    #         + torch.exp(block_lse - new_lse) * block_out
    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)

    return out, lse
```

数学推导如下：

```
(1) new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
(2) new_out = torch.exp(lse - new_lse) * out
            + torch.exp(block_lse - new_lse) * block_out

# 推导 (1)
log(e^lse) + log(1+e^{block_lse-lse})
    = log(e^lse + e^block_lse)
    = log(e^new_lse)
    = new_lse

# 推导 (2)
令：
out = X / e^lse，block_out = Y / e^block_lse

且我们知道：
new_out = (X+Y) / e^{lse + block_lse}

所以：
new_out = (out * e^lse + block_out * e^block_lse)
        / e^{lse + block_lse}
        = e^{-block_lse} * out + e^{-lse} * block_out
        = e^{lse - new_lse} * out
        + e^{block_lse - new_lse} * block_out
```

再补充一点。为什么这里可以保证数值稳定性？因为 `block_lse` 和 `block_out` 都是由 flash_attn API 返回的，可以认为是数值稳定的。上述更新公式本身也是数值稳定的。因此，整个实现是数值稳定的。

### 回答问题 2

从图 1 可以看出，在第一步中，我们只需设置 `causal=True`，而在后续所有步骤中，我们可以设置 `causal=False`，因为 query 是在关注前缀 KV 块。

通过编排 flash_attn API 的 `causal` 参数，每一步我们只需调用一次 flash_attn 即可完成计算。

代码位于 [ring_flash_attn.py](https://github.com/zhuzilin/ring-flash-attention/blob/main/ring_flash_attn/ring_flash_attn.py) 的 `ring_flash_attn_forward` 函数中：

```python
"causal": causal and step == 0
```

## Ring attention 反向传播

本地 `dQ` 需要看到所有前缀 KV 块和本地 `dout`。`dK` 和 `dV` 需要看到所有后续的 Q 块和 `dout`。因此，`dQ` 可以留在本地 GPU 上，KV 沿环传递。反过来，`dKV` 应沿环传递以访问每个本地的 `Q` 和 `dout`。

为了实现这个求和，反向传播算法将 `dk` 和 `dv` 缓冲区与 $K$ 和 $V$ 块一起沿环传递。在循环的每一步中：

1. GPU $i$ 计算局部梯度（`block_dk_buffer`、`block_dv_buffer`）。
2. 将这些新计算的局部梯度加到从前一个 GPU 接收到的传递累加缓冲区（`next_dk`、`next_dv`）上（`dk = block_dk_buffer + next_dk`，`dv = block_dv_buffer + next_dv`）。
3. 使用 `next_dk, next_dv = d_kv_comm.send_recv_kv(dk, dv)` 将更新后的 `dk` 和 `dv` 缓冲区发送给下一个 GPU。

见图 2 以及 [ring_flash_attn.py](https://github.com/zhuzilin/ring-flash-attention/blob/main/ring_flash_attn/ring_flash_attn.py) 中 `ring_flash_attn_backward` 函数的代码：

![Ring attention 反向传播逻辑](ring-flash-attn-backward.png)

## 朴素 ring flash attention 的通信开销

**前向传播：**
假设序列的 K 块存储大小为 $S$，有 $P$ 个处理器（因此每个子序列的 K 块大小为 $S/P$）。在前向传播中，有 $P$ 个通信步骤，每个 GPU 发送一个 K 块和一个 V 块（$2 \cdot S/P$）。这导致每个 GPU 的通信开销为 $2S$，总通信开销为 $2PS$。

**反向传播：**
同样有 $P$ 个通信步骤，每个 GPU 发送一个 K 块、一个 V 块、一个 dK 块和一个 dV 块（$C \cdot S/P$，其中 $C$ 是取决于梯度精度的常数）。这导致每个 GPU 的通信开销为 $CS$，总通信开销为 $CPS$。

总体通信开销与 GPU 数量呈线性增长，且均匀分布在网络互联上，表明了优秀的可扩展性。然而，在朴素的 Ring Attention 中，由于因果掩码（causal masking），计算负担严重不均衡。

## Zigzag ring attention

朴素的 ring attention 无法均衡地分配计算量。可以看到，被分配到序列靠前部分的 GPU 会更早进入空闲状态。

Zigzag 块分配策略解决了这个问题。图 3 展示了一个 4 个 GPU、16 个块的示例。四个 GPU 被分配的子序列编号为：

$$0 \; 1 \; 14 \; 15 \mid 2 \; 3 \; 12 \; 13 \mid 4 \; 5 \; 10 \; 11 \mid 6 \; 7 \; 8 \; 9$$

这有两个优势：(1) 计算分配更加均衡；(2) 每一步仍然只需调用一次 flash_attn API，只需对输入进行切片即可。例如，当 `0 1 14 15` 的 KV 进入拥有 Q `2 3 12 13` 的 GPU 时，送入 flash_attn API 的 KV 和 Q 将分别是 `0 1` 和 `2 3 12 13`，设置 `causal=False`。

性能对比见[此处](https://github.com/zhuzilin/ring-flash-attention/tree/main?tab=readme-ov-file#performance-summary)。

![Zigzag ring flash attention](ring-flash-attn-zigzag.png)
