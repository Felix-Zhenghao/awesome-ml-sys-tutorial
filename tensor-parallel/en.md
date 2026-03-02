[English](en.md) | [中文](zh.md)

# Tensor Parallel, Sequence Parallel and Loss Parallel

After reading this blog, you will understand tensor parallel, sequence parallel and loss parallel and how they are implemented on real model in-depth.

The [torch.distributed documentation on TP and SP](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html) is well-written but too concise. Also, some illustrations are confusing. I will reconstruct and expand it substantially with concise but sufficient explanation on code, illustrations and communication pattern.

After reading this, I encourage you to skim through the pytorch doc for specific programming guidance (for instance, when should we set `use_local_output=False`).

## Column Wise TP and Row Wise TP

TP splits the individual matrices (tensors) within a single layer across multiple GPUs. See figure 1.

There are two types of tensor parallel strategies: row-wise TP and column-wise TP. The difference is whether the **weight matrix** is split row-wise or col-wise across different GPUs. See figure 1 for examples.

![Column-wise TP and Row-wise TP](colTP-rowTP-illus.jpg)

## Orchestrate ColWise TP and RowWise TP in a Transformer Layer

In the torch.distributed tutorial, a transformer layer orchestrates two types of TP in the following way:
```python
layer_tp_plan = {
    "attention.wq": ColwiseParallel(use_local_output=False),
    "attention.wk": ColwiseParallel(use_local_output=False),
    "attention.wv": ColwiseParallel(use_local_output=False),
    "attention.wo": RowwiseParallel(),
    "feed_forward.up_proj": ColwiseParallel(),
    "feed_forward.down_proj": RowwiseParallel(),
}
```

![TP forward pass in a Transformer layer (TP=2)](tp-forward-transformer.png)
![TP backward pass in a Transformer layer (TP=2)](tp-backward-transformer.png)

This process is illustrated in figure 2 (a TP=2 config). Note that normalization, activation function and residual connection are omitted for simplicity. Now let's analyze the process.

### Communication cost in forward and backward pass

As we can see there are two all-reduce's on forward pass and two on backward pass. Here is the conclusion: **ColWise TP yields all-reduce in backward pass and RowWise TP yields all-reduce in forward pass.** This is intuitive.

- The input of ColWise TP is typically the same on all GPUs (result from forward all-reduce). Therefore, it separately contributes to different forward branches of different GPUs. Therefore, we need to gather and sum the gradient from all branches. This leads to an all-reduce in backward pass.
- The input of RowWise TP is sharded column-wise. Therefore, each forward branch gets different parts of the input. So we want to merge the branches (all-reduce) in the forward pass (see figure 1), but the backward pass is much easier.

According to the [ring all-reduce blog](/TechBlog/ring-all-reduce/), the communication cost of one all-reduce is approximately $2S$ per GPU where $S$ is the storage size of data to be reduced. Here we have four all-reduce's in total and all operates on a tensor of shape $[N,C]$ (typical $N = \text{bsz} \times \text{seq{\_}len}$ and $C$ is model dimension). So the total communication cost is $4 \times 2 \times (N \times C \times \text{byte{\_}per{\_}element})$ **per GPU per layer**. It is a balanced but expensive cost.

### How TP reduces memory burden on each GPU

Easy. Because weights and thus activation is sharded across all TP GPUs. Memory is reduced to approximately $\frac{1}{\text{TP}}$ for each GPU.

## Add Sequence Parallel to The Orchestration

This is how torch.distributed doc describes SP.[^1]

[^1]: Sequence Parallel works on top of the Tensor Parallel illustrated above. Compared with basic Tensor Parallel, which only shards tensors within the Attention modules and FeedForward modules and keep their module inputs and outputs (namely activations in the forward pass and gradients in the backward pass) replicated, Sequence Parallel keeps them sharded on the sequence dimension.

See how torch.distributed orchestrates TP and SP. I will walk through this chunk of code in very great details. However, it is recommended to read the [ring all-reduce blog](/TechBlog/ring-all-reduce/) first or at least understand why **`all_reduce = reduce_scatter + all_gather`** (figure 4).

![All-reduce = Reduce-scatter + All-gather](all-reduce-reduce-scatter-all-gather.png)

```python
layer_tp_plan = {
    # Now the input and output of SequenceParallel has Shard(1)
    # layouts, to represent the input/output tensors sharded on
    # the sequence dimension
    "attention_norm": SequenceParallel(),
    "attention": PrepareModuleInput(
        input_layouts=(Shard(1), Replicate()),
        desired_input_layouts=(Replicate(), Replicate()),
    ),
    "attention.wq": ColwiseParallel(use_local_output=False),
    "attention.wk": ColwiseParallel(use_local_output=False),
    "attention.wv": ColwiseParallel(use_local_output=False),
    "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
    "ffn_norm": SequenceParallel(),
    "feed_forward": PrepareModuleInput(
        input_layouts=(Shard(1),),
        desired_input_layouts=(Replicate(),),
    ),
    "feed_forward.w1": ColwiseParallel(),
    "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
    "feed_forward.w3": ColwiseParallel(),
}
```

First, what is SP? The tensor (shape=`(B,S,C)`) is sharded along `dim=1` and each GPU gets a chunk (sub-sequence). When doing layer norm and activation function, each GPU does this on its local sub-sequence. This is valid because both normalization and the activation functions are computed completely independently for each token.

Recall that forward of a TransformerBlock is:
```python
def forward(self, x):
    h = x + self.attention(self.attention_norm(x))
    out = h + self.feed_forward(self.ffn_norm(h))
    return out
```

I will take `"ffn_norm": SequenceParallel()` as an example to show the orchestration between TP and SP.

Note that in the code, the output layouts of `attention.wo` is `Shard(1)`. This means that the output tensor after applying `wo` is sharded along sequence length dimension. According to Figure 2, after applying `wo` we should do all-reduce. Now since all-reduce can be divided into a reduce-scatter phase and an all-gather phase, and each GPU holds the partial result for a sub-sequence after the reduce-scatter phase, we can stop at the end of the reduce-scatter phase and do `ffn_norm` on the sub-sequence rather than the whole sequence! Perfect and clever.

But, we still need to do the full all-reduce after that (in this case, move on to the all-gather phase). This is what the following code chunk does. It tells torch.distributed that the input of `ffn` is currently sharded along `dim=1` and we want it to be a full tensor rather than a sharded tensor (`Replicate()`). The torch.distributed will do all-gather for us before sending the input tensor to `ffn`.
```python
"feed_forward": PrepareModuleInput(
    input_layouts=(Shard(1),),
    desired_input_layouts=(Replicate(),),
)
```

This is how SP and TP orchestrate. RowWise TP does all-reduce. SP tells RowWise TP: stop at the reduce-scatter phase and do some computation with partial result on each GPU, and after that do all-gather to finish the all-reduce.

## Loss Parallel

```python
"norm": SequenceParallel(),
"output": ColwiseParallel(
    input_layouts=Shard(1),
    use_local_output=False, # use DTensor as the output
),
```

The above is the loss parallel tp mesh. In loss parallel, instead of gathering the logits to compute standard cross-entropy, the logits remain sharded across the GPUs (shape: $B \times S \times \frac{V}{\text{TP}}$).

Loss parallel breaks Cross-Entropy down into its two mathematical components: Log-Softmax and Negative Log-Likelihood (NLL) Loss.

### Log-Softmax

The formula for Log-Softmax for a given logit $x_i$ is:

$$\text{LogSoftmax}(x_i) = x_i - \max(x) - \log\left(\sum \exp(x_k - \max(x))\right)$$

To compute this, every GPU needs two global numbers: the global maximum logit (to prevent numerical overflow) and the global sum of exponentials (the denominator). The computation technique is the same as flash_attn online softmax.

### Negative Log-Likelihood (NLL) Loss

In this phase, we just need to find the log likelihood of the target vocab index and do a reduction(sum) along the sequence dimension.
