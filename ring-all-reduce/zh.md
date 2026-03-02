[English](en.md) | [中文](zh.md)

# Ring All-Reduce

Ring all-reduce 通过将处理器组织成"环"的形式，高效地完成 all-reduce 通信模式。

> All-reduce：GPU A 拥有数据 X；GPU B 拥有数据 Y => 最终两者都拥有 X+Y（此处为 reduce_sum，也可以是 reduce_max、min 等）。

假设我们有四个处理器（GPU）。见图 1。数据首先被分成四个块（与 GPU 数量相同），然后在"环"中逐步传递和接收。在图 1 所示流程结束时，每个 GPU 都持有一个可作为最终结果（此处为求和）的数据块。

![阶段 1：环形归约](phase-1-ring-reduce.png)

接着，见图 2。我们只需做一次 all-gather 通信，让每个 GPU 将其结果块发送给所有其他 GPU。之后，所有 GPU 都将拥有全部结果数据块，all-reduce 就完成了。

> All-gather：GPU A 拥有数据 X；GPU B 拥有数据 Y => 最终两者都拥有数据 X 和 Y。

![阶段 2：All-Gather](phase-2-ring-reduce.png)

现在，我们来计算 ring all-reduce 的通信开销。

假设我们有 P 个处理器，数据的存储大小为 S。数据被分成 P 个块，每个块大小为 $S/P$。在第一阶段（图 1）中，共有 $P-1$ 步，每一步有 $P$ 个数据块被发送和接收。因此，第一阶段总共发送的数据量为 $(P-1) \cdot [ (S/P) \cdot P ] = (P-1)S$。在第二阶段，每个 GPU 发送 $(P-1) \cdot (S/P)$ 的数据，共有 $P$ 个 GPU，因此第二阶段总共发送的数据量为 $(P-1) \cdot [ (S/P) \cdot P ] = (P-1)S$。总通信开销为 $2(P-1)S$，与处理器数量呈线性增长。每个 GPU 发送和接收 $2 \cdot \frac{P-1}{P} \cdot S \approx 2S$ 的数据，因此通信压力在各 GPU 之间分布得非常均匀，表明该方案具有良好的可扩展性。
