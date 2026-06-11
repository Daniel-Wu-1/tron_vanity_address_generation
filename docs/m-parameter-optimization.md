<p align="right">
  <a href="../README.md">返回 README</a>
</p>

# M 参数与高端 GPU 调优说明

本文说明 `M` 参数在本项目中的含义、为什么它不是越大越好，以及在 RTX 4090 / RTX 5090 / A100 / H100 等高端 GPU 上如何进一步测试和提高性能。

---

## M 是什么

这里的 `M` 对应代码里的 `POINTS_PER_THREAD`，含义是：

> 每个 CUDA 线程同时维护 M 条独立的椭圆曲线点链，并对这 M 个点共享一次 Montgomery 批量求逆。

在 secp256k1 仿射点加中，最贵的操作是模逆。朴素做法是每个点做 1 次模逆，而一次模逆大约需要 256 次平方和约 128 次乘法量级的工作。

Montgomery 批量求逆可以让 M 个点共享 1 次大模逆，再用额外乘法把每个点的逆元还原出来。

理论摊销成本大致如下：

| M | 理论摊销成本 | 说明 |
|---|---:|---|
| 1 | ~256 mul/点 | 朴素做法，每个点 1 次大模逆 |
| 8 | ~35 mul/点 | 8 个点共享 1 次模逆 |
| 16 | ~19 mul/点 | 当前默认高性能配置 |
| 32 | ~11 mul/点 | 理论更低，但寄存器压力显著增加 |

从公式上看，M 越大，每个点摊到的模逆成本越低。

但这只是 ECC 模逆部分的理论成本，不等于整体程序一定更快。

---

## 为什么 M 不是越大越好

每个 CUDA 线程需要同时保存 M 组点坐标和中间变量，例如：

- `px`
- `py`
- `dx`
- `dy`
- `prefix`
- 批量求逆临时变量
- Base58 / Keccak / SHA 过程中的局部变量

M 增大后，单线程寄存器使用量会快速上升。

如果寄存器不够，编译器会把一部分变量 spill 到 local memory。CUDA 里的 local memory 并不是 CPU 意义上的“本地内存”，它通常会走显存/缓存路径，延迟远高于寄存器。

所以 M 太大时可能出现：

- registers per thread 过高
- occupancy 下降
- local memory spill 增加
- warp stall 增加
- SM 利用率看起来很高，但实际每秒地址数下降

因此，M 的最优值取决于具体 GPU、驱动版本、编译器、运行模式和 kernel 的寄存器分配情况。

---

## 不同 GPU 上的可能表现

在 RTX 3060 上，`M=16` 通常比 `M=8` 更快，但 `M=32` 可能因为寄存器溢出到 local memory，反而变慢。

在 RTX 4090 / RTX 5090 / A100 / H100 这类高端 GPU 上，情况可能不同：

- 寄存器资源更充足
- 调度能力更强
- L1/L2 缓存更强
- SM 数量更多
- 整体吞吐更高

因此，`M=24` 或 `M=32` 有可能成为更优点。

但也要注意：当 ECC 模逆已经被 `M=16` 充分摊薄后，瓶颈可能转移到其他部分：

- Keccak-256
- SHA-256
- Base58 编码
- 命中结果写回
- CPU 复核

尤其在 prefix 模式下，完整 Base58 无法早退，继续增大 M 的收益会更容易递减。

---

## 当前代码的 M 限制

当前代码启动时会自动测试 `M=8` 和 `M=16`：

```python
M_CANDIDATES = [16, 8]
```

程序会分别编译不同 M 的 kernel，实际跑一小段 benchmark，然后选择吞吐更高的配置。

这个设计是正确的，因为 M 的最佳值应该实测，而不是靠猜。

不过，当前代码结构还有一个重要限制：**不能直接把 M 改成 24 或 32**。

原因在于 kernel 里把 `p_idx` 打包进了 `thread_id` 的低 4 位：

```c
out[idx].thread_id = ((u32)tid << 4) | ((u32)p & 0xF);
```

Python 端再这样解包：

```python
real_tid = packed >> 4
p_idx = packed & 0xF
```

这意味着 `p_idx` 最多只能表示 `0..15`，也就是当前结构天然只支持：

```text
M <= 16
```

如果直接把 M 改成 32，`p=16..31` 会被截断成 `0..15`。结果是 GPU 命中反推私钥时会使用错误的 point index，导致 CPU 校验失败后丢弃命中，甚至影响统计结果。

---

## 如何解除 M <= 16 的限制

如果要支持 `M=24` / `M=32`，建议先修改 `MatchRecord`，把 `thread_id` 和 `point_idx` 分开保存。

CUDA 结构可以改成：

```c
struct MatchRecord {
    u32  thread_id;
    u32  point_idx;
    u64  step;
    char address[34];
    char _pad2[6];
};
```

kernel 写入时改成：

```c
out[idx].thread_id = (u32)tid;
out[idx].point_idx = (u32)p;
out[idx].step = step_offset + (u64)step;
```

Python 端 `MATCH_DTYPE` 对应改成：

```python
MATCH_DTYPE = np.dtype([
    ("thread_id", np.uint32),
    ("point_idx", np.uint32),
    ("step",      np.uint64),
    ("address",   np.uint8, 34),
    ("_pad2",     np.uint8, 6),
], align=True)
```

读取命中时改成：

```python
real_tid = int(m["thread_id"])
p_idx = int(m["point_idx"])
```

这样就不会被低 4 位限制卡住，M 可以继续扩展。

---

## 高端 GPU 上的推荐调优流程

如果你使用 RTX 5090 等高端 GPU，希望进一步压榨性能，建议按下面流程做。

### 1. 先验证原始版本

先确认原始 `M=8/16` 版本能通过：

```bash
python verify_kernel.py
```

确保 GPU 地址派生和 CPU 参考实现完全一致。

### 2. 修改 MatchRecord

按上文方式把 `thread_id` 和 `point_idx` 拆开，解除 `M <= 16` 的结构限制。

### 3. 扩展 M 候选值

修改：

```python
M_CANDIDATES = [16, 8]
```

为：

```python
M_CANDIDATES = [32, 24, 16, 8]
```

也可以更激进地测试：

```python
M_CANDIDATES = [48, 40, 32, 24, 16, 8]
```

但 M 越大越可能触发寄存器 spill，所以不要只看理论值。

### 4. 扩展 verify_kernel.py

把验证范围从：

```python
for M in (8, 16):
```

扩展成：

```python
for M in (8, 16, 24, 32):
```

如果你还测试 M=40/48，也同步加入验证。

### 5. 分模式 benchmark

不同搜索模式的瓶颈不同，建议分别测试：

| 模式 | 主要瓶颈 | M 增大收益 |
|---|---|---|
| 纯 prefix | 完整 Base58 + Keccak/SHA + ECC | 中等 |
| suffix | Base58 tail early-exit + ECC | 可能较高 |
| 尾部重复 | Base58 tail early-exit + ECC | 可能较高 |
| 很宽松的条件 | 命中写回和 CPU 校验 | M 不是主要瓶颈 |

不要只用一种模式的 benchmark 代表全部情况。

### 6. 观察真实吞吐

最终判断标准应该是实际输出的地址数/秒，而不是理论 `mul/点`。

如果 M=32 比 M=16 慢，通常说明：

- 寄存器使用量太高
- occupancy 下降
- local memory spill 增加
- 编译器生成的代码不再适合当前 GPU

这时应回退到 M=24 或 M=16。

如果 M=32 更快，可以继续尝试 M=40/48，但必须继续验证正确性和观察 spill。

---

## 建议观察的性能指标

可以用 Nsight Compute 或编译器输出观察：

- registers per thread
- local memory per thread
- occupancy
- achieved occupancy
- SM utilization
- memory throughput
- instruction throughput
- warp stall reasons

重点关注：

```text
是否因为 M 增大导致 registers/thread 暴涨
是否出现 local memory spill
是否 occupancy 明显下降
是否每秒地址数真的提高
```

如果每秒地址数没有提高，即使理论模逆摊销更低，也不值得增大 M。

---

## 简短结论

- `M` 是每个 CUDA 线程同时处理的点链数量。
- 增大 M 可以降低 ECC 模逆的摊销成本。
- 但 M 增大会增加寄存器压力，可能导致 local memory spill。
- 当前代码安全支持的上限是 `M=16`，因为 `p_idx` 只用了低 4 位打包。
- RTX 5090 等高端 GPU 上，`M=24/32` 可能更快，但必须先修改 `MatchRecord`。
- 最优 M 不应该靠猜，应该让程序启动时自动 benchmark 选择。
- 最终以真实地址吞吐为准，而不是只看理论公式。
