<p align="right">
  <a href="../README.en.md">Back to README</a>
</p>

# M Parameter and High-End GPU Tuning

This document explains what the `M` parameter means in this project, why a larger M is not always faster, and how to further test and improve performance on high-end GPUs such as RTX 4090 / RTX 5090 / A100 / H100.

---

## What Is M

Here, `M` corresponds to `POINTS_PER_THREAD` in the code. It means:

> Each CUDA thread maintains M independent elliptic-curve point chains, and these M points share one Montgomery batch inversion.

In secp256k1 affine point addition, the most expensive operation is modular inversion. The naive method performs 1 modular inversion per point, while one modular inversion roughly costs 256 squaring steps plus about 128 multiplications.

Montgomery batch inversion allows M points to share 1 large modular inversion, then uses extra multiplications to recover the inverse for each point.

The theoretical amortized cost is roughly:

| M | Theoretical amortized cost | Notes |
|---|---:|---|
| 1 | ~256 mul/point | Naive method, 1 large modular inverse per point |
| 8 | ~35 mul/point | 8 points share 1 modular inverse |
| 16 | ~19 mul/point | Current default high-performance configuration |
| 32 | ~11 mul/point | Theoretically lower, but register pressure increases significantly |

From the formula alone, a larger M reduces the modular-inversion cost per point.

But this is only the theoretical cost of the ECC inversion part. It does not mean the whole program will always become faster.

---

## Why M Is Not Always Better When Larger

Each CUDA thread must keep M groups of point coordinates and intermediate variables, such as:

- `px`
- `py`
- `dx`
- `dy`
- `prefix`
- temporary variables used by batch inversion
- local variables used by Base58 / Keccak / SHA

As M increases, register usage per thread rises quickly.

If registers are insufficient, the compiler spills some variables into local memory. CUDA local memory is not "local" in the CPU sense; it usually goes through global memory/cache paths and has much higher latency than registers.

When M is too large, you may see:

- registers per thread becomes too high
- occupancy drops
- local memory spill increases
- warp stalls increase
- SM utilization looks high, but addresses/second decreases

Therefore, the optimal M depends on the specific GPU, driver version, compiler, search mode, and register allocation of the kernel.

---

## Possible Behavior on Different GPUs

On RTX 3060, `M=16` is usually faster than `M=8`, but `M=32` may become slower because registers spill into local memory.

On high-end GPUs such as RTX 4090 / RTX 5090 / A100 / H100, the situation may be different:

- more register resources
- stronger scheduling
- stronger L1/L2 cache
- more SMs
- higher overall throughput

Therefore, `M=24` or `M=32` may become a better point.

However, note that once ECC inversion has already been sufficiently amortized by `M=16`, the bottleneck may move to other parts:

- Keccak-256
- SHA-256
- Base58 encoding
- hit-result writeback
- CPU verification

Especially in pure prefix mode, full Base58 cannot early-exit, so the benefit from increasing M is more likely to diminish.

---

## Current M Limitation in the Code

The current code automatically tests `M=8` and `M=16` at startup:

```python
M_CANDIDATES = [16, 8]
```

The program compiles kernels with different M values, runs a short benchmark, and then selects the configuration with higher throughput.

This design is correct, because the best M should be measured rather than guessed.

However, the current code structure has an important limitation: **you cannot directly change M to 24 or 32**.

The reason is that the kernel packs `p_idx` into the low 4 bits of `thread_id`:

```c
out[idx].thread_id = ((u32)tid << 4) | ((u32)p & 0xF);
```

The Python side unpacks it like this:

```python
real_tid = packed >> 4
p_idx = packed & 0xF
```

This means `p_idx` can only represent `0..15`, so the current structure naturally supports only:

```text
M <= 16
```

If you directly change M to 32, `p=16..31` will be truncated into `0..15`. The GPU hit will then use the wrong point index when reconstructing the private key, causing CPU verification to fail and discard hits, and possibly affecting statistics.

---

## How to Remove the M <= 16 Limitation

To support `M=24` / `M=32`, it is recommended to modify `MatchRecord` first and store `thread_id` and `point_idx` separately.

The CUDA struct can be changed to:

```c
struct MatchRecord {
    u32  thread_id;
    u32  point_idx;
    u64  step;
    char address[34];
    char _pad2[6];
};
```

The kernel write path becomes:

```c
out[idx].thread_id = (u32)tid;
out[idx].point_idx = (u32)p;
out[idx].step = step_offset + (u64)step;
```

The Python-side `MATCH_DTYPE` should be updated accordingly:

```python
MATCH_DTYPE = np.dtype([
    ("thread_id", np.uint32),
    ("point_idx", np.uint32),
    ("step",      np.uint64),
    ("address",   np.uint8, 34),
    ("_pad2",     np.uint8, 6),
], align=True)
```

When reading hits:

```python
real_tid = int(m["thread_id"])
p_idx = int(m["point_idx"])
```

This removes the low-4-bit packing limit, allowing M to be extended further.

---

## Recommended Tuning Flow on High-End GPUs

If you use a high-end GPU such as RTX 5090 and want to squeeze more performance, follow this process.

### 1. Verify the Original Version First

First confirm the original `M=8/16` version passes:

```bash
python verify_kernel.py
```

Make sure GPU address derivation fully matches the CPU reference implementation.

### 2. Modify MatchRecord

Split `thread_id` and `point_idx` as described above to remove the `M <= 16` structural limitation.

### 3. Expand M Candidates

Change:

```python
M_CANDIDATES = [16, 8]
```

to:

```python
M_CANDIDATES = [32, 24, 16, 8]
```

You can also test more aggressively:

```python
M_CANDIDATES = [48, 40, 32, 24, 16, 8]
```

But the larger M is, the more likely register spilling becomes. Do not judge only by the theoretical value.

### 4. Extend verify_kernel.py

Change the verification range from:

```python
for M in (8, 16):
```

to:

```python
for M in (8, 16, 24, 32):
```

If you also test M=40/48, add them to verification as well.

### 5. Benchmark by Mode

Different search modes have different bottlenecks, so test them separately:

| Mode | Main bottleneck | Benefit from larger M |
|---|---|---|
| Pure prefix | Full Base58 + Keccak/SHA + ECC | Medium |
| Suffix | Base58 tail early-exit + ECC | Potentially high |
| Tail repetition | Base58 tail early-exit + ECC | Potentially high |
| Very loose condition | Hit writeback and CPU verification | M is not the main bottleneck |

Do not use one mode's benchmark to represent all modes.

### 6. Watch Real Throughput

The final criterion should be actual addresses/second, not theoretical `mul/point`.

If M=32 is slower than M=16, it usually means:

- register usage is too high
- occupancy drops
- local memory spill increases
- compiler-generated code no longer fits the current GPU well

In that case, fall back to M=24 or M=16.

If M=32 is faster, you can continue trying M=40/48, but you must keep verifying correctness and watching spill.

---

## Recommended Metrics to Observe

Use Nsight Compute or compiler output to inspect:

- registers per thread
- local memory per thread
- occupancy
- achieved occupancy
- SM utilization
- memory throughput
- instruction throughput
- warp stall reasons

Focus on:

```text
Does registers/thread jump sharply when M increases?
Is there local memory spill?
Does occupancy drop significantly?
Does addresses/second actually improve?
```

If addresses/second does not improve, increasing M is not worth it even if the theoretical inversion cost is lower.

---

## Short Conclusion

- `M` is the number of point chains processed by each CUDA thread.
- Increasing M reduces the amortized ECC modular-inversion cost.
- But increasing M also increases register pressure and may cause local memory spill.
- The current code safely supports up to `M=16`, because `p_idx` is packed into only 4 low bits.
- On high-end GPUs such as RTX 5090, `M=24/32` may be faster, but `MatchRecord` must be modified first.
- The best M should not be guessed; the program should benchmark and choose it at startup.
- Real address throughput is the final standard, not the theoretical formula alone.
