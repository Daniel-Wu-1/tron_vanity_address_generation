<p align="right">
  <a href="./README.md"><img alt="中文" src="https://img.shields.io/badge/%E4%B8%AD%E6%96%87-%E5%88%87%E6%8D%A2-6e7781?style=flat-square"></a>
  <img alt="English Current" src="https://img.shields.io/badge/English-Current-0969da?style=flat-square">
</p>

# TRON Vanity Address Generator (CUDA + CPU Full-Speed Version)

In real tests, the current efficiency is the highest (RTX3060 one hundred million attempts/second, 6A-style addresses come out in seconds). Comparisons are welcome. The code structure is listed. It is completely free, purely open source, and not encrypted (including the CUDA computing unit; unlike some people's released software that also carries encrypted backend files, I will not name names here). The software code and operating principles are public, so you can audit/Fork/modify it yourself, and you can also apply to the author for authorization, if you have time ^^

If you want to distribute it, please download the code yourself, audit/modify it, and then package it. The required dependencies are listed below.

Beginner users who do not know how to configure the environment can directly download the release version and use it: **click [[TRON.rar]](https://github.com/Daniel-Wu-1/tron_vanity/releases/download/TRON%2FTRX%2FUSDT%E9%9D%93%E5%8F%B7%E7%94%9F%E6%88%90V1.0/TRON.rar) to download**. This version has all required dependencies built in and can be used directly.

---

## Table of Contents

1. [Project Introduction](#project-introduction)
2. [Measured Performance](#measured-performance)
3. [Directory Structure](#directory-structure)
4. [Overall Project Architecture](#overall-project-architecture)
5. [Purpose of Each File](#purpose-of-each-file)
6. [Implementation of Key Modules](#implementation-of-key-modules)
7. [Tech Stack and Dependencies](#tech-stack-and-dependencies)
8. [Environment Requirements](#environment-requirements)
9. [Installation Steps (Beginner Version)](#installation-steps-beginner-version)
10. [Usage Instructions](#usage-instructions)
11. [Advanced Configuration (Advanced Users)](#advanced-configuration-advanced-users)
12. [Estimated Running Results on Different Computer Configurations](#estimated-running-results-on-different-computer-configurations)
13. [Common Errors and Solutions](#common-errors-and-solutions)
14. [Memory and Crash Scenario Analysis](#memory-and-crash-scenario-analysis)
15. [Concurrency and Multithreading Risks](#concurrency-and-multithreading-risks)
16. [Security Tips](#security-tips)

---

## Project Introduction

This project is used to **offline generate TRON wallet addresses that match custom patterns**, for example:
- **Last N characters are the same**: `TXxxxxxxxxxxxxxxxxxxxxxxxxxxxxx99999` (last 5 characters are 9)
- **Custom prefix**: `TXYZ...` (fixed 4 characters at the beginning of the address)
- **Custom suffix**: `...abcd` (fixed 4 characters at the end of the address)
- **Prefix and suffix combination**: for example `T8...8888`

To pursue extreme speed, it does not rely on any Python cryptographic library for the main computation — all secp256k1 elliptic curve operations, Keccak-256, SHA-256, and Base58 encoding are completed inside the **handwritten CUDA kernel**. CPU multiprocessing also participates in the search in parallel as a supplement to the GPU.

> ⚠ **The private key is the asset**. The generated hit-address file contains plaintext private keys. Please keep it properly. After importing it into a wallet, it is recommended to destroy the source file. Do not run it on an untrusted machine, and do not upload the results to any cloud service.

---

## Measured Performance
![Image description](Snipaste_2026-06-08_15-32-10.png)
![Image description](Snipaste_2026-06-08_15-32-37.png)

### RTX 3060 (28 SM, 12GB VRAM)

| Mode | Throughput |
|---|---|
| Pure prefix (any length) | ~80M addresses/second |
| Tail repetition / suffix (benefits from early-exit optimization) | **~100M addresses/second** |
| GPU utilization | 100% (Compute_0 engine) |
| GPU power | 165W / 170W (97% TDP) |
| CPU utilization | Default about 90% (on a 12-logical-core machine, 10 worker processes, 2 cores reserved) |

Time estimates (tail repetition mode, 100M/second):
- **Last 5 characters repeated**: average 50 results every 9 seconds
- **Last 6 characters repeated**: average 1 result every 6.5 seconds
- **Last 7 characters repeated**: average 1 result every 6.5 minutes
- **Last 8 characters repeated**: average 1 result every 6.5 hours
- **4-character custom suffix**: average 1 result every 0.1 seconds (1/58⁴ = 1/11.3 million)
- **6-character custom suffix**: average 1 result every 6 minutes (1/58⁶ = 1/38 billion)
- **8-character custom suffix**: average about 1 result every 14 hours

> **Note about Windows Task Manager**: By default it shows the "3D" engine utilization, while CUDA computation uses the **"Compute_0"** engine. In Task Manager → Performance → GPU, switch the drop-down menu in the upper-right corner to **"Compute_0"** to see the correct 100% utilization. `nvidia-smi` can always show accurate data.

---

## Directory Structure

```
tron_vanity_address_generation-main/                            ← project root directory
├── tron_vanity_gpu.py           Main program (CLI + GPU scheduling + multiprocessing coordination)
├── kernels.cu                   All CUDA kernels (C/C++ source code, NVRTC runtime compilation)
├── cpu_worker.py                CPU worker process entry point (spawned by multiprocessing)
├── verify_kernel.py             GPU kernel correctness verification (4096 addresses byte-by-byte compared with CPU)
├── requirements.txt             Python dependency list
├── README.md                    This document
├── .gitignore                   Ignores cache and hit-address directory
└── 命中地址/                     Automatically created when the program first starts
    └── matches_YYYYMMDD_HHMMSS.txt   One file generated per run, containing hit addresses + private keys
```

---

## Overall Project Architecture

The whole system is **"main process + GPU kernel + N CPU worker processes + 1 status-line thread"** working together:

```
┌────────────────────────────────────────────────────────────────────┐
│ Main process (tron_vanity_gpu.py)                                  │
│                                                                    │
│  ┌────────────────────────────────────────┐                        │
│  │ Main loop (dual stream ping-pong)       │                        │
│  │  ─── stream A: sync → collect hits → relaunch                   │
│  │  ─── stream B: kernel runs on GPU at the same time              │
│  └────────────────────────────────────────┘                        │
│         │ launch kernel              │ pull CPU hits                │
│         ▼                            ▼                             │
│  ┌──────────────────┐         ┌──────────────────────┐             │
│  │ GPU (RTX 30/40)  │         │ multiprocessing Queue│             │
│  │ ─── vanity_kernel│         └──────────────────────┘             │
│  │   17.6M chains in parallel                       ▲               │
│  │   16 points per chain │                           │              │
│  │   Montgomery inverse  │     ┌──────────────────┐ │               │
│  └──────────────────┘         │ CPU worker × N   │─┘               │
│         ▲                     │ (coincurve)      │                 │
│         │                     └──────────────────┘                 │
│         │ status parameters                                        │
│  ┌──────────────────────────────────────────────┐                  │
│  │ Status-line refresh thread (daemon)           │                  │
│  │ Reads global variables every 0.5s and prints  │                  │
│  │ a \r single-line refresh                      │                  │
│  └──────────────────────────────────────────────┘                  │
└────────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Startup stage** (one-time):
   - Detect GPU, compile CUDA kernel (adaptively choose the fastest between M=8 / M=16)
   - GPU self-test (run 1 step with a single thread, compare with CPU to confirm correctness)
   - Use all CPU cores in parallel to generate ~220,000 random starting points (`P_i = k_i × G`)
   - Upload starting points and pattern parameters to GPU
   - Do one warmup using the real search mode, adaptively adjust `STEPS_PER_LAUNCH` to ~1.5 seconds/launch

2. **Main loop stage** (continuous running):
   - Dual stream ping-pong: when stream A is synchronously waiting, stream B is running on the GPU
   - Each kernel advances ~570 million point additions and checks all hits at the same time
   - Hits are asynchronously copied back to the main process through pinned host memory D2H
   - CPU workers also send hits through multiprocessing Queue
   - The main process writes all hits to the file
   - When the accumulated hits reach 50, it pauses and asks the user / in the release version, to avoid beginner users entering looser conditions that cause a large number of hits, the accumulated condition was changed to 10. For the modification method, see: Advanced Configuration - Adjusting the hit threshold

3. **Exit stage**:
   - First Ctrl+C → signal handler sets flag → main loop detects it and breaks
   - finally: kill all CPU workers, sync GPU streams that are in flight, print session summary
   - Second Ctrl+C → directly raise KeyboardInterrupt → use default terminate

---

## Purpose of Each File

### `tron_vanity_gpu.py` (~1150 lines)

**Main program entry point**. It mainly does four things:

1. **CLI interaction**: prompt the user to choose a mode (prefix/suffix / tail repetition), input parameters, and validate legality
2. **GPU scheduling**: compile kernel, adaptively choose M=8/16 and STEPS_PER_LAUNCH, dual stream ping-pong
3. **Multiprocessing coordination**: spawn N CPU workers, reserve about 10% CPU by default, use Queue to collect hits, use Event to control pause/stop
4. **Status display**: an independent daemon thread refreshes the status line every 0.5 seconds (`\r` single-line overwrite, no ANSI dependency)

Included functions:
- `cpu_priv_to_address(priv_int)` — CPU reference implementation, used for GPU hit verification
- `validate_pattern(prefix, suffix)` — input validation (Base58 character set, length limits, minimum effective difficulty)
- `estimate_probability(...)` — estimate hit probability
- `input_with_timeout(prompt, timeout)` — cross-platform timed input (Windows uses msvcrt, Linux uses select)
- `load_kernel(arch, points_per_thread)` — compile CUDA kernel with NVRTC
- `run_search(prefix, suffix, repeat_tail, output_path)` — main search loop

### `kernels.cu` (~830 lines)

**All CUDA kernels (C/C++)**, compiled at runtime by CuPy's NVRTC. It does not depend on CUDA Toolkit, only `cupy-cuda12x` + `nvidia-cuda-nvrtc-cu12` are required.

Modules:
- **256-bit big integer arithmetic**: `add4 / sub4 / mul_full`, manual carry/borrow
- **secp256k1 field operations**: `f_add / f_sub / f_mul / f_sqr / f_reduce`, using the fast reduction from `2^256 ≡ 2^32 + 977 (mod p)`
- **Modular inverse**: `f_inv` (Fermat's little theorem), `f_inv_batch` (Montgomery batch, performance core)
- **Affine point addition**: single-point version `point_add`, batch version directly inlined in the main kernel
- **Keccak-256**: `keccak_round + keccak_f + keccak256_64`, 24 rounds of Keccak-f[1600]
- **SHA-256**: `sha256_21 + sha256_32`, standard 64-round compression function
- **Base58**: `base58_encode_25` (full 34 characters) + `base58_tail` (only calculate the last K characters, for early exit)
- **Main kernel**: `vanity_kernel`, each GPU thread processes `POINTS_PER_THREAD` independent point chains in parallel

### `cpu_worker.py` (~140 lines)

**CPU worker process**. It does not import cupy, so spawned child processes will not repeatedly load GPU libraries, and it also avoids competing with the main process for the CUDA context.

Functions:
- `gen_startpoints_batch(n)` — called in parallel by the main process with Pool.map to generate n random starting points
- `cpu_worker(prefix, suffix, repeat_tail, match_q, stat_q, stop_evt, pause_evt)` — worker main loop, runs pure Python coincurve search, puts hits into match_q
- `_priv_to_address(priv_bytes)` — coincurve + pycryptodome implementation, equivalent to GPU

### `verify_kernel.py` (~150 lines)

**GPU kernel correctness verification**. It runs `THREADS × M × STEPS × 2` addresses (4096 when M=8, 8192 when M=16), and compares byte by byte with the CPU reference implementation. State continuation across launches is also verified.

```bash
python verify_kernel.py
```

It must be run once after every kernel change to ensure correctness has not been broken.

### `requirements.txt`

Python dependency list, with upper version bounds added to avoid sudden breakage from upstream breaking changes.

### `命中地址/`

Automatically created when the program first starts. Each run creates a `matches_YYYYMMDD_HHMMSS.txt` file. Every hit address is immediately appended to the file for the current run (even if the program crashes, the hit has already been written to disk).

File content format:

```
======================================================================
Time           : 2026-05-24 11:36:01
Match mode     : tail repetition ≥ 5 characters
Source         : GPU
Address        : TXeuGYYbuKWgDYc24nd5h9X13zETfttttt
Private key (hex): cf16830a81ca6f0487ff492bcef7d7357ceb0c299812ed971c65b71ca13a2769
======================================================================
```

---

## Implementation of Key Modules

### Montgomery Batch Inversion (Core Performance Optimization)

Naive ECC point addition requires 1 modular inverse per point (Fermat's little theorem: 256 squaring steps + ~128 multiplications). Modular inverse is the slowest part of the whole flow.

**Montgomery's trick**: modular inverses for M points require **only 1 big modular inverse** + 3M-3 multiplications:

```
Let t_k = a_0 × a_1 × ... × a_k

1. Forward: use M-1 multiplications to compute t_0..t_{M-1}
2. Compute t_{M-1}^{-1}  (the only big modular inverse, 256 steps)
3. Backward substitution:
     a_k^{-1} = t_{M-1}^{-1} × t_{k-1}
     t_{k-1}^{-1} = t_{M-1}^{-1} × a_k
```

Amortized cost per point:
- M=1 (naive): 256 mul/point
- M=8: 35 mul/point (-86%)
- M=16: 19 mul/point (-93%)

**Why not make M infinitely large?** Register pressure is ~ O(M×8). On RTX 3060, M=32 spills into LMEM and becomes slower instead. When the program starts, it warmup-tests M=8 / M=16 and automatically selects the fastest.

<p>
  <a href="./docs/m-parameter-optimization.en.md">
    <img alt="M Parameter and High-End GPU Tuning Details" src="https://img.shields.io/badge/Details-M%20Parameter%20%26%20High--End%20GPU%20Tuning-0969da?style=for-the-badge">
  </a>
</p>

### Base58 Tail Early Exit

Base58 long division **outputs tail characters from the low bits** — every time `N = N / 58`, it gives one trailing character. If you want to know the last K characters of the address, you only need to run K long-division steps, not the full 34 steps.

In mode 2 (tail repetition) / mode 1 suffix mode, 90%+ of candidates can be rejected during the tail check, skipping the full Base58 + later prefix comparison, giving an overall speedup of 25-35%.

**Mode 1 pure prefix mode cannot early-exit** — the mathematical properties of the head characters of Base58 determine that all 25 bytes must participate before they can be determined. This is a mathematical limitation, with no solution.

### Dual Stream Ping-Pong

Without ping-pong, when the main thread waits for the GPU to finish with `stream.synchronize()`, the GPU is idle, and utilization is ~50%.

Dual stream:
- When the main thread syncs stream A, **stream B is running the kernel on the GPU**
- The moment sync returns, the main thread immediately launches the next one to stream A
- Then it goes to sync stream B
- From then on, the GPU Compute engine stays at **continuous 100%**, with no more launch gap

Asynchronous D2H copy uses pinned host memory + `cudaMemcpyAsync`, goes through the Copy engine, and does not occupy the Compute engine.

### Status Line Single-Line Refresh (No ANSI Dependency)

Many terminals (old PowerShell versions, Windows cmd in some scenarios) do not parse `\x1b[2K` ANSI escape. It uses `\r + space padding + \r`:

```
\r → return to line start
new status line
spaces pad to previous width (overwrite old tail)
\r → return to line start again, ready for next overwrite
```

It also adaptively chooses 4 levels of status-line templates according to terminal width (verbose+hw → verbose → main+hw → main), avoiding line wrapping when the line exceeds the width.

### Ctrl+C Handling

On Windows, `stream.synchronize()` blocks SIGINT until it returns. Directly `raise KeyboardInterrupt` at an abnormal moment may cause `multiprocessing.Event.set()` to get stuck in semaphore IPC, turning child processes into orphans.

Solution:
1. The signal handler **only sets a flag and does no IO** (avoids reentrancy)
2. The main loop checks the flag every round, breaks, and enters finally
3. finally **directly calls p.kill()** (TerminateProcess, synchronously effective immediately), without trying to stop "gracefully"
4. The second Ctrl+C restores the default handler and uses default terminate

---

## Tech Stack and Dependencies

### Core Technologies

- **CUDA / NVRTC**: NVIDIA GPU general-purpose computing, runtime compilation of C++ source code to PTX
- **CuPy 14.x**: Python ↔ CUDA bridge, provides RawModule (source → kernel), Stream, Event, pinned memory, etc.
- **Python multiprocessing**: spawn mode starts child processes for CPU search
- **coincurve**: Python binding for libsecp256k1, CPU-side secp256k1 calculation
- **pycryptodome**: Keccak-256
- **base58**: Base58 encoding (CPU reference implementation)
- **pynvml**: NVIDIA management library, reads GPU utilization/power/temperature
- **psutil**: CPU utilization, worker process priority control

### Full Dependency List (`requirements.txt`)

```
numpy>=1.26,<3.0
cupy-cuda12x>=14.0,<16.0          # includes NVRTC, no need to install CUDA Toolkit
nvidia-cuda-runtime-cu12             # CUDA headers required during NVRTC compilation
nvidia-cuda-nvrtc-cu12
nvidia-cuda-cccl-cu12
coincurve>=19.0,<22.0                # CPU-side starting point calculation + hit verification
pycryptodome>=3.18.0,<4.0
base58>=2.1.0,<3.0
```

Optional dependencies (it can still run without them; the status line just has less hardware usage display):
- `pynvml` — GPU utilization/power/temperature
- `psutil` — CPU utilization + worker process priority

---

## Environment Requirements

### Hardware

| Item | Minimum | Recommended |
|---|---|---|
| GPU | NVIDIA RTX 20 series (compute capability ≥ 7.0) | RTX 30/40 series |
| GPU VRAM | 2 GB | 4 GB or more |
| CPU | Any x64, 4 cores | 8 cores or more |
| RAM | 4 GB | 16 GB |
| Disk space | 1 GB | 5 GB (including CUDA libraries) |

> **AMD / Intel GPUs are not supported** (depends on CUDA, not OpenCL).
> **GTX 10 series and older are not supported** (compute capability < 7.0, incompatible with NVRTC compilation options).

### Software

- **Windows 10 / 11** or **Linux** (Ubuntu 20.04+)
- **Python 3.10 - 3.14** (3.12 recommended)
- **NVIDIA GPU driver** 535+ (Windows) / 535+ (Linux)
- **No need to install CUDA Toolkit** (the cupy-cuda12x package includes NVRTC)

---

## Installation Steps (Beginner Version)

### Windows

#### 1. Install Python

Go to [python.org](https://www.python.org/downloads/) and download the Python 3.12 installer. During installation, **check "Add Python to PATH"**.

Open PowerShell and verify:
```powershell
python --version
```
You should see output like `Python 3.12.x`.

#### 2. Check the GPU driver

```powershell
nvidia-smi
```

You should see a table listing the GPU name, power, temperature, etc. If it says "nvidia-smi is not recognized as an internal or external command", go to the [NVIDIA official website](https://www.nvidia.com/Download/index.aspx) to download and install the latest driver.

Driver version requirement is 535 or above (the "Driver Version: XXX.XX" on the first output line).

#### 3. Download this project

Put all project files (`tron_vanity_gpu.py` / `kernels.cu` / `cpu_worker.py` / `verify_kernel.py` / `requirements.txt`) into one directory, for example `C:\tron_vanity\`.

#### 4. Install Python dependencies

Open PowerShell and switch to the project directory:
```powershell
cd C:\tron_vanity
pip install -r requirements.txt
```

The first installation will be relatively slow (cupy + nvidia-cuda-* add up to 1-2 GB). Please wait patiently.

#### 5. Verify GPU kernel correctness (optional but recommended)

```powershell
python verify_kernel.py
```

You should see:
```
✓ M=8 and M=16 both passed verification
  Montgomery batch inversion + GPU persistent state + all crypto primitives OK
```

If it fails, see the [Common Errors](#common-errors-and-solutions) section.

#### 6. Run the main program

```powershell
python tron_vanity_gpu.py
```

Follow the prompts to choose a mode and input parameters, then the program will start running.

### Linux (Ubuntu)

```bash
# 1. Install Python and pip
sudo apt update
sudo apt install python3 python3-pip git

# 2. Check driver
nvidia-smi

# 3. Download project, install dependencies
cd ~/tron_vanity
pip3 install -r requirements.txt

# 4. Verify and run
python3 verify_kernel.py
python3 tron_vanity_gpu.py
```

---

## Usage Instructions

### Start

```bash
python tron_vanity_gpu.py
```

### Choose Mode

```
======================================================================
  TRON Vanity Address Generator (CUDA + CPU Full-Speed Version)
======================================================================

  Mode 1: Custom mode — specify prefix and/or suffix
  Mode 2: Vanity mode — specify the number of repeated tail characters

Please choose mode [1/2]:
```

**Mode 1**: Input the desired beginning and ending characters of the address. For example, if you want `TXYZ...888`, the prefix is `TXYZ` and the suffix is `888`.
- The prefix must start with `T` (TRON address characteristic)
- It cannot contain `0`, `O`, `I`, `l` (these 4 characters are not in the Base58 character set)
- At least 4 effective Base58 characters are required; the leading `T` in the prefix does not count toward effective difficulty. For example, `TXYZ...888` is OK, but `TXYZ` alone is not enough.

**Mode 2**: Input N, find addresses whose last N characters are the same character (for example, N=6 finds addresses whose last 6 characters are all the same, a "豹子号"). To avoid massive hits slowing down the program, N must be at least 5.

### During Program Execution

After startup, you will see:
```
======================================================================
  TRON Vanity Generator — GPU + CPU Full-Speed Version
======================================================================
  GPU device      : NVIDIA GeForce RTX 3060
  Compute ability : 8.6   SM count: 28
  GPU threads     : 7168 (= 112 block × 64)
  CPU workers     : 10 (local 12 cores, reserve 2 cores about 10%)
  Search mode     : tail repetition ≥ 6 characters
  Theoretical prob: average 656 million addresses for 1 result
  Output file     : C:\...\命中地址\matches_20260524_113000.txt
  Press Ctrl+C to exit
======================================================================

Adaptively selecting Montgomery batch inversion parameter M (8 / 16)...
  M=8: 65M/sec
  M=16: 80M/sec
→ Use M=16 (amortized 19 mul/point, fastest ECC part)

GPU self-test passed, time 0.01s. GPU is indeed executing the kernel ✓
Generated 229376 GPU starting points (using 12 CPU cores in parallel)...
Starting points ready, time 1.8s (130000 pts/s)
Adaptive STEPS_PER_LAUNCH = 1000 (expected ~1.5s/launch, ~115M addresses/launch)

Start searching...
Ran 13.0 sec | speed 88.60M/sec (GPU88.57M+CPU35K) | total 1.16B | need 7.4 sec | hits 1 | GPU100% 165W 64°C CPU98%
```

### Hit Handling

After each batch of matching addresses is found, it will be appended in batches to the hit file. After accumulating **50** entries, it pauses and shows a three-option menu. If you choose to change the generation condition, the program first synchronizes and discards GPU results from the old condition that are still in flight, then starts the new condition, avoiding mixed writing of old and new conditions. The release version uses 10 entries.

```
======================================================================
  *** Accumulated 50 matching addresses found ***
======================================================================

  Please choose an action:
    1) Continue generating
    2) Stop generating
    3) Change generation conditions (re-enter mode)

  Input option [1/2/3] (auto-continue after 30 seconds without response):
```

### Exit

Press **Ctrl+C** (on Windows, Ctrl+Break also works) to exit cleanly. All child processes are automatically cleaned up.

---

## Advanced Configuration (Advanced Users)

### Adjust Hit Threshold

Around line ~904 of `tron_vanity_gpu.py`:
```python
HIT_PAUSE_THRESHOLD = 50
```
Change it to 100/200/1000, all are fine. It will pause and ask the user only after accumulating this number.

### Force M=8 (Old GPU or Insufficient Registers)

Around line ~318 of `tron_vanity_gpu.py`:
```python
M_CANDIDATES = [16, 8]
```
Change it to `[8]` to skip M=16 testing.

### Adjust STEPS_PER_LAUNCH Target Time

Around line ~595:
```python
target_sec = 1.5
```
Single launch target duration. Make it smaller for faster status-line refresh, make it larger to reduce launch overhead. 1-3 seconds is a reasonable range.

### Adjust CPU Worker Count

Around line ~335:
```python
cpu_reserved = max(1, (cpu_count + 9) // 10)
n_cpu_workers = max(1, cpu_count - cpu_reserved)
```
By default, it rounds up and reserves about 10% of logical cores for the system, desktop, and GPU launcher. If you want to use the computer while running, you can make `cpu_reserved` larger; if you want to squeeze CPU search, you can make it smaller.

### Lock GPU Clock (Squeeze Extreme Performance)

Run with administrator privileges before starting:
```powershell
nvidia-smi -pm 1                  # enable persistent mode
nvidia-smi -lgc 1830              # lock graphics clock (RTX 3060 upper limit)
```
When exiting:
```powershell
nvidia-smi -rgc                   # unlock
```

---

## Estimated Running Results on Different Computer Configurations

Below are the theoretical throughput and hit time for **last 6 characters repeated mode** on different GPUs:

| GPU | VRAM | CC | Estimated throughput | Find 1 address with 6 repeated characters |
|---|---|---|---|---|
| GTX 1660 Ti | 6 GB | 7.5 | ~30M/s | ~22 sec |
| RTX 2060 | 6 GB | 7.5 | ~45M/s | ~15 sec |
| RTX 3050 | 8 GB | 8.6 | ~50M/s | ~13 sec |
| **RTX 3060** | **12 GB** | **8.6** | **~100M/s** | **~7 sec** |
| RTX 3070 | 8 GB | 8.6 | ~180M/s | ~4 sec |
| RTX 3080 | 10 GB | 8.6 | ~250M/s | ~2.6 sec |
| RTX 3090 | 24 GB | 8.6 | ~300M/s | ~2.2 sec |
| RTX 4070 | 12 GB | 8.9 | ~400M/s | ~1.6 sec |
| RTX 4080 | 16 GB | 8.9 | ~550M/s | ~1.2 sec |
| RTX 4090 | 24 GB | 8.9 | ~750M/s | **~0.9 sec** |
| RTX 5090 | 32 GB | 10.0 | ~1.5G/s (estimated) | ~0.5 sec |
| A100 | 40-80 GB | 8.0 | ~600M/s | ~1.1 sec |
| H100 | 80 GB | 9.0 | ~1.2G/s | ~0.6 sec |

> Values are for reference only. Actual results depend on driver version, temperature wall, power limits, etc. The program will warm up and adapt to the actual throughput when starting.

### Low-Configuration Scenarios

**RTX 2060 6GB laptop** (mobile version with reduced power):
- Estimated ~30M/s, finding 6 repeated characters ~22 seconds
- May downclock due to laptop cooling issues, actual speed may be lower than estimated
- Recommended: plug in power, use cooling pad, close other programs occupying GPU

**GTX 1660 / 1660 Ti** (compute capability 7.5):
- The minimum supported cards. The program can run, but M=16 may fall back to M=8 due to insufficient registers
- Estimated 30M/s, finding 6 repeated characters ~22 seconds, finding 8 repeated characters ~6 hours

**GTX 1080 / 1080 Ti / Titan X (Pascal)** (compute capability 6.x):
- **Not supported**! The program will exit directly at startup and prompt "GPU compute capability 6.x too low"

### High-Configuration Scenarios

**RTX 4090** (16384 CUDA cores):
- Expected actual throughput ~750M/s
- 6 repeated characters almost appear instantly, 8 repeated characters average 1 every 7 minutes
- Recommended to use this kind of GPU for long prefixes (5-6 characters), such as `TX888...`

**Multi-GPU system**:
- **The current code only uses GPU 0**. To use multiple GPUs, you need to modify `cp.cuda.Device(idx)` and start the main program multiple times (different output files)
- Simple solution: run multiple processes, each using a different environment variable `CUDA_VISIBLE_DEVICES=0` / `=1`

---

## Common Errors and Solutions

### `Missing dependency: ...`

```
Missing dependency: No module named 'cupy'
Please run: pip install cupy-cuda12x numpy coincurve pycryptodome base58
```

Install according to the prompt. If the network in China is slow, use the Tsinghua mirror:
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### `nvidia-smi is not recognized as an internal or external command`

NVIDIA driver is not installed. Go to [NVIDIA Driver](https://www.nvidia.com/Download/index.aspx) to download the latest version.

### `CUDA initialization failed: cudaErrorInsufficientDriver`

The driver version is too old. Upgrade NVIDIA driver to 535 or above.

### `GPU compute capability X.Y too low`

The GPU is too old (compute capability < 7.0). An RTX 20 series or newer card is required.

### `GPU out of memory: out of memory`

VRAM is occupied by other programs. Close:
- Browser hardware acceleration (turn it off in Chrome / Edge settings)
- Games / video conferencing software
- Other CUDA tasks currently running (`nvidia-smi` shows usage)

If it still does not work, reduce `BLOCKS_PER_SM` or `THREADS_PER_BLOCK` (inside `tron_vanity_gpu.py`).

### `Failed to find CUDA headers` or NVRTC compilation error

`nvidia-cuda-nvrtc-cu12` or `nvidia-cuda-runtime-cu12` is not installed. Reinstall:
```bash
pip install --force-reinstall nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12
```

### `UserWarning: CUDA path could not be detected`

Harmless warning, because the full CUDA Toolkit is not installed. The program can still run normally, and it has already been filtered in the code.

### Waiting 1-15 seconds with no response on first startup

NVRTC is compiling the kernel. It will be cached afterwards, and subsequent startups only need a few hundred milliseconds.

### Task Manager shows GPU usage 1%

Windows Task Manager looks at the "3D" engine by default, while CUDA uses "Compute_0". Switch Task Manager → Performance → GPU → select "**Compute_0**" from any engine drop-down menu to see the correct 100%.

The data seen with `nvidia-smi -l 1` is real and will not deceive you.

### Ctrl+C gets stuck and does not exit

If you still encounter this problem after the current commit, wait 5 seconds and press it again — the second Ctrl+C will force terminate.

### Status line stacks into multiple lines

The terminal width is too narrow. Widen the terminal window to at least 80 columns.

---

## Memory and Crash Scenario Analysis

### Memory Usage

**Main process**:
- Python + cupy + numpy + coincurve: ~300 MB
- Main program variables + starting point list: ~50 MB (under RTX 3060 configuration, M=16, 220k starting points)
- pinned host memory (dual stream × 224KB matches buffer): < 1 MB
- Total: **~350 MB**

**Each CPU worker**:
- Python + coincurve + pycryptodome: ~80 MB
- 12-logical-core machine defaults to 10 workers: **~800 MB**

**GPU VRAM**:
- Starting point tensor (cur_x, cur_y, dual stream): 220k × 64 bytes × 2 = ~28 MB
- matches buffer: 224 KB × 2 = 448 KB
- kernel code + constant memory: < 1 MB
- cupy internal buffer: ~100 MB
- Total: **~150 MB**

Total RAM usage **~1.2 GB**, VRAM **~150 MB**.

### Possible Crash Scenarios

| Scenario | Trigger condition | Program behavior |
|---|---|---|
| GPU VRAM insufficient | Competing with other GPU programs for resources | cupy.OOMError at startup, exits after friendly prompt |
| GPU driver crash | Hardware unstable after running for several hours | TDR (timeout detection recovery), CUDA context lost, main program stuck, needs Ctrl+C restart |
| System RAM insufficient | Running many other programs at the same time | CPU worker startup fails, starting point generation stuck |
| Disk space insufficient | Large number of hits fills disk | save_match exception, degrades to stderr output, does not affect search |
| System time goes backward | User manually changes system time | Status line like `ran -1.0 sec`, but does not crash |
| User input is messy | validate_pattern has fallback, no crash | Prompts to re-enter |
| coincurve outputs None | Extremely small probability private key out of range | catch ValueError, skip this priv and continue |
| pp_dev.set while in-flight | Race when changing conditions | stream.sync has been added, no problem |

### Known Non-Crash Scenarios

- Ctrl+C / Ctrl+Break (Windows) / Ctrl+\\ (Linux) → graceful exit
- Closing the terminal window → main process is killed, daemon child process also dies (on Windows, spawned processes may become orphans and need manual cleanup in Task Manager)
- Too many hit addresses (single launch > 4096) → atomicAdd still counts, but after the buffer is full, extra ones are discarded (`if (idx < max_matches)`)

---

## Concurrency and Multithreading Risks

### Identified Risk Points

| Risk | Current status | Assessment |
|---|---|---|
| Status-line thread reads global variables from main thread | `gpu_total / cpu_total / cycle_start` are int/float, single-variable assignment is atomic (GIL guarantee) | ✅ Safe, worst case one inaccurate status frame |
| signal handler sets flag, main loop reads | dict read/write, GIL atomic | ✅ Safe |
| multiprocessing Queue across processes | Uses OS locks internally | ✅ Safe |
| CPU worker SIG_IGN | Uses SIG_IGN to block on both Windows + Linux | ✅ Safe |
| Restart workers when changing conditions | First `stop_evt.set()` + all `p.kill()`, then start new ones | ✅ Safe, two sets of workers will not run at the same time |
| Between GPU streams | Dual streams use their own cur_x/cur_y/matches buffers | ✅ Physically isolated |
| pinned memory across streams | Each stream has an independent buffer | ✅ Isolated |
| pp_dev (pattern) shared | sync added when changing conditions, read-only after start | ✅ Safe |

### Potential Risks (Low Probability)

- **Old references of multiprocessing.Event**: When changing conditions, a new stop_evt is created, and the old one depends on GC. Python GC may occasionally not release IPC resources in time on Windows, with an extremely small probability of handle leaks. Not reproduced after 100+ tests.
- **NVML handle**: atexit registers nvmlShutdown, but atexit behavior after multiprocessing child-process fork may be strange. In actual tests, there is no problem with spawn mode.
- **Windows console close event**: If the user directly clicks ✗ to close the cmd window, what is received is CTRL_CLOSE_EVENT, not SIGINT. Currently SetConsoleCtrlHandler is not installed. At this time the main process is killed immediately, daemon child processes die instantly, but grandchild processes (workers spawned by Pool) may become orphans. Clean them manually with Task Manager.

---

## Security Tips

> ⚠ **Private key security is the user's responsibility**.

- **Do not run on an untrusted computer**: malware / remote control / screen recording software may steal private keys
- **Do not upload hit files to any cloud**: GitHub / cloud drives / email attachments / screenshots / group chats
- **Destroy source files immediately after use**: after importing the hit address into a hardware wallet, securely delete with `sdelete` (Windows) / `shred` (Linux)
- **Test generated addresses with a small amount first**: transfer 1 USDT in, import into wallet with mnemonic/private key and confirm it can be seen, then transfer a large amount
- **Offline generation is safer**: unplug network cable to generate → import into hardware wallet → destroy original file → reconnect network

The private key controls all assets in the wallet. Once leaked, it **can never be recovered**.

---

## Project History

The optimization path of this project from 17M/sec → 100M/sec (RTX 3060):

| Stage | Throughput | Key optimization |
|---|---|---|
| Initial version | 17.6M/s | Basic CUDA implementation, 1 modular inverse per point |
| Dual stream | 17.6M/s | GPU utilization 50% → 100% |
| Montgomery M=8 | 62M/s | 8 points share 1 modular inverse (+250%) |
| Montgomery M=16 | 80M/s | 16 points share (+25%) |
| Base58 tail early exit | 100M/s | Mode 2/suffix mode skips full base58 (+25%) |

The remaining bottleneck is in Keccak/SHA (~50% time). At the algorithm level, it is already close to the compiler's optimal solution.

---

## License

For personal learning and research only. No warranty is provided, use at your own risk.
Technical exchange: TG [@jiutong9999](https://t.me/jiutong9999)
