import os
import sys
import time
import hashlib
import threading
import multiprocessing as mp
import signal
import atexit
from datetime import datetime
from queue import Empty
if getattr(sys, "frozen", False):
    _base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(sys.executable)))
    for _rel in ("", os.path.join("nvidia", "cuda_runtime", "bin"), os.path.join("nvidia", "cuda_nvrtc", "bin")):
        _dll_dir = os.path.join(_base, _rel)
        if os.path.isdir(_dll_dir):
            try:
                os.add_dll_directory(_dll_dir)
            except Exception:
                pass
            os.environ["PATH"] = _dll_dir + os.pathsep + os.environ.get("PATH", "")

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.SetConsoleOutputCP(65001)
        k32.SetConsoleCP(65001)
    except Exception:
        pass
try:
    import warnings
    warnings.filterwarnings("ignore", message=".*CUDA path could not be detected.*")
    import cupy as cp
    import numpy as np
except ImportError as e:
    sys.stderr.write(
        "缺少依赖: {}\n请运行: pip install cupy-cuda12x numpy coincurve pycryptodome base58\n".format(e)
    )
    sys.exit(1)
try:
    import coincurve
    from Crypto.Hash import keccak as keccak_lib
    import base58
except ImportError as e:
    sys.stderr.write("缺少依赖: {}\n请运行: pip install coincurve pycryptodome base58\n".format(e))
    sys.exit(1)
try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    _HAS_NVML = True
    atexit.register(lambda: pynvml.nvmlShutdown())
except Exception:
    _NVML_HANDLE = None
    _HAS_NVML = False
try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False
def _gpu_stats():
    if not _HAS_NVML or _NVML_HANDLE is None:
        return None
    try:
        u = pynvml.nvmlDeviceGetUtilizationRates(_NVML_HANDLE).gpu
        p = pynvml.nvmlDeviceGetPowerUsage(_NVML_HANDLE) / 1000.0
        t = pynvml.nvmlDeviceGetTemperature(_NVML_HANDLE, pynvml.NVML_TEMPERATURE_GPU)
        m = pynvml.nvmlDeviceGetMemoryInfo(_NVML_HANDLE).used / (1024 * 1024)
        return (u, p, t, m)
    except Exception:
        return None
def _cpu_percent():
    if not _HAS_PSUTIL:
        return None
    try:
        return psutil.cpu_percent(interval=None)
    except Exception:
        return None
SECP256K1_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
BASE58_SET = set(BASE58_ALPHABET)
ADDRESS_LEN = 34
def cpu_priv_to_address(priv_int: int) -> str:
    pub = coincurve.PublicKey.from_valid_secret(priv_int.to_bytes(32, "big")).format(compressed=False)
    k = keccak_lib.new(digest_bits=256)
    k.update(pub[1:])
    payload = b"\x41" + k.digest()[-20:]
    checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    return base58.b58encode(payload + checksum).decode()
def validate_pattern(prefix: str, suffix: str):
    errs = []
    min_effective_chars = 4
    if prefix:
        if prefix[0] != "T":
            errs.append("前缀必须以 'T' 开头 (所有 TRON 地址都以 T 开头)")
        bad = sorted({c for c in prefix if c not in BASE58_SET})
        if bad:
            errs.append("前缀含非 Base58 字符: {} (Base58 不含 0、O、I、l)".format(bad))
        if len(prefix) > ADDRESS_LEN:
            errs.append("前缀过长 (地址总长 34)")
    if suffix:
        bad = sorted({c for c in suffix if c not in BASE58_SET})
        if bad:
            errs.append("后缀含非 Base58 字符: {}".format(bad))
        if len(suffix) > ADDRESS_LEN:
            errs.append("后缀过长 (地址总长 34)")
    if prefix and suffix and len(prefix) + len(suffix) > ADDRESS_LEN:
        errs.append("前缀+后缀长度之和超过 34")
    effective_chars = max(0, len(prefix) - 1) + len(suffix)
    if effective_chars < min_effective_chars:
        errs.append(
            "条件太宽泛, 至少需要 4 个有效 Base58 字符 "
            "(前缀开头的 T 不计入; 例如 TX888 或后缀 8888)"
        )
    return errs
def estimate_probability(prefix: str, suffix: str, repeat_tail: int) -> float:
    p = 1.0
    if prefix:
        extra = len(prefix) - 1
        if extra > 0:
            p *= (1.0 / 58) ** extra
    if suffix:
        p *= (1.0 / 58) ** len(suffix)
    if repeat_tail > 0:
        p *= 58.0 * (1.0 / 58) ** repeat_tail
    return p
def input_with_timeout(prompt: str, timeout: float):
    sys.stdout.write(prompt)
    sys.stdout.flush()
    if sys.platform == "win32":
        import msvcrt
        start = time.time()
        buf = []
        while time.time() - start < timeout:
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ("\r", "\n"):
                    sys.stdout.write("\n")
                    return "".join(buf)
                if ch == "\x03":
                    raise KeyboardInterrupt
                if ch == "\b":
                    if buf:
                        buf.pop()
                        sys.stdout.write("\b \b")
                        sys.stdout.flush()
                else:
                    buf.append(ch)
                    sys.stdout.write(ch)
                    sys.stdout.flush()
            time.sleep(0.03)
        sys.stdout.write("\n")
        return None
    import select
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if ready:
        line = sys.stdin.readline()
        if not line:
            return None
        return line.rstrip("\n")
    sys.stdout.write("\n")
    return None
def fmt_time(seconds):
    if seconds is None or seconds != seconds or seconds == float("inf") or seconds < 0:
        return "?"
    if seconds < 1:    return "{:.0f}毫秒".format(seconds * 1000)
    if seconds < 60:   return "{:.1f}秒".format(seconds)
    if seconds < 3600: return "{:.1f}分".format(seconds / 60)
    if seconds < 86400:return "{:.1f}小时".format(seconds / 3600)
    if seconds < 86400 * 365: return "{:.1f}天".format(seconds / 86400)
    return "{:.1f}年".format(seconds / 86400 / 365)
def fmt_num(n):
    if n < 1e3:
        return "{:.0f}".format(n)
    if n < 1e4:
        return "{:.0f}".format(n)
    if n < 1e8:
        v = n / 1e4
        if v < 10:    return "{:.2f}万".format(v)
        if v < 100:   return "{:.1f}万".format(v)
        return "{:.0f}万".format(v)
    if n < 1e12:
        v = n / 1e8
        if v < 10:    return "{:.2f}亿".format(v)
        if v < 100:   return "{:.1f}亿".format(v)
        return "{:.0f}亿".format(v)
    v = n / 1e12
    if v < 10:    return "{:.2f}万亿".format(v)
    if v < 100:   return "{:.1f}万亿".format(v)
    return "{:.0f}万亿".format(v)
APP_BASE_DIR = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__))
RESOURCE_BASE_DIR = getattr(sys, "_MEIPASS", APP_BASE_DIR)
KERNEL_PATH = os.path.join(RESOURCE_BASE_DIR, "kernels.cu")
PATTERN_DTYPE = np.dtype([
    ("prefix_len", np.int32),
    ("suffix_len", np.int32),
    ("repeat_n",   np.int32),
    ("prefix",     np.uint8, 40),
    ("suffix",     np.uint8, 40),
], align=True)
MATCH_DTYPE = np.dtype([
    ("thread_id", np.uint32),
    ("_pad",      np.uint32),
    ("step",      np.uint64),
    ("address",   np.uint8, 34),
    ("_pad2",     np.uint8, 6),
], align=True)
def load_kernel(arch=None, points_per_thread=8):
    with open(KERNEL_PATH, "r", encoding="utf-8") as f:
        src = f.read()
    options = ["-std=c++14", "--use_fast_math",
               "-DPOINTS_PER_THREAD={}".format(points_per_thread)]
    if arch:
        options.append("-arch=" + arch)
    module = cp.RawModule(
        code=src,
        options=tuple(options),
        backend="nvrtc",
        name_expressions=("vanity_kernel",),
    )
    return module.get_function("vanity_kernel")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpu_worker import cpu_worker, gen_startpoints_batch
def run_search(prefix: str, suffix: str, repeat_tail: int, output_path: str):
    try:
        n_dev = cp.cuda.runtime.getDeviceCount()
    except Exception as e:
        sys.stderr.write(
            "✗ CUDA 初始化失败: {}\n"
            "  请检查:\n"
            "    1) NVIDIA 显卡驱动是否已安装 (终端跑 nvidia-smi 应能看到设备)\n"
            "    2) 显卡是否支持 CUDA (compute capability ≥ 7.0, 即 RTX 20 系及以上)\n"
            "    3) Python 是否能找到 CUDA 库 (重启电脑或重装 cupy-cuda12x)\n".format(e))
        sys.exit(1)
    if n_dev == 0:
        sys.stderr.write("✗ 没检测到 NVIDIA GPU. 请确认驱动已装, nvidia-smi 能列出设备.\n")
        sys.exit(1)
    try:
        props = cp.cuda.runtime.getDeviceProperties(0)
    except Exception as e:
        sys.stderr.write("✗ 读取 GPU 0 信息失败: {}\n".format(e))
        sys.exit(1)
    gpu_name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
    n_sms = props["multiProcessorCount"]
    cc_major = props["major"]
    cc_minor = props["minor"]
    if cc_major < 7:
        sys.stderr.write(
            "✗ GPU 计算能力 {}.{} 太低 (需要 ≥ 7.0, 即 RTX 20 系或更新).\n"
            "  当前 GPU: {}\n".format(cc_major, cc_minor, gpu_name))
        sys.exit(1)
    M_CANDIDATES = [16, 8]
    POINTS_PER_THREAD = None
    THREADS_PER_BLOCK = 64
    BLOCKS_PER_SM = 4
    n_blocks = n_sms * BLOCKS_PER_SM
    n_threads = n_blocks * THREADS_PER_BLOCK
    STEPS_PER_LAUNCH = 128
    MAX_MATCHES = 4096
    MATCH_QUEUE_SIZE = 8192
    HIT_PAUSE_THRESHOLD = 50
    N_STREAMS = 2
    cpu_count = mp.cpu_count()
    cpu_reserved = max(1, (cpu_count + 9) // 10)
    n_cpu_workers = max(1, cpu_count - cpu_reserved)
    parts = []
    if prefix:      parts.append("前缀={}".format(prefix))
    if suffix:      parts.append("后缀={}".format(suffix))
    if repeat_tail: parts.append("尾部重复≥{}位".format(repeat_tail))
    pattern_desc = " 且 ".join(parts)
    prob = estimate_probability(prefix, suffix, repeat_tail)
    print()
    print("=" * 70)
    print("  TRON 靓号生成器 — GPU + CPU 全速版")
    print("=" * 70)
    print("  GPU 设备     : {}".format(gpu_name))
    print("  计算能力     : {}.{}   SM 数: {}".format(cc_major, cc_minor, n_sms))
    print("  GPU 并发线程 : {} (= {} block × {})".format(n_threads, n_blocks, THREADS_PER_BLOCK))
    print("  CPU 工作进程 : {} (本机 {} 核, 预留 {} 核约 10%)".format(
        n_cpu_workers, cpu_count, cpu_reserved))
    print("  搜索模式     : {}".format(pattern_desc))
    if prob > 0:
        print("  理论概率     : 平均 {} 个地址出 1 个".format(fmt_num(1 / prob)))
    print("  输出文件     : {}".format(output_path))
    print("  按 Ctrl+C 退出")
    print("=" * 70)
    print()
    print()
    print("自适应选择 Montgomery 批量求逆参数 M (8 / 16)...")
    arch = "sm_{}{}".format(cc_major, cc_minor)
    def _bench_m(m_val):
        try:
            k = load_kernel(arch=arch, points_per_thread=m_val)
        except Exception as e:
            return None, "编译失败: {}".format(e)
        bench_threads = 256
        bench_blocks = 4
        bench_chains = bench_threads * m_val
        pub_bench = coincurve.PublicKey.from_valid_secret((2).to_bytes(32, "big")).format(compressed=False)
        x_bench = int.from_bytes(pub_bench[1:33], "big")
        y_bench = int.from_bytes(pub_bench[33:65], "big")
        sx = np.empty(bench_chains * 4, dtype=np.uint64)
        sy = np.empty(bench_chains * 4, dtype=np.uint64)
        for j in range(4):
            sx[j::4] = (x_bench >> (64 * j)) & 0xFFFFFFFFFFFFFFFF
            sy[j::4] = (y_bench >> (64 * j)) & 0xFFFFFFFFFFFFFFFF
        cur_x = cp.asarray(sx)
        cur_y = cp.asarray(sy)
        pp_b = np.zeros(1, dtype=PATTERN_DTYPE)
        pp_b["repeat_n"] = 30
        pp_b_dev = cp.asarray(pp_b)
        m_d = cp.zeros(64, dtype=MATCH_DTYPE)
        c_d = cp.zeros(1, dtype=cp.uint32)
        STEPS = 16
        try:
            k((bench_blocks,), (THREADS_PER_BLOCK,),
              (cur_x, cur_y, np.int32(STEPS), np.uint64(0),
                pp_b_dev, m_d, c_d, np.uint32(64)))
            cp.cuda.Stream.null.synchronize()
        except Exception as e:
            return None, "运行失败: {}".format(e)
        STEPS = 64
        cp.cuda.runtime.memsetAsync(c_d.data.ptr, 0, 4, 0)
        cp.cuda.Stream.null.synchronize()
        t0 = time.time()
        k((bench_blocks,), (THREADS_PER_BLOCK,),
          (cur_x, cur_y, np.int32(STEPS), np.uint64(0),
           pp_b_dev, m_d, c_d, np.uint32(64)))
        cp.cuda.Stream.null.synchronize()
        elapsed = time.time() - t0
        addrs = bench_threads * STEPS * m_val
        return (addrs / elapsed, k), None
    best_rate, best_M, best_kernel = 0.0, None, None
    for M_try in M_CANDIDATES:
        result, err = _bench_m(M_try)
        if err:
            print("  M={}: {}".format(M_try, err))
            continue
        rate, k_obj = result
        print("  M={}: {:.1f}M/秒".format(M_try, rate / 1e6))
        if rate > best_rate:
            best_rate, best_M, best_kernel = rate, M_try, k_obj
    if best_kernel is None:
        sys.stderr.write(
            "\n✗ CUDA 内核所有 M 值都编译失败, GPU 不兼容.\n"
            "  可能原因:\n"
            "    1) NVRTC 找不到 CUDA 头文件 (重装 nvidia-cuda-nvrtc-cu12 和 nvidia-cuda-runtime-cu12)\n"
            "    2) GPU 驱动版本太老, 升级 NVIDIA 驱动到 535+\n"
            "    3) GPU 计算能力 < 7.0 (RTX 20 系以下不支持)\n")
        sys.exit(1)
    POINTS_PER_THREAD = best_M
    kernel = best_kernel
    BATCH = n_threads * STEPS_PER_LAUNCH * POINTS_PER_THREAD
    print("→ 选用 M={} (摊销 {:.0f} mul/点, ECC 部分最快)".format(
        POINTS_PER_THREAD, 256 / POINTS_PER_THREAD + 3))
    print()
    print("GPU 自检: 单线程执行 1 步 (M={}个点)...".format(POINTS_PER_THREAD))
    self_t0 = time.time()
    try:
        k_tests = list(range(1, POINTS_PER_THREAD + 1))
        sx_test = np.zeros(POINTS_PER_THREAD * 4, dtype=np.uint64)
        sy_test = np.zeros(POINTS_PER_THREAD * 4, dtype=np.uint64)
        for p, k_test in enumerate(k_tests):
            pub_test = coincurve.PublicKey.from_valid_secret(k_test.to_bytes(32, "big")).format(compressed=False)
            x_test = int.from_bytes(pub_test[1:33], "big")
            y_test = int.from_bytes(pub_test[33:65], "big")
            for j in range(4):
                sx_test[p * 4 + j] = (x_test >> (64 * j)) & 0xFFFFFFFFFFFFFFFF
                sy_test[p * 4 + j] = (y_test >> (64 * j)) & 0xFFFFFFFFFFFFFFFF
        cur_x_test = cp.asarray(sx_test)
        cur_y_test = cp.asarray(sy_test)
        pp_test = np.zeros(1, dtype=PATTERN_DTYPE)
        pp_test["prefix_len"] = 1
        pp_test["suffix_len"] = 0
        pp_test["repeat_n"] = 0
        pp_test["prefix"][0, 0] = ord("T")
        pp_test_dev = cp.asarray(pp_test)
        m_test = cp.zeros(POINTS_PER_THREAD * 2, dtype=MATCH_DTYPE)
        c_test = cp.zeros(1, dtype=cp.uint32)
        kernel(
            (1,), (1,),
            (cur_x_test, cur_y_test, np.int32(1), np.uint64(0),
             pp_test_dev, m_test, c_test, np.uint32(POINTS_PER_THREAD * 2))
        )
        cp.cuda.Stream.null.synchronize()
        n = int(c_test.get()[0])
        if n != POINTS_PER_THREAD:
            raise RuntimeError("GPU 内核未执行 (期望 {} 命中, 实际 {}). GPU 可能没有响应.".format(POINTS_PER_THREAD, n))
        matches_back = m_test[:n].get()
        for m in matches_back:
            packed = int(m["thread_id"])
            p_idx = packed & 0xF
            addr_gpu = bytes(m["address"]).decode("ascii")
            addr_cpu = cpu_priv_to_address(k_tests[p_idx])
            if addr_gpu != addr_cpu:
                raise RuntimeError("GPU 自检不匹配 (point={}): \n  GPU: {}\n  CPU: {}".format(p_idx, addr_gpu, addr_cpu))
        print("GPU 自检通过, 用时 {:.2f}s. GPU 确实在执行内核 ✓".format(time.time() - self_t0))
    except Exception as e:
        print()
        print("✗ GPU 自检失败: {}".format(e))
        print("  请检查: nvidia-smi 是否能看到 GPU; 显存是否充足; 驱动是否最新")
        sys.exit(1)
    print()
    pp = np.zeros(1, dtype=PATTERN_DTYPE)
    pp["prefix_len"] = len(prefix)
    pp["suffix_len"] = len(suffix)
    pp["repeat_n"]   = repeat_tail
    if prefix: pp["prefix"][0, :len(prefix)] = np.frombuffer(prefix.encode(), dtype=np.uint8)
    if suffix: pp["suffix"][0, :len(suffix)] = np.frombuffer(suffix.encode(), dtype=np.uint8)
    pp_dev = cp.asarray(pp)
    try:
        matches_dev_list = [cp.zeros(MAX_MATCHES, dtype=MATCH_DTYPE) for _ in range(N_STREAMS)]
        count_dev_list   = [cp.zeros(1, dtype=cp.uint32) for _ in range(N_STREAMS)]
        matches_host_list = [cp.cuda.alloc_pinned_memory(MATCH_DTYPE.itemsize * MAX_MATCHES)
                             for _ in range(N_STREAMS)]
        count_host_list   = [cp.cuda.alloc_pinned_memory(4) for _ in range(N_STREAMS)]
    except cp.cuda.memory.OutOfMemoryError as e:
        sys.stderr.write(
            "✗ GPU 显存不足: {}\n"
            "  当前 GPU: {} (需要至少 ~500MB 显存空间)\n"
            "  请关掉其他占用 GPU 的程序 (浏览器硬件加速、游戏、其他 CUDA 任务) 后重试.\n".format(e, gpu_name))
        sys.exit(1)
    matches_host_np_list = [
        np.frombuffer(matches_host_list[s], dtype=MATCH_DTYPE, count=MAX_MATCHES)
        for s in range(N_STREAMS)
    ]
    count_host_np_list = [
        np.frombuffer(count_host_list[s], dtype=np.uint32, count=1)
        for s in range(N_STREAMS)
    ]
    pts_per_stream = n_threads * POINTS_PER_THREAD
    total_pts = pts_per_stream * N_STREAMS
    print("生成 {} 个 GPU 起始点 ({} 套 × {} 线程 × {} 点, 用 {} 个 CPU 核并行)...".format(
        total_pts, N_STREAMS, n_threads, POINTS_PER_THREAD, cpu_count))
    t1 = time.time()
    n_chunks = cpu_count * 4
    chunk = (total_pts + n_chunks - 1) // n_chunks
    tasks = []
    remain = total_pts
    while remain > 0:
        c = min(chunk, remain)
        tasks.append(c)
        remain -= c
    ctx_pool = mp.get_context("spawn")
    try:
        with ctx_pool.Pool(processes=cpu_count) as pool:
            results = pool.map(gen_startpoints_batch, tasks)
    except Exception as e:
        sys.stderr.write(
            "✗ 起点生成失败: {}\n"
            "  可能原因: 内存不足 / coincurve 安装损坏 / 子进程被杀\n".format(e))
        sys.exit(1)
    base_keys_all = []
    sx_all = np.zeros(total_pts * 4, dtype=np.uint64)
    sy_all = np.zeros(total_pts * 4, dtype=np.uint64)
    idx = 0
    for xb, yb, privs in results:
        n_in_chunk = len(privs)
        for i in range(n_in_chunk):
            x_be = xb[i*32:(i+1)*32]
            y_be = yb[i*32:(i+1)*32]
            x = int.from_bytes(x_be, "big")
            y = int.from_bytes(y_be, "big")
            for j in range(4):
                sx_all[idx*4 + j] = (x >> (64 * j)) & 0xFFFFFFFFFFFFFFFF
                sy_all[idx*4 + j] = (y >> (64 * j)) & 0xFFFFFFFFFFFFFFFF
            base_keys_all.append(privs[i])
            idx += 1
    stream_state = []
    for s in range(N_STREAMS):
        beg = s * pts_per_stream
        end = (s + 1) * pts_per_stream
        sx_slice = sx_all[beg*4:end*4]
        sy_slice = sy_all[beg*4:end*4]
        base_keys = base_keys_all[beg:end]
        cur_x_dev = cp.asarray(sx_slice)
        cur_y_dev = cp.asarray(sy_slice)
        stream_state.append({
            "cur_x": cur_x_dev,
            "cur_y": cur_y_dev,
            "base_keys": base_keys,
            "step_offset": 0,
        })
    print("起点准备完成, 用时 {:.1f}s ({:.0f} pts/s)".format(
        time.time() - t1, total_pts / max(time.time() - t1, 0.001)))
    print()
    print("自适应调参: 测量单步耗时...")
    pp_warmup_dev = cp.asarray(pp)
    m_warmup = cp.zeros(MAX_MATCHES, dtype=MATCH_DTYPE)
    c_warmup = cp.zeros(1, dtype=cp.uint32)
    warmup_state = stream_state[0]
    cp.cuda.Stream.null.synchronize()
    t_warm = time.time()
    WARMUP_STEPS = 64
    kernel(
        (n_blocks,), (THREADS_PER_BLOCK,),
        (warmup_state["cur_x"], warmup_state["cur_y"],
         np.int32(WARMUP_STEPS), np.uint64(0),
         pp_warmup_dev, m_warmup, c_warmup, np.uint32(MAX_MATCHES))
    )
    cp.cuda.Stream.null.synchronize()
    elapsed_warm = time.time() - t_warm
    warmup_state["step_offset"] += WARMUP_STEPS
    warmup_hits = int(c_warmup.get()[0])
    target_sec = 1.5
    per_step = elapsed_warm / WARMUP_STEPS
    STEPS_PER_LAUNCH = max(16, int(target_sec / max(per_step, 1e-6)))
    STEPS_PER_LAUNCH = max(16, min(STEPS_PER_LAUNCH, 4096))
    BATCH = n_threads * STEPS_PER_LAUNCH * POINTS_PER_THREAD
    print("  warmup: {} 步用 {:.2f}s ({:.2f}ms/步)".format(WARMUP_STEPS, elapsed_warm, per_step * 1000))
    if warmup_hits > MAX_MATCHES:
        print("  警告: 当前条件命中率过高, warmup 已超过 GPU 命中缓冲上限 {}".format(MAX_MATCHES))
    print("  自适应 STEPS_PER_LAUNCH = {} (预期 ~{:.1f}s/launch, ~{:.1f}M 地址/launch)".format(
        STEPS_PER_LAUNCH, STEPS_PER_LAUNCH * per_step, BATCH / 1e6))
    print()
    ctx = mp.get_context("spawn")
    match_q = ctx.Queue(maxsize=MATCH_QUEUE_SIZE)
    stat_q  = ctx.Queue(maxsize=4096)
    stop_evt  = ctx.Event()
    pause_evt = ctx.Event()
    cpu_procs = []
    for _ in range(n_cpu_workers):
        p = ctx.Process(target=cpu_worker,
                        args=(prefix, suffix, repeat_tail,
                              match_q, stat_q, stop_evt, pause_evt),
                        daemon=True)
        p.start()
        cpu_procs.append(p)
    session_start = time.time()
    cycle_start   = session_start
    gpu_total     = 0
    cpu_total     = 0
    matches_found = 0
    session_gpu_total = 0
    session_cpu_total = 0
    session_matches_found = 0
    gpu_dropped_matches = 0
    cpu_dropped_matches = 0
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(N_STREAMS)]
    in_flight = [False] * N_STREAMS
    def save_matches(records):
        if not records:
            return
        try:
            with open(output_path, "a", encoding="utf-8") as f:
                now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                for priv_hex, addr, source in records:
                    f.write("=" * 70 + "\n")
                    f.write("时间          : {}\n".format(now_text))
                    f.write("匹配模式      : {}\n".format(pattern_desc))
                    f.write("来源          : {}\n".format(source))
                    f.write("地址          : {}\n".format(addr))
                    f.write("私钥 (hex)    : {}\n".format(priv_hex))
                    f.write("=" * 70 + "\n\n")
        except OSError as e:
            sys.stderr.write(
                "\n⚠ 写命中文件失败 ({}): {}\n"
                "  本批 {} 个命中未能写入文件, 请从终端输出或重新运行获取.\n".format(
                    output_path, e, len(records)))
    def drain_cpu_stats():
        nonlocal cpu_total, cpu_dropped_matches
        added = 0
        dropped = 0
        try:
            while True:
                item = stat_q.get_nowait()
                if isinstance(item, tuple):
                    added += item[0]
                    dropped += item[1]
                else:
                    added += item
        except Empty:
            pass
        cpu_total += added
        cpu_dropped_matches += dropped
        return added
    def drain_cpu_matches():
        ms = []
        try:
            while True:
                ms.append(match_q.get_nowait())
        except Empty:
            pass
        return ms
    hw_cache = {"t": 0.0, "gpu": None, "cpu": None}
    def get_hw_stats(now):
        if now - hw_cache["t"] >= 1.0:
            hw_cache["gpu"] = _gpu_stats()
            hw_cache["cpu"] = _cpu_percent()
            hw_cache["t"] = now
        return hw_cache["gpu"], hw_cache["cpu"]
    status_stop = threading.Event()
    status_paused = threading.Event()
    status_state = {"matches_found": 0, "last_width": 0}
    def _display_width(s):
        w = 0
        for ch in s:
            o = ord(ch)
            if o < 0x80:
                w += 1
            else:
                w += 2
        return w
    def _term_cols():
        try:
            return os.get_terminal_size().columns
        except Exception:
            return 80
    def status_loop():
        while not status_stop.is_set():
            try:
                time.sleep(0.5)
                if status_paused.is_set():
                    continue
                now = time.time()
                cycle_elapsed = now - cycle_start
                elapsed = max(cycle_elapsed, 0.001)
                g_rate = gpu_total / elapsed
                c_rate = cpu_total / elapsed
                total_rate = g_rate + c_rate
                eta = fmt_time(1.0 / (prob * total_rate)) if (total_rate > 0 and prob > 0) else "?"
                gs, cpu_pct = get_hw_stats(now)
                hw_parts = []
                if gs is not None:
                    u, p, t, _m = gs
                    hw_parts.append("GPU{:.0f}% {:.0f}W {:.0f}°C".format(u, p, t))
                if cpu_pct is not None:
                    hw_parts.append("CPU{:.0f}%".format(cpu_pct))
                hw_str = " | " + " ".join(hw_parts) if hw_parts else ""
                main_part = "已跑{} | {}/秒 | 累计{} | 还需{} | 命中{}".format(
                    fmt_time(cycle_elapsed),
                    fmt_num(total_rate),
                    fmt_num(gpu_total + cpu_total),
                    eta,
                    status_state["matches_found"],
                )
                verbose_part = "已跑{} | 速度{}/秒 (GPU{}+CPU{}) | 累计{} | 还需{} | 命中{}".format(
                    fmt_time(cycle_elapsed),
                    fmt_num(total_rate), fmt_num(g_rate), fmt_num(c_rate),
                    fmt_num(gpu_total + cpu_total), eta,
                    status_state["matches_found"],
                )
                cols = _term_cols()
                budget = max(40, cols - 2)
                for line in (verbose_part + hw_str, verbose_part, main_part + hw_str, main_part):
                    if _display_width(line) <= budget:
                        break
                else:
                    line = main_part
                cur_w = _display_width(line)
                pad = max(0, status_state["last_width"] - cur_w)
                sys.stdout.write("\r" + line + (" " * pad) + "\r")
                sys.stdout.flush()
                status_state["last_width"] = cur_w
            except BaseException:
                time.sleep(1.0)
    print("开始搜索...")
    print()
    def launch_stream(idx):
        st = streams[idx]
        state = stream_state[idx]
        m_dev = matches_dev_list[idx]
        c_dev = count_dev_list[idx]
        m_host = matches_host_list[idx]
        c_host = count_host_list[idx]
        with st:
            cp.cuda.runtime.memsetAsync(c_dev.data.ptr, 0, 4, st.ptr)
            kernel(
                (n_blocks,), (THREADS_PER_BLOCK,),
                (state["cur_x"], state["cur_y"],
                 np.int32(STEPS_PER_LAUNCH),
                 np.uint64(state["step_offset"]),
                 pp_dev,
                 m_dev, c_dev,
                 np.uint32(MAX_MATCHES)),
                stream=st
            )
            cp.cuda.runtime.memcpyAsync(
                c_host.ptr, c_dev.data.ptr, 4,
                cp.cuda.runtime.memcpyDeviceToHost, st.ptr
            )
            cp.cuda.runtime.memcpyAsync(
                m_host.ptr, m_dev.data.ptr,
                MATCH_DTYPE.itemsize * MAX_MATCHES,
                cp.cuda.runtime.memcpyDeviceToHost, st.ptr
            )
        state["step_offset"] += STEPS_PER_LAUNCH
        in_flight[idx] = True
    def collect_stream(idx):
        nonlocal gpu_dropped_matches
        streams[idx].synchronize()
        in_flight[idx] = False
        n_hits = int(count_host_np_list[idx][0])
        if n_hits == 0:
            return []
        n_clamp = min(n_hits, MAX_MATCHES)
        if n_hits > MAX_MATCHES:
            gpu_dropped_matches += n_hits - MAX_MATCHES
            sys.stderr.write(
                "\n[警告] GPU 命中缓冲已满: 本批命中 {}, 保存 {}, 丢弃 {}. "
                "请提高匹配难度或增大 MAX_MATCHES.\n".format(
                    n_hits, MAX_MATCHES, n_hits - MAX_MATCHES)
            )
        arr = matches_host_np_list[idx][:n_clamp]
        state = stream_state[idx]
        base_keys = state["base_keys"]
        results = []
        for m in arr:
            packed = int(m["thread_id"])
            real_tid = packed >> 4
            p_idx = packed & 0xF
            step = int(m["step"])
            addr_gpu = bytes(m["address"]).decode("ascii", errors="replace")
            if real_tid >= n_threads or p_idx >= POINTS_PER_THREAD:
                continue
            priv_int = (base_keys[real_tid * POINTS_PER_THREAD + p_idx] + step) % SECP256K1_N
            addr_cpu = cpu_priv_to_address(priv_int)
            if addr_cpu != addr_gpu:
                sys.stderr.write(
                    "\n[警告] GPU/CPU 校验不一致, 丢弃: GPU={} CPU={}\n".format(addr_gpu, addr_cpu)
                )
                continue
            results.append((priv_int.to_bytes(32, "big").hex(), addr_gpu, "GPU"))
        return results
    submit_order = []
    status_thread = threading.Thread(target=status_loop, daemon=True)
    status_thread.start()
    interrupted = {"flag": False}
    sig_names = ["SIGINT"]
    if hasattr(signal, "SIGBREAK"):
        sig_names.append("SIGBREAK")
    prev_handlers = {}
    for name in sig_names:
        prev_handlers[name] = signal.getsignal(getattr(signal, name))
    def _on_sig(signum, frame):
        if not interrupted["flag"]:
            interrupted["flag"] = True
        else:
            for n in sig_names:
                signal.signal(getattr(signal, n), prev_handlers[n])
            raise KeyboardInterrupt
    for name in sig_names:
        signal.signal(getattr(signal, name), _on_sig)
    try:
        for s in range(N_STREAMS):
            launch_stream(s)
            submit_order.append(s)
        while not interrupted["flag"]:
            idx = submit_order.pop(0)
            gpu_matches_raw = collect_stream(idx)
            if interrupted["flag"]:
                sys.stdout.write("\n[收到中断, 正在清理...]\n")
                sys.stdout.flush()
                break
            launch_stream(idx)
            submit_order.append(idx)
            gpu_total += BATCH
            session_gpu_total += BATCH
            cpu_added = drain_cpu_stats()
            session_cpu_total += cpu_added
            cpu_matches_raw_pairs = drain_cpu_matches()
            cpu_matches_raw = [(h, a, "CPU") for (h, a) in cpu_matches_raw_pairs]
            all_matches = gpu_matches_raw + cpu_matches_raw
            if not all_matches:
                continue
            save_matches(all_matches)
            matches_found += len(all_matches)
            session_matches_found += len(all_matches)
            status_state["matches_found"] = matches_found
            if matches_found < HIT_PAUSE_THRESHOLD:
                continue
            pause_evt.set()
            status_paused.set()
            sys.stdout.write("\r" + (" " * status_state["last_width"]) + "\r")
            status_state["last_width"] = 0
            sys.stdout.flush()
            print()
            print("=" * 70)
            print("  *** 已累计找到 {} 个匹配地址 ***".format(matches_found))
            print("=" * 70)
            print("  匹配模式     : {}".format(pattern_desc))
            print("  本轮用时     : {}".format(fmt_time(time.time() - cycle_start)))
            print("  已保存到     : {}".format(output_path))
            print("  (所有命中含私钥已写入文件, 请妥善保管)")
            print("=" * 70)
            print()
            print("  请选择操作:")
            print("    1) 继续生成")
            print("    2) 停止生成")
            print("    3) 更改生成条件 (重新输入模式)")
            print()
            ans = input_with_timeout("  输入选项 [1/2/3] (30 秒无回应自动继续): ", 30.0)
            choice = (ans or "").strip()
            if choice == "2" or choice.lower() in ("n", "no", "否", "不", "stop"):
                print("\n已停止。")
                return
            if choice == "3":
                print()
                print("-" * 70)
                print("  更改生成条件")
                print("-" * 70)
                while True:
                    mode = input("  选择模式 [1=自定义前后缀 / 2=尾部重复]: ").strip()
                    if mode in ("1", "2"): break
                    print("  无效输入")
                new_prefix, new_suffix, new_repeat = "", "", 0
                if mode == "1":
                    new_prefix, new_suffix = prompt_mode_1()
                else:
                    new_repeat = prompt_mode_2()
                prefix, suffix, repeat_tail = new_prefix, new_suffix, new_repeat
                pp[:] = 0
                pp["prefix_len"] = len(prefix)
                pp["suffix_len"] = len(suffix)
                pp["repeat_n"] = repeat_tail
                if prefix: pp["prefix"][0, :len(prefix)] = np.frombuffer(prefix.encode(), dtype=np.uint8)
                if suffix: pp["suffix"][0, :len(suffix)] = np.frombuffer(suffix.encode(), dtype=np.uint8)
                for s_idx in range(N_STREAMS):
                    if in_flight[s_idx]:
                        try:
                            streams[s_idx].synchronize()
                            in_flight[s_idx] = False
                        except BaseException: pass
                submit_order[:] = []
                pp_dev.set(pp)
                parts = []
                if prefix:      parts.append("前缀={}".format(prefix))
                if suffix:      parts.append("后缀={}".format(suffix))
                if repeat_tail: parts.append("尾部重复≥{}位".format(repeat_tail))
                pattern_desc = " 且 ".join(parts)
                prob = estimate_probability(prefix, suffix, repeat_tail)
                stop_evt.set()
                for p in cpu_procs:
                    try: p.kill()
                    except BaseException: pass
                for p in cpu_procs:
                    try: p.join(timeout=0.2)
                    except BaseException: pass
                cpu_procs[:] = []
                old_cpu_added = drain_cpu_stats()
                session_cpu_total += old_cpu_added
                drain_cpu_matches()
                stop_evt = ctx.Event()
                for _ in range(n_cpu_workers):
                    cpw = ctx.Process(target=cpu_worker,
                                      args=(prefix, suffix, repeat_tail,
                                            match_q, stat_q, stop_evt, pause_evt),
                                      daemon=True)
                    cpw.start()
                    cpu_procs.append(cpw)
                print()
                print("新模式已应用: {}".format(pattern_desc))
                if prob > 0:
                    print("理论概率: 平均 {} 个地址出 1 个".format(fmt_num(1/prob)))
                print()
            else:
                if ans is None:
                    print("  (30 秒无回应, 自动继续)")
            print()
            print("继续搜索...")
            print()
            matches_found = 0
            status_state["matches_found"] = 0
            cycle_start = time.time()
            gpu_total = 0
            cpu_total = 0
            leftover_cpu = drain_cpu_stats()
            session_cpu_total += leftover_cpu
            drain_cpu_matches()
            pause_evt.clear()
            status_paused.clear()
            if not submit_order:
                for s in range(N_STREAMS):
                    launch_stream(s)
                    submit_order.append(s)
    except KeyboardInterrupt:
        print("\n\n用户中断。")
    finally:
        for name in sig_names:
            try: signal.signal(getattr(signal, name), prev_handlers[name])
            except BaseException: pass
        for p in cpu_procs:
            if p.is_alive():
                try: p.kill()
                except BaseException: pass
        for p in cpu_procs:
            try: p.join(timeout=0.2)
            except BaseException: pass
        status_stop.set()
        try: status_thread.join(timeout=0.5)
        except BaseException: pass
        for s in range(N_STREAMS):
            if in_flight[s]:
                try: streams[s].synchronize()
                except BaseException: pass
        elapsed = time.time() - session_start
        print()
        print("=" * 70)
        print("  会话总结")
        print("=" * 70)
        print("  总用时       : {}".format(fmt_time(elapsed)))
        print("  GPU 生成     : {} 个地址".format(fmt_num(session_gpu_total)))
        print("  CPU 生成     : {} 个地址".format(fmt_num(session_cpu_total)))
        print("  总共生成     : {} 个地址".format(fmt_num(session_gpu_total + session_cpu_total)))
        print("  找到匹配     : {} 个".format(session_matches_found))
        if gpu_dropped_matches or cpu_dropped_matches:
            print("  丢弃命中     : GPU {} 个, CPU {} 个".format(
                fmt_num(gpu_dropped_matches), fmt_num(cpu_dropped_matches)))
        if elapsed > 0:
            print("  平均速度     : {} 个地址/秒".format(
                fmt_num((session_gpu_total + session_cpu_total) / elapsed)))
        print("=" * 70)
def prompt_mode_1():
    while True:
        prefix = input("请输入自定义前缀 (留空表示无, 必须以 T 开头): ").strip()
        suffix = input("请输入自定义后缀 (留空表示无): ").strip()
        if not prefix and not suffix:
            print("错误: 前缀和后缀至少填写一项。\n")
            continue
        errs = validate_pattern(prefix, suffix)
        if errs:
            print("输入有误:")
            for e in errs: print("  - {}".format(e))
            print()
            continue
        return prefix, suffix
def prompt_mode_2():
    while True:
        s = input("请输入尾号重复字符数 N (地址末尾 N 位为相同字符, 建议 5-8): ").strip()
        try: n = int(s)
        except ValueError:
            print("错误: 请输入整数。\n"); continue
        if n < 5:
            print("错误: N 太小会产生海量命中并拖慢程序, 请至少输入 5。\n"); continue
        if n > 34:
            print("错误: 太大, 地址只有 34 位。\n"); continue
        return n
def main():
    print("=" * 70)
    print("  TRON 靓号地址生成器 (CUDA + CPU 全速版)")
    print("=" * 70)
    print()
    print("  模式 1: 自定义模式 — 指定前缀和/或后缀")
    print("  模式 2: 靓号模式   — 指定尾部重复字符位数")
    print()
    while True:
        mode = input("请选择模式 [1/2]: ").strip()
        if mode in ("1", "2"): break
        print("无效输入。\n")
    prefix = ""; suffix = ""; repeat_tail = 0
    if mode == "1":
        prefix, suffix = prompt_mode_1()
    else:
        repeat_tail = prompt_mode_2()
    out_dir = os.path.join(APP_BASE_DIR, "命中地址")

    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as e:
        sys.stderr.write(
            "✗ 无法创建输出目录 {}: {}\n"
            "  请检查脚本所在目录是否有写权限\n".format(out_dir, e))
        sys.exit(1)
    output_path = os.path.join(
        out_dir, "matches_{}.txt".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    run_search(prefix, suffix, repeat_tail, output_path)
if __name__ == "__main__":
    mp.freeze_support()
    main()