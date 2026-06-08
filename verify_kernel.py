import os
import sys
import secrets
import time
import warnings
warnings.filterwarnings("ignore", message=".*CUDA path could not be detected.*")
import numpy as np
import cupy as cp
import coincurve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tron_vanity_gpu import (
    PATTERN_DTYPE, MATCH_DTYPE, load_kernel,
    cpu_priv_to_address, SECP256K1_N
)
def verify_with_m(M):
    props = cp.cuda.runtime.getDeviceProperties(0)
    cc_major = props["major"]
    cc_minor = props["minor"]
    print("\n=== 测试 M={} ===".format(M))
    print("加载并编译 CUDA 内核 (arch=sm_{}{})...".format(cc_major, cc_minor))
    t0 = time.time()
    kernel = load_kernel(arch="sm_{}{}".format(cc_major, cc_minor),
                         points_per_thread=M)
    print("编译完成: {:.1f}s".format(time.time() - t0))
    THREADS = 4
    STEPS = 64
    n_chains = THREADS * M
    base_keys = []
    sx = np.zeros(n_chains * 4, dtype=np.uint64)
    sy = np.zeros(n_chains * 4, dtype=np.uint64)
    for i in range(n_chains):
        k = secrets.randbelow(SECP256K1_N - 1) + 1
        base_keys.append(k)
        pub = coincurve.PublicKey.from_valid_secret(k.to_bytes(32, "big")).format(compressed=False)
        x = int.from_bytes(pub[1:33], "big")
        y = int.from_bytes(pub[33:65], "big")
        for j in range(4):
            sx[i*4 + j] = (x >> (64 * j)) & 0xFFFFFFFFFFFFFFFF
            sy[i*4 + j] = (y >> (64 * j)) & 0xFFFFFFFFFFFFFFFF
    start_x = cp.asarray(sx)
    start_y = cp.asarray(sy)
    pp = np.zeros(1, dtype=PATTERN_DTYPE)
    pp["prefix_len"] = 1
    pp["suffix_len"] = 0
    pp["repeat_n"] = 0
    pp["prefix"][0, 0] = ord("T")
    pp_dev = cp.asarray(pp)
    MAX_MATCHES = THREADS * STEPS * M * 4 + 10
    matches_dev = cp.zeros(MAX_MATCHES, dtype=MATCH_DTYPE)
    count_dev = cp.zeros(1, dtype=cp.uint32)
    print("启动内核 (THREADS={}, M={}, STEPS={} x 2 次)...".format(THREADS, M, STEPS))
    for launch_idx in range(2):
        offset = launch_idx * STEPS
        kernel(
            (1,), (THREADS,),
            (start_x, start_y,
             np.int32(STEPS),
             np.uint64(offset),
             pp_dev,
             matches_dev, count_dev, np.uint32(MAX_MATCHES))
        )
    cp.cuda.Stream.null.synchronize()
    n_hits = int(count_dev.get()[0])
    expected = THREADS * STEPS * M * 2
    print("内核报告命中: {} (期望 {})".format(n_hits, expected))
    if n_hits != expected:
        print("⚠ 命中数不匹配")
    n_check = min(n_hits, MAX_MATCHES)
    matches = matches_dev[:n_check].get()
    mismatches = 0
    for m in matches:
        packed = int(m["thread_id"])
        real_tid = packed >> 4
        p_idx = packed & 0xF
        step = int(m["step"])
        gpu_addr = bytes(m["address"]).decode("ascii", errors="replace")
        if real_tid >= THREADS or p_idx >= M:
            print("✗ 越界 tid={} p={}".format(real_tid, p_idx))
            mismatches += 1
            continue
        chain_idx = real_tid * M + p_idx
        priv_int = (base_keys[chain_idx] + step) % SECP256K1_N
        cpu_addr = cpu_priv_to_address(priv_int)
        if gpu_addr != cpu_addr:
            mismatches += 1
            if mismatches <= 5:
                print("✗ 不匹配 tid={} p={} step={}".format(real_tid, p_idx, step))
                print("    GPU: {}".format(gpu_addr))
                print("    CPU: {}".format(cpu_addr))
    if mismatches == 0:
        print("✓ M={}: 全部 {} 个 GPU 地址 (2 次启动, 状态延续) 与 CPU 完全一致".format(M, n_check))
        return True
    else:
        print("✗ M={}: 发现 {} 个不匹配".format(M, mismatches))
        return False
def main():
    try:
        cp.cuda.Device(0).use()
        n_dev = cp.cuda.runtime.getDeviceCount()
        if n_dev == 0:
            print("✗ 没检测到 NVIDIA GPU. 请确认驱动已安装, nvidia-smi 能看到设备.")
            return 1
    except Exception as e:
        print("✗ CUDA 初始化失败: {}".format(e))
        print("  请确认 NVIDIA 驱动已安装, GPU 可用 (跑 nvidia-smi 测试).")
        return 1
    all_ok = True
    for M in (8, 16):
        ok = verify_with_m(M)
        if not ok:
            all_ok = False
    print()
    if all_ok:
        print("✓ M=8 和 M=16 两种配置都通过验证")
        print("  Montgomery 批量求逆 + GPU 持久化状态 + 全部加密原语 OK")
        return 0
    else:
        return 1
if __name__ == "__main__":
    sys.exit(main())