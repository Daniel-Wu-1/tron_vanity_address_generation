import os
import sys
import time
import secrets
import hashlib
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
import coincurve
from Crypto.Hash import keccak as keccak_lib
import base58
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False
def _set_low_priority():
    if not _HAS_PSUTIL:
        return
    try:
        p = psutil.Process(os.getpid())
        if sys.platform == "win32":
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            p.nice(5)
    except Exception:
        pass
def _priv_to_address(priv_bytes: bytes) -> str:
    pub = coincurve.PublicKey.from_valid_secret(priv_bytes).format(compressed=False)
    k = keccak_lib.new(digest_bits=256)
    k.update(pub[1:])
    payload = b"\x41" + k.digest()[-20:]
    checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    return base58.b58encode(payload + checksum).decode()
def _matches(addr: str, prefix: str, suffix: str, repeat_tail: int) -> bool:
    if prefix and not addr.startswith(prefix):
        return False
    if suffix and not addr.endswith(suffix):
        return False
    if repeat_tail > 0:
        tail = addr[-repeat_tail:]
        if tail != tail[0] * repeat_tail:
            return False
    return True
def gen_startpoints_batch(n):
    SECP256K1_N_LOCAL = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    xs = bytearray(n * 32)
    ys = bytearray(n * 32)
    privs = [0] * n
    for i in range(n):
        k = secrets.randbelow(SECP256K1_N_LOCAL - 1) + 1
        privs[i] = k
        pub = coincurve.PublicKey.from_valid_secret(k.to_bytes(32, "big")).format(compressed=False)
        xs[i*32:(i+1)*32] = pub[1:33]
        ys[i*32:(i+1)*32] = pub[33:65]
    return bytes(xs), bytes(ys), privs
def cpu_worker(prefix: str, suffix: str, repeat_tail: int,
               match_q, stat_q, stop_evt, pause_evt):
    import signal as _signal
    for sig_name in ("SIGINT", "SIGBREAK"):
        if hasattr(_signal, sig_name):
            try:
                _signal.signal(getattr(_signal, sig_name), _signal.SIG_IGN)
            except Exception:
                pass
    _set_low_priority()
    local = 0
    dropped_matches = 0
    last_report = time.time()
    REPORT_EVERY = 1000
    try:
        while not stop_evt.is_set():
            if pause_evt.is_set():
                time.sleep(0.05)
                continue
            for _ in range(REPORT_EVERY):
                if stop_evt.is_set() or pause_evt.is_set():
                    break
                priv = secrets.token_bytes(32)
                try:
                    addr = _priv_to_address(priv)
                except ValueError:
                    continue
                local += 1
                if _matches(addr, prefix, suffix, repeat_tail):
                    try:
                        match_q.put((priv.hex(), addr), timeout=1.0)
                    except Exception:
                        dropped_matches += 1
            now = time.time()
            if now - last_report >= 0.25:
                try:
                    stat_q.put_nowait((local, dropped_matches))
                except Exception:
                    pass
                local = 0
                dropped_matches = 0
                last_report = now
    except (KeyboardInterrupt, BrokenPipeError, EOFError):
        pass