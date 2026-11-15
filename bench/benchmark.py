#!/usr/bin/env python3
"""Runtime and peak-memory benchmark for lagcorr.analyze.

Wall-clock time is measured *without* tracemalloc, because Python's allocation
tracer adds a large constant overhead that would distort the O(n log n)
growth curve.  Peak Python-level memory is measured in a separate process via
the ``--memory`` flag; the driver combines both.  A maximum-size run with a
heavily missing signal shows that gaps do not change the cost.

Usage:
  python3 bench/benchmark.py             # full table (re-execs itself)
  python3 bench/benchmark.py --memory N  # single peak-memory measurement
"""

import os
import random
import resource
import sys
import time
import tracemalloc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lagcorr import analyze  # noqa: E402

MAX_N = 32768


def make_signal(n, miss_prob, rng):
    left = [rng.randint(-10**6, 10**6) if rng.random() >= miss_prob
            else None for _ in range(n)]
    right = [rng.randint(-10**6, 10**6) if rng.random() >= miss_prob
             else None for _ in range(n)]
    return left, right


def memory_only(n):
    """One traced run; prints peak traced MiB and current RSS MiB."""
    rng = random.Random(424242)
    left, right = make_signal(n, 0.1, rng)
    tracemalloc.start()
    out = analyze(left, right)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KiB on Linux
    print(f"{peak / 2**20:.3f} {rss / 2**10:.3f} {len(out)}")


def timed_run(n, miss_prob, rng, repeats=1):
    left, right = make_signal(n, miss_prob, rng)
    best = float("inf")
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = analyze(left, right)
        best = min(best, time.perf_counter() - t0)
    return best, out


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--memory":
        memory_only(int(sys.argv[2]))
        return

    rng = random.Random(424242)
    sizes = [256, 512, 1024, 2048, 4096, 8192, 16384, MAX_N]
    print(f"{'n':>6} {'lags':>7} {'time (s)':>10} {'peak MiB':>10} "
          f"{'rss MiB':>9} {'x growth':>22}")
    prev_n = prev_t = None
    for n in sizes:
        # Timing: untraced, best of two for the larger sizes.
        elapsed, out = timed_run(n, 0.1, rng, repeats=2 if n >= 2048 else 1)
        assert len(out) == 2 * n - 1
        # Peak memory: separate traced subprocess so timing stays clean.
        import subprocess
        proc = subprocess.run(
            [sys.executable, "-I", os.path.abspath(__file__),
             "--memory", str(n)],
            capture_output=True, text=True, check=True)
        peak_mib, rss_mib, _ = proc.stdout.split()
        growth = ""
        if prev_t is not None:
            expected = (n * (n.bit_length() - 1)) / (
                prev_n * (prev_n.bit_length() - 1))
            growth = f"x{elapsed / prev_t:.2f} (theory x{expected:.2f})"
        print(f"{n:>6} {2*n-1:>7} {elapsed:>10.4f} "
              f"{float(peak_mib):>10.2f} {float(rss_mib):>9.2f} {growth:>22}")
        prev_n, prev_t = n, elapsed

    # Maximum size with a heavily missing signal (~50% gaps on both sides).
    elapsed, out = timed_run(MAX_N, 0.5, rng)
    print()
    print(f"max n={MAX_N}, ~50% missing on both channels: "
          f"time={elapsed:.4f}s (untraced), lags={len(out)}")

    # Structural checks on max-size output.
    assert [r["lag"] for r in out] == list(range(1 - MAX_N, MAX_N))
    assert all(0 <= r["count"] <= MAX_N for r in out)
    print("structural checks on max-size output passed")


if __name__ == "__main__":
    main()
