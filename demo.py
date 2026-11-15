#!/usr/bin/env python3
"""Demonstration of lagcorr.analyze on signals with missing samples.

Two synthetic channels share a buried pulse pattern.  The right channel is a
delayed, noisy copy of the left one with random gaps (``None``) on both
sides.  The script prints the lag records around the best alignment and shows
how counts shrink with the gaps, then runs a tiny fully printed example so the
exact per-lag behavior (including corr=None with fewer than two pairs or a
constant side) is visible.
"""

import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lagcorr import analyze  # noqa: E402


def small_printed_example():
    print("=== Small example with gaps ===")
    left = [1, 2, None, 4, 5]
    right = [None, 2, 3, 4, None]
    print(f"left  = {left}")
    print(f"right = {right}")
    for rec in analyze(left, right):
        corr = "None" if rec["corr"] is None else f"{rec['corr']:+.6f}"
        print(f"  lag={rec['lag']:+d}  count={rec['count']}  corr={corr}")
    print()


def buried_pulse_demo():
    print("=== Buried pulse with 30% missing samples ===")
    rng = random.Random(7)
    n = 96
    true_lag = 12

    def pulse(i):
        # Smooth triangular pulse train plus a weak deterministic carrier.
        phase = i % 24
        tri = 1.0 - abs(phase - 12) / 12.0
        return 80 * tri + 20 * math.sin(i * 0.5)

    clean = [pulse(i) for i in range(n)]
    left = []
    right = []
    for i in range(n):
        # right is left shifted by true_lag with independent noise; both
        # channels lose ~30% of their samples at random positions.
        lv = int(clean[i] + rng.randint(-3, 3))
        left.append(None if rng.random() < 0.3 else lv)
        if i - true_lag < 0:
            right.append(None if rng.random() < 0.3 else rng.randint(-5, 5))
        else:
            rv = int(clean[i - true_lag] + rng.randint(-3, 3))
            right.append(None if rng.random() < 0.3 else rv)

    records = analyze(left, right)
    valid = [r for r in records if r["corr"] is not None]
    best = max(valid, key=lambda r: r["corr"])
    top = sorted(valid, key=lambda r: -r["corr"])[:5]

    present_l = sum(v is not None for v in left)
    present_r = sum(v is not None for v in right)
    print(f"n={n}, present samples: left={present_l}, right={present_r}")
    print(f"injected shift: right[i] ~= left[i - {true_lag}] "
          f"(i.e. lag d = +{true_lag})")
    print("top five lags by Pearson corr:")
    for rec in top:
        print(f"  lag={rec['lag']:+3d}  count={rec['count']:3d}  "
              f"corr={rec['corr']:+.6f}")
    print(f"detected lag = {best['lag']:+d} "
          f"(count={best['count']}, corr={best['corr']:+.6f})")
    assert best["lag"] == true_lag, best
    # The record count at the aligned lag never exceeds the overlap.
    assert best["count"] <= n - abs(true_lag)
    print("missing-sample lag analysis passed")


if __name__ == "__main__":
    small_printed_example()
    buried_pulse_demo()
