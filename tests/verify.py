#!/usr/bin/env python3
"""Independent verification of lagcorr.analyze.

The reference implementation here is deliberately written from scratch in the
most direct style (an explicit per-lag, per-pair scan) so that it shares no
code with the FFT-based implementation under test.  Correlations are checked
two ways: against a float reference computed with ``statistics.fmean`` style
sums and, for small cases, against a high-precision decimal reference.
"""

import math
import os
import random
import sys
from decimal import Decimal, getcontext

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lagcorr import analyze  # noqa: E402

getcontext().prec = 60
TOL = 1e-10


def reference(left, right):
    """Straightforward O(n^2) enumeration, independent of lagcorr."""
    n = len(left)
    out = []
    for d in range(1 - n, n):
        pairs = []
        for i in range(n):
            j = i + d
            if 0 <= j < n and left[i] is not None and right[j] is not None:
                pairs.append((left[i], right[j]))
        c = len(pairs)
        corr = None
        if c >= 2:
            xs = [p[0] for p in pairs]
            ys = [p[1] for p in pairs]
            mx = sum(xs) / c
            my = sum(ys) / c
            vx = sum((x - mx) ** 2 for x in xs)
            vy = sum((y - my) ** 2 for y in ys)
            cov = sum((x - mx) * (y - my) for x, y in pairs)
            if vx > 0.0 and vy > 0.0:
                corr = cov / math.sqrt(vx * vy)
        out.append({"lag": d, "count": c, "corr": corr})
    return out


def decimal_corr(pairs):
    """High precision Pearson coefficient for a list of (x, y) pairs."""
    c = Decimal(len(pairs))
    sx = sum((Decimal(x) for x, _ in pairs), Decimal(0))
    sy = sum((Decimal(y) for _, y in pairs), Decimal(0))
    sxx = sum((Decimal(x) * Decimal(x) for x, _ in pairs), Decimal(0))
    syy = sum((Decimal(y) * Decimal(y) for _, y in pairs), Decimal(0))
    sxy = sum((Decimal(x) * Decimal(y) for x, y in pairs), Decimal(0))
    num = c * sxy - sx * sy
    vx = c * sxx - sx * sx
    vy = c * syy - sy * sy
    if vx <= 0 or vy <= 0:
        return None
    return float(num / (vx * vy).sqrt())


def assert_records_match(got, ref, left, right, label):
    assert len(got) == len(ref), (label, len(got), len(ref))
    assert [r["lag"] for r in got] == list(range(1 - len(left), len(left))), \
        label + ": lags not ascending / complete"
    for g, r in zip(got, ref):
        assert g["lag"] == r["lag"]
        assert g["count"] == r["count"], (label, g, r)
        if r["corr"] is None:
            assert g["corr"] is None, (label, g, r)
        else:
            assert g["corr"] is not None, (label, g, r)
            err = abs(g["corr"] - r["corr"])
            assert err <= TOL, (label, g, r, err)
            assert -1.0 - 1e-12 <= g["corr"] <= 1.0 + 1e-12, (label, g)


def run():
    failures = 0

    # --- fixed examples --------------------------------------------------
    # The example from the specification.
    left, right = [1, 2], [2, 4]
    got = analyze(left, right)
    expected = [
        {"lag": -1, "count": 1, "corr": None},
        {"lag": 0, "count": 2, "corr": 1.0},
        {"lag": 1, "count": 1, "corr": None},
    ]
    assert got == expected, got
    assert left == [1, 2] and right == [2, 4], "inputs were mutated"

    # Empty input.
    assert analyze([], []) == []

    # n = 1: two lags, each a single pair -> corr None.
    got1 = analyze([5], [None])
    assert got1 == [
        {"lag": 0, "count": 0, "corr": None},
    ]
    got1b = analyze([5], [7])
    assert got1b == [{"lag": 0, "count": 1, "corr": None}]

    # Constant side -> zero variance -> None even with enough pairs.
    gotc = analyze([3, 3, 3, 3], [1, 2, 3, 4])
    assert all(r["corr"] is None for r in gotc)
    assert gotc[3]["count"] == 4

    # Perfect anti-correlation and a missing sample in the middle.
    la = [1, None, 3, 4, 5]
    lb = [10, 9, 8, 7, 6]
    got = analyze(la, lb)
    ref = reference(la, lb)
    assert_records_match(got, ref, la, lb, "missing+anti")
    assert got[4]["corr"] is not None and abs(got[4]["corr"] + 1.0) < 1e-12

    # --- exhaustive small enumeration with random data -------------------
    rng = random.Random(20261115)
    for trial in range(400):
        n = rng.randint(0, 12)
        left = []
        right = []
        for _ in range(n):
            left.append(None if rng.random() < 0.35
                        else rng.randint(-5, 5))
            right.append(None if rng.random() < 0.35
                         else rng.randint(-5, 5))
        left_snapshot = list(left)
        right_snapshot = list(right)
        got = analyze(left, right)
        ref = reference(left, right)
        assert_records_match(got, ref, left, right, f"random n={n} t={trial}")
        assert left == left_snapshot and right == right_snapshot, \
            "inputs mutated"

    # Decimal cross-check on a subset, including duplicate values and gaps.
    for trial in range(50):
        n = rng.randint(4, 10)
        left = [rng.randint(-100, 100) for _ in range(n)]
        right = [rng.randint(-100, 100) for _ in range(n)]
        if rng.random() < 0.7:
            for k in rng.sample(range(n), rng.randint(1, n // 2 + 1)):
                left[k] = None
        if rng.random() < 0.7:
            for k in rng.sample(range(n), rng.randint(1, n // 2 + 1)):
                right[k] = None
        got = analyze(left, right)
        for rec in got:
            d = rec["lag"]
            pairs = [(left[i], right[i + d])
                     for i in range(n)
                     if 0 <= i + d < n
                     and left[i] is not None and right[i + d] is not None]
            if len(pairs) >= 2:
                dc = decimal_corr(pairs)
                if dc is None:
                    assert rec["corr"] is None, (left, right, rec)
                else:
                    err = abs(rec["corr"] - dc)
                    assert err <= 1e-12, (left, right, rec, dc, err)

    # --- precision stress: large offset, tiny relative fluctuations ------
    # y = 10**6 + tiny variations, x = shifted copy; correlation must stay
    # extremely close to 1 even though the means are huge relative to variance.
    n = 400
    base = 1_000_000
    left = [base + (i % 7) - 3 for i in range(n)]
    right = [None] * 20 + left[:-20]
    got = analyze(left, right)
    zero = next(r for r in got if r["lag"] == 0)
    assert zero["count"] == n - 20
    # right[k] = left[k-20]; the pairing left[i] with right[i+d] therefore
    # aligns identical values at lag d = +20, where corr must be exactly 1.
    aligned = next(r for r in got if r["lag"] == 20)
    assert aligned["count"] == n - 20
    assert abs(aligned["corr"] - 1.0) <= 1e-12, aligned
    ref = reference(left, right)
    assert_records_match(got, ref, left, right, "precision stress")

    # Values at the allowed magnitude extremes, including negatives.
    n = 64
    left = [rng.choice([-10**6, 10**6, None]) for _ in range(n)]
    right = [rng.choice([-10**6, 10**6, None]) for _ in range(n)]
    got = analyze(left, right)
    ref = reference(left, right)
    assert_records_match(got, ref, left, right, "extreme magnitudes")

    # --- medium random fuzz vs the independent reference -----------------
    for n in (63, 64, 65, 127, 200):
        left = [rng.randint(-10**6, 10**6) if rng.random() < 0.85 else None
                for _ in range(n)]
        right = [rng.randint(-10**6, 10**6) if rng.random() < 0.85 else None
                 for _ in range(n)]
        got = analyze(left, right)
        ref = reference(left, right)
        assert_records_match(got, ref, left, right, f"fuzz n={n}")

    print("all verification checks passed")


if __name__ == "__main__":
    run()
    print("done")
