# lagcorr

Cross-lag Pearson correlation for two equally-sized sampled channels with
missing samples, implemented with **only the Python standard library**.

```python
from lagcorr import analyze

analyze([1, 2], [2, 4])
# [{'lag': -1, 'count': 1, 'corr': None},
#  {'lag':  0, 'count': 2, 'corr': 1.0},
#  {'lag':  1, 'count': 1, 'corr': None}]
```

## Semantics

`analyze(left, right)` accepts two equal-length lists (`0 <= n <= 32768`) whose
elements are either `None` (missing) or non-boolean integers with
`|value| <= 10**6`. The inputs are read only and never modified.

It returns one dictionary per lag, in ascending lag order:

```text
{"lag": d, "count": c, "corr": r}
```

At lag `d`, sample `left[i]` is paired with `right[i + d]`. Only pairs whose
two indices are in range and whose two values are both present (not `None`)
count. For non-empty input every lag from `-(n-1)` through `n-1` is emitted;
empty input returns `[]`.

* `count` is the exact number of valid pairs.
* `corr` is the Pearson correlation coefficient of those pairs. It is
  `None` when fewer than two pairs exist or when the values on either side
  have zero variance (a constant side). Otherwise the absolute error versus
  the exact coefficient is below `1e-10`.

## Algorithm

### 1. Pearson correlation from six per-lag sums

For the valid pairs at one lag, let `c` be their count and let `Sx, Sy, Sxx,
Syy, Sxy` be the sums of x, y, x², y² and x·y over those pairs. With centered
sums kept as integers,

```text
num = c*Sxy - Sx*Sy
vx  = c*Sxx - Sx*Sx
vy  = c*Syy - Sy*Sy
corr = num / sqrt(vx * vy)          # undefined when c < 2 or vx == 0 or vy == 0
```

Every one of the six sums is, across all lags simultaneously, a
**cross-correlation** of masked sequences. Missing samples are encoded with a
0/1 mask and zeroed value limbs, so invalid pairs contribute nothing.

### 2. Cross-correlation as convolution (no per-lag scan)

The left channel is reversed. With `l'[q] = l[n-1-q]`, the ordinary
convolution satisfies

```text
conv(l', r)[m] = sum_i l[i] * r[i + d],   m = (n-1) + d
```

so convolution slot `m = (n-1) + d` holds exactly the lag-`d` pairing. All
sums for all lags therefore come from a constant number of convolutions, with
no loop that scans every lag.

Convolutions use an in-place radix-2 iterative Cooley–Tukey FFT on complex
lists. The FFT size is the next power of two at least `2n-1`. Two real
forward transforms share one complex FFT through the standard real-pair
identity, and two independent real convolutions share one inverse FFT by
packing their spectrum products into the real and imaginary parts of one
complex array.

### 3. Exact integer recovery with 11-bit limbs

A direct FFT on values up to `10**6` produces convolution coefficients up to
`n * 10**12 ≈ 3.3e16`, far beyond the 2^53 exact-integer range of a float;
rounding could not be trusted and the centered-sum subtraction would
catastrophically cancel.

Instead each present value is first shifted by `10**6` (making it
non-negative) and split into base-`B = 2**11 = 2048` limbs, each at most
2047; the mask is itself a 0/1 limb. Every convolution coefficient is then
bounded by

```text
n * 2047**2  <  32768 * 4_190_209  ≈ 1.38e11
```

The FFT round-off per output coefficient is bounded by the standard FFT error
model `~ eps * log2(N) * sum|a*b|`, which at the maximum size is on the order
of `1e-4` — more than six orders of magnitude below `0.5`. `round()` on each
coefficient therefore restores the **exact integer** convolution. The shifted
limb convolutions are recombined algebraically and the `10**6` shift is
undone, so all six Pearson sums are exact integers. Only the final
`sqrt`/division is floating point.

### 4. Final floating-point precision

The integer centered sums are converted to float only at the last step. Their
magnitude is at most `c * 10**12 ≈ 3.3e16`, represented with a relative error
below `~1e-16`. The single square root and division add another `~1e-16`,
giving a total relative error around `1e-15`, comfortably inside the required
`1e-10` absolute tolerance even for large offsets with tiny fluctuations (the
test suite includes a signal centered at `10**6` whose variation is only
±3). Results are clamped to `[-1, 1]` solely to remove a possible last-ULP
float overshoot; genuinely constant sides still report `corr = None`.

## Complexity

Let `N` be the FFT size, `N = O(n)` (the next power of two ≥ `2n-1`).

* A constant number of FFTs is performed (six forward, eight inverse), each
  `O(N log N)`; the pointwise products, limb splits, reassembly and the final
  per-lag assembly are `O(n)`. **Total time `O(n log(n+1))`.**
* All arrays have length `N = O(n)` or `2n-1 = O(n)` and only a constant
  number are live. **Total space `O(n+1)`.**
* There is no `O(n²)` per-lag scan anywhere in the implementation.

Measured on this machine (Python 3.11, standard-library FFT) at the maximum
`n = 32768`: about 2.2 s wall clock and roughly 157 MiB peak Python-allocated
memory; doubling `n` grows runtime by the `O(n log n)` factor of about 2.1–2.2.
Run `python3 bench/benchmark.py` to reproduce.

## Repository layout

```text
lagcorr/              package (analyze lives in lagcorr/lagcorr.py)
tests/verify.py       independent O(n^2) + Decimal cross-checks
bench/benchmark.py    doubling-size timing and peak-memory measurements
demo.py               missing-sample offset analysis demonstration
.pointwise-runtime.json   reproducible command records
```

## Reproducing

```bash
python3 tests/verify.py      # correctness (independent enumeration, Decimal)
python3 bench/benchmark.py   # timing at max size and doubling sizes + memory
python3 demo.py              # missing-sample lag analysis walk-through
```

All commands run with `-I` (isolated mode) as recorded in
`.pointwise-runtime.json`; no third-party packages are installed or needed.
