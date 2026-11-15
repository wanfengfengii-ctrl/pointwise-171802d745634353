"""lagcorr -- cross-lag Pearson correlation with missing samples.

Only the Python standard library is used.  ``analyze(left, right)`` returns one
record ``{"lag", "count", "corr"}`` per lag in ascending order, where at lag
``d`` the paired samples are ``left[i]`` and ``right[i + d]``; pairs with an
out-of-range index or a ``None`` on either side are ignored.

The six per-lag sums needed for Pearson correlation are all cross-correlations,
hence convolutions of the (masked) input sequences.  They are evaluated in
``O(n log n)`` with an in-place radix-2 Cooley-Tukey FFT instead of scanning
every lag.  To recover the integer sums *exactly* from floating point FFTs,
every operand is split into 11-bit limbs: each convolution coefficient is then
small enough that the FFT round-off is far below 0.5 and ``round`` restores
the exact integer (see README.md for the full error budget).

FFT packing: two real forward transforms share one complex FFT via the standard
real-pair identity, and two independent real convolutions share one inverse
FFT by packing their (complex) spectrum products into the real and imaginary
parts of a single complex array.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

__all__ = ["analyze"]

# Input values satisfy |v| <= 10**6; shifting by this offset makes every
# present sample non-negative so it can be split into non-negative limbs.
_SHIFT = 1_000_000
# Limb width.  With B = 2**11 every limb (and mask) is <= 2047, so a
# convolution coefficient is at most n * 2047**2 < 1.38e11; FFT round-off stays
# several orders of magnitude below 0.5 (see README.md).
_BITS = 11
_BASE = 1 << _BITS          # 2048
_MASK = _BASE - 1           # 2047


# ---------------------------------------------------------------------------
# FFT (radix-2 iterative Cooley-Tukey, complex lists, standard library only)
# ---------------------------------------------------------------------------
def _fft(a: List[complex], invert: bool) -> None:
    """In-place radix-2 DFT; ``len(a)`` must be a power of two."""
    n = len(a)

    # Bit-reversal permutation.
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]

    # Standard sign convention: forward uses the negative exponent, the
    # inverse uses the positive one and divides by n.
    sign = -1.0 if not invert else 1.0
    length = 2
    while length <= n:
        half = length >> 1
        ang = sign * 2.0 * math.pi / length
        wlen = complex(math.cos(ang), math.sin(ang))
        for i in range(0, n, length):
            w = 1 + 0j
            for jj in range(half):
                u = a[i + jj]
                v = a[i + jj + half] * w
                a[i + jj] = u + v
                a[i + jj + half] = u - v
                w *= wlen
        length <<= 1

    if invert:
        inv_n = 1.0 / n
        for i in range(n):
            a[i] *= inv_n


def _real_pair_forward(x: List[int], y: List[int], size: int, n: int
                       ) -> Tuple[List[complex], List[complex]]:
    """Spectra X, Y of two real arrays using one complex forward FFT."""
    z = [complex(x[k], y[k]) for k in range(n)]
    z += [0j] * (size - n)
    _fft(z, False)
    xs = [0j] * size
    ys = [0j] * size
    # Bin 0 and the Nyquist bin are real for real inputs; z[k] there already
    # equals X[k] + i*Y[k] with X, Y purely real.
    xs[0] = complex(z[0].real, 0.0)
    ys[0] = complex(z[0].imag, 0.0)
    half = size >> 1
    if half > 0:
        xs[half] = complex(z[half].real, 0.0)
        ys[half] = complex(z[half].imag, 0.0)
    for k in range(1, half):
        zc = z[size - k].conjugate()
        xs[k] = 0.5 * (z[k] + zc)
        ys[k] = -0.5j * (z[k] - zc)
        # Conjugate symmetry of real transforms.
        xs[size - k] = xs[k].conjugate()
        ys[size - k] = ys[k].conjugate()
    return xs, ys


def _paired_inverse(p1: List[complex], p2: List[complex], size: int,
                    n_out: int) -> Tuple[List[int], List[int]]:
    """Two exact integer convolutions through one inverse FFT.

    The two (complex) spectrum products are packed as the real and imaginary
    parts of a single complex array; linearity of the inverse FFT separates
    the two real convolution results.
    """
    # Pack spectrum product p1 as the real component and p2 as the imaginary
    # component of one array; the inverse FFT separates the two real
    # convolutions by linearity.
    h = [complex(p1[k].real + 1j * p1[k].imag,
                 p2[k].real + 1j * p2[k].imag) for k in range(size)]
    _fft(h, True)
    ab = [int(round(h[k].real)) for k in range(n_out)]
    cd = [int(round(h[k].imag)) for k in range(n_out)]
    return ab, cd


def _single_inverse(p: List[complex], size: int, n_out: int) -> List[int]:
    h = list(p)
    _fft(h, True)
    return [int(round(h[k].real)) for k in range(n_out)]


def _mul(a: List[complex], b: List[complex], size: int) -> List[complex]:
    return [a[k] * b[k] for k in range(size)]


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------
def _build_side(values: Sequence[Optional[int]], reverse: bool
                ) -> Tuple[List[int], List[int], List[int]]:
    """Return (mask, low limb, high limb) for one channel.

    A missing sample (``None``) yields a zero mask bit and zero limbs.
    Present values are shifted by ``_SHIFT`` before splitting into 11-bit
    limbs.  ``reverse=True`` reverses the arrays; the L channel is reversed so
    that convolution slot m = (n - 1) + d holds the lag-d pairing
    left[i] / right[i + d].
    """
    n = len(values)
    idx = range(n - 1, -1, -1) if reverse else range(n)
    mask = [0] * n
    lo = [0] * n
    hi = [0] * n
    for out, i in enumerate(idx):
        v = values[i]
        if v is None:
            continue
        sv = v + _SHIFT
        mask[out] = 1
        lo[out] = sv & _MASK
        hi[out] = sv >> _BITS
    return mask, lo, hi


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def analyze(left: Sequence[Optional[int]],
            right: Sequence[Optional[int]]) -> List[dict]:
    """Cross-lag Pearson correlation of two equal-length sampled channels.

    See the module docstring for semantics.  The inputs are never mutated.
    """
    n = len(left)
    if n == 0:
        return []
    if len(right) != n:
        raise ValueError("left and right must have equal length")

    # Reverse the L channel: conv(reverse(l), r)[m] = sum_i l[i]*r[i+d] at
    # slot m = (n - 1) + d, exactly the specification's lag-d pairing.
    ml, llo, lhi = _build_side(left, reverse=True)
    mr, rlo, rhi = _build_side(right, reverse=False)

    # Next power of two >= 2n - 1.
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    n_out = 2 * n - 1

    # Pointwise squared-limb arrays (linear-size work).
    loL_sq = [v * v for v in llo]
    loL_hiL = [a * b for a, b in zip(llo, lhi)]
    hiL_sq = [v * v for v in lhi]
    loR_sq = [v * v for v in rlo]
    loR_hiR = [a * b for a, b in zip(rlo, rhi)]
    hiR_sq = [v * v for v in rhi]

    # Six packed forward FFTs cover all twelve real input arrays.  Each call
    # returns the two full complex spectra of its real/imaginary inputs.
    ML, LO_L = _real_pair_forward(ml, llo, size, n)
    HL, LO_R = _real_pair_forward(lhi, rlo, size, n)
    MR, HR = _real_pair_forward(mr, rhi, size, n)
    SQ_LOL, SQ_LHLH = _real_pair_forward(loL_sq, loL_hiL, size, n)
    SQ_HHL, SQ_LOR = _real_pair_forward(hiL_sq, loR_sq, size, n)
    SQ_LRHR, SQ_HHR = _real_pair_forward(loR_hiR, hiR_sq, size, n)

    # 15 limb convolutions.  Writing L' = loL + B*hiL and R' = loR + B*hiR,
    # an L-side quantity is always convolved with an R-side quantity (the
    # mask array stands for the constant value 1):
    #   C1 = M_L*M_R                            (pair counts)
    #   C2 = loL*M_R,  C3 = M_L*loR             (shifted single-side sums)
    #   sum L'R' = loL*loR
    #              + B*(loL*hiR + hiL*loR) + B^2*hiL*hiR
    #   sum (L')^2 = (loL^2)*M_R
    #              + 2B*(loL*hiL)*M_R + B^2*(hiL^2)*M_R
    #   sum (R')^2 = M_L*(loR^2)
    #              + 2B*M_L*(loR*hiR) + B^2*M_L*(hiR^2)
    # 14 share 7 inverse FFTs; the last runs alone.
    pair_specs = [
        (_mul(ML, MR, size), _mul(LO_L, MR, size)),          # C1, C2
        (_mul(ML, LO_R, size), _mul(LO_L, LO_R, size)),      # C3, loL*loR
        (_mul(LO_L, HR, size), _mul(HL, LO_R, size)),        # loL*hiR, hiL*loR
        (_mul(HL, HR, size), _mul(HL, MR, size)),            # hiL*hiR, hiL*MR
        (_mul(ML, HR, size), _mul(SQ_LOL, MR, size)),        # ML*hiR, loL^2*MR
        (_mul(SQ_LHLH, MR, size), _mul(SQ_HHL, MR, size)),   # L cross, hiL^2
        (_mul(ML, SQ_LOR, size), _mul(ML, SQ_LRHR, size)),   # R lo sq, cross
    ]
    convs: List[List[int]] = []
    for p1, p2 in pair_specs:
        convs.extend(_paired_inverse(p1, p2, size, n_out))
    convs.append(_single_inverse(_mul(ML, SQ_HHR, size), size, n_out))

    (c1, c2, c3, c_lolo, c_lohi_r, c_hilo_l, c_hihi, c_himr, c_mhi,
     c_ll2, c_lhlh, c_hh2, c_rr2, c_rlrh, c_rh2) = convs

    # Reassemble the shifted-value convolutions from their limbs.
    b = _BASE
    b2 = b * b
    c5 = [c_ll2[k] + 2 * b * c_lhlh[k] + b2 * c_hh2[k]
          for k in range(n_out)]
    c6 = [c_rr2[k] + 2 * b * c_rlrh[k] + b2 * c_rh2[k]
          for k in range(n_out)]
    sum_l = [c2[k] + b * c_himr[k] for k in range(n_out)]
    sum_r = [c3[k] + b * c_mhi[k] for k in range(n_out)]
    sum_lr = [c_lolo[k]
              + b * (c_lohi_r[k] + c_hilo_l[k])
              + b2 * c_hihi[k] for k in range(n_out)]

    # Undo the value shift L = L' - S, R = R' - S.  All sums stay exact
    # integers; only the final sqrt/divide below is floating point.
    s = _SHIFT
    s2 = s * s
    cnt = c1
    sx = [sum_l[k] - s * cnt[k] for k in range(n_out)]
    sy = [sum_r[k] - s * cnt[k] for k in range(n_out)]
    sxy = [sum_lr[k] - s * (sum_l[k] + sum_r[k]) + s2 * cnt[k]
           for k in range(n_out)]
    sxx = [c5[k] - 2 * s * sum_l[k] + s2 * cnt[k] for k in range(n_out)]
    syy = [c6[k] - 2 * s * sum_r[k] + s2 * cnt[k] for k in range(n_out)]

    results: List[dict] = []
    n_mid = n - 1
    # Slot m = (n - 1) + d holds lag d, ascending from -(n - 1) to n - 1.
    for k in range(n_out):
        d = k - n_mid
        c = cnt[k]
        corr: Optional[float]
        if c < 2:
            corr = None
        else:
            # Exact-integer centered sums; only the final step is floating.
            num = c * sxy[k] - sx[k] * sy[k]
            vx = c * sxx[k] - sx[k] * sx[k]
            vy = c * syy[k] - sy[k] * sy[k]
            if vx <= 0 or vy <= 0:
                corr = None
            else:
                corr = num / math.sqrt(vx * vy)
                # Guard against only the last ulp of float overshoot.
                if corr > 1.0:
                    corr = 1.0
                elif corr < -1.0:
                    corr = -1.0
        results.append({"lag": d, "count": c, "corr": corr})
    return results
