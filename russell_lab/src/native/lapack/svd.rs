//! Native singular value decomposition: `dgesvd` and the unblocked pieces it is
//! built from.
//!
//! # The pipeline
//!
//! | Step | LAPACK | What it does |
//! |---|---|---|
//! | 1 | `dgebd2` | orthogonal reduction to bidiagonal form, `Q^T A P = B` |
//! | 2 | `dorgbr` | forms `Q` and `P` explicitly from the reflectors `dgebd2` left behind |
//! | 3 | `dbdsqr` | implicit QR on the bidiagonal, accumulating the singular vectors |
//!
//! `dgesvd` scales the matrix first when its norm falls outside the safe range,
//! exactly as the reference does, and undoes that scaling on the singular values
//! at the end.
//!
//! # Which path
//!
//! The reference has several paths. For `m >= n` it either bidiagonalizes `A`
//! directly or, when `m` is much larger than `n`, first reduces it with a QR
//! factorization; for `m < n` it does the transposed analog with an LQ
//! factorization. Only the direct bidiagonalization path is implemented here. The
//! QR/LQ pre-reduction is a performance optimization: it changes the rounding,
//! not the singular values, and the singular values are what the parity test
//! compares. Both shapes are supported; the `m < n` case exercises `dbdsqr`'s
//! lower-bidiagonal branch.
//!
//! # `dbdsqr` without the `dlasq1` fast path
//!
//! `dbdsqr` has a fast path (`ncvt = nru = ncc = 0`) that calls `dlasq1`. It is
//! never taken from here, because `dgesvd` always asks for at least one set of
//! singular vectors, so only the rotation-accumulating path is implemented. That
//! path is also what the reference uses for the small matrices this crate sees:
//! the blocked `dorgqr`/`dorglq` reduce to `dorg2r`/`dorgl2` for a single block,
//! and `dbdsqr`'s rotations are unblocked.
//!
//! Everything is column-major: element `(i, j)` of an `m x n` matrix is
//! `data[i + j * m]`. Indices in the transcriptions below are kept 1-based where
//! the Fortran is, so `d[i - 1]` is Fortran's `D(I)`; that makes the comparison
//! against the reference line by line.

use super::LapackError;
use super::auxiliary::dlascl_g;
use super::eig::{dlarfg, dlartg};

/// `DLAMCH('E')`: the relative machine precision, `eps/2` (LAPACK assumes
/// rounding).
const EPS: f64 = f64::EPSILON * 0.5;
/// `DLAMCH('P')`: the precision, `eps * base`.
const PREC: f64 = f64::EPSILON;
/// `DLAMCH('S')`: the safe minimum, the smallest normal positive number.
const SAFMIN: f64 = f64::MIN_POSITIVE;

/// Fortran's `SIGN(a, b)`: `|a|` with the sign of `b`, and `+` for `b >= 0`.
///
/// Not `f64::signum`, which follows the sign bit of `-0.0` and would disagree
/// with Fortran on a zero argument.
#[inline]
fn fsign(a: f64, b: f64) -> f64 {
    if b >= 0.0 { a.abs() } else { -a.abs() }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Reflectors
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// `DLARF1F`, left: `C := (I - tau v v^T) C` with the leading `1` of `v` implicit.
///
/// `v_tail` is `v[1..]`, so `v = [1, v_tail]`; `c` is `m x n` column-major with
/// leading dimension `ldc`. This is the `DLARF` update, written in the "1F" form
/// the reference uses so that the reflector's `1` never has to be placed in the
/// matrix (where `dgebd2` keeps `beta` on the diagonal instead).
fn larf_left(tau: f64, v_tail: &[f64], c: &mut [f64], m: usize, n: usize, ldc: usize) {
    debug_assert_eq!(v_tail.len() + 1, m);
    if tau == 0.0 || m == 0 || n == 0 {
        return;
    }
    let mut w = vec![0.0; n];
    for j in 0..n {
        let mut acc = c[j * ldc];
        for i in 1..m {
            acc += v_tail[i - 1] * c[i + j * ldc];
        }
        w[j] = acc;
    }
    for j in 0..n {
        // `v = [1, v_tail]`, so row 0 takes no tail factor.
        c[j * ldc] -= tau * w[j];
        for i in 1..m {
            c[i + j * ldc] -= tau * w[j] * v_tail[i - 1];
        }
    }
}

/// `DLARF1F`, right: `C := C (I - tau v v^T)` with the leading `1` of `v` implicit.
fn larf_right(tau: f64, v_tail: &[f64], c: &mut [f64], m: usize, n: usize, ldc: usize) {
    debug_assert_eq!(v_tail.len() + 1, n);
    if tau == 0.0 || m == 0 || n == 0 {
        return;
    }
    let mut w = vec![0.0; m];
    for i in 0..m {
        let mut acc = c[i];
        for j in 1..n {
            acc += c[i + j * ldc] * v_tail[j - 1];
        }
        w[i] = acc;
    }
    for i in 0..m {
        c[i] -= tau * w[i];
        for j in 1..n {
            c[i + j * ldc] -= tau * w[i] * v_tail[j - 1];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Scaling
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Plane rotations and the 2x2 kernels
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// `DLASR` with `PIVOT = 'V'`: apply a sequence of plane rotations.
///
/// `side_left` selects `L` (rotates rows) or `R` (rotates columns); `forward`
/// selects `F` (ascending `j`) or `B` (descending). Only `PIVOT = 'V'` is used by
/// `dbdsqr`, so that is the only pivot form implemented.
fn dlasr_v(side_left: bool, forward: bool, m: usize, n: usize, c: &[f64], s: &[f64], a: &mut [f64], lda: usize) {
    if m == 0 || n == 0 {
        return;
    }
    if side_left {
        let count = m - 1;
        for jj in 0..count {
            let j = if forward { jj } else { count - 1 - jj };
            let ctemp = c[j];
            let stemp = s[j];
            if ctemp != 1.0 || stemp != 0.0 {
                for i in 0..n {
                    let temp = a[(j + 1) + i * lda];
                    a[(j + 1) + i * lda] = ctemp * temp - stemp * a[j + i * lda];
                    a[j + i * lda] = stemp * temp + ctemp * a[j + i * lda];
                }
            }
        }
    } else {
        let count = n - 1;
        for jj in 0..count {
            let j = if forward { jj } else { count - 1 - jj };
            let ctemp = c[j];
            let stemp = s[j];
            if ctemp != 1.0 || stemp != 0.0 {
                for i in 0..m {
                    let temp = a[i + (j + 1) * lda];
                    a[i + (j + 1) * lda] = ctemp * temp - stemp * a[i + j * lda];
                    a[i + j * lda] = stemp * temp + ctemp * a[i + j * lda];
                }
            }
        }
    }
}

/// `DLAS2`: the singular values of `[f g; 0 h]`, returned as `(ssmin, ssmax)`.
///
/// No vectors, so no square roots of differences: this is the cheap form used by
/// `dbdsqr` only to choose a shift.
pub(crate) fn dlas2(f: f64, g: f64, h: f64) -> (f64, f64) {
    let fa = f.abs();
    let ga = g.abs();
    let ha = h.abs();
    let fhmn = fa.min(ha);
    let fhmx = fa.max(ha);
    if fhmn == 0.0 {
        let ssmin = 0.0;
        let ssmax = if fhmx == 0.0 {
            ga
        } else {
            fhmx.max(ga) * (1.0 + (fhmx.min(ga) / fhmx.max(ga)).powi(2)).sqrt()
        };
        (ssmin, ssmax)
    } else if ga < fhmx {
        let a_s = 1.0 + fhmn / fhmx;
        let a_t = (fhmx - fhmn) / fhmx;
        let a_u = (ga / fhmx).powi(2);
        let c = 2.0 / ((a_s * a_s + a_u).sqrt() + (a_t * a_t + a_u).sqrt());
        (fhmn * c, fhmx / c)
    } else {
        let a_u = fhmx / ga;
        if a_u == 0.0 {
            ((fhmn * fhmx) / ga, ga)
        } else {
            let a_s = 1.0 + fhmn / fhmx;
            let a_t = (fhmx - fhmn) / fhmx;
            let c = 1.0 / ((1.0 + (a_s * a_u).powi(2)).sqrt() + (1.0 + (a_t * a_u).powi(2)).sqrt());
            let ssmin = (fhmn * c) * a_u * 2.0;
            (ssmin, ga / (c + c))
        }
    }
}

/// `DLASV2`: the SVD of `[f g; 0 h]`, singular values and both rotation pairs.
///
/// Returns `(ssmin, ssmax, snr, csr, snl, csl)`, in the reference's argument
/// order. `(csr, snr)` is the rotation from the right and `(csl, snl)` the one
/// from the left; `dbdsqr` applies them to `VT` and `U` respectively.
pub(crate) fn dlasv2(f: f64, g: f64, h: f64) -> (f64, f64, f64, f64, f64, f64) {
    let half = 0.5;
    let mut ft = f;
    let mut fa = ft.abs();
    let mut ht = h;
    let mut ha = ht.abs();

    // PMAX records which of f, g, h is the largest in magnitude; TSIGN at the end
    // needs it.
    let mut pmax = 1;
    let swap = ha > fa;
    if swap {
        pmax = 3;
        std::mem::swap(&mut ft, &mut ht);
        std::mem::swap(&mut fa, &mut ha);
    }

    let gt = g;
    let ga = gt.abs();

    let mut ssmin = 0.0;
    let mut ssmax = 0.0;
    let mut clt = 0.0;
    let mut crt = 0.0;
    let mut slt = 0.0;
    let mut srt = 0.0;

    if ga == 0.0 {
        // The matrix is already diagonal.
        ssmin = ha;
        ssmax = fa;
        clt = 1.0;
        crt = 1.0;
        slt = 0.0;
        srt = 0.0;
    } else {
        let mut gasmal = true;
        if ga > fa {
            pmax = 2;
            if (fa / ga) < EPS {
                // `f` is negligible: the rotation is a permutation.
                gasmal = false;
                ssmax = ga;
                ssmin = if ha > 1.0 { fa / (ga / ha) } else { (fa / ga) * ha };
                clt = 1.0;
                slt = ht / gt;
                srt = 1.0;
                crt = ft / gt;
            }
        }
        if gasmal {
            // Normal case: `ga` is not huge relative to `fa`.
            let d = fa - ha;
            let l = if d == fa { 1.0 } else { d / fa };
            let m = gt / ft;
            let t = 2.0 - l;
            let mm = m * m;
            let tt = t * t;
            let s = (tt + mm).sqrt();
            let r = if l == 0.0 { m.abs() } else { (l * l + mm).sqrt() };
            let a = half * (s + r);
            ssmin = ha / a;
            ssmax = fa * a;
            let t2 = if mm == 0.0 {
                if l == 0.0 {
                    fsign(2.0, ft) * fsign(1.0, gt)
                } else {
                    gt / fsign(d, ft) + m / t
                }
            } else {
                (m / (s + t) + m / (r + l)) * (1.0 + a)
            };
            let l2 = (t2 * t2 + 4.0).sqrt();
            crt = 2.0 / l2;
            srt = t2 / l2;
            clt = (crt + srt * m) / a;
            slt = (ht / ft) * srt / a;
        }
    }

    let (csr, snr, csl, snl);
    if swap {
        csl = srt;
        snl = crt;
        csr = slt;
        snr = clt;
    } else {
        csl = clt;
        snl = slt;
        csr = crt;
        snr = srt;
    }

    // Correct the signs so that `ssmax` and `ssmin` are the true singular values.
    let tsign = if pmax == 1 {
        fsign(1.0, csr) * fsign(1.0, csl) * fsign(1.0, f)
    } else if pmax == 2 {
        fsign(1.0, snr) * fsign(1.0, csl) * fsign(1.0, g)
    } else {
        fsign(1.0, snr) * fsign(1.0, snl) * fsign(1.0, h)
    };
    ssmax = fsign(ssmax, tsign);
    ssmin = fsign(ssmin, tsign * fsign(1.0, f) * fsign(1.0, h));

    (ssmin, ssmax, snr, csr, snl, csl)
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Bidiagonalization
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// `DGEBD2`: reduce an `m x n` matrix to bidiagonal form, unblocked.
///
/// On exit `a` (leading dimension `lda`) holds the bidiagonal matrix on its
/// diagonal and first off-diagonal, with the reflectors of `Q` below the diagonal
/// and those of `P` above it. `d` and `e` receive the bidiagonal entries, `tauq`
/// and `taup` the reflector scalars.
pub fn dgebd2(
    a: &mut [f64],
    lda: usize,
    m: usize,
    n: usize,
    d: &mut [f64],
    e: &mut [f64],
    tauq: &mut [f64],
    taup: &mut [f64],
) {
    let zero = 0.0;
    if m >= n {
        // Reduce to upper bidiagonal form.
        for i in 1..=n {
            // H(i) annihilates A(i+1:m, i).
            let alpha = a[(i - 1) + (i - 1) * lda];
            let x: Vec<f64> = ((i + 1)..=m).map(|r| a[(r - 1) + (i - 1) * lda]).collect();
            let (beta, tau, v) = dlarfg(alpha, &x);
            a[(i - 1) + (i - 1) * lda] = beta;
            tauq[i - 1] = tau;
            for (k, r) in ((i + 1)..=m).enumerate() {
                a[(r - 1) + (i - 1) * lda] = v[k + 1];
            }
            d[i - 1] = beta;

            if i < n {
                // Apply H(i) from the left to A(i:m, i+1:n).
                let start = (i - 1) + i * lda;
                larf_left(tau, &v[1..], &mut a[start..], m - i + 1, n - i, lda);

                // G(i) annihilates A(i, i+2:n).
                let alpha = a[(i - 1) + i * lda];
                let x: Vec<f64> = ((i + 2)..=n).map(|c| a[(i - 1) + (c - 1) * lda]).collect();
                let (beta, tau, v) = dlarfg(alpha, &x);
                a[(i - 1) + i * lda] = beta;
                taup[i - 1] = tau;
                for (k, c) in ((i + 2)..=n).enumerate() {
                    a[(i - 1) + (c - 1) * lda] = v[k + 1];
                }
                e[i - 1] = beta;

                // Apply G(i) from the right to A(i+1:m, i+1:n).
                let start = i + i * lda;
                larf_right(tau, &v[1..], &mut a[start..], m - i, n - i, lda);
            } else {
                taup[i - 1] = zero;
            }
        }
    } else {
        // Reduce to lower bidiagonal form.
        for i in 1..=m {
            // G(i) annihilates A(i, i+1:n).
            let alpha = a[(i - 1) + (i - 1) * lda];
            let x: Vec<f64> = ((i + 1)..=n).map(|c| a[(i - 1) + (c - 1) * lda]).collect();
            let (beta, tau, v) = dlarfg(alpha, &x);
            a[(i - 1) + (i - 1) * lda] = beta;
            taup[i - 1] = tau;
            for (k, c) in ((i + 1)..=n).enumerate() {
                a[(i - 1) + (c - 1) * lda] = v[k + 1];
            }
            d[i - 1] = beta;

            if i < m {
                // Apply G(i) from the right to A(i+1:m, i:n).
                let start = i + (i - 1) * lda;
                larf_right(tau, &v[1..], &mut a[start..], m - i, n - i + 1, lda);

                // H(i) annihilates A(i+2:m, i).
                let alpha = a[i + (i - 1) * lda];
                let x: Vec<f64> = ((i + 2)..=m).map(|r| a[(r - 1) + (i - 1) * lda]).collect();
                let (beta, tau, v) = dlarfg(alpha, &x);
                a[i + (i - 1) * lda] = beta;
                tauq[i - 1] = tau;
                for (k, r) in ((i + 2)..=m).enumerate() {
                    a[(r - 1) + (i - 1) * lda] = v[k + 1];
                }
                e[i - 1] = beta;

                // Apply H(i) from the left to A(i+1:m, i+1:n).
                let start = i + i * lda;
                larf_left(tau, &v[1..], &mut a[start..], m - i, n - i, lda);
            } else {
                tauq[i - 1] = zero;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Generating the orthogonal factors
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// `DORG2R`: generate the first `n` columns of an `m x m` orthogonal matrix `Q`
/// from the reflectors `dgebd2` stored below the diagonal of `a`.
fn dorg2r(m: usize, n: usize, k: usize, a: &mut [f64], lda: usize, tau: &[f64]) {
    // Fortran's IF( N.LE.0 ); for a `usize` that is exactly `n == 0`.
    if n == 0 {
        return;
    }
    // Set the columns after the first k to the identity.
    for j in (k + 1)..=n {
        for l in 1..=m {
            a[(l - 1) + (j - 1) * lda] = 0.0;
        }
        a[(j - 1) + (j - 1) * lda] = 1.0;
    }
    // Form the first k columns in reverse order.
    for i in (1..=k).rev() {
        if i < n {
            let v_tail: Vec<f64> = ((i + 1)..=m).map(|r| a[(r - 1) + (i - 1) * lda]).collect();
            let start = (i - 1) + i * lda;
            larf_left(tau[i - 1], &v_tail, &mut a[start..], m - i + 1, n - i, lda);
        }
        if i < m {
            for l in (i + 1)..=m {
                a[(l - 1) + (i - 1) * lda] *= -tau[i - 1];
            }
        }
        a[(i - 1) + (i - 1) * lda] = 1.0 - tau[i - 1];
        for l in 1..=(i - 1) {
            a[(l - 1) + (i - 1) * lda] = 0.0;
        }
    }
}

/// `DORGL2`: generate the first `m` rows of an `m x n` matrix `Q` with orthogonal
/// rows from the reflectors `dgebd2` stored above the diagonal of `a`.
fn dorgl2(m: usize, n: usize, k: usize, a: &mut [f64], lda: usize, tau: &[f64]) {
    // Fortran's IF( M.LE.0 ); for a `usize` that is exactly `m == 0`.
    if m == 0 {
        return;
    }
    if k < m {
        for j in 1..=n {
            for l in (k + 1)..=m {
                a[(l - 1) + (j - 1) * lda] = 0.0;
            }
            if j > k && j <= m {
                a[(j - 1) + (j - 1) * lda] = 1.0;
            }
        }
    }
    for i in (1..=k).rev() {
        if i < n {
            let v_tail: Vec<f64> = ((i + 1)..=n).map(|c| a[(i - 1) + (c - 1) * lda]).collect();
            if i < m {
                let start = i + (i - 1) * lda;
                larf_right(tau[i - 1], &v_tail, &mut a[start..], m - i, n - i + 1, lda);
            }
            for c in (i + 1)..=n {
                a[(i - 1) + (c - 1) * lda] *= -tau[i - 1];
            }
        }
        a[(i - 1) + (i - 1) * lda] = 1.0 - tau[i - 1];
        for l in 1..=(i - 1) {
            a[(i - 1) + (l - 1) * lda] = 0.0;
        }
    }
}

/// `DORGBR`: form `Q` (`want_q`) or `P` from the reflectors `dgebd2` left in `a`.
///
/// `a` is `m x n`; for `want_q` the reflectors sit below the diagonal and `Q` is
/// generated by `dorg2r`, for `P` they sit above and `dorgl2` is used. When
/// the number of reflectors `k` does not match the shape `dorg2r`/`dorgl2`
/// expect, the reference shifts the reflectors one row or column first, and that
/// shift is transcribed here.
pub fn dorgbr(want_q: bool, m: usize, n: usize, k: usize, a: &mut [f64], lda: usize, tau: &[f64]) {
    if m == 0 || n == 0 {
        return;
    }
    if want_q {
        if m >= k {
            dorg2r(m, n, k, a, lda, tau);
        } else {
            // Shift the reflectors one column to the left and insert the identity
            // column, so that `dorg2r` finds them where it expects.
            for j in (2..=m).rev() {
                a[(j - 1) * lda] = 0.0;
                for i in (j + 1)..=m {
                    a[(i - 1) + (j - 1) * lda] = a[(i - 1) + (j - 2) * lda];
                }
            }
            a[0] = 1.0;
            for i in 2..=m {
                a[i - 1] = 0.0;
            }
            if m > 1 {
                dorg2r(m - 1, m - 1, m - 1, &mut a[(1) + (1) * lda..], lda, tau);
            }
        }
    } else if k < n {
        dorgl2(m, n, k, a, lda, tau);
    } else {
        // Shift the reflectors one row down, exactly as above but along the other
        // diagonal. `A(I, 1)` is row `I` of column 1, i.e. offset `i - 1`, not
        // `(i - 1) * lda` -- confusing the two zeroes a whole column of
        // reflectors and leaves `P` non-orthogonal.
        a[0] = 1.0;
        for i in 2..=n {
            a[i - 1] = 0.0;
        }
        for j in 2..=n {
            for i in (2..=j - 1).rev() {
                a[(i - 1) + (j - 1) * lda] = a[(i - 2) + (j - 1) * lda];
            }
            a[(j - 1) * lda] = 0.0;
        }
        if n > 1 {
            dorgl2(n - 1, n - 1, n - 1, &mut a[(1) + (1) * lda..], lda, tau);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Bidiagonal SVD
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// `DBDSQR`: the SVD of an `n x n` bidiagonal matrix, accumulating the vectors.
///
/// `lower` selects `UPLO = 'L'`. `d` holds the diagonal, `e` the off-diagonal.
/// `vt` is `ldvt x ncvt` with the first `n` rows meaningful, `u` is `ldu x nru`
/// with the first `n` columns meaningful; both are updated by the right-hand and
/// left-hand orthogonal factors. `c` (an `ldc x ncc` matrix) is accepted for
/// fidelity with the reference but this driver always calls with `ncc = 0`.
///
/// Returns `info`: `0` on success, or the number of off-diagonal entries that did
/// not converge (LAPACK's `INFO > 0`).
#[allow(clippy::too_many_arguments)]
pub fn dbdsqr(
    lower: bool,
    n: usize,
    ncvt: usize,
    nru: usize,
    ncc: usize,
    d: &mut [f64],
    e: &mut [f64],
    vt: &mut [f64],
    ldvt: usize,
    u: &mut [f64],
    ldu: usize,
    c: &mut [f64],
    ldc: usize,
) -> usize {
    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;
    const NEGONE: f64 = -1.0;
    const HNDRTH: f64 = 0.01;
    const TEN: f64 = 10.0;
    const HNDRD: f64 = 100.0;
    const MEIGTH: f64 = -0.125;
    const MAXITR: usize = 6;

    if n == 0 {
        return 0;
    }

    let mut work = vec![0.0; 4 * n];
    let nm1 = n - 1;
    let nm12 = nm1 + nm1;
    let nm13 = nm12 + nm1;
    let mut idir: usize = 0;

    let unfl = SAFMIN;

    if lower {
        // Make the bidiagonal upper, accumulating the rotations on the left.
        for i in 1..=nm1 {
            let (cs, sn, r) = dlartg(d[i - 1], e[i - 1]);
            d[i - 1] = r;
            e[i - 1] = sn * d[i];
            d[i] *= cs;
            work[i - 1] = cs;
            work[nm1 + i - 1] = sn;
        }
        if nru > 0 {
            dlasr_v(false, true, nru, n, &work[0..], &work[nm1..], u, ldu);
        }
        if ncc > 0 {
            dlasr_v(true, true, n, ncc, &work[0..], &work[nm1..], c, ldc);
        }
    }

    // TOLMUL = MAX( TEN, MIN( HNDRD, EPS**MEIGTH ) ).
    let tolmul = TEN.max(HNDRD.min(EPS.powf(MEIGTH)));
    let tol = tolmul * EPS;

    let mut smax = ZERO;
    for i in 1..=n {
        smax = smax.max(d[i - 1].abs());
    }
    for i in 1..=nm1 {
        smax = smax.max(e[i - 1].abs());
    }
    let mut smin = ZERO;
    let thresh;
    if tol >= ZERO {
        let mut sminoa = d[0].abs();
        if sminoa != ZERO {
            let mut mu = sminoa;
            for i in 2..=n {
                mu = d[i - 1].abs() * (mu / (mu + e[i - 2].abs()));
                sminoa = sminoa.min(mu);
                if sminoa == ZERO {
                    break;
                }
            }
        }
        sminoa /= (n as f64).sqrt();
        thresh = (tol * sminoa).max(MAXITR as f64 * (n as f64 * (n as f64 * unfl)));
    } else {
        thresh = (tol.abs() * smax).max(MAXITR as f64 * (n as f64 * (n as f64 * unfl)));
    }

    let maxitdivn = MAXITR * n;
    let mut iterdivn = 0usize;
    let mut iter: isize = -1;
    let mut oldll: isize = -1;
    let mut oldm: isize = -1;

    let mut m = n;
    let info = 0usize;

    // Begin main iteration loop (Fortran's label 60).
    loop {
        if m <= 1 {
            // All singular values converged: go to the final cleanup.
            break;
        }
        if iter >= n as isize {
            iter -= n as isize;
            iterdivn += 1;
            if iterdivn >= maxitdivn {
                // Failure to converge: count the unconverged off-diagonals.
                let mut count = 0usize;
                for i in 1..=nm1 {
                    if e[i - 1] != ZERO {
                        count += 1;
                    }
                }
                return count;
            }
        }

        // Find the diagonal block of the matrix to work on.
        if tol < ZERO && d[m - 1].abs() <= thresh {
            d[m - 1] = ZERO;
        }
        smax = d[m - 1].abs();
        let mut ll = 0usize;
        let mut split = false;
        for lll in 1..=m - 1 {
            ll = m - lll;
            let abss = d[ll - 1].abs();
            let abse = e[ll - 1].abs();
            if tol < ZERO && abss <= thresh {
                d[ll - 1] = ZERO;
            }
            if abse <= thresh {
                split = true;
                break;
            }
            smax = smax.max(abss).max(abse);
        }
        if !split {
            ll = 0;
        } else {
            e[ll - 1] = ZERO;
            if ll == m - 1 {
                // Convergence of the bottom singular value.
                m -= 1;
                continue;
            }
        }
        // Fortran's label 90.
        ll += 1;

        if ll == m - 1 {
            // A 2 by 2 block: handle it separately.
            let (sigmn, sigmax, sinr, cosr, sinl, cosl) = dlasv2(d[m - 2], e[m - 2], d[m - 1]);
            d[m - 2] = sigmax;
            e[m - 2] = ZERO;
            d[m - 1] = sigmn;
            if ncvt > 0 {
                for k in 0..ncvt {
                    let x = vt[(m - 2) + k * ldvt];
                    let y = vt[(m - 1) + k * ldvt];
                    vt[(m - 2) + k * ldvt] = cosr * x + sinr * y;
                    vt[(m - 1) + k * ldvt] = cosr * y - sinr * x;
                }
            }
            if nru > 0 {
                for k in 0..nru {
                    let x = u[k + (m - 2) * ldu];
                    let y = u[k + (m - 1) * ldu];
                    u[k + (m - 2) * ldu] = cosl * x + sinl * y;
                    u[k + (m - 1) * ldu] = cosl * y - sinl * x;
                }
            }
            if ncc > 0 {
                for k in 0..ncc {
                    let x = c[(m - 2) + k * ldc];
                    let y = c[(m - 1) + k * ldc];
                    c[(m - 2) + k * ldc] = cosl * x + sinl * y;
                    c[(m - 1) + k * ldc] = cosl * y - sinl * x;
                }
            }
            m -= 2;
            continue;
        }

        // On a new submatrix, choose the shift direction (from the larger end
        // diagonal element toward the smaller).
        if ll as isize > oldm || (m as isize) < oldll {
            idir = if d[ll - 1].abs() >= d[m - 1].abs() { 1 } else { 2 };
        }

        if idir == 1 {
            if e[m - 2].abs() <= tol.abs() * d[m - 1].abs() || (tol < ZERO && e[m - 2].abs() <= thresh) {
                e[m - 2] = ZERO;
                continue;
            }
            if tol >= ZERO {
                let mut mu = d[ll - 1].abs();
                smin = mu;
                let mut deflated = false;
                for lll in ll..=m - 1 {
                    if e[lll - 1].abs() <= tol * mu {
                        e[lll - 1] = ZERO;
                        deflated = true;
                        break;
                    }
                    mu = d[lll].abs() * (mu / (mu + e[lll - 1].abs()));
                    smin = smin.min(mu);
                }
                if deflated {
                    continue;
                }
            }
        } else {
            if e[ll - 1].abs() <= tol.abs() * d[ll - 1].abs() || (tol < ZERO && e[ll - 1].abs() <= thresh) {
                e[ll - 1] = ZERO;
                continue;
            }
            if tol >= ZERO {
                let mut mu = d[m - 1].abs();
                smin = mu;
                let mut deflated = false;
                for lll in (ll..=m - 1).rev() {
                    if e[lll - 1].abs() <= tol * mu {
                        e[lll - 1] = ZERO;
                        deflated = true;
                        break;
                    }
                    mu = d[lll - 1].abs() * (mu / (mu + e[lll - 1].abs()));
                    smin = smin.min(mu);
                }
                if deflated {
                    continue;
                }
            }
        }
        oldll = ll as isize;
        oldm = m as isize;

        // Choose the shift.
        let shift;
        if tol >= ZERO && (n as f64) * tol * (smin / smax) <= PREC.max(HNDRTH * tol) {
            shift = ZERO;
        } else {
            let (sh, sll);
            if idir == 1 {
                sll = d[ll - 1].abs();
                sh = dlas2(d[m - 2], e[m - 2], d[m - 1]).0;
            } else {
                sll = d[m - 1].abs();
                sh = dlas2(d[ll - 1], e[ll - 1], d[ll]).0;
            }
            shift = if sll > ZERO && (sh / sll).powi(2) < EPS {
                ZERO
            } else {
                sh
            };
        }

        iter += m as isize - ll as isize;

        if shift == ZERO {
            // Simplified QR iteration, no shift.
            if idir == 1 {
                // Chase the bulge from top to bottom.
                let mut cs = ONE;
                let mut oldcs = ONE;
                let mut sn;
                let mut oldsn = ZERO;
                for i in ll..=m - 1 {
                    let (c, s, r) = dlartg(d[i - 1] * cs, e[i - 1]);
                    cs = c;
                    sn = s;
                    if i > ll {
                        e[i - 2] = oldsn * r;
                    }
                    let (c, s, r) = dlartg(oldcs * r, d[i] * sn);
                    oldcs = c;
                    oldsn = s;
                    d[i - 1] = r;
                    work[i - ll] = cs;
                    work[i - ll + nm1] = sn;
                    work[i - ll + nm12] = oldcs;
                    work[i - ll + nm13] = oldsn;
                }
                let h = d[m - 1] * cs;
                d[m - 1] = h * oldcs;
                e[m - 2] = h * oldsn;

                if ncvt > 0 {
                    dlasr_v(
                        true,
                        true,
                        m - ll + 1,
                        ncvt,
                        &work[0..],
                        &work[nm1..],
                        &mut vt[ll - 1..],
                        ldvt,
                    );
                }
                if nru > 0 {
                    dlasr_v(
                        false,
                        true,
                        nru,
                        m - ll + 1,
                        &work[nm12..],
                        &work[nm13..],
                        &mut u[(ll - 1) * ldu..],
                        ldu,
                    );
                }
                if ncc > 0 {
                    dlasr_v(
                        true,
                        true,
                        m - ll + 1,
                        ncc,
                        &work[nm12..],
                        &work[nm13..],
                        &mut c[ll - 1..],
                        ldc,
                    );
                }
                if e[m - 2].abs() <= thresh {
                    e[m - 2] = ZERO;
                }
            } else {
                // Chase the bulge from bottom to top.
                let mut cs = ONE;
                let mut oldcs = ONE;
                let mut sn;
                let mut oldsn = ZERO;
                for i in (ll + 1..=m).rev() {
                    let (c, s, r) = dlartg(d[i - 1] * cs, e[i - 2]);
                    cs = c;
                    sn = s;
                    if i < m {
                        e[i - 1] = oldsn * r;
                    }
                    let (c, s, r) = dlartg(oldcs * r, d[i - 2] * sn);
                    oldcs = c;
                    oldsn = s;
                    d[i - 1] = r;
                    work[i - ll - 1] = cs;
                    work[i - ll - 1 + nm1] = -sn;
                    work[i - ll - 1 + nm12] = oldcs;
                    work[i - ll - 1 + nm13] = -oldsn;
                }
                let h = d[ll - 1] * cs;
                d[ll - 1] = h * oldcs;
                e[ll - 1] = h * oldsn;

                if ncvt > 0 {
                    dlasr_v(
                        true,
                        false,
                        m - ll + 1,
                        ncvt,
                        &work[nm12..],
                        &work[nm13..],
                        &mut vt[ll - 1..],
                        ldvt,
                    );
                }
                if nru > 0 {
                    dlasr_v(
                        false,
                        false,
                        nru,
                        m - ll + 1,
                        &work[0..],
                        &work[nm1..],
                        &mut u[(ll - 1) * ldu..],
                        ldu,
                    );
                }
                if ncc > 0 {
                    dlasr_v(
                        true,
                        false,
                        m - ll + 1,
                        ncc,
                        &work[0..],
                        &work[nm1..],
                        &mut c[ll - 1..],
                        ldc,
                    );
                }
                if e[ll - 1].abs() <= thresh {
                    e[ll - 1] = ZERO;
                }
            }
        } else if idir == 1 {
            // Nonzero shift, chasing from top to bottom.
            let mut f = (d[ll - 1].abs() - shift) * (fsign(ONE, d[ll - 1]) + shift / d[ll - 1]);
            let mut g = e[ll - 1];
            for i in ll..=m - 1 {
                let (cosr, sinr, r) = dlartg(f, g);
                if i > ll {
                    e[i - 2] = r;
                }
                f = cosr * d[i - 1] + sinr * e[i - 1];
                e[i - 1] = cosr * e[i - 1] - sinr * d[i - 1];
                g = sinr * d[i];
                d[i] *= cosr;
                let (cosl, sinl, r) = dlartg(f, g);
                d[i - 1] = r;
                f = cosl * e[i - 1] + sinl * d[i];
                d[i] = cosl * d[i] - sinl * e[i - 1];
                if i < m - 1 {
                    g = sinl * e[i];
                    e[i] *= cosl;
                }
                work[i - ll] = cosr;
                work[i - ll + nm1] = sinr;
                work[i - ll + nm12] = cosl;
                work[i - ll + nm13] = sinl;
            }
            e[m - 2] = f;

            if ncvt > 0 {
                dlasr_v(
                    true,
                    true,
                    m - ll + 1,
                    ncvt,
                    &work[0..],
                    &work[nm1..],
                    &mut vt[ll - 1..],
                    ldvt,
                );
            }
            if nru > 0 {
                dlasr_v(
                    false,
                    true,
                    nru,
                    m - ll + 1,
                    &work[nm12..],
                    &work[nm13..],
                    &mut u[(ll - 1) * ldu..],
                    ldu,
                );
            }
            if ncc > 0 {
                dlasr_v(
                    true,
                    true,
                    m - ll + 1,
                    ncc,
                    &work[nm12..],
                    &work[nm13..],
                    &mut c[ll - 1..],
                    ldc,
                );
            }
            if e[m - 2].abs() <= thresh {
                e[m - 2] = ZERO;
            }
        } else {
            // Nonzero shift, chasing from bottom to top.
            let mut f = (d[m - 1].abs() - shift) * (fsign(ONE, d[m - 1]) + shift / d[m - 1]);
            let mut g = e[m - 2];
            for i in (ll + 1..=m).rev() {
                let (cosr, sinr, r) = dlartg(f, g);
                if i < m {
                    e[i - 1] = r;
                }
                f = cosr * d[i - 1] + sinr * e[i - 2];
                e[i - 2] = cosr * e[i - 2] - sinr * d[i - 1];
                g = sinr * d[i - 2];
                d[i - 2] *= cosr;
                let (cosl, sinl, r) = dlartg(f, g);
                d[i - 1] = r;
                f = cosl * e[i - 2] + sinl * d[i - 2];
                d[i - 2] = cosl * d[i - 2] - sinl * e[i - 2];
                if i > ll + 1 {
                    g = sinl * e[i - 3];
                    e[i - 3] *= cosl;
                }
                work[i - ll - 1] = cosr;
                work[i - ll - 1 + nm1] = -sinr;
                work[i - ll - 1 + nm12] = cosl;
                work[i - ll - 1 + nm13] = -sinl;
            }
            e[ll - 1] = f;

            if e[ll - 1].abs() <= thresh {
                e[ll - 1] = ZERO;
            }
            if ncvt > 0 {
                dlasr_v(
                    true,
                    false,
                    m - ll + 1,
                    ncvt,
                    &work[nm12..],
                    &work[nm13..],
                    &mut vt[ll - 1..],
                    ldvt,
                );
            }
            if nru > 0 {
                dlasr_v(
                    false,
                    false,
                    nru,
                    m - ll + 1,
                    &work[0..],
                    &work[nm1..],
                    &mut u[(ll - 1) * ldu..],
                    ldu,
                );
            }
            if ncc > 0 {
                dlasr_v(
                    true,
                    false,
                    m - ll + 1,
                    ncc,
                    &work[0..],
                    &work[nm1..],
                    &mut c[ll - 1..],
                    ldc,
                );
            }
        }
    }

    // All singular values converged: make them positive.
    for i in 1..=n {
        if d[i - 1] == ZERO {
            // Avoid -ZERO.
            d[i - 1] = ZERO;
        }
        if d[i - 1] < ZERO {
            d[i - 1] = -d[i - 1];
            if ncvt > 0 {
                for k in 0..ncvt {
                    vt[(i - 1) + k * ldvt] *= NEGONE;
                }
            }
        }
    }

    // Insertion sort into decreasing order, one transposition per singular value.
    for i in 1..=nm1 {
        let mut isub = 1usize;
        let mut smallest = d[0];
        for j in 2..=(n + 1 - i) {
            if d[j - 1] <= smallest {
                isub = j;
                smallest = d[j - 1];
            }
        }
        if isub != n + 1 - i {
            d[isub - 1] = d[n - i];
            d[n - i] = smallest;
            if ncvt > 0 {
                for k in 0..ncvt {
                    vt.swap((isub - 1) + k * ldvt, (n - i) + k * ldvt);
                }
            }
            if nru > 0 {
                for k in 0..nru {
                    u.swap(k + (isub - 1) * ldu, k + (n - i) * ldu);
                }
            }
            if ncc > 0 {
                for k in 0..ncc {
                    c.swap((isub - 1) + k * ldc, (n - i) + k * ldc);
                }
            }
        }
    }
    info
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// The driver
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// `DGESVD` with `JOBU = 'A'` and `JOBVT = 'A'`: the full SVD of an `m x n`
/// matrix.
///
/// On entry `a` is the `m x n` matrix (column-major, leading dimension `m`) and is
/// destroyed. On exit `s` holds the `min(m, n)` singular values in decreasing
/// order, `u` the `m x m` orthogonal matrix, and `vt` the `n x n` matrix `V^T`,
/// so that `A = U S V^T` with `S` the `m x n` diagonal of `s`.
///
/// # Errors
///
/// [`LapackError::BadShape`] if a buffer's length disagrees with `(m, n)`, and
/// [`LapackError::Convergence`] if `dbdsqr` fails to converge (the driver's
/// `INFO > 0`, which the caller reports as "algorithm did not converge").
pub fn dgesvd(
    a: &mut [f64],
    m: usize,
    n: usize,
    s: &mut [f64],
    u: &mut [f64],
    vt: &mut [f64],
) -> Result<(), LapackError> {
    if a.len() != m * n {
        return Err(LapackError::BadShape);
    }
    if m == 0 || n == 0 {
        return Ok(());
    }
    let minmn = m.min(n);
    if s.len() != minmn || u.len() != m * m || vt.len() != n * n {
        return Err(LapackError::BadShape);
    }

    // Scale the matrix into the safe range up front; the singular values are
    // scaled back at the end.
    let eps = PREC;
    let smlnum = SAFMIN.sqrt() / eps;
    let bignum = 1.0 / smlnum;
    let anrm = super::super::blas::dlange('M', a, m, n);
    let mut iscl = false;
    if anrm > 0.0 && anrm < smlnum {
        iscl = true;
        dlascl_g(anrm, smlnum, a, m, m, n);
    } else if anrm > bignum {
        iscl = true;
        dlascl_g(anrm, bignum, a, m, m, n);
    }

    let mut e = vec![0.0; minmn];
    let mut tauq = vec![0.0; minmn];
    let mut taup = vec![0.0; minmn];

    let info;
    if m >= n {
        // Bidiagonalize A directly: Q^T A P = B with B upper bidiagonal.
        dgebd2(a, m, m, n, s, &mut e, &mut tauq, &mut taup);

        // Left singular vectors: copy the reflectors into U and generate Q.
        for j in 0..n {
            for i in j..m {
                u[i + j * m] = a[i + j * m];
            }
        }
        dorgbr(true, m, m, n, u, m, &tauq);

        // Right singular vectors: copy the reflectors into VT and generate P.
        // `a` is `m x n` with leading dimension `m`, while `vt` is `n x n`; the
        // strides differ for a tall matrix.
        for j in 0..n {
            for i in 0..=j {
                vt[i + j * n] = a[i + j * m];
            }
        }
        dorgbr(false, n, n, n, vt, n, &taup);

        info = dbdsqr(false, n, n, m, 0, s, &mut e, vt, n, u, m, &mut [], 0);
    } else {
        // Bidiagonalize A directly: Q^T A P = B with B lower bidiagonal.
        dgebd2(a, m, m, n, s, &mut e, &mut tauq, &mut taup);

        // Left singular vectors: the reflectors live in the m x m lower triangle.
        for j in 0..m {
            for i in j..m {
                u[i + j * m] = a[i + j * m];
            }
        }
        dorgbr(true, m, m, n, u, m, &tauq);

        // Right singular vectors: the reflectors live in the m x n upper triangle,
        // but VT itself is n x n.
        for j in 0..n {
            for i in 0..=j.min(m - 1) {
                vt[i + j * n] = a[i + j * m];
            }
        }
        dorgbr(false, n, n, m, vt, n, &taup);

        info = dbdsqr(true, m, n, m, 0, s, &mut e, vt, n, u, m, &mut [], 0);
    }

    if iscl {
        if anrm > bignum {
            dlascl_g(bignum, anrm, s, minmn, minmn, 1);
        } else if anrm < smlnum {
            dlascl_g(smlnum, anrm, s, minmn, minmn, 1);
        }
    }

    if info != 0 {
        return Err(LapackError::Convergence);
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic rectangular matrix, the same construction the parity test
    /// uses, so a failure here points at a shape the parity test also covers.
    fn rect_sample(m: usize, n: usize) -> Vec<f64> {
        let mut a = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                a[i + j * m] = (((i * 7 + j * 13 + 3) % 11) as f64) - 5.0 + (i as f64) * 0.25 - (j as f64) * 0.1;
            }
        }
        a
    }

    /// The 1-norm of a dense column-major matrix: LAPACK's `DLANGE('1', ...)`.
    fn norm_1(a: &[f64], m: usize, n: usize) -> f64 {
        let mut best = 0.0f64;
        for j in 0..n {
            let mut sum = 0.0;
            for i in 0..m {
                sum += a[i + j * m].abs();
            }
            best = best.max(sum);
        }
        best
    }

    /// `C = A B` for dense column-major `m x k` and `k x n` factors.
    fn mat_mul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
        let mut c = vec![0.0; m * n];
        for j in 0..n {
            for i in 0..m {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += a[i + l * m] * b[l + j * k];
                }
                c[i + j * m] = sum;
            }
        }
        c
    }

    /// The ratio LAPACK's `dort01` uses for an orthonormal set of `n` columns:
    /// `norm(I - Q^T Q) / (n * eps)`.
    fn orthogonality_ratio(q: &[f64], m: usize, n: usize) -> f64 {
        let mut diff = 0.0;
        for j in 0..n {
            for i in 0..n {
                let mut dot = 0.0;
                for k in 0..m {
                    dot += q[k + i * m] * q[k + j * m];
                }
                diff += (dot - if i == j { 1.0 } else { 0.0 }).abs();
            }
        }
        diff / (n as f64 * f64::EPSILON)
    }

    /// Reconstruction and orthogonality for every shape the parity test uses.
    ///
    /// This is the reference-free half of that test: it cannot check the singular
    /// values against LAPACK, but it does catch wrong vectors, wrong pairing, and
    /// non-orthogonal factors.
    #[test]
    fn dgesvd_reconstructs_for_all_shapes() {
        let mut bad = Vec::new();
        for (m, n) in [
            (1usize, 1usize),
            (2, 2),
            (3, 2),
            (2, 3),
            (4, 3),
            (3, 4),
            (6, 4),
            (4, 6),
            (5, 5),
            (3, 3),
            (4, 4),
            (8, 6),
        ] {
            let a = rect_sample(m, n);
            let mut work = a.clone();
            let min_mn = m.min(n);
            let mut s = vec![0.0; min_mn];
            let mut u = vec![0.0; m * m];
            let mut vt = vec![0.0; n * n];
            dgesvd(&mut work, m, n, &mut s, &mut u, &mut vt).unwrap();
            let scale = a.iter().fold(1.0f64, |acc, x| acc.max(x.abs()));
            let mut worst = 0.0f64;
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for k in 0..min_mn {
                        sum += u[i + k * m] * s[k] * vt[k + j * n];
                    }
                    worst = worst.max((sum - a[i + j * m]).abs());
                }
            }
            let mut u_orth = 0.0f64;
            for i in 0..m {
                for j in 0..m {
                    let mut dot = 0.0;
                    for k in 0..m {
                        dot += u[k + i * m] * u[k + j * m];
                    }
                    u_orth = u_orth.max((dot - if i == j { 1.0 } else { 0.0 }).abs());
                }
            }
            let mut vt_orth = 0.0f64;
            for i in 0..n {
                for j in 0..n {
                    let mut dot = 0.0;
                    for k in 0..n {
                        dot += vt[i + k * n] * vt[j + k * n];
                    }
                    vt_orth = vt_orth.max((dot - if i == j { 1.0 } else { 0.0 }).abs());
                }
            }
            println!(
                "{m}x{n}: recon {:.3e}, U orth {:.3e}, V orth {:.3e}, s = {:?}",
                worst / scale,
                u_orth,
                vt_orth,
                s
            );
            if worst / scale > 1e-10 || u_orth > 1e-10 || vt_orth > 1e-10 {
                bad.push((m, n));
            }
        }
        assert!(bad.is_empty(), "shapes with a bad SVD: {bad:?}");
    }

    /// `dgebd2` followed by `dorgbr` must produce an orthogonal `P^T`.
    ///
    /// `dorgbr` with `VECT = 'P'` returns `P^T`, and when `K == N` it has to shift
    /// the reflectors before handing them to the LQ kernel. That shift is where a
    /// transposed index hides, so it is checked here on its own as well as through
    /// the full `dgesvd` reconstruction test.
    #[test]
    fn dgebd2_and_dorgbr_give_an_orthogonal_p_transpose() {
        for (m, n) in [(3usize, 3usize), (4, 3), (5, 5), (6, 4)] {
            let mut a = rect_sample(m, n);
            let mut d = vec![0.0; m.min(n)];
            let mut e = vec![0.0; m.min(n)];
            let mut tq = vec![0.0; m.min(n)];
            let mut tp = vec![0.0; m.min(n)];
            dgebd2(&mut a, m, m, n, &mut d, &mut e, &mut tq, &mut tp);

            let mut vt = vec![0.0; n * n];
            for j in 0..n {
                for i in 0..=j {
                    vt[i + j * n] = a[i + j * m];
                }
            }
            dorgbr(false, n, n, n, &mut vt, n, &tp);

            for i in 0..n {
                for j in 0..n {
                    let mut dot = 0.0;
                    for k in 0..n {
                        dot += vt[i + k * n] * vt[j + k * n];
                    }
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (dot - want).abs() < 1e-12,
                        "{m}x{n}: P^T rows {i},{j} are not orthonormal ({dot})"
                    );
                }
            }
        }
    }

    /// A 2x2 matrix whose SVD is known exactly: A = diag(3, 2).
    #[test]
    fn dgesvd_of_a_diagonal_matrix_keeps_the_order() {
        let mut a = vec![3.0, 0.0, 0.0, 2.0];
        let mut s = vec![0.0; 2];
        let mut u = vec![0.0; 4];
        let mut vt = vec![0.0; 4];
        dgesvd(&mut a, 2, 2, &mut s, &mut u, &mut vt).unwrap();
        assert!((s[0] - 3.0).abs() < 1e-14, "s0 = {}", s[0]);
        assert!((s[1] - 2.0).abs() < 1e-14, "s1 = {}", s[1]);
        assert!((u[0] - 1.0).abs() < 1e-14);
        assert!((vt[3] - 1.0).abs() < 1e-14);
    }

    /// A tall matrix, checked by reconstructing `U S V^T`.
    #[test]
    fn dgesvd_reconstructs_a_tall_matrix() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3 x 2
        let mut work = a.clone();
        let mut s = vec![0.0; 2];
        let mut u = vec![0.0; 9];
        let mut vt = vec![0.0; 4];
        dgesvd(&mut work, 3, 2, &mut s, &mut u, &mut vt).unwrap();
        for i in 0..3 {
            for j in 0..2 {
                let mut sum = 0.0;
                for k in 0..2 {
                    sum += u[i + k * 3] * s[k] * vt[k + j * 2];
                }
                assert!((sum - a[i + j * 3]).abs() < 1e-12, "({i},{j}) = {sum}");
            }
        }
    }

    /// A wide matrix (the lower-bidiagonal branch).
    #[test]
    fn dgesvd_reconstructs_a_wide_matrix() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 x 3
        let mut work = a.clone();
        let mut s = vec![0.0; 2];
        let mut u = vec![0.0; 4];
        let mut vt = vec![0.0; 9];
        dgesvd(&mut work, 2, 3, &mut s, &mut u, &mut vt).unwrap();
        for i in 0..2 {
            for j in 0..3 {
                let mut sum = 0.0;
                for k in 0..2 {
                    sum += u[i + k * 2] * s[k] * vt[k + j * 3];
                }
                assert!((sum - a[i + j * 2]).abs() < 1e-12, "({i},{j}) = {sum}");
            }
        }
    }

    /// `DLASV2` on a diagonal matrix.
    #[test]
    fn dlasv2_handles_a_diagonal_block() {
        let (ssmin, ssmax, snr, csr, snl, csl) = dlasv2(3.0, 0.0, 2.0);
        assert!((ssmax.abs() - 3.0).abs() < 1e-14);
        assert!((ssmin.abs() - 2.0).abs() < 1e-14);
        assert_eq!((snr, csr, snl, csl), (0.0, 1.0, 0.0, 1.0));
    }

    /// `dbdsqr`'s non-convergence reaches the caller as `Convergence`.
    ///
    /// A NaN matrix keeps the QR sweeps' convergence test false, so `dbdsqr`
    /// runs out of iterations and sets `INFO > 0`. The driver must surface it:
    /// `mat_svd` reports this variant as "Algorithm did not converge", so
    /// returning `Singular` here -- which is what it used to do -- named the
    /// wrong condition.
    #[test]
    fn dgesvd_reports_non_convergence() {
        for (m, n) in [(4usize, 4usize), (6, 4), (4, 6)] {
            let mut a = vec![f64::NAN; m * n];
            let mut s = vec![0.0; m.min(n)];
            let mut u = vec![0.0; m * m];
            let mut vt = vec![0.0; n * n];
            assert_eq!(
                dgesvd(&mut a, m, n, &mut s, &mut u, &mut vt),
                Err(LapackError::Convergence),
                "dgesvd {m}x{n} must report non-convergence"
            );
        }
    }

    /// LAPACK's `dbdt01` and `dort01`: `dgebd2` followed by the two `dorgbr` calls
    /// must give `A = Q B P^T`, with `B` bidiagonal and `Q`, `P` orthogonal, for
    /// tall, square and wide matrices.
    ///
    /// The existing `dgebd2`/`dorgbr` test only checks that `P^T` comes out
    /// orthogonal, which a wrong pairing of reflectors and taus can satisfy. This
    /// reconstructs the matrix, which is the property the reduction exists for,
    /// and covers the wide (`m < n`, lower bidiagonal) branch that no test touched.
    ///
    /// The ratios are LAPACK's:
    ///
    ///   ratio 1 = norm(A - Q B P^T) / (max(m,n) * norm(A) * eps)
    ///   ratio 2 = norm(I - Q^T Q)   / (m * eps)
    ///   ratio 3 = norm(I - P P^T)   / (n * eps)
    #[test]
    fn dgebd2_and_dorgbr_reconstruct_the_bidiagonal_form() {
        let eps = f64::EPSILON;
        for (m, n) in [(2usize, 2usize), (3, 2), (3, 3), (4, 3), (6, 4), (2, 3), (3, 4), (4, 6)] {
            let a0 = rect_sample(m, n);
            let mut a = a0.clone();
            let minmn = m.min(n);
            let mut d = vec![0.0; minmn];
            let mut e = vec![0.0; minmn];
            let mut tauq = vec![0.0; minmn];
            let mut taup = vec![0.0; minmn];
            dgebd2(&mut a, m, m, n, &mut d, &mut e, &mut tauq, &mut taup);

            // The reflectors for `Q` (m x m) and for `P^T` (n x n), copied out of
            // `a` and generated exactly as `dgesvd` does it.
            let mut q = vec![0.0; m * m];
            if m >= n {
                for j in 0..n {
                    for i in j..m {
                        q[i + j * m] = a[i + j * m];
                    }
                }
            } else {
                for j in 0..m {
                    for i in j..m {
                        q[i + j * m] = a[i + j * m];
                    }
                }
            }
            dorgbr(true, m, m, n, &mut q, m, &tauq);

            let mut vt = vec![0.0; n * n];
            if m >= n {
                for j in 0..n {
                    for i in 0..=j {
                        vt[i + j * n] = a[i + j * m];
                    }
                }
            } else {
                for j in 0..n {
                    for i in 0..=j.min(m - 1) {
                        vt[i + j * n] = a[i + j * m];
                    }
                }
            }
            let kp = if m >= n { n } else { m };
            dorgbr(false, n, n, kp, &mut vt, n, &taup);

            // B, in the bidiagonal form `dgebd2` produces: upper for m >= n,
            // lower otherwise.
            let mut b = vec![0.0; m * n];
            for i in 0..minmn {
                b[i + i * m] = d[i];
                if i + 1 < minmn {
                    if m >= n {
                        b[i + (i + 1) * m] = e[i];
                    } else {
                        b[(i + 1) + i * m] = e[i];
                    }
                }
            }

            let qb = mat_mul(&q, &b, m, m, n);
            let mut rec = vec![0.0; m * n];
            for j in 0..n {
                for i in 0..m {
                    let mut sum = 0.0;
                    for k in 0..n {
                        sum += qb[i + k * m] * vt[k + j * n];
                    }
                    rec[i + j * m] = sum;
                }
            }
            let a_norm = norm_1(&a0, m, n).max(f64::MIN_POSITIVE);
            let mut diff = 0.0;
            for idx in 0..m * n {
                diff += (a0[idx] - rec[idx]).abs();
            }
            let ratio1 = diff / (m.max(n) as f64 * a_norm * eps);
            let ratio2 = orthogonality_ratio(&q, m, m);
            let ratio3 = orthogonality_ratio(&vt, n, n);
            assert!(ratio1 <= 20.0, "{m}x{n}: dbdt01 ratio = {ratio1} exceeds 20");
            assert!(ratio2 <= 20.0, "{m}x{n}: Q orthogonality ratio = {ratio2} exceeds 20");
            assert!(ratio3 <= 20.0, "{m}x{n}: P orthogonality ratio = {ratio3} exceeds 20");
        }
    }

    /// LAPACK's `dbdt03`: `dbdsqr` on a bidiagonal matrix must produce
    /// `S = U^T B V`, checked directly rather than through `dgesvd`.
    ///
    /// `dbdsqr` had no direct test: everything about it was inferred from
    /// `dgesvd` reconstructing a dense matrix, two reductions away. Feeding it a
    /// bidiagonal matrix and asking for the decomposition tests the routine on
    /// the only input it ever sees.
    #[test]
    fn dbdsqr_reconstructs_the_singular_value_decomposition() {
        let eps = f64::EPSILON;
        let mut cases: Vec<(usize, Vec<f64>, Vec<f64>)> = Vec::new();
        for n in [2usize, 3, 5, 8, 13] {
            let d = (0..n).map(|i| 1.0 + 0.5 * (i as f64)).collect();
            let mut e = vec![0.0; n];
            for i in 0..n - 1 {
                e[i] = 0.75 - 0.125 * ((i * 3) % 4) as f64;
            }
            cases.push((n, d, e));
        }
        // Rank deficient: a zero diagonal entry.
        let mut d = vec![1.0, 0.0, 2.0, 3.0];
        let mut e = vec![0.5, 0.25, 0.125, 0.0];
        cases.push((4, d.clone(), e.clone()));
        // Wide range of magnitudes in the same matrix.
        d = vec![1e8, 1.0, 1e-8, 1.0];
        e = vec![1.0, 1e-8, 1.0, 0.0];
        cases.push((4, d, e));

        for (n, d_in, mut e) in cases {
            let mut d = d_in.clone();
            // B, the bidiagonal matrix, before dbdsqr destroys it.
            let mut b = vec![0.0; n * n];
            for i in 0..n {
                b[i + i * n] = d[i];
                if i + 1 < n {
                    b[i + (i + 1) * n] = e[i];
                }
            }
            let b_norm = norm_1(&b, n, n).max(f64::MIN_POSITIVE);

            let mut u = vec![0.0; n * n];
            let mut vt = vec![0.0; n * n];
            for i in 0..n {
                u[i + i * n] = 1.0;
                vt[i + i * n] = 1.0;
            }
            let mut c = vec![0.0; 1];
            let info = dbdsqr(false, n, n, n, 0, &mut d, &mut e, &mut vt, n, &mut u, n, &mut c, 1);
            assert_eq!(info, 0, "n={n}: dbdsqr did not converge");

            // S = U^T B V, compared with the singular values it reported.
            let mut ub = vec![0.0; n * n];
            for j in 0..n {
                for i in 0..n {
                    let mut sum = 0.0;
                    for k in 0..n {
                        sum += u[k + i * n] * b[k + j * n];
                    }
                    ub[i + j * n] = sum;
                }
            }
            let mut s = vec![0.0; n * n];
            for j in 0..n {
                for i in 0..n {
                    let mut sum = 0.0;
                    for k in 0..n {
                        sum += ub[i + k * n] * vt[j + k * n];
                    }
                    s[i + j * n] = sum;
                }
            }
            let mut diff = 0.0;
            for j in 0..n {
                for i in 0..n {
                    let want = if i == j { d[i] } else { 0.0 };
                    diff += (s[i + j * n] - want).abs();
                }
            }
            let ratio1 = diff / (b_norm * n as f64 * eps);
            let ratio2 = orthogonality_ratio(&u, n, n);
            let ratio3 = orthogonality_ratio(&vt, n, n);
            assert!(ratio1 <= 20.0, "n={n}: dbdt03 ratio = {ratio1} exceeds 20");
            assert!(ratio2 <= 20.0, "n={n}: U orthogonality ratio = {ratio2} exceeds 20");
            assert!(ratio3 <= 20.0, "n={n}: V orthogonality ratio = {ratio3} exceeds 20");
        }
    }

    /// `dbdsqr`'s modes: singular values only (no vectors requested), the lower
    /// bidiagonal form, the `NCC > 0` transformation of a right-hand block, and a
    /// matrix with an exact zero singular value.
    ///
    /// The existing test always asks for both full vector sets from an upper
    /// bidiagonal matrix, which leaves the other four argument shapes unexecuted
    /// -- and `NCC > 0` is what `dgebrd` callers rely on to accumulate `Q^T C`.
    #[test]
    fn dbdsqr_covers_the_lower_form_and_the_vector_modes() {
        let eps = f64::EPSILON;

        // Upper bidiagonal with a zero singular value: values only.
        let n = 4;
        let d_in = [1.0, 0.0, 2.0, 3.0];
        let e_in = [0.5, 0.5, 0.25, 0.0];
        let mut d = d_in.to_vec();
        let mut e = e_in.to_vec();
        let mut vt: Vec<f64> = vec![];
        let mut u: Vec<f64> = vec![];
        let mut c: Vec<f64> = vec![];
        let info = dbdsqr(false, n, 0, 0, 0, &mut d, &mut e, &mut vt, 1, &mut u, 1, &mut c, 1);
        assert_eq!(info, 0);
        for k in 0..n {
            assert!(d[k] >= 0.0, "s[{k}] = {} is negative", d[k]);
        }
        for k in 1..n {
            assert!(d[k - 1] >= d[k], "s is not descending: {d:?}");
        }
        assert!(d[n - 1] < 1e-14, "one singular value must be zero: {d:?}");

        // Lower bidiagonal, full vectors: the same matrix transposed gives the
        // same singular values.
        let n = 3;
        let mut d_lo = [1.0, 2.0, 3.0];
        let mut e_lo = [0.5, 0.25, 0.0];
        let mut vt = vec![0.0; 9];
        let mut u = vec![0.0; 9];
        for i in 0..3 {
            vt[i + i * 3] = 1.0;
            u[i + i * 3] = 1.0;
        }
        let info = dbdsqr(true, n, n, n, 0, &mut d_lo, &mut e_lo, &mut vt, n, &mut u, n, &mut c, 1);
        assert_eq!(info, 0);
        for k in 1..n {
            assert!(d_lo[k - 1] >= d_lo[k], "lower: s not descending: {d_lo:?}");
        }

        // NCC > 0: the caller's C is transformed by the accumulated orthogonal
        // factor, `C := Q^T C`. Feeding the identity recovers `Q^T` itself, which
        // must be orthogonal and must agree with the `U` reported for the same
        // matrix up to the column signs the QR iteration is free to choose.
        let n = 4;
        let d_in = [1.0, 2.0, 3.0, 4.0];
        let e_in = [0.5, 0.25, 0.125, 0.0];

        // Pass 1: the singular vectors.
        let (mut d_u, mut e_u) = (d_in.to_vec(), e_in.to_vec());
        let mut vt_u = vec![0.0; n * n];
        let mut u_u = vec![0.0; n * n];
        let mut c_empty: Vec<f64> = vec![];
        let info = dbdsqr(
            false,
            n,
            n,
            n,
            0,
            &mut d_u,
            &mut e_u,
            &mut vt_u,
            n,
            &mut u_u,
            n,
            &mut c_empty,
            1,
        );
        assert_eq!(info, 0);

        // Pass 2: C = identity, so the output is Q^T.
        let (mut d_c, mut e_c) = (d_in.to_vec(), e_in.to_vec());
        let mut vt_c = vec![0.0; n * n];
        let mut u_c = vec![0.0; n * n];
        let mut c = vec![0.0; n * n];
        for i in 0..n {
            c[i + i * n] = 1.0;
        }
        let info = dbdsqr(
            false, n, 0, 0, n, &mut d_c, &mut e_c, &mut vt_c, 1, &mut u_c, 1, &mut c, n,
        );
        assert_eq!(info, 0);

        // Q^T (as returned in C) is orthogonal ...
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0;
                for k in 0..n {
                    dot += c[k + i * n] * c[k + j * n];
                }
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((dot - want).abs() < 1e-13, "Q^T Q ({i},{j}) = {dot}");
            }
        }
        // ... and the transformation preserves inner products, which is what an
        // orthogonal factor means. Comparing against `U` column by column would
        // be wrong: the rotations reach `C` in the order the iteration generates
        // them, before the singular values are sorted and their vectors signed,
        // so the columns of `C` come out permuted relative to `U`'s.
        let ncc = 2;
        let mut b_in = vec![0.0; n * ncc];
        for i in 0..n {
            for j in 0..ncc {
                b_in[i + j * n] = 1.0 + (i as f64) * 0.5 - (j as f64) * 0.25;
            }
        }
        let (mut d_b, mut e_b) = (d_in.to_vec(), e_in.to_vec());
        let mut b_out = b_in.clone();
        let info = dbdsqr(
            false, n, 0, 0, ncc, &mut d_b, &mut e_b, &mut vt_c, 1, &mut u_c, 1, &mut b_out, n,
        );
        assert_eq!(info, 0);
        for i in 0..ncc {
            for j in 0..ncc {
                let mut got = 0.0;
                let mut want = 0.0;
                for k in 0..n {
                    got += b_out[k + i * n] * b_out[k + j * n];
                    want += b_in[k + i * n] * b_in[k + j * n];
                }
                assert!(
                    (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                    "inner product ({i},{j}) changed: {got} != {want}"
                );
            }
        }

        // n = 1 is a single entry and no rotations.
        let mut d = vec![2.5];
        let mut e = vec![0.0];
        let mut vt = vec![1.0];
        let mut u = vec![1.0];
        let info = dbdsqr(
            false,
            1,
            1,
            1,
            0,
            &mut d,
            &mut e,
            &mut vt,
            1,
            &mut u,
            1,
            &mut c_empty,
            1,
        );
        assert_eq!(info, 0);
        assert_eq!(d[0], 2.5);
        let _ = eps;
    }
}
