//! The generalized eigenproblem `A v = lambda B v` for real and complex
//! matrices: the QZ algorithm.
//!
//! This is a transcription of LAPACK 3.12.1's `dggev`/`zggev` drivers and the
//! routines they call, with 1-based Fortran indices translated mechanically to
//! 0-based Rust. The point of following the reference literally is that the
//! results agree with it bit for bit, which the parity tests rely on.
//!
//! | Routine | Role |
//! |---|---|
//! | [`dggev`], [`zggev`] | the drivers: balance, reduce, run the QZ iteration, back-substitute |
//! | `dggbal`, `zggbal`, `dggbak`, `zggbak` | balancing (permutation and diagonal scaling) and its inverse |
//! | `dgeqr2`, `dorm2r`, `dorg2r` | the QR of `B` and its application, from `dgeqrf`/`dormqr`/`dorgqr` |
//! | `zgeqr2`, `zunm2r`, `zung2r` | the complex QR of `B` and its application, from `zgeqrf`/`zunmqr`/`zungqr` |
//! | `dgghrd`, `zgghrd` | reduction to Hessenberg-triangular form |
//! | `dhgeqz`, `zhgeqz` | the QZ iteration, producing the generalized Schur form |
//! | `dtgevc`, `ztgevc` | left and right eigenvectors by back-substitution |
//! | `dlag2`, `dlasv2`, `dlaln2`, `dladiv`, `zlartg` | the 2x2 eigenproblem and the Givens rotations |
//!
//! The blocked variants in LAPACK's `dgeqrf`, `dormqr`, `dorgqr` (and their
//! complex counterparts) fall through to the unblocked `dgeqr2`, `dorm2r`,
//! `dorg2r` (`zgeqr2`, `zunm2r`, `zung2r`) for blocks no larger than the tuned
//! block size, which covers every matrix these routines are asked about here;
//! the unblocked form is what is transcribed.
//!
//! The complex routines live in their own section, "The complex generalized
//! eigenproblem (ZGGEV)", just above the tests.
//!
//! Conventions match the rest of [`super`]: column-major, 0-based indices, and
//! workspace allocated internally.

use super::LapackError;
use super::auxiliary::{DLAMCH_E, dlaln2, dlapy2, dlapy3, dlascl_g, zladiv, zlarf_left, zlascl_g};
use super::blas;
use super::blas::dznrm2;
use super::eig::{dlarfg, dlartg};
use crate::Complex64;

/// `DLAMCH('S')`: the safe minimum, below which `1/x` overflows.
const SAFMIN: f64 = f64::MIN_POSITIVE;
/// `DLAMCH('P')` / `DLAMCH('E') * DLAMCH('B')`: the unit roundoff times the base.
const ULP: f64 = f64::EPSILON;

// ---------------------------------------------------------------------------
// Small helpers, transcribed from the BLAS and LAPACK routines named.
// ---------------------------------------------------------------------------

/// `dlassq`-equivalent Frobenius norm `dlanhs('F')` of the upper Hessenberg part
/// of an `n x n` matrix: entries `(i, j)` with `i <= j + 1`.
fn dlanhs_f(a: &[f64], ld: usize, n: usize) -> f64 {
    // Scaled sum of squares, one column at a time, in the reference's order.
    let mut scale = 0.0f64;
    let mut sumsq = 1.0f64;
    for j in 0..n {
        let count = n.min(j + 2);
        for i in 0..count {
            let x = a[i + j * ld].abs();
            if x != 0.0 {
                if scale < x {
                    let t = scale / x;
                    sumsq = 1.0 + sumsq * t * t;
                    scale = x;
                } else {
                    let t = x / scale;
                    sumsq += t * t;
                }
            }
        }
    }
    scale * sumsq.sqrt()
}

/// `dlag2`: the eigenvalues of the generalized 2x2 problem
/// `A - w B` (upper triangular `B`), returning `(s1, s2, wr1, wr2, wi)`.
///
/// `a11..a22` are the entries of `A`, `b11, b12, b22` those of `B`, and `safmin`
/// the caller's threshold.
#[allow(clippy::too_many_arguments)]
fn dlag2(
    a11: f64,
    a21: f64,
    a12: f64,
    a22: f64,
    b11: f64,
    b12: f64,
    b22: f64,
    safmin: f64,
) -> (f64, f64, f64, f64, f64) {
    const FUZZY1: f64 = 1.0 + 1.0e-5;
    const HALF: f64 = 0.5;
    const ONE: f64 = 1.0;
    const ZERO: f64 = 0.0;

    let rtmin = safmin.sqrt();
    let rtmax = ONE / rtmin;
    let safmax = ONE / safmin;

    // Scale A.
    let anorm = (a11.abs() + a21.abs()).max(a12.abs() + a22.abs()).max(safmin);
    let ascale = ONE / anorm;
    let a11 = ascale * a11;
    let a21 = ascale * a21;
    let a12 = ascale * a12;
    let a22 = ascale * a22;

    // Perturb B if necessary to ensure non-singularity.
    let mut b11 = b11;
    let mut b22 = b22;
    let bmin = rtmin * b11.abs().max(b12.abs()).max(b22.abs()).max(rtmin);
    if b11.abs() < bmin {
        b11 = bmin.copysign(b11);
    }
    if b22.abs() < bmin {
        b22 = bmin.copysign(b22);
    }

    // Scale B.
    let bnorm = b11.abs().max(b12.abs() + b22.abs()).max(safmin);
    let bsize = b11.abs().max(b22.abs());
    let bscale = ONE / bsize;
    b11 *= bscale;
    let b12 = b12 * bscale;
    b22 *= bscale;

    // Compute the larger eigenvalue by the method of C. van Loan.
    let binv11 = ONE / b11;
    let binv22 = ONE / b22;
    let s1 = a11 * binv11;
    let s2 = a22 * binv22;
    let as12;
    let abi22;
    let pp;
    let shift;
    let ss = a21 * (binv11 * binv22);
    if s1.abs() <= s2.abs() {
        as12 = a12 - s1 * b12;
        let as22 = a22 - s1 * b22;
        abi22 = as22 * binv22 - ss * b12;
        pp = HALF * abi22;
        shift = s1;
    } else {
        as12 = a12 - s2 * b12;
        let as11 = a11 - s2 * b11;
        abi22 = -ss * b12;
        pp = HALF * (as11 * binv11 + abi22);
        shift = s2;
    }
    let qq = ss * as12;
    let discr;
    let r;
    if (pp * rtmin).abs() >= ONE {
        discr = (rtmin * pp) * (rtmin * pp) + qq * safmin;
        r = discr.abs().sqrt() * rtmax;
    } else if pp * pp + qq.abs() <= safmin {
        discr = (rtmax * pp) * (rtmax * pp) + qq * safmax;
        r = discr.abs().sqrt() * rtmin;
    } else {
        discr = pp * pp + qq;
        r = discr.abs().sqrt();
    }

    let wr1;
    let wr2;
    let wi;
    if discr >= ZERO || r == ZERO {
        let sum = pp + r.copysign(pp);
        let diff = pp - r.copysign(pp);
        let wbig = shift + sum;
        let mut wsmall = shift + diff;
        if HALF * wbig.abs() > wsmall.abs().max(safmin) {
            let wdet = (a11 * a22 - a12 * a21) * (binv11 * binv22);
            wsmall = wdet / wbig;
        }
        if pp > abi22 {
            wr1 = wbig.min(wsmall);
            wr2 = wbig.max(wsmall);
        } else {
            wr1 = wbig.max(wsmall);
            wr2 = wbig.min(wsmall);
        }
        wi = ZERO;
    } else {
        wr1 = shift + pp;
        wr2 = wr1;
        wi = r;
    }

    // Further scaling to avoid over/underflow.
    let c1 = bsize * (safmin * ONE.max(ascale));
    let c2 = safmin * ONE.max(bnorm);
    let c3 = bsize * safmin;
    let c4 = if ascale <= ONE && bsize <= ONE {
        ONE.min((ascale / safmin) * bsize)
    } else {
        ONE
    };
    let c5 = if ascale <= ONE || bsize <= ONE {
        ONE.min(ascale * bsize)
    } else {
        ONE
    };

    // Scale the first eigenvalue.
    let mut wr1 = wr1;
    let mut wr2 = wr2;
    let mut wi = wi;
    let scale1;
    let mut scale2;
    let wabs = wr1.abs() + wi.abs();
    let mut wsize = safmin
        .max(c1)
        .max(FUZZY1 * (wabs * c2 + c3))
        .max(c4.min(HALF * wabs.max(c5)));
    if wsize != ONE {
        let wscale = ONE / wsize;
        if wsize > ONE {
            scale1 = (ascale.max(bsize) * wscale) * ascale.min(bsize);
        } else {
            scale1 = (ascale.min(bsize) * wscale) * ascale.max(bsize);
        }
        wr1 *= wscale;
        if wi != ZERO {
            wi *= wscale;
            wr2 = wr1;
            scale2 = scale1;
        } else {
            scale2 = scale1;
        }
    } else {
        scale1 = ascale * bsize;
        scale2 = scale1;
    }

    // Scale the second eigenvalue (if real).
    if wi == ZERO {
        wsize = safmin
            .max(c1)
            .max(FUZZY1 * (wr2.abs() * c2 + c3))
            .max(c4.min(HALF * wr2.abs().max(c5)));
        if wsize != ONE {
            let wscale = ONE / wsize;
            if wsize > ONE {
                scale2 = (ascale.max(bsize) * wscale) * ascale.min(bsize);
            } else {
                scale2 = (ascale.min(bsize) * wscale) * ascale.max(bsize);
            }
            wr2 *= wscale;
        } else {
            scale2 = ascale * bsize;
        }
    }
    (scale1, scale2, wr1, wr2, wi)
}

/// `dlasv2`: the singular value decomposition of the upper triangular
/// `[[f, g], [0, h]]`, returned as `(ssmin, ssmax, snr, csr, snl, csl)`.
fn dlasv2(f: f64, g: f64, h: f64) -> (f64, f64, f64, f64, f64, f64) {
    const ZERO: f64 = 0.0;
    const HALF: f64 = 0.5;
    const ONE: f64 = 1.0;
    const TWO: f64 = 2.0;
    const FOUR: f64 = 4.0;

    let mut ft = f;
    let mut fa = ft.abs();
    let mut ht = h;
    let mut ha = ht.abs();
    let mut pmax = 1usize;
    let swap = ha > fa;
    if swap {
        pmax = 3;
        std::mem::swap(&mut ft, &mut ht);
        std::mem::swap(&mut fa, &mut ha);
    }
    let gt = g;
    let ga = gt.abs();
    let mut ssmin;
    let mut ssmax;
    let mut clt;
    let mut crt;
    let mut slt;
    let mut srt;
    if ga == ZERO {
        // Diagonal matrix.
        ssmin = ha;
        ssmax = fa;
        clt = ONE;
        crt = ONE;
        slt = ZERO;
        srt = ZERO;
    } else {
        let mut gasmal = true;
        clt = ZERO;
        crt = ZERO;
        slt = ZERO;
        srt = ZERO;
        ssmin = ZERO;
        ssmax = ZERO;
        if ga > fa {
            pmax = 2;
            if (fa / ga) < DLAMCH_E {
                // Case of very large GA.
                gasmal = false;
                ssmax = ga;
                ssmin = if ha > ONE { fa / (ga / ha) } else { (fa / ga) * ha };
                clt = ONE;
                slt = ht / gt;
                srt = ONE;
                crt = ft / gt;
            }
        }
        if gasmal {
            // Normal case.
            let d = fa - ha;
            let l = if d == fa { ONE } else { d / fa };
            let m = gt / ft;
            let t = TWO - l;
            let mm = m * m;
            let tt = t * t;
            let s = (tt + mm).sqrt();
            let r = if l == ZERO { m.abs() } else { (l * l + mm).sqrt() };
            let a = HALF * (s + r);
            ssmin = ha / a;
            ssmax = fa * a;
            let t2 = if mm == ZERO {
                if l == ZERO {
                    TWO.copysign(ft) * ONE.copysign(gt)
                } else {
                    gt / d.copysign(ft) + m / t
                }
            } else {
                (m / (s + t) + m / (r + l)) * (ONE + a)
            };
            let l2 = (t2 * t2 + FOUR).sqrt();
            crt = TWO / l2;
            srt = t2 / l2;
            clt = (crt + srt * m) / a;
            slt = (ht / ft) * srt / a;
        }
    }
    let (csl, snl, csr, snr) = if swap {
        (srt, crt, slt, clt)
    } else {
        (clt, slt, crt, srt)
    };

    // Correct the signs of SSMAX and SSMIN.
    let tsign = match pmax {
        1 => ONE.copysign(csr) * ONE.copysign(csl) * ONE.copysign(f),
        2 => ONE.copysign(snr) * ONE.copysign(csl) * ONE.copysign(g),
        _ => ONE.copysign(snr) * ONE.copysign(snl) * ONE.copysign(h),
    };
    ssmax = ssmax.copysign(tsign);
    ssmin = ssmin.copysign(tsign * ONE.copysign(f) * ONE.copysign(h));
    (ssmin, ssmax, snr, csr, snl, csl)
}

// ---------------------------------------------------------------------------
// Points of the driver's pipeline.
// ---------------------------------------------------------------------------

/// `dlarf1f` with `SIDE = 'L'`: apply `H = I - tau v v^T` to the `m x ncol` block
/// of `a` at rows `r0..r0+m`, columns `c0..c0+ncol`.
///
/// `v` is explicit and has length `m`, with `v[0] = 1` (LAPACK does not store
/// that first entry; this transcription does, which removes the special case
/// while keeping the same arithmetic).
fn larf1f_left(a: &mut [f64], lda: usize, r0: usize, c0: usize, m: usize, ncol: usize, v: &[f64], tau: f64) {
    if tau == 0.0 || m == 0 || ncol == 0 {
        return;
    }
    // The last non-zero entry of v, 1-based, at least 1.
    let mut lastv = m;
    while lastv > 1 && v[lastv - 1] == 0.0 {
        lastv -= 1;
    }
    if lastv == 1 {
        // v = e1, so H scales the first row by (1 - tau).
        let factor = 1.0 - tau;
        for j in 0..ncol {
            a[r0 + (c0 + j) * lda] *= factor;
        }
        return;
    }
    for j in 0..ncol {
        // w = C(2:lastv, j)^T v(2:lastv), then plus C(1, j)*v(1) with v(1) = 1.
        let mut w = 0.0;
        for t in 1..lastv {
            w += a[(r0 + t) + (c0 + j) * lda] * v[t];
        }
        w += a[r0 + (c0 + j) * lda];
        a[r0 + (c0 + j) * lda] -= tau * w;
        for t in 1..lastv {
            a[(r0 + t) + (c0 + j) * lda] -= tau * v[t] * w;
        }
    }
}

/// Rebuild the explicit reflector `v` (with `v[0] = 1`) stored in column `col` of
/// `a`, rows `r0..r0+m`.
fn householder_column(a: &[f64], lda: usize, r0: usize, col: usize, m: usize) -> Vec<f64> {
    let mut v = vec![0.0; m];
    v[0] = 1.0;
    for t in 1..m {
        v[t] = a[(r0 + t) + col * lda];
    }
    v
}

/// `dgeqr2`: QR of the `m x ncol` block of `a` at rows `r0..`, columns `c0..`,
/// with the reflectors left in place and `tau` of length `min(m, ncol)`.
fn dgeqr2(a: &mut [f64], lda: usize, r0: usize, c0: usize, m: usize, ncol: usize, tau: &mut [f64]) {
    let k = m.min(ncol);
    for i in 0..k {
        let alpha = a[(r0 + i) + (c0 + i) * lda];
        let tail: Vec<f64> = (1..(m - i)).map(|t| a[(r0 + i + t) + (c0 + i) * lda]).collect();
        let (beta, taui, v) = dlarfg(alpha, &tail);
        a[(r0 + i) + (c0 + i) * lda] = beta;
        for t in 1..(m - i) {
            a[(r0 + i + t) + (c0 + i) * lda] = v[t];
        }
        tau[i] = taui;
        if i + 1 < ncol {
            larf1f_left(a, lda, r0 + i, c0 + i + 1, m - i, ncol - i - 1, &v, taui);
        }
    }
}

/// `dorm2r` with `SIDE = 'L'`, `TRANS = 'T'`: apply `Q^T` from the left, where
/// `Q = H(1)...H(k)` is built from the reflectors stored in the block of `bref`
/// at rows `rb..`, columns `cb..`, to the block of `a` at rows `r0..`, columns
/// `c0..` (both `m x ncol` / with `k` reflectors).
#[allow(clippy::too_many_arguments)]
fn dorm2r_left_trans(
    a: &mut [f64],
    lda: usize,
    r0: usize,
    c0: usize,
    m: usize,
    ncol: usize,
    bref: &[f64],
    ldb: usize,
    rb: usize,
    cb: usize,
    k: usize,
    tau: &[f64],
) {
    for i in 0..k {
        let mi = m - i;
        let v = householder_column(bref, ldb, rb + i, cb + i, mi);
        larf1f_left(a, lda, r0 + i, c0, mi, ncol, &v, tau[i]);
    }
}

/// `dorg2r`: generate the first `n` columns of the orthogonal `Q = H(1)...H(k)`
/// into the `m x n` block of `a` at rows `r0..`, columns `c0..`, whose reflectors
/// are stored there.
#[allow(clippy::too_many_arguments)]
fn dorg2r(a: &mut [f64], lda: usize, r0: usize, c0: usize, m: usize, n: usize, k: usize, tau: &[f64]) {
    // Fortran's IF( N.LE.0 ); for a `usize` that is exactly `n == 0`.
    if n == 0 {
        return;
    }
    // Initialize columns k..n to the corresponding columns of the unit matrix.
    for j in k..n {
        for l in 0..m {
            a[(r0 + l) + (c0 + j) * lda] = 0.0;
        }
        a[(r0 + j) + (c0 + j) * lda] = 1.0;
    }
    let mut i = k;
    while i > 0 {
        i -= 1;
        if i < n {
            let v = householder_column(a, lda, r0 + i, c0 + i, m - i);
            larf1f_left(a, lda, r0 + i, c0 + i + 1, m - i, n - i - 1, &v, tau[i]);
        }
        if i < m {
            for t in (i + 1)..m {
                a[(r0 + t) + (c0 + i) * lda] *= -tau[i];
            }
        }
        a[(r0 + i) + (c0 + i) * lda] = 1.0 - tau[i];
        for l in 0..i {
            a[(r0 + l) + (c0 + i) * lda] = 0.0;
        }
    }
}

/// Rotate columns `j1` and `j2` of an `n x n` matrix over rows `r0..r0+count`,
/// as `drot` does: `x' = c x + s y`, `y' = c y - s x`.
fn rot_col_range(a: &mut [f64], ld: usize, j1: usize, j2: usize, r0: usize, count: usize, c: f64, s: f64) {
    for i in r0..(r0 + count) {
        let x = a[i + j1 * ld];
        let y = a[i + j2 * ld];
        a[i + j1 * ld] = c * x + s * y;
        a[i + j2 * ld] = c * y - s * x;
    }
}

/// Rotate rows `i1` and `i2` of a matrix over columns `j0..j0+count`, as `drot`
/// does: `x' = c x + s y`, `y' = c y - s x`.
fn rot_row_range(a: &mut [f64], ld: usize, i1: usize, i2: usize, j0: usize, count: usize, c: f64, s: f64) {
    for j in j0..(j0 + count) {
        let x = a[i1 + j * ld];
        let y = a[i2 + j * ld];
        a[i1 + j * ld] = c * x + s * y;
        a[i2 + j * ld] = c * y - s * x;
    }
}

/// `dgghrd`: reduce the pair `(a, b)` to generalized Hessenberg form
/// (`a` upper Hessenberg, `b` upper triangular) using orthogonal transformations,
/// accumulating them into `q` and `z` when asked. `ilo` and `ihi` (0-based) bound
/// the active block.
#[allow(clippy::too_many_arguments)]
fn dgghrd(
    compq: bool,
    compz: bool,
    n: usize,
    ilo: usize,
    ihi: usize,
    a: &mut [f64],
    lda: usize,
    b: &mut [f64],
    ldb: usize,
    q: &mut [f64],
    ldq: usize,
    z: &mut [f64],
    ldz: usize,
) {
    if n <= 1 || ihi < ilo + 2 {
        // Still zero the strictly lower triangle of B, as the reference does
        // before its quick return on N <= 1.
        if n > 1 {
            for jcol in 0..(n - 1) {
                for jrow in (jcol + 1)..n {
                    b[jrow + jcol * ldb] = 0.0;
                }
            }
        }
        return;
    }

    // Zero the lower triangle of B.
    for jcol in 0..(n - 1) {
        for jrow in (jcol + 1)..n {
            b[jrow + jcol * ldb] = 0.0;
        }
    }

    // Reduce A and B. `jcol` and `jrow` are 0-based.
    for jcol in ilo..=(ihi - 2) {
        let mut jrow = ihi;
        loop {
            // Step 1: rotate rows jrow-1, jrow to kill a(jrow, jcol).
            let temp = a[(jrow - 1) + jcol * lda];
            let (c, s, r) = dlartg(temp, a[jrow + jcol * lda]);
            a[(jrow - 1) + jcol * lda] = r;
            a[jrow + jcol * lda] = 0.0;
            // DROT(N-JCOL, A(JROW-1,JCOL+1), LDA, A(JROW,JCOL+1), LDA, ...)
            if jcol < n {
                rot_row_range(a, lda, jrow - 1, jrow, jcol + 1, n - (jcol + 1), c, s);
            }
            // DROT(N+2-JROW, B(JROW-1,JROW-1), LDB, B(JROW,JROW-1), LDB, ...)
            rot_row_range(b, ldb, jrow - 1, jrow, jrow - 1, n + 1 - jrow, c, s);
            if compq {
                rot_col_range(q, ldq, jrow - 1, jrow, 0, n, c, s);
            }

            // Step 2: rotate columns jrow, jrow-1 to kill b(jrow, jrow-1).
            let temp = b[jrow + jrow * ldb];
            let (c, s, r) = dlartg(temp, b[jrow + (jrow - 1) * ldb]);
            b[jrow + jrow * ldb] = r;
            b[jrow + (jrow - 1) * ldb] = 0.0;
            // DROT(IHI, A(1,JROW), 1, A(1,JROW-1), 1, ...): IHI is 1-based in the
            // reference, so the row count is `ihi + 1` here.
            rot_col_range(a, lda, jrow, jrow - 1, 0, ihi + 1, c, s);
            // DROT(JROW-1, B(1,JROW), 1, B(1,JROW-1), 1, ...): JROW-1 in 1-based
            // is `jrow` in 0-based, which is also the count.
            rot_col_range(b, ldb, jrow, jrow - 1, 0, jrow, c, s);
            if compz {
                rot_col_range(z, ldz, jrow, jrow - 1, 0, n, c, s);
            }

            if jrow <= jcol + 2 {
                break;
            }
            jrow -= 1;
        }
    }
}

/// `dggbal` with `JOB = 'P'`: permute the pair `(a, b)` to isolate eigenvalues
/// and return `(ilo, ihi, lscale, rscale)` (0-based `ilo`, exclusive `ihi`).
///
/// The permutation part is all `dggev` asks for, so the conjugate-gradient
/// balancing that `JOB = 'B'` adds is not transcribed.
fn dggbal_p(n: usize, a: &mut [f64], lda: usize, b: &mut [f64], ldb: usize) -> (usize, usize, Vec<f64>, Vec<f64>) {
    let mut lscale = vec![0.0; n];
    let mut rscale = vec![0.0; n];
    if n == 0 {
        return (0, 0, lscale, rscale);
    }
    if n == 1 {
        lscale[0] = 1.0;
        rscale[0] = 1.0;
        return (0, 1, lscale, rscale);
    }
    // 1-based working values, as in the reference.
    let mut k = 1usize;
    let mut l = n;
    let mut lm1 = 0usize;
    let mut flow = 0u8; // 0: first entry, 1: from label 20, 2: from label 90
    'outer: loop {
        if flow == 1 {
            l = lm1;
            if l == 1 {
                rscale[0] = 1.0;
                lscale[0] = 1.0;
                break 'outer;
            }
        } else if flow == 2 {
            k += 1;
        }

        // label 30: find a row with one nonzero in columns 1..L.
        lm1 = l - 1;
        let mut found_flow = 0u8;
        let mut m = 0usize;
        let mut ifound = 0usize;
        let mut jfound = 0usize;
        let mut row_permuted = false;
        'row: for i in (1..=l).rev() {
            let mut j = 0usize;
            let mut nonz = false;
            for jj in 1..=lm1 {
                if a[(i - 1) + (jj - 1) * lda] != 0.0 || b[(i - 1) + (jj - 1) * ldb] != 0.0 {
                    j = jj;
                    nonz = true;
                    break;
                }
            }
            if !nonz {
                j = l;
            } else {
                // label 50
                let jp1 = j + 1;
                let mut found = false;
                for jj in jp1..=l {
                    if a[(i - 1) + (jj - 1) * lda] != 0.0 || b[(i - 1) + (jj - 1) * ldb] != 0.0 {
                        found = true;
                        break;
                    }
                }
                if found {
                    continue 'row;
                }
                j = jp1 - 1;
            }
            // label 70
            m = l;
            ifound = i;
            jfound = j;
            found_flow = 1;
            row_permuted = true;
            break 'row;
        }
        if row_permuted {
            // label 160
            lscale[m - 1] = ifound as f64;
            if ifound != m {
                // DSWAP(N-K+1, A(I,K), LDA, A(M,K), LDA)
                swap_rows(a, lda, ifound - 1, m - 1, k - 1, n - k + 1);
                swap_rows(b, ldb, ifound - 1, m - 1, k - 1, n - k + 1);
            }
            // label 170
            rscale[m - 1] = jfound as f64;
            if jfound != m {
                // DSWAP(L, A(1,J), 1, A(1,M), 1)
                swap_cols(a, lda, jfound - 1, m - 1, 0, l);
                swap_cols(b, ldb, jfound - 1, m - 1, 0, l);
            }
            let _ = found_flow;
            flow = 1;
            continue 'outer;
        }

        // label 100: find a column with one nonzero in rows K..N.
        let mut col_permuted = false;
        loop {
            let mut m2 = 0usize;
            let mut ifound2 = 0usize;
            let mut jfound2 = 0usize;
            'col: for j in k..=l {
                let mut i = 0usize;
                let mut nonz = false;
                for ii in k..=lm1 {
                    if a[(ii - 1) + (j - 1) * lda] != 0.0 || b[(ii - 1) + (j - 1) * ldb] != 0.0 {
                        i = ii;
                        nonz = true;
                        break;
                    }
                }
                if !nonz {
                    i = l;
                } else {
                    let ip1 = i + 1;
                    let mut found = false;
                    for ii in ip1..=l {
                        if a[(ii - 1) + (j - 1) * lda] != 0.0 || b[(ii - 1) + (j - 1) * ldb] != 0.0 {
                            found = true;
                            break;
                        }
                    }
                    if found {
                        continue 'col;
                    }
                    i = ip1 - 1;
                }
                // label 140
                m2 = k;
                ifound2 = i;
                jfound2 = j;
                col_permuted = true;
                break 'col;
            }
            if !col_permuted {
                break 'outer;
            }
            // label 160 with IFLOW = 2
            lscale[m2 - 1] = ifound2 as f64;
            if ifound2 != m2 {
                swap_rows(a, lda, ifound2 - 1, m2 - 1, k - 1, n - k + 1);
                swap_rows(b, ldb, ifound2 - 1, m2 - 1, k - 1, n - k + 1);
            }
            rscale[m2 - 1] = jfound2 as f64;
            if jfound2 != m2 {
                swap_cols(a, lda, jfound2 - 1, m2 - 1, 0, l);
                swap_cols(b, ldb, jfound2 - 1, m2 - 1, 0, l);
            }
            // label 180 -> 90 -> 100
            k += 1;
        }
    }
    // label 190 with JOB = 'P'
    for i in (k - 1)..l {
        lscale[i] = 1.0;
        rscale[i] = 1.0;
    }
    (k - 1, l, lscale, rscale)
}

/// Swap the `count` entries of rows `i` and `m` starting at column `j0`.
fn swap_rows<T: Copy>(a: &mut [T], ld: usize, i: usize, m: usize, j0: usize, count: usize) {
    if i == m {
        return;
    }
    for j in j0..(j0 + count) {
        a.swap(i + j * ld, m + j * ld);
    }
}

/// Swap the `count` entries of columns `j` and `m` starting at row `i0`.
fn swap_cols<T: Copy>(a: &mut [T], ld: usize, j: usize, m: usize, i0: usize, count: usize) {
    if j == m {
        return;
    }
    for i in i0..(i0 + count) {
        a.swap(i + j * ld, i + m * ld);
    }
}

/// `dggbak`/`zggbak` with `JOB = 'P'`: undo the permutation recorded in
/// `lscale`/`rscale` on `m` eigenvectors stored in the columns of `v`. `right`
/// selects the right (`rscale`) or left (`lscale`) permutation.
///
/// The reference has one routine per type; the body only swaps rows of `v`, so
/// [`dggbak_p`] and [`zggbak_p`] are thin wrappers over this.
#[allow(clippy::too_many_arguments)]
fn ggbak_p<T: Copy>(
    right: bool,
    n: usize,
    ilo: usize,
    ihi: usize,
    lscale: &[f64],
    rscale: &[f64],
    m: usize,
    v: &mut [T],
    ldv: usize,
) {
    if n == 0 || m == 0 {
        return;
    }
    // Note that `ilo == ihi` still reaches the permutation: the reference only
    // skips the backward *balance*, then jumps to the permutation block.
    // `ilo` is 0-based and `ihi` 0-based inclusive here.
    let scale = if right { rscale } else { lscale };
    // `scale` stores 1-based partner indices, as the reference does. The
    // reference swaps `DSWAP(M, V(I,1), LDV, V(K,1), LDV)`: the increment is
    // LDV, so it swaps *rows* I and K of V across all M columns.
    if ilo > 0 {
        for i in (0..ilo).rev() {
            let k = scale[i] as usize;
            if k == i + 1 {
                continue;
            }
            swap_rows(v, ldv, i, k - 1, 0, m);
        }
    }
    for i in (ihi + 1)..n {
        let k = scale[i] as usize;
        if k == i + 1 {
            continue;
        }
        swap_rows(v, ldv, i, k - 1, 0, m);
    }
}

/// `dggbak` with `JOB = 'P'`, for real eigenvectors: see [`ggbak_p`].
#[allow(clippy::too_many_arguments)]
fn dggbak_p(
    right: bool,
    n: usize,
    ilo: usize,
    ihi: usize,
    lscale: &[f64],
    rscale: &[f64],
    m: usize,
    v: &mut [f64],
    ldv: usize,
) {
    ggbak_p(right, n, ilo, ihi, lscale, rscale, m, v, ldv)
}

/// `zggbak` with `JOB = 'P'`, for complex eigenvectors: see [`ggbak_p`].
#[allow(clippy::too_many_arguments)]
fn zggbak_p(
    right: bool,
    n: usize,
    ilo: usize,
    ihi: usize,
    lscale: &[f64],
    rscale: &[f64],
    m: usize,
    v: &mut [Complex64],
    ldv: usize,
) {
    ggbak_p(right, n, ilo, ihi, lscale, rscale, m, v, ldv)
}

/// `dhgeqz` with `JOB = 'S'` or `'E'`: the QZ iteration.
///
/// On entry `h` and `t` are in generalized Hessenberg form (`h` upper Hessenberg,
/// `t` upper triangular); on exit they hold the generalized Schur form and the
/// generalized eigenvalues are in `alphar`/`alphai`/`beta`. The orthogonal
/// factors are accumulated into `q` and `z` when `ilq`/`ilz`. Returns the
/// reference's `INFO` (0 on success, `1..=n` for non-convergence, `n+1` for the
/// "impossible" drop-through).
#[allow(clippy::too_many_arguments)]
fn dhgeqz(
    ilschr: bool,
    ilq: bool,
    ilz: bool,
    n: usize,
    ilo: usize,
    ihi: usize,
    h: &mut [f64],
    ldh: usize,
    t: &mut [f64],
    ldt: usize,
    alphar: &mut [f64],
    alphai: &mut [f64],
    beta: &mut [f64],
    q: &mut [f64],
    ldq: usize,
    z: &mut [f64],
    ldz: usize,
) -> usize {
    const SAFETY: f64 = 1.0e2;
    const HALF: f64 = 0.5;
    const ONE: f64 = 1.0;
    const ZERO: f64 = 0.0;

    // Fortran's IF( N.LE.0 ); for a `usize` that is exactly `n == 0`.
    if n == 0 {
        return 0;
    }

    let in_ = ihi + 1 - ilo;
    let safmin = SAFMIN;
    let safmax = ONE / safmin;
    let ulp = ULP;
    let anorm = dlanhs_f(&h[(ilo + ilo * ldh)..], ldh, in_);
    let bnorm = dlanhs_f(&t[(ilo + ilo * ldt)..], ldt, in_);
    let atol = safmin.max(ulp * anorm);
    let btol = safmin.max(ulp * bnorm);
    let ascale = ONE / safmin.max(anorm);
    let bscale = ONE / safmin.max(bnorm);

    // Set eigenvalues ihi+1:n.
    for j in (ihi + 1)..n {
        if t[j + j * ldt] < ZERO {
            if ilschr {
                for jr in 0..=j {
                    h[jr + j * ldh] = -h[jr + j * ldh];
                    t[jr + j * ldt] = -t[jr + j * ldt];
                }
            } else {
                h[j + j * ldh] = -h[j + j * ldh];
                t[j + j * ldt] = -t[j + j * ldt];
            }
            if ilz {
                for jr in 0..n {
                    z[jr + j * ldz] = -z[jr + j * ldz];
                }
            }
        }
        alphar[j] = h[j + j * ldh];
        alphai[j] = ZERO;
        beta[j] = t[j + j * ldt];
    }

    // If ihi < ilo, skip the QZ steps.
    if ihi < ilo {
        return 0;
    }

    let mut ilast = ihi;
    let (mut ifrstm, mut ilastm) = if ilschr { (0usize, n - 1) } else { (ilo, ihi) };
    let mut iiter = 0usize;
    let mut eshift = ZERO;
    let maxit = 30 * (ihi - ilo + 1);

    // Per-iteration destination, mirroring the reference's GOTO labels.
    enum Next {
        /// Label 80, reached through label 70 when `via70`.
        Deflate { via70: bool },
        /// Label 110 with the given IFIRST.
        QzStep(usize),
    }

    for _jiter in 0..maxit {
        let next: Next;

        // ---- Split the matrix if possible. ----
        if ilast == ilo {
            next = Next::Deflate { via70: false };
        } else if h[ilast + (ilast - 1) * ldh].abs()
            <= safmin.max(ulp * (h[ilast + ilast * ldh].abs() + h[(ilast - 1) + (ilast - 1) * ldh].abs()))
        {
            h[ilast + (ilast - 1) * ldh] = ZERO;
            next = Next::Deflate { via70: false };
        } else if t[ilast + ilast * ldt].abs() <= btol {
            t[ilast + ilast * ldt] = ZERO;
            next = Next::Deflate { via70: true };
        } else {
            // General case: j < ilast.
            let mut chosen: Option<Next> = None;
            let mut j = ilast - 1;
            while j >= ilo {
                let ilazro = if j == ilo {
                    true
                } else if h[j + (j - 1) * ldh].abs()
                    <= safmin.max(ulp * (h[j + j * ldh].abs() + h[(j - 1) + (j - 1) * ldh].abs()))
                {
                    h[j + (j - 1) * ldh] = ZERO;
                    true
                } else {
                    false
                };

                if t[j + j * ldt].abs() < btol {
                    t[j + j * ldt] = ZERO;

                    // Test 1a: two consecutive small subdiagonals in A.
                    let mut ilazr2 = false;
                    if !ilazro {
                        let mut temp = h[j + (j - 1) * ldh].abs();
                        let mut temp2 = h[j + j * ldh].abs();
                        let tempr = temp.max(temp2);
                        if tempr < ONE && tempr != ZERO {
                            temp /= tempr;
                            temp2 /= tempr;
                        }
                        if temp * (ascale * h[(j + 1) + j * ldh].abs()) <= temp2 * (ascale * atol) {
                            ilazr2 = true;
                        }
                    }

                    if ilazro || ilazr2 {
                        // Split a 1x1 block off at the top.
                        let mut jch = j;
                        loop {
                            let temp = h[jch + jch * ldh];
                            let (c, s, r) = dlartg(temp, h[(jch + 1) + jch * ldh]);
                            h[jch + jch * ldh] = r;
                            h[(jch + 1) + jch * ldh] = ZERO;
                            rot_row_range(h, ldh, jch, jch + 1, jch + 1, ilastm - jch, c, s);
                            rot_row_range(t, ldt, jch, jch + 1, jch + 1, ilastm - jch, c, s);
                            if ilq {
                                rot_col_range(q, ldq, jch, jch + 1, 0, n, c, s);
                            }
                            if ilazr2 {
                                h[jch + (jch - 1) * ldh] *= c;
                            }
                            ilazr2 = false;
                            if t[(jch + 1) + (jch + 1) * ldt].abs() >= btol {
                                if jch + 1 >= ilast {
                                    chosen = Some(Next::Deflate { via70: false });
                                } else {
                                    chosen = Some(Next::QzStep(jch + 1));
                                }
                                break;
                            }
                            t[(jch + 1) + (jch + 1) * ldt] = ZERO;
                            // The reference loop is JCH = J, ILAST-1 inclusive;
                            // `ilast - 1` is its last 0-based index.
                            if jch >= ilast - 1 {
                                chosen = Some(Next::Deflate { via70: true });
                                break;
                            }
                            jch += 1;
                        }
                    } else {
                        // Only test 2 passed: chase the zero to T(ilast, ilast).
                        // The reference loop is JCH = J, ILAST-1 inclusive, so the
                        // 0-based range runs to `ilast - 1`.
                        for jch in j..ilast {
                            let temp = t[jch + (jch + 1) * ldt];
                            let (c, s, r) = dlartg(temp, t[(jch + 1) + (jch + 1) * ldt]);
                            t[jch + (jch + 1) * ldt] = r;
                            t[(jch + 1) + (jch + 1) * ldt] = ZERO;
                            if jch < ilastm - 1 {
                                rot_row_range(t, ldt, jch, jch + 1, jch + 2, ilastm - jch - 1, c, s);
                            }
                            rot_row_range(h, ldh, jch, jch + 1, jch - 1, ilastm - jch + 2, c, s);
                            if ilq {
                                rot_col_range(q, ldq, jch, jch + 1, 0, n, c, s);
                            }
                            let temp = h[(jch + 1) + jch * ldh];
                            let (c, s, r) = dlartg(temp, h[(jch + 1) + (jch - 1) * ldh]);
                            h[(jch + 1) + jch * ldh] = r;
                            h[(jch + 1) + (jch - 1) * ldh] = ZERO;
                            rot_col_range(h, ldh, jch, jch - 1, ifrstm, jch + 1 - ifrstm, c, s);
                            rot_col_range(t, ldt, jch, jch - 1, ifrstm, jch - ifrstm, c, s);
                            if ilz {
                                rot_col_range(z, ldz, jch, jch - 1, 0, n, c, s);
                            }
                        }
                        chosen = Some(Next::Deflate { via70: true });
                    }
                    break;
                } else if ilazro {
                    chosen = Some(Next::QzStep(j));
                    break;
                }

                if j == ilo {
                    break;
                }
                j -= 1;
            }
            match chosen {
                Some(nx) => next = nx,
                None => return n + 1,
            }
        }

        match next {
            Next::Deflate { via70 } => {
                if via70 {
                    // Label 70: clear H(ilast, ilast-1) to split off a 1x1 block.
                    let temp = h[ilast + ilast * ldh];
                    let (c, s, r) = dlartg(temp, h[ilast + (ilast - 1) * ldh]);
                    h[ilast + ilast * ldh] = r;
                    h[ilast + (ilast - 1) * ldh] = ZERO;
                    rot_col_range(h, ldh, ilast, ilast - 1, ifrstm, ilast - ifrstm, c, s);
                    rot_col_range(t, ldt, ilast, ilast - 1, ifrstm, ilast - ifrstm, c, s);
                    if ilz {
                        rot_col_range(z, ldz, ilast, ilast - 1, 0, n, c, s);
                    }
                }
                // Label 80: standardize B, set alphar, alphai and beta.
                if t[ilast + ilast * ldt] < ZERO {
                    if ilschr {
                        for j in ifrstm..=ilast {
                            h[j + ilast * ldh] = -h[j + ilast * ldh];
                            t[j + ilast * ldt] = -t[j + ilast * ldt];
                        }
                    } else {
                        h[ilast + ilast * ldh] = -h[ilast + ilast * ldh];
                        t[ilast + ilast * ldt] = -t[ilast + ilast * ldt];
                    }
                    if ilz {
                        for j in 0..n {
                            z[j + ilast * ldz] = -z[j + ilast * ldz];
                        }
                    }
                }
                alphar[ilast] = h[ilast + ilast * ldh];
                alphai[ilast] = ZERO;
                beta[ilast] = t[ilast + ilast * ldt];

                if ilast == ilo {
                    return 0;
                }
                ilast -= 1;
                iiter = 0;
                eshift = ZERO;
                if !ilschr {
                    ilastm = ilast;
                    if ifrstm > ilast {
                        ifrstm = ilo;
                    }
                }
            }

            Next::QzStep(ifirst) => {
                // Label 110: the QZ step.
                iiter += 1;
                if !ilschr {
                    ifrstm = ifirst;
                }

                // Compute single shifts.
                let mut s1;
                let mut wr;
                let wi;
                let mut do_double = false;
                if (iiter / 10) * 10 == iiter {
                    // Exceptional shift.
                    if (maxit as f64 * safmin) * h[ilast + (ilast - 1) * ldh].abs()
                        < t[(ilast - 1) + (ilast - 1) * ldt].abs()
                    {
                        eshift = h[ilast + (ilast - 1) * ldh] / t[(ilast - 1) + (ilast - 1) * ldt];
                    } else {
                        eshift += ONE / (safmin * maxit as f64);
                    }
                    s1 = ONE;
                    wr = eshift;
                } else {
                    // Shifts from the bottom-right 2x2 block.
                    let (s1v, s2v, wrv, wr2v, wiv) = dlag2(
                        h[(ilast - 1) + (ilast - 1) * ldh],
                        h[ilast + (ilast - 1) * ldh],
                        h[(ilast - 1) + ilast * ldh],
                        h[ilast + ilast * ldh],
                        t[(ilast - 1) + (ilast - 1) * ldt],
                        t[(ilast - 1) + ilast * ldt],
                        t[ilast + ilast * ldt],
                        safmin * SAFETY,
                    );
                    let mut s1l = s1v;
                    let mut s2l = s2v;
                    let mut wrl = wrv;
                    let mut wr2l = wr2v;
                    if ((wrl / s1l) * t[ilast + ilast * ldt] - h[ilast + ilast * ldh]).abs()
                        > ((wr2l / s2l) * t[ilast + ilast * ldt] - h[ilast + ilast * ldh]).abs()
                    {
                        std::mem::swap(&mut wrl, &mut wr2l);
                        std::mem::swap(&mut s1l, &mut s2l);
                    }
                    s1 = s1l;
                    wr = wrl;
                    wi = wiv;
                    if wi != ZERO {
                        do_double = true;
                    }
                }

                if !do_double {
                    // Fiddle with the shift to avoid overflow.
                    let temp = ascale.min(ONE) * (HALF * safmax);
                    let mut scale = if s1 > temp { temp / s1 } else { ONE };
                    let temp = bscale.min(ONE) * (HALF * safmax);
                    if wr.abs() > temp {
                        scale = scale.min(temp / wr.abs());
                    }
                    s1 *= scale;
                    wr *= scale;

                    // Check for two consecutive small subdiagonals.
                    let mut istart = ifirst;
                    let mut j = ilast - 1;
                    while j > ifirst {
                        let mut temp = (s1 * h[j + (j - 1) * ldh]).abs();
                        let mut temp2 = (s1 * h[j + j * ldh] - wr * t[j + j * ldt]).abs();
                        let tempr = temp.max(temp2);
                        if tempr < ONE && tempr != ZERO {
                            temp /= tempr;
                            temp2 /= tempr;
                        }
                        if (ascale * h[(j + 1) + j * ldh]).abs() * temp <= (ascale * atol) * temp2 {
                            istart = j;
                            break;
                        }
                        if j == ifirst + 1 {
                            break;
                        }
                        j -= 1;
                    }

                    // Implicit single-shift QZ sweep.
                    let temp = s1 * h[istart + istart * ldh] - wr * t[istart + istart * ldt];
                    let temp2 = s1 * h[(istart + 1) + istart * ldh];
                    let (mut c, mut s, _tempr) = dlartg(temp, temp2);

                    for j in istart..=(ilast - 1) {
                        if j > istart {
                            let temp = h[j + (j - 1) * ldh];
                            let (c2, s2, r) = dlartg(temp, h[(j + 1) + (j - 1) * ldh]);
                            h[j + (j - 1) * ldh] = r;
                            h[(j + 1) + (j - 1) * ldh] = ZERO;
                            c = c2;
                            s = s2;
                        }
                        for jc in j..=ilastm {
                            let temp = c * h[j + jc * ldh] + s * h[(j + 1) + jc * ldh];
                            h[(j + 1) + jc * ldh] = -s * h[j + jc * ldh] + c * h[(j + 1) + jc * ldh];
                            h[j + jc * ldh] = temp;
                            let temp2 = c * t[j + jc * ldt] + s * t[(j + 1) + jc * ldt];
                            t[(j + 1) + jc * ldt] = -s * t[j + jc * ldt] + c * t[(j + 1) + jc * ldt];
                            t[j + jc * ldt] = temp2;
                        }
                        if ilq {
                            for jr in 0..n {
                                let temp = c * q[jr + j * ldq] + s * q[jr + (j + 1) * ldq];
                                q[jr + (j + 1) * ldq] = -s * q[jr + j * ldq] + c * q[jr + (j + 1) * ldq];
                                q[jr + j * ldq] = temp;
                            }
                        }

                        let temp = t[(j + 1) + (j + 1) * ldt];
                        let (c2, s2, r) = dlartg(temp, t[(j + 1) + j * ldt]);
                        t[(j + 1) + (j + 1) * ldt] = r;
                        t[(j + 1) + j * ldt] = ZERO;
                        c = c2;
                        s = s2;

                        for jr in ifrstm..=(j + 2).min(ilast) {
                            let temp = c * h[jr + (j + 1) * ldh] + s * h[jr + j * ldh];
                            h[jr + j * ldh] = -s * h[jr + (j + 1) * ldh] + c * h[jr + j * ldh];
                            h[jr + (j + 1) * ldh] = temp;
                        }
                        for jr in ifrstm..=j {
                            let temp = c * t[jr + (j + 1) * ldt] + s * t[jr + j * ldt];
                            t[jr + j * ldt] = -s * t[jr + (j + 1) * ldt] + c * t[jr + j * ldt];
                            t[jr + (j + 1) * ldt] = temp;
                        }
                        if ilz {
                            for jr in 0..n {
                                let temp = c * z[jr + (j + 1) * ldz] + s * z[jr + j * ldz];
                                z[jr + j * ldz] = -s * z[jr + (j + 1) * ldz] + c * z[jr + j * ldz];
                                z[jr + (j + 1) * ldz] = temp;
                            }
                        }
                    }
                } else if ifirst + 1 == ilast {
                    // Special case: 2x2 block with complex eigenvalues.
                    // Step 1: standardize so B = diag(B11, B22) with B11 >= 0.
                    let (ssmin, ssmax, snr, csr, snl, csl) = dlasv2(
                        t[(ilast - 1) + (ilast - 1) * ldt],
                        t[(ilast - 1) + ilast * ldt],
                        t[ilast + ilast * ldt],
                    );
                    let mut b22 = ssmin;
                    let mut b11 = ssmax;
                    let mut sr = snr;
                    let mut cr = csr;
                    let sl = snl;
                    let cl = csl;
                    if b11 < ZERO {
                        cr = -cr;
                        sr = -sr;
                        b11 = -b11;
                        b22 = -b22;
                    }
                    rot_row_range(h, ldh, ilast - 1, ilast, ilast - 1, ilastm + 1 - ifirst, cl, sl);
                    rot_col_range(h, ldh, ilast - 1, ilast, ifrstm, ilast + 1 - ifrstm, cr, sr);
                    if ilast < ilastm {
                        rot_row_range(t, ldt, ilast - 1, ilast, ilast + 1, ilastm - ilast, cl, sl);
                    }
                    if ifrstm < ilast - 1 {
                        rot_col_range(t, ldt, ilast - 1, ilast, ifrstm, ifirst - ifrstm, cr, sr);
                    }
                    if ilq {
                        rot_col_range(q, ldq, ilast - 1, ilast, 0, n, cl, sl);
                    }
                    if ilz {
                        rot_col_range(z, ldz, ilast - 1, ilast, 0, n, cr, sr);
                    }
                    t[(ilast - 1) + (ilast - 1) * ldt] = b11;
                    t[(ilast - 1) + ilast * ldt] = ZERO;
                    t[ilast + (ilast - 1) * ldt] = ZERO;
                    t[ilast + ilast * ldt] = b22;

                    if b22 < ZERO {
                        for j in ifrstm..=ilast {
                            h[j + ilast * ldh] = -h[j + ilast * ldh];
                            t[j + ilast * ldt] = -t[j + ilast * ldt];
                        }
                        if ilz {
                            for j in 0..n {
                                z[j + ilast * ldz] = -z[j + ilast * ldz];
                            }
                        }
                        b22 = -b22;
                    }

                    // Step 2: recompute the shift and the eigenvalue data.
                    let (s1v, _temp, wrv, _temp2, wiv) = dlag2(
                        h[(ilast - 1) + (ilast - 1) * ldh],
                        h[ilast + (ilast - 1) * ldh],
                        h[(ilast - 1) + ilast * ldh],
                        h[ilast + ilast * ldh],
                        t[(ilast - 1) + (ilast - 1) * ldt],
                        t[(ilast - 1) + ilast * ldt],
                        t[ilast + ilast * ldt],
                        safmin * SAFETY,
                    );
                    if wiv == ZERO {
                        // Standardization perturbed the shift onto the real line.
                        continue;
                    }
                    let s1inv = ONE / s1v;
                    let a11 = h[(ilast - 1) + (ilast - 1) * ldh];
                    let a21 = h[ilast + (ilast - 1) * ldh];
                    let a12 = h[(ilast - 1) + ilast * ldh];
                    let a22 = h[ilast + ilast * ldh];

                    // Complex Givens rotation on the right.
                    let c11r = s1v * a11 - wrv * b11;
                    let c11i = -wiv * b11;
                    let c12 = s1v * a12;
                    let c21 = s1v * a21;
                    let c22r = s1v * a22 - wrv * b22;
                    let c22i = -wiv * b22;
                    let mut cz;
                    let szr;
                    let szi;
                    if c11r.abs() + c11i.abs() + c12.abs() > c21.abs() + c22r.abs() + c22i.abs() {
                        let t1 = dlapy3(c12, c11r, c11i);
                        cz = c12 / t1;
                        szr = -c11r / t1;
                        szi = -c11i / t1;
                    } else {
                        cz = dlapy2(c22r, c22i);
                        if cz <= safmin {
                            cz = ZERO;
                            szr = ONE;
                            szi = ZERO;
                        } else {
                            let tempr = c22r / cz;
                            let tempi = c22i / cz;
                            let t1 = dlapy2(cz, c21);
                            cz /= t1;
                            szr = -c21 * tempr / t1;
                            szi = c21 * tempi / t1;
                        }
                    }

                    // Givens rotation on the left.
                    let an = a11.abs() + a12.abs() + a21.abs() + a22.abs();
                    let bn = b11.abs() + b22.abs();
                    let wabs = wrv.abs() + wiv.abs();
                    let mut cq;
                    let sqr;
                    let sqi;
                    if s1v * an > wabs * bn {
                        cq = cz * b11;
                        sqr = szr * b22;
                        sqi = -szi * b22;
                    } else {
                        let a1r = cz * a11 + szr * a12;
                        let a1i = szi * a12;
                        let a2r = cz * a21 + szr * a22;
                        let a2i = szi * a22;
                        cq = dlapy2(a1r, a1i);
                        if cq <= safmin {
                            cq = ZERO;
                            sqr = ONE;
                            sqi = ZERO;
                        } else {
                            let tempr = a1r / cq;
                            let tempi = a1i / cq;
                            sqr = tempr * a2r + tempi * a2i;
                            sqi = tempi * a2r - tempr * a2i;
                        }
                    }
                    let t1 = dlapy3(cq, sqr, sqi);
                    let cq = cq / t1;
                    let sqr = sqr / t1;
                    let sqi = sqi / t1;

                    // Diagonal elements of QBZ.
                    let tempr = sqr * szr - sqi * szi;
                    let tempi = sqr * szi + sqi * szr;
                    let b1r = cq * cz * b11 + tempr * b22;
                    let b1i = tempi * b22;
                    let b1a = dlapy2(b1r, b1i);
                    let b2r = cq * cz * b22 + tempr * b11;
                    let b2i = -tempi * b11;
                    let b2a = dlapy2(b2r, b2i);

                    // Normalize so beta > 0 and Im(alpha1) > 0.
                    beta[ilast - 1] = b1a;
                    beta[ilast] = b2a;
                    alphar[ilast - 1] = (wrv * b1a) * s1inv;
                    alphai[ilast - 1] = (wiv * b1a) * s1inv;
                    alphar[ilast] = (wrv * b2a) * s1inv;
                    alphai[ilast] = -(wiv * b2a) * s1inv;

                    // Step 3: go to the next block.
                    if ifirst == ilo {
                        return 0;
                    }
                    ilast = ifirst - 1;
                    iiter = 0;
                    eshift = ZERO;
                    if !ilschr {
                        ilastm = ilast;
                        if ifrstm > ilast {
                            ifrstm = ilo;
                        }
                    }
                } else {
                    // Usual case: 3x3 or larger block, Francis double shift.
                    let ad11 =
                        (ascale * h[(ilast - 1) + (ilast - 1) * ldh]) / (bscale * t[(ilast - 1) + (ilast - 1) * ldt]);
                    let ad21 = (ascale * h[ilast + (ilast - 1) * ldh]) / (bscale * t[(ilast - 1) + (ilast - 1) * ldt]);
                    let ad12 = (ascale * h[(ilast - 1) + ilast * ldh]) / (bscale * t[ilast + ilast * ldt]);
                    let ad22 = (ascale * h[ilast + ilast * ldh]) / (bscale * t[ilast + ilast * ldt]);
                    let u12 = t[(ilast - 1) + ilast * ldt] / t[ilast + ilast * ldt];
                    let ad11l = (ascale * h[ifirst + ifirst * ldh]) / (bscale * t[ifirst + ifirst * ldt]);
                    let ad21l = (ascale * h[(ifirst + 1) + ifirst * ldh]) / (bscale * t[ifirst + ifirst * ldt]);
                    let ad12l =
                        (ascale * h[ifirst + (ifirst + 1) * ldh]) / (bscale * t[(ifirst + 1) + (ifirst + 1) * ldt]);
                    let ad22l = (ascale * h[(ifirst + 1) + (ifirst + 1) * ldh])
                        / (bscale * t[(ifirst + 1) + (ifirst + 1) * ldt]);
                    let ad32l = (ascale * h[(ifirst + 2) + (ifirst + 1) * ldh])
                        / (bscale * t[(ifirst + 1) + (ifirst + 1) * ldt]);
                    let u12l = t[ifirst + (ifirst + 1) * ldt] / t[(ifirst + 1) + (ifirst + 1) * ldt];

                    let v1 = (ad11 - ad11l) * (ad22 - ad11l) - ad12 * ad21
                        + ad21 * u12 * ad11l
                        + (ad12l - ad11l * u12l) * ad21l;
                    let v2 = ((ad22l - ad11l) - ad21l * u12l - (ad11 - ad11l) - (ad22 - ad11l) + ad21 * u12) * ad21l;
                    let v3 = ad32l * ad21l;

                    let istart = ifirst;
                    let (_beta_v, mut tau, vout) = dlarfg(v1, &[v2, v3]);
                    let mut vv = [ONE, vout[1], vout[2]];

                    for j in istart..=(ilast - 2) {
                        if j > istart {
                            let alpha = h[j + (j - 1) * ldh];
                            let (beta_v, tau2, v2v) =
                                dlarfg(alpha, &[h[(j + 1) + (j - 1) * ldh], h[(j + 2) + (j - 1) * ldh]]);
                            h[j + (j - 1) * ldh] = beta_v;
                            vv = [ONE, v2v[1], v2v[2]];
                            tau = tau2;
                            h[(j + 1) + (j - 1) * ldh] = ZERO;
                            h[(j + 2) + (j - 1) * ldh] = ZERO;
                        }
                        let t2 = tau * vv[1];
                        let t3 = tau * vv[2];
                        for jc in j..=ilastm {
                            let temp = h[j + jc * ldh] + vv[1] * h[(j + 1) + jc * ldh] + vv[2] * h[(j + 2) + jc * ldh];
                            h[j + jc * ldh] -= temp * tau;
                            h[(j + 1) + jc * ldh] -= temp * t2;
                            h[(j + 2) + jc * ldh] -= temp * t3;
                            let temp2 = t[j + jc * ldt] + vv[1] * t[(j + 1) + jc * ldt] + vv[2] * t[(j + 2) + jc * ldt];
                            t[j + jc * ldt] -= temp2 * tau;
                            t[(j + 1) + jc * ldt] -= temp2 * t2;
                            t[(j + 2) + jc * ldt] -= temp2 * t3;
                        }
                        if ilq {
                            for jr in 0..n {
                                let temp =
                                    q[jr + j * ldq] + vv[1] * q[jr + (j + 1) * ldq] + vv[2] * q[jr + (j + 2) * ldq];
                                q[jr + j * ldq] -= temp * tau;
                                q[jr + (j + 1) * ldq] -= temp * t2;
                                q[jr + (j + 2) * ldq] -= temp * t3;
                            }
                        }

                        // Zero the j-th column of B (see DLAGBC in the reference).
                        let mut ilpivt = false;
                        let temp = t[(j + 1) + (j + 1) * ldt].abs().max(t[(j + 1) + (j + 2) * ldt].abs());
                        let temp2 = t[(j + 2) + (j + 1) * ldt].abs().max(t[(j + 2) + (j + 2) * ldt].abs());
                        let mut w11;
                        let mut w12;
                        let mut w21;
                        let mut w22;
                        let mut u1;
                        let mut u2;
                        let mut scale = ONE;
                        if temp.max(temp2) < safmin {
                            scale = ZERO;
                            u1 = ONE;
                            u2 = ZERO;
                        } else {
                            if temp >= temp2 {
                                w11 = t[(j + 1) + (j + 1) * ldt];
                                w21 = t[(j + 2) + (j + 1) * ldt];
                                w12 = t[(j + 1) + (j + 2) * ldt];
                                w22 = t[(j + 2) + (j + 2) * ldt];
                                u1 = t[(j + 1) + j * ldt];
                                u2 = t[(j + 2) + j * ldt];
                            } else {
                                w21 = t[(j + 1) + (j + 1) * ldt];
                                w11 = t[(j + 2) + (j + 1) * ldt];
                                w22 = t[(j + 1) + (j + 2) * ldt];
                                w12 = t[(j + 2) + (j + 2) * ldt];
                                u2 = t[(j + 1) + j * ldt];
                                u1 = t[(j + 2) + j * ldt];
                            }
                            // Swap columns if necessary.
                            if w12.abs() > w11.abs() {
                                ilpivt = true;
                                let tt = w12;
                                let t22 = w22;
                                w12 = w11;
                                w22 = w21;
                                w11 = tt;
                                w21 = t22;
                            }
                            // LU-factor.
                            let tt = w21 / w11;
                            u2 -= tt * u1;
                            w22 -= tt * w12;
                            // Compute SCALE and solve.
                            if w22.abs() < safmin {
                                scale = ZERO;
                                u2 = ONE;
                                u1 = -w12 / w11;
                            } else {
                                if w22.abs() < u2.abs() {
                                    scale = (w22 / u2).abs();
                                }
                                if w11.abs() < u1.abs() {
                                    scale = scale.min((w11 / u1).abs());
                                }
                                u2 = (scale * u2) / w22;
                                u1 = (scale * u1 - w12 * u2) / w11;
                            }
                        }
                        if ilpivt {
                            std::mem::swap(&mut u1, &mut u2);
                        }
                        let t1 = (scale * scale + u1 * u1 + u2 * u2).sqrt();
                        let tau = ONE + scale / t1;
                        let vs = -ONE / (scale + t1);
                        let vv2 = [ONE, vs * u1, vs * u2];

                        let t2 = tau * vv2[1];
                        let t3 = tau * vv2[2];
                        for jr in ifrstm..=(j + 3).min(ilast) {
                            let temp =
                                h[jr + j * ldh] + vv2[1] * h[jr + (j + 1) * ldh] + vv2[2] * h[jr + (j + 2) * ldh];
                            h[jr + j * ldh] -= temp * tau;
                            h[jr + (j + 1) * ldh] -= temp * t2;
                            h[jr + (j + 2) * ldh] -= temp * t3;
                        }
                        for jr in ifrstm..=(j + 2) {
                            let temp =
                                t[jr + j * ldt] + vv2[1] * t[jr + (j + 1) * ldt] + vv2[2] * t[jr + (j + 2) * ldt];
                            t[jr + j * ldt] -= temp * tau;
                            t[jr + (j + 1) * ldt] -= temp * t2;
                            t[jr + (j + 2) * ldt] -= temp * t3;
                        }
                        if ilz {
                            for jr in 0..n {
                                let temp =
                                    z[jr + j * ldz] + vv2[1] * z[jr + (j + 1) * ldz] + vv2[2] * z[jr + (j + 2) * ldz];
                                z[jr + j * ldz] -= temp * tau;
                                z[jr + (j + 1) * ldz] -= temp * t2;
                                z[jr + (j + 2) * ldz] -= temp * t3;
                            }
                        }
                        t[(j + 1) + j * ldt] = ZERO;
                        t[(j + 2) + j * ldt] = ZERO;
                    }

                    // Last elements: Givens rotations.
                    let j = ilast - 1;
                    let temp = h[j + (j - 1) * ldh];
                    let (c, s, r) = dlartg(temp, h[(j + 1) + (j - 1) * ldh]);
                    h[j + (j - 1) * ldh] = r;
                    h[(j + 1) + (j - 1) * ldh] = ZERO;
                    for jc in j..=ilastm {
                        let temp = c * h[j + jc * ldh] + s * h[(j + 1) + jc * ldh];
                        h[(j + 1) + jc * ldh] = -s * h[j + jc * ldh] + c * h[(j + 1) + jc * ldh];
                        h[j + jc * ldh] = temp;
                        let temp2 = c * t[j + jc * ldt] + s * t[(j + 1) + jc * ldt];
                        t[(j + 1) + jc * ldt] = -s * t[j + jc * ldt] + c * t[(j + 1) + jc * ldt];
                        t[j + jc * ldt] = temp2;
                    }
                    if ilq {
                        for jr in 0..n {
                            let temp = c * q[jr + j * ldq] + s * q[jr + (j + 1) * ldq];
                            q[jr + (j + 1) * ldq] = -s * q[jr + j * ldq] + c * q[jr + (j + 1) * ldq];
                            q[jr + j * ldq] = temp;
                        }
                    }
                    let temp = t[(j + 1) + (j + 1) * ldt];
                    let (c, s, r) = dlartg(temp, t[(j + 1) + j * ldt]);
                    t[(j + 1) + (j + 1) * ldt] = r;
                    t[(j + 1) + j * ldt] = ZERO;
                    for jr in ifrstm..=ilast {
                        let temp = c * h[jr + (j + 1) * ldh] + s * h[jr + j * ldh];
                        h[jr + j * ldh] = -s * h[jr + (j + 1) * ldh] + c * h[jr + j * ldh];
                        h[jr + (j + 1) * ldh] = temp;
                    }
                    for jr in ifrstm..=(ilast - 1) {
                        let temp = c * t[jr + (j + 1) * ldt] + s * t[jr + j * ldt];
                        t[jr + j * ldt] = -s * t[jr + (j + 1) * ldt] + c * t[jr + j * ldt];
                        t[jr + (j + 1) * ldt] = temp;
                    }
                    if ilz {
                        for jr in 0..n {
                            let temp = c * z[jr + (j + 1) * ldz] + s * z[jr + j * ldz];
                            z[jr + j * ldz] = -s * z[jr + (j + 1) * ldz] + c * z[jr + j * ldz];
                            z[jr + (j + 1) * ldz] = temp;
                        }
                    }
                }
            }
        }
    }

    // Drop-through: non-convergence.
    ilast + 1
}

/// 1-based accessors, so the transcription of `dtgevc` reads like the Fortran.
#[inline]
fn mget(a: &[f64], ld: usize, i: usize, j: usize) -> f64 {
    a[(i - 1) + (j - 1) * ld]
}

/// 1-based setter matching [`mget`].
#[inline]
fn mset(a: &mut [f64], ld: usize, i: usize, j: usize, x: f64) {
    a[(i - 1) + (j - 1) * ld] = x;
}

/// `dtgevc` with `HOWMNY = 'B'` and `SIDE = 'B'`, `'L'` or `'R'`: the left
/// (`compl`) and right (`compr`) generalized eigenvectors of the pair `(s, p)`,
/// which are on entry the generalized Schur form and the Schur vectors and on
/// exit the eigenvectors.
///
/// `HOWMNY = 'B'` means every eigenvector is computed and back-transformed with
/// the Schur vectors supplied in `vl`/`vr`. `SELECT` is not referenced in that
/// case, so it is absent here. Returns the reference's `INFO`.
#[allow(clippy::too_many_arguments)]
fn dtgevc(
    compl: bool,
    compr: bool,
    n: usize,
    s: &[f64],
    lds: usize,
    p: &[f64],
    ldp: usize,
    vl: &mut [f64],
    ldvl: usize,
    vr: &mut [f64],
    ldvr: usize,
) -> usize {
    const SAFETY: f64 = 1.0e2;
    const ONE: f64 = 1.0;

    if n == 0 {
        return 0;
    }

    let safmin = SAFMIN;
    let ulp = ULP;
    let small = safmin * n as f64 / ulp;
    // The reference computes BIG = 1/SAFMIN first and then overwrites it.
    let big = ONE / small;
    let bignum = ONE / (safmin * n as f64);

    // 1-based workspace, as in the reference.
    let mut work = vec![0.0; 6 * n + 1];

    // Column norms of the strictly upper triangular parts.
    let mut anorm = mget(s, lds, 1, 1).abs();
    if n > 1 {
        anorm += mget(s, lds, 2, 1).abs();
    }
    let mut bnorm = mget(p, ldp, 1, 1).abs();
    work[1] = 0.0;
    work[n + 1] = 0.0;
    for j in 2..=n {
        let mut temp = 0.0;
        let mut temp2 = 0.0;
        let iend = if mget(s, lds, j, j - 1) == 0.0 { j - 1 } else { j - 2 };
        for i in 1..=iend {
            temp += mget(s, lds, i, j).abs();
            temp2 += mget(p, ldp, i, j).abs();
        }
        work[j] = temp;
        work[n + j] = temp2;
        for i in (iend + 1)..=(j + 1).min(n) {
            temp += mget(s, lds, i, j).abs();
            temp2 += mget(p, ldp, i, j).abs();
        }
        anorm = anorm.max(temp);
        bnorm = bnorm.max(temp2);
    }
    let ascale = ONE / anorm.max(safmin);
    let bscale = ONE / bnorm.max(safmin);

    // --- Left eigenvectors ---
    if compl {
        let mut ieig = 0usize;
        let mut ilcplx = false;
        for je in 1..=n {
            if ilcplx {
                ilcplx = false;
                continue;
            }
            let mut nw = 1usize;
            if je < n && mget(s, lds, je + 1, je) != 0.0 {
                ilcplx = true;
                nw = 2;
            }
            // ILALL is true, so ILCOMP is always true.

            if !ilcplx && mget(s, lds, je, je).abs() <= safmin && mget(p, ldp, je, je).abs() <= safmin {
                // Singular pencil: return the unit eigenvector.
                ieig += 1;
                for jr in 1..=n {
                    mset(vl, ldvl, jr, ieig, 0.0);
                }
                mset(vl, ldvl, ieig, ieig, 1.0);
                continue;
            }

            for jr in 1..=(nw * n) {
                work[2 * n + jr] = 0.0;
            }

            let acoef;
            let bcoefr;
            let bcoefi;
            let mut acoefa;
            let mut bcoefa;
            let mut xmax;
            if !ilcplx {
                // Real eigenvalue.
                let temp = ONE
                    / (mget(s, lds, je, je).abs() * ascale)
                        .max(mget(p, ldp, je, je).abs() * bscale)
                        .max(safmin);
                let salfar = (temp * mget(s, lds, je, je)) * ascale;
                let sbeta = (temp * mget(p, ldp, je, je)) * bscale;
                let mut a = sbeta * ascale;
                let mut br = salfar * bscale;
                let bi = 0.0;
                let mut scale = ONE;
                let lsa = sbeta.abs() >= safmin && a.abs() < small;
                let lsb = salfar.abs() >= safmin && br.abs() < small;
                if lsa {
                    scale = (small / sbeta.abs()) * anorm.min(big);
                }
                if lsb {
                    scale = scale.max((small / salfar.abs()) * bnorm.min(big));
                }
                if lsa || lsb {
                    scale = scale.min(ONE / (safmin * ONE.max(a.abs()).max(br.abs())));
                    if lsa {
                        a = ascale * (scale * sbeta);
                    } else {
                        a *= scale;
                    }
                    if lsb {
                        br = bscale * (scale * salfar);
                    } else {
                        br *= scale;
                    }
                }
                acoef = a;
                bcoefr = br;
                bcoefi = bi;
                acoefa = acoef.abs();
                bcoefa = bcoefr.abs();
                work[2 * n + je] = ONE;
                xmax = ONE;
            } else {
                // Complex eigenvalue.
                let (a0, _temp, br0, _temp2, bi0) = dlag2(
                    mget(s, lds, je, je),
                    mget(s, lds, je + 1, je),
                    mget(s, lds, je, je + 1),
                    mget(s, lds, je + 1, je + 1),
                    mget(p, ldp, je, je),
                    mget(p, ldp, je, je + 1),
                    mget(p, ldp, je + 1, je + 1),
                    safmin * SAFETY,
                );
                let mut a = a0;
                let mut br = br0;
                let mut bi = -bi0;
                if bi == 0.0 {
                    return je;
                }
                acoefa = a.abs();
                bcoefa = br.abs() + bi.abs();
                let mut scale = ONE;
                if acoefa * ulp < safmin && acoefa >= safmin {
                    scale = (safmin / ulp) / acoefa;
                }
                if bcoefa * ulp < safmin && bcoefa >= safmin {
                    scale = scale.max((safmin / ulp) / bcoefa);
                }
                if safmin * acoefa > ascale {
                    scale = ascale / (safmin * acoefa);
                }
                if safmin * bcoefa > bscale {
                    scale = scale.min(bscale / (safmin * bcoefa));
                }
                if scale != ONE {
                    a *= scale;
                    acoefa = a.abs();
                    br *= scale;
                    bi *= scale;
                    bcoefa = br.abs() + bi.abs();
                }
                // First two components of the eigenvector.
                let temp = a * mget(s, lds, je + 1, je);
                let temp2r = a * mget(s, lds, je, je) - br * mget(p, ldp, je, je);
                let temp2i = -bi * mget(p, ldp, je, je);
                if temp.abs() > temp2r.abs() + temp2i.abs() {
                    work[2 * n + je] = ONE;
                    work[3 * n + je] = 0.0;
                    work[2 * n + je + 1] = -temp2r / temp;
                    work[3 * n + je + 1] = -temp2i / temp;
                } else {
                    work[2 * n + je + 1] = ONE;
                    work[3 * n + je + 1] = 0.0;
                    let temp = a * mget(s, lds, je, je + 1);
                    work[2 * n + je] = (br * mget(p, ldp, je + 1, je + 1) - a * mget(s, lds, je + 1, je + 1)) / temp;
                    work[3 * n + je] = bi * mget(p, ldp, je + 1, je + 1) / temp;
                }
                acoef = a;
                bcoefr = br;
                bcoefi = bi;
                xmax = (work[2 * n + je].abs() + work[3 * n + je].abs())
                    .max(work[2 * n + je + 1].abs() + work[3 * n + je + 1].abs());
            }

            let dmin = (ulp * acoefa * anorm).max(ulp * bcoefa * bnorm).max(safmin);

            // Triangular solve of (a A - b B)^T y = 0.
            let mut il2by2 = false;
            for j in (je + nw)..=n {
                if il2by2 {
                    il2by2 = false;
                    continue;
                }
                let na;
                let bdiag1 = mget(p, ldp, j, j);
                let bdiag2;
                if j < n && mget(s, lds, j + 1, j) != 0.0 {
                    il2by2 = true;
                    bdiag2 = mget(p, ldp, j + 1, j + 1);
                    na = 2;
                } else {
                    bdiag2 = 0.0;
                    na = 1;
                }

                // Check whether scaling is necessary for the dot products.
                let xscale = ONE / ONE.max(xmax);
                let mut temp = work[j].max(work[n + j]).max(acoefa * work[j] + bcoefa * work[n + j]);
                if il2by2 {
                    temp = temp
                        .max(work[j + 1])
                        .max(work[n + j + 1])
                        .max(acoefa * work[j + 1] + bcoefa * work[n + j + 1]);
                }
                if temp > bignum * xscale {
                    for jw in 0..nw {
                        for jr in je..j {
                            work[(jw + 2) * n + jr] *= xscale;
                        }
                    }
                    xmax *= xscale;
                }

                // Dot products: sum_{k=je}^{j-1} conjg(a S(k,j) - b P(k,j)) x(k).
                let mut sums = [[0.0f64; 3]; 3];
                let mut sump = [[0.0f64; 3]; 3];
                for jw in 1..=nw {
                    for ja in 1..=na {
                        let mut ss = 0.0;
                        let mut sp = 0.0;
                        for jr in je..j {
                            ss += mget(s, lds, jr, j + ja - 1) * work[(jw + 1) * n + jr];
                            sp += mget(p, ldp, jr, j + ja - 1) * work[(jw + 1) * n + jr];
                        }
                        sums[ja][jw] = ss;
                        sump[ja][jw] = sp;
                    }
                }
                let mut sumv = [[0.0f64; 3]; 3];
                for ja in 1..=na {
                    if ilcplx {
                        sumv[ja][1] = -acoef * sums[ja][1] + bcoefr * sump[ja][1] - bcoefi * sump[ja][2];
                        sumv[ja][2] = -acoef * sums[ja][2] + bcoefr * sump[ja][2] + bcoefi * sump[ja][1];
                    } else {
                        sumv[ja][1] = -acoef * sums[ja][1] + bcoefr * sump[ja][1];
                    }
                }

                // A is S(J:J+1, J:J+1); for a 1x1 block only A(1,1) is read, and
                // J can be N, so the neighbors must not be evaluated eagerly.
                let (sa11, sa21, sa12, sa22) = if na == 2 {
                    (
                        mget(s, lds, j, j),
                        mget(s, lds, j + 1, j),
                        mget(s, lds, j, j + 1),
                        mget(s, lds, j + 1, j + 1),
                    )
                } else {
                    (mget(s, lds, j, j), 0.0, 0.0, 0.0)
                };
                let res = dlaln2(
                    true, na, nw, dmin, acoef, sa11, sa21, sa12, sa22, bdiag1, bdiag2, sumv[1][1], sumv[2][1],
                    sumv[1][2], sumv[2][2], bcoefr, bcoefi,
                );
                work[2 * n + j] = res.x11;
                if na == 2 {
                    work[2 * n + j + 1] = res.x21;
                }
                if nw == 2 {
                    work[3 * n + j] = res.x12;
                    if na == 2 {
                        work[3 * n + j + 1] = res.x22;
                    }
                }
                let scale = res.scale;
                let xnorm = res.xnorm;
                if scale < ONE {
                    for jw in 0..nw {
                        for jr in je..j {
                            work[(jw + 2) * n + jr] *= scale;
                        }
                    }
                    xmax *= scale;
                }
                xmax = xmax.max(xnorm);
            }

            // Copy the eigenvector to VL, back-transforming (HOWMNY = 'B').
            ieig += 1;
            for jw in 0..nw {
                for i in 1..=n {
                    let mut acc = 0.0;
                    for c in 0..(n + 1 - je) {
                        acc += mget(vl, ldvl, i, je + c) * work[(jw + 2) * n + je + c];
                    }
                    work[(jw + 4) * n + i] = acc;
                }
            }
            for jw in 0..nw {
                for jr in 1..=n {
                    let x = work[(jw + 4) * n + jr];
                    mset(vl, ldvl, jr, ieig + jw, x);
                }
            }

            // Scale the eigenvector.
            let ibeg = 1;
            let mut xmax = 0.0f64;
            if ilcplx {
                for j in ibeg..=n {
                    xmax = xmax.max(mget(vl, ldvl, j, ieig).abs() + mget(vl, ldvl, j, ieig + 1).abs());
                }
            } else {
                for j in ibeg..=n {
                    xmax = xmax.max(mget(vl, ldvl, j, ieig).abs());
                }
            }
            if xmax > safmin {
                let xs = ONE / xmax;
                for jw in 0..nw {
                    for jr in ibeg..=n {
                        let x = xs * mget(vl, ldvl, jr, ieig + jw);
                        mset(vl, ldvl, jr, ieig + jw, x);
                    }
                }
            }
            ieig += nw - 1;
        }
    }

    // --- Right eigenvectors ---
    if compr {
        let mut ieig = n + 1; // 1-based index of the next free column
        let mut ilcplx = false;
        for je in (1..=n).rev() {
            if ilcplx {
                ilcplx = false;
                continue;
            }
            let mut nw = 1usize;
            if je > 1 && mget(s, lds, je, je - 1) != 0.0 {
                ilcplx = true;
                nw = 2;
            }
            // ILALL is true, so ILCOMP is always true.

            if !ilcplx && mget(s, lds, je, je).abs() <= safmin && mget(p, ldp, je, je).abs() <= safmin {
                // Singular pencil: unit eigenvector.
                ieig -= 1;
                let col = ieig;
                for jr in 1..=n {
                    mset(vr, ldvr, jr, col, 0.0);
                }
                mset(vr, ldvr, col, col, 1.0);
                continue;
            }

            for jw in 0..nw {
                for jr in 1..=n {
                    work[(jw + 2) * n + jr] = 0.0;
                }
            }

            let acoef;
            let bcoefr;
            let bcoefi;
            let mut acoefa;
            let mut bcoefa;
            let mut xmax;
            if !ilcplx {
                // Real eigenvalue.
                let temp = ONE
                    / (mget(s, lds, je, je).abs() * ascale)
                        .max(mget(p, ldp, je, je).abs() * bscale)
                        .max(safmin);
                let salfar = (temp * mget(s, lds, je, je)) * ascale;
                let sbeta = (temp * mget(p, ldp, je, je)) * bscale;
                let mut a = sbeta * ascale;
                let mut br = salfar * bscale;
                let bi = 0.0;
                let mut scale = ONE;
                let lsa = sbeta.abs() >= safmin && a.abs() < small;
                let lsb = salfar.abs() >= safmin && br.abs() < small;
                if lsa {
                    scale = (small / sbeta.abs()) * anorm.min(big);
                }
                if lsb {
                    scale = scale.max((small / salfar.abs()) * bnorm.min(big));
                }
                if lsa || lsb {
                    scale = scale.min(ONE / (safmin * ONE.max(a.abs()).max(br.abs())));
                    if lsa {
                        a = ascale * (scale * sbeta);
                    } else {
                        a *= scale;
                    }
                    if lsb {
                        br = bscale * (scale * salfar);
                    } else {
                        br *= scale;
                    }
                }
                acoef = a;
                bcoefr = br;
                bcoefi = bi;
                acoefa = acoef.abs();
                bcoefa = bcoefr.abs();
                work[2 * n + je] = ONE;
                xmax = ONE;
                // Contribution from column JE of A and B to the sums.
                for jr in 1..je {
                    work[2 * n + jr] = bcoefr * mget(p, ldp, jr, je) - acoef * mget(s, lds, jr, je);
                }
            } else {
                // Complex eigenvalue (2x2 block in rows/columns je-1:je).
                let (a0, _temp, br0, _temp2, bi0) = dlag2(
                    mget(s, lds, je - 1, je - 1),
                    mget(s, lds, je, je - 1),
                    mget(s, lds, je - 1, je),
                    mget(s, lds, je, je),
                    mget(p, ldp, je - 1, je - 1),
                    mget(p, ldp, je - 1, je),
                    mget(p, ldp, je, je),
                    safmin * SAFETY,
                );
                let mut a = a0;
                let mut br = br0;
                let mut bi = bi0;
                if bi == 0.0 {
                    return je - 1;
                }
                acoefa = a.abs();
                bcoefa = br.abs() + bi.abs();
                let mut scale = ONE;
                if acoefa * ulp < safmin && acoefa >= safmin {
                    scale = (safmin / ulp) / acoefa;
                }
                if bcoefa * ulp < safmin && bcoefa >= safmin {
                    scale = scale.max((safmin / ulp) / bcoefa);
                }
                if safmin * acoefa > ascale {
                    scale = ascale / (safmin * acoefa);
                }
                if safmin * bcoefa > bscale {
                    scale = scale.min(bscale / (safmin * bcoefa));
                }
                if scale != ONE {
                    a *= scale;
                    acoefa = a.abs();
                    br *= scale;
                    bi *= scale;
                    bcoefa = br.abs() + bi.abs();
                }
                // First two components and their contributions to the sums.
                let temp = a * mget(s, lds, je, je - 1);
                let temp2r = a * mget(s, lds, je, je) - br * mget(p, ldp, je, je);
                let temp2i = -bi * mget(p, ldp, je, je);
                if temp.abs() >= temp2r.abs() + temp2i.abs() {
                    work[2 * n + je] = ONE;
                    work[3 * n + je] = 0.0;
                    work[2 * n + je - 1] = -temp2r / temp;
                    work[3 * n + je - 1] = -temp2i / temp;
                } else {
                    work[2 * n + je - 1] = ONE;
                    work[3 * n + je - 1] = 0.0;
                    let temp = a * mget(s, lds, je - 1, je);
                    work[2 * n + je] = (br * mget(p, ldp, je - 1, je - 1) - a * mget(s, lds, je - 1, je - 1)) / temp;
                    work[3 * n + je] = bi * mget(p, ldp, je - 1, je - 1) / temp;
                }
                xmax = (work[2 * n + je].abs() + work[3 * n + je].abs())
                    .max(work[2 * n + je - 1].abs() + work[3 * n + je - 1].abs());

                let creala = a * work[2 * n + je - 1];
                let cimaga = a * work[3 * n + je - 1];
                let crealb = br * work[2 * n + je - 1] - bi * work[3 * n + je - 1];
                let cimagb = bi * work[2 * n + je - 1] + br * work[3 * n + je - 1];
                let cre2a = a * work[2 * n + je];
                let cim2a = a * work[3 * n + je];
                let cre2b = br * work[2 * n + je] - bi * work[3 * n + je];
                let cim2b = bi * work[2 * n + je] + br * work[3 * n + je];
                for jr in 1..(je - 1) {
                    work[2 * n + jr] = -creala * mget(s, lds, jr, je - 1) + crealb * mget(p, ldp, jr, je - 1)
                        - cre2a * mget(s, lds, jr, je)
                        + cre2b * mget(p, ldp, jr, je);
                    work[3 * n + jr] = -cimaga * mget(s, lds, jr, je - 1) + cimagb * mget(p, ldp, jr, je - 1)
                        - cim2a * mget(s, lds, jr, je)
                        + cim2b * mget(p, ldp, jr, je);
                }
                acoef = a;
                bcoefr = br;
                bcoefi = bi;
            }

            let dmin = (ulp * acoefa * anorm).max(ulp * bcoefa * bnorm).max(safmin);

            // Columnwise triangular solve of (a A - b B) x = 0.
            let mut il2by2 = false;
            let start = je as isize - nw as isize;
            let mut jj = start;
            while jj >= 1 {
                let j = jj as usize;
                if !il2by2 && j > 1 && mget(s, lds, j, j - 1) != 0.0 {
                    il2by2 = true;
                    jj -= 1;
                    continue;
                }
                let bdiag1 = mget(p, ldp, j, j);
                let na;
                let bdiag2;
                if il2by2 {
                    na = 2;
                    bdiag2 = mget(p, ldp, j + 1, j + 1);
                } else {
                    na = 1;
                    bdiag2 = 0.0;
                }

                // A is S(J:J+1, J:J+1); only A(1,1) is read for a 1x1 block.
                let (sa11, sa21, sa12, sa22) = if na == 2 {
                    (
                        mget(s, lds, j, j),
                        mget(s, lds, j + 1, j),
                        mget(s, lds, j, j + 1),
                        mget(s, lds, j + 1, j + 1),
                    )
                } else {
                    (mget(s, lds, j, j), 0.0, 0.0, 0.0)
                };
                let res = dlaln2(
                    false,
                    na,
                    nw,
                    dmin,
                    acoef,
                    sa11,
                    sa21,
                    sa12,
                    sa22,
                    bdiag1,
                    bdiag2,
                    work[2 * n + j],
                    work[2 * n + j + 1],
                    work[3 * n + j],
                    work[3 * n + j + 1],
                    bcoefr,
                    bcoefi,
                );
                if res.scale < ONE {
                    for jw in 0..nw {
                        for jr in 1..=je {
                            work[(jw + 2) * n + jr] *= res.scale;
                        }
                    }
                }
                xmax = (res.scale * xmax).max(res.xnorm);

                // WORK((JW+1)*N+J+JA-1) = SUM(JA,JW).
                work[2 * n + j] = res.x11;
                if na == 2 {
                    work[2 * n + j + 1] = res.x21;
                }
                if nw == 2 {
                    work[3 * n + j] = res.x12;
                    if na == 2 {
                        work[3 * n + j + 1] = res.x22;
                    }
                }

                // w = w + x(j) * (a S(*,j) - b P(*,j)), with scaling.
                if j > 1 {
                    let xscale = ONE / ONE.max(xmax);
                    let mut temp = acoefa * work[j] + bcoefa * work[n + j];
                    if il2by2 {
                        temp = temp.max(acoefa * work[j + 1] + bcoefa * work[n + j + 1]);
                    }
                    temp = temp.max(acoefa).max(bcoefa);
                    if temp > bignum * xscale {
                        for jw in 0..nw {
                            for jr in 1..=je {
                                work[(jw + 2) * n + jr] *= xscale;
                            }
                        }
                        xmax *= xscale;
                    }
                    for ja in 1..=na {
                        if ilcplx {
                            let creala = acoef * work[2 * n + j + ja - 1];
                            let cimaga = acoef * work[3 * n + j + ja - 1];
                            let crealb = bcoefr * work[2 * n + j + ja - 1] - bcoefi * work[3 * n + j + ja - 1];
                            let cimagb = bcoefi * work[2 * n + j + ja - 1] + bcoefr * work[3 * n + j + ja - 1];
                            for jr in 1..j {
                                work[2 * n + jr] = work[2 * n + jr] - creala * mget(s, lds, jr, j + ja - 1)
                                    + crealb * mget(p, ldp, jr, j + ja - 1);
                                work[3 * n + jr] = work[3 * n + jr] - cimaga * mget(s, lds, jr, j + ja - 1)
                                    + cimagb * mget(p, ldp, jr, j + ja - 1);
                            }
                        } else {
                            let creala = acoef * work[2 * n + j + ja - 1];
                            let crealb = bcoefr * work[2 * n + j + ja - 1];
                            for jr in 1..j {
                                work[2 * n + jr] = work[2 * n + jr] - creala * mget(s, lds, jr, j + ja - 1)
                                    + crealb * mget(p, ldp, jr, j + ja - 1);
                            }
                        }
                    }
                }
                il2by2 = false;
                jj -= 1;
            }

            // Copy the eigenvector to VR, back-transforming (HOWMNY = 'B').
            ieig -= nw;
            let ieig_col = ieig;
            for jw in 0..nw {
                for jr in 1..=n {
                    work[(jw + 4) * n + jr] = work[(jw + 2) * n + 1] * mget(vr, ldvr, jr, 1);
                }
                for jc in 2..=je {
                    for jr in 1..=n {
                        work[(jw + 4) * n + jr] += work[(jw + 2) * n + jc] * mget(vr, ldvr, jr, jc);
                    }
                }
            }
            for jw in 0..nw {
                for jr in 1..=n {
                    let x = work[(jw + 4) * n + jr];
                    mset(vr, ldvr, jr, ieig_col + jw, x);
                }
            }

            // Scale the eigenvector.
            let iend = n;
            let mut xmax = 0.0f64;
            if ilcplx {
                for j in 1..=iend {
                    xmax = xmax.max(mget(vr, ldvr, j, ieig_col).abs() + mget(vr, ldvr, j, ieig_col + 1).abs());
                }
            } else {
                for j in 1..=iend {
                    xmax = xmax.max(mget(vr, ldvr, j, ieig_col).abs());
                }
            }
            if xmax > safmin {
                let xs = ONE / xmax;
                for jw in 0..nw {
                    for jr in 1..=iend {
                        let x = xs * mget(vr, ldvr, jr, ieig_col + jw);
                        mset(vr, ldvr, jr, ieig_col + jw, x);
                    }
                }
            }
        }
    }

    0
}

/// Normalize the eigenvectors in the columns of `v`: for a real eigenvalue the
/// largest `|v(i,j)|` is 1, for a complex pair the largest `|Re| + |Im|` is 1.
/// Negative `alphai` marks the second half of a pair and is skipped.
#[allow(clippy::too_many_arguments)]
fn dggev_normalize(n: usize, smlnum: f64, alphai: &[f64], v: &mut [f64], ldv: usize) {
    for jc in 0..n {
        if alphai[jc] < 0.0 {
            continue;
        }
        let mut temp = 0.0f64;
        if alphai[jc] == 0.0 {
            for jr in 0..n {
                temp = temp.max(v[jr + jc * ldv].abs());
            }
        } else {
            for jr in 0..n {
                temp = temp.max(v[jr + jc * ldv].abs() + v[jr + (jc + 1) * ldv].abs());
            }
        }
        if temp < smlnum {
            continue;
        }
        let temp = 1.0 / temp;
        if alphai[jc] == 0.0 {
            for jr in 0..n {
                v[jr + jc * ldv] *= temp;
            }
        } else {
            for jr in 0..n {
                v[jr + jc * ldv] *= temp;
                v[jr + (jc + 1) * ldv] *= temp;
            }
        }
    }
}

/// `dlaset('Full', n, n, 0, 1, a)`: set `a` to the identity.
fn dlaset_identity(a: &mut [f64], ld: usize, n: usize) {
    for j in 0..n {
        for i in 0..n {
            a[i + j * ld] = if i == j { 1.0 } else { 0.0 };
        }
    }
}

/// `dggev`: the generalized eigenvalues of the pair `(a, b)` and, optionally,
/// the left and/or right generalized eigenvectors.
///
/// This follows LAPACK 3.12.1's `dggev` with `JOBVL`/`JOBVR` decoded into
/// `calc_vl`/`calc_vr`. On exit `a` and `b` are overwritten, `alphar`, `alphai`
/// and `beta` hold the eigenvalues as `(alphar + i alphai) / beta`, and `vl`/`vr`
/// hold the eigenvectors in LAPACK's compact storage: for a conjugate pair,
/// columns `j` and `j + 1` are the real and imaginary parts of `v(j)`.
///
/// # Errors
///
/// [`LapackError::Convergence`] when the reference's `INFO` is non-zero: `1..=n`
/// for QZ non-convergence, `n+1..=2n` for an argument it rejected, and `n + 2`
/// when `dtgevc` failed to compute the eigenvectors.
#[allow(clippy::too_many_arguments)]
pub fn dggev(
    calc_vl: bool,
    calc_vr: bool,
    n: usize,
    a: &mut [f64],
    b: &mut [f64],
    alphar: &mut [f64],
    alphai: &mut [f64],
    beta: &mut [f64],
    vl: &mut [f64],
    vr: &mut [f64],
) -> Result<(), LapackError> {
    assert_eq!(a.len(), n * n, "dggev: a must be n*n");
    assert_eq!(b.len(), n * n, "dggev: b must be n*n");
    assert_eq!(alphar.len(), n, "dggev: alphar must have n entries");
    assert_eq!(alphai.len(), n, "dggev: alphai must have n entries");
    assert_eq!(beta.len(), n, "dggev: beta must have n entries");
    // The eigenvector matrices are only touched when requested, so an empty
    // slice is accepted when the corresponding JOB is 'N'.
    if calc_vl {
        assert_eq!(vl.len(), n * n, "dggev: vl must be n*n");
    }
    if calc_vr {
        assert_eq!(vr.len(), n * n, "dggev: vr must be n*n");
    }

    if n == 0 {
        return Ok(());
    }
    let ilv = calc_vl || calc_vr;
    let lda = n;
    let ldb = n;

    // Machine constants.
    let eps = ULP; // DLAMCH('P')
    let smlnum0 = SAFMIN;
    let bignum0 = 1.0 / smlnum0;
    let smlnum = smlnum0.sqrt() / eps;
    let bignum = 1.0 / smlnum;

    // Scale A if its norm is outside [smlnum, bignum].
    let anrm = blas::dlange('M', a, n, n);
    let mut ilascl = false;
    let mut anrmto = 0.0;
    if anrm > 0.0 && anrm < smlnum {
        anrmto = smlnum;
        ilascl = true;
    } else if anrm > bignum {
        anrmto = bignum;
        ilascl = true;
    }
    if ilascl {
        dlascl_g(anrm, anrmto, a, lda, n, n);
    }

    // Scale B likewise.
    let bnrm = blas::dlange('M', b, n, n);
    let mut ilbscl = false;
    let mut bnrmto = 0.0;
    if bnrm > 0.0 && bnrm < smlnum {
        bnrmto = smlnum;
        ilbscl = true;
    } else if bnrm > bignum {
        bnrmto = bignum;
        ilbscl = true;
    }
    if ilbscl {
        dlascl_g(bnrm, bnrmto, b, ldb, n, n);
    }
    let _ = bignum0;

    // Permute the matrices to isolate eigenvalues.
    let (ilo, ihi, lscale, rscale) = dggbal_p(n, a, lda, b, ldb);
    let ihi0 = ihi - 1; // 0-based inclusive

    // Reduce B to triangular form (QR of B).
    let irows = ihi - ilo;
    // `n + 1 - ilo` in the reference counts from 1; with 0-based `ilo` it is `n - ilo`.
    let icols = if ilv { n - ilo } else { irows };

    let mut tau = vec![0.0; irows.max(1)];
    dgeqr2(b, ldb, ilo, ilo, irows, icols, &mut tau);

    // Apply the orthogonal transformation to A.
    dorm2r_left_trans(a, lda, ilo, ilo, irows, icols, b, ldb, ilo, ilo, irows, &tau);

    // Initialize VL then generate Q in it.
    if calc_vl {
        dlaset_identity(vl, n, n);
        if irows > 1 {
            // DLACPY('L', IROWS-1, IROWS-1, B(ILO+1,ILO), LDB, VL(ILO+1,ILO), LDVL)
            for r in 1..irows {
                for c in 1..=r {
                    vl[(ilo + r) + (ilo + c - 1) * n] = b[(ilo + r) + (ilo + c - 1) * n];
                }
            }
        }
        dorg2r(vl, n, ilo, ilo, irows, irows, irows, &tau);
    }

    // Initialize VR.
    if calc_vr {
        dlaset_identity(vr, n, n);
    }

    // Reduce to generalized Hessenberg form.
    dgghrd(calc_vl, calc_vr, n, ilo, ihi0, a, lda, b, ldb, vl, n, vr, n);

    // Perform the QZ algorithm.
    let ierr = dhgeqz(
        ilv, calc_vl, calc_vr, n, ilo, ihi0, a, lda, b, ldb, alphar, alphai, beta, vl, n, vr, n,
    );
    // The QZ iteration's INFO is 1..=n for non-convergence and n+1..=2n for an
    // argument it rejected; both are errors here.
    if ierr != 0 {
        return Err(LapackError::Convergence);
    }

    // Compute eigenvectors.
    if ilv {
        let ierr = dtgevc(calc_vl, calc_vr, n, a, lda, b, ldb, vl, n, vr, n);
        // The eigenvector back-substitution failed (LAPACK's INFO = N+2).
        if ierr != 0 {
            return Err(LapackError::Convergence);
        }
        // Undo balancing on VL and VR, and normalize.
        if calc_vl {
            // DGGBAK('P', 'L', ...) permutes with LSCALE.
            dggbak_p(false, n, ilo, ihi0, &lscale, &rscale, n, vl, n);
            dggev_normalize(n, smlnum, alphai, vl, n);
        }
        if calc_vr {
            // DGGBAK('P', 'R', ...) permutes with RSCALE.
            dggbak_p(true, n, ilo, ihi0, &lscale, &rscale, n, vr, n);
            dggev_normalize(n, smlnum, alphai, vr, n);
        }
    }

    // Undo scaling if necessary.
    if ilascl {
        for i in 0..n {
            alphar[i] *= anrm / anrmto;
            alphai[i] *= anrm / anrmto;
        }
    }
    if ilbscl {
        for i in 0..n {
            beta[i] *= bnrm / bnrmto;
        }
    }

    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// The complex generalized eigenproblem (ZGGEV)
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// A literal transcription of LAPACK 3.12.1's `zggev` driver and the complex
// routines it calls: `zggbal`, `zgghrd`, `zhgeqz`, `ztgevc`, `zggbak`, `zgeqr2`
// (standing in for the unblocked `zgeqrf`), `zunm2r` (for `zunmqr`), `zung2r`
// (for `zungqr`), `zlartg`, `zlascl`, `zlange` and `zlanhs`. The layout mirrors
// the real [`dggev`] above; what differs is that every plane rotation carries a
// conjugate (`zrot`), as does every reflector application (`zlarf`), and that
// `zhgeqz` needs no double-shift path because the complex Schur form is already
// triangular. `zggbal`'s complex balancing is not transcribed: `zggev` asks only
// for `JOB = 'P'`, the permutation.
//
// The unblocked `zgeqr2`/`zunm2r`/`zung2r` are what `zgeqrf`/`zunmqr`/`zungqr`
// fall through to for these block sizes, exactly as in the real port.
//
// `zlarfg`, the complex `zlarf` and `zladiv` are duplicated from
// [`super::zsvd`] (which took `zlarfg` from [`super::zeig`]) rather than made
// public there, following the precedent that those modules are not to be
// disturbed.

/// The complex zero, named so the arithmetic reads like the Fortran `CZERO`.
#[inline]
fn zzero() -> Complex64 {
    Complex64::new(0.0, 0.0)
}

/// The complex one, named so the arithmetic reads like the Fortran `CONE`.
#[inline]
fn zone() -> Complex64 {
    Complex64::new(1.0, 0.0)
}

/// The reference's `ABS1(z)`: `|Re(z)| + |Im(z)|`.
#[inline]
fn zabs1(z: Complex64) -> f64 {
    z.re.abs() + z.im.abs()
}

/// Fortran's `SIGN(a, b)`: `|a|` with `b`'s sign, where `b == 0` counts as positive.
#[inline]
fn zsign_f(a: f64, b: f64) -> f64 {
    if b < 0.0 { -a.abs() } else { a.abs() }
}

/// `zlarfg`: an elementary reflector `H = I - tau * v * v^H` with `H x = beta e1`.
///
/// Copied from [`super::zsvd`]'s kernel of the same name (itself copied from
/// [`super::zeig`]'s); they are private there and those modules are not to be
/// disturbed, so the text is duplicated rather than shared.
fn zlarfg(alpha: &mut Complex64, x: &mut [Complex64]) -> Complex64 {
    let mut xnrm = dznrm2(x);
    let mut alphr = alpha.re;
    let mut alphi = alpha.im;

    if xnrm == 0.0 && alphi == 0.0 {
        return zzero();
    }

    let safmin = f64::MIN_POSITIVE / (f64::EPSILON / 2.0);
    let rsafmn = 1.0 / safmin;

    let mut beta = -zsign_f(dlapy3(alphr, alphi, xnrm), alphr);
    let mut knt = 0usize;
    if beta.abs() < safmin {
        loop {
            knt += 1;
            for xi in x.iter_mut() {
                *xi *= rsafmn;
            }
            beta *= rsafmn;
            alphi *= rsafmn;
            alphr *= rsafmn;
            if !(beta.abs() < safmin && knt < 20) {
                break;
            }
        }
        xnrm = dznrm2(x);
        beta = -zsign_f(dlapy3(alphr, alphi, xnrm), alphr);
    }

    let tau = Complex64::new((beta - alphr) / beta, -alphi / beta);
    let al = zladiv(zone(), Complex64::new(alphr - beta, alphi));
    for xi in x.iter_mut() {
        *xi *= al;
    }
    for _ in 0..knt {
        beta *= safmin;
    }
    *alpha = Complex64::new(beta, 0.0);
    tau
}

/// `zlartg`: the complex Givens rotation. Returns `(c, s, r)` with `c` real and
/// `s`/`r` complex such that `[c, s; -conj(s), c] [f; g] = [r; 0]`, scaled so
/// that no intermediate overflows or underflows.
fn zlartg(f: Complex64, g: Complex64) -> (f64, Complex64, Complex64) {
    fn abssq(t: Complex64) -> f64 {
        t.re * t.re + t.im * t.im
    }
    let safmin = SAFMIN;
    let safmax = 1.0 / SAFMIN;
    let rtmin = safmin.sqrt();

    let mut c;
    let s;
    let mut r;
    if g == zzero() {
        c = 1.0;
        s = zzero();
        r = f;
    } else if f == zzero() {
        c = 0.0;
        if g.re == 0.0 {
            r = Complex64::new(g.im.abs(), 0.0);
            s = g.conj() / r;
        } else if g.im == 0.0 {
            r = Complex64::new(g.re.abs(), 0.0);
            s = g.conj() / r;
        } else {
            let g1 = g.re.abs().max(g.im.abs());
            let rtmax = (safmax / 2.0).sqrt();
            if g1 > rtmin && g1 < rtmax {
                // Unscaled.
                let d = abssq(g).sqrt();
                s = g.conj() / d;
                r = Complex64::new(d, 0.0);
            } else {
                // Scaled.
                let u = safmax.min(safmin.max(g1));
                let gs = g / u;
                let d = abssq(gs).sqrt();
                s = gs.conj() / d;
                r = Complex64::new(d * u, 0.0);
            }
        }
    } else {
        let f1 = f.re.abs().max(f.im.abs());
        let g1 = g.re.abs().max(g.im.abs());
        let mut rtmax = (safmax / 4.0).sqrt();
        if f1 > rtmin && f1 < rtmax && g1 > rtmin && g1 < rtmax {
            // Unscaled.
            let f2 = abssq(f);
            let g2 = abssq(g);
            let h2 = f2 + g2;
            if f2 >= h2 * safmin {
                c = (f2 / h2).sqrt();
                r = f / c;
                rtmax *= 2.0;
                if f2 > rtmin && h2 < rtmax {
                    s = g.conj() * (f / (f2 * h2).sqrt());
                } else {
                    s = g.conj() * (r / h2);
                }
            } else {
                let d = (f2 * h2).sqrt();
                c = f2 / d;
                if c >= safmin {
                    r = f / c;
                } else {
                    r = f * (h2 / d);
                }
                s = g.conj() * (f / d);
            }
        } else {
            // Scaled.
            let u = safmax.min(safmin.max(f1).max(g1));
            let gs = g / u;
            let g2 = abssq(gs);
            let (fs, w, f2, h2);
            if f1 / u < rtmin {
                let v = safmax.min(safmin.max(f1));
                w = v / u;
                fs = f / v;
                f2 = abssq(fs);
                h2 = f2 * w * w + g2;
            } else {
                w = 1.0;
                fs = f / u;
                f2 = abssq(fs);
                h2 = f2 + g2;
            }
            if f2 >= h2 * safmin {
                c = (f2 / h2).sqrt();
                r = fs / c;
                rtmax *= 2.0;
                if f2 > rtmin && h2 < rtmax {
                    s = gs.conj() * (fs / (f2 * h2).sqrt());
                } else {
                    s = gs.conj() * (r / h2);
                }
            } else {
                let d = (f2 * h2).sqrt();
                c = f2 / d;
                if c >= safmin {
                    r = fs / c;
                } else {
                    r = fs * (h2 / d);
                }
                s = gs.conj() * (fs / d);
            }
            c *= w;
            r *= u;
        }
    }
    (c, s, r)
}

/// `zlange('M')`: the largest complex modulus in the `m x n` matrix.
fn zlange_m(a: &[Complex64], ld: usize, m: usize, n: usize) -> f64 {
    let mut value = 0.0f64;
    for j in 0..n {
        for i in 0..m {
            let temp = a[i + j * ld].norm();
            if value < temp {
                value = temp;
            }
        }
    }
    value
}

/// `zlanhs('F')`: the Frobenius norm of the upper Hessenberg part of an `n x n`
/// matrix (entries `(i, j)` with `i <= j + 1`), via `zlassq`'s scaled sum of
/// squares over the real and imaginary parts.
fn zlanhs_f(a: &[Complex64], ld: usize, n: usize) -> f64 {
    let mut scale = 0.0f64;
    let mut sumsq = 1.0f64;
    for j in 0..n {
        let count = n.min(j + 2);
        for i in 0..count {
            let z = a[i + j * ld];
            for absxi in [z.re.abs(), z.im.abs()] {
                if absxi != 0.0 {
                    if scale < absxi {
                        let t = scale / absxi;
                        sumsq = 1.0 + sumsq * t * t;
                        scale = absxi;
                    } else {
                        let t = absxi / scale;
                        sumsq += t * t;
                    }
                }
            }
        }
    }
    scale * sumsq.sqrt()
}

/// `zrot` on the rows `i1` and `i2` over `count` columns from `j0`:
/// `x' = c x + s y`, `y' = c y - conj(s) x`.
fn zrot_rows(a: &mut [Complex64], ld: usize, i1: usize, i2: usize, j0: usize, count: usize, c: f64, s: Complex64) {
    for j in j0..(j0 + count) {
        let x = a[i1 + j * ld];
        let y = a[i2 + j * ld];
        a[i1 + j * ld] = c * x + s * y;
        a[i2 + j * ld] = c * y - s.conj() * x;
    }
}

/// `zrot` on the columns `j1` and `j2` over `count` rows from `i0`, with the
/// same `x' = c x + s y`, `y' = c y - conj(s) x`.
fn zrot_cols(a: &mut [Complex64], ld: usize, j1: usize, j2: usize, i0: usize, count: usize, c: f64, s: Complex64) {
    for i in i0..(i0 + count) {
        let x = a[i + j1 * ld];
        let y = a[i + j2 * ld];
        a[i + j1 * ld] = c * x + s * y;
        a[i + j2 * ld] = c * y - s.conj() * x;
    }
}

/// `zgemv('N', m, ncols, 1, a, lda, x, 1, 0, y, 1)`: `y := A x`, where `x`
/// multiplies the `ncols` columns of `a` starting at column `col0`.
fn zgemv_n(a: &[Complex64], lda: usize, m: usize, col0: usize, ncols: usize, x: &[Complex64], y: &mut [Complex64]) {
    for i in 0..m {
        let mut acc = zzero();
        for j in 0..ncols {
            acc += a[i + (col0 + j) * lda] * x[j];
        }
        y[i] = acc;
    }
}

/// `zgeqr2`: QR of the `m x ncol` block of `a` at `(r0, c0)`, with the
/// reflectors left in place and `tau` of length `min(m, ncol)`.
fn zgeqr2(a: &mut [Complex64], lda: usize, r0: usize, c0: usize, m: usize, ncol: usize, tau: &mut [Complex64]) {
    let k = m.min(ncol);
    for i in 0..k {
        let mut alpha = a[(r0 + i) + (c0 + i) * lda];
        let mut x: Vec<Complex64> = (1..(m - i)).map(|t| a[(r0 + i + t) + (c0 + i) * lda]).collect();
        let taui = zlarfg(&mut alpha, &mut x);
        a[(r0 + i) + (c0 + i) * lda] = alpha;
        for (t, v) in x.iter().enumerate() {
            a[(r0 + i + 1 + t) + (c0 + i) * lda] = *v;
        }
        tau[i] = taui;
        if i + 1 < ncol {
            // `zlarf1f('Left', m-i, ncol-i-1, A(i,i), 1, conj(tau[i]), A(i,i+1), lda, work)`:
            // the reflector's implicit `1` sits at `(i, i)`.
            let mrows = m - i;
            let ncols = ncol - i - 1;
            let mut v = vec![zzero(); mrows];
            v[0] = zone();
            for t in 1..mrows {
                v[t] = a[(r0 + i + t) + (c0 + i) * lda];
            }
            zlarf_left(a, lda, r0 + i, mrows, c0 + i + 1, ncols, &v, taui.conj());
        }
    }
}

/// `zunm2r` with `SIDE = 'L'`, `TRANS = 'C'`: apply `Q^H` from the left, where
/// `Q` is built from the reflectors stored in the block of `bref` at
/// `(rb, cb)`, to the block of `a` at `(r0, c0)`.
#[allow(clippy::too_many_arguments)]
fn zunm2r_left_conj(
    a: &mut [Complex64],
    lda: usize,
    r0: usize,
    c0: usize,
    m: usize,
    ncol: usize,
    bref: &[Complex64],
    ldb: usize,
    rb: usize,
    cb: usize,
    k: usize,
    tau: &[Complex64],
) {
    for i in 0..k {
        let mi = m - i;
        let mut v = vec![zzero(); mi];
        v[0] = zone();
        for t in 1..mi {
            v[t] = bref[(rb + i + t) + (cb + i) * ldb];
        }
        zlarf_left(a, lda, r0 + i, mi, c0, ncol, &v, tau[i].conj());
    }
}

/// `zung2r`: generate the first `n` columns of the unitary `Q = H(1)...H(k)`
/// into the `m x n` block of `a` at `(r0, c0)`, whose reflectors are stored
/// there in the first `k` columns.
///
/// The text follows [`super::zsvd`]'s `zung2r`, with the `(r0, c0)` offset of
/// the block added.
#[allow(clippy::too_many_arguments)]
fn zung2r_sub(a: &mut [Complex64], ld: usize, r0: usize, c0: usize, m: usize, n: usize, k: usize, tau: &[Complex64]) {
    if n == 0 {
        return;
    }
    // Set the columns after the first k to the identity.
    for j in k..n {
        for l in 0..m {
            a[(r0 + l) + (c0 + j) * ld] = zzero();
        }
        a[(r0 + j) + (c0 + j) * ld] = zone();
    }
    // Form the first k columns in reverse order.
    for i in (0..k).rev() {
        if i < n - 1 {
            let mrows = m - i;
            let ncols = n - i - 1;
            let mut v = vec![zzero(); mrows];
            v[0] = zone();
            for t in 1..mrows {
                v[t] = a[(r0 + i + t) + (c0 + i) * ld];
            }
            zlarf_left(a, ld, r0 + i, mrows, c0 + i + 1, ncols, &v, tau[i]);
        }
        if i < m - 1 {
            for t in (i + 1)..m {
                a[(r0 + t) + (c0 + i) * ld] *= -tau[i];
            }
        }
        a[(r0 + i) + (c0 + i) * ld] = zone() - tau[i];
        for l in 0..i {
            a[(r0 + l) + (c0 + i) * ld] = zzero();
        }
    }
}

/// `zgghrd`: reduce the pair `(a, b)` to generalized Hessenberg form (`a` upper
/// Hessenberg, `b` upper triangular) using unitary transformations, accumulating
/// them into `q` and `z` when asked. `ilo` and `ihi` (0-based, inclusive) bound
/// the active block.
#[allow(clippy::too_many_arguments)]
fn zgghrd(
    ilq: bool,
    ilz: bool,
    n: usize,
    ilo: usize,
    ihi: usize,
    a: &mut [Complex64],
    lda: usize,
    b: &mut [Complex64],
    ldb: usize,
    q: &mut [Complex64],
    ldq: usize,
    z: &mut [Complex64],
    ldz: usize,
) {
    if n <= 1 || ihi < ilo + 2 {
        // The reference zeroes the strict lower triangle of B before its quick
        // return on the block size.
        if n > 1 {
            for jcol in 0..(n - 1) {
                for jrow in (jcol + 1)..n {
                    b[jrow + jcol * ldb] = zzero();
                }
            }
        }
        return;
    }

    // Zero the lower triangle of B.
    for jcol in 0..(n - 1) {
        for jrow in (jcol + 1)..n {
            b[jrow + jcol * ldb] = zzero();
        }
    }

    for jcol in ilo..=(ihi - 2) {
        let mut jrow = ihi;
        loop {
            // Step 1: rotate rows jrow-1, jrow to kill a(jrow, jcol).
            let ctemp = a[(jrow - 1) + jcol * lda];
            let (c, s, r) = zlartg(ctemp, a[jrow + jcol * lda]);
            a[(jrow - 1) + jcol * lda] = r;
            a[jrow + jcol * lda] = zzero();
            // zrot(N-JCOL, A(JROW-1,JCOL+1), LDA, A(JROW,JCOL+1), LDA, ...)
            zrot_rows(a, lda, jrow - 1, jrow, jcol + 1, n - jcol - 1, c, s);
            // zrot(N+2-JROW, B(JROW-1,JROW-1), LDB, B(JROW,JROW-1), LDB, ...)
            zrot_rows(b, ldb, jrow - 1, jrow, jrow - 1, n + 1 - jrow, c, s);
            if ilq {
                zrot_cols(q, ldq, jrow - 1, jrow, 0, n, c, s.conj());
            }

            // Step 2: rotate columns jrow, jrow-1 to kill b(jrow, jrow-1).
            let ctemp = b[jrow + jrow * ldb];
            let (c, s, r) = zlartg(ctemp, b[jrow + (jrow - 1) * ldb]);
            b[jrow + jrow * ldb] = r;
            b[jrow + (jrow - 1) * ldb] = zzero();
            // zrot(IHI, A(1,JROW), 1, A(1,JROW-1), 1, ...): IHI is 1-based, so
            // the row count is `ihi + 1` here.
            zrot_cols(a, lda, jrow, jrow - 1, 0, ihi + 1, c, s);
            // zrot(JROW-1, B(1,JROW), 1, B(1,JROW-1), 1, ...).
            zrot_cols(b, ldb, jrow, jrow - 1, 0, jrow, c, s);
            if ilz {
                zrot_cols(z, ldz, jrow, jrow - 1, 0, n, c, s);
            }

            if jrow <= jcol + 2 {
                break;
            }
            jrow -= 1;
        }
    }
}

/// `zggbal` with `JOB = 'P'`: permute the pair `(a, b)` to isolate eigenvalues
/// and return `(ilo, ihi, lscale, rscale)` (0-based `ilo`, exclusive `ihi`).
///
/// The permutation section of `zggbal` is character for character the same as
/// `dggbal`'s (only the element type differs), so this is the same port as
/// [`dggbal_p`] with complex elements. The conjugate-gradient balancing that
/// `JOB = 'B'` adds is not transcribed, as `zggev` does not ask for it.
fn zggbal_p(
    n: usize,
    a: &mut [Complex64],
    lda: usize,
    b: &mut [Complex64],
    ldb: usize,
) -> (usize, usize, Vec<f64>, Vec<f64>) {
    let mut lscale = vec![0.0; n];
    let mut rscale = vec![0.0; n];
    if n == 0 {
        return (0, 0, lscale, rscale);
    }
    if n == 1 {
        lscale[0] = 1.0;
        rscale[0] = 1.0;
        return (0, 1, lscale, rscale);
    }
    // 1-based working values, as in the reference.
    let mut k = 1usize;
    let mut l = n;
    let mut lm1 = 0usize;
    let mut flow = 0u8; // 0: first entry, 1: from label 20, 2: from label 90
    'outer: loop {
        if flow == 1 {
            l = lm1;
            if l == 1 {
                rscale[0] = 1.0;
                lscale[0] = 1.0;
                break 'outer;
            }
        } else if flow == 2 {
            k += 1;
        }

        // label 30: find a row with one nonzero in columns 1..L.
        lm1 = l - 1;
        let mut found_flow = 0u8;
        let mut m = 0usize;
        let mut ifound = 0usize;
        let mut jfound = 0usize;
        let mut row_permuted = false;
        'row: for i in (1..=l).rev() {
            let mut j = 0usize;
            let mut nonz = false;
            for jj in 1..=lm1 {
                if a[(i - 1) + (jj - 1) * lda] != zzero() || b[(i - 1) + (jj - 1) * ldb] != zzero() {
                    j = jj;
                    nonz = true;
                    break;
                }
            }
            if !nonz {
                j = l;
            } else {
                // label 50
                let jp1 = j + 1;
                let mut found = false;
                for jj in jp1..=l {
                    if a[(i - 1) + (jj - 1) * lda] != zzero() || b[(i - 1) + (jj - 1) * ldb] != zzero() {
                        found = true;
                        break;
                    }
                }
                if found {
                    continue 'row;
                }
                j = jp1 - 1;
            }
            // label 70
            m = l;
            ifound = i;
            jfound = j;
            found_flow = 1;
            row_permuted = true;
            break 'row;
        }
        if row_permuted {
            // label 160
            lscale[m - 1] = ifound as f64;
            if ifound != m {
                // ZSWAP(N-K+1, A(I,K), LDA, A(M,K), LDA)
                swap_rows(a, lda, ifound - 1, m - 1, k - 1, n - k + 1);
                swap_rows(b, ldb, ifound - 1, m - 1, k - 1, n - k + 1);
            }
            // label 170
            rscale[m - 1] = jfound as f64;
            if jfound != m {
                // ZSWAP(L, A(1,J), 1, A(1,M), 1)
                swap_cols(a, lda, jfound - 1, m - 1, 0, l);
                swap_cols(b, ldb, jfound - 1, m - 1, 0, l);
            }
            let _ = found_flow;
            flow = 1;
            continue 'outer;
        }

        // label 100: find a column with one nonzero in rows K..N.
        let mut col_permuted = false;
        loop {
            let mut m2 = 0usize;
            let mut ifound2 = 0usize;
            let mut jfound2 = 0usize;
            'col: for j in k..=l {
                let mut i = 0usize;
                let mut nonz = false;
                for ii in k..=lm1 {
                    if a[(ii - 1) + (j - 1) * lda] != zzero() || b[(ii - 1) + (j - 1) * ldb] != zzero() {
                        i = ii;
                        nonz = true;
                        break;
                    }
                }
                if !nonz {
                    i = l;
                } else {
                    let ip1 = i + 1;
                    let mut found = false;
                    for ii in ip1..=l {
                        if a[(ii - 1) + (j - 1) * lda] != zzero() || b[(ii - 1) + (j - 1) * ldb] != zzero() {
                            found = true;
                            break;
                        }
                    }
                    if found {
                        continue 'col;
                    }
                    i = ip1 - 1;
                }
                // label 140
                m2 = k;
                ifound2 = i;
                jfound2 = j;
                col_permuted = true;
                break 'col;
            }
            if !col_permuted {
                break 'outer;
            }
            // label 160 with IFLOW = 2
            lscale[m2 - 1] = ifound2 as f64;
            if ifound2 != m2 {
                swap_rows(a, lda, ifound2 - 1, m2 - 1, k - 1, n - k + 1);
                swap_rows(b, ldb, ifound2 - 1, m2 - 1, k - 1, n - k + 1);
            }
            rscale[m2 - 1] = jfound2 as f64;
            if jfound2 != m2 {
                swap_cols(a, lda, jfound2 - 1, m2 - 1, 0, l);
                swap_cols(b, ldb, jfound2 - 1, m2 - 1, 0, l);
            }
            // label 180 -> 90 -> 100
            k += 1;
        }
    }
    // label 190 with JOB = 'P'
    for i in (k - 1)..l {
        lscale[i] = 1.0;
        rscale[i] = 1.0;
    }
    (k - 1, l, lscale, rscale)
}

/// The standardization of one diagonal element that ends a deflation in
/// `zhgeqz` (labels 10, 60 and 200 of the reference): make `t(j,j)` real and
/// non-negative, scale the parts of the columns above it by the conjugate, and
/// record `alpha`/`beta`.
#[allow(clippy::too_many_arguments)]
fn zhgeqz_set_eigenvalue(
    j: usize,
    ilschr: bool,
    ilz: bool,
    ifrstm: usize,
    h: &mut [Complex64],
    ldh: usize,
    t: &mut [Complex64],
    ldt: usize,
    z: &mut [Complex64],
    ldz: usize,
    n: usize,
    alpha: &mut [Complex64],
    beta: &mut [Complex64],
) {
    let absb = t[j + j * ldt].norm();
    if absb > SAFMIN {
        let signbc = (t[j + j * ldt] / absb).conj();
        t[j + j * ldt] = Complex64::new(absb, 0.0);
        if ilschr {
            // zscal(j-1, signbc, t(1,j), 1) and zscal(j, signbc, h(1,j), 1).
            for jr in ifrstm..j {
                t[jr + j * ldt] *= signbc;
            }
            for jr in ifrstm..=j {
                h[jr + j * ldh] *= signbc;
            }
        } else {
            h[j + j * ldh] *= signbc;
        }
        if ilz {
            for jr in 0..n {
                z[jr + j * ldz] *= signbc;
            }
        }
    } else {
        t[j + j * ldt] = zzero();
    }
    alpha[j] = h[j + j * ldh];
    beta[j] = t[j + j * ldt];
}

/// `zhgeqz` with `JOB = 'S'` or `'E'`: the complex QZ iteration.
///
/// On entry `h` and `t` are in generalized Hessenberg form (`h` upper Hessenberg,
/// `t` upper triangular); on exit they hold the generalized Schur form and the
/// generalized eigenvalues are in `alpha`/`beta` as `alpha / beta`. The unitary
/// factors are accumulated into `q` and `z` when `ilq`/`ilz`. Returns the
/// reference's `INFO`: 0 on success, `1..=n` for non-convergence and `2*n + 1`
/// for the "impossible" drop-through.
#[allow(clippy::too_many_arguments)]
fn zhgeqz(
    ilschr: bool,
    ilq: bool,
    ilz: bool,
    n: usize,
    ilo: usize,
    ihi: usize,
    h: &mut [Complex64],
    ldh: usize,
    t: &mut [Complex64],
    ldt: usize,
    alpha: &mut [Complex64],
    beta: &mut [Complex64],
    q: &mut [Complex64],
    ldq: usize,
    z: &mut [Complex64],
    ldz: usize,
) -> usize {
    // Fortran's IF( N.LE.0 ); for a `usize` that is exactly `n == 0`.
    if n == 0 {
        return 0;
    }

    let in_ = (ihi + 1).saturating_sub(ilo);
    let safmin = SAFMIN;
    let ulp = ULP;
    let anorm = zlanhs_f(&h[(ilo + ilo * ldh)..], ldh, in_);
    let bnorm = zlanhs_f(&t[(ilo + ilo * ldt)..], ldt, in_);
    let atol = safmin.max(ulp * anorm);
    let btol = safmin.max(ulp * bnorm);
    let ascale = 1.0 / safmin.max(anorm);
    let bscale = 1.0 / safmin.max(bnorm);

    // Set eigenvalues ihi+1..n.
    for j in (ihi + 1)..n {
        zhgeqz_set_eigenvalue(j, ilschr, ilz, 0, h, ldh, t, ldt, z, ldz, n, alpha, beta);
    }

    // If ihi < ilo, skip the QZ steps.
    if ihi < ilo {
        for j in 0..ilo {
            zhgeqz_set_eigenvalue(j, ilschr, ilz, 0, h, ldh, t, ldt, z, ldz, n, alpha, beta);
        }
        return 0;
    }

    let mut ilast = ihi;
    let (mut ifrstm, mut ilastm) = if ilschr { (0usize, n - 1) } else { (ilo, ihi) };
    let mut iiter = 0usize;
    let mut eshift = zzero();
    let maxit = 30 * (ihi - ilo + 1);

    /// The destination of one pass of the reference's loop, mirroring its labels.
    enum Next {
        /// Label 60 (through label 50 when `via50`).
        Deflate { via50: bool },
        /// Label 70, the QZ step, with `ifirst`.
        QzStep(usize),
    }

    for _jiter in 0..maxit {
        let next: Next = if ilast == ilo {
            Next::Deflate { via50: false }
        } else if zabs1(h[ilast + (ilast - 1) * ldh])
            <= safmin.max(ulp * (zabs1(h[ilast + ilast * ldh]) + zabs1(h[(ilast - 1) + (ilast - 1) * ldh])))
        {
            h[ilast + (ilast - 1) * ldh] = zzero();
            Next::Deflate { via50: false }
        } else if t[ilast + ilast * ldt].norm() <= btol {
            t[ilast + ilast * ldt] = zzero();
            Next::Deflate { via50: true }
        } else {
            // General case: j < ilast.
            let mut chosen: Option<Next> = None;
            let mut j = ilast - 1;
            while j >= ilo {
                // Test 1: h(j,j-1) = 0 or j = ilo.
                let ilazro = if j == ilo {
                    true
                } else if zabs1(h[j + (j - 1) * ldh])
                    <= safmin.max(ulp * (zabs1(h[j + j * ldh]) + zabs1(h[(j - 1) + (j - 1) * ldh])))
                {
                    h[j + (j - 1) * ldh] = zzero();
                    true
                } else {
                    false
                };

                // Test 2: t(j,j) = 0.
                if t[j + j * ldt].norm() < btol {
                    t[j + j * ldt] = zzero();

                    // Test 1a: two consecutive small subdiagonals in A.
                    let mut ilazr2 = false;
                    if !ilazro
                        && zabs1(h[j + (j - 1) * ldh]) * (ascale * zabs1(h[(j + 1) + j * ldh]))
                            <= zabs1(h[j + j * ldh]) * (ascale * atol)
                    {
                        ilazr2 = true;
                    }

                    if ilazro || ilazr2 {
                        // Split a 1x1 block off at the top; the reference loop
                        // is JCH = J, ILAST-1.
                        let mut jch = j;
                        loop {
                            let ctemp = h[jch + jch * ldh];
                            let (c, s, r) = zlartg(ctemp, h[(jch + 1) + jch * ldh]);
                            h[jch + jch * ldh] = r;
                            h[(jch + 1) + jch * ldh] = zzero();
                            zrot_rows(h, ldh, jch, jch + 1, jch + 1, ilastm - jch, c, s);
                            zrot_rows(t, ldt, jch, jch + 1, jch + 1, ilastm - jch, c, s);
                            if ilq {
                                zrot_cols(q, ldq, jch, jch + 1, 0, n, c, s.conj());
                            }
                            if ilazr2 {
                                h[jch + (jch - 1) * ldh] *= c;
                            }
                            ilazr2 = false;
                            if zabs1(t[(jch + 1) + (jch + 1) * ldt]) >= btol {
                                if jch + 1 >= ilast {
                                    chosen = Some(Next::Deflate { via50: false });
                                } else {
                                    chosen = Some(Next::QzStep(jch + 1));
                                }
                                break;
                            }
                            t[(jch + 1) + (jch + 1) * ldt] = zzero();
                            if jch >= ilast - 1 {
                                chosen = Some(Next::Deflate { via50: true });
                                break;
                            }
                            jch += 1;
                        }
                    } else {
                        // Only test 2 passed: chase the zero to T(ilast, ilast).
                        // The reference loop is JCH = J, ILAST-1.
                        for jch in j..ilast {
                            let ctemp = t[jch + (jch + 1) * ldt];
                            let (c, s, r) = zlartg(ctemp, t[(jch + 1) + (jch + 1) * ldt]);
                            t[jch + (jch + 1) * ldt] = r;
                            t[(jch + 1) + (jch + 1) * ldt] = zzero();
                            if jch < ilastm - 1 {
                                zrot_rows(t, ldt, jch, jch + 1, jch + 2, ilastm - jch - 1, c, s);
                            }
                            zrot_rows(h, ldh, jch, jch + 1, jch - 1, ilastm - jch + 2, c, s);
                            if ilq {
                                zrot_cols(q, ldq, jch, jch + 1, 0, n, c, s.conj());
                            }
                            let ctemp = h[(jch + 1) + jch * ldh];
                            let (c, s, r) = zlartg(ctemp, h[(jch + 1) + (jch - 1) * ldh]);
                            h[(jch + 1) + jch * ldh] = r;
                            h[(jch + 1) + (jch - 1) * ldh] = zzero();
                            zrot_cols(h, ldh, jch, jch - 1, ifrstm, jch + 1 - ifrstm, c, s);
                            zrot_cols(t, ldt, jch, jch - 1, ifrstm, jch - ifrstm, c, s);
                            if ilz {
                                zrot_cols(z, ldz, jch, jch - 1, 0, n, c, s);
                            }
                        }
                        chosen = Some(Next::Deflate { via50: true });
                    }
                    break;
                } else if ilazro {
                    // Only test 1 passed: work on j:ilast.
                    chosen = Some(Next::QzStep(j));
                    break;
                }

                if j == ilo {
                    break;
                }
                j -= 1;
            }
            match chosen {
                Some(nx) => nx,
                None => return 2 * n + 1,
            }
        };

        match next {
            Next::Deflate { via50 } => {
                if via50 {
                    // Label 50: clear h(ilast, ilast-1) to split off a 1x1 block.
                    let ctemp = h[ilast + ilast * ldh];
                    let (c, s, r) = zlartg(ctemp, h[ilast + (ilast - 1) * ldh]);
                    h[ilast + ilast * ldh] = r;
                    h[ilast + (ilast - 1) * ldh] = zzero();
                    zrot_cols(h, ldh, ilast, ilast - 1, ifrstm, ilast - ifrstm, c, s);
                    zrot_cols(t, ldt, ilast, ilast - 1, ifrstm, ilast - ifrstm, c, s);
                    if ilz {
                        zrot_cols(z, ldz, ilast, ilast - 1, 0, n, c, s);
                    }
                }
                // Label 60: standardize B and record alpha/beta.
                zhgeqz_set_eigenvalue(ilast, ilschr, ilz, ifrstm, h, ldh, t, ldt, z, ldz, n, alpha, beta);
                if ilast == ilo {
                    // Label 190: set eigenvalues 1:ilo-1 and finish.
                    for j in 0..ilo {
                        zhgeqz_set_eigenvalue(j, ilschr, ilz, 0, h, ldh, t, ldt, z, ldz, n, alpha, beta);
                    }
                    return 0;
                }
                ilast -= 1;
                iiter = 0;
                eshift = zzero();
                if !ilschr {
                    ilastm = ilast;
                    if ifrstm > ilast {
                        ifrstm = ilo;
                    }
                }
            }

            Next::QzStep(ifirst) => {
                // Label 70: the QZ step.
                iiter += 1;
                if !ilschr {
                    ifrstm = ifirst;
                }

                // Compute the shift.
                let shift: Complex64;
                if (iiter / 10) * 10 != iiter {
                    // The Wilkinson shift: the eigenvalue of the bottom-right
                    // 2x2 block of A inv(B) nearest the (ilast, ilast) element.
                    let u12 = (bscale * t[(ilast - 1) + ilast * ldt]) / (bscale * t[ilast + ilast * ldt]);
                    let ad11 =
                        (ascale * h[(ilast - 1) + (ilast - 1) * ldh]) / (bscale * t[(ilast - 1) + (ilast - 1) * ldt]);
                    let ad21 = (ascale * h[ilast + (ilast - 1) * ldh]) / (bscale * t[(ilast - 1) + (ilast - 1) * ldt]);
                    let ad12 = (ascale * h[(ilast - 1) + ilast * ldh]) / (bscale * t[ilast + ilast * ldt]);
                    let ad22 = (ascale * h[ilast + ilast * ldh]) / (bscale * t[ilast + ilast * ldt]);
                    let abi22 = ad22 - u12 * ad21;
                    let abi12 = ad12 - u12 * ad11;
                    let mut sh = abi22;
                    let ctemp = abi12.sqrt() * ad21.sqrt();
                    if ctemp != zzero() {
                        let x = 0.5 * (ad11 - sh);
                        let temp2 = zabs1(x);
                        let temp = zabs1(ctemp).max(zabs1(x));
                        let mut y = temp * ((x / temp) * (x / temp) + (ctemp / temp) * (ctemp / temp)).sqrt();
                        if temp2 > 0.0 && (x / temp2).re * y.re + (x / temp2).im * y.im < 0.0 {
                            y = -y;
                        }
                        sh -= ctemp * zladiv(ctemp, x + y);
                    }
                    shift = sh;
                } else {
                    // Exceptional shift.
                    if (iiter / 20) * 20 == iiter && bscale * zabs1(t[ilast + ilast * ldt]) > safmin {
                        eshift += (ascale * h[ilast + ilast * ldh]) / (bscale * t[ilast + ilast * ldt]);
                    } else {
                        eshift +=
                            (ascale * h[ilast + (ilast - 1) * ldh]) / (bscale * t[(ilast - 1) + (ilast - 1) * ldt]);
                    }
                    shift = eshift;
                }

                // Check for two consecutive small subdiagonals.
                let mut istart = ifirst;
                let mut ctemp = ascale * h[ifirst + ifirst * ldh] - shift * (bscale * t[ifirst + ifirst * ldt]);
                let mut found = false;
                let mut j = ilast - 1;
                while j > ifirst {
                    istart = j;
                    ctemp = ascale * h[j + j * ldh] - shift * (bscale * t[j + j * ldt]);
                    let mut temp = zabs1(ctemp);
                    let mut temp2 = ascale * zabs1(h[(j + 1) + j * ldh]);
                    let tempr = temp.max(temp2);
                    if tempr < 1.0 && tempr != 0.0 {
                        temp /= tempr;
                        temp2 /= tempr;
                    }
                    if zabs1(h[j + (j - 1) * ldh]) * temp2 <= temp * atol {
                        found = true;
                        break;
                    }
                    if j == ifirst + 1 {
                        break;
                    }
                    j -= 1;
                }
                if !found {
                    istart = ifirst;
                    ctemp = ascale * h[ifirst + ifirst * ldh] - shift * (bscale * t[ifirst + ifirst * ldt]);
                }

                // Implicit-shift QZ sweep.
                let ctemp2 = ascale * h[(istart + 1) + istart * ldh];
                let (mut c, mut s, _r) = zlartg(ctemp, ctemp2);
                for j in istart..ilast {
                    if j > istart {
                        let ct = h[j + (j - 1) * ldh];
                        let (c2, s2, r) = zlartg(ct, h[(j + 1) + (j - 1) * ldh]);
                        h[j + (j - 1) * ldh] = r;
                        h[(j + 1) + (j - 1) * ldh] = zzero();
                        c = c2;
                        s = s2;
                    }
                    for jc in j..=ilastm {
                        let ct = c * h[j + jc * ldh] + s * h[(j + 1) + jc * ldh];
                        h[(j + 1) + jc * ldh] = -s.conj() * h[j + jc * ldh] + c * h[(j + 1) + jc * ldh];
                        h[j + jc * ldh] = ct;
                        let ct2 = c * t[j + jc * ldt] + s * t[(j + 1) + jc * ldt];
                        t[(j + 1) + jc * ldt] = -s.conj() * t[j + jc * ldt] + c * t[(j + 1) + jc * ldt];
                        t[j + jc * ldt] = ct2;
                    }
                    if ilq {
                        for jr in 0..n {
                            let ct = c * q[jr + j * ldq] + s.conj() * q[jr + (j + 1) * ldq];
                            q[jr + (j + 1) * ldq] = -s * q[jr + j * ldq] + c * q[jr + (j + 1) * ldq];
                            q[jr + j * ldq] = ct;
                        }
                    }
                    let ct = t[(j + 1) + (j + 1) * ldt];
                    let (c2, s2, r) = zlartg(ct, t[(j + 1) + j * ldt]);
                    t[(j + 1) + (j + 1) * ldt] = r;
                    t[(j + 1) + j * ldt] = zzero();
                    c = c2;
                    s = s2;
                    for jr in ifrstm..=(j + 2).min(ilast) {
                        let ct = c * h[jr + (j + 1) * ldh] + s * h[jr + j * ldh];
                        h[jr + j * ldh] = -s.conj() * h[jr + (j + 1) * ldh] + c * h[jr + j * ldh];
                        h[jr + (j + 1) * ldh] = ct;
                    }
                    for jr in ifrstm..=j {
                        let ct = c * t[jr + (j + 1) * ldt] + s * t[jr + j * ldt];
                        t[jr + j * ldt] = -s.conj() * t[jr + (j + 1) * ldt] + c * t[jr + j * ldt];
                        t[jr + (j + 1) * ldt] = ct;
                    }
                    if ilz {
                        for jr in 0..n {
                            let ct = c * z[jr + (j + 1) * ldz] + s * z[jr + j * ldz];
                            z[jr + j * ldz] = -s.conj() * z[jr + (j + 1) * ldz] + c * z[jr + j * ldz];
                            z[jr + (j + 1) * ldz] = ct;
                        }
                    }
                }
            }
        }
    }

    // Drop-through: non-convergence.
    ilast + 1
}

/// The coefficients `a`, `b` of `(a A - b B) x = 0` and the denominator floor
/// `dmin` that `ztgevc` computes identically for the left and the right
/// eigenvectors.
struct ZtgevcCoeffs {
    acoeff: f64,
    bcoeff: Complex64,
    acoefa: f64,
    bcoefa: f64,
    dmin: f64,
}

/// The shared coefficient block of `ztgevc`'s two eigenvector loops.
#[allow(clippy::too_many_arguments)]
fn ztgevc_coeffs(
    je: usize,
    s: &[Complex64],
    lds: usize,
    p: &[Complex64],
    ldp: usize,
    ascale: f64,
    bscale: f64,
    anorm: f64,
    bnorm: f64,
    safmin: f64,
    small: f64,
    big: f64,
    ulp: f64,
) -> ZtgevcCoeffs {
    let temp = 1.0
        / (zabs1(s[je + je * lds]) * ascale)
            .max(p[je + je * ldp].re.abs() * bscale)
            .max(safmin);
    let salpha = (temp * s[je + je * lds]) * ascale;
    let sbeta = (temp * p[je + je * ldp].re) * bscale;
    let mut acoeff = sbeta * ascale;
    let mut bcoeff = salpha * bscale;

    // Scale to avoid underflow.
    let lsa = sbeta.abs() >= safmin && acoeff.abs() < small;
    let lsb = zabs1(salpha) >= safmin && zabs1(bcoeff) < small;
    let mut scale = 1.0;
    if lsa {
        scale = (small / sbeta.abs()) * anorm.min(big);
    }
    if lsb {
        scale = scale.max((small / zabs1(salpha)) * bnorm.min(big));
    }
    if lsa || lsb {
        scale = scale.min(1.0 / (safmin * 1.0f64.max(acoeff.abs()).max(zabs1(bcoeff))));
        if lsa {
            acoeff = ascale * (scale * sbeta);
        } else {
            acoeff *= scale;
        }
        if lsb {
            bcoeff = bscale * (scale * salpha);
        } else {
            bcoeff = scale * bcoeff;
        }
    }
    let acoefa = acoeff.abs();
    let bcoefa = zabs1(bcoeff);
    let dmin = (ulp * acoefa * anorm).max(ulp * bcoefa * bnorm).max(safmin);
    ZtgevcCoeffs {
        acoeff,
        bcoeff,
        acoefa,
        bcoefa,
        dmin,
    }
}

/// `ztgevc` with `HOWMNY = 'B'`: all right (and/or left) generalized
/// eigenvectors of the triangular pair `(s, p)` by back-substitution,
/// back-transformed with the Schur vectors left in `vl`/`vr`.
///
/// `compl`/`compr` select the left and right eigenvectors. `zggev` always asks
/// for `HOWMNY = 'B'`, so the `SELECT`/`MM` machinery is not transcribed.
/// Returns `true` when the reference would return a nonzero `INFO`.
#[allow(clippy::too_many_arguments)]
fn ztgevc(
    compl: bool,
    compr: bool,
    n: usize,
    s: &[Complex64],
    lds: usize,
    p: &[Complex64],
    ldp: usize,
    vl: &mut [Complex64],
    ldvl: usize,
    vr: &mut [Complex64],
    ldvr: usize,
) -> bool {
    if n == 0 {
        return false;
    }

    // The diagonal of B must be real; `zhgeqz` has already made it so.
    for j in 0..n {
        if p[j + j * ldp].im != 0.0 {
            return true;
        }
    }

    let safmin = SAFMIN;
    let ulp = ULP;
    let small = safmin * (n as f64) / ulp;
    let big = 1.0 / small;
    let bignum = 1.0 / (safmin * (n as f64));

    // The 1-norm of each column of the strictly upper triangular part of A and
    // B, to check for overflow in the triangular solver.
    let mut rwork = vec![0.0f64; 2 * n];
    let mut anorm = zabs1(s[0]);
    let mut bnorm = zabs1(p[0]);
    rwork[0] = 0.0;
    rwork[n] = 0.0;
    for j in 1..n {
        rwork[j] = 0.0;
        rwork[n + j] = 0.0;
        for i in 0..j {
            rwork[j] += zabs1(s[i + j * lds]);
            rwork[n + j] += zabs1(p[i + j * ldp]);
        }
        anorm = anorm.max(rwork[j] + zabs1(s[j + j * lds]));
        bnorm = bnorm.max(rwork[n + j] + zabs1(p[j + j * ldp]));
    }
    let ascale = 1.0 / anorm.max(safmin);
    let bscale = 1.0 / bnorm.max(safmin);

    let mut work = vec![zzero(); 2 * n];

    // Left eigenvectors.
    if compl {
        for je in 0..n {
            if zabs1(s[je + je * lds]) <= safmin && p[je + je * ldp].re.abs() <= safmin {
                // Singular matrix pencil: return the unit eigenvector.
                for jr in 0..n {
                    vl[jr + je * ldvl] = zzero();
                }
                vl[je + je * ldvl] = zone();
                continue;
            }

            let cf = ztgevc_coeffs(
                je, s, lds, p, ldp, ascale, bscale, anorm, bnorm, safmin, small, big, ulp,
            );
            let acoeff = cf.acoeff;
            let bcoeff = cf.bcoeff;
            let acoefa = cf.acoefa;
            let bcoefa = cf.bcoefa;
            let dmin = cf.dmin;

            let mut xmax = 1.0f64;
            for jr in 0..n {
                work[jr] = zzero();
            }
            work[je] = zone();

            // Triangular solve of (a A - b B)^H y = 0, rowwise.
            for j in (je + 1)..n {
                let temp = 1.0 / xmax;
                if acoefa * rwork[j] + bcoefa * rwork[n + j] > bignum * temp {
                    for jr in je..j {
                        work[jr] *= temp;
                    }
                    xmax = 1.0;
                }
                let mut suma = zzero();
                let mut sumb = zzero();
                for jr in je..j {
                    suma += s[jr + j * lds].conj() * work[jr];
                    sumb += p[jr + j * ldp].conj() * work[jr];
                }
                let mut sum = acoeff * suma - bcoeff.conj() * sumb;
                let mut d = (acoeff * s[j + j * lds] - bcoeff * p[j + j * ldp]).conj();
                if zabs1(d) <= dmin {
                    d = Complex64::new(dmin, 0.0);
                }
                if zabs1(d) < 1.0 && zabs1(sum) >= bignum * zabs1(d) {
                    let temp = 1.0 / zabs1(sum);
                    for jr in je..j {
                        work[jr] *= temp;
                    }
                    xmax *= temp;
                    sum *= temp;
                }
                work[j] = zladiv(-sum, d);
                xmax = xmax.max(zabs1(work[j]));
            }

            // Back transform with the Schur vectors in columns je..n of vl:
            // `zgemv('N', n, n+1-je, 1, VL(1,je), ldvl, WORK(je), 1, 0, ...)`.
            let (wv, wout) = work.split_at_mut(n);
            zgemv_n(vl, ldvl, n, je, n - je, &wv[je..n], wout);

            let mut ibeg = 0usize;
            let mut xmax2 = 0.0f64;
            for jr in ibeg..n {
                xmax2 = xmax2.max(zabs1(wout[jr]));
            }
            if xmax2 > safmin {
                let temp = 1.0 / xmax2;
                for jr in ibeg..n {
                    vl[jr + je * ldvl] = wout[jr] * temp;
                }
            } else {
                ibeg = n;
            }
            for jr in 0..ibeg {
                vl[jr + je * ldvl] = zzero();
            }
        }
    }

    // Right eigenvectors.
    if compr {
        for je in (0..n).rev() {
            if zabs1(s[je + je * lds]) <= safmin && p[je + je * ldp].re.abs() <= safmin {
                for jr in 0..n {
                    vr[jr + je * ldvr] = zzero();
                }
                vr[je + je * ldvr] = zone();
                continue;
            }

            let cf = ztgevc_coeffs(
                je, s, lds, p, ldp, ascale, bscale, anorm, bnorm, safmin, small, big, ulp,
            );
            let acoeff = cf.acoeff;
            let bcoeff = cf.bcoeff;
            let acoefa = cf.acoefa;
            let bcoefa = cf.bcoefa;
            let dmin = cf.dmin;

            for jr in 0..n {
                work[jr] = zzero();
            }
            for jr in 0..je {
                work[jr] = acoeff * s[jr + je * lds] - bcoeff * p[jr + je * ldp];
            }
            work[je] = zone();

            // Triangular solve of (a A - b B) x = 0, columnwise.
            for j in (0..je).rev() {
                let mut d = acoeff * s[j + j * lds] - bcoeff * p[j + j * ldp];
                if zabs1(d) <= dmin {
                    d = Complex64::new(dmin, 0.0);
                }
                if zabs1(d) < 1.0 && zabs1(work[j]) >= bignum * zabs1(d) {
                    let temp = 1.0 / zabs1(work[j]);
                    for jr in 0..=je {
                        work[jr] *= temp;
                    }
                }
                work[j] = zladiv(-work[j], d);
                if j > 0 {
                    if zabs1(work[j]) > 1.0 {
                        let temp = 1.0 / zabs1(work[j]);
                        if acoefa * rwork[j] + bcoefa * rwork[n + j] >= bignum * temp {
                            for jr in 0..=je {
                                work[jr] *= temp;
                            }
                        }
                    }
                    let ca = acoeff * work[j];
                    let cb = bcoeff * work[j];
                    for jr in 0..j {
                        work[jr] = work[jr] + ca * s[jr + j * lds] - cb * p[jr + j * ldp];
                    }
                }
            }

            // Back transform with the Schur vectors in columns 0..=je of vr:
            // `zgemv('N', n, je, 1, VR, ldvr, WORK, 1, 0, ...)`.
            let (wv, wout) = work.split_at_mut(n);
            zgemv_n(vr, ldvr, n, 0, je + 1, &wv[0..=je], wout);

            let mut iend = n;
            let mut xmax2 = 0.0f64;
            for jr in 0..iend {
                xmax2 = xmax2.max(zabs1(wout[jr]));
            }
            if xmax2 > safmin {
                let temp = 1.0 / xmax2;
                for jr in 0..iend {
                    vr[jr + je * ldvr] = wout[jr] * temp;
                }
            } else {
                iend = 0;
            }
            for jr in iend..n {
                vr[jr + je * ldvr] = zzero();
            }
        }
    }

    false
}

/// `zlaset('Full', n, n, 0, 1, a)`: set `a` to the identity.
fn zlaset_identity(a: &mut [Complex64], n: usize) {
    for j in 0..n {
        for i in 0..n {
            a[i + j * n] = if i == j { zone() } else { zzero() };
        }
    }
}

/// Normalize the columns of `v` so that the largest `ABS1` entry is 1, as the
/// reference's `zggev` does after undoing the balancing. Columns whose largest
/// entry is below `smlnum` are left alone.
fn zggev_normalize(n: usize, smlnum: f64, v: &mut [Complex64], ldv: usize) {
    for jc in 0..n {
        let mut temp = 0.0f64;
        for jr in 0..n {
            temp = temp.max(zabs1(v[jr + jc * ldv]));
        }
        if temp < smlnum {
            continue;
        }
        let temp = 1.0 / temp;
        for jr in 0..n {
            v[jr + jc * ldv] *= temp;
        }
    }
}

/// `zggev`: the generalized eigenvalues of the pair `(a, b)` and, optionally,
/// the left and/or right generalized eigenvectors.
///
/// This follows LAPACK 3.12.1's `zggev` with `JOBVL`/`JOBVR` decoded into
/// `calc_vl`/`calc_vr`. On exit `a` and `b` are overwritten, `alpha` and `beta`
/// hold the eigenvalues as `alpha / beta` (`beta` real and non-negative), and
/// `vl`/`vr` hold the eigenvectors in columns.
///
/// # Errors
///
/// [`LapackError::Convergence`] when the reference's `INFO` is non-zero: `1..=n`
/// for QZ non-convergence, `n+1..=2n` for an argument it rejected, and `n + 2`
/// when `ztgevc` failed to compute the eigenvectors.
#[allow(clippy::too_many_arguments)]
pub fn zggev(
    calc_vl: bool,
    calc_vr: bool,
    n: usize,
    a: &mut [Complex64],
    b: &mut [Complex64],
    alpha: &mut [Complex64],
    beta: &mut [Complex64],
    vl: &mut [Complex64],
    vr: &mut [Complex64],
) -> Result<(), LapackError> {
    assert_eq!(a.len(), n * n, "zggev: a must be n*n");
    assert_eq!(b.len(), n * n, "zggev: b must be n*n");
    assert_eq!(alpha.len(), n, "zggev: alpha must have n entries");
    assert_eq!(beta.len(), n, "zggev: beta must have n entries");
    if calc_vl {
        assert_eq!(vl.len(), n * n, "zggev: vl must be n*n");
    }
    if calc_vr {
        assert_eq!(vr.len(), n * n, "zggev: vr must be n*n");
    }

    if n == 0 {
        return Ok(());
    }
    let ilv = calc_vl || calc_vr;
    let lda = n;
    let ldb = n;

    // Machine constants.
    let eps = ULP; // DLAMCH('E') * DLAMCH('B')
    let smlnum0 = SAFMIN;
    let smlnum = smlnum0.sqrt() / eps;
    let bignum = 1.0 / smlnum;

    // Scale A if its norm is outside [smlnum, bignum].
    let anrm = zlange_m(a, n, n, n);
    let mut ilascl = false;
    let mut anrmto = 0.0;
    if anrm > 0.0 && anrm < smlnum {
        anrmto = smlnum;
        ilascl = true;
    } else if anrm > bignum {
        anrmto = bignum;
        ilascl = true;
    }
    if ilascl {
        zlascl_g(anrm, anrmto, a, lda, n, n);
    }

    // Scale B likewise.
    let bnrm = zlange_m(b, n, n, n);
    let mut ilbscl = false;
    let mut bnrmto = 0.0;
    if bnrm > 0.0 && bnrm < smlnum {
        bnrmto = smlnum;
        ilbscl = true;
    } else if bnrm > bignum {
        bnrmto = bignum;
        ilbscl = true;
    }
    if ilbscl {
        zlascl_g(bnrm, bnrmto, b, ldb, n, n);
    }

    // Permute the matrices to isolate eigenvalues.
    let (ilo, ihi, lscale, rscale) = zggbal_p(n, a, lda, b, ldb);
    let ihi0 = ihi - 1; // 0-based inclusive

    // Reduce B to triangular form (QR of B).
    let irows = ihi - ilo;
    let icols = if ilv { n - ilo } else { irows };
    let mut tau = vec![zzero(); irows.max(1)];
    zgeqr2(b, ldb, ilo, ilo, irows, icols, &mut tau);

    // Apply the unitary transformation to A.
    zunm2r_left_conj(a, lda, ilo, ilo, irows, icols, b, ldb, ilo, ilo, irows, &tau);

    // Initialize VL then generate Q in it.
    if calc_vl {
        zlaset_identity(vl, n);
        if irows > 1 {
            // zlacpy('L', irows-1, irows-1, B(ilo+1,ilo), ldb, VL(ilo+1,ilo), n).
            for r in 1..irows {
                for c in 1..=r {
                    vl[(ilo + r) + (ilo + c - 1) * n] = b[(ilo + r) + (ilo + c - 1) * n];
                }
            }
        }
        zung2r_sub(vl, n, ilo, ilo, irows, irows, irows, &tau);
    }

    // Initialize VR.
    if calc_vr {
        zlaset_identity(vr, n);
    }

    // Reduce to generalized Hessenberg form.
    zgghrd(calc_vl, calc_vr, n, ilo, ihi0, a, lda, b, ldb, vl, n, vr, n);

    // Perform the QZ algorithm.
    let ierr = zhgeqz(
        ilv, calc_vl, calc_vr, n, ilo, ihi0, a, lda, b, ldb, alpha, beta, vl, n, vr, n,
    );
    // The QZ iteration's INFO is 1..=n for non-convergence and n+1..=2n for an
    // argument it rejected; both are errors here.
    if ierr != 0 {
        return Err(LapackError::Convergence);
    }

    // Compute the eigenvectors.
    if ilv {
        let bad = ztgevc(calc_vl, calc_vr, n, a, lda, b, ldb, vl, n, vr, n);
        // The eigenvector back-substitution failed (LAPACK's INFO = N+2).
        if bad {
            return Err(LapackError::Convergence);
        }
        // Undo the balancing on VL and VR, and normalize.
        if calc_vl {
            zggbak_p(false, n, ilo, ihi0, &lscale, &rscale, n, vl, n);
            zggev_normalize(n, smlnum, vl, n);
        }
        if calc_vr {
            zggbak_p(true, n, ilo, ihi0, &lscale, &rscale, n, vr, n);
            zggev_normalize(n, smlnum, vr, n);
        }
    }

    // Undo the scaling if necessary.
    if ilascl {
        let f = anrm / anrmto;
        for i in 0..n {
            alpha[i] *= f;
        }
    }
    if ilbscl {
        let f = bnrm / bnrmto;
        for i in 0..n {
            beta[i] *= f;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run `dggev` on copies of the pair and return `(alphar, alphai, beta, u, v)`.
    fn solve_full(a: &[f64], b: &[f64], n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut aw = a.to_vec();
        let mut bw = b.to_vec();
        let mut ar = vec![0.0; n];
        let mut ai = vec![0.0; n];
        let mut be = vec![0.0; n];
        let mut u = vec![0.0; n * n];
        let mut v = vec![0.0; n * n];
        dggev(
            true, true, n, &mut aw, &mut bw, &mut ar, &mut ai, &mut be, &mut u, &mut v,
        )
        .expect("dggev converged");
        (ar, ai, be, u, v)
    }

    /// Like [`solve_full`], for the right eigenvectors only.
    fn solve(a: &[f64], b: &[f64], n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let (ar, ai, be, _u, v) = solve_full(a, b, n);
        (ar, ai, be, v)
    }

    /// `beta` at this fraction of the matrix scale or below counts as zero: the
    /// eigenvalue is then infinite, and the condition is `B v = 0`, not
    /// `A v = lambda B v`.
    fn beta_is_zero(beta: f64, scale: f64) -> bool {
        beta.abs() <= 1e-10 * scale
    }

    fn matrix_scale(a: &[f64]) -> f64 {
        a.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1.0)
    }

    /// The worst `A v - lambda B v` over all right eigenvectors, in complex
    /// arithmetic for the conjugate pairs stored in LAPACK's compact layout.
    fn worst_residual(a: &[f64], b: &[f64], n: usize, ar: &[f64], ai: &[f64], be: &[f64], v: &[f64]) -> f64 {
        let scale = matrix_scale(b);
        let mut worst = 0.0f64;
        let mut j = 0usize;
        while j < n {
            let nw = if ai[j] != 0.0 { 2 } else { 1 };
            let lr = if beta_is_zero(be[j], scale) { 0.0 } else { ar[j] / be[j] };
            let li = if beta_is_zero(be[j], scale) { 0.0 } else { ai[j] / be[j] };
            for i in 0..n {
                let mut avr = 0.0;
                let mut avi = 0.0;
                let mut bvr = 0.0;
                let mut bvi = 0.0;
                for k in 0..n {
                    let ur = v[k + j * n];
                    let ui = if nw == 2 { v[k + (j + 1) * n] } else { 0.0 };
                    avr += a[i + k * n] * ur;
                    avi += a[i + k * n] * ui;
                    bvr += b[i + k * n] * ur;
                    bvi += b[i + k * n] * ui;
                }
                if beta_is_zero(be[j], scale) {
                    worst = worst.max(bvr.abs().max(bvi.abs()));
                } else {
                    worst = worst.max((avr - (lr * bvr - li * bvi)).abs());
                    worst = worst.max((avi - (lr * bvi + li * bvr)).abs());
                }
            }
            j += nw;
        }
        worst
    }

    /// The worst `u^H A - lambda u^H B` over all left eigenvectors, i.e.
    /// `A^H u = conj(lambda) B^H u`; for an infinite eigenvalue, `B^H u = 0`.
    ///
    /// `mat_gen_eigen_lr_works` never looks at `u` (and neither does the parity
    /// test for `dggev`), so a wrong left eigenvector passes there. This checks
    /// what those leave out.
    fn worst_left_residual(a: &[f64], b: &[f64], n: usize, ar: &[f64], ai: &[f64], be: &[f64], u: &[f64]) -> f64 {
        let scale = matrix_scale(b);
        let mut worst = 0.0f64;
        let mut j = 0usize;
        while j < n {
            let nw = if ai[j] != 0.0 { 2 } else { 1 };
            let infinite = beta_is_zero(be[j], scale);
            let lr = if infinite { 0.0 } else { ar[j] / be[j] };
            let li = if infinite { 0.0 } else { ai[j] / be[j] };
            for i in 0..n {
                // (A^H u)_i = sum_k conj(A_ki) u_k, with A real here.
                let mut ahr = 0.0;
                let mut ahi = 0.0;
                let mut bhr = 0.0;
                let mut bhi = 0.0;
                for k in 0..n {
                    let ur = u[k + j * n];
                    let ui = if nw == 2 { u[k + (j + 1) * n] } else { 0.0 };
                    ahr += a[k + i * n] * ur;
                    ahi += a[k + i * n] * ui;
                    bhr += b[k + i * n] * ur;
                    bhi += b[k + i * n] * ui;
                }
                if infinite {
                    worst = worst.max(bhr.abs().max(bhi.abs()));
                } else {
                    // RHS = conj(lambda) * (B^H u).
                    let rhs_r = lr * bhr + li * bhi;
                    let rhs_i = lr * bhi - li * bhr;
                    worst = worst.max((ahr - rhs_r).abs().max((ahi - rhs_i).abs()));
                }
            }
            j += nw;
        }
        worst
    }

    #[test]
    fn dggev_is_exact_on_a_diagonal_pair() {
        // A = diag(1, 2, 3), B = I: the eigenvalues are the diagonal of A and
        // nothing has to be reduced, so the answer is exactly representable.
        let a = [1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0];
        let b = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let (ar, ai, be, vr) = solve(&a, &b, 3);
        assert_eq!(ar, vec![1.0, 2.0, 3.0]);
        assert_eq!(ai, vec![0.0, 0.0, 0.0]);
        assert_eq!(be, vec![1.0, 1.0, 1.0]);
        // The eigenvectors are the coordinate vectors, normalized to max abs 1.
        for j in 0..3 {
            for i in 0..3 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert_eq!(vr[i + j * 3], want);
            }
        }
    }

    #[test]
    fn dggev_matches_the_reference_on_a_general_3x3_pair() {
        // The pair from `mat_gen_eigen_works` (column-major). B is singular, so
        // the third eigenvalue is infinite and reported as beta = 0.
        let a = [10.0, 1.0, 1.0, 1.0, 2.0, 1.0, 2.0, -1.0, 2.0];
        let b = [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0];
        let (ar, ai, be, vr) = solve(&a, &b, 3);
        let ar_ref = [-5.040877033060623, 1.9878762745291856, 4.490731195102491];
        let be_ref = [1.0510183823362071, 12.076677914155423, 0.0];
        for i in 0..3 {
            assert!((ar[i] - ar_ref[i]).abs() < 1e-13, "alphar[{i}] = {}", ar[i]);
            assert!(ai[i].abs() < 1e-13, "alphai[{i}] = {}", ai[i]);
            assert!((be[i] - be_ref[i]).abs() < 1e-13, "beta[{i}] = {}", be[i]);
        }
        let vr_ref = [
            0.7711213016981152,
            0.4700458576415154,
            -1.0, // column 0
            0.1548092699389088,
            -1.0,
            -0.5655472382416965, // column 1
            0.5000000000000012,
            -1.0,
            0.4999999999999993, // column 2 (beta = 0)
        ];
        for i in 0..9 {
            assert!((vr[i] - vr_ref[i]).abs() < 1e-13, "vr[{i}] = {}", vr[i]);
        }
    }

    #[test]
    fn dggev_satisfies_the_defining_equation() {
        // A complex conjugate pair: A has no real eigenvalues, so the 2x2 path
        // through `dlasv2` and the complex eigenvector recovery are exercised.
        let complex_pair_a = [0.0, -1.0, 1.0, 0.0];
        let complex_pair_b = [1.0, 0.0, 0.0, 1.0];
        let (ar, ai, be, vr) = solve(&complex_pair_a, &complex_pair_b, 2);
        assert!(ai[0].abs() > 0.0, "the pair should have complex eigenvalues");
        assert!(worst_residual(&complex_pair_a, &complex_pair_b, 2, &ar, &ai, &be, &vr) < 1e-14);

        // A symmetric positive definite B (the acceptance test's construction)
        // over several sizes, so no eigenvalue is infinite.
        for n in [1usize, 2, 3, 5, 8] {
            let mut ra = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    ra[i + j * n] = if i == j {
                        n as f64 + 1.0 + i as f64 * 0.5
                    } else {
                        (((i * 7 + j * 13) % 11) as f64) - 5.0
                    };
                }
            }
            let mut b = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    let mut sum = 0.0;
                    for k in 0..n {
                        sum += ra[k + i * n] * ra[k + j * n];
                    }
                    b[i + j * n] = if i == j { sum + n as f64 } else { sum };
                }
            }
            let (ar, ai, be, vr) = solve(&ra, &b, n);
            let resid = worst_residual(&ra, &b, n, &ar, &ai, &be, &vr);
            assert!(resid / matrix_scale(&b) < 1e-10, "n={n}: worst residual = {resid}");
        }
    }

    #[test]
    fn dggev_left_eigenvectors_satisfy_their_defining_equation() {
        // Pairs whose right eigenvectors are checked by the other tests: a
        // general one with a singular B (so one eigenvalue is infinite), one with
        // a complex conjugate pair, and a 4x4 pseudo-random pair whose pencil has
        // a complex pair (so the left solve takes `dlaln2`'s LTRANS = .TRUE.
        // complex branch).
        let cases: [(&[f64], &[f64], usize); 3] = [
            (
                &[10.0, 1.0, 1.0, 1.0, 2.0, 1.0, 2.0, -1.0, 2.0],
                &[1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0],
                3,
            ),
            (&[0.0, -1.0, 1.0, 0.0], &[1.0, 0.0, 0.0, 1.0], 2),
            (
                &[
                    0.24526520538822338,
                    -0.5935434796942392,
                    0.7317641783236293,
                    0.9592431913013717,
                    0.09950539312702711,
                    0.24321636829860016,
                    -0.22561210646640695,
                    0.5765589928454418,
                    0.17938593451307838,
                    -0.8901599342432223,
                    0.9295132922669012,
                    0.7758235044857111,
                    -0.4335690440811617,
                    0.28055693413894156,
                    -0.3584680296066671,
                    0.09441012627953915,
                ],
                &[
                    5.020367963895579,
                    -0.38515843098893854,
                    0.5855832463742635,
                    -0.2920155364457001,
                    -0.38515843098893854,
                    5.461515045558352,
                    -1.0368325236725757,
                    0.9204718542517483,
                    0.5855832463742635,
                    -1.0368325236725757,
                    5.2345654931927745,
                    -0.5344466716537862,
                    -0.2920155364457001,
                    0.9204718542517483,
                    -0.5344466716537862,
                    5.759993732066009,
                ],
                4,
            ),
        ];
        for (a, b, n) in cases {
            let (ar, ai, be, u, v) = solve_full(a, b, n);
            let right = worst_residual(a, b, n, &ar, &ai, &be, &v);
            let left = worst_left_residual(a, b, n, &ar, &ai, &be, &u);
            assert!(right < 1e-12, "worst right residual = {right}");
            assert!(left < 1e-12, "worst left residual = {left}");
        }
    }

    #[test]
    fn dggev_handles_isolated_eigenvalues() {
        // Block-diagonal pencils: the first (or last) row/column has a single
        // non-zero, which is what `dggbal`'s permutation isolates before the QZ
        // iteration. The remaining 2x2 pencil  [[1, 2], [3, 4]] - lambda I
        // [[2, 0], [0, 1]]  has  lambda = (9 +- sqrt(97)) / 4.
        let lambda2 = (9.0 + 97.0f64.sqrt()) / 4.0;
        let lambda3 = (9.0 - 97.0f64.sqrt()) / 4.0;

        let cases: [(&[f64], &[f64], [f64; 3]); 3] = [
            (
                // Isolated first block.
                &[3.0, 0.0, 0.0, 0.0, 1.0, 3.0, 0.0, 2.0, 4.0],
                &[5.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0],
                [0.6, lambda2, lambda3],
            ),
            (
                // Isolated last block.
                &[1.0, 3.0, 0.0, 2.0, 4.0, 0.0, 0.0, 0.0, 6.0],
                &[2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0],
                [3.0, lambda2, lambda3],
            ),
            (
                // Upper triangular: every eigenvalue is isolated by the *column*
                // test (label 100 of `dggbal`), which the row test above does not
                // reach.
                &[1.0, 0.0, 0.0, 2.0, 4.0, 0.0, 3.0, 5.0, 6.0],
                &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 4.0, 6.0],
            ),
        ];
        for (a, b, want) in cases {
            let n = 3;
            let (ar, ai, be, vr) = solve(a, b, n);
            let mut lambda: Vec<f64> = (0..n).map(|j| ar[j] / be[j]).collect();
            lambda.sort_by(|x, y| x.partial_cmp(y).unwrap());
            let mut want = want;
            want.sort_by(|x, y| x.partial_cmp(y).unwrap());
            for j in 0..n {
                assert!(
                    (lambda[j] - want[j]).abs() < 1e-12,
                    "lambda[{j}] = {} but want {}",
                    lambda[j],
                    want[j]
                );
                assert!(ai[j].abs() < 1e-12, "alphai[{j}] = {}", ai[j]);
            }
            let resid = worst_residual(a, b, n, &ar, &ai, &be, &vr);
            assert!(resid < 1e-12, "worst residual = {resid}");
        }
    }

    #[test]
    fn dggev_survives_a_pseudo_random_sweep() {
        // Deterministic pseudo-random pencils (a Lehmer generator, so the test is
        // reproducible). Larger sizes mean more QZ sweeps, including the
        // exceptional shifts taken every tenth iteration; half of the pencils
        // have a singular `b`, so the `T(j,j) = 0` deflation and infinite
        // eigenvalues are exercised too. Both left and right eigenvectors are
        // checked.
        let mut state: u64 = 1;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
        };
        for trial in 0..150usize {
            let n = 1 + (trial % 8);
            let mut a = vec![0.0; n * n];
            for x in a.iter_mut() {
                *x = next() * 3.0;
            }
            let mut b = vec![0.0; n * n];
            for x in b.iter_mut() {
                *x = next() * 3.0;
            }
            if trial % 2 == 0 && n > 1 {
                // Make the last row a combination of the first two.
                for j in 0..n {
                    b[(n - 1) + j * n] = b[j * n] + 2.0 * b[1 + j * n];
                }
            }
            let (ar, ai, be, u, v) = solve_full(&a, &b, n);
            let scale = matrix_scale(&b);
            let right = worst_residual(&a, &b, n, &ar, &ai, &be, &v);
            let left = worst_left_residual(&a, &b, n, &ar, &ai, &be, &u);
            assert!(right / scale < 1e-8, "trial {trial} n={n}: right residual = {right}");
            assert!(left / scale < 1e-8, "trial {trial} n={n}: left residual = {left}");
        }
    }

    /// Run the native `zggev` on copies of the pair and return
    /// `(alpha, beta, u, v)`.
    #[allow(clippy::type_complexity)]
    fn zggev_solve(
        a: &[Complex64],
        b: &[Complex64],
        n: usize,
    ) -> (Vec<Complex64>, Vec<Complex64>, Vec<Complex64>, Vec<Complex64>) {
        let mut aw = a.to_vec();
        let mut bw = b.to_vec();
        let mut alpha = vec![zzero(); n];
        let mut beta = vec![zzero(); n];
        let mut u = vec![zzero(); n * n];
        let mut v = vec![zzero(); n * n];
        zggev(true, true, n, &mut aw, &mut bw, &mut alpha, &mut beta, &mut u, &mut v).expect("zggev converged");
        (alpha, beta, u, v)
    }

    #[test]
    fn zggev_satisfies_the_defining_equation() {
        // A genuinely complex pair (every entry has a non-zero imaginary part,
        // so a missing conjugate somewhere cannot cancel out) with a Hermitian
        // positive definite `B = R^H R + n I`, chosen so that no eigenvalue is
        // infinite.
        const N: usize = 4;
        #[rustfmt::skip]
        let a = [
            Complex64::new( 1.0,  2.0), Complex64::new( 0.0, -1.0), Complex64::new( 2.0,  1.0), Complex64::new( 1.0, -1.0),
            Complex64::new( 3.0, -1.0), Complex64::new( 2.0,  3.0), Complex64::new( 0.0,  2.0), Complex64::new( 2.0,  0.0),
            Complex64::new( 2.0,  0.0), Complex64::new( 1.0,  1.0), Complex64::new( 3.0, -1.0), Complex64::new( 0.0,  1.0),
            Complex64::new( 0.0,  1.0), Complex64::new( 4.0, -2.0), Complex64::new( 1.0,  0.0), Complex64::new( 2.0,  2.0),
        ];
        #[rustfmt::skip]
        let r = [
            Complex64::new( 1.0,  1.0), Complex64::new( 0.0,  0.0), Complex64::new( 1.0,  0.0), Complex64::new( 0.0,  0.0),
            Complex64::new( 0.0,  0.0), Complex64::new( 2.0, -1.0), Complex64::new( 0.0,  0.0), Complex64::new( 1.0,  0.0),
            Complex64::new( 1.0,  0.0), Complex64::new( 0.0,  0.0), Complex64::new( 1.0,  2.0), Complex64::new( 0.0,  0.0),
            Complex64::new( 0.0,  0.0), Complex64::new( 1.0,  0.0), Complex64::new( 0.0,  0.0), Complex64::new( 2.0,  1.0),
        ];
        // B = R^H R + n I.
        let mut b = [zzero(); N * N];
        for i in 0..N {
            for j in 0..N {
                let mut sum = zzero();
                for k in 0..N {
                    sum += r[k + i * N].conj() * r[k + j * N];
                }
                b[i + j * N] = if i == j {
                    sum + Complex64::new(N as f64, 0.0)
                } else {
                    sum
                };
            }
        }

        let (alpha, beta, u, v) = zggev_solve(&a, &b, N);

        // `B` is definite, so every `beta` is non-zero (and, as the reference
        // requires, real and non-negative). If this ever stops holding, the
        // residual below would compare infinities instead of failing honestly.
        for j in 0..N {
            assert!(beta[j].im == 0.0, "beta[{j}] should be real: {}", beta[j]);
            assert!(beta[j].re > 1e-8, "beta[{j}] = {} should be non-zero", beta[j]);
        }

        // The defining equation, for every right eigenvector:
        //     A v_j = lambda_j B v_j,   lambda_j = alpha_j / beta_j.
        for j in 0..N {
            let lambda = alpha[j] / beta[j];
            for i in 0..N {
                let mut av = zzero();
                let mut bv = zzero();
                for k in 0..N {
                    av += a[i + k * N] * v[k + j * N];
                    bv += b[i + k * N] * v[k + j * N];
                }
                let resid = av - lambda * bv;
                assert!(
                    resid.norm() < 1e-11,
                    "right eigenvector {j}, row {i}: residual = {resid}"
                );
            }
        }

        // The left eigenvectors satisfy u_j^H A = lambda_j u_j^H B.
        for j in 0..N {
            let lambda = alpha[j] / beta[j];
            for i in 0..N {
                let mut au = zzero();
                let mut bu = zzero();
                for k in 0..N {
                    au += u[k + j * N].conj() * a[k + i * N];
                    bu += u[k + j * N].conj() * b[k + i * N];
                }
                let resid = au - lambda * bu;
                assert!(
                    resid.norm() < 1e-11,
                    "left eigenvector {j}, col {i}: residual = {resid}"
                );
            }
        }

        // The values LAPACK 3.12.1's own `zggev` returns for this pair, in the
        // same order, so the QZ iteration and the balancing are checked against
        // the reference and not merely self-consistently.
        let alpha_ref = [
            Complex64::new(3.920168438253699e-01, 5.255147830858519e+00),
            Complex64::new(-1.150893246652945e+00, 9.439744406542160e-01),
            Complex64::new(5.214715653920524e+00, 1.627879167945076e+00),
            Complex64::new(2.174485077144718e+00, -1.398564397480211e+00),
        ];
        let beta_ref = [
            Complex64::new(6.520111926148862e+00, 0.0),
            Complex64::new(7.567188908625723e+00, 0.0),
            Complex64::new(1.057904013366485e+01, 0.0),
            Complex64::new(9.962479757893947e+00, 0.0),
        ];
        for j in 0..N {
            assert!((alpha[j] - alpha_ref[j]).norm() < 1e-10, "alpha[{j}] = {}", alpha[j]);
            assert!((beta[j] - beta_ref[j]).norm() < 1e-10, "beta[{j}] = {}", beta[j]);
        }
    }

    /// A deterministic pseudo-random sweep of complex pencils, half of them with
    /// a singular `B`, so the infinite eigenvalues and the `t(j,j) = 0`
    /// deflation are exercised. Both the right and the left defining equations
    /// are checked with the backward-error measure appropriate to each
    /// eigenvalue (a near-zero `beta` makes `lambda = alpha / beta` huge, so its
    /// residual is judged against `|lambda|` too).
    #[test]
    fn zggev_survives_a_pseudo_random_sweep() {
        fn next(state: &mut u64) -> f64 {
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*state >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
        }
        let mut state: u64 = 20261016;
        for trial in 0..160usize {
            let n = 1 + (trial % 8);
            // Fill column-major, as the routine and the reference both expect.
            let mut a = vec![zzero(); n * n];
            let mut b = vec![zzero(); n * n];
            for j in 0..n {
                for i in 0..n {
                    a[i + j * n] = Complex64::new(next(&mut state) * 3.0, next(&mut state) * 3.0);
                    b[i + j * n] = Complex64::new(next(&mut state) * 3.0, next(&mut state) * 3.0);
                }
            }
            if trial % 2 == 0 && n > 1 {
                // Make the last row a combination of the first two, so `B` is
                // singular and one eigenvalue is infinite.
                for j in 0..n {
                    b[(n - 1) + j * n] = b[j * n] + Complex64::new(2.0, 0.0) * b[1 + j * n];
                }
            }
            let (alpha, beta, u, v) = zggev_solve(&a, &b, n);
            let ascale = 1.0f64.max(a.iter().map(|z| z.norm()).fold(0.0, f64::max));
            let bscale = 1.0f64.max(b.iter().map(|z| z.norm()).fold(0.0, f64::max));
            for j in 0..n {
                assert!(
                    beta[j].im == 0.0 && beta[j].re >= 0.0,
                    "trial {trial}: beta[{j}] = {} should be real and non-negative",
                    beta[j]
                );
                // `0 / 0` (a singular pencil) leaves an arbitrary vector, so
                // only the genuinely infinite eigenvalues carry information.
                let finite = beta[j].re != 0.0;
                let infinite = !finite && alpha[j].norm() > 1e-8 * ascale;
                if !finite && !infinite {
                    continue;
                }
                let lambda = if finite { alpha[j] / beta[j] } else { zzero() };
                let denom = if finite {
                    ascale + lambda.norm() * bscale
                } else {
                    bscale
                };
                for i in 0..n {
                    let mut av = zzero();
                    let mut bv = zzero();
                    let mut au = zzero();
                    let mut bu = zzero();
                    for k in 0..n {
                        av += a[i + k * n] * v[k + j * n];
                        bv += b[i + k * n] * v[k + j * n];
                        au += u[k + j * n].conj() * a[k + i * n];
                        bu += u[k + j * n].conj() * b[k + i * n];
                    }
                    let (r, l) = if finite {
                        ((av - lambda * bv).norm(), (au - lambda * bu).norm())
                    } else {
                        (bv.norm(), bu.norm())
                    };
                    assert!(
                        r / denom < 1e-9,
                        "trial {trial} n={n}: right ({i},{j}) residual {}",
                        r / denom
                    );
                    assert!(
                        l / denom < 1e-9,
                        "trial {trial} n={n}: left ({i},{j}) residual {}",
                        l / denom
                    );
                }
            }
        }
    }

    /// Non-convergence of the QZ iteration reaches the caller as an error.
    ///
    /// A NaN pair keeps `dhgeqz`/`zhgeqz`'s convergence test false, so the
    /// iteration exhausts its limit and reports `INFO > 0`; the driver must turn
    /// that into `LapackError::Convergence` rather than returning partly converged
    /// alpha/beta. Reverting either `return Err(...)` to `return ierr` makes this
    /// test fail.
    #[test]
    fn qz_non_convergence_is_reported_rather_than_returned_as_a_result() {
        let n = 4;
        let mut a = vec![f64::NAN; n * n];
        let mut b = vec![f64::NAN; n * n];
        let mut ar = vec![0.0; n];
        let mut ai = vec![0.0; n];
        let mut be = vec![0.0; n];
        let mut vl = vec![0.0; n * n];
        let mut v = vec![0.0; n * n];
        assert_eq!(
            dggev(
                true, true, n, &mut a, &mut b, &mut ar, &mut ai, &mut be, &mut vl, &mut v
            ),
            Err(LapackError::Convergence),
            "dggev must report non-convergence"
        );

        let mut ca = vec![Complex64::new(f64::NAN, 0.0); n * n];
        let mut cb = vec![Complex64::new(f64::NAN, 0.0); n * n];
        let mut alpha = vec![Complex64::new(0.0, 0.0); n];
        let mut beta = vec![Complex64::new(0.0, 0.0); n];
        let mut cvl = vec![Complex64::new(0.0, 0.0); n * n];
        let mut cv = vec![Complex64::new(0.0, 0.0); n * n];
        assert_eq!(
            zggev(
                true, true, n, &mut ca, &mut cb, &mut alpha, &mut beta, &mut cvl, &mut cv
            ),
            Err(LapackError::Convergence),
            "zggev must report non-convergence"
        );
    }
}
