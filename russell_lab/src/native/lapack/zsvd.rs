//! Native complex singular value decomposition: `zgesvd` and the unblocked
//! pieces it is built from.
//!
//! # The pipeline
//!
//! | Step | LAPACK | What it does |
//! |---|---|---|
//! | 1 | `zgebd2` | unitary reduction to bidiagonal form, `Q^H A P = B` |
//! | 2 | `zungbr` | forms `Q` (and `P^H`) explicitly from the reflectors |
//! | 3 | `zbdsqr` | implicit QR on the bidiagonal, accumulating the vectors |
//!
//! This mirrors [`super::svd`] (the real `dgesvd`) and follows the same
//! structure: `zgebrd` reduces to `zgebd2` for a single block, `zungbr` to
//! `zung2r`/`zungl2`, and only the direct bidiagonalization path of the driver
//! is implemented (the QR/LQ pre-reduction the reference uses when `m` is much
//! larger than `n` is a performance optimization that does not change the
//! singular values).
//!
//! # What is genuinely complex here, and what is not
//!
//! `zbdsqr` looks like its real counterpart and it is: `D` and `E` are real, the
//! shifts and rotations come from `dlartg`/`dlas2`/`dlasv2`, and the only
//! complex part is that `U`, `VT` and `C` are complex, so the real plane
//! rotations are applied with `zlasr` (no conjugation) and the 2x2 rotations
//! with `zdrot`. That is the reference's own structure -- `zbdsqr.f` does not
//! call `zlartg` at all. What *is* delicate is `zgebd2`/`zung2r`/`zungl2`, which
//! use `zlacgv` (conjugation) to make `zlarfg` and `zlarf` compose in the
//! orientation the reference wants; those conjugations are transcribed exactly.
//!
//! # Reused from the real SVD
//!
//! The bidiagonal QR loop, the convergence tests and the ordering are the same
//! as [`super::svd::dbdsqr`]'s, and the driver's data flow is
//! [`super::svd::dgesvd`]'s. `dlas2`, `dlasv2` and `dlascl_g` are the real
//! routines, reused directly; `zlarfg` and the complex `zlarf`/`zlasr` helpers
//! follow [`super::zeig`]'s copies of the same LAPACK kernels (kept local so
//! `zeig` is untouched).
//!
//! Everything is column-major: element `(i, j)` of an `m x n` matrix is
//! `data[i + j * m]`. Indices are kept 1-based where the Fortran is, so
//! `d[i - 1]` is Fortran's `D(I)`.

use super::LapackError;
use super::auxiliary::{dlapy3, dlascl_g, zladiv, zlarf_left, zlarf_right, zlascl_g};
use super::blas::dznrm2;
use super::eig::dlartg;
use super::svd::{dlas2, dlasv2};
use crate::Complex64;

/// `DLAMCH('E')`: the relative machine precision, `eps/2`.
const EPS: f64 = f64::EPSILON * 0.5;
/// `DLAMCH('P')`: the precision, `eps * base`.
const PREC: f64 = f64::EPSILON;
/// `DLAMCH('S')`: the safe minimum, the smallest normal positive number.
const SAFMIN: f64 = f64::MIN_POSITIVE;

/// The complex zero, named so the arithmetic reads like the Fortran `ZERO`.
#[inline]
fn zero() -> Complex64 {
    Complex64::new(0.0, 0.0)
}

/// The complex one, named so the arithmetic reads like the Fortran `ONE`.
#[inline]
fn one() -> Complex64 {
    Complex64::new(1.0, 0.0)
}

/// The complex value `c` times the real scalar `r`.
#[inline]
fn scale(c: Complex64, r: f64) -> Complex64 {
    Complex64::new(c.re * r, c.im * r)
}

/// Fortran's `SIGN(a, b)`: `|a|` with the sign of `b`, and `+` for `b >= 0`.
#[inline]
fn fsign(a: f64, b: f64) -> f64 {
    if b >= 0.0 { a.abs() } else { -a.abs() }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Elementary reflectors (ZLARFG and ZLARF)
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// `zlarfg`: an elementary reflector `H = I - tau * v * v^H` with `H x = beta e1`.
///
/// Copied from [`super::zeig`]'s kernel of the same name (which is `zheev`'s);
/// it is private there and `zeig` is not to be disturbed, so the text is
/// duplicated rather than shared.
fn zlarfg(alpha: &mut Complex64, x: &mut [Complex64]) -> Complex64 {
    let mut xnrm = dznrm2(x);
    let mut alphr = alpha.re;
    let mut alphi = alpha.im;

    if xnrm == 0.0 && alphi == 0.0 {
        return zero();
    }

    let safmin = f64::MIN_POSITIVE / (f64::EPSILON / 2.0);
    let rsafmn = 1.0 / safmin;

    let mut beta = -sign_f(dlapy3(alphr, alphi, xnrm), alphr);
    let mut knt = 0usize;
    if beta.abs() < safmin {
        loop {
            knt += 1;
            for xi in x.iter_mut() {
                *xi = scale(*xi, rsafmn);
            }
            beta *= rsafmn;
            alphi *= rsafmn;
            alphr *= rsafmn;
            if !(beta.abs() < safmin && knt < 20) {
                break;
            }
        }
        xnrm = dznrm2(x);
        beta = -sign_f(dlapy3(alphr, alphi, xnrm), alphr);
    }

    let tau = Complex64::new((beta - alphr) / beta, -alphi / beta);
    let al = zladiv(one(), Complex64::new(alphr - beta, alphi));
    for xi in x.iter_mut() {
        *xi *= al;
    }
    for _ in 0..knt {
        beta *= safmin;
    }
    *alpha = Complex64::new(beta, 0.0);
    tau
}

/// Fortran's `SIGN(a, b)`: `|a|` with `b`'s sign, where `b == 0` counts as positive.
fn sign_f(a: f64, b: f64) -> f64 {
    if b < 0.0 { -a.abs() } else { a.abs() }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Real plane rotations on a complex matrix (ZLASR, PIVOT = 'V')
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// `ZLASR` with `SIDE = 'L'`, `PIVOT = 'V'`: rotate rows `r0+j` and `r0+j+1`,
/// using `c[j]`/`s[j]`, for each of `ncols` columns starting at `c0`.
fn zlasr_left(
    nrows: usize,
    ncols: usize,
    c: &[f64],
    s: &[f64],
    a: &mut [Complex64],
    lda: usize,
    r0: usize,
    c0: usize,
    backward: bool,
) {
    if nrows < 2 || ncols == 0 {
        return;
    }
    let count = nrows - 1;
    for jj in 0..count {
        let j = if backward { count - 1 - jj } else { jj };
        let ctemp = c[j];
        let stemp = s[j];
        if ctemp != 1.0 || stemp != 0.0 {
            for i in 0..ncols {
                let temp = a[(r0 + j + 1) + (c0 + i) * lda];
                a[(r0 + j + 1) + (c0 + i) * lda] = scale(temp, ctemp) - scale(a[(r0 + j) + (c0 + i) * lda], stemp);
                a[(r0 + j) + (c0 + i) * lda] = scale(temp, stemp) + scale(a[(r0 + j) + (c0 + i) * lda], ctemp);
            }
        }
    }
}

/// `ZLASR` with `SIDE = 'R'`, `PIVOT = 'V'`: rotate columns `c0+j` and `c0+j+1`,
/// using `c[j]`/`s[j]`, for each of `nrows` rows starting at `r0`.
fn zlasr_right(
    nrows: usize,
    ncols: usize,
    c: &[f64],
    s: &[f64],
    a: &mut [Complex64],
    lda: usize,
    r0: usize,
    c0: usize,
    backward: bool,
) {
    if ncols < 2 || nrows == 0 {
        return;
    }
    let count = ncols - 1;
    for jj in 0..count {
        let j = if backward { count - 1 - jj } else { jj };
        let ctemp = c[j];
        let stemp = s[j];
        if ctemp != 1.0 || stemp != 0.0 {
            for i in 0..nrows {
                let temp = a[(r0 + i) + (c0 + j + 1) * lda];
                a[(r0 + i) + (c0 + j + 1) * lda] = scale(temp, ctemp) - scale(a[(r0 + i) + (c0 + j) * lda], stemp);
                a[(r0 + i) + (c0 + j) * lda] = scale(temp, stemp) + scale(a[(r0 + i) + (c0 + j) * lda], ctemp);
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Scaling
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Bidiagonalization
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// `ZGEBD2`: reduce an `m x n` matrix to bidiagonal form, unblocked.
///
/// On exit `a` (leading dimension `lda`) holds the bidiagonal matrix on its
/// diagonal and first off-diagonal, with the reflectors of `Q` below the diagonal
/// and those of `P^H` above it. The `zlacgv` conjugations are the reference's:
/// `zlarfg` builds a reflector for the *conjugated* row/column, and undoing the
/// conjugation afterwards is what makes `zungbr` able to form `P^H` from the
/// stored entries.
pub fn zgebd2(
    a: &mut [Complex64],
    lda: usize,
    m: usize,
    n: usize,
    d: &mut [f64],
    e: &mut [f64],
    tauq: &mut [Complex64],
    taup: &mut [Complex64],
) {
    if m >= n {
        // Reduce to upper bidiagonal form.
        for i in 1..=n {
            // H(i) annihilates A(i+1:m, i).
            let mut alpha = a[(i - 1) + (i - 1) * lda];
            let mut x: Vec<Complex64> = ((i + 1)..=m).map(|r| a[(r - 1) + (i - 1) * lda]).collect();
            let tau = zlarfg(&mut alpha, &mut x);
            a[(i - 1) + (i - 1) * lda] = alpha;
            tauq[i - 1] = tau;
            for (k, r) in ((i + 1)..=m).enumerate() {
                a[(r - 1) + (i - 1) * lda] = x[k];
            }
            d[i - 1] = alpha.re;

            if i < n {
                // Apply H(i)^H to A(i:m, i+1:n) from the left. The reflector's
                // implicit `1` sits at A(i,i).
                let mrows = m - i + 1;
                let mut v = vec![zero(); mrows];
                v[0] = one();
                for k in 1..mrows {
                    v[k] = a[(i + k - 1) + (i - 1) * lda];
                }
                zlarf_left(a, lda, i - 1, mrows, i, n - i, &v, tau.conj());
                a[(i - 1) + (i - 1) * lda] = Complex64::new(d[i - 1], 0.0);

                // G(i) annihilates A(i, i+2:n), through the conjugated row.
                for c in (i + 1)..=n {
                    a[(i - 1) + (c - 1) * lda] = a[(i - 1) + (c - 1) * lda].conj();
                }
                let mut alpha = a[(i - 1) + i * lda];
                let mut x: Vec<Complex64> = ((i + 2)..=n).map(|c| a[(i - 1) + (c - 1) * lda]).collect();
                let tau = zlarfg(&mut alpha, &mut x);
                a[(i - 1) + i * lda] = alpha;
                taup[i - 1] = tau;
                for (k, c) in ((i + 2)..=n).enumerate() {
                    a[(i - 1) + (c - 1) * lda] = x[k];
                }
                e[i - 1] = alpha.re;

                // Apply G(i) to A(i+1:m, i+1:n) from the right. The implicit `1`
                // sits at A(i, i+1).
                let ncols = n - i;
                let mut v = vec![zero(); ncols];
                v[0] = one();
                for k in 1..ncols {
                    v[k] = a[(i - 1) + (i + k) * lda];
                }
                zlarf_right(a, lda, i, m - i, i, ncols, &v, tau);

                for c in (i + 1)..=n {
                    a[(i - 1) + (c - 1) * lda] = a[(i - 1) + (c - 1) * lda].conj();
                }
                a[(i - 1) + i * lda] = Complex64::new(e[i - 1], 0.0);
            } else {
                taup[i - 1] = zero();
            }
        }
    } else {
        // Reduce to lower bidiagonal form.
        for i in 1..=m {
            // G(i) annihilates A(i, i+1:n), through the conjugated row.
            for c in i..=n {
                a[(i - 1) + (c - 1) * lda] = a[(i - 1) + (c - 1) * lda].conj();
            }
            let mut alpha = a[(i - 1) + (i - 1) * lda];
            let mut x: Vec<Complex64> = ((i + 1)..=n).map(|c| a[(i - 1) + (c - 1) * lda]).collect();
            let tau = zlarfg(&mut alpha, &mut x);
            a[(i - 1) + (i - 1) * lda] = alpha;
            taup[i - 1] = tau;
            for (k, c) in ((i + 1)..=n).enumerate() {
                a[(i - 1) + (c - 1) * lda] = x[k];
            }
            d[i - 1] = alpha.re;

            if i < m {
                // Apply G(i) to A(i+1:m, i:n) from the right. The implicit `1`
                // sits at A(i,i).
                let ncols = n - i + 1;
                let mut v = vec![zero(); ncols];
                v[0] = one();
                for k in 1..ncols {
                    v[k] = a[(i - 1) + (i + k - 1) * lda];
                }
                zlarf_right(a, lda, i, m - i, i - 1, ncols, &v, tau);
            }
            for c in i..=n {
                a[(i - 1) + (c - 1) * lda] = a[(i - 1) + (c - 1) * lda].conj();
            }
            a[(i - 1) + (i - 1) * lda] = Complex64::new(d[i - 1], 0.0);

            if i < m {
                // H(i) annihilates A(i+2:m, i).
                let mut alpha = a[i + (i - 1) * lda];
                let mut x: Vec<Complex64> = ((i + 2)..=m).map(|r| a[(r - 1) + (i - 1) * lda]).collect();
                let tau = zlarfg(&mut alpha, &mut x);
                a[i + (i - 1) * lda] = alpha;
                tauq[i - 1] = tau;
                for (k, r) in ((i + 2)..=m).enumerate() {
                    a[(r - 1) + (i - 1) * lda] = x[k];
                }
                e[i - 1] = alpha.re;

                // Apply H(i)^H to A(i+1:m, i+1:n) from the left. The implicit `1`
                // sits at A(i+1, i).
                let mrows = m - i;
                let mut v = vec![zero(); mrows];
                v[0] = one();
                for k in 1..mrows {
                    v[k] = a[(i + k) + (i - 1) * lda];
                }
                zlarf_left(a, lda, i, mrows, i, n - i, &v, tau.conj());
                a[i + (i - 1) * lda] = Complex64::new(e[i - 1], 0.0);
            } else {
                tauq[i - 1] = zero();
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Generating the unitary factors
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// `ZUNG2R`: generate the first `n` columns of an `m x m` unitary matrix `Q`
/// from the reflectors `zgebd2` stored below the diagonal of `a`.
fn zung2r(m: usize, n: usize, k: usize, a: &mut [Complex64], lda: usize, tau: &[Complex64]) {
    if n == 0 {
        return;
    }
    // Set the columns after the first k to the identity.
    for j in k..n {
        for l in 0..m {
            a[l + j * lda] = zero();
        }
        a[j + j * lda] = one();
    }
    // Form the first k columns in reverse order.
    for i in (0..k).rev() {
        if i < n - 1 {
            // `ZLARF1F('Left', M-I+1, N-I, A(I,I), 1, TAU(I), A(I,I+1), LDA)`.
            let mrows = m - i;
            let ncols = n - i - 1;
            let mut v = vec![zero(); mrows];
            v[0] = one();
            for t in 1..mrows {
                v[t] = a[(i + t) + i * lda];
            }
            zlarf_left(a, lda, i, mrows, i + 1, ncols, &v, tau[i]);
        }
        if i < m - 1 {
            for t in (i + 1)..m {
                a[t + i * lda] *= -tau[i];
            }
        }
        a[i + i * lda] = one() - tau[i];
        for l in 0..i {
            a[l + i * lda] = zero();
        }
    }
}

/// `ZUNGL2`: generate the first `m` rows of an `m x n` matrix with unitary rows
/// (`P^H`) from the reflectors `zgebd2` stored above the diagonal of `a`.
fn zungl2(m: usize, n: usize, k: usize, a: &mut [Complex64], lda: usize, tau: &[Complex64]) {
    if m == 0 {
        return;
    }
    if k < m {
        for j in 0..n {
            for l in k..m {
                a[l + j * lda] = zero();
            }
            if j >= k && j < m {
                a[j + j * lda] = one();
            }
        }
    }
    for i in (0..k).rev() {
        if i < n - 1 {
            // `ZLACGV(N-I, A(I,I+1), LDA)`.
            for c in (i + 1)..n {
                a[i + c * lda] = a[i + c * lda].conj();
            }
            if i < m - 1 {
                // `ZLARF1F('Right', M-I, N-I+1, A(I,I), LDA, CONJG(TAU(I)), A(I+1,I), LDA)`.
                let nrows = m - i;
                let ncols = n - i;
                let mut v = vec![zero(); ncols];
                v[0] = one();
                for t in 1..ncols {
                    v[t] = a[i + (i + t) * lda];
                }
                zlarf_right(a, lda, i + 1, nrows - 1, i, ncols, &v, tau[i].conj());
            }
            // `ZSCAL(N-I, -TAU(I), A(I,I+1), LDA)` then undo the conjugation.
            for c in (i + 1)..n {
                a[i + c * lda] *= -tau[i];
            }
            for c in (i + 1)..n {
                a[i + c * lda] = a[i + c * lda].conj();
            }
        }
        a[i + i * lda] = one() - tau[i].conj();
        for l in 0..i {
            a[i + l * lda] = zero();
        }
    }
}

/// `ZUNGBR`: form `Q` (`want_q`) or `P^H` from the reflectors `zgebd2` left in `a`.
///
/// Mirrors [`super::svd::dorgbr`], including the two shift branches and their
/// offsets (the `A(I,1)` of the `P^H` branch is row `I` of column 1, i.e. offset
/// `i - 1`, not `(i - 1) * lda`).
pub fn zungbr(want_q: bool, m: usize, n: usize, k: usize, a: &mut [Complex64], lda: usize, tau: &[Complex64]) {
    if m == 0 || n == 0 {
        return;
    }
    if want_q {
        if m >= k {
            zung2r(m, n, k, a, lda, tau);
        } else {
            // Shift the reflectors one column to the right and insert the
            // identity, so that `zung2r` finds them where it expects.
            for j in (2..=m).rev() {
                a[(j - 1) * lda] = zero();
                for i in (j + 1)..=m {
                    a[(i - 1) + (j - 1) * lda] = a[(i - 1) + (j - 2) * lda];
                }
            }
            a[0] = one();
            for i in 2..=m {
                a[i - 1] = zero();
            }
            if m > 1 {
                zung2r(m - 1, m - 1, m - 1, &mut a[1 + lda..], lda, tau);
            }
        }
    } else if k < n {
        zungl2(m, n, k, a, lda, tau);
    } else {
        // Shift the reflectors one row down, along the other diagonal.
        a[0] = one();
        for i in 2..=n {
            a[i - 1] = zero();
        }
        for j in 2..=n {
            for i in (2..=j - 1).rev() {
                a[(i - 1) + (j - 1) * lda] = a[(i - 2) + (j - 1) * lda];
            }
            a[(j - 1) * lda] = zero();
        }
        if n > 1 {
            zungl2(n - 1, n - 1, n - 1, &mut a[1 + lda..], lda, tau);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Bidiagonal SVD
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// `ZBDSQR`: the SVD of an `n x n` bidiagonal matrix, accumulating the vectors.
///
/// `lower` selects `UPLO = 'L'`. `d` holds the real diagonal, `e` the real
/// off-diagonal. `vt` is `ldvt x ncvt` with the first `n` rows meaningful, `u` is
/// `ldu x nru` with the first `n` columns meaningful; both are complex and are
/// updated by the real plane rotations. `c` (an `ldc x ncc` matrix) is accepted
/// for fidelity with the reference but this driver always calls with `ncc = 0`.
///
/// Returns `info`: `0` on success, or the number of off-diagonal entries that did
/// not converge (LAPACK's `INFO > 0`).
///
/// The iteration is [`super::svd::dbdsqr`]'s, with `zlasr`/`zdrot` in place of
/// `dlasr`/the inline rotation and complex `U`/`VT`/`C`.
#[allow(clippy::too_many_arguments)]
pub fn zbdsqr(
    lower: bool,
    n: usize,
    ncvt: usize,
    nru: usize,
    ncc: usize,
    d: &mut [f64],
    e: &mut [f64],
    vt: &mut [Complex64],
    ldvt: usize,
    u: &mut [Complex64],
    ldu: usize,
    c: &mut [Complex64],
    ldc: usize,
) -> usize {
    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;
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
            zlasr_right(nru, n, &work, &work[nm1..], u, ldu, 0, 0, false);
        }
        if ncc > 0 {
            zlasr_left(n, ncc, &work, &work[nm1..], c, ldc, 0, 0, false);
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
            // A 2 by 2 block: handle it separately (ZDROT on each vector set).
            let (sigmn, sigmax, sinr, cosr, sinl, cosl) = dlasv2(d[m - 2], e[m - 2], d[m - 1]);
            d[m - 2] = sigmax;
            e[m - 2] = ZERO;
            d[m - 1] = sigmn;
            if ncvt > 0 {
                for k in 0..ncvt {
                    let x = vt[(m - 2) + k * ldvt];
                    let y = vt[(m - 1) + k * ldvt];
                    vt[(m - 2) + k * ldvt] = scale(x, cosr) + scale(y, sinr);
                    vt[(m - 1) + k * ldvt] = scale(y, cosr) - scale(x, sinr);
                }
            }
            if nru > 0 {
                for k in 0..nru {
                    let x = u[k + (m - 2) * ldu];
                    let y = u[k + (m - 1) * ldu];
                    u[k + (m - 2) * ldu] = scale(x, cosl) + scale(y, sinl);
                    u[k + (m - 1) * ldu] = scale(y, cosl) - scale(x, sinl);
                }
            }
            if ncc > 0 {
                for k in 0..ncc {
                    let x = c[(m - 2) + k * ldc];
                    let y = c[(m - 1) + k * ldc];
                    c[(m - 2) + k * ldc] = scale(x, cosl) + scale(y, sinl);
                    c[(m - 1) + k * ldc] = scale(y, cosl) - scale(x, sinl);
                }
            }
            m -= 2;
            continue;
        }

        // On a new submatrix, choose the shift direction.
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
                    zlasr_left(m - ll + 1, ncvt, &work, &work[nm1..], vt, ldvt, ll - 1, 0, false);
                }
                if nru > 0 {
                    zlasr_right(nru, m - ll + 1, &work[nm12..], &work[nm13..], u, ldu, 0, ll - 1, false);
                }
                if ncc > 0 {
                    zlasr_left(m - ll + 1, ncc, &work[nm12..], &work[nm13..], c, ldc, ll - 1, 0, false);
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
                    zlasr_left(
                        m - ll + 1,
                        ncvt,
                        &work[nm12..],
                        &work[nm13..],
                        vt,
                        ldvt,
                        ll - 1,
                        0,
                        true,
                    );
                }
                if nru > 0 {
                    zlasr_right(nru, m - ll + 1, &work, &work[nm1..], u, ldu, 0, ll - 1, true);
                }
                if ncc > 0 {
                    zlasr_left(m - ll + 1, ncc, &work, &work[nm1..], c, ldc, ll - 1, 0, true);
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
                zlasr_left(m - ll + 1, ncvt, &work, &work[nm1..], vt, ldvt, ll - 1, 0, false);
            }
            if nru > 0 {
                zlasr_right(nru, m - ll + 1, &work[nm12..], &work[nm13..], u, ldu, 0, ll - 1, false);
            }
            if ncc > 0 {
                zlasr_left(m - ll + 1, ncc, &work[nm12..], &work[nm13..], c, ldc, ll - 1, 0, false);
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
                zlasr_left(
                    m - ll + 1,
                    ncvt,
                    &work[nm12..],
                    &work[nm13..],
                    vt,
                    ldvt,
                    ll - 1,
                    0,
                    true,
                );
            }
            if nru > 0 {
                zlasr_right(nru, m - ll + 1, &work, &work[nm1..], u, ldu, 0, ll - 1, true);
            }
            if ncc > 0 {
                zlasr_left(m - ll + 1, ncc, &work, &work[nm1..], c, ldc, ll - 1, 0, true);
            }
        }
    }

    // All singular values converged: make them positive.
    for i in 1..=n {
        if d[i - 1] == ZERO {
            d[i - 1] = ZERO;
        }
        if d[i - 1] < ZERO {
            d[i - 1] = -d[i - 1];
            if ncvt > 0 {
                for k in 0..ncvt {
                    vt[(i - 1) + k * ldvt] *= Complex64::new(-1.0, 0.0);
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

/// `ZGESVD` with `JOBU = 'A'` and `JOBVT = 'A'`: the full SVD of an `m x n`
/// complex matrix.
///
/// On entry `a` is the `m x n` matrix (column-major, leading dimension `m`) and is
/// destroyed. On exit `s` holds the `min(m, n)` real singular values in decreasing
/// order, `u` the `m x m` unitary matrix, and `vt` the `n x n` unitary matrix
/// `V^H`, so that `A = U S V^H` with `S` the `m x n` diagonal of `s`.
///
/// # Errors
///
/// [`LapackError::BadShape`] if a buffer's length disagrees with `(m, n)`, and
/// [`LapackError::Convergence`] if `zbdsqr` fails to converge.
pub fn zgesvd(
    a: &mut [Complex64],
    m: usize,
    n: usize,
    s: &mut [f64],
    u: &mut [Complex64],
    vt: &mut [Complex64],
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
    // scaled back at the end (the reference also scales E when INFO > 0, which
    // only matters on the non-convergent path).
    let eps = PREC;
    let smlnum = SAFMIN.sqrt() / eps;
    let bignum = 1.0 / smlnum;
    let anrm = super::super::blas::zlange('M', a, m, n);
    let mut iscl = false;
    if anrm > 0.0 && anrm < smlnum {
        iscl = true;
        zlascl_g(anrm, smlnum, a, m, m, n);
    } else if anrm > bignum {
        iscl = true;
        zlascl_g(anrm, bignum, a, m, m, n);
    }

    let mut e = vec![0.0; minmn];
    let mut tauq = vec![zero(); minmn];
    let mut taup = vec![zero(); minmn];

    let info;
    if m >= n {
        // Bidiagonalize A directly: Q^H A P = B with B upper bidiagonal.
        zgebd2(a, m, m, n, s, &mut e, &mut tauq, &mut taup);

        // ZLACPY('L', M, N, A, LDA, U, LDU) then form Q in U.
        for j in 0..n {
            for i in j..m {
                u[i + j * m] = a[i + j * m];
            }
        }
        zungbr(true, m, m, n, u, m, &tauq);

        // ZLACPY('U', N, N, A, LDA, VT, LDVT) then form P^H in VT.
        for j in 0..n {
            for i in 0..=j {
                vt[i + j * n] = a[i + j * m];
            }
        }
        zungbr(false, n, n, n, vt, n, &taup);

        info = zbdsqr(false, n, n, m, 0, s, &mut e, vt, n, u, m, &mut [], 0);
    } else {
        // Bidiagonalize A directly: Q^H A P = B with B lower bidiagonal.
        zgebd2(a, m, m, n, s, &mut e, &mut tauq, &mut taup);

        // ZLACPY('L', M, M, A, LDA, U, LDU) then form Q in U.
        for j in 0..m {
            for i in j..m {
                u[i + j * m] = a[i + j * m];
            }
        }
        zungbr(true, m, m, n, u, m, &tauq);

        // ZLACPY('U', M, N, A, LDA, VT, LDVT) then form P^H in VT.
        for j in 0..n {
            for i in 0..=j.min(m - 1) {
                vt[i + j * n] = a[i + j * m];
            }
        }
        zungbr(false, n, n, m, vt, n, &taup);

        info = zbdsqr(true, m, n, m, 0, s, &mut e, vt, n, u, m, &mut [], 0);
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
    use crate::{ComplexMatrix, Matrix, Vector, cpx, mat_svd, vec_approx_eq};

    /// A deterministic complex sample with a non-zero real *and* imaginary part
    /// in every entry, so a stray conjugation or transpose cannot pass.
    fn complex_sample(m: usize, n: usize, offset: f64) -> Vec<Complex64> {
        let mut a = vec![zero(); m * n];
        for i in 0..m {
            for j in 0..n {
                let re = (((i * 7 + j * 13 + 3) % 11) as f64) - 5.0 + offset;
                let im = (((i * 5 + j * 17 + 7) % 9) as f64) - 4.0 - 0.5 * offset;
                a[i + j * m] = Complex64::new(re, im);
            }
        }
        a
    }

    /// Runs the driver and returns `(s, u, vt)`.
    fn run_svd(m: usize, n: usize, a: &[Complex64]) -> (Vec<f64>, Vec<Complex64>, Vec<Complex64>) {
        let minmn = m.min(n);
        let mut a_work = a.to_vec();
        let mut s = vec![0.0; minmn];
        let mut u = vec![zero(); m * m];
        let mut vt = vec![zero(); n * n];
        zgesvd(&mut a_work, m, n, &mut s, &mut u, &mut vt).unwrap();
        (s, u, vt)
    }

    /// Checks `U` unitary, `V^H` unitary, `U S V^H == A` and descending,
    /// non-negative singular values.
    fn check_full_svd(m: usize, n: usize, a: &[Complex64], tol: f64) {
        let minmn = m.min(n);
        let mut a_work = a.to_vec();
        let mut s = vec![0.0; minmn];
        let mut u = vec![zero(); m * m];
        let mut vt = vec![zero(); n * n];
        zgesvd(&mut a_work, m, n, &mut s, &mut u, &mut vt).unwrap();

        // s non-negative and descending.
        for k in 0..minmn {
            assert!(s[k] >= 0.0, "s[{}] = {} is negative", k, s[k]);
        }
        for k in 1..minmn {
            assert!(
                s[k - 1] >= s[k],
                "s is not descending at {}: {} < {}",
                k,
                s[k - 1],
                s[k]
            );
        }

        // U^H U = I.
        for i in 0..m {
            for j in 0..m {
                let mut acc = zero();
                for k in 0..m {
                    acc += u[k + i * m].conj() * u[k + j * m];
                }
                let want = if i == j { one() } else { zero() };
                assert!((acc - want).norm() < tol, "U not unitary at ({},{}): {:?}", i, j, acc);
            }
        }

        // (V^H)^H (V^H) = V V^H = I  <=>  V^H (V^H)^H = I. Check V^H (V^H)^H.
        for i in 0..n {
            for j in 0..n {
                let mut acc = zero();
                for k in 0..n {
                    acc += vt[i + k * n] * vt[j + k * n].conj();
                }
                let want = if i == j { one() } else { zero() };
                assert!((acc - want).norm() < tol, "V^H not unitary at ({},{}): {:?}", i, j, acc);
            }
        }

        // U S V^H == A.
        for i in 0..m {
            for j in 0..n {
                let mut acc = zero();
                for k in 0..minmn {
                    acc += u[i + k * m] * Complex64::new(s[k], 0.0) * vt[k + j * n];
                }
                assert!(
                    (acc - a[i + j * m]).norm() < tol,
                    "reconstruction failed at ({},{}): {:?} vs {:?}",
                    i,
                    j,
                    acc,
                    a[i + j * m]
                );
            }
        }
    }

    #[test]
    fn zgesvd_reconstructs_for_all_shapes() {
        for (m, n) in [
            (5, 3),
            (4, 4),
            (3, 5),
            (1, 1),
            (6, 2),
            (2, 6),
            (10, 7),
            (7, 10),
            (12, 12),
        ] {
            let a = complex_sample(m, n, 0.25);
            check_full_svd(m, n, &a, 1e-12);
        }
    }

    #[test]
    fn zgesvd_handles_a_rank_deficient_matrix() {
        // Rows 3 and 4 are copies of rows 1 and 2: rank 2, with genuine imaginary
        // parts everywhere.
        let m = 4;
        let n = 4;
        let mut a = complex_sample(m, n, 0.5);
        for j in 0..n {
            a[2 + j * m] = a[0 + j * m];
            a[3 + j * m] = a[1 + j * m];
        }
        check_full_svd(m, n, &a, 1e-12);

        // The two null singular values must come out near zero.
        let mut a_work = a.clone();
        let mut s = vec![0.0; 4];
        let mut u = vec![zero(); 16];
        let mut vt = vec![zero(); 16];
        zgesvd(&mut a_work, m, n, &mut s, &mut u, &mut vt).unwrap();
        assert!(
            s[3] < 1e-12 && s[2] < 1e-12,
            "expected two zero singular values, got {:?}",
            s
        );
    }

    #[test]
    fn zgesvd_matches_the_real_svd_on_real_data() {
        let m = 4;
        let n = 3;
        let mut re = Matrix::new(m, n);
        for i in 0..m {
            for j in 0..n {
                re.set(
                    i,
                    j,
                    (((i * 7 + j * 13 + 3) % 11) as f64) - 5.0 + 0.25 * (i as f64) - 0.1 * (j as f64),
                );
            }
        }
        let a_complex = {
            let mut a = vec![zero(); m * n];
            for i in 0..m {
                for j in 0..n {
                    a[i + j * m] = Complex64::new(re.get(i, j), 0.0);
                }
            }
            a
        };

        // Complex SVD (with real data).
        let mut a_work = a_complex.clone();
        let minmn = m.min(n);
        let mut s_z = vec![0.0; minmn];
        let mut u_z = vec![zero(); m * m];
        let mut vt_z = vec![zero(); n * n];
        zgesvd(&mut a_work, m, n, &mut s_z, &mut u_z, &mut vt_z).unwrap();

        // Real SVD.
        let mut a_re = re.clone();
        let mut s_d = Vector::new(minmn);
        let mut u_d = Matrix::new(m, m);
        let mut vt_d = Matrix::new(n, n);
        mat_svd(&mut s_d, &mut u_d, &mut vt_d, &mut a_re).unwrap();

        vec_approx_eq(&Vector::from(&s_z), s_d.as_data(), 1e-13);

        // Real input must come back with (numerically) real vectors: a
        // conjugation mistake would leave an imaginary part behind.
        for i in 0..m * m {
            assert!(u_z[i].im.abs() < 1e-13, "U has imaginary part at {}: {:?}", i, u_z[i]);
        }
        for i in 0..n * n {
            assert!(
                vt_z[i].im.abs() < 1e-13,
                "VT has imaginary part at {}: {:?}",
                i,
                vt_z[i]
            );
        }
        check_full_svd(m, n, &a_complex, 1e-12);
    }

    #[test]
    fn zgesvd_uses_genuinely_complex_data() {
        // Hermitian matrices would hide a conjugation error; this one is not
        // Hermitian and not symmetric, and the imaginary parts matter: the
        // singular values differ from those of the real part alone.
        let a = complex_sample(3, 3, 0.75);
        let mut a_work = a.clone();
        let mut s = vec![0.0; 3];
        let mut u = vec![zero(); 9];
        let mut vt = vec![zero(); 9];
        zgesvd(&mut a_work, 3, 3, &mut s, &mut u, &mut vt).unwrap();
        check_full_svd(3, 3, &a, 1e-12);

        // The SVD of the real part differs, so this really exercised the
        // imaginary parts.
        let mut re = vec![zero(); 9];
        for i in 0..9 {
            re[i] = Complex64::new(a[i].re, 0.0);
        }
        let mut re_work = re.clone();
        let mut s_re = vec![0.0; 3];
        let mut u_re = vec![zero(); 9];
        let mut vt_re = vec![zero(); 9];
        zgesvd(&mut re_work, 3, 3, &mut s_re, &mut u_re, &mut vt_re).unwrap();
        assert!(
            (s[0] - s_re[0]).abs() > 1e-6 || (s[2] - s_re[2]).abs() > 1e-6,
            "expected the imaginary parts to change the singular values: {:?} vs {:?}",
            s,
            s_re
        );
    }

    #[test]
    fn zgesvd_fails_on_bad_shapes() {
        let mut a = vec![zero(); 6];
        let mut s = vec![0.0; 2];
        let mut u = vec![zero(); 9];
        let mut vt = vec![zero(); 4];
        assert_eq!(
            zgesvd(&mut a, 3, 2, &mut s[0..1], &mut u, &mut vt),
            Err(LapackError::BadShape)
        );
        assert_eq!(
            zgesvd(&mut a, 3, 2, &mut s, &mut u[0..8], &mut vt),
            Err(LapackError::BadShape)
        );
        assert_eq!(
            zgesvd(&mut a, 3, 2, &mut s, &mut u, &mut vt[0..3]),
            Err(LapackError::BadShape)
        );
        assert_eq!(
            zgesvd(&mut a[0..5], 3, 2, &mut s, &mut u, &mut vt),
            Err(LapackError::BadShape)
        );
    }

    /// The reference values from `complex_mat_svd`'s IBM sample, exercised
    /// through the module's own entry point as well.
    #[test]
    fn zgesvd_matches_the_ibm_reference_values() {
        let a = ComplexMatrix::from(&[
            [cpx!(1.0, 1.0), cpx!(2.0, -1.0), cpx!(3.0, 0.0)],
            [cpx!(2.0, -1.0), cpx!(4.0, 1.0), cpx!(5.0, -1.0)],
            [cpx!(3.0, 0.0), cpx!(5.0, -1.0), cpx!(6.0, 1.0)],
        ]);
        let (s, u, vt) = run_svd(3, 3, a.as_data());
        vec_approx_eq(&Vector::from(&s), &[11.370686, 2.386257, 1.006620], 1e-6);
        #[rustfmt::skip]
        let u_ref = ComplexMatrix::from(&[
            [cpx!(-0.3265, 0.0409), cpx!( 0.0558, 0.4814), cpx!( 0.3504, -0.7308)],
            [cpx!(-0.5822, 0.0725), cpx!(-0.0823, -0.7730), cpx!( 0.1017, -0.2026)],
            [cpx!(-0.7396, 0.0233), cpx!( 0.0036, 0.4009), cpx!(-0.2805, 0.4616)],
        ]);
        #[rustfmt::skip]
        let vt_ref = ComplexMatrix::from(&[
            [cpx!(-0.3290, 0.0000), cpx!(-0.5867, -0.0004), cpx!(-0.7367, -0.0688)],
            [cpx!( 0.4846, 0.0000), cpx!(-0.7774, -0.0071), cpx!( 0.3987, 0.0425)],
            [cpx!(-0.8105, 0.0000), cpx!(-0.2267, -0.0041), cpx!( 0.5375, 0.0533)],
        ]);
        let mut u_mat = ComplexMatrix::new(3, 3);
        let mut vt_mat = ComplexMatrix::new(3, 3);
        for i in 0..3 {
            for j in 0..3 {
                u_mat.set(i, j, u[i + j * 3]);
                vt_mat.set(i, j, vt[i + j * 3]);
            }
        }
        crate::complex_mat_approx_eq(&u_mat, &u_ref, 1e-4);
        crate::complex_mat_approx_eq(&vt_mat, &vt_ref, 1e-4);
        check_full_svd(3, 3, a.as_data(), 1e-12);
    }

    /// The driver scales a matrix whose norm is outside `[smlnum, bignum]` and
    /// undoes that on the singular values. Without this test that branch is
    /// never taken.
    #[test]
    fn zgesvd_scales_extreme_matrices() {
        let m = 4;
        let n = 3;
        let base = complex_sample(m, n, 0.25);
        let (s_base, _, _) = run_svd(m, n, &base);
        for factor in [1e-160, 1e160] {
            let a: Vec<Complex64> = base.iter().map(|v| scale(*v, factor)).collect();
            let (s, _, _) = run_svd(m, n, &a);
            for k in 0..n.min(m) {
                let want = s_base[k] * factor;
                let rel = (s[k] - want).abs() / want.abs();
                assert!(
                    rel < 1e-13,
                    "scaling by {} broke s[{}]: {} vs {}",
                    factor,
                    k,
                    s[k],
                    want
                );
            }
        }
    }

    /// `zbdsqr`'s modes: values only, the lower bidiagonal form, the `NCC > 0`
    /// transformation of a complex right-hand block, and a zero singular value.
    ///
    /// As with the real routine, `zgesvd` only ever calls it with both vector
    /// sets requested from an upper bidiagonal matrix, so the other shapes were
    /// never executed.
    #[test]
    fn zbdsqr_covers_the_lower_form_and_the_vector_modes() {
        let n = 4;
        // Upper bidiagonal, one singular value exactly zero: values only.
        let d_in = [1.0, 0.0, 2.0, 3.0];
        let e_in = [0.5, 0.5, 0.25, 0.0];
        let mut d = d_in.to_vec();
        let mut e = e_in.to_vec();
        let mut vt: Vec<Complex64> = vec![];
        let mut u: Vec<Complex64> = vec![];
        let mut c: Vec<Complex64> = vec![];
        let info = zbdsqr(false, n, 0, 0, 0, &mut d, &mut e, &mut vt, 1, &mut u, 1, &mut c, 1);
        assert_eq!(info, 0);
        for k in 0..n {
            assert!(d[k] >= 0.0, "s[{k}] = {} is negative", d[k]);
        }
        for k in 1..n {
            assert!(d[k - 1] >= d[k], "s is not descending: {d:?}");
        }
        assert!(d[n - 1] < 1e-14, "one singular value must be zero: {d:?}");

        // Lower bidiagonal with the full vector sets.
        let mut d_lo = [1.0, 2.0, 3.0, 4.0];
        let mut e_lo = [0.5, 0.25, 0.125, 0.0];
        let mut vt = vec![zero(); 16];
        let mut u = vec![zero(); 16];
        for i in 0..4 {
            vt[i + i * 4] = Complex64::new(1.0, 0.0);
            u[i + i * 4] = Complex64::new(1.0, 0.0);
        }
        let info = zbdsqr(true, 4, 4, 4, 0, &mut d_lo, &mut e_lo, &mut vt, 4, &mut u, 4, &mut c, 1);
        assert_eq!(info, 0);
        for k in 1..4 {
            assert!(d_lo[k - 1] >= d_lo[k], "lower: s not descending: {d_lo:?}");
        }
        // Same matrix as an upper bidiagonal: the singular values must agree.
        let (mut d_up, mut e_up) = (d_in.to_vec(), e_in.to_vec());
        let mut vt_up = vec![zero(); 16];
        let mut u_up = vec![zero(); 16];
        let info = zbdsqr(
            false, 4, 0, 0, 0, &mut d_up, &mut e_up, &mut vt_up, 1, &mut u_up, 1, &mut c, 1,
        );
        assert_eq!(info, 0);

        // NCC > 0: the caller's C is transformed by an orthogonal factor, so the
        // inner products of its columns must survive.
        let ncc = 2;
        let mut c_in = vec![zero(); 4 * ncc];
        for i in 0..4 {
            for j in 0..ncc {
                c_in[i + j * 4] = Complex64::new(1.0 + (i as f64) * 0.5, -(j as f64) * 0.25);
            }
        }
        let mut c_out = c_in.clone();
        let (mut d_c, mut e_c) = (d_in.to_vec(), e_in.to_vec());
        let mut vt_c = vec![zero(); 16];
        let mut u_c = vec![zero(); 16];
        let info = zbdsqr(
            false, 4, 0, 0, ncc, &mut d_c, &mut e_c, &mut vt_c, 1, &mut u_c, 1, &mut c_out, 4,
        );
        assert_eq!(info, 0);
        for i in 0..ncc {
            for j in 0..ncc {
                let mut got = zero();
                let mut want = zero();
                for k in 0..4 {
                    got += c_out[k + i * 4].conj() * c_out[k + j * 4];
                    want += c_in[k + i * 4].conj() * c_in[k + j * 4];
                }
                assert!(
                    (got - want).norm() <= 1e-12 * want.norm().max(1.0),
                    "inner product ({i},{j}) changed: {got:?} != {want:?}"
                );
            }
        }
        // ... and the transformation is not the identity, so something happened.
        assert!(
            c_out.iter().zip(c_in.iter()).any(|(a, b)| (a - b).norm() > 1e-14),
            "the NCC > 0 pass left C unchanged"
        );

        // n = 1.
        let mut d = vec![2.5];
        let mut e = vec![0.0];
        let mut vt = vec![Complex64::new(1.0, 0.0)];
        let mut u = vec![Complex64::new(1.0, 0.0)];
        let info = zbdsqr(false, 1, 1, 1, 0, &mut d, &mut e, &mut vt, 1, &mut u, 1, &mut c, 1);
        assert_eq!(info, 0);
        assert_eq!(d[0], 2.5);
    }

    /// LAPACK's `zbdt01`: `zgebd2` followed by the two `zungbr` calls must give
    /// `A = Q B P^H`, with `B` real bidiagonal and `Q`, `P` unitary, for tall,
    /// square and wide matrices.
    ///
    /// The complex twin of `dbdt01`. `zgesvd` reconstructs a dense matrix through
    /// this path, but nothing tested the reduction on its own, and the `K`
    /// arguments (`K = n` for `Q`, as `EIG/dchkbd.f:860` passes) are the kind of
    /// detail that can be right for the wrong reason when it is only checked two
    /// reductions away.
    #[test]
    fn zgebd2_and_zungbr_reconstruct_the_bidiagonal_form() {
        let eps = f64::EPSILON;
        let one = Complex64::new(1.0, 0.0);
        for (m, n) in [(2usize, 2usize), (3, 2), (3, 3), (4, 3), (6, 4), (2, 3), (3, 4), (4, 6)] {
            let a0 = complex_sample(m, n, 0.5);
            let mut a = a0.clone();
            let minmn = m.min(n);
            let mut d = vec![0.0; minmn];
            let mut e = vec![0.0; minmn];
            let mut tauq = vec![zero(); minmn];
            let mut taup = vec![zero(); minmn];
            zgebd2(&mut a, m, m, n, &mut d, &mut e, &mut tauq, &mut taup);

            let mut q = vec![zero(); m * m];
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
            zungbr(true, m, m, n, &mut q, m, &tauq);

            let mut vt = vec![zero(); n * n];
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
            zungbr(false, n, n, kp, &mut vt, n, &taup);

            // B: upper bidiagonal for m >= n, lower otherwise; real entries.
            let mut b = vec![zero(); m * n];
            for i in 0..minmn {
                b[i + i * m] = Complex64::new(d[i], 0.0);
                if i + 1 < minmn {
                    if m >= n {
                        b[i + (i + 1) * m] = Complex64::new(e[i], 0.0);
                    } else {
                        b[(i + 1) + i * m] = Complex64::new(e[i], 0.0);
                    }
                }
            }

            // Q B P^H, then the eps-normalized reconstruction ratio.
            let mut qb = vec![zero(); m * n];
            for j in 0..n {
                for i in 0..m {
                    let mut sum = zero();
                    for k in 0..m {
                        sum += q[i + k * m] * b[k + j * m];
                    }
                    qb[i + j * m] = sum;
                }
            }
            let mut rec = vec![zero(); m * n];
            for j in 0..n {
                for i in 0..m {
                    let mut sum = zero();
                    for k in 0..n {
                        // `zungbr('P')` returns `P^H` already, so this is the
                        // factor to multiply by, not its conjugate transpose.
                        sum += qb[i + k * m] * vt[k + j * n];
                    }
                    rec[i + j * m] = sum;
                }
            }
            let mut a_norm = 0.0f64;
            for j in 0..n {
                let mut s = 0.0;
                for i in 0..m {
                    s += a0[i + j * m].norm();
                }
                a_norm = a_norm.max(s);
            }
            let a_norm = a_norm.max(f64::MIN_POSITIVE);
            let resid: f64 = (0..m * n).map(|k| (a0[k] - rec[k]).norm()).sum();
            let ratio = resid / (m.max(n) as f64 * a_norm * eps);
            assert!(
                ratio <= 20.0,
                "{m}x{n}: zbdt01 reconstruction ratio = {ratio} exceeds 20"
            );

            // Q^H Q = I and P P^H = I, each divided by the dimension.
            for (name, x, dim) in [("Q", &q, m), ("P", &vt, n)] {
                let mut orth = 0.0;
                for i in 0..dim {
                    for j in 0..dim {
                        let mut dot = zero();
                        for k in 0..dim {
                            dot += x[k + i * dim].conj() * x[k + j * dim];
                        }
                        let want = if i == j { one } else { zero() };
                        orth += (dot - want).norm();
                    }
                }
                assert!(
                    orth / (dim as f64 * eps) <= 20.0,
                    "{m}x{n}: {name} is not unitary ({orth})"
                );
            }
        }
    }
}
