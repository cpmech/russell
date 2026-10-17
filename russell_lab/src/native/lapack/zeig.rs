//! Native complex Hermitian eigen-decomposition: `zheev` and the unblocked
//! pieces it is built from.
//!
//! # The pipeline
//!
//! | Step | LAPACK | What it does |
//! |---|---|---|
//! | 1 | `zhetd2` | unitary reduction to Hermitian tridiagonal form, `T = Q^H A Q` |
//! | 2 | `zungtr` | forms `Q` explicitly from the reflectors `zhetd2` left behind |
//! | 3 | `zsteqr` | implicit QL/QR on the tridiagonal, accumulating `Q Y` |
//!
//! This is `zheev`'s unblocked route, exactly as [`super::sym`] is `dsyev`'s:
//! `zhetrd` reduces to `zhetd2` for a single block, and `zungql`/`zungqr` reduce
//! to `zung2l`/`zung2r`. `zheev` scales the matrix first when its norm falls
//! outside the safe range, as the reference does, and undoes that scaling on the
//! eigenvalues at the end. Eigenvalues come out in ascending order and the
//! eigenvectors overwrite the input matrix as columns.
//!
//! Everything is column-major: element `(i, j)` of an `n x n` matrix is
//! `data[i + j * n]`.
//!
//! # Only the named triangle is touched
//!
//! `zhetd2`'s `uplo` selects which triangle of `a` is *read*; the other one may
//! hold anything. The Hermitian matrix-vector product and the rank-2 update in
//! `zhetd2` therefore use the stored triangle alone and reconstruct the
//! missing half by conjugation, which is exactly what `zhemv` and `zher2` do.
//! A routine that read the wrong half would still look plausible on a real
//! matrix, which is why the parity test uses genuinely complex data.
//!
//! # The `1` in a Householder vector
//!
//! `zlarfg` returns `v` with `v[0] == 1` and the tail in the entries after it,
//! and each driver decides where that `1` sits in the matrix. As in the real
//! code, `zung2l` builds its vectors with the `1` last and `zung2r` with it
//! first; the shared application is `zlarf_left`, which the caller feeds the
//! explicit vector.

use super::LapackError;
use super::aux::{dlae2, dlaev2, dlapy2, dlapy3, zladiv, zlarf_left, zlarf_right, zlascl_g};
use super::blas::dznrm2;
use super::eig::dlartg;
use crate::Complex64;
use num_complex::ComplexFloat;

/// The zero element, named so the arithmetic reads like the Fortran `ZERO`.
fn zero() -> Complex64 {
    Complex64::new(0.0, 0.0)
}

/// The one element, named so the arithmetic reads like the Fortran `ONE`.
fn one() -> Complex64 {
    Complex64::new(1.0, 0.0)
}

/// A complex number built from real and imaginary parts, matching `DCMPLX`.
fn cpx(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

/// `zheev`: all eigenvalues and eigenvectors of a complex Hermitian matrix.
///
/// `a` is `n x n` column-major and is **destroyed**: on exit it holds the
/// orthonormal eigenvectors as columns. `w` receives the eigenvalues in
/// ascending order. `upper` selects which triangle of `a` is *read* on entry
/// (`'U'` or `'L'`); the other triangle may hold anything, though the output
/// overwrites the whole matrix.
///
/// The contract is `zheev(JOBZ='V', ...)`: eigenvectors are always computed,
/// because that is all `russell_lab`'s caller asks for.
///
/// # Errors
///
/// [`LapackError::BadShape`] if `a.len() != n * n` or `w.len() != n`.
pub fn zheev(a: &mut [Complex64], w: &mut [f64], n: usize, upper: bool) -> Result<(), LapackError> {
    if a.len() != n * n || w.len() != n {
        return Err(LapackError::BadShape);
    }
    if n == 0 {
        return Ok(());
    }
    if n == 1 {
        w[0] = a[0].re;
        a[0] = one();
        return Ok(());
    }

    // Machine constants, named as in the reference. `DLAMCH('E')` is half the
    // spacing, not `f64::EPSILON`.
    let safmin = f64::MIN_POSITIVE;
    let eps = f64::EPSILON / 2.0;
    let smlnum = safmin / eps;
    let bignum = 1.0 / smlnum;
    let rmin = smlnum.sqrt();
    let rmax = bignum.sqrt();

    // `ZLANHE('M', UPLO, ...)`: largest absolute entry of the stored triangle,
    // with the diagonal treated as real (`DBLE(A(J,J))`).
    let mut anrm = 0.0f64;
    for j in 0..n {
        let (i0, i1) = if upper { (0, j) } else { (j + 1, n) };
        for i in i0..i1 {
            let t = a[i + j * n].norm();
            if t > anrm {
                anrm = t;
            }
        }
        let t = a[j + j * n].re.abs();
        if t > anrm {
            anrm = t;
        }
    }

    let mut iscale = false;
    let mut sigma = 1.0f64;
    if anrm > 0.0 && anrm < rmin {
        iscale = true;
        sigma = rmin / anrm;
    } else if anrm > rmax {
        iscale = true;
        sigma = rmax / anrm;
    }
    if iscale {
        zlascl_tri(1.0, sigma, a, n, upper);
    }

    let mut d = vec![0.0; n];
    let mut e = vec![0.0; n - 1];
    let mut tau = vec![zero(); n - 1];
    zhetd2(upper, n, a, &mut d, &mut e, &mut tau);
    zungtr(upper, n, a, n, &tau);

    // `ZSTEQR('V', ...)`: `a` enters holding Q and leaves holding Q*Y, the
    // eigenvectors of A.
    let mut work = vec![0.0; 2 * n - 2];
    let info = zsteqr(1, n, &mut d, &mut e, a, n, &mut work);
    // `INFO > 0`: the QL/QR iteration did not converge. As in `dsyev`, these are
    // partly converged values, not a decomposition -- and as there, no test
    // reaches the branch (a NaN matrix takes `zsteqr`'s norm-gated early return).
    if info != 0 {
        return Err(LapackError::Convergence);
    }

    if iscale {
        let factor = 1.0 / sigma;
        for v in d.iter_mut() {
            *v *= factor;
        }
    }
    w.copy_from_slice(&d);
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// `zhetd2`: reduction of a complex Hermitian matrix to tridiagonal form,
/// `T = Q^H A Q`.
///
/// Reads and writes only the triangle named by `upper`. On exit the diagonal of
/// `T` is in `d`, its off-diagonal in `e`, and the reflectors that build `Q` are
/// stored in the matrix: for `'U'` the vector of `H(i)` fills `a(0:i-1, i+1)`,
/// for `'L'` it fills `a(i+2:n, i)`.
///
/// The two rank-2 updates are `zhemv` and `zher2` written out, because the whole
/// computation is a single column at a time and a general level-2 kernel would
/// hide the conjugation that is the only place this routine can go wrong.
fn zhetd2(upper: bool, n: usize, a: &mut [Complex64], d: &mut [f64], e: &mut [f64], tau: &mut [Complex64]) {
    if n == 0 {
        return;
    }
    if upper {
        // Fortran: A(N,N) = DBLE(A(N,N)); DO I = N-1, 1, -1
        a[(n - 1) + (n - 1) * n] = cpx(a[(n - 1) + (n - 1) * n].re, 0.0);
        for i in (0..n - 1).rev() {
            let len = i + 1; // Fortran I
            let mut alpha = a[i + (i + 1) * n];
            let mut x: Vec<Complex64> = (0..i).map(|t| a[t + (i + 1) * n]).collect();
            let taui = zlarfg(&mut alpha, &mut x);
            for (t, &v) in x.iter().enumerate() {
                a[t + (i + 1) * n] = v;
            }
            e[i] = alpha.re; // E(I) = DBLE(ALPHA), ALPHA now holds BETA

            if taui != zero() {
                // Apply H(i) from both sides to A(1:i,1:i). The reflector vector
                // is [tail, 1]: the `1` sits at row i of column i+1.
                a[i + (i + 1) * n] = one();
                let v: Vec<Complex64> = (0..len).map(|t| a[t + (i + 1) * n]).collect();

                // x := tau * A * v, stored aside (the reference reuses TAU here).
                let mut xw = zhemv(upper, len, taui, a, n, 0, 0, &v);

                // w := x - 1/2 * tau * (x^H * v) * v
                let mut z = zero();
                for t in 0..len {
                    z += xw[t].conj() * v[t];
                }
                let alpha2 = -0.5 * taui * z;
                for t in 0..len {
                    xw[t] += alpha2 * v[t];
                }

                // A := A - v*w^H - w*v^H, upper triangle only.
                zher2(true, len, -one(), &v, &xw, a, n, 0);
            } else {
                a[i + i * n] = cpx(a[i + i * n].re, 0.0);
            }
            a[i + (i + 1) * n] = cpx(e[i], 0.0);
            d[i + 1] = a[(i + 1) + (i + 1) * n].re;
            tau[i] = taui;
        }
        d[0] = a[0].re;
    } else {
        // Fortran: A(1,1) = DBLE(A(1,1)); DO I = 1, N-1
        a[0] = cpx(a[0].re, 0.0);
        for i in 0..n - 1 {
            let len = n - i - 1; // Fortran N-I
            let mut alpha = a[(i + 1) + i * n];
            let mut x: Vec<Complex64> = ((i + 2)..n).map(|t| a[t + i * n]).collect();
            let taui = zlarfg(&mut alpha, &mut x);
            for (k, &v) in x.iter().enumerate() {
                a[(i + 2 + k) + i * n] = v;
            }
            e[i] = alpha.re;

            if taui != zero() {
                // The reflector vector is [1, tail]: the `1` sits at row i+1.
                a[(i + 1) + i * n] = one();
                let v: Vec<Complex64> = ((i + 1)..n).map(|t| a[t + i * n]).collect();

                let mut xw = zhemv(upper, len, taui, a, n, i + 1, i + 1, &v);

                let mut z = zero();
                for t in 0..len {
                    z += xw[t].conj() * v[t];
                }
                let alpha2 = -0.5 * taui * z;
                for t in 0..len {
                    xw[t] += alpha2 * v[t];
                }

                // A := A - v*w^H - w*v^H, lower triangle only.
                zher2(false, len, -one(), &v, &xw, a, n, i + 1);
            } else {
                a[(i + 1) + (i + 1) * n] = cpx(a[(i + 1) + (i + 1) * n].re, 0.0);
            }
            a[(i + 1) + i * n] = cpx(e[i], 0.0);
            d[i] = a[i + i * n].re;
            tau[i] = taui;
        }
        d[n - 1] = a[(n - 1) + (n - 1) * n].re;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// `zungtr`: forms the unitary matrix `Q` from the reflectors `zhetd2` stored.
///
/// For `'U'`, `Q = H(n-1) ... H(2) H(1)`; for `'L'`, `Q = H(1) H(2) ... H(n-1)`.
/// The reflectors are first shifted one column so the `1` of each lies on the
/// diagonal of the block that is generated, and the last row/column (or first)
/// is set to the corresponding unit-vector entries as the reference does.
///
/// `lda` is the leading dimension of `a`, which is `n` everywhere here but is
/// kept because `zung2l`/`zung2r` are also called on a sub-block of it.
fn zungtr(upper: bool, n: usize, a: &mut [Complex64], lda: usize, tau: &[Complex64]) {
    if n == 0 {
        return;
    }
    if upper {
        for j in 0..n - 1 {
            for i in 0..j {
                a[i + j * lda] = a[i + (j + 1) * lda];
            }
            a[(n - 1) + j * lda] = zero();
        }
        for i in 0..n - 1 {
            a[i + (n - 1) * lda] = zero();
        }
        a[(n - 1) + (n - 1) * lda] = one();
        // `ZUNGQL` on the leading (n-1)x(n-1) block; unblocked it is `ZUNG2L`.
        zung2l(n - 1, n - 1, n - 1, a, lda, tau);
    } else {
        for j in (1..n).rev() {
            a[j * lda] = zero();
            for i in (j + 1)..n {
                a[i + j * lda] = a[i + (j - 1) * lda];
            }
        }
        a[0] = one();
        for i in 1..n {
            a[i] = zero();
        }
        // `ZUNGQR` on the trailing (n-1)x(n-1) block; unblocked it is `ZUNG2R`.
        zung2r(n - 1, n - 1, n - 1, a, lda, tau, 1, 1);
    }
}

/// `zung2l`: the unblocked `ZUNGQL`. Generates the first `k` columns of
/// `Q = H(k) ... H(1)` in `A(0:m, 0:ncols)`.
///
/// The reflector `H(i)` is taken from column `ncols-k+i`; its implicit `1` is the
/// element on the diagonal at row `m-ncols+i` (the reference's `A(M-N+II, II)`),
/// so the explicit entries are the rows above it. That is `ZLARF1L`'s convention.
fn zung2l(m: usize, ncols: usize, k: usize, a: &mut [Complex64], lda: usize, tau: &[Complex64]) {
    if ncols == 0 {
        return;
    }
    for j in 0..(ncols - k) {
        for l in 0..m {
            a[l + j * lda] = zero();
        }
        a[(m - ncols + j) + j * lda] = one();
    }
    for i in 0..k {
        let ii = ncols - k + i;
        let mrows = m - ncols + ii + 1; // Fortran M-N+II
        if mrows == 0 {
            continue;
        }
        // `ZLARF1L` convention: the `1` is the *last* entry of v.
        a[(mrows - 1) + ii * lda] = one();
        let mut v = vec![zero(); mrows];
        for t in 0..mrows - 1 {
            v[t] = a[t + ii * lda];
        }
        v[mrows - 1] = one();
        zlarf_left(a, lda, 0, mrows, 0, ii, &v, tau[i]);
        for t in 0..mrows - 1 {
            a[t + ii * lda] *= -tau[i];
        }
        a[(mrows - 1) + ii * lda] = one() - tau[i];
        for l in mrows..m {
            a[l + ii * lda] = zero();
        }
    }
}

/// `zung2r`: the unblocked `ZUNGQR`. Generates the last `ncols-k` columns of
/// `Q = H(k) ... H(1)` in the block of `a` starting at `(r0, c0)`.
///
/// The reflector `H(i)` is column `i` of the block from row `i` down, with its
/// implicit `1` at the top (`ZLARF1F` convention).
fn zung2r(m: usize, ncols: usize, k: usize, a: &mut [Complex64], lda: usize, tau: &[Complex64], r0: usize, c0: usize) {
    if ncols == 0 {
        return;
    }
    for j in k..ncols {
        for l in 0..m {
            a[(r0 + l) + (c0 + j) * lda] = zero();
        }
        a[(r0 + j) + (c0 + j) * lda] = one();
    }
    for i in (0..k).rev() {
        if i < ncols - 1 {
            let mrows = m - i; // Fortran M-I+1
            let ncolc = ncols - i - 1;
            let mut v = vec![zero(); mrows];
            v[0] = one();
            for t in 1..mrows {
                v[t] = a[(r0 + i + t) + (c0 + i) * lda];
            }
            zlarf_left(a, lda, r0 + i, mrows, c0 + i + 1, ncolc, &v, tau[i]);
        }
        if i < m - 1 {
            for t in 1..m - i {
                a[(r0 + i + t) + (c0 + i) * lda] *= -tau[i];
            }
        }
        a[(r0 + i) + (c0 + i) * lda] = one() - tau[i];
        for l in 0..i {
            a[(r0 + l) + (c0 + i) * lda] = zero();
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// `zhemv` for `y := alpha * A * x` with `y` zero on entry.
///
/// `a` holds the `len x len` Hermitian block whose `(r0, r0)` entry is the
/// block's diagonal (so a sub-block of a larger matrix can be named). Only the
/// triangle named by `upper` -- relative to the block -- is read; the other half
/// is reconstructed by conjugation. The diagonal is taken as real, as `zhemv`
/// does (`DBLE(A(J,J))`).
fn zhemv(
    upper: bool,
    len: usize,
    alpha: Complex64,
    a: &[Complex64],
    lda: usize,
    r0: usize,
    c0: usize,
    x: &[Complex64],
) -> Vec<Complex64> {
    let mut y = vec![zero(); len];
    if len == 0 || alpha == zero() {
        return y;
    }
    if upper {
        for j in 0..len {
            let temp1 = alpha * x[j];
            let mut temp2 = zero();
            for i in 0..j {
                let aij = a[(r0 + i) + (c0 + j) * lda];
                y[i] += temp1 * aij;
                temp2 += aij.conj() * x[i];
            }
            y[j] += temp1 * a[(r0 + j) + (c0 + j) * lda].re + alpha * temp2;
        }
    } else {
        for j in 0..len {
            let temp1 = alpha * x[j];
            let mut temp2 = zero();
            y[j] += temp1 * a[(r0 + j) + (c0 + j) * lda].re;
            for i in (j + 1)..len {
                let aij = a[(r0 + i) + (c0 + j) * lda];
                y[i] += temp1 * aij;
                temp2 += aij.conj() * x[i];
            }
            y[j] += alpha * temp2;
        }
    }
    y
}

/// `zher2`: `A := alpha * x * y^H + conj(alpha) * y * x^H + A`, one triangle.
///
/// The diagonal is forced real, as the reference's `DBLE(A(J,J))` does; without
/// that, rounding in the two conjugate products would leave a small imaginary
/// part on the diagonal of a matrix that is supposed to be Hermitian.
#[allow(clippy::too_many_arguments)]
fn zher2(
    upper: bool,
    len: usize,
    alpha: Complex64,
    x: &[Complex64],
    y: &[Complex64],
    a: &mut [Complex64],
    lda: usize,
    r0: usize,
) {
    if len == 0 || alpha == zero() {
        return;
    }
    if upper {
        for j in 0..len {
            let d = r0 + j;
            if x[j] != zero() || y[j] != zero() {
                let temp1 = alpha * y[j].conj();
                let temp2 = (alpha * x[j]).conj();
                a[d + d * lda] = cpx(a[d + d * lda].re + (x[j] * temp1 + y[j] * temp2).re, 0.0);
                for i in 0..j {
                    a[(r0 + i) + d * lda] += x[i] * temp1 + y[i] * temp2;
                }
            } else {
                a[d + d * lda] = cpx(a[d + d * lda].re, 0.0);
            }
        }
    } else {
        for j in 0..len {
            let d = r0 + j;
            if x[j] != zero() || y[j] != zero() {
                let temp1 = alpha * y[j].conj();
                let temp2 = (alpha * x[j]).conj();
                a[d + d * lda] = cpx(a[d + d * lda].re + (x[j] * temp1 + y[j] * temp2).re, 0.0);
                for i in (j + 1)..len {
                    a[(r0 + i) + d * lda] += x[i] * temp1 + y[i] * temp2;
                }
            } else {
                a[d + d * lda] = cpx(a[d + d * lda].re, 0.0);
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// `zlarfg`: an elementary reflector `H = I - tau * v * v^H` with `H x = beta e1`.
///
/// `alpha` is the entry to annihilate and `x` the rest of the column; both are
/// updated in place, `alpha` becoming the real `beta` and `x` the scaled tail.
/// The return is `tau`; `tau == 0` marks the identity (`x` was zero and `alpha`
/// real), in which case neither argument is touched in a meaningful way.
///
/// The scaling follows the reference: `beta = -sign(alpha) * dlapy3(Re, Im, ||x||)`
/// has the *opposite* sign to `alpha`'s real part, and when `beta` comes out
/// subnormal the whole problem is scaled up by `safmin` repeatedly (at most 20
/// times) and re-solved, then `beta` is scaled back. The vector is consistently
/// scaled by `1/(alpha - beta)` afterwards, so the shifted problem and the
/// original one have the same reflector direction.
fn zlarfg(alpha: &mut Complex64, x: &mut [Complex64]) -> Complex64 {
    // The reference's `N <= 0` early return cannot arise here: `N` is
    // `x.len() + 1`, and `x` is the column tail.
    let mut xnrm = dznrm2(x);
    let mut alphr = alpha.re;
    let mut alphi = alpha.im;

    if xnrm == 0.0 && alphi == 0.0 {
        // H = I.
        return zero();
    }

    // `SAFMIN = DLAMCH('S') / DLAMCH('E')` -- the threshold below which the norm
    // and beta themselves lose accuracy.
    let safmin = f64::MIN_POSITIVE / (f64::EPSILON / 2.0);
    let rsafmn = 1.0 / safmin;

    let mut beta = -sign_f(dlapy3(alphr, alphi, xnrm), alphr);
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
        // The reference rebinds ALPHA to the scaled (ALPHR, ALPHI) here.
        beta = -sign_f(dlapy3(alphr, alphi, xnrm), alphr);
    }

    let tau = cpx((beta - alphr) / beta, -alphi / beta);
    let al = zladiv(one(), cpx(alphr, alphi) - cpx(beta, 0.0));
    for xi in x.iter_mut() {
        *xi *= al;
    }
    for _ in 0..knt {
        beta *= safmin;
    }
    *alpha = cpx(beta, 0.0);
    tau
}

/// Fortran's `SIGN(a, b)`: `|a|` with `b`'s sign, where `b == 0` counts as
/// positive (so `-0.0` does not flip it).
fn sign_f(a: f64, b: f64) -> f64 {
    if b < 0.0 { -a.abs() } else { a.abs() }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// `zsteqr`: eigenvalues and eigenvectors of a symmetric tridiagonal matrix by
/// the implicit QL/QR method, for a complex eigenvector matrix.
///
/// This is `dsteqr`'s iteration with `zlasr`'s complex plane rotations: the
/// tridiagonal itself is real (`d`, `e`), but `z` is complex and the rotations
/// are applied to it without conjugation.
///
/// `icompz` mirrors `COMPZ`: `0` for eigenvalues only, `1` when `z` enters
/// holding the reduction matrix `Q`, `2` to initialize `z` to the identity. Only
/// `1` is used by `zheev`, but all three are kept because the branches differ
/// (the `1`/`2` cases use `dlaev2` and accumulate the rotations into `z`, the `0`
/// case uses `dlae2`). Returns the reference's `INFO`: the number of off-diagonal
/// elements that failed to converge.
fn zsteqr(
    icompz: i32,
    n: usize,
    d: &mut [f64],
    e: &mut [f64],
    z: &mut [Complex64],
    ldz: usize,
    work: &mut [f64],
) -> usize {
    let mut info = 0usize;
    if n == 0 {
        return 0;
    }
    if n == 1 {
        if icompz == 2 {
            z[0] = one();
        }
        return 0;
    }

    const MAXIT: usize = 30;
    let eps = f64::EPSILON / 2.0;
    let eps2 = eps * eps;
    let safmin = f64::MIN_POSITIVE;
    let safmax = 1.0 / safmin;
    let ssfmax = safmax.sqrt() / 3.0;
    let ssfmin = safmin.sqrt() / eps2;

    if icompz == 2 {
        for j in 0..n {
            for i in 0..n {
                z[i + j * ldz] = if i == j { one() } else { zero() };
            }
        }
    }

    let nmaxit = n * MAXIT;
    let mut jtot = 0usize;
    let mut l1 = 0usize; // Fortran L1 minus one
    let nm1 = n - 1;

    'outer: loop {
        // Fortran: IF( L1.GT.N ) GO TO 160
        if l1 >= n {
            break 'outer;
        }
        if l1 > 0 {
            e[l1 - 1] = 0.0;
        }

        // Determine where the matrix splits.
        let mut m = n - 1;
        if l1 < nm1 {
            // Fortran: DO M = L1, NM1
            for mm in l1..nm1 {
                let tst = e[mm].abs();
                if tst == 0.0 {
                    m = mm;
                    break;
                }
                if tst <= (d[mm].abs().sqrt() * d[mm + 1].abs().sqrt()) * eps {
                    e[mm] = 0.0;
                    m = mm;
                    break;
                }
            }
        }

        let mut l = l1;
        let lsv = l;
        let mut lend = m;
        let lendsv = lend;
        l1 = m + 1;
        if lend == l {
            continue 'outer;
        }

        // Scale the block if it is outside the safe range.
        let mut anorm = 0.0f64;
        for i in l..=lend {
            let t = d[i].abs();
            if t > anorm {
                anorm = t;
            }
        }
        for i in l..lend {
            let t = e[i].abs();
            if t > anorm {
                anorm = t;
            }
        }
        let mut iscale = 0i32;
        if anorm == 0.0 {
            continue 'outer;
        }
        if anorm > ssfmax {
            iscale = 1;
            dlascl_vec(anorm, ssfmax, &mut d[l..=lend]);
            if lend > l {
                dlascl_vec(anorm, ssfmax, &mut e[l..lend]);
            }
        } else if anorm < ssfmin {
            iscale = 2;
            dlascl_vec(anorm, ssfmin, &mut d[l..=lend]);
            if lend > l {
                dlascl_vec(anorm, ssfmin, &mut e[l..lend]);
            }
        }

        // Choose between QL and QR iteration: the larger end is reduced with QR.
        if d[lend].abs() < d[l].abs() {
            lend = lsv;
            l = lendsv;
        }

        if lend > l {
            // ------------------------------------------------------ QL iteration
            'ql: loop {
                // Look for a small subdiagonal element.
                let mut m = lend;
                if l != lend {
                    let lendm1 = lend - 1;
                    for mm in l..=lendm1 {
                        let tst = e[mm] * e[mm];
                        if tst <= (eps2 * d[mm].abs()) * d[mm + 1].abs() + safmin {
                            m = mm;
                            break;
                        }
                    }
                }
                if m < lend {
                    e[m] = 0.0;
                }
                let p = d[l];
                if m == l {
                    // Eigenvalue found.
                    d[l] = p;
                    l += 1;
                    if l <= lend {
                        continue 'ql;
                    }
                    break 'ql;
                }

                // A 2x2 block left: solve it in closed form.
                if m == l + 1 {
                    let (rt1, rt2, c, s) = if icompz > 0 {
                        dlaev2(d[l], e[l], d[l + 1])
                    } else {
                        let (rt1, rt2) = dlae2(d[l], e[l], d[l + 1]);
                        (rt1, rt2, 0.0, 0.0)
                    };
                    if icompz > 0 {
                        work[l] = c;
                        work[n - 1 + l] = s;
                        zlasr_v(n, 2, &work[l..l + 1], &work[n - 1 + l..n + l], z, ldz, l, true);
                    }
                    d[l] = rt1;
                    d[l + 1] = rt2;
                    e[l] = 0.0;
                    l += 2;
                    if l <= lend {
                        continue 'ql;
                    }
                    break 'ql;
                }

                if jtot == nmaxit {
                    break 'ql;
                }
                jtot += 1;

                // Form shift.
                let mut g = (d[l + 1] - p) / (2.0 * e[l]);
                let mut r = dlapy2(g, 1.0);
                g = d[m] - p + (e[l] / (g + r.abs() * g.signum()));

                let mut s = 1.0;
                let mut c = 1.0;
                let mut p = 0.0;

                // Inner loop, from the bottom of the block upwards.
                let mm1 = m - 1;
                for i in (l..=mm1).rev() {
                    let f = s * e[i];
                    let b = c * e[i];
                    let (cc, ss, rr) = dlartg(g, f);
                    c = cc;
                    s = ss;
                    if i != m - 1 {
                        e[i + 1] = rr;
                    }
                    g = d[i + 1] - p;
                    r = (d[i] - g) * s + 2.0 * c * b;
                    p = s * r;
                    d[i + 1] = g + p;
                    g = c * r - b;
                    if icompz > 0 {
                        work[i] = c;
                        work[n - 1 + i] = -s;
                    }
                }

                if icompz > 0 {
                    let mm = m - l + 1;
                    let nc = mm - 1;
                    zlasr_v(
                        n,
                        mm,
                        &work[l..l + nc],
                        &work[n - 1 + l..n - 1 + l + nc],
                        z,
                        ldz,
                        l,
                        true,
                    );
                }
                d[l] -= p;
                e[l] = g;
                // Back to the top of the QL loop.
            }
        } else {
            // ------------------------------------------------------ QR iteration
            'qr: loop {
                // Look for a small superdiagonal element.
                let mut m = lend;
                if l != lend {
                    for mm in ((lend + 1)..=l).rev() {
                        let tst = e[mm - 1] * e[mm - 1];
                        if tst <= (eps2 * d[mm].abs()) * d[mm - 1].abs() + safmin {
                            m = mm;
                            break;
                        }
                    }
                }
                if m > lend {
                    e[m - 1] = 0.0;
                }
                let p = d[l];
                if m == l {
                    // Eigenvalue found. `l` may reach 0 and then step below the
                    // block, which in Fortran means "block finished" -- in
                    // 0-based terms that is the break, not a decrement.
                    d[l] = p;
                    if l == 0 {
                        break 'qr;
                    }
                    l -= 1;
                    if l >= lend {
                        continue 'qr;
                    }
                    break 'qr;
                }

                // A 2x2 block left: solve it in closed form.
                if l > 0 && m == l - 1 {
                    let (rt1, rt2, c, s) = if icompz > 0 {
                        dlaev2(d[l - 1], e[l - 1], d[l])
                    } else {
                        let (rt1, rt2) = dlae2(d[l - 1], e[l - 1], d[l]);
                        (rt1, rt2, 0.0, 0.0)
                    };
                    if icompz > 0 {
                        work[m] = c;
                        work[n - 1 + m] = s;
                        zlasr_v(n, 2, &work[m..m + 1], &work[n - 1 + m..n + m], z, ldz, l - 1, false);
                    }
                    d[l - 1] = rt1;
                    d[l] = rt2;
                    e[l - 1] = 0.0;
                    // Fortran allows L to fall to LEND-1 here; 0-based that is
                    // below the block, so the block is finished.
                    if l < 2 {
                        break 'qr;
                    }
                    l -= 2;
                    if l >= lend {
                        continue 'qr;
                    }
                    break 'qr;
                }

                if jtot == nmaxit {
                    break 'qr;
                }
                jtot += 1;

                // Form shift.
                let mut g = (d[l - 1] - p) / (2.0 * e[l - 1]);
                let mut r = dlapy2(g, 1.0);
                g = d[m] - p + (e[l - 1] / (g + r.abs() * g.signum()));

                let mut s = 1.0;
                let mut c = 1.0;
                let mut p = 0.0;

                // Inner loop, from the top of the block downwards.
                let lm1 = l - 1;
                for i in m..=lm1 {
                    let f = s * e[i];
                    let b = c * e[i];
                    let (cc, ss, rr) = dlartg(g, f);
                    c = cc;
                    s = ss;
                    if i != m {
                        e[i - 1] = rr;
                    }
                    g = d[i] - p;
                    r = (d[i + 1] - g) * s + 2.0 * c * b;
                    p = s * r;
                    d[i] = g + p;
                    g = c * r - b;
                    if icompz > 0 {
                        work[i] = c;
                        work[n - 1 + i] = s;
                    }
                }

                if icompz > 0 {
                    let mm = l - m + 1;
                    let nc = mm - 1;
                    zlasr_v(
                        n,
                        mm,
                        &work[m..m + nc],
                        &work[n - 1 + m..n - 1 + m + nc],
                        z,
                        ldz,
                        m,
                        false,
                    );
                }
                d[l] -= p;
                e[lm1] = g;
                // Back to the top of the QR loop.
            }
        }

        // Undo scaling if necessary.
        if iscale == 1 {
            dlascl_vec(ssfmax, anorm, &mut d[lsv..=lendsv]);
            if lendsv > lsv {
                dlascl_vec(ssfmax, anorm, &mut e[lsv..lendsv]);
            }
        } else if iscale == 2 {
            dlascl_vec(ssfmin, anorm, &mut d[lsv..=lendsv]);
            if lendsv > lsv {
                dlascl_vec(ssfmin, anorm, &mut e[lsv..lendsv]);
            }
        }

        // No convergence after N*MAXIT iterations.
        if jtot < nmaxit {
            continue 'outer;
        }
        for i in 0..n - 1 {
            if e[i] != 0.0 {
                info += 1;
            }
        }
        return info;
    }

    // Order the eigenvalues, swapping the eigenvector columns along with them.
    if icompz == 0 {
        // A plain selection sort is enough: eigenvalues only.
        for i in 0..n - 1 {
            let mut k = i;
            let mut p = d[i];
            for j in (i + 1)..n {
                if d[j] < p {
                    k = j;
                    p = d[j];
                }
            }
            if k != i {
                d[k] = d[i];
                d[i] = p;
            }
        }
    } else {
        // Selection sort, chosen by the reference to minimize swaps of the
        // eigenvector columns.
        for i in 0..n - 1 {
            let mut k = i;
            let mut p = d[i];
            for j in (i + 1)..n {
                if d[j] < p {
                    k = j;
                    p = d[j];
                }
            }
            if k != i {
                d[k] = d[i];
                d[i] = p;
                for row in 0..n {
                    z.swap(row + i * ldz, row + k * ldz);
                }
            }
        }
    }
    0
}

/// `zlasr` restricted to `SIDE='R'`, `PIVOT='V'`: applies a sequence of plane
/// rotations to columns `col0:col0+ncols` of `a` on the right.
///
/// `c` and `s` hold the `ncols-1` real cosines and sines. `backward` selects
/// `DIRECT='B'` (the QL case), `false` selects `'F'` (the QR case). Only the
/// first `ncols-1` entries of `c`/`s` are read, matching the reference's loops.
fn zlasr_v(m: usize, ncols: usize, c: &[f64], s: &[f64], a: &mut [Complex64], lda: usize, col0: usize, backward: bool) {
    if m == 0 || ncols < 2 {
        return;
    }
    debug_assert!(c.len() >= ncols - 1 && s.len() >= ncols - 1);
    let apply = |a: &mut [Complex64], j: usize, ctemp: f64, stemp: f64| {
        for i in 0..m {
            let temp = a[i + (col0 + j) * lda];
            a[i + (col0 + j) * lda] = temp * ctemp - a[i + (col0 + j - 1) * lda] * stemp;
            a[i + (col0 + j - 1) * lda] = temp * stemp + a[i + (col0 + j - 1) * lda] * ctemp;
        }
    };
    if backward {
        for j in (1..ncols).rev() {
            let ctemp = c[j - 1];
            let stemp = s[j - 1];
            if ctemp != 1.0 || stemp != 0.0 {
                apply(a, j, ctemp, stemp);
            }
        }
    } else {
        for j in 1..ncols {
            let ctemp = c[j - 1];
            let stemp = s[j - 1];
            if ctemp != 1.0 || stemp != 0.0 {
                apply(a, j, ctemp, stemp);
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// `dlascl` for a contiguous real vector: multiplies it by `cto / cfrom` in
/// stages, so a single factor that would overflow or underflow never appears.
fn dlascl_vec(cfrom: f64, cto: f64, a: &mut [f64]) {
    if a.is_empty() {
        return;
    }
    let smlnum = f64::MIN_POSITIVE;
    let bignum = 1.0 / smlnum;
    let mut cfromc = cfrom;
    let mut ctoc = cto;
    loop {
        let done;
        let mul;
        let cfrom1 = cfromc * smlnum;
        if cfrom1 == cfromc {
            // cfromc is infinite; multiply by a correctly signed zero or a NaN.
            mul = ctoc / cfromc;
            done = true;
        } else {
            let cto1 = ctoc / bignum;
            if cto1 == ctoc {
                // ctoc is zero or infinite: it is itself the correct factor.
                mul = ctoc;
                done = true;
                cfromc = 1.0;
            } else if cfrom1.abs() > ctoc.abs() && ctoc != 0.0 {
                mul = smlnum;
                done = false;
                cfromc = cfrom1;
            } else if cto1.abs() > cfromc.abs() {
                mul = bignum;
                done = false;
                ctoc = cto1;
            } else {
                mul = ctoc / cfromc;
                done = true;
                if mul == 1.0 {
                    return;
                }
            }
        }
        for x in a.iter_mut() {
            *x *= mul;
        }
        if done {
            return;
        }
    }
}

/// `zlascl` restricted to `TYPE=UPLO`: multiplies the stored triangle of a
/// complex Hermitian matrix by `cto / cfrom`, in the same stages as
/// [`dlascl_vec`].
///
/// `zheev` uses this only to move the matrix into the safe exponent range, so a
/// single factor would in principle do; the staging is kept because it is the
/// reference's, and the extreme case is exactly the one the scaling exists for.
fn zlascl_tri(cfrom: f64, cto: f64, a: &mut [Complex64], n: usize, upper: bool) {
    if n == 0 {
        return;
    }
    let smlnum = f64::MIN_POSITIVE;
    let bignum = 1.0 / smlnum;
    let mut cfromc = cfrom;
    let mut ctoc = cto;
    loop {
        let done;
        let mul;
        let cfrom1 = cfromc * smlnum;
        if cfrom1 == cfromc {
            mul = ctoc / cfromc;
            done = true;
        } else {
            let cto1 = ctoc / bignum;
            if cto1 == ctoc {
                mul = ctoc;
                done = true;
                cfromc = 1.0;
            } else if cfrom1.abs() > ctoc.abs() && ctoc != 0.0 {
                mul = smlnum;
                done = false;
                cfromc = cfrom1;
            } else if cto1.abs() > cfromc.abs() {
                mul = bignum;
                done = false;
                ctoc = cto1;
            } else {
                mul = ctoc / cfromc;
                done = true;
                if mul == 1.0 {
                    return;
                }
            }
        }
        for j in 0..n {
            let (i0, i1) = if upper { (0, j + 1) } else { (j, n) };
            for i in i0..i1 {
                a[i + j * n] *= mul;
            }
        }
        if done {
            return;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// zgeev: eigenvalues and eigenvectors of a complex general matrix
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// `|Re(z)| + |Im(z)|`, the reference's `CABS1`.
fn cabs1(z: Complex64) -> f64 {
    z.re.abs() + z.im.abs()
}

/// `DZNRM2` over a strided slice: the Euclidean norm of `count` elements
/// starting at `start` with the given stride, scaled so nothing overflows.
fn dznrm2_strided(a: &[Complex64], start: usize, stride: usize, count: usize) -> f64 {
    let mut scale = 0.0f64;
    let mut ssq = 1.0f64;
    for t in 0..count {
        let ax = a[start + t * stride].norm();
        if ax != 0.0 {
            if scale < ax {
                let r = scale / ax;
                ssq = 1.0 + ssq * r * r;
                scale = ax;
            } else {
                let r = ax / scale;
                ssq += r * r;
            }
        }
    }
    scale * ssq.sqrt()
}

/// `ZLACPY` with `UPLO = 'L'`: copy the lower triangle (diagonal included).
fn zlacpy_lower(a: &[Complex64], n: usize, b: &mut [Complex64]) {
    for j in 0..n {
        for i in j..n {
            b[i + j * n] = a[i + j * n];
        }
    }
}

/// Normalize each column to Euclidean norm one with the largest component real,
/// exactly as `zgeev` does with `DZNRM2`, `IDAMAX` and `ZSCAL`.
fn znormalize_columns(v: &mut [Complex64], n: usize) {
    for i in 0..n {
        let scl = 1.0 / dznrm2(&v[i * n..(i + 1) * n]);
        for k in 0..n {
            v[k + i * n] *= scl;
        }
        let mut kbest = 0usize;
        let mut best = -1.0f64;
        for k in 0..n {
            let z = v[k + i * n];
            let t = z.re * z.re + z.im * z.im;
            if t > best {
                best = t;
                kbest = k;
            }
        }
        let tmp = v[kbest + i * n].conj() / best.sqrt();
        for k in 0..n {
            v[k + i * n] *= tmp;
        }
        v[kbest + i * n] = cpx(v[kbest + i * n].re, 0.0);
    }
}

/// `zgeev`: the eigenvalues and, optionally, the left and right eigenvectors of a
/// complex general square matrix.
///
/// This is `zgeev.f`'s unblocked route, transcribed literally:
///
/// 1. scale `a` when its largest entry lies outside `[SMLNUM, BIGNUM]`;
/// 2. `zgebal` (`JOB = 'B'`): permute and scale to improve conditioning;
/// 3. `zgehrd` (the unblocked `zgehd2`): reduce to upper Hessenberg form,
///    `a = Q H Q^H`, keeping the reflectors;
/// 4. `zunghr` forms `Q`, then `zhseqr` (`JOB = 'S'`, `COMPZ = 'V'`) reduces
///    `H` to the complex Schur form with the unblocked `zlahqr`, accumulating
///    the Schur vectors;
/// 5. `ztrevc3` with `HOWMNY = 'B'`: the eigenvectors of the triangular `T`,
///    back-transformed by the Schur vectors;
/// 6. `zgebak`: undo the balancing;
/// 7. normalize the eigenvectors and unscale the eigenvalues.
///
/// `a` is `n x n` column-major and is destroyed. `w` receives the eigenvalues.
/// `vl` and `vr`, when given, are `n x n` and receive the left and right
/// eigenvectors as columns; the left eigenvector `u(j)` satisfies
/// `u(j)^H a = lambda(j) u(j)^H`.
///
/// # Errors
///
/// [`LapackError::Convergence`] if the QR iteration in step 4 does not converge;
/// LAPACK reports that as `INFO > 0`, naming the 1-based index at which it gave
/// up, and the outputs then hold partly converged values.
///
/// The blocked `zlaqr0` fallback that `zhseqr`
/// selects for large `n` is not implemented; every matrix goes through the
/// single-shift `zlahqr`, which is what the reference itself does for `n <= 15`.
pub fn zgeev(
    w: &mut [Complex64],
    a: &mut [Complex64],
    n: usize,
    mut vl: Option<&mut [Complex64]>,
    mut vr: Option<&mut [Complex64]>,
) -> Result<(), LapackError> {
    if n == 0 {
        return Ok(());
    }
    let wantvl = vl.is_some();
    let wantvr = vr.is_some();

    // Machine constants, named as `zgeev` names them.
    let eps = f64::EPSILON; // DLAMCH('P')
    let smlnum0 = f64::MIN_POSITIVE; // DLAMCH('S')
    let smlnum = smlnum0.sqrt() / eps;
    let bignum = 1.0 / smlnum;

    // Scale `a` if its largest element lies outside `[SMLNUM, BIGNUM]`.
    let anrm = super::super::blas::zlange('M', a, n, n);
    let mut scalea = false;
    let mut cscale = 1.0;
    if anrm > 0.0 && anrm < smlnum {
        scalea = true;
        cscale = smlnum;
    } else if anrm > bignum {
        scalea = true;
        cscale = bignum;
    }
    if scalea {
        zlascl_g(anrm, cscale, a, n, n, n);
    }

    // Balance the matrix.
    let (ilo, ihi, bal_scale) = zgebal(a, n);

    // Reduce to upper Hessenberg form, keeping the reflectors in `a` and `tau`.
    let mut tau = vec![zero(); n - 1];
    zgehrd(a, n, ilo, ihi, &mut tau);

    // QR iteration, accumulating the Schur vectors in `vl` (or `vr`).
    let info = if wantvl {
        let z = vl.as_deref_mut().unwrap();
        zlacpy_lower(a, n, z);
        zunghr(n, ilo, ihi, z, &tau);
        let info = zhseqr(a, w, n, ilo, ihi, true, Some(z));
        if wantvr {
            let zr = vr.as_deref_mut().unwrap();
            zr.copy_from_slice(z);
        }
        info
    } else if wantvr {
        let z = vr.as_deref_mut().unwrap();
        zlacpy_lower(a, n, z);
        zunghr(n, ilo, ihi, z, &tau);
        zhseqr(a, w, n, ilo, ihi, true, Some(z))
    } else {
        zhseqr(a, w, n, ilo, ihi, false, None)
    };

    if info == 0 {
        if wantvl || wantvr {
            let side = if wantvl && wantvr {
                b'B'
            } else if wantvl {
                b'L'
            } else {
                b'R'
            };
            ztrevc3(side, n, a, vl.as_deref_mut(), vr.as_deref_mut());
        }
        if wantvl {
            let z = vl.unwrap();
            zgebak(z, n, ilo, ihi, &bal_scale, true);
            znormalize_columns(z, n);
        }
        if wantvr {
            let z = vr.unwrap();
            zgebak(z, n, ilo, ihi, &bal_scale, false);
            znormalize_columns(z, n);
        }
    }

    // Undo the scaling of the eigenvalues.
    if scalea {
        let m = n - info.min(n);
        if m > 0 {
            zlascl_g(cscale, anrm, &mut w[info..], m, m, 1);
        }
        if info > 0 && ilo > 0 {
            zlascl_g(cscale, anrm, &mut w[..ilo], ilo, ilo, 1);
        }
    }
    // `INFO > 0` is zhseqr's non-convergence: `w` and the eigenvector buffers hold
    // partly converged values, so this is an error rather than a result.
    if info != 0 {
        return Err(LapackError::Convergence);
    }
    Ok(())
}

/// `zhseqr` with `JOB = 'S'` and `COMPZ = 'N'` or `'V'`, restricted to the
/// unblocked path.
///
/// `h` is `n x n` and upper Hessenberg on entry; on exit it is the complex Schur
/// form `T` (upper triangular). `w` receives the eigenvalues. `ilo`/`ihi` are
/// 0-based. The entries outside the active block are already triangular, so their
/// eigenvalues are copied straight through, as the reference does.
///
/// The reference falls back to the blocked `zlaqr0` either for large `n` or when
/// the unblocked iteration fails; that path is not implemented here, so a
/// failure is returned as-is.
fn zhseqr(
    h: &mut [Complex64],
    w: &mut [Complex64],
    n: usize,
    ilo: usize,
    ihi: usize,
    wantz: bool,
    z: Option<&mut [Complex64]>,
) -> usize {
    if n == 0 {
        return 0;
    }
    for i in 0..ilo {
        w[i] = h[i + i * n];
    }
    for i in (ihi + 1)..n {
        w[i] = h[i + i * n];
    }
    if ilo >= ihi {
        if ilo < n {
            w[ilo] = h[ilo + ilo * n];
        }
        return 0;
    }
    zlahqr(wantz, n, ilo, ihi, h, w, z)
}

/// `zlahqr` (`zlahqr.f`, LAPACK 3.12.1) with `WANTT = .TRUE.`: the complex Schur
/// form and Schur vectors of an upper Hessenberg matrix by the single-shift
/// implicit QR iteration.
///
/// `ilo`/`ihi` are 0-based and delimit the active block. `z`, when given, enters
/// holding the unitary matrix from `zgehrd` and leaves holding the Schur
/// vectors. The return value is LAPACK's `INFO`: `0`, or the (1-based) index that
/// failed to converge.
fn zlahqr(
    wantz: bool,
    n: usize,
    ilo: usize,
    ihi: usize,
    h: &mut [Complex64],
    w: &mut [Complex64],
    mut z: Option<&mut [Complex64]>,
) -> usize {
    // ==== clear out the trash ====
    if ihi >= 3 {
        for j in ilo..=(ihi - 3) {
            h[(j + 2) + j * n] = zero();
            h[(j + 3) + j * n] = zero();
        }
    }
    if ilo + 2 <= ihi {
        h[ihi + (ihi - 2) * n] = zero();
    }
    // ==== ensure that subdiagonal entries are real ====
    for i in (ilo + 1)..=ihi {
        let t = h[i + (i - 1) * n];
        if t.im != 0.0 {
            let mut sc = t / cabs1(t);
            sc = sc.conj() / sc.norm();
            h[i + (i - 1) * n] = cpx(t.norm(), 0.0);
            for j in i..n {
                h[i + j * n] *= sc;
            }
            let m = (n - 1).min(i + 1);
            for j in 0..=m {
                h[j + i * n] *= sc.conj();
            }
            if wantz {
                if let Some(zz) = z.as_deref_mut() {
                    for j in 0..n {
                        zz[j + i * n] *= sc.conj();
                    }
                }
            }
        }
    }

    let nh = ihi - ilo + 1;
    let safmin = f64::MIN_POSITIVE;
    let ulp = f64::EPSILON;
    let smlnum = safmin * (nh as f64 / ulp);
    let itmax = 30 * nh.max(10);
    let i1 = 0usize;
    let i2 = n - 1;
    let mut kdefl = 0usize;
    let mut i = ihi as isize;
    while i >= ilo as isize {
        let i0 = i as usize;
        let mut l = ilo;
        let mut converged_at = None;
        for _its in 0..=itmax {
            // Look for a single small subdiagonal element.
            let mut k0 = l;
            if i0 > l {
                for k in ((l + 1)..=i0).rev() {
                    let mut hit = cabs1(h[k + (k - 1) * n]) <= smlnum;
                    if !hit {
                        let mut tst = cabs1(h[(k - 1) + (k - 1) * n]) + cabs1(h[k + k * n]);
                        if tst == 0.0 {
                            if k >= ilo + 2 {
                                tst += h[(k - 1) + (k - 2) * n].re.abs();
                            }
                            if k < ihi {
                                tst += h[(k + 1) + k * n].re.abs();
                            }
                        }
                        if h[k + (k - 1) * n].re.abs() <= ulp * tst {
                            let hk1 = h[k + (k - 1) * n];
                            let hk2 = h[(k - 1) + k * n];
                            let hkk = h[k + k * n];
                            let diff = h[(k - 1) + (k - 1) * n] - hkk;
                            let ab = cabs1(hk1).max(cabs1(hk2));
                            let ba = cabs1(hk1).min(cabs1(hk2));
                            let aa = cabs1(hkk).max(cabs1(diff));
                            let bb = cabs1(hkk).min(cabs1(diff));
                            let s = aa + ab;
                            if ba * (ab / s) <= smlnum.max(ulp * (bb * (aa / s))) {
                                hit = true;
                            }
                        }
                    }
                    if hit {
                        k0 = k;
                        break;
                    }
                }
            }
            l = k0;
            if l > ilo {
                h[l + (l - 1) * n] = zero();
            }
            if l >= i0 {
                converged_at = Some(l);
                break;
            }
            kdefl += 1;

            // Shift: every tenth iteration an exceptional shift, otherwise
            // Wilkinson's.
            let t: Complex64;
            if kdefl % 20 == 0 {
                let s = 0.75 * h[i0 + (i0 - 1) * n].re.abs();
                t = cpx(s, 0.0) + h[i0 + i0 * n];
            } else if kdefl % 10 == 0 {
                let s = 0.75 * h[(l + 1) + l * n].re.abs();
                t = cpx(s, 0.0) + h[l + l * n];
            } else {
                let mut tt = h[i0 + i0 * n];
                let u = h[(i0 - 1) + i0 * n].sqrt() * h[i0 + (i0 - 1) * n].sqrt();
                let s0 = cabs1(u);
                if s0 != 0.0 {
                    let x = 0.5 * (h[(i0 - 1) + (i0 - 1) * n] - tt);
                    let sx = cabs1(x);
                    let s = s0.max(cabs1(x));
                    let mut y = s * ((x / s) * (x / s) + (u / s) * (u / s)).sqrt();
                    if sx > 0.0 && (x / sx).re * y.re + (x / sx).im * y.im < 0.0 {
                        y = -y;
                    }
                    tt -= u * zladiv(u, x + y);
                }
                t = tt;
            }

            // Look for two consecutive small subdiagonal elements.
            let mut m0 = l;
            let mut v = [zero(); 2];
            let mut found_m = false;
            if i0 >= l + 2 {
                for m in ((l + 1)..=(i0 - 1)).rev() {
                    let h11 = h[m + m * n];
                    let h22 = h[(m + 1) + (m + 1) * n];
                    let mut h11s = h11 - t;
                    let mut h21 = h[(m + 1) + m * n].re;
                    let s = cabs1(h11s) + h21.abs();
                    h11s /= s;
                    h21 /= s;
                    v[0] = h11s;
                    v[1] = cpx(h21, 0.0);
                    let h10 = h[m + (m - 1) * n].re;
                    if h10.abs() * h21.abs() <= ulp * (cabs1(h11s) * (cabs1(h11) + cabs1(h22))) {
                        m0 = m;
                        found_m = true;
                        break;
                    }
                }
            }
            if !found_m {
                // The fallthrough block uses only the (1,1) entry; the reference
                // also reads H(L+1,L+1) here but never uses it.
                let h11 = h[l + l * n];
                let mut h11s = h11 - t;
                let mut h21 = h[(l + 1) + l * n].re;
                let s = cabs1(h11s) + h21.abs();
                h11s /= s;
                h21 /= s;
                v[0] = h11s;
                v[1] = cpx(h21, 0.0);
            }

            // Single-shift QR step.
            for k in m0..=(i0 - 1) {
                if k > m0 {
                    v[0] = h[k + (k - 1) * n];
                    v[1] = h[(k + 1) + (k - 1) * n];
                }
                let mut alpha = v[0];
                let mut xtail = [v[1]];
                let t1 = zlarfg(&mut alpha, &mut xtail);
                let v2 = xtail[0];
                if k > m0 {
                    h[k + (k - 1) * n] = alpha;
                    h[(k + 1) + (k - 1) * n] = zero();
                }
                let t2 = (t1 * v2).re;

                // Apply G from the left to columns K..I2.
                for j in k..=i2 {
                    let sum = t1.conj() * h[k + j * n] + cpx(t2, 0.0) * h[(k + 1) + j * n];
                    h[k + j * n] -= sum;
                    h[(k + 1) + j * n] -= sum * v2;
                }
                // Apply G from the right to rows I1..min(K+2,I).
                let jmax = (k + 2).min(i0);
                for j in i1..=jmax {
                    let sum = t1 * h[j + k * n] + cpx(t2, 0.0) * h[j + (k + 1) * n];
                    h[j + k * n] -= sum;
                    h[j + (k + 1) * n] -= sum * v2.conj();
                }
                if wantz {
                    if let Some(zz) = z.as_deref_mut() {
                        for j in 0..n {
                            let sum = t1 * zz[j + k * n] + cpx(t2, 0.0) * zz[j + (k + 1) * n];
                            zz[j + k * n] -= sum;
                            zz[j + (k + 1) * n] -= sum * v2.conj();
                        }
                    }
                }

                if k == m0 && m0 > l {
                    // The iteration started at M > L: extra scaling keeps
                    // H(M,M-1) real.
                    let mut temp = one() - t1;
                    temp = temp / temp.norm();
                    h[(m0 + 1) + m0 * n] *= temp.conj();
                    if m0 + 2 <= i0 {
                        h[(m0 + 2) + (m0 + 1) * n] *= temp;
                    }
                    for j in m0..=i0 {
                        if j != m0 + 1 {
                            if i2 > j {
                                for jj in (j + 1)..=i2 {
                                    h[j + jj * n] *= temp;
                                }
                            }
                            for jj in i1..j {
                                h[jj + j * n] *= temp.conj();
                            }
                            if wantz {
                                if let Some(zz) = z.as_deref_mut() {
                                    for jj in 0..n {
                                        zz[jj + j * n] *= temp.conj();
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Ensure that H(I,I-1) is real.
            let temp = h[i0 + (i0 - 1) * n];
            if temp.im != 0.0 {
                let rtemp = temp.norm();
                h[i0 + (i0 - 1) * n] = cpx(rtemp, 0.0);
                let temp = temp / rtemp;
                if i2 > i0 {
                    for j in (i0 + 1)..=i2 {
                        h[i0 + j * n] *= temp.conj();
                    }
                }
                for j in i1..i0 {
                    h[j + i0 * n] *= temp;
                }
                if wantz {
                    if let Some(zz) = z.as_deref_mut() {
                        for j in 0..n {
                            zz[j + i0 * n] *= temp;
                        }
                    }
                }
            }
        }

        match converged_at {
            Some(l) => {
                w[i0] = h[i0 + i0 * n];
                kdefl = 0;
                i = l as isize - 1;
            }
            None => return i0 + 1,
        }
    }
    0
}

/// `ztrevc3` (`ztrevc3.f`, LAPACK 3.12.1) with `HOWMNY = 'B'` and the unblocked
/// `NB = 1` back-transformation.
///
/// On entry `t` is the complex Schur form (upper triangular) and `vl`/`vr` hold
/// the unitary Schur vectors `Q`. On exit the requested matrices hold the left
/// and/or right eigenvectors of the original matrix.
///
/// `side` is `b'B'`, `b'L'` or `b'R'`.
fn ztrevc3(side: u8, n: usize, t: &mut [Complex64], vl: Option<&mut [Complex64]>, vr: Option<&mut [Complex64]>) {
    if n == 0 {
        return;
    }
    let bothv = side == b'B';
    let rightv = side == b'R' || bothv;
    let leftv = side == b'L' || bothv;

    let unfl = f64::MIN_POSITIVE;
    let ulp = f64::EPSILON;
    let smlnum = unfl * (n as f64 / ulp);

    // Store the diagonal of T and the 1-norms of the strictly upper triangular
    // part of each column.
    let mut diag = vec![zero(); n];
    for i in 0..n {
        diag[i] = t[i + i * n];
    }
    let mut rwork = vec![0.0f64; n];
    for j in 1..n {
        let mut s = 0.0;
        for i in 0..j {
            s += cabs1(t[i + j * n]);
        }
        rwork[j] = s;
    }

    let mut work = vec![zero(); n];
    let mut xbuf = vec![zero(); n];

    if rightv {
        let vr = vr.unwrap();
        for ki in (0..n).rev() {
            let smin = (ulp * cabs1(t[ki + ki * n])).max(smlnum);
            work[ki] = one();
            for k in 0..ki {
                work[k] = -t[k + ki * n];
            }
            for k in 0..ki {
                t[k + k * n] -= t[ki + ki * n];
                if cabs1(t[k + k * n]) < smin {
                    t[k + k * n] = cpx(smin, 0.0);
                }
            }
            let mut scale = 1.0;
            if ki > 0 {
                zlatrs_upper(ki, false, t, n, 0, 0, &mut work[..ki], &mut scale, &rwork);
                work[ki] = cpx(scale, 0.0);
                for i in 0..n {
                    let mut s = cpx(scale, 0.0) * vr[i + ki * n];
                    for j in 0..ki {
                        s += vr[i + j * n] * work[j];
                    }
                    xbuf[i] = s;
                }
                for i in 0..n {
                    vr[i + ki * n] = xbuf[i];
                }
            }
            let mut ii = 0usize;
            let mut best = cabs1(vr[ki * n]);
            for i in 1..n {
                let c = cabs1(vr[i + ki * n]);
                if c > best {
                    best = c;
                    ii = i;
                }
            }
            let remax = 1.0 / cabs1(vr[ii + ki * n]);
            for i in 0..n {
                vr[i + ki * n] *= remax;
            }
            for k in 0..ki {
                t[k + k * n] = diag[k];
            }
        }
    }

    if leftv {
        let vl = vl.unwrap();
        for ki in 0..n {
            let smin = (ulp * cabs1(t[ki + ki * n])).max(smlnum);
            work[ki] = one();
            for k in (ki + 1)..n {
                work[k] = -t[ki + k * n].conj();
            }
            for k in (ki + 1)..n {
                t[k + k * n] -= t[ki + ki * n];
                if cabs1(t[k + k * n]) < smin {
                    t[k + k * n] = cpx(smin, 0.0);
                }
            }
            let mut scale = 1.0;
            if ki + 1 < n {
                zlatrs_upper(
                    n - ki - 1,
                    true,
                    t,
                    n,
                    ki + 1,
                    ki + 1,
                    &mut work[(ki + 1)..],
                    &mut scale,
                    &rwork,
                );
                work[ki] = cpx(scale, 0.0);
                for i in 0..n {
                    let mut s = cpx(scale, 0.0) * vl[i + ki * n];
                    for j in (ki + 1)..n {
                        s += vl[i + j * n] * work[j];
                    }
                    xbuf[i] = s;
                }
                for i in 0..n {
                    vl[i + ki * n] = xbuf[i];
                }
            }
            let mut ii = 0usize;
            let mut best = cabs1(vl[ki * n]);
            for i in 1..n {
                let c = cabs1(vl[i + ki * n]);
                if c > best {
                    best = c;
                    ii = i;
                }
            }
            let remax = 1.0 / cabs1(vl[ii + ki * n]);
            for i in 0..n {
                vl[i + ki * n] *= remax;
            }
            for k in (ki + 1)..n {
                t[k + k * n] = diag[k];
            }
        }
    }
}

/// `zlatrs` restricted to `UPLO = 'U'`, `DIAG = 'N'` and `NORMIN = 'Y'`: solve
/// `A x = scale * b` (or `A^H x = scale * b`) with scaling to avoid overflow.
///
/// `a` holds the leading `n x n` upper triangular block whose `(r0, c0)` entry is
/// its top-left corner; `lda` is its leading dimension. `cnorm` holds the
/// reference's `CNORM` array (1-norms of the strictly upper part of the columns
/// of the Schur form), read from index zero.
///
/// The reference chooses between a plain `zt rsv` and the scaled loop based on a
/// growth estimate; when no scaling triggers the two are the same arithmetic, so
/// only the scaled loop is implemented here. `scale` is always meaningful, and
/// `x` is overwritten with `scale * A^-1 b`.
fn zlatrs_upper(
    n: usize,
    conj_trans: bool,
    a: &[Complex64],
    lda: usize,
    r0: usize,
    c0: usize,
    x: &mut [Complex64],
    scale: &mut f64,
    cnorm_in: &[f64],
) {
    *scale = 1.0;
    if n == 0 {
        return;
    }
    let smlnum = f64::MIN_POSITIVE / f64::EPSILON;
    let bignum = 1.0 / smlnum;
    let half = 0.5;
    let one_r = 1.0;
    let two = 2.0;

    // NORMIN = 'Y': CNORM is given. TSCAL rescales it when it is huge.
    let mut cnorm = cnorm_in[..n].to_vec();
    let mut tmax = cnorm[0];
    for c in cnorm.iter().skip(1) {
        if *c > tmax {
            tmax = *c;
        }
    }
    let tscal = if tmax <= bignum * half {
        one_r
    } else if tmax <= f64::MAX {
        let t = half / (smlnum * tmax);
        for c in cnorm.iter_mut() {
            *c *= t;
        }
        t
    } else {
        // CNORM has overflowed. The reference recomputes it from A, which cannot
        // matter for finite input; leaving it unscaled is the safe choice.
        one_r
    };

    let mut xmax = 0.0f64;
    for v in x.iter() {
        xmax = xmax.max(cabs1(*v));
    }
    if xmax > bignum * half {
        *scale = (bignum * half) / xmax;
        for v in x.iter_mut() {
            *v *= *scale;
        }
        xmax = bignum;
    } else {
        xmax *= two;
    }

    if !conj_trans {
        // Solve A x = scale b, from the bottom up.
        for jj in (0..n).rev() {
            let mut xj = cabs1(x[jj]);
            let tjjs = a[(r0 + jj) + (c0 + jj) * lda] * tscal;
            let tjj = cabs1(tjjs);
            if tjj > smlnum {
                if tjj < one_r && xj > tjj * bignum {
                    let rec = one_r / xj;
                    for v in x.iter_mut() {
                        *v *= rec;
                    }
                    *scale *= rec;
                    xmax *= rec;
                }
                x[jj] = zladiv(x[jj], tjjs);
                xj = cabs1(x[jj]);
            } else if tjj > 0.0 {
                if xj > tjj * bignum {
                    let mut rec = (tjj * bignum) / xj;
                    if cnorm[jj] > one_r {
                        rec /= cnorm[jj];
                    }
                    for v in x.iter_mut() {
                        *v *= rec;
                    }
                    *scale *= rec;
                    xmax *= rec;
                }
                x[jj] = zladiv(x[jj], tjjs);
                xj = cabs1(x[jj]);
            } else {
                for v in x.iter_mut() {
                    *v = zero();
                }
                x[jj] = one();
                xj = one_r;
                *scale = 0.0;
                xmax = 0.0;
            }
            if xj > one_r {
                let mut rec = one_r / xj;
                if cnorm[jj] > (bignum - xmax) * rec {
                    rec *= half;
                    for v in x.iter_mut() {
                        *v *= rec;
                    }
                    *scale *= rec;
                }
            } else if xj * cnorm[jj] > (bignum - xmax) {
                for v in x.iter_mut() {
                    *v *= half;
                }
                *scale *= half;
            }
            if jj > 0 {
                let alpha = -x[jj] * tscal;
                for i in 0..jj {
                    x[i] += alpha * a[(r0 + i) + (c0 + jj) * lda];
                }
                let mut im = 0usize;
                let mut best = cabs1(x[0]);
                for i in 1..jj {
                    let c = cabs1(x[i]);
                    if c > best {
                        best = c;
                        im = i;
                    }
                }
                xmax = cabs1(x[im]);
            }
        }
    } else {
        // Solve A^H x = scale b, from the top down.
        for jj in 0..n {
            let mut xj = cabs1(x[jj]);
            let mut uscal = cpx(tscal, 0.0);
            let rec0 = one_r / xmax.max(one_r);
            if cnorm[jj] > (bignum - xj) * rec0 {
                let mut rec = rec0 * half;
                let tjjs = a[(r0 + jj) + (c0 + jj) * lda].conj() * tscal;
                let tjj = cabs1(tjjs);
                if tjj > one_r {
                    rec = one_r.min(rec * tjj);
                    uscal = zladiv(uscal, tjjs);
                }
                if rec < one_r {
                    for v in x.iter_mut() {
                        *v *= rec;
                    }
                    *scale *= rec;
                    xmax *= rec;
                }
            }
            let tjjs_here = a[(r0 + jj) + (c0 + jj) * lda].conj() * tscal;
            let mut csumj = zero();
            if uscal == one() {
                for i in 0..jj {
                    csumj += a[(r0 + i) + (c0 + jj) * lda].conj() * x[i];
                }
            } else {
                for i in 0..jj {
                    csumj += (a[(r0 + i) + (c0 + jj) * lda].conj() * uscal) * x[i];
                }
            }
            if uscal == cpx(tscal, 0.0) {
                x[jj] -= csumj;
                xj = cabs1(x[jj]);
                let tjjs = tjjs_here;
                let tjj = cabs1(tjjs);
                if tjj > smlnum {
                    if tjj < one_r && xj > tjj * bignum {
                        let rec = one_r / xj;
                        for v in x.iter_mut() {
                            *v *= rec;
                        }
                        *scale *= rec;
                        xmax *= rec;
                    }
                    x[jj] = zladiv(x[jj], tjjs);
                } else if tjj > 0.0 {
                    if xj > tjj * bignum {
                        let rec = (tjj * bignum) / xj;
                        for v in x.iter_mut() {
                            *v *= rec;
                        }
                        *scale *= rec;
                        xmax *= rec;
                    }
                    x[jj] = zladiv(x[jj], tjjs);
                } else {
                    for v in x.iter_mut() {
                        *v = zero();
                    }
                    x[jj] = one();
                    *scale = 0.0;
                    xmax = 0.0;
                }
            } else {
                x[jj] = zladiv(x[jj], tjjs_here) - csumj;
            }
            xmax = xmax.max(cabs1(x[jj]));
        }
    }
    *scale /= tscal;
}

/// `zgebal` with `JOB = 'B'` (`zgebal.f`, LAPACK 3.12.1): permute and scale a
/// complex general matrix to improve the conditioning of its eigenvalues.
///
/// Returns the 0-based `ilo`/`ihi` bounding the active block and the `scale`
/// vector, whose entries inside `ilo..=ihi` are the scaling factors and whose
/// entries outside hold the 1-based source index of the permutation (as
/// `zgebak` expects).
pub(super) fn zgebal(a: &mut [Complex64], n: usize) -> (usize, usize, Vec<f64>) {
    let mut scale = vec![1.0f64; n];
    if n == 0 {
        return (0, 0, scale);
    }
    let mut k = 0usize;
    let mut l = n - 1;

    // Row and column exchange.
    let mut noconv = true;
    while noconv {
        noconv = false;
        // Search for rows isolating an eigenvalue and push them down.
        let i_start = l;
        let mut i = i_start as isize;
        while i >= 0 {
            let iu = i as usize;
            let mut canswap = true;
            for j in 0..=l {
                if iu != j && a[iu + j * n] != zero() {
                    canswap = false;
                    break;
                }
            }
            if canswap {
                scale[l] = (iu + 1) as f64;
                if iu != l {
                    for r in 0..=l {
                        a.swap(r + iu * n, r + l * n);
                    }
                    for c in k..n {
                        a.swap(iu + c * n, l + c * n);
                    }
                }
                noconv = true;
                if l == 0 {
                    return (0, 0, scale);
                }
                l -= 1;
            }
            i -= 1;
        }
    }

    // Search for columns isolating an eigenvalue and push them left.
    noconv = true;
    while noconv {
        noconv = false;
        let j_start = k;
        let mut j = j_start;
        while j <= l {
            let mut canswap = true;
            for i in k..=l {
                if i != j && a[i + j * n] != zero() {
                    canswap = false;
                    break;
                }
            }
            if canswap {
                scale[k] = (j + 1) as f64;
                if j != k {
                    for r in 0..=l {
                        a.swap(r + j * n, r + k * n);
                    }
                    for c in k..n {
                        a.swap(j + c * n, k + c * n);
                    }
                }
                noconv = true;
                k += 1;
            }
            j += 1;
        }
    }

    for i in k..=l {
        scale[i] = 1.0;
    }

    // Iterative loop for norm reduction.
    let sfmin1 = f64::MIN_POSITIVE / f64::EPSILON;
    let sfmax1 = 1.0 / sfmin1;
    let sfmin2 = sfmin1 * 2.0;
    let sfmax2 = 1.0 / sfmin2;
    let mut noconv = true;
    while noconv {
        noconv = false;
        for i in k..=l {
            let c = dznrm2_strided(a, k + i * n, 1, l - k + 1);
            let r = dznrm2_strided(a, i + k * n, n, l - k + 1);
            let mut ica = 0usize;
            let mut best_ca = cabs1(a[i * n]);
            for t in 1..=l {
                let cc = cabs1(a[t + i * n]);
                if cc > best_ca {
                    best_ca = cc;
                    ica = t;
                }
            }
            let ca = a[ica + i * n].norm();
            let mut ira = 0usize;
            let mut best_ra = cabs1(a[i + k * n]);
            for t in 1..(n - k) {
                let cc = cabs1(a[i + (k + t) * n]);
                if cc > best_ra {
                    best_ra = cc;
                    ira = t;
                }
            }
            let ra = a[i + (ira + k) * n].norm();

            // Guard against zero C or R due to underflow.
            if c == 0.0 || r == 0.0 || (c + ca + r + ra).is_nan() {
                continue;
            }

            let mut cc = c;
            let mut ca2 = ca;
            let mut rr = r;
            let mut ra2 = ra;
            let mut f: f64 = 1.0;
            let mut g = rr / 2.0;
            let s = cc + rr;
            while cc < g && f.max(cc).max(ca2) < sfmax2 && rr.min(g).min(ra2) > sfmin2 {
                f *= 2.0;
                cc *= 2.0;
                ca2 *= 2.0;
                rr /= 2.0;
                g /= 2.0;
                ra2 /= 2.0;
            }
            g = cc / 2.0;
            while g >= rr && rr.max(ra2) < sfmax2 && f.min(cc).min(g).min(ca2) > sfmin2 {
                f /= 2.0;
                cc /= 2.0;
                g /= 2.0;
                ca2 /= 2.0;
                rr *= 2.0;
                ra2 *= 2.0;
            }

            // Now balance.
            if cc + rr >= 0.95 * s {
                continue;
            }
            if f < 1.0 && scale[i] < 1.0 && f * scale[i] <= sfmin1 {
                continue;
            }
            if f > 1.0 && scale[i] > 1.0 && scale[i] >= sfmax1 / f {
                continue;
            }
            let ginv = 1.0 / f;
            scale[i] *= f;
            noconv = true;

            // ZSCAL(N-K+1, GINV, A(I,K), LDA) then ZSCAL(L, F, A(1,I), 1): row I
            // across columns K..N, then column I across rows 0..=L. Writing the
            // first one as `a[t + k * n]` for `t in i..n` scales column K, not
            // row I: it corrupts the matrix instead of applying a similarity, and
            // the scaling loop then diverges to the underflow limit (seen as a
            // `scale` entry of 2e-292 instead of a power of two).
            for cc in k..n {
                a[i + cc * n] *= ginv;
            }
            for rr in 0..=l {
                a[rr + i * n] *= f;
            }
        }
    }

    (k, l, scale)
}

/// `zgebak` with `JOB = 'B'`: undo the balancing that `zgebal` recorded.
///
/// `left` selects `SIDE = 'L'` (which divides each row by its scaling factor)
/// instead of `SIDE = 'R'` (which multiplies). `ilo`/`ihi` are 0-based.
fn zgebak(v: &mut [Complex64], n: usize, ilo: usize, ihi: usize, scale: &[f64], left: bool) {
    if n == 0 {
        return;
    }
    if ilo != ihi {
        if !left {
            for i in ilo..=ihi {
                let s = scale[i];
                for j in 0..n {
                    v[i + j * n] *= s;
                }
            }
        } else {
            for i in ilo..=ihi {
                let s = 1.0 / scale[i];
                for j in 0..n {
                    v[i + j * n] *= s;
                }
            }
        }
    }
    // Backward permutation: I walks down from ILO-1 to 1, then up from IHI+1 to N.
    for ii in 1..=n {
        let i1 = ii;
        if i1 > ilo && i1 <= ihi + 1 {
            continue;
        }
        let i1 = if i1 < ilo + 1 { ilo + 1 - ii } else { i1 };
        let k1 = scale[i1 - 1] as usize;
        if k1 == i1 {
            continue;
        }
        for j in 0..n {
            v.swap((i1 - 1) + j * n, (k1 - 1) + j * n);
        }
    }
}

/// `zgehrd` (the unblocked `zgehd2` path): reduce a complex general matrix to
/// upper Hessenberg form, `a = Q H Q^H`.
///
/// `ilo`/`ihi` are 0-based and delimit the active block; `tau[i]` receives the
/// reflector of step `i` for `i` in `ilo..ihi`. The reflectors are stored in the
/// zeroed part of `a`, as LAPACK packs them.
fn zgehrd(a: &mut [Complex64], n: usize, ilo: usize, ihi: usize, tau: &mut [Complex64]) {
    if n == 0 || ihi <= ilo {
        return;
    }
    for i in ilo..ihi {
        // Reflector H(i) annihilates A(i+2:ihi, i).
        let mut alpha = a[(i + 1) + i * n];
        let taui = {
            let tail = &mut a[(i + 2) + i * n..=ihi + i * n];
            zlarfg(&mut alpha, tail)
        };
        a[(i + 1) + i * n] = alpha;
        tau[i] = taui;

        // The explicit reflector is [1, tail] with the 1 at row i+1.
        let len = ihi - i;
        let mut v = vec![zero(); len];
        v[0] = one();
        for t in 1..len {
            v[t] = a[(i + 1 + t) + i * n];
        }

        // Apply H(i) to A(0:ihi, i+1:ihi) from the right.
        zlarf_right(a, n, 0, ihi + 1, i + 1, ihi - i, &v, taui);
        // Apply H(i)^H to A(i+1:ihi, i+1:n) from the left.
        zlarf_left(a, n, i + 1, ihi - i, i + 1, n - 1 - i, &v, taui.conj());
    }
}

/// `zunghr`: form the unitary matrix `Q` from the reflectors `zgehrd` stored.
///
/// The reflectors are shifted one column to the right and the first `ilo` and
/// last `n-ihi-1` rows and columns are set to those of the identity, then the
/// unblocked `zungqr` ([`zung2r`]) generates the active block. `ilo`/`ihi` are
/// 0-based.
fn zunghr(n: usize, ilo: usize, ihi: usize, a: &mut [Complex64], tau: &[Complex64]) {
    if n == 0 {
        return;
    }
    let nh = ihi - ilo;
    // Shift the reflectors one column to the right.
    if ihi > ilo {
        for j in ((ilo + 1)..=ihi).rev() {
            for i in 0..j {
                a[i + j * n] = zero();
            }
            for i in (j + 1)..=ihi {
                a[i + j * n] = a[i + (j - 1) * n];
            }
            for i in (ihi + 1)..n {
                a[i + j * n] = zero();
            }
        }
    }
    // Columns 0..=ilo and ihi+1..n become unit vectors.
    for j in 0..=ilo {
        for i in 0..n {
            a[i + j * n] = zero();
        }
        a[j + j * n] = one();
    }
    for j in (ihi + 1)..n {
        for i in 0..n {
            a[i + j * n] = zero();
        }
        a[j + j * n] = one();
    }
    if nh > 0 {
        zung2r(nh, nh, nh, a, n, &tau[ilo..ilo + nh], ilo + 1, ilo + 1);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    fn mat_vec(a: &[Complex64], x: &[Complex64], n: usize) -> Vec<Complex64> {
        let mut y = vec![zero(); n];
        for i in 0..n {
            let mut s = zero();
            for j in 0..n {
                s += a[i + j * n] * x[j];
            }
            y[i] = s;
        }
        y
    }

    fn check_right(a: &[Complex64], vr: &[Complex64], w: &[Complex64], n: usize, tol: f64) {
        for j in 0..n {
            let v: Vec<Complex64> = (0..n).map(|i| vr[i + j * n]).collect();
            let av = mat_vec(a, &v, n);
            for i in 0..n {
                let d = (av[i] - w[j] * v[i]).norm();
                assert!(d <= tol, "right residual {d} at ({i},{j})");
            }
        }
    }

    fn check_left(a: &[Complex64], vl: &[Complex64], w: &[Complex64], n: usize, tol: f64) {
        for j in 0..n {
            let u: Vec<Complex64> = (0..n).map(|i| vl[i + j * n]).collect();
            let mut y = vec![zero(); n];
            for i in 0..n {
                let mut s = zero();
                for k in 0..n {
                    s += a[k + i * n].conj() * u[k];
                }
                y[i] = s;
            }
            for i in 0..n {
                let d = (y[i] - w[j].conj() * u[i]).norm();
                assert!(d <= tol, "left residual {d} at ({i},{j})");
            }
        }
    }

    /// The 3x3 real companion matrix used by `complex_mat_eigen`'s own test.
    fn companion3() -> Vec<Complex64> {
        vec![
            cpx(0.0, 0.0),
            cpx(0.0, 0.0),
            cpx(1.0, 0.0),
            cpx(1.0, 0.0),
            cpx(0.0, 0.0),
            cpx(0.0, 0.0),
            cpx(0.0, 0.0),
            cpx(1.0, 0.0),
            cpx(0.0, 0.0),
        ]
    }

    /// The genuinely complex 4x4 matrix of the `complex_mat_eigen` test.
    fn complex4() -> Vec<Complex64> {
        #[rustfmt::skip]
        let rows = [
            [cpx(5.0, 9.0), cpx(5.0,  5.0), cpx(-6.0, -6.0), cpx(-7.0, -7.0)],
            [cpx(3.0, 3.0), cpx(6.0, 10.0), cpx(-5.0, -5.0), cpx(-6.0, -6.0)],
            [cpx(2.0, 2.0), cpx(3.0,  3.0), cpx(-1.0,  3.0), cpx(-5.0, -5.0)],
            [cpx(1.0, 1.0), cpx(2.0,  2.0), cpx(-3.0, -3.0), cpx( 0.0,  4.0)],
        ];
        let mut a = vec![zero(); 16];
        for i in 0..4 {
            for j in 0..4 {
                a[i + j * 4] = rows[i][j];
            }
        }
        a
    }

    #[test]
    fn zgeev_right_eigenvectors_of_the_companion_matrix() {
        let a_orig = companion3();
        let mut a = a_orig.clone();
        let n = 3;
        let mut w = vec![zero(); n];
        let mut vr = vec![zero(); n * n];
        zgeev(&mut w, &mut a, n, None, Some(&mut vr)).unwrap();
        let s3 = 3.0f64.sqrt();
        let expected = [cpx(-0.5, s3 / 2.0), cpx(-0.5, -s3 / 2.0), cpx(1.0, 0.0)];
        for j in 0..n {
            assert!((w[j] - expected[j]).norm() < 1e-14, "w[{j}] = {:?}", w[j]);
        }
        check_right(&a_orig, &vr, &w, n, 1e-14);
    }

    #[test]
    fn zgeev_left_and_right_eigenvectors_of_a_complex_matrix() {
        let a_orig = complex4();
        let mut a = a_orig.clone();
        let n = 4;
        let mut w = vec![zero(); n];
        let mut vl = vec![zero(); n * n];
        let mut vr = vec![zero(); n * n];
        zgeev(&mut w, &mut a, n, Some(&mut vl), Some(&mut vr)).unwrap();
        check_right(&a_orig, &vr, &w, n, 1e-13);
        check_left(&a_orig, &vl, &w, n, 1e-13);
    }

    #[test]
    fn zgeev_repeated_eigenvalues() {
        #[rustfmt::skip]
        let rows = [
            [cpx(2.0, 0.0), cpx(0.0, 0.0), cpx(0.0, 0.0), cpx(0.0, 0.0)],
            [cpx(1.0, 0.0), cpx(2.0, 0.0), cpx(0.0, 0.0), cpx(0.0, 0.0)],
            [cpx(0.0, 0.0), cpx(1.0, 0.0), cpx(3.0, 0.0), cpx(0.0, 0.0)],
            [cpx(0.0, 0.0), cpx(0.0, 0.0), cpx(1.0, 0.0), cpx(3.0, 0.0)],
        ];
        let n = 4;
        let mut a_orig = vec![zero(); n * n];
        for i in 0..n {
            for j in 0..n {
                a_orig[i + j * n] = rows[i][j];
            }
        }
        let mut a = a_orig.clone();
        let mut w = vec![zero(); n];
        let mut vr = vec![zero(); n * n];
        zgeev(&mut w, &mut a, n, None, Some(&mut vr)).unwrap();
        check_right(&a_orig, &vr, &w, n, 1e-14);
    }

    #[test]
    fn zgeev_small_sizes() {
        // 1x1.
        let mut a = vec![cpx(3.0, -2.0)];
        let mut w = vec![zero(); 1];
        let mut vr = vec![zero(); 1];
        let mut vl = vec![zero(); 1];
        zgeev(&mut w, &mut a, 1, Some(&mut vl), Some(&mut vr)).unwrap();
        assert!((w[0] - cpx(3.0, -2.0)).norm() < 1e-15);
        assert!((vr[0] - one()).norm() < 1e-15);
        assert!((vl[0] - one()).norm() < 1e-15);

        // 2x2 with a complex conjugate pair.
        let a_orig = vec![cpx(1.0, 1.0), cpx(2.0, 0.0), cpx(0.0, -1.0), cpx(3.0, 0.0)];
        let n = 2;
        let mut a = a_orig.clone();
        let mut w = vec![zero(); n];
        let mut vr = vec![zero(); n * n];
        let mut vl = vec![zero(); n * n];
        zgeev(&mut w, &mut a, n, Some(&mut vl), Some(&mut vr)).unwrap();
        check_right(&a_orig, &vr, &w, n, 1e-14);
        check_left(&a_orig, &vl, &w, n, 1e-14);
    }

    /// Larger, pseudo-random complex matrices, so the QR iteration has to take
    /// several steps and the exceptional shifts are reached, plus a matrix whose
    /// norm forces `zgeev`'s scaling branch.
    #[test]
    fn zgeev_random_sizes_and_scaled_matrices() {
        let mut state: u64 = 0x0123_4567_89ab_cdef;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        };

        for n in [6usize, 9, 12] {
            let mut a_orig = vec![zero(); n * n];
            for z in a_orig.iter_mut() {
                *z = cpx(next() * 10.0, next() * 10.0);
            }
            let mut a = a_orig.clone();
            let mut w = vec![zero(); n];
            let mut vr = vec![zero(); n * n];
            let mut vl = vec![zero(); n * n];
            zgeev(&mut w, &mut a, n, Some(&mut vl), Some(&mut vr)).unwrap();
            let scale = a_orig.iter().fold(0.0f64, |m, z| m.max(z.norm())).max(1.0);
            check_right(&a_orig, &vr, &w, n, 1e-12 * scale);
            check_left(&a_orig, &vl, &w, n, 1e-12 * scale);
        }

        // An entry far below SMLNUM triggers `ANRM < SMLNUM` and the scaling of
        // `a`; the unscaled eigenvalues must still satisfy the defining equation.
        for factor in [1e-150, 1e150] {
            let n = 3;
            let base = companion3();
            let a_orig: Vec<Complex64> = base.iter().map(|z| *z * factor).collect();
            let mut a = a_orig.clone();
            let mut w = vec![zero(); n];
            let mut vr = vec![zero(); n * n];
            zgeev(&mut w, &mut a, n, None, Some(&mut vr)).unwrap();
            check_right(&a_orig, &vr, &w, n, 1e-13 * factor);
        }
    }

    /// A real symmetric tridiagonal matrix as dense complex data, from its
    /// diagonal `d` and off-diagonal `e` (`e[n-1]` is scratch, as in LAPACK).
    fn tridiagonal_complex(d: &[f64], e: &[f64], n: usize) -> Vec<Complex64> {
        let mut a = vec![zero(); n * n];
        for i in 0..n {
            a[i + i * n] = cpx(d[i], 0.0);
            if i + 1 < n {
                a[i + (i + 1) * n] = cpx(e[i], 0.0);
                a[(i + 1) + i * n] = cpx(e[i], 0.0);
            }
        }
        a
    }

    /// The 1-norm of dense complex data.
    fn norm_1_complex(a: &[Complex64], n: usize) -> f64 {
        let mut best = 0.0f64;
        for j in 0..n {
            let mut sum = 0.0;
            for i in 0..n {
                sum += a[i + j * n].norm();
            }
            best = best.max(sum);
        }
        best
    }

    /// `zsteqr` in its three modes: eigenvalues only, accumulating into a
    /// caller's complex basis (`COMPZ = 'V'`), and starting from the identity.
    ///
    /// The complex eigen-solver reaches it only through `zheev`, with `icompz = 1`
    /// and an identity basis, so the accumulation into a genuinely complex `Z`
    /// and the eigenvalues-only path were never executed.
    #[test]
    fn zsteqr_covers_its_modes() {
        let eps = f64::EPSILON;

        // icompz = 2: identity start, real tridiagonal, complex Z on exit.
        for n in [2usize, 3, 5, 8] {
            let d0: Vec<f64> = (0..n).map(|i| 1.0 + 1.5 * (i as f64)).collect();
            let mut e0 = vec![0.0; n];
            for i in 0..n - 1 {
                e0[i] = 0.5 + 0.125 * ((i * 3) % 4) as f64;
            }
            let t = tridiagonal_complex(&d0, &e0, n);
            let mut d = d0.clone();
            let mut e = e0.clone();
            let mut z = vec![zero(); n * n];
            let mut work = vec![0.0; 2 * n.max(1)];
            let info = zsteqr(2, n, &mut d, &mut e, &mut z, n, &mut work);
            assert_eq!(info, 0, "n={n}: zsteqr did not converge");
            let w = d.clone();
            // T Z = Z W
            let mut tu = vec![zero(); n * n];
            for j in 0..n {
                for i in 0..n {
                    let mut sum = zero();
                    for k in 0..n {
                        sum += t[i + k * n] * z[k + j * n];
                    }
                    tu[i + j * n] = sum - cpx(w[j], 0.0) * z[i + j * n];
                }
            }
            let resid: f64 = tu.iter().map(|v| v.norm()).sum::<f64>() / (norm_1_complex(&t, n) * n as f64 * eps);
            assert!(resid <= 20.0, "n={n}: zsteqr residual = {resid} exceeds 20");
            // Z^H Z = I
            let mut orth = 0.0;
            for j in 0..n {
                for i in 0..n {
                    let mut dot = zero();
                    for k in 0..n {
                        dot += z[k + i * n].conj() * z[k + j * n];
                    }
                    orth += (dot - cpx(if i == j { 1.0 } else { 0.0 }, 0.0)).norm();
                }
            }
            assert!(orth / (n as f64 * eps) <= 20.0, "n={n}: Z is not unitary");
        }

        // icompz = 1 with a genuinely complex basis: Z enters unitary and leaves
        // holding the eigenvectors of A = Z T Z^H.
        let n = 4;
        let d0 = vec![2.0, 3.0, 5.0, 7.0];
        let e0 = vec![0.5, 0.25, 0.125, 0.0];
        let t = tridiagonal_complex(&d0, &e0, n);
        // Z_in = diag(exp(i theta_j))
        let mut z_in = vec![zero(); n * n];
        for j in 0..n {
            let theta = 0.3 * (j as f64 + 1.0);
            z_in[j + j * n] = cpx(theta.cos(), theta.sin());
        }
        // A = Z T Z^H
        let mut a = vec![zero(); n * n];
        for i in 0..n {
            for j in 0..n {
                a[i + j * n] = z_in[i + i * n] * t[i + j * n] * z_in[j + j * n].conj();
            }
        }
        let mut d = d0.clone();
        let mut e = e0.clone();
        let mut z = z_in.clone();
        let mut work = vec![0.0; 2 * n];
        let info = zsteqr(1, n, &mut d, &mut e, &mut z, n, &mut work);
        assert_eq!(info, 0);
        let w = d.clone();
        let mut resid = 0.0;
        for j in 0..n {
            for i in 0..n {
                let mut sum = zero();
                for k in 0..n {
                    sum += a[i + k * n] * z[k + j * n];
                }
                resid += (sum - cpx(w[j], 0.0) * z[i + j * n]).norm();
            }
        }
        let ratio = resid / (norm_1_complex(&a, n) * n as f64 * eps);
        assert!(ratio <= 20.0, "icompz = 1: residual = {ratio} exceeds 20");

        // icompz = 0: eigenvalues only, Z untouched, and the same spectrum as the
        // run that computed the vectors.
        let (mut d_ref, mut e_ref) = (d0.clone(), e0.clone());
        let mut z_ref = vec![zero(); n * n];
        let info = zsteqr(2, n, &mut d_ref, &mut e_ref, &mut z_ref, n, &mut work);
        assert_eq!(info, 0);
        let mut d = d0.clone();
        let mut e = e0.clone();
        let mut z = vec![cpx(7.0, 7.0); n * n];
        let info = zsteqr(0, n, &mut d, &mut e, &mut z, n, &mut work);
        assert_eq!(info, 0);
        assert!(z.iter().all(|v| *v == cpx(7.0, 7.0)), "icompz = 0 must not touch Z");
        let mut got = d.clone();
        got.sort_by(|x, y| x.partial_cmp(y).unwrap());
        for k in 0..n {
            assert!(
                (got[k] - d_ref[k]).abs() <= 1e-13 * d_ref[k].abs().max(1.0),
                "icompz = 0 gave {got:?}, the vector run gave {d_ref:?}"
            );
        }

        // n = 1: no iteration.
        let mut d = vec![4.5];
        let mut e = vec![0.0];
        let mut z = vec![zero()];
        let info = zsteqr(2, 1, &mut d, &mut e, &mut z, 1, &mut work);
        assert_eq!(info, 0);
        assert_eq!(d[0], 4.5);
        assert_eq!(z[0], cpx(1.0, 0.0));
    }

    /// `zheev`, the complex Hermitian driver, has had no direct test: it was
    /// exercised only through `ComplexMatrix::eigen_herm`.
    ///
    /// A 2x2 with a known spectrum, then a 4x4 checked by the defining equation
    /// with LAPACK's eps-normalized residual, both stored triangles.
    #[test]
    fn zheev_solves_the_hermitian_eigenproblem() {
        let eps = f64::EPSILON;

        // [[2, 1-i], [1+i, 3]] has eigenvalues 1 and 4.
        let mut a = vec![cpx(2.0, 0.0), cpx(1.0, 1.0), cpx(1.0, -1.0), cpx(3.0, 0.0)];
        let mut w = vec![0.0; 2];
        zheev(&mut a, &mut w, 2, false).expect("zheev");
        assert!((w[0] - 1.0).abs() < 1e-14, "w = {w:?}");
        assert!((w[1] - 4.0).abs() < 1e-14, "w = {w:?}");

        // A 4x4 Hermitian built as B + B^H, so the spectrum is not symmetric and
        // repeated eigenvalues are unlikely.
        for upper in [false, true] {
            let n = 4;
            let mut b = vec![zero(); n * n];
            for i in 0..n {
                for j in 0..n {
                    let re = (((i * 7 + j * 13 + 3) % 11) as f64) - 5.0;
                    let im = (((i * 5 + j * 17 + 7) % 9) as f64) - 4.0;
                    b[i + j * n] = cpx(re, im);
                }
            }
            let mut a = vec![zero(); n * n];
            for i in 0..n {
                for j in 0..n {
                    a[i + j * n] = b[i + j * n] + b[j + i * n].conj();
                }
            }
            let a0 = a.clone();
            let mut w = vec![0.0; n];
            zheev(&mut a, &mut w, n, upper).expect("zheev");

            // Eigenvalues ascending.
            for k in 1..n {
                assert!(w[k - 1] <= w[k], "upper = {upper}: w is not ascending: {w:?}");
            }
            // A Z = Z diag(w), eps-normalized.
            let mut resid = 0.0;
            for j in 0..n {
                for i in 0..n {
                    let mut sum = zero();
                    for k in 0..n {
                        sum += a0[i + k * n] * a[k + j * n];
                    }
                    resid += (sum - cpx(w[j], 0.0) * a[i + j * n]).norm();
                }
            }
            let ratio = resid / (norm_1_complex(&a0, n) * n as f64 * eps);
            assert!(ratio <= 20.0, "upper = {upper}: zheev residual = {ratio} exceeds 20");
            // Z^H Z = I.
            let mut orth = 0.0;
            for j in 0..n {
                for i in 0..n {
                    let mut dot = zero();
                    for k in 0..n {
                        dot += a[k + i * n].conj() * a[k + j * n];
                    }
                    orth += (dot - cpx(if i == j { 1.0 } else { 0.0 }, 0.0)).norm();
                }
            }
            assert!(orth / (n as f64 * eps) <= 20.0, "upper = {upper}: Z not unitary");
        }

        // Degenerate sizes and the argument check.
        let mut none: Vec<Complex64> = vec![];
        let mut w0: Vec<f64> = vec![];
        zheev(&mut none, &mut w0, 0, false).expect("n = 0");
        let mut one = vec![cpx(3.5, 0.0)];
        let mut w1 = vec![0.0];
        zheev(&mut one, &mut w1, 1, false).expect("n = 1");
        assert_eq!(w1[0], 3.5);
        let mut a = vec![zero(); 4];
        let mut w = vec![0.0; 3];
        assert_eq!(zheev(&mut a, &mut w, 2, false), Err(LapackError::BadShape));
        let mut w = vec![0.0; 2];
        assert_eq!(zheev(&mut a[0..3], &mut w, 2, false), Err(LapackError::BadShape));
    }
}
