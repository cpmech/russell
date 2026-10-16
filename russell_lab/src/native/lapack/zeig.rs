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
//! [`zhetd2`] therefore use the stored triangle alone and reconstruct the
//! missing half by conjugation, which is exactly what `zhemv` and `zher2` do.
//! A routine that read the wrong half would still look plausible on a real
//! matrix, which is why the parity test uses genuinely complex data.
//!
//! # The `1` in a Householder vector
//!
//! `zlarfg` returns `v` with `v[0] == 1` and the tail in the entries after it,
//! and each driver decides where that `1` sits in the matrix. As in the real
//! code, `zung2l` builds its vectors with the `1` last and `zung2r` with it
//! first; the shared application is [`zlarf_left`], which the caller feeds the
//! explicit vector.

use super::LapackError;
use super::eig::dlartg;
use crate::Complex64;

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
    debug_assert_eq!(info, 0, "zheev: zsteqr failed to converge");

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
            a[t + ii * lda] = a[t + ii * lda] * (-tau[i]);
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
                a[(r0 + i + t) + (c0 + i) * lda] = a[(r0 + i + t) + (c0 + i) * lda] * (-tau[i]);
            }
        }
        a[(r0 + i) + (c0 + i) * lda] = one() - tau[i];
        for l in 0..i {
            a[(r0 + l) + (c0 + i) * lda] = zero();
        }
    }
}

/// Applies `H = I - tau * v * v^H` to `C = A(r0:r0+m, c0:c0+ncols)` from the left.
///
/// This is `ZLARF`'s body for `SIDE='L'`. `ZLARF1F`/`ZLARF1L` differ only in
/// where the implicit `1` of `v` sits; here the caller has already put it in `v`,
/// and the last-nonzero-column scan those routines do is an optimisation that
/// changes nothing but which entries are multiplied by zero. `v` has length `m`.
fn zlarf_left(
    a: &mut [Complex64],
    lda: usize,
    r0: usize,
    m: usize,
    c0: usize,
    ncols: usize,
    v: &[Complex64],
    tau: Complex64,
) {
    if tau == zero() || m == 0 || ncols == 0 {
        return;
    }
    let mut w = vec![zero(); ncols];
    for (j, wj) in w.iter_mut().enumerate() {
        let mut s = zero();
        for i in 0..m {
            s += a[(r0 + i) + (c0 + j) * lda] * v[i].conj();
        }
        *wj = s;
    }
    for (j, &wj) in w.iter().enumerate() {
        for i in 0..m {
            a[(r0 + i) + (c0 + j) * lda] -= tau * v[i] * wj;
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
                *xi = *xi * rsafmn;
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
        *xi = *xi * al;
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

/// `DLAPY3`: `sqrt(x^2 + y^2 + z^2)`, scaled so it cannot overflow.
fn dlapy3(x: f64, y: f64, z: f64) -> f64 {
    let xabs = x.abs();
    let yabs = y.abs();
    let zabs = z.abs();
    let w = xabs.max(yabs).max(zabs);
    if w == 0.0 {
        0.0
    } else {
        let d1 = xabs / w;
        let d2 = yabs / w;
        let d3 = zabs / w;
        w * (d1 * d1 + d2 * d2 + d3 * d3).sqrt()
    }
}

/// `DZNRM2`: the Euclidean norm of a complex vector, scaled so that neither the
/// squares nor the sum can overflow.
fn dznrm2(x: &[Complex64]) -> f64 {
    let mut scale = 0.0f64;
    let mut ssq = 1.0f64;
    for c in x {
        let a = c.norm();
        if a != 0.0 {
            if scale < a {
                let r = scale / a;
                ssq = 1.0 + ssq * r * r;
                scale = a;
            } else {
                let r = a / scale;
                ssq += r * r;
            }
        }
    }
    scale * ssq.sqrt()
}

/// `zladiv`: the complex quotient `x / y`, avoiding overflow in `y`'s products.
///
/// The reference routes this through `dladiv`; the two-branch form below is the
/// same algorithm without the extreme-exponent rescaling, which `zlarfg`'s
/// `1/(alpha - beta)` -- where `beta` is by construction comparable to `alpha`
/// once the subnormal case has been scaled away -- does not reach.
fn zladiv(x: Complex64, y: Complex64) -> Complex64 {
    if y.re.abs() >= y.im.abs() {
        let r = y.im / y.re;
        let den = y.re + r * y.im;
        cpx((x.re + r * x.im) / den, (x.im - r * x.re) / den)
    } else {
        let r = y.re / y.im;
        let den = y.im + r * y.re;
        cpx((x.re * r + x.im) / den, (x.im * r - x.re) / den)
    }
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
/// holding the reduction matrix `Q`, `2` to initialise `z` to the identity. Only
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
        if l1 + 1 <= nm1 {
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
                d[l] = d[l] - p;
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
                d[l] = d[l] - p;
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
        // Selection sort, chosen by the reference to minimise swaps of the
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
                a[i + j * n] = a[i + j * n] * mul;
            }
        }
        if done {
            return;
        }
    }
}

/// `dlapy2`: `sqrt(x^2 + y^2)` without unnecessary overflow or underflow.
fn dlapy2(x: f64, y: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if y.is_nan() {
        return y;
    }
    let xabs = x.abs();
    let yabs = y.abs();
    let w = xabs.max(yabs);
    let z = xabs.min(yabs);
    if z == 0.0 || w > f64::MAX {
        w
    } else {
        w * (1.0 + (z / w) * (z / w)).sqrt()
    }
}

/// `dlae2`: the eigenvalues of the symmetric 2x2 block `[[a, b], [b, c]]`.
fn dlae2(a: f64, b: f64, c: f64) -> (f64, f64) {
    let sm = a + c;
    let df = a - c;
    let adf = df.abs();
    let tb = b + b;
    let ab = tb.abs();
    let (acmx, acmn) = if a.abs() > c.abs() { (a, c) } else { (c, a) };
    let rt = if adf > ab {
        adf * (1.0 + (ab / adf) * (ab / adf)).sqrt()
    } else if adf < ab {
        ab * (1.0 + (adf / ab) * (adf / ab)).sqrt()
    } else {
        ab * 2.0f64.sqrt()
    };
    if sm < 0.0 {
        let rt1 = 0.5 * (sm - rt);
        // The order matters: the smaller eigenvalue is fully accurate only when
        // computed from the larger one this way.
        let rt2 = (acmx / rt1) * acmn - (b / rt1) * b;
        (rt1, rt2)
    } else if sm > 0.0 {
        let rt1 = 0.5 * (sm + rt);
        let rt2 = (acmx / rt1) * acmn - (b / rt1) * b;
        (rt1, rt2)
    } else {
        (0.5 * rt, -0.5 * rt)
    }
}

/// `dlaev2`: the eigenvalues and the eigenvector rotation of the symmetric 2x2
/// block `[[a, b], [b, c]]`.
///
/// Returns `(rt1, rt2, cs1, sn1)`; the eigenvector is `(cs1, sn1)` and the signs
/// are chosen exactly as the reference chooses them, which is what pins the
/// eigenvector signs of the whole decomposition.
fn dlaev2(a: f64, b: f64, c: f64) -> (f64, f64, f64, f64) {
    let sm = a + c;
    let df = a - c;
    let adf = df.abs();
    let tb = b + b;
    let ab = tb.abs();
    let (acmx, acmn) = if a.abs() > c.abs() { (a, c) } else { (c, a) };
    let rt = if adf > ab {
        adf * (1.0 + (ab / adf) * (ab / adf)).sqrt()
    } else if adf < ab {
        ab * (1.0 + (adf / ab) * (adf / ab)).sqrt()
    } else {
        ab * 2.0f64.sqrt()
    };

    let (rt1, rt2, sgn1);
    if sm < 0.0 {
        rt1 = 0.5 * (sm - rt);
        sgn1 = -1;
        rt2 = (acmx / rt1) * acmn - (b / rt1) * b;
    } else if sm > 0.0 {
        rt1 = 0.5 * (sm + rt);
        sgn1 = 1;
        rt2 = (acmx / rt1) * acmn - (b / rt1) * b;
    } else {
        rt1 = 0.5 * rt;
        rt2 = -0.5 * rt;
        sgn1 = 1;
    }

    let (cs, sgn2) = if df >= 0.0 { (df + rt, 1) } else { (df - rt, -1) };
    let acs = cs.abs();
    let (mut cs1, mut sn1) = if acs > ab {
        let ct = -tb / cs;
        let sn1 = 1.0 / (1.0 + ct * ct).sqrt();
        (ct * sn1, sn1)
    } else if ab == 0.0 {
        (1.0, 0.0)
    } else {
        let tn = -cs / tb;
        let cs1 = 1.0 / (1.0 + tn * tn).sqrt();
        (cs1, tn * cs1)
    };
    if sgn1 == sgn2 {
        let tn = cs1;
        cs1 = -sn1;
        sn1 = tn;
    }
    (rt1, rt2, cs1, sn1)
}
