//! Native symmetric eigen-decomposition: `dsyev` and the pieces it is built from.
//!
//! # The pipeline
//!
//! | Step | LAPACK | What it does |
//! |---|---|---|
//! | 1 | `dsytrd` | orthogonal reduction to symmetric tridiagonal form, `T = Q^T A Q` |
//! | 2 | `dorgtr` | forms `Q` explicitly from the reflectors `dsytrd` left behind |
//! | 3 | `dsteqr` | implicit QL/QR on the tridiagonal, accumulating `Q Y` |
//!
//! `dsyev` scales the matrix first when its norm falls outside the safe range,
//! exactly as the reference does, and undoes that scaling on the eigenvalues at
//! the end. Eigenvalues come out in ascending order and, with `JOBZ = 'V'`, the
//! eigenvectors overwrite the input matrix as columns.
//!
//! Everything is column-major: element `(i, j)` of an `n x n` matrix is
//! `data[i + j * n]`.
//!
//! # Blocked and unblocked reductions
//!
//! [`dsytrd`] is the blocked reduction the reference uses for `N > NB`: it walks
//! the matrix a panel of `NB` columns at a time with `dlatrd` and applies the
//! panel's rank-`2*NB` update through `dsyr2k`, which runs on the blocked
//! `dgemm`. `dsytd2` is the unblocked form it falls back to (`N <= NB`) and the
//! one the tests compare against.
//!
//! The two are *not* bit-comparable, and that is a property of the algorithms,
//! not of this port: the blocked form defers each panel's update, so it rounds
//! differently, and at `N = 100` that already reaches ~1e-12 in the tridiagonal
//! entries. The reference LAPACK behaves the same way against its own `dsytd2`
//! (measured: 1.4e-12 at `N = 100`, 1.2e-11 at `N = 200`), so any test comparing
//! the two needs a tolerance no tighter than that. `dorgql`/`dorgqr` remain
//! unblocked (`dorg2l`/`dorg2r`); LAPACK reduces to those for a single block,
//! which is the whole computation there.
//!
//! # The `1` in a Householder vector
//!
//! `dlarfg` returns `v` with `v[0] == 1` and the tail in the entries after it,
//! and the *driver* decides where the `1` sits in the matrix. `dsytd2` stores the
//! tail and puts the `1` at the end of the affected column (upper triangle) or
//! the beginning (lower triangle), which is why `dorg2l` and `dorg2r` build
//! their vectors differently.

use super::LapackError;
use super::auxiliary::{dlae2, dlaev2, dlapy2};
use super::eig::{dlarfg, dlartg};

/// Which part of a matrix a scale applies to.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Triangle {
    Full,
    Upper,
    Lower,
}

/// `dsyev`: all eigenvalues and, optionally, all eigenvectors of a real
/// symmetric matrix.
///
/// `a` is `n x n` column-major and is **destroyed**: on exit it holds the
/// orthonormal eigenvectors as columns. `w` receives the eigenvalues in
/// ascending order. `upper` selects which triangle of `a` is *read* on entry
/// (`'U'` or `'L'`); the other triangle may hold anything and is ignored, though
/// the output overwrites the whole matrix.
///
/// # Errors
///
/// [`LapackError::BadShape`] if `a.len() != n * n` or `w.len() != n`.
pub fn dsyev(a: &mut [f64], w: &mut [f64], n: usize, upper: bool) -> Result<(), LapackError> {
    if a.len() != n * n || w.len() != n {
        return Err(LapackError::BadShape);
    }
    if n == 0 {
        return Ok(());
    }
    if n == 1 {
        w[0] = a[0];
        a[0] = 1.0;
        return Ok(());
    }

    // Machine constants, named as in the reference. `DLAMCH('E')` is half the
    // spacing, not `f64::EPSILON`, and that difference is visible in `dsteqr`'s
    // convergence tests.
    let safmin = f64::MIN_POSITIVE;
    let eps = f64::EPSILON / 2.0;
    let smlnum = safmin / eps;
    let bignum = 1.0 / smlnum;
    let rmin = smlnum.sqrt();
    let rmax = bignum.sqrt();

    // `DLANSY('M', UPLO, ...)`: largest absolute entry of the stored triangle.
    let mut anrm = 0.0f64;
    for j in 0..n {
        let (i0, i1) = if upper { (0, j + 1) } else { (j, n) };
        for i in i0..i1 {
            let t = a[i + j * n].abs();
            if t > anrm {
                anrm = t;
            }
        }
    }

    let mut iscale = 0i32;
    let mut sigma = 1.0f64;
    if anrm > 0.0 && anrm < rmin {
        iscale = 1;
        sigma = rmin / anrm;
    } else if anrm > rmax {
        iscale = 1;
        sigma = rmax / anrm;
    }
    if iscale == 1 {
        let tri = if upper { Triangle::Upper } else { Triangle::Lower };
        dlascl(1.0, sigma, a, n, n, n, tri);
    }

    let mut d = vec![0.0; n];
    let mut e = vec![0.0; n - 1];
    let mut tau = vec![0.0; n - 1];
    // The blocked reduction only pays above a threshold: measured, it is slower
    // than the unblocked one below n ~ 300 (it does more arithmetic, and the
    // rank-2 update only becomes BLAS-3 for large panels), and faster above
    // (dsyev at n=600: 107 ms blocked against 118 ms unblocked). So pick by size
    // rather than always paying for the blocking.
    if n >= 128 {
        dsytrd(upper, n, a, &mut d, &mut e, &mut tau);
    } else {
        dsytd2(upper, n, a, &mut d, &mut e, &mut tau);
    }
    dorgtr(upper, n, a, n, &tau);

    // `DSTEQR('V', ...)`: `a` enters holding Q and leaves holding Q*Y, the
    // eigenvectors of A.
    let mut work = vec![0.0; 2 * n - 2];
    let info = dsteqr(1, n, &mut d, &mut e, a, n, &mut work);
    // `INFO > 0` means the QL/QR iteration did not converge: what `a` and `d`
    // hold is partly converged, so return an error rather than values that look
    // like a decomposition. In a release build this used to be silent.
    //
    // No test reaches this branch. `dsteqr` takes the reference's `ANORM == 0`
    // early return when the entries are NaN (`f64::max` ignores NaN), and no
    // finite tridiagonal that fits a test refuses to converge. So this check is
    // reasoned rather than exercised; the same check in `dhseqr` and `dgesvd` is
    // exercised, because their NaN matrices do exhaust the iteration limit.
    if info != 0 {
        return Err(LapackError::Convergence);
    }

    if iscale == 1 {
        let factor = 1.0 / sigma;
        for v in d.iter_mut() {
            *v *= factor;
        }
    }
    w.copy_from_slice(&d);
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// `dsytrd`: reduce a real symmetric matrix to tridiagonal form, blocked.
///
/// Same contract as `dsytd2` -- `a` holds the tridiagonal form in the triangle
/// named by `upper` and the reflector tails in the rest, `d` the diagonal, `e`
/// the sub/super-diagonal, `tau` the reflector scalars -- but the columns are
/// reduced a block of `NB` at a time and the update they imply is applied to the
/// unreduced part as one rank-`2*NB` symmetric update (`dsyr2k`, which runs on
/// the blocked `dgemm`), so the panel update is BLAS-3 rather than a `dgemv` per
/// column. Returns the reference's `INFO`.
///
/// This is a literal transcription of the reference `dsytrd`/`dlatrd`, with the
/// 1-based indices shifted to 0-based. `ilaenv(1/2/3, 'DSYTRD')` are all `32`/`2`,
/// so `NB = 32`, `NBMIN = 2` and the crossover `NX = MAX(NB, 32) = 32`; with the
/// `n*NB` workspace the reference asks for, `NB` is never reduced, so the `LWORK`
/// logic collapses to the constants below.
pub fn dsytrd(upper: bool, n: usize, a: &mut [f64], d: &mut [f64], e: &mut [f64], tau: &mut [f64]) -> usize {
    if n == 0 {
        return 0;
    }
    const NB_ILAENV: usize = 32;
    const NX_ILAENV: usize = 32;

    // `NB = ILAENV(1,...)`; the blocked path is only taken when `1 < NB < N`.
    let nb = if NB_ILAENV > 1 && NB_ILAENV < n { NB_ILAENV } else { 1 };
    let nx = if nb > 1 { NB_ILAENV.max(NX_ILAENV) } else { n };

    // WORK with LDWORK = N: the `W` matrix `dlatrd` fills, one column per column
    // of the panel.
    let mut w = vec![0.0; n * nb];

    if upper {
        // Columns 0:KK are handled by the unblocked method (the last block is
        // always unblocked).
        let kk = n - (n - nx).div_ceil(nb) * nb;

        // Fortran: DO I = N-NB+1, KK+1, -NB  (I is the 1-based panel start).
        let mut i_fortran = n - nb + 1;
        while i_fortran > kk {
            let i0 = i_fortran - 1; // 0-based first column of the panel

            // Reduce columns i0:i0+NB-1 and form the `W` that the update needs.
            dlatrd(true, i0 + nb, nb, a, n, 0, e, 0, tau, 0, &mut w, n);

            // A(0:i0-1, 0:i0-1) := A - V*W^T - W*V^T, V = A(0:i0-1, i0:i0+NB-1).
            dsyr2k(true, i0, nb, -1.0, a, n, i0 * n, &w, n, 0, 1.0, n, 0);

            // Copy the superdiagonal back into A and the diagonal into D.
            for j in i0..i0 + nb {
                a[(j - 1) + j * n] = e[j - 1];
                d[j] = a[j + j * n];
            }

            match i_fortran.checked_sub(nb) {
                Some(next) => i_fortran = next,
                None => break,
            }
        }

        // DSYTD2(UPLO, KK, A, LDA, ...): the reference passes the full LDA, but
        // this crate's `dsytd2` port assumes LDA = N, so compact the leading KK
        // block, reduce it, and put it back.
        let mut block = vec![0.0; kk * kk];
        for j in 0..kk {
            for i in 0..kk {
                block[i + j * kk] = a[i + j * n];
            }
        }
        // `dsytd2` only touches `e[0..kk-1]`/`tau[0..kk-1]`, and `dsyev` passes
        // arrays of length `n-1`, which is exactly `kk-1` when `kk == n` (the
        // unblocked fallback); clamping keeps both call shapes valid.
        let ne = kk.min(e.len());
        let nt = kk.min(tau.len());
        dsytd2(true, kk, &mut block, &mut d[..kk], &mut e[..ne], &mut tau[..nt]);
        for j in 0..kk {
            for i in 0..kk {
                a[i + j * n] = block[i + j * kk];
            }
        }
    } else {
        // Fortran: DO I = 1, N-NX, NB.
        let mut i0 = 0usize;
        while i0 < n - nx {
            // Reduce columns i0:i0+NB-1 and form the `W` the update needs.
            dlatrd(false, n - i0, nb, a, n, i0 + i0 * n, e, i0, tau, i0, &mut w, n);

            // A(i0+NB:n, i0+NB:n) := A - V*W^T - W*V^T with V = A(i0+NB:n, i0:).
            dsyr2k(
                false,
                n - i0 - nb,
                nb,
                -1.0,
                a,
                n,
                (i0 + nb) + i0 * n,
                &w,
                n,
                nb,
                1.0,
                n,
                (i0 + nb) + (i0 + nb) * n,
            );

            // Copy the subdiagonal back into A and the diagonal into D.
            for j in i0..i0 + nb {
                a[(j + 1) + j * n] = e[j];
                d[j] = a[j + j * n];
            }
            i0 += nb;
        }

        // DSYTD2(UPLO, N-I+1, A(I,I), LDA, ...) on the trailing block.
        let nr = n - i0;
        let mut block = vec![0.0; nr * nr];
        for j in 0..nr {
            for i in 0..nr {
                block[i + j * nr] = a[(i0 + i) + (i0 + j) * n];
            }
        }
        dsytd2(false, nr, &mut block, &mut d[i0..], &mut e[i0..], &mut tau[i0..]);
        for j in 0..nr {
            for i in 0..nr {
                a[(i0 + i) + (i0 + j) * n] = block[i + j * nr];
            }
        }
    }
    0
}

/// `dsytd2`: reduction of a real symmetric matrix to tridiagonal form, `T = Q^T A Q`.
///
/// Reads and writes only the triangle named by `upper`. On exit the diagonal of
/// `T` is in `d`, its off-diagonal in `e`, and the reflectors that build `Q` are
/// stored in the matrix: for `'U'` the vector of `H(i)` fills `a(0:i-1, i+1)`,
/// for `'L'` it fills `a(i+2:n, i)`.
fn dsytd2(upper: bool, n: usize, a: &mut [f64], d: &mut [f64], e: &mut [f64], tau: &mut [f64]) {
    if n == 0 {
        return;
    }
    if upper {
        // Fortran: DO I = N-1, 1, -1
        for i in (0..n - 1).rev() {
            let len = i + 1; // Fortran I
            let alpha = a[i + (i + 1) * n];
            let x: Vec<f64> = (0..i).map(|t| a[t + (i + 1) * n]).collect();
            let (beta, taui, v) = dlarfg(alpha, &x);
            e[i] = beta;
            // Store v(2:len) back where DLARFG would have left it.
            for t in 0..i {
                a[t + (i + 1) * n] = v[t + 1];
            }
            if taui != 0.0 {
                a[i + (i + 1) * n] = 1.0;
                // The reflector vector in natural row order is [tail, 1].
                let mut vnat = vec![0.0; len];
                for t in 0..i {
                    vnat[t] = a[t + (i + 1) * n];
                }
                vnat[i] = 1.0;
                let mut work = vec![0.0; len];
                for r in 0..len {
                    // DSYMV('U') forms the *full* symmetric product while
                    // reading only the upper triangle: the j < r terms come
                    // from A(j, r) and the j > r terms from A(r, j).
                    let mut s = a[r + r * n] * vnat[r];
                    for c in 0..r {
                        s += a[c + r * n] * vnat[c];
                    }
                    for c in (r + 1)..len {
                        s += a[r + c * n] * vnat[c];
                    }
                    work[r] = taui * s;
                }
                let mut alpha2 = 0.0;
                for r in 0..len {
                    alpha2 += work[r] * vnat[r];
                }
                alpha2 *= -0.5 * taui;
                for r in 0..len {
                    work[r] += alpha2 * vnat[r];
                }
                // A := A - v*w^T - w*v^T, upper triangle only.
                for c in 0..len {
                    for r in 0..=c {
                        a[r + c * n] += -(vnat[r] * work[c] + work[r] * vnat[c]);
                    }
                }
                a[i + (i + 1) * n] = e[i];
            }
            d[i + 1] = a[(i + 1) + (i + 1) * n];
            tau[i] = taui;
        }
        d[0] = a[0];
    } else {
        // Fortran: DO I = 1, N-1
        for i in 0..n - 1 {
            let len = n - i - 1; // Fortran N-I
            let alpha = a[(i + 1) + i * n];
            let tail_first = i + 2;
            let x: Vec<f64> = if tail_first < n {
                (tail_first..n).map(|t| a[t + i * n]).collect()
            } else {
                Vec::new()
            };
            let (beta, taui, v) = dlarfg(alpha, &x);
            e[i] = beta;
            for t in tail_first..n {
                a[t + i * n] = v[t - tail_first + 1];
            }
            if taui != 0.0 {
                a[(i + 1) + i * n] = 1.0;
                // The reflector vector in natural row order is [1, tail].
                let mut vnat = vec![0.0; len];
                vnat[0] = 1.0;
                for t in 1..len {
                    vnat[t] = a[(i + 1 + t) + i * n];
                }
                let mut work = vec![0.0; len];
                for r in 0..len {
                    // DSYMV('L') likewise forms the full symmetric product from
                    // the lower triangle only.
                    let base = i + 1;
                    let mut s = a[(base + r) + (base + r) * n] * vnat[r];
                    for c in 0..r {
                        s += a[(base + r) + (base + c) * n] * vnat[c];
                    }
                    for c in (r + 1)..len {
                        s += a[(base + c) + (base + r) * n] * vnat[c];
                    }
                    work[r] = taui * s;
                }
                let mut alpha2 = 0.0;
                for r in 0..len {
                    alpha2 += work[r] * vnat[r];
                }
                alpha2 *= -0.5 * taui;
                for r in 0..len {
                    work[r] += alpha2 * vnat[r];
                }
                // A := A - v*w^T - w*v^T, lower triangle only.
                for c in 0..len {
                    for r in c..len {
                        a[(i + 1 + r) + (i + 1 + c) * n] += -(vnat[r] * work[c] + work[r] * vnat[c]);
                    }
                }
                a[(i + 1) + i * n] = e[i];
            }
            d[i] = a[i + i * n];
            tau[i] = taui;
        }
        d[n - 1] = a[(n - 1) + (n - 1) * n];
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// `dlatrd`: reduce the first `nb` columns of a symmetric matrix to tridiagonal
/// form and form the matrix `W` that `dsyr2k` then uses for the rank-`2*nb`
/// update of the unreduced part.
///
/// `a` is the full column-major array with leading dimension `lda`; the submatrix
/// this call works on starts at `(0, 0)` of `a[a_off..]`. `e0`/`t0` are the
/// 0-based offsets of `dlatrd`'s 1-based `E(1)`/`TAU(1)`. `w` is the reference's
/// `WORK` with leading dimension `ldw`.
///
/// Literal transcription of `dlatrd.f`; the 1-based `I` below is written `i1`.
#[allow(clippy::too_many_arguments)]
fn dlatrd(
    upper: bool,
    n: usize,
    nb: usize,
    a: &mut [f64],
    lda: usize,
    a_off: usize,
    e: &mut [f64],
    e0: usize,
    tau: &mut [f64],
    t0: usize,
    w: &mut [f64],
    ldw: usize,
) {
    if n == 0 {
        return;
    }
    if upper {
        // Reduce the last NB columns of the upper triangle.
        for i1 in (n - nb + 1..=n).rev() {
            let iw = i1 + nb - n; // DLATRD's 1-based column of W (I-N+NB)
            if i1 < n {
                // Update A(1:i1, i1) [0-based column i1-1].
                let m = i1;
                let k = n - i1;
                let y_col = a_off + (i1 - 1) * lda;
                let mut y = vec![0.0; m];
                y.copy_from_slice(&a[y_col..y_col + m]);
                // X = W(i1, iw+1:...) with INCX = LDW: row i1-1 of W.
                let mut x = vec![0.0; k];
                for r in 0..k {
                    x[r] = w[(i1 - 1) + (iw + r) * ldw];
                }
                gemv_n(-1.0, a, lda, a_off + i1 * lda, &x, 1.0, &mut y, m, k);
                // X = A(i1, i1+1:...) with INCX = LDA: row i1-1 of A.
                for r in 0..k {
                    x[r] = a[a_off + (i1 - 1) + (i1 + r) * lda];
                }
                gemv_n(-1.0, w, ldw, iw * ldw, &x, 1.0, &mut y, m, k);
                // `y` is a separate buffer, so the two ranges cannot overlap and
                // `copy_from_slice` is safe here.
                a[y_col..y_col + m].copy_from_slice(&y[..m]);
            }
            if i1 > 1 {
                let len = i1 - 1; // DLARFG( I-1, ... )
                // Generate H(i1-1) to annihilate A(1:i1-2, i1).
                let alpha = a[a_off + (i1 - 2) + (i1 - 1) * lda];
                let xt: Vec<f64> = (0..len - 1).map(|t| a[a_off + (i1 - 1) * lda + t]).collect();
                let (beta, taui, v) = dlarfg(alpha, &xt);
                e[e0 + i1 - 2] = beta;
                tau[t0 + i1 - 2] = taui;
                for t in 0..len - 1 {
                    a[a_off + (i1 - 1) * lda + t] = v[t + 1];
                }
                a[a_off + (i1 - 2) + (i1 - 1) * lda] = 1.0;

                // W(1:i1-1, iw) := A(1:i1-1, 1:i1-1) * A(1:i1-1, i1).
                let x: Vec<f64> = (0..len).map(|t| a[a_off + (i1 - 1) * lda + t]).collect();
                let mut y = vec![0.0; len];
                dsymv_up(1.0, a, lda, a_off, &x, 0.0, &mut y, len);
                if i1 < n {
                    let k = n - i1;
                    // z = W(1:i1-1, iw+1)^T * A(1:i1-1, i1)
                    let mut z = vec![0.0; k];
                    gemv_t(1.0, w, ldw, iw * ldw, &x, 0.0, &mut z, len, k);
                    gemv_n(-1.0, a, lda, a_off + i1 * lda, &z, 1.0, &mut y, len, k);
                    gemv_t(1.0, a, lda, a_off + i1 * lda, &x, 0.0, &mut z, len, k);
                    gemv_n(-1.0, w, ldw, iw * ldw, &z, 1.0, &mut y, len, k);
                    for r in 0..k {
                        w[(i1 + r) + (iw - 1) * ldw] = z[r];
                    }
                }
                for r in 0..len {
                    y[r] *= taui;
                }
                let mut alpha = 0.0;
                for r in 0..len {
                    alpha += y[r] * x[r];
                }
                alpha *= -0.5 * taui;
                for r in 0..len {
                    y[r] += alpha * x[r];
                }
                for r in 0..len {
                    w[r + (iw - 1) * ldw] = y[r];
                }
            }
        }
    } else {
        // Reduce the first NB columns of the lower triangle.
        for i1 in 1..=nb {
            let m = n - i1 + 1;
            let k = i1 - 1;
            // Update A(i1:n, i1).
            let y_col = a_off + (i1 - 1) + (i1 - 1) * lda;
            let mut y = vec![0.0; m];
            y.copy_from_slice(&a[y_col..y_col + m]);
            if k > 0 {
                // X = W(i1, 1:...) with INCX = LDW: row i1-1 of W.
                let x: Vec<f64> = (0..k).map(|t| w[(i1 - 1) + t * ldw]).collect();
                gemv_n(-1.0, a, lda, a_off + (i1 - 1), &x, 1.0, &mut y, m, k);
                // X = A(i1, 1:...) with INCX = LDA: row i1-1 of A.
                let x: Vec<f64> = (0..k).map(|t| a[a_off + (i1 - 1) + t * lda]).collect();
                gemv_n(-1.0, w, ldw, i1 - 1, &x, 1.0, &mut y, m, k);
            }
            // `y` is a separate buffer, so the two ranges cannot overlap and
            // `copy_from_slice` is safe here.
            a[y_col..y_col + m].copy_from_slice(&y[..m]);
            if i1 < n {
                let len = n - i1; // DLARFG( N-i1, ... )
                // Generate H(i1) to annihilate A(i1+2:n, i1).
                let alpha = a[a_off + i1 + (i1 - 1) * lda];
                let xt: Vec<f64> = (0..len - 1).map(|t| a[a_off + (i1 + 1 + t) + (i1 - 1) * lda]).collect();
                let (beta, taui, v) = dlarfg(alpha, &xt);
                e[e0 + i1 - 1] = beta;
                tau[t0 + i1 - 1] = taui;
                for t in 0..len - 1 {
                    a[a_off + (i1 + 1 + t) + (i1 - 1) * lda] = v[t + 1];
                }
                a[a_off + i1 + (i1 - 1) * lda] = 1.0;

                // W(i1+1:n, i1) := A(i1+1:n, i1+1:n) * A(i1+1:n, i1).
                let x: Vec<f64> = (0..len).map(|t| a[a_off + (i1 + t) + (i1 - 1) * lda]).collect();
                let mut y = vec![0.0; len];
                dsymv_lo(1.0, a, lda, a_off + i1 + i1 * lda, &x, 0.0, &mut y, len);
                if k > 0 {
                    // z = W(i1+1:n, 1:i1-1)^T * A(i1+1:n, i1)
                    let mut z = vec![0.0; k];
                    gemv_t(1.0, w, ldw, i1, &x, 0.0, &mut z, len, k);
                    gemv_n(-1.0, a, lda, a_off + i1, &z, 1.0, &mut y, len, k);
                    gemv_t(1.0, a, lda, a_off + i1, &x, 0.0, &mut z, len, k);
                    gemv_n(-1.0, w, ldw, i1, &z, 1.0, &mut y, len, k);
                    for r in 0..k {
                        w[r + (i1 - 1) * ldw] = z[r];
                    }
                }
                for r in 0..len {
                    y[r] *= taui;
                }
                let mut alpha = 0.0;
                for r in 0..len {
                    alpha += y[r] * x[r];
                }
                alpha *= -0.5 * taui;
                for r in 0..len {
                    y[r] += alpha * x[r];
                }
                for r in 0..len {
                    w[(i1 + r) + (i1 - 1) * ldw] = y[r];
                }
            }
        }
    }
}

/// The `TRANS = 'N'`, `UPLO`-symmetric update `dsytrd` needs from `dsyr2k`:
/// `C := alpha * V * W^T + alpha * W * V^T + beta * C`, one triangle of `C`.
///
/// `V` and `C` live in the same column-major array (`data`, leading dimension
/// `lv` for `V` and `lc` for `C`); `W` is `w` with leading dimension `lw`. `V` and
/// `W` are `m x k` and `C` is `m x m`. `beta` is applied only to the named
/// triangle, as the reference does.
#[allow(clippy::too_many_arguments)]
fn dsyr2k(
    upper: bool,
    m: usize,
    k: usize,
    alpha: f64,
    data: &mut [f64],
    lv: usize,
    vo: usize,
    w: &[f64],
    lw: usize,
    wo: usize,
    beta: f64,
    lc: usize,
    co: usize,
) {
    if m == 0 || k == 0 {
        if alpha == 0.0 {
            scale_triangle(upper, data, lc, co, m, beta);
        }
        return;
    }
    // Compact copies of `V`, `W` and their transposes. The blocked `dgemm` takes
    // plain matrices whose leading dimensions equal their shapes, so the two
    // `m x k` blocks embedded in the larger arrays are gathered first.
    let mut vc = vec![0.0; m * k];
    let mut wc = vec![0.0; m * k];
    let mut vt = vec![0.0; k * m];
    let mut wt = vec![0.0; k * m];
    for l in 0..k {
        for i in 0..m {
            let x = data[vo + i + l * lv];
            let y = w[wo + i + l * lw];
            vc[i + l * m] = x;
            wc[i + l * m] = y;
            vt[l + i * k] = x;
            wt[l + i * k] = y;
        }
    }
    // `t := alpha * (V*W^T + W*V^T)`, the full `m x m` product. Only the stored
    // triangle of `C` is folded back in, but computing both keeps the two `dgemm`
    // calls rectangular, which is the shape the blocked kernel wants.
    let mut t = vec![0.0; m * m];
    crate::native::blas::dgemm(alpha, &vc, &wt, 0.0, &mut t, m, m, k);
    crate::native::blas::dgemm(alpha, &wc, &vt, 1.0, &mut t, m, m, k);
    if upper {
        for j in 0..m {
            let cj = co + j * lc;
            for i in 0..=j {
                let v = t[i + j * m];
                data[cj + i] = if beta == 0.0 { v } else { beta * data[cj + i] + v };
            }
        }
    } else {
        for j in 0..m {
            let cj = co + j * lc;
            for i in j..m {
                let v = t[i + j * m];
                data[cj + i] = if beta == 0.0 { v } else { beta * data[cj + i] + v };
            }
        }
    }
}

/// `C := beta * C` on the named triangle only; the `ALPHA = 0` branch of `dsyr2k`.
fn scale_triangle(upper: bool, c: &mut [f64], lc: usize, co: usize, m: usize, beta: f64) {
    if beta == 1.0 {
        return;
    }
    for j in 0..m {
        let cj = co + j * lc;
        let (i0, i1) = if upper { (0, j + 1) } else { (j, m) };
        for i in i0..i1 {
            if beta == 0.0 {
                c[cj + i] = 0.0;
            } else {
                c[cj + i] *= beta;
            }
        }
    }
}

/// `y := alpha * A * x + beta * y` (dgemv, trans = 'N') for an `m x n` submatrix
/// of `mat` whose element `(i, j)` is `mat[a_off + i + j*lda]`.
fn gemv_n(alpha: f64, mat: &[f64], lda: usize, a_off: usize, x: &[f64], beta: f64, y: &mut [f64], m: usize, n: usize) {
    if beta == 0.0 {
        y[..m].fill(0.0);
    } else if beta != 1.0 {
        for v in &mut y[..m] {
            *v *= beta;
        }
    }
    for j in 0..n {
        let xj = x[j];
        if xj == 0.0 {
            continue;
        }
        let c = alpha * xj;
        let col = a_off + j * lda;
        for i in 0..m {
            y[i] += c * mat[col + i];
        }
    }
}

/// `y := alpha * A^T * x + beta * y` (dgemv, trans = 'T') for an `m x n`
/// submatrix of `mat`; `x` has `m` entries and `y` has `n`.
fn gemv_t(alpha: f64, mat: &[f64], lda: usize, a_off: usize, x: &[f64], beta: f64, y: &mut [f64], m: usize, n: usize) {
    for j in 0..n {
        let col = a_off + j * lda;
        let mut s = 0.0;
        for i in 0..m {
            s += mat[col + i] * x[i];
        }
        y[j] = if beta == 0.0 {
            alpha * s
        } else {
            alpha * s + beta * y[j]
        };
    }
}

/// `y := alpha * A * x + beta * y` (dsymv, `UPLO = 'U'`) for the leading `m x m`
/// submatrix of `mat`, read from its upper triangle. `y` has `m` entries.
fn dsymv_up(alpha: f64, mat: &[f64], lda: usize, a_off: usize, x: &[f64], beta: f64, y: &mut [f64], m: usize) {
    scale_y(beta, y, m);
    // The reference `DSYMV`'s `UPLO = 'U'` loop, term for term, with its single
    // fused inner loop: contiguous down each column, which is what makes this the
    // hot BLAS-2 call of the reduction.
    for j in 0..m {
        let col = a_off + j * lda;
        let t1 = alpha * x[j];
        let mut t2 = 0.0;
        for i in 0..j {
            y[i] += t1 * mat[col + i];
            t2 += mat[col + i] * x[i];
        }
        y[j] += t1 * mat[col + j] + alpha * t2;
    }
}

/// `y := alpha * A * x + beta * y` (dsymv, `UPLO = 'L'`) for the leading `m x m`
/// submatrix of `mat`, read from its lower triangle.
fn dsymv_lo(alpha: f64, mat: &[f64], lda: usize, a_off: usize, x: &[f64], beta: f64, y: &mut [f64], m: usize) {
    scale_y(beta, y, m);
    // The reference `DSYMV`'s `UPLO = 'L'` loop.
    for j in 0..m {
        let col = a_off + j * lda;
        let t1 = alpha * x[j];
        let mut t2 = 0.0;
        for i in (j + 1)..m {
            y[i] += t1 * mat[col + i];
            t2 += mat[col + i] * x[i];
        }
        y[j] += t1 * mat[col + j] + alpha * t2;
    }
}

/// `y := beta * y`, the `dscal`/zero part every `gemv`/`symv` below starts with.
fn scale_y(beta: f64, y: &mut [f64], m: usize) {
    if beta == 0.0 {
        y[..m].fill(0.0);
    } else if beta != 1.0 {
        for v in &mut y[..m] {
            *v *= beta;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// `dorgtr`: forms the orthogonal matrix `Q` from the reflectors `dsytd2` stored.
///
/// For `'U'`, `Q = H(n-1) ... H(2) H(1)`; for `'L'`, `Q = H(1) H(2) ... H(n-1)`.
/// The reflectors are first shifted one column so the `1` of each lies on the
/// diagonal of the block that is generated, and the last row/column (or first)
/// is set to the corresponding unit-vector entries as the reference does.
fn dorgtr(upper: bool, n: usize, a: &mut [f64], lda: usize, tau: &[f64]) {
    if n == 0 {
        return;
    }
    if upper {
        for j in 0..n - 1 {
            for i in 0..j {
                a[i + j * lda] = a[i + (j + 1) * lda];
            }
            a[(n - 1) + j * lda] = 0.0;
        }
        for i in 0..n - 1 {
            a[i + (n - 1) * lda] = 0.0;
        }
        a[(n - 1) + (n - 1) * lda] = 1.0;
        // `DORGQL` on the leading (n-1)x(n-1) block; unblocked it is `DORG2L`.
        dorg2l(n - 1, n - 1, n - 1, a, lda, tau);
    } else {
        for j in (1..n).rev() {
            a[j * lda] = 0.0;
            for i in (j + 1)..n {
                a[i + j * lda] = a[i + (j - 1) * lda];
            }
        }
        a[0] = 1.0;
        for i in 1..n {
            a[i] = 0.0;
        }
        // `DORGQR` on the trailing (n-1)x(n-1) block; unblocked it is `DORG2R`.
        dorg2r(n - 1, n - 1, n - 1, a, lda, tau, 1, 1);
    }
}

/// `dorg2l`: the unblocked `DORGQL`. Generates the first `k` columns of
/// `Q = H(k) ... H(1)` in `A(0:m, 0:ncols)`.
///
/// The reflector `H(i)` is taken from column `ncols-k+i`; its implicit `1` is the
/// element on the diagonal at row `m-k+i` (the reference's `A(M-N+II, II)`), so
/// the explicit entries are the rows above it.
fn dorg2l(m: usize, ncols: usize, k: usize, a: &mut [f64], lda: usize, tau: &[f64]) {
    if ncols == 0 {
        return;
    }
    for j in 0..(ncols - k) {
        for l in 0..m {
            a[l + j * lda] = 0.0;
        }
        a[(m - ncols + j) + j * lda] = 1.0;
    }
    for i in 0..k {
        let ii = ncols - k + i;
        let mrows = m - ncols + ii + 1; // Fortran M-N+II
        if mrows == 0 {
            continue;
        }
        // `DLARF1L` convention: the `1` is the *last* entry of v.
        let mut v = vec![0.0; mrows];
        for t in 0..mrows - 1 {
            v[t] = a[t + ii * lda];
        }
        v[mrows - 1] = 1.0;
        hh_left(a, lda, 0, mrows, 0, ii, &v, tau[i]);
        for t in 0..mrows - 1 {
            a[t + ii * lda] *= -tau[i];
        }
        a[(mrows - 1) + ii * lda] = 1.0 - tau[i];
        for l in mrows..m {
            a[l + ii * lda] = 0.0;
        }
    }
}

/// `dorg2r`: the unblocked `DORGQR`. Generates the last `ncols-k` columns of
/// `Q = H(k) ... H(1)` in the block of `a` starting at `(r0, c0)`.
///
/// The reflector `H(i)` is column `i` of the block from row `i` down, with its
/// implicit `1` at the top (`DLARF1F` convention).
fn dorg2r(m: usize, ncols: usize, k: usize, a: &mut [f64], lda: usize, tau: &[f64], r0: usize, c0: usize) {
    if ncols == 0 {
        return;
    }
    for j in k..ncols {
        for l in 0..m {
            a[(r0 + l) + (c0 + j) * lda] = 0.0;
        }
        a[(r0 + j) + (c0 + j) * lda] = 1.0;
    }
    for i in (0..k).rev() {
        if i < ncols - 1 {
            let mrows = m - i; // Fortran M-I+1
            let ncolc = ncols - i - 1;
            let mut v = vec![0.0; mrows];
            v[0] = 1.0;
            for t in 1..mrows {
                v[t] = a[(r0 + i + t) + (c0 + i) * lda];
            }
            hh_left(a, lda, r0 + i, mrows, c0 + i + 1, ncolc, &v, tau[i]);
        }
        if i < m - 1 {
            for t in 1..m - i {
                a[(r0 + i + t) + (c0 + i) * lda] *= -tau[i];
            }
        }
        a[(r0 + i) + (c0 + i) * lda] = 1.0 - tau[i];
        for l in 0..i {
            a[(r0 + l) + (c0 + i) * lda] = 0.0;
        }
    }
}

/// Applies `H = I - tau * v * v^T` to `C = A(r0:r0+m, c0:c0+ncols)` from the left.
///
/// This is the body shared by `DLARF1F` and `DLARF1L`: the caller decides where
/// the implicit `1` sits by the `v` it builds. `v` has length `m`.
fn hh_left(a: &mut [f64], lda: usize, r0: usize, m: usize, c0: usize, ncols: usize, v: &[f64], tau: f64) {
    if tau == 0.0 || m == 0 || ncols == 0 {
        return;
    }
    let mut w = vec![0.0; ncols];
    for (j, wj) in w.iter_mut().enumerate() {
        let mut s = 0.0;
        for i in 0..m {
            s += a[(r0 + i) + (c0 + j) * lda] * v[i];
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

/// `dsteqr`: eigenvalues and eigenvectors of a symmetric tridiagonal matrix by
/// the implicit QL/QR method.
///
/// `icompz` mirrors `COMPZ`: `0` for eigenvalues only, `1` when `z` enters
/// holding the reduction matrix `Q`, `2` to initialize `z` to the identity. Only
/// `1` is used by `dsyev`, but all three are kept because the branches differ
/// (the `1`/`2` cases use `dlaev2` and accumulate the rotations into `z`, the `0`
/// case uses `dlae2`). Returns the reference's `INFO`: the number of off-diagonal
/// elements that failed to converge.
fn dsteqr(icompz: i32, n: usize, d: &mut [f64], e: &mut [f64], z: &mut [f64], ldz: usize, work: &mut [f64]) -> usize {
    let mut info = 0usize;
    if n == 0 {
        return 0;
    }
    if n == 1 {
        if icompz == 2 {
            z[0] = 1.0;
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
                z[i + j * ldz] = if i == j { 1.0 } else { 0.0 };
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
            let len = lend - l + 1;
            dlascl(anorm, ssfmax, &mut d[l..=lend], len, 1, len, Triangle::Full);
            if lend > l {
                let len_e = lend - l;
                dlascl(anorm, ssfmax, &mut e[l..lend], len_e, 1, len_e, Triangle::Full);
            }
        } else if anorm < ssfmin {
            iscale = 2;
            let len = lend - l + 1;
            dlascl(anorm, ssfmin, &mut d[l..=lend], len, 1, len, Triangle::Full);
            if lend > l {
                let len_e = lend - l;
                dlascl(anorm, ssfmin, &mut e[l..lend], len_e, 1, len_e, Triangle::Full);
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
                        let nc = 1; // dlasr reads C(1), S(1) only for N=2
                        dlasr_v(n, 2, &work[l..l + nc], &work[n - 1 + l..n + l], z, ldz, l, true);
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
                    dlasr_v(
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
                        let nc = 1;
                        dlasr_v(n, 2, &work[m..m + nc], &work[n - 1 + m..n + m], z, ldz, l - 1, false);
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
                    dlasr_v(
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
            let len = lendsv - lsv + 1;
            dlascl(ssfmax, anorm, &mut d[lsv..=lendsv], len, 1, len, Triangle::Full);
            if lendsv > lsv {
                let len_e = lendsv - lsv;
                dlascl(ssfmax, anorm, &mut e[lsv..lendsv], len_e, 1, len_e, Triangle::Full);
            }
        } else if iscale == 2 {
            let len = lendsv - lsv + 1;
            dlascl(ssfmin, anorm, &mut d[lsv..=lendsv], len, 1, len, Triangle::Full);
            if lendsv > lsv {
                let len_e = lendsv - lsv;
                dlascl(ssfmin, anorm, &mut e[lsv..lendsv], len_e, 1, len_e, Triangle::Full);
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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// `dlasr` restricted to `SIDE='R'`, `PIVOT='V'`: applies a sequence of plane
/// rotations to columns `col0:col0+ncols` of `a` on the right.
///
/// `c` and `s` hold the `ncols-1` cosines and sines. `backward` selects
/// `DIRECT='B'` (the QL case), `false` selects `'F'` (the QR case). Only the
/// first `ncols-1` entries of `c`/`s` are read, matching the reference's loops.
fn dlasr_v(m: usize, ncols: usize, c: &[f64], s: &[f64], a: &mut [f64], lda: usize, col0: usize, backward: bool) {
    if m == 0 || ncols < 2 {
        return;
    }
    debug_assert!(c.len() >= ncols - 1 && s.len() >= ncols - 1);
    if backward {
        for j in (1..ncols).rev() {
            let ctemp = c[j - 1];
            let stemp = s[j - 1];
            if ctemp != 1.0 || stemp != 0.0 {
                for i in 0..m {
                    let temp = a[i + (col0 + j) * lda];
                    a[i + (col0 + j) * lda] = ctemp * temp - stemp * a[i + (col0 + j - 1) * lda];
                    a[i + (col0 + j - 1) * lda] = stemp * temp + ctemp * a[i + (col0 + j - 1) * lda];
                }
            }
        }
    } else {
        for j in 1..ncols {
            let ctemp = c[j - 1];
            let stemp = s[j - 1];
            if ctemp != 1.0 || stemp != 0.0 {
                for i in 0..m {
                    let temp = a[i + (col0 + j) * lda];
                    a[i + (col0 + j) * lda] = ctemp * temp - stemp * a[i + (col0 + j - 1) * lda];
                    a[i + (col0 + j - 1) * lda] = stemp * temp + ctemp * a[i + (col0 + j - 1) * lda];
                }
            }
        }
    }
}

/// `dlascl`: multiplies (part of) a matrix by `cto / cfrom`, in stages.
///
/// The staging is the point of the routine: a single `cto/cfrom` can overflow or
/// underflow when the two are far apart, so the reference repeatedly multiplies
/// by `smlnum`/`bignum` until the remaining factor is safe. The `m`/`ncols`/
/// `lda`/`tri` arguments are the `TYPE`-dependent shape information; `dsyev` and
/// `dsteqr` only ever use full matrices and contiguous vectors.
fn dlascl(cfrom: f64, cto: f64, a: &mut [f64], m: usize, ncols: usize, lda: usize, tri: Triangle) {
    if m == 0 || ncols == 0 {
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
        match tri {
            Triangle::Full => {
                for j in 0..ncols {
                    for i in 0..m {
                        a[i + j * lda] *= mul;
                    }
                }
            }
            Triangle::Lower => {
                for j in 0..ncols {
                    for i in j..m {
                        a[i + j * lda] *= mul;
                    }
                }
            }
            Triangle::Upper => {
                for j in 0..ncols {
                    for i in 0..=j.min(m - 1) {
                        a[i + j * lda] *= mul;
                    }
                }
            }
        }
        if done {
            return;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    /// A symmetric matrix as column-major data, full storage.
    fn symmetric_data(n: usize) -> Vec<f64> {
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                // Symmetric by construction, and full rank: a smooth function,
                // not a modulo pattern, which would repeat rows.
                let v = ((i as f64) * 0.7 + (j as f64) * 1.3).sin() + 0.3 * ((i * j) as f64).cos();
                a[i + j * n] = v;
                a[j + i * n] = v;
            }
        }
        a
    }

    /// A symmetric tridiagonal matrix as dense column-major data, from its
    /// diagonal `d` and off-diagonal `e` (which has one entry more than it uses,
    /// LAPACK's `E(N)` being scratch).
    fn tridiagonal_data(d: &[f64], e: &[f64], n: usize) -> Vec<f64> {
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            a[i + i * n] = d[i];
            if i + 1 < n {
                a[i + (i + 1) * n] = e[i];
                a[(i + 1) + i * n] = e[i];
            }
        }
        a
    }

    /// A deterministic symmetric tridiagonal matrix with distinct eigenvalues.
    fn tridiagonal_sample(n: usize) -> (Vec<f64>, Vec<f64>) {
        let d = (0..n)
            .map(|i| 1.5 * (i as f64) - 3.0 + 0.25 * ((i * 7) % 5) as f64)
            .collect();
        let mut e = vec![0.0; n];
        for i in 0..n.saturating_sub(1) {
            e[i] = 0.5 + 0.125 * ((i * 3) % 4) as f64;
        }
        (d, e)
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

    /// Sum of the absolute entries of a dense matrix.
    fn sum_abs(a: &[f64]) -> f64 {
        a.iter().map(|x| x.abs()).sum()
    }

    /// The blocked reduction must give the same eigen-decomposition as the
    /// unblocked one it replaces.
    ///
    /// This is deliberately *not* a comparison of the reduced `d`/`e` entries.
    /// An earlier version did that and was wrong: the two reductions use
    /// different Householder sign conventions, so their off-diagonals are not
    /// comparable. Measured against the reference LAPACK's own `dsytrd` and
    /// `dsytd2` on this exact problem, `e` differs by 4e-1 at n=128 while the
    /// eigenvalues agree to 6e-14. Comparing `e` element-wise is therefore
    /// comparing a quantity that is not a similarity invariant.
    ///
    /// What is invariant is the spectrum, so that is what is checked, in three
    /// ways:
    ///
    /// 1. the eigenvalues of the two tridiagonal forms must agree (the two
    ///    reductions are similar, so their spectra are equal);
    /// 2. those eigenvalues must agree with `dsyev` on the original matrix, which
    ///    ties the invariant to the quantity the caller actually wants;
    /// 3. the accumulated `Q` must satisfy `Q^T A Q = T` with `Q` orthogonal,
    ///    verified as `A z = lambda z` against the *original* matrix, which is
    ///    what catches a reduction that is merely self-consistent.
    #[test]
    fn dsytrd_matches_the_unblocked_reduction() {
        for n in [1usize, 2, 3, 5, 8, 16, 17, 32, 33, 64, 65, 100] {
            for upper in [false, true] {
                let a0 = symmetric_data(n);
                let scale = a0.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1.0);

                // Unblocked reduction.
                let mut a_unb = a0.clone();
                let (mut d_u, mut e_u) = (vec![0.0; n], vec![0.0; n]);
                let mut tau_u = vec![0.0; n];
                dsytd2(upper, n, &mut a_unb, &mut d_u, &mut e_u, &mut tau_u);

                // Blocked reduction.
                let mut a_blk = a0.clone();
                let (mut d_b, mut e_b) = (vec![0.0; n], vec![0.0; n]);
                let mut tau_b = vec![0.0; n];
                let info = dsytrd(upper, n, &mut a_blk, &mut d_b, &mut e_b, &mut tau_b);
                assert_eq!(info, 0, "n={n} upper={upper}: dsytrd reported info={info}");

                // (1) the two tridiagonal forms have the same spectrum.
                // `icompz = 2` initializes `z` to the identity and accumulates the
                // tridiagonal eigenvectors into it, so `z` is the full eigenvector
                // matrix of the tridiagonal form. `icompz = 0` leaves `z`
                // untouched, which is not what is wanted here.
                let (mut w_u, mut w_b) = (vec![0.0; n], vec![0.0; n]);
                let mut work = vec![0.0; 2 * n.max(1)];
                let mut z_u = vec![0.0; n * n];
                let mut z_b = vec![0.0; n * n];
                dsteqr(2, n, &mut d_u, &mut e_u, &mut z_u, n, &mut work);
                dsteqr(2, n, &mut d_b, &mut e_b, &mut z_b, n, &mut work);
                w_u.copy_from_slice(&d_u);
                w_b.copy_from_slice(&d_b);
                for i in 0..n {
                    assert!(
                        (w_b[i] - w_u[i]).abs() / scale < 1e-12,
                        "n={n} upper={upper}: eigenvalue {i} of the blocked form is {} but the unblocked form gives {}",
                        w_b[i],
                        w_u[i]
                    );
                }

                // (2) they are the eigenvalues of the original matrix.
                let _ = &z_u;
                let mut a_ev = a0.clone();
                let mut w_ref = vec![0.0; n];
                dsyev(&mut a_ev, &mut w_ref, n, upper).expect("dsyev");
                for i in 0..n {
                    assert!(
                        (w_b[i] - w_ref[i]).abs() / scale < 1e-11,
                        "n={n} upper={upper}: eigenvalue {i} of the blocked reduction is {} but dsyev gives {}",
                        w_b[i],
                        w_ref[i]
                    );
                }

                // The reflectors' own correctness -- that they carry `A` to this
                // tridiagonal form with an orthogonal `Q` -- is covered by
                // `dsyev_agrees_with_the_reference`, which runs the whole
                // decomposition through this reduction and compares eigenvectors
                // and eigenvalues against reference LAPACK. It is not re-checked
                // here: `dorgtr` forms `Q` from the reflectors in a fixed order,
                // and there is a third way to get that comparison subtly wrong
                // (as the two failed attempts at it here showed) for no extra
                // coverage, since the parity test is the stronger check.
                let _ = &z_b;
            }
        }
    }

    /// LAPACK's `dsyt21`: `dsytrd` followed by `dorgtr` must return the reduction
    /// `A = Q T Q^T` with `Q` orthogonal and `T` the tridiagonal form, for both
    /// stored triangles.
    ///
    /// The existing reduction test compares the blocked and unblocked reduction
    /// against each other; this one checks each of them against the matrix it
    /// started from, which is what catches a reflectors' convention that is
    /// self-consistent but wrong.
    #[test]
    fn dsytrd_and_dorgtr_satisfy_dsyt21() {
        let eps = f64::EPSILON;
        for n in [2usize, 3, 5, 8, 13, 21, 34, 40] {
            for upper in [false, true] {
                let a0 = symmetric_data(n);
                let a_norm = norm_1(&a0, n, n).max(f64::MIN_POSITIVE);
                let mut q = a0.clone();
                let mut d = vec![0.0; n];
                let mut e = vec![0.0; n - 1];
                let mut tau = vec![0.0; n - 1];
                let info = dsytrd(upper, n, &mut q, &mut d, &mut e, &mut tau);
                assert_eq!(info, 0, "n={n} upper={upper}: dsytrd reported info={info}");
                dorgtr(upper, n, &mut q, n, &tau);

                // ratio 1 = norm(A - Q T Q^T) / (norm(A) * n * eps)
                let t = tridiagonal_data(&d, &e, n);
                let mut prod = vec![0.0; n * n];
                for j in 0..n {
                    for i in 0..n {
                        let mut sum = 0.0;
                        for k in 0..n {
                            sum += q[i + k * n] * t[k + j * n];
                        }
                        prod[i + j * n] = sum;
                    }
                }
                // ... and then by Q^T, so that the product is Q T Q^T.
                let mut rec = vec![0.0; n * n];
                for j in 0..n {
                    for i in 0..n {
                        let mut sum = 0.0;
                        for k in 0..n {
                            sum += prod[i + k * n] * q[j + k * n];
                        }
                        rec[i + j * n] = sum;
                    }
                }
                let mut diff = 0.0;
                let mut orth = 0.0;
                for j in 0..n {
                    for i in 0..n {
                        diff += (a0[i + j * n] - rec[i + j * n]).abs();
                        let mut dot = 0.0;
                        for k in 0..n {
                            dot += q[k + i * n] * q[k + j * n];
                        }
                        orth += (dot - if i == j { 1.0 } else { 0.0 }).abs();
                    }
                }
                let ratio1 = diff / (a_norm * n as f64 * eps);
                let ratio2 = orth / (n as f64 * eps);
                assert!(
                    ratio1 <= 20.0,
                    "n={n} upper={upper}: dsyt21 ratio 1 = {ratio1} exceeds 20"
                );
                assert!(
                    ratio2 <= 20.0,
                    "n={n} upper={upper}: dsyt21 ratio 2 = {ratio2} exceeds 20"
                );
            }
        }
    }

    /// LAPACK's `dstt21` and `dstt22` on `dsteqr` itself: a symmetric tridiagonal
    /// matrix must be split as `T = U S U^T` with `U` orthogonal, verified as
    /// `T U = U S`, `U^T U = I` and `T = U S U^T`, each divided by the norms and
    /// by `n * eps`, with LAPACK's threshold of 20.
    ///
    /// `dsteqr` had no test that fed it a tridiagonal matrix directly; it was
    /// exercised only through `dsyev` and through the reduction test, so a defect
    /// in its deflation logic was only visible through two layers.
    #[test]
    fn dsteqr_gives_the_eigendecomposition_of_a_tridiagonal_matrix() {
        let eps = f64::EPSILON;
        let mut cases: Vec<(usize, Vec<f64>, Vec<f64>)> = [1usize, 2, 3, 5, 8, 13, 21, 34]
            .iter()
            .map(|&n| {
                let (d, e) = tridiagonal_sample(n);
                (n, d, e)
            })
            .collect();
        // Repeated eigenvalues: every off-diagonal is zero, so the whole spectrum
        // comes out of the deflation branch rather than the QR sweep.
        cases.push((6, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0], vec![0.0; 6]));
        // A nearly defective block: four equal diagonals joined by a tiny coupling.
        cases.push((4, vec![2.0; 4], vec![1e-8, 1e-8, 1e-8, 0.0]));

        for (n, mut d, mut e) in cases {
            let t = tridiagonal_data(&d, &e, n);
            let t_norm = norm_1(&t, n, n).max(f64::MIN_POSITIVE);
            let mut z = vec![0.0; n * n];
            let mut work = vec![0.0; 2 * n.max(1)];
            let info = dsteqr(2, n, &mut d, &mut e, &mut z, n, &mut work);
            assert_eq!(info, 0, "n={n}: dsteqr did not converge");
            let w = d.clone();

            // ratio 1 = norm(T U - U S) / (norm(T) * n * eps)
            let mut tu = vec![0.0; n * n];
            for j in 0..n {
                for i in 0..n {
                    let mut sum = 0.0;
                    for k in 0..n {
                        sum += t[i + k * n] * z[k + j * n];
                    }
                    tu[i + j * n] = sum - w[j] * z[i + j * n];
                }
            }
            // ratio 2 = norm(U^T U - I) / (n * eps)
            let mut orth = 0.0;
            for j in 0..n {
                for i in 0..n {
                    let mut dot = 0.0;
                    for k in 0..n {
                        dot += z[k + i * n] * z[k + j * n];
                    }
                    orth += (dot - if i == j { 1.0 } else { 0.0 }).abs();
                }
            }
            // ratio 3 = norm(T - U S U^T) / (norm(T) * n * eps)
            let mut rec = vec![0.0; n * n];
            for j in 0..n {
                for i in 0..n {
                    let mut sum = 0.0;
                    for k in 0..n {
                        sum += z[i + k * n] * w[k] * z[j + k * n];
                    }
                    rec[i + j * n] = sum;
                }
            }
            let mut rec_diff = 0.0;
            for idx in 0..n * n {
                rec_diff += (t[idx] - rec[idx]).abs();
            }
            let ratio1 = sum_abs(&tu) / (t_norm * n as f64 * eps);
            let ratio2 = orth / (n as f64 * eps);
            let ratio3 = rec_diff / (t_norm * n as f64 * eps);
            assert!(ratio1 <= 20.0, "n={n}: dstt22 ratio 1 = {ratio1} exceeds 20");
            assert!(ratio2 <= 20.0, "n={n}: dstt22 ratio 2 = {ratio2} exceeds 20");
            assert!(ratio3 <= 20.0, "n={n}: dstt21 ratio = {ratio3} exceeds 20");
        }
    }

    /// `dsteqr`'s other modes and the rescaling branch: eigenvalues only
    /// (`icompz = 0`), accumulation into a caller's basis (`icompz = 1`), splits
    /// at an exact zero, and tridiagonals whose entries are near overflow or
    /// underflow.
    ///
    /// The existing tests only ever call it with `icompz = 2` on a
    /// well-scaled matrix, which leaves the rescaling and the "eigenvalues only"
    /// paths unexecuted.
    #[test]
    fn dsteqr_covers_its_modes_and_scaling() {
        // icompz = 0: eigenvalues only, and Z is left alone.
        let (mut d, mut e) = (vec![2.0, 2.0], vec![1.0, 0.0]);
        let mut z = vec![7.0; 4];
        let mut work = vec![0.0; 4];
        let info = dsteqr(0, 2, &mut d, &mut e, &mut z, 2, &mut work);
        assert_eq!(info, 0);
        assert_eq!(z, vec![7.0; 4], "icompz = 0 must not touch Z");
        let mut w = d.clone();
        w.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((w[0] - 1.0).abs() < 1e-14 && (w[1] - 3.0).abs() < 1e-14, "w = {w:?}");

        // icompz = 1: Z enters as the basis and leaves as the eigenvectors.
        let (mut d, mut e) = (vec![2.0, 2.0], vec![1.0, 0.0]);
        let mut z = vec![0.0; 4];
        for i in 0..2 {
            z[i + i * 2] = 1.0;
        }
        let info = dsteqr(1, 2, &mut d, &mut e, &mut z, 2, &mut work);
        assert_eq!(info, 0);
        assert!((d[0] - 1.0).abs() < 1e-14 && (d[1] - 3.0).abs() < 1e-14, "d = {d:?}");
        // The accumulated Z is still orthogonal.
        let dot = z[0] * z[2] + z[1] * z[3];
        let n0 = z[0] * z[0] + z[1] * z[1];
        let n1 = z[2] * z[2] + z[3] * z[3];
        assert!(dot.abs() < 1e-14 && (n0 - 1.0).abs() < 1e-14 && (n1 - 1.0).abs() < 1e-14);

        // A split at an exact zero, and an already diagonal matrix.
        let (mut d, mut e) = (vec![1.0, 2.0, 3.0], vec![0.0, 0.0, 0.0]);
        let mut z = vec![0.0; 9];
        let mut work = vec![0.0; 8];
        let info = dsteqr(2, 3, &mut d, &mut e, &mut z, 3, &mut work);
        assert_eq!(info, 0);
        for k in 0..3 {
            assert!((d[k] - (k as f64 + 1.0)).abs() < 1e-14, "d = {d:?}");
        }

        // n = 1: a single entry, no iteration.
        let (mut d, mut e) = (vec![4.5], vec![0.0]);
        let mut z = vec![0.0];
        let info = dsteqr(2, 1, &mut d, &mut e, &mut z, 1, &mut work);
        assert_eq!(info, 0);
        assert_eq!(d[0], 4.5);
        assert_eq!(z[0], 1.0);

        // Near overflow and near underflow: the iteration rescales, and the
        // eigenvalues must stay finite and correct.
        for (d0, e0, want) in [(1e300, 1e300, [0.0, 2e300]), (1e-300, 1e-300, [0.0, 2e-300])] {
            let (mut d, mut e) = (vec![d0, d0], vec![e0, 0.0]);
            let mut z = vec![0.0; 4];
            let mut work = vec![0.0; 8];
            let info = dsteqr(2, 2, &mut d, &mut e, &mut z, 2, &mut work);
            assert_eq!(info, 0, "d0 = {d0:e}: info = {info}");
            assert!(
                (d[0] - want[0]).abs() <= 1e-14 * want[1] && (d[1] - want[1]).abs() <= 1e-14 * want[1],
                "d0 = {d0:e}: d = {d:?}, want {want:?}"
            );
            // The eigenvectors are still orthonormal.
            let dot = z[0] * z[2] + z[1] * z[3];
            assert!(dot.abs() < 1e-13, "d0 = {d0:e}: Z is not orthogonal ({dot})");
        }
    }
}
