//! Native symmetric eigen-decomposition: `dsyev` and the unblocked pieces it is
//! built from.
//!
//! # The pipeline
//!
//! | Step | LAPACK | What it does |
//! |---|---|---|
//! | 1 | `dsytd2` | orthogonal reduction to symmetric tridiagonal form, `T = Q^T A Q` |
//! | 2 | `dorgtr` | forms `Q` explicitly from the reflectors `dsytd2` left behind |
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
//! # Why the unblocked route
//!
//! The blocked `dsytrd`/`dorgql`/`dorgqr` need `dlarft`/`dlarfb` and a workspace
//! whose size comes from `ilaenv`. The unblocked `dsytd2` and the
//! `dorg2l`/`dorg2r` pair compute the same factors with nothing but `dlarfg`,
//! and at the sizes this crate sees the difference is not worth the extra moving
//! parts. LAPACK's own `dorgql`/`dorgqr` reduce to `dorg2l`/`dorg2r` for a single
//! block, which is the whole computation here.
//!
//! # The `1` in a Householder vector
//!
//! `dlarfg` returns `v` with `v[0] == 1` and the tail in the entries after it,
//! and the *driver* decides where the `1` sits in the matrix. `dsytd2` stores the
//! tail and puts the `1` at the end of the affected column (upper triangle) or
//! the beginning (lower triangle), which is why [`dorg2l`] and [`dorg2r`] build
//! their vectors differently.

use super::LapackError;
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
    dsytd2(upper, n, a, &mut d, &mut e, &mut tau);
    dorgtr(upper, n, a, n, &tau);

    // `DSTEQR('V', ...)`: `a` enters holding Q and leaves holding Q*Y, the
    // eigenvectors of A.
    let mut work = vec![0.0; 2 * n - 2];
    let info = dsteqr(1, n, &mut d, &mut e, a, n, &mut work);
    debug_assert_eq!(info, 0, "dsyev: dsteqr failed to converge");

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

/// `dsytd2`: reduction of a real symmetric matrix to tridiagonal form, `T = Q^T A Q`.
///
/// Reads and writes only the triangle named by `upper`. On exit the diagonal of
/// `T` is in `d`, its off-diagonal in `e`, and the reflectors that build `Q` are
/// stored in the matrix: for `'U'` the vector of `H(i)` fills `a(0:i-1, i+1)`,
/// for `'L'` it fills `a(i+2:n, i)`.
/// `dsytrd`: reduce a real symmetric matrix to tridiagonal form, blocked.
///
/// Same contract as [`dsytd2`] -- `a` holds the tridiagonal form in the triangle
/// named by `upper` and the reflector tails in the rest, `d` the diagonal, `e`
/// the sub/super-diagonal, `tau` the reflector scalars -- but the reflectors are
/// accumulated into a block reflector and applied with `dgemm`, so the panel
/// updates are BLAS-3 rather than a `dgemv` per column. Returns the reference's
/// `INFO`.
pub fn dsytrd(upper: bool, n: usize, a: &mut [f64], d: &mut [f64], e: &mut [f64], tau: &mut [f64]) -> usize {
    // Placeholder: delegates to the unblocked form so the interface is fixed and
    // the acceptance test can be written first. The blocked body replaces this.
    dsytd2(upper, n, a, d, e, tau);
    0
}

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
                alpha2 = -0.5 * taui * alpha2;
                for r in 0..len {
                    work[r] += alpha2 * vnat[r];
                }
                // A := A - v*w^T - w*v^T, upper triangle only.
                for c in 0..len {
                    for r in 0..=c {
                        a[r + c * n] += -1.0 * (vnat[r] * work[c] + work[r] * vnat[c]);
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
                alpha2 = -0.5 * taui * alpha2;
                for r in 0..len {
                    work[r] += alpha2 * vnat[r];
                }
                // A := A - v*w^T - w*v^T, lower triangle only.
                for c in 0..len {
                    for r in c..len {
                        a[(i + 1 + r) + (i + 1 + c) * n] += -1.0 * (vnat[r] * work[c] + work[r] * vnat[c]);
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
/// holding the reduction matrix `Q`, `2` to initialise `z` to the identity. Only
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
                d[l] = d[l] - p;
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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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

    /// The blocked reduction must produce the same tridiagonal form and the same
    /// reflectors as the unblocked one it replaces.
    ///
    /// This is the acceptance test for `dsytrd`: the eigen-decomposition it feeds
    /// is checked elsewhere, but a reduction can be *almost* right -- same
    /// eigenvalues, slightly wrong `Q` -- and only this comparison catches that.
    ///
    /// Sizes deliberately straddle the block size and its remainder, because the
    /// blocked path only differs from the unblocked one when a block boundary
    /// falls inside the matrix.
    #[test]
    fn dsytrd_matches_the_unblocked_reduction() {
        for n in [1usize, 2, 3, 5, 8, 16, 17, 32, 33, 64, 65, 100] {
            for upper in [false, true] {
                let a0 = symmetric_data(n);

                let mut a_unb = a0.clone();
                let mut d_unb = vec![0.0; n];
                let mut e_unb = vec![0.0; n];
                let mut tau_unb = vec![0.0; n];
                dsytd2(upper, n, &mut a_unb, &mut d_unb, &mut e_unb, &mut tau_unb);

                let mut a_blk = a0.clone();
                let mut d_blk = vec![0.0; n];
                let mut e_blk = vec![0.0; n];
                let mut tau_blk = vec![0.0; n];
                let info = dsytrd(upper, n, &mut a_blk, &mut d_blk, &mut e_blk, &mut tau_blk);
                assert_eq!(info, 0, "n={n} upper={upper}: dsytrd reported info={info}");

                // The diagonal and off-diagonal of the reduced form.
                for i in 0..n {
                    assert!(
                        (d_blk[i] - d_unb[i]).abs() < 1e-12,
                        "n={n} upper={upper}: d[{i}] is {} but the unblocked reduction gives {}",
                        d_blk[i],
                        d_unb[i]
                    );
                    assert!(
                        (e_blk[i] - e_unb[i]).abs() < 1e-12,
                        "n={n} upper={upper}: e[{i}] is {} but the unblocked reduction gives {}",
                        e_blk[i],
                        e_unb[i]
                    );
                    assert!(
                        (tau_blk[i] - tau_unb[i]).abs() < 1e-12,
                        "n={n} upper={upper}: tau[{i}] is {} but the unblocked reduction gives {}",
                        tau_blk[i],
                        tau_unb[i]
                    );
                }

                // The off-diagonal of the stored triangle is exactly the
                // sub/super-diagonal; the rest of the triangle must hold the
                // reflector tails. Compare the whole stored triangle: the blocked
                // form applies the reflectors in a different order, but the
                // entries DLARFG writes must land in the same places.
                for j in 0..n {
                    for i in 0..n {
                        let stored = if upper { i <= j } else { i >= j };
                        if !stored {
                            continue;
                        }
                        // Skip the tridiagonal itself, compared above; every
                        // other entry is a reflector tail.
                        if i == j || i + 1 == j || j + 1 == i {
                            continue;
                        }
                        let want = a_unb[i + j * n];
                        let got = a_blk[i + j * n];
                        // A zero tail in one and a stale value in the other is
                        // the failure mode worth catching, so compare both.
                        assert!(
                            (got - want).abs() < 1e-12,
                            "n={n} upper={upper}: reflector tail ({i},{j}) is {got} but the unblocked reduction leaves {want}"
                        );
                    }
                }
            }
        }
    }
}
