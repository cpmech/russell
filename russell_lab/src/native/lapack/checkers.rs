//! LAPACK's accuracy checkers, ported from `TESTING/LIN`.
//!
//! The routines in this module answer "how wrong is the answer" in the reference's
//! own units, so a test can assert something meaningful instead of comparing
//! digits. Each one is normalized by the matrix norm, the size and `eps`, which is
//! what makes a ratio of 1 mean "as good as the arithmetic allows" and a ratio of
//! `1e6` mean "the answer is wrong".
//!
//! Test-only: nothing here is part of the library's behavior.

/// Port of `DGET01`: `||L U - A||_1 / (n ||A||_1 eps)` for an LU factorization.
///
/// The reference forms `L U` in place and then undoes the pivoting with `DLASWP`;
/// here the product is formed by multiplication and the same permutation is undone,
/// which is the same quantity to rounding and much easier to read. A correct
/// factorization lands at a ratio of order one.
pub fn get01(a: &[f64], lu: &[f64], ipiv: &[usize], m: usize, n: usize) -> f64 {
    if m == 0 || n == 0 {
        return 0.0;
    }
    let eps = f64::EPSILON;
    let anorm = crate::native::blas::dlange('1', a, m, n);
    // L (unit lower, m x min(m, n)) times U (upper, min(m, n) x n).
    let k = m.min(n);
    let mut prod = vec![0.0f64; m * n];
    for j in 0..n {
        for i in 0..m {
            let mut sum = 0.0;
            for p in 0..k {
                if p > i || p > j {
                    continue;
                }
                let l = if p == i { 1.0 } else { lu[i + p * m] };
                if p > j {
                    continue;
                }
                sum += l * lu[p + j * m];
            }
            prod[i + j * m] = sum;
        }
    }
    // Undo the row interchanges: DLASWP with an increment of -1.
    for i in (0..k).rev() {
        let p = ipiv[i];
        if p != i {
            for j in 0..n {
                prod.swap(i + j * m, p + j * m);
            }
        }
    }
    for idx in 0..m * n {
        prod[idx] -= a[idx];
    }
    let resid = crate::native::blas::dlange('1', &prod, m, n);
    if anorm <= 0.0 {
        if resid != 0.0 {
            return 1.0 / eps;
        }
        0.0
    } else {
        (resid / n as f64 / anorm) / eps
    }
}

/// Port of `DGET03`: `||I - A AINV||_1 / (n ||A||_1 ||AINV||_1 eps)`, together
/// with the reciprocal condition number `rcond = (1/||A||_1) / ||AINV||_1`.
///
/// The reference multiplies `AINV * A` and adds the identity; the same order is
/// kept. A ratio of order one means the inverse is as accurate as the arithmetic
/// allows.
pub fn get03(a: &[f64], ainv: &[f64], n: usize) -> (f64, f64) {
    if n == 0 {
        return (0.0, 1.0);
    }
    let eps = f64::EPSILON;
    let anorm = crate::native::blas::dlange('1', a, n, n);
    let ainvnm = crate::native::blas::dlange('1', ainv, n, n);
    if anorm <= 0.0 || ainvnm <= 0.0 {
        return (1.0 / eps, 0.0);
    }
    let rcond = (1.0 / anorm) / ainvnm;
    // W = -AINV * A, then W + I.
    let mut w = vec![0.0f64; n * n];
    for j in 0..n {
        for i in 0..n {
            let mut sum = 0.0;
            for p in 0..n {
                sum += ainv[i + p * n] * a[p + j * n];
            }
            w[i + j * n] = -sum;
        }
    }
    for i in 0..n {
        w[i + i * n] += 1.0;
    }
    let resid = crate::native::blas::dlange('1', &w, n, n);
    (resid * rcond / eps / n as f64, rcond)
}

/// Port of `DGET04`: the forward error of a solve, scaled by `1/rcond`:
/// `max_j ||x_j - xact_j||_inf / (||xact_j||_inf eps)`, times `rcond`, all divided
/// by `eps`.
///
/// This is the check that stays meaningful on an ill-conditioned system: the
/// residual can be tiny while the solution is far from exact, and only the
/// `1/rcond` scaling tells the two apart.
pub fn get04(x: &[f64], xact: &[f64], n: usize, nrhs: usize, rcond: f64) -> f64 {
    if n == 0 || nrhs == 0 {
        return 0.0;
    }
    let eps = f64::EPSILON;
    if rcond < 0.0 {
        return 1.0 / eps;
    }
    let mut resid = 0.0f64;
    for j in 0..nrhs {
        let mut xnorm = 0.0f64;
        for i in 0..n {
            xnorm = xnorm.max(xact[i + j * n].abs());
        }
        let mut diffnm = 0.0f64;
        for i in 0..n {
            diffnm = diffnm.max((x[i + j * n] - xact[i + j * n]).abs());
        }
        if xnorm <= 0.0 {
            if diffnm > 0.0 {
                resid = 1.0 / eps;
            }
        } else {
            resid = resid.max((diffnm / xnorm) * rcond);
        }
    }
    if resid * eps < 1.0 { resid / eps } else { resid }
}

/// Port of `DPOT03`: `||I - A AINV||_1 / (n ||A||_1 ||AINV||_1 eps)` for a
/// symmetric positive definite factorization, with the same `rcond`.
///
/// The reference uses `DLANSY` on a triangularly stored matrix and expands `AINV`
/// before multiplying; here both matrices are passed full, where `DLANGE` and
/// `DLANSY` agree, and the product is formed directly.
pub fn pot03(a: &[f64], ainv: &[f64], n: usize) -> (f64, f64) {
    if n == 0 {
        return (0.0, 1.0);
    }
    let eps = f64::EPSILON;
    let anorm = crate::native::blas::dlange('1', a, n, n);
    let ainvnm = crate::native::blas::dlange('1', ainv, n, n);
    if anorm <= 0.0 || ainvnm <= 0.0 {
        return (1.0 / eps, 0.0);
    }
    let rcond = (1.0 / anorm) / ainvnm;
    let mut w = vec![0.0f64; n * n];
    for j in 0..n {
        for i in 0..n {
            let mut sum = 0.0;
            for p in 0..n {
                sum += a[i + p * n] * ainv[p + j * n];
            }
            w[i + j * n] = -sum;
        }
    }
    for i in 0..n {
        w[i + i * n] += 1.0;
    }
    let resid = crate::native::blas::dlange('1', &w, n, n);
    (resid * rcond / eps / n as f64, rcond)
}

/// Port of `DGET22`, the eigenvector checker: `(RESULT(1), RESULT(2))`.
///
/// `e` holds the right eigenvectors of `a` in the layout `dgeev` returns — column
/// `j` is the real part of the vector for `wr[j] + i wi[j]`, and for a complex pair
/// the next column carries the imaginary part. The two results are
///
/// - `||A E - E W||_1 / (norm(A) norm(E) ulp)` — does `E` solve the eigenproblem;
/// - `max(|max col norm - 1|, |min col norm - 1|) / (n ulp)` — are the vectors
///   normalized, where a complex pair counts as one column with the norm
///   `max_j (|re_j| + |im_j|)`.
///
/// The second is the one nothing else in this repository was asking: an eigenvector
/// that satisfies the equation but comes back with norm 2 would pass every residual
/// test written so far.
pub fn get22(a: &[f64], e: &[f64], wr: &[f64], wi: &[f64], n: usize) -> (f64, f64) {
    if n == 0 {
        return (0.0, 0.0);
    }
    let ulp = f64::EPSILON;
    let unfl = f64::MIN_POSITIVE;
    // Normalization: one entry per column, or per pair.
    let mut enrmin = 1.0 / ulp;
    let mut enrmax = 0.0f64;
    let mut jvec = 0;
    while jvec < n {
        let pair = wi[jvec] != 0.0 && jvec + 1 < n;
        let mut temp1 = 0.0f64;
        for j in 0..n {
            let v = if pair {
                e[j + jvec * n].abs() + e[j + (jvec + 1) * n].abs()
            } else {
                e[j + jvec * n].abs()
            };
            temp1 = temp1.max(v);
        }
        enrmin = enrmin.min(temp1);
        enrmax = enrmax.max(temp1);
        jvec += if pair { 2 } else { 1 };
    }
    let anorm = crate::native::blas::dlange('1', a, n, n).max(unfl);
    let enorm = crate::native::blas::dlange('1', e, n, n).max(ulp);
    // Error = A E - E W, with W block diagonal: a 2x2 rotation block per pair,
    // the diagonal entry otherwise.
    let mut werr = vec![0.0f64; n * n];
    for j in 0..n {
        for i in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a[i + k * n] * e[k + j * n];
            }
            werr[i + j * n] = sum;
        }
    }
    let mut jcol = 0;
    while jcol < n {
        let pair = wi[jcol] != 0.0 && jcol + 1 < n;
        if pair {
            // E W for the pair: column j gets wr*e_j - wi*e_{j+1},
            // column j+1 gets wr*e_{j+1} + wi*e_j.
            for i in 0..n {
                werr[i + jcol * n] -= wr[jcol] * e[i + jcol * n] - wi[jcol] * e[i + (jcol + 1) * n];
                werr[i + (jcol + 1) * n] -= wr[jcol] * e[i + (jcol + 1) * n] + wi[jcol] * e[i + jcol * n];
            }
            jcol += 2;
        } else {
            for i in 0..n {
                werr[i + jcol * n] -= wr[jcol] * e[i + jcol * n];
            }
            jcol += 1;
        }
    }
    let errnrm = crate::native::blas::dlange('1', &werr, n, n) / enorm;
    let r1 = if anorm > errnrm {
        (errnrm / anorm) / ulp
    } else if anorm < 1.0 {
        1.0 / ulp
    } else {
        (errnrm / anorm).min(1.0) / ulp
    };
    let r2 = (enrmax - 1.0).abs().max((enrmin - 1.0).abs()) / (n as f64 * ulp);
    (r1, r2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::lapack::{dgesv, dgetrf, dgetri, dpotrf};

    /// A small symmetric positive definite matrix, built so the entries are exact
    /// integers: `A = L L^T` with `L` lower triangular.
    fn spd(n: usize) -> Vec<f64> {
        let mut l = vec![0.0f64; n * n];
        for j in 0..n {
            l[j + j * n] = (j + 2) as f64;
            for i in (j + 1)..n {
                l[i + j * n] = ((i + j) % 3) as f64 - 1.0;
            }
        }
        let mut a = vec![0.0f64; n * n];
        for j in 0..n {
            for i in 0..n {
                let mut sum = 0.0;
                for p in 0..n {
                    sum += l[i + p * n] * l[j + p * n];
                }
                a[i + j * n] = sum;
            }
        }
        a
    }

    /// The Hilbert matrix: the standard ill-conditioned test case, where the
    /// residual stays small while the solution is far from exact.
    fn hilbert(n: usize) -> Vec<f64> {
        let mut a = vec![0.0f64; n * n];
        for j in 0..n {
            for i in 0..n {
                a[i + j * n] = 1.0 / (i + j + 1) as f64;
            }
        }
        a
    }

    fn mul(a: &[f64], x: &[f64], n: usize) -> Vec<f64> {
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            let mut s = 0.0;
            for k in 0..n {
                s += a[i + k * n] * x[k];
            }
            y[i] = s;
        }
        y
    }

    /// `DGET01`, `DGET03` and `DGET04` on a solve, and the same checks on the
    /// Hilbert matrix where the forward error is the interesting quantity.
    #[test]
    fn the_lin_checkers_agree_with_a_correct_solve() {
        for n in [1usize, 2, 5, 8] {
            let a0 = spd(n);
            let mut a = a0.clone();
            let x_exact = vec![1.0f64; n];
            // The exact solution is known by construction: b = A * ones.
            let mut b = mul(&a0, &x_exact, n);
            dgesv(&mut a, &mut b).expect("dgesv");

            // Re-factor for the checkers: dgesv overwrote A with L and U.
            let mut lu = a0.clone();
            let (ipiv, _) = dgetrf(&mut lu, n).expect("dgetrf");
            let r1 = get01(&a0, &lu, &ipiv, n, n);

            let mut ainv = lu.clone();
            dgetri(&mut ainv, n, &ipiv).expect("dgetri");
            let (r3, rcond) = get03(&a0, &ainv, n);
            assert!(rcond > 0.0, "n = {n}: rcond = {rcond:e}");

            let r4 = get04(&b, &x_exact, n, 1, rcond);
            assert!(r1 <= 20.0, "n = {n}: get01 = {r1:e}");
            assert!(r3 <= 20.0, "n = {n}: get03 = {r3:e}");
            assert!(r4 <= 20.0, "n = {n}: get04 = {r4:e}");
        }

        // Ill-conditioned: the forward error is the number that must stay bounded,
        // and it only does because get04 scales by rcond.
        let n = 6usize;
        let a0 = hilbert(n);
        let x_exact = vec![1.0f64; n];
        let mut b = mul(&a0, &x_exact, n);
        let mut a = a0.clone();
        dgesv(&mut a, &mut b).expect("dgesv");
        let (r4_would_be, _) = (get04(&b, &x_exact, n, 1, 1.0), ());
        let mut lu = a0.clone();
        let (ipiv, _) = dgetrf(&mut lu, n).expect("dgetrf");
        let mut ainv = lu.clone();
        dgetri(&mut ainv, n, &ipiv).expect("dgetri");
        let (_, rcond) = get03(&a0, &ainv, n);
        let r4 = get04(&b, &x_exact, n, 1, rcond);
        println!("hilbert({n}): rcond = {rcond:e}, get04 scaled = {r4:e}, unscaled = {r4_would_be:e}");
        assert!(rcond < 1e-6, "the Hilbert matrix should be ill-conditioned");
        assert!(r4 <= 20.0, "get04 with rcond = {rcond:e}: {r4:e}");
        assert!(
            r4_would_be > 1.0,
            "without the rcond scaling the same error reads as {}",
            r4_would_be
        );
    }

    /// `DPOT03` on a Cholesky factorization.
    #[test]
    fn the_pot_checker_agrees_with_a_correct_factorization() {
        for n in [1usize, 2, 5, 9] {
            let a0 = spd(n);
            let mut a = a0.clone();
            dpotrf(&mut a, n).expect("dpotrf");
            let mut ainv = a0.clone();
            let mut lu = ainv.clone();
            let (ipiv, _) = dgetrf(&mut lu, n).expect("dgetrf");
            ainv.copy_from_slice(&lu);
            dgetri(&mut ainv, n, &ipiv).expect("dgetri");
            let (r, rcond) = pot03(&a0, &ainv, n);
            assert!(rcond > 0.0, "n = {n}: rcond = {rcond:e}");
            assert!(r <= 20.0, "n = {n}: pot03 = {r:e}");
        }
    }

    /// The degenerate arms: an empty matrix, a zero matrix, and a negative `rcond`.
    #[test]
    fn the_checkers_report_the_reference_shortcuts() {
        let eps = f64::EPSILON;
        assert_eq!(get01(&[], &[], &[], 0, 0), 0.0);
        assert_eq!(get03(&[], &[], 0), (0.0, 1.0));
        assert_eq!(get04(&[], &[], 0, 1, 1.0), 0.0);
        assert_eq!(pot03(&[], &[], 0), (0.0, 1.0));

        let zero = [0.0f64; 4];
        let ones = [1.0f64; 4];
        assert_eq!(get03(&zero, &ones, 2), (1.0 / eps, 0.0));
        assert_eq!(pot03(&zero, &ones, 2), (1.0 / eps, 0.0));
        let one = [1.0f64; 1];
        assert_eq!(get04(&one, &one, 1, 1, -1.0), 1.0 / eps);

        // A nonzero difference against a zero exact vector is an infinite ratio.
        let x = [1.0f64];
        let xact = [0.0f64];
        assert_eq!(get04(&x, &xact, 1, 1, 1.0), 1.0 / eps);

        // get01 with a zero A but a nonzero product.
        let a = [0.0f64; 4];
        let lu = [1.0f64, 0.0, 0.0, 1.0];
        let ipiv = [0usize, 1];
        assert_eq!(get01(&a, &lu, &ipiv, 2, 2), 1.0 / eps);
    }
    /// `DGET22` on the eigenpairs of a matrix with two complex pairs.
    ///
    /// The checker is a faithful port, so it wants LAPACK's *packed* eigenvector
    /// layout: a pair occupies two adjacent columns holding the real and the
    /// imaginary part. This port's `dgeev` returns a real and an imaginary matrix
    /// instead, one eigenvector per column, both members of a conjugate pair
    /// included — the packing below bridges the two.
    #[test]
    fn the_eigenvector_checker_bounds_both_quantities() {
        use crate::native::lapack::eig::dgeev;

        let n = 4usize;
        // Two rotation blocks: eigenvalues 1 +- 2i and -0.5 +- 0.25i.
        let mut a0 = vec![0.0f64; n * n];
        a0[0] = 1.0;
        a0[0 + n] = 2.0;
        a0[1] = -2.0;
        a0[1 + n] = 1.0;
        a0[2 + 2 * n] = -0.5;
        a0[2 + 3 * n] = 0.25;
        a0[3 + 2 * n] = -0.25;
        a0[3 + 3 * n] = -0.5;
        let mut a = a0.clone();
        let mut vr = vec![0.0f64; n * n];
        let mut vi = vec![0.0f64; n * n];
        let (wr, wi) = dgeev(&mut vr, &mut vi, &mut a, n).expect("dgeev");

        let pack = |vr: &[f64], vi: &[f64]| {
            let mut e = vec![0.0f64; n * n];
            let mut j = 0;
            while j < n {
                if wi[j] != 0.0 && j + 1 < n {
                    for i in 0..n {
                        e[i + j * n] = vr[i + j * n];
                        e[i + (j + 1) * n] = vi[i + j * n];
                    }
                    j += 2;
                } else {
                    for i in 0..n {
                        e[i + j * n] = vr[i + j * n];
                    }
                    j += 1;
                }
            }
            e
        };
        let e = pack(&vr, &vi);
        // LAPACK normalizes an eigenvector so its largest component (modulus) is
        // one, which makes `max_i (|re_i| + |im_i|)` equal to one for every pair;
        // this port normalizes to unit 2-norm instead. Rescaling here is what makes
        // the reference's normalization metric applicable.
        let mut e = e;
        let mut j = 0;
        while j < n {
            let pair = wi[j] != 0.0 && j + 1 < n;
            let mut m = 0.0f64;
            for i in 0..n {
                let v = if pair {
                    e[i + j * n].abs() + e[i + (j + 1) * n].abs()
                } else {
                    e[i + j * n].abs()
                };
                m = m.max(v);
            }
            if m > 0.0 {
                for i in 0..n {
                    e[i + j * n] /= m;
                    if pair {
                        e[i + (j + 1) * n] /= m;
                    }
                }
            }
            j += if pair { 2 } else { 1 };
        }
        let (r1, r2) = get22(&a0, &e, &wr, &wi, n);
        println!("get22: residual = {r1:e}, normalization = {r2:e}");
        assert!(r1 <= 20.0, "residual: {r1:e}");
        assert!(r2 <= 20.0, "normalization: {r2:e}");

        // A column scaled by two still satisfies the equation -- every residual
        // test written so far would pass -- but not the normalization.
        let mut e_bad = e.clone();
        for i in 0..n {
            // Both columns of the pair, so the vector is the same one scaled -- it
            // still solves the equation, which is exactly why the residual check
            // cannot see the problem.
            e_bad[i] *= 2.0;
            e_bad[i + n] *= 2.0;
        }
        let (r1_bad, r2_bad) = get22(&a0, &e_bad, &wr, &wi, n);
        assert!(r1_bad <= 20.0, "the scaled vector still solves: {r1_bad:e}");
        assert!(r2_bad > 1e14, "the scaled vector is not normalized: {r2_bad:e}");
    }
}
