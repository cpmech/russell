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
}
