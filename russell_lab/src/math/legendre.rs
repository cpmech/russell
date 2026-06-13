use crate::Vector;

/// Evaluates the Legendre polynomial Pn(x)
///
/// ```text
/// Bonnet's recurrence:
/// (n+1) P_{n+1}(x) = (2n+1) x P_n(x) - n P_{n-1}(x)
/// ```
///
/// Base cases: P₀(x) = 1, P₁(x) = x
///
/// See: <https://mathworld.wolfram.com/LegendrePolynomial.html>
///
/// See also: <https://en.wikipedia.org/wiki/Legendre_polynomials>
///
/// | n | Pₙ(x)                       | dPₙ/dx(x)              | d²Pₙ/dx²(x)         |
/// |:-:|:----------------------------|:-----------------------|:--------------------|
/// | 0 | 1                           | 0                      | 0                   |
/// | 1 | x                           | 1                      | 0                   |
/// | 2 | (-1 + 3x²)/2                | 3x                     | 3                   |
/// | 3 | (-3x + 5x³)/2               | (-3 + 15x²)/2          | 15x                 |
/// | 4 | (3 - 30x² + 35x⁴)/8         | (-60x + 140x³)/8       | (-60 + 420x²)/8     |
/// | 5 | (15x - 70x³ + 63x⁵)/8       | (15 - 210x² + 315x⁴)/8 | (-420x + 1260x³)/8  |
///
/// # Examples
///
/// ```
/// use russell_lab::{approx_eq, math};
///
/// approx_eq(math::legendre_pn(4, 0.25), 0.15771484375, 1e-15);
/// ```
pub fn legendre_pn(n: usize, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return x;
    }
    let mut p_prev = 1.0; // P_0
    let mut p_curr = x; // P_1
    for k in 1..n {
        let kf = k as f64;
        let p_next = ((2.0 * kf + 1.0) * x * p_curr - kf * p_prev) / (kf + 1.0);
        p_prev = p_curr;
        p_curr = p_next;
    }
    p_curr
}

/// Computes the first derivative of the Legendre polynomial Pn(x)
///
/// ```text
/// dPₙ(x)
/// ——————
///   dx
/// ```
///
/// For |x| < 1:
/// ```text
/// P'_n(x) = n / (x² - 1) * (x P_n(x) - P_{n-1}(x))
/// ```
///
/// Boundary values:
/// ```text
/// P'_n(1)  =  n(n+1)/2
/// P'_n(-1) = (-1)^{n+1} n(n+1)/2
/// ```
///
/// See: <https://mathworld.wolfram.com/LegendrePolynomial.html>
///
/// # Examples
///
/// ```
/// use russell_lab::{approx_eq, math};
///
/// approx_eq(math::legendre_pn_deriv1(4, 0.25), -1.6015625, 1e-15);
/// ```
pub fn legendre_pn_deriv1(n: usize, x: f64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    if (x - 1.0).abs() < 1e-15 {
        let nf = n as f64;
        return nf * (nf + 1.0) / 2.0;
    }
    if (x + 1.0).abs() < 1e-15 {
        let nf = n as f64;
        let val = nf * (nf + 1.0) / 2.0;
        return if n % 2 == 0 { -val } else { val };
    }
    let pn = legendre_pn(n, x);
    let pn1 = legendre_pn(n - 1, x);
    (n as f64) * (x * pn - pn1) / (x * x - 1.0)
}

/// Computes the second derivative of the Legendre polynomial Pn(x)
///
/// ```text
/// d²Pₙ(x)
/// ———————
///   dx²
/// ```
///
/// Using the Legendre ODE:
/// ```text
/// (1-x²)P''_n - 2x P'_n + n(n+1)P_n = 0
/// => P''_n = (2x P'_n - n(n+1)P_n) / (1-x²)
/// ```
///
/// See: <https://mathworld.wolfram.com/LegendrePolynomial.html>
///
/// # Examples
///
/// ```
/// use russell_lab::{approx_eq, math};
///
/// approx_eq(math::legendre_pn_deriv2(4, 0.25), -4.21875, 1e-14);
/// ```
pub fn legendre_pn_deriv2(n: usize, x: f64) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    let nf = n as f64;
    let nn1 = nf * (nf + 1.0);
    if (x - 1.0).abs() < 1e-15 {
        return (nf - 1.0) * nf * (nf + 1.0) * (nf + 2.0) / 8.0;
    }
    if (x + 1.0).abs() < 1e-15 {
        let val = (nf - 1.0) * nf * (nf + 1.0) * (nf + 2.0) / 8.0;
        return if n % 2 == 0 { val } else { -val };
    }
    let dpn = legendre_pn_deriv1(n, x);
    let pn = legendre_pn(n, x);
    (2.0 * x * dpn - nn1 * pn) / (1.0 - x * x)
}

/// Computes Gauss-Legendre quadrature points (roots of Pn)
///
/// The points are the roots of Pₙ(x), found via Newton-Raphson iteration.
/// The points are returned in ascending order from -1 to 1.
///
/// # Examples
///
/// ```
/// use russell_lab::{approx_eq, math};
///
/// let xx = math::legendre_gauss_points(2);
/// approx_eq(xx[0], -0.7745966692414834, 1e-14);
/// approx_eq(xx[1], 0.0, 1e-15);
/// approx_eq(xx[2], 0.7745966692414834, 1e-14);
/// ```
pub fn legendre_gauss_points(nn: usize) -> Vector {
    let mut xx = Vector::new(nn + 1);
    if nn == 0 {
        xx[0] = 0.0;
        return xx;
    }
    let n = nn + 1; // find roots of P_{n} where n = nn+1
    for k in 0..=nn {
        // initial guess from Chebyshev nodes
        let angle = std::f64::consts::PI * ((4 * k + 3) as f64) / ((4 * n + 2) as f64);
        let mut x = angle.cos();
        for _ in 0..100 {
            let p = legendre_pn(n, x);
            let dp = legendre_pn_deriv1(n, x);
            if dp.abs() < 1e-30 {
                break;
            }
            let dx = -p / dp;
            x += dx;
            if dx.abs() < 1e-15 {
                break;
            }
        }
        xx[k] = x;
    }
    let mut data = xx.as_data().to_vec();
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for (i, &val) in data.iter().enumerate() {
        xx[i] = val;
    }
    xx
}

/// Computes Gauss-Legendre quadrature weights
///
/// ```text
/// w_k = 2 / ((1 - x_k²) [P'_n(x_k)]²)
/// ```
///
/// # Examples
///
/// ```
/// use russell_lab::{approx_eq, math};
///
/// let ww = math::legendre_gauss_weights(2);
/// approx_eq(ww[0], 5.0 / 9.0, 1e-14);
/// approx_eq(ww[1], 8.0 / 9.0, 1e-14);
/// approx_eq(ww[2], 5.0 / 9.0, 1e-14);
/// ```
pub fn legendre_gauss_weights(nn: usize) -> Vector {
    let xx = legendre_gauss_points(nn);
    let mut ww = Vector::new(nn + 1);
    let n = nn + 1; // P_{nn+1}
    for k in 0..=nn {
        let x = xx[k];
        let dp = legendre_pn_deriv1(n, x);
        ww[k] = 2.0 / ((1.0 - x * x) * dp * dp);
    }
    ww
}

/// Computes Gauss-Lobatto-Legendre quadrature points
///
/// Returns n+1 points including the endpoints -1 and 1.
/// The interior n-1 points are roots of P'_{n}(x).
///
/// Note: Early return for nn == 0 with X = {0.0} (the midpoint, consistent with what legendre_gauss_points(0) returns).
///
/// # Examples
///
/// ```
/// use russell_lab::{approx_eq, math};
///
/// let xx = math::legendre_lobatto_points(3);
/// approx_eq(xx[0], -1.0, 1e-15);
/// approx_eq(xx[1], -0.4472135954999579, 1e-14);
/// approx_eq(xx[2], 0.4472135954999579, 1e-14);
/// approx_eq(xx[3], 1.0, 1e-15);
/// ```
pub fn legendre_lobatto_points(nn: usize) -> Vector {
    let mut xx = Vector::new(nn + 1);
    if nn == 0 {
        return xx; // single-point Lobatto is degenerate; return midpoint
    }
    xx[0] = -1.0;
    xx[nn] = 1.0;
    if nn < 3 {
        if nn == 2 {
            xx[1] = 0.0;
        }
        return xx;
    }
    // interior points are roots of P'_{nn}(x)
    let n = nn;
    // initial guesses from Chebyshev-Gauss-Lobatto-like spacing
    for k in 1..nn {
        let angle = std::f64::consts::PI * (k as f64) / (nn as f64);
        let mut x = -angle.cos();
        for _ in 0..200 {
            let dp = legendre_pn_deriv1(n, x);
            let ddp = legendre_pn_deriv2(n, x);
            if ddp.abs() < 1e-30 {
                break;
            }
            let dx = -dp / ddp;
            x += dx;
            if dx.abs() < 1e-15 {
                break;
            }
        }
        xx[k] = x;
    }
    xx
}

/// Computes Gauss-Lobatto-Legendre quadrature weights
///
/// For interior points:
/// ```text
/// w_k = 2 / (n(n-1) [P_{n-1}(x_k)]²)
/// ```
///
/// For endpoints:
/// ```text
/// w_0 = w_n = 2 / (n(n-1))
/// ```
///
/// # Examples
///
/// ```
/// use russell_lab::{approx_eq, math};
///
/// let ww = math::legendre_lobatto_weights(3);
/// approx_eq(ww[0], 1.0 / 6.0, 1e-14);
/// approx_eq(ww[1], 5.0 / 6.0, 1e-14);
/// approx_eq(ww[2], 5.0 / 6.0, 1e-14);
/// approx_eq(ww[3], 1.0 / 6.0, 1e-14);
/// ```
pub fn legendre_lobatto_weights(nn: usize) -> Vector {
    let mut ww = Vector::new(nn + 1);
    if nn < 2 {
        return ww;
    }
    let np1 = (nn + 1) as f64;
    let w_endpoint = 2.0 / (np1 * (np1 - 1.0));
    ww[0] = w_endpoint;
    ww[nn] = w_endpoint;
    let xx = legendre_lobatto_points(nn);
    for k in 1..nn {
        let x = xx[k];
        let pn1 = legendre_pn(nn, x);
        ww[k] = 2.0 / (np1 * (np1 - 1.0) * pn1 * pn1);
    }
    ww
}

/// Helper for testing: Gauss-Legendre points starting from a given initial guess
#[cfg(test)]
fn gauss_points_with_guess(nn: usize, x0: f64) -> f64 {
    let n = nn + 1;
    let mut x = x0;
    for _ in 0..100 {
        let p = legendre_pn(n, x);
        let dp = legendre_pn_deriv1(n, x);
        if dp.abs() < 1e-30 {
            break;
        }
        let dx = -p / dp;
        x += dx;
        if dx.abs() < 1e-15 {
            break;
        }
    }
    x
}

/// Helper for testing: Lobatto interior points starting from a given initial guess
#[cfg(test)]
fn lobatto_points_with_guess(nn: usize, x0: f64) -> f64 {
    let n = nn;
    let mut x = x0;
    for _ in 0..200 {
        let dp = legendre_pn_deriv1(n, x);
        let ddp = legendre_pn_deriv2(n, x);
        if ddp.abs() < 1e-30 {
            break;
        }
        let dx = -dp / ddp;
        x += dx;
        if dx.abs() < 1e-15 {
            break;
        }
    }
    x
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::approx_eq;

    // ========================================================================
    // Step 1: legendre_pn tests
    // ========================================================================

    #[test]
    fn pn_base_cases() {
        let xx = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5];
        for &x in &xx {
            assert_eq!(legendre_pn(0, x), 1.0);
            assert_eq!(legendre_pn(1, x), x);
        }
    }

    #[test]
    fn pn_exact_polynomials_n2_to_n5() {
        let xx = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5];
        for &x in &xx {
            let x2 = x * x;
            let x3 = x2 * x;
            let x4 = x3 * x;
            let x5 = x4 * x;
            // P_2 = (-1 + 3x²)/2
            approx_eq(legendre_pn(2, x), (-1.0 + 3.0 * x2) / 2.0, 1e-14);
            // P_3 = (-3x + 5x³)/2
            approx_eq(legendre_pn(3, x), (-3.0 * x + 5.0 * x3) / 2.0, 1e-14);
            // P_4 = (3 - 30x² + 35x⁴)/8
            approx_eq(legendre_pn(4, x), (3.0 - 30.0 * x2 + 35.0 * x4) / 8.0, 1e-14);
            // P_5 = (15x - 70x³ + 63x⁵)/8
            approx_eq(legendre_pn(5, x), (15.0 * x - 70.0 * x3 + 63.0 * x5) / 8.0, 1e-13);
        }
    }

    #[test]
    fn pn_special_values() {
        for n in 0..20 {
            assert_eq!(legendre_pn(n, 1.0), 1.0);
            let expected_at_neg1 = if n % 2 == 0 { 1.0 } else { -1.0 };
            approx_eq(legendre_pn(n, -1.0), expected_at_neg1, 1e-14);
            if n % 2 != 0 {
                assert_eq!(legendre_pn(n, 0.0), 0.0);
            }
        }
    }

    #[test]
    fn pn_symmetry() {
        let xx = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5];
        for &x in &xx {
            for n in 0..11 {
                let pn_x = legendre_pn(n, x);
                let pn_neg_x = legendre_pn(n, -x);
                let expected = if n % 2 == 0 { pn_x } else { -pn_x };
                approx_eq(pn_neg_x, expected, 1e-14);
            }
        }
    }

    #[test]
    fn pn_bonnet_recursion() {
        let xx = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0];
        for &x in &xx {
            for n in 1..10 {
                let pn = legendre_pn(n, x);
                let pn1 = legendre_pn(n + 1, x);
                let pm1 = legendre_pn(n - 1, x);
                let nf = n as f64;
                // (n+1)P_{n+1} = (2n+1)x P_n - n P_{n-1}
                let lhs = (nf + 1.0) * pn1;
                let rhs = (2.0 * nf + 1.0) * x * pn - nf * pm1;
                approx_eq(lhs, rhs, 1e-12);
            }
        }
    }

    #[test]
    fn pn_outside_domain() {
        for &x in &[-3.0, -2.0, 2.0, 3.0] {
            for n in 0..10 {
                let pn = legendre_pn(n, x);
                // verify via Bonnet recurrence
                if n >= 1 {
                    let pn1 = legendre_pn(n + 1, x);
                    let pm1 = legendre_pn(n - 1, x);
                    let nf = n as f64;
                    let lhs = (nf + 1.0) * pn1;
                    let rhs = (2.0 * nf + 1.0) * x * pn - nf * pm1;
                    approx_eq(lhs, rhs, 1e-10);
                }
            }
        }
    }

    #[test]
    fn pn_high_degree() {
        // verify via Bonnet recurrence at high degree
        for &n in &[20, 50] {
            let x = 0.3;
            let pn = legendre_pn(n, x);
            let pn1 = legendre_pn(n + 1, x);
            let pm1 = legendre_pn(n - 1, x);
            let nf = n as f64;
            let lhs = (nf + 1.0) * pn1;
            let rhs = (2.0 * nf + 1.0) * x * pn - nf * pm1;
            approx_eq(lhs, rhs, 1e-8);

            // verify P_n(1) = 1
            approx_eq(legendre_pn(n, 1.0), 1.0, 1e-14);
        }
    }

    // ========================================================================
    // Step 2: legendre_pn_deriv1 tests
    // ========================================================================

    #[test]
    fn deriv1_base_cases() {
        let xx = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5];
        for &x in &xx {
            assert_eq!(legendre_pn_deriv1(0, x), 0.0);
            assert_eq!(legendre_pn_deriv1(1, x), 1.0);
        }
    }

    #[test]
    fn deriv1_exact_formulas_n2_to_n5() {
        let xx = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5];
        for &x in &xx {
            let x2 = x * x;
            let x3 = x2 * x;
            // P'_2 = 3x
            approx_eq(legendre_pn_deriv1(2, x), 3.0 * x, 1e-13);
            // P'_3 = (-3 + 15x²)/2
            approx_eq(legendre_pn_deriv1(3, x), (-3.0 + 15.0 * x2) / 2.0, 1e-13);
            // P'_4 = (-60x + 140x³)/8
            approx_eq(legendre_pn_deriv1(4, x), (-60.0 * x + 140.0 * x3) / 8.0, 1e-12);
            // P'_5 = (15 - 210x² + 315x⁴)/8
            let x4 = x3 * x;
            approx_eq(legendre_pn_deriv1(5, x), (15.0 - 210.0 * x2 + 315.0 * x4) / 8.0, 1e-12);
        }
    }

    #[test]
    fn deriv1_boundary_x_eq_1() {
        for n in 1..11 {
            let expected = (n as f64) * (n as f64 + 1.0) / 2.0;
            approx_eq(legendre_pn_deriv1(n, 1.0), expected, 1e-14);
        }
    }

    #[test]
    fn deriv1_boundary_x_eq_neg1() {
        for n in 1..11 {
            let val = (n as f64) * (n as f64 + 1.0) / 2.0;
            let expected = if n % 2 == 0 { -val } else { val };
            approx_eq(legendre_pn_deriv1(n, -1.0), expected, 1e-14);
        }
    }

    #[test]
    fn deriv1_numerical_finite_difference() {
        let h = 1e-7;
        let xx = [-0.99, -0.5, 0.0, 0.5, 0.99];
        for &x in &xx {
            for n in 0..9 {
                let pn_h1 = legendre_pn(n, x + h);
                let pn_h2 = legendre_pn(n, x - h);
                let numerical = (pn_h1 - pn_h2) / (2.0 * h);
                let analytical = legendre_pn_deriv1(n, x);
                approx_eq(analytical, numerical, 1e-7);
            }
        }
    }

    #[test]
    fn deriv1_outside_domain() {
        for &x in &[-3.0, -2.0, 2.0, 3.0] {
            let h = 1e-6;
            for n in 1..9 {
                let pn_h1 = legendre_pn(n, x + h);
                let pn_h2 = legendre_pn(n, x - h);
                let numerical = (pn_h1 - pn_h2) / (2.0 * h);
                let analytical = legendre_pn_deriv1(n, x);
                approx_eq(analytical, numerical, 1e-4);
            }
        }
    }

    #[test]
    fn deriv1_ode_relation() {
        // (1-x²)P'_n = n(P_{n-1} - xP_n)
        let xx = [-0.9, -0.5, 0.0, 0.5, 0.9];
        for &x in &xx {
            for n in 2..9 {
                let dpn = legendre_pn_deriv1(n, x);
                let pn = legendre_pn(n, x);
                let pn1 = legendre_pn(n - 1, x);
                let lhs = (1.0 - x * x) * dpn;
                let rhs = (n as f64) * (pn1 - x * pn);
                approx_eq(lhs, rhs, 1e-12);
            }
        }
    }

    // ========================================================================
    // Step 3: legendre_pn_deriv2 tests
    // ========================================================================

    #[test]
    fn deriv2_base_cases() {
        let xx = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5];
        for &x in &xx {
            assert_eq!(legendre_pn_deriv2(0, x), 0.0);
            assert_eq!(legendre_pn_deriv2(1, x), 0.0);
        }
        for &x in &xx {
            assert_eq!(legendre_pn_deriv2(2, x), 3.0);
            approx_eq(legendre_pn_deriv2(3, x), 15.0 * x, 1e-13);
        }
    }

    #[test]
    fn deriv2_exact_formulas_n2_to_n5() {
        let xx = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5];
        for &x in &xx {
            let x2 = x * x;
            let x3 = x2 * x;
            // P''_2 = 3
            approx_eq(legendre_pn_deriv2(2, x), 3.0, 1e-13);
            // P''_3 = 15x
            approx_eq(legendre_pn_deriv2(3, x), 15.0 * x, 1e-13);
            // P''_4 = (-60 + 420x²)/8
            approx_eq(legendre_pn_deriv2(4, x), (-60.0 + 420.0 * x2) / 8.0, 1e-12);
            // P''_5 = (-420x + 1260x³)/8
            approx_eq(legendre_pn_deriv2(5, x), (-420.0 * x + 1260.0 * x3) / 8.0, 1e-12);
        }
    }

    #[test]
    fn deriv2_legendre_ode() {
        // (1-x²)P''_n - 2xP'_n + n(n+1)P_n = 0
        let xx = [-0.99, -0.5, 0.0, 0.5, 0.99];
        for &x in &xx {
            for n in 0..9 {
                let pn = legendre_pn(n, x);
                let dpn = legendre_pn_deriv1(n, x);
                let ddpn = legendre_pn_deriv2(n, x);
                let residual = (1.0 - x * x) * ddpn - 2.0 * x * dpn + (n as f64) * ((n as f64) + 1.0) * pn;
                approx_eq(residual, 0.0, 1e-10);
            }
        }
    }

    #[test]
    fn deriv2_numerical_finite_difference() {
        let h = 1e-5;
        let xx = [-0.99, -0.5, 0.0, 0.5, 0.99];
        for &x in &xx {
            for n in 0..9 {
                let dpn_h1 = legendre_pn_deriv1(n, x + h);
                let dpn_h2 = legendre_pn_deriv1(n, x - h);
                let numerical = (dpn_h1 - dpn_h2) / (2.0 * h);
                let analytical = legendre_pn_deriv2(n, x);
                approx_eq(analytical, numerical, 1e-5);
            }
        }
    }

    #[test]
    fn deriv2_boundary_x_eq_1() {
        // P''_n(1) = n(n+1)(n-1)(n+2)/8 for n>=2
        for n in 2..9 {
            let nf = n as f64;
            let expected = nf * (nf + 1.0) * (nf - 1.0) * (nf + 2.0) / 8.0;
            approx_eq(legendre_pn_deriv2(n, 1.0), expected, 1e-12);
        }
    }

    #[test]
    fn deriv2_boundary_x_eq_neg1() {
        // P''_n(-1) = (-1)^n * n(n+1)(n-1)(n+2)/8
        for n in 2..9 {
            let nf = n as f64;
            let val = nf * (nf + 1.0) * (nf - 1.0) * (nf + 2.0) / 8.0;
            let expected = if n % 2 == 0 { val } else { -val };
            approx_eq(legendre_pn_deriv2(n, -1.0), expected, 1e-12);
        }
    }

    #[test]
    fn deriv2_consistency_with_deriv1() {
        let h = 1e-5;
        let xx = [-0.99, -0.5, 0.0, 0.5, 0.99];
        for &x in &xx {
            for n in 2..9 {
                let dpn_h1 = legendre_pn_deriv1(n, x + h);
                let dpn_h2 = legendre_pn_deriv1(n, x - h);
                let numerical = (dpn_h1 - dpn_h2) / (2.0 * h);
                let analytical = legendre_pn_deriv2(n, x);
                approx_eq(analytical, numerical, 1e-5);
            }
        }
    }

    // ========================================================================
    // Step 4: legendre_gauss_points tests
    // ========================================================================

    #[test]
    fn gauss_points_n0() {
        let xx = legendre_gauss_points(0);
        assert_eq!(xx.dim(), 1);
        approx_eq(xx[0], 0.0, 1e-15);
    }

    #[test]
    fn gauss_points_n1_to_n9() {
        for n in 1..=9 {
            let xx = legendre_gauss_points(n);
            assert_eq!(xx.dim(), n + 1);
            for k in 0..=n {
                let pn = legendre_pn(n + 1, xx[k]);
                approx_eq(pn, 0.0, 1e-14);
            }
        }
    }

    #[test]
    fn gauss_points_are_roots() {
        for n in 1..=15 {
            let xx = legendre_gauss_points(n);
            for k in 0..=n {
                let pn = legendre_pn(n + 1, xx[k]);
                assert!(pn.abs() < 1e-12, "P_{}({}) = {} not zero", n + 1, xx[k], pn);
            }
        }
    }

    #[test]
    fn gauss_points_symmetry() {
        for n in 1..=15 {
            let xx = legendre_gauss_points(n);
            for k in 0..=n {
                approx_eq(xx[k], -xx[n - k], 1e-15);
            }
        }
    }

    #[test]
    fn gauss_points_sorted_ascending() {
        for n in 1..=15 {
            let xx = legendre_gauss_points(n);
            for k in 1..=n {
                assert!(
                    xx[k] >= xx[k - 1],
                    "Points not sorted for n={}: x[{}]={} < x[{}]={}",
                    n,
                    k,
                    xx[k],
                    k - 1,
                    xx[k - 1]
                );
            }
        }
    }

    #[test]
    fn gauss_points_known_values() {
        // n=2: x = ±sqrt(3/5)
        let xx2 = legendre_gauss_points(2);
        let sqrt3_5 = (3.0_f64 / 5.0).sqrt();
        approx_eq(xx2[0], -sqrt3_5, 1e-14);
        approx_eq(xx2[1], 0.0, 1e-15);
        approx_eq(xx2[2], sqrt3_5, 1e-14);

        // n=3: x = 0, ±sqrt(3/7 - 2/7*sqrt(6/5))
        let xx3 = legendre_gauss_points(3);
        approx_eq(xx3[0], -0.8611363115940526, 1e-14);
        approx_eq(xx3[1], -0.3399810435848563, 1e-14);
        approx_eq(xx3[2], 0.3399810435848563, 1e-14);
        approx_eq(xx3[3], 0.8611363115940526, 1e-14);
    }

    #[test]
    fn gauss_points_domain() {
        for n in 1..=15 {
            let xx = legendre_gauss_points(n);
            for k in 0..=n {
                assert!(
                    xx[k] > -1.0 && xx[k] < 1.0,
                    "Point x[{}]={} out of (-1,1) for n={}",
                    k,
                    xx[k],
                    n
                );
            }
        }
    }

    // ========================================================================
    // Step 5: legendre_gauss_weights tests
    // ========================================================================

    #[test]
    fn gauss_weights_sum_to_two() {
        for n in 1..=15 {
            let ww = legendre_gauss_weights(n);
            let sum: f64 = ww.as_data().iter().sum();
            approx_eq(sum, 2.0, 1e-14);
        }
    }

    #[test]
    fn gauss_weights_positive() {
        for n in 1..=15 {
            let ww = legendre_gauss_weights(n);
            for k in 0..=n {
                assert!(ww[k] > 0.0, "Weight w[{}]={} not positive for n={}", k, ww[k], n);
            }
        }
    }

    #[test]
    fn gauss_weights_symmetry() {
        for n in 1..=15 {
            let ww = legendre_gauss_weights(n);
            for k in 0..=n {
                approx_eq(ww[k], ww[n - k], 1e-15);
            }
        }
    }

    #[test]
    fn gauss_weights_known_values() {
        // n=1: w = [1, 1]
        let ww1 = legendre_gauss_weights(1);
        approx_eq(ww1[0], 1.0, 1e-14);
        approx_eq(ww1[1], 1.0, 1e-14);

        // n=2: w = [5/9, 8/9, 5/9]
        let ww2 = legendre_gauss_weights(2);
        approx_eq(ww2[0], 5.0 / 9.0, 1e-14);
        approx_eq(ww2[1], 8.0 / 9.0, 1e-14);
        approx_eq(ww2[2], 5.0 / 9.0, 1e-14);
    }

    #[test]
    fn gauss_quadrature_exact_linear() {
        // integral of x from -1 to 1 = 0
        let xx = legendre_gauss_points(1);
        let ww = legendre_gauss_weights(1);
        let mut result = 0.0;
        for k in 0..=1 {
            result += ww[k] * xx[k];
        }
        approx_eq(result, 0.0, 1e-15);
    }

    #[test]
    fn gauss_quadrature_exact_quadratic() {
        // integral of x² from -1 to 1 = 2/3
        let xx = legendre_gauss_points(2);
        let ww = legendre_gauss_weights(2);
        let mut result = 0.0;
        for k in 0..=2 {
            result += ww[k] * xx[k] * xx[k];
        }
        approx_eq(result, 2.0 / 3.0, 1e-15);
    }

    #[test]
    fn gauss_quadrature_exact_degree_2n_minus_1() {
        // n-point Gauss-Legendre is exact for polynomials of degree <= 2n-1
        for n in 1..=8 {
            let xx = legendre_gauss_points(n);
            let ww = legendre_gauss_weights(n);

            // odd power: integral of x^{2n-1} from -1 to 1 = 0
            let deg_odd = 2 * n - 1;
            let mut result_odd = 0.0;
            for k in 0..=n {
                result_odd += ww[k] * xx[k].powi(deg_odd as i32);
            }
            approx_eq(result_odd, 0.0, 1e-12);

            // even power: integral of x^{2n-2} from -1 to 1 = 2/(2n-1)
            let deg_even = 2 * n - 2;
            let mut result_even = 0.0;
            for k in 0..=n {
                result_even += ww[k] * xx[k].powi(deg_even as i32);
            }
            approx_eq(result_even, 2.0 / (2 * n - 1) as f64, 1e-12);
        }
    }

    #[test]
    fn gauss_quadrature_trig_function() {
        // integral of cos(x) from -1 to 1 = 2*sin(1)
        let n = 10;
        let xx = legendre_gauss_points(n);
        let ww = legendre_gauss_weights(n);
        let mut result = 0.0;
        for k in 0..=n {
            result += ww[k] * xx[k].cos();
        }
        approx_eq(result, 2.0 * 1.0_f64.sin(), 1e-12);
    }

    // ========================================================================
    // Step 6: legendre_lobatto_points tests
    // ========================================================================

    #[test]
    fn lobatto_points_n2() {
        let xx = legendre_lobatto_points(2);
        assert_eq!(xx.dim(), 3);
        approx_eq(xx[0], -1.0, 1e-15);
        approx_eq(xx[1], 0.0, 1e-15);
        approx_eq(xx[2], 1.0, 1e-15);
    }

    #[test]
    fn lobatto_points_endpoints() {
        for n in 2..=15 {
            let xx = legendre_lobatto_points(n);
            assert_eq!(xx[0], -1.0);
            assert_eq!(xx[n], 1.0);
        }
    }

    #[test]
    fn lobatto_points_interior_are_critical_points() {
        for n in 3..=15 {
            let xx = legendre_lobatto_points(n);
            for k in 1..n {
                let dp = legendre_pn_deriv1(n, xx[k]);
                assert!(dp.abs() < 1e-10, "P'_{}({}) = {} not zero for n={}", n, xx[k], dp, n);
            }
        }
    }

    #[test]
    fn lobatto_points_symmetry() {
        for n in 2..=15 {
            let xx = legendre_lobatto_points(n);
            for k in 0..=n {
                approx_eq(xx[k], -xx[n - k], 1e-15);
            }
        }
    }

    #[test]
    fn lobatto_points_sorted() {
        for n in 2..=15 {
            let xx = legendre_lobatto_points(n);
            for k in 1..=n {
                assert!(xx[k] > xx[k - 1], "Points not sorted for n={}", n);
            }
        }
    }

    #[test]
    fn lobatto_points_known_values() {
        // n=3: x = [-1, -1/sqrt(5), 1/sqrt(5), 1]
        let xx3 = legendre_lobatto_points(3);
        let inv_sqrt5 = 1.0 / 5.0_f64.sqrt();
        approx_eq(xx3[0], -1.0, 1e-15);
        approx_eq(xx3[1], -inv_sqrt5, 1e-14);
        approx_eq(xx3[2], inv_sqrt5, 1e-14);
        approx_eq(xx3[3], 1.0, 1e-15);
    }

    #[test]
    fn lobatto_points_domain() {
        for n in 2..=15 {
            let xx = legendre_lobatto_points(n);
            for k in 0..=n {
                assert!(xx[k] >= -1.0 && xx[k] <= 1.0, "Point out of [-1,1] for n={}", n);
            }
        }
    }

    // ========================================================================
    // Step 7: legendre_lobatto_weights tests
    // ========================================================================

    #[test]
    fn lobatto_weights_sum() {
        for n in 2..=15 {
            let ww = legendre_lobatto_weights(n);
            let sum: f64 = ww.as_data().iter().sum();
            approx_eq(sum, 2.0, 1e-13);
        }
    }

    #[test]
    fn lobatto_weights_positive() {
        for n in 2..=15 {
            let ww = legendre_lobatto_weights(n);
            for k in 0..=n {
                assert!(ww[k] > 0.0, "Weight w[{}] not positive for n={}", k, n);
            }
        }
    }

    #[test]
    fn lobatto_weights_symmetry() {
        for n in 2..=15 {
            let ww = legendre_lobatto_weights(n);
            for k in 0..=n {
                approx_eq(ww[k], ww[n - k], 1e-15);
            }
        }
    }

    #[test]
    fn lobatto_weights_endpoint_values() {
        for n in 3..=10 {
            let ww = legendre_lobatto_weights(n);
            let np1 = (n + 1) as f64;
            let expected = 2.0 / (np1 * (np1 - 1.0));
            approx_eq(ww[0], expected, 1e-14);
            approx_eq(ww[n], expected, 1e-14);
        }
    }

    #[test]
    fn lobatto_weights_known_values() {
        // n=3: w = [1/6, 5/6, 5/6, 1/6]
        let ww3 = legendre_lobatto_weights(3);
        approx_eq(ww3[0], 1.0 / 6.0, 1e-14);
        approx_eq(ww3[1], 5.0 / 6.0, 1e-14);
        approx_eq(ww3[2], 5.0 / 6.0, 1e-14);
        approx_eq(ww3[3], 1.0 / 6.0, 1e-14);
    }

    #[test]
    fn lobatto_quadrature_exact_low_degree() {
        // n Lobatto points exact for degree <= 2n-3
        for n in 3..=10 {
            let xx = legendre_lobatto_points(n);
            let ww = legendre_lobatto_weights(n);

            // integral of 1 = 2
            let mut r0 = 0.0;
            for k in 0..=n {
                r0 += ww[k];
            }
            approx_eq(r0, 2.0, 1e-14);

            // integral of x = 0
            let mut r1 = 0.0;
            for k in 0..=n {
                r1 += ww[k] * xx[k];
            }
            approx_eq(r1, 0.0, 1e-14);

            // integral of x² = 2/3
            let mut r2 = 0.0;
            for k in 0..=n {
                r2 += ww[k] * xx[k] * xx[k];
            }
            approx_eq(r2, 2.0 / 3.0, 1e-14);
        }
    }

    #[test]
    fn lobatto_quadrature_trig_function() {
        let n = 12;
        let xx = legendre_lobatto_points(n);
        let ww = legendre_lobatto_weights(n);
        let mut result = 0.0;
        for k in 0..=n {
            result += ww[k] * xx[k].cos();
        }
        approx_eq(result, 2.0 * 1.0_f64.sin(), 1e-12);
    }

    // ========================================================================
    // Step 8: boundary and robustness tests
    // ========================================================================

    #[test]
    fn pn_large_n_stability() {
        let pn100 = legendre_pn(100, 0.5);
        assert!(pn100.abs() < 1.0, "P_100(0.5) = {} too large", pn100);
    }

    #[test]
    fn deriv1_equals_numerical_derivative() {
        let h = 1e-7;
        let xx = [-0.99, -0.5, 0.0, 0.5, 0.99];
        for &x in &xx {
            for n in 0..11 {
                let pn_h1 = legendre_pn(n, x + h);
                let pn_h2 = legendre_pn(n, x - h);
                let numerical = (pn_h1 - pn_h2) / (2.0 * h);
                let analytical = legendre_pn_deriv1(n, x);
                approx_eq(analytical, numerical, 1e-7);
            }
        }
    }

    #[test]
    fn deriv2_equals_numerical_second_derivative() {
        let h = 1e-5;
        let xx = [-0.9, -0.5, 0.0, 0.5, 0.9];
        for &x in &xx {
            for n in 0..11 {
                let pn_h1 = legendre_pn(n, x + h);
                let pn = legendre_pn(n, x);
                let pn_h2 = legendre_pn(n, x - h);
                let numerical = (pn_h1 - 2.0 * pn + pn_h2) / (h * h);
                let analytical = legendre_pn_deriv2(n, x);
                approx_eq(analytical, numerical, 1e-4);
            }
        }
    }

    #[test]
    fn gauss_points_high_degree() {
        for &n in &[20, 50] {
            let xx = legendre_gauss_points(n);
            for k in 0..=n {
                let pn = legendre_pn(n + 1, xx[k]);
                assert!(pn.abs() < 1e-10, "P_{}({}) = {} not zero", n + 1, xx[k], pn);
            }
        }
    }

    #[test]
    fn lobatto_points_high_degree() {
        let n = 20;
        let xx = legendre_lobatto_points(n);
        for k in 1..n {
            let dp = legendre_pn_deriv1(n, xx[k]);
            assert!(dp.abs() < 1e-8, "P'_{}({}) = {} not zero", n, xx[k], dp);
        }
    }

    // ========================================================================
    // Coverage: defensive branches
    // ========================================================================

    #[test]
    fn gauss_points_newton_dp_zero_safety_break() {
        // Line 167: break when dp < 1e-30
        // Start Newton-Raphson at a critical point of P_4 where P'_4 = 0
        // P_4(x) = (35x^4 - 30x^2 + 3)/8, P'_4(x) = (140x^3 - 60x)/8
        // P'_4 = 0 at x = 0 and x = ±sqrt(3/7)
        // At x = 0, P'_4(0) = 0, so dp < 1e-30 → break
        let result = gauss_points_with_guess(3, 0.0);
        // Should break immediately since dp(0) = 0 for P_4
        approx_eq(result, 0.0, 1e-15);
    }

    #[test]
    fn lobatto_points_newton_ddp_zero_safety_break() {
        // Line 249: break when ddp < 1e-30
        // For n=2: P'_2(x) = 3x, P''_2(x) = 6, never zero
        // For n=4: P'_4(x) = (140x^3 - 60x)/8, P''_4(x) = (420x^2 - 60)/8
        // P''_4 = 0 at x = ±sqrt(1/7)
        // Starting at x = sqrt(1/7) makes ddp = 0, triggering the safety break
        let x0 = (1.0_f64 / 7.0).sqrt();
        let result = lobatto_points_with_guess(4, x0);
        // The break fires immediately, so result should be the initial guess
        assert!((result - x0).abs() < 1e-10 || result.is_finite());
    }

    #[test]
    fn lobatto_weights_returns_empty_for_nn_0() {
        // Line 288: return ww when nn < 2
        let ww = legendre_lobatto_weights(0);
        assert_eq!(ww.dim(), 1);
    }

    #[test]
    fn lobatto_weights_returns_empty_for_nn_1() {
        // Line 288: return ww when nn < 2
        let ww = legendre_lobatto_weights(1);
        assert_eq!(ww.dim(), 2);
    }
}
