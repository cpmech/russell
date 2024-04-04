#![allow(unused, non_snake_case)]

use super::PI;

/// Evaluates the Chebyshev polynomial of first kind Tn(x) using the trigonometric functions
///
/// ```text
///         ⎧ (-1)ⁿ cosh[n⋅acosh(-x)]   if x < -1
/// Tₙ(x) = ⎨       cosh[n⋅acosh( x)]   if x > 1
///         ⎩       cos [n⋅acos ( x)]   if |x| ≤ 1
/// ```
///
/// | n | Tₙ(x)               | dTₙ/dx(x)         | d²Tₙ/dx²(x)     |
/// |:-:|:--------------------|:------------------|:----------------|
/// | 0 | 1                   | 0                 | 0               |
/// | 1 | x                   | 1                 | 0               |
/// | 2 | -1 + 2 x²           | 4 x               | 4               |
/// | 3 | -3 x + 4 x³         | -3 + 12 x²        | 24 x            |
/// | 4 | 1 - 8 x² + 8 x⁴     | -16 x + 32 x³     | -16 + 96 x²     |
/// | 5 | 5 x - 20 x³ + 16 x⁵ | 5 - 60 x² + 80 x⁴ | -120 x + 320 x³ |
pub fn chebyshev_tn(n: usize, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return x;
    }
    let p = n as f64;
    if x < -1.0 {
        if (n & 1) == 0 {
            // n is even
            return f64::cosh(p * f64::acosh(-x));
        }
        return -f64::cosh(p * f64::acosh(-x));
    }
    if x > 1.0 {
        return f64::cosh(p * f64::acosh(x));
    }
    f64::cos(p * f64::acos(x))
}

/// Computes the first derivative of the Chebyshev T(n, x) function
///
/// ```text
/// dTₙ(x)
/// ——————
///   dx
/// ```
///
/// | n | Tₙ(x)               | dTₙ/dx(x)         | d²Tₙ/dx²(x)     |
/// |:-:|:--------------------|:------------------|:----------------|
/// | 0 | 1                   | 0                 | 0               |
/// | 1 | x                   | 1                 | 0               |
/// | 2 | -1 + 2 x²           | 4 x               | 4               |
/// | 3 | -3 x + 4 x³         | -3 + 12 x²        | 24 x            |
/// | 4 | 1 - 8 x² + 8 x⁴     | -16 x + 32 x³     | -16 + 96 x²     |
/// | 5 | 5 x - 20 x³ + 16 x⁵ | 5 - 60 x² + 80 x⁴ | -120 x + 320 x³ |
pub fn chebyshev_tn_deriv1(n: usize, x: f64) -> f64 {
    let p = n as f64;
    if x > -1.0 && x < 1.0 {
        let t = f64::acos(x);
        let d1 = -p * f64::sin(p * t); // derivatives of cos(n⋅t) with respect to t
        let s = f64::sin(t);
        return -d1 / s;
    }
    if x == 1.0 {
        return p * p;
    }
    if x == -1.0 {
        if (n & 1) == 0 {
            // n is even ⇒ n+1 is odd
            return -p * p;
        }
        return p * p; // n is odd ⇒ n+1 is even
    }
    if x < -1.0 {
        return -f64::powf(-1.0, p) * (p * f64::sinh(p * f64::acosh(-x))) / f64::sqrt(x * x - 1.0);
    }
    // x > +1
    (p * f64::sinh(p * f64::acosh(x))) / f64::sqrt(x * x - 1.0)
}

/// Computes the second derivative of the Chebyshev Tn(x) function
///
/// Returns:
///
/// ```text
/// d²Tₙ(x)
/// ———————
///   dx²
/// ```
///
/// | n | Tₙ(x)               | dTₙ/dx(x)         | d²Tₙ/dx²(x)     |
/// |:-:|:--------------------|:------------------|:----------------|
/// | 0 | 1                   | 0                 | 0               |
/// | 1 | x                   | 1                 | 0               |
/// | 2 | -1 + 2 x²           | 4 x               | 4               |
/// | 3 | -3 x + 4 x³         | -3 + 12 x²        | 24 x            |
/// | 4 | 1 - 8 x² + 8 x⁴     | -16 x + 32 x³     | -16 + 96 x²     |
/// | 5 | 5 x - 20 x³ + 16 x⁵ | 5 - 60 x² + 80 x⁴ | -120 x + 320 x³ |
pub fn chebyshev_tn_deriv2(n: usize, x: f64) -> f64 {
    let p = n as f64;
    let pp = p * p;
    let t = f64::acos(x);
    if x > -1.0 && x < 1.0 {
        let d1 = -p * f64::sin(p * t); // derivatives of cos(n⋅t) with respect to t
        let d2 = -pp * f64::cos(p * t);
        let c = f64::cos(t);
        let s = f64::sin(t);
        return (s * d2 - c * d1) / (s * s * s);
    }
    if x == -1.0 {
        if (n & 1) != 0 {
            // n is odd
            return -(pp * pp - pp) / 3.0;
        }
        return (pp * pp - pp) / 3.0;
    }
    if x == 1.0 {
        return (pp * pp - pp) / 3.0;
    }
    let d = x * x - 1.0;
    if x < -1.0 {
        let r = (pp * f64::cosh(p * f64::acosh(-x))) / d + (p * x * f64::sinh(p * f64::acosh(-x))) / f64::powf(d, 1.5);
        if (n & 1) == 0 {
            // n is even
            return r;
        }
        return -r;
    }
    // x > +1
    (pp * f64::cosh(p * f64::acosh(x))) / d - (p * x * f64::sinh(p * f64::acosh(x))) / f64::powf(d, 1.5)
}

/// Computes Chebyshev-Gauss points considering symmetry
///
/// ```text
///             ⎛  (2i+1)⋅π  ⎞
/// X[i] = -cos ⎜ —————————— ⎟
///             ⎝   2N + 2   ⎠
///
/// i = 0 ... N
/// ```
pub fn chebyshev_gauss_points(N: usize) -> Vec<f64> {
    let mut X = vec![0.0; N + 1];
    let n = N as f64;
    let d = 2.0 * n + 2.0;
    if (N & 1) == 0 {
        // even number of segments
        let l = N / 2;
        for i in 0..l {
            X[N - i] = f64::cos(((2 * i + 1) as f64) * PI / d);
            X[i] = -X[N - i];
        }
    } else {
        // odd number of segments
        let l = (N + 3) / 2 - 1;
        for i in 0..l {
            X[N - i] = f64::cos(((2 * i + 1) as f64) * PI / d);
            if i < l {
                X[i] = -X[N - i];
            }
        }
    }
    X
}

/// Computes Chebyshev-Gauss-Lobatto points considering symmetry
///
/// Uses the sin(x) function:
///
/// ```text
///             ⎛  π⋅(N-2i)  ⎞
/// X[i] = -sin ⎜ —————————— ⎟
///             ⎝    2⋅N     ⎠
///
/// i = 0 ... N
/// ```
///
/// Another option is:
///
/// ```text
///             ⎛  i⋅π  ⎞
/// X[i] = -cos ⎜ ————— ⎟
///             ⎝   N   ⎠
///
/// i = 0 ... N
/// ```
///
/// # Reference
///
/// * Baltensperger R and Trummer MR (2003) Spectral differencing with a twist,
///   SIAM Journal Scientific Computation 24(5):1465-1487
pub fn chebyshev_lobatto_points(N: usize) -> Vec<f64> {
    let mut X = vec![0.0; N + 1];
    X[0] = -1.0;
    X[N] = 1.0;
    if N < 3 {
        return X;
    }
    let n = N as f64;
    let d = 2.0 * n;
    if (N & 1) == 0 {
        // even number of segments
        let l = N / 2;
        for i in 1..l {
            X[N - i] = f64::sin(PI * (n - 2.0 * (i as f64)) / d);
            X[i] = -X[N - i];
        }
    } else {
        // odd number of segments
        let l = (N + 3) / 2 - 1;
        for i in 1..l {
            X[N - i] = f64::sin(PI * (n - 2.0 * (i as f64)) / d);
            if i < l {
                X[i] = -X[N - i];
            }
        }
    }
    X
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{chebyshev_tn, chebyshev_tn_deriv1, chebyshev_tn_deriv2};
    use crate::approx_eq;

    /// Checks the symmetry of segments in a set of points
    fn check_segment_symmetry(xx: &[f64]) {
        if xx.len() < 2 {
            panic!("the length of the array must be at least 2");
        }
        let l = xx.len() - 1; // last
        if -xx[0] != xx[l] {
            panic!("first and last coordinates must be equal one with another");
        }
        let even = l % 2 == 0;
        let mut i_max = l / 2 + 1;
        if !even {
            i_max = (l + 1) / 2;
        }
        for i in 1..i_max {
            let dxa = xx[i] - xx[i - 1];
            let dxb = xx[l - i + 1] - xx[l - i];
            if dxa != dxb {
                panic!("dxa must be equal to dxb");
            }
        }
    }

    #[test]
    fn chebyshev_t_and_derivatives_work() {
        let nn = 5;
        let mut xx: Vec<_> = (0..(nn + 1))
            .into_iter()
            .map(|i| -1.5 + (i as f64) * 3.0 / (nn as f64))
            .collect();
        xx.push(-1.0);
        xx.push(1.0);
        for x in xx {
            println!("x = {:?}, T3(x) = {:?}", x, chebyshev_tn(3, x));
            // n = 0
            assert_eq!(chebyshev_tn(0, x), 1.0);
            assert_eq!(chebyshev_tn_deriv1(0, x), 0.0);
            assert_eq!(chebyshev_tn_deriv2(0, x), 0.0);
            // n = 1
            assert_eq!(chebyshev_tn(1, x), x);
            assert_eq!(chebyshev_tn_deriv1(1, x), 1.0);
            assert_eq!(chebyshev_tn_deriv2(1, x), 0.0);
            // n = 2
            let x2 = x * x;
            approx_eq(chebyshev_tn(2, x), -1.0 + 2.0 * x2, 1e-15);
            approx_eq(chebyshev_tn_deriv1(2, x), 4.0 * x, 1e-15);
            approx_eq(chebyshev_tn_deriv2(2, x), 4.0, 1e-14);
            // n = 3
            let x3 = x * x2;
            approx_eq(chebyshev_tn(3, x), -3.0 * x + 4.0 * x3, 1e-14);
            approx_eq(chebyshev_tn_deriv1(3, x), -3.0 + 12.0 * x2, 1e-13);
            approx_eq(chebyshev_tn_deriv2(3, x), 24.0 * x, 1e-13);
            // n = 4
            let x4 = x * x3;
            approx_eq(chebyshev_tn(4, x), 1.0 - 8.0 * x2 + 8.0 * x4, 1e-14);
            approx_eq(chebyshev_tn_deriv1(4, x), -16.0 * x + 32.0 * x3, 1e-13);
            approx_eq(chebyshev_tn_deriv2(4, x), -16.0 + 96.0 * x2, 1e-13);
        }
    }
}
