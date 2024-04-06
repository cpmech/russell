use super::PI;

/// Evaluates the Chebyshev polynomial of first kind Tn(x) using the trigonometric functions
///
/// ```text
///         ⎧ (-1)ⁿ cosh[n⋅acosh(-x)]   if x < -1
/// Tₙ(x) = ⎨       cosh[n⋅acosh( x)]   if x > 1
///         ⎩       cos [n⋅acos ( x)]   if |x| ≤ 1
/// ```
///
/// See: <https://mathworld.wolfram.com/ChebyshevPolynomialoftheFirstKind.html>
///
/// See also: <https://en.wikipedia.org/wiki/Chebyshev_polynomials>
///
/// | n | Tₙ(x)               | dTₙ/dx(x)         | d²Tₙ/dx²(x)     |
/// |:-:|:--------------------|:------------------|:----------------|
/// | 0 | 1                   | 0                 | 0               |
/// | 1 | x                   | 1                 | 0               |
/// | 2 | -1 + 2 x²           | 4 x               | 4               |
/// | 3 | -3 x + 4 x³         | -3 + 12 x²        | 24 x            |
/// | 4 | 1 - 8 x² + 8 x⁴     | -16 x + 32 x³     | -16 + 96 x²     |
/// | 5 | 5 x - 20 x³ + 16 x⁵ | 5 - 60 x² + 80 x⁴ | -120 x + 320 x³ |
/// |...| ...                 | ...               | ....            |
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
/// This is the first derivative of [chebyshev_tn]
///
/// See: <https://mathworld.wolfram.com/ChebyshevPolynomialoftheFirstKind.html>
///
/// See also: <https://en.wikipedia.org/wiki/Chebyshev_polynomials>
///
/// | n | Tₙ(x)               | dTₙ/dx(x)         | d²Tₙ/dx²(x)     |
/// |:-:|:--------------------|:------------------|:----------------|
/// | 0 | 1                   | 0                 | 0               |
/// | 1 | x                   | 1                 | 0               |
/// | 2 | -1 + 2 x²           | 4 x               | 4               |
/// | 3 | -3 x + 4 x³         | -3 + 12 x²        | 24 x            |
/// | 4 | 1 - 8 x² + 8 x⁴     | -16 x + 32 x³     | -16 + 96 x²     |
/// | 5 | 5 x - 20 x³ + 16 x⁵ | 5 - 60 x² + 80 x⁴ | -120 x + 320 x³ |
/// |...| ...                 | ...               | ....            |
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
/// This is the second derivative of [chebyshev_tn] and the first derivative of [chebyshev_tn_deriv1]
///
/// See: <https://mathworld.wolfram.com/ChebyshevPolynomialoftheFirstKind.html>
///
/// See also: <https://en.wikipedia.org/wiki/Chebyshev_polynomials>
///
/// | n | Tₙ(x)               | dTₙ/dx(x)         | d²Tₙ/dx²(x)     |
/// |:-:|:--------------------|:------------------|:----------------|
/// | 0 | 1                   | 0                 | 0               |
/// | 1 | x                   | 1                 | 0               |
/// | 2 | -1 + 2 x²           | 4 x               | 4               |
/// | 3 | -3 x + 4 x³         | -3 + 12 x²        | 24 x            |
/// | 4 | 1 - 8 x² + 8 x⁴     | -16 x + 32 x³     | -16 + 96 x²     |
/// | 5 | 5 x - 20 x³ + 16 x⁵ | 5 - 60 x² + 80 x⁴ | -120 x + 320 x³ |
/// |...| ...                 | ...               | ....            |
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

/// Computes Chebyshev-Gauss points with symmetry
///
/// ```text
///             ⎛  (2i+1)⋅π  ⎞
/// X[i] = -cos ⎜ —————————— ⎟
///             ⎝   2N + 2   ⎠
///
/// i = 0 ... N
/// ```
pub fn chebyshev_gauss_points(nn: usize) -> Vec<f64> {
    let mut xx = vec![0.0; nn + 1];
    let nf = nn as f64;
    let d = 2.0 * nf + 2.0;
    let l = if (nn & 1) == 0 {
        // even number of segments
        nn / 2
    } else {
        // odd number of segments
        (nn + 3) / 2 - 1
    };
    for i in 0..l {
        xx[nn - i] = f64::cos(((2 * i + 1) as f64) * PI / d);
        xx[i] = -xx[nn - i];
    }
    xx
}

/// Computes Chebyshev-Gauss-Lobatto points with symmetry
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
pub fn chebyshev_lobatto_points(nn: usize) -> Vec<f64> {
    let mut xx = vec![0.0; nn + 1];
    xx[0] = -1.0;
    xx[nn] = 1.0;
    if nn < 3 {
        return xx;
    }
    let nf = nn as f64;
    let d = 2.0 * nf;
    let l = if (nn & 1) == 0 {
        // even number of segments
        nn / 2
    } else {
        // odd number of segments
        (nn + 3) / 2 - 1
    };
    for i in 1..l {
        xx[nn - i] = f64::sin(PI * (nf - 2.0 * (i as f64)) / d);
        xx[i] = -xx[nn - i];
    }
    xx
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{
        chebyshev_gauss_points, chebyshev_lobatto_points, chebyshev_tn, chebyshev_tn_deriv1, chebyshev_tn_deriv2,
    };
    use crate::math::{SQRT_2, SQRT_3};
    use crate::{approx_eq, vec_approx_eq};

    /// Checks the symmetry of segments in a set of points
    fn check_segment_symmetry(xx: &[f64]) {
        if xx.len() < 2 {
            panic!("the length of the array must be at least 2");
        }
        let l = xx.len() - 1; // last
        if -xx[0] != xx[l] {
            panic!("first and last coordinates must be equal with opposite signs");
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
    fn check_segment_symmetry_works_ok_1() {
        check_segment_symmetry(&[-1.0, -0.2, 0.0, 0.2, 1.0]);
        check_segment_symmetry(&[-1.0, -0.2, 0.2, 1.0]);
    }

    #[test]
    #[should_panic(expected = "the length of the array must be at least 2")]
    fn check_segment_symmetry_works_bad_1() {
        check_segment_symmetry(&[-1.0]);
    }

    #[test]
    #[should_panic(expected = "first and last coordinates must be equal with opposite signs")]
    fn check_segment_symmetry_works_bad_2() {
        check_segment_symmetry(&[-1.0, -0.4, 0.0, 0.2, -1.0]);
    }

    #[test]
    #[should_panic(expected = "dxa must be equal to dxb")]
    fn check_segment_symmetry_works_bad_3() {
        check_segment_symmetry(&[-1.0, -0.4, 0.0, 0.2, 1.0]);
    }

    #[test]
    #[should_panic(expected = "dxa must be equal to dxb")]
    fn check_segment_symmetry_works_bad_4() {
        check_segment_symmetry(&[-1.0, -0.4, 0.2, 1.0]);
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

    #[test]
    fn chebyshev_gauss_points_works() {
        let xx = chebyshev_gauss_points(1);
        let xx_ref = vec![-1.0 / SQRT_2, 1.0 / SQRT_2];
        vec_approx_eq(&xx, &xx_ref, 1e-15);
        check_segment_symmetry(&xx);

        let xx = chebyshev_gauss_points(2);
        let xx_ref = vec![-SQRT_3 / 2.0, 0.0, SQRT_3 / 2.0];
        vec_approx_eq(&xx, &xx_ref, 1e-15);
        check_segment_symmetry(&xx);

        let xx = chebyshev_gauss_points(3);
        let xx_ref = vec![
            -9.238795325112867e-01,
            -3.826834323650898e-01,
            3.826834323650897e-01,
            9.238795325112867e-01,
        ];
        vec_approx_eq(&xx, &xx_ref, 1e-15);
        check_segment_symmetry(&xx);

        let xx = chebyshev_gauss_points(4);
        let xx_ref = vec![
            -9.510565162951535e-01,
            -5.877852522924731e-01,
            -6.123233995736766e-17,
            5.877852522924730e-01,
            9.510565162951535e-01,
        ];
        vec_approx_eq(&xx, &xx_ref, 1e-15);
        check_segment_symmetry(&xx);

        let xx = chebyshev_gauss_points(5);
        let xx_ref = vec![
            -9.659258262890683e-01,
            -7.071067811865476e-01,
            -2.588190451025210e-01,
            2.588190451025206e-01,
            7.071067811865475e-01,
            9.659258262890682e-01,
        ];
        vec_approx_eq(&xx, &xx_ref, 1e-15);
        check_segment_symmetry(&xx);

        let xx = chebyshev_gauss_points(6);
        let xx_ref = vec![
            -9.749279121818236e-01,
            -7.818314824680298e-01,
            -4.338837391175582e-01,
            -6.123233995736766e-17,
            4.338837391175581e-01,
            7.818314824680298e-01,
            9.749279121818236e-01,
        ];
        vec_approx_eq(&xx, &xx_ref, 1e-15);
        check_segment_symmetry(&xx);

        let xx = chebyshev_gauss_points(7);
        let xx_ref = vec![
            -9.807852804032304e-01,
            -8.314696123025452e-01,
            -5.555702330196023e-01,
            -1.950903220161283e-01,
            1.950903220161282e-01,
            5.555702330196020e-01,
            8.314696123025453e-01,
            9.807852804032304e-01,
        ];
        vec_approx_eq(&xx, &xx_ref, 1e-15);
        check_segment_symmetry(&xx);

        let xx = chebyshev_gauss_points(8);
        let xx_ref = vec![
            -9.848077530122080e-01,
            -8.660254037844387e-01,
            -6.427876096865394e-01,
            -3.420201433256688e-01,
            -6.123233995736766e-17,
            3.420201433256687e-01,
            6.427876096865394e-01,
            8.660254037844387e-01,
            9.848077530122080e-01,
        ];
        vec_approx_eq(&xx, &xx_ref, 1e-15);
        check_segment_symmetry(&xx);

        let xx = chebyshev_gauss_points(9);
        let xx_ref = vec![
            -9.876883405951378e-01,
            -8.910065241883679e-01,
            -7.071067811865476e-01,
            -4.539904997395468e-01,
            -1.564344650402309e-01,
            1.564344650402308e-01,
            4.539904997395467e-01,
            7.071067811865475e-01,
            8.910065241883678e-01,
            9.876883405951377e-01,
        ];
        vec_approx_eq(&xx, &xx_ref, 1e-15);
        check_segment_symmetry(&xx);
    }

    #[test]
    fn chebyshev_lobatto_points_works() {
        let xx = chebyshev_lobatto_points(1);
        let xx_ref = vec![-1.0, 1.0];
        vec_approx_eq(&xx, &xx_ref, 1e-15);
        check_segment_symmetry(&xx);

        let xx = chebyshev_lobatto_points(2);
        let xx_ref = vec![-1.0, 0.0, 1.0];
        vec_approx_eq(&xx, &xx_ref, 1e-15);
        check_segment_symmetry(&xx);

        let xx = chebyshev_lobatto_points(3);
        let xx_ref = vec![-1.0, -0.5, 0.5, 1.0];
        vec_approx_eq(&xx, &xx_ref, 1e-15);
        check_segment_symmetry(&xx);

        let xx = chebyshev_lobatto_points(4);
        let xx_ref = vec![
            -1.000000000000000e+00,
            -7.071067811865476e-01,
            -6.123233995736766e-17,
            7.071067811865475e-01,
            1.000000000000000e+00,
        ];
        vec_approx_eq(&xx, &xx_ref, 1e-15);
        check_segment_symmetry(&xx);

        let xx = chebyshev_lobatto_points(5);
        let xx_ref = vec![
            -1.000000000000000e+00,
            -8.090169943749475e-01,
            -3.090169943749475e-01,
            3.090169943749473e-01,
            8.090169943749473e-01,
            1.000000000000000e+00,
        ];
        vec_approx_eq(&xx, &xx_ref, 1e-15);
        check_segment_symmetry(&xx);

        let xx = chebyshev_lobatto_points(6);
        let xx_ref = vec![
            -1.000000000000000e+00,
            -8.660254037844387e-01,
            -5.000000000000001e-01,
            -6.123233995736766e-17,
            4.999999999999998e-01,
            8.660254037844385e-01,
            1.000000000000000e+00,
        ];
        vec_approx_eq(&xx, &xx_ref, 1e-15);
        check_segment_symmetry(&xx);

        let xx = chebyshev_lobatto_points(7);
        let xx_ref = vec![
            -1.000000000000000e+00,
            -9.009688679024191e-01,
            -6.234898018587336e-01,
            -2.225209339563144e-01,
            2.225209339563143e-01,
            6.234898018587335e-01,
            9.009688679024190e-01,
            1.000000000000000e+00,
        ];
        vec_approx_eq(&xx, &xx_ref, 1e-15);
        check_segment_symmetry(&xx);

        let xx = chebyshev_lobatto_points(8);
        let xx_ref = vec![
            -1.000000000000000e+00,
            -9.238795325112867e-01,
            -7.071067811865476e-01,
            -3.826834323650898e-01,
            -6.123233995736766e-17,
            3.826834323650897e-01,
            7.071067811865475e-01,
            9.238795325112867e-01,
            1.000000000000000e+00,
        ];
        vec_approx_eq(&xx, &xx_ref, 1e-15);
        check_segment_symmetry(&xx);

        let xx = chebyshev_lobatto_points(9);
        let xx_ref = vec![
            -1.000000000000000e+00,
            -9.396926207859084e-01,
            -7.660444431189780e-01,
            -5.000000000000001e-01,
            -1.736481776669304e-01,
            1.736481776669303e-01,
            4.999999999999998e-01,
            7.660444431189779e-01,
            9.396926207859083e-01,
            1.000000000000000e+00,
        ];
        vec_approx_eq(&xx, &xx_ref, 1e-15);
        check_segment_symmetry(&xx);
    }
}
