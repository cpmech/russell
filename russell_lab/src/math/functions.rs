/// Evaluates the sign function
///
/// ```text
///           │ -1   if x < 0
/// sign(x) = ┤  0   if x = 0
///           │  1   if x > 0
///
///           |x|    x
/// sign(x) = ——— = ———
///            x    |x|
///
/// sign(x) = 2 · heaviside(x) - 1
/// ```
///
/// Reference: <https://en.wikipedia.org/wiki/Sign_function>
#[inline]
pub fn sign(x: f64) -> f64 {
    if x < 0.0 {
        -1.0
    } else if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

/// Evaluates the ramp function (Macaulay brackets)
///
/// ```text
/// ramp(x) = │ 0   if x < 0
///           │ x   otherwise
///
/// ramp(x) = max(x, 0)
///
///           x + |x|
/// ramp(x) = ———————
///              2
///
/// ramp(x) =〈x〉  (Macaulay brackets)
///
/// ramp(x) = x · heaviside(x)
/// ```
///
/// Reference: <https://en.wikipedia.org/wiki/Ramp_function>
#[inline]
pub fn ramp(x: f64) -> f64 {
    if x < 0.0 {
        0.0
    } else {
        x
    }
}

/// Evaluates the Heaviside step function (derivative of ramp(x))
///
/// ```text
///                │ 0    if x < 0
/// heaviside(x) = ┤ 1/2  if x = 0
///                │ 1    if x > 0
///
/// heaviside(x) = ½ + ½ · sign(x)
/// ```
///
/// Reference: <https://en.wikipedia.org/wiki/Heaviside_step_function>
#[inline]
pub fn heaviside(x: f64) -> f64 {
    if x < 0.0 {
        0.0
    } else if x > 0.0 {
        1.0
    } else {
        0.5
    }
}

/// Evaluates the boxcar function
///
/// ```text
///                 │ 0    if x < a or  x > b
/// boxcar(x;a,b) = ┤ 1/2  if x = a or  x = b
///                 │ 1    if x > a and x < b
///
/// boxcar(x;a,b) = heaviside(x-a) - heaviside(x-b)
/// ```
///
/// Note: `a ≤ x ≤ b` with `b ≥ a` **not** being checked.
///
/// Reference: <https://en.wikipedia.org/wiki/Boxcar_function>
#[inline]
pub fn boxcar(x: f64, a: f64, b: f64) -> f64 {
    if x < a || x > b {
        0.0
    } else if x > a && x < b {
        1.0
    } else {
        0.5
    }
}

/// Evaluates the standard logistic function
///
/// ```text
///                   1
/// logistic(x) = ———————————
///               1 + exp(-x)
/// ```
///
/// Reference: <https://en.wikipedia.org/wiki/Logistic_function>
#[inline]
pub fn logistic(x: f64) -> f64 {
    1.0 / (1.0 + f64::exp(-x))
}

/// Returns the first derivative of the standard logistic function
///
/// Reference: <https://en.wikipedia.org/wiki/Logistic_function>
#[inline]
pub fn logistic_deriv(x: f64) -> f64 {
    let f = logistic(x);
    f * (1.0 - f)
}

/// Evaluates the smooth ramp function
///
/// ```text
///                  │ 0   if -β·x > 500
///                  |
/// smooth_ramp(x) = │     log(1 + exp(-β·x))
///                  │ x + ——————————————————  otherwise
///                  │            β
/// ```
#[inline]
pub fn smooth_ramp(x: f64, beta: f64) -> f64 {
    if -beta * x > 500.0 {
        return 0.0;
    }
    x + f64::ln(1.0 + f64::exp(-beta * x)) / beta
}

/// Returns the first derivative of smooth_ramp
#[inline]
pub fn smooth_ramp_deriv(x: f64, beta: f64) -> f64 {
    if -beta * x > 500.0 {
        return 0.0;
    }
    return 1.0 / (1.0 + f64::exp(-beta * x));
}

/// Returns the second derivative of smooth_ramp
#[inline]
pub fn smooth_ramp_deriv2(x: f64, beta: f64) -> f64 {
    if beta * x > 500.0 {
        return 0.0;
    }
    beta * f64::exp(beta * x) / f64::powf(f64::exp(beta * x) + 1.0, 2.0)
}

/// Evaluates the superquadric function involving sin(x)
///
/// ```text
/// suq_sin(x;k) = sign(sin(x)) · |sin(x)|ᵏ
/// ```
///
/// `suq_sin(x;k)` is the `f(ω;m)` function from <https://en.wikipedia.org/wiki/Superquadrics>
#[inline]
pub fn suq_sin(x: f64, k: f64) -> f64 {
    sign(f64::sin(x)) * f64::powf(f64::abs(f64::sin(x)), k)
}

/// Evaluates the superquadric function involving cos(x)
///
/// ```text
/// suq_cos(x;k) = sign(cos(x)) · |cos(x)|ᵏ
/// ```
///
/// `suq_cos(x;k)` is the `g(ω;m)` function from <https://en.wikipedia.org/wiki/Superquadrics>
#[inline]
pub fn suq_cos(x: f64, k: f64) -> f64 {
    sign(f64::cos(x)) * f64::powf(f64::abs(f64::cos(x)), k)
}

const FACTORIAL_22: [f64; 23] = [
    1.0,                      // 0
    1.0,                      // 1
    2.0,                      // 2
    6.0,                      // 3
    24.0,                     // 4
    120.0,                    // 5
    720.0,                    // 6
    5040.0,                   // 7
    40320.0,                  // 8
    362880.0,                 // 9
    3628800.0,                // 10
    39916800.0,               // 11
    479001600.0,              // 12
    6227020800.0,             // 13
    87178291200.0,            // 14
    1307674368000.0,          // 15
    20922789888000.0,         // 16
    355687428096000.0,        // 17
    6402373705728000.0,       // 18
    121645100408832000.0,     // 19
    2432902008176640000.0,    // 20
    51090942171709440000.0,   // 21
    1124000727777607680000.0, // 22
];

/// Returns the factorial of n smaller than or equal to 22 by table lookup
///
/// # Panics
///
/// Will panic if n > 22
///
/// # Reference
///
/// According to the reference, factorials up to 22! have exact double precision representations
/// (52 bits of mantissa, not counting powers of two that are absorbed into the exponent)
///
/// * Press WH, Teukolsky SA, Vetterling WT, Flannery BP (2007) Numerical Recipes: The Art of
///   Scientific Computing. Third Edition. Cambridge University Press. 1235p.
pub fn factorial_lookup_22(n: usize) -> f64 {
    if n > 22 {
        panic!("factorial_lookup_22 requires n ≤ 22");
    }
    FACTORIAL_22[n]
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{approx_eq, deriv_approx_eq};
    use std::f64::consts::PI;

    #[test]
    fn sign_ramp_heaviside_boxcar_work() {
        let xx = [-2.0, -1.6, -1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 1.6, 2.0];
        let (a, b) = (-1.2, 0.4);
        for x in xx {
            let s = sign(x);
            let r = ramp(x);
            let h = heaviside(x);
            let bxc = boxcar(x, a, b);
            if x == 0.0 {
                assert_eq!(s, 0.0);
            } else {
                assert_eq!(s, f64::abs(x) / x);
            }
            assert_eq!(s, 2.0 * h - 1.0);
            assert_eq!(r, f64::max(x, 0.0));
            assert_eq!(r, (x + f64::abs(x)) / 2.0);
            assert_eq!(r, x * h);
            assert_eq!(h, 0.5 + 0.5 * s);
            assert_eq!(bxc, heaviside(x - a) - heaviside(x - b));
        }
    }

    #[test]
    fn logistic_and_deriv_work() {
        struct Arguments {}
        let args = &mut Arguments {};
        let f = |x: f64, _: &mut Arguments| logistic(x);
        let xx = [-2.0, -1.6, -1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 1.6, 2.0];
        for x in xx {
            let l = logistic(x);
            let d = logistic_deriv(x);
            approx_eq(l, 0.5 + 0.5 * f64::tanh(x / 2.0), 1e-14);
            deriv_approx_eq(d, x, args, 1e-10, f);
        }
    }

    #[test]
    fn smooth_ramp_and_deriv_work() {
        assert_eq!(smooth_ramp(-1.0, 500.1), 0.0);
        assert_eq!(smooth_ramp(-1.0, 499.9), 0.0);
        assert_eq!(smooth_ramp_deriv(-1.0, 500.1), 0.0);
        approx_eq(smooth_ramp_deriv(-1.0, 499.99), 0.0, 1e-15);
        assert_eq!(smooth_ramp_deriv2(1.0, 500.1), 0.0);
        approx_eq(smooth_ramp_deriv2(1.0, 499.99), 0.0, 1e-15);
        let beta = 2.0;
        struct Arguments {
            beta: f64,
        }
        let args = &mut Arguments { beta };
        let f = |x: f64, args: &mut Arguments| smooth_ramp(x, args.beta);
        let g = |x: f64, args: &mut Arguments| smooth_ramp_deriv(x, args.beta);
        let xx = [-2.0, -1.6, -1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 1.6, 2.0];
        for x in xx {
            let d = smooth_ramp_deriv(x, beta);
            let d2 = smooth_ramp_deriv2(x, beta);
            deriv_approx_eq(d, x, args, 1e-9, f);
            deriv_approx_eq(d2, x, args, 1e-9, g);
        }
    }

    #[test]
    fn suq_sin_and_cos_work() {
        approx_eq(suq_sin(0.0, 1.0), 0.0, 1e-14);
        approx_eq(suq_sin(PI, 1.0), 0.0, 1e-14);
        approx_eq(suq_sin(PI / 2.0, 0.0), 1.0, 1e-14);
        approx_eq(suq_sin(PI / 2.0, 1.0), 1.0, 1e-14);
        approx_eq(suq_sin(PI / 2.0, 2.0), 1.0, 1e-14);
        approx_eq(suq_sin(PI / 4.0, 2.0), 0.5, 1e-14);
        approx_eq(suq_sin(-PI / 4.0, 2.0), -0.5, 1e-14);

        approx_eq(suq_cos(0.0, 1.0), 1.0, 1e-14);
        approx_eq(suq_cos(PI, 1.0), -1.0, 1e-14);
        approx_eq(suq_cos(PI / 2.0, 0.0), 1.0, 1e-14); // because sign(cos(pi/2))=1
        approx_eq(suq_cos(PI / 2.0, 1.0), 0.0, 1e-14);
        approx_eq(suq_cos(PI / 2.0, 2.0), 0.0, 1e-14);
        approx_eq(suq_cos(PI / 4.0, 2.0), 0.5, 1e-14);
        approx_eq(suq_cos(-PI / 4.0, 2.0), 0.5, 1e-14);
    }

    #[test]
    #[should_panic(expected = "factorial_lookup_22 requires n ≤ 22")]
    fn factorial_lookup_22_captures_error() {
        factorial_lookup_22(23);
    }

    #[test]
    fn factorial_lookup_22_works() {
        assert_eq!(factorial_lookup_22(0), 1.0);
        assert_eq!(factorial_lookup_22(1), 1.0);
        assert_eq!(factorial_lookup_22(2), 2.0);
        assert_eq!(factorial_lookup_22(3), 6.0);
        assert_eq!(factorial_lookup_22(4), 24.0);
        assert_eq!(factorial_lookup_22(10), 3628800.0,);
        assert_eq!(factorial_lookup_22(22), 1124000727_7776076800_00.0);
    }
}
