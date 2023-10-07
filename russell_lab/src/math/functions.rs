/// Implements the sign function
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

// Implements the ramp function (Macaulay brackets)
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

/// Implements the Heaviside step function (derivative of ramp(x))
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

/// Implements the boxcar function
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

/// Implements the standard logistic function
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

/// Implements a smooth ramp function
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

/// Implements the superquadric function involving sin(x)
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

/// Implements the superquadric auxiliary involving cos(x)
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
}
