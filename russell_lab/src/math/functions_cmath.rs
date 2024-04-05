extern "C" {
    fn c_ln_gamma(x: f64) -> f64;
    fn c_frexp(x: f64, exp: *mut i32) -> f64;
    fn c_ldexp(frac: f64, exp: i32) -> f64;
}

/// Evaluates the natural logarithm of Γ(x) and its sign
///
/// Returns `(ln(Γ(x)), sign)` where sign is -1 or 1
///
/// # Special cases
///
/// * `ln(Γ(+Inf))     = +Inf`
/// * `ln(Γ(0))        = +Inf`
/// * `ln(Γ(-integer)) = +Inf`
/// * `ln(Γ(-Inf))     = -Inf`
/// * `ln(Γ(NaN))      = NaN`
#[inline]
pub fn ln_gamma(x: f64) -> (f64, i32) {
    // TODO: implement the sign
    unsafe { (c_ln_gamma(x), 1) }
}

/// Gets the significand and exponent of a number
///
/// Breaks a value x into a normalized fraction and an integral power of two
///
/// Returns `(frac, exp)` satisfying:
///
/// ```text
/// x == frac · 2^exp
/// ```
///
/// with the absolute value of frac in the interval [0.5, 1)
///
/// Special cases:
///
///	* `frexp(±0.0) = ±0.0, 0`
///	* `frexp(±Inf) = ±Inf, 0`
///	* `frexp(NaN)  = NaN,  0`
///
/// Reference: <https://cplusplus.com/reference/cmath/frexp/>
pub fn frexp(x: f64) -> (f64, i32) {
    if x == 0.0 || f64::is_nan(x) {
        return (x, 0); // correctly return -0 or NaN
    } else if f64::is_infinite(x) || f64::is_nan(x) {
        return (x, 0);
    }
    unsafe {
        let mut exp: i32 = 0;
        let frac = c_frexp(x, &mut exp);
        (frac, exp)
    }
}

/// Generates a number from significand and exponent
///
/// Returns:
///
/// ```text
/// x = frac · 2^exp
/// ```
///
/// Returns the result of multiplying x (the significand) by 2 raised to the power of exp (the exponent).
///
/// Special cases:
///
/// * `ldexp(±0.0, exp) = ±0.0`
/// * `ldexp(±Inf, exp) = ±Inf`
/// * `ldexp(NaN,  exp) = NaN`
///
/// Reference: <https://cplusplus.com/reference/cmath/ldexp/>
pub fn ldexp(frac: f64, exp: i32) -> f64 {
    if frac == 0.0 || f64::is_nan(frac) {
        return frac; // correctly return -0 or NaN
    } else if f64::is_infinite(frac) || f64::is_nan(frac) {
        return frac;
    }
    unsafe { c_ldexp(frac, exp) }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{frexp, ldexp, ln_gamma};
    use crate::approx_eq;
    use crate::math::ONE_BY_3;

    #[test]
    fn ln_gamma_works() {
        assert!(ln_gamma(f64::NAN).0.is_nan());
        assert_eq!(ln_gamma(-1.0).0, f64::INFINITY);
        assert_eq!(ln_gamma(0.0).0, f64::INFINITY);
        assert_eq!(ln_gamma(1.0).0, 0.0);
        assert_eq!(ln_gamma(2.0).0, 0.0);

        // Mathematica
        // res = Table[{x, NumberForm[N[LogGamma[x], 50], 50]}, {x, {0.1, 1/3, 0.5, 3, 10, 33}}]
        // Export["test.txt", res, "Table", "FieldSeparators" -> ", "]
        let mathematica = [
            (0.1, 1e-15, 2.252712651734206),
            (ONE_BY_3, 1e-15, 0.98542064692776706918717403697796139173555649638589),
            (0.5, 1e-50, 0.5723649429247001),
            (3.0, 1e-50, 0.69314718055994530941723212145817656807550013436026),
            (10.0, 1e-14, 12.801827480081469611207717874566706164281149255663),
            (33.0, 1e-13, 81.557959456115037178502968666011206687099284403417),
        ];
        for (x, tol, reference) in mathematica {
            // println!("x = {:?}", x);
            approx_eq(ln_gamma(x).0, reference, tol)
        }
    }

    #[test]
    fn frexp_works() {
        assert_eq!(frexp(-0.0), (-0.0, 0));
        assert_eq!(frexp(0.0), (0.0, 0));
        assert_eq!(frexp(f64::NEG_INFINITY), (f64::NEG_INFINITY, 0));
        assert_eq!(frexp(f64::INFINITY), (f64::INFINITY, 0));
        assert!(frexp(f64::NAN).0.is_nan());
        assert_eq!(frexp(8.0), (0.5, 4));
    }

    #[test]
    fn ldexp_works() {
        assert_eq!(ldexp(-0.0, 0), -0.0);
        assert_eq!(ldexp(0.0, 0), 0.0);
        assert_eq!(ldexp(f64::NEG_INFINITY, 0), f64::NEG_INFINITY);
        assert_eq!(ldexp(f64::INFINITY, 0), f64::INFINITY);
        assert!(ldexp(f64::NAN, 0).is_nan());
        assert_eq!(ldexp(0.5, 4), 8.0);
    }
}
