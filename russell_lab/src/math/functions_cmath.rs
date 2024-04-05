extern "C" {
    fn c_frexp(x: f64, exp: *mut i32) -> f64;
    fn c_ldexp(frac: f64, exp: i32) -> f64;
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
    use super::{frexp, ldexp};

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
