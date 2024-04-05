/// (modf) Splits a floating-point number into an integer and a fractional part
///
/// Returns `(integer, fractional)` where:
///
/// ```text
/// integer + fractional = x
/// ```
///
/// # Notes
///
/// * This function is also known as **modf**
/// * Both integer and fractional values will have the same sign as x
///
/// # Special cases
///
/// * `mod_f(±Inf) = ±Inf, NaN`
/// * `mod_f(NaN) = NaN, NaN`
///
/// # Examples
///
/// ```
/// # use russell_lab::math;
/// let (integer, fractional) = math::split_integer_fractional(3.141593);
/// assert_eq!(
///     format!("integer = {:?}, fractional = {:.6}", integer, fractional),
///     "integer = 3.0, fractional = 0.141593"
/// );
/// ```
pub fn split_integer_fractional(x: f64) -> (f64, f64) {
    let mut u = x.to_bits();
    let e = ((u >> 52 & 0x7ff) as i32) - 0x3ff;

    // no fractional part
    let integer: f64;
    if e >= 52 {
        integer = x;
        if e == 0x400 && (u << 12) != 0 {
            return (integer, x); // nan
        }
        u &= 1 << 63;
        return (integer, f64::from_bits(u));
    }

    // no integral part
    if e < 0 {
        u &= 1 << 63;
        integer = f64::from_bits(u);
        return (integer, x);
    }

    let mask: u64 = ((!0) >> 12) >> e;
    if (u & mask) == 0 {
        integer = x;
        u &= 1 << 63;
        return (integer, f64::from_bits(u));
    }
    u &= !mask;
    integer = f64::from_bits(u);
    (integer, x - integer)
}

/// Reports whether x is a negative integer or not
///
/// # Special cases
///
/// * `is_negative_integer(NaN)  = false`
/// * `is_negative_integer(±Inf) = false`
///
/// # Examples
///
/// ```
/// # use russell_lab::math;
/// assert_eq!(math::is_negative_integer(-1.23), false);
/// assert_eq!(math::is_negative_integer(2.0), false);
/// assert_eq!(math::is_negative_integer(-2.0), true);
/// ```
pub fn is_negative_integer(x: f64) -> bool {
    if f64::is_finite(x) {
        if x < 0.0 {
            let (_, xf) = split_integer_fractional(x);
            xf == 0.0
        } else {
            false
        }
    } else {
        false
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{is_negative_integer, split_integer_fractional};

    #[test]
    fn split_integer_fractional_works() {
        let values = [123.239459191, 3956969101.20101, -2.3303];
        for x in values {
            let (integer, fractional) = split_integer_fractional(x);
            assert_eq!(integer + fractional, x);
        }

        // #include <stdio.h>
        // #include <math.h>
        // int main () {
        //   double param, fractional, integer;
        //   param = -1.0/0.0;
        //   fractional = modf (param , &integer);
        //   printf ("%f = %f + %f \n", param, integer, fractional);
        //   return 0;
        // }
        // // Output:
        // // -inf = -inf + -0.000000

        // special cases
        assert_eq!(split_integer_fractional(f64::NEG_INFINITY), (f64::NEG_INFINITY, -0.0)); // note that Go returns (-Inf, NaN); but the C code above returns (-Inf, -0.0)
        assert_eq!(split_integer_fractional(-0.0), (-0.0, -0.0));
        assert_eq!(split_integer_fractional(f64::INFINITY), (f64::INFINITY, 0.0)); // note that Go returns (Inf, NaN); but the C code above returns (Inf, 0.0)
        let (integer, fractional) = split_integer_fractional(f64::NAN);
        assert!(integer.is_nan());
        assert!(fractional.is_nan());
    }

    #[test]
    fn verify_function_is_sign_negative() {
        // Go uses `sign_bit := f64::to_bits(x) & (1 << 63) != 0`
        // In rust, we write `sign_bit = f64::is_sign_negative(x)`
        let values = [1.0, -1.0, f64::NEG_INFINITY, f64::INFINITY, f64::NAN];
        for x in values {
            assert_eq!(f64::is_sign_negative(x), f64::to_bits(x) & (1 << 63) != 0);
        }
    }

    #[test]
    fn is_negative_integer_works() {
        // true
        assert_eq!(is_negative_integer(-3.0), true);
        assert_eq!(is_negative_integer(-10.0), true);
        assert_eq!(is_negative_integer(f64::MIN), true);
        // false
        assert_eq!(is_negative_integer(-3.14), false);
        assert_eq!(is_negative_integer(-0.0), false);
        assert_eq!(is_negative_integer(0.0), false);
        assert_eq!(is_negative_integer(1.0), false);
        assert_eq!(is_negative_integer(f64::NEG_INFINITY), false);
        assert_eq!(is_negative_integer(f64::INFINITY), false);
        assert_eq!(is_negative_integer(f64::NAN), false);
    }
}
