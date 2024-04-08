// Part of the code is based on files frexp.go and ldexp.go from Go (1.22.1),
// having the following copyright notice:
//
// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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
/// * This function is also known as **modf** in [C/C++](https://cplusplus.com/reference/cmath/modf/) and [Go](https://pkg.go.dev/math@go1.22.2#Modf)
/// * Both integer and fractional values will have the same sign as x
///
/// # Special cases
///
/// * `float_split(±Inf) = ±Inf, NaN`
/// * `float_split(NaN) = NaN, NaN`
///
/// # Examples
///
/// ```
/// # use russell_lab::math;
/// let (integer, fractional) = math::float_split(3.141593);
/// assert_eq!(
///     format!("integer = {:?}, fractional = {:.6}", integer, fractional),
///     "integer = 3.0, fractional = 0.141593"
/// );
/// ```
pub fn float_split(x: f64) -> (f64, f64) {
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

/// Reports whether a floating-point number corresponds to a negative integer number or not
///
/// # Special cases
///
/// * `float_is_neg_integer(NaN)  = false`
/// * `float_is_neg_integer(±Inf) = false`
///
/// # Examples
///
/// ```
/// # use russell_lab::math;
/// assert_eq!(math::float_is_neg_integer(-1.23), false);
/// assert_eq!(math::float_is_neg_integer(2.0), false);
/// assert_eq!(math::float_is_neg_integer(-2.0), true);
/// ```
pub fn float_is_neg_integer(x: f64) -> bool {
    if f64::is_finite(x) {
        if x < 0.0 {
            let (_, xf) = float_split(x);
            xf == 0.0
        } else {
            false
        }
    } else {
        false
    }
}

/// Reports whether a floating-point number corresponds to an integer number or not
///
/// # Special cases
///
/// * `float_is_integer(NaN)  = false`
/// * `float_is_integer(±Inf) = false`
///
/// # Examples
///
/// ```
/// # use russell_lab::math;
/// assert_eq!(math::float_is_integer(-1.23), false);
/// assert_eq!(math::float_is_integer(-2.0), true);
/// assert_eq!(math::float_is_integer(3.0), true);
/// ```
pub fn float_is_integer(x: f64) -> bool {
    if f64::is_finite(x) {
        let (_, xf) = float_split(x);
        xf == 0.0
    } else {
        false
    }
}

/// (frexp) Decomposes a floating-point number into a mantissa and exponent parts
///
/// Returns `(mantissa, exponent)` satisfying:
///
/// ```text
/// x == mantissa · 2^exponent
/// ```
///
/// with the absolute value of mantissa in the interval `[0.5, 1)`
///
/// # Notes
///
/// * This function is also known as **frexp** in [C/C++](https://cplusplus.com/reference/cmath/frexp/) and [Go](https://pkg.go.dev/math@go1.22.2#Frexp)
///
/// # Special cases
///
///	* `float_decompose(±0.0) = (±0.0, 0)`
///	* `float_decompose(±Inf) = (±Inf, 0)`
///	* `float_decompose(NaN)  = (NaN,  0)`
///
/// # Examples
///
/// ```
/// # use russell_lab::math;
/// let x = 0.5 * 2.0 * 2.0 * 2.0 * 2.0;
/// assert_eq!(math::float_decompose(x), (0.5, 4));
/// ```
pub fn float_decompose(x: f64) -> (f64, i32) {
    // handle special cases
    if x == 0.0 || f64::is_infinite(x) || f64::is_nan(x) {
        return (x, 0);
    }
    // normalize
    let (xx, mut e) = if f64::abs(x) < f64::MIN_POSITIVE {
        (x * ((1_u64 << 52) as f64), -52)
    } else {
        (x, 0)
    };
    // decomposition
    let mut y = xx.to_bits();
    e += ((y >> 52) & 0x7ff) as i32 - 0x3fe;
    y &= 0x800fffffffffffff;
    y |= 0x3fe0000000000000;
    (f64::from_bits(y), e)
}

/// (ldexp) Composes a floating-point number from a mantissa and exponent
///
/// Returns:
///
/// ```text
/// x = mantissa · 2^exponent
/// ```
///
/// # Notes
///
/// * This function is also known as **ldexp** in [C/C++](https://cplusplus.com/reference/cmath/ldexp/) and [Go](https://pkg.go.dev/math@go1.22.2#Ldexp)
///
/// # Special cases
///
/// * `float_compose(±0.0, exponent) = ±0.0`
/// * `float_compose(±Inf, exponent) = ±Inf`
/// * `float_compose(NaN,  exponent) = NaN`
///
/// # Examples
///
/// ```
/// # use russell_lab::math;
/// let x = 0.5 * 2.0 * 2.0 * 2.0 * 2.0;
/// assert_eq!(math::float_compose(0.5,  4), x);
/// assert_eq!(math::float_compose(8.0,  0), 8.0); // 8 · 2⁰
/// assert_eq!(math::float_compose(8.0, -1), 4.0); // 8 · 2⁻¹
/// assert_eq!(math::float_compose(8.0, -2), 2.0); // 8 · 2⁻²
/// assert_eq!(math::float_compose(8.0, -3), 1.0); // 8 · 2⁻³
/// assert_eq!(math::float_compose(8.0, -4), 0.5); // 8 · 2⁻⁴
/// ```
pub fn float_compose(mantissa: f64, exponent: i32) -> f64 {
    // handle special cases
    if mantissa == 0.0 || f64::is_infinite(mantissa) || f64::is_nan(mantissa) {
        return mantissa;
    }
    // normalize
    let (frac, e) = if f64::abs(mantissa) < f64::MIN_POSITIVE {
        (mantissa * ((1_u64 << 52) as f64), -52)
    } else {
        (mantissa, 0)
    };
    // composition
    let mut exp = exponent + e;
    let mut x = frac.to_bits();
    exp += ((x >> 52) & 0x7ff) as i32 - 0x3ff;
    if exp < -1075 {
        return f64::copysign(0.0, frac); // underflow
    }
    if exp > 1023 {
        // overflow
        if frac < 0.0 {
            return f64::NEG_INFINITY;
        }
        return f64::INFINITY;
    }
    // de-normalize
    let mut m = 1.0;
    if exp < -1022 {
        exp += 53;
        m = 1.0 / ((1_u64 << 53) as f64); // 2**-53
    }
    // results
    x &= 0x800fffffffffffff;
    x |= ((exp + 1023) as u64) << 52;
    m * f64::from_bits(x)
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{float_compose, float_decompose, float_is_integer, float_is_neg_integer, float_split};
    use crate::{approx_eq, assert_alike};

    #[test]
    fn float_split_works() {
        let values = [123.239459191, 3956969101.20101, -2.3303];
        for x in values {
            let (integer, fractional) = float_split(x);
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
        assert_eq!(float_split(f64::NEG_INFINITY), (f64::NEG_INFINITY, -0.0)); // note that Go returns (-Inf, NaN); but the C code above returns (-Inf, -0.0)
        assert_eq!(float_split(-0.0), (-0.0, -0.0));
        assert_eq!(float_split(f64::INFINITY), (f64::INFINITY, 0.0)); // note that Go returns (Inf, NaN); but the C code above returns (Inf, 0.0)
        let (integer, fractional) = float_split(f64::NAN);
        assert!(integer.is_nan());
        assert!(fractional.is_nan());
    }

    #[test]
    fn verify_go_sign_bit_function() {
        // Go uses `sign_bit := f64::to_bits(x) & (1 << 63) != 0`
        // In rust, we write `sign_bit = f64::is_sign_negative(x)`
        let values = [1.0, -1.0, f64::NEG_INFINITY, f64::INFINITY, f64::NAN];
        for x in values {
            assert_eq!(f64::is_sign_negative(x), f64::to_bits(x) & (1 << 63) != 0);
        }
    }

    #[test]
    fn float_is_neg_integer_works() {
        // true
        assert_eq!(float_is_neg_integer(-3.0), true);
        assert_eq!(float_is_neg_integer(-10.0), true);
        assert_eq!(float_is_neg_integer(f64::MIN), true);
        // false
        assert_eq!(float_is_neg_integer(-3.14), false);
        assert_eq!(float_is_neg_integer(-0.0), false);
        assert_eq!(float_is_neg_integer(0.0), false);
        assert_eq!(float_is_neg_integer(1.0), false);
        assert_eq!(float_is_neg_integer(f64::NEG_INFINITY), false);
        assert_eq!(float_is_neg_integer(f64::INFINITY), false);
        assert_eq!(float_is_neg_integer(f64::NAN), false);
    }

    #[test]
    fn float_is_integer_works() {
        // true
        assert_eq!(float_is_integer(-3.0), true);
        assert_eq!(float_is_integer(-10.0), true);
        assert_eq!(float_is_integer(f64::MIN), true);
        assert_eq!(float_is_integer(f64::MAX), true);
        assert_eq!(float_is_integer(3.0), true);
        assert_eq!(float_is_integer(10.0), true);
        assert_eq!(float_is_integer(-0.0), true);
        assert_eq!(float_is_integer(0.0), true);
        // false
        assert_eq!(float_is_integer(-3.14), false);
        assert_eq!(float_is_integer(3.14), false);
        assert_eq!(float_is_integer(f64::NEG_INFINITY), false);
        assert_eq!(float_is_integer(f64::INFINITY), false);
        assert_eq!(float_is_integer(f64::NAN), false);
    }

    #[test]
    fn float_decompose_works() {
        assert_eq!(float_decompose(-0.0), (-0.0, 0));
        assert_eq!(float_decompose(0.0), (0.0, 0));
        assert_eq!(float_decompose(f64::NEG_INFINITY), (f64::NEG_INFINITY, 0));
        assert_eq!(float_decompose(f64::INFINITY), (f64::INFINITY, 0));
        assert!(float_decompose(f64::NAN).0.is_nan());
        assert_eq!(float_decompose(8.0), (0.5, 4));
    }

    #[test]
    fn float_compose_works() {
        assert_eq!(float_compose(-0.0, 0), -0.0);
        assert_eq!(float_compose(0.0, 0), 0.0);
        assert_eq!(float_compose(f64::NEG_INFINITY, 0), f64::NEG_INFINITY);
        assert_eq!(float_compose(f64::INFINITY, 0), f64::INFINITY);
        assert!(float_compose(f64::NAN, 0).is_nan());
        assert_eq!(float_compose(0.5, 4), 8.0);
        assert_eq!(float_compose(1.0, 3), 8.0);
        assert_eq!(float_compose(2.0, 2), 8.0);
        assert_eq!(float_compose(4.0, 1), 8.0);
        assert_eq!(float_compose(8.0, 0), 8.0);
        assert_eq!(float_compose(8.0, -1), 4.0);
        assert_eq!(float_compose(8.0, -2), 2.0);
        assert_eq!(float_compose(8.0, -3), 1.0);
        assert_eq!(float_compose(8.0, -4), 0.5);
    }

    // The code below is based on all_test.go file from Go (1.22.1)

    const VALUES: [f64; 10] = [
        4.9790119248836735e+00,
        7.7388724745781045e+00,
        -2.7688005719200159e-01,
        -5.0106036182710749e+00,
        9.6362937071984173e+00,
        2.9263772392439646e+00,
        5.2290834314593066e+00,
        2.7279399104360102e+00,
        1.8253080916808550e+00,
        -8.6859247685756013e+00,
    ];

    const SOLUTION_MODF: [[f64; 2]; 10] = [
        [4.0000000000000000e+00, 9.7901192488367350108546816e-01],
        [7.0000000000000000e+00, 7.3887247457810456552351752e-01],
        [-0.0, -2.7688005719200159404635997e-01],
        [-5.0000000000000000e+00, -1.060361827107492160848778e-02],
        [9.0000000000000000e+00, 6.3629370719841737980004837e-01],
        [2.0000000000000000e+00, 9.2637723924396464525443662e-01],
        [5.0000000000000000e+00, 2.2908343145930665230025625e-01],
        [2.0000000000000000e+00, 7.2793991043601025126008608e-01],
        [1.0000000000000000e+00, 8.2530809168085506044576505e-01],
        [-8.0000000000000000e+00, -6.8592476857560136238589621e-01],
    ];

    #[test]
    fn test_float_split() {
        for (i, v) in VALUES.iter().enumerate() {
            let (f, g) = float_split(*v);
            assert_eq!(SOLUTION_MODF[i][0], f);
            assert_eq!(SOLUTION_MODF[i][1], g);
        }
    }

    // frexp and ldexp ------------------------------------------------------------------------

    struct Fi {
        f: f64,
        i: i32,
    }

    #[rustfmt::skip]
    const SOLUTION_FREXP: [Fi; 10] = [
        Fi { f: 6.2237649061045918750e-01,  i:  3 },
        Fi { f: 9.6735905932226306250e-01,  i:  3 },
        Fi { f: -5.5376011438400318000e-01, i: -1 },
        Fi { f: -6.2632545228388436250e-01, i:  3 },
        Fi { f: 6.02268356699901081250e-01, i:  4 },
        Fi { f: 7.3159430981099115000e-01,  i:  2 },
        Fi { f: 6.5363542893241332500e-01,  i:  3 },
        Fi { f: 6.8198497760900255000e-01,  i:  2 },
        Fi { f: 9.1265404584042750000e-01,  i:  1 },
        Fi { f: -5.4287029803597508250e-01, i:  4 },
    ];

    const SC_VALUES_FREXP: [f64; 5] = [f64::NEG_INFINITY, -0.0, 0.0, f64::INFINITY, f64::NAN];

    #[rustfmt::skip]
    const SC_SOLUTION_FREXP: [Fi; 5] = [
        Fi { f: f64::NEG_INFINITY, i: 0 },
        Fi { f: -0.0,              i: 0 },
        Fi { f: 0.0,               i: 0 },
        Fi { f: f64::INFINITY,     i: 0 },
        Fi { f: f64::NAN,          i: 0 },
    ];

    #[rustfmt::skip]
    const SC_VALUES_LDEXP: [Fi; 11] = [
        Fi { f:  0.0,              i: 0     },
        Fi { f:  0.0,              i: -1075 },
        Fi { f:  0.0,              i: 1024  },
        Fi { f: -0.0,              i: 0     },
        Fi { f: -0.0,              i: -1075 },
        Fi { f: -0.0,              i: 1024  },
        Fi { f: f64::INFINITY,     i: 0     },
        Fi { f: f64::INFINITY,     i: -1024 },
        Fi { f: f64::NEG_INFINITY, i: 0     },
        Fi { f: f64::NEG_INFINITY, i: -1024 },
        Fi { f: f64::NAN,          i: -1024 },
    ];

    const SC_SOLUTION_LDEXP: [f64; 11] = [
        0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        -0.0,
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];

    // arguments and expected results for boundary cases
    const SMALLEST_NONZERO_F64: f64 =
        4.9406564584124654417656879286822137236505980261432476442558568250067550727020875186529983636163599238e-324; // from Go

    const SMALLEST_NORMAL_F64: f64 = 2.2250738585072014e-308; // 2**-1022

    const LARGEST_SUBNORMAL_F64: f64 = SMALLEST_NORMAL_F64 - SMALLEST_NONZERO_F64;

    const BC_VALUES_FREXP: [f64; 8] = [
        SMALLEST_NORMAL_F64,
        LARGEST_SUBNORMAL_F64,
        SMALLEST_NONZERO_F64,
        f64::MAX,
        -SMALLEST_NORMAL_F64,
        -LARGEST_SUBNORMAL_F64,
        -SMALLEST_NONZERO_F64,
        -f64::MAX,
    ];

    #[rustfmt::skip]
    const BC_SOLUTION_FREXP: [Fi; 8] = [
        Fi { f: 0.5,                  i: -1021 },
        Fi { f: 0.99999999999999978,  i: -1022 },
        Fi { f: 0.5,                  i: -1073 },
        Fi { f: 0.99999999999999989,  i:  1024 },
        Fi { f: -0.5,                 i: -1021 },
        Fi { f: -0.99999999999999978, i: -1022 },
        Fi { f: -0.5,                 i: -1073 },
        Fi { f: -0.99999999999999989, i:  1024 },
    ];

    #[rustfmt::skip]
    const BC_VALUES_LDEXP: [Fi; 10] = [
        Fi { f: SMALLEST_NORMAL_F64,   i: -52 },
        Fi { f: LARGEST_SUBNORMAL_F64, i: -51 },
        Fi { f: SMALLEST_NONZERO_F64,  i:  1074 },
        Fi { f: f64::MAX,                i: -(1023 + 1074) },
        Fi { f:  1.0,                    i: -1075 },
        Fi { f: -1.0,                    i: -1075 },
        Fi { f:  1.0,                    i:  1024 },
        Fi { f: -1.0,                    i:  1024 },
        Fi { f:  1.0000000000000002,     i: -1075 },
        Fi { f:  1.0,                    i: -1075 },
    ];

    const BC_SOLUTION_LDEXP: [f64; 10] = [
        SMALLEST_NONZERO_F64,
        1e-323, // 2**-1073
        1.0,
        1e-323, // 2**-1073
        0.0,
        -0.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        SMALLEST_NONZERO_F64,
        0.0,
    ];

    #[test]
    fn test_float_decompose() {
        for (i, v) in VALUES.iter().enumerate() {
            let (f, e) = float_decompose(*v);
            approx_eq(SOLUTION_FREXP[i].f, f, 1e-50);
            assert_eq!(SOLUTION_FREXP[i].i, e);
        }
        // special cases
        for (i, v) in SC_VALUES_FREXP.iter().enumerate() {
            let (f, e) = float_decompose(*v);
            assert_alike(SC_SOLUTION_FREXP[i].f, f);
            assert_eq!(SC_SOLUTION_FREXP[i].i, e);
        }
        // boundary cases
        for (i, v) in BC_VALUES_FREXP.iter().enumerate() {
            let (f, e) = float_decompose(*v);
            assert_alike(BC_SOLUTION_FREXP[i].f, f);
            assert_eq!(BC_SOLUTION_FREXP[i].i, e);
        }
    }

    #[test]
    fn test_float_compose() {
        // test composition from FREXP data
        for (i, v) in VALUES.iter().enumerate() {
            let f = float_compose(SOLUTION_FREXP[i].f, SOLUTION_FREXP[i].i);
            approx_eq(*v, f, 1e-50);
        }
        for (i, v) in SC_VALUES_FREXP.iter().enumerate() {
            let f = float_compose(SC_SOLUTION_FREXP[i].f, SC_SOLUTION_FREXP[i].i);
            assert_alike(*v, f);
        }
        for (i, v) in BC_VALUES_FREXP.iter().enumerate() {
            let f = float_compose(BC_SOLUTION_FREXP[i].f, BC_SOLUTION_FREXP[i].i);
            assert_alike(*v, f);
        }
        // special cases
        for (i, v) in SC_VALUES_LDEXP.iter().enumerate() {
            let f = float_compose(v.f, v.i);
            assert_alike(SC_SOLUTION_LDEXP[i], f);
        }
        // boundary cases
        for (i, v) in BC_VALUES_LDEXP.iter().enumerate() {
            let f = float_compose(v.f, v.i);
            assert_alike(BC_SOLUTION_LDEXP[i], f);
        }
    }
}
