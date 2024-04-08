use super::{float_compose, float_decompose};

// The code is based on mod.go file from Go (1.22.1)
//
// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/// Returns the floating-point remainder of x/y
///
/// The magnitude of the result is less than y and its sign agrees with that of x
///
/// # Special cases
///
/// * `modulo(±Inf, y) = NaN`
/// * `modulo(NaN, y) = NaN`
/// * `modulo(x, 0) = NaN`
/// * `modulo(x, ±Inf) = x`
/// * `modulo(x, NaN) = NaN`
pub fn modulo(x: f64, y: f64) -> f64 {
    if y == 0.0 || f64::is_infinite(x) || f64::is_nan(x) || f64::is_nan(y) {
        return f64::NAN;
    }
    let y = f64::abs(y);

    let (y_frac, y_exp) = float_decompose(y);
    let mut r = x;
    if x < 0.0 {
        r = -x;
    }

    while r >= y {
        let (r_frac, mut r_exp) = float_decompose(r);
        if r_frac < y_frac {
            r_exp = r_exp - 1;
        }
        r = r - float_compose(y, r_exp - y_exp);
    }

    if x < 0.0 {
        r = -r;
    }
    r
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::modulo;
    use crate::assert_alike;
    use crate::math::PI;

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

    const SOLUTION: [f64; 10] = [
        4.197615023265299782906368e-02,
        2.261127525421895434476482e+00,
        3.231794108794261433104108e-02,
        4.989396381728925078391512e+00,
        3.637062928015826201999516e-01,
        1.220868282268106064236690e+00,
        4.770916568540693347699744e+00,
        1.816180268691969246219742e+00,
        8.734595415957246977711748e-01,
        1.314075231424398637614104e+00,
    ];

    const SC_VALUES: [[f64; 2]; 34] = [
        [f64::NEG_INFINITY, f64::NEG_INFINITY],
        [f64::NEG_INFINITY, -PI],
        [f64::NEG_INFINITY, 0.0],
        [f64::NEG_INFINITY, PI],
        [f64::NEG_INFINITY, f64::INFINITY],
        [f64::NEG_INFINITY, f64::NAN],
        [-PI, f64::NEG_INFINITY],
        [-PI, 0.0],
        [-PI, f64::INFINITY],
        [-PI, f64::NAN],
        [-0.0, f64::NEG_INFINITY],
        [-0.0, 0.0],
        [-0.0, f64::INFINITY],
        [-0.0, f64::NAN],
        [0.0, f64::NEG_INFINITY],
        [0.0, 0.0],
        [0.0, f64::INFINITY],
        [0.0, f64::NAN],
        [PI, f64::NEG_INFINITY],
        [PI, 0.0],
        [PI, f64::INFINITY],
        [PI, f64::NAN],
        [f64::INFINITY, f64::NEG_INFINITY],
        [f64::INFINITY, -PI],
        [f64::INFINITY, 0.0],
        [f64::INFINITY, PI],
        [f64::INFINITY, f64::INFINITY],
        [f64::INFINITY, f64::NAN],
        [f64::NAN, f64::NEG_INFINITY],
        [f64::NAN, -PI],
        [f64::NAN, 0.0],
        [f64::NAN, PI],
        [f64::NAN, f64::INFINITY],
        [f64::NAN, f64::NAN],
    ];

    const SC_SOLUTION: [f64; 34] = [
        f64::NAN, // modulo(-Inf, -Inf)
        f64::NAN, // modulo(-Inf, -PI)
        f64::NAN, // modulo(-Inf, 0)
        f64::NAN, // modulo(-Inf, PI)
        f64::NAN, // modulo(-Inf, +Inf)
        f64::NAN, // modulo(-Inf, NaN)
        -PI,      // modulo(-PI, -Inf)
        f64::NAN, // modulo(-PI, 0)
        -PI,      // modulo(-PI, +Inf)
        f64::NAN, // modulo(-PI, NaN)
        -0.0,     // modulo(-0, -Inf)
        f64::NAN, // modulo(-0, 0)
        -0.0,     // modulo(-0, Inf)
        f64::NAN, // modulo(-0, NaN)
        0.0,      // modulo(0, -Inf)
        f64::NAN, // modulo(0, 0)
        0.0,      // modulo(0, +Inf)
        f64::NAN, // modulo(0, NaN)
        PI,       // modulo(PI, -Inf)
        f64::NAN, // modulo(PI, 0)
        PI,       // modulo(PI, +Inf)
        f64::NAN, // modulo(PI, NaN)
        f64::NAN, // modulo(+Inf, -Inf)
        f64::NAN, // modulo(+Inf, -PI)
        f64::NAN, // modulo(+Inf, 0)
        f64::NAN, // modulo(+Inf, PI)
        f64::NAN, // modulo(+Inf, +Inf)
        f64::NAN, // modulo(+Inf, NaN)
        f64::NAN, // modulo(NaN, -Inf)
        f64::NAN, // modulo(NaN, -PI)
        f64::NAN, // modulo(NaN, 0)
        f64::NAN, // modulo(NaN, PI)
        f64::NAN, // modulo(NaN, +Inf)
        f64::NAN, // modulo(NaN, NaN)
    ];

    #[test]
    fn test_modulo() {
        for (i, v) in VALUES.iter().enumerate() {
            let f = modulo(10.0, *v);
            assert_eq!(SOLUTION[i], f);
        }
        for (i, v) in SC_VALUES.iter().enumerate() {
            let f = modulo(v[0], v[1]);
            assert_alike(SC_SOLUTION[i], f);
        }
        // verify precision of result for extreme inputs
        let f = modulo(5.9790119248836734e+200, 1.1258465975523544);
        assert_eq!(0.6447968302508578, f);
    }
}
