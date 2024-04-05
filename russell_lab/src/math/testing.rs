//////////////////////////////////////////////////////////////////
// This file contains data and functions for unit tests.        //
// The code is based on all_test.go file from Go (1.22.1)       //
//////////////////////////////////////////////////////////////////
// Copyright 2009 The Go Authors. All rights reserved.          //
// Use of this source code is governed by a BSD-style           //
// license that can be found in the LICENSE file.               //
//////////////////////////////////////////////////////////////////

#[allow(dead_code)]
pub(super) const MATH_TEST_VALUES: [f64; 10] = [
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

fn tolerance(a: f64, b: f64, e: f64) -> bool {
    // multiplying by e here can underflow de-normal values to zero.
    // check a==b so that at least if a and b are small and identical we say they match.
    if a == b {
        return true;
    }
    let mut d = a - b;
    if d < 0.0 {
        d = -d
    }
    // note: b is correct (expected) value, a is actual value.
    // make error tolerance a fraction of b, not a.
    let mut ee = e;
    if b != 0.0 {
        ee = ee * b;
        if ee < 0.0 {
            ee = -ee;
        }
    }
    d < ee
}

pub(super) fn close(a: f64, b: f64) -> bool {
    tolerance(a, b, 1e-14)
}

pub(super) fn very_close(a: f64, b: f64) -> bool {
    tolerance(a, b, 4e-16)
}

pub(super) fn so_close(a: f64, b: f64, e: f64) -> bool {
    tolerance(a, b, e)
}

pub(super) fn alike(a: f64, b: f64) -> bool {
    if f64::is_nan(a) && f64::is_nan(b) {
        true
    } else if a == b {
        a.is_sign_negative() == b.is_sign_negative()
    } else {
        false
    }
}
