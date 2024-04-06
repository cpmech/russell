#![allow(non_upper_case_globals)]

use russell_lab::math;

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//// These tests are based on all_test.go file from Go (1.22.1),      ////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Copyright 2009 The Go Authors. All rights reserved.                  //
// Use of this source code is governed by a BSD-style                   //
// license that can be found in the LICENSE file.                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

// Defines an auxiliary structure holding a float and an integer
struct Pair {
    f: f64,
    i: i32,
}

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

// The expected results below were computed by the high precision calculators
// at https://keisan.casio.com/.  More exact input values (array vf[], above)
// were obtained by printing them with "%.26f".  The answers were calculated
// to 26 digits (by using the "Digit number" drop-down control of each
// calculator).

const SOLUTION_FREXP: [Pair; 10] = [
    Pair {
        f: 6.2237649061045918750e-01,
        i: 3,
    },
    Pair {
        f: 9.6735905932226306250e-01,
        i: 3,
    },
    Pair {
        f: -5.5376011438400318000e-01,
        i: -1,
    },
    Pair {
        f: -6.2632545228388436250e-01,
        i: 3,
    },
    Pair {
        f: 6.02268356699901081250e-01,
        i: 4,
    },
    Pair {
        f: 7.3159430981099115000e-01,
        i: 2,
    },
    Pair {
        f: 6.5363542893241332500e-01,
        i: 3,
    },
    Pair {
        f: 6.8198497760900255000e-01,
        i: 2,
    },
    Pair {
        f: 9.1265404584042750000e-01,
        i: 1,
    },
    Pair {
        f: -5.4287029803597508250e-01,
        i: 4,
    },
];

const SPECIAL_CASES_FREXP: [f64; 5] = [f64::NEG_INFINITY, -0.0, 0.0, f64::INFINITY, f64::NAN];

const SPECIAL_CASES_SOLUTION_FREXP: [Pair; 5] = [
    Pair {
        f: f64::NEG_INFINITY,
        i: 0,
    },
    Pair { f: -0.0, i: 0 },
    Pair { f: 0.0, i: 0 },
    Pair { f: f64::INFINITY, i: 0 },
    Pair { f: f64::NAN, i: 0 },
];

const SPECIAL_CASES_LDEXP: [Pair; 11] = [
    Pair { f: 0.0, i: 0 },
    Pair { f: 0.0, i: -1075 },
    Pair { f: 0.0, i: 1024 },
    Pair { f: -0.0, i: 0 },
    Pair { f: -0.0, i: -1075 },
    Pair { f: -0.0, i: 1024 },
    Pair { f: f64::INFINITY, i: 0 },
    Pair {
        f: f64::INFINITY,
        i: -1024,
    },
    Pair {
        f: f64::NEG_INFINITY,
        i: 0,
    },
    Pair {
        f: f64::NEG_INFINITY,
        i: -1024,
    },
    Pair { f: f64::NAN, i: -1024 },
    // Fi {
    //     f: 10.0,
    //     i: 72057594037927936,
    // }, // from Go
    // Fi {
    //     f: 10.0,
    //     i: -72057594037927936,
    // },
];

const SPECIAL_CASES_SOLUTION_LDEXP: [f64; 13] = [
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
    f64::INFINITY,
    0.0,
];

// arguments and expected results for boundary cases
const SmallestNonzeroFloat64: f64 =
    4.9406564584124654417656879286822137236505980261432476442558568250067550727020875186529983636163599238e-324; // from Go

const SmallestNormalFloat64: f64 = 2.2250738585072014e-308; // 2**-1022

const LargestSubnormalFloat64: f64 = SmallestNormalFloat64 - SmallestNonzeroFloat64;

const vffrexpBC: [f64; 8] = [
    SmallestNormalFloat64,
    LargestSubnormalFloat64,
    SmallestNonzeroFloat64,
    f64::MAX,
    -SmallestNormalFloat64,
    -LargestSubnormalFloat64,
    -SmallestNonzeroFloat64,
    -f64::MAX,
];

const frexpBC: [Pair; 8] = [
    Pair { f: 0.5, i: -1021 },
    Pair {
        f: 0.99999999999999978,
        i: -1022,
    },
    Pair { f: 0.5, i: -1073 },
    Pair {
        f: 0.99999999999999989,
        i: 1024,
    },
    Pair { f: -0.5, i: -1021 },
    Pair {
        f: -0.99999999999999978,
        i: -1022,
    },
    Pair { f: -0.5, i: -1073 },
    Pair {
        f: -0.99999999999999989,
        i: 1024,
    },
];

const vfldexpBC: [Pair; 10] = [
    Pair {
        f: SmallestNormalFloat64,
        i: -52,
    },
    Pair {
        f: LargestSubnormalFloat64,
        i: -51,
    },
    Pair {
        f: SmallestNonzeroFloat64,
        i: 1074,
    },
    Pair {
        f: f64::MAX,
        i: -(1023 + 1074),
    },
    Pair { f: 1.0, i: -1075 },
    Pair { f: -1.0, i: -1075 },
    Pair { f: 1.0, i: 1024 },
    Pair { f: -1.0, i: 1024 },
    Pair {
        f: 1.0000000000000002,
        i: -1075,
    },
    Pair { f: 1.0, i: -1075 },
];

const ldexpBC: [f64; 10] = [
    SmallestNonzeroFloat64,
    1e-323, // 2**-1073
    1.0,
    1e-323, // 2**-1073
    0.0,
    -0.0,
    f64::INFINITY,
    f64::NEG_INFINITY,
    SmallestNonzeroFloat64,
    0.0,
];

fn tolerance(a: f64, b: f64, e: f64) -> bool {
    // Multiplying by e here can underflow de-normal values to zero.
    // Check a==b so that at least if a and b are small and identical
    // we say they match.
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

fn very_close(a: f64, b: f64) -> bool {
    tolerance(a, b, 4e-16)
}

fn alike(a: f64, b: f64) -> bool {
    if f64::is_nan(a) && f64::is_nan(b) {
        return true;
    } else if a == b {
        return a.is_sign_negative() == b.is_sign_negative();
    }
    false
}

#[test]
fn test_frexp() {
    for i in 0..VALUES.len() {
        let (f, j) = math::float_decompose(VALUES[i]);
        if !very_close(SOLUTION_FREXP[i].f, f) || SOLUTION_FREXP[i].i != j {
            println!(
                "frexp({}) = ({}, {}); want ({}, {})",
                VALUES[i], f, j, SOLUTION_FREXP[i].f, SOLUTION_FREXP[i].i
            );
            panic!("frexp failed");
        }
    }
    for i in 0..SPECIAL_CASES_FREXP.len() {
        let (f, j) = math::float_decompose(SPECIAL_CASES_FREXP[i]);
        if !alike(SPECIAL_CASES_SOLUTION_FREXP[i].f, f) || SPECIAL_CASES_SOLUTION_FREXP[i].i != j {
            println!(
                "frexp({}) = ({}, {}); want ({}, {})",
                SPECIAL_CASES_FREXP[i], f, j, SPECIAL_CASES_SOLUTION_FREXP[i].f, SPECIAL_CASES_SOLUTION_FREXP[i].i
            );
            panic!("frexp special cases failed");
        }
    }
    for i in 0..vffrexpBC.len() {
        let (f, j) = math::float_decompose(vffrexpBC[i]);
        if !alike(frexpBC[i].f, f) || frexpBC[i].i != j {
            println!(
                "frexp({}) = ({}, {}); want ({}, {})",
                vffrexpBC[i], f, j, frexpBC[i].f, frexpBC[i].i
            );
            panic!("frexp boundary cases failed");
        }
    }
}

#[test]
fn test_ldexp() {
    for i in 0..VALUES.len() {
        let f = math::float_compose(SOLUTION_FREXP[i].f, SOLUTION_FREXP[i].i);
        if !very_close(VALUES[i], f) {
            println!(
                "ldexp({}, {}) = {}, want {}",
                SOLUTION_FREXP[i].f, SOLUTION_FREXP[i].i, f, VALUES[i]
            );
            panic!("ldexp failed");
        }
    }
    for i in 0..SPECIAL_CASES_FREXP.len() {
        let f = math::float_compose(SPECIAL_CASES_SOLUTION_FREXP[i].f, SPECIAL_CASES_SOLUTION_FREXP[i].i);
        if !alike(SPECIAL_CASES_FREXP[i], f) {
            println!(
                "ldexp({}, {}) = {}, want {}",
                SPECIAL_CASES_SOLUTION_FREXP[i].f, SPECIAL_CASES_SOLUTION_FREXP[i].i, f, SPECIAL_CASES_FREXP[i]
            );
            panic!("ldexp special cases failed");
        }
    }
    for i in 0..SPECIAL_CASES_LDEXP.len() {
        let f = math::float_compose(SPECIAL_CASES_LDEXP[i].f, SPECIAL_CASES_LDEXP[i].i);
        if !alike(SPECIAL_CASES_SOLUTION_LDEXP[i], f) {
            println!(
                "ldexp({}, {}) = {}, want {}",
                SPECIAL_CASES_LDEXP[i].f, SPECIAL_CASES_LDEXP[i].i, f, SPECIAL_CASES_SOLUTION_LDEXP[i]
            );
            panic!("ldexp from frexp failed");
        }
    }
    for i in 0..vffrexpBC.len() {
        let f = math::float_compose(frexpBC[i].f, frexpBC[i].i);
        if !alike(vffrexpBC[i], f) {
            println!(
                "ldexp({}, {}) = {}, want {}",
                frexpBC[i].f, frexpBC[i].i, f, vffrexpBC[i]
            );
            panic!("ldexp boundary cases failed");
        }
    }
    for i in 0..vfldexpBC.len() {
        let f = math::float_compose(vfldexpBC[i].f, vfldexpBC[i].i);
        if !alike(ldexpBC[i], f) {
            println!(
                "Ldexp({}, {}) = {}, want {}",
                vfldexpBC[i].f, vfldexpBC[i].i, f, ldexpBC[i]
            );
            panic!("ldexp from frexp boundary cases failed");
        }
    }
}
