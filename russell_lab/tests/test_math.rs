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

const SOLUTION_ERF: [f64; 10] = [
    5.1865354817738701906913566e-01,
    7.2623875834137295116929844e-01,
    -3.123458688281309990629839e-02,
    -5.2143121110253302920437013e-01,
    8.2704742671312902508629582e-01,
    3.2101767558376376743993945e-01,
    5.403990312223245516066252e-01,
    3.0034702916738588551174831e-01,
    2.0369924417882241241559589e-01,
    -7.8069386968009226729944677e-01,
];

const SOLUTION_ERFC: [f64; 10] = [
    4.8134645182261298093086434e-01,
    2.7376124165862704883070156e-01,
    1.0312345868828130999062984e+00,
    1.5214312111025330292043701e+00,
    1.7295257328687097491370418e-01,
    6.7898232441623623256006055e-01,
    4.596009687776754483933748e-01,
    6.9965297083261411448825169e-01,
    7.9630075582117758758440411e-01,
    1.7806938696800922672994468e+00,
];

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

const SOLUTION_GAMMA: [f64; 10] = [
    2.3254348370739963835386613898e+01,
    2.991153837155317076427529816e+03,
    -4.561154336726758060575129109e+00,
    7.719403468842639065959210984e-01,
    1.6111876618855418534325755566e+05,
    1.8706575145216421164173224946e+00,
    3.4082787447257502836734201635e+01,
    1.579733951448952054898583387e+00,
    9.3834586598354592860187267089e-01,
    -2.093995902923148389186189429e-05,
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

const SPECIAL_CASES_ERF: [f64; 7] = [f64::NEG_INFINITY, -0.0, 0.0, f64::INFINITY, f64::NAN, -1000.0, 1000.0];

const SPECIAL_CASES_SOLUTION_ERF: [f64; 7] = [-1.0, -0.0, 0.0, 1.0, f64::NAN, -1.0, 1.0];

const SPECIAL_CASES_ERFC: [f64; 5] = [f64::NEG_INFINITY, f64::INFINITY, f64::NAN, -1000.0, 1000.0];

const SPECIAL_CASES_SOLUTION_ERFC: [f64; 5] = [2.0, 0.0, f64::NAN, 2.0, 0.0];

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

const VALUES_FOR_GAMMA: [[f64; 2]; 71] = [
    [f64::INFINITY, f64::INFINITY],
    [f64::NEG_INFINITY, f64::NAN],
    [0.0, f64::INFINITY],
    [-0.0, f64::NEG_INFINITY],
    [f64::NAN, f64::NAN],
    [-1.0, f64::NAN],
    [-2.0, f64::NAN],
    [-3.0, f64::NAN],
    [-1e16, f64::NAN],
    [-1e300, f64::NAN],
    [1.7e308, f64::INFINITY],
    [0.5, 1.772453850905516],
    [1.5, 0.886226925452758],
    [2.5, 1.329340388179137],
    [3.5, 3.3233509704478426],
    [-0.5, -3.544907701811032],
    [-1.5, 2.363271801207355],
    [-2.5, -0.9453087204829419],
    [-3.5, 0.2700882058522691],
    [0.1, 9.51350769866873],
    [0.01, 99.4325851191506],
    [1e-08, 9.999999942278434e+07],
    [1e-16, 1e+16],
    [0.001, 999.4237724845955],
    [1e-16, 1e+16],
    [1e-308, 1e+308],
    [5.6e-309, 1.7857142857142864e+308],
    [5.5e-309, f64::INFINITY],
    [1e-309, f64::INFINITY],
    [1e-323, f64::INFINITY],
    [5e-324, f64::INFINITY],
    [-0.1, -10.686287021193193],
    [-0.01, -100.58719796441078],
    [-1e-08, -1.0000000057721567e+08],
    [-1e-16, -1e+16],
    [-0.001, -1000.5782056293586],
    [-1e-16, -1e+16],
    [-1e-308, -1e+308],
    [-5.6e-309, -1.7857142857142864e+308],
    [-5.5e-309, f64::NEG_INFINITY],
    [-1e-309, f64::NEG_INFINITY],
    [-1e-323, f64::NEG_INFINITY],
    [-5e-324, f64::NEG_INFINITY],
    [-0.9999999999999999, -9.007199254740992e+15],
    [-1.0000000000000002, 4.5035996273704955e+15],
    [-1.9999999999999998, 2.2517998136852485e+15],
    [-2.0000000000000004, -1.1258999068426235e+15],
    [-100.00000000000001, -7.540083334883109e-145],
    [-99.99999999999999, 7.540083334884096e-145],
    [17.0, 2.0922789888e+13],
    [171.0, 7.257415615307999e+306],
    [171.6, 1.5858969096672565e+308],
    [171.624, 1.7942117599248104e+308],
    [171.625, f64::INFINITY],
    [172.0, f64::INFINITY],
    [2000.0, f64::INFINITY],
    [-100.5, -3.3536908198076787e-159],
    [-160.5, -5.255546447007829e-286],
    [-170.5, -3.3127395215386074e-308],
    [-171.5, 1.9316265431712e-310],
    [-176.5, -1.196e-321],
    [-177.5, 5e-324],
    [-178.5, -0.0],
    [-179.5, 0.0],
    [-201.0001, 0.0],
    [-202.9999, -0.0],
    [-1000.5, -0.0],
    [-1.0000000003e+09, -0.0],
    [-4.5035996273704955e+15, 0.0],
    [-63.349078729022985, 4.177797167776188e-88],
    [-127.45117632943295, 1.183111089623681e-214],
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

fn close(a: f64, b: f64) -> bool {
    tolerance(a, b, 1e-14)
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
fn test_erf() {
    for i in 0..VALUES.len() {
        let a = VALUES[i] / 10.0;
        let f = math::erf(a);
        if !very_close(SOLUTION_ERF[i], f) {
            println!("erf({}) = {}, want {}", a, f, SOLUTION_ERF[i]);
            panic!("erf failed");
        }
    }
    for i in 0..(SPECIAL_CASES_ERF.len()) {
        let f = math::erf(SPECIAL_CASES_ERF[i]);
        if !alike(SPECIAL_CASES_SOLUTION_ERF[i], f) {
            println!(
                "erf({}) = {}, want {}",
                SPECIAL_CASES_ERF[i], f, SPECIAL_CASES_SOLUTION_ERF[i]
            );
            panic!("erf special cases failed");
        }
    }
}

#[test]
fn test_erfc() {
    for i in 0..VALUES.len() {
        let a = VALUES[i] / 10.0;
        let f = math::erfc(a);
        if !very_close(SOLUTION_ERFC[i], f) {
            println!("erfc({}) = {}, want {}", a, f, SOLUTION_ERFC[i]);
            panic!("erfc failed");
        }
    }
    for i in 0..SPECIAL_CASES_ERFC.len() {
        let f = math::erfc(SPECIAL_CASES_ERFC[i]);
        if !alike(SPECIAL_CASES_SOLUTION_ERFC[i], f) {
            println!(
                "erfc({}) = {}, want {}",
                SPECIAL_CASES_ERFC[i], f, SPECIAL_CASES_SOLUTION_ERFC[i]
            );
            panic!("erfc special cases failed");
        }
    }
}

#[test]
fn test_frexp() {
    for i in 0..VALUES.len() {
        let (f, j) = math::frexp(VALUES[i]);
        if !very_close(SOLUTION_FREXP[i].f, f) || SOLUTION_FREXP[i].i != j {
            println!(
                "frexp({}) = ({}, {}); want ({}, {})",
                VALUES[i], f, j, SOLUTION_FREXP[i].f, SOLUTION_FREXP[i].i
            );
            panic!("frexp failed");
        }
    }
    for i in 0..SPECIAL_CASES_FREXP.len() {
        let (f, j) = math::frexp(SPECIAL_CASES_FREXP[i]);
        if !alike(SPECIAL_CASES_SOLUTION_FREXP[i].f, f) || SPECIAL_CASES_SOLUTION_FREXP[i].i != j {
            println!(
                "frexp({}) = ({}, {}); want ({}, {})",
                SPECIAL_CASES_FREXP[i], f, j, SPECIAL_CASES_SOLUTION_FREXP[i].f, SPECIAL_CASES_SOLUTION_FREXP[i].i
            );
            panic!("frexp special cases failed");
        }
    }
    for i in 0..vffrexpBC.len() {
        let (f, j) = math::frexp(vffrexpBC[i]);
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
        let f = math::ldexp(SOLUTION_FREXP[i].f, SOLUTION_FREXP[i].i);
        if !very_close(VALUES[i], f) {
            println!(
                "ldexp({}, {}) = {}, want {}",
                SOLUTION_FREXP[i].f, SOLUTION_FREXP[i].i, f, VALUES[i]
            );
            panic!("ldexp failed");
        }
    }
    for i in 0..SPECIAL_CASES_FREXP.len() {
        let f = math::ldexp(SPECIAL_CASES_SOLUTION_FREXP[i].f, SPECIAL_CASES_SOLUTION_FREXP[i].i);
        if !alike(SPECIAL_CASES_FREXP[i], f) {
            println!(
                "ldexp({}, {}) = {}, want {}",
                SPECIAL_CASES_SOLUTION_FREXP[i].f, SPECIAL_CASES_SOLUTION_FREXP[i].i, f, SPECIAL_CASES_FREXP[i]
            );
            panic!("ldexp special cases failed");
        }
    }
    for i in 0..SPECIAL_CASES_LDEXP.len() {
        let f = math::ldexp(SPECIAL_CASES_LDEXP[i].f, SPECIAL_CASES_LDEXP[i].i);
        if !alike(SPECIAL_CASES_SOLUTION_LDEXP[i], f) {
            println!(
                "ldexp({}, {}) = {}, want {}",
                SPECIAL_CASES_LDEXP[i].f, SPECIAL_CASES_LDEXP[i].i, f, SPECIAL_CASES_SOLUTION_LDEXP[i]
            );
            panic!("ldexp from frexp failed");
        }
    }
    for i in 0..vffrexpBC.len() {
        let f = math::ldexp(frexpBC[i].f, frexpBC[i].i);
        if !alike(vffrexpBC[i], f) {
            println!(
                "ldexp({}, {}) = {}, want {}",
                frexpBC[i].f, frexpBC[i].i, f, vffrexpBC[i]
            );
            panic!("ldexp boundary cases failed");
        }
    }
    for i in 0..vfldexpBC.len() {
        let f = math::ldexp(vfldexpBC[i].f, vfldexpBC[i].i);
        if !alike(ldexpBC[i], f) {
            println!(
                "Ldexp({}, {}) = {}, want {}",
                vfldexpBC[i].f, vfldexpBC[i].i, f, ldexpBC[i]
            );
            panic!("ldexp from frexp boundary cases failed");
        }
    }
}

#[test]
fn test_modf() {
    for i in 0..VALUES.len() {
        let (f, g) = math::split_integer_fractional(VALUES[i]);
        if !very_close(SOLUTION_MODF[i][0], f) || !very_close(SOLUTION_MODF[i][1], g) {
            println!(
                "split_integer_fractional({}) = ({}, {}); want ({}, {})",
                VALUES[i], f, g, SOLUTION_MODF[i][0], SOLUTION_MODF[i][1]
            );
            panic!("split_integer_fractional failed");
        }
    }
}

#[test]
fn test_gamma() {
    for i in 0..VALUES.len() {
        let f = math::gamma(VALUES[i]);
        if !close(SOLUTION_GAMMA[i], f) {
            println!("gamma({}) = {}, want {}", VALUES[i], f, SOLUTION_GAMMA[i]);
            panic!("gamma failed");
        }
    }
    for g in VALUES_FOR_GAMMA {
        let f = math::gamma(g[0]);
        let ok = if f64::is_nan(g[1]) || f64::is_infinite(g[1]) || g[1] == 0.0 || f == 0.0 {
            alike(g[1], f)
        } else if g[0] > -50.0 && g[0] <= 171.0 {
            very_close(g[1], f)
        } else {
            close(g[1], f)
        };
        if !ok {
            println!("gamma({}) = {}, want {}", g[0], f, g[1]);
            panic!("gamma special cases failed");
        }
    }
}
