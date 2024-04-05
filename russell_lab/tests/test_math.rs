#![allow(unused, non_upper_case_globals, non_snake_case)]

use num_traits::Signed;
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

const vf: [f64; 10] = [
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

const erf: [f64; 10] = [
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

const erfc: [f64; 10] = [
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

const erfinv: [f64; 10] = [
    4.746037673358033586786350696e-01,
    8.559054432692110956388764172e-01,
    -2.45427830571707336251331946e-02,
    -4.78116683518973366268905506e-01,
    1.479804430319470983648120853e+00,
    2.654485787128896161882650211e-01,
    5.027444534221520197823192493e-01,
    2.466703532707627818954585670e-01,
    1.632011465103005426240343116e-01,
    -1.06672334642196900710000389e+00,
];

struct Fi {
    f: f64,
    i: i32,
}

const frexp: [Fi; 10] = [
    Fi {
        f: 6.2237649061045918750e-01,
        i: 3,
    },
    Fi {
        f: 9.6735905932226306250e-01,
        i: 3,
    },
    Fi {
        f: -5.5376011438400318000e-01,
        i: -1,
    },
    Fi {
        f: -6.2632545228388436250e-01,
        i: 3,
    },
    Fi {
        f: 6.02268356699901081250e-01,
        i: 4,
    },
    Fi {
        f: 7.3159430981099115000e-01,
        i: 2,
    },
    Fi {
        f: 6.5363542893241332500e-01,
        i: 3,
    },
    Fi {
        f: 6.8198497760900255000e-01,
        i: 2,
    },
    Fi {
        f: 9.1265404584042750000e-01,
        i: 1,
    },
    Fi {
        f: -5.4287029803597508250e-01,
        i: 4,
    },
];

const gamma: [f64; 10] = [
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

const bessel_j0: [f64; 10] = [
    -1.8444682230601672018219338e-01,
    2.27353668906331975435892e-01,
    9.809259936157051116270273e-01,
    -1.741170131426226587841181e-01,
    -2.1389448451144143352039069e-01,
    -2.340905848928038763337414e-01,
    -1.0029099691890912094586326e-01,
    -1.5466726714884328135358907e-01,
    3.252650187653420388714693e-01,
    -8.72218484409407250005360235e-03,
];

const bessel_j1: [f64; 10] = [
    -3.251526395295203422162967e-01,
    1.893581711430515718062564e-01,
    -1.3711761352467242914491514e-01,
    3.287486536269617297529617e-01,
    1.3133899188830978473849215e-01,
    3.660243417832986825301766e-01,
    -3.4436769271848174665420672e-01,
    4.329481396640773768835036e-01,
    5.8181350531954794639333955e-01,
    -2.7030574577733036112996607e-01,
];

const bessel_j2: [f64; 10] = [
    5.3837518920137802565192769e-02,
    -1.7841678003393207281244667e-01,
    9.521746934916464142495821e-03,
    4.28958355470987397983072e-02,
    2.4115371837854494725492872e-01,
    4.842458532394520316844449e-01,
    -3.142145220618633390125946e-02,
    4.720849184745124761189957e-01,
    3.122312022520957042957497e-01,
    7.096213118930231185707277e-02,
];

const bessel_jM3: [f64; 10] = [
    -3.684042080996403091021151e-01,
    2.8157665936340887268092661e-01,
    4.401005480841948348343589e-04,
    3.629926999056814081597135e-01,
    3.123672198825455192489266e-02,
    -2.958805510589623607540455e-01,
    -3.2033177696533233403289416e-01,
    -2.592737332129663376736604e-01,
    -1.0241334641061485092351251e-01,
    -2.3762660886100206491674503e-01,
];

const lgamma: [Fi; 10] = [
    Fi {
        f: 3.146492141244545774319734e+00,
        i: 1,
    },
    Fi {
        f: 8.003414490659126375852113e+00,
        i: 1,
    },
    Fi {
        f: 1.517575735509779707488106e+00,
        i: -1,
    },
    Fi {
        f: -2.588480028182145853558748e-01,
        i: 1,
    },
    Fi {
        f: 1.1989897050205555002007985e+01,
        i: 1,
    },
    Fi {
        f: 6.262899811091257519386906e-01,
        i: 1,
    },
    Fi {
        f: 3.5287924899091566764846037e+00,
        i: 1,
    },
    Fi {
        f: 4.5725644770161182299423372e-01,
        i: 1,
    },
    Fi {
        f: -6.363667087767961257654854e-02,
        i: 1,
    },
    Fi {
        f: -1.077385130910300066425564e+01,
        i: -1,
    },
];

const modf: [[f64; 2]; 10] = [
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

const bessel_y0: [f64; 10] = [
    -3.053399153780788357534855e-01,
    1.7437227649515231515503649e-01,
    -8.6221781263678836910392572e-01,
    -3.100664880987498407872839e-01,
    1.422200649300982280645377e-01,
    4.000004067997901144239363e-01,
    -3.3340749753099352392332536e-01,
    4.5399790746668954555205502e-01,
    4.8290004112497761007536522e-01,
    2.7036697826604756229601611e-01,
];

const bessel_y1: [f64; 10] = [
    0.15494213737457922210218611,
    -0.2165955142081145245075746,
    -2.4644949631241895201032829,
    0.1442740489541836405154505,
    0.2215379960518984777080163,
    0.3038800915160754150565448,
    0.0691107642452362383808547,
    0.2380116417809914424860165,
    -0.20849492979459761009678934,
    0.0242503179793232308250804,
];

const bessel_y2: [f64; 10] = [
    0.3675780219390303613394936,
    -0.23034826393250119879267257,
    -16.939677983817727205631397,
    0.367653980523052152867791,
    -0.0962401471767804440353136,
    -0.1923169356184851105200523,
    0.35984072054267882391843766,
    -0.2794987252299739821654982,
    -0.7113490692587462579757954,
    -0.2647831587821263302087457,
];

const bessel_yM3: [f64; 10] = [
    -0.14035984421094849100895341,
    -0.097535139617792072703973,
    242.25775994555580176377379,
    -0.1492267014802818619511046,
    0.26148702629155918694500469,
    0.56675383593895176530394248,
    -0.206150264009006981070575,
    0.64784284687568332737963658,
    1.3503631555901938037008443,
    0.1461869756579956803341844,
];

const vferfSC: [f64; 7] = [f64::NEG_INFINITY, -0.0, 0.0, f64::INFINITY, f64::NAN, -1000.0, 1000.0];

const erfSC: [f64; 7] = [-1.0, -0.0, 0.0, 1.0, f64::NAN, -1.0, 1.0];

const vferfcSC: [f64; 5] = [f64::NEG_INFINITY, f64::INFINITY, f64::NAN, -1000.0, 1000.0];

const erfcSC: [f64; 5] = [2.0, 0.0, f64::NAN, 2.0, 0.0];

const vferfinvSC: [f64; 6] = [1.0, -1.0, 0.0, f64::NEG_INFINITY, f64::INFINITY, f64::NAN];

const erfinvSC: [f64; 6] = [f64::INFINITY, f64::NEG_INFINITY, 0.0, f64::NAN, f64::NAN, f64::NAN];

const vferfcinvSC: [f64; 6] = [0.0, 2.0, 1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN];

const erfcinvSC: [f64; 6] = [f64::INFINITY, f64::NEG_INFINITY, 0.0, f64::NAN, f64::NAN, f64::NAN];

const vffrexpSC: [f64; 5] = [f64::NEG_INFINITY, -0.0, 0.0, f64::INFINITY, f64::NAN];

const frexpSC: [Fi; 5] = [
    Fi {
        f: f64::NEG_INFINITY,
        i: 0,
    },
    Fi { f: -0.0, i: 0 },
    Fi { f: 0.0, i: 0 },
    Fi { f: f64::INFINITY, i: 0 },
    Fi { f: f64::NAN, i: 0 },
];

const vfgamma: [[f64; 2]; 71] = [
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

const vfbessel_j0SC: [f64; 4] = [f64::NEG_INFINITY, 0.0, f64::INFINITY, f64::NAN];

const bessel_j0SC: [f64; 4] = [0.0, 1.0, 0.0, f64::NAN];

const bessel_j1SC: [f64; 4] = [0.0, 0.0, 0.0, f64::NAN];

const bessel_j2SC: [f64; 4] = [0.0, 0.0, 0.0, f64::NAN];

const bessel_jM3SC: [f64; 4] = [0.0, 0.0, 0.0, f64::NAN];

const vfldexpSC: [Fi; 11] = [
    Fi { f: 0.0, i: 0 },
    Fi { f: 0.0, i: -1075 },
    Fi { f: 0.0, i: 1024 },
    Fi { f: -0.0, i: 0 },
    Fi { f: -0.0, i: -1075 },
    Fi { f: -0.0, i: 1024 },
    Fi { f: f64::INFINITY, i: 0 },
    Fi {
        f: f64::INFINITY,
        i: -1024,
    },
    Fi {
        f: f64::NEG_INFINITY,
        i: 0,
    },
    Fi {
        f: f64::NEG_INFINITY,
        i: -1024,
    },
    Fi { f: f64::NAN, i: -1024 },
    // Fi {
    //     f: 10.0,
    //     i: 72057594037927936,
    // }, // from Go
    // Fi {
    //     f: 10.0,
    //     i: -72057594037927936,
    // },
];

const ldexpSC: [f64; 13] = [
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

const vflgammaSC: [f64; 7] = [f64::NEG_INFINITY, -3.0, 0.0, 1.0, 2.0, f64::INFINITY, f64::NAN];

const lgammaSC: [Fi; 7] = [
    Fi {
        f: f64::NEG_INFINITY,
        i: 1,
    },
    Fi { f: f64::INFINITY, i: 1 },
    Fi { f: f64::INFINITY, i: 1 },
    Fi { f: 0.0, i: 1 },
    Fi { f: 0.0, i: 1 },
    Fi { f: f64::INFINITY, i: 1 },
    Fi { f: f64::NAN, i: 1 },
];

const vfmodfSC: [f64; 4] = [f64::NEG_INFINITY, -0.0, f64::INFINITY, f64::NAN];

// #include <stdio.h>
// #include <math.h>
// int main () {
//   double param, fractpart, intpart;
//   param = -1.0/0.0;
//   fractpart = modf (param , &intpart);
//   printf ("%f = %f + %f \n", param, intpart, fractpart);
//   return 0;
// }
// // Output:
// // -inf = -inf + -0.000000

const modfSC: [[f64; 2]; 4] = [
    [f64::NEG_INFINITY, -0.0], // note that Go returns (-Inf, NaN); but the C code above returns (-Inf, -0.0)
    [-0.0, -0.0],
    [f64::INFINITY, 0.0], // note that Go returns (Inf, NaN); but the C code above returns (Inf, 0.0)
    [f64::NAN, f64::NAN],
];

const vfsignbitSC: [f64; 5] = [f64::NEG_INFINITY, -0.0, 0.0, f64::INFINITY, f64::NAN];

const signbitSC: [bool; 5] = [true, true, false, false, false];

const vfbessel_y0SC: [f64; 5] = [f64::NEG_INFINITY, 0.0, f64::INFINITY, f64::NAN, -1.0];

const bessel_y0SC: [f64; 5] = [f64::NAN, f64::NEG_INFINITY, 0.0, f64::NAN, f64::NAN];

const bessel_y1SC: [f64; 5] = [f64::NAN, f64::NEG_INFINITY, 0.0, f64::NAN, f64::NAN];

const bessel_y2SC: [f64; 5] = [f64::NAN, f64::NEG_INFINITY, 0.0, f64::NAN, f64::NAN];

const bessel_yM3SC: [f64; 5] = [f64::NAN, f64::INFINITY, 0.0, f64::NAN, f64::NAN];

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

const frexpBC: [Fi; 8] = [
    Fi { f: 0.5, i: -1021 },
    Fi {
        f: 0.99999999999999978,
        i: -1022,
    },
    Fi { f: 0.5, i: -1073 },
    Fi {
        f: 0.99999999999999989,
        i: 1024,
    },
    Fi { f: -0.5, i: -1021 },
    Fi {
        f: -0.99999999999999978,
        i: -1022,
    },
    Fi { f: -0.5, i: -1073 },
    Fi {
        f: -0.99999999999999989,
        i: 1024,
    },
];

const vfldexpBC: [Fi; 10] = [
    Fi {
        f: SmallestNormalFloat64,
        i: -52,
    },
    Fi {
        f: LargestSubnormalFloat64,
        i: -51,
    },
    Fi {
        f: SmallestNonzeroFloat64,
        i: 1074,
    },
    Fi {
        f: f64::MAX,
        i: -(1023 + 1074),
    },
    Fi { f: 1.0, i: -1075 },
    Fi { f: -1.0, i: -1075 },
    Fi { f: 1.0, i: 1024 },
    Fi { f: -1.0, i: 1024 },
    Fi {
        f: 1.0000000000000002,
        i: -1075,
    },
    Fi { f: 1.0, i: -1075 },
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

fn so_close(a: f64, b: f64, e: f64) -> bool {
    tolerance(a, b, e)
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
    for i in 0..vf.len() {
        let a = vf[i] / 10.0;
        let f = math::erf(a);
        if !very_close(erf[i], f) {
            println!("erf({}) = {}, want {}", a, f, erf[i]);
            panic!("erf failed");
        }
    }
    for i in 0..(vferfSC.len()) {
        let f = math::erf(vferfSC[i]);
        if !alike(erfSC[i], f) {
            println!("erf({}) = {}, want {}", vferfSC[i], f, erfSC[i]);
            panic!("erf special cases failed");
        }
    }
}

#[test]
fn test_erfc() {
    for i in 0..vf.len() {
        let a = vf[i] / 10.0;
        let f = math::erfc(a);
        if !very_close(erfc[i], f) {
            println!("erfc({}) = {}, want {}", a, f, erfc[i]);
            panic!("erfc failed");
        }
    }
    for i in 0..vferfcSC.len() {
        let f = math::erfc(vferfcSC[i]);
        if !alike(erfcSC[i], f) {
            println!("erfc({}) = {}, want {}", vferfcSC[i], f, erfcSC[i]);
            panic!("erfc special cases failed");
        }
    }
}

// #[test]
fn test_erf_inv() {
    for i in 0..vf.len() {
        let a = vf[i] / 10.0;
        let f = math::erf_inv(a);
        if !very_close(erfinv[i], f) {
            println!("erf_inv({}) = {}, want {}", a, f, erfinv[i]);
            panic!("erf_inv failed");
        }
    }
    for i in 0..vferfinvSC.len() {
        let f = math::erf_inv(vferfinvSC[i]);
        if !alike(erfinvSC[i], f) {
            println!("erf_inv({}) = {}, want {}", vferfinvSC[i], f, erfinvSC[i]);
            panic!("erf_inv special cases failed");
        }
    }
    let mut x = -0.9;
    while x <= 0.90 {
        let f = math::erf(math::erf_inv(x));
        if !close(x, f) {
            println!("erf(erf_inv({})) = {}, want {}", x, f, x);
            panic!("erf(erf_inv(x)) = x failed");
        }
        x += 1e-2;
    }
    let mut x = -0.9;
    while x <= 0.90 {
        let f = math::erf_inv(math::erf(x));
        if !close(x, f) {
            println!("erf_inv(erf({})) = {}, want {}", x, f, x);
            panic!("erf_inv(erf(x)) = x failed");
        }
        x += 1e-2;
    }
}

// #[test]
fn test_erfc_inv() {
    for i in 0..vf.len() {
        let a = 1.0 - (vf[i] / 10.0);
        let f = math::erfc_inv(a);
        if !very_close(erfinv[i], f) {
            println!("erfc_inv({}) = {}, want {}", a, f, erfinv[i]);
            panic!("erfc_inv failed");
        }
    }
    for i in 0..vferfcinvSC.len() {
        let f = math::erfc_inv(vferfcinvSC[i]);
        if !alike(erfcinvSC[i], f) {
            println!("erfc_inv({}) = {}, want {}", vferfcinvSC[i], f, erfcinvSC[i]);
            panic!("erfc_inv special cases failed");
        }
    }
    let mut x = 0.1;
    while x <= 1.9 {
        let f = math::erfc(math::erfc_inv(x));
        if !close(x, f) {
            println!("erfc(erfc_inv({})) = {}, want {}", x, f, x);
            panic!("erfc(erfc_inv(x)) = x");
        }
        x += 1e-2;
    }
    let mut x = 0.1;
    while x <= 1.9 {
        let f = math::erfc_inv(math::erfc(x));
        if !close(x, f) {
            println!("erfc_inv(erfc({})) = {}, want {}", x, f, x);
            panic!("erfc_inv(erfc(x)) = x");
        }
        x += 1e-2;
    }
}

#[test]
fn test_frexp() {
    for i in 0..vf.len() {
        let (f, j) = math::frexp(vf[i]);
        if !very_close(frexp[i].f, f) || frexp[i].i != j {
            println!(
                "frexp({}) = ({}, {}); want ({}, {})",
                vf[i], f, j, frexp[i].f, frexp[i].i
            );
            panic!("frexp failed");
        }
    }
    for i in 0..vffrexpSC.len() {
        let (f, j) = math::frexp(vffrexpSC[i]);
        if !alike(frexpSC[i].f, f) || frexpSC[i].i != j {
            println!(
                "frexp({}) = ({}, {}); want ({}, {})",
                vffrexpSC[i], f, j, frexpSC[i].f, frexpSC[i].i
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
    for i in 0..vf.len() {
        let f = math::ldexp(frexp[i].f, frexp[i].i);
        if !very_close(vf[i], f) {
            println!("ldexp({}, {}) = {}, want {}", frexp[i].f, frexp[i].i, f, vf[i]);
            panic!("ldexp failed");
        }
    }
    for i in 0..vffrexpSC.len() {
        let f = math::ldexp(frexpSC[i].f, frexpSC[i].i);
        if !alike(vffrexpSC[i], f) {
            println!(
                "ldexp({}, {}) = {}, want {}",
                frexpSC[i].f, frexpSC[i].i, f, vffrexpSC[i]
            );
            panic!("ldexp special cases failed");
        }
    }
    for i in 0..vfldexpSC.len() {
        let f = math::ldexp(vfldexpSC[i].f, vfldexpSC[i].i);
        if !alike(ldexpSC[i], f) {
            println!(
                "ldexp({}, {}) = {}, want {}",
                vfldexpSC[i].f, vfldexpSC[i].i, f, ldexpSC[i]
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
    for i in 0..vf.len() {
        let (f, g) = math::split_integer_fractional(vf[i]);
        if !very_close(modf[i][0], f) || !very_close(modf[i][1], g) {
            println!(
                "split_integer_fractional({}) = ({}, {}); want ({}, {})",
                vf[i], f, g, modf[i][0], modf[i][1]
            );
            panic!("split_integer_fractional failed");
        }
    }
    for i in 0..vfmodfSC.len() {
        let (f, g) = math::split_integer_fractional(vfmodfSC[i]);
        if !alike(modfSC[i][0], f) || !alike(modfSC[i][1], g) {
            println!(
                "split_integer_fractional({}) = ({}, {}); want ({}, {})",
                vfmodfSC[i], f, g, modfSC[i][0], modfSC[i][1]
            );
            panic!("split_integer_fractional special cases failed");
        }
    }
}

#[test]
fn test_gamma() {
    for i in 0..vf.len() {
        let f = math::gamma(vf[i]);
        if !close(gamma[i], f) {
            println!("gamma({}) = {}, want {}", vf[i], f, gamma[i]);
            panic!("gamma failed");
        }
    }
    for g in vfgamma {
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

// #[test]
fn test_ln_gamma() {
    for i in 0..vf.len() {
        let (f, s) = math::ln_gamma(vf[i]);
        if !close(lgamma[i].f, f) || lgamma[i].i != s {
            println!(
                "ln_gamma({}) = {}, {}, want {}, {}",
                vf[i], f, s, lgamma[i].f, lgamma[i].i
            );
            panic!("ln_gamma failed");
        }
    }
    for i in 0..vflgammaSC.len() {
        let (f, s) = math::ln_gamma(vflgammaSC[i]);
        if !alike(lgammaSC[i].f, f) || lgammaSC[i].i != s {
            println!(
                "ln_gamma({}) = {}, {}, want {}, {}",
                vflgammaSC[i], f, s, lgammaSC[i].f, lgammaSC[i].i
            );
            panic!("ln_gamma special cases failed");
        }
    }
}

#[test]
fn test_bessel_j0() {
    for i in 0..vf.len() {
        let f = math::bessel_j0(vf[i]);
        if !so_close(bessel_j0[i], f, 4e-14) {
            println!("bessel_j0({}) = {}, want {}", vf[i], f, bessel_j0[i]);
            panic!("bessel_j0 failed");
        }
    }
    for i in 0..vfbessel_j0SC.len() {
        let f = math::bessel_j0(vfbessel_j0SC[i]);
        if !alike(bessel_j0SC[i], f) {
            println!("bessel_j0({}) = {}, want {}", vfbessel_j0SC[i], f, bessel_j0SC[i]);
            panic!("bessel_j0 special cases failed");
        }
    }
}

#[test]
fn test_bessel_j1() {
    for i in 0..vf.len() {
        let f = math::bessel_j1(vf[i]);
        if !close(bessel_j1[i], f) {
            println!("bessel_j1({}) = {}, want {}", vf[i], f, bessel_j1[i]);
            panic!("bessel_j1 failed");
        }
    }
    for i in 0..vfbessel_j0SC.len() {
        let f = math::bessel_j1(vfbessel_j0SC[i]);
        if !alike(bessel_j1SC[i], f) {
            println!("bessel_j1({}) = {}, want {}", vfbessel_j0SC[i], f, bessel_j1SC[i]);
            panic!("bessel_j1 special cases failed");
        }
    }
}

#[test]
fn test_bessel_jn() {
    for i in 0..vf.len() {
        let f = math::bessel_jn(2, vf[i]);
        if !close(bessel_j2[i], f) {
            println!("bessel_jn(2, {}) = {}, want {}", vf[i], f, bessel_j2[i]);
            panic!("bessel_jn(2, x) failed");
        }
        let f = math::bessel_jn(-3, vf[i]);
        if !close(bessel_jM3[i], f) {
            println!("bessel_Jn(-3, {}) = {}, want {}", vf[i], f, bessel_jM3[i]);
            panic!("bessel_jn(-3, x) failed");
        }
    }
    for i in 0..vfbessel_j0SC.len() {
        let f = math::bessel_jn(2, vfbessel_j0SC[i]);
        if !alike(bessel_j2SC[i], f) {
            println!("bessel_jn(2, {}) = {}, want {}", vfbessel_j0SC[i], f, bessel_j2SC[i]);
            panic!("bessel_jn(2, x) special cases failed");
        }
        let f = math::bessel_jn(-3, vfbessel_j0SC[i]);
        if !alike(bessel_jM3SC[i], f) {
            println!("bessel_jn(-3, {}) = {}, want {}", vfbessel_j0SC[i], f, bessel_jM3SC[i]);
            panic!("bessel_jn(-3, x) special cases failed");
        }
    }
}

#[test]
fn test_bessel_y0() {
    for i in 0..vf.len() {
        let a = f64::abs(vf[i]);
        let f = math::bessel_y0(a);
        if !close(bessel_y0[i], f) {
            println!("bessel_y0({}) = {}, want {}", a, f, bessel_y0[i]);
            panic!("bessel_y0 failed");
        }
    }
    for i in 0..vfbessel_y0SC.len() {
        let f = math::bessel_y0(vfbessel_y0SC[i]);
        if !alike(bessel_y0SC[i], f) {
            println!("bessel_y0({}) = {}, want {}", vfbessel_y0SC[i], f, bessel_y0SC[i]);
            panic!("bessel_y0 special cases failed");
        }
    }
}

#[test]
fn test_bessel_y1() {
    for i in 0..vf.len() {
        let a = f64::abs(vf[i]);
        let f = math::bessel_y1(a);
        if !so_close(bessel_y1[i], f, 2e-14) {
            println!("bessel_y1({}) = {}, want {}", a, f, bessel_y1[i]);
            panic!("bessel_y1 failed");
        }
    }
    for i in 0..vfbessel_y0SC.len() {
        let f = math::bessel_y1(vfbessel_y0SC[i]);
        if !alike(bessel_y1SC[i], f) {
            println!("bessel_y1({}) = {}, want {}", vfbessel_y0SC[i], f, bessel_y1SC[i]);
            panic!("bessel_y1 special cases failed");
        }
    }
}

#[test]
fn test_bessel_yn() {
    for i in 0..vf.len() {
        let a = f64::abs(vf[i]);
        let f = math::bessel_yn(2, a);
        if !close(bessel_y2[i], f) {
            println!("bessel_yn(2, {}) = {}, want {}", a, f, bessel_y2[i]);
            panic!("bessel_yn(2, x) failed");
        }
        let f = math::bessel_yn(-3, a);
        if !close(bessel_yM3[i], f) {
            println!("bessel_yn(-3, {}) = {}, want {}", a, f, bessel_yM3[i]);
            panic!("bessel_yn(-3, x) failed");
        }
    }
    for i in 0..vfbessel_y0SC.len() {
        let f = math::bessel_yn(2, vfbessel_y0SC[i]);
        if !alike(bessel_y2SC[i], f) {
            println!("bessel_yn(2, {}) = {}, want {}", vfbessel_y0SC[i], f, bessel_y2SC[i]);
            panic!("bessel_yn(2, x) special cases failed");
        }
        let f = math::bessel_yn(-3, vfbessel_y0SC[i]);
        if !alike(bessel_yM3SC[i], f) {
            println!("bessel_yn(-3, {}) = {}, want {}", vfbessel_y0SC[i], f, bessel_yM3SC[i]);
            panic!("bessel_yn(-3, x) special cases failed");
        }
    }
    let f = math::bessel_yn(0, 0.0);
    if !alike(f64::NEG_INFINITY, f) {
        println!("bessel_yn(0, 0) = {}, want {}", f, f64::NEG_INFINITY);
        panic!("bessel_yn(0, 0) failed");
    }
}
