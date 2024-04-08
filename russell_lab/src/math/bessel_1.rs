use super::{PI, SQRT_PI, TWO_129, TWO_M27};

// This implementation is based on j1.go file from Go (1.22.1),
// which, in turn, is based on the FreeBSD code as explained below.
//
// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Bessel function of the first and second kinds of order one.
//
// The original C code and the long comment below are
// from FreeBSD's /usr/src/lib/msun/src/e_j1.c and
// came with this notice. The go code is a simplified
// version of the original C.
//
// ====================================================
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this
// software is freely granted, provided that this notice
// is preserved.
// ====================================================

// R0/S0 on [0, 2]
const R00: f64 = -6.25000000000000000000e-02; // 0xBFB0000000000000
const R01: f64 = 1.40705666955189706048e-03; // 0x3F570D9F98472C61
const R02: f64 = -1.59955631084035597520e-05; // 0xBEF0C5C6BA169668
const R03: f64 = 4.96727999609584448412e-08; // 0x3E6AAAFA46CA0BD9
const S01: f64 = 1.91537599538363460805e-02; // 0x3F939D0B12637E53
const S02: f64 = 1.85946785588630915560e-04; // 0x3F285F56B9CDF664
const S03: f64 = 1.17718464042623683263e-06; // 0x3EB3BFF8333F8498
const S04: f64 = 5.04636257076217042715e-09; // 0x3E35AC88C97DFF2C
const S05: f64 = 1.23542274426137913908e-11; // 0x3DAB2ACFCFB97ED8

/// Evaluates the Bessel function J1(x) for any real x
///
/// Special cases:
///
/// * `J1(NaN)  = NaN`
/// * `J1(±Inf) = 0.0`
/// * `J1(0.0)  = 0.0`
pub fn bessel_j1(x: f64) -> f64 {
    //
    // For tiny x, use j1(x) = x/2 - x**3/16 + x**5/384 - ...
    //
    // Reduce x to |x| since j1(x)=-j1(-x), and:
    //
    // for x in (0,2)
    //
    // j1(x) = x/2 + x*z*R0/S0,  where z = x*x;
    //
    // (precision:  |j1/x - 1/2 - R0/S0 |<2**-61.51 )
    //
    // for x in (2,inf)
    //
    // j1(x) = sqrt(2/(pi*x))*(p1(x)*cos(x1)-q1(x)*sin(x1))
    // y1(x) = sqrt(2/(pi*x))*(p1(x)*sin(x1)+q1(x)*cos(x1))
    //
    // where x1 = x-3*pi/4.
    //
    // Compute sin(x1),cos(x1) as follows:
    //
    // cos(x1) =  cos(x)cos(3pi/4)+sin(x)sin(3pi/4)
    //         =  1/sqrt(2) * (sin(x) - cos(x))
    // sin(x1) =  sin(x)cos(3pi/4)-cos(x)sin(3pi/4)
    //         = -1/sqrt(2) * (sin(x) + cos(x))
    //
    // To avoid cancellation, use:
    //
    // sin(x) +- cos(x) = -cos(2x)/(sin(x) -+ cos(x))

    if f64::is_nan(x) {
        return f64::NAN;
    } else if f64::is_infinite(x) {
        return 0.0;
    } else if x == 0.0 {
        return 0.0;
    }

    let (xx, negative) = if x < 0.0 { (-x, true) } else { (x, false) };

    if xx >= 2.0 {
        let (s, c) = f64::sin_cos(xx);
        let mut ss = -s - c;
        let mut cc = s - c;

        // make sure x+x does not overflow
        if xx < f64::MAX / 2.0 {
            let z = f64::cos(xx + xx);
            if s * c > 0.0 {
                cc = z / ss;
            } else {
                ss = z / cc;
            }
        }

        let z = if xx > TWO_129 {
            (1.0 / SQRT_PI) * cc / f64::sqrt(xx)
        } else {
            let u = pone(xx);
            let v = qone(xx);
            (1.0 / SQRT_PI) * (u * cc - v * ss) / f64::sqrt(xx)
        };

        if negative {
            return -z;
        } else {
            return z;
        }
    }

    if xx < TWO_M27 {
        return 0.5 * xx;
    }

    let mut z = xx * xx;
    let mut r = z * (R00 + z * (R01 + z * (R02 + z * R03)));
    let s = 1.0 + z * (S01 + z * (S02 + z * (S03 + z * (S04 + z * S05))));
    r *= xx;
    z = 0.5 * xx + r / s;

    if negative {
        return -z;
    } else {
        return z;
    }
}

// constant computed with Mathematica
const TWO_M54: f64 = 5.5511151231257827021181583404541015625000000000000e-17; // 2**-54 0x3c90000000000000 Mathematica: N[2^-54, 50]

const U00: f64 = -1.96057090646238940668e-01; // 0xBFC91866143CBC8A
const U01: f64 = 5.04438716639811282616e-02; // 0x3FA9D3C776292CD1
const U02: f64 = -1.91256895875763547298e-03; // 0xBF5F55E54844F50F
const U03: f64 = 2.35252600561610495928e-05; // 0x3EF8AB038FA6B88E
const U04: f64 = -9.19099158039878874504e-08; // 0xBE78AC00569105B8
const V00: f64 = 1.99167318236649903973e-02; // 0x3F94650D3F4DA9F0
const V01: f64 = 2.02552581025135171496e-04; // 0x3F2A8C896C257764
const V02: f64 = 1.35608801097516229404e-06; // 0x3EB6C05A894E8CA6
const V03: f64 = 6.22741452364621501295e-09; // 0x3E3ABF1D5BA69A86
const V04: f64 = 1.66559246207992079114e-11; // 0x3DB25039DACA772A

/// Evaluates the Bessel function Y1(x) for positive real x
///
/// Special cases:
///
/// * `Y1(x < 0.0) = NaN`
/// * `Y1(NaN)     = NaN`
/// * `Y1(+Inf)    = 0.0`
/// * `Y1(0.0)     = -Inf`
pub fn bessel_y1(x: f64) -> f64 {
    //
    // Screen out x<=0 cases: y1(0)=-inf, y1(x<0)=NaN
    //
    // For x<2. Since
    //
    // y1(x) = 2/pi*(j1(x)*(ln(x/2)+Euler)-1/x-x/2+5/64*x**3-...)
    //
    // therefore y1(x)-2/pi*j1(x)*ln(x)-1/x is an odd function.
    //
    // Use the following function to approximate y1,
    //
    // y1(x) = x*U(z)/V(z) + (2/pi)*(j1(x)*ln(x)-1/x), z= x**2
    //
    // where for x in [0,2] (abs err less than 2**-65.89)
    //
    // U(z) = U0[0] + U0[1]*z + ... + U0[4]*z**4
    // V(z) = 1  + v0[0]*z + ... + v0[4]*z**5
    //
    // Note: For tiny x, 1/x dominate y1 and hence
    // y1(tiny) = -2/pi/tiny, (choose tiny<2**-54)
    //
    // For x>=2.
    //
    // y1(x) = sqrt(2/(pi*x))*(p1(x)*sin(x1)+q1(x)*cos(x1))
    //
    // where x1 = x-3*pi/4.
    //
    // Compute sin(x1),cos(x1) by method mentioned in bessel_j1

    if x < 0.0 || f64::is_nan(x) {
        return f64::NAN;
    } else if f64::is_infinite(x) {
        // this handles +Inf since x < 0.0 was handled already
        return 0.0;
    } else if x == 0.0 {
        return f64::NEG_INFINITY;
    }

    if x >= 2.0 {
        let (s, c) = f64::sin_cos(x);
        let mut ss = -s - c;
        let mut cc = s - c;

        // make sure x+x does not overflow
        if x < f64::MAX / 2.0 {
            let z = f64::cos(x + x);
            if s * c > 0.0 {
                cc = z / ss;
            } else {
                ss = z / cc;
            }
        }

        let z = if x > TWO_129 {
            (1.0 / SQRT_PI) * ss / f64::sqrt(x)
        } else {
            let u = pone(x);
            let v = qone(x);
            (1.0 / SQRT_PI) * (u * ss + v * cc) / f64::sqrt(x)
        };

        return z;
    }

    if x <= TWO_M54 {
        return -(2.0 / PI) / x;
    }

    let z = x * x;
    let u = U00 + z * (U01 + z * (U02 + z * (U03 + z * U04)));
    let v = 1.0 + z * (V00 + z * (V01 + z * (V02 + z * (V03 + z * V04))));
    x * (u / v) + (2.0 / PI) * (bessel_j1(x) * f64::ln(x) - 1.0 / x)
}

// For x >= 8, the asymptotic expansions of pone is
//
// 1 + 15/128 s**2 - 4725/2**15 s**4 - ..., where s = 1/x.
//
// We approximate pone by
//
// pone(x) = 1 + (R/S)
//
// where R = pr0 + pr1*s**2 + pr2*s**4 + ... + pr5*s**10
//
// S = 1 + ps0*s**2 + ... + ps4*s**10
//
// and
//
// | pone(x)-1-R/S | <= 2**(-60.06)

// for x in [inf, 8]=1/[0,0.125]
const P1R8: [f64; 6] = [
    0.00000000000000000000e+00, // 0x0000000000000000
    1.17187499999988647970e-01, // 0x3FBDFFFFFFFFFCCE
    1.32394806593073575129e+01, // 0x402A7A9D357F7FCE
    4.12051854307378562225e+02, // 0x4079C0D4652EA590
    3.87474538913960532227e+03, // 0x40AE457DA3A532CC
    7.91447954031891731574e+03, // 0x40BEEA7AC32782DD
];
const P1S8: [f64; 5] = [
    1.14207370375678408436e+02, // 0x405C8D458E656CAC
    3.65093083420853463394e+03, // 0x40AC85DC964D274F
    3.69562060269033463555e+04, // 0x40E20B8697C5BB7F
    9.76027935934950801311e+04, // 0x40F7D42CB28F17BB
    3.08042720627888811578e+04, // 0x40DE1511697A0B2D
];

// for x in [8,4.5454] = 1/[0.125,0.22001]
const P1R5: [f64; 6] = [
    1.31990519556243522749e-11, // 0x3DAD0667DAE1CA7D
    1.17187493190614097638e-01, // 0x3FBDFFFFE2C10043
    6.80275127868432871736e+00, // 0x401B36046E6315E3
    1.08308182990189109773e+02, // 0x405B13B9452602ED
    5.17636139533199752805e+02, // 0x40802D16D052D649
    5.28715201363337541807e+02, // 0x408085B8BB7E0CB7
];
const P1S5: [f64; 5] = [
    5.92805987221131331921e+01, // 0x404DA3EAA8AF633D
    9.91401418733614377743e+02, // 0x408EFB361B066701
    5.35326695291487976647e+03, // 0x40B4E9445706B6FB
    7.84469031749551231769e+03, // 0x40BEA4B0B8A5BB15
    1.50404688810361062679e+03, // 0x40978030036F5E51
];

// for x in[4.5453,2.8571] = 1/[0.2199,0.35001]
const P1R3: [f64; 6] = [
    3.02503916137373618024e-09, // 0x3E29FC21A7AD9EDD
    1.17186865567253592491e-01, // 0x3FBDFFF55B21D17B
    3.93297750033315640650e+00, // 0x400F76BCE85EAD8A
    3.51194035591636932736e+01, // 0x40418F489DA6D129
    9.10550110750781271918e+01, // 0x4056C3854D2C1837
    4.85590685197364919645e+01, // 0x4048478F8EA83EE5
];
const P1S3: [f64; 5] = [
    3.47913095001251519989e+01, // 0x40416549A134069C
    3.36762458747825746741e+02, // 0x40750C3307F1A75F
    1.04687139975775130551e+03, // 0x40905B7C5037D523
    8.90811346398256432622e+02, // 0x408BD67DA32E31E9
    1.03787932439639277504e+02, // 0x4059F26D7C2EED53
];

// for x in [2.8570,2] = 1/[0.3499,0.5]
const P1R2: [f64; 6] = [
    1.07710830106873743082e-07, // 0x3E7CE9D4F65544F4
    1.17176219462683348094e-01, // 0x3FBDFF42BE760D83
    2.36851496667608785174e+00, // 0x4002F2B7F98FAEC0
    1.22426109148261232917e+01, // 0x40287C377F71A964
    1.76939711271687727390e+01, // 0x4031B1A8177F8EE2
    5.07352312588818499250e+00, // 0x40144B49A574C1FE
];
const P1S2: [f64; 5] = [
    2.14364859363821409488e+01, // 0x40356FBD8AD5ECDC
    1.25290227168402751090e+02, // 0x405F529314F92CD5
    2.32276469057162813669e+02, // 0x406D08D8D5A2DBD9
    1.17679373287147100768e+02, // 0x405D6B7ADA1884A9
    8.36463893371618283368e+00, // 0x4020BAB1F44E5192
];

fn pone(x: f64) -> f64 {
    let (p, q) = if x >= 8.0 {
        (&P1R8, &P1S8)
    } else if x >= 4.5454 {
        (&P1R5, &P1S5)
    } else if x >= 2.8571 {
        (&P1R3, &P1S3)
    } else if x >= 2.0 {
        (&P1R2, &P1S2)
    } else {
        panic!("INTERNAL ERROR: x must be ≥ 2.0 for pone");
    };
    let z = 1.0 / (x * x);
    let r = p[0] + z * (p[1] + z * (p[2] + z * (p[3] + z * (p[4] + z * p[5]))));
    let s = 1.0 + z * (q[0] + z * (q[1] + z * (q[2] + z * (q[3] + z * q[4]))));
    1.0 + r / s
}

// For x >= 8, the asymptotic expansions of qone is
//
// 3/8 s - 105/1024 s**3 - ..., where s = 1/x.
//
// Approximate qone by
//
// qone(x) = s*(0.375 + (R/S))
//
// where
//
// R = qr1*s**2 + qr2*s**4 + ... + qr5*s**10
// S = 1 + qs1*s**2 + ... + qs6*s**12
//
// and
//
// | qone(x)/s -0.375-R/S | <= 2**(-61.13)

// for x in [inf, 8] = 1/[0,0.125]
const Q1R8: [f64; 6] = [
    0.00000000000000000000e+00,  // 0x0000000000000000
    -1.02539062499992714161e-01, // 0xBFBA3FFFFFFFFDF3
    -1.62717534544589987888e+01, // 0xC0304591A26779F7
    -7.59601722513950107896e+02, // 0xC087BCD053E4B576
    -1.18498066702429587167e+04, // 0xC0C724E740F87415
    -4.84385124285750353010e+04, // 0xC0E7A6D065D09C6A
];
const Q1S8: [f64; 6] = [
    1.61395369700722909556e+02,  // 0x40642CA6DE5BCDE5
    7.82538599923348465381e+03,  // 0x40BE9162D0D88419
    1.33875336287249578163e+05,  // 0x4100579AB0B75E98
    7.19657723683240939863e+05,  // 0x4125F65372869C19
    6.66601232617776375264e+05,  // 0x412457D27719AD5C
    -2.94490264303834643215e+05, // 0xC111F9690EA5AA18
];

// for x in [8,4.5454] = 1/[0.125,0.22001]
const Q1R5: [f64; 6] = [
    -2.08979931141764104297e-11, // 0xBDB6FA431AA1A098
    -1.02539050241375426231e-01, // 0xBFBA3FFFCB597FEF
    -8.05644828123936029840e+00, // 0xC0201CE6CA03AD4B
    -1.83669607474888380239e+02, // 0xC066F56D6CA7B9B0
    -1.37319376065508163265e+03, // 0xC09574C66931734F
    -2.61244440453215656817e+03, // 0xC0A468E388FDA79D
];
const Q1S5: [f64; 6] = [
    8.12765501384335777857e+01,  // 0x405451B2FF5A11B2
    1.99179873460485964642e+03,  // 0x409F1F31E77BF839
    1.74684851924908907677e+04,  // 0x40D10F1F0D64CE29
    4.98514270910352279316e+04,  // 0x40E8576DAABAD197
    2.79480751638918118260e+04,  // 0x40DB4B04CF7C364B
    -4.71918354795128470869e+03, // 0xC0B26F2EFCFFA004
];

// for x in [4.5454,2.8571] = 1/[0.2199,0.35001] ???
const Q1R3: [f64; 6] = [
    -5.07831226461766561369e-09, // 0xBE35CFA9D38FC84F
    -1.02537829820837089745e-01, // 0xBFBA3FEB51AEED54
    -4.61011581139473403113e+00, // 0xC01270C23302D9FF
    -5.78472216562783643212e+01, // 0xC04CEC71C25D16DA
    -2.28244540737631695038e+02, // 0xC06C87D34718D55F
    -2.19210128478909325622e+02, // 0xC06B66B95F5C1BF6
];
const Q1S3: [f64; 6] = [
    4.76651550323729509273e+01,  // 0x4047D523CCD367E4
    6.73865112676699709482e+02,  // 0x40850EEBC031EE3E
    3.38015286679526343505e+03,  // 0x40AA684E448E7C9A
    5.54772909720722782367e+03,  // 0x40B5ABBAA61D54A6
    1.90311919338810798763e+03,  // 0x409DBC7A0DD4DF4B
    -1.35201191444307340817e+02, // 0xC060E670290A311F
];

// for x in [2.8570,2] = 1/[0.3499,0.5]
const Q1R2: [f64; 6] = [
    -1.78381727510958865572e-07, // 0xBE87F12644C626D2
    -1.02517042607985553460e-01, // 0xBFBA3E8E9148B010
    -2.75220568278187460720e+00, // 0xC006048469BB4EDA
    -1.96636162643703720221e+01, // 0xC033A9E2C168907F
    -4.23253133372830490089e+01, // 0xC04529A3DE104AAA
    -2.13719211703704061733e+01, // 0xC0355F3639CF6E52
];
const Q1S2: [f64; 6] = [
    2.95333629060523854548e+01,  // 0x403D888A78AE64FF
    2.52981549982190529136e+02,  // 0x406F9F68DB821CBA
    7.57502834868645436472e+02,  // 0x4087AC05CE49A0F7
    7.39393205320467245656e+02,  // 0x40871B2548D4C029
    1.55949003336666123687e+02,  // 0x40637E5E3C3ED8D4
    -4.95949898822628210127e+00, // 0xC013D686E71BE86B
];

fn qone(x: f64) -> f64 {
    let (p, q) = if x >= 8.0 {
        (&Q1R8, &Q1S8)
    } else if x >= 4.5454 {
        (&Q1R5, &Q1S5)
    } else if x >= 2.8571 {
        (&Q1R3, &Q1S3)
    } else if x >= 2.0 {
        (&Q1R2, &Q1S2)
    } else {
        panic!("INTERNAL ERROR: x must be ≥ 2.0 for qone");
    };
    let z = 1.0 / (x * x);
    let r = p[0] + z * (p[1] + z * (p[2] + z * (p[3] + z * (p[4] + z * p[5]))));
    let s = 1.0 + z * (q[0] + z * (q[1] + z * (q[2] + z * (q[3] + z * (q[4] + z * q[5])))));
    (0.375 + r / s) / x
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{bessel_j1, bessel_y1, pone, qone, TWO_129, TWO_M54};
    use crate::{approx_eq, assert_alike};

    #[test]
    #[should_panic(expected = "INTERNAL ERROR: x must be ≥ 2.0 for pone")]
    fn pone_panics_on_wrong_input() {
        pone(1.99);
    }

    #[test]
    #[should_panic(expected = "INTERNAL ERROR: x must be ≥ 2.0 for qone")]
    fn qone_panics_on_wrong_input() {
        qone(1.99);
    }

    #[test]
    fn bessel_j1_handles_special_cases() {
        assert!(bessel_j1(f64::NAN).is_nan());
        assert_eq!(bessel_j1(f64::NEG_INFINITY), 0.0);
        assert_eq!(bessel_j1(f64::INFINITY), 0.0);
        assert_eq!(bessel_j1(0.0), 0.0);
    }

    #[test]
    fn bessel_y1_handles_special_cases() {
        assert!(bessel_y1(f64::NEG_INFINITY).is_nan());
        assert!(bessel_y1(-0.01).is_nan());
        assert!(bessel_y1(f64::NAN).is_nan());
        assert_eq!(bessel_y1(f64::INFINITY), 0.0);
        assert_eq!(bessel_y1(0.0), f64::NEG_INFINITY);
    }

    #[test]
    fn bessel_j1_works() {
        // Mathematica: N[BesselJ[1, -123], 100]
        approx_eq(
            bessel_j1(-123.0),
            -0.02156735149890660940086328069036361009670280158394365333670828669644907165480535175323397489229874718,
            1e-17,
        );

        // Mathematica: N[BesselJ[1, -5], 100]
        assert_eq!(
            bessel_j1(-5.0),
            0.3275791375914652220377343219101691327608499046240540186864806450648753089914574086157808718640701942
        );

        // Mathematica: N[BesselJ[1, -2], 100]
        approx_eq(
            bessel_j1(-2.0),
            -0.5767248077568733872024482422691370869203026897196754401211390207640871162896121849483995433063402923,
            1e-15,
        );

        // Mathematica: N[BesselJ[1, -1], 100]
        assert_eq!(
            bessel_j1(-1.0),
            -0.4400505857449335159596822037189149131273723019927652511367581717801382224780155479307965923811982542
        );

        // Mathematica: N[BesselJ[1, 10^-9], 100]
        assert_eq!(
            bessel_j1(1e-9),
            4.999999999999999999375000000000000000026041666666666666666124131944444444444451226128472222222222166e-10
        );

        // Mathematica: N[BesselJ[1, 1], 100]
        assert_eq!(
            bessel_j1(1.0),
            0.4400505857449335159596822037189149131273723019927652511367581717801382224780155479307965923811982542
        );

        // Mathematica: N[BesselJ[1, 2], 100]
        approx_eq(
            bessel_j1(2.0),
            0.5767248077568733872024482422691370869203026897196754401211390207640871162896121849483995433063402923,
            1e-15,
        );

        // Mathematica: N[BesselJ[1, 5], 100]
        assert_eq!(
            bessel_j1(5.0),
            -0.3275791375914652220377343219101691327608499046240540186864806450648753089914574086157808718640701942
        );

        // Mathematica: N[BesselJ[1, 123], 100]
        approx_eq(
            bessel_j1(123.0),
            0.02156735149890660940086328069036361009670280158394365333670828669644907165480535175323397489229874718,
            1e-17,
        );
    }

    #[test]
    fn bessel_y1_works() {
        // Mathematica: N[BesselY[0, 10^-9], 100]
        approx_eq(
            bessel_y1(1e-9),
            -6.366197723675813498680125340511460635200799230620073016920009919302213806667149128128951250692787669e8,
            1e-6,
        );

        // Mathematica: N[BesselY[0, 1], 100]
        approx_eq(
            bessel_y1(1.0),
            -0.7812128213002887165471500000479648205499063907164446078438332461277843915385602167276292380048056346,
            1e-17,
        );

        // Mathematica: N[BesselY[0, 2], 100]
        approx_eq(
            bessel_y1(2.0),
            -0.1070324315409375468883707722774766366874808982350538605257945572313199774994906724960750929874561422,
            1e-16,
        );

        // Mathematica: N[BesselY[0, 5], 100]
        approx_eq(
            bessel_y1(5.0),
            0.1478631433912268448010506754880372053734652184104756133006976595086882659262171067506925531074677045,
            1e-16,
        );

        // Mathematica: N[BesselY[0, 123], 100]
        approx_eq(
            bessel_y1(123.0),
            0.06863489010254926652096999352892516176822669293542798648532852667873692449570610866356639175413101604,
            1e-17,
        );
    }

    #[test]
    fn bessel_j1_edge_cases_work() {
        //
        // x > f64::MAX / 2.0
        //
        // println!("x = {:?}", (f64::MAX / 2.0)); // 8.988465674311579e307
        // println!("j1(x) = {:?}", bessel_j1(8.988465674311579e307));
        // Reference value from Go 1.22.1
        approx_eq(bessel_j1(f64::MAX / 2.0), 5.936112522662019e-155, 1e-310);

        //
        // x > TWO_129
        //
        // Mathematica: N[BesselJ[1, 2  2^129], 100]
        approx_eq(
            bessel_j1(2.0 * TWO_129),
            -2.148812412212001397545607680138144226557017053953123779067461809570040430645106137686812136124982505e-20,
            1e-100,
        );
    }

    #[test]
    fn bessel_y1_edge_cases_work() {
        //
        // x > f64::MAX / 2.0
        //
        // println!("x = {:?}", (f64::MAX / 2.0)); // 8.988465674311579e307
        // println!("y1(x) = {:?}", bessel_y1(8.988465674311579e307));
        // Reference value from Go 1.22.1
        approx_eq(bessel_y1(f64::MAX / 2.0), -5.965640685080747e-155, 1e-310);

        //
        // x <= TWO_M54
        //
        approx_eq(
            bessel_y1(TWO_M54),
            -1.146832227844531729100095246803011708838780681972379423086390888013625612852380226098068118499203402e16,
            1e-100,
        );
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

    const SOLUTION_J1: [f64; 10] = [
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

    const SOLUTION_Y1: [f64; 10] = [
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

    const SC_VALUES: [f64; 4] = [f64::NEG_INFINITY, 0.0, f64::INFINITY, f64::NAN];

    const SC_SOLUTION_J1: [f64; 4] = [0.0, 0.0, 0.0, f64::NAN];

    const SC_SOLUTION_Y1: [f64; 5] = [f64::NAN, f64::NEG_INFINITY, 0.0, f64::NAN, f64::NAN];

    #[test]
    fn test_bessel_j1() {
        for (i, v) in VALUES.iter().enumerate() {
            let f = bessel_j1(*v);
            approx_eq(SOLUTION_J1[i], f, 1e-15);
        }
        for (i, v) in SC_VALUES.iter().enumerate() {
            let f = bessel_j1(*v);
            assert_alike(SC_SOLUTION_J1[i], f);
        }
    }

    #[test]
    fn test_bessel_y1() {
        for (i, v) in VALUES.iter().enumerate() {
            let a = f64::abs(*v);
            let f = bessel_y1(a);
            approx_eq(SOLUTION_Y1[i], f, 1e-15);
        }
        for (i, v) in SC_VALUES.iter().enumerate() {
            let f = bessel_y1(*v);
            assert_alike(SC_SOLUTION_Y1[i], f);
        }
    }
}
