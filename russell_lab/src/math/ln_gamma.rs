use super::{modulo, PI};

// This implementation is based on gamma.go file from Go (1.22.1),
// which, in turn, is based on the code described below.
//
// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Floating-point logarithm of the Gamma function.
//
// The original C code and the long comment below are
// from FreeBSD's /usr/src/lib/msun/src/e_lgamma_r.c and
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

// Method:
//   1. Argument Reduction for 0 < x <= 8
//      Since gamma(1+s)=s*gamma(s), for x in [0,8], we may
//      reduce x to a number in [1.5,2.5] by
//              lgamma(1+s) = log(s) + lgamma(s)
//      for example,
//              lgamma(7.3) = log(6.3) + lgamma(6.3)
//                          = log(6.3*5.3) + lgamma(5.3)
//                          = log(6.3*5.3*4.3*3.3*2.3) + lgamma(2.3)
//   2. Polynomial approximation of lgamma around its
//      minimum (ymin=1.461632144968362245) to maintain monotonicity.
//      On [ymin-0.23, ymin+0.27] (i.e., [1.23164,1.73163]), use
//              Let z = x-ymin;
//              lgamma(x) = -1.214862905358496078218 + z**2*poly(z)
//              poly(z) is a 14 degree polynomial.
//   2. Rational approximation in the primary interval [2,3]
//      We use the following approximation:
//              s = x-2.0;
//              lgamma(x) = 0.5*s + s*P(s)/Q(s)
//      with accuracy
//              |P/Q - (lgamma(x)-0.5s)| < 2**-61.71
//      Our algorithms are based on the following observation
//
//                             zeta(2)-1    2    zeta(3)-1    3
// lgamma(2+s) = s*(1-Euler) + --------- * s  -  --------- * s  + ...
//                                 2                 3
//
//      where Euler = 0.5772156649... is the Euler constant, which
//      is very close to 0.5.
//
//   3. For x>=8, we have
//      lgamma(x)~(x-0.5)log(x)-x+0.5*log(2pi)+1/(12x)-1/(360x**3)+....
//      (better formula:
//         lgamma(x)~(x-0.5)*(log(x)-1)-.5*(log(2pi)-1) + ...)
//      Let z = 1/x, then we approximation
//              f(z) = lgamma(x) - (x-0.5)(log(x)-1)
//      by
//                                  3       5             11
//              w = w0 + w1*z + w2*z  + w3*z  + ... + w6*z
//      where
//              |w - f(z)| < 2**-58.74
//
//   4. For negative x, since (G is gamma function)
//              -x*G(-x)*G(x) = pi/sin(pi*x),
//      we have
//              G(x) = pi/(sin(pi*x)*(-x)*G(-x))
//      since G(-x) is positive, sign(G(x)) = sign(sin(pi*x)) for x<0
//      Hence, for x<0, sign_gam = sign(sin(pi*x)) and
//              lgamma(x) = log(|Gamma(x)|)
//                        = log(pi/(|x*sin(pi*x)|)) - lgamma(-x);
//      Note: one should avoid computing pi*(-x) directly in the
//            computation of sin(pi*(-x)).
//
//   5. Special Cases
//              lgamma(2+s) ~ s*(1-Euler) for tiny s
//              lgamma(1)=lgamma(2)=0
//              lgamma(x) ~ -log(x) for tiny x
//              lgamma(0) = lgamma(inf) = inf
//              lgamma(-integer) = +-inf
//
//

const LA: [f64; 12] = [
    7.72156649015328655494e-02, // 0x3FB3C467E37DB0C8
    3.22467033424113591611e-01, // 0x3FD4A34CC4A60FAD
    6.73523010531292681824e-02, // 0x3FB13E001A5562A7
    2.05808084325167332806e-02, // 0x3F951322AC92547B
    7.38555086081402883957e-03, // 0x3F7E404FB68FEFE8
    2.89051383673415629091e-03, // 0x3F67ADD8CCB7926B
    1.19270763183362067845e-03, // 0x3F538A94116F3F5D
    5.10069792153511336608e-04, // 0x3F40B6C689B99C00
    2.20862790713908385557e-04, // 0x3F2CF2ECED10E54D
    1.08011567247583939954e-04, // 0x3F1C5088987DFB07
    2.52144565451257326939e-05, // 0x3EFA7074428CFA52
    4.48640949618915160150e-05, // 0x3F07858E90A45837
];

const LR: [f64; 7] = [
    1.0,                        // placeholder
    1.39200533467621045958e+00, // 0x3FF645A762C4AB74
    7.21935547567138069525e-01, // 0x3FE71A1893D3DCDC
    1.71933865632803078993e-01, // 0x3FC601EDCCFBDF27
    1.86459191715652901344e-02, // 0x3F9317EA742ED475
    7.77942496381893596434e-04, // 0x3F497DDACA41A95B
    7.32668430744625636189e-06, // 0x3EDEBAF7A5B38140
];

const LS: [f64; 7] = [
    -7.72156649015328655494e-02, // 0xBFB3C467E37DB0C8
    2.14982415960608852501e-01,  // 0x3FCB848B36E20878
    3.25778796408930981787e-01,  // 0x3FD4D98F4F139F59
    1.46350472652464452805e-01,  // 0x3FC2BB9CBEE5F2F7
    2.66422703033638609560e-02,  // 0x3F9B481C7E939961
    1.84028451407337715652e-03,  // 0x3F5E26B67368F239
    3.19475326584100867617e-05,  // 0x3F00BFECDD17E945
];

const LT: [f64; 15] = [
    4.83836122723810047042e-01,  // 0x3FDEF72BC8EE38A2
    -1.47587722994593911752e-01, // 0xBFC2E4278DC6C509
    6.46249402391333854778e-02,  // 0x3FB08B4294D5419B
    -3.27885410759859649565e-02, // 0xBFA0C9A8DF35B713
    1.79706750811820387126e-02,  // 0x3F9266E7970AF9EC
    -1.03142241298341437450e-02, // 0xBF851F9FBA91EC6A
    6.10053870246291332635e-03,  // 0x3F78FCE0E370E344
    -3.68452016781138256760e-03, // 0xBF6E2EFFB3E914D7
    2.25964780900612472250e-03,  // 0x3F6282D32E15C915
    -1.40346469989232843813e-03, // 0xBF56FE8EBF2D1AF1
    8.81081882437654011382e-04,  // 0x3F4CDF0CEF61A8E9
    -5.38595305356740546715e-04, // 0xBF41A6109C73E0EC
    3.15632070903625950361e-04,  // 0x3F34AF6D6C0EBBF7
    -3.12754168375120860518e-04, // 0xBF347F24ECC38C38
    3.35529192635519073543e-04,  // 0x3F35FD3EE8C2D3F4
];

const LU: [f64; 6] = [
    -7.72156649015328655494e-02, // 0xBFB3C467E37DB0C8
    6.32827064025093366517e-01,  // 0x3FE4401E8B005DFF
    1.45492250137234768737e+00,  // 0x3FF7475CD119BD6F
    9.77717527963372745603e-01,  // 0x3FEF497644EA8450
    2.28963728064692451092e-01,  // 0x3FCD4EAEF6010924
    1.33810918536787660377e-02,  // 0x3F8B678BBF2BAB09
];

const LV: [f64; 6] = [
    1.0,
    2.45597793713041134822e+00, // 0x4003A5D7C2BD619C
    2.12848976379893395361e+00, // 0x40010725A42B18F5
    7.69285150456672783825e-01, // 0x3FE89DFBE45050AF
    1.04222645593369134254e-01, // 0x3FBAAE55D6537C88
    3.21709242282423911810e-03, // 0x3F6A5ABB57D0CF61
];

const LW: [f64; 7] = [
    4.18938533204672725052e-01,  // 0x3FDACFE390C97D69
    8.33333333333329678849e-02,  // 0x3FB555555555553B
    -2.77777777728775536470e-03, // 0xBF66C16C16B02E5C
    7.93650558643019558500e-04,  // 0x3F4A019F98CF38B6
    -5.95187557450339963135e-04, // 0xBF4380CB8C0FE741
    8.36339918996282139126e-04,  // 0x3F4B67BA4CDAD5D1
    -1.63092934096575273989e-03, // 0xBF5AB89D0B9E43E4
];

const Y_MIN: f64 = 1.461632144968362245;

// Mathematica: N[2^52, 50]
const TWO_52: f64 = 4.5035996273704960000000000000000000000000000000000e15; // 1 << 52; 0x4330000000000000

// Mathematica: N[2^53, 50]
const TWO_53: f64 = 9.0071992547409920000000000000000000000000000000000e15; // 1 << 53; 0x4340000000000000

// Mathematica: N[2^58, 50]
const TWO_58: f64 = 2.8823037615171174400000000000000000000000000000000e17; // 1 << 58; 0x4390000000000000

// Mathematica: N[2^-70, 64]
const TINY: f64 = 8.470329472543003390683225006796419620513916015625000000000000000e-22; // 1.0 / (1 << 70); 0x3b90000000000000

const TC: f64 = 1.46163214496836224576e+00; // 0x3FF762D86356BE3F

const TF: f64 = -1.21486290535849611461e-01; // 0xBFBF19B9BCC38A42

// TT = -(tail of TF)
const TT: f64 = -3.63867699703950536541e-18; // 0xBC50C7CAA48A971F

/// Evaluates the natural logarithm and sign of Γ(x)
///
/// Returns `(ln_gamma_x, sign)` where the sign is -1 or 1.
///
/// See: <https://mathworld.wolfram.com/LogGammaFunction.html>
///
/// See also: <https://en.wikipedia.org/wiki/Gamma_function>
///
/// # Special cases
///
/// * `ln(Γ(+Inf))     = +Inf`
/// * `ln(Γ(0))        = +Inf`
/// * `ln(Γ(-integer)) = +Inf`
/// * `ln(Γ(-Inf))     = -Inf`
/// * `ln(Γ(NaN))      = NaN`
pub fn ln_gamma(x: f64) -> (f64, i32) {
    // special cases
    if f64::is_nan(x) {
        return (x, 1);
    } else if f64::is_infinite(x) {
        return (x, 1);
    } else if x == 0.0 {
        return (f64::INFINITY, 1);
    }

    let mut negative = false;
    let mut xx = x;
    if xx < 0.0 {
        xx = -xx;
        negative = true;
    }

    let mut sign = 1;
    if xx < TINY {
        // if |x| < 2**-70, return -log(|x|)
        if negative {
            sign = -1;
        }
        return (-f64::ln(xx), sign);
    }

    let mut n_adj: f64 = 0.0;
    if negative {
        if xx >= TWO_52 {
            // |x| >= 2**52, must be -integer
            return (f64::INFINITY, sign);
        }
        let t = sin_pi(xx);
        if t == 0.0 {
            return (f64::INFINITY, sign); // -integer case
        }
        n_adj = f64::ln(PI / f64::abs(t * xx));
        if t < 0.0 {
            sign = -1;
        }
    }

    if xx == 1.0 || xx == 2.0 {
        // purge off 1 and 2
        return (0.0, sign);
    }

    let mut lgamma: f64;
    if xx < 2.0 {
        // use lgamma(x) = lgamma(x+1) - log(x)
        let (y, i) = if xx <= 0.9 {
            lgamma = -f64::ln(xx);
            if xx >= (Y_MIN - 1.0 + 0.27) {
                // 0.7316 <= x <=  0.9
                (1.0 - xx, 0)
            } else if xx >= (Y_MIN - 1.0 - 0.27) {
                // 0.2316 <= x < 0.7316
                (xx - (TC - 1.0), 1)
            } else {
                // 0 < x < 0.2316
                (xx, 2)
            }
        } else {
            lgamma = 0.0;
            if xx >= (Y_MIN + 0.27) {
                // 1.7316 <= x < 2
                (2.0 - xx, 0)
            } else if xx >= (Y_MIN - 0.27) {
                // 1.2316 <= x < 1.7316
                (xx - TC, 1)
            } else {
                // 0.9 < x < 1.2316
                (xx - 1.0, 2)
            }
        };
        if i == 0 {
            let z = y * y;
            let p1 = LA[0] + z * (LA[2] + z * (LA[4] + z * (LA[6] + z * (LA[8] + z * LA[10]))));
            let p2 = z * (LA[1] + z * (LA[3] + z * (LA[5] + z * (LA[7] + z * (LA[9] + z * LA[11])))));
            let p = y * p1 + p2;
            lgamma += p - 0.5 * y;
        } else if i == 1 {
            let z = y * y;
            let w = z * y;
            let p1 = LT[0] + w * (LT[3] + w * (LT[6] + w * (LT[9] + w * LT[12]))); // parallel comp
            let p2 = LT[1] + w * (LT[4] + w * (LT[7] + w * (LT[10] + w * LT[13])));
            let p3 = LT[2] + w * (LT[5] + w * (LT[8] + w * (LT[11] + w * LT[14])));
            let p = z * p1 - (TT - w * (p2 + y * p3));
            lgamma += TF + p;
        } else {
            let p1 = y * (LU[0] + y * (LU[1] + y * (LU[2] + y * (LU[3] + y * (LU[4] + y * LU[5])))));
            let p2 = 1.0 + y * (LV[1] + y * (LV[2] + y * (LV[3] + y * (LV[4] + y * LV[5]))));
            lgamma += -0.5 * y + p1 / p2;
        }
    } else if xx < 8.0 {
        // 2 <= x < 8
        let i = xx as i32;
        let y = xx - (i as f64);
        let p = y * (LS[0] + y * (LS[1] + y * (LS[2] + y * (LS[3] + y * (LS[4] + y * (LS[5] + y * LS[6]))))));
        let q = 1.0 + y * (LR[1] + y * (LR[2] + y * (LR[3] + y * (LR[4] + y * (LR[5] + y * LR[6])))));
        lgamma = 0.5 * y + p / q;
        let mut z = 1.0; // ln_gamma(1+s) = ln(s) + ln_gamma(s)
        if i >= 7 {
            z *= y + 6.0;
        }
        if i >= 6 {
            z *= y + 5.0;
        }
        if i >= 5 {
            z *= y + 4.0;
        }
        if i >= 4 {
            z *= y + 3.0;
        }
        if i >= 3 {
            z *= y + 2.0;
            lgamma += f64::ln(z);
        }
    } else if xx < TWO_58 {
        // 8 <= x < 2**58
        let t = f64::ln(xx);
        let z = 1.0 / xx;
        let y = z * z;
        let w = LW[0] + z * (LW[1] + y * (LW[2] + y * (LW[3] + y * (LW[4] + y * (LW[5] + y * LW[6])))));
        lgamma = (xx - 0.5) * (t - 1.0) + w;
    } else {
        // 2**58 <= x <= Inf
        lgamma = xx * (f64::ln(xx) - 1.0);
    }
    if negative {
        lgamma = n_adj - lgamma;
    }
    (lgamma, sign)
}

// helper function for negative x
fn sin_pi(x: f64) -> f64 {
    if x < 0.25 {
        return -f64::sin(PI * x);
    }

    // argument reduction
    let mut xx = x;
    let mut z = f64::floor(xx);
    let mut n: i32;
    if z != xx {
        // inexact
        xx = modulo(xx, 2.0);
        n = (xx * 4.0) as i32;
    } else {
        if xx >= TWO_53 {
            // x must be even
            xx = 0.0;
            n = 0;
        } else {
            if xx < TWO_52 {
                z = xx + TWO_52; // exact
            }
            n = (1 & f64::to_bits(z)) as i32;
            xx = n as f64;
            n <<= 2;
        }
    }
    if n == 0 {
        xx = f64::sin(PI * xx);
    } else if n == 1 || n == 2 {
        xx = f64::cos(PI * (0.5 - xx));
    } else if n == 3 || n == 4 {
        xx = f64::sin(PI * (1.0 - xx));
    } else if n == 5 || n == 6 {
        xx = -f64::cos(PI * (xx - 1.5));
    } else {
        xx = f64::sin(PI * (xx - 2.0));
    }
    -xx
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{ln_gamma, TINY, TWO_52};
    use crate::math::ONE_BY_3;
    use crate::{approx_eq, assert_alike};

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
    fn ln_gamma_works_special_case() {
        let x = TINY / 2.0;
        let (y, s) = ln_gamma(x);
        // Mathematica: N[LogGamma[2^-71], 50]
        approx_eq(y, 49.213449819756116968623236163187614885368991246517, 1e-50);
        assert_eq!(s, 1);

        // Mathematica: N[LogGamma[-2^-71], 50] (using the real part of the result)
        let (y, s) = ln_gamma(-x);
        approx_eq(y, 49.213449819756116968623725083873457781352028127685, 1e-50);
        assert_eq!(s, -1);

        let x = -TWO_52;
        let (y, s) = ln_gamma(x);
        assert_eq!(y, f64::INFINITY);
        assert_eq!(s, 1);
    }

    // The code below is based on all_test.go file from Go (1.22.1)

    struct Fi {
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

    #[rustfmt::skip]
    const SOLUTION: [Fi; 10] = [
        Fi { f: 3.146492141244545774319734e+00,  i: 1 },
        Fi { f: 8.003414490659126375852113e+00,  i: 1 },
        Fi { f: 1.517575735509779707488106e+00,  i: -1 },
        Fi { f: -2.588480028182145853558748e-01, i: 1 },
        Fi { f: 1.1989897050205555002007985e+01, i: 1 },
        Fi { f: 6.262899811091257519386906e-01,  i: 1 },
        Fi { f: 3.5287924899091566764846037e+00, i: 1 },
        Fi { f: 4.5725644770161182299423372e-01, i: 1 },
        Fi { f: -6.363667087767961257654854e-02, i: 1 },
        Fi { f: -1.077385130910300066425564e+01, i: -1 },
    ];

    const SC_VALUES: [f64; 7] = [f64::NEG_INFINITY, -3.0, 0.0, 1.0, 2.0, f64::INFINITY, f64::NAN];

    #[rustfmt::skip]
    const SC_SOLUTION: [Fi; 7] = [
        Fi { f: f64::NEG_INFINITY, i: 1 },
        Fi { f: f64::INFINITY, i: 1 },
        Fi { f: f64::INFINITY, i: 1 },
        Fi { f: 0.0, i: 1 },
        Fi { f: 0.0, i: 1 },
        Fi { f: f64::INFINITY, i: 1 },
        Fi { f: f64::NAN, i: 1 },
    ];

    #[test]
    fn test_ln_gamma() {
        for (i, v) in VALUES.iter().enumerate() {
            let (f, s) = ln_gamma(*v);
            approx_eq(SOLUTION[i].f, f, 1e-14);
            assert_eq!(SOLUTION[i].i, s);
        }
        for (i, v) in SC_VALUES.iter().enumerate() {
            let (f, s) = ln_gamma(*v);
            assert_alike(SC_SOLUTION[i].f, f);
            assert_eq!(SC_SOLUTION[i].i, s);
        }
    }
}
