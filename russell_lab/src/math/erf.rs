//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//// This implementation is based on erf.go file from Go (1.22.1),    ////
//// which, in turn, is based on the FreeBSD code as explained below. ////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Copyright 2010 The Go Authors. All rights reserved.                  //
// Use of this source code is governed by a BSD-style                   //
// license that can be found in the LICENSE file.                       //
//                                                                      //
// Floating-point error function and complementary error function.      //
//                                                                      //
// The original C code and the long comment below are                   //
// from FreeBSD's /usr/src/lib/msun/src/s_erf.c and                     //
// came with this notice. The go code is a simplified                   //
// version of the original C.                                           //
//                                                                      //
// ====================================================                 //
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.    //
//                                                                      //
// Developed at SunPro, a Sun Microsystems, Inc. business.              //
// Permission to use, copy, modify, and distribute this                 //
// software is freely granted, provided that this notice                //
// is preserved.                                                        //
// ====================================================                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

// double erf(double x)
// double erfc(double x)
//                           x
//                    2      |\
//     erf(x)  =  ---------  | exp(-t*t)dt
//                 sqrt(pi) \|
//                           0
//
//     erfc(x) =  1-erf(x)
//  Note that
//              erf(-x) = -erf(x)
//              erfc(-x) = 2 - erfc(x)
//
// Method:
//      1. For |x| in [0, 0.84375]
//          erf(x)  = x + x*R(x**2)
//          erfc(x) = 1 - erf(x)           if x in [-.84375,0.25]
//                  = 0.5 + ((0.5-x)-x*R)  if x in [0.25,0.84375]
//         where R = P/Q where P is an odd poly of degree 8 and
//         Q is an odd poly of degree 10.
//                                               -57.90
//                      | R - (erf(x)-x)/x | <= 2
//
//
//         Remark. The formula is derived by noting
//          erf(x) = (2/sqrt(pi))*(x - x**3/3 + x**5/10 - x**7/42 + ....)
//         and that
//          2/sqrt(pi) = 1.128379167095512573896158903121545171688
//         is close to one. The interval is chosen because the fix
//         point of erf(x) is near 0.6174 (i.e., erf(x)=x when x is
//         near 0.6174), and by some experiment, 0.84375 is chosen to
//         guarantee the error is less than one ulp for erf.
//
//      2. For |x| in [0.84375,1.25], let s = |x| - 1, and
//         c = 0.84506291151 rounded to single (24 bits)
//              erf(x)  = sign(x) * (c  + P1(s)/Q1(s))
//              erfc(x) = (1-c)  - P1(s)/Q1(s) if x > 0
//                        1+(c+P1(s)/Q1(s))    if x < 0
//              |P1/Q1 - (erf(|x|)-c)| <= 2**-59.06
//         Remark: here we use the taylor series expansion at x=1.
//              erf(1+s) = erf(1) + s*Poly(s)
//                       = 0.845.. + P1(s)/Q1(s)
//         That is, we use rational approximation to approximate
//                      erf(1+s) - (c = (single)0.84506291151)
//         Note that |P1/Q1|< 0.078 for x in [0.84375,1.25]
//         where
//              P1(s) = degree 6 poly in s
//              Q1(s) = degree 6 poly in s
//
//      3. For x in [1.25,1/0.35(~2.857143)],
//              erfc(x) = (1/x)*exp(-x*x-0.5625+R1/S1)
//              erf(x)  = 1 - erfc(x)
//         where
//              R1(z) = degree 7 poly in z, (z=1/x**2)
//              S1(z) = degree 8 poly in z
//
//      4. For x in [1/0.35,28]
//              erfc(x) = (1/x)*exp(-x*x-0.5625+R2/S2) if x > 0
//                      = 2.0 - (1/x)*exp(-x*x-0.5625+R2/S2) if -6<x<0
//                      = 2.0 - tiny            (if x <= -6)
//              erf(x)  = sign(x)*(1.0 - erfc(x)) if x < 6, else
//              erf(x)  = sign(x)*(1.0 - tiny)
//         where
//              R2(z) = degree 6 poly in z, (z=1/x**2)
//              S2(z) = degree 7 poly in z
//
//      Note1:
//         To compute exp(-x*x-0.5625+R/S), let s be a single
//         precision number and s = x; then
//              -x*x = -s*s + (s-x)*(s+x)
//              exp(-x*x-0.5626+R/S) =
//                      exp(-s*s-0.5625)*exp((s-x)*(s+x)+R/S);
//      Note2:
//         Here 4 and 5 make use of the asymptotic series
//                        exp(-x*x)
//              erfc(x) ~ ---------- * ( 1 + Poly(1/x**2) )
//                        x*sqrt(pi)
//         We use rational approximation to approximate
//              g(s)=f(1/x**2) = log(erfc(x)*x) - x*x + 0.5625
//         Here is the error bound for R1/S1 and R2/S2
//              |R1/S1 - f(x)|  < 2**(-62.57)
//              |R2/S2 - f(x)|  < 2**(-61.52)
//
//      5. For inf > x >= 28
//              erf(x)  = sign(x) *(1 - tiny)  (raise inexact)
//              erfc(x) = tiny*tiny (raise underflow) if x > 0
//                      = 2 - tiny if x<0
//
//      7. Special case:
//              erf(0)  = 0, erf(inf)  = 1, erf(-inf) = -1,
//              erfc(0) = 1, erfc(inf) = 0, erfc(-inf) = 2,
//              erfc/erf(NaN) is NaN

const ERX: f64 = 8.45062911510467529297e-01; // 0x3FEB0AC160000000

// coefficients for approximation to  erf in [0, 0.84375]
const EFX: f64 = 1.28379167095512586316e-01; // 0x3FC06EBA8214DB69
const EFX8: f64 = 1.02703333676410069053e+00; // 0x3FF06EBA8214DB69
const PP0: f64 = 1.28379167095512558561e-01; // 0x3FC06EBA8214DB68
const PP1: f64 = -3.25042107247001499370e-01; // 0xBFD4CD7D691CB913
const PP2: f64 = -2.84817495755985104766e-02; // 0xBF9D2A51DBD7194F
const PP3: f64 = -5.77027029648944159157e-03; // 0xBF77A291236668E4
const PP4: f64 = -2.37630166566501626084e-05; // 0xBEF8EAD6120016AC
const QQ1: f64 = 3.97917223959155352819e-01; // 0x3FD97779CDDADC09
const QQ2: f64 = 6.50222499887672944485e-02; // 0x3FB0A54C5536CEBA
const QQ3: f64 = 5.08130628187576562776e-03; // 0x3F74D022C4D36B0F
const QQ4: f64 = 1.32494738004321644526e-04; // 0x3F215DC9221C1A10
const QQ5: f64 = -3.96022827877536812320e-06; // 0xBED09C4342A26120

// coefficients for approximation to  erf  in [0.84375, 1.25]
const PA0: f64 = -2.36211856075265944077e-03; // 0xBF6359B8BEF77538
const PA1: f64 = 4.14856118683748331666e-01; // 0x3FDA8D00AD92B34D
const PA2: f64 = -3.72207876035701323847e-01; // 0xBFD7D240FBB8C3F1
const PA3: f64 = 3.18346619901161753674e-01; // 0x3FD45FCA805120E4
const PA4: f64 = -1.10894694282396677476e-01; // 0xBFBC63983D3E28EC
const PA5: f64 = 3.54783043256182359371e-02; // 0x3FA22A36599795EB
const PA6: f64 = -2.16637559486879084300e-03; // 0xBF61BF380A96073F
const QA1: f64 = 1.06420880400844228286e-01; // 0x3FBB3E6618EEE323
const QA2: f64 = 5.40397917702171048937e-01; // 0x3FE14AF092EB6F33
const QA3: f64 = 7.18286544141962662868e-02; // 0x3FB2635CD99FE9A7
const QA4: f64 = 1.26171219808761642112e-01; // 0x3FC02660E763351F
const QA5: f64 = 1.36370839120290507362e-02; // 0x3F8BEDC26B51DD1C
const QA6: f64 = 1.19844998467991074170e-02; // 0x3F888B545735151D

// coefficients for approximation to  erfc in [1.25, 1/0.35]
const RA0: f64 = -9.86494403484714822705e-03; // 0xBF843412600D6435
const RA1: f64 = -6.93858572707181764372e-01; // 0xBFE63416E4BA7360
const RA2: f64 = -1.05586262253232909814e+01; // 0xC0251E0441B0E726
const RA3: f64 = -6.23753324503260060396e+01; // 0xC04F300AE4CBA38D
const RA4: f64 = -1.62396669462573470355e+02; // 0xC0644CB184282266
const RA5: f64 = -1.84605092906711035994e+02; // 0xC067135CEBCCABB2
const RA6: f64 = -8.12874355063065934246e+01; // 0xC054526557E4D2F2
const RA7: f64 = -9.81432934416914548592e+00; // 0xC023A0EFC69AC25C
const SA1: f64 = 1.96512716674392571292e+01; // 0x4033A6B9BD707687
const SA2: f64 = 1.37657754143519042600e+02; // 0x4061350C526AE721
const SA3: f64 = 4.34565877475229228821e+02; // 0x407B290DD58A1A71
const SA4: f64 = 6.45387271733267880336e+02; // 0x40842B1921EC2868
const SA5: f64 = 4.29008140027567833386e+02; // 0x407AD02157700314
const SA6: f64 = 1.08635005541779435134e+02; // 0x405B28A3EE48AE2C
const SA7: f64 = 6.57024977031928170135e+00; // 0x401A47EF8E484A93
const SA8: f64 = -6.04244152148580987438e-02; // 0xBFAEEFF2EE749A62

// coefficients for approximation to  erfc in [1/.35, 28]
const RB0: f64 = -9.86494292470009928597e-03; // 0xBF84341239E86F4A
const RB1: f64 = -7.99283237680523006574e-01; // 0xBFE993BA70C285DE
const RB2: f64 = -1.77579549177547519889e+01; // 0xC031C209555F995A
const RB3: f64 = -1.60636384855821916062e+02; // 0xC064145D43C5ED98
const RB4: f64 = -6.37566443368389627722e+02; // 0xC083EC881375F228
const RB5: f64 = -1.02509513161107724954e+03; // 0xC09004616A2E5992
const RB6: f64 = -4.83519191608651397019e+02; // 0xC07E384E9BDC383F
const SB1: f64 = 3.03380607434824582924e+01; // 0x403E568B261D5190
const SB2: f64 = 3.25792512996573918826e+02; // 0x40745CAE221B9F0A
const SB3: f64 = 1.53672958608443695994e+03; // 0x409802EB189D5118
const SB4: f64 = 3.19985821950859553908e+03; // 0x40A8FFB7688C246A
const SB5: f64 = 2.55305040643316442583e+03; // 0x40A3F219CEDF3BE6
const SB6: f64 = 4.74528541206955367215e+02; // 0x407DA874E79FE763
const SB7: f64 = -2.24409524465858183362e+01; // 0xC03670E242712D62

const VERY_TINY: f64 = 2.848094538889218e-306; // 0x0080000000000000

// 2**-56
// Mathematica: N[2^-56, 50]
const TINY: f64 = 1.3877787807814456755295395851135253906250000000000e-17;

// 2**-28
// Mathematica: N[2^-28, 50]
const SMALL: f64 = 3.7252902984619140625000000000000000000000000000000e-9;

/// Evaluates the error function
///
/// ```text
///                z
///           2   ⌠
/// erf(z) = ———  │  exp(-t²) dt
///          √π   ⌡
///              0
/// ```
///
/// See: <https://en.wikipedia.org/wiki/Error_function>
///
/// # Special cases
///
/// * `erf(+Inf) = 1`
/// * `erf(-Inf) = -1`
/// * `erf(NaN) = NaN`
pub fn erf(x: f64) -> f64 {
    // special cases
    if f64::is_nan(x) {
        return f64::NAN;
    } else if x == f64::INFINITY {
        return 1.0;
    } else if x == f64::NEG_INFINITY {
        return -1.0;
    }
    let mut sign = false;
    let mut x = x;
    if x < 0.0 {
        x = -x;
        sign = true;
    }
    if x < 0.84375 {
        // |x| < 0.84375
        let mut temp: f64;
        if x < SMALL {
            // |x| < 2**-28
            if x < VERY_TINY {
                temp = 0.125 * (8.0 * x + EFX8 * x); // avoid underflow
            } else {
                temp = x + EFX * x;
            }
        } else {
            let z = x * x;
            let r = PP0 + z * (PP1 + z * (PP2 + z * (PP3 + z * PP4)));
            let s = 1.0 + z * (QQ1 + z * (QQ2 + z * (QQ3 + z * (QQ4 + z * QQ5))));
            let y = r / s;
            temp = x + x * y;
        }
        if sign {
            return -temp;
        }
        return temp;
    }
    if x < 1.25 {
        // 0.84375 <= |x| < 1.25
        let s = x - 1.0;
        let P = PA0 + s * (PA1 + s * (PA2 + s * (PA3 + s * (PA4 + s * (PA5 + s * PA6)))));
        let Q = 1.0 + s * (QA1 + s * (QA2 + s * (QA3 + s * (QA4 + s * (QA5 + s * QA6)))));
        if sign {
            return -ERX - P / Q;
        }
        return ERX + P / Q;
    }
    if x >= 6.0 {
        // inf > |x| >= 6
        if sign {
            return -1.0;
        }
        return 1.0;
    }
    let s = 1.0 / (x * x);
    let mut R: f64;
    let mut S: f64;
    if x < 1.0 / 0.35 {
        // |x| < 1 / 0.35  ~ 2.857143
        R = RA0 + s * (RA1 + s * (RA2 + s * (RA3 + s * (RA4 + s * (RA5 + s * (RA6 + s * RA7))))));
        S = 1.0 + s * (SA1 + s * (SA2 + s * (SA3 + s * (SA4 + s * (SA5 + s * (SA6 + s * (SA7 + s * SA8)))))));
    } else {
        // |x| >= 1 / 0.35  ~ 2.857143
        R = RB0 + s * (RB1 + s * (RB2 + s * (RB3 + s * (RB4 + s * (RB5 + s * RB6)))));
        S = 1.0 + s * (SB1 + s * (SB2 + s * (SB3 + s * (SB4 + s * (SB5 + s * (SB6 + s * SB7))))));
    }
    let z = f64::from_bits(f64::to_bits(x) & 0xffffffff00000000); // pseudo-single (20-bit) precision x
    let r = f64::exp(-z * z - 0.5625) * f64::exp((z - x) * (z + x) + R / S);
    if sign {
        return r / x - 1.0;
    }
    return 1.0 - r / x;
}

/// Evaluates the complementary error function
///
/// ```text
/// erfc(z) = 1 - erf(z)
/// ```
///
/// See: <https://en.wikipedia.org/wiki/Error_function>
///
/// # Special cases
///
/// * `erfc(+Inf) = 0`
/// * `erfc(-Inf) = 2`
/// * `erfc(NaN) = NaN`
pub fn erfc(x: f64) -> f64 {
    // special cases
    if f64::is_nan(x) {
        return f64::NAN;
    } else if x == f64::INFINITY {
        return 0.0;
    } else if x == f64::NEG_INFINITY {
        return 2.0;
    }
    let mut sign = false;
    let mut x = x;
    if x < 0.0 {
        x = -x;
        sign = true;
    }
    if x < 0.84375 {
        // |x| < 0.84375
        let mut temp: f64;
        if x < TINY {
            // |x| < 2**-56
            temp = x;
        } else {
            let z = x * x;
            let r = PP0 + z * (PP1 + z * (PP2 + z * (PP3 + z * PP4)));
            let s = 1.0 + z * (QQ1 + z * (QQ2 + z * (QQ3 + z * (QQ4 + z * QQ5))));
            let y = r / s;
            if x < 0.25 {
                // |x| < 1/4
                temp = x + x * y;
            } else {
                temp = 0.5 + (x * y + (x - 0.5));
            }
        }
        if sign {
            return 1.0 + temp;
        }
        return 1.0 - temp;
    }
    if x < 1.25 {
        // 0.84375 <= |x| < 1.25
        let s = x - 1.0;
        let P = PA0 + s * (PA1 + s * (PA2 + s * (PA3 + s * (PA4 + s * (PA5 + s * PA6)))));
        let Q = 1.0 + s * (QA1 + s * (QA2 + s * (QA3 + s * (QA4 + s * (QA5 + s * QA6)))));
        if sign {
            return 1.0 + ERX + P / Q;
        }
        return 1.0 - ERX - P / Q;
    }
    if x < 28.0 {
        // |x| < 28
        let s = 1.0 / (x * x);
        let mut R: f64;
        let mut S: f64;
        if x < 1.0 / 0.35 {
            // |x| < 1 / 0.35 ~ 2.857143
            R = RA0 + s * (RA1 + s * (RA2 + s * (RA3 + s * (RA4 + s * (RA5 + s * (RA6 + s * RA7))))));
            S = 1.0 + s * (SA1 + s * (SA2 + s * (SA3 + s * (SA4 + s * (SA5 + s * (SA6 + s * (SA7 + s * SA8)))))));
        } else {
            // |x| >= 1 / 0.35 ~ 2.857143
            if sign && x > 6.0 {
                return 2.0; // x < -6
            }
            R = RB0 + s * (RB1 + s * (RB2 + s * (RB3 + s * (RB4 + s * (RB5 + s * RB6)))));
            S = 1.0 + s * (SB1 + s * (SB2 + s * (SB3 + s * (SB4 + s * (SB5 + s * (SB6 + s * SB7))))));
        }
        let z = f64::from_bits(f64::to_bits(x) & 0xffffffff00000000); // pseudo-single (20-bit) precision x
        let r = f64::exp(-z * z - 0.5625) * f64::exp((z - x) * (z + x) + R / S);
        if sign {
            return 2.0 - r / x;
        }
        return r / x;
    }
    if sign {
        return 2.0;
    }
    return 0.0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{erf, erfc};
    use crate::approx_eq;

    #[test]
    fn erf_works_1() {
        assert_eq!(erf(0.0), 0.0);
        approx_eq(erf(0.3), 0.328626759, 1e-9);
        approx_eq(erf(1.0), 0.842700793, 1e-9);
        approx_eq(erf(1.8), 0.989090502, 1e-9);
        approx_eq(erf(3.5), 0.999999257, 1e-9);
        assert!(erf(f64::NAN).is_nan());
        approx_eq(
            erf(-1.0),
            -0.84270079294971486934122063508260925929606699796630291,
            1e-11,
        );
        assert_eq!(erf(0.0), 0.0);
        assert_eq!(
            erf(1e-15),
            0.0000000000000011283791670955126615773132947717431253912942469337536
        );
        assert_eq!(erf(0.1), 0.1124629160182848984047122510143040617233925185058162);
        approx_eq(erf(0.2), 0.22270258921047846617645303120925671669511570710081967, 1e-16);
        assert_eq!(erf(0.3), 0.32862675945912741618961798531820303325847175931290341);
        assert_eq!(erf(0.4), 0.42839235504666847645410962730772853743532927705981257);
        approx_eq(erf(0.5), 0.5204998778130465376827466538919645287364515757579637, 1e-9);
        approx_eq(erf(1.0), 0.84270079294971486934122063508260925929606699796630291, 1e-11);
        approx_eq(erf(1.5), 0.96610514647531072706697626164594785868141047925763678, 1e-11);
        approx_eq(erf(2.0), 0.99532226501895273416206925636725292861089179704006008, 1e-11);
        approx_eq(erf(2.5), 0.99959304798255504106043578426002508727965132259628658, 1e-13);
        approx_eq(erf(3.0), 0.99997790950300141455862722387041767962015229291260075, 1e-11);
        assert_eq!(erf(4.0), 0.99999998458274209971998114784032651311595142785474641);
        assert_eq!(erf(5.0), 0.99999999999846254020557196514981165651461662110988195);
        assert_eq!(erf(6.0), 0.99999999999999997848026328750108688340664960081261537);
        assert_eq!(erf(f64::INFINITY), 1.0);
        assert_eq!(erf(f64::NEG_INFINITY), -1.0);
    }

    #[test]
    fn erf_works_2() {
        let mathematica = [
            (-5.0, 1e-50, -0.9999999999984626),
            (-4.5, 1e-50, -0.9999999998033839),
            (-4.0, 1e-50, -0.9999999845827421),
            (-3.5, 1e-50, -0.9999992569016276),
            (-3.0, 1e-50, -0.9999779095030014),
            (-2.5, 1e-50, -0.999593047982555),
            (-2.0, 1e-50, -0.9953222650189527),
            (-1.5, 1e-50, -0.9661051464753108),
            (-1.0, 1e-50, -0.8427007929497149),
            (-0.5, 1e-50, -0.5204998778130465),
            (0.0, 1e-50, 0.),
            (0.5, 1e-50, 0.5204998778130465),
            (1.0, 1e-50, 0.8427007929497149),
            (1.5, 1e-50, 0.9661051464753108),
            (2.0, 1e-50, 0.9953222650189527),
            (2.5, 1e-50, 0.999593047982555),
            (3.0, 1e-50, 0.9999779095030014),
            (3.5, 1e-50, 0.9999992569016276),
            (4.0, 1e-50, 0.9999999845827421),
            (4.5, 1e-50, 0.9999999998033839),
            (5.0, 1e-50, 0.9999999999984626),
        ];
        for (x, tol, reference) in mathematica {
            approx_eq(erf(x), reference, tol);
        }
    }

    #[test]
    #[rustfmt::skip]
    fn erfc_works_1() {
        assert!(erfc(f64::NAN).is_nan());
        approx_eq(erfc(-1.0), 1.8427007929497148693412206350826092592960669979663028, 1e-11);
        assert_eq!(erfc(0.0), 1.0);
        approx_eq(erfc(0.1), 0.88753708398171510159528774898569593827660748149418343, 1e-15);
        assert_eq!(erfc(0.2), 0.77729741078952153382354696879074328330488429289918085);
        assert_eq!(erfc(0.3), 0.67137324054087258381038201468179696674152824068709621);
        approx_eq(erfc(0.4), 0.57160764495333152354589037269227146256467072294018715, 1e-15);
        approx_eq(erfc(0.5), 0.47950012218695346231725334610803547126354842424203654, 1e-9);
        approx_eq(erfc(1.0), 0.15729920705028513065877936491739074070393300203369719, 1e-11);
        approx_eq(erfc(1.5), 0.033894853524689272933023738354052141318589520742363247, 1e-11);
        approx_eq(erfc(2.0), 0.0046777349810472658379307436327470713891082029599399245, 1e-11);
        approx_eq(erfc(2.5), 0.00040695201744495893956421573997491272034867740371342016, 1e-13);
        approx_eq(erfc(3.0), 0.00002209049699858544137277612958232037984770708739924966, 1e-11);
        approx_eq(erfc(4.0), 0.000000015417257900280018852159673486884048572145253589191167, 1e-18);
        approx_eq(erfc(5.0), 0.0000000000015374597944280348501883434853833788901180503147233804, 1e-22);
        approx_eq(erfc(6.0), 2.1519736712498913116593350399187384630477514061688559e-17, 1e-26);
        approx_eq(erfc(10.0), 2.0884875837625447570007862949577886115608181193211634e-45, 1e-55);
        approx_eq(erfc(15.0), 7.2129941724512066665650665586929271099340909298253858e-100, 1e-109);
        approx_eq(erfc(20.0), 5.3958656116079009289349991679053456040882726709236071e-176, 1e-186);
        assert_eq!(erfc(30.0), 2.5646562037561116000333972775014471465488897227786155e-393);
        assert_eq!(erfc(50.0), 2.0709207788416560484484478751657887929322509209953988e-1088);
        assert_eq!(erfc(80.0), 2.3100265595063985852034904366341042118385080919280966e-2782);
        assert_eq!(erfc(f64::INFINITY), 0.0);
        assert_eq!(erfc(f64::NEG_INFINITY), 2.0);
    }

    #[test]
    fn erfc_works_2() {
        let mathematica = [
            (-5.0, 1e-50, 1.9999999999984626),
            (-4.5, 1e-50, 1.999999999803384),
            (-4.0, 1e-50, 1.999999984582742),
            (-3.5, 1e-50, 1.9999992569016276),
            (-3.0, 1e-50, 1.9999779095030015),
            (-2.5, 1e-50, 1.999593047982555),
            (-2.0, 1e-50, 1.9953222650189528),
            (-1.5, 1e-50, 1.9661051464753108),
            (-1.0, 1e-15, 1.842700792949715),
            (-0.5, 1e-50, 1.5204998778130465),
            (0.0, 1e-50, 1.0),
            (0.5, 1e-50, 0.4795001221869535),
            (1.0, 1e-50, 0.15729920705028513),
            (1.5, 1e-50, 0.033894853524689274),
            (2.0, 1e-50, 0.004677734981047265),
            (2.5, 1e-50, 0.0004069520174449589),
            (3.0, 1e-50, 0.000022090496998585438),
            (3.5, 1e-50, 7.430983723414128e-7),
            (4.0, 1e-50, 1.541725790028002e-8),
            (4.5, 1e-50, 1.9661604415428873e-10),
            (5.0, 1e-50, 1.5374597944280351e-12),
        ];
        for (x, tol, reference) in mathematica {
            // println!("x = {:?}", x);
            approx_eq(erfc(x), reference, tol);
        }
    }
}
