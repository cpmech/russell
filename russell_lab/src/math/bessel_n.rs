use super::{bessel_j0, bessel_j1, bessel_y0, bessel_y1, SQRT_PI};

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//// This implementation is based on j1.go file from Go (1.22.1),     ////
//// which, in turn, is based on the FreeBSD code as explained below. ////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Copyright 2010 The Go Authors. All rights reserved.                  //
// Use of this source code is governed by a BSD-style                   //
// license that can be found in the LICENSE file.                       //
//                                                                      //
// Bessel function of the first and second kinds of order n.            //
//                                                                      //
// The original C code and the long comment below are                   //
// from FreeBSD's /usr/src/lib/msun/src/e_jn.c and                      //
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
//
// Special cases:
//      y0(0)=y1(0)=yn(n,0) = -inf with division by zero signal;
//      y0(-ve)=y1(-ve)=yn(n,-ve) are NaN with invalid signal.
// Note 2. About jn(n,x), yn(n,x)
//      For n=0, j0(x) is called,
//      for n=1, j1(x) is called,
//      for n<x, forward recursion is used starting
//      from values of j0(x) and j1(x).
//      for n>x, a continued fraction approximation to
//      j(n,x)/j(n-1,x) is evaluated and then backward
//      recursion is used starting from a supposed value
//      for j(n,x). The resulting value of j(0,x) is
//      compared with the actual value to correct the
//      supposed value of j(n,x).
//
//      yn(n,x) is similar in all respects, except
//      that forward recursion is used for all
//      values of n>1.

// 2**-29 0x3e10000000000000 Mathematica: N[2^-29, 50]
const TWO_M29: f64 = 1.8626451492309570312500000000000000000000000000000e-9;

// 2**302 0x52D0000000000000 Mathematica: N[2^302, 100]
const TWO_302: f64 =
    8.148143905337944345073782753637512644205873574663745002544561797417525199053346824733589504000000000e90;

/// Evaluates the order-n Bessel function of the first kind
///
/// The special cases are:
///
///	* `Jn(n, NaN)  = NaN`
///	* `Jn(n, ±Inf) = 0.0`
pub fn bessel_jn(n: i32, x: f64) -> f64 {
    if f64::is_nan(x) {
        return f64::NAN;
    } else if f64::is_infinite(x) {
        return 0.0;
    }

    // J(-n, x) = (-1)**n * J(n, x), J(n, -x) = (-1)**n * J(n, x)
    // Thus, J(-n, x) = J(n, -x)

    if n == 0 {
        return bessel_j0(x);
    }
    if x == 0.0 {
        return 0.0;
    }

    let (nn, mut xx) = if n < 0 { (-n, -x) } else { (n, x) };

    if nn == 1 {
        return bessel_j1(xx);
    }

    let mut negative = false;
    if xx < 0.0 {
        xx = -xx;
        if nn & 1 == 1 {
            negative = true; // odd n and negative x
        }
    }

    let mut b: f64;
    if (nn as f64) <= xx {
        if xx >= TWO_302 {
            // Safe to use J(n+1,x)=2n/x *J(n,x)-J(n-1,x)
            //
            // (x >> n**2)
            // Jn(x) = cos(x-(2n+1)*pi/4)*sqrt(2/x*pi)
            // Yn(x) = sin(x-(2n+1)*pi/4)*sqrt(2/x*pi)
            // Let s=sin(x), c=cos(x),
            // xn=x-(2n+1)*pi/4, sqt2 = sqrt(2),then
            //
            // n  sin(xn)*sqt2  cos(xn)*sqt2
            // ------------------------------
            // 0   s-c           c+s
            // 1  -s-c          -c+s
            // 2  -s+c          -c-s
            // 3   s+c           c-s

            let (s, c) = f64::sin_cos(xx);

            let temp = match nn & 3 {
                0 => c + s,
                1 => -c + s,
                2 => -c - s,
                _ => c - s, // 3
            };
            b = (1.0 / SQRT_PI) * temp / f64::sqrt(xx);
        } else {
            let mut a = bessel_j0(xx);
            b = bessel_j1(xx);
            for i in 1..nn {
                let b_copy = b;
                b = b * (((i + i) as f64) / xx) - a; // avoid underflow
                a = b_copy;
            }
        }
    } else {
        if xx < TWO_M29 {
            // x is tiny, return the first Taylor expansion of J(n,x)
            // J(n,x) = 1/n!*(x/2)**n  - ...
            if nn > 33 {
                // underflow
                b = 0.0;
            } else {
                let temp = xx * 0.5;
                b = temp;
                let mut a = 1.0;
                for i in 2..(nn + 1) {
                    a *= i as f64; // a = n!
                    b *= temp; // b = (x/2)**n
                }
                b /= a;
            }
        } else {
            // use backward recurrence
            //                     x      x**2      x**2
            // J(n,x)/J(n-1,x) =  ----   ------   ------   .....
            //                     2n  - 2(n+1) - 2(n+2)
            //
            //                     1      1        1
            // (for large x)   =  ----  ------   ------   .....
            //                     2n   2(n+1)   2(n+2)
            //                     -- - ------ - ------ -
            //                      x     x         x
            //
            // Let w = 2n/x and h=2/x, then the above quotient
            // is equal to the continued fraction:
            //               1
            //   = -----------------------
            //                  1
            //      w - -----------------
            //                     1
            //           w+h - ---------
            //                  w+2h - ...
            //
            // To determine how many terms needed, let
            // Q(0) = w, Q(1) = w(w+h) - 1,
            // Q(k) = (w+k*h)*Q(k-1) - Q(k-2),
            // When Q(k) > 1e4	good for single
            // When Q(k) > 1e9	good for double
            // When Q(k) > 1e17	good for quadruple

            // determine k
            let w = ((nn + nn) as f64) / xx;
            let h = 2.0 / xx;
            let mut q0 = w;
            let mut z = w + h;
            let mut q1 = w * z - 1.0;
            let mut k = 1;
            while q1 < 1e9 {
                k += 1;
                z += h;
                let q1_copy = q1;
                q1 = z * q1 - q0;
                q0 = q1_copy;
            }
            let m = nn + nn;
            let mut t = 0.0;
            let mut i = 2 * (nn + k);
            while i >= m {
                t = 1.0 / ((i as f64) / xx - t);
                i -= 2;
            }
            let mut a = t;
            b = 1.0;

            // estimate log((2/x)**n*n!) = n*log(2/x)+n*ln(n)
            // Hence, if n*(log(2n/x)) > ...
            // single 8.8722839355e+01
            // double 7.09782712893383973096e+02
            // long double 1.1356523406294143949491931077970765006170e+04
            // then recurrent value may overflow and the result is likely underflow to zero

            let mut tmp = nn as f64;
            let v = 2.0 / xx;
            tmp = tmp * f64::ln(f64::abs(v * tmp));
            let mut i = nn - 1;
            if tmp < 7.09782712893383973096e+02 {
                while i > 0 {
                    let di = (i + i) as f64;
                    let b_copy = b;
                    b = b * di / xx - a;
                    a = b_copy;
                    i -= 1;
                }
            } else {
                while i > 0 {
                    let di = (i + i) as f64;
                    let b_copy = b;
                    b = b * di / xx - a;
                    a = b_copy;
                    // scale b to avoid spurious overflow
                    if b > 1e100 {
                        a /= b;
                        t /= b;
                        b = 1.0;
                    }
                    i -= 1;
                }
            }
            b = t * bessel_j0(xx) / b;
        }
    }

    if negative {
        return -b;
    } else {
        return b;
    }
}

/// Evaluates the order-n Bessel function of the second kind
///
/// The special cases are:
///
/// * `Yn(n,  +Inf) = 0`
/// * `Yn(n ≥ 0, 0) = -Inf`
/// * `Yn(n < 0, 0) = +Inf if n is odd, -Inf if n is even`
/// * `Yn(n, x < 0) = NaN`
/// * `Yn(n,   NaN) = NaN`
pub fn bessel_yn(n: i32, x: f64) -> f64 {
    if x < 0.0 || f64::is_nan(x) {
        return f64::NAN;
    } else if f64::is_infinite(x) {
        return 0.0;
    }

    if n == 0 {
        return bessel_y0(x);
    }
    if x == 0.0 {
        if n < 0 && n & 1 == 1 {
            return f64::INFINITY;
        }
        return f64::NEG_INFINITY;
    }

    let mut nn = n;
    let mut negative = false;
    if n < 0 {
        nn = -n;
        if nn & 1 == 1 {
            negative = true; // sign true if n < 0 && |n| odd
        }
    }

    if nn == 1 {
        if negative {
            return -bessel_y1(x);
        } else {
            return bessel_y1(x);
        }
    }

    let mut b: f64;
    if x >= TWO_302 {
        // (x >> n**2)
        // Jn(x) = cos(x-(2n+1)*pi/4)*sqrt(2/x*pi)
        // Yn(x) = sin(x-(2n+1)*pi/4)*sqrt(2/x*pi)
        // Let s=sin(x), c=cos(x),
        // xn=x-(2n+1)*pi/4, sqt2 = sqrt(2),then
        //
        // n  sin(xn)*sqt2  cos(xn)*sqt2
        // -----------------------------
        // 0   s-c           c+s
        // 1  -s-c          -c+s
        // 2  -s+c          -c-s
        // 3   s+c           c-s

        let (s, c) = f64::sin_cos(x);

        let temp = match nn & 3 {
            0 => s - c,
            1 => -s - c,
            2 => -s + c,
            _ => s + c, // 3
        };
        b = (1.0 / SQRT_PI) * temp / f64::sqrt(x);
    } else {
        let mut a = bessel_y0(x);
        b = bessel_y1(x);
        for i in 1..nn {
            if f64::is_infinite(b) {
                break;
            }
            let b_copy = b;
            b = (((i + i) as f64) / x) * b - a;
            a = b_copy;
        }
    }

    if negative {
        return -b;
    } else {
        return b;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{bessel_jn, bessel_yn};
    use crate::{approx_eq, math::bessel_n::TWO_302};

    #[test]
    fn bessel_jn_handles_special_cases() {
        assert!(bessel_jn(0, f64::NAN).is_nan());
        assert_eq!(bessel_jn(0, f64::NEG_INFINITY), 0.0);
        assert_eq!(bessel_jn(0, f64::INFINITY), 0.0);
        assert_eq!(bessel_jn(0, 0.0), 1.0);
    }

    #[test]
    fn bessel_yn_handles_special_cases() {
        assert!(bessel_yn(0, f64::NEG_INFINITY).is_nan());
        assert!(bessel_yn(0, -0.01).is_nan());
        assert!(bessel_yn(0, f64::NAN).is_nan());
        assert_eq!(bessel_yn(0, f64::INFINITY), 0.0);
        assert_eq!(bessel_yn(0, 0.0), f64::NEG_INFINITY);
    }

    #[test]
    fn bessel_jn_and_yn_with_n0_n1_work() {
        // Mathematica: N[BesselJ[0, 1], 100]
        assert_eq!(
            bessel_jn(0, 1.0),
            0.7651976865579665514497175261026632209092742897553252418615475491192789122152724401671806000989156340
        );

        // Mathematica: N[BesselY[0, 2], 100]
        assert_eq!(
            bessel_yn(0, 2.0),
            0.5103756726497451195966065927271578732681392270858461355718392719327313950418217977114130750041427489
        );

        // Mathematica: N[BesselJ[1, -5], 100]
        assert_eq!(
            bessel_jn(1, -5.0),
            0.3275791375914652220377343219101691327608499046240540186864806450648753089914574086157808718640701942
        );

        // Mathematica: N[BesselY[0, 1], 100]
        approx_eq(
            bessel_yn(1, 1.0),
            -0.7812128213002887165471500000479648205499063907164446078438332461277843915385602167276292380048056346,
            1e-17,
        );
    }

    #[test]
    fn bessel_jn_with_negative_n_works() {
        assert_eq!(bessel_jn(-123, 0.0), 0.0);

        // Mathematica:
        // Table[{-n, -2 + 5 k, N[BesselJ[-n, -2 + 5 k], 50]}, {n, 7, 2, -1}, {k, 0, 4}]
        #[rustfmt::skip]
        let mathematica = [
            (-7, -2.0, 1e-50,  0.00017494407486827416850658563045969580180848490306832),
            (-7,  3.0, 1e-18, -0.0025472944518046937591483625954343408982676393225066),
            (-7,  8.0, 1e-15, -0.32058907797982630385588191830690893183963113907509),
            (-7, 13.0, 1e-16,  0.24057094958616050699434490979185588377377524166962),
            (-7, 18.0, 1e-16, -0.051399275982175232583387686447788728801996398416596),
            (-6, -2.0, 1e-18,  0.0012024289717899932754583496358906732318577721676433),
            (-6,  3.0, 1e-17,  0.011393932332213069421014905764605140571689330060906),
            (-6,  8.0, 1e-50,  0.33757590011359307746413007118545048428713953845814),
            (-6, 13.0, 1e-16, -0.11803067213023636247514119833000825982731433577354),
            (-6, 18.0, 1e-16, -0.15595623419531116610693081507932322547298903175230),
            (-5, -2.0, 1e-50,  0.0070396297558716854842435121848843435893381481027912),
            (-5,  3.0, 1e-16, -0.043028434877047583924911260462986221388489680921116),
            (-5,  8.0, 1e-50, -0.18577477219056331234031318847126679459107816861212),
            (-5, 13.0, 1e-16, -0.13161955992748078778652226517954056701010047018638),
            (-5, 18.0, 1e-16,  0.15537009877904934332134156316733754578398908625146),
            (-4, -2.0, 1e-50,  0.033995719807568434145759211288531044714832968346313),
            (-4,  3.0, 1e-16,  0.13203418392461221032868929577868226405660960634281),
            (-4,  8.0, 1e-16, -0.10535743487538893703873858559636699104829182769299),
            (-4, 13.0, 1e-16,  0.21927648745906773769554294077580869598893008207073), 
            (-4, 18.0, 1e-50,  0.069639512651394864261741057764135700037439539390376),
            (-3, -2.0, 1e-16,  0.12894324947440205109879333296923983526999372528246), 
            (-3,  3.0, 1e-50, -0.30906272225525164361826019494683314942913593599306),
            (-3,  8.0, 1e-16,  0.29113220706595224937905177406763378563936999630511),
            (-3, 13.0, 1e-16, -0.0033198169704070507953503137594186305215488110879089),
            (-3, 18.0, 1e-16, -0.18632099329078039410433758884028674580062888153607),
            (-2, -2.0, 1e-15,  0.35283402861563771915062078761918846109514820750107),
            (-2,  3.0, 1e-16,  0.48609126058589107690783109411498403480166226564330),
            (-2,  8.0, 1e-50, -0.11299172042407524999555024495435834818123566953584),
            (-2, 13.0, 1e-16, -0.21774426424195679117461202673300009728667678464554),
            (-2, 18.0, 1e-18, -0.0075325148878013995602951948173734514372299122116846),
        ];
        for (n, x, tol, reference) in mathematica {
            // println!("n = {}, x = {:?}", n, x);
            approx_eq(bessel_jn(n, x), reference, tol);
        }
    }

    #[test]
    fn bessel_jn_with_positive_n_works() {
        assert_eq!(bessel_jn(-123, 0.0), 0.0);

        // Mathematica:
        // Table[{n, -1 + 4 k, N[BesselJ[n, -1 + 4 k], 50]}, {n, 2, 7}, {k, 0, 4}]
        #[rustfmt::skip]
        let mathematica = [
            (2, -1.0, 1e-16,  0.11490348493190048046964688133516660534547031423021),
            (2,  3.0, 1e-50,  0.48609126058589107690783109411498403480166226564330), 
            (2,  7.0, 1e-50, -0.30141722008594012027859360795340085850208704473688),
            (2, 11.0, 1e-16,  0.13904751877870126995714895549718353849136554460924),
            (2, 15.0, 1e-50,  0.041571677975250474720149258888763224726202033971064),
            (3, -1.0, 1e-50,  -0.019563353982668405918905321621751508254508954928056),
            (3,  3.0, 1e-50,  0.30906272225525164361826019494683314942913593599306),
            (3,  7.0, 1e-50, -0.16755558799533423603151111263420177673348957104840),
            (3, 11.0, 1e-50,  0.22734803305806741748578524511861337818174780553224),
            (3, 15.0, 1e-16, -0.19401825782012263455509760970658677543654210243494),
            (4, -1.0, 1e-50,  0.0024766389641099550437850483953424441815834153381295),
            (4,  3.0, 1e-16,  0.13203418392461221032868929577868226405660960634281),
            (4,  7.0, 1e-16,  0.15779814466136791796586979712408504987338169812397),
            (4, 11.0, 1e-16, -0.015039500747028133146720639977939877664957650682562),
            (4, 15.0, 1e-16, -0.11917898110329952854218830277139793490081887494504),
            (5, -1.0, 1e-19, -0.00024975773021123443137506554098804519815836777698007),
            (5,  3.0, 1e-16,  0.043028434877047583924911260462986221388489680921116),
            (5,  7.0, 1e-50,  0.34789632475118328513536230934744183373164008319008),
            (5, 11.0, 1e-16, -0.23828585178317878704703661964802419830171700602865),
            (5, 15.0, 1e-16,  0.13045613456502955266593051489517454348943870246425),
            (6, -1.0, 1e-20,  0.000020938338002389269965607014538007800000262431671225),
            (6,  3.0, 1e-17,  0.011393932332213069421014905764605140571689330060906),
            (6,  7.0, 1e-50,  0.33919660498317963222750493051511756974324699214757),
            (6, 11.0, 1e-16, -0.20158400087404349144149446879299121170023962752530),
            (6, 15.0, 1e-16,  0.20614973747998589698614197936818096389377800992121),
            (7, -1.0, 1e-21, -1.5023258174368082122186334680484018447814030746246e-6),
            (7,  3.0, 1e-18,  0.0025472944518046937591483625954343408982676393225066),
            (7,  7.0, 1e-16,  0.23358356950569608439750328582133114297106904620575),
            (7, 11.0, 1e-16,  0.018376032647858614565406290055670149174182866910139),
            (7, 15.0, 1e-16,  0.034463655418959164922983068599370227625583705472715),
        ];
        for (n, x, tol, reference) in mathematica {
            // println!("n = {}, x = {:?}", n, x);
            approx_eq(bessel_jn(n, x), reference, tol);
        }
    }

    #[test]
    fn bessel_yn_with_negative_n_works() {
        // Mathematica: N[BesselY[-6, 2], 100]
        approx_eq(
            bessel_yn(-6, 2.0),
            -46.91400241607927421302779586696514823847870394405369892095755850291572783603804499784058944393818635,
            1e-13,
        );

        // Mathematica: N[BesselY[-4, 1], 100]
        assert_eq!(
            bessel_yn(-4, 1.0),
            -33.27842302897211869549331562045944156703605865974332188810733834018737081598321620902994512845428393,
        );

        // Mathematica: N[BesselY[-3, 1], 100]
        assert_eq!(
            bessel_yn(-3, 1.0),
            5.821517605964728847761757064429814395160614827377814441751008113342489898063975002297713467470519189
        );

        // Mathematica: N[BesselY[-2, 1], 100]
        approx_eq(
            bessel_yn(-2, 1.0),
            -1.650682606816254391077226766119444803927630304523564762398710339867568572400633804756335676368831206,
            1e-17,
        );

        // Mathematica: N[BesselY[-1, 1], 100]
        assert_eq!(
            bessel_yn(-1, 1.0),
            0.7812128213002887165471500000479648205499063907164446078438332461277843915385602167276292380048056346
        );
    }

    #[test]
    fn bessel_yn_with_positive_n_works() {
        // Mathematica: N[BesselY[6, 2], 100]
        approx_eq(
            bessel_yn(6, 2.0),
            -46.91400241607927421302779586696514823847870394405369892095755850291572783603804499784058944393818635,
            1e-13,
        );

        // Mathematica: N[BesselY[4, 1], 100]
        assert_eq!(
            bessel_yn(4, 1.0),
            -33.27842302897211869549331562045944156703605865974332188810733834018737081598321620902994512845428393,
        );

        // Mathematica: N[BesselY[3, 1], 100]
        assert_eq!(
            bessel_yn(3, 1.0),
            -5.821517605964728847761757064429814395160614827377814441751008113342489898063975002297713467470519189
        );

        // Mathematica: N[BesselY[2, 1], 100]
        approx_eq(
            bessel_yn(2, 1.0),
            -1.650682606816254391077226766119444803927630304523564762398710339867568572400633804756335676368831206,
            1e-16,
        );

        // Mathematica: N[BesselY[1, 1], 100]
        assert_eq!(
            bessel_yn(1, 1.0),
            -0.7812128213002887165471500000479648205499063907164446078438332461277843915385602167276292380048056346
        );
    }

    #[test]
    fn bessel_jn_edge_cases_work() {
        // Mathematica: N[BesselJ[2, 2^302], 55]
        approx_eq(
            bessel_jn(2, TWO_302),
            1.933303565664127367623773247390803574928962081463125585e-47,
            1e-61,
        );

        // Mathematica: Table[{n, N[BesselJ[n, 2^302], 50]}, {n, 2, 7}]
        let mathematica = [
            (3, 2.78849211658424e-46),
            (4, -1.933303565664e-47),
            (5, -2.78849211658424e-46),
            (6, 1.933303565664e-47),
            (7, 2.78849211658424e-46),
        ];
        for (n, reference) in mathematica {
            approx_eq(bessel_jn(n, TWO_302), reference, 1e-59);
        }
    }
}
