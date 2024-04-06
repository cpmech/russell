use super::{float_is_neg_int, EULER, PI};

// This implementation is based on gamma.go file from Go (1.22.1),
// which, in turn, is based on the code described below.
//
// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// The original C code, the long comment, and the constants
// below are from http://netlib.sandia.gov/cephes/cprob/gamma.c.
// The go code is a simplified version of the original C.
//
// Cephes Math Library Release 2.8:  June, 2000
// Copyright 1984, 1987, 1989, 1992, 2000 by Stephen L. Moshier
//
// The readme file at http://netlib.sandia.gov/cephes/ says:
//    Some software in this archive may be from the book _Methods and
// Programs for Mathematical Functions_ (Prentice-Hall or Simon & Schuster
// International, 1989) or from the Cephes Mathematical Library, a
// commercial product. In either event, it is copyrighted by the author.
// What you see here may be used freely but it comes with no support or
// guarantee.

const GP: [f64; 7] = [
    1.60119522476751861407e-04,
    1.19135147006586384913e-03,
    1.04213797561761569935e-02,
    4.76367800457137231464e-02,
    2.07448227648435975150e-01,
    4.94214826801497100753e-01,
    9.99999999999999996796e-01,
];

const GQ: [f64; 8] = [
    -2.31581873324120129819e-05,
    5.39605580493303397842e-04,
    -4.45641913851797240494e-03,
    1.18139785222060435552e-02,
    3.58236398605498653373e-02,
    -2.34591795718243348568e-01,
    7.14304917030273074085e-02,
    1.00000000000000000320e+00,
];

const GS: [f64; 5] = [
    7.87311395793093628397e-04,
    -2.29549961613378126380e-04,
    -2.68132617805781232825e-03,
    3.47222221605458667310e-03,
    8.33333333333482257126e-02,
];

const SQRT_TWO_PI: f64 = 2.506628274631000502417;

const MAX_STIRLING: f64 = 143.01608;

// Evaluates the Gamma using Stirling's formula.
// The pair of results must be multiplied together to get the actual answer.
// The multiplication is left to the caller so that, if careful, the caller can avoid
// infinity for 172 <= x <= 180.
// The polynomial is valid for 33 <= x <= 172; larger values are only used
// in reciprocal and produce de-normalized floats. The lower precision there
// masks any imprecision in the polynomial.
fn stirling(x: f64) -> (f64, f64) {
    if x > 200.0 {
        return (f64::INFINITY, 1.0);
    }
    let mut w = 1.0 / x;
    w = 1.0 + w * ((((GS[0] * w + GS[1]) * w + GS[2]) * w + GS[3]) * w + GS[4]);
    let mut y1 = f64::exp(x);
    let mut y2 = 1.0;
    if x > MAX_STIRLING {
        // avoid Pow() overflow
        let v = f64::powf(x, 0.5 * x - 0.25);
        y2 = v / y1;
        y1 = v;
    } else {
        y1 = f64::powf(x, x - 0.5) / y1;
    }
    (y1, SQRT_TWO_PI * w * y2)
}

/// Evaluates the Gamma function Γ(x)
///
/// See: <https://mathworld.wolfram.com/GammaFunction.html>
///
/// See also: <https://en.wikipedia.org/wiki/Gamma_function>
///
/// # Special cases
///
/// * `Γ(+Inf) = +Inf`
/// * `Γ(+0)   = +Inf`
/// * `Γ(-0)   = -Inf`
/// * `Γ(x)    = NaN  for integer x < 0`
/// * `Γ(-Inf) = NaN`
/// * `Γ(NaN)  = NaN`
pub fn gamma(x: f64) -> f64 {
    // special cases
    if float_is_neg_int(x) || x == f64::NEG_INFINITY || f64::is_nan(x) {
        return f64::NAN;
    } else if x == f64::INFINITY {
        return f64::INFINITY;
    } else if x == 0.0 {
        if f64::is_sign_negative(x) {
            return f64::NEG_INFINITY;
        }
        return f64::INFINITY;
    }
    let mut q = f64::abs(x);
    let mut p = f64::floor(q);
    if q > 33.0 {
        if x >= 0.0 {
            let (y1, y2) = stirling(x);
            return y1 * y2;
        }
        // Note: x is negative but (checked above) not a negative integer,
        // so x must be small enough to be in range for conversion to int64.
        // If |x| were >= 2⁶³ it would have to be an integer.
        let mut sign_gam = 1;
        let ip = p as i64;
        if ip & 1 == 0 {
            sign_gam = -1;
        }
        let mut z = q - p;
        if z > 0.5 {
            p = p + 1.0;
            z = q - p;
        }
        z = q * f64::sin(PI * z);
        if z == 0.0 {
            if sign_gam < 0 {
                return f64::NEG_INFINITY;
            }
            return f64::INFINITY;
        }
        let (sq1, sq2) = stirling(q);
        let abs_z = f64::abs(z);
        let d = abs_z * sq1 * sq2;
        if f64::is_infinite(d) {
            z = PI / abs_z / sq1 / sq2;
        } else {
            z = PI / d;
        }
        return (sign_gam as f64) * z;
    }

    // Reduce argument
    let mut xx = x;
    let mut z = 1.0;
    while xx >= 3.0 {
        xx = xx - 1.0;
        z = z * xx;
    }
    while xx < 0.0 {
        if xx > -1e-09 {
            if xx == 0.0 {
                return f64::INFINITY;
            }
            return z / ((1.0 + EULER * xx) * xx);
        }
        z = z / xx;
        xx = xx + 1.0;
    }
    while xx < 2.0 {
        if xx < 1e-09 {
            if xx == 0.0 {
                return f64::INFINITY;
            }
            return z / ((1.0 + EULER * xx) * xx);
        }
        z = z / xx;
        xx = xx + 1.0;
    }

    if xx == 2.0 {
        return z;
    }

    xx = xx - 2.0;
    p = (((((xx * GP[0] + GP[1]) * xx + GP[2]) * xx + GP[3]) * xx + GP[4]) * xx + GP[5]) * xx + GP[6];
    q = ((((((xx * GQ[0] + GQ[1]) * xx + GQ[2]) * xx + GQ[3]) * xx + GQ[4]) * xx + GQ[5]) * xx + GQ[6]) * xx + GQ[7];
    z * p / q
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::gamma;
    use crate::math::PI;
    use crate::{approx_eq, assert_alike};

    #[test]
    #[rustfmt::skip]
    fn gamma_works() {
        assert!(gamma(f64::NAN).is_nan());
        approx_eq(gamma(0.5), 1.772454, 1e-6);
        approx_eq(gamma(1.000001e-35), 9.9999900000099999900000099999899999522784235098567139293e+34, 1e20);
        approx_eq(gamma(1.000001e-10), 9.99998999943278432519738283781280989934496494539074049002e+9, 1e-5);
        approx_eq(gamma(1.000001e-5), 99999.32279432557746387178953902739303931424932435387031653234, 1e-10);
        approx_eq(gamma(1.000001e-2), 99.43248512896257405886134437203369035261893114349805309870831, 1e-13);
        approx_eq(gamma(-4.8), -0.06242336135475955314181664931547009890495158793105543559676, 1e-13);
        approx_eq(gamma(-1.5), 2.363271801207354703064223311121526910396732608163182837618410, 1e-13);
        approx_eq(gamma(-0.5), -3.54490770181103205459633496668229036559509891224477425642761, 1e-13);
        approx_eq(gamma(1.0e-5 + 1.0e-16), 99999.42279322556767360213300482199406241771308740302819426480, 1e-9);
        approx_eq(gamma(0.1), 9.513507698668731836292487177265402192550578626088377343050000, 1e-14);
        approx_eq(gamma(1.0 - 1.0e-14), 1.000000000000005772156649015427511664653698987042926067639529, 1e-15);
        approx_eq(gamma(1.0), 1.0, 1e-15);
        approx_eq(gamma(1.0 + 1.0e-14), 0.99999999999999422784335098477029953441189552403615306268023, 1e-15);
        approx_eq(gamma(1.5), 0.886226925452758013649083741670572591398774728061193564106903, 1e-14);
        approx_eq(gamma(PI / 2.0), 0.890560890381539328010659635359121005933541962884758999762766, 1e-15);
        assert_eq!(gamma(2.0), 1.0);
        approx_eq(gamma(2.5), 1.329340388179137020473625612505858887098162092091790346160355, 1e-13);
        approx_eq(gamma(3.0), 2.0, 1e-14);
        approx_eq(gamma(PI), 2.288037795340032417959588909060233922889688153356222441199380, 1e-13);
        approx_eq(gamma(3.5), 3.323350970447842551184064031264647217745405230229475865400889, 1e-14);
        approx_eq(gamma(4.0), 6.0, 1e-13);
        approx_eq(gamma(4.5), 11.63172839656744892914422410942626526210891830580316552890311, 1e-12);
        approx_eq(gamma(5.0 - 1.0e-14), 23.99999999999963853175957637087420162718107213574617032780374, 1e-13);
        approx_eq(gamma(5.0), 24.0, 1e-12);
        approx_eq(gamma(5.0 + 1.0e-14), 24.00000000000036146824042363510111050137786752408660789873592, 1e-12);
        approx_eq(gamma(5.5), 52.34277778455352018114900849241819367949013237611424488006401, 1e-12);
        approx_eq(gamma(10.1), 454760.7514415859508673358368319076190405047458218916492282448, 1e-7);
        approx_eq(gamma(150.0 + 1.0e-12), 3.8089226376496421386707466577615064443807882167327097140e+260, 1e248);
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

    const SOLUTION: [f64; 10] = [
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

    // test inputs inspired by Python test suite
    const EXTRA: [[f64; 2]; 71] = [
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

    #[test]
    fn test_gamma() {
        for (i, v) in VALUES.iter().enumerate() {
            let f = gamma(*v);
            // println!("Γ({:?}) = {:?} =? {:?}", *v, f, SOLUTION[i]);
            if SOLUTION[i] > 160_000.0 {
                approx_eq(SOLUTION[i], f, 1e-10);
            } else {
                approx_eq(SOLUTION[i], f, 1e-14);
            }
        }
        for g in EXTRA {
            let f = gamma(g[0]);
            // println!("Γ({:?}) = {:?} =? {:?}", g[0], f, g[1]);
            if f64::is_nan(g[1]) || f64::is_infinite(g[1]) || g[1] == 0.0 || f == 0.0 {
                assert_alike(g[1], f);
            } else {
                let a = f64::abs(g[1]);
                let tol = if a > 1e+300 {
                    2e292
                } else if a > 1e+15 {
                    2.1
                } else if a > 99999999.0 {
                    1e-7
                } else if a > 999.0 {
                    1e-12
                } else if a > 99.0 {
                    1e-13
                } else {
                    1e-14
                };
                approx_eq(g[1], f, tol);
            };
        }
    }
}
