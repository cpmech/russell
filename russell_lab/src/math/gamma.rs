use super::{EULER, PI};

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
//// This implementation is based on gamma.go file from Go (1.22.1),     ////
//// which, in turn, is based on the code described below.               ////
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
//                                                                         //
// Copyright 2010 The Go Authors. All rights reserved.                     //
// Use of this source code is governed by a BSD-style                      //
// license that can be found in the LICENSE file.                          //
//                                                                         //
// The original C code, the long comment, and the constants                //
// below are from http://netlib.sandia.gov/cephes/cprob/gamma.c.           //
// The go code is a simplified version of the original C.                  //
//                                                                         //
//      tgamma.c                                                           //
//                                                                         //
// Cephes Math Library Release 2.8:  June, 2000                            //
// Copyright 1984, 1987, 1989, 1992, 2000 by Stephen L. Moshier            //
//                                                                         //
// The readme file at http://netlib.sandia.gov/cephes/ says:               //
//    Some software in this archive may be from the book _Methods and      //
// Programs for Mathematical Functions_ (Prentice-Hall or Simon & Schuster //
// International, 1989) or from the Cephes Mathematical Library, a         //
// commercial product. In either event, it is copyrighted by the author.   //
// What you see here may be used freely but it comes with no support or    //
// guarantee.                                                              //
//                                                                         //
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

const GAM_P: [f64; 7] = [
    1.60119522476751861407e-04,
    1.19135147006586384913e-03,
    1.04213797561761569935e-02,
    4.76367800457137231464e-02,
    2.07448227648435975150e-01,
    4.94214826801497100753e-01,
    9.99999999999999996796e-01,
];

const GAM_Q: [f64; 8] = [
    -2.31581873324120129819e-05,
    5.39605580493303397842e-04,
    -4.45641913851797240494e-03,
    1.18139785222060435552e-02,
    3.58236398605498653373e-02,
    -2.34591795718243348568e-01,
    7.14304917030273074085e-02,
    1.00000000000000000320e+00,
];

const GAM_S: [f64; 5] = [
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
    w = 1.0 + w * ((((GAM_S[0] * w + GAM_S[1]) * w + GAM_S[2]) * w + GAM_S[3]) * w + GAM_S[4]);
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
/// Reference: <https://www.cplusplus.com/reference/cmath/tgamma/>
///
/// # Special cases
///
/// * `Γ(+Inf) = +Inf`
/// * `Γ(+0)   = +Inf`
/// * `Γ(-0)   = -Inf`
/// * `Γ(x)    = NaN  for integer x < 0`
/// * `Γ(-Inf) = NaN`
/// * `Γ(NaN)  = NaN`
pub fn gamma(x_in: f64) -> f64 {
    // special cases
    if is_neg_int(x_in) || x_in == f64::NEG_INFINITY || f64::is_nan(x_in) {
        return f64::NAN;
    } else if x_in == f64::INFINITY {
        return f64::INFINITY;
    } else if x_in == 0.0 {
        if sign_bit(x_in) {
            return f64::NEG_INFINITY;
        }
        return f64::INFINITY;
    }
    let mut q = f64::abs(x_in);
    let mut p = f64::floor(q);
    if q > 33.0 {
        if x_in >= 0.0 {
            let (y1, y2) = stirling(x_in);
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
    let mut xx = x_in;
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

    p = (((((xx * GAM_P[0] + GAM_P[1]) * xx + GAM_P[2]) * xx + GAM_P[3]) * xx + GAM_P[4]) * xx + GAM_P[5]) * xx
        + GAM_P[6];

    q = ((((((xx * GAM_Q[0] + GAM_Q[1]) * xx + GAM_Q[2]) * xx + GAM_Q[3]) * xx + GAM_Q[4]) * xx + GAM_Q[5]) * xx
        + GAM_Q[6])
        * xx
        + GAM_Q[7];

    z * p / q
}

// Reports whether x is negative or negative zero
fn sign_bit(x: f64) -> bool {
    f64::to_bits(x) & (1 << 63) != 0
}

/// Reports wether x is negative integer or not
fn is_neg_int(x: f64) -> bool {
    if x < 0.0 {
        let (_, xf) = modf(x);
        return xf == 0.0;
    }
    return false;
}

/// Returns integer and fractional floating-point numbers that sum to f
///
/// Both values have the same sign as f.
///
/// Returns `(integer, fractional)`
///
/// # Special cases
///
/// * `mod_f(±Inf) = ±Inf, NaN`
/// * `mod_f(NaN) = NaN, NaN`
pub fn modf(x: f64) -> (f64, f64) {
    let mut u = x.to_bits();
    let e = ((u >> 52 & 0x7ff) as i32) - 0x3ff;

    // no fractional part
    let integer: f64;
    if e >= 52 {
        integer = x;
        if e == 0x400 && (u << 12) != 0 {
            return (integer, x); // nan
        }
        u &= 1 << 63;
        return (integer, f64::from_bits(u));
    }

    // no integral part
    if e < 0 {
        u &= 1 << 63;
        integer = f64::from_bits(u);
        return (integer, x);
    }

    let mask: u64 = ((!0) >> 12) >> e;
    if (u & mask) == 0 {
        integer = x;
        u &= 1 << 63;
        return (integer, f64::from_bits(u));
    }
    u &= !mask;
    integer = f64::from_bits(u);
    (integer, x - integer)
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::gamma;
    use crate::approx_eq;
    use crate::math::PI;

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
}
