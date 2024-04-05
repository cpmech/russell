extern "C" {
    fn c_gamma(x: f64) -> f64;
    fn c_ln_gamma(x: f64) -> f64;
    fn c_frexp(x: f64, exp: *mut i32) -> f64;
    fn c_ldexp(frac: f64, exp: i32) -> f64;
}

/// Evaluates the Gamma function Γ(x)
///
/// Reference: <https://www.cplusplus.com/reference/cmath/tgamma/>
#[inline]
pub fn gamma(x: f64) -> f64 {
    unsafe { c_gamma(x) }
}

/// Evaluates the natural logarithm of Γ(x)
///
/// Reference: <https://cplusplus.com/reference/cmath/lgamma/>
///
/// **WARNING:** This function is not thread-safe! see, e.g., <https://en.cppreference.com/w/c/numeric/math/lgamma>
#[inline]
pub fn ln_gamma(x: f64) -> f64 {
    unsafe { c_ln_gamma(x) }
}

/// Gets the significand and exponent of a number
///
/// Breaks a value x into a normalized fraction and an integral power of two
///
/// Returns `(frac, exp)` satisfying:
///
/// ```text
/// x == frac · 2^exp
/// ```
///
/// with the absolute value of frac in the interval [0.5, 1)
///
/// Special cases:
///
///	* `frexp(±0.0) = ±0.0, 0`
///	* `frexp(±Inf) = ±Inf, 0`
///	* `frexp(NaN)  = NaN,  0`
///
/// Reference: <https://cplusplus.com/reference/cmath/frexp/>
pub fn frexp(x: f64) -> (f64, i32) {
    if x == 0.0 || f64::is_nan(x) {
        return (x, 0); // correctly return -0 or NaN
    } else if f64::is_infinite(x) || f64::is_nan(x) {
        return (x, 0);
    }
    unsafe {
        let mut exp: i32 = 0;
        let frac = c_frexp(x, &mut exp);
        (frac, exp)
    }
}

/// Generates a number from significand and exponent
///
/// Returns:
///
/// ```text
/// x = frac · 2^exp
/// ```
///
/// Returns the result of multiplying x (the significand) by 2 raised to the power of exp (the exponent).
///
/// Special cases:
///
/// * `ldexp(±0.0, exp) = ±0.0`
/// * `ldexp(±Inf, exp) = ±Inf`
/// * `ldexp(NaN,  exp) = NaN`
///
/// Reference: <https://cplusplus.com/reference/cmath/ldexp/>
pub fn ldexp(frac: f64, exp: i32) -> f64 {
    if frac == 0.0 || f64::is_nan(frac) {
        return frac; // correctly return -0 or NaN
    } else if f64::is_infinite(frac) || f64::is_nan(frac) {
        return frac;
    }
    unsafe { c_ldexp(frac, exp) }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{frexp, gamma, ldexp, ln_gamma};
    use crate::approx_eq;
    use crate::math::{ONE_BY_3, PI};

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
        assert_eq!(gamma(1.0 - 1.0e-14), 1.000000000000005772156649015427511664653698987042926067639529);
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

    #[test]
    fn ln_gamma_works() {
        assert!(ln_gamma(f64::NAN).is_nan());
        assert_eq!(ln_gamma(-1.0), f64::INFINITY);
        assert_eq!(ln_gamma(0.0), f64::INFINITY);
        assert_eq!(ln_gamma(1.0), 0.0);
        assert_eq!(ln_gamma(2.0), 0.0);

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
            approx_eq(ln_gamma(x), reference, tol)
        }
    }

    #[test]
    fn frexp_works() {
        assert_eq!(frexp(-0.0), (-0.0, 0));
        assert_eq!(frexp(0.0), (0.0, 0));
        assert_eq!(frexp(f64::NEG_INFINITY), (f64::NEG_INFINITY, 0));
        assert_eq!(frexp(f64::INFINITY), (f64::INFINITY, 0));
        assert!(frexp(f64::NAN).0.is_nan());
        assert_eq!(frexp(8.0), (0.5, 4));
    }

    #[test]
    fn ldexp_works() {
        assert_eq!(ldexp(-0.0, 0), -0.0);
        assert_eq!(ldexp(0.0, 0), 0.0);
        assert_eq!(ldexp(f64::NEG_INFINITY, 0), f64::NEG_INFINITY);
        assert_eq!(ldexp(f64::INFINITY, 0), f64::INFINITY);
        assert!(ldexp(f64::NAN, 0).is_nan());
        assert_eq!(ldexp(0.5, 4), 8.0);
    }
}
