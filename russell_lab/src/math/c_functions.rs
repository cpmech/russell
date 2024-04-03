extern "C" {
    fn c_erf(x: f64) -> f64;
    fn c_erfc(x: f64) -> f64;
    fn c_gamma(x: f64) -> f64;
    fn c_frexp(x: f64, exp: *mut i32) -> f64;
    fn c_ldexp(frac: f64, exp: i32) -> f64;
}

/// Returns the error function (wraps C-code: erf)
///
/// Reference: <https://www.cplusplus.com/reference/cmath/erf/>
///
/// See also: <https://en.wikipedia.org/wiki/Error_function>
#[inline]
pub fn erf(x: f64) -> f64 {
    unsafe { c_erf(x) }
}

/// Returns the complementary error function (wraps C-code: erfc)
///
/// Reference: <https://www.cplusplus.com/reference/cmath/erfc/>
///
/// See also: <https://en.wikipedia.org/wiki/Error_function>
#[inline]
pub fn erfc(x: f64) -> f64 {
    unsafe { c_erfc(x) }
}

/// Returns the Gamma function Γ (wraps C-code: tgamma)
///
/// Reference: <https://www.cplusplus.com/reference/cmath/tgamma/>
#[inline]
pub fn gamma(x: f64) -> f64 {
    unsafe { c_gamma(x) }
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
/// The special cases are:
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
/// The special cases are:
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
    use super::{erf, erfc, frexp, gamma, ldexp};
    use crate::approx_eq;
    use crate::math::PI;

    #[test]
    fn erf_works() {
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
    #[rustfmt::skip]
    fn erfc_works() {
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
