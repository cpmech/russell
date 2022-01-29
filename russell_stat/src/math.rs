extern "C" {
    fn c_erf(x: f64) -> f64;
    fn c_gamma(x: f64) -> f64;
}

/// Returns the error function (wraps C-code: erf)
///
/// Code from: https://www.cplusplus.com/reference/cmath/erf/
///
/// https://en.wikipedia.org/wiki/Error_function
#[inline]
pub fn erf(x: f64) -> f64 {
    unsafe { c_erf(x) }
}

/// Returns the Gamma function Γ (wraps C-code: tgamma)
///
/// Code from: https://www.cplusplus.com/reference/cmath/tgamma/
#[inline]
pub fn gamma(x: f64) -> f64 {
    unsafe { c_gamma(x) }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use crate::{erf, gamma};
    use russell_chk::assert_approx_eq;

    #[test]
    fn erf_works() {
        assert_eq!(erf(0.0), 0.0);
        assert_approx_eq!(erf(0.3), 0.328626759, 1e-9);
        assert_approx_eq!(erf(1.0), 0.842700793, 1e-9);
        assert_approx_eq!(erf(1.8), 0.989090502, 1e-9);
        assert_approx_eq!(erf(3.5), 0.999999257, 1e-9);
    }

    #[test]
    fn gamma_works() {
        assert_approx_eq!(gamma(0.5), 1.772454, 1e-6);
    }
}
