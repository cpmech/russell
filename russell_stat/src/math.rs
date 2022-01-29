extern "C" {
    fn c_erf(x: f64) -> f64;
    fn c_gamma(x: f64) -> f64;
}

#[inline]
pub fn erf(x: f64) -> f64 {
    unsafe { c_erf(x) }
}

#[inline]
pub fn gamma(x: f64) -> f64 {
    unsafe { c_gamma(x) }
}
