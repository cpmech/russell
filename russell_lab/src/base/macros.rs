/// Calls Complex64::new(real, imag)
///
/// ```
/// use russell_lab::cpx;
/// use num_complex::Complex64;
///
/// let x = cpx!(1.0, 2.0);
/// let y = cpx!(3.0, 4.0);
/// let z = x + y;
/// assert_eq!(z.re, 4.0);
/// assert_eq!(z.im, 6.0);
/// ```
#[macro_export]
macro_rules! cpx {
    ($real:expr, $imag:expr) => {{
        Complex64::new($real, $imag)
    }};
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use crate::cpx;
    use num_complex::Complex64;

    #[test]
    fn cpx_works() {
        let x = cpx!(1.0, 2.0);
        assert_eq!(x.re, 1.0);
        assert_eq!(x.im, 2.0);
    }
}
