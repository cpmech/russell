/// Allocates a new Complex64 number
///
/// This macro simply calls `Complex64::new(real, imag)`
///
/// **Note:** When using this macro, the `Complex64` type must be imported as well.
///
/// # Examples
///
/// ```
/// use russell_lab::{cpx, Complex64};
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
    use crate::{cpx, Complex64};

    #[test]
    fn cpx_works() {
        let x = cpx!(1.0, 2.0);
        assert_eq!(x.re, 1.0);
        assert_eq!(x.im, 2.0);
    }
}
