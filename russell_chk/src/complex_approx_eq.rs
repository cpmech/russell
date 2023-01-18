use num_complex::Complex;
use num_traits::{Num, NumCast};

/// Panics if two numbers are not approximately equal to each other
///
/// # Input
///
/// `a` -- Left value
/// `b` -- Right value
/// `tol: f64` -- Error tolerance: panics occurs if `|a.re - b.re| > tol` or `|a.im - b.im| > tol`
///
/// # Examples
///
/// ## Accepts small error
///
/// ```
/// use russell_chk::complex_approx_eq;
/// use num_complex::Complex64;
///
/// fn main() {
///     let a = Complex64::new(3.0000001, 2.0000001);
///     let b = Complex64::new(3.0, 2.0);
///     complex_approx_eq(a, b, 1e-6);
/// }
/// ```
///
/// ## Panics on different values
///
/// ```should_panic
/// use russell_chk::complex_approx_eq;
/// use num_complex::Complex64;
///
/// fn main() {
///     let a = Complex64::new(1.0, 3.0);
///     let b = Complex64::new(2.0, 3.0);
///     complex_approx_eq(a, b, 1e-6);
/// }
/// ```
///
/// ```should_panic
/// use russell_chk::complex_approx_eq;
/// use num_complex::Complex64;
///
/// fn main() {
///     let a = Complex64::new(1.0, 3.0);
///     let b = Complex64::new(1.0, 4.0);
///     complex_approx_eq(a, b, 1e-6);
/// }
/// ```
pub fn complex_approx_eq<T>(a: Complex<T>, b: Complex<T>, tol: f64)
where
    T: Num + NumCast + Copy,
{
    let diff_re = f64::abs(a.re.to_f64().unwrap() - b.re.to_f64().unwrap());
    if diff_re > tol {
        panic!("complex numbers are not approximately equal. diff_re = {:?}", diff_re);
    }
    let diff_im = f64::abs(a.im.to_f64().unwrap() - b.im.to_f64().unwrap());
    if diff_im > tol {
        panic!("complex numbers are not approximately equal. diff_im = {:?}", diff_im);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::complex_approx_eq;
    use num_complex::{Complex32, Complex64};

    #[test]
    #[should_panic(expected = "complex numbers are not approximately equal. diff_re = 0.5")]
    fn panics_on_different_values_re() {
        complex_approx_eq(Complex64::new(2.0, 3.0), Complex64::new(2.5, 3.0), 1e-1);
    }

    #[test]
    #[should_panic(expected = "complex numbers are not approximately equal. diff_im = 0.5")]
    fn panics_on_different_values_im() {
        complex_approx_eq(Complex64::new(2.0, 3.0), Complex64::new(2.0, 3.5), 1e-1);
    }

    #[test]
    #[should_panic(expected = "complex numbers are not approximately equal. diff_re = 0.5")]
    fn panics_on_different_values_f32_re() {
        complex_approx_eq(Complex32::new(2f32, 3f32), Complex32::new(2.5f32, 3f32), 1e-1);
    }

    #[test]
    #[should_panic(expected = "complex numbers are not approximately equal. diff_im = 0.5")]
    fn panics_on_different_values_f32_im() {
        complex_approx_eq(Complex32::new(2f32, 3f32), Complex32::new(2f32, 3.5f32), 1e-1);
    }

    #[test]
    fn accepts_approx_equal_values() {
        let tol = 0.03;

        let a = Complex64::new(2.0, 3.0);
        let b = Complex64::new(2.02, 3.0);
        complex_approx_eq(a, b, tol);

        let a = Complex64::new(2.0, 3.0);
        let b = Complex64::new(2.0, 3.02);
        complex_approx_eq(a, b, tol);
    }

    #[test]
    fn accepts_approx_equal_values_f32() {
        let tol = 0.03;

        let a = Complex32::new(2.0, 3.0);
        let b = Complex32::new(2.0, 3.02);
        complex_approx_eq(a, b, tol);

        let a = Complex32::new(2.0, 3.0);
        let b = Complex32::new(2.02, 3.0);
        complex_approx_eq(a, b, tol);
    }
}
