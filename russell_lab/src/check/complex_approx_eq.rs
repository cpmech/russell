use num_complex::Complex;
use num_traits::{Num, NumCast};

/// Panics if two numbers are not approximately equal to each other
///
/// # Panics
///
/// 1. Will panic if NAN, INFINITY, or NEG_INFINITY is found
/// 2. Will panic if the absolute difference of each real/imag part is greater than the tolerance
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
/// use russell_lab::{complex_approx_eq, cpx};
/// use num_complex::Complex64;
///
/// fn main() {
///     let a = cpx!(3.0000001, 2.0000001);
///     let b = cpx!(3.0, 2.0);
///     complex_approx_eq(a, b, 1e-6);
/// }
/// ```
///
/// ## Panics on different values
///
/// ### Real part
///
/// ```should_panic
/// use russell_lab::{complex_approx_eq, cpx};
/// use num_complex::Complex64;
///
/// fn main() {
///     let a = cpx!(1.0, 3.0);
///     let b = cpx!(2.0, 3.0);
///     complex_approx_eq(a, b, 1e-6);
/// }
/// ```
///
/// ### Imaginary part
///
/// ```should_panic
/// use russell_lab::{complex_approx_eq, cpx};
/// use num_complex::Complex64;
///
/// fn main() {
///     let a = cpx!(1.0, 3.0);
///     let b = cpx!(1.0, 4.0);
///     complex_approx_eq(a, b, 1e-6);
/// }
/// ```
pub fn complex_approx_eq<T>(a: Complex<T>, b: Complex<T>, tol: f64)
where
    T: Num + NumCast + Copy,
{
    let aa_re = a.re.to_f64().unwrap();
    let bb_re = b.re.to_f64().unwrap();

    if aa_re.is_nan() {
        panic!("the real part of the first number is NaN");
    }
    if bb_re.is_nan() {
        panic!("the real part of the second number is NaN");
    }
    if aa_re.is_infinite() {
        panic!("the real part of the first number is Inf");
    }
    if bb_re.is_infinite() {
        panic!("the real part of the second number is Inf");
    }

    let aa_im = a.im.to_f64().unwrap();
    let bb_im = b.im.to_f64().unwrap();

    if aa_im.is_nan() {
        panic!("the imaginary part of the first number is NaN");
    }
    if bb_im.is_nan() {
        panic!("the imaginary part of the second number is NaN");
    }
    if aa_im.is_infinite() {
        panic!("the imaginary part of the first number is Inf");
    }
    if bb_im.is_infinite() {
        panic!("the imaginary part of the second number is Inf");
    }

    let diff_re = f64::abs(aa_re - bb_re);
    if diff_re > tol {
        panic!("complex numbers are not approximately equal. diff_re = {:?}", diff_re);
    }

    let diff_im = f64::abs(aa_im - bb_im);
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
    #[should_panic(expected = "the real part of the first number is NaN")]
    fn panics_on_nan_real_1() {
        complex_approx_eq(Complex64::new(f64::NAN, 0.0), Complex64::new(2.5, 0.0), 1e-1);
    }

    #[test]
    #[should_panic(expected = "the real part of the second number is NaN")]
    fn panics_on_nan_real_2() {
        complex_approx_eq(Complex64::new(2.5, 0.0), Complex64::new(f64::NAN, 0.0), 1e-1);
    }

    #[test]
    #[should_panic(expected = "the real part of the first number is Inf")]
    fn panics_on_inf_real_1() {
        complex_approx_eq(Complex64::new(f64::INFINITY, 0.0), Complex64::new(2.5, 0.0), 1e-1);
    }

    #[test]
    #[should_panic(expected = "the real part of the second number is Inf")]
    fn panics_on_inf_real_2() {
        complex_approx_eq(Complex64::new(2.5, 0.0), Complex64::new(f64::INFINITY, 0.0), 1e-1);
    }

    #[test]
    #[should_panic(expected = "the real part of the first number is Inf")]
    fn panics_on_inf_real_3() {
        complex_approx_eq(Complex64::new(f64::NEG_INFINITY, 0.0), Complex64::new(2.5, 0.0), 1e-1);
    }

    #[test]
    #[should_panic(expected = "the real part of the second number is Inf")]
    fn panics_on_inf_real_4() {
        complex_approx_eq(Complex64::new(2.5, 0.0), Complex64::new(f64::NEG_INFINITY, 0.0), 1e-1);
    }

    // ------

    #[test]
    #[should_panic(expected = "the imaginary part of the first number is NaN")]
    fn panics_on_nan_imag_1() {
        complex_approx_eq(Complex64::new(0.0, f64::NAN), Complex64::new(2.5, 0.0), 1e-1);
    }

    #[test]
    #[should_panic(expected = "the imaginary part of the second number is NaN")]
    fn panics_on_nan_imag_2() {
        complex_approx_eq(Complex64::new(2.5, 0.0), Complex64::new(0.0, f64::NAN), 1e-1);
    }

    #[test]
    #[should_panic(expected = "the imaginary part of the first number is Inf")]
    fn panics_on_inf_imag_1() {
        complex_approx_eq(Complex64::new(0.0, f64::INFINITY), Complex64::new(2.5, 0.0), 1e-1);
    }

    #[test]
    #[should_panic(expected = "the imaginary part of the second number is Inf")]
    fn panics_on_inf_imag_2() {
        complex_approx_eq(Complex64::new(2.5, 0.0), Complex64::new(0.0, f64::INFINITY), 1e-1);
    }

    #[test]
    #[should_panic(expected = "the imaginary part of the first number is Inf")]
    fn panics_on_inf_imag_3() {
        complex_approx_eq(Complex64::new(0.0, f64::NEG_INFINITY), Complex64::new(2.5, 0.0), 1e-1);
    }

    #[test]
    #[should_panic(expected = "the imaginary part of the second number is Inf")]
    fn panics_on_inf_imag_4() {
        complex_approx_eq(Complex64::new(2.5, 0.0), Complex64::new(0.0, f64::NEG_INFINITY), 1e-1);
    }

    // ------

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
