use super::ComplexVector;
use crate::{complex_array_approx_eq, AsArray1D};
use num_complex::Complex64;

/// Panics if two complex vectors are not approximately equal to each other
///
/// # Panics
///
/// 1. Will panic if the dimensions are different
/// 2. Will panic if NAN, INFINITY, or NEG_INFINITY is found
/// 3. Will panic if the absolute difference of components is greater than the tolerance
///
/// # Examples
///
/// ## Accepts small error
///
/// ```
/// use russell_lab::{complex_vec_approx_eq, cpx, ComplexVector};
/// use num_complex::Complex64;
///
/// fn main() {
///     let a = ComplexVector::from(&[cpx!(3.0000001, 2.0000001), cpx!(1.0, 2.0)]);
///     let b = ComplexVector::from(&[cpx!(3.0, 2.0),             cpx!(1.0, 2.0)]);
///     complex_vec_approx_eq(&a, &b, 1e-6);
/// }
/// ```
///
/// ## Panics on different values
///
/// ### Real part
///
/// ```should_panic
/// use russell_lab::{complex_vec_approx_eq, cpx, ComplexVector};
/// use num_complex::Complex64;
///
/// fn main() {
///     let a = ComplexVector::from(&[cpx!(1.0, 3.0), cpx!(1.0, 2.0)]);
///     let b = ComplexVector::from(&[cpx!(2.0, 3.0), cpx!(1.0, 2.0)]);
///     complex_vec_approx_eq(&a, &b, 1e-6);
/// }
/// ```
///
/// ### Imaginary part
///
/// ```should_panic
/// use russell_lab::{complex_vec_approx_eq, cpx, ComplexVector};
/// use num_complex::Complex64;
///
/// fn main() {
///     let a = ComplexVector::from(&[cpx!(1.0, 3.0), cpx!(1.0, 2.0)]);
///     let b = ComplexVector::from(&[cpx!(1.0, 4.0), cpx!(1.0, 2.0)]);
///     complex_vec_approx_eq(&a, &b, 1e-6);
/// }
/// ```
pub fn complex_vec_approx_eq<'a, T>(u: &ComplexVector, v: &'a T, tol: f64)
where
    T: AsArray1D<'a, Complex64>,
{
    complex_array_approx_eq(u.as_data(), v.as_slice(), tol);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::complex_vec_approx_eq;
    use crate::ComplexVector;

    #[test]
    fn complex_vec_approx_eq_works() {
        let u = ComplexVector::from(&[1.0, 2.0, 3.0]);
        let v = ComplexVector::from(&[1.0, 2.0, 3.01]);
        complex_vec_approx_eq(&u, &v, 0.011);
    }

    #[test]
    #[should_panic(expected = "vectors are not approximately equal. diff_re[2] =")]
    fn complex_vec_approx_eq_panics() {
        let u = ComplexVector::from(&[1.0, 2.0, 3.0]);
        let v = ComplexVector::from(&[1.0, 2.0, 3.01]);
        complex_vec_approx_eq(&u, &v, 0.009);
    }
}
