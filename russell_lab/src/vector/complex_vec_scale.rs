use super::ComplexVector;
use crate::to_i32;
use num_complex::Complex64;

extern "C" {
    // Scales a vector by a constant
    // <https://www.netlib.org/lapack/explore-html/d2/d74/zscal_8f.html>
    fn cblas_zscal(n: i32, alpha: *const Complex64, x: *mut Complex64, incx: i32);
}

/// Scales a complex vector by a constant
///
/// ```text
/// u := alpha * u
/// ```
///
/// See also: <https://www.netlib.org/lapack/explore-html/d2/d74/zscal_8f.html>
///
/// # Example
///
/// ```
/// use num_complex::Complex64;
/// use russell_lab::{complex_vec_scale, cpx, ComplexVector};
///
/// fn main() {
///     let mut u = ComplexVector::from(&[cpx!(1.0, 1.0), cpx!(2.0, -2.0), cpx!(3.0, 3.0)]);
///     complex_vec_scale(&mut u, cpx!(0.5, -0.5));
///     println!("{}", u);
///     let correct = "┌      ┐\n\
///                    │ 1+0i │\n\
///                    │ 0-2i │\n\
///                    │ 3+0i │\n\
///                    └      ┘";
///     assert_eq!(format!("{}", u), correct);
/// }
/// ```
pub fn complex_vec_scale(v: &mut ComplexVector, alpha: Complex64) {
    let n_i32: i32 = to_i32(v.dim());
    unsafe {
        cblas_zscal(n_i32, &alpha, v.as_mut_data().as_mut_ptr(), 1);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::complex_vec_scale;
    use crate::{complex_vec_approx_eq, cpx, ComplexVector};
    use num_complex::Complex64;

    #[test]
    fn complex_vec_scale_works() {
        let mut u = ComplexVector::from(&[6.0, 9.0, 12.0]);
        complex_vec_scale(&mut u, cpx!(1.0 / 3.0, 0.0));
        let correct = &[cpx!(2.0, 0.0), cpx!(3.0, 0.0), cpx!(4.0, 0.0)];
        complex_vec_approx_eq(&u, correct, 1e-15);
    }
}
