use super::ComplexMatrix;
use crate::to_i32;
use num_complex::Complex64;

extern "C" {
    // Scales a vector by a constant
    // <https://www.netlib.org/lapack/explore-html/d2/d74/zscal_8f.html>
    fn cblas_zscal(n: i32, alpha: *const Complex64, x: *mut Complex64, incx: i32);

}

/// (zscal) Scales matrix (complex version)
///
/// ```text
/// a := alpha * a
/// ```
///
/// See also <https://www.netlib.org/lapack/explore-html/d2/d74/zscal_8f.html>
///
/// # Example
///
/// ```
/// use num_complex::Complex64;
/// use russell_lab::*;
///
/// fn main() -> Result<(), StrError> {
///     #[rustfmt::skip]
///     let mut a = ComplexMatrix::from(&[
///         [cpx!( 6.0, 3.0), cpx!( 9.0, -3.0)],
///         [cpx!(-6.0, 3.0), cpx!(-9.0, -3.0)],
///     ]);
///
///     complex_mat_scale(&mut a, cpx!(1.0 / 3.0, 0.0));
///
///     #[rustfmt::skip]
///     let a_correct = ComplexMatrix::from(&[
///         [cpx!( 2.0, 1.0), cpx!( 3.0, -1.0)],
///         [cpx!(-2.0, 1.0), cpx!(-3.0, -1.0)],
///     ]);
///
///     complex_mat_approx_eq(&a, &a_correct, 1e-15);
///     Ok(())
/// }
/// ```
pub fn complex_mat_scale(a: &mut ComplexMatrix, alpha: Complex64) {
    let (m, n) = a.dims();
    let mn_i32 = to_i32(m * n);
    unsafe {
        cblas_zscal(mn_i32, &alpha, a.as_mut_data().as_mut_ptr(), 1);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{complex_mat_scale, ComplexMatrix};
    use crate::{complex_mat_approx_eq, cpx};
    use num_complex::Complex64;

    #[test]
    fn complex_mat_scale_works() {
        #[rustfmt::skip]
        let mut a = ComplexMatrix::from(&[
            [cpx!( 6.0, 1.0), cpx!( 9.0, -1.0), cpx!( 12.0, 1.0)],
            [cpx!(-6.0, 1.0), cpx!(-9.0, -1.0), cpx!(-12.0, 2.0)],
        ]);
        complex_mat_scale(&mut a, cpx!(2.0, 3.0));
        #[rustfmt::skip]
        let correct = ComplexMatrix::from(&[
            [cpx!(  9.0,  20.0), cpx!( 21.0,  25.0), cpx!( 21.0,  38.0)],
            [cpx!(-15.0, -16.0), cpx!(-15.0, -29.0), cpx!(-30.0, -32.0)],
        ]);
        complex_mat_approx_eq(&a, &correct, 1e-15);
    }
}
