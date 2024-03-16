use super::Matrix;
use crate::to_i32;

extern "C" {
    // Scales a vector by a constant
    // <https://www.netlib.org/lapack/explore-html/d4/dd0/dscal_8f.html>
    fn cblas_dscal(n: i32, alpha: f64, x: *const f64, incx: i32);
}

/// (dscal) Scales matrix
///
/// ```text
/// a := alpha * a
/// ```
///
/// See also <https://www.netlib.org/lapack/explore-html/d4/dd0/dscal_8f.html>
///
/// # Example
///
/// ```
/// use russell_lab::{mat_scale, Matrix};
///
/// fn main() {
///     let mut a = Matrix::from(&[
///         [1.0, 2.0, 3.0],
///         [4.0, 5.0, 6.0],
///     ]);
///
///     mat_scale(&mut a, 0.5);
///
///     let correct = "┌             ┐\n\
///                    │ 0.5   1 1.5 │\n\
///                    │   2 2.5   3 │\n\
///                    └             ┘";
///
///     assert_eq!(format!("{}", a), correct);
/// }
/// ```
pub fn mat_scale(a: &mut Matrix, alpha: f64) {
    let (m, n) = a.dims();
    let mn_i32 = to_i32(m * n);
    unsafe {
        cblas_dscal(mn_i32, alpha, a.as_mut_data().as_mut_ptr(), 1);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{mat_scale, Matrix};
    use crate::mat_approx_eq;

    #[test]
    fn mat_scale_works() {
        #[rustfmt::skip]
        let mut a = Matrix::from(&[
            [ 6.0,  9.0,  12.0],
            [-6.0, -9.0, -12.0],
        ]);
        mat_scale(&mut a, 1.0 / 3.0);
        #[rustfmt::skip]
        let correct = &[
            [ 2.0,  3.0,  4.0],
            [-2.0, -3.0, -4.0],
        ];
        mat_approx_eq(&a, correct, 1e-15);
    }
}
