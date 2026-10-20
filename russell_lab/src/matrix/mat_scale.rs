use super::Matrix;

/// (dscal) Scales matrix
///
/// ```text
/// a := alpha * a
/// ```
///
/// See also <https://www.netlib.org/lapack/explore-html/d4/dd0/dscal_8f.html>
///
/// # Examples
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
    crate::base::blas_dispatch::dscal(alpha, a.as_mut_data());
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{Matrix, mat_scale};
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
