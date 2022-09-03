use super::Matrix;
use russell_openblas::{dscal, to_i32};

/// Scales matrix
///
/// ```text
/// a := alpha * a
/// ```
///
/// # Example
///
/// ```
/// use russell_lab::{scale_matrix, Matrix};
///
/// fn main() {
///     let mut a = Matrix::from(&[
///         [1.0, 2.0, 3.0],
///         [4.0, 5.0, 6.0],
///     ]);
///
///     scale_matrix(&mut a, 0.5);
///
///     let correct = "┌             ┐\n\
///                    │ 0.5   1 1.5 │\n\
///                    │   2 2.5   3 │\n\
///                    └             ┘";
///
///     assert_eq!(format!("{}", a), correct);
/// }
/// ```
pub fn scale_matrix(a: &mut Matrix, alpha: f64) {
    let mut data = a.as_mut_data();
    let n: i32 = to_i32(data.len());
    dscal(n, alpha, &mut data, 1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{scale_matrix, Matrix};
    use crate::mat_approx_eq;

    #[test]
    fn scale_matrix_works() {
        #[rustfmt::skip]
        let mut a = Matrix::from(&[
            [ 6.0,  9.0,  12.0],
            [-6.0, -9.0, -12.0],
        ]);
        scale_matrix(&mut a, 1.0 / 3.0);
        #[rustfmt::skip]
        let correct = &[
            [ 2.0,  3.0,  4.0],
            [-2.0, -3.0, -4.0],
        ];
        mat_approx_eq(&a, correct, 1e-15);
    }
}
