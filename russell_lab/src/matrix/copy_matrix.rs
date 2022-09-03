use super::Matrix;
use crate::StrError;
use russell_openblas::{dcopy, to_i32};

/// Copies matrix
///
/// ```text
/// b := a
/// ```
///
/// # Example
///
/// ```
/// use russell_lab::{copy_matrix, Matrix, StrError};
///
/// fn main() -> Result<(), StrError> {
///     let a = Matrix::from(&[
///         [1.0, 2.0, 3.0],
///         [4.0, 5.0, 6.0],
///     ]);
///     let mut b = Matrix::from(&[
///         [-1.0, -2.0, -3.0],
///         [-4.0, -5.0, -6.0],
///     ]);
///     copy_matrix(&mut b, &a)?;
///     let correct = "┌       ┐\n\
///                    │ 1 2 3 │\n\
///                    │ 4 5 6 │\n\
///                    └       ┘";
///     assert_eq!(format!("{}", b), correct);
///     Ok(())
/// }
/// ```
pub fn copy_matrix(b: &mut Matrix, a: &Matrix) -> Result<(), StrError> {
    let (m, n) = b.dims();
    if a.nrow() != m || a.ncol() != n {
        return Err("matrices are incompatible");
    }
    let n_i32: i32 = to_i32(m * n);
    dcopy(n_i32, a.as_data(), 1, b.as_mut_data(), 1);
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{copy_matrix, Matrix};
    use crate::mat_approx_eq;
    use crate::StrError;

    #[test]
    fn copy_matrix_fails_on_wrong_dimensions() {
        let a_2x2 = Matrix::new(2, 2);
        let a_2x1 = Matrix::new(2, 1);
        let a_1x2 = Matrix::new(1, 2);
        let mut b_2x2 = Matrix::new(2, 2);
        let mut b_2x1 = Matrix::new(2, 1);
        let mut b_1x2 = Matrix::new(1, 2);
        assert_eq!(copy_matrix(&mut b_2x2, &a_2x1), Err("matrices are incompatible"));
        assert_eq!(copy_matrix(&mut b_2x2, &a_1x2), Err("matrices are incompatible"));
        assert_eq!(copy_matrix(&mut b_2x1, &a_2x2), Err("matrices are incompatible"));
        assert_eq!(copy_matrix(&mut b_1x2, &a_2x2), Err("matrices are incompatible"));
    }

    #[test]
    fn copy_matrix_works() -> Result<(), StrError> {
        #[rustfmt::skip]
        let a = Matrix::from(&[
            [10.0, 20.0, 30.0],
            [40.0, 50.0, 60.0],
        ]);
        #[rustfmt::skip]
        let mut b = Matrix::from(&[
            [100.0, 200.0, 300.0],
            [400.0, 500.0, 600.0],
        ]);
        copy_matrix(&mut b, &a)?;
        #[rustfmt::skip]
        let correct = &[
            [10.0, 20.0, 30.0],
            [40.0, 50.0, 60.0],
        ];
        mat_approx_eq(&b, correct, 1e-15);
        Ok(())
    }
}
