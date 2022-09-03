use super::Matrix;
use crate::StrError;
use crate::NATIVE_VERSUS_OPENBLAS_BOUNDARY;
use russell_openblas::{add_vectors_native, add_vectors_oblas};

/// Performs the addition of two matrices
///
/// ```text
/// c := α⋅a + β⋅b
/// ```
///
/// # Example
///
/// ```
/// use russell_lab::{add_matrices, Matrix, StrError};
///
/// fn main() -> Result<(), StrError> {
///     let a = Matrix::from(&[
///         [ 10.0,  20.0,  30.0,  40.0],
///         [-10.0, -20.0, -30.0, -40.0],
///     ]);
///     let b = Matrix::from(&[
///         [ 2.0,  1.5,  1.0,  0.5],
///         [-2.0, -1.5, -1.0, -0.5],
///     ]);
///     let mut c = Matrix::new(2, 4);
///     add_matrices(&mut c, 0.1, &a, 2.0, &b)?;
///     let correct = "┌             ┐\n\
///                    │  5  5  5  5 │\n\
///                    │ -5 -5 -5 -5 │\n\
///                    └             ┘";
///     assert_eq!(format!("{}", c), correct);
///     Ok(())
/// }
/// ```
pub fn add_matrices(c: &mut Matrix, alpha: f64, a: &Matrix, beta: f64, b: &Matrix) -> Result<(), StrError> {
    let (m, n) = c.dims();
    if a.nrow() != m || a.ncol() != n || b.nrow() != m || b.ncol() != n {
        return Err("matrices are incompatible");
    }
    if m == 0 && n == 0 {
        return Ok(());
    }
    if m * n > NATIVE_VERSUS_OPENBLAS_BOUNDARY {
        add_vectors_oblas(c.as_mut_data(), alpha, a.as_data(), beta, b.as_data());
    } else {
        add_vectors_native(c.as_mut_data(), alpha, a.as_data(), beta, b.as_data());
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{add_matrices, Matrix};
    use crate::mat_approx_eq;
    use crate::StrError;

    #[test]
    fn add_matrices_fail_on_wrong_dims() {
        let a_2x2 = Matrix::new(2, 2);
        let a_2x3 = Matrix::new(2, 3);
        let a_3x2 = Matrix::new(3, 2);
        let b_2x2 = Matrix::new(2, 2);
        let b_2x3 = Matrix::new(2, 3);
        let b_3x2 = Matrix::new(3, 2);
        let mut c_2x2 = Matrix::new(2, 2);
        assert_eq!(
            add_matrices(&mut c_2x2, 1.0, &a_2x3, 1.0, &b_2x2),
            Err("matrices are incompatible")
        );
        assert_eq!(
            add_matrices(&mut c_2x2, 1.0, &a_3x2, 1.0, &b_2x2),
            Err("matrices are incompatible")
        );
        assert_eq!(
            add_matrices(&mut c_2x2, 1.0, &a_2x2, 1.0, &b_2x3),
            Err("matrices are incompatible")
        );
        assert_eq!(
            add_matrices(&mut c_2x2, 1.0, &a_2x2, 1.0, &b_3x2),
            Err("matrices are incompatible")
        );
    }

    #[test]
    fn add_matrices_works() -> Result<(), StrError> {
        const NOISE: f64 = 1234.567;
        #[rustfmt::skip]
        let a = Matrix::from(&[
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
        ]);
        #[rustfmt::skip]
        let b = Matrix::from(&[
            [0.5, 1.0, 1.5, 2.0],
            [0.5, 1.0, 1.5, 2.0],
            [0.5, 1.0, 1.5, 2.0],
        ]);
        let mut c = Matrix::from(&[
            [NOISE, NOISE, NOISE, NOISE],
            [NOISE, NOISE, NOISE, NOISE],
            [NOISE, NOISE, NOISE, NOISE],
        ]);
        add_matrices(&mut c, 1.0, &a, -4.0, &b)?;
        #[rustfmt::skip]
        let correct = &[
            [-1.0, -2.0, -3.0, -4.0],
            [-1.0, -2.0, -3.0, -4.0],
            [-1.0, -2.0, -3.0, -4.0],
        ];
        mat_approx_eq(&c, correct, 1e-15);
        Ok(())
    }

    #[test]
    fn add_matrix_oblas_works() -> Result<(), StrError> {
        const NOISE: f64 = 1234.567;
        let a = Matrix::from(&[
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        ]);
        let b = Matrix::from(&[
            [0.5, 1.0, 1.5, 2.0, 2.5],
            [0.5, 1.0, 1.5, 2.0, 2.5],
            [0.5, 1.0, 1.5, 2.0, 2.5],
            [0.5, 1.0, 1.5, 2.0, 2.5],
            [0.5, 1.0, 1.5, 2.0, 2.5],
        ]);
        let mut c = Matrix::from(&[
            [NOISE, NOISE, NOISE, NOISE, NOISE],
            [NOISE, NOISE, NOISE, NOISE, NOISE],
            [NOISE, NOISE, NOISE, NOISE, NOISE],
            [NOISE, NOISE, NOISE, NOISE, NOISE],
            [NOISE, NOISE, NOISE, NOISE, NOISE],
        ]);
        add_matrices(&mut c, 1.0, &a, -4.0, &b)?;
        #[rustfmt::skip]
        let correct = &[
            [-1.0, -2.0, -3.0, -4.0, -5.0],
            [-1.0, -2.0, -3.0, -4.0, -5.0],
            [-1.0, -2.0, -3.0, -4.0, -5.0],
            [-1.0, -2.0, -3.0, -4.0, -5.0],
            [-1.0, -2.0, -3.0, -4.0, -5.0],
        ];
        mat_approx_eq(&c, correct, 1e-15);
        Ok(())
    }

    #[test]
    fn add_matrices_skip() -> Result<(), StrError> {
        let a = Matrix::new(0, 0);
        let b = Matrix::new(0, 0);
        let mut c = Matrix::new(0, 0);
        add_matrices(&mut c, 1.0, &a, 1.0, &b)?;
        assert_eq!(a.as_data().len(), 0);
        Ok(())
    }
}
