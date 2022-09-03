use super::ComplexMatrix;
use crate::StrError;
use crate::NATIVE_VERSUS_OPENBLAS_BOUNDARY;
use num_complex::Complex64;
use russell_openblas::{complex_add_vectors_native, complex_add_vectors_oblas};

/// Performs the addition of two matrices
///
/// ```text
/// c := α⋅a + β⋅b
/// ```
///
/// # Example
///
/// ```
/// use russell_lab::{complex_add_matrices, ComplexMatrix, StrError};
/// use num_complex::Complex64;
///
/// fn main() -> Result<(), StrError> {
///     let a = ComplexMatrix::from(&[
///         [ 10.0,  20.0,  30.0,  40.0],
///         [-10.0, -20.0, -30.0, -40.0],
///     ]);
///     let b = ComplexMatrix::from(&[
///         [ 2.0,  1.5,  1.0,  0.5],
///         [-2.0, -1.5, -1.0, -0.5],
///     ]);
///     let mut c = ComplexMatrix::new(2, 4);
///     let alpha = Complex64::new(0.1, 0.0);
///     let beta = Complex64::new(2.0, 0.0);
///     complex_add_matrices(&mut c, alpha, &a, beta, &b)?;
///     let correct = "┌                         ┐\n\
///                    │  5+0i  5+0i  5+0i  5+0i │\n\
///                    │ -5+0i -5+0i -5+0i -5+0i │\n\
///                    └                         ┘";
///     assert_eq!(format!("{}", c), correct);
///     Ok(())
/// }
/// ```
pub fn complex_add_matrices(
    c: &mut ComplexMatrix,
    alpha: Complex64,
    a: &ComplexMatrix,
    beta: Complex64,
    b: &ComplexMatrix,
) -> Result<(), StrError> {
    let (m, n) = c.dims();
    if a.nrow() != m || a.ncol() != n || b.nrow() != m || b.ncol() != n {
        return Err("matrices are incompatible");
    }
    if m == 0 && n == 0 {
        return Ok(());
    }
    if m * n > NATIVE_VERSUS_OPENBLAS_BOUNDARY {
        complex_add_vectors_oblas(c.as_mut_data(), alpha, a.as_data(), beta, b.as_data());
    } else {
        complex_add_vectors_native(c.as_mut_data(), alpha, a.as_data(), beta, b.as_data());
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{complex_add_matrices, ComplexMatrix};
    use crate::complex_mat_approx_eq;
    use crate::StrError;
    use num_complex::Complex64;

    #[test]
    fn complex_add_matrices_fail_on_wrong_dims() {
        let a_2x2 = ComplexMatrix::new(2, 2);
        let a_2x3 = ComplexMatrix::new(2, 3);
        let a_3x2 = ComplexMatrix::new(3, 2);
        let b_2x2 = ComplexMatrix::new(2, 2);
        let b_2x3 = ComplexMatrix::new(2, 3);
        let b_3x2 = ComplexMatrix::new(3, 2);
        let mut c_2x2 = ComplexMatrix::new(2, 2);
        let alpha = Complex64::new(1.0, 0.0);
        let beta = Complex64::new(1.0, 0.0);
        assert_eq!(
            complex_add_matrices(&mut c_2x2, alpha, &a_2x3, beta, &b_2x2),
            Err("matrices are incompatible")
        );
        assert_eq!(
            complex_add_matrices(&mut c_2x2, alpha, &a_3x2, beta, &b_2x2),
            Err("matrices are incompatible")
        );
        assert_eq!(
            complex_add_matrices(&mut c_2x2, alpha, &a_2x2, beta, &b_2x3),
            Err("matrices are incompatible")
        );
        assert_eq!(
            complex_add_matrices(&mut c_2x2, alpha, &a_2x2, beta, &b_3x2),
            Err("matrices are incompatible")
        );
    }

    #[test]
    fn complex_add_matrices_works() -> Result<(), StrError> {
        const NOISE: Complex64 = Complex64::new(1234.567, 3456.789);
        #[rustfmt::skip]
        let a = ComplexMatrix::from(&[
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
        ]);
        #[rustfmt::skip]
        let b = ComplexMatrix::from(&[
            [0.5, 1.0, 1.5, 2.0],
            [0.5, 1.0, 1.5, 2.0],
            [0.5, 1.0, 1.5, 2.0],
        ]);
        let mut c = ComplexMatrix::from(&[
            [NOISE, NOISE, NOISE, NOISE],
            [NOISE, NOISE, NOISE, NOISE],
            [NOISE, NOISE, NOISE, NOISE],
        ]);
        let alpha = Complex64::new(1.0, 0.0);
        let beta = Complex64::new(-4.0, 0.0);
        complex_add_matrices(&mut c, alpha, &a, beta, &b)?;
        #[rustfmt::skip]
        let correct = &[
            [Complex64::new(-1.0, 0.0), Complex64::new(-2.0, 0.0), Complex64::new(-3.0, 0.0), Complex64::new(-4.0, 0.0)],
            [Complex64::new(-1.0, 0.0), Complex64::new(-2.0, 0.0), Complex64::new(-3.0, 0.0), Complex64::new(-4.0, 0.0)],
            [Complex64::new(-1.0, 0.0), Complex64::new(-2.0, 0.0), Complex64::new(-3.0, 0.0), Complex64::new(-4.0, 0.0)],
        ];
        complex_mat_approx_eq(&c, correct, 1e-15);
        Ok(())
    }

    #[test]
    fn complex_add_matrix_oblas_works() -> Result<(), StrError> {
        const NOISE: Complex64 = Complex64::new(1234.567, 3456.789);
        let a = ComplexMatrix::from(&[
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        ]);
        let b = ComplexMatrix::from(&[
            [0.5, 1.0, 1.5, 2.0, 2.5],
            [0.5, 1.0, 1.5, 2.0, 2.5],
            [0.5, 1.0, 1.5, 2.0, 2.5],
            [0.5, 1.0, 1.5, 2.0, 2.5],
            [0.5, 1.0, 1.5, 2.0, 2.5],
        ]);
        let mut c = ComplexMatrix::from(&[
            [NOISE, NOISE, NOISE, NOISE, NOISE],
            [NOISE, NOISE, NOISE, NOISE, NOISE],
            [NOISE, NOISE, NOISE, NOISE, NOISE],
            [NOISE, NOISE, NOISE, NOISE, NOISE],
            [NOISE, NOISE, NOISE, NOISE, NOISE],
        ]);
        let alpha = Complex64::new(1.0, 0.0);
        let beta = Complex64::new(-4.0, 0.0);
        complex_add_matrices(&mut c, alpha, &a, beta, &b)?;
        #[rustfmt::skip]
        let correct = &[
            [Complex64::new(-1.0,0.0), Complex64::new(-2.0,0.0), Complex64::new(-3.0,0.0), Complex64::new(-4.0,0.0), Complex64::new(-5.0,0.0)],
            [Complex64::new(-1.0,0.0), Complex64::new(-2.0,0.0), Complex64::new(-3.0,0.0), Complex64::new(-4.0,0.0), Complex64::new(-5.0,0.0)],
            [Complex64::new(-1.0,0.0), Complex64::new(-2.0,0.0), Complex64::new(-3.0,0.0), Complex64::new(-4.0,0.0), Complex64::new(-5.0,0.0)],
            [Complex64::new(-1.0,0.0), Complex64::new(-2.0,0.0), Complex64::new(-3.0,0.0), Complex64::new(-4.0,0.0), Complex64::new(-5.0,0.0)],
            [Complex64::new(-1.0,0.0), Complex64::new(-2.0,0.0), Complex64::new(-3.0,0.0), Complex64::new(-4.0,0.0), Complex64::new(-5.0,0.0)],
        ];
        complex_mat_approx_eq(&c, correct, 1e-15);
        Ok(())
    }

    #[test]
    fn complex_add_matrices_skip() -> Result<(), StrError> {
        let a = ComplexMatrix::new(0, 0);
        let b = ComplexMatrix::new(0, 0);
        let mut c = ComplexMatrix::new(0, 0);
        let alpha = Complex64::new(1.0, 0.0);
        let beta = Complex64::new(1.0, 0.0);
        complex_add_matrices(&mut c, alpha, &a, beta, &b)?;
        assert_eq!(c.as_data().len(), 0);
        Ok(())
    }
}
