use crate::ComplexMatrix;
use crate::Matrix;
use crate::StrError;
use num_complex::Complex64;

/// Zips two arrays (real and imag) to make a new ComplexMatrix
///
/// # Examples
///
/// ```
/// use russell_lab::*;
///
/// fn main() -> Result<(), StrError> {
///     let mut a = ComplexMatrix::new(2, 3);
///     let real = Matrix::from(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
///     let imag = Matrix::from(&[[0.1, 0.2, 0.3], [-0.4, -0.5, -0.6]]);
///     complex_mat_zip(&mut a, &real, &imag)?;
///     assert_eq!(
///         format!("{}", a),
///         "┌                      ┐\n\
///          │ 1+0.1i 2+0.2i 3+0.3i │\n\
///          │ 4-0.4i 5-0.5i 6-0.6i │\n\
///          └                      ┘"
///     );
///     Ok(())
/// }
/// ```
pub fn complex_mat_zip(a: &mut ComplexMatrix, real: &Matrix, imag: &Matrix) -> Result<(), StrError> {
    let (nrow, ncol) = a.dims();
    let (nrow_re, ncol_re) = real.dims();
    let (nrow_im, ncol_im) = imag.dims();
    if nrow_re != nrow || ncol_re != ncol || nrow_im != nrow || ncol_im != ncol {
        return Err("matrices are incompatible");
    }
    for i in 0..nrow {
        for j in 0..ncol {
            a.set(i, j, Complex64::new(real.get(i, j), imag.get(i, j)));
        }
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::complex_mat_zip;
    use crate::{ComplexMatrix, Matrix};

    #[test]
    fn complex_mat_zip_handles_errors() {
        let mut a = ComplexMatrix::new(2, 2);
        let wrong_1x2 = Matrix::new(1, 2);
        let wrong_2x1 = Matrix::new(2, 1);
        let ok = Matrix::new(2, 2);
        assert_eq!(
            complex_mat_zip(&mut a, &wrong_1x2, &ok).err(),
            Some("matrices are incompatible")
        );
        assert_eq!(
            complex_mat_zip(&mut a, &wrong_2x1, &ok).err(),
            Some("matrices are incompatible")
        );
        assert_eq!(
            complex_mat_zip(&mut a, &ok, &wrong_1x2).err(),
            Some("matrices are incompatible")
        );
        assert_eq!(
            complex_mat_zip(&mut a, &ok, &wrong_2x1).err(),
            Some("matrices are incompatible")
        );
    }

    #[test]
    fn complex_mat_zip_works() {
        let real = Matrix::from(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let imag = Matrix::from(&[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]);
        let mut c = ComplexMatrix::new(2, 3);
        complex_mat_zip(&mut c, &real, &imag).unwrap();
        assert_eq!(
            format!("{}", c),
            "┌                      ┐\n\
             │ 1+0.1i 2+0.2i 3+0.3i │\n\
             │ 4+0.4i 5+0.5i 6+0.6i │\n\
             └                      ┘"
        );
    }
}
