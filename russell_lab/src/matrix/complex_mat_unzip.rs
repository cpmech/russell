use crate::ComplexMatrix;
use crate::Matrix;
use crate::StrError;

/// Zips two arrays (real and imag) to make a new ComplexMatrix
///
/// # Example
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
pub fn complex_mat_unzip(real: &mut Matrix, imag: &mut Matrix, a: &ComplexMatrix) -> Result<(), StrError> {
    let (nrow, ncol) = a.dims();
    let (nrow_re, ncol_re) = real.dims();
    let (nrow_im, ncol_im) = imag.dims();
    if nrow_re != nrow || ncol_re != ncol || nrow_im != nrow || ncol_im != ncol {
        return Err("matrices are incompatible");
    }
    for i in 0..nrow {
        for j in 0..ncol {
            real.set(i, j, a.get(i, j).re);
            imag.set(i, j, a.get(i, j).im);
        }
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::complex_mat_unzip;
    use crate::{cpx, mat_approx_eq, ComplexMatrix, Matrix};
    use num_complex::Complex64;

    #[test]
    fn complex_mat_unzip_handles_errors() {
        let a = ComplexMatrix::new(2, 2);
        let mut wrong_1x2 = Matrix::new(1, 2);
        let mut wrong_2x1 = Matrix::new(2, 1);
        let mut ok = Matrix::new(2, 2);
        assert_eq!(
            complex_mat_unzip(&mut wrong_1x2, &mut ok, &a).err(),
            Some("matrices are incompatible")
        );
        assert_eq!(
            complex_mat_unzip(&mut wrong_2x1, &mut ok, &a).err(),
            Some("matrices are incompatible")
        );
        assert_eq!(
            complex_mat_unzip(&mut ok, &mut wrong_1x2, &a).err(),
            Some("matrices are incompatible")
        );
        assert_eq!(
            complex_mat_unzip(&mut ok, &mut wrong_2x1, &a).err(),
            Some("matrices are incompatible")
        );
    }

    #[test]
    fn complex_mat_unzip_works() {
        let a = ComplexMatrix::from(&[
            [cpx!(1.0, 0.1), cpx!(2.0, 0.2), cpx!(3.0, 0.3)],
            [cpx!(4.0, 0.4), cpx!(5.0, 0.5), cpx!(6.0, 0.6)],
        ]);
        let mut real = Matrix::new(2, 3);
        let mut imag = Matrix::new(2, 3);
        complex_mat_unzip(&mut real, &mut imag, &a).unwrap();
        mat_approx_eq(&real, &[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], 1e-15);
        mat_approx_eq(&imag, &[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], 1e-15);
    }
}
