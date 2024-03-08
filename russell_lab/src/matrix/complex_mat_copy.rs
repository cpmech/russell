use super::ComplexMatrix;
use crate::{to_i32, StrError};
use num_complex::Complex64;

extern "C" {
    // Copies a vector into another
    // <https://www.netlib.org/lapack/explore-html/d6/d53/zcopy_8f.html>
    fn cblas_zcopy(n: i32, x: *const Complex64, incx: i32, y: *mut Complex64, incy: i32);
}

/// Copies complex matrix
///
/// ```text
/// b := a
/// ```
///
/// # Example
///
/// ```
/// use num_complex::Complex64;
/// use russell_lab::{cpx, complex_mat_copy, ComplexMatrix, StrError};
///
/// fn main() -> Result<(), StrError> {
///     let a = ComplexMatrix::from(&[
///         [cpx!(1.0, 1.0), cpx!(2.0, -1.0), cpx!(3.0, 1.0)],
///         [cpx!(4.0, 1.0), cpx!(5.0, -1.0), cpx!(6.0, 1.0)],
///     ]);
///     let mut b = ComplexMatrix::from(&[
///         [cpx!(-1.0, 1.0), cpx!(-2.0, 2.0), cpx!(-3.0, 3.0)],
///         [cpx!(-4.0, 4.0), cpx!(-5.0, 5.0), cpx!(-6.0, 6.0)],
///     ]);
///     complex_mat_copy(&mut b, &a)?;
///     let correct = "┌                ┐\n\
///                    │ 1+1i 2-1i 3+1i │\n\
///                    │ 4+1i 5-1i 6+1i │\n\
///                    └                ┘";
///     assert_eq!(format!("{}", b), correct);
///     Ok(())
/// }
/// ```
pub fn complex_mat_copy(b: &mut ComplexMatrix, a: &ComplexMatrix) -> Result<(), StrError> {
    let (m, n) = b.dims();
    if a.nrow() != m || a.ncol() != n {
        return Err("matrices are incompatible");
    }
    let n_i32: i32 = to_i32(m * n);
    unsafe {
        cblas_zcopy(n_i32, a.as_data().as_ptr(), 1, b.as_mut_data().as_mut_ptr(), 1);
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{complex_mat_copy, ComplexMatrix};
    use crate::{complex_mat_approx_eq, cpx};
    use num_complex::Complex64;

    #[test]
    fn complex_mat_copy_fails_on_wrong_dimensions() {
        let a_2x2 = ComplexMatrix::new(2, 2);
        let a_2x1 = ComplexMatrix::new(2, 1);
        let a_1x2 = ComplexMatrix::new(1, 2);
        let mut b_2x2 = ComplexMatrix::new(2, 2);
        let mut b_2x1 = ComplexMatrix::new(2, 1);
        let mut b_1x2 = ComplexMatrix::new(1, 2);
        assert_eq!(complex_mat_copy(&mut b_2x2, &a_2x1), Err("matrices are incompatible"));
        assert_eq!(complex_mat_copy(&mut b_2x2, &a_1x2), Err("matrices are incompatible"));
        assert_eq!(complex_mat_copy(&mut b_2x1, &a_2x2), Err("matrices are incompatible"));
        assert_eq!(complex_mat_copy(&mut b_1x2, &a_2x2), Err("matrices are incompatible"));
    }

    #[test]
    fn complex_mat_copy_works() {
        #[rustfmt::skip]
        let a = ComplexMatrix::from(&[
            [cpx!(10.0, 1.0), cpx!(20.0, 2.0), cpx!(30.0, 3.0)],
            [cpx!(40.0, 4.0), cpx!(50.0, 5.0), cpx!(60.0, 6.0)],
        ]);
        #[rustfmt::skip]
        let mut b = ComplexMatrix::from(&[
            [100.0, 200.0, 300.0],
            [400.0, 500.0, 600.0],
        ]);
        complex_mat_copy(&mut b, &a).unwrap();
        #[rustfmt::skip]
        let correct = &[
            [cpx!(10.0, 1.0), cpx!(20.0, 2.0), cpx!(30.0, 3.0)],
            [cpx!(40.0, 4.0), cpx!(50.0, 5.0), cpx!(60.0, 6.0)],
        ];
        complex_mat_approx_eq(&b, correct, 1e-15);
    }
}
