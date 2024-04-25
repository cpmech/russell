use super::ComplexMatrix;
use crate::{to_i32, Complex64, StrError};

extern "C" {
    // Computes constant times a vector plus a vector
    // <https://www.netlib.org/lapack/explore-html/d7/db2/zaxpy_8f.html>
    fn cblas_zaxpy(n: i32, alpha: *const Complex64, x: *const Complex64, incx: i32, y: *mut Complex64, incy: i32);
}

/// (zaxpy) Updates matrix based on another matrix
///
/// ```text
/// b += α⋅a
/// ```
///
/// See also: <https://www.netlib.org/lapack/explore-html/d7/db2/zaxpy_8f.html>
///
/// # Examples
///
/// ```
/// use russell_lab::*;
///
/// fn main() -> Result<(), StrError> {
///     let a = ComplexMatrix::from(&[
///         [10.0, 20.0, 30.0],
///         [40.0, 50.0, 60.0],
///     ]);
///     let mut b = ComplexMatrix::from(&[
///         [10.0, 20.0, 30.0],
///         [40.0, 50.0, 60.0],
///     ]);
///     complex_mat_update(&mut b, cpx!(0.1, 0.0), &a)?;
///     let correct = "┌                   ┐\n\
///                    │ 11+0i 22+0i 33+0i │\n\
///                    │ 44+0i 55+0i 66+0i │\n\
///                    └                   ┘";
///     assert_eq!(format!("{}", b), correct);
///     Ok(())
/// }
/// ```
pub fn complex_mat_update(b: &mut ComplexMatrix, alpha: Complex64, a: &ComplexMatrix) -> Result<(), StrError> {
    let (m, n) = b.dims();
    if a.nrow() != m || a.ncol() != n {
        return Err("matrices are incompatible");
    }
    let mn_i32 = to_i32(m * n);
    unsafe {
        cblas_zaxpy(mn_i32, &alpha, a.as_data().as_ptr(), 1, b.as_mut_data().as_mut_ptr(), 1);
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::complex_mat_update;
    use crate::{complex_mat_approx_eq, cpx, Complex64, ComplexMatrix};

    #[test]
    fn complex_mat_update_fail_on_wrong_dims() {
        let a_2x2 = ComplexMatrix::new(2, 2);
        let a_2x1 = ComplexMatrix::new(2, 1);
        let a_1x2 = ComplexMatrix::new(1, 2);
        let mut b_2x2 = ComplexMatrix::new(2, 2);
        let mut b_2x1 = ComplexMatrix::new(2, 1);
        let mut b_1x2 = ComplexMatrix::new(1, 2);
        assert_eq!(
            complex_mat_update(&mut b_2x2, cpx!(2.0, 0.0), &a_2x1),
            Err("matrices are incompatible")
        );
        assert_eq!(
            complex_mat_update(&mut b_2x2, cpx!(2.0, 0.0), &a_1x2),
            Err("matrices are incompatible")
        );
        assert_eq!(
            complex_mat_update(&mut b_2x1, cpx!(2.0, 0.0), &a_2x2),
            Err("matrices are incompatible")
        );
        assert_eq!(
            complex_mat_update(&mut b_1x2, cpx!(2.0, 0.0), &a_2x2),
            Err("matrices are incompatible")
        );
    }

    #[test]
    fn complex_mat_update_works() {
        // real only
        #[rustfmt::skip]
        let a = ComplexMatrix::from(&[
            [cpx!(10.0, 0.0), cpx!(20.0, 0.0), cpx!(30.0, 0.0)],
            [cpx!(40.0, 0.0), cpx!(50.0, 0.0), cpx!(60.0, 0.0)],
        ]);
        #[rustfmt::skip]
        let mut b = ComplexMatrix::from(&[
            [cpx!(100.0, 0.0), cpx!(200.0, 0.0), cpx!(300.0, 0.0)],
            [cpx!(400.0, 0.0), cpx!(500.0, 0.0), cpx!(600.0, 0.0)],
        ]);
        complex_mat_update(&mut b, cpx!(2.0, 0.0), &a).unwrap();
        #[rustfmt::skip]
        let correct = &[
            [cpx!(120.0, 0.0), cpx!(240.0, 0.0), cpx!(360.0, 0.0)],
            [cpx!(480.0, 0.0), cpx!(600.0, 0.0), cpx!(720.0, 0.0)],
        ];
        complex_mat_approx_eq(&b, correct, 1e-15);

        // real and imag
        #[rustfmt::skip]
        let a = ComplexMatrix::from(&[
            [cpx!(10.0, 1.0), cpx!(20.0, 2.0), cpx!(30.0, 3.0)],
            [cpx!(40.0, 4.0), cpx!(50.0, 5.0), cpx!(60.0, 6.0)],
        ]);
        #[rustfmt::skip]
        let mut b = ComplexMatrix::from(&[
            [cpx!(100.0, -1.0), cpx!(200.0, 1.0), cpx!(300.0, -1.0)],
            [cpx!(400.0, -1.0), cpx!(500.0, 1.0), cpx!(600.0, -1.0)],
        ]);
        complex_mat_update(&mut b, cpx!(2.0, 1.0), &a).unwrap();
        #[rustfmt::skip]
        let correct = &[
            [cpx!(119.0, 11.0), cpx!(238.0, 25.0), cpx!(357.0, 35.0)],
            [cpx!(476.0, 47.0), cpx!(595.0, 61.0), cpx!(714.0, 71.0)],
        ];
        complex_mat_approx_eq(&b, correct, 1e-15);
    }
}
