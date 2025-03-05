use super::ComplexMatrix;
use crate::{array_minus_op_complex, StrError};

/// Performs the subtraction of two matrices
///
/// ```text
/// c := a - b
/// ```
///
/// # Examples
///
/// ```
/// use russell_lab::*;
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
///     complex_mat_minus(&mut c, &a, &b)?;
///     let correct = "┌                                     ┐\n\
///                    │     8+0i  18.5+0i    29+0i  39.5+0i │\n\
///                    │    -8+0i -18.5+0i   -29+0i -39.5+0i │\n\
///                    └                                     ┘";
///     assert_eq!(format!("{}", c), correct);
///     Ok(())
/// }
/// ```
pub fn complex_mat_minus(c: &mut ComplexMatrix, a: &ComplexMatrix, b: &ComplexMatrix) -> Result<(), StrError> {
    let (m, n) = c.dims();
    if a.nrow() != m || a.ncol() != n || b.nrow() != m || b.ncol() != n {
        return Err("matrices are incompatible");
    }
    array_minus_op_complex(c.as_mut_data(), a.as_data(), b.as_data())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{complex_mat_minus, ComplexMatrix};
    use crate::{complex_mat_approx_eq, cpx, Complex64};

    #[test]
    fn complex_mat_minus_fails_on_wrong_dims() {
        let a_2x2 = ComplexMatrix::new(2, 2);
        let a_2x3 = ComplexMatrix::new(2, 3);
        let a_3x2 = ComplexMatrix::new(3, 2);
        let b_2x2 = ComplexMatrix::new(2, 2);
        let b_2x3 = ComplexMatrix::new(2, 3);
        let b_3x2 = ComplexMatrix::new(3, 2);
        let mut c_2x2 = ComplexMatrix::new(2, 2);
        assert_eq!(
            complex_mat_minus(&mut c_2x2, &a_2x3, &b_2x2),
            Err("matrices are incompatible")
        );
        assert_eq!(
            complex_mat_minus(&mut c_2x2, &a_3x2, &b_2x2),
            Err("matrices are incompatible")
        );
        assert_eq!(
            complex_mat_minus(&mut c_2x2, &a_2x2, &b_2x3),
            Err("matrices are incompatible")
        );
        assert_eq!(
            complex_mat_minus(&mut c_2x2, &a_2x2, &b_3x2),
            Err("matrices are incompatible")
        );
    }

    #[test]
    fn complex_mat_minus_works() {
        const NOISE: Complex64 = cpx!(1234.567, 3456.789);
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
        complex_mat_minus(&mut c, &a, &b).unwrap();
        #[rustfmt::skip]
        let correct = &[
            [cpx!(0.5, 0.0), cpx!(1.0, 0.0), cpx!(1.5, 0.0), cpx!(2.0, 0.0)],
            [cpx!(0.5, 0.0), cpx!(1.0, 0.0), cpx!(1.5, 0.0), cpx!(2.0, 0.0)],
            [cpx!(0.5, 0.0), cpx!(1.0, 0.0), cpx!(1.5, 0.0), cpx!(2.0, 0.0)],
        ];
        complex_mat_approx_eq(&c, correct, 1e-15);
    }

    #[test]
    fn complex_add_matrix_5x5_works() {
        const NOISE: Complex64 = cpx!(1234.567, 3456.789);
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
        complex_mat_minus(&mut c, &a, &b).unwrap();
        #[rustfmt::skip]
        let correct = &[
            [cpx!(0.5,0.0), cpx!(1.0,0.0), cpx!(1.5,0.0), cpx!(2.0,0.0), cpx!(2.5,0.0)],
            [cpx!(0.5,0.0), cpx!(1.0,0.0), cpx!(1.5,0.0), cpx!(2.0,0.0), cpx!(2.5,0.0)],
            [cpx!(0.5,0.0), cpx!(1.0,0.0), cpx!(1.5,0.0), cpx!(2.0,0.0), cpx!(2.5,0.0)],
            [cpx!(0.5,0.0), cpx!(1.0,0.0), cpx!(1.5,0.0), cpx!(2.0,0.0), cpx!(2.5,0.0)],
            [cpx!(0.5,0.0), cpx!(1.0,0.0), cpx!(1.5,0.0), cpx!(2.0,0.0), cpx!(2.5,0.0)],
        ];
        complex_mat_approx_eq(&c, correct, 1e-15);
    }

    #[test]
    fn complex_mat_minus_skip() {
        let a = ComplexMatrix::new(0, 0);
        let b = ComplexMatrix::new(0, 0);
        let mut c = ComplexMatrix::new(0, 0);
        complex_mat_minus(&mut c, &a, &b).unwrap();
        assert_eq!(c.as_data().len(), 0);
    }
}
