use super::ComplexMatrix;
use crate::{array_plus_opx_complex, Complex64, StrError};

/// Performs the addition of two matrices
///
/// ```text
/// c := α⋅a + β⋅b
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
///     let alpha = cpx!(0.1, 0.0);
///     let beta = cpx!(2.0, 0.0);
///     complex_mat_add(&mut c, alpha, &a, beta, &b)?;
///     let correct = "┌                         ┐\n\
///                    │  5+0i  5+0i  5+0i  5+0i │\n\
///                    │ -5+0i -5+0i -5+0i -5+0i │\n\
///                    └                         ┘";
///     assert_eq!(format!("{}", c), correct);
///     Ok(())
/// }
/// ```
pub fn complex_mat_add(
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
    array_plus_opx_complex(c.as_mut_data(), alpha, a.as_data(), beta, b.as_data())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{complex_mat_add, ComplexMatrix};
    use crate::{complex_mat_approx_eq, cpx, Complex64};

    #[test]
    fn complex_mat_add_fails_on_wrong_dims() {
        let a_2x2 = ComplexMatrix::new(2, 2);
        let a_2x3 = ComplexMatrix::new(2, 3);
        let a_3x2 = ComplexMatrix::new(3, 2);
        let b_2x2 = ComplexMatrix::new(2, 2);
        let b_2x3 = ComplexMatrix::new(2, 3);
        let b_3x2 = ComplexMatrix::new(3, 2);
        let mut c_2x2 = ComplexMatrix::new(2, 2);
        let alpha = cpx!(1.0, 0.0);
        let beta = cpx!(1.0, 0.0);
        assert_eq!(
            complex_mat_add(&mut c_2x2, alpha, &a_2x3, beta, &b_2x2),
            Err("matrices are incompatible")
        );
        assert_eq!(
            complex_mat_add(&mut c_2x2, alpha, &a_3x2, beta, &b_2x2),
            Err("matrices are incompatible")
        );
        assert_eq!(
            complex_mat_add(&mut c_2x2, alpha, &a_2x2, beta, &b_2x3),
            Err("matrices are incompatible")
        );
        assert_eq!(
            complex_mat_add(&mut c_2x2, alpha, &a_2x2, beta, &b_3x2),
            Err("matrices are incompatible")
        );
    }

    #[test]
    fn complex_mat_add_works() {
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
        let alpha = cpx!(1.0, 0.0);
        let beta = cpx!(-4.0, 0.0);
        complex_mat_add(&mut c, alpha, &a, beta, &b).unwrap();
        #[rustfmt::skip]
        let correct = &[
            [cpx!(-1.0, 0.0), cpx!(-2.0, 0.0), cpx!(-3.0, 0.0), cpx!(-4.0, 0.0)],
            [cpx!(-1.0, 0.0), cpx!(-2.0, 0.0), cpx!(-3.0, 0.0), cpx!(-4.0, 0.0)],
            [cpx!(-1.0, 0.0), cpx!(-2.0, 0.0), cpx!(-3.0, 0.0), cpx!(-4.0, 0.0)],
        ];
        complex_mat_approx_eq(&c, correct, 1e-15);
    }

    #[test]
    fn complex_add_matrix_oblas_works() {
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
        let alpha = cpx!(1.0, 0.0);
        let beta = cpx!(-4.0, 0.0);
        complex_mat_add(&mut c, alpha, &a, beta, &b).unwrap();
        #[rustfmt::skip]
        let correct = &[
            [cpx!(-1.0,0.0), cpx!(-2.0,0.0), cpx!(-3.0,0.0), cpx!(-4.0,0.0), cpx!(-5.0,0.0)],
            [cpx!(-1.0,0.0), cpx!(-2.0,0.0), cpx!(-3.0,0.0), cpx!(-4.0,0.0), cpx!(-5.0,0.0)],
            [cpx!(-1.0,0.0), cpx!(-2.0,0.0), cpx!(-3.0,0.0), cpx!(-4.0,0.0), cpx!(-5.0,0.0)],
            [cpx!(-1.0,0.0), cpx!(-2.0,0.0), cpx!(-3.0,0.0), cpx!(-4.0,0.0), cpx!(-5.0,0.0)],
            [cpx!(-1.0,0.0), cpx!(-2.0,0.0), cpx!(-3.0,0.0), cpx!(-4.0,0.0), cpx!(-5.0,0.0)],
        ];
        complex_mat_approx_eq(&c, correct, 1e-15);
    }

    #[test]
    fn complex_mat_add_skip() {
        let a = ComplexMatrix::new(0, 0);
        let b = ComplexMatrix::new(0, 0);
        let mut c = ComplexMatrix::new(0, 0);
        let alpha = cpx!(1.0, 0.0);
        let beta = cpx!(1.0, 0.0);
        complex_mat_add(&mut c, alpha, &a, beta, &b).unwrap();
        assert_eq!(c.as_data().len(), 0);
    }
}
