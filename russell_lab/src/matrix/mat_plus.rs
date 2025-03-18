use super::Matrix;
use crate::{array_plus_op, StrError};

/// Performs the addition of two matrices
///
/// ```text
/// c := a + b
/// ```
///
/// # Examples
///
/// ```
/// use russell_lab::{mat_plus, Matrix, StrError};
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
///     mat_plus(&mut c, &a, &b)?;
///     let correct = "┌                         ┐\n\
///                    │    12  21.5    31  40.5 │\n\
///                    │   -12 -21.5   -31 -40.5 │\n\
///                    └                         ┘";
///     assert_eq!(format!("{}", c), correct);
///     Ok(())
/// }
/// ```
pub fn mat_plus(c: &mut Matrix, a: &Matrix, b: &Matrix) -> Result<(), StrError> {
    let (m, n) = c.dims();
    if a.nrow() != m || a.ncol() != n || b.nrow() != m || b.ncol() != n {
        return Err("matrices are incompatible");
    }
    array_plus_op(c.as_mut_data(), a.as_data(), b.as_data())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{mat_plus, Matrix};
    use crate::mat_approx_eq;

    #[test]
    fn mat_plus_fail_on_wrong_dims() {
        let a_2x2 = Matrix::new(2, 2);
        let a_2x3 = Matrix::new(2, 3);
        let a_3x2 = Matrix::new(3, 2);
        let b_2x2 = Matrix::new(2, 2);
        let b_2x3 = Matrix::new(2, 3);
        let b_3x2 = Matrix::new(3, 2);
        let mut c_2x2 = Matrix::new(2, 2);
        assert_eq!(mat_plus(&mut c_2x2, &a_2x3, &b_2x2), Err("matrices are incompatible"));
        assert_eq!(mat_plus(&mut c_2x2, &a_3x2, &b_2x2), Err("matrices are incompatible"));
        assert_eq!(mat_plus(&mut c_2x2, &a_2x2, &b_2x3), Err("matrices are incompatible"));
        assert_eq!(mat_plus(&mut c_2x2, &a_2x2, &b_3x2), Err("matrices are incompatible"));
    }

    #[test]
    fn mat_plus_works() {
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
        mat_plus(&mut c, &a, &b).unwrap();
        #[rustfmt::skip]
        let correct = &[
            [1.5, 3.0, 4.5, 6.0],
            [1.5, 3.0, 4.5, 6.0],
            [1.5, 3.0, 4.5, 6.0],
        ];
        mat_approx_eq(&c, correct, 1e-15);
    }

    #[test]
    fn add_matrix_5x5_works() {
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
        mat_plus(&mut c, &a, &b).unwrap();
        #[rustfmt::skip]
        let correct = &[
            [1.5, 3.0, 4.5, 6.0, 7.5],
            [1.5, 3.0, 4.5, 6.0, 7.5],
            [1.5, 3.0, 4.5, 6.0, 7.5],
            [1.5, 3.0, 4.5, 6.0, 7.5],
            [1.5, 3.0, 4.5, 6.0, 7.5],
        ];
        mat_approx_eq(&c, correct, 1e-15);
    }

    #[test]
    fn mat_plus_skip() {
        let a = Matrix::new(0, 0);
        let b = Matrix::new(0, 0);
        let mut c = Matrix::new(0, 0);
        mat_plus(&mut c, &a, &b).unwrap();
        assert_eq!(a.as_data().len(), 0);
    }
}
