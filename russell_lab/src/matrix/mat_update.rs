use super::Matrix;
use crate::StrError;
use russell_openblas::{daxpy, to_i32};

/// Updates matrix based on another matrix
///
/// ```text
/// b += α⋅a
/// ```
///
/// # Example
///
/// ```
/// use russell_lab::{mat_update, Matrix, StrError};
///
/// fn main() -> Result<(), StrError> {
///     let a = Matrix::from(&[
///         [10.0, 20.0, 30.0],
///         [40.0, 50.0, 60.0],
///     ]);
///     let mut b = Matrix::from(&[
///         [10.0, 20.0, 30.0],
///         [40.0, 50.0, 60.0],
///     ]);
///     mat_update(&mut b, 0.1, &a)?;
///     let correct = "┌          ┐\n\
///                    │ 11 22 33 │\n\
///                    │ 44 55 66 │\n\
///                    └          ┘";
///     assert_eq!(format!("{}", b), correct);
///     Ok(())
/// }
/// ```
pub fn mat_update(b: &mut Matrix, alpha: f64, a: &Matrix) -> Result<(), StrError> {
    let (m, n) = b.dims();
    if a.nrow() != m || a.ncol() != n {
        return Err("matrices are incompatible");
    }
    let mn_i32: i32 = to_i32(m * n);
    daxpy(mn_i32, alpha, a.as_data(), 1, b.as_mut_data(), 1);
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{mat_update, Matrix};
    use crate::mat_approx_eq;

    #[test]
    fn mat_update_fail_on_wrong_dims() {
        let a_2x2 = Matrix::new(2, 2);
        let a_2x1 = Matrix::new(2, 1);
        let a_1x2 = Matrix::new(1, 2);
        let mut b_2x2 = Matrix::new(2, 2);
        let mut b_2x1 = Matrix::new(2, 1);
        let mut b_1x2 = Matrix::new(1, 2);
        assert_eq!(mat_update(&mut b_2x2, 1.0, &a_2x1), Err("matrices are incompatible"));
        assert_eq!(mat_update(&mut b_2x2, 1.0, &a_1x2), Err("matrices are incompatible"));
        assert_eq!(mat_update(&mut b_2x1, 1.0, &a_2x2), Err("matrices are incompatible"));
        assert_eq!(mat_update(&mut b_1x2, 1.0, &a_2x2), Err("matrices are incompatible"));
    }

    #[test]
    fn mat_update_works() {
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
        mat_update(&mut b, 2.0, &a).unwrap();
        #[rustfmt::skip]
        let correct = &[
            [120.0, 240.0, 360.0],
            [480.0, 600.0, 720.0],
        ];
        mat_approx_eq(&b, correct, 1e-15);
    }
}
