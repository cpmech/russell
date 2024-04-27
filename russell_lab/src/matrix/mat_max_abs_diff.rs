use super::Matrix;
use crate::StrError;

/// Finds the maximum absolute difference between the components of two matrices
///
/// Returns (i,j,max_abs_diff)
///
/// ```text
/// max_abs_diff := max_ij ( |aᵢⱼ - bᵢⱼ| )
/// ```
///
/// # Warning
///
/// This function may be slow for large matrices.
///
/// # Examples
///
/// ```
/// use russell_lab::*;
///
/// fn main() -> Result<(), StrError> {
///     let a = Matrix::from(&[
///         [ 10.0,  20.0],
///         [-10.0, -20.0],
///     ]);
///     let b = Matrix::from(&[
///         [ 10.0,  20.0],
///         [-10.0, -20.01],
///     ]);
///     let (i, j, max_abs_diff) = mat_max_abs_diff(&a, &b)?;
///     assert_eq!(i, 1);
///     assert_eq!(j, 1);
///     approx_eq(max_abs_diff, 0.01, 1e-14);
///     Ok(())
/// }
/// ```
pub fn mat_max_abs_diff(a: &Matrix, b: &Matrix) -> Result<(usize, usize, f64), StrError> {
    let (m, n) = a.dims();
    if b.nrow() != m || b.ncol() != n {
        return Err("matrices are incompatible");
    }
    let (mut i_found, mut j_found, mut max_abs_diff) = (0, 0, 0.0);
    for i in 0..m {
        for j in 0..n {
            let abs_diff = f64::abs(a.get(i, j) - b.get(i, j));
            if abs_diff > max_abs_diff {
                i_found = i;
                j_found = j;
                max_abs_diff = abs_diff;
            }
        }
    }
    Ok((i_found, j_found, max_abs_diff))
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{mat_max_abs_diff, Matrix};

    #[test]
    fn mat_max_abs_diff_fail_on_wrong_dims() {
        let a_2x2 = Matrix::new(2, 2);
        let b_2x3 = Matrix::new(2, 3);
        let b_3x2 = Matrix::new(3, 2);
        assert_eq!(mat_max_abs_diff(&a_2x2, &b_2x3), Err("matrices are incompatible"));
        assert_eq!(mat_max_abs_diff(&a_2x2, &b_3x2), Err("matrices are incompatible"));
    }

    #[test]
    fn mat_max_abs_diff_works() {
        #[rustfmt::skip]
        let a = Matrix::from(&[
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
        ]);
        #[rustfmt::skip]
        let b = Matrix::from(&[
            [0.5, 1.0, 1.5, 2.0],
            [0.5, 1.0, 1.5, -2.0],
        ]);
        let (i, j, max_abs_diff) = mat_max_abs_diff(&a, &b).unwrap();
        assert_eq!(i, 1);
        assert_eq!(j, 3);
        assert_eq!(max_abs_diff, 6.0);
    }
}
