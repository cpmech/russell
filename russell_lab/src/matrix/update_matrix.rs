use super::*;
use russell_openblas::*;

/// Updates matrix based on another matrix (axpy)
///
/// ```text
/// b += α⋅a
/// ```
///
/// # Example
///
/// ```
/// # fn main() -> Result<(), &'static str> {
/// use russell_lab::*;
/// let a = Matrix::from(&[
///     &[10.0, 20.0, 30.0],
///     &[40.0, 50.0, 60.0],
/// ])?;
/// let mut b = Matrix::from(&[
///     &[10.0, 20.0, 30.0],
///     &[40.0, 50.0, 60.0],
/// ])?;
/// update_matrix(&mut b, 0.1, &a)?;
/// let correct = "┌          ┐\n\
///                │ 11 22 33 │\n\
///                │ 44 55 66 │\n\
///                └          ┘";
/// assert_eq!(format!("{}", b), correct);
/// # Ok(())
/// # }
/// ```
pub fn update_matrix(b: &mut Matrix, alpha: f64, a: &Matrix) -> Result<(), &'static str> {
    if a.nrow != b.nrow || a.ncol != b.ncol {
        return Err("matrices have wrong dimensions");
    }
    let n_i32: i32 = to_i32(b.data.len());
    daxpy(n_i32, alpha, &a.data, 1, &mut b.data, 1);
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use russell_chk::*;

    #[test]
    fn update_matrix_works() -> Result<(), &'static str> {
        #[rustfmt::skip]
        let a = Matrix::from(&[
            &[10.0, 20.0, 30.0],
            &[40.0, 50.0, 60.0],
        ])?;
        #[rustfmt::skip]
        let mut b = Matrix::from(&[
            &[100.0, 200.0, 300.0],
            &[400.0, 500.0, 600.0],
        ])?;
        update_matrix(&mut b, 2.0, &a)?;
        #[rustfmt::skip]
        let correct = slice_to_colmajor(&[
            &[120.0, 240.0, 360.0],
            &[480.0, 600.0, 720.0],
        ])?;
        assert_vec_approx_eq!(b.data, correct, 1e-15);
        Ok(())
    }

    #[test]
    fn update_matrix_fail_on_wrong_dimensions() {
        let a_2x2 = Matrix::new(2, 2);
        let a_2x1 = Matrix::new(2, 1);
        let a_1x2 = Matrix::new(1, 2);
        let mut b_2x2 = Matrix::new(2, 2);
        let mut b_2x1 = Matrix::new(2, 1);
        let mut b_1x2 = Matrix::new(1, 2);
        assert_eq!(
            update_matrix(&mut b_2x2, 1.0, &a_2x1),
            Err("matrices have wrong dimensions")
        );
        assert_eq!(
            update_matrix(&mut b_2x2, 1.0, &a_1x2),
            Err("matrices have wrong dimensions")
        );
        assert_eq!(
            update_matrix(&mut b_2x1, 1.0, &a_2x2),
            Err("matrices have wrong dimensions")
        );
        assert_eq!(
            update_matrix(&mut b_1x2, 1.0, &a_2x2),
            Err("matrices have wrong dimensions")
        );
    }
}
