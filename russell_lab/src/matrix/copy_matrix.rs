use super::*;
use russell_openblas::*;

/// Copies matrix
///
/// ```text
/// b := a
/// ```
///
/// # Example
///
/// ```
/// # fn main() -> Result<(), &'static str> {
/// use russell_lab::*;
/// let a = Matrix::from(&[
///     &[1.0, 2.0, 3.0],
///     &[4.0, 5.0, 6.0],
/// ])?;
/// let mut b = Matrix::from(&[
///     &[-1.0, -2.0, -3.0],
///     &[-4.0, -5.0, -6.0],
/// ])?;
/// copy_matrix(&mut b, &a)?;
/// let correct = "┌       ┐\n\
///                │ 1 2 3 │\n\
///                │ 4 5 6 │\n\
///                └       ┘";
/// assert_eq!(format!("{}", b), correct);
/// # Ok(())
/// # }
/// ```
pub fn copy_matrix(b: &mut Matrix, a: &Matrix) -> Result<(), &'static str> {
    if a.nrow != b.nrow || a.ncol != b.ncol {
        return Err("matrices have wrong dimensions");
    }
    let n_i32: i32 = to_i32(b.data.len());
    dcopy(n_i32, &a.data, 1, &mut b.data, 1);
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use russell_chk::*;

    #[test]
    fn copy_matrix_works() -> Result<(), &'static str> {
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
        copy_matrix(&mut b, &a)?;
        #[rustfmt::skip]
        let correct = slice_to_colmajor(&[
            &[10.0, 20.0, 30.0],
            &[40.0, 50.0, 60.0],
        ])?;
        assert_vec_approx_eq!(b.data, correct, 1e-15);
        Ok(())
    }

    #[test]
    fn copy_matrix_fails_on_wrong_dimensions() {
        let a_2x2 = Matrix::new(2, 2);
        let a_2x1 = Matrix::new(2, 1);
        let a_1x2 = Matrix::new(1, 2);
        let mut b_2x2 = Matrix::new(2, 2);
        let mut b_2x1 = Matrix::new(2, 1);
        let mut b_1x2 = Matrix::new(1, 2);
        assert_eq!(
            copy_matrix(&mut b_2x2, &a_2x1),
            Err("matrices have wrong dimensions")
        );
        assert_eq!(
            copy_matrix(&mut b_2x2, &a_1x2),
            Err("matrices have wrong dimensions")
        );
        assert_eq!(
            copy_matrix(&mut b_2x1, &a_2x2),
            Err("matrices have wrong dimensions")
        );
        assert_eq!(
            copy_matrix(&mut b_1x2, &a_2x2),
            Err("matrices have wrong dimensions")
        );
    }
}
