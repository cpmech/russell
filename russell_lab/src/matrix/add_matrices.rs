use super::*;
use russell_openblas::*;

const NATIVE_VERSUS_OPENBLAS_BOUNDARY: usize = 16;

/// Performs the addition of two matrices
///
/// ```text
/// c := α⋅a + β⋅b
/// ```
///
/// # Example
///
/// ```
/// # fn main() -> Result<(), &'static str> {
/// use russell_lab::*;
/// let a = Matrix::from(&[
///     &[ 10.0,  20.0,  30.0,  40.0],
///     &[-10.0, -20.0, -30.0, -40.0],
/// ])?;
/// let b = Matrix::from(&[
///     &[ 2.0,  1.5,  1.0,  0.5],
///     &[-2.0, -1.5, -1.0, -0.5],
/// ])?;
/// let mut c = Matrix::new(2, 4);
/// add_matrices(&mut c, 0.1, &a, 2.0, &b)?;
/// let correct = "┌             ┐\n\
///                │  5  5  5  5 │\n\
///                │ -5 -5 -5 -5 │\n\
///                └             ┘";
/// assert_eq!(format!("{}", c), correct);
/// # Ok(())
/// # }
/// ```
pub fn add_matrices(
    c: &mut Matrix,
    alpha: f64,
    a: &Matrix,
    beta: f64,
    b: &Matrix,
) -> Result<(), &'static str> {
    if a.nrow != c.nrow || a.ncol != c.ncol || b.nrow != c.nrow || b.ncol != c.ncol {
        return Err("matrices have wrong dimensions");
    }
    if c.nrow == 0 && c.ncol == 0 {
        return Ok(());
    }
    if c.data.len() > NATIVE_VERSUS_OPENBLAS_BOUNDARY {
        add_vectors_oblas(&mut c.data, alpha, &a.data, beta, &b.data);
    } else {
        add_vectors_native(&mut c.data, alpha, &a.data, beta, &b.data);
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use russell_chk::*;

    #[test]
    fn add_matrices_works() -> Result<(), &'static str> {
        const NOISE: f64 = 1234.567;
        let a = Matrix::from(&[
            &[1.0, 2.0, 3.0, 4.0],
            &[1.0, 2.0, 3.0, 4.0],
            &[1.0, 2.0, 3.0, 4.0],
        ])?;
        let b = Matrix::from(&[
            &[0.5, 1.0, 1.5, 2.0],
            &[0.5, 1.0, 1.5, 2.0],
            &[0.5, 1.0, 1.5, 2.0],
        ])?;
        let mut c = Matrix::from(&[
            &[NOISE, NOISE, NOISE, NOISE],
            &[NOISE, NOISE, NOISE, NOISE],
            &[NOISE, NOISE, NOISE, NOISE],
        ])?;
        add_matrices(&mut c, 1.0, &a, -4.0, &b)?;
        #[rustfmt::skip]
        let correct = slice_to_colmajor(&[
            &[-1.0, -2.0, -3.0, -4.0],
            &[-1.0, -2.0, -3.0, -4.0],
            &[-1.0, -2.0, -3.0, -4.0],
        ])?;
        assert_vec_approx_eq!(c.data, correct, 1e-15);
        Ok(())
    }

    #[test]
    fn add_matrix_oblas_works() -> Result<(), &'static str> {
        const NOISE: f64 = 1234.567;
        let a = Matrix::from(&[
            &[1.0, 2.0, 3.0, 4.0, 5.0],
            &[1.0, 2.0, 3.0, 4.0, 5.0],
            &[1.0, 2.0, 3.0, 4.0, 5.0],
            &[1.0, 2.0, 3.0, 4.0, 5.0],
            &[1.0, 2.0, 3.0, 4.0, 5.0],
        ])?;
        let b = Matrix::from(&[
            &[0.5, 1.0, 1.5, 2.0, 2.5],
            &[0.5, 1.0, 1.5, 2.0, 2.5],
            &[0.5, 1.0, 1.5, 2.0, 2.5],
            &[0.5, 1.0, 1.5, 2.0, 2.5],
            &[0.5, 1.0, 1.5, 2.0, 2.5],
        ])?;
        let mut c = Matrix::from(&[
            &[NOISE, NOISE, NOISE, NOISE, NOISE],
            &[NOISE, NOISE, NOISE, NOISE, NOISE],
            &[NOISE, NOISE, NOISE, NOISE, NOISE],
            &[NOISE, NOISE, NOISE, NOISE, NOISE],
            &[NOISE, NOISE, NOISE, NOISE, NOISE],
        ])?;
        add_matrices(&mut c, 1.0, &a, -4.0, &b)?;
        #[rustfmt::skip]
        let correct = slice_to_colmajor(&[
            &[-1.0, -2.0, -3.0, -4.0, -5.0],
            &[-1.0, -2.0, -3.0, -4.0, -5.0],
            &[-1.0, -2.0, -3.0, -4.0, -5.0],
            &[-1.0, -2.0, -3.0, -4.0, -5.0],
            &[-1.0, -2.0, -3.0, -4.0, -5.0],
        ])?;
        assert_vec_approx_eq!(c.data, correct, 1e-15);
        Ok(())
    }

    #[test]
    fn add_matrices_handle_wrong_dimensions() {
        let a_2x2 = Matrix::new(2, 2);
        let a_2x3 = Matrix::new(2, 3);
        let a_3x2 = Matrix::new(3, 2);
        let b_2x2 = Matrix::new(2, 2);
        let b_2x3 = Matrix::new(2, 3);
        let b_3x2 = Matrix::new(3, 2);
        let mut c_2x2 = Matrix::new(2, 2);
        assert_eq!(
            add_matrices(&mut c_2x2, 1.0, &a_2x3, 1.0, &b_2x2),
            Err("matrices have wrong dimensions")
        );
        assert_eq!(
            add_matrices(&mut c_2x2, 1.0, &a_3x2, 1.0, &b_2x2),
            Err("matrices have wrong dimensions")
        );
        assert_eq!(
            add_matrices(&mut c_2x2, 1.0, &a_2x2, 1.0, &b_2x3),
            Err("matrices have wrong dimensions")
        );
        assert_eq!(
            add_matrices(&mut c_2x2, 1.0, &a_2x2, 1.0, &b_3x2),
            Err("matrices have wrong dimensions")
        );
    }

    #[test]
    fn add_matrices_skip() -> Result<(), &'static str> {
        let a = Matrix::new(0, 0);
        let b = Matrix::new(0, 0);
        let mut c = Matrix::new(0, 0);
        add_matrices(&mut c, 1.0, &a, 1.0, &b)?;
        let correct: &[f64] = &[];
        assert_vec_approx_eq!(c.data, correct, 1e-15);
        Ok(())
    }
}
