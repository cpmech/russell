use super::*;
use russell_openblas::*;
use std::convert::TryInto;

/// Copies matrix
///
/// ```text
/// b := a
/// ```
///
/// # Examples
///
/// ```
/// use russell_lab::*;
/// let a = Matrix::from(&[
///     &[1.0, 2.0, 3.0],
///     &[4.0, 5.0, 6.0],
/// ]);
/// let mut b = Matrix::from(&[
///     &[-1.0, -2.0, -3.0],
///     &[-4.0, -5.0, -6.0],
/// ]);
/// copy_matrix(&mut b, &a);
/// let correct = "┌       ┐\n\
///                │ 1 2 3 │\n\
///                │ 4 5 6 │\n\
///                └       ┘";
/// assert_eq!(format!("{}", b), correct);
/// ```
pub fn copy_matrix(b: &mut Matrix, a: &Matrix) {
    if a.nrow != b.nrow || a.ncol != b.ncol {
        panic!("the matrices must have the same dimensions");
    }
    let n_i32: i32 = b.data.len().try_into().unwrap();
    dcopy(n_i32, &a.data, 1, &mut b.data, 1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use russell_chk::*;

    #[test]
    fn copy_matrix_works() {
        #[rustfmt::skip]
        let a = Matrix::from(&[
            &[10.0, 20.0, 30.0],
            &[40.0, 50.0, 60.0],
        ]);
        #[rustfmt::skip]
        let mut b = Matrix::from(&[
            &[100.0, 200.0, 300.0],
            &[400.0, 500.0, 600.0],
        ]);
        copy_matrix(&mut b, &a);
        #[rustfmt::skip]
        let correct =slice_to_colmajor(&[
            &[10.0, 20.0, 30.0],
            &[40.0, 50.0, 60.0],
        ]);
        assert_vec_approx_eq!(b.data, correct, 1e-15);
    }
}
