use super::*;
use russell_openblas::*;
use std::convert::TryInto;

/// Updates matrix based on another matrix (axpy)
///
/// ```text
/// b += alpha * a
/// ```
///
/// # Examples
///
/// ```
/// use russell_lab::*;
/// let a = Matrix::from(&[
///     &[10.0, 20.0, 30.0],
///     &[40.0, 50.0, 60.0],
/// ]);
/// let mut b = Matrix::from(&[
///     &[10.0, 20.0, 30.0],
///     &[40.0, 50.0, 60.0],
/// ]);
/// update_matrix(&mut b, 0.1, &a);
/// let correct = "┌          ┐\n\
///                │ 11 22 33 │\n\
///                │ 44 55 66 │\n\
///                └          ┘";
/// assert_eq!(format!("{}", b), correct);
/// ```
pub fn update_matrix(b: &mut Matrix, alpha: f64, a: &Matrix) {
    if a.nrow != b.nrow || a.ncol != b.ncol {
        panic!("the matrices must have the same dimensions");
    }
    let n_i32: i32 = b.data.len().try_into().unwrap();
    daxpy(n_i32, alpha, &a.data, 1, &mut b.data, 1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use russell_chk::*;

    #[test]
    fn update_matrix_works() {
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
        update_matrix(&mut b, 2.0, &a);
        #[rustfmt::skip]
        let correct =slice_to_colmajor(&[
            &[120.0, 240.0, 360.0],
            &[480.0, 600.0, 720.0],
        ]);
        assert_vec_approx_eq!(b.data, correct, 1e-15);
    }
}
