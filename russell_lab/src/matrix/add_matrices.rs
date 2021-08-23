use super::*;
use russell_openblas::*;

const NATIVE_VERSUS_OPENBLAS_BOUNDARY: usize = 16;

/// Performs the addition of two vectors
///
/// ```text
/// c := alpha * a + beta * b
/// ```
///
/// # Examples
///
/// ```
/// use russell_lab::*;
/// let a = Matrix::from(&[
///     &[ 10.0,  20.0,  30.0,  40.0],
///     &[-10.0, -20.0, -30.0, -40.0],
/// ]);
/// let b = Matrix::from(&[
///     &[ 2.0,  1.5,  1.0,  0.5],
///     &[-2.0, -1.5, -1.0, -0.5],
/// ]);
/// let mut c = Matrix::new(2, 4);
/// add_matrices(&mut c, 0.1, &a, 2.0, &b);
/// let correct = "┌             ┐\n\
///                │  5  5  5  5 │\n\
///                │ -5 -5 -5 -5 │\n\
///                └             ┘";
/// assert_eq!(format!("{}", c), correct);
/// ```
///
pub fn add_matrices(c: &mut Matrix, alpha: f64, a: &Matrix, beta: f64, b: &Matrix) {
    if a.nrow != c.nrow {
        #[rustfmt::skip]
        panic!("the nrow of matrix [a] (={}) must equal the nrow of matrix [c] (={})", a.nrow, c.nrow);
    }
    if a.ncol != c.ncol {
        #[rustfmt::skip]
        panic!("the ncol of matrix [a] (={}) must equal the ncol of matrix [c] (={})", a.ncol, c.ncol);
    }
    if b.nrow != c.nrow {
        #[rustfmt::skip]
        panic!("the nrow of matrix [b] (={}) must equal the nrow of matrix [c] (={})", b.nrow, c.nrow);
    }
    if b.ncol != c.ncol {
        #[rustfmt::skip]
        panic!("the ncol of matrix [b] (={}) must equal the ncol of matrix [c] (={})", b.ncol, c.ncol);
    }
    if c.nrow == 0 && c.ncol == 0 {
        return;
    }
    if c.data.len() > NATIVE_VERSUS_OPENBLAS_BOUNDARY {
        add_vectors_oblas(&mut c.data, alpha, &a.data, beta, &b.data);
    } else {
        add_vectors_native(&mut c.data, alpha, &a.data, beta, &b.data);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use russell_chk::*;

    #[test]
    fn add_matrices_works() {
        const NOISE: f64 = 1234.567;
        let a = Matrix::from(&[
            &[1.0, 2.0, 3.0, 4.0],
            &[1.0, 2.0, 3.0, 4.0],
            &[1.0, 2.0, 3.0, 4.0],
        ]);
        let b = Matrix::from(&[
            &[0.5, 1.0, 1.5, 2.0],
            &[0.5, 1.0, 1.5, 2.0],
            &[0.5, 1.0, 1.5, 2.0],
        ]);
        let mut c = Matrix::from(&[
            &[NOISE, NOISE, NOISE, NOISE],
            &[NOISE, NOISE, NOISE, NOISE],
            &[NOISE, NOISE, NOISE, NOISE],
        ]);
        add_matrices(&mut c, 1.0, &a, -4.0, &b);
        #[rustfmt::skip]
        let correct =slice_to_colmajor(&[
            &[-1.0, -2.0, -3.0, -4.0],
            &[-1.0, -2.0, -3.0, -4.0],
            &[-1.0, -2.0, -3.0, -4.0],
        ]);
        assert_vec_approx_eq!(c.data, correct, 1e-15);
    }
}
