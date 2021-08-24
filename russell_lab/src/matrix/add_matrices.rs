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
        panic!("nrow of matrix [a] (={}) must equal nrow of matrix [c] (={})", a.nrow, c.nrow);
    }
    if a.ncol != c.ncol {
        #[rustfmt::skip]
        panic!("ncol of matrix [a] (={}) must equal ncol of matrix [c] (={})", a.ncol, c.ncol);
    }
    if b.nrow != c.nrow {
        #[rustfmt::skip]
        panic!("nrow of matrix [b] (={}) must equal nrow of matrix [c] (={})", b.nrow, c.nrow);
    }
    if b.ncol != c.ncol {
        #[rustfmt::skip]
        panic!("ncol of matrix [b] (={}) must equal ncol of matrix [c] (={})", b.ncol, c.ncol);
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

    #[test]
    fn add_matrix_oblas_works() {
        const NOISE: f64 = 1234.567;
        let a = Matrix::from(&[
            &[1.0, 2.0, 3.0, 4.0, 5.0],
            &[1.0, 2.0, 3.0, 4.0, 5.0],
            &[1.0, 2.0, 3.0, 4.0, 5.0],
            &[1.0, 2.0, 3.0, 4.0, 5.0],
            &[1.0, 2.0, 3.0, 4.0, 5.0],
        ]);
        let b = Matrix::from(&[
            &[0.5, 1.0, 1.5, 2.0, 2.5],
            &[0.5, 1.0, 1.5, 2.0, 2.5],
            &[0.5, 1.0, 1.5, 2.0, 2.5],
            &[0.5, 1.0, 1.5, 2.0, 2.5],
            &[0.5, 1.0, 1.5, 2.0, 2.5],
        ]);
        let mut c = Matrix::from(&[
            &[NOISE, NOISE, NOISE, NOISE, NOISE],
            &[NOISE, NOISE, NOISE, NOISE, NOISE],
            &[NOISE, NOISE, NOISE, NOISE, NOISE],
            &[NOISE, NOISE, NOISE, NOISE, NOISE],
            &[NOISE, NOISE, NOISE, NOISE, NOISE],
        ]);
        add_matrices(&mut c, 1.0, &a, -4.0, &b);
        #[rustfmt::skip]
        let correct =slice_to_colmajor(&[
            &[-1.0, -2.0, -3.0, -4.0, -5.0],
            &[-1.0, -2.0, -3.0, -4.0, -5.0],
            &[-1.0, -2.0, -3.0, -4.0, -5.0],
            &[-1.0, -2.0, -3.0, -4.0, -5.0],
            &[-1.0, -2.0, -3.0, -4.0, -5.0],
        ]);
        assert_vec_approx_eq!(c.data, correct, 1e-15);
    }

    #[test]
    #[should_panic(expected = "nrow of matrix [a] (=4) must equal nrow of matrix [c] (=3)")]
    fn add_matrices_panic_1() {
        let a = Matrix::new(4, 4);
        let b = Matrix::new(3, 4);
        let mut c = Matrix::new(3, 4);
        add_matrices(&mut c, 1.0, &a, 1.0, &b);
    }

    #[test]
    #[should_panic(expected = "ncol of matrix [a] (=3) must equal ncol of matrix [c] (=4)")]
    fn add_matrices_panic_2() {
        let a = Matrix::new(3, 3);
        let b = Matrix::new(3, 4);
        let mut c = Matrix::new(3, 4);
        add_matrices(&mut c, 1.0, &a, 1.0, &b);
    }

    #[test]
    #[should_panic(expected = "nrow of matrix [b] (=4) must equal nrow of matrix [c] (=3)")]
    fn add_matrices_panic_3() {
        let a = Matrix::new(3, 4);
        let b = Matrix::new(4, 4);
        let mut c = Matrix::new(3, 4);
        add_matrices(&mut c, 1.0, &a, 1.0, &b);
    }

    #[test]
    #[should_panic(expected = "ncol of matrix [b] (=3) must equal ncol of matrix [c] (=4)")]
    fn add_matrices_panic_4() {
        let a = Matrix::new(3, 4);
        let b = Matrix::new(3, 3);
        let mut c = Matrix::new(3, 4);
        add_matrices(&mut c, 1.0, &a, 1.0, &b);
    }

    #[test]
    fn add_matrices_skip() {
        let a = Matrix::new(0, 0);
        let b = Matrix::new(0, 0);
        let mut c = Matrix::new(0, 0);
        add_matrices(&mut c, 1.0, &a, 1.0, &b);
        let correct: &[f64] = &[];
        assert_vec_approx_eq!(c.data, correct, 1e-15);
    }
}
