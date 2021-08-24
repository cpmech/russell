use crate::matrix::*;
use crate::vector::*;
use russell_openblas::*;
use std::convert::TryInto;

/// Performs the outer (tensor) product between two vectors resulting in a matrix
///
/// ```text
///   a  :=  alpha * u  outer  v
/// (m,n)           (m)       (n)
/// ```
///
/// # Note
///
/// The rows of matrix a must equal the length of vector u and
/// the columns of matrix a must equal the length of vector v
///
/// # Examples
///
/// ```
/// use russell_lab::*;
/// let u = Vector::from(&[1.0, 2.0, 3.0]);
/// let v = Vector::from(&[5.0, -2.0, 0.0, 1.0]);
/// let mut a = Matrix::new(u.dim(), v.dim());
/// outer(&mut a, 1.0, &u, &v);
/// let correct = "┌             ┐\n\
///                │  5 -2  0  1 │\n\
///                │ 10 -4  0  2 │\n\
///                │ 15 -6  0  3 │\n\
///                └             ┘";
/// assert_eq!(format!("{}", a), correct);
/// ```
pub fn outer(a: &mut Matrix, alpha: f64, u: &Vector, v: &Vector) {
    let m = u.data.len();
    let n = v.data.len();
    if a.nrow != m {
        #[rustfmt::skip]
        panic!("nrow of matrix a (={}) must equal dim of vector u (={})", a.nrow, m);
    }
    if a.ncol != n {
        #[rustfmt::skip]
        panic!("ncol of matrix a (={}) must equal dim of vector v (={})", a.ncol, n);
    }
    let m_i32: i32 = m.try_into().unwrap();
    let n_i32: i32 = n.try_into().unwrap();
    let lda_i32 = m_i32;
    dger(
        m_i32,
        n_i32,
        alpha,
        &u.data,
        1,
        &v.data,
        1,
        &mut a.data,
        lda_i32,
    );
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use russell_chk::*;

    #[test]
    fn outer_works() {
        let u = Vector::from(&[1.0, 2.0, 3.0]);
        let v = Vector::from(&[5.0, -2.0, 0.0, 1.0]);
        let mut a = Matrix::new(u.data.len(), v.data.len());
        outer(&mut a, 3.0, &u, &v);
        #[rustfmt::skip]
        let correct = slice_to_colmajor(&[
            &[15.0,  -6.0, 0.0, 3.0],
            &[30.0, -12.0, 0.0, 6.0],
            &[45.0, -18.0, 0.0, 9.0],
        ]);
        assert_vec_approx_eq!(a.data, correct, 1e-15);
    }

    #[test]
    fn outer_works_1() {
        let u = Vector::from(&[1.0, 2.0, 3.0, 4.0]);
        let v = Vector::from(&[1.0, 1.0, -2.0]);
        let mut a = Matrix::new(u.data.len(), v.data.len());
        outer(&mut a, 1.0, &u, &v);
        #[rustfmt::skip]
        let correct = slice_to_colmajor(&[
            &[1.0, 1.0, -2.0],
            &[2.0, 2.0, -4.0],
            &[3.0, 3.0, -6.0],
            &[4.0, 4.0, -8.0],
        ]);
        assert_vec_approx_eq!(a.data, correct, 1e-15);
    }

    #[test]
    #[should_panic(expected = "nrow of matrix a (=1) must equal dim of vector u (=2)")]
    fn mat_vec_mul_panic_1() {
        let u = Vector::new(2);
        let v = Vector::new(3);
        let mut a = Matrix::new(1, 3);
        outer(&mut a, 1.0, &u, &v);
    }

    #[test]
    #[should_panic(expected = "ncol of matrix a (=1) must equal dim of vector v (=3)")]
    fn mat_vec_mul_panic_2() {
        let u = Vector::new(2);
        let v = Vector::new(3);
        let mut a = Matrix::new(2, 1);
        outer(&mut a, 1.0, &u, &v);
    }
}
