use crate::matrix::*;
use crate::vector::*;
use russell_openblas::*;
use std::convert::TryInto;

/// Performs the matrix-vector multiplication resulting in a vector
///
/// ```text
///  v  := alpha * a   multiply  u
/// (m)          (m,n)          (n)
/// ```
///
/// # Note
///
/// The length of vector u must equal the rows of matrix a and
/// the length of vector v must equal the columns of matrix a
///
/// # Examples
///
/// ```
/// use russell_lab::*;
/// let a = Matrix::from(&[
///     &[ 5.0, -2.0, 1.0],
///     &[-4.0,  0.0, 2.0],
///     &[15.0, -6.0, 0.0],
///     &[ 3.0,  5.0, 1.0],
/// ]);
/// let u = Vector::from(&[1.0, 2.0, 3.0]);
/// let mut v = Vector::new(a.nrow());
/// mat_vec_mul(&mut v, 0.5, &a, &u);
/// let correct = "┌     ┐\n\
///                │   2 │\n\
///                │   1 │\n\
///                │ 1.5 │\n\
///                │   8 │\n\
///                └     ┘";
/// assert_eq!(format!("{}", v), correct);
/// ```
///
pub fn mat_vec_mul(v: &mut Vector, alpha: f64, a: &Matrix, u: &Vector) {
    let m = v.data.len();
    let n = u.data.len();
    if m != a.nrow {
        #[rustfmt::skip]
        panic!("dim of vector v (={}) must equal nrow of matrix a (={})", m, a.nrow);
    }
    if n != a.ncol {
        #[rustfmt::skip]
        panic!("dim of vector u (={}) must equal ncol of matrix a (={})", n, a.ncol);
    }
    let m_i32: i32 = m.try_into().unwrap();
    let n_i32: i32 = n.try_into().unwrap();
    let lda_i32 = m_i32;
    dgemv(
        false,
        m_i32,
        n_i32,
        alpha,
        &a.data,
        lda_i32,
        &u.data,
        1,
        0.0,
        &mut v.data,
        1,
    );
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use russell_chk::*;

    #[test]
    fn mat_vec_mul_works() {
        #[rustfmt::skip]
        let a = Matrix::from(&[
            &[ 5.0, -2.0, 0.0, 1.0],
            &[10.0, -4.0, 0.0, 2.0],
            &[15.0, -6.0, 0.0, 3.0],
        ]);
        let u = Vector::from(&[1.0, 3.0, 8.0, 5.0]);
        let mut v = Vector::new(a.nrow());
        mat_vec_mul(&mut v, 1.0, &a, &u);
        let correct = &[4.0, 8.0, 12.0];
        assert_vec_approx_eq!(v.data, correct, 1e-15);
    }

    #[test]
    #[should_panic(expected = "dim of vector v (=4) must equal nrow of matrix a (=3)")]
    fn mat_vec_mul_panic_1() {
        let u = Vector::new(4);
        let a = Matrix::new(3, 4);
        let mut v = Vector::new(4);
        mat_vec_mul(&mut v, 1.0, &a, &u);
    }

    #[test]
    #[should_panic(expected = "dim of vector u (=3) must equal ncol of matrix a (=4)")]
    fn mat_vec_mul_panic_2() {
        let u = Vector::new(3);
        let a = Matrix::new(3, 4);
        let mut v = Vector::new(3);
        mat_vec_mul(&mut v, 1.0, &a, &u);
    }
}
