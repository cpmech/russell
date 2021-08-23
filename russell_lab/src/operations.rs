use super::vector::*;
use super::*;
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
///
pub fn outer(a: &mut Matrix, alpha: f64, u: &Vector, v: &Vector) {
    let m = u.data.len();
    let n = v.data.len();
    if a.nrow != m {
        panic!(
            "the number of rows of matrix a (={}) must equal the size of vector u (={})",
            a.nrow, m
        );
    }
    if a.ncol != n {
        panic!(
            "the number of columns of matrix a (={}) must equal the size of vector v (={})",
            a.ncol, n
        );
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
        panic!(
            "the size of lhs vector v (={}) must be equal to the number of rows of matrix a (={})",
            m, a.nrow
        );
    }
    if n != a.ncol {
        panic!(
            "the size of rhs vector u (={}) must be equal to the number of columns of matrix a (={})", 
            n, a.ncol
        );
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

/*
/// Performs the addition of two vectors
///
/// ```text
/// w := alpha * u + beta * v
/// ```
///
/// # Examples
///
/// ```
/// use russell_lab::*;
/// let u = Vector::from(&[10.0, 20.0, 30.0, 40.0]);
/// let v = Vector::from(&[2.0, 1.5, 1.0, 0.5]);
/// let mut w = Vector::new(4);
/// add_vectors(&mut w, 0.1, &u, 2.0, &v);
/// let correct = "┌   ┐\n\
///                │ 5 │\n\
///                │ 5 │\n\
///                │ 5 │\n\
///                │ 5 │\n\
///                └   ┘";
/// assert_eq!(format!("{}", w), correct);
/// ```
///
pub fn add_vectors(w: &mut Vector, alpha: f64, u: &Vector, beta: f64, v: &Vector) {
    let n = w.data.len();
    if u.data.len() != n {
        #[rustfmt::skip]
        panic!("the length of vector [u] (={}) must equal the length of vector [w] (={})", u.data.len(), n);
    }
    if v.data.len() != n {
        #[rustfmt::skip]
        panic!("the length of vector [v] (={}) must equal the length of vector [w] (={})", v.data.len(), n);
    }
    const SIZE_LIMIT: usize = 99;
    let use_openblas = n > SIZE_LIMIT;
    if use_openblas {
        let n_i32: i32 = n.try_into().unwrap();
        // w := v
        dcopy(n_i32, &v.data, 1, &mut w.data, 1);
        // w := beta * v
        dscal(n_i32, beta, &mut w.data, 1);
        // w := alpha*u + w
        daxpy(n_i32, alpha, &u.data, 1, &mut w.data, 1);
    } else {
        simd_add_vectors(w, alpha, u, beta, v);
        /*
        let m = n % 4;
        for i in 0..m {
            w.data[i] = alpha * u.data[i] + beta * v.data[i];
        }
        for i in (m..n).step_by(4) {
            w.data[i + 0] = alpha * u.data[i + 0] + beta * v.data[i + 0];
            w.data[i + 1] = alpha * u.data[i + 1] + beta * v.data[i + 1];
            w.data[i + 2] = alpha * u.data[i + 2] + beta * v.data[i + 2];
            w.data[i + 3] = alpha * u.data[i + 3] + beta * v.data[i + 3];
        }
        */
    }
}
*/

/// v += alpha * u (daxpy)
// pub fn update_vector(v: &mut Vector, alpha: f64, u: &Vector) {
// TODO
// remember to clear v
// daxpy
// }

/// v := u
// pub fn copy_vector(v: &mut Vector, u: &Vector) {
// TODO
// }

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
    fn outer_alt_works() {
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

}
