use super::*;
use russell_openblas::*;
use std::convert::TryInto;

/// Performs the inner (dot) product between two vectors resulting in a scalar value
///
/// ```text
///  s := u dot v
/// ```
///
/// # Note
///
/// The lengths of both vectors may be different; the smaller length will be selected.
///
/// # Examples
///
/// ```
/// use russell_lab::*;
/// let u = Vector::from(&[1.0, 2.0, 3.0]);
/// let v = Vector::from(&[5.0, -2.0, 0.0, 1.0]);
/// let s = inner(&u, &v);
/// assert_eq!(s, 1.0);
/// ```
///
pub fn inner(u: &Vector, v: &Vector) -> f64 {
    let n = if u.data.len() < v.data.len() {
        u.data.len()
    } else {
        v.data.len()
    };
    ddot(n.try_into().unwrap(), &u.data, 1, &v.data, 1)
}

/// Performs the outer (tensor) product between two vectors resulting in a matrix
///
/// ```text
///  a := u outer v
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
/// outer(&mut a, &u, &v);
/// let correct = "┌             ┐\n\
///                │  5 -2  0  1 │\n\
///                │ 10 -4  0  2 │\n\
///                │ 15 -6  0  3 │\n\
///                └             ┘";
/// assert_eq!(format!("{}", a), correct);
/// ```
///
pub fn outer(a: &mut Matrix, u: &Vector, v: &Vector) {
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
        1.0,
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
///  u := a multiply v
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
/// mat_vec_mul(&mut v, &a, &u);
/// let correct = "┌    ┐\n\
///                │  4 │\n\
///                │  2 │\n\
///                │  3 │\n\
///                │ 16 │\n\
///                └    ┘";
/// assert_eq!(format!("{}", v), correct);
/// ```
///
pub fn mat_vec_mul(v: &mut Vector, a: &Matrix, u: &Vector) {
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
        1.0,
        &a.data,
        lda_i32,
        &u.data,
        1,
        0.0,
        &mut v.data,
        1,
    );
}

/// Performs the matrix-matrix multiplication resulting in a matrix
///
/// ```text
///   c  := alpha *  a   multiply   b
/// (m,n)          (m,k)          (k,n)
/// ```
///
/// # Panics
///
/// This function panics if the matrix dimensions are incorrect
///
/// # Examples
///
/// ```
/// use russell_lab::*;
/// let a = Matrix::from(&[
///     &[1.0, 2.0],
///     &[3.0, 4.0],
///     &[5.0, 6.0],
/// ]);
/// let b = Matrix::from(&[
///     &[-1.0, -2.0, -3.0],
///     &[-4.0, -5.0, -6.0],
/// ]);
/// let mut c = Matrix::new(3, 3);
/// mat_mat_mul(&mut c, 1.0, &a, &b);
/// let correct = "┌             ┐\n\
///                │  -9 -12 -15 │\n\
///                │ -19 -26 -33 │\n\
///                │ -29 -40 -51 │\n\
///                └             ┘";
/// assert_eq!(format!("{}", c), correct);
/// ```
///
pub fn mat_mat_mul(c: &mut Matrix, alpha: f64, a: &Matrix, b: &Matrix) {
    if a.nrow != c.nrow {
        panic!("the number of rows of matrix [a] (={}) must be equal to the number of rows of matrix [c] (={})", a.nrow, c.nrow);
    }
    if b.nrow != a.ncol {
        panic!("the number of rows of matrix [b] (={}) must be equal to the number of columns of matrix [a] (={})", b.nrow, a.ncol);
    }
    if b.ncol != c.ncol {
        panic!("the number of columns of matrix [b] (={}) must be equal to the number of columns of matrix [c] (={})", b.ncol, c.ncol);
    }
    let m_i32: i32 = c.nrow.try_into().unwrap();
    let n_i32: i32 = c.ncol.try_into().unwrap();
    let k_i32: i32 = a.ncol.try_into().unwrap();
    let lda_i32: i32 = a.nrow.try_into().unwrap();
    let ldb_i32: i32 = b.nrow.try_into().unwrap();
    dgemm(
        false,
        false,
        m_i32,
        n_i32,
        k_i32,
        alpha,
        &a.data,
        lda_i32,
        &b.data,
        ldb_i32,
        0.0,
        &mut c.data,
        m_i32,
    );
}

/// Performs the scaling of a vector
///
/// ```text
/// u := alpha * u
/// ```
///
/// # Examples
///
/// ```
/// use russell_lab::*;
/// let mut u = Vector::from(&[1.0, 2.0, 3.0]);
/// scale_vector(&mut u, 0.5);
/// let correct = "┌     ┐\n\
///                │ 0.5 │\n\
///                │   1 │\n\
///                │ 1.5 │\n\
///                └     ┘";
/// assert_eq!(format!("{}", u), correct);
/// ```
///
pub fn scale_vector(u: &mut Vector, alpha: f64) {
    let n: i32 = u.data.len().try_into().unwrap();
    dscal(n, alpha, &mut u.data, 1);
}

/// w := u + v
pub fn add_vectors(w: &mut Vector, u: &Vector, v: &Vector) {
    // TODO
    // remember to clear w using dscal
    // daxpy
}

/// v += alpha * u (daxpy)
pub fn update_vector(v: &mut Vector, alpha: f64, u: &Vector) {
    // TODO
    // remember to clear v
    // daxpy
}

/// v := u
pub fn copy_vector(v: &mut Vector, u: &Vector) {
    // TODO
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use russell_chk::*;

    #[test]
    fn inner_works() {
        const IGNORED: f64 = 100000.0;
        let x = Vector::from(&[20.0, 10.0, 30.0, IGNORED]);
        let y = Vector::from(&[-15.0, -5.0, -24.0]);
        assert_eq!(inner(&x, &y), -1070.0);
    }

    #[test]
    fn outer_works() {
        let u = Vector::from(&[1.0, 2.0, 3.0]);
        let v = Vector::from(&[5.0, -2.0, 0.0, 1.0]);
        let mut a = Matrix::new(u.data.len(), v.data.len());
        outer(&mut a, &u, &v);
        #[rustfmt::skip]
        let correct = slice_to_colmajor(&[
            &[ 5.0, -2.0, 0.0, 1.0],
            &[10.0, -4.0, 0.0, 2.0],
            &[15.0, -6.0, 0.0, 3.0],
        ]);
        assert_vec_approx_eq!(a.data, correct, 1e-15);
    }

    #[test]
    fn outer_alt_works() {
        let u = Vector::from(&[1.0, 2.0, 3.0, 4.0]);
        let v = Vector::from(&[1.0, 1.0, -2.0]);
        let mut a = Matrix::new(u.data.len(), v.data.len());
        outer(&mut a, &u, &v);
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
        mat_vec_mul(&mut v, &a, &u);
        let correct = &[4.0, 8.0, 12.0];
        assert_vec_approx_eq!(v.data, correct, 1e-15);
    }

    #[test]
    fn mat_mat_mul_works() {
        let a = Matrix::from(&[
            // 2 x 3
            &[1.0, 2.00, 3.0],
            &[0.5, 0.75, 1.5],
        ]);
        let b = Matrix::from(&[
            // 3 x 4
            &[0.1, 0.5, 0.5, 0.75],
            &[0.2, 2.0, 2.0, 2.00],
            &[0.3, 0.5, 0.5, 0.50],
        ]);
        let mut c = Matrix::new(2, 4);
        // c := 2⋅a⋅b
        mat_mat_mul(&mut c, 2.0, &a, &b);
        #[rustfmt::skip]
        let correct =slice_to_colmajor(&[
            &[2.80, 12.0, 12.0, 12.50],
            &[1.30,  5.0,  5.0, 5.25],
        ]);
        assert_vec_approx_eq!(c.data, correct, 1e-15);
    }

    #[test]
    fn scale_vector_works() {
        let mut u = Vector::from(&[6.0, 9.0, 12.0]);
        scale_vector(&mut u, 1.0 / 3.0);
        let correct = &[2.0, 3.0, 4.0];
        assert_vec_approx_eq!(u.data, correct, 1e-15);
    }
}
