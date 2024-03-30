use crate::matrix::Matrix;
use crate::vector::Vector;
use crate::{to_i32, StrError, CBLAS_COL_MAJOR, CBLAS_NO_TRANS};

extern "C" {
    // Performs one of the matrix-vector multiplication
    // <https://www.netlib.org/lapack/explore-html/dc/da8/dgemv_8f.html>
    fn cblas_dgemv(
        layout: i32,
        transa: i32,
        m: i32,
        n: i32,
        alpha: f64,
        a: *const f64,
        lda: i32,
        x: *const f64,
        incx: i32,
        beta: f64,
        y: *mut f64,
        incy: i32,
    );
}

/// (dgemv) Performs the matrix-vector multiplication
///
/// ```text
///  v  :=  α ⋅  a   ⋅  u
/// (m)        (m,n)   (n)
/// ```
///
/// See also: <https://www.netlib.org/lapack/explore-html/dc/da8/dgemv_8f.html>
///
/// # Note
///
/// The length of vector `u` must equal the number of columns of matrix `a` and
/// the length of vector `v` must equal the number of rows of matrix `a`.
///
/// # Example
///
/// ```
/// use russell_lab::{mat_vec_mul, Matrix, Vector, StrError};
///
/// fn main() -> Result<(), StrError> {
///     let a = Matrix::from(&[
///         [ 5.0, -2.0, 1.0],
///         [-4.0,  0.0, 2.0],
///         [15.0, -6.0, 0.0],
///         [ 3.0,  5.0, 1.0],
///     ]);
///     let u = Vector::from(&[1.0, 2.0, 3.0]);
///     let mut v = Vector::new(a.nrow());
///     mat_vec_mul(&mut v, 0.5, &a, &u)?;
///     let correct = "┌     ┐\n\
///                    │   2 │\n\
///                    │   1 │\n\
///                    │ 1.5 │\n\
///                    │   8 │\n\
///                    └     ┘";
///     assert_eq!(format!("{}", v), correct);
///     Ok(())
/// }
/// ```
pub fn mat_vec_mul(v: &mut Vector, alpha: f64, a: &Matrix, u: &Vector) -> Result<(), StrError> {
    let m = v.dim();
    let n = u.dim();
    if m != a.nrow() || n != a.ncol() {
        return Err("matrix and vectors are incompatible");
    }
    if m == 0 {
        return Ok(());
    }
    if n == 0 {
        v.fill(0.0);
        return Ok(());
    }
    let m_i32: i32 = to_i32(m);
    let n_i32: i32 = to_i32(n);
    let incx = 1;
    let incy = 1;
    let beta = 0.0;
    unsafe {
        cblas_dgemv(
            CBLAS_COL_MAJOR,
            CBLAS_NO_TRANS,
            m_i32,
            n_i32,
            alpha,
            a.as_data().as_ptr(),
            m_i32,
            u.as_data().as_ptr(),
            incx,
            beta,
            v.as_mut_data().as_mut_ptr(),
            incy,
        );
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{mat_vec_mul, Matrix, Vector};
    use crate::{vec_approx_eq, vec_norm, Norm};

    #[test]
    fn mat_vec_mul_fails_on_wrong_dims() {
        let u = Vector::new(2);
        let a_1x2 = Matrix::new(1, 2);
        let a_3x1 = Matrix::new(3, 1);
        let mut v = Vector::new(3);
        assert_eq!(
            mat_vec_mul(&mut v, 1.0, &a_1x2, &u),
            Err("matrix and vectors are incompatible")
        );
        assert_eq!(
            mat_vec_mul(&mut v, 1.0, &a_3x1, &u),
            Err("matrix and vectors are incompatible")
        );
    }

    #[test]
    fn mat_vec_mul_works() {
        #[rustfmt::skip]
        let a = Matrix::from(&[
            [ 5.0, -2.0, 0.0, 1.0],
            [10.0, -4.0, 0.0, 2.0],
            [15.0, -6.0, 0.0, 3.0],
        ]);
        let u = Vector::from(&[1.0, 3.0, 8.0, 5.0]);
        let mut v = Vector::new(a.nrow());
        mat_vec_mul(&mut v, 1.0, &a, &u).unwrap();
        let correct = &[4.0, 8.0, 12.0];
        vec_approx_eq(v.as_data(), correct, 1e-15);
    }

    #[test]
    fn mat_vec_mul_zero_works() {
        let a_0x0 = Matrix::new(0, 0);
        let a_0x1 = Matrix::new(0, 1);
        let a_1x0 = Matrix::new(1, 0);
        let u0 = Vector::new(0);
        let u1 = Vector::new(1);
        let mut v0 = Vector::new(0);
        let mut v1 = Vector::new(1);
        mat_vec_mul(&mut v0, 1.0, &a_0x0, &u0).unwrap();
        assert_eq!(v0.as_data(), &[] as &[f64]);
        mat_vec_mul(&mut v0, 1.0, &a_0x1, &u1).unwrap();
        assert_eq!(v0.as_data(), &[] as &[f64]);
        mat_vec_mul(&mut v1, 1.0, &a_1x0, &u0).unwrap();
        assert_eq!(v1.as_data(), &[0.0]);
    }

    #[test]
    fn mat_vec_mul_works_range() {
        // v  :=  a  ⋅ u
        // (m)  (m,n) (n)
        for m in [0, 7, 15_usize] {
            for n in [0, 4, 8_usize] {
                let a = Matrix::filled(m, n, 1.0);
                let u = Vector::filled(n, 1.0);
                let mut v = Vector::new(m);
                mat_vec_mul(&mut v, 1.0, &a, &u).unwrap();
                if m == 0 {
                    assert_eq!(vec_norm(&v, Norm::Max), 0.0);
                } else {
                    assert_eq!(vec_norm(&v, Norm::Max), n as f64);
                }
            }
        }
    }
}
