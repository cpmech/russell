use crate::matrix::ComplexMatrix;
use crate::vector::ComplexVector;
use crate::{to_i32, StrError, CBLAS_COL_MAJOR, CBLAS_TRANS};
use num_complex::Complex64;

extern "C" {
    // Performs one of the matrix-vector multiplication (complex version)
    // <http://www.netlib.org/lapack/explore-html/db/d40/zgemv_8f.html>
    fn cblas_zgemv(
        layout: i32,
        transa: i32,
        m: i32,
        n: i32,
        alpha: *const Complex64,
        a: *const Complex64,
        lda: i32,
        x: *const Complex64,
        incx: i32,
        beta: *const Complex64,
        y: *mut Complex64,
        incy: i32,
    );
}

/// Performs the vector-matrix multiplication resulting in a vector (complex version)
///
/// ```text
///  v  :=  α ⋅  u  ⋅  a  
/// (n)         (m)  (m,n)
/// ```
///
/// or
///
/// ```text
///  v  :=  α ⋅   aᵀ  ⋅  u
/// (n)         (n,m)   (m)  
/// ```
///
/// # Note
///
/// The length of vector `u` must equal the number of rows of matrix `a` and
/// the length of vector `v` must equal the number of columns of matrix `a`.
///
/// # Examples
///
/// ```
/// use russell_lab::*;
/// use num_complex::Complex64;
///
/// fn main() -> Result<(), StrError> {
///     #[rustfmt::skip]
///     let a = ComplexMatrix::from(&[
///         [ 5.0, -2.0, 0.0, 1.0],
///         [10.0, -4.0, 0.0, 2.0],
///         [15.0, -6.0, 0.0, 3.0],
///     ]);
///     let u = ComplexVector::from(&[1.0, 3.0, 8.0]);
///     let mut v = ComplexVector::new(a.ncol());
///     let one = Complex64::new(1.0, 0.0);
///     complex_vec_mat_mul(&mut v, one, &u, &a).unwrap();
///     let correct = &[
///         Complex64::new(155.0, 0.0),
///         Complex64::new(-62.0, 0.0),
///         Complex64::new(0.0, 0.0),
///         Complex64::new(31.0, 0.0),
///     ];
///     complex_vec_approx_eq(v.as_data(), correct, 1e-15);
///     Ok(())
/// }
/// ```
pub fn complex_vec_mat_mul(
    v: &mut ComplexVector,
    alpha: Complex64,
    u: &ComplexVector,
    a: &ComplexMatrix,
) -> Result<(), StrError> {
    let n = v.dim();
    let m = u.dim();
    if m != a.nrow() || n != a.ncol() {
        return Err("matrix and vectors are incompatible");
    }
    if m == 0 || n == 0 {
        return Ok(());
    }
    let m_i32: i32 = to_i32(m);
    let n_i32: i32 = to_i32(n);
    let incx = 1;
    let incy = 1;
    let beta = Complex64::new(0.0, 0.0);
    unsafe {
        cblas_zgemv(
            CBLAS_COL_MAJOR,
            CBLAS_TRANS,
            m_i32,
            n_i32,
            &alpha,
            a.as_data().as_ptr(),
            m_i32,
            u.as_data().as_ptr(),
            incx,
            &beta,
            v.as_mut_data().as_mut_ptr(),
            incy,
        );
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{complex_vec_mat_mul, ComplexMatrix, ComplexVector};
    use crate::complex_vec_approx_eq;
    use num_complex::Complex64;

    #[test]
    fn vec_mat_mul_fails_on_wrong_dims() {
        let u = ComplexVector::new(2);
        let a_1x2 = ComplexMatrix::new(1, 2);
        let a_3x1 = ComplexMatrix::new(3, 1);
        let mut v = ComplexVector::new(3);
        let one = Complex64::new(1.0, 0.0);
        assert_eq!(
            complex_vec_mat_mul(&mut v, one, &u, &a_1x2),
            Err("matrix and vectors are incompatible")
        );
        assert_eq!(
            complex_vec_mat_mul(&mut v, one, &u, &a_3x1),
            Err("matrix and vectors are incompatible")
        );
    }

    #[test]
    fn vec_mat_mul_works() {
        #[rustfmt::skip]
        let a = ComplexMatrix::from(&[
            [ 5.0, -2.0, 0.0, 1.0],
            [10.0, -4.0, 0.0, 2.0],
            [15.0, -6.0, 0.0, 3.0],
        ]);
        let u = ComplexVector::from(&[1.0, 3.0, 8.0]);
        let mut v = ComplexVector::new(a.ncol());
        let one = Complex64::new(1.0, 0.0);
        complex_vec_mat_mul(&mut v, one, &u, &a).unwrap();
        let correct = &[
            Complex64::new(155.0, 0.0),
            Complex64::new(-62.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(31.0, 0.0),
        ];
        complex_vec_approx_eq(v.as_data(), correct, 1e-15);
    }

    #[test]
    fn vec_mat_mul_zero_works() {
        let a_0x0 = ComplexMatrix::new(0, 0);
        let a_0x1 = ComplexMatrix::new(0, 1);
        let a_1x0 = ComplexMatrix::new(1, 0);
        let u0 = ComplexVector::new(0);
        let u1 = ComplexVector::new(1);
        let mut v0 = ComplexVector::new(0);
        let mut v1 = ComplexVector::new(1);
        let one = Complex64::new(1.0, 0.0);
        let zero = Complex64::new(0.0, 0.0);
        complex_vec_mat_mul(&mut v0, one, &u0, &a_0x0).unwrap();
        assert_eq!(v0.as_data(), &[] as &[Complex64]);
        complex_vec_mat_mul(&mut v1, one, &u0, &a_0x1).unwrap();
        assert_eq!(v1.as_data(), &[zero]);
        complex_vec_mat_mul(&mut v0, one, &u1, &a_1x0).unwrap();
        assert_eq!(v0.as_data(), &[] as &[Complex64]);
    }
}
