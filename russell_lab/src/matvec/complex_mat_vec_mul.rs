use crate::matrix::ComplexMatrix;
use crate::vector::ComplexVector;
use crate::{to_i32, StrError, CBLAS_COL_MAJOR, CBLAS_NO_TRANS};
use num_complex::Complex64;

extern "C" {
    // Performs one of the matrix-vector multiplication
    // <https://www.netlib.org/lapack/explore-html/db/d40/zgemv_8f.html>
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

/// Performs the matrix-vector multiplication
///
/// ```text
///  v  :=  α ⋅  a   ⋅  u
/// (m)        (m,n)   (n)
/// ```
///
/// # Note
///
/// The length of vector `u` must equal the number of columns of matrix `a` and
/// the length of vector `v` must equal the number of rows of matrix `a`.
///
/// # Example
///
/// ```
/// use num_complex::Complex64;
/// use russell_lab::*;
///
/// fn main() -> Result<(), StrError> {
///     let a = ComplexMatrix::from(&[
///         [ 5.0, -2.0, 1.0],
///         [-4.0,  0.0, 2.0],
///         [15.0, -6.0, 0.0],
///         [ 3.0,  5.0, 1.0],
///     ]);
///     let u = ComplexVector::from(&[1.0, 2.0, 3.0]);
///     let mut v = ComplexVector::new(a.nrow());
///     let half = cpx!(0.5, 0.0);
///     complex_mat_vec_mul(&mut v, half, &a, &u)?;
///     let correct = &[
///         cpx!(2.0, 0.0),
///         cpx!(1.0, 0.0),
///         cpx!(1.5, 0.0),
///         cpx!(8.0, 0.0),
///     ];
///     complex_vec_approx_eq(v.as_data(), correct, 1e-15);
///     Ok(())
/// }
/// ```
pub fn complex_mat_vec_mul(
    v: &mut ComplexVector,
    alpha: Complex64,
    a: &ComplexMatrix,
    u: &ComplexVector,
) -> Result<(), StrError> {
    let m = v.dim();
    let n = u.dim();
    if m != a.nrow() || n != a.ncol() {
        return Err("matrix and vectors are incompatible");
    }
    if m == 0 {
        return Ok(());
    }
    let zero = Complex64::new(0.0, 0.0);
    if n == 0 {
        v.fill(zero);
        return Ok(());
    }
    let m_i32: i32 = to_i32(m);
    let n_i32: i32 = to_i32(n);
    let incx = 1;
    let incy = 1;
    unsafe {
        cblas_zgemv(
            CBLAS_COL_MAJOR,
            CBLAS_NO_TRANS,
            m_i32,
            n_i32,
            &alpha,
            a.as_data().as_ptr(),
            m_i32,
            u.as_data().as_ptr(),
            incx,
            &zero,
            v.as_mut_data().as_mut_ptr(),
            incy,
        );
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{complex_mat_vec_mul, ComplexMatrix, ComplexVector};
    use crate::{complex_vec_approx_eq, cpx};
    use num_complex::Complex64;

    #[test]
    fn complex_mat_vec_mul_fails_on_wrong_dims() {
        let u = ComplexVector::new(2);
        let a_1x2 = ComplexMatrix::new(1, 2);
        let a_3x1 = ComplexMatrix::new(3, 1);
        let mut v = ComplexVector::new(3);
        let one = cpx!(1.0, 0.0);
        assert_eq!(
            complex_mat_vec_mul(&mut v, one, &a_1x2, &u),
            Err("matrix and vectors are incompatible")
        );
        assert_eq!(
            complex_mat_vec_mul(&mut v, one, &a_3x1, &u),
            Err("matrix and vectors are incompatible")
        );
    }

    #[test]
    fn complex_mat_vec_mul_works() {
        #[rustfmt::skip]
        let a = ComplexMatrix::from(&[
            [ 5.0, -2.0, 0.0, 1.0],
            [10.0, -4.0, 0.0, 2.0],
            [15.0, -6.0, 0.0, 3.0],
        ]);
        let u = ComplexVector::from(&[1.0, 3.0, 8.0, 5.0]);
        let mut v = ComplexVector::new(a.nrow());
        let one = cpx!(1.0, 0.0);
        complex_mat_vec_mul(&mut v, one, &a, &u).unwrap();
        let correct = &[cpx!(4.0, 0.0), cpx!(8.0, 0.0), cpx!(12.0, 0.0)];
        complex_vec_approx_eq(v.as_data(), correct, 1e-15);
    }

    #[test]
    fn complex_mat_vec_mul_zero_works() {
        let a_0x0 = ComplexMatrix::new(0, 0);
        let a_0x1 = ComplexMatrix::new(0, 1);
        let a_1x0 = ComplexMatrix::new(1, 0);
        let u0 = ComplexVector::new(0);
        let u1 = ComplexVector::new(1);
        let mut v0 = ComplexVector::new(0);
        let mut v1 = ComplexVector::new(1);
        let one = cpx!(1.0, 0.0);
        let zero = cpx!(0.0, 0.0);
        complex_mat_vec_mul(&mut v0, one, &a_0x0, &u0).unwrap();
        assert_eq!(v0.as_data(), &[] as &[Complex64]);
        complex_mat_vec_mul(&mut v0, one, &a_0x1, &u1).unwrap();
        assert_eq!(v0.as_data(), &[] as &[Complex64]);
        complex_mat_vec_mul(&mut v1, one, &a_1x0, &u0).unwrap();
        assert_eq!(v1.as_data(), &[zero]);
    }
}
