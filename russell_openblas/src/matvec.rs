use super::{cblas_transpose, to_i32, CBLAS_ROW_MAJOR, LAPACK_ROW_MAJOR};
use crate::StrError;
use num_complex::Complex64;

#[rustfmt::skip]
extern "C" {
    // from /usr/include/x86_64-linux-gnu/cblas.h
    fn cblas_dgemv(order: i32, trans: i32, m: i32, n: i32, alpha: f64, a: *const f64, lda: i32, x: *const f64, incx: i32, beta: f64, y: *mut f64, incy: i32);
    fn cblas_zgemv(order: i32, trans: i32, m: i32, n: i32, alpha: *const Complex64, a: *const Complex64, lda: i32, x: *const Complex64, incx: i32, beta: *const Complex64, y: *mut Complex64, incy: i32);
    fn cblas_dger(order: i32, m: i32, n: i32, alpha: f64, x: *const f64, incx: i32, y: *const f64, incy: i32, a: *mut f64, lda: i32);
    // from /usr/include/lapacke.h
    fn LAPACKE_dgesv(matrix_layout: i32, n: i32, nrhs: i32, a: *mut f64, lda: i32, ipiv: *mut i32, b: *mut f64, ldb: i32) -> i32;
    fn LAPACKE_zgesv(matrix_layout: i32, n: i32, nrhs: i32, a: *mut Complex64, lda: i32, ipiv: *mut i32, b: *mut Complex64, ldb: i32) -> i32;
}

/// Performs the rank 1 operation (tensor product)
///
/// ```text
///   a := α ⋅ x ⋅ yᵀ +  a
/// (m,n)     (m) (n)  (m,n)
/// ```
///
/// # Important
///
/// * The data must be in **row-major** order
///
/// # Reference
///
/// <http://www.netlib.org/lapack/explore-html/dc/da8/dger_8f.html>
///
#[inline]
pub fn dger(m: i32, n: i32, alpha: f64, x: &[f64], incx: i32, y: &[f64], incy: i32, a: &mut [f64]) {
    unsafe {
        cblas_dger(
            CBLAS_ROW_MAJOR,
            m,
            n,
            alpha,
            x.as_ptr(),
            incx,
            y.as_ptr(),
            incy,
            a.as_mut_ptr(),
            n,
        );
    }
}

/// Performs one of the matrix-vector multiplication
///
/// ```text
///  y := α ⋅ a  ⋅ x  +  β ⋅ y
/// (m)     (m,n) (n)       (m)
///
/// or
///
///  y := α ⋅  aᵀ ⋅ x  +  β ⋅ y
/// (m)      (m,n) (n)       (m)
/// ```
///
/// # Important
///
/// * The data must be in **row-major** order
///
/// # Reference
///
/// <http://www.netlib.org/lapack/explore-html/dc/da8/dgemv_8f.html>
///
#[inline]
pub fn dgemv(
    trans: bool,
    m: i32,
    n: i32,
    alpha: f64,
    a: &[f64],
    x: &[f64],
    incx: i32,
    beta: f64,
    y: &mut [f64],
    incy: i32,
) {
    unsafe {
        cblas_dgemv(
            CBLAS_ROW_MAJOR,
            cblas_transpose(trans),
            m,
            n,
            alpha,
            a.as_ptr(),
            n,
            x.as_ptr(),
            incx,
            beta,
            y.as_mut_ptr(),
            incy,
        );
    }
}

/// Performs one of the matrix-vector multiplication (complex version)
///
/// ```text
///  y := α ⋅ a  ⋅ x  +  β ⋅ y
/// (m)     (m,n) (n)       (m)
///
/// or
///
///  y := α ⋅  aᵀ ⋅ x  +  β ⋅ y
/// (m)      (m,n) (n)       (m)
/// ```
///
/// # Important
///
/// * The data must be in **row-major** order
///
/// # Reference
///
/// <http://www.netlib.org/lapack/explore-html/db/d40/zgemv_8f.html>
///
#[inline]
pub fn zgemv(
    trans: bool,
    m: i32,
    n: i32,
    alpha: Complex64,
    a: &[Complex64],
    x: &[Complex64],
    incx: i32,
    beta: Complex64,
    y: &mut [Complex64],
    incy: i32,
) {
    unsafe {
        cblas_zgemv(
            CBLAS_ROW_MAJOR,
            cblas_transpose(trans),
            m,
            n,
            &alpha,
            a.as_ptr(),
            n,
            x.as_ptr(),
            incx,
            &beta,
            y.as_mut_ptr(),
            incy,
        );
    }
}

/// Computes the solution to a real system of linear equations
///
/// The system is:
///
/// ```text
///   A  ⋅  X =   B
/// (n,n)  (n)  (n,nrhs)
/// ```
///
/// where A is an N-by-N matrix and X and B are N-by-NRHS matrices.
///
/// The LU decomposition with partial pivoting and row interchanges is
/// used to factor A as
///
/// ```text
/// A = P ⋅ L ⋅ U,
/// ```
///
/// where P is a permutation matrix, L is unit lower triangular, and U is
/// upper triangular. The factored form of A is then used to solve the
/// system of equations A * X = B.
///
/// # Note
///
/// 1. The length of ipiv must be equal to `n`
/// 2. The matrix will be modified
///
/// # Important
///
/// * The data must be in **row-major** order
///
/// # Reference
///
/// <http://www.netlib.org/lapack/explore-html/d8/d72/dgesv_8f.html>
///
#[inline]
pub fn dgesv(n: i32, nrhs: i32, a: &mut [f64], ipiv: &mut [i32], b: &mut [f64]) -> Result<(), StrError> {
    unsafe {
        let ipiv_len: i32 = to_i32(ipiv.len());
        if ipiv_len != n {
            return Err("the length of ipiv must equal n");
        }
        let info = LAPACKE_dgesv(
            LAPACK_ROW_MAJOR,
            n,
            nrhs,
            a.as_mut_ptr(),
            n,
            ipiv.as_mut_ptr(),
            b.as_mut_ptr(),
            nrhs,
        );
        if info != 0_i32 {
            return Err("LAPACK dgesv failed");
        }
    }
    Ok(())
}

/// Computes the solution to a real system of linear equations (complex version)
///
/// The system is:
///
/// ```text
///   A  ⋅  X =   B
/// (n,n)  (n)  (n,nrhs)
/// ```
///
/// where A is an N-by-N matrix and X and B are N-by-NRHS matrices.
///
/// The LU decomposition with partial pivoting and row interchanges is
/// used to factor A as
///
/// ```text
/// A = P ⋅ L ⋅ U,
/// ```
///
/// where P is a permutation matrix, L is unit lower triangular, and U is
/// upper triangular. The factored form of A is then used to solve the
/// system of equations A * X = B.
///
/// # Note
///
/// 1. The length of ipiv must be equal to `n`
/// 2. The matrix will be modified
///
/// # Important
///
/// * The data must be in **row-major** order
///
/// # Reference
///
/// <http://www.netlib.org/lapack/explore-html/d1/ddc/zgesv_8f.html>
///
#[inline]
pub fn zgesv(n: i32, nrhs: i32, a: &mut [Complex64], ipiv: &mut [i32], b: &mut [Complex64]) -> Result<(), StrError> {
    unsafe {
        let ipiv_len: i32 = to_i32(ipiv.len());
        if ipiv_len != n {
            return Err("the length of ipiv must equal n");
        }
        let info = LAPACKE_zgesv(
            LAPACK_ROW_MAJOR,
            n,
            nrhs,
            a.as_mut_ptr(),
            n,
            ipiv.as_mut_ptr(),
            b.as_mut_ptr(),
            nrhs,
        );
        if info != 0_i32 {
            return Err("LAPACK zgesv failed");
        }
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{dgemv, dger, dgesv, zgemv, zgesv};
    use crate::{to_i32, StrError};
    use num_complex::Complex64;
    use russell_chk::{assert_complex_vec_approx_eq, assert_vec_approx_eq};

    #[test]
    fn dger_works() {
        #[rustfmt::skip]
        let mut a = [
            100.0, 100.0, 100.0,
            100.0, 100.0, 100.0,
            100.0, 100.0, 100.0,
            100.0, 100.0, 100.0,
        ];
        let u = &[1.0, 2.0, 3.0, 4.0];
        let v = &[4.0, 3.0, 2.0];
        let m = 4; // m = nrow(a) = len(u)
        let n = 3; // n = ncol(a) = len(v)
        let alpha = 0.5;
        dger(m, n, alpha, u, 1, v, 1, &mut a);
        // a = 100 + 0.5⋅u⋅vᵀ
        #[rustfmt::skip]
        let correct = [
            102.0, 101.5, 101.0,
            104.0, 103.0, 102.0,
            106.0, 104.5, 103.0,
            108.0, 106.0, 104.0,
        ];
        assert_vec_approx_eq!(a, correct, 1e-15);
    }

    #[test]
    fn dgemv_works() {
        // allocate matrix
        #[rustfmt::skip]
        let a = [
            0.1, 0.2, 0.3,
            1.0, 0.2, 0.3,
            2.0, 0.2, 0.3,
            3.0, 0.2, 0.3,
        ];

        // perform mv
        let (alpha, beta) = (0.5, 2.0);
        let mut x = [20.0, 10.0, 30.0];
        let mut y = [3.0, 1.0, 2.0, 4.0];
        dgemv(false, 4, 3, alpha, &a, &x, 1, beta, &mut y, 1);
        assert_vec_approx_eq!(y, &[12.5, 17.5, 29.5, 43.5], 1e-15);

        // perform mv with transpose
        dgemv(true, 4, 3, alpha, &a, &y, 1, beta, &mut x, 1);
        assert_vec_approx_eq!(x, &[144.125, 30.3, 75.45], 1e-15);

        // check that a is unmodified
        assert_vec_approx_eq!(a, &[0.1, 0.2, 0.3, 1.0, 0.2, 0.3, 2.0, 0.2, 0.3, 3.0, 0.2, 0.3], 1e-15);
    }

    #[test]
    fn zgemv_works() {
        // allocate matrix
        #[rustfmt::skip]
        let a = [
            Complex64::new(0.1, 3.0), Complex64::new(0.2, 0.0), Complex64::new(0.3, -0.3),
            Complex64::new(1.0, 2.0), Complex64::new(0.2, 0.0), Complex64::new(0.3, -0.4),
            Complex64::new(2.0, 1.0), Complex64::new(0.2, 0.0), Complex64::new(0.3, -0.5),
            Complex64::new(3.0, 0.1), Complex64::new(0.2, 0.0), Complex64::new(0.3, -0.6),
        ];
        let a_clone = a.clone();

        // perform mv
        let (alpha, beta) = (Complex64::new(0.5, 1.0), Complex64::new(2.0, 1.0));
        let mut x = [
            Complex64::new(20.0, 0.0),
            Complex64::new(10.0, 0.0),
            Complex64::new(30.0, 0.0),
        ];
        let mut y = [
            Complex64::new(3.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(4.0, 0.0),
        ];
        zgemv(false, 4, 3, alpha, &a, &x, 1, beta, &mut y, 1);
        let y_correct = [
            Complex64::new(-38.5, 41.5),
            Complex64::new(-10.5, 46.0),
            Complex64::new(24.5, 55.5),
            Complex64::new(59.5, 67.0),
        ];
        assert_complex_vec_approx_eq!(y, y_correct, 1e-15);

        // perform mv with transpose
        zgemv(true, 4, 3, alpha, &a, &y, 1, beta, &mut x, 1);
        let x_correct = [
            Complex64::new(-248.875, 82.5),
            Complex64::new(-18.5, 38.0),
            Complex64::new(83.85, 154.7),
        ];
        assert_complex_vec_approx_eq!(x, x_correct, 1e-13);

        // check that a is unmodified
        assert_complex_vec_approx_eq!(a, a_clone, 1e-15);
    }

    #[test]
    fn dgesv_fails() {
        let m = 2;
        let mut a = vec![0.0; m * m];
        let mut b = vec![0.0; m];
        let mut ipiv = vec![0; m];
        let m_i32 = to_i32(m);
        let nrhs = 1_i32;
        assert_eq!(
            dgesv(m_i32, nrhs, &mut a, &mut ipiv, &mut b),
            Err("LAPACK dgesv failed")
        );
    }

    #[test]
    fn dgesv_fails_on_wrong_ipiv() {
        let m = 2;
        let mut a = [1.0, 0.0, 0.0, 1.0];
        let mut b = vec![0.0; m];
        let mut ipiv = vec![0; 1]; // << ERROR
        let m_i32 = to_i32(m);
        let nrhs = 1_i32;
        assert_eq!(
            dgesv(m_i32, nrhs, &mut a, &mut ipiv, &mut b),
            Err("the length of ipiv must equal n")
        );
    }

    #[test]
    fn dgesv_works() -> Result<(), StrError> {
        // matrix
        #[rustfmt::skip]
        let mut a = [
            2.0,  3.0,  0.0, 0.0, 0.0,
            3.0,  0.0,  4.0, 0.0, 6.0,
            0.0, -1.0, -3.0, 2.0, 0.0,
            0.0,  0.0,  1.0, 0.0, 0.0,
            0.0,  4.0,  2.0, 0.0, 1.0,
        ];

        // right-hand-side
        let mut b = vec![8.0, 45.0, -3.0, 3.0, 19.0];

        // solve b := x := A⁻¹ b
        let (n, nrhs) = (5_i32, 1_i32);
        let mut ipiv = vec![0; n as usize];
        dgesv(n, nrhs, &mut a, &mut ipiv, &mut b)?;

        // check
        let correct = &[1.0, 2.0, 3.0, 4.0, 5.0];
        assert_vec_approx_eq!(b, correct, 1e-14);
        Ok(())
    }

    #[test]
    fn zgesv_fails() {
        let m = 2;
        let mut a = vec![Complex64::new(0.0, 0.0); m * m];
        let mut b = vec![Complex64::new(0.0, 0.0); m];
        let mut ipiv = vec![0; m];
        let m_i32 = to_i32(m);
        let nrhs = 1_i32;
        assert_eq!(
            zgesv(m_i32, nrhs, &mut a, &mut ipiv, &mut b),
            Err("LAPACK zgesv failed")
        );
    }

    #[test]
    fn zgesv_fails_on_wrong_ipiv() {
        let m = 2;
        let mut a = [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];
        let mut b = vec![Complex64::new(0.0, 0.0); m];
        let mut ipiv = vec![0; 1]; // << ERROR
        let m_i32 = to_i32(m);
        let nrhs = 1_i32;
        assert_eq!(
            zgesv(m_i32, nrhs, &mut a, &mut ipiv, &mut b),
            Err("the length of ipiv must equal n")
        );
    }

    #[test]
    fn zgesv_works_1() -> Result<(), StrError> {
        // matrix
        #[rustfmt::skip]
        let mut a = [
            Complex64::new(2.0, 0.0), Complex64::new( 3.0, 0.0), Complex64::new( 0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(3.0, 0.0), Complex64::new( 0.0, 0.0), Complex64::new( 4.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(6.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0), Complex64::new(-3.0, 0.0), Complex64::new(2.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new( 0.0, 0.0), Complex64::new( 1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new( 4.0, 0.0), Complex64::new( 2.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
        ];

        // right-hand-side
        let mut b = vec![
            Complex64::new(8.0, 0.0),
            Complex64::new(45.0, 0.0),
            Complex64::new(-3.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(19.0, 0.0),
        ];

        // solve b := x := A⁻¹ b
        let (n, nrhs) = (5_i32, 1_i32);
        let mut ipiv = vec![0; n as usize];
        zgesv(n, nrhs, &mut a, &mut ipiv, &mut b)?;

        // check
        let correct = &[
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
            Complex64::new(5.0, 0.0),
        ];
        assert_complex_vec_approx_eq!(b, correct, 1e-14);
        Ok(())
    }

    #[test]
    fn zgesv_works_2() -> Result<(), StrError> {
        // NOTE: zgesv performs poorly in this problem.
        // The same problem happens in python (likely using lapack too)

        // matrix
        #[rustfmt::skip]
        let mut a = [
            Complex64::new(19.730,  0.000), Complex64::new(12.110, - 1.000), Complex64::new( 0.000, 5.000), Complex64::new( 0.000,  0.000), Complex64::new( 0.000, 0.000),
            Complex64::new( 0.000, -0.510), Complex64::new(32.300,   7.000), Complex64::new(23.070, 0.000), Complex64::new( 0.000,  1.000), Complex64::new( 0.000, 0.000),
            Complex64::new( 0.000,  0.000), Complex64::new( 0.000, - 0.510), Complex64::new(70.000, 7.300), Complex64::new( 3.950,  0.000), Complex64::new(19.000, 31.830),
            Complex64::new( 0.000,  0.000), Complex64::new( 0.000,   0.000), Complex64::new( 1.000, 1.100), Complex64::new(50.170,  0.000), Complex64::new(45.510, 0.000),
            Complex64::new( 0.000,  0.000), Complex64::new( 0.000,   0.000), Complex64::new( 0.000, 0.000), Complex64::new( 0.000, -9.351), Complex64::new(55.000, 0.000),
        ];

        // right-hand-side
        let mut b = [
            Complex64::new(77.38, 8.82),
            Complex64::new(157.48, 19.8),
            Complex64::new(1175.62, 20.69),
            Complex64::new(912.12, -801.75),
            Complex64::new(550.00, -1060.4),
        ];

        // solution
        let x_correct = [
            Complex64::new(3.3, -1.00),
            Complex64::new(1.0, 0.17),
            Complex64::new(5.5, 0.00),
            Complex64::new(9.0, 0.00),
            Complex64::new(10.0, -17.75),
        ];

        // run test
        // solve b := x := A⁻¹ b
        let (n, nrhs) = (5_i32, 1_i32);
        let mut ipiv = vec![0; n as usize];
        zgesv(n, nrhs, &mut a, &mut ipiv, &mut b)?;
        assert_complex_vec_approx_eq!(b, x_correct, 0.00049);

        // compare with python results
        let x_python = [
            Complex64::new(3.299687426933794e+00, -1.000372829305209e+00),
            Complex64::new(9.997606020636992e-01, 1.698383755401385e-01),
            Complex64::new(5.500074759292877e+00, -4.556001293922560e-05),
            Complex64::new(8.999787912842375e+00, -6.662818244209770e-05),
            Complex64::new(1.000001132800243e+01, -1.774987242230929e+01),
        ];
        assert_complex_vec_approx_eq!(b, x_python, 1e-13);

        // check ipiv
        assert_eq!(ipiv, [1, 2, 3, 4, 5]);
        Ok(())
    }
}
