use super::*;

#[rustfmt::skip]
extern "C" {
    fn cblas_dgemv(order: i32, trans: i32, m: i32, n: i32, alpha: f64, a: *const f64, lda: i32, x: *const f64, incx: i32, beta: f64, y: *mut f64, incy: i32);
    fn cblas_dger(order: i32, m: i32, n: i32, alpha: f64, x: *const f64, incx: i32, y: *const f64, incy: i32, a: *mut f64, lda: i32);
    fn LAPACKE_dgesv(matrix_layout: i32, n: i32, nrhs: i32, a: *mut f64, lda: i32, ipiv: *mut i32, b: *mut f64, ldb: i32) -> i32;
}

/// Performs the rank 1 operation (tensor product)
///
/// ```text
/// A := alpha*x*y**T + A
/// ```
///
/// # Note
///
/// alpha is a scalar, x is an m element vector, y is an n element
/// vector and A is an m by n matrix.
///
/// # Reference
///
/// <http://www.netlib.org/lapack/explore-html/dc/da8/dger_8f.html>
///
#[inline]
pub fn dger(m: i32, n: i32, alpha: f64, x: &[f64], incx: i32, y: &[f64], incy: i32, a: &mut [f64], lda: i32) {
    unsafe {
        cblas_dger(
            CBLAS_COL_MAJOR,
            m,
            n,
            alpha,
            x.as_ptr(),
            incx,
            y.as_ptr(),
            incy,
            a.as_mut_ptr(),
            lda,
        );
    }
}

/// Performs one of the matrix-vector multiplication
///
/// ```text
/// y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,
/// ```
///
/// # Note
///
/// alpha and beta are scalars, x and y are vectors and A is an m by n matrix.
///
/// ```text
/// trans=false     y := alpha*A*x + beta*y.
///
/// trans=true      y := alpha*A**T*x + beta*y.
/// ```
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
    lda: i32,
    x: &[f64],
    incx: i32,
    beta: f64,
    y: &mut [f64],
    incy: i32,
) {
    unsafe {
        cblas_dgemv(
            CBLAS_COL_MAJOR,
            cblas_transpose(trans),
            m,
            n,
            alpha,
            a.as_ptr(),
            lda,
            x.as_ptr(),
            incx,
            beta,
            y.as_mut_ptr(),
            incy,
        );
    }
}

/// Computes the solution to a real system of linear equations.
///
/// The system is:
///
/// ```text
///    A * X = B,
/// ```
/// where A is an N-by-N matrix and X and B are N-by-NRHS matrices.
///
/// The LU decomposition with partial pivoting and row interchanges is
/// used to factor A as
///
/// ```text
///    A = P * L * U,
/// ```
///
/// where P is a permutation matrix, L is unit lower triangular, and U is
/// upper triangular.  The factored form of A is then used to solve the
/// system of equations A * X = B.
///
/// # Note
///
/// 1. The length of ipiv must be equal to `n`
/// 2. The matrix will be modified
///
/// # Reference
///
/// <http://www.netlib.org/lapack/explore-html/d8/d72/dgesv_8f.html>
///
#[inline]
pub fn dgesv(
    n: i32,
    nrhs: i32,
    a: &mut [f64],
    lda: i32,
    ipiv: &mut [i32],
    b: &mut [f64],
    ldb: i32,
) -> Result<(), &'static str> {
    unsafe {
        let ipiv_len: i32 = to_i32(ipiv.len());
        if ipiv_len != n {
            return Err("the length of ipiv must equal n");
        }
        let info = LAPACKE_dgesv(
            LAPACK_COL_MAJOR,
            n,
            nrhs,
            a.as_mut_ptr(),
            lda,
            ipiv.as_mut_ptr(),
            b.as_mut_ptr(),
            ldb,
        );
        if info != 0_i32 {
            return Err("LAPACK failed");
        }
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use russell_chk::*;

    #[test]
    fn dger_works() -> Result<(), &'static str> {
        #[rustfmt::skip]
        let mut a = slice_to_colmajor(&[
            &[100.0, 100.0, 100.0],
            &[100.0, 100.0, 100.0],
            &[100.0, 100.0, 100.0],
            &[100.0, 100.0, 100.0],
        ])?;
        let u = &[1.0, 2.0, 3.0, 4.0];
        let v = &[4.0, 3.0, 2.0];
        let m = 4; // m = nrow(a) = len(u)
        let n = 3; // n = ncol(a) = len(v)
        let lda = 4;
        let alpha = 0.5;
        dger(m, n, alpha, u, 1, v, 1, &mut a, lda);
        // a = 100 + 0.5⋅u⋅vᵀ
        let correct = slice_to_colmajor(&[
            &[102.0, 101.5, 101.0],
            &[104.0, 103.0, 102.0],
            &[106.0, 104.5, 103.0],
            &[108.0, 106.0, 104.0],
        ])?;
        assert_vec_approx_eq!(a, correct, 1e-15);
        Ok(())
    }

    #[test]
    fn dgemv_works() -> Result<(), &'static str> {
        // allocate matrix
        #[rustfmt::skip]
        let a = slice_to_colmajor(&[
            &[0.1, 0.2, 0.3],
            &[1.0, 0.2, 0.3],
            &[2.0, 0.2, 0.3],
            &[3.0, 0.2, 0.3]
        ])?;

        // perform mv
        let (m, n) = (4, 3);
        let (alpha, beta) = (0.5, 2.0);
        let mut x = [20.0, 10.0, 30.0];
        let mut y = [3.0, 1.0, 2.0, 4.0];
        let (lda, incx, incy) = (m, 1, 1);
        dgemv(false, m, n, alpha, &a, lda, &x, incx, beta, &mut y, incy);
        assert_vec_approx_eq!(y, &[12.5, 17.5, 29.5, 43.5], 1e-15);

        // perform mv with transpose
        dgemv(true, m, n, alpha, &a, lda, &y, incy, beta, &mut x, incx);
        assert_vec_approx_eq!(x, &[144.125, 30.3, 75.45], 1e-15);

        // check that a is unmodified
        assert_vec_approx_eq!(a, &[0.1, 1.0, 2.0, 3.0, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3], 1e-15);
        Ok(())
    }

    #[test]
    fn dgesv_works() -> Result<(), &'static str> {
        // matrix
        #[rustfmt::skip]
        let mut a = slice_to_colmajor(&[
            &[2.0,  3.0,  0.0, 0.0, 0.0],
            &[3.0,  0.0,  4.0, 0.0, 6.0],
            &[0.0, -1.0, -3.0, 2.0, 0.0],
            &[0.0,  0.0,  1.0, 0.0, 0.0],
            &[0.0,  4.0,  2.0, 0.0, 1.0],
        ])?;

        // right-hand-side
        let mut b = vec![8.0, 45.0, -3.0, 3.0, 19.0];

        // solve b := x := A⁻¹ b
        let (n, lda, ldb, nrhs) = (5_i32, 5_i32, 5_i32, 1_i32);
        let mut ipiv = vec![0; n as usize];
        dgesv(n, nrhs, &mut a, lda, &mut ipiv, &mut b, ldb)?;

        // check
        let correct = &[1.0, 2.0, 3.0, 4.0, 5.0];
        assert_vec_approx_eq!(b, correct, 1e-15);
        Ok(())
    }
}
