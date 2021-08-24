use super::*;

#[rustfmt::skip]
extern "C" {
    fn cblas_dgemv(order: i32, trans: i32, m: i32, n: i32, alpha: f64, a: *const f64, lda: i32, x: *const f64, incx: i32, beta: f64, y: *mut f64, incy: i32);
    fn cblas_dger(order: i32, m: i32, n: i32, alpha: f64, x: *const f64, incx: i32, y: *const f64, incy: i32, a: *mut f64, lda: i32);
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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use russell_chk::*;

    #[test]
    fn dger_works() {
        #[rustfmt::skip]
        let mut a = slice_to_colmajor(&[
            &[100.0, 100.0, 100.0],
            &[100.0, 100.0, 100.0],
            &[100.0, 100.0, 100.0],
            &[100.0, 100.0, 100.0],
        ]);
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
        ]);
        assert_vec_approx_eq!(a, correct, 1e-15);
    }

    #[test]
    fn dgemv_works() {
        // allocate matrix
        #[rustfmt::skip]
        let a = slice_to_colmajor(&[
            &[0.1, 0.2, 0.3],
            &[1.0, 0.2, 0.3],
            &[2.0, 0.2, 0.3],
            &[3.0, 0.2, 0.3]
        ]);

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
    }
}
