use super::*;

#[rustfmt::skip]
extern "C" {
    fn cblas_ddot(n: i32, x: *const f64, incx: i32, y: *const f64, incy: i32) -> f64;
    fn cblas_dscal(n: i32, alpha: f64, x: *const f64, incx: i32);
    fn cblas_daxpy(n: i32, alpha: f64, x: *const f64, incx: i32, y: *const f64, incy: i32);
    fn cblas_dgemv(order: i32, trans: i32, m: i32, n: i32, alpha: f64, a: *const f64, lda: i32, x: *const f64, incx: i32, beta: f64, y: *mut f64, incy: i32);
    fn cblas_dger(order: i32, m: i32, n: i32, alpha: f64, x: *const f64, incx: i32, y: *const f64, incy: i32, a: *mut f64, lda: i32);
    fn cblas_dgemm(order: i32, transa: i32, transb: i32, m: i32, n: i32, k: i32, alpha: f64, a: *const f64, lda: i32, b: *const f64, ldb: i32, beta: f64, c: *mut f64, ldc: i32);
}

/// Calculates the dot product of two vectors.
///
/// ```text
/// x dot y
/// ```
///
/// # Note
///
/// Uses unrolled loops for increments equal to one.
///
/// # Reference
///
/// <http://www.netlib.org/lapack/explore-html/d5/df6/ddot_8f.html>
///
pub fn ddot(n: i32, x: &[f64], incx: i32, y: &[f64], incy: i32) -> f64 {
    unsafe { cblas_ddot(n, x.as_ptr(), incx, y.as_ptr(), incy) }
}

/// Scales a vector by a constant.
///
/// ```text
/// x := alpha * x
/// ```
///
/// # Note
///
/// Uses unrolled loops for increment equal to 1.
///
/// # Reference
///
/// <http://www.netlib.org/lapack/explore-html/d4/dd0/dscal_8f.html>
///
pub fn dscal(n: i32, alpha: f64, x: &mut [f64], incx: i32) {
    unsafe {
        cblas_dscal(n, alpha, x.as_ptr(), incx);
    }
}

/// Computes constant times a vector plus a vector.
///
/// ```text
/// y += alpha*x + y
/// ```
///
/// # Reference
///
/// <http://www.netlib.org/lapack/explore-html/d9/dcd/daxpy_8f.html>
///
pub fn daxpy(n: i32, alpha: f64, x: &[f64], incx: i32, y: &mut [f64], incy: i32) {
    unsafe {
        cblas_daxpy(n, alpha, x.as_ptr(), incx, y.as_ptr(), incy);
    }
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

/// Performs one of the matrix-matrix multiplications
///
/// ```text
/// false,false:  C_{m,n} := α ⋅ A_{m,k} ⋅ B_{k,n}  +  β ⋅ C_{m,n}
/// false,true:   C_{m,n} := α ⋅ A_{m,k} ⋅ B_{n,k}  +  β ⋅ C_{m,n}
/// true, false:  C_{m,n} := α ⋅ A_{k,m} ⋅ B_{k,n}  +  β ⋅ C_{m,n}
/// true, true:   C_{m,n} := α ⋅ A_{k,m} ⋅ B_{n,k}  +  β ⋅ C_{m,n}
/// ```
///
/// ```text
/// C := alpha*op(A)*op(B) + beta*C
/// ```
///
/// # Note
///
/// op(X) = X   or   op(X) = X**T
///
/// alpha and beta are scalars, and A, B and C are matrices, with op( A )
/// an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.
///
/// # Reference
///
/// <http://www.netlib.org/lapack/explore-html/d7/d2b/dgemm_8f.html>
///
pub fn dgemm(
    trans_a: bool,
    trans_b: bool,
    m: i32,
    n: i32,
    k: i32,
    alpha: f64,
    a: &[f64],
    lda: i32,
    b: &[f64],
    ldb: i32,
    beta: f64,
    c: &mut [f64],
    ldc: i32,
) {
    unsafe {
        cblas_dgemm(
            CBLAS_COL_MAJOR,
            cblas_transpose(trans_a),
            cblas_transpose(trans_b),
            m,
            n,
            k,
            alpha,
            a.as_ptr(),
            lda,
            b.as_ptr(),
            ldb,
            beta,
            c.as_mut_ptr(),
            ldc,
        );
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use russell_chk::*;

    #[test]
    fn ddot_works() {
        const IGNORED: f64 = 100000.0;
        let x = [20.0, 10.0, 30.0, IGNORED, IGNORED];
        let y = [-15.0, -5.0, -24.0, IGNORED, IGNORED, IGNORED];
        let (n, incx, incy) = (3, 1, 1);
        assert_eq!(ddot(n, &x, incx, &y, incy), -1070.0);
    }

    #[test]
    fn dscal_works() {
        const IGNORED: f64 = 100000.0;
        let alpha = 0.5;
        let mut x = [20.0, 10.0, -30.0, IGNORED, IGNORED];
        let (n, incx) = (3, 1);
        dscal(n, alpha, &mut x, incx);
        assert_vec_approx_eq!(x, &[10.0, 5.0, -15.0, IGNORED, IGNORED], 1e-15);
    }

    #[test]
    fn daxpy_works() {
        const IGNORED: f64 = 100000.0;
        let alpha = 0.5;
        let x = [20.0, 10.0, 48.0, IGNORED, IGNORED];
        let mut y = [-15.0, -5.0, -24.0, IGNORED, IGNORED, IGNORED];
        let (n, incx, incy) = (3, 1, 1);
        daxpy(n, alpha, &x, incx, &mut y, incy);
        assert_vec_approx_eq!(x, &[20.0, 10.0, 48.0, IGNORED, IGNORED], 1e-15);
        assert_vec_approx_eq!(y, &[-5.0, 0.0, 0.0, IGNORED, IGNORED, IGNORED], 1e-15);
    }

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

    #[test]
    fn dgemm_notrans_notrans_works() {
        // 0.5⋅a⋅b + 2⋅c

        // allocate matrices
        #[rustfmt::skip]
        let a = slice_to_colmajor(&[ // 4 x 5
            &[1.0, 2.0,  0.0, 1.0, -1.0],
            &[2.0, 3.0, -1.0, 1.0,  1.0],
            &[1.0, 2.0,  0.0, 4.0, -1.0],
            &[4.0, 0.0,  3.0, 1.0,  1.0],
        ]);
        #[rustfmt::skip]
        let b = slice_to_colmajor(&[ // 5 x 3
            &[1.0, 0.0, 0.0],
            &[0.0, 0.0, 3.0],
            &[0.0, 0.0, 1.0],
            &[1.0, 0.0, 1.0],
            &[0.0, 2.0, 0.0],
        ]);
        #[rustfmt::skip]
        let mut c = slice_to_colmajor(&[ // 4 x 3
            &[ 0.50, 0.0,  0.25],
            &[ 0.25, 0.0, -0.25],
            &[-0.25, 0.0,  0.00],
            &[-0.25, 0.0,  0.00],
        ]);

        // sizes
        let m = 4; // m = nrow(a) = a.M = nrow(c)
        let k = 5; // k = ncol(a) = a.N = nrow(b)
        let n = 3; // n = ncol(b) = b.N = ncol(c)

        // run dgemm
        let (trans_a, trans_b) = (false, false);
        let (alpha, beta) = (0.5, 2.0);
        let (lda, ldb, ldc) = (4, 5, 4);
        dgemm(trans_a, trans_b, m, n, k, alpha, &a, lda, &b, ldb, beta, &mut c, ldc);

        // check
        #[rustfmt::skip]
        let correct = slice_to_colmajor(&[
            &[2.0, -1.0, 4.0],
            &[2.0,  1.0, 4.0],
            &[2.0, -1.0, 5.0],
            &[2.0,  1.0, 2.0],
        ]);
        assert_vec_approx_eq!(c, correct, 1e-15);
    }

    #[test]
    fn dgemm_notrans_trans_works() {
        // 0.5⋅a⋅bᵀ + 2⋅c"

        // allocate matrices
        #[rustfmt::skip]
        let a = slice_to_colmajor(&[ // 4 x 5
            &[1.0, 2.0,  0.0, 1.0, -1.0],
            &[2.0, 3.0, -1.0, 1.0,  1.0],
            &[1.0, 2.0,  0.0, 4.0, -1.0],
            &[4.0, 0.0,  3.0, 1.0,  1.0],
        ]);
        #[rustfmt::skip]
        let b = slice_to_colmajor(&[ // 3 x 5
            &[1.0, 0.0, 0.0, 1.0, 0.0],
            &[0.0, 0.0, 0.0, 0.0, 2.0],
            &[0.0, 3.0, 1.0, 1.0, 0.0],
        ]);
        #[rustfmt::skip]
        let mut c = slice_to_colmajor(&[ // 4 x 3
            &[ 0.50, 0.0,  0.25],
            &[ 0.25, 0.0, -0.25],
            &[-0.25, 0.0,  0.00],
            &[-0.25, 0.0,  0.00],
        ]);

        // sizes
        let m = 4; // m = nrow(a)        = a.M = nrow(c)
        let k = 5; // k = ncol(a)        = a.N = nrow(trans(b))
        let n = 3; // n = ncol(trans(b)) = b.M = ncol(c)

        // run dgemm
        let (trans_a, trans_b) = (false, true);
        let (alpha, beta) = (0.5, 2.0);
        let (lda, ldb, ldc) = (4, 3, 4);
        dgemm(trans_a, trans_b, m, n, k, alpha, &a, lda, &b, ldb, beta, &mut c, ldc);

        // check
        #[rustfmt::skip]
        let correct = slice_to_colmajor(&[
            &[2.0, -1.0, 4.0],
            &[2.0,  1.0, 4.0],
            &[2.0, -1.0, 5.0],
            &[2.0,  1.0, 2.0],
        ]);
        assert_vec_approx_eq!(c, correct, 1e-15);
    }

    #[test]
    fn dgemm_trans_notrans_works() {
        // 0.5⋅aᵀ⋅b + 2⋅c

        // allocate matrices
        #[rustfmt::skip]
        let a = slice_to_colmajor(&[ // 5 x 4
            &[ 1.0,  2.0,  1.0, 4.0],
            &[ 2.0,  3.0,  2.0, 0.0],
            &[ 0.0, -1.0,  0.0, 3.0],
            &[ 1.0,  1.0,  4.0, 1.0],
            &[-1.0,  1.0, -1.0, 1.0],
        ]);
        #[rustfmt::skip]
        let b = slice_to_colmajor(&[ // 5 x 3
            &[1.0, 0.0, 0.0],
            &[0.0, 0.0, 3.0],
            &[0.0, 0.0, 1.0],
            &[1.0, 0.0, 1.0],
            &[0.0, 2.0, 0.0],
        ]);
        #[rustfmt::skip]
        let mut c = slice_to_colmajor(&[ // 4 x 3
            &[ 0.50, 0.0,  0.25],
            &[ 0.25, 0.0, -0.25],
            &[-0.25, 0.0,  0.00],
            &[-0.25, 0.0,  0.00],
        ]);

        // sizes
        let m = 4; // m = nrow(trans(a)) = a.N = nrow(c)
        let k = 5; // k = ncol(trans(a)) = a.M = nrow(trans(b))
        let n = 3; // n = ncol(b)        = b.N = ncol(c)

        // run dgemm
        let (trans_a, trans_b) = (true, false);
        let (alpha, beta) = (0.5, 2.0);
        let (lda, ldb, ldc) = (5, 5, 4);
        dgemm(trans_a, trans_b, m, n, k, alpha, &a, lda, &b, ldb, beta, &mut c, ldc);

        // check
        #[rustfmt::skip]
        let correct = slice_to_colmajor(&[
            &[2.0, -1.0, 4.0],
            &[2.0,  1.0, 4.0],
            &[2.0, -1.0, 5.0],
            &[2.0,  1.0, 2.0],
        ]);
        assert_vec_approx_eq!(c, correct, 1e-15);
    }

    #[test]
    fn dgemm_trans_trans_works() {
        // 0.5⋅aᵀ⋅bᵀ + 2⋅c

        // allocate matrices
        #[rustfmt::skip]
        let a = slice_to_colmajor(&[ // 5 x 4
            &[ 1.0,  2.0,  1.0, 4.0],
            &[ 2.0,  3.0,  2.0, 0.0],
            &[ 0.0, -1.0,  0.0, 3.0],
            &[ 1.0,  1.0,  4.0, 1.0],
            &[-1.0,  1.0, -1.0, 1.0],
        ]);
        #[rustfmt::skip]
        let b = slice_to_colmajor(&[ // 3 x 5
            &[1.0, 0.0, 0.0, 1.0, 0.0],
            &[0.0, 0.0, 0.0, 0.0, 2.0],
            &[0.0, 3.0, 1.0, 1.0, 0.0],
        ]);
        #[rustfmt::skip]
        let mut c = slice_to_colmajor(&[ // 4 x 3
            &[ 0.50, 0.0,  0.25],
            &[ 0.25, 0.0, -0.25],
            &[-0.25, 0.0,  0.00],
            &[-0.25, 0.0,  0.00],
        ]);

        // sizes
        let m = 4; // m = nrow(trans(a)) = a.N = nrow(c)
        let k = 5; // k = ncol(trans(a)) = a.M = nrow(trans(b))
        let n = 3; // n = ncol(trans(b)) = b.M = ncol(c)

        // run dgemm
        let (trans_a, trans_b) = (true, true);
        let (alpha, beta) = (0.5, 2.0);
        let (lda, ldb, ldc) = (5, 3, 4);
        dgemm(trans_a, trans_b, m, n, k, alpha, &a, lda, &b, ldb, beta, &mut c, ldc);

        // check
        #[rustfmt::skip]
        let correct = slice_to_colmajor(&[
            &[2.0, -1.0, 4.0],
            &[2.0,  1.0, 4.0],
            &[2.0, -1.0, 5.0],
            &[2.0,  1.0, 2.0],
        ]);
        assert_vec_approx_eq!(c, correct, 1e-15);
    }
}
