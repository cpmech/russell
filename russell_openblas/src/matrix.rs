use super::*;

#[rustfmt::skip]
extern "C" {
    fn cblas_dgemm(order: i32, transa: i32, transb: i32, m: i32, n: i32, k: i32, alpha: f64, a: *const f64, lda: i32, b: *const f64, ldb: i32, beta: f64, c: *mut f64, ldc: i32);
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
#[inline]
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
