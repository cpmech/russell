use super::*;
use std::convert::TryInto;

#[rustfmt::skip]
extern "C" {
    fn cblas_dgemm(order: i32, transa: i32, transb: i32, m: i32, n: i32, k: i32, alpha: f64, a: *const f64, lda: i32, b: *const f64, ldb: i32, beta: f64, c: *mut f64, ldc: i32);
    fn LAPACKE_dgesv(matrix_layout: i32, n: i32, nrhs: i32, a: *mut f64, lda: i32, ipiv: *mut i32, b: *mut f64, ldb: i32) -> i32;
    fn LAPACKE_dgesvd(matrix_layout: i32, jobu: u8, jobvt: u8, m: i32, n: i32, a: *mut f64, lda: i32, s: *mut f64, u: *mut f64, ldu: i32, vt: *mut f64, ldvt: i32, superb: *mut f64) -> i32;
    
    // fn cblas_dsyrk(order: i32, uplo: i32, trans: i32, n: i32, k: i32, alpha: f64, a: *mut f64, lda: i32, beta: f64, c: *mut f64, ldc: i32);
    // fn LAPACKE_dgetrf(matrix_layout: i32, m: i32, n: i32, a: *mut f64, lda: i32, ipiv: *mut i32) -> i32;
    // fn LAPACKE_dgetri(matrix_layout: i32, n: i32, a: *mut f64, lda: i32, ipiv: *const i32) -> i32;
    // fn LAPACKE_dgeev(matrix_layout: i32, jobvl: char, jobvr: char, n: i32, a: *mut f64, lda: i32, wr: *mut f64, wi: *mut f64, vl: *mut f64, ldvl: i32, vr: *mut f64, ldvr: i32) -> i32;
    // fn LAPACKE_dpotrf(matrix_layout: i32, uplo: char, n: i32, a: *mut f64, lda: i32) -> i32;
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
        let ipiv_len: i32 = ipiv.len().try_into().unwrap();
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

/// Computes the singular value decomposition (SVD) of a real M-by-N matrix A, optionally computing the left and/or right singular vectors.
///
/// The SVD is written
///
/// ```text
///    A = U * SIGMA * transpose(V)
/// ```
///
/// where SIGMA is an M-by-N matrix which is zero except for its
/// min(m,n) diagonal elements, U is an M-by-M orthogonal matrix, and
/// V is an N-by-N orthogonal matrix.  The diagonal elements of SIGMA
/// are the singular values of A; they are real and non-negative, and
/// are returned in descending order.  The first min(m,n) columns of
/// U and V are the left and right singular vectors of A.
///
/// # Note
///
/// 1. The routine returns V**T, not V.
/// 2. The matrix will be modified
/// 3. jobu and jobvt are c_char and can be passed as b'A'
///    (see LAPACK reference for further options)
///
/// # Reference
///
/// <http://www.netlib.org/lapack/explore-html/d8/d2d/dgesvd_8f.html>
///
pub fn dgesvd(
    jobu: u8,
    jobvt: u8,
    m: i32,
    n: i32,
    a: &mut [f64],
    lda: i32,
    s: &mut [f64],
    u: &mut [f64],
    ldu: i32,
    vt: &mut [f64],
    ldvt: i32,
    superb: &mut [f64],
) -> Result<(), &'static str> {
    unsafe {
        let info = LAPACKE_dgesvd(
            LAPACK_COL_MAJOR,
            jobu,
            jobvt,
            m,
            n,
            a.as_mut_ptr(),
            lda,
            s.as_mut_ptr(),
            u.as_mut_ptr(),
            ldu,
            vt.as_mut_ptr(),
            ldvt,
            superb.as_mut_ptr(),
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
        ]);

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

    #[test]
    fn dgesvd_works() -> Result<(), &'static str> {
        // matrix
        #[rustfmt::skip]
        let mut a = slice_to_colmajor(&[
            &[1.0, 0.0, 0.0, 0.0, 2.0],
            &[0.0, 0.0, 3.0, 0.0, 0.0],
            &[0.0, 0.0, 0.0, 0.0, 0.0],
            &[0.0, 2.0, 0.0, 0.0, 0.0],
        ]);
        let a_copy = a.to_vec();

        // dimensions
        let (m, n) = (4_usize, 5_usize);
        let min_mn = if m < n { m } else { n };
        let (lda, ldu, ldvt) = (m, m, n);

        // allocate output arrays
        let mut s = vec![0.0; min_mn as usize];
        let mut u = vec![0.0; (m * m) as usize];
        let mut vt = vec![0.0; (n * n) as usize];
        let mut superb = vec![0.0; min_mn as usize];

        // perform SVD
        let (jobu, jobvt) = (b'A', b'A');
        dgesvd(
            jobu,
            jobvt,
            m.try_into().unwrap(),
            n.try_into().unwrap(),
            &mut a,
            lda.try_into().unwrap(),
            &mut s,
            &mut u,
            ldu.try_into().unwrap(),
            &mut vt,
            ldvt.try_into().unwrap(),
            &mut superb,
        )?;

        // check
        #[rustfmt::skip]
        let u_correct = slice_to_colmajor(&[
            &[0.0, 1.0, 0.0,  0.0],
            &[1.0, 0.0, 0.0,  0.0],
            &[0.0, 0.0, 0.0, -1.0],
            &[0.0, 0.0, 1.0,  0.0],
        ]);
        let s_correct = &[3.0, f64::sqrt(5.0), 2.0, 0.0];
        let s2 = f64::sqrt(0.2);
        let s8 = f64::sqrt(0.8);
        #[rustfmt::skip]
        let vt_correct = slice_to_colmajor(&[
            &[0.0, 0.0, 1.0, 0.0, 0.0],
            &[ s2, 0.0, 0.0, 0.0,  s8],
            &[0.0, 1.0, 0.0, 0.0, 0.0],
            &[0.0, 0.0, 0.0, 1.0, 0.0],
            &[-s8, 0.0, 0.0, 0.0,  s2],
        ]);
        assert_vec_approx_eq!(u, u_correct, 1e-15);
        assert_vec_approx_eq!(s, s_correct, 1e-15);
        assert_vec_approx_eq!(vt, vt_correct, 1e-15);

        // check SVD
        let mut usv = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                for k in 0..min_mn {
                    usv[i + j * m] += u[i + k * m] * s[k] * vt[k + j * n];
                }
            }
        }
        assert_vec_approx_eq!(usv, a_copy, 1e-15);
        Ok(())
    }

    #[test]
    fn dgesvd_1_works() -> Result<(), &'static str> {
        // matrix
        let s33 = f64::sqrt(3.0) / 3.0;
        #[rustfmt::skip]
        let mut a = slice_to_colmajor(&[
            &[-s33, -s33, 1.0],
            &[ s33, -s33, 1.0],
            &[-s33,  s33, 1.0],
            &[ s33,  s33, 1.0],
        ]);
        let a_copy = a.to_vec();

        // dimensions
        let (m, n) = (4_usize, 3_usize);
        let min_mn = if m < n { m } else { n };
        let (lda, ldu, ldvt) = (m, m, n);

        // allocate output arrays
        let mut s = vec![0.0; min_mn as usize];
        let mut u = vec![0.0; (m * m) as usize];
        let mut vt = vec![0.0; (n * n) as usize];
        let mut superb = vec![0.0; min_mn as usize];

        // perform SVD
        let (jobu, jobvt) = (b'A', b'A');
        dgesvd(
            jobu,
            jobvt,
            m.try_into().unwrap(),
            n.try_into().unwrap(),
            &mut a,
            lda.try_into().unwrap(),
            &mut s,
            &mut u,
            ldu.try_into().unwrap(),
            &mut vt,
            ldvt.try_into().unwrap(),
            &mut superb,
        )?;

        // check
        #[rustfmt::skip]
        let u_correct = slice_to_colmajor(&[
            &[-0.5, -0.5, -0.5,  0.5],
            &[-0.5, -0.5,  0.5, -0.5],
            &[-0.5,  0.5, -0.5, -0.5],
            &[-0.5,  0.5,  0.5,  0.5],
        ]);
        let s_correct = &[2.0, 2.0 / f64::sqrt(3.0), 2.0 / f64::sqrt(3.0)];
        #[rustfmt::skip]
        let vt_correct = slice_to_colmajor(&[
            &[0.0, 0.0, -1.0],
            &[0.0, 1.0,  0.0],
            &[1.0, 0.0,  0.0],
        ]);
        assert_vec_approx_eq!(u, u_correct, 1e-15);
        assert_vec_approx_eq!(s, s_correct, 1e-15);
        assert_vec_approx_eq!(vt, vt_correct, 1e-15);

        // check SVD
        let mut usv = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                for k in 0..min_mn {
                    usv[i + j * m] += u[i + k * m] * s[k] * vt[k + j * n];
                }
            }
        }
        assert_vec_approx_eq!(usv, a_copy, 1e-15);
        Ok(())
    }
}
