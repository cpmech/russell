use super::*;
use std::convert::TryInto;

#[rustfmt::skip]
extern "C" {
    fn cblas_dgemm(order: i32, transa: i32, transb: i32, m: i32, n: i32, k: i32, alpha: f64, a: *const f64, lda: i32, b: *const f64, ldb: i32, beta: f64, c: *mut f64, ldc: i32);
    fn cblas_dsyrk(order: i32, uplo: i32, trans: i32, n: i32, k: i32, alpha: f64, a: *const f64, lda: i32, beta: f64, c: *mut f64, ldc: i32);
    fn LAPACKE_dgesv(matrix_layout: i32, n: i32, nrhs: i32, a: *mut f64, lda: i32, ipiv: *mut i32, b: *mut f64, ldb: i32) -> i32;
    fn LAPACKE_dgesvd(matrix_layout: i32, jobu: u8, jobvt: u8, m: i32, n: i32, a: *mut f64, lda: i32, s: *mut f64, u: *mut f64, ldu: i32, vt: *mut f64, ldvt: i32, superb: *mut f64) -> i32;
    fn LAPACKE_dgetrf(matrix_layout: i32, m: i32, n: i32, a: *mut f64, lda: i32, ipiv: *mut i32) -> i32;
    fn LAPACKE_dgetri(matrix_layout: i32, n: i32, a: *mut f64, lda: i32, ipiv: *const i32) -> i32;
    fn LAPACKE_dpotrf(matrix_layout: i32, uplo: u8, n: i32, a: *mut f64, lda: i32) -> i32;
    fn LAPACKE_dgeev(matrix_layout: i32, jobvl: u8, jobvr: u8, n: i32, a: *mut f64, lda: i32, wr: *mut f64, wi: *mut f64, vl: *mut f64, ldvl: i32, vr: *mut f64, ldvr: i32) -> i32;
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

/// Performs one of the symmetric rank k operations
///
/// ```text
///     C := alpha*A*A**T + beta*C
/// ```
///
/// or
///
/// ```text
///     C := alpha*A**T*A + beta*C
/// ```
///
/// where alpha and beta are scalars, C is an n by n symmetric matrix
/// and A is an n by k matrix in the first case and a k by n matrix
/// in the second case.
///
/// # Reference
///
/// <http://www.netlib.org/lapack/explore-html/dc/d05/dsyrk_8f.html>
///
#[inline]
pub fn dsyrk(
    up: bool,
    trans: bool,
    n: i32,
    k: i32,
    alpha: f64,
    a: &[f64],
    lda: i32,
    beta: f64,
    c: &mut [f64],
    ldc: i32,
) {
    unsafe {
        cblas_dsyrk(
            CBLAS_COL_MAJOR,
            cblas_uplo(up),
            cblas_transpose(trans),
            n,
            k,
            alpha,
            a.as_ptr(),
            lda,
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
#[inline]
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

/// Computes an LU factorization of a general M-by-N matrix A using partial pivoting with row interchanges.
///
/// The factorization has the form
///
/// ```text
///    A = P * L * U
/// ```
///
/// where P is a permutation matrix, L is lower triangular with unit
/// diagonal elements (lower trapezoidal if m > n), and U is upper
/// triangular (upper trapezoidal if m < n).
///
/// This is the right-looking Level 3 BLAS version of the algorithm.
///
/// # Note
///
/// 1. matrix 'a' will be modified
/// 2. ipiv indices are 1-based (i.e. Fortran)
/// 3. See **dgetri** to use the factorization in finding the inverse matrix
///
/// # Reference
///
/// <http://www.netlib.org/lapack/explore-html/d3/d6a/dgetrf_8f.html>
///
#[inline]
pub fn dgetrf(m: i32, n: i32, a: &mut [f64], lda: i32, ipiv: &mut [i32]) -> Result<(), &'static str> {
    unsafe {
        let info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, m, n, a.as_mut_ptr(), lda, ipiv.as_mut_ptr());
        if info != 0_i32 {
            return Err("LAPACK failed");
        }
    }
    Ok(())
}

/// Computes the inverse of a matrix using the LU factorization computed by DGETRF.
///
/// This method inverts U and then computes inv(A) by solving the system
///
/// ```text
///    inv(A)*L = inv(U) for inv(A).
/// ```
///
/// # Note
///
/// 1. See **dgetrf** to compute the factorization
///
/// # Reference
///
/// <http://www.netlib.org/lapack/explore-html/df/da4/dgetri_8f.html>
///
#[inline]
pub fn dgetri(n: i32, a: &mut [f64], lda: i32, ipiv: &[i32]) -> Result<(), &'static str> {
    unsafe {
        let info = LAPACKE_dgetri(LAPACK_COL_MAJOR, n, a.as_mut_ptr(), lda, ipiv.as_ptr());
        if info != 0_i32 {
            return Err("LAPACK failed");
        }
    }
    Ok(())
}

/// Computes the Cholesky factorization of a real symmetric positive definite matrix A.
///
/// The factorization has the form
///
/// ```text
///    A = U**T * U,  if UPLO = 'U'
/// ```
///
/// or
///
/// ```text
///    A = L  * L**T,  if UPLO = 'L'
/// ```
///
/// where U is an upper triangular matrix and L is lower triangular.
///
/// This is the block version of the algorithm, calling Level 3 BLAS.
///
/// # Reference
///
/// <http://www.netlib.org/lapack/explore-html/d0/d8a/dpotrf_8f.html>
///
#[inline]
pub fn dpotrf(up: bool, n: i32, a: &mut [f64], lda: i32) -> Result<(), &'static str> {
    unsafe {
        let info = LAPACKE_dpotrf(LAPACK_COL_MAJOR, lapack_uplo(up), n, a.as_mut_ptr(), lda);
        if info != 0_i32 {
            return Err("LAPACK failed");
        }
    }
    Ok(())
}

/// Computes for an N-by-N real non-symmetric matrix A, the
/// eigenvalues and, optionally, the left and/or right eigenvectors.
///
/// The right eigenvector v(j) of A satisfies
///
/// ```text
///    A * v(j) = lambda(j) * v(j)
/// ```
///
/// where lambda(j) is its eigenvalue.
///
/// The left eigenvector u(j) of A satisfies
///
/// ```text
///    u(j)**H * A = lambda(j) * u(j)**H
/// ```
///
/// where u(j)**H denotes the conjugate-transpose of u(j).
///
/// The computed eigenvectors are normalized to have Euclidean norm
/// equal to 1 and largest component real.
///
/// # Notes
///
/// * If calc_vl==false, you may pass an empty vl array and must set ldvl=1
/// * If calc_vr==false, you may pass an empty vr array and must set ldvr=1
///
/// # Reference
///
/// <http://www.netlib.org/lapack/explore-html/d9/d28/dgeev_8f.html>
///
#[inline]
pub fn dgeev(
    calc_vl: bool,
    calc_vr: bool,
    n: i32,
    a: &mut [f64],
    lda: i32,
    wr: &mut [f64],
    wi: &mut [f64],
    vl: &mut [f64],
    ldvl: i32,
    vr: &mut [f64],
    ldvr: i32,
) -> Result<(), &'static str> {
    unsafe {
        let info = LAPACKE_dgeev(
            LAPACK_COL_MAJOR,
            lapack_job_vlr(calc_vl),
            lapack_job_vlr(calc_vr),
            n,
            a.as_mut_ptr(),
            lda,
            wr.as_mut_ptr(),
            wi.as_mut_ptr(),
            vl.as_mut_ptr(),
            ldvl,
            vr.as_mut_ptr(),
            ldvr,
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
    fn dsyrk_works() -> Result<(), &'static str> {
        // matrix c
        #[rustfmt::skip]
        let mut c_up = slice_to_colmajor(&[
            &[ 3.0,  0.0, -3.0,  0.0],
            &[ 0.0,  3.0,  1.0,  2.0],
            &[ 0.0,  0.0,  4.0,  1.0],
            &[ 0.0,  0.0,  0.0,  3.0],
        ]);
        #[rustfmt::skip]
        let mut c_lo = slice_to_colmajor(&[
            &[ 3.0,  0.0,  0.0,  0.0],
            &[ 0.0,  3.0,  0.0,  0.0],
            &[-3.0,  1.0,  4.0,  0.0],
            &[ 0.0,  2.0,  1.0,  3.0],
        ]);

        // n-size
        let n = 4_i32; // =c.ncol

        // matrix a
        #[rustfmt::skip]
        let a = slice_to_colmajor(&[
            &[ 1.0,  2.0,  1.0,  1.0, -1.0,  0.0],
            &[ 2.0,  2.0,  1.0,  0.0,  0.0,  0.0],
            &[ 3.0,  1.0,  3.0,  1.0,  2.0, -1.0],
            &[ 1.0,  0.0,  1.0, -1.0,  0.0,  0.0],
        ]);

        // k-size
        let k = 6_i32; // =a.ncol

        // constants
        let (alpha, beta) = (3.0, -1.0);

        // run dsyrk with up part of matrix c
        let trans = false;
        let (lda, ldc) = (n, n);
        dsyrk(true, trans, n, k, alpha, &a, lda, beta, &mut c_up, ldc);

        // check results: c := up(3⋅a⋅aᵀ - c)
        #[rustfmt::skip]
        let c_up_correct = slice_to_colmajor(&[
            &[21.0, 21.0, 24.0,  3.0],
            &[ 0.0, 24.0, 32.0,  7.0],
            &[ 0.0,  0.0, 71.0, 14.0],
            &[ 0.0,  0.0,  0.0,  6.0],
        ]);
        assert_vec_approx_eq!(c_up, c_up_correct, 1e-15);

        // run dsyrk with lo part of matrix c
        dsyrk(false, trans, n, k, alpha, &a, lda, beta, &mut c_lo, ldc);

        // check results: c := lo(3⋅a⋅aᵀ - c)
        #[rustfmt::skip]
        let c_lo_correct = slice_to_colmajor(&[
            &[21.0,  0.0,  0.0,  0.0],
            &[21.0, 24.0,  0.0,  0.0],
            &[24.0, 32.0, 71.0,  0.0],
            &[ 3.0,  7.0, 14.0,  6.0],
        ]);
        assert_vec_approx_eq!(c_lo, c_lo_correct, 1e-15);
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

    #[test]
    fn dgetrf_and_dgetri_work() -> Result<(), &'static str> {
        // matrix
        #[rustfmt::skip]
        let mut a = slice_to_colmajor(&[
            &[1.0, 2.0,  0.0, 1.0],
            &[2.0, 3.0, -1.0, 1.0],
            &[1.0, 2.0,  0.0, 4.0],
            &[4.0, 0.0,  3.0, 1.0],
        ]);
        let a_copy = a.to_vec();
        let (m, n) = (4_usize, 4_usize);
        let min_mn = if m < n { m } else { n };

        // run dgetrf
        let m_i32 = m.try_into().unwrap();
        let n_i32 = n.try_into().unwrap();
        let lda_i32 = m_i32;
        let mut ipiv = vec![0_i32; min_mn];
        dgetrf(m_i32, n_i32, &mut a, lda_i32, &mut ipiv)?;

        // check ipiv
        let ipiv_correct = &[4_i32, 2_i32, 3_i32, 4_i32];
        assert_eq!(ipiv, ipiv_correct);

        // check LU
        #[rustfmt::skip]
        let lu_correct = slice_to_colmajor(&[
            &[4.0e+00, 0.000000000000000e+00,  3.000000000000000e+00,  1.000000000000000e+00],
            &[5.0e-01, 3.000000000000000e+00, -2.500000000000000e+00,  5.000000000000000e-01],
            &[2.5e-01, 6.666666666666666e-01,  9.166666666666665e-01,  3.416666666666667e+00],
            &[2.5e-01, 6.666666666666666e-01,  1.000000000000000e+00, -3.000000000000000e+00],
        ]);
        assert_vec_approx_eq!(a, lu_correct, 1e-15);

        // run dgetri
        dgetri(n_i32, &mut a, lda_i32, &ipiv)?;

        // check inverse matrix
        #[rustfmt::skip]
        let ai_correct = slice_to_colmajor(&[
            &[-8.484848484848487e-01,  5.454545454545455e-01,  3.030303030303039e-02,  1.818181818181818e-01],
            &[ 1.090909090909091e+00, -2.727272727272728e-01, -1.818181818181817e-01, -9.090909090909091e-02],
            &[ 1.242424242424243e+00, -7.272727272727273e-01, -1.515151515151516e-01,  9.090909090909088e-02],
            &[-3.333333333333333e-01,  0.000000000000000e+00,  3.333333333333333e-01,  0.000000000000000e+00],
        ]);
        assert_vec_approx_eq!(a, ai_correct, 1e-15);

        // check again: a⋅a⁻¹ = I
        for i in 0..m {
            for j in 0..n {
                let mut res = 0.0;
                for k in 0..m {
                    res += a_copy[i + k * m] * ai_correct[k + j * m];
                }
                if i == j {
                    assert_approx_eq!(res, 1.0, 1e-13);
                } else {
                    assert_approx_eq!(res, 0.0, 1e-13);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn dpotrf_works() -> Result<(), &'static str> {
        // matrix a
        #[rustfmt::skip]
        let mut a_up = slice_to_colmajor(&[
            &[ 3.0,  0.0, -3.0,  0.0],
            &[ 0.0,  3.0,  1.0,  2.0],
            &[ 0.0,  0.0,  4.0,  1.0],
            &[ 0.0,  0.0,  0.0,  3.0],
        ]);
        #[rustfmt::skip]
        let mut a_lo = slice_to_colmajor(&[
            &[ 3.0,  0.0,  0.0,  0.0],
            &[ 0.0,  3.0,  0.0,  0.0],
            &[-3.0,  1.0,  4.0,  0.0],
            &[ 0.0,  2.0,  1.0,  3.0],
        ]);

        // n-size
        let n = 4_i32; // =a.ncol

        // run dpotrf with up part of matrix a
        let lda = n;
        dpotrf(true, n, &mut a_up, lda)?;

        // check Cholesky
        #[rustfmt::skip]
        let a_up_correct = slice_to_colmajor(&[
            &[ 1.732050807568877e+00,  0.000000000000000e+00, -1.732050807568878e+00,  0.000000000000000e+00],
            &[ 0.000000000000000e+00,  1.732050807568877e+00,  5.773502691896258e-01,  1.154700538379252e+00],
            &[ 0.000000000000000e+00,  0.000000000000000e+00,  8.164965809277251e-01,  4.082482904638632e-01],
            &[ 0.000000000000000e+00,  0.000000000000000e+00,  0.000000000000000e+00,  1.224744871391589e+00],
        ]);
        assert_vec_approx_eq!(a_up, a_up_correct, 1e-15);

        // run dpotrf with lo part of matrix a
        dpotrf(false, n, &mut a_lo, lda)?;

        // check Cholesky
        #[rustfmt::skip]
        let a_lo_correct = slice_to_colmajor(&[
            &[ 1.732050807568877e+00,  0.000000000000000e+00,  0.000000000000000e+00,  0.000000000000000e+00],
            &[ 0.000000000000000e+00,  1.732050807568877e+00,  0.000000000000000e+00,  0.000000000000000e+00],
            &[-1.732050807568878e+00,  5.773502691896258e-01,  8.164965809277251e-01,  0.000000000000000e+00],
            &[ 0.000000000000000e+00,  1.154700538379252e+00,  4.082482904638632e-01,  1.224744871391589e+00],
        ]);
        assert_vec_approx_eq!(a_lo, a_lo_correct, 1e-15);
        Ok(())
    }
}
