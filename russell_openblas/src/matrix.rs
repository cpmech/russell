use super::{cblas_transpose, cblas_uplo, lapack_job_vlr, lapack_uplo, CBLAS_ROW_MAJOR, LAPACK_ROW_MAJOR};

#[rustfmt::skip]
extern "C" {
    fn cblas_dgemm(order: i32, transa: i32, transb: i32, m: i32, n: i32, k: i32, alpha: f64, a: *const f64, lda: i32, b: *const f64, ldb: i32, beta: f64, c: *mut f64, ldc: i32);
    fn cblas_dsyrk(order: i32, uplo: i32, trans: i32, n: i32, k: i32, alpha: f64, a: *const f64, lda: i32, beta: f64, c: *mut f64, ldc: i32);
    fn LAPACKE_dlange(matrix_layout: i32, norm: u8, m:i32, n:i32, a: *const f64, lda:i32) -> f64;
    fn LAPACKE_dgesvd(matrix_layout: i32, jobu: u8, jobvt: u8, m: i32, n: i32, a: *mut f64, lda: i32, s: *mut f64, u: *mut f64, ldu: i32, vt: *mut f64, ldvt: i32, superb: *mut f64) -> i32;
    fn LAPACKE_dgetrf(matrix_layout: i32, m: i32, n: i32, a: *mut f64, lda: i32, ipiv: *mut i32) -> i32;
    fn LAPACKE_dgetri(matrix_layout: i32, n: i32, a: *mut f64, lda: i32, ipiv: *const i32) -> i32;
    fn LAPACKE_dpotrf(matrix_layout: i32, uplo: u8, n: i32, a: *mut f64, lda: i32) -> i32;
    fn LAPACKE_dgeev(matrix_layout: i32, jobvl: u8, jobvr: u8, n: i32, a: *mut f64, lda: i32, wr: *mut f64, wi: *mut f64, vl: *mut f64, ldvl: i32, vr: *mut f64, ldvr: i32) -> i32;
}

/// Performs the matrix-matrix multiplication
///
/// Computes one of:
///
/// ```text
/// trans_a = false, trans_b = false:
///
///   c  := α ⋅  a  ⋅  b  +  β ⋅  c
/// (m,n)      (m,k) (k,n)      (m,n)
///
///
/// trans_a = false, trans_b = true:
///
///   c  := α ⋅  a  ⋅  bᵀ  +  β ⋅  c
/// (m,n)      (m,k) (k,n)       (m,n)
///
///
/// trans_a = true, trans_b = false:
///
///   c  := α ⋅  aᵀ  ⋅  b  +  β ⋅  c
/// (m,n)      (m,k) (k,n)       (m,n)
///
///
/// trans_a = true, trans_b = true:
///
///   c  := α ⋅  aᵀ ⋅  bᵀ +  β ⋅  c
/// (m,n)      (m,k) (k,n)      (m,n)
/// ```
///
/// # Important
///
/// * The data must be in **row-major** order
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
    b: &[f64],
    beta: f64,
    c: &mut [f64],
) {
    let lda = if trans_a { m } else { k };
    let ldb = if trans_b { k } else { n };
    unsafe {
        cblas_dgemm(
            CBLAS_ROW_MAJOR,
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
            n,
        );
    }
}

/// Performs one of the symmetric rank k operations
///
/// Computes one of:
///
/// ```text
///   c := α ⋅ a  ⋅  aᵀ  +  β ⋅  c
/// (n,n)    (n,k) (k,n)       (n,n)
///
/// or
///
///   c := α ⋅ aᵀ  ⋅  a  +  β ⋅  c
/// (n,n)    (n,k)  (k,n)      (n,n)
/// ```
///
/// # Important
///
/// * The data must be in **row-major** order
///
/// # Reference
///
/// <http://www.netlib.org/lapack/explore-html/dc/d05/dsyrk_8f.html>
///
#[inline]
pub fn dsyrk(up: bool, trans: bool, n: i32, k: i32, alpha: f64, a: &[f64], beta: f64, c: &mut [f64]) {
    let lda = if trans { n } else { k };
    unsafe {
        cblas_dsyrk(
            CBLAS_ROW_MAJOR,
            cblas_uplo(up),
            cblas_transpose(trans),
            n,
            k,
            alpha,
            a.as_ptr(),
            lda,
            beta,
            c.as_mut_ptr(),
            n,
        );
    }
}

/// Computes the matrix norm
///
/// Computes one of:
///
/// ```text
/// ‖a‖_1 = max_j ( Σ_i |aij| )
///
/// ‖a‖_∞ = max_i ( Σ_j |aij| )
///
/// ‖a‖_F = sqrt(Σ_i Σ_j aij⋅aij) == ‖a‖_2
///
/// ‖a‖_max = max_ij ( |aij| )
/// ```
///
/// # Input
///
/// * norm == b'1' -- computes the 1-norm
/// * norm == b'I' -- computes the infinity-norm
/// * norm == b'F' -- computes the Frobenius-norm
/// * norm == b'M' -- computes max(abs(a(i,j)))
///
/// # Important
///
/// * The data must be in **row-major** order
///
/// # Reference
///
/// <http://www.netlib.org/lapack/explore-html/dc/d09/dlange_8f.html>
///
#[inline]
pub fn dlange(norm: u8, m: i32, n: i32, a: &[f64]) -> f64 {
    unsafe { LAPACKE_dlange(LAPACK_ROW_MAJOR, norm, m, n, a.as_ptr(), n) }
}

/// Computes the singular value decomposition (SVD)
///
/// The SVD is written as follows:
///
/// ```text
///   A  =   U  ⋅ SIGMA ⋅ Vᵀ
/// (m,n)  (m,m)  (m,n)  (n,n)
/// ```
///
/// where SIGMA is an M-by-N matrix which is zero except for its
/// min(m,n) diagonal elements, U is an M-by-M orthogonal matrix, and
/// V is an N-by-N orthogonal matrix. The diagonal elements of SIGMA
/// are the singular values of A; they are real and non-negative, and
/// are returned in descending order. The first min(m,n) columns of
/// U and V are the left and right singular vectors of A.
///
/// # Note
///
/// 1. The routine returns Vᵀ, not V.
/// 2. The matrix will be modified
/// 3. `jobu` and `jobvt` are c_char and can be passed as b'A'
///    (see LAPACK reference for further options)
/// 4. `superb` is a work area of size min(m,n)-1; e.g., use min(m,n)
///
/// # Important
///
/// * The data must be in **row-major** order
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
    s: &mut [f64],
    u: &mut [f64],
    vt: &mut [f64],
    superb: &mut [f64],
) -> Result<(), &'static str> {
    unsafe {
        let info = LAPACKE_dgesvd(
            LAPACK_ROW_MAJOR,
            jobu,
            jobvt,
            m,
            n,
            a.as_mut_ptr(),
            n,
            s.as_mut_ptr(),
            u.as_mut_ptr(),
            m,
            vt.as_mut_ptr(),
            n,
            superb.as_mut_ptr(),
        );
        if info != 0_i32 {
            return Err("LAPACK dgesvd failed");
        }
    }
    Ok(())
}

/// Computes an LU factorization of a general (m,n) matrix
///
/// The factorization has the form:
///
/// ```text
/// A  =  P ⋅ L ⋅ U
/// ```
///
/// where P is a permutation matrix, L is lower triangular with unit
/// diagonal elements (lower trapezoidal if m > n), and U is upper
/// triangular (upper trapezoidal if m < n).
///
/// # Note
///
/// 1. matrix `a` will be modified
/// 2. ipiv indices are 1-based (i.e. Fortran)
/// 3. See **dgetri** to use the factorization in finding the inverse matrix
///
/// # Important
///
/// * The data must be in **row-major** order
///
/// # Reference
///
/// <http://www.netlib.org/lapack/explore-html/d3/d6a/dgetrf_8f.html>
///
#[inline]
pub fn dgetrf(m: i32, n: i32, a: &mut [f64], ipiv: &mut [i32]) -> Result<(), &'static str> {
    unsafe {
        let info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, m, n, a.as_mut_ptr(), n, ipiv.as_mut_ptr());
        if info != 0_i32 {
            return Err("LAPACK dgetrf failed");
        }
    }
    Ok(())
}

/// Computes the inverse of a matrix using the LU factorization computed by dgetrf
///
/// This method inverts U and then computes inv(A) by solving the system
///
/// ```text
/// inv(A)*L = inv(U) for inv(A)
/// ```
///
/// # Note
///
/// 1. See **dgetrf** to compute the factorization
///
/// # Important
///
/// * The data must be in **row-major** order
///
/// # Reference
///
/// <http://www.netlib.org/lapack/explore-html/df/da4/dgetri_8f.html>
///
#[inline]
pub fn dgetri(n: i32, a: &mut [f64], ipiv: &[i32]) -> Result<(), &'static str> {
    unsafe {
        let info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, a.as_mut_ptr(), n, ipiv.as_ptr());
        if info != 0_i32 {
            return Err("LAPACK dgetri failed");
        }
    }
    Ok(())
}

/// Computes the Cholesky factorization of a real symmetric positive definite matrix
///
/// The factorization has the form
///
/// ```text
/// up = true:
///
/// A = Uᵀ ⋅ U
///
/// or
///
/// up = false:
///
/// A = L ⋅ Lᵀ
/// ```
///
/// where U is an upper triangular matrix and L is lower triangular.
///
/// # Important
///
/// * The data must be in **row-major** order
///
/// # Reference
///
/// <http://www.netlib.org/lapack/explore-html/d0/d8a/dpotrf_8f.html>
///
#[inline]
pub fn dpotrf(up: bool, n: i32, a: &mut [f64]) -> Result<(), &'static str> {
    unsafe {
        let info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, lapack_uplo(up), n, a.as_mut_ptr(), n);
        if info != 0_i32 {
            return Err("LAPACK dpotrf failed");
        }
    }
    Ok(())
}

/// Computes the eigenvalues and eigenvectors of a general matrix
///
/// The right eigenvector v(j) of A satisfies
///
/// ```text
/// A ⋅ v(j) = lambda(j) ⋅ v(j)
/// ```
///
/// where lambda(j) is its eigenvalue.
///
/// The left eigenvector u(j) of A satisfies
///
/// ```text
/// u(j)ᴴ ⋅ A = lambda(j) ⋅ u(j)ᴴ
/// ```
///
/// where u(j)ᴴ denotes the conjugate-transpose of u(j).
///
/// The computed eigenvectors are normalized to have Euclidean norm
/// equal to 1 and largest component real.
///
/// # Notes
///
/// 1. The matrix will be modified
/// 2. If calc_vl==false, you may pass an empty array
/// 3. If calc_vr==false, you may pass an empty array
///
/// # Important
///
/// * The data must be in **row-major** order
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
    wr: &mut [f64],
    wi: &mut [f64],
    vl: &mut [f64],
    vr: &mut [f64],
) -> Result<(), &'static str> {
    unsafe {
        let info = LAPACKE_dgeev(
            LAPACK_ROW_MAJOR,
            lapack_job_vlr(calc_vl),
            lapack_job_vlr(calc_vr),
            n,
            a.as_mut_ptr(),
            n,
            wr.as_mut_ptr(),
            wi.as_mut_ptr(),
            vl.as_mut_ptr(),
            n, // should be 1 if !calc_vl; but lapack works differently in row-major
            vr.as_mut_ptr(),
            n, // should be 1 if !calc_vr; but lapack works differently in row-major
        );
        if info != 0_i32 {
            return Err("LAPACK dgeev failed");
        }
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{dgeev, dgemm, dgesvd, dgetrf, dgetri, dlange, dpotrf, dsyrk};
    use crate::conversions::{dgeev_data, dgeev_data_lr};
    use crate::to_i32;
    use russell_chk::*;

    #[test]
    fn dgemm_notrans_notrans_works() {
        // 0.5⋅a⋅b + 2⋅c

        // allocate matrices
        #[rustfmt::skip]
        let a = [ // (m, k) = (4, 5)
            1.0, 2.0,  0.0, 1.0, -1.0,
            2.0, 3.0, -1.0, 1.0,  1.0,
            1.0, 2.0,  0.0, 4.0, -1.0,
            4.0, 0.0,  3.0, 1.0,  1.0,
        ];
        #[rustfmt::skip]
        let b = [ // (k, n) = (5, 3)
            1.0, 0.0, 0.0,
            0.0, 0.0, 3.0,
            0.0, 0.0, 1.0,
            1.0, 0.0, 1.0,
            0.0, 2.0, 0.0,
        ];
        #[rustfmt::skip]
        let mut c = [ // (m, n) = (4, 3)
             0.50, 0.0,  0.25,
             0.25, 0.0, -0.25,
            -0.25, 0.0,  0.00,
            -0.25, 0.0,  0.00,
        ];

        // sizes
        let m = 4; // m = nrow(a) = a.M = nrow(c)
        let k = 5; // k = ncol(a) = a.N = nrow(b)
        let n = 3; // n = ncol(b) = b.N = ncol(c)

        // run dgemm
        let (trans_a, trans_b) = (false, false);
        let (alpha, beta) = (0.5, 2.0);
        dgemm(trans_a, trans_b, m, n, k, alpha, &a, &b, beta, &mut c);

        // check
        #[rustfmt::skip]
        let correct = [
            2.0, -1.0, 4.0,
            2.0,  1.0, 4.0,
            2.0, -1.0, 5.0,
            2.0,  1.0, 2.0,
        ];
        assert_vec_approx_eq!(c, correct, 1e-15);
    }

    #[test]
    fn dgemm_notrans_trans_works() {
        // 0.5⋅a⋅bᵀ + 2⋅c"

        // allocate matrices
        #[rustfmt::skip]
        let a = [ // (m, k) = (4, 5)
            1.0, 2.0,  0.0, 1.0, -1.0,
            2.0, 3.0, -1.0, 1.0,  1.0,
            1.0, 2.0,  0.0, 4.0, -1.0,
            4.0, 0.0,  3.0, 1.0,  1.0,
        ];
        #[rustfmt::skip]
        let b = [ // (n, k) = (3, 5)
            1.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 2.0,
            0.0, 3.0, 1.0, 1.0, 0.0,
        ];
        #[rustfmt::skip]
        let mut c = [ // (m, n) = (4, 3)
             0.50, 0.0,  0.25,
             0.25, 0.0, -0.25,
            -0.25, 0.0,  0.00,
            -0.25, 0.0,  0.00,
        ];

        // sizes
        let m = 4; // m = nrow(a)        = a.M = nrow(c)
        let k = 5; // k = ncol(a)        = a.N = nrow(trans(b))
        let n = 3; // n = ncol(trans(b)) = b.M = ncol(c)

        // run dgemm
        let (trans_a, trans_b) = (false, true);
        let (alpha, beta) = (0.5, 2.0);
        dgemm(trans_a, trans_b, m, n, k, alpha, &a, &b, beta, &mut c);

        // check
        #[rustfmt::skip]
        let correct = [
            2.0, -1.0, 4.0,
            2.0,  1.0, 4.0,
            2.0, -1.0, 5.0,
            2.0,  1.0, 2.0,
        ];
        assert_vec_approx_eq!(c, correct, 1e-15);
    }

    #[test]
    fn dgemm_trans_notrans_works() {
        // 0.5⋅aᵀ⋅b + 2⋅c

        // allocate matrices
        #[rustfmt::skip]
        let a = [ // (k, m) = (5, 4)
             1.0,  2.0,  1.0, 4.0,
             2.0,  3.0,  2.0, 0.0,
             0.0, -1.0,  0.0, 3.0,
             1.0,  1.0,  4.0, 1.0,
            -1.0,  1.0, -1.0, 1.0,
        ];
        #[rustfmt::skip]
        let b = [ // (k, n) = (5, 3)
            1.0, 0.0, 0.0,
            0.0, 0.0, 3.0,
            0.0, 0.0, 1.0,
            1.0, 0.0, 1.0,
            0.0, 2.0, 0.0,
        ];
        #[rustfmt::skip]
        let mut c = [ // (m, n) = (4, 3)
             0.50, 0.0,  0.25,
             0.25, 0.0, -0.25,
            -0.25, 0.0,  0.00,
            -0.25, 0.0,  0.00,
        ];

        // sizes
        let m = 4; // m = nrow(trans(a)) = a.N = nrow(c)
        let k = 5; // k = ncol(trans(a)) = a.M = nrow(trans(b))
        let n = 3; // n = ncol(b)        = b.N = ncol(c)

        // run dgemm
        let (trans_a, trans_b) = (true, false);
        let (alpha, beta) = (0.5, 2.0);
        dgemm(trans_a, trans_b, m, n, k, alpha, &a, &b, beta, &mut c);

        // check
        #[rustfmt::skip]
        let correct = [
            2.0, -1.0, 4.0,
            2.0,  1.0, 4.0,
            2.0, -1.0, 5.0,
            2.0,  1.0, 2.0,
        ];
        assert_vec_approx_eq!(c, correct, 1e-15);
    }

    #[test]
    fn dgemm_trans_trans_works() {
        // 0.5⋅aᵀ⋅bᵀ + 2⋅c

        // allocate matrices
        #[rustfmt::skip]
        let a = [ // (k, m) = (5, 4)
             1.0,  2.0,  1.0, 4.0,
             2.0,  3.0,  2.0, 0.0,
             0.0, -1.0,  0.0, 3.0,
             1.0,  1.0,  4.0, 1.0,
            -1.0,  1.0, -1.0, 1.0,
        ];
        #[rustfmt::skip]
        let b = [ // (n, k) = (3, 5)
            1.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 2.0,
            0.0, 3.0, 1.0, 1.0, 0.0,
        ];
        #[rustfmt::skip]
        let mut c = [ // (m, n) = (4, 3)
             0.50, 0.0,  0.25,
             0.25, 0.0, -0.25,
            -0.25, 0.0,  0.00,
            -0.25, 0.0,  0.00,
        ];

        // sizes
        let m = 4; // m = nrow(trans(a)) = a.N = nrow(c)
        let k = 5; // k = ncol(trans(a)) = a.M = nrow(trans(b))
        let n = 3; // n = ncol(trans(b)) = b.M = ncol(c)

        // run dgemm
        let (trans_a, trans_b) = (true, true);
        let (alpha, beta) = (0.5, 2.0);
        dgemm(trans_a, trans_b, m, n, k, alpha, &a, &b, beta, &mut c);

        // check
        #[rustfmt::skip]
        let correct = [
            2.0, -1.0, 4.0,
            2.0,  1.0, 4.0,
            2.0, -1.0, 5.0,
            2.0,  1.0, 2.0,
        ];
        assert_vec_approx_eq!(c, correct, 1e-15);
    }

    #[test]
    fn dsyrk_works() {
        // matrix c
        #[rustfmt::skip]
        let mut c_up = [
            3.0,  0.0, -3.0,  0.0,
            0.0,  3.0,  1.0,  2.0,
            0.0,  0.0,  4.0,  1.0,
            0.0,  0.0,  0.0,  3.0,
        ];
        #[rustfmt::skip]
        let mut c_lo = [
             3.0,  0.0,  0.0,  0.0,
             0.0,  3.0,  0.0,  0.0,
            -3.0,  1.0,  4.0,  0.0,
             0.0,  2.0,  1.0,  3.0,
        ];

        // n-size
        let n = 4_i32; // =c.ncol

        // matrix a
        #[rustfmt::skip]
        let a = [
            1.0,  2.0,  1.0,  1.0, -1.0,  0.0,
            2.0,  2.0,  1.0,  0.0,  0.0,  0.0,
            3.0,  1.0,  3.0,  1.0,  2.0, -1.0,
            1.0,  0.0,  1.0, -1.0,  0.0,  0.0,
        ];

        // k-size
        let k = 6_i32; // =a.ncol

        // constants
        let (alpha, beta) = (3.0, -1.0);

        // run dsyrk with up part of matrix c
        let trans = false;
        dsyrk(true, trans, n, k, alpha, &a, beta, &mut c_up);

        // check results: c := up(3⋅a⋅aᵀ - c)
        #[rustfmt::skip]
        let c_up_correct = [
            21.0, 21.0, 24.0,  3.0,
             0.0, 24.0, 32.0,  7.0,
             0.0,  0.0, 71.0, 14.0,
             0.0,  0.0,  0.0,  6.0,
        ];
        assert_vec_approx_eq!(c_up, c_up_correct, 1e-15);

        // run dsyrk with lo part of matrix c
        dsyrk(false, trans, n, k, alpha, &a, beta, &mut c_lo);

        // check results: c := lo(3⋅a⋅aᵀ - c)
        #[rustfmt::skip]
        let c_lo_correct = [
            21.0,  0.0,  0.0,  0.0,
            21.0, 24.0,  0.0,  0.0,
            24.0, 32.0, 71.0,  0.0,
             3.0,  7.0, 14.0,  6.0,
        ];
        assert_vec_approx_eq!(c_lo, c_lo_correct, 1e-15);
    }

    #[test]
    fn dlange_works() {
        #[rustfmt::skip]
        let a = [
            -3.0, 5.0, 7.0,
             2.0, 6.0, 4.0,
             0.0, 2.0, 8.0,
        ];
        let norm_one = dlange(b'1', 3, 3, &a);
        let norm_inf = dlange(b'I', 3, 3, &a);
        let norm_fro = dlange(b'F', 3, 3, &a);
        let norm_max = dlange(b'M', 3, 3, &a);
        assert_eq!(norm_one, 19.0);
        assert_eq!(norm_inf, 15.0);
        assert_eq!(norm_fro, f64::sqrt(207.0));
        assert_eq!(norm_max, 8.0);
    }

    #[test]
    fn dgesvd_fails() {
        let (m, n) = (2_usize, 3_usize);
        let min_mn = if m < n { m } else { n };
        let mut a = vec![0.0; m * n];
        let mut s = vec![0.0; min_mn as usize];
        let mut u = vec![0.0; (m * m) as usize];
        let mut vt = vec![0.0; (n * n) as usize];
        let mut superb = vec![0.0; min_mn as usize];
        assert_eq!(
            dgesvd(
                b'A',
                b'X', // <<<< ERROR
                to_i32(m),
                to_i32(n),
                &mut a,
                &mut s,
                &mut u,
                &mut vt,
                &mut superb,
            ),
            Err("LAPACK dgesvd failed")
        );
    }

    #[test]
    fn dgesvd_works() -> Result<(), &'static str> {
        // matrix
        #[rustfmt::skip]
        let mut a = [
            1.0, 0.0, 0.0, 0.0, 2.0,
            0.0, 0.0, 3.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 2.0, 0.0, 0.0, 0.0,
        ];
        let a_copy = a.to_vec();

        // dimensions
        let (m, n) = (4_usize, 5_usize);
        let min_mn = if m < n { m } else { n };

        // allocate output arrays
        let mut s = vec![0.0; min_mn as usize];
        let mut u = vec![0.0; (m * m) as usize];
        let mut vt = vec![0.0; (n * n) as usize];
        let mut superb = vec![0.0; min_mn as usize];

        // perform the SVD
        dgesvd(
            b'A',
            b'A',
            to_i32(m),
            to_i32(n),
            &mut a,
            &mut s,
            &mut u,
            &mut vt,
            &mut superb,
        )?;

        // check
        #[rustfmt::skip]
        let u_correct = [
            0.0, 1.0, 0.0,  0.0,
            1.0, 0.0, 0.0,  0.0,
            0.0, 0.0, 0.0, -1.0,
            0.0, 0.0, 1.0,  0.0,
        ];
        let s_correct = &[3.0, f64::sqrt(5.0), 2.0, 0.0];
        let s2 = f64::sqrt(0.2);
        let s8 = f64::sqrt(0.8);
        #[rustfmt::skip]
        let vt_correct = [
            0.0, 0.0, 1.0, 0.0, 0.0,
             s2, 0.0, 0.0, 0.0,  s8,
            0.0, 1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 0.0,
            -s8, 0.0, 0.0, 0.0,  s2,
        ];
        assert_vec_approx_eq!(u, u_correct, 1e-15);
        assert_vec_approx_eq!(s, s_correct, 1e-15);
        assert_vec_approx_eq!(vt, vt_correct, 1e-15);

        // check SVD
        let mut usv = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                for k in 0..min_mn {
                    usv[i * n + j] += u[i * m + k] * s[k] * vt[k * n + j];
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
        let mut a = [
            -s33, -s33, 1.0,
             s33, -s33, 1.0,
            -s33,  s33, 1.0,
             s33,  s33, 1.0,
        ];
        let a_copy = a.to_vec();

        // dimensions
        let (m, n) = (4_usize, 3_usize);
        let min_mn = if m < n { m } else { n };

        // allocate output arrays
        let mut s = vec![0.0; min_mn as usize];
        let mut u = vec![0.0; (m * m) as usize];
        let mut vt = vec![0.0; (n * n) as usize];
        let mut superb = vec![0.0; min_mn as usize];

        // perform SVD
        dgesvd(
            b'A',
            b'A',
            to_i32(m),
            to_i32(n),
            &mut a,
            &mut s,
            &mut u,
            &mut vt,
            &mut superb,
        )?;

        // check
        #[rustfmt::skip]
        let u_correct = [
            -0.5, -0.5, -0.5,  0.5,
            -0.5, -0.5,  0.5, -0.5,
            -0.5,  0.5, -0.5, -0.5,
            -0.5,  0.5,  0.5,  0.5,
        ];
        let s_correct = &[2.0, 2.0 / f64::sqrt(3.0), 2.0 / f64::sqrt(3.0)];
        #[rustfmt::skip]
        let vt_correct = [
            0.0, 0.0, -1.0,
            0.0, 1.0,  0.0,
            1.0, 0.0,  0.0,
        ];
        assert_vec_approx_eq!(u, u_correct, 1e-15);
        assert_vec_approx_eq!(s, s_correct, 1e-15);
        assert_vec_approx_eq!(vt, vt_correct, 1e-15);

        // check SVD
        let mut usv = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                for k in 0..min_mn {
                    usv[i * n + j] += u[i * m + k] * s[k] * vt[k * n + j];
                }
            }
        }
        assert_vec_approx_eq!(usv, a_copy, 1e-15);
        Ok(())
    }

    #[test]
    fn dgetrf_and_dgetri_fail() {
        let (m, n) = (2, 2);
        let min_mn = if m < n { m } else { n };
        let m_i32 = to_i32(m);
        let n_i32 = to_i32(n);
        let mut a = vec![0.0; m * n];
        let mut ipiv = vec![0_i32; min_mn];
        assert_eq!(dgetrf(m_i32, n_i32, &mut a, &mut ipiv), Err("LAPACK dgetrf failed"));
        assert_eq!(dgetri(n_i32, &mut a, &ipiv), Err("LAPACK dgetri failed"));
    }

    #[test]
    fn dgetrf_and_dgetri_work() -> Result<(), &'static str> {
        // matrix
        #[rustfmt::skip]
        let mut a = [
            1.0, 2.0,  0.0, 1.0,
            2.0, 3.0, -1.0, 1.0,
            1.0, 2.0,  0.0, 4.0,
            4.0, 0.0,  3.0, 1.0,
        ];
        let a_copy = a.to_vec();
        let (m, n) = (4_usize, 4_usize);
        let min_mn = if m < n { m } else { n };

        // run dgetrf
        let m_i32 = to_i32(m);
        let n_i32 = to_i32(n);
        let mut ipiv = vec![0_i32; min_mn];
        dgetrf(m_i32, n_i32, &mut a, &mut ipiv)?;

        // check ipiv
        let ipiv_correct = &[4_i32, 2_i32, 3_i32, 4_i32];
        assert_eq!(ipiv, ipiv_correct);

        // check LU
        #[rustfmt::skip]
        let lu_correct = [
            4.0e+00, 0.000000000000000e+00,  3.000000000000000e+00,  1.000000000000000e+00,
            5.0e-01, 3.000000000000000e+00, -2.500000000000000e+00,  5.000000000000000e-01,
            2.5e-01, 6.666666666666666e-01,  9.166666666666665e-01,  3.416666666666667e+00,
            2.5e-01, 6.666666666666666e-01,  1.000000000000000e+00, -3.000000000000000e+00,
        ];
        assert_vec_approx_eq!(a, lu_correct, 1e-15);

        // run dgetri
        dgetri(n_i32, &mut a, &ipiv)?;

        // check inverse matrix
        #[rustfmt::skip]
        let ai_correct = [
            -8.484848484848487e-01,  5.454545454545455e-01,  3.030303030303039e-02,  1.818181818181818e-01,
             1.090909090909091e+00, -2.727272727272728e-01, -1.818181818181817e-01, -9.090909090909091e-02,
             1.242424242424243e+00, -7.272727272727273e-01, -1.515151515151516e-01,  9.090909090909088e-02,
            -3.333333333333333e-01,  0.000000000000000e+00,  3.333333333333333e-01,  0.000000000000000e+00,
        ];
        assert_vec_approx_eq!(a, ai_correct, 1e-15);

        // check again: a⋅a⁻¹ = I
        for i in 0..m {
            for j in 0..n {
                let mut res = 0.0;
                for k in 0..m {
                    res += a_copy[i * n + k] * ai_correct[k * n + j];
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
    fn dpotrf_fails() {
        let mut a = vec![0.0; 4];
        assert_eq!(dpotrf(true, 2_i32, &mut a), Err("LAPACK dpotrf failed"));
    }

    #[test]
    fn dpotrf_works() -> Result<(), &'static str> {
        // matrix a
        #[rustfmt::skip]
        let mut a_up = [
            3.0,  0.0, -3.0,  0.0,
            0.0,  3.0,  1.0,  2.0,
            0.0,  0.0,  4.0,  1.0,
            0.0,  0.0,  0.0,  3.0,
        ];
        #[rustfmt::skip]
        let mut a_lo = [
             3.0,  0.0,  0.0,  0.0,
             0.0,  3.0,  0.0,  0.0,
            -3.0,  1.0,  4.0,  0.0,
             0.0,  2.0,  1.0,  3.0,
        ];

        // n-size
        let n = 4_i32; // =a.ncol

        // run dpotrf with up part of matrix a
        dpotrf(true, n, &mut a_up)?;

        // check Cholesky
        #[rustfmt::skip]
        let a_up_correct = [
            1.732050807568877e+00,  0.000000000000000e+00, -1.732050807568878e+00,  0.000000000000000e+00,
            0.000000000000000e+00,  1.732050807568877e+00,  5.773502691896258e-01,  1.154700538379252e+00,
            0.000000000000000e+00,  0.000000000000000e+00,  8.164965809277251e-01,  4.082482904638632e-01,
            0.000000000000000e+00,  0.000000000000000e+00,  0.000000000000000e+00,  1.224744871391589e+00,
        ];
        assert_vec_approx_eq!(a_up, a_up_correct, 1e-15);

        // run dpotrf with lo part of matrix a
        dpotrf(false, n, &mut a_lo)?;

        // check Cholesky
        #[rustfmt::skip]
        let a_lo_correct = [
             1.732050807568877e+00,  0.000000000000000e+00,  0.000000000000000e+00,  0.000000000000000e+00,
             0.000000000000000e+00,  1.732050807568877e+00,  0.000000000000000e+00,  0.000000000000000e+00,
            -1.732050807568878e+00,  5.773502691896258e-01,  8.164965809277251e-01,  0.000000000000000e+00,
             0.000000000000000e+00,  1.154700538379252e+00,  4.082482904638632e-01,  1.224744871391589e+00,
        ];
        assert_vec_approx_eq!(a_lo, a_lo_correct, 1e-15);
        Ok(())
    }

    #[test]
    fn dgeev_fails() {
        let m = 1_usize;
        let mut a = vec![0.0; m * m];
        let mut wr = vec![0.0; m]; // eigenvalues (real part)
        let mut wi = vec![0.0; m]; // eigenvalues (imaginary part)
        let mut vl = vec![0.0; m * m]; // left eigenvectors
        let mut vr = vec![0.0; m * m]; // right eigenvectors
        let wrong = -1_i32; // <<< wrong
        assert_eq!(
            dgeev(true, true, wrong, &mut a, &mut wr, &mut wi, &mut vl, &mut vr),
            Err("LAPACK dgeev failed")
        );
    }

    #[test]
    fn dgeev_works() -> Result<(), &'static str> {
        // matrix a
        #[rustfmt::skip]
        let mut a = [
             0.35,  0.45, -0.14, -0.17,
             0.09,  0.07, -0.54,  0.35,
            -0.44, -0.33, -0.03,  0.17,
             0.25, -0.32, -0.13,  0.11,
        ];
        let mut a_copy1 = a.to_vec();
        let mut a_copy2 = a.to_vec();

        // n-size
        let n = 4_i32; // =a.nrow=a.ncol

        // eigen-arrays
        let sz = n as usize;
        let mut wr = vec![0.0; sz]; // eigenvalues (real part)
        let mut wi = vec![0.0; sz]; // eigenvalues (imaginary part)
        let mut vl = vec![0.0; sz * sz]; // left eigenvectors
        let mut vr = vec![0.0; sz * sz]; // right eigenvectors

        // compute eigen-things
        dgeev(true, true, n, &mut a, &mut wr, &mut wi, &mut vl, &mut vr)?;

        // check eigenvalues
        #[rustfmt::skip]
        let wr_correct = [
             7.994821225862098e-01,
            -9.941245329507467e-02,
            -9.941245329507467e-02,
            -1.006572159960587e-01,
        ];
        #[rustfmt::skip]
        let wi_correct = [
             0.0,
             4.007924719897546e-01,
            -4.007924719897546e-01,
             0.0,
        ];
        assert_vec_approx_eq!(wr, wr_correct, 1e-15);
        assert_vec_approx_eq!(wi, wi_correct, 1e-15);

        // extract eigenvalues from dgeev data
        let mut vl_real = vec![0.0; sz * sz];
        let mut vl_imag = vec![0.0; sz * sz];
        let mut vr_real = vec![0.0; sz * sz];
        let mut vr_imag = vec![0.0; sz * sz];
        dgeev_data_lr(&mut vl_real, &mut vl_imag, &mut vr_real, &mut vr_imag, &wi, &vl, &vr)?;

        // check left eigenvectors
        #[rustfmt::skip]
        let vl_real_correct = [
            -6.244707486379453e-01,  5.330229831716200e-01,  5.330229831716200e-01,  6.641410231734539e-01,
            -5.994889025288728e-01, -2.666163325181558e-01, -2.666163325181558e-01, -1.068153340034493e-01,
             4.999156725721429e-01,  3.455257668600027e-01,  3.455257668600027e-01,  7.293254091191846e-01,
            -2.708616172576073e-02, -2.540814367391268e-01, -2.540814367391268e-01,  1.248664621625170e-01,
        ];
        #[rustfmt::skip]
        let vl_imag_correct = [
            0.0,  0.0,                    0.0,                   0.0,
            0.0,  4.041362636762622e-01, -4.041362636762622e-01, 0.0,
            0.0,  3.152853126680209e-01, -3.152853126680209e-01, 0.0,
            0.0, -4.451133008385643e-01,  4.451133008385643e-01, 0.0,
        ];
        assert_vec_approx_eq!(vl_real, vl_real_correct, 1e-15);
        assert_vec_approx_eq!(vl_imag, vl_imag_correct, 1e-15);

        // check right eigenvectors
        #[rustfmt::skip]
        let vr_real_correct = [
            -6.550887675124076e-01,-1.933015482642217e-01,-1.933015482642217e-01, 1.253326972309026e-01,
            -5.236294609021240e-01, 2.518565317267399e-01, 2.518565317267399e-01, 3.320222155717508e-01,
             5.362184613722345e-01, 9.718245844328152e-02, 9.718245844328152e-02, 5.938377595573312e-01,
            -9.560677820122976e-02, 6.759540542547480e-01, 6.759540542547480e-01, 7.220870298624550e-01,
        ];
        #[rustfmt::skip]
        let vr_imag_correct = [
            0.0,  2.546315719275843e-01, -2.546315719275843e-01, 0.0,
            0.0, -5.224047347116287e-01,  5.224047347116287e-01, 0.0,
            0.0, -3.083837558972283e-01,  3.083837558972283e-01, 0.0,
            0.0,  0.0,                    0.0,                   0.0,
        ];
        assert_vec_approx_eq!(vr_real, vr_real_correct, 1e-15);
        assert_vec_approx_eq!(vr_imag, vr_imag_correct, 1e-15);

        // clear output arrays
        wr.iter_mut().map(|x| *x = 0.0).count();
        wi.iter_mut().map(|x| *x = 0.0).count();
        vl.iter_mut().map(|x| *x = 0.0).count();
        vr.iter_mut().map(|x| *x = 0.0).count();
        let n_zeros = vec![0.0; sz];
        let nn_zeros = vec![0.0; sz * sz];
        assert_eq!(wr, n_zeros);
        assert_eq!(wi, n_zeros);
        assert_eq!(vl, nn_zeros);
        assert_eq!(vr, nn_zeros);

        // auxiliary
        let mut empty: Vec<f64> = Vec::new();

        // compute eigen-things again, vl only
        dgeev(true, false, n, &mut a_copy1, &mut wr, &mut wi, &mut vl, &mut empty)?;

        // extract eigenvalues from dgeev data
        vl_real.iter_mut().map(|x| *x = 0.0).count();
        vl_imag.iter_mut().map(|x| *x = 0.0).count();
        dgeev_data(&mut vl_real, &mut vl_imag, &wi, &vl)?;

        // check left eigenvalues
        assert_vec_approx_eq!(vl_real, vl_real_correct, 1e-15);

        // compute eigen-things again, vr only
        dgeev(false, true, n, &mut a_copy2, &mut wr, &mut wi, &mut empty, &mut vr)?;

        // extract eigenvalues from dgeev data
        vr_real.iter_mut().map(|x| *x = 0.0).count();
        vr_imag.iter_mut().map(|x| *x = 0.0).count();
        dgeev_data(&mut vr_real, &mut vr_imag, &wi, &vr)?;

        // check left eigenvalues
        assert_vec_approx_eq!(vr_real, vr_real_correct, 1e-15);

        // done
        Ok(())
    }
}
