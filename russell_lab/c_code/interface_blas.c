#include <inttypes.h>
#include <stdio.h>

#ifdef USE_INTEL_MKL
#include "mkl.h"
#define COMPLEX64 MKL_Complex16
#define FN_DGESV dgesv_
#define FN_ZGESV zgesv_
#define FN_DLANGE dlange_
#define FN_ZLANGE zlange_
#define FN_DPOTRF dpotrf_
#define FN_DSYEV dsyev_
#define FN_DGEEV dgeev_
#define FN_ZGEEV zgeev_
#define FN_ZHEEV zheev_
#define FN_DGGEV dggev_
#define FN_ZGGEV zggev_
#define FN_DGESVD dgesvd_
#define FN_DGETRF dgetrf_
#define FN_DGETRI dgetri_
#define FN_ZGETRF zgetrf_
#define FN_ZGETRI zgetri_
#else
#include "cblas.h"
#include "lapack.h"
#define COMPLEX64 lapack_complex_double
#define FN_DGESV LAPACK_dgesv
#define FN_ZGESV LAPACK_zgesv
#define FN_DLANGE LAPACK_dlange
#define FN_ZLANGE LAPACK_zlange
#define FN_DPOTRF LAPACK_dpotrf
#define FN_DSYEV LAPACK_dsyev
#define FN_DGEEV LAPACK_dgeev
#define FN_ZGEEV LAPACK_zgeev
#define FN_ZHEEV LAPACK_zheev
#define FN_DGGEV LAPACK_dggev
#define FN_ZGGEV LAPACK_zggev
#define FN_DGESVD LAPACK_dgesvd
#define FN_DGETRF LAPACK_dgetrf
#define FN_DGETRI LAPACK_dgetri
#define FN_ZGETRF LAPACK_zgetrf
#define FN_ZGETRI LAPACK_zgetri
#endif

#include "constants.h"

// Here, we assume that:
//
// 1. OPENBLAS_CONST = const
// 2. CBLAS_INDEX = int
// 3. blasint = int
// 4. MKL_INT = int
//
// i.e., this will only work with i32 and f64
//
// Note: CBLAS uses "pointer to void" to represent complex numbers.
//
// OpenBLAS include files:
//
// 1. /usr/include/x86_64-linux-gnu/cblas.h
// 2. /usr/include/lapack.h
//
// Intel MKL include files:
//
// 1. /opt/intel/oneapi/mkl/latest/include/mkl_cblas.h
// 2. /opt/intel/oneapi/mkl/latest/include/mkl_lapack.h

int32_t c_using_intel_mkl() {
#ifdef USE_INTEL_MKL
    return 1;
#else
    return 0;
#endif
}

void c_set_num_threads(int32_t n) {
#ifdef USE_INTEL_MKL
    MKL_Set_Num_Threads(n);
#else
    openblas_set_num_threads(n);
#endif
}

int32_t c_get_num_threads() {
#ifdef USE_INTEL_MKL
    return mkl_get_max_threads();
#else
    return openblas_get_num_threads();
#endif
}

// Computes the solution to a system of linear equations
// <https://www.netlib.org/lapack/explore-html/d8/d72/dgesv_8f.html>
void c_dgesv(const int32_t *n,
             const int32_t *nrhs,
             double *a,
             const int32_t *lda,
             int32_t *ipiv,
             double *b,
             const int32_t *ldb,
             int32_t *info) {
    FN_DGESV(n, nrhs, a, lda, ipiv, b, ldb, info);
}

// Computes the solution to a real system of linear equations (complex version)
// <https://www.netlib.org/lapack/explore-html/d1/ddc/zgesv_8f.html>
void c_zgesv(const int32_t *n,
             const int32_t *nrhs,
             COMPLEX64 *a,
             const int32_t *lda,
             int32_t *ipiv,
             COMPLEX64 *b,
             const int32_t *ldb,
             int32_t *info) {
    FN_ZGESV(n, nrhs, a, lda, ipiv, b, ldb, info);
}

// Computes the matrix norm
// <https://www.netlib.org/lapack/explore-html/dc/d09/dlange_8f.html>
double c_dlange(int32_t norm_code,
                const int32_t *m,
                const int32_t *n,
                const double *a,
                const int32_t *lda,
                double *work) {
    const char *norm = norm_code == NORM_EUC || norm_code == NORM_FRO ? "F"
                       : norm_code == NORM_INF                        ? "I"
                       : norm_code == NORM_MAX                        ? "M"
                                                                      : "O";
    return FN_DLANGE(norm, m, n, a, lda, work);
}

// Computes the matrix norm (complex version)
// <https://www.netlib.org/lapack/explore-html/d5/d8f/zlange_8f.html>
double c_zlange(int32_t norm_code,
                const int32_t *m,
                const int32_t *n,
                const COMPLEX64 *a,
                const int32_t *lda,
                double *work) {
    const char *norm = norm_code == NORM_EUC || norm_code == NORM_FRO ? "F"
                       : norm_code == NORM_INF                        ? "I"
                       : norm_code == NORM_MAX                        ? "M"
                                                                      : "O";
    return FN_ZLANGE(norm, m, n, a, lda, work);
}

// Computes the Cholesky factorization of a real symmetric positive definite matrix
// <https://www.netlib.org/lapack/explore-html/d0/d8a/dpotrf_8f.html>
void c_dpotrf(C_BOOL upper,
              const int32_t *n,
              double *a,
              const int32_t *lda,
              int32_t *info) {
    const char *uplo = upper == C_TRUE ? "U" : "L";
    FN_DPOTRF(uplo, n, a, lda, info);
}

// Computes the eigenvalues and eigenvectors of a symmetric matrix
// <https://netlib.org/lapack/explore-html/dd/d4c/dsyev_8f.html>
void c_dsyev(C_BOOL calc_v,
             C_BOOL upper,
             const int32_t *n,
             double *a,
             const int32_t *lda,
             double *w,
             double *work,
             const int32_t *lwork,
             int32_t *info) {
    const char *jobz = calc_v == C_TRUE ? "V" : "N";
    const char *uplo = upper == C_TRUE ? "U" : "L";
    FN_DSYEV(jobz, uplo, n, a, lda, w, work, lwork, info);
}

// Computes the eigenvalues and eigenvectors of a general matrix
// <https://www.netlib.org/lapack/explore-html/d9/d28/dgeev_8f.html>
void c_dgeev(C_BOOL calc_vl,
             C_BOOL calc_vr,
             const int32_t *n,
             double *a,
             const int32_t *lda,
             double *wr,
             double *wi,
             double *vl,
             const int32_t *ldvl,
             double *vr,
             const int32_t *ldvr,
             double *work,
             const int32_t *lwork,
             int32_t *info) {
    const char *jobvl = calc_vl == C_TRUE ? "V" : "N";
    const char *jobvr = calc_vr == C_TRUE ? "V" : "N";
    FN_DGEEV(jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, work, lwork, info);
}

// Computes the eigenvalues and, optionally, the left and/or right eigenvectors for GE matrices
// <https://www.netlib.org/lapack/explore-html/dd/dba/zgeev_8f.html>
void c_zgeev(
    C_BOOL calc_vl,
    C_BOOL calc_vr,
    const int32_t *n,
    COMPLEX64 *a,
    const int32_t *lda,
    COMPLEX64 *w,
    COMPLEX64 *vl,
    const int32_t *ldvl,
    COMPLEX64 *vr,
    const int32_t *ldvr,
    COMPLEX64 *work,
    const int32_t *lwork,
    double *rwork,
    int32_t *info) {
    const char *jobvl = calc_vl == C_TRUE ? "V" : "N";
    const char *jobvr = calc_vr == C_TRUE ? "V" : "N";
    FN_ZGEEV(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr, work, lwork, rwork, info);
}

// Computes the eigenvalues and, optionally, the left and/or right eigenvectors for HE matrices
// <https://www.netlib.org/lapack/explore-html/d6/dee/zheev_8f.html>
void c_zheev(
    C_BOOL calc_v,
    C_BOOL upper,
    int32_t const *n,
    COMPLEX64 *a,
    int32_t const *lda,
    double *w,
    COMPLEX64 *work,
    int32_t const *lwork,
    double *rwork,
    int32_t *info) {
    const char *jobz = calc_v == C_TRUE ? "V" : "N";
    const char *uplo = upper == C_TRUE ? "U" : "L";
    FN_ZHEEV(jobz, uplo, n, a, lda, w, work, lwork, rwork, info);
}

// Computes the eigenvalues and, optionally, the left and/or right eigenvectors for GE matrices
// <https://www.netlib.org/lapack/explore-html/d9/d52/dggev_8f.html>
void c_dggev(
    C_BOOL calc_vl,
    C_BOOL calc_vr,
    const int32_t *n,
    double *a,
    const int32_t *lda,
    double *b,
    const int32_t *ldb,
    double *alphar,
    double *alphai,
    double *beta,
    double *vl,
    const int32_t *ldvl,
    double *vr,
    const int32_t *ldvr,
    double *work,
    const int32_t *lwork,
    int32_t *info) {
    const char *jobvl = calc_vl == C_TRUE ? "V" : "N";
    const char *jobvr = calc_vr == C_TRUE ? "V" : "N";
    FN_DGGEV(jobvl, jobvr, n, a, lda, b, ldb, alphar, alphai, beta, vl, ldvl, vr, ldvr, work, lwork, info);
}

// Computes the eigenvalues and, optionally, the left and/or right eigenvectors for GE matrices
// <https://www.netlib.org/lapack/explore-html/d3/d47/zggev_8f.html>
void c_zggev(
    C_BOOL calc_vl,
    C_BOOL calc_vr,
    const int32_t *n,
    COMPLEX64 *a,
    const int32_t *lda,
    COMPLEX64 *b,
    const int32_t *ldb,
    COMPLEX64 *alpha,
    COMPLEX64 *beta,
    COMPLEX64 *vl,
    const int32_t *ldvl,
    COMPLEX64 *vr,
    const int32_t *ldvr,
    COMPLEX64 *work,
    const int32_t *lwork,
    double *rwork,
    int32_t *info) {
    const char *jobvl = calc_vl == C_TRUE ? "V" : "N";
    const char *jobvr = calc_vr == C_TRUE ? "V" : "N";
    FN_ZGGEV(jobvl, jobvr, n, a, lda, b, ldb, alpha, beta, vl, ldvl, vr, ldvr, work, lwork, rwork, info);
}

// Computes the singular value decomposition (SVD)
// <https://www.netlib.org/lapack/explore-html/d8/d2d/dgesvd_8f.html>
void c_dgesvd(int32_t jobu_code,
              int32_t jobvt_code,
              const int32_t *m,
              const int32_t *n,
              double *a,
              const int32_t *lda,
              double *s,
              double *u,
              const int32_t *ldu,
              double *vt,
              const int32_t *ldvt,
              double *work,
              const int32_t *lwork,
              int32_t *info) {
    const char *jobu = jobu_code == SVD_CODE_A   ? "A"
                       : jobu_code == SVD_CODE_S ? "S"
                       : jobu_code == SVD_CODE_O ? "O"
                                                 : "N";
    const char *jobvt = jobvt_code == SVD_CODE_A   ? "A"
                        : jobvt_code == SVD_CODE_S ? "S"
                        : jobvt_code == SVD_CODE_O ? "O"
                                                   : "N";
    FN_DGESVD(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info);
}

// Computes the LU factorization of a general (m,n) matrix
// <https://www.netlib.org/lapack/explore-html/d3/d6a/dgetrf_8f.html>
void c_dgetrf(const int32_t *m,
              const int32_t *n,
              double *a,
              const int32_t *lda,
              int32_t *ipiv,
              int32_t *info) {
    FN_DGETRF(m, n, a, lda, ipiv, info);
}

// Computes the inverse of a matrix using the LU factorization computed by dgetrf
// <https://www.netlib.org/lapack/explore-html/df/da4/dgetri_8f.html>
void c_dgetri(const int32_t *n,
              double *a,
              const int32_t *lda,
              const int32_t *ipiv,
              double *work,
              const int32_t *lwork,
              int32_t *info) {
    FN_DGETRI(n, a, lda, ipiv, work, lwork, info);
}

// Computes the LU factorization of a general (m,n) matrix
// <https://www.netlib.org/lapack/explore-html/dd/dd1/zgetrf_8f.html>
void c_zgetrf(const int32_t *m,
              const int32_t *n,
              COMPLEX64 *a,
              const int32_t *lda,
              int32_t *ipiv,
              int32_t *info) {
    FN_ZGETRF(m, n, a, lda, ipiv, info);
}

// Computes the inverse of a matrix using the LU factorization computed by zgetrf
// <https://www.netlib.org/lapack/explore-html/d0/db3/zgetri_8f.html>
void c_zgetri(const int32_t *n,
              COMPLEX64 *a,
              const int32_t *lda,
              const int32_t *ipiv,
              COMPLEX64 *work,
              const int32_t *lwork,
              int32_t *info) {
    FN_ZGETRI(n, a, lda, ipiv, work, lwork, info);
}
