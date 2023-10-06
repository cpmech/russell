#include <inttypes.h>
#include <stdio.h>

#ifdef USE_INTEL_MKL
#include "mkl.h"
#define COMPLEX64 MKL_Complex16
#define FN_DGESV dgesv_
#define FN_DLANGE dlange_
#define FN_ZLANGE zlange_
#define FN_DPOTRF dpotrf_
#define FN_DSYEV dsyev_
#define FN_DGEEV dgeev_
#define FN_DGESVD dgesvd_
#else
#include "cblas.h"
#include "lapack.h"
#define COMPLEX64 lapack_complex_double
#define FN_DGESV LAPACK_dgesv
#define FN_DLANGE LAPACK_dlange
#define FN_ZLANGE LAPACK_zlange
#define FN_DPOTRF LAPACK_dpotrf
#define FN_DSYEV LAPACK_dsyev
#define FN_DGEEV LAPACK_dgeev
#define FN_DGESVD LAPACK_dgesvd
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

// <http://www.netlib.org/lapack/explore-html/d8/d72/dgesv_8f.html>
void c_dgesv(const int32_t *n, const int32_t *nrhs, double *a,
             const int32_t *lda, int32_t *ipiv, double *b, const int32_t *ldb, int32_t *info) {
    dgesv_(n, nrhs, a, lda, ipiv, b, ldb, info);
}

// <http://www.netlib.org/lapack/explore-html/dc/d09/dlange_8f.html>
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

// <http://www.netlib.org/lapack/explore-html/d5/d8f/zlange_8f.html>
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

// <http://www.netlib.org/lapack/explore-html/d0/d8a/dpotrf_8f.html>
void c_dpotrf(C_BOOL upper,
              const int32_t *n,
              double *a,
              const int32_t *lda,
              int32_t *info) {
    const char *uplo = upper == C_TRUE ? "U" : "L";
    FN_DPOTRF(uplo, n, a, lda, info);
}

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

// <http://www.netlib.org/lapack/explore-html/d9/d28/dgeev_8f.html>
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

// <http://www.netlib.org/lapack/explore-html/d8/d2d/dgesvd_8f.html>
void c_dgesvd(int32_t jobu_code,
              int32_t jobvt_code,
              const MKL_INT *m,
              const MKL_INT *n,
              double *a,
              const MKL_INT *lda,
              double *s,
              double *u,
              const MKL_INT *ldu,
              double *vt,
              const MKL_INT *ldvt,
              double *work,
              const MKL_INT *lwork,
              MKL_INT *info) {
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
