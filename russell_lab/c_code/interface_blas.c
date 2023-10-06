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
    if (norm_code == NORM_EUC || norm_code == NORM_FRO) {
        return FN_DLANGE("F", m, n, a, lda, work);
    } else if (norm_code == NORM_INF) {
        return FN_DLANGE("I", m, n, a, lda, work);
    } else if (norm_code == NORM_MAX) {
        return FN_DLANGE("M", m, n, a, lda, work);
    } else {
        return FN_DLANGE("O", m, n, a, lda, work); // norm_code == NORM_ONE
    }
}

// <http://www.netlib.org/lapack/explore-html/d5/d8f/zlange_8f.html>
double c_zlange(int32_t norm_code,
                const int32_t *m,
                const int32_t *n,
                const COMPLEX64 *a,
                const int32_t *lda,
                double *work) {
    if (norm_code == NORM_EUC || norm_code == NORM_FRO) {
        return FN_ZLANGE("F", m, n, a, lda, work);
    } else if (norm_code == NORM_INF) {
        return FN_ZLANGE("I", m, n, a, lda, work);
    } else if (norm_code == NORM_MAX) {
        return FN_ZLANGE("M", m, n, a, lda, work);
    } else {
        return FN_ZLANGE("O", m, n, a, lda, work); // norm_code == NORM_ONE
    }
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
