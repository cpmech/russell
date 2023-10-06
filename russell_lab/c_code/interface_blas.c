#include <inttypes.h>
#include <stdio.h>

#ifdef USE_INTEL_MKL
#include "mkl.h"
#define COMPLEX64 MKL_Complex16
#define FN_DLANGE dlange_
#define FN_ZLANGE zlange_
#else
#include "cblas.h"
#include "lapack.h"
#define COMPLEX64 lapack_complex_double
#define FN_DLANGE LAPACK_dlange
#define FN_ZLANGE LAPACK_zlange
#endif

#include "constants.h"

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

double c_dlange(int32_t norm_code, const int32_t *m, const int32_t *n,
                const double *a, const int32_t *lda, double *work) {
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

double c_zlange(int32_t norm_code, const int32_t *m, const int32_t *n,
                const COMPLEX64 *a, const int32_t *lda, double *work) {
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

// Here, we assume that (SUBSTITUTIONS):
//
// 1. OPENBLAS_CONST = const
// 2. blasint = int
// 3. MKL_INT = int
// 4. CBLAS_INDEX = int
//
// i.e., this will only work with i32 and f64
//
// Note: CBLAS uses "pointer to void" to represent complex numbers.
//
// vector //////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// --- OpenBLAS --------------------------------------------------------------------------------------------------------
//
// From: /usr/include/x86_64-linux-gnu/cblas.h
//
// double cblas_ddot(OPENBLAS_CONST blasint n, OPENBLAS_CONST double *x, OPENBLAS_CONST blasint incx, OPENBLAS_CONST double *y, OPENBLAS_CONST blasint incy);
// void cblas_dcopy(OPENBLAS_CONST blasint n, OPENBLAS_CONST double *x, OPENBLAS_CONST blasint incx, double *y, OPENBLAS_CONST blasint incy);
// void cblas_zcopy(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx, void *y, OPENBLAS_CONST blasint incy);
// void cblas_dscal(OPENBLAS_CONST blasint N, OPENBLAS_CONST double alpha, double *X, OPENBLAS_CONST blasint incX);
// void cblas_zscal(OPENBLAS_CONST blasint N, OPENBLAS_CONST void *alpha, void *X, OPENBLAS_CONST blasint incX);
// void cblas_daxpy(OPENBLAS_CONST blasint n, OPENBLAS_CONST double alpha, OPENBLAS_CONST double *x, OPENBLAS_CONST blasint incx, double *y, OPENBLAS_CONST blasint incy);
// void cblas_zaxpy(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx, void *y, OPENBLAS_CONST blasint incy);
// double cblas_dnrm2 (OPENBLAS_CONST blasint N, OPENBLAS_CONST double *X, OPENBLAS_CONST blasint incX);
// double cblas_dasum (OPENBLAS_CONST blasint n, OPENBLAS_CONST double *x, OPENBLAS_CONST blasint incx);
// CBLAS_INDEX cblas_idamax(OPENBLAS_CONST blasint n, OPENBLAS_CONST double *x, OPENBLAS_CONST blasint incx);
//
// --- Intel MKL -------------------------------------------------------------------------------------------------------
//
// From: /opt/intel/oneapi/mkl/latest/include/mkl_cblas.h
//
// double cblas_ddot(const MKL_INT N, const double *X, const MKL_INT incX, const double *Y, const MKL_INT incY) NOTHROW;
// void cblas_dcopy(const MKL_INT N, const double *X, const MKL_INT incX, double *Y, const MKL_INT incY) NOTHROW;
// void cblas_zcopy(const MKL_INT N, const void *X, const MKL_INT incX, void *Y, const MKL_INT incY) NOTHROW;
// void cblas_dscal(const MKL_INT N, const double alpha, double *X, const MKL_INT incX) NOTHROW;
// void cblas_zscal(const MKL_INT N, const void *alpha, void *X, const MKL_INT incX) NOTHROW;
// void cblas_daxpy(const MKL_INT N, const double alpha, const double *X, const MKL_INT incX, double *Y, const MKL_INT incY) NOTHROW;
// void cblas_zaxpy(const MKL_INT N, const void *alpha, const void *X, const MKL_INT incX, void *Y, const MKL_INT incY) NOTHROW;
// double cblas_dnrm2(const MKL_INT N, const double *X, const MKL_INT incX) NOTHROW;
// double cblas_dasum(const MKL_INT N, const double *X, const MKL_INT incX) NOTHROW;
// CBLAS_INDEX cblas_idamax(const MKL_INT N, const double *X, const MKL_INT incX) NOTHROW;
//
// matrix //////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// --- OpenBLAS --------------------------------------------------------------------------------------------------------
//
// From: /usr/include/x86_64-linux-gnu/cblas.h
//
// typedef enum CBLAS_ORDER     {CblasRowMajor=101, CblasColMajor=102} CBLAS_ORDER;
// typedef enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113, CblasConjNoTrans=114} CBLAS_TRANSPOSE;
// typedef enum CBLAS_UPLO      {CblasUpper=121, CblasLower=122} CBLAS_UPLO;
// typedef enum CBLAS_DIAG      {CblasNonUnit=131, CblasUnit=132} CBLAS_DIAG;
// typedef enum CBLAS_SIDE      {CblasLeft=141, CblasRight=142} CBLAS_SIDE;
// typedef CBLAS_ORDER CBLAS_LAYOUT;
//
// void cblas_dgemm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA,
//                  OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N,
//                  OPENBLAS_CONST blasint K, OPENBLAS_CONST double alpha, OPENBLAS_CONST double *A,
//                  OPENBLAS_CONST blasint lda, OPENBLAS_CONST double *B, OPENBLAS_CONST blasint ldb,
//                  OPENBLAS_CONST double beta, double *C, OPENBLAS_CONST blasint ldc);
// void cblas_zgemm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA,
//                  OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N,
//                  OPENBLAS_CONST blasint K, OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A,
//                  OPENBLAS_CONST blasint lda, OPENBLAS_CONST void *B, OPENBLAS_CONST blasint ldb,
//                  OPENBLAS_CONST void *beta, void *C, OPENBLAS_CONST blasint ldc);
// void cblas_dsyrk(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_UPLO Uplo,
//                  OPENBLAS_CONST enum CBLAS_TRANSPOSE Trans, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
//                  OPENBLAS_CONST double alpha, OPENBLAS_CONST double *A, OPENBLAS_CONST blasint lda,
//                  OPENBLAS_CONST double beta, double *C, OPENBLAS_CONST blasint ldc);
// void cblas_zsyrk(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_UPLO Uplo,
//                  OPENBLAS_CONST enum CBLAS_TRANSPOSE Trans, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
//                  OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda,
//                  OPENBLAS_CONST void *beta, void *C, OPENBLAS_CONST blasint ldc);
// void cblas_zherk(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_UPLO Uplo,
//                  OPENBLAS_CONST enum CBLAS_TRANSPOSE Trans, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
//                  OPENBLAS_CONST double alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda,
//                  OPENBLAS_CONST double beta, void *C, OPENBLAS_CONST blasint ldc);
//
// --- Intel MKL -------------------------------------------------------------------------------------------------------
//
// From: /opt/intel/oneapi/mkl/latest/include/mkl_cblas.h
//
// enum CBLAS_LAYOUT {CblasRowMajor=101, CblasColMajor=102};
// enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};
// enum CBLAS_UPLO {CblasUpper=121, CblasLower=122};
// enum CBLAS_DIAG {CblasNonUnit=131, CblasUnit=132};
// enum CBLAS_SIDE {CblasLeft=141, CblasRight=142};
// enum CBLAS_STORAGE {CblasPacked=151};
// enum CBLAS_IDENTIFIER {CblasAMatrix=161, CblasBMatrix=162};
// enum CBLAS_OFFSET {CblasRowOffset=171, CblasColOffset=172, CblasFixOffset=173};
// typedef CBLAS_LAYOUT CBLAS_ORDER; /* this for backward compatibility with CBLAS_ORDER */
//
// void cblas_dgemm(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
//                  const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N,
//                  const MKL_INT K, const double alpha, const double *A,
//                  const MKL_INT lda, const double *B, const MKL_INT ldb,
//                  const double beta, double *C, const MKL_INT ldc) NOTHROW;
// void cblas_zgemm(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
//                  const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N,
//                  const MKL_INT K, const void *alpha, const void *A,
//                  const MKL_INT lda, const void *B, const MKL_INT ldb,
//                  const void *beta, void *C, const MKL_INT ldc) NOTHROW;
// void cblas_dsyrk(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
//                  const CBLAS_TRANSPOSE Trans, const MKL_INT N, const MKL_INT K,
//                  const double alpha, const double *A, const MKL_INT lda,
//                  const double beta, double *C, const MKL_INT ldc) NOTHROW;
// void cblas_zsyrk(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
//                  const CBLAS_TRANSPOSE Trans, const MKL_INT N, const MKL_INT K,
//                  const void *alpha, const void *A, const MKL_INT lda,
//                  const void *beta, void *C, const MKL_INT ldc) NOTHROW;
// void cblas_zherk(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
//                  const CBLAS_TRANSPOSE Trans, const MKL_INT N, const MKL_INT K,
//                  const double alpha, const void *A, const MKL_INT lda,
//                  const double beta, void *C, const MKL_INT ldc) NOTHROW;
//
// matrix-vector ///////////////////////////////////////////////////////////////////////////////////////////////////////
//
// --- OpenBLAS --------------------------------------------------------------------------------------------------------
//
// From: /usr/include/x86_64-linux-gnu/cblas.h
//
// void cblas_dgemv(OPENBLAS_CONST enum CBLAS_ORDER order,
//                  OPENBLAS_CONST enum CBLAS_TRANSPOSE trans,  OPENBLAS_CONST blasint m, OPENBLAS_CONST blasint n,
//                  OPENBLAS_CONST double alpha, OPENBLAS_CONST double  *a, OPENBLAS_CONST blasint lda,
//                  OPENBLAS_CONST double  *x, OPENBLAS_CONST blasint incx,  OPENBLAS_CONST double beta,
//                  double  *y, OPENBLAS_CONST blasint incy);
//
// --- Intel MKL -------------------------------------------------------------------------------------------------------
//
// From: /opt/intel/oneapi/mkl/latest/include/mkl_cblas.h
//
// void cblas_dgemv(const CBLAS_LAYOUT Layout,
//                  const CBLAS_TRANSPOSE TransA, const MKL_INT M, const MKL_INT N,
//                  const double alpha, const double *A, const MKL_INT lda,
//                  const double *X, const MKL_INT incX, const double beta,
//                  double *Y, const MKL_INT incY) NOTHROW;
//
// LAPACK //////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// --- OpenBLAS --------------------------------------------------------------------------------------------------------
//
// From: /usr/include/lapack.h
//
// NOTE: The functions with char arguments will have to be wrapped and called with the LAPACK_ prefix.
//
// void LAPACK_dgesv(lapack_int const* n, lapack_int const* nrhs, double* A,
//                   lapack_int const* lda, lapack_int* ipiv, double* B, lapack_int const* ldb,
//                   lapack_int* info );
// double LAPACK_dlange(char const *norm, lapack_int const *m, lapack_int const *n,
//                           double const *A, lapack_int const *lda, double *work);
// double LAPACK_zlange(char const *norm, lapack_int const *m, lapack_int const *n,
//                           lapack_complex_double const *A, lapack_int const *lda, double *work);
//
// --- Intel MKL -------------------------------------------------------------------------------------------------------
//
// From: /opt/intel/oneapi/mkl/latest/include/mkl_lapack.h
// (note the trailing underscore!)
//
// void dgesv_( const MKL_INT* n, const MKL_INT* nrhs, double* a,
//             const MKL_INT* lda, MKL_INT* ipiv, double* b, const MKL_INT* ldb,
//             MKL_INT* info ) NOTHROW;
// double dlange_( const char* norm, const MKL_INT* m, const MKL_INT* n,
//                 const double* a, const MKL_INT* lda, double* work ) NOTHROW;
// double zlange_( const char* norm, const MKL_INT* m, const MKL_INT* n,
//                 const MKL_Complex16* a, const MKL_INT* lda, double* work ) NOTHROW;
//