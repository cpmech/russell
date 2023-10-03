#include <inttypes.h>
#include <stdio.h>

#ifdef USE_INTEL_MKL
#include "mkl.h"
#else
#include "cblas.h"
#endif

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

// vector //////////////////////////////////////////////////////////////////////////////////////////////////////////////

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
// Assuming that:
//
// 1. OPENBLAS_CONST = const
// 2. blasint = int
// 3. MKL_INT = int
// 4. CBLAS_INDEX = int
//
// i.e., this will only work with i32 and f64
//
// By making the above substitutions in both cblas.h and mkl_cblas.h,
// we arrive at the following common C-interface:
//
// double cblas_ddot(const int N, const double *X, const int incX, const double *Y, const int incY);
// void cblas_dcopy(const int N, const double *X, const int incX, double *Y, const int incY);
// void cblas_zcopy(const int N, const void *X, const int incX, void *Y, const int incY);
// void cblas_dscal(const int N, const double alpha, double *X, const int incX);
// void cblas_zscal(const int N, const void *alpha, void *X, const int incX);
// void cblas_daxpy(const int N, const double alpha, const double *X, const int incX, double *Y, const int incY);
// void cblas_zaxpy(const int N, const void *alpha, const void *X, const int incX, void *Y, const int incY);
// double cblas_dnrm2(const int N, const double *X, const int incX);
// double cblas_dasum(const int N, const double *X, const int incX);
// int cblas_idamax(const int N, const double *X, const int incX);
//
// Note the "pointer to void" to represent complex numbers.
