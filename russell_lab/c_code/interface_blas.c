#include <inttypes.h>
#include <stdio.h>

// The vendor headers, the `COMPLEX64` type and the prototypes of every entry
// point in this file live in `interface.h`. Including it here is what makes a
// definition that drifts from the declaration a compile error.
#include "interface.h"

#ifdef USE_INTEL_MKL
#define FN_DGESV dgesv_
#define FN_ZGESV zgesv_
#define FN_DLANGE dlange_
#define FN_ZLANGE zlange_
#define FN_DPOTRF dpotrf_
#define FN_ZPOTRF zpotrf_
#define FN_DSYEV dsyev_
#define FN_DGEEV dgeev_
#define FN_ZGEEV zgeev_
#define FN_ZHEEV zheev_
#define FN_DGGEV dggev_
#define FN_ZGGEV zggev_
#define FN_DGESVD dgesvd_
#define FN_ZGESVD zgesvd_
#define FN_DGETRF dgetrf_
#define FN_DGETRI dgetri_
#define FN_ZGETRF zgetrf_
#define FN_ZGETRI zgetri_
#else
#define FN_DGESV LAPACK_dgesv
#define FN_ZGESV LAPACK_zgesv
#define FN_DLANGE LAPACK_dlange
#define FN_ZLANGE LAPACK_zlange
#define FN_DPOTRF LAPACK_dpotrf
#define FN_ZPOTRF LAPACK_zpotrf
#define FN_DSYEV LAPACK_dsyev
#define FN_DGEEV LAPACK_dgeev
#define FN_ZGEEV LAPACK_zgeev
#define FN_ZHEEV LAPACK_zheev
#define FN_DGGEV LAPACK_dggev
#define FN_ZGGEV LAPACK_zggev
#define FN_DGESVD LAPACK_dgesvd
#define FN_ZGESVD LAPACK_zgesvd
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

int32_t c_using_intel_mkl(void) {
#ifdef USE_INTEL_MKL
    return 1;
#else
    return 0;
#endif
}

#ifndef USE_INTEL_MKL
// A vendor-neutral `cblas.h` does not declare these OpenBLAS threading calls,
// even though the symbols are there whenever `libblas` is OpenBLAS underneath
// (as with Nix's blas/lapack). Declaring them here is harmless when `cblas.h`
// already does, as the signatures match.
extern void openblas_set_num_threads(int num_threads);
extern int openblas_get_num_threads(void);
#endif

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

// ---------------------------------------------------------------------------------------------
// BLAS entry points routed by src/base/blas_dispatch.rs
// ---------------------------------------------------------------------------------------------
//
// Each wrapper is one CBLAS call with the arguments the crate's own kernels take,
// so the Rust side has a single signature per routine regardless of the library.
// Two conventions differ between the real and the complex CBLAS routines: the
// real ones take `alpha`/`beta` by value, while the complex ones take pointers to
// the scalar (CBLAS represents a complex value as `void *`). The wrappers keep
// the crate's convention on both sides, which is why `c_zgemm` takes
// `COMPLEX64 *` for the scalars.
//
// CBLAS's index type is `blasint` on OpenBLAS and `MKL_INT` on MKL; both are
// `int` in the configurations this shim supports, as noted at the top of the file.

// c := alpha a b + beta c (both no-trans), all matrices column-major
void c_dgemm(double alpha,
             const double *a,
             const double *b,
             double beta,
             double *c,
             const int32_t *m,
             const int32_t *n,
             const int32_t *k) {
    cblas_dgemm(CblasColMajor,
                CblasNoTrans,
                CblasNoTrans,
                (int)(*m),
                (int)(*n),
                (int)(*k),
                alpha,
                a,
                (int)(*m),
                b,
                (int)(*k),
                beta,
                c,
                (int)(*m));
}

// y := alpha a x + beta y
void c_dgemv(double alpha,
             const double *a,
             const double *x,
             double beta,
             double *y,
             const int32_t *m,
             const int32_t *n) {
    cblas_dgemv(CblasColMajor, CblasNoTrans, (int)(*m), (int)(*n), alpha, a, (int)(*m), x, 1, beta, y, 1);
}

// y := alpha aᵀ x + beta y
void c_dgemv_trans(double alpha,
                   const double *a,
                   const double *x,
                   double beta,
                   double *y,
                   const int32_t *m,
                   const int32_t *n) {
    cblas_dgemv(CblasColMajor, CblasTrans, (int)(*m), (int)(*n), alpha, a, (int)(*m), x, 1, beta, y, 1);
}

// x ⋅ y
double c_ddot(const double *x, const double *y, const int32_t *n) {
    return cblas_ddot((int)(*n), x, 1, y, 1);
}

// ‖x‖₂
double c_dnrm2(const double *x, const int32_t *n) {
    return cblas_dnrm2((int)(*n), x, 1);
}

// ‖x‖₁
double c_dasum(const double *x, const int32_t *n) {
    return cblas_dasum((int)(*n), x, 1);
}

// Index of the first maximum absolute value, 0-based as CBLAS returns it.
int32_t c_idamax(const double *x, const int32_t *n) {
    if (*n <= 0) {
        return 0;
    }
    return (int32_t)cblas_idamax((int)(*n), x, 1);
}

// y := alpha x + y
void c_daxpy(double alpha, const double *x, double *y, const int32_t *n) {
    cblas_daxpy((int)(*n), alpha, x, 1, y, 1);
}

// c := alpha a b + beta c for complex data (both no-trans), using the same
// pointer-to-scalar convention as cblas_zgemm itself
void c_zgemm(const COMPLEX64 *alpha,
             const COMPLEX64 *a,
             const COMPLEX64 *b,
             const COMPLEX64 *beta,
             COMPLEX64 *c,
             const int32_t *m,
             const int32_t *n,
             const int32_t *k) {
    cblas_zgemm(CblasColMajor,
                CblasNoTrans,
                CblasNoTrans,
                (int)(*m),
                (int)(*n),
                (int)(*k),
                (const void *)alpha,
                (const void *)a,
                (int)(*m),
                (const void *)b,
                (int)(*k),
                (const void *)beta,
                (void *)c,
                (int)(*m));
}

// c := alpha aᵀ b + beta c; a is k×m (lda = k), b is k×n (ldb = k)
void c_dgemm_tn(double alpha,
                const double *a,
                const double *b,
                double beta,
                double *c,
                const int32_t *m,
                const int32_t *n,
                const int32_t *k) {
    cblas_dgemm(CblasColMajor,
                CblasTrans,
                CblasNoTrans,
                (int)(*m),
                (int)(*n),
                (int)(*k),
                alpha,
                a,
                (int)(*k),
                b,
                (int)(*k),
                beta,
                c,
                (int)(*m));
}

// c := alpha a bᵀ + beta c; a is m×k (lda = m), b is n×k (ldb = n)
void c_dgemm_nt(double alpha,
                const double *a,
                const double *b,
                double beta,
                double *c,
                const int32_t *m,
                const int32_t *n,
                const int32_t *k) {
    cblas_dgemm(CblasColMajor,
                CblasNoTrans,
                CblasTrans,
                (int)(*m),
                (int)(*n),
                (int)(*k),
                alpha,
                a,
                (int)(*m),
                b,
                (int)(*n),
                beta,
                c,
                (int)(*m));
}

// c := alpha aᵀ bᵀ + beta c; a is k×m (lda = k), b is n×k (ldb = n)
void c_dgemm_tt(double alpha,
                const double *a,
                const double *b,
                double beta,
                double *c,
                const int32_t *m,
                const int32_t *n,
                const int32_t *k) {
    cblas_dgemm(CblasColMajor,
                CblasTrans,
                CblasTrans,
                (int)(*m),
                (int)(*n),
                (int)(*k),
                alpha,
                a,
                (int)(*k),
                b,
                (int)(*n),
                beta,
                c,
                (int)(*m));
}

// c := alpha aᵀ b + beta c for complex data (not conjugated)
void c_zgemm_tn(const COMPLEX64 *alpha,
                const COMPLEX64 *a,
                const COMPLEX64 *b,
                const COMPLEX64 *beta,
                COMPLEX64 *c,
                const int32_t *m,
                const int32_t *n,
                const int32_t *k) {
    cblas_zgemm(CblasColMajor,
                CblasTrans,
                CblasNoTrans,
                (int)(*m),
                (int)(*n),
                (int)(*k),
                (const void *)alpha,
                (const void *)a,
                (int)(*k),
                (const void *)b,
                (int)(*k),
                (const void *)beta,
                (void *)c,
                (int)(*m));
}

// c := alpha a bᵀ + beta c for complex data (not conjugated)
void c_zgemm_nt(const COMPLEX64 *alpha,
                const COMPLEX64 *a,
                const COMPLEX64 *b,
                const COMPLEX64 *beta,
                COMPLEX64 *c,
                const int32_t *m,
                const int32_t *n,
                const int32_t *k) {
    cblas_zgemm(CblasColMajor,
                CblasNoTrans,
                CblasTrans,
                (int)(*m),
                (int)(*n),
                (int)(*k),
                (const void *)alpha,
                (const void *)a,
                (int)(*m),
                (const void *)b,
                (int)(*n),
                (const void *)beta,
                (void *)c,
                (int)(*m));
}

// c := alpha aᵀ bᵀ + beta c for complex data (not conjugated)
void c_zgemm_tt(const COMPLEX64 *alpha,
                const COMPLEX64 *a,
                const COMPLEX64 *b,
                const COMPLEX64 *beta,
                COMPLEX64 *c,
                const int32_t *m,
                const int32_t *n,
                const int32_t *k) {
    cblas_zgemm(CblasColMajor,
                CblasTrans,
                CblasTrans,
                (int)(*m),
                (int)(*n),
                (int)(*k),
                (const void *)alpha,
                (const void *)a,
                (int)(*k),
                (const void *)b,
                (int)(*n),
                (const void *)beta,
                (void *)c,
                (int)(*m));
}

// y := alpha a x + beta y for complex data
void c_zgemv(const COMPLEX64 *alpha,
             const COMPLEX64 *a,
             const COMPLEX64 *x,
             const COMPLEX64 *beta,
             COMPLEX64 *y,
             const int32_t *m,
             const int32_t *n) {
    cblas_zgemv(CblasColMajor,
                CblasNoTrans,
                (int)(*m),
                (int)(*n),
                (const void *)alpha,
                (const void *)a,
                (int)(*m),
                (const void *)x,
                1,
                (const void *)beta,
                (void *)y,
                1);
}

// y := alpha aᵀ x + beta y for complex data (not conjugated)
void c_zgemv_trans(const COMPLEX64 *alpha,
                   const COMPLEX64 *a,
                   const COMPLEX64 *x,
                   const COMPLEX64 *beta,
                   COMPLEX64 *y,
                   const int32_t *m,
                   const int32_t *n) {
    cblas_zgemv(CblasColMajor,
                CblasTrans,
                (int)(*m),
                (int)(*n),
                (const void *)alpha,
                (const void *)a,
                (int)(*m),
                (const void *)x,
                1,
                (const void *)beta,
                (void *)y,
                1);
}

// a := alpha x yᵀ + a (dger, the unconjugated rank-1 update)
void c_dger(double alpha, const double *x, const double *y, double *a, const int32_t *m, const int32_t *n) {
    cblas_dger(CblasColMajor, (int)(*m), (int)(*n), alpha, x, 1, y, 1, a, (int)(*m));
}

// a := alpha x yᵀ + a for complex data. BLAS's `zger` is the *unconjugated*
// rank-1 update, which CBLAS spells `zgeru` (`zgerc` conjugates y).
void c_zger(const COMPLEX64 *alpha,
            const COMPLEX64 *x,
            const COMPLEX64 *y,
            COMPLEX64 *a,
            const int32_t *m,
            const int32_t *n) {
    cblas_zgeru(CblasColMajor,
                (int)(*m),
                (int)(*n),
                (const void *)alpha,
                (const void *)x,
                1,
                (const void *)y,
                1,
                (void *)a,
                (int)(*m));
}

// c := alpha a aᵀ + beta c, or c := alpha aᵀ a + beta c when `transposed`.
//
// The leading dimension follows the *stored* shape, which is the whole trap here:
// `a` is n×k without the transpose and k×n with it, so lda is n and k
// respectively. Only the requested triangle of c is written.
void c_dsyrk(double alpha,
             const double *a,
             double beta,
             double *c,
             const int32_t *n,
             const int32_t *k,
             C_BOOL upper,
             C_BOOL transposed) {
    cblas_dsyrk(CblasColMajor,
                upper == C_TRUE ? CblasUpper : CblasLower,
                transposed == C_TRUE ? CblasTrans : CblasNoTrans,
                (int)(*n),
                (int)(*k),
                alpha,
                a,
                transposed == C_TRUE ? (int)(*k) : (int)(*n),
                beta,
                c,
                (int)(*n));
}

// y := alpha x + y for complex data
void c_zaxpy(const COMPLEX64 *alpha, const COMPLEX64 *x, COMPLEX64 *y, const int32_t *n) {
    cblas_zaxpy((int)(*n), (const void *)alpha, (const void *)x, 1, (void *)y, 1);
}

// x := alpha x for complex data
void c_zscal(const COMPLEX64 *alpha, COMPLEX64 *x, const int32_t *n) {
    cblas_zscal((int)(*n), (const void *)alpha, (void *)x, 1);
}

// ‖x‖₂ for complex data
double c_dznrm2(const COMPLEX64 *x, const int32_t *n) {
    return cblas_dznrm2((int)(*n), (const void *)x, 1);
}

// c := alpha a aᵀ + beta c, or c := alpha aᵀ a + beta c when `transposed`.
//
// The *symmetric* rank-k update for complex data: the transpose is plain, not
// conjugated, which is what makes it `zsyrk` and not `zherk`. Alpha and beta are
// complex pointers, as CBLAS takes them for the complex routines. Same
// leading-dimension rule as `c_dsyrk`.
void c_zsyrk(const COMPLEX64 *alpha,
             const COMPLEX64 *a,
             const COMPLEX64 *beta,
             COMPLEX64 *c,
             const int32_t *n,
             const int32_t *k,
             C_BOOL upper,
             C_BOOL transposed) {
    cblas_zsyrk(CblasColMajor,
                upper == C_TRUE ? CblasUpper : CblasLower,
                transposed == C_TRUE ? CblasTrans : CblasNoTrans,
                (int)(*n),
                (int)(*k),
                (const void *)alpha,
                (const void *)a,
                transposed == C_TRUE ? (int)(*k) : (int)(*n),
                (const void *)beta,
                (void *)c,
                (int)(*n));
}

// c := alpha a aᴴ + beta c, or c := alpha aᴴ a + beta c when `conj_transposed`.
//
// `alpha` and `beta` are real, as `zherk` requires. Same leading-dimension rule
// as `c_dsyrk`.
void c_zherk(double alpha,
             const COMPLEX64 *a,
             double beta,
             COMPLEX64 *c,
             const int32_t *n,
             const int32_t *k,
             C_BOOL upper,
             C_BOOL conj_transposed) {
    cblas_zherk(CblasColMajor,
                upper == C_TRUE ? CblasUpper : CblasLower,
                conj_transposed == C_TRUE ? CblasConjTrans : CblasNoTrans,
                (int)(*n),
                (int)(*k),
                alpha,
                (const void *)a,
                conj_transposed == C_TRUE ? (int)(*k) : (int)(*n),
                beta,
                (void *)c,
                (int)(*n));
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

// Computes the solution to a real system of linear equations
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

// Computes the matrix norm
// <https://www.netlib.org/lapack/explore-html/d5/d8f/zlange_8f.html>
//
// No Rust caller: `blas_dispatch::zlange` measured LAPACK's zlange as no faster
// than the native kernel on every norm and slower on the one-norm, so it stays
// native. Kept as the twin of `c_dlange`, which is routed, so a future call site
// finds the wrapper rather than writing it.
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

// Computes the Cholesky factorization of a complex Hermitian positive definite matrix A.
// <https://www.netlib.org/lapack/explore-html/d1/db9/zpotrf_8f.html>
void c_zpotrf(C_BOOL upper,
              const int32_t *n,
              COMPLEX64 *a,
              const int32_t *lda,
              int32_t *info) {
    const char *uplo = upper == C_TRUE ? "U" : "L";
    FN_ZPOTRF(uplo, n, a, lda, info);
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

// Computes the singular value decomposition (SVD)
// <https://www.netlib.org/lapack/explore-html/d6/d42/zgesvd_8f.html>
void c_zgesvd(int32_t jobu_code,
              int32_t jobvt_code,
              const int32_t *m,
              const int32_t *n,
              COMPLEX64 *a,
              const int32_t *lda,
              double *s,
              COMPLEX64 *u,
              const int32_t *ldu,
              COMPLEX64 *vh,
              const int32_t *ldvt,
              COMPLEX64 *work,
              const int32_t *lwork,
              double *rwork,
              int32_t *info) {
    const char *jobu = jobu_code == SVD_CODE_A   ? "A"
                       : jobu_code == SVD_CODE_S ? "S"
                       : jobu_code == SVD_CODE_O ? "O"
                                                 : "N";
    const char *jobvt = jobvt_code == SVD_CODE_A   ? "A"
                        : jobvt_code == SVD_CODE_S ? "S"
                        : jobvt_code == SVD_CODE_O ? "O"
                                                   : "N";
    FN_ZGESVD(jobu, jobvt, m, n, a, lda, s, u, ldu, vh, ldvt, work, lwork, rwork, info);
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
