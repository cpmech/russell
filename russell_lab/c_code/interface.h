#pragma once

// Prototypes for the C shim, `interface_blas.c`.
//
// The shim is the crate's only C layer: `build.rs` compiles this one source
// against OpenBLAS, Intel MKL or a locally built reference LAPACK, and the Rust
// side declares the `c_*` entry points it calls in
// `src/base/blas_dispatch.rs` and `src/base/lapack_dispatch.rs`.
//
// Two things are centralized here rather than repeated in the shim:
//
// 1. The include of the vendor headers, which is also what decides the
//    `COMPLEX64` type. MKL spells it `MKL_Complex16`; the reference LAPACK's
//    `lapack.h` spells it `lapack_complex_double`. Both are two `double`s in
//    sequence, which is what `Complex64` is on the Rust side.
// 2. The prototypes themselves, so a definition that drifts from its callers'
//    expectation is a compile error in this translation unit rather than a
//    stack corruption at run time.
//
// Note for the `dev_lapack` build: `build.rs` derives a copy of the shim with
// every `c_*` identifier renamed to `dev_lapack_c_*`. That copy includes this
// header too and therefore also sees the unprefixed declarations; they are
// simply never defined or called in that translation unit, which C allows. Do
// not make these `static inline` or otherwise give them a definition here, or
// the two copies would collide.

#include <inttypes.h>

#ifdef USE_INTEL_MKL
#include "mkl.h"
#define COMPLEX64 MKL_Complex16
#else
#include "cblas.h"
#include "lapack.h"
#define COMPLEX64 lapack_complex_double
#endif

#include "constants.h"

// ---------------------------------------------------------------------------------------------
// Library selection and threading
// ---------------------------------------------------------------------------------------------

int32_t c_using_intel_mkl(void);
void c_set_num_threads(int32_t n);
int32_t c_get_num_threads(void);

// ---------------------------------------------------------------------------------------------
// BLAS entry points routed by src/base/blas_dispatch.rs
// ---------------------------------------------------------------------------------------------

void c_dgemm(double alpha,
             const double *a,
             const double *b,
             double beta,
             double *c,
             const int32_t *m,
             const int32_t *n,
             const int32_t *k);

void c_dgemm_tn(double alpha,
                const double *a,
                const double *b,
                double beta,
                double *c,
                const int32_t *m,
                const int32_t *n,
                const int32_t *k);

void c_dgemm_nt(double alpha,
                const double *a,
                const double *b,
                double beta,
                double *c,
                const int32_t *m,
                const int32_t *n,
                const int32_t *k);

void c_dgemm_tt(double alpha,
                const double *a,
                const double *b,
                double beta,
                double *c,
                const int32_t *m,
                const int32_t *n,
                const int32_t *k);

void c_dgemv(double alpha, const double *a, const double *x, double beta, double *y, const int32_t *m, const int32_t *n);

void c_dgemv_trans(double alpha,
                   const double *a,
                   const double *x,
                   double beta,
                   double *y,
                   const int32_t *m,
                   const int32_t *n);

double c_ddot(const double *x, const double *y, const int32_t *n);
double c_dnrm2(const double *x, const int32_t *n);
double c_dasum(const double *x, const int32_t *n);
int32_t c_idamax(const double *x, const int32_t *n);
void c_daxpy(double alpha, const double *x, double *y, const int32_t *n);
void c_dger(double alpha, const double *x, const double *y, double *a, const int32_t *m, const int32_t *n);

void c_dsyrk(double alpha,
             const double *a,
             double beta,
             double *c,
             const int32_t *n,
             const int32_t *k,
             C_BOOL upper,
             C_BOOL transposed);

void c_zgemm(const COMPLEX64 *alpha,
             const COMPLEX64 *a,
             const COMPLEX64 *b,
             const COMPLEX64 *beta,
             COMPLEX64 *c,
             const int32_t *m,
             const int32_t *n,
             const int32_t *k);

void c_zgemm_tn(const COMPLEX64 *alpha,
                const COMPLEX64 *a,
                const COMPLEX64 *b,
                const COMPLEX64 *beta,
                COMPLEX64 *c,
                const int32_t *m,
                const int32_t *n,
                const int32_t *k);

void c_zgemm_nt(const COMPLEX64 *alpha,
                const COMPLEX64 *a,
                const COMPLEX64 *b,
                const COMPLEX64 *beta,
                COMPLEX64 *c,
                const int32_t *m,
                const int32_t *n,
                const int32_t *k);

void c_zgemm_tt(const COMPLEX64 *alpha,
                const COMPLEX64 *a,
                const COMPLEX64 *b,
                const COMPLEX64 *beta,
                COMPLEX64 *c,
                const int32_t *m,
                const int32_t *n,
                const int32_t *k);

void c_zgemv(const COMPLEX64 *alpha,
             const COMPLEX64 *a,
             const COMPLEX64 *x,
             const COMPLEX64 *beta,
             COMPLEX64 *y,
             const int32_t *m,
             const int32_t *n);

void c_zgemv_trans(const COMPLEX64 *alpha,
                   const COMPLEX64 *a,
                   const COMPLEX64 *x,
                   const COMPLEX64 *beta,
                   COMPLEX64 *y,
                   const int32_t *m,
                   const int32_t *n);

void c_zger(const COMPLEX64 *alpha,
            const COMPLEX64 *x,
            const COMPLEX64 *y,
            COMPLEX64 *a,
            const int32_t *m,
            const int32_t *n);

void c_zsyrk(const COMPLEX64 *alpha,
             const COMPLEX64 *a,
             const COMPLEX64 *beta,
             COMPLEX64 *c,
             const int32_t *n,
             const int32_t *k,
             C_BOOL upper,
             C_BOOL transposed);

void c_zherk(double alpha,
             const COMPLEX64 *a,
             double beta,
             COMPLEX64 *c,
             const int32_t *n,
             const int32_t *k,
             C_BOOL upper,
             C_BOOL conj_transposed);

void c_zaxpy(const COMPLEX64 *alpha, const COMPLEX64 *x, COMPLEX64 *y, const int32_t *n);
void c_zscal(const COMPLEX64 *alpha, COMPLEX64 *x, const int32_t *n);
double c_dznrm2(const COMPLEX64 *x, const int32_t *n);

double c_dlange(int32_t norm_code,
                const int32_t *m,
                const int32_t *n,
                const double *a,
                const int32_t *lda,
                double *work);

double c_zlange(int32_t norm_code,
                const int32_t *m,
                const int32_t *n,
                const COMPLEX64 *a,
                const int32_t *lda,
                double *work);

// ---------------------------------------------------------------------------------------------
// LAPACK entry points routed by src/base/lapack_dispatch.rs
// ---------------------------------------------------------------------------------------------
//
// These are the Fortran routines, not CBLAS, so their arguments are LAPACK's:
// a leading dimension per matrix, a caller-supplied workspace and `lwork`, and
// LAPACK's integer `info` out-parameter. The wrapper's job is to supply the
// character arguments and their hidden lengths, and nothing else -- the
// workspace sizing belongs to the caller, which is the only side that knows how
// much it is willing to allocate.
//
// `info` is LAPACK's convention, not the crate's: `info < 0` names an illegal
// argument (as `-info`), `info > 0` means the routine could not finish.
// Translating it is done in `lapack_dispatch.rs`, once, so that the accelerated
// path and the native path report the same error.

void c_dgesv(const int32_t *n,
             const int32_t *nrhs,
             double *a,
             const int32_t *lda,
             int32_t *ipiv,
             double *b,
             const int32_t *ldb,
             int32_t *info);

void c_zgesv(const int32_t *n,
             const int32_t *nrhs,
             COMPLEX64 *a,
             const int32_t *lda,
             int32_t *ipiv,
             COMPLEX64 *b,
             const int32_t *ldb,
             int32_t *info);

void c_dgetrf(const int32_t *m, const int32_t *n, double *a, const int32_t *lda, int32_t *ipiv, int32_t *info);

void c_dgetri(const int32_t *n,
              double *a,
              const int32_t *lda,
              const int32_t *ipiv,
              double *work,
              const int32_t *lwork,
              int32_t *info);

void c_zgetrf(const int32_t *m, const int32_t *n, COMPLEX64 *a, const int32_t *lda, int32_t *ipiv, int32_t *info);

void c_zgetri(const int32_t *n,
              COMPLEX64 *a,
              const int32_t *lda,
              const int32_t *ipiv,
              COMPLEX64 *work,
              const int32_t *lwork,
              int32_t *info);

void c_dpotrf(C_BOOL upper, const int32_t *n, double *a, const int32_t *lda, int32_t *info);

void c_zpotrf(C_BOOL upper, const int32_t *n, COMPLEX64 *a, const int32_t *lda, int32_t *info);

void c_dsyev(C_BOOL calc_v,
             C_BOOL upper,
             const int32_t *n,
             double *a,
             const int32_t *lda,
             double *w,
             double *work,
             const int32_t *lwork,
             int32_t *info);

void c_zheev(C_BOOL calc_v,
             C_BOOL upper,
             const int32_t *n,
             COMPLEX64 *a,
             const int32_t *lda,
             double *w,
             COMPLEX64 *work,
             const int32_t *lwork,
             double *rwork,
             int32_t *info);

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
             int32_t *info);

void c_zgeev(C_BOOL calc_vl,
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
             int32_t *info);

void c_dggev(C_BOOL calc_vl,
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
             int32_t *info);

void c_zggev(C_BOOL calc_vl,
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
             int32_t *info);

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
              int32_t *info);

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
              int32_t *info);
