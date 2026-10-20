//! BLAS entry points, dispatched to the native kernels or to a linked library.
//!
//! Every function here has the same signature as its counterpart in
//! [`crate::native::blas`], so a call site names the *operation* rather than the
//! backend: the default build forwards to the native Rust kernel, and a build
//! with an accelerator forwards to the C shim (`c_code/interface_blas.c`), which
//! calls CBLAS.
//!
//! # Precedence
//!
//! `intel_mkl` wins when both features are enabled. The crate's gates are of two
//! kinds: `any(feature = "open_blas", feature = "intel_mkl")` for "an accelerator
//! is in use", which is most of them and does not care which, and
//! `all(feature = "open_blas", not(feature = "intel_mkl"))` wherever OpenBLAS
//! specifically has to be chosen -- `build.rs`'s linking being the one that
//! matters, since only that decides which library defines the symbols. Enabling
//! both is therefore not an error: it is MKL.
//!
//! # What is here
//!
//! One entry per BLAS operation the crate uses, with the crate's own argument
//! shapes -- slices, `usize` lengths, `bool` flags -- rather than CBLAS's:
//!
//! * level 1: `ddot`, `daxpy`, `dscal`, `dcopy`, `dnrm2`, `dasum`, `idamax`,
//!   `dznrm2`, `zaxpy`, `zscal`, `zcopy`;
//! * level 2: `dgemv`, `dgemv_trans`, `dger`, `zgemv`, `zgemv_trans`, `zger`;
//! * level 3: `dgemm` and its three transposed combinations (`dgemm_tn`,
//!   `dgemm_nt`, `dgemm_tt`), `dsyrk`, `zgemm` and its three, `zsyrk`, `zherk`;
//! * `dlange` and `zlange`, which are LAPACK's rather than BLAS's but are norms
//!   over the same buffer and are needed by `mat_norm`/`complex_mat_norm`.
//!
//! This is now the only place in the crate that names a `cblas_*` symbol or
//! chooses between a native BLAS kernel and the shim: every BLAS call site goes
//! through here. (The `c_*` LAPACK wrappers in `src/matrix` and `src/matvec` are a
//! different layer and were not part of this.)
//!
//! Five entries have no call site -- `dgemm_nt`, `dgemm_tt`, `zgemm_nt`,
//! `zgemm_tt` and `zger`. They are here because this module is the surface a call
//! site has to use, and leaving one of a symmetric set out would make the next
//! call site reopen it; the tests are their callers, which is what the `allow`s
//! record.
//!
//! `zsyrk` is **not** `zherk`. Both are rank-k updates of an `n x n` matrix from
//! an `n x k` one, but `zsyrk` transposes plainly and takes complex `alpha`/`beta`
//! (the result is complex symmetric), while `zherk` conjugates and takes real
//! scalars (the result is Hermitian, with a real diagonal). They are separate CBLAS
//! entry points and separate native kernels for exactly that reason; the shim's
//! `c_zsyrk` sends `CblasTrans`, never `CblasConjTrans`.
//!
//! # Which entries reach the library
//!
//! Sending everything to the library is the uniform choice, and for level 1 it is
//! partly the wrong one: these routines are memory-bound and short, so the call
//! into C can cost more than the loop it replaces. Each was measured instead of
//! assumed, with `cargo bench --bench backends_benchmark` (level1 group) against
//! MKL 2026.0 pinned to one thread, at the three sizes the numbers below are
//! quoted for. `native -> CBLAS`, ns per call, mean of two runs:
//!
//! | routine | n = 128 | n = 1,024 | n = 8,192 | routed |
//! |---|---|---|---|---|
//! | `daxpy` | 12 -> 8.9 | 98 -> 46 | 1,011 -> 820 | yes |
//! | `dcopy` | 6.4 -> 8.9 | 36 -> 37 | 866 -> 913 | **no** |
//! | `dscal` | 7.7 -> 8.1 | 63 -> 28 | 682 -> 724 | **no** |
//! | `ddot` | 27 -> 9.2 | 337 -> 52 | 2,817 -> 738 | yes |
//! | `dnrm2` | 124 -> 31 | 1,073 -> 189 | 8,665 -> 1,456 | yes |
//! | `dasum` | 28 -> 7.2 | 339 -> 34 | 2,832 -> 315 | yes |
//! | `idamax` | 80 -> 59 | 699 -> 95 | 5,860 -> 413 | yes |
//! | `zaxpy` | 68 -> 18 | 533 -> 119 | 4,376 -> 1,655 | yes |
//! | `zcopy` | 8.7 -> 13 | 69 -> 100 | 1,737 -> 2,018 | **no** |
//! | `zscal` | 45 -> 14 | 356 -> 93 | 2,868 -> 1,422 | yes |
//! | `dznrm2` | 587 -> 54 | 4,718 -> 369 | 37,774 -> 2,887 | yes |
//!
//! Three conclusions, and the last two are the awkward ones:
//!
//! * The reductions and the complex scalings are 3-14x faster through the library
//!   at every size, which is why they were worth routing.
//! * The two copies are not. `copy_from_slice` compiles to a single `memcpy`,
//!   which the library cannot beat and a call only makes slower, so `dcopy` and
//!   `zcopy` stay native. `daxpy` is the same shape plus an add, and its win is
//!   the thinnest of the routed entries for that reason.
//! * `dscal` is the one judgment call: 2.2x faster through the library at
//!   n = 1,024 and 6% slower at n = 8,192, where both sides are limited by memory
//!   bandwidth. Routing it would trade one size for the other rather than win, so
//!   it is left native -- the same `if size > threshold` problem `dsyev` has, and
//!   no size test can express it because the loss and the win are at opposite ends.
//!
//! Read the n = 128 column with care: those calls are 5-12 ns, and the native
//! column moved by up to 40% between the two runs there, so a difference under
//! about 20% at that size is not a result. The n = 1,024 and n = 8,192 columns
//! reproduced within about 3% (`zcopy` by 7%), and the decisions rest on those.
//!
//! The level-2 and level-3 entries are O(n^2) or O(n^3), so the call overhead
//! does not matter and the only question is whether the library's kernel is
//! better. Measured through the public entry point -- plain vs `--all-features`,
//! same binary, same data -- it is:
//!
//! | call site | n = 128 | n = 512 | n = 1,024 | routed |
//! |---|---|---|---|---|
//! | `complex_mat_vec_mul` (zgemv) | 9.0 -> 1.7 us | 136 -> 40 us | 563 -> 171 us | yes |
//! | `complex_mat_sym_rank_op` (zsyrk) | 777 -> 105 us | 143 -> 6.2 ms | 1.37 s -> 50 ms | yes |
//! | `mat_norm`, Frobenius (dlange) | 17.3 -> 4.1 us | 276 -> 51 us | 1,106 -> 194 us | yes |
//! | `complex_mat_norm`, one-norm (zlange) | 39.7 -> 50.2 us | 647 -> 752 us | 2,682 -> 3,055 us | **no** |
//!
//! `zsyrk` is where this dispatch was worth the most outside the level-3 products:
//! the native complex symmetric rank-k update is a scalar triple loop, and the
//! library is 7-27x faster depending on size. `zlange` is the one library kernel
//! that loses on every norm -- see its own comment.
//!
//! # Why the shim, and not CBLAS directly
//!
//! The shim is the crate's only C layer, and `build.rs` selects which library it
//! compiles against. Calling CBLAS from here would mean a second set of
//! declarations per call site and a second library assumption in the Rust code,
//! which is exactly what `interface_blas.c` exists to centralize.

use crate::Complex64;

#[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
use crate::internal::{C_FALSE, C_TRUE, NORM_EUC, NORM_INF, NORM_MAX, NORM_ONE};
#[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
use crate::to_i32;

#[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
unsafe extern "C" {
    // All of these are the shim's own entry points; each is a two-line wrapper
    // around one CBLAS call (see `c_code/interface_blas.c`).
    fn c_dgemm(
        alpha: f64,
        a: *const f64,
        b: *const f64,
        beta: f64,
        c: *mut f64,
        m: *const i32,
        n: *const i32,
        k: *const i32,
    );
    fn c_dgemv(alpha: f64, a: *const f64, x: *const f64, beta: f64, y: *mut f64, m: *const i32, n: *const i32);
    fn c_dgemv_trans(alpha: f64, a: *const f64, x: *const f64, beta: f64, y: *mut f64, m: *const i32, n: *const i32);
    fn c_ddot(x: *const f64, y: *const f64, n: *const i32) -> f64;
    fn c_dnrm2(x: *const f64, n: *const i32) -> f64;
    fn c_dasum(x: *const f64, n: *const i32) -> f64;
    fn c_idamax(x: *const f64, n: *const i32) -> i32;
    fn c_daxpy(alpha: f64, x: *const f64, y: *mut f64, n: *const i32);
    fn c_dlange(norm_code: i32, m: *const i32, n: *const i32, a: *const f64, lda: *const i32, work: *mut f64) -> f64;
    fn c_zgemm(
        alpha: *const Complex64,
        a: *const Complex64,
        b: *const Complex64,
        beta: *const Complex64,
        c: *mut Complex64,
        m: *const i32,
        n: *const i32,
        k: *const i32,
    );
    fn c_dgemm_tn(
        alpha: f64,
        a: *const f64,
        b: *const f64,
        beta: f64,
        c: *mut f64,
        m: *const i32,
        n: *const i32,
        k: *const i32,
    );
    fn c_dgemm_nt(
        alpha: f64,
        a: *const f64,
        b: *const f64,
        beta: f64,
        c: *mut f64,
        m: *const i32,
        n: *const i32,
        k: *const i32,
    );
    fn c_dgemm_tt(
        alpha: f64,
        a: *const f64,
        b: *const f64,
        beta: f64,
        c: *mut f64,
        m: *const i32,
        n: *const i32,
        k: *const i32,
    );
    fn c_zgemm_tn(
        alpha: *const Complex64,
        a: *const Complex64,
        b: *const Complex64,
        beta: *const Complex64,
        c: *mut Complex64,
        m: *const i32,
        n: *const i32,
        k: *const i32,
    );
    fn c_zgemm_nt(
        alpha: *const Complex64,
        a: *const Complex64,
        b: *const Complex64,
        beta: *const Complex64,
        c: *mut Complex64,
        m: *const i32,
        n: *const i32,
        k: *const i32,
    );
    fn c_zgemm_tt(
        alpha: *const Complex64,
        a: *const Complex64,
        b: *const Complex64,
        beta: *const Complex64,
        c: *mut Complex64,
        m: *const i32,
        n: *const i32,
        k: *const i32,
    );
    fn c_dger(alpha: f64, x: *const f64, y: *const f64, a: *mut f64, m: *const i32, n: *const i32);
    fn c_zger(
        alpha: *const Complex64,
        x: *const Complex64,
        y: *const Complex64,
        a: *mut Complex64,
        m: *const i32,
        n: *const i32,
    );
    fn c_dsyrk(
        alpha: f64,
        a: *const f64,
        beta: f64,
        c: *mut f64,
        n: *const i32,
        k: *const i32,
        upper: i32,
        transposed: i32,
    );
    fn c_zherk(
        alpha: f64,
        a: *const Complex64,
        beta: f64,
        c: *mut Complex64,
        n: *const i32,
        k: *const i32,
        upper: i32,
        conj_transposed: i32,
    );
    fn c_zaxpy(alpha: *const Complex64, x: *const Complex64, y: *mut Complex64, n: *const i32);
    fn c_zscal(alpha: *const Complex64, x: *mut Complex64, n: *const i32);
    fn c_dznrm2(x: *const Complex64, n: *const i32) -> f64;
    fn c_zgemv(
        alpha: *const Complex64,
        a: *const Complex64,
        x: *const Complex64,
        beta: *const Complex64,
        y: *mut Complex64,
        m: *const i32,
        n: *const i32,
    );
    fn c_zgemv_trans(
        alpha: *const Complex64,
        a: *const Complex64,
        x: *const Complex64,
        beta: *const Complex64,
        y: *mut Complex64,
        m: *const i32,
        n: *const i32,
    );
    fn c_zsyrk(
        alpha: *const Complex64,
        a: *const Complex64,
        beta: *const Complex64,
        c: *mut Complex64,
        n: *const i32,
        k: *const i32,
        upper: i32,
        transposed: i32,
    );
}

/// `C := alpha * A * B + beta * C` (dgemm, both trans = 'N').
///
/// # Panics
///
/// Panics if the shapes are inconsistent.
pub fn dgemm(alpha: f64, a: &[f64], b: &[f64], beta: f64, c: &mut [f64], m: usize, n: usize, k: usize) {
    assert_eq!(a.len(), m * k, "dgemm: a must be m*k");
    assert_eq!(b.len(), k * n, "dgemm: b must be k*n");
    assert_eq!(c.len(), m * n, "dgemm: c must be m*n");
    if m == 0 || n == 0 {
        return;
    }
    #[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
    {
        let m_i32 = to_i32(m);
        let n_i32 = to_i32(n);
        let k_i32 = to_i32(k);
        // Safety: every slice was checked against `m`, `n` and `k` above, and
        // the shim is linked in this configuration.
        unsafe {
            c_dgemm(
                alpha,
                a.as_ptr(),
                b.as_ptr(),
                beta,
                c.as_mut_ptr(),
                &m_i32,
                &n_i32,
                &k_i32,
            );
        }
    }
    #[cfg(not(any(feature = "open_blas", feature = "intel_mkl")))]
    crate::native::blas::dgemm(alpha, a, b, beta, c, m, n, k);
}

/// `C := alpha * A^T * B + beta * C` (dgemm, trans = 'T','N').
///
/// `a` is `k x m`, `b` is `k x n`, `c` is `m x n`, all column-major.
///
/// # Panics
///
/// Panics if the shapes are inconsistent.
pub fn dgemm_tn(alpha: f64, a: &[f64], b: &[f64], beta: f64, c: &mut [f64], m: usize, n: usize, k: usize) {
    assert_eq!(a.len(), k * m, "dgemm_tn: a must be k*m");
    assert_eq!(b.len(), k * n, "dgemm_tn: b must be k*n");
    assert_eq!(c.len(), m * n, "dgemm_tn: c must be m*n");
    if m == 0 || n == 0 {
        return;
    }
    #[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
    {
        let m_i32 = to_i32(m);
        let n_i32 = to_i32(n);
        let k_i32 = to_i32(k);
        // Safety: every slice was checked against `m`, `n` and `k` above, and
        // the shim is linked in this configuration.
        unsafe {
            c_dgemm_tn(
                alpha,
                a.as_ptr(),
                b.as_ptr(),
                beta,
                c.as_mut_ptr(),
                &m_i32,
                &n_i32,
                &k_i32,
            )
        };
    }
    #[cfg(not(any(feature = "open_blas", feature = "intel_mkl")))]
    crate::native::blas::dgemm_tn(alpha, a, b, beta, c, m, n, k);
}

/// `C := alpha * A * B^T + beta * C` (dgemm, trans = 'N','T').
///
/// `a` is `m x k`, `b` is `n x k`, `c` is `m x n`, all column-major.
///
/// No call site yet: the crate's own products are `A B` and `A^T B`, so this is
/// exercised by the tests only. It is here because the dispatcher is the surface a
/// call site has to use, and leaving one of the three transposed combinations out
/// would make the next `A B^T` call site reopen this file.
///
/// # Panics
///
/// Panics if the shapes are inconsistent.
#[allow(dead_code)]
pub fn dgemm_nt(alpha: f64, a: &[f64], b: &[f64], beta: f64, c: &mut [f64], m: usize, n: usize, k: usize) {
    assert_eq!(a.len(), m * k, "dgemm_nt: a must be m*k");
    assert_eq!(b.len(), n * k, "dgemm_nt: b must be n*k");
    assert_eq!(c.len(), m * n, "dgemm_nt: c must be m*n");
    if m == 0 || n == 0 {
        return;
    }
    #[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
    {
        let m_i32 = to_i32(m);
        let n_i32 = to_i32(n);
        let k_i32 = to_i32(k);
        // Safety: every slice was checked against `m`, `n` and `k` above, and
        // the shim is linked in this configuration.
        unsafe {
            c_dgemm_nt(
                alpha,
                a.as_ptr(),
                b.as_ptr(),
                beta,
                c.as_mut_ptr(),
                &m_i32,
                &n_i32,
                &k_i32,
            )
        };
    }
    #[cfg(not(any(feature = "open_blas", feature = "intel_mkl")))]
    crate::native::blas::dgemm_nt(alpha, a, b, beta, c, m, n, k);
}

/// `C := alpha * A^T * B^T + beta * C` (dgemm, trans = 'T','T').
///
/// `a` is `k x m`, `b` is `n x k`, `c` is `m x n`, all column-major.
///
/// No call site yet: the crate's own products are `A B` and `A^T B`, so this is
/// exercised by the tests only. It is here because the dispatcher is the surface a
/// call site has to use, and leaving one of the three transposed combinations out
/// would make the next `A B^T` call site reopen this file.
///
/// # Panics
///
/// Panics if the shapes are inconsistent.
#[allow(dead_code)]
pub fn dgemm_tt(alpha: f64, a: &[f64], b: &[f64], beta: f64, c: &mut [f64], m: usize, n: usize, k: usize) {
    assert_eq!(a.len(), k * m, "dgemm_tt: a must be k*m");
    assert_eq!(b.len(), n * k, "dgemm_tt: b must be n*k");
    assert_eq!(c.len(), m * n, "dgemm_tt: c must be m*n");
    if m == 0 || n == 0 {
        return;
    }
    #[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
    {
        let m_i32 = to_i32(m);
        let n_i32 = to_i32(n);
        let k_i32 = to_i32(k);
        // Safety: every slice was checked against `m`, `n` and `k` above, and
        // the shim is linked in this configuration.
        unsafe {
            c_dgemm_tt(
                alpha,
                a.as_ptr(),
                b.as_ptr(),
                beta,
                c.as_mut_ptr(),
                &m_i32,
                &n_i32,
                &k_i32,
            )
        };
    }
    #[cfg(not(any(feature = "open_blas", feature = "intel_mkl")))]
    crate::native::blas::dgemm_tt(alpha, a, b, beta, c, m, n, k);
}

/// `y := alpha * A * x + beta * y` (dgemv, trans = 'N').
///
/// # Panics
///
/// Panics if the shapes are inconsistent.
pub fn dgemv(alpha: f64, a: &[f64], x: &[f64], beta: f64, y: &mut [f64], m: usize, n: usize) {
    assert_eq!(a.len(), m * n, "dgemv: a must be m*n");
    assert_eq!(x.len(), n, "dgemv: x must have n entries");
    assert_eq!(y.len(), m, "dgemv: y must have m entries");
    #[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
    {
        let m_i32 = to_i32(m);
        let n_i32 = to_i32(n);
        // Safety: every slice was checked against `m` and `n` above, and the
        // shim is linked in this configuration.
        unsafe {
            c_dgemv(alpha, a.as_ptr(), x.as_ptr(), beta, y.as_mut_ptr(), &m_i32, &n_i32);
        }
    }
    #[cfg(not(any(feature = "open_blas", feature = "intel_mkl")))]
    crate::native::blas::dgemv(alpha, a, x, beta, y, m, n);
}

/// `y := alpha * A^T * x + beta * y` (dgemv, trans = 'T').
///
/// `a` is `m x n` column-major, so `x` has `m` entries and `y` has `n`.
///
/// # Panics
///
/// Panics if the shapes are inconsistent.
pub fn dgemv_trans(alpha: f64, a: &[f64], x: &[f64], beta: f64, y: &mut [f64], m: usize, n: usize) {
    assert_eq!(a.len(), m * n, "dgemv_trans: a must be m*n");
    assert_eq!(x.len(), m, "dgemv_trans: x must have m entries");
    assert_eq!(y.len(), n, "dgemv_trans: y must have n entries");
    #[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
    {
        let m_i32 = to_i32(m);
        let n_i32 = to_i32(n);
        // Safety: every slice was checked against `m` and `n` above, and the
        // shim is linked in this configuration.
        unsafe {
            c_dgemv_trans(alpha, a.as_ptr(), x.as_ptr(), beta, y.as_mut_ptr(), &m_i32, &n_i32);
        }
    }
    #[cfg(not(any(feature = "open_blas", feature = "intel_mkl")))]
    crate::native::blas::dgemv_trans(alpha, a, x, beta, y, m, n);
}

/// `A := alpha * x * y^T + A` (dger), with `A` `m x n` column-major.
///
/// # Panics
///
/// Panics if the shapes are inconsistent.
pub fn dger(alpha: f64, x: &[f64], y: &[f64], a: &mut [f64], m: usize, n: usize) {
    assert_eq!(a.len(), m * n, "dger: a must be m*n");
    assert_eq!(x.len(), m, "dger: x must have m entries");
    assert_eq!(y.len(), n, "dger: y must have n entries");
    if m == 0 || n == 0 {
        return;
    }
    #[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
    {
        let m_i32 = to_i32(m);
        let n_i32 = to_i32(n);
        // Safety: every slice was checked against `m` and `n` above, and the
        // shim is linked in this configuration.
        unsafe { c_dger(alpha, x.as_ptr(), y.as_ptr(), a.as_mut_ptr(), &m_i32, &n_i32) };
    }
    #[cfg(not(any(feature = "open_blas", feature = "intel_mkl")))]
    crate::native::blas::dger(alpha, x, y, a, m, n);
}

/// `x . y` (ddot).
///
/// Mismatched lengths read the shorter vector, as BLAS does.
pub fn ddot(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n == 0 {
        return 0.0;
    }
    #[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
    {
        let n_i32 = to_i32(n);
        // Safety: both slices have at least `n` entries, and the shim is linked
        // in this configuration.
        unsafe { c_ddot(x.as_ptr(), y.as_ptr(), &n_i32) }
    }
    #[cfg(not(any(feature = "open_blas", feature = "intel_mkl")))]
    crate::native::blas::ddot(&x[..n], &y[..n])
}

/// `y := alpha * x + y` (daxpy), where `y` and `x` are the same length.
///
/// # Panics
///
/// Panics if the slices differ in length.
pub fn daxpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    assert_eq!(x.len(), y.len(), "daxpy: vectors must have the same length");
    if y.is_empty() {
        return;
    }
    #[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
    {
        let n_i32 = to_i32(y.len());
        // Safety: `x` and `y` have the same length, checked above, and the shim
        // is linked in this configuration.
        unsafe { c_daxpy(alpha, x.as_ptr(), y.as_mut_ptr(), &n_i32) };
    }
    #[cfg(not(any(feature = "open_blas", feature = "intel_mkl")))]
    crate::native::blas::daxpy(alpha, x, y);
}

/// `y := x` (dcopy).
///
/// **Not routed.** The native kernel is a `copy_from_slice`, i.e. one `memcpy`,
/// and the library call adds a call it cannot win back; see the level-1
/// measurements on this module.
///
/// # Panics
///
/// Panics if the slices differ in length.
pub fn dcopy(x: &[f64], y: &mut [f64]) {
    crate::native::blas::dcopy(x, y);
}

/// `x := alpha * x` (dscal).
///
/// **Not routed.** Measured against MKL, this is the one level-1 routine that
/// wins through the library at one size and loses at another: 2.2x faster at
/// n = 1024, 6% slower at n = 8192, and within noise at n = 128 (see the
/// measurements on this module). The large-size case is memory-bandwidth-bound on
/// both sides, so routing would trade one size for the other rather than win, and
/// the native loop is kept.
pub fn dscal(alpha: f64, x: &mut [f64]) {
    crate::native::blas::dscal(alpha, x);
}

/// Euclidean norm `sqrt(x . x)` (dnrm2).
pub fn dnrm2(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    #[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
    {
        let n_i32 = to_i32(x.len());
        // Safety: `x` has `n` entries, and the shim is linked in this
        // configuration.
        unsafe { c_dnrm2(x.as_ptr(), &n_i32) }
    }
    #[cfg(not(any(feature = "open_blas", feature = "intel_mkl")))]
    crate::native::blas::dnrm2(x)
}

/// Sum of absolute values `|x|_1` (dasum).
pub fn dasum(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    #[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
    {
        let n_i32 = to_i32(x.len());
        // Safety: `x` has `n` entries, and the shim is linked in this
        // configuration.
        unsafe { c_dasum(x.as_ptr(), &n_i32) }
    }
    #[cfg(not(any(feature = "open_blas", feature = "intel_mkl")))]
    crate::native::blas::dasum(x)
}

/// Index of the first entry with the largest absolute value (idamax).
///
/// 0-based, unlike BLAS's own `idamax`. An empty slice yields 0.
pub fn idamax(x: &[f64]) -> usize {
    if x.is_empty() {
        return 0;
    }
    #[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
    {
        let n_i32 = to_i32(x.len());
        // Safety: `x` has `n` entries, and the shim is linked in this
        // configuration. CBLAS's index is 0-based; a negative one would mean the
        // library disagreed with the guard above, so it is not cast blindly.
        let i = unsafe { c_idamax(x.as_ptr(), &n_i32) };
        assert!(i >= 0, "idamax: the library returned a negative index");
        i as usize
    }
    #[cfg(not(any(feature = "open_blas", feature = "intel_mkl")))]
    crate::native::blas::idamax(x)
}

/// `nrm := |A|` for the one-, infinity- or Frobenius-norm (dlange).
///
/// `norm` is the BLAS character: `'1'`/`'O'` for the one-norm, `'I'` for
/// infinity, `'F'`/`'E'` for Frobenius and `'M'` for the maximum absolute entry.
///
/// # Panics
///
/// Panics on an unrecognized norm character.
pub fn dlange(norm: char, a: &[f64], m: usize, n: usize) -> f64 {
    assert_eq!(a.len(), m * n, "dlange: a must be m*n");
    #[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
    {
        // Resolved here so an unrecognized character is rejected in every
        // configuration, with the same message the native routine uses.
        let norm_code = norm_discriminant(norm);
        let m_i32 = to_i32(m);
        let n_i32 = to_i32(n);
        let lda = m_i32;
        // The infinity norm is the only one LAPACK's dlange needs workspace
        // for; `m` entries hold the running row sums.
        let mut work = if norm == 'I' || norm == 'i' {
            vec![0.0; m]
        } else {
            Vec::new()
        };
        // Safety: `a` was checked against `m` and `n` above, `work` has `m`
        // entries where LAPACK needs them, and the shim is linked in this
        // configuration.
        unsafe { c_dlange(norm_code, &m_i32, &n_i32, a.as_ptr(), &lda, work.as_mut_ptr()) }
    }
    #[cfg(not(any(feature = "open_blas", feature = "intel_mkl")))]
    crate::native::blas::dlange(norm, a, m, n)
}

/// The `Norm` discriminant the shim expects for a BLAS norm character.
///
/// `c_code/interface_blas.c` is the source of truth for the mapping: it sends
/// `0|1` to `"F"`, `2` to `"I"`, `3` to `"M"` and anything else to `"O"`. So
/// `'O'` must become `NORM_ONE` here, not a one-norm character in its own right;
/// mapping by position instead of by constant is the mistake recorded in
/// `AGENTS.md`.
#[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
fn norm_discriminant(norm: char) -> i32 {
    match norm {
        'M' | 'm' => NORM_MAX as i32,
        '1' | 'O' | 'o' => NORM_ONE as i32,
        'I' | 'i' => NORM_INF as i32,
        'F' | 'f' | 'E' | 'e' => NORM_EUC as i32,
        other => panic!("dlange: unrecognized norm '{other}'; expected one of 1, O, I, F, E, M"),
    }
}

/// `nrm := |A|` for a complex matrix (zlange), as a real number.
///
/// `norm` is the BLAS character, as in [`dlange`]. The one- and infinity-norms
/// use the modulus of each entry, as LAPACK specifies.
///
/// **Not routed, unlike [`dlange`].** Measured through `complex_mat_norm` against
/// MKL, LAPACK's `zlange` is 23% slower at n = 128 and 14% slower at n = 1024 for
/// the one-norm, and no better than the Rust kernel for the Frobenius and max
/// norms (1.00-1.03). The real `dlange` goes the other way -- 4-6x faster through
/// LAPACK -- so this is a property of the complex kernel, not of the norm code:
/// LAPACK's `zlange` walks the matrix twice to build `dlassq`'s scaled sums, and
/// the native twin needs neither a scaling pass nor a workspace. The shim's
/// `c_zlange` is kept for the pairing with `c_dlange` and has no Rust caller.
///
/// # Panics
///
/// Panics on an unrecognized norm character.
pub fn zlange(norm: char, a: &[Complex64], m: usize, n: usize) -> f64 {
    crate::native::blas::zlange(norm, a, m, n)
}

/// `C := alpha * A * A^T + beta * C`, or `C := alpha * A^T * A + beta * C` when
/// `transposed` (dsyrk).
///
/// `a` is `n x k` column-major without the transpose and `k x n` with it; its
/// length is `n*k` either way, which is exactly why `transposed` cannot be
/// inferred from the shapes and the caller has to state it. `c` is `n x n` and
/// only the `upper` (or lower) triangle of it is written -- the other triangle
/// keeps what it held, as BLAS specifies.
///
/// The native counterpart is [`crate::native::blas::dsyrk_rank`], whose extra
/// `transposed` argument this entry already had to carry.
///
/// # Panics
///
/// Panics if the shapes are inconsistent.
pub fn dsyrk(alpha: f64, a: &[f64], beta: f64, c: &mut [f64], n: usize, k: usize, upper: bool, transposed: bool) {
    assert_eq!(a.len(), n * k, "dsyrk: a must be n*k");
    assert_eq!(c.len(), n * n, "dsyrk: c must be n*n");
    if n == 0 {
        return;
    }
    #[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
    {
        let n_i32 = to_i32(n);
        let k_i32 = to_i32(k);
        // Safety: `a` and `c` were checked against `n` and `k` above, and the
        // shim is linked in this configuration.
        unsafe {
            c_dsyrk(
                alpha,
                a.as_ptr(),
                beta,
                c.as_mut_ptr(),
                &n_i32,
                &k_i32,
                if upper { C_TRUE } else { C_FALSE },
                if transposed { C_TRUE } else { C_FALSE },
            );
        }
    }
    #[cfg(not(any(feature = "open_blas", feature = "intel_mkl")))]
    crate::native::blas::dsyrk_rank(alpha, a, beta, c, n, k, upper, transposed);
}

/// `y := alpha * x + y` for complex data (zaxpy).
///
/// # Panics
///
/// Panics if the slices differ in length.
pub fn zaxpy(alpha: Complex64, x: &[Complex64], y: &mut [Complex64]) {
    assert_eq!(x.len(), y.len(), "zaxpy: vectors must have the same length");
    if y.is_empty() {
        return;
    }
    #[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
    {
        let n_i32 = to_i32(y.len());
        // Safety: `x` and `y` have the same length, checked above, and the shim
        // is linked in this configuration.
        unsafe { c_zaxpy(&alpha, x.as_ptr(), y.as_mut_ptr(), &n_i32) };
    }
    #[cfg(not(any(feature = "open_blas", feature = "intel_mkl")))]
    crate::native::blas::zaxpy(alpha, x, y);
}

/// `y := x` for complex data (zcopy).
///
/// **Not routed**, for the same reason as [`dcopy`]: the native kernel is one
/// `memcpy` and the library call cannot beat it.
///
/// # Panics
///
/// Panics if the slices differ in length.
pub fn zcopy(x: &[Complex64], y: &mut [Complex64]) {
    crate::native::blas::zcopy(x, y);
}

/// `x := alpha * x` for complex data (zscal).
pub fn zscal(alpha: Complex64, x: &mut [Complex64]) {
    if x.is_empty() {
        return;
    }
    #[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
    {
        let n_i32 = to_i32(x.len());
        // Safety: `x` has `n` entries by construction, and the shim is linked in
        // this configuration.
        unsafe { c_zscal(&alpha, x.as_mut_ptr(), &n_i32) };
    }
    #[cfg(not(any(feature = "open_blas", feature = "intel_mkl")))]
    crate::native::blas::zscal(alpha, x);
}

/// Euclidean norm of a complex vector (dznrm2), as a real number.
///
/// LAPACK names this `dznrm2` because its result is real even though the input is
/// complex; the native twin is [`crate::native::blas::dznrm2`].
pub fn dznrm2(x: &[Complex64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    #[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
    {
        let n_i32 = to_i32(x.len());
        // Safety: `x` has `n` entries, and the shim is linked in this
        // configuration.
        unsafe { c_dznrm2(x.as_ptr(), &n_i32) }
    }
    #[cfg(not(any(feature = "open_blas", feature = "intel_mkl")))]
    crate::native::blas::dznrm2(x)
}

/// `C := alpha * A * B + beta * C` for complex data (zgemm, both 'N').
///
/// # Panics
///
/// Panics if the shapes are inconsistent.
pub fn zgemm(
    alpha: Complex64,
    a: &[Complex64],
    b: &[Complex64],
    beta: Complex64,
    c: &mut [Complex64],
    m: usize,
    n: usize,
    k: usize,
) {
    assert_eq!(a.len(), m * k, "zgemm: a must be m*k");
    assert_eq!(b.len(), k * n, "zgemm: b must be k*n");
    assert_eq!(c.len(), m * n, "zgemm: c must be m*n");
    if m == 0 || n == 0 {
        return;
    }
    #[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
    {
        let m_i32 = to_i32(m);
        let n_i32 = to_i32(n);
        let k_i32 = to_i32(k);
        // `Complex64` is `#[repr(C)]` `{ re, im }`, which is what both CBLAS's
        // `void *` complex convention and the shim's `COMPLEX64` expect, so the
        // scalars go by pointer exactly as `cblas_zgemm` takes them.
        // Safety: every slice was checked against `m`, `n` and `k` above, and
        // the shim is linked in this configuration.
        unsafe {
            c_zgemm(
                &alpha,
                a.as_ptr(),
                b.as_ptr(),
                &beta,
                c.as_mut_ptr(),
                &m_i32,
                &n_i32,
                &k_i32,
            );
        }
    }
    #[cfg(not(any(feature = "open_blas", feature = "intel_mkl")))]
    crate::native::blas::zgemm(alpha, a, b, beta, c, m, n, k);
}

/// `C := alpha * A^T * B + beta * C` for complex data (zgemm, 'T','N'). Not
/// conjugated.
///
/// # Panics
///
/// Panics if the shapes are inconsistent.
pub fn zgemm_tn(
    alpha: Complex64,
    a: &[Complex64],
    b: &[Complex64],
    beta: Complex64,
    c: &mut [Complex64],
    m: usize,
    n: usize,
    k: usize,
) {
    assert_eq!(a.len(), k * m, "zgemm_tn: a must be k*m");
    assert_eq!(b.len(), k * n, "zgemm_tn: b must be k*n");
    assert_eq!(c.len(), m * n, "zgemm_tn: c must be m*n");
    if m == 0 || n == 0 {
        return;
    }
    #[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
    {
        let m_i32 = to_i32(m);
        let n_i32 = to_i32(n);
        let k_i32 = to_i32(k);
        // Safety: every slice was checked against `m`, `n` and `k` above, and
        // the shim is linked in this configuration.
        unsafe {
            c_zgemm_tn(
                &alpha,
                a.as_ptr(),
                b.as_ptr(),
                &beta,
                c.as_mut_ptr(),
                &m_i32,
                &n_i32,
                &k_i32,
            )
        };
    }
    #[cfg(not(any(feature = "open_blas", feature = "intel_mkl")))]
    crate::native::blas::zgemm_tn(alpha, a, b, beta, c, m, n, k);
}

/// `C := alpha * A * B^T + beta * C` for complex data (zgemm, 'N','T'). Not
/// conjugated.
///
/// No call site yet, for the same reason as [`dgemm_nt`]: `A B^T` and `A^T B^T`
/// have no product in this crate, so the tests are the only callers.
///
/// # Panics
///
/// Panics if the shapes are inconsistent.
#[allow(dead_code)]
pub fn zgemm_nt(
    alpha: Complex64,
    a: &[Complex64],
    b: &[Complex64],
    beta: Complex64,
    c: &mut [Complex64],
    m: usize,
    n: usize,
    k: usize,
) {
    assert_eq!(a.len(), m * k, "zgemm_nt: a must be m*k");
    assert_eq!(b.len(), n * k, "zgemm_nt: b must be n*k");
    assert_eq!(c.len(), m * n, "zgemm_nt: c must be m*n");
    if m == 0 || n == 0 {
        return;
    }
    #[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
    {
        let m_i32 = to_i32(m);
        let n_i32 = to_i32(n);
        let k_i32 = to_i32(k);
        // Safety: every slice was checked against `m`, `n` and `k` above, and
        // the shim is linked in this configuration.
        unsafe {
            c_zgemm_nt(
                &alpha,
                a.as_ptr(),
                b.as_ptr(),
                &beta,
                c.as_mut_ptr(),
                &m_i32,
                &n_i32,
                &k_i32,
            )
        };
    }
    #[cfg(not(any(feature = "open_blas", feature = "intel_mkl")))]
    crate::native::blas::zgemm_nt(alpha, a, b, beta, c, m, n, k);
}

/// `C := alpha * A^T * B^T + beta * C` for complex data (zgemm, 'T','T'). Not
/// conjugated.
///
/// No call site yet, for the same reason as [`dgemm_nt`]: `A B^T` and `A^T B^T`
/// have no product in this crate, so the tests are the only callers.
///
/// # Panics
///
/// Panics if the shapes are inconsistent.
#[allow(dead_code)]
pub fn zgemm_tt(
    alpha: Complex64,
    a: &[Complex64],
    b: &[Complex64],
    beta: Complex64,
    c: &mut [Complex64],
    m: usize,
    n: usize,
    k: usize,
) {
    assert_eq!(a.len(), k * m, "zgemm_tt: a must be k*m");
    assert_eq!(b.len(), n * k, "zgemm_tt: b must be n*k");
    assert_eq!(c.len(), m * n, "zgemm_tt: c must be m*n");
    if m == 0 || n == 0 {
        return;
    }
    #[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
    {
        let m_i32 = to_i32(m);
        let n_i32 = to_i32(n);
        let k_i32 = to_i32(k);
        // Safety: every slice was checked against `m`, `n` and `k` above, and
        // the shim is linked in this configuration.
        unsafe {
            c_zgemm_tt(
                &alpha,
                a.as_ptr(),
                b.as_ptr(),
                &beta,
                c.as_mut_ptr(),
                &m_i32,
                &n_i32,
                &k_i32,
            )
        };
    }
    #[cfg(not(any(feature = "open_blas", feature = "intel_mkl")))]
    crate::native::blas::zgemm_tt(alpha, a, b, beta, c, m, n, k);
}

/// `A := alpha * x * y^T + A` for complex data (zger). Not conjugated: BLAS's
/// `zger` is the unconjugated rank-1 update, which the shim sends to
/// `cblas_zgeru`.
///
/// No call site: the crate has no complex outer-product function, so this exists
/// for symmetry with [`dger`] and is exercised by the tests only.
///
/// # Panics
///
/// Panics if the shapes are inconsistent.
#[allow(dead_code)]
pub fn zger(alpha: Complex64, x: &[Complex64], y: &[Complex64], a: &mut [Complex64], m: usize, n: usize) {
    assert_eq!(a.len(), m * n, "zger: a must be m*n");
    assert_eq!(x.len(), m, "zger: x must have m entries");
    assert_eq!(y.len(), n, "zger: y must have n entries");
    if m == 0 || n == 0 {
        return;
    }
    #[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
    {
        let m_i32 = to_i32(m);
        let n_i32 = to_i32(n);
        // Safety: every slice was checked against `m` and `n` above, and the
        // shim is linked in this configuration.
        unsafe { c_zger(&alpha, x.as_ptr(), y.as_ptr(), a.as_mut_ptr(), &m_i32, &n_i32) };
    }
    #[cfg(not(any(feature = "open_blas", feature = "intel_mkl")))]
    crate::native::blas::zger(alpha, x, y, a, m, n);
}

/// `y := alpha * A * x + beta * y` for complex data (zgemv, trans = 'N').
///
/// # Panics
///
/// Panics if the shapes are inconsistent.
pub fn zgemv(
    alpha: Complex64,
    a: &[Complex64],
    x: &[Complex64],
    beta: Complex64,
    y: &mut [Complex64],
    m: usize,
    n: usize,
) {
    assert_eq!(a.len(), m * n, "zgemv: a must be m*n");
    assert_eq!(x.len(), n, "zgemv: x must have n entries");
    assert_eq!(y.len(), m, "zgemv: y must have m entries");
    #[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
    {
        let m_i32 = to_i32(m);
        let n_i32 = to_i32(n);
        // The complex scalars go by pointer, as CBLAS's own `zgemv` takes them.
        // Safety: every slice was checked against `m` and `n` above, and the shim
        // is linked in this configuration.
        unsafe {
            c_zgemv(&alpha, a.as_ptr(), x.as_ptr(), &beta, y.as_mut_ptr(), &m_i32, &n_i32);
        }
    }
    #[cfg(not(any(feature = "open_blas", feature = "intel_mkl")))]
    crate::native::blas::zgemv(alpha, a, x, beta, y, m, n);
}

/// `y := alpha * A^T * x + beta * y` for complex data (zgemv, trans = 'T').
///
/// `a` is `m x n` column-major, so `x` has `m` entries and `y` has `n`. **Not
/// conjugated**: the conjugated product is `'C'` and a different operation.
///
/// # Panics
///
/// Panics if the shapes are inconsistent.
pub fn zgemv_trans(
    alpha: Complex64,
    a: &[Complex64],
    x: &[Complex64],
    beta: Complex64,
    y: &mut [Complex64],
    m: usize,
    n: usize,
) {
    assert_eq!(a.len(), m * n, "zgemv_trans: a must be m*n");
    assert_eq!(x.len(), m, "zgemv_trans: x must have m entries");
    assert_eq!(y.len(), n, "zgemv_trans: y must have n entries");
    #[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
    {
        let m_i32 = to_i32(m);
        let n_i32 = to_i32(n);
        // Safety: every slice was checked against `m` and `n` above, and the shim
        // is linked in this configuration.
        unsafe {
            c_zgemv_trans(&alpha, a.as_ptr(), x.as_ptr(), &beta, y.as_mut_ptr(), &m_i32, &n_i32);
        }
    }
    #[cfg(not(any(feature = "open_blas", feature = "intel_mkl")))]
    crate::native::blas::zgemv_trans(alpha, a, x, beta, y, m, n);
}

/// `C := alpha * A * A^H + beta * C`, or `C := alpha * A^H * A + beta * C` when
/// `conj_transposed` (zherk).
///
/// `a` is `n x k` column-major without the conjugation and `k x n` with it, and
/// `c` is `n x n` Hermitian: only the requested triangle is written and its
/// diagonal is exactly real. `alpha` and `beta` are real, as `zherk` requires.
///
/// **The transpose here is conjugated.** That is what makes this a Hermitian
/// rank-k update and not a symmetric one; `zherk` with `'N'` computes `A A^H`
/// and with `'C'` computes `A^H A`, and the two conjugate different operands, so
/// they are different kernels (see `crate::native::blas::zherk_rank`).
///
/// # Panics
///
/// Panics if the shapes are inconsistent.
pub fn zherk(
    alpha: f64,
    a: &[Complex64],
    beta: f64,
    c: &mut [Complex64],
    n: usize,
    k: usize,
    upper: bool,
    conj_transposed: bool,
) {
    assert_eq!(a.len(), n * k, "zherk: a must be n*k");
    assert_eq!(c.len(), n * n, "zherk: c must be n*n");
    if n == 0 {
        return;
    }
    #[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
    {
        let n_i32 = to_i32(n);
        let k_i32 = to_i32(k);
        // Safety: `a` and `c` were checked against `n` and `k` above, and the
        // shim is linked in this configuration.
        unsafe {
            c_zherk(
                alpha,
                a.as_ptr(),
                beta,
                c.as_mut_ptr(),
                &n_i32,
                &k_i32,
                if upper { C_TRUE } else { C_FALSE },
                if conj_transposed { C_TRUE } else { C_FALSE },
            );
        }
    }
    #[cfg(not(any(feature = "open_blas", feature = "intel_mkl")))]
    if conj_transposed {
        // `A^H A`, contracting over A's k rows: a different kernel, not the same
        // one with the operands exchanged.
        crate::native::blas::zherk_rank(alpha, a, beta, c, n, k, upper);
    } else {
        crate::native::blas::zherk(alpha, a, beta, c, n, k, upper);
    }
}

/// `C := alpha * A * A^T + beta * C`, or `C := alpha * A^T * A + beta * C` when
/// `transposed` (zsyrk).
///
/// **The symmetric rank-k update, not the Hermitian one.** The transpose is
/// plain, so `alpha` and `beta` are complex and the result is complex symmetric
/// rather than Hermitian: no conjugation, and no real diagonal. [`zherk`] is the
/// other kernel, and passing one's shapes to the other is silently wrong for
/// complex data -- the two agree only when the imaginary parts vanish.
///
/// `a` is `n x k` column-major without the transpose and `k x n` with it; its
/// length is `n*k` either way, so the caller states `transposed` rather than the
/// shapes implying it. `c` is `n x n` and only the `upper` (or lower) triangle of
/// it is written.
///
/// The native counterparts are [`crate::native::blas::zsyrk`] and, for the
/// transposed orientation, [`crate::native::blas::zsyrk_rank`].
///
/// # Panics
///
/// Panics if the shapes are inconsistent.
pub fn zsyrk(
    alpha: Complex64,
    a: &[Complex64],
    beta: Complex64,
    c: &mut [Complex64],
    n: usize,
    k: usize,
    upper: bool,
    transposed: bool,
) {
    assert_eq!(a.len(), n * k, "zsyrk: a must be n*k");
    assert_eq!(c.len(), n * n, "zsyrk: c must be n*n");
    if n == 0 {
        return;
    }
    #[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
    {
        let n_i32 = to_i32(n);
        let k_i32 = to_i32(k);
        // Safety: `a` and `c` were checked against `n` and `k` above, and the shim
        // is linked in this configuration.
        unsafe {
            c_zsyrk(
                &alpha,
                a.as_ptr(),
                &beta,
                c.as_mut_ptr(),
                &n_i32,
                &k_i32,
                if upper { C_TRUE } else { C_FALSE },
                if transposed { C_TRUE } else { C_FALSE },
            );
        }
    }
    #[cfg(not(any(feature = "open_blas", feature = "intel_mkl")))]
    if transposed {
        // `A^T A`, contracting over A's k rows: a different kernel, not the same
        // one with the operands exchanged.
        crate::native::blas::zsyrk_rank(alpha, a, beta, c, n, k, upper);
    } else {
        crate::native::blas::zsyrk(alpha, a, beta, c, n, k, upper);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpx;

    /// Deterministic, non-trivial data: a transposed index cannot coincide with
    /// the right one over these.
    fn data(n: usize) -> Vec<f64> {
        (0..n).map(|i| ((i * 7 % 19) as f64) - 9.0).collect()
    }

    fn assert_close(got: f64, want: f64, what: &str) {
        let scale = want.abs().max(1.0);
        assert!((got - want).abs() / scale < 1e-13, "{what}: got {got} but want {want}");
    }

    /// The complex twin, comparing the modulus of the difference.
    fn assert_close_cpx(got: Complex64, want: Complex64, what: &str) {
        let scale = want.norm().max(1.0);
        assert!((got - want).norm() / scale < 1e-13, "{what}: got {got} but want {want}");
    }

    /// `assert_eq!` on floats would be wrong here: the native kernel and the
    /// library sum in different orders, so only the value is comparable.
    #[test]
    fn dgemm_matches_the_native_kernel() {
        let (m, n, k) = (7usize, 5usize, 9usize);
        let (a, b) = (data(m * k), data(k * n));
        let c0 = data(m * n);
        for (alpha, beta) in [(1.0, 0.0), (2.5, -1.5), (-1.0, 1.0)] {
            let mut want = c0.clone();
            crate::native::blas::dgemm(alpha, &a, &b, beta, &mut want, m, n, k);
            let mut got = c0.clone();
            dgemm(alpha, &a, &b, beta, &mut got, m, n, k);
            for i in 0..m * n {
                assert_close(
                    got[i],
                    want[i],
                    &format!("dgemm m={m} n={n} k={k} alpha={alpha} beta={beta} [{i}]"),
                );
            }
        }
    }

    #[test]
    fn dgemv_and_dgemv_trans_match_the_native_kernel() {
        let (m, n) = (8usize, 6usize);
        let a = data(m * n);

        let x = data(n);
        let y0 = data(m);
        for (alpha, beta) in [(1.0, 0.0), (0.5, 2.0)] {
            let mut want = y0.clone();
            crate::native::blas::dgemv(alpha, &a, &x, beta, &mut want, m, n);
            let mut got = y0.clone();
            dgemv(alpha, &a, &x, beta, &mut got, m, n);
            for i in 0..m {
                assert_close(got[i], want[i], &format!("dgemv [{i}]"));
            }
        }

        // Transposed: x has m entries, y has n.
        let xt = data(m);
        let yt0 = data(n);
        for (alpha, beta) in [(1.0, 0.0), (0.5, 2.0)] {
            let mut want = yt0.clone();
            crate::native::blas::dgemv_trans(alpha, &a, &xt, beta, &mut want, m, n);
            let mut got = yt0.clone();
            dgemv_trans(alpha, &a, &xt, beta, &mut got, m, n);
            for i in 0..n {
                assert_close(got[i], want[i], &format!("dgemv_trans [{i}]"));
            }
        }
    }

    #[test]
    fn level1_routines_match_the_native_kernel() {
        let x = data(23);
        let y = data(23);
        assert_close(ddot(&x, &y), crate::native::blas::ddot(&x, &y), "ddot");
        assert_close(dnrm2(&x), crate::native::blas::dnrm2(&x), "dnrm2");
        assert_close(dasum(&x), crate::native::blas::dasum(&x), "dasum");
        assert_eq!(idamax(&x), crate::native::blas::idamax(&x));
    }

    /// The documented degenerate cases must survive the library too: CBLAS's
    /// `idamax` on an empty vector is not something to rely on, so it is guarded.
    #[test]
    fn level1_routines_handle_empty_slices() {
        assert_eq!(ddot(&[], &[]), 0.0);
        assert_eq!(dnrm2(&[]), 0.0);
        assert_eq!(dasum(&[]), 0.0);
        assert_eq!(idamax(&[]), 0);
        daxpy(1.0, &[], &mut []);
        dcopy(&[], &mut []);
        dscal(2.0, &mut []);
        assert_eq!(dznrm2(&[]), 0.0);
        zaxpy(cpx!(1.0, 1.0), &[], &mut []);
        zcopy(&[], &mut []);
        zscal(cpx!(1.0, 1.0), &mut []);
    }

    /// The level-1 routines that write an output vector, against their native
    /// kernels.
    ///
    /// The comparison is per entry and relative: a library that contracts
    /// `alpha * x` into an FMA is not required to be bit-identical to the Rust
    /// loop. The copy is the exception -- it moves bits, so it is compared
    /// exactly.
    #[test]
    fn level1_updates_match_the_native_kernel() {
        let x = data(23);
        let y0 = data(23);
        for alpha in [1.0, -2.5, 0.0] {
            let mut want = y0.clone();
            crate::native::blas::daxpy(alpha, &x, &mut want);
            let mut got = y0.clone();
            daxpy(alpha, &x, &mut got);
            for i in 0..got.len() {
                assert_close(got[i], want[i], &format!("daxpy alpha={alpha} [{i}]"));
            }

            let mut want = y0.clone();
            crate::native::blas::dscal(alpha, &mut want);
            let mut got = y0.clone();
            dscal(alpha, &mut got);
            for i in 0..got.len() {
                assert_close(got[i], want[i], &format!("dscal alpha={alpha} [{i}]"));
            }
        }

        let mut want = vec![0.0; x.len()];
        crate::native::blas::dcopy(&x, &mut want);
        let mut got = vec![0.0; x.len()];
        dcopy(&x, &mut got);
        assert_eq!(got, want, "dcopy");
    }

    /// The complex twins of the level-1 updates, plus the complex norm.
    #[test]
    fn complex_level1_routines_match_the_native_kernel() {
        let x: Vec<Complex64> = (0..23)
            .map(|i| cpx!(((i % 5) as f64) - 2.0, ((i % 7) as f64) - 3.0))
            .collect();
        let y0: Vec<Complex64> = (0..23).map(|i| cpx!((i % 3) as f64, ((i % 4) as f64) - 1.5)).collect();

        for alpha in [cpx!(1.0, 0.0), cpx!(0.5, -1.5)] {
            let mut want = y0.clone();
            crate::native::blas::zaxpy(alpha, &x, &mut want);
            let mut got = y0.clone();
            zaxpy(alpha, &x, &mut got);
            for i in 0..got.len() {
                assert_close_cpx(got[i], want[i], &format!("zaxpy alpha={alpha} [{i}]"));
            }

            let mut want = y0.clone();
            crate::native::blas::zscal(alpha, &mut want);
            let mut got = y0.clone();
            zscal(alpha, &mut got);
            for i in 0..got.len() {
                assert_close_cpx(got[i], want[i], &format!("zscal alpha={alpha} [{i}]"));
            }
        }

        let mut want = vec![cpx!(0.0, 0.0); x.len()];
        crate::native::blas::zcopy(&x, &mut want);
        let mut got = vec![cpx!(0.0, 0.0); x.len()];
        zcopy(&x, &mut got);
        assert_eq!(got, want, "zcopy");

        assert_close(dznrm2(&x), crate::native::blas::dznrm2(&x), "dznrm2");
    }

    /// The complex twin of the norm sweep, plus its own rejection of an unknown
    /// character: the panic message has to name `zlange`, not `dlange`.
    #[test]
    fn zlange_matches_the_native_kernel_for_every_norm() {
        let (m, n) = (4usize, 6usize);
        let a: Vec<Complex64> = (0..m * n)
            .map(|i| cpx!(((i * 3) % 7) as f64 - 3.0, ((i * 5) % 11) as f64 - 5.0))
            .collect();
        for norm in ['1', 'O', 'I', 'F', 'E', 'M'] {
            let got = zlange(norm, &a, m, n);
            let want = crate::native::blas::zlange(norm, &a, m, n);
            assert_close(got, want, &format!("zlange '{norm}'"));
        }
    }

    #[test]
    #[should_panic(expected = "unrecognized norm")]
    fn zlange_rejects_an_unknown_norm() {
        zlange('X', &[cpx!(1.0, 0.0)], 1, 1);
    }

    /// Both orientations of the complex matrix-vector product.
    ///
    /// `m != n` is what makes a wrong leading dimension or a swapped `x`/`y`
    /// extent visible; with a square matrix either mistake would still agree.
    #[test]
    fn zgemv_and_zgemv_trans_match_the_native_kernel() {
        let (m, n) = (8usize, 6usize);
        let a: Vec<Complex64> = (0..m * n)
            .map(|i| cpx!(((i % 3) as f64) - 1.0, ((i % 5) as f64) - 2.0))
            .collect();

        let x: Vec<Complex64> = (0..n).map(|i| cpx!(((i % 4) as f64) - 1.5, (i % 2) as f64)).collect();
        let y0: Vec<Complex64> = (0..m).map(|i| cpx!((i % 2) as f64, 0.5)).collect();
        // Transposed: x has m entries, y has n.
        let xt: Vec<Complex64> = (0..m).map(|i| cpx!(((i % 3) as f64) - 1.0, (i % 4) as f64)).collect();
        let yt0: Vec<Complex64> = (0..n).map(|i| cpx!(0.5, ((i % 3) as f64) - 1.0)).collect();

        for (alpha, beta) in [(cpx!(1.0, 0.0), cpx!(0.0, 0.0)), (cpx!(0.5, -1.5), cpx!(2.0, 0.25))] {
            let mut want = y0.clone();
            crate::native::blas::zgemv(alpha, &a, &x, beta, &mut want, m, n);
            let mut got = y0.clone();
            zgemv(alpha, &a, &x, beta, &mut got, m, n);
            for i in 0..m {
                assert_close_cpx(got[i], want[i], &format!("zgemv [{i}]"));
            }

            let mut want = yt0.clone();
            crate::native::blas::zgemv_trans(alpha, &a, &xt, beta, &mut want, m, n);
            let mut got = yt0.clone();
            zgemv_trans(alpha, &a, &xt, beta, &mut got, m, n);
            for i in 0..n {
                assert_close_cpx(got[i], want[i], &format!("zgemv_trans [{i}]"));
            }
        }
    }

    #[test]
    fn dlange_matches_the_native_kernel_for_every_norm() {
        let (m, n) = (4usize, 6usize);
        let a = data(m * n);
        for norm in ['1', 'O', 'I', 'F', 'E', 'M'] {
            let got = dlange(norm, &a, m, n);
            let want = crate::native::blas::dlange(norm, &a, m, n);
            assert_close(got, want, &format!("dlange '{norm}'"));
        }
    }

    #[test]
    #[should_panic(expected = "unrecognized norm")]
    fn dlange_rejects_an_unknown_norm() {
        dlange('X', &[1.0], 1, 1);
    }

    #[test]
    fn zgemm_matches_the_native_kernel() {
        let (m, n, k) = (5usize, 4usize, 6usize);
        let a: Vec<Complex64> = (0..m * k).map(|i| cpx!((i % 3) as f64, (i % 5) as f64 - 2.0)).collect();
        let b: Vec<Complex64> = (0..k * n).map(|i| cpx!((i % 4) as f64 - 1.0, (i % 7) as f64)).collect();
        let c0: Vec<Complex64> = (0..m * n).map(|i| cpx!((i % 2) as f64, 0.5)).collect();
        for (alpha, beta) in [(cpx!(1.0, 0.0), cpx!(0.0, 0.0)), (cpx!(0.5, -1.5), cpx!(2.0, 0.25))] {
            let mut want = c0.clone();
            crate::native::blas::zgemm(alpha, &a, &b, beta, &mut want, m, n, k);
            let mut got = c0.clone();
            zgemm(alpha, &a, &b, beta, &mut got, m, n, k);
            for i in 0..m * n {
                assert!(
                    (got[i] - want[i]).norm() < 1e-13,
                    "zgemm [{i}]: got {} but want {}",
                    got[i],
                    want[i]
                );
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    type RealGemm = fn(f64, &[f64], &[f64], f64, &mut [f64], usize, usize, usize);
    type ComplexGemm = fn(Complex64, &[Complex64], &[Complex64], Complex64, &mut [Complex64], usize, usize, usize);

    /// The three transposed real combinations, each against its native kernel.
    ///
    /// `A` and `B` are stored in the orientation each variant expects, and the
    /// three dimensions differ, so a leading dimension taken from the wrong axis
    /// shows up in the result instead of being absorbed by a coincidence.
    #[test]
    fn transposed_dgemm_matches_the_native_kernels() {
        let (m, n, k) = (7usize, 5usize, 9usize);
        let a_t = data(k * m); // A stored k x m, as 'T' expects
        let a_n = data(m * k); // A stored m x k, as 'N' expects
        let b_n = data(k * n);
        let b_t = data(n * k);
        let c0 = data(m * n);
        let cases: [(&str, RealGemm, RealGemm, &[f64], &[f64]); 3] = [
            ("dgemm_tn", dgemm_tn, crate::native::blas::dgemm_tn, &a_t, &b_n),
            ("dgemm_nt", dgemm_nt, crate::native::blas::dgemm_nt, &a_n, &b_t),
            ("dgemm_tt", dgemm_tt, crate::native::blas::dgemm_tt, &a_t, &b_t),
        ];
        for (label, lib, native, a, b) in cases {
            for (alpha, beta) in [(1.0, 0.0), (2.5, -1.5), (0.0, 3.0)] {
                let mut want = c0.clone();
                native(alpha, a, b, beta, &mut want, m, n, k);
                let mut got = c0.clone();
                lib(alpha, a, b, beta, &mut got, m, n, k);
                for i in 0..m * n {
                    assert_close(got[i], want[i], &format!("{label} alpha={alpha} beta={beta} [{i}]"));
                }
            }
        }
    }

    /// The same for the complex twins. The transpose here is the plain one, so
    /// complex data that is not Hermitian is what distinguishes a conjugation
    /// mistake from a correct result.
    #[test]
    fn transposed_zgemm_matches_the_native_kernels() {
        let (m, n, k) = (5usize, 4usize, 6usize);
        let x = |len: usize| -> Vec<Complex64> {
            (0..len)
                .map(|i| cpx!((i % 3) as f64 - 1.0, (i % 5) as f64 - 2.0))
                .collect()
        };
        let a_t = x(k * m);
        let a_n = x(m * k);
        let b_n = x(k * n);
        let b_t = x(n * k);
        let c0 = x(m * n);
        let cases: [(&str, ComplexGemm, ComplexGemm, &[Complex64], &[Complex64]); 3] = [
            ("zgemm_tn", zgemm_tn, crate::native::blas::zgemm_tn, &a_t, &b_n),
            ("zgemm_nt", zgemm_nt, crate::native::blas::zgemm_nt, &a_n, &b_t),
            ("zgemm_tt", zgemm_tt, crate::native::blas::zgemm_tt, &a_t, &b_t),
        ];
        for (label, lib, native, a, b) in cases {
            for (alpha, beta) in [(cpx!(1.0, 0.0), cpx!(0.0, 0.0)), (cpx!(0.5, -1.5), cpx!(2.0, 0.25))] {
                let mut want = c0.clone();
                native(alpha, a, b, beta, &mut want, m, n, k);
                let mut got = c0.clone();
                lib(alpha, a, b, beta, &mut got, m, n, k);
                for i in 0..m * n {
                    assert!(
                        (got[i] - want[i]).norm() < 1e-13,
                        "{label} alpha={alpha} beta={beta} [{i}]: got {} but want {}",
                        got[i],
                        want[i]
                    );
                }
            }
        }
    }

    #[test]
    fn dger_matches_the_native_kernel() {
        let (m, n) = (6usize, 5usize);
        let x = data(m);
        let y = data(n);
        let a0 = data(m * n);
        for alpha in [1.0, -2.5, 0.0] {
            let mut want = a0.clone();
            crate::native::blas::dger(alpha, &x, &y, &mut want, m, n);
            let mut got = a0.clone();
            dger(alpha, &x, &y, &mut got, m, n);
            for i in 0..m * n {
                assert_close(got[i], want[i], &format!("dger alpha={alpha} [{i}]"));
            }
        }
    }

    #[test]
    fn zger_matches_the_native_kernel() {
        let (m, n) = (4usize, 6usize);
        let x: Vec<Complex64> = (0..m).map(|i| cpx!((i % 3) as f64 - 1.0, (i % 2) as f64)).collect();
        let y: Vec<Complex64> = (0..n)
            .map(|i| cpx!((i % 4) as f64 - 1.5, (i % 5) as f64 - 2.0))
            .collect();
        let a0: Vec<Complex64> = (0..m * n).map(|i| cpx!((i % 2) as f64, 0.5)).collect();
        for alpha in [cpx!(1.0, 0.0), cpx!(0.5, -1.5)] {
            let mut want = a0.clone();
            crate::native::blas::zger(alpha, &x, &y, &mut want, m, n);
            let mut got = a0.clone();
            zger(alpha, &x, &y, &mut got, m, n);
            for i in 0..m * n {
                assert!(
                    (got[i] - want[i]).norm() < 1e-13,
                    "zger [{i}]: got {} but want {}",
                    got[i],
                    want[i]
                );
            }
        }
    }

    /// Both orientations and both triangles of the symmetric rank-k update.
    ///
    /// `n != k` matters: the two orientations contract over `a`'s columns and over
    /// its rows respectively, so with `n == k` a dispatch that passed the wrong
    /// one would still agree. The untaken triangle is also checked, because "only
    /// the requested triangle is written" is part of the contract.
    #[test]
    fn dsyrk_matches_the_native_kernel() {
        let (n, k) = (5usize, 8usize);
        // a is n x k in the first case and k x n in the second; both are n*k long.
        let a = data(n * k);
        let c0 = data(n * n);
        for transposed in [false, true] {
            for upper in [false, true] {
                let mut want = c0.clone();
                crate::native::blas::dsyrk_rank(2.5, &a, -1.5, &mut want, n, k, upper, transposed);
                let mut got = c0.clone();
                dsyrk(2.5, &a, -1.5, &mut got, n, k, upper, transposed);
                for j in 0..n {
                    for i in 0..n {
                        let idx = i + j * n;
                        let in_triangle = if upper { i <= j } else { i >= j };
                        assert_close(
                            got[idx],
                            want[idx],
                            &format!("dsyrk transposed={transposed} upper={upper} ({i},{j})"),
                        );
                        if !in_triangle {
                            assert_eq!(got[idx], c0[idx], "dsyrk must leave ({i},{j}) alone");
                        }
                    }
                }
            }
        }
    }

    /// The same sweep for the Hermitian twin, where the "transpose" is conjugated
    /// and the diagonal must come out exactly real.
    #[test]
    fn zherk_matches_the_native_kernel() {
        let (n, k) = (4usize, 6usize);
        let a: Vec<Complex64> = (0..n * k)
            .map(|i| cpx!((i % 3) as f64 - 1.0, (i % 5) as f64 - 2.0))
            .collect();
        let c0: Vec<Complex64> = (0..n * n).map(|i| cpx!((i % 2) as f64, 0.5)).collect();
        for conj_transposed in [false, true] {
            for upper in [false, true] {
                let native = if conj_transposed {
                    crate::native::blas::zherk_rank as fn(f64, &[Complex64], f64, &mut [Complex64], usize, usize, bool)
                } else {
                    crate::native::blas::zherk as fn(f64, &[Complex64], f64, &mut [Complex64], usize, usize, bool)
                };
                let mut want = c0.clone();
                native(2.5, &a, -1.5, &mut want, n, k, upper);
                let mut got = c0.clone();
                zherk(2.5, &a, -1.5, &mut got, n, k, upper, conj_transposed);
                for j in 0..n {
                    for i in 0..n {
                        let idx = i + j * n;
                        assert!(
                            (got[idx] - want[idx]).norm() < 1e-13,
                            "zherk conj_transposed={conj_transposed} upper={upper} ({i},{j}): got {} but want {}",
                            got[idx],
                            want[idx]
                        );
                        if i == j {
                            assert_eq!(got[idx].im, 0.0, "the Hermitian diagonal must be exactly real");
                        }
                        let in_triangle = if upper { i <= j } else { i >= j };
                        if !in_triangle {
                            assert_eq!(got[idx], c0[idx], "zherk must leave ({i},{j}) alone");
                        }
                    }
                }
            }
        }
    }

    /// Both orientations and both triangles of the complex *symmetric* rank-k
    /// update, against the native kernels.
    ///
    /// `n != k` matters for the same reason as in [`dsyrk`]: the two orientations
    /// contract over different axes, so passing the untransposed `k` would agree
    /// only on a square `a`. The untaken triangle is checked too, because leaving
    /// it alone is part of the contract.
    ///
    /// This is `zsyrk`, not `zherk`: nothing here is conjugated, and the test data
    /// has imaginary parts throughout, so a dispatch that conjugated would differ
    /// in every off-diagonal entry.
    #[test]
    fn zsyrk_matches_the_native_kernel() {
        let (n, k) = (5usize, 8usize);
        let a: Vec<Complex64> = (0..n * k)
            .map(|i| cpx!(((i % 3) as f64) - 1.0, ((i % 5) as f64) - 2.0))
            .collect();
        let c0: Vec<Complex64> = (0..n * n).map(|i| cpx!((i % 2) as f64, 0.5)).collect();
        let (alpha, beta) = (cpx!(2.5, -1.0), cpx!(-1.5, 0.25));
        type Zsyrk = fn(Complex64, &[Complex64], Complex64, &mut [Complex64], usize, usize, bool);
        for transposed in [false, true] {
            let native: Zsyrk = if transposed {
                crate::native::blas::zsyrk_rank
            } else {
                crate::native::blas::zsyrk
            };
            for upper in [false, true] {
                let mut want = c0.clone();
                native(alpha, &a, beta, &mut want, n, k, upper);
                let mut got = c0.clone();
                zsyrk(alpha, &a, beta, &mut got, n, k, upper, transposed);
                for j in 0..n {
                    for i in 0..n {
                        let idx = i + j * n;
                        assert_close_cpx(
                            got[idx],
                            want[idx],
                            &format!("zsyrk transposed={transposed} upper={upper} ({i},{j})"),
                        );
                        let in_triangle = if upper { i <= j } else { i >= j };
                        if !in_triangle {
                            assert_eq!(got[idx], c0[idx], "zsyrk must leave ({i},{j}) alone");
                        }
                    }
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// The rank-1 and rank-k entries against values captured from CBLAS.
///
/// # Provenance
///
/// The same throwaway C driver as the transposed-`gemm` fixtures in
/// `crate::native::blas::reference_tests` (source and full output in the commit
/// message that added these entries) also called `cblas_dger`, `cblas_zgeru`,
/// `cblas_dsyrk` and `cblas_zherk` on its deterministic data, and printed the
/// inputs as well as the results. What follows is that output, transcribed.
///
/// OpenBLAS and MKL 2026.0 agree on these to 1.5e-14 relative at worst, which is
/// why the comparison is 1e-13 relative: a wrong leading dimension or a contracted
/// length taken from the wrong axis is wrong by O(1), so the tolerance cannot hide
/// one. The two `zherk` diagonals are exactly real in both libraries and are
/// asserted exactly, as is the untouched triangle.
#[cfg(test)]
mod reference_tests {
    use super::*;
    use crate::cpx;

    fn assert_close(got: f64, want: f64, what: &str) {
        let scale = want.abs().max(1.0);
        assert!(
            (got - want).abs() / scale < 1e-13,
            "{what}: got {got} but the reference says {want}"
        );
    }

    /// `A := 1.5 x y^T + A`, A is `4 x 3`.
    #[test]
    fn dger_matches_cblas() {
        let x: &[f64] = &[242.66666666666666, 55.0, 287.66666666666669, 144.0];

        let y: &[f64] = &[-321.33333333333331, -217.33333333333334, 54.666666666666664];

        let a0: &[f64] = &[
            -59.166666666666664,
            134.66666666666666,
            -4.0,
            -142.0,
            -115.66666666666667,
            21.0,
            -86.333333333333329,
            94.166666666666671,
            18.666666666666668,
            -108.0,
            -93.666666666666671,
            57.166666666666664,
        ];

        let want: &[f64] = &[
            -117024.5,
            -26375.333333333332,
            -138659.33333333334,
            -69550.0,
            -79225.0,
            -17909.0,
            -93865.666666666672,
            -46849.833333333336,
            19917.333333333332,
            4402.0,
            23495.0,
            11865.166666666666,
        ];
        let (m, n) = (4usize, 3usize);
        let mut a = a0.to_vec();
        dger(1.5, x, y, &mut a, m, n);
        for i in 0..m * n {
            assert_close(a[i], want[i], &format!("dger [{i}]"));
        }
    }

    /// `A := (1 - 0.5i) x y^T + A`, A is `2 x 3`. Not conjugated: `cblas_zgeru`.
    ///
    /// If the shim ever routed this to `cblas_zgerc` instead, every off-diagonal
    /// entry's imaginary part would change sign -- which is exactly what the
    /// fixture catches.
    #[test]
    fn zger_matches_cblas() {
        let x: &[Complex64] = &[cpx!(1.0, 0.5), cpx!(-2.0, 0.25), cpx!(2.0, -1.5), cpx!(0.0, 1.0)];

        let y: &[Complex64] = &[cpx!(0.5, 1.0), cpx!(2.0, -0.5), cpx!(-1.0, 0.75)];

        let a0: &[Complex64] = &[
            cpx!(9.9166666666666661, -38.583333333333336),
            cpx!(-79.166666666666671, -35.333333333333336),
            cpx!(-4.166666666666667, 63.166666666666664),
            cpx!(68.0, -17.666666666666668),
            cpx!(-24.166666666666668, -16.916666666666668),
            cpx!(-78.916666666666671, -55.5),
        ];

        let want: &[Complex64] = &[
            cpx!(10.541666666666666, -37.333333333333336),
            cpx!(-81.354166666666671, -36.583333333333336),
            cpx!(-1.666666666666667, 62.541666666666664),
            cpx!(64.875, -14.229166666666668),
            cpx!(-25.416666666666668, -15.979166666666668),
            cpx!(-77.979166666666671, -58.15625),
        ];
        let (m, n) = (2usize, 3usize);
        let mut a = a0.to_vec();
        // The driver's x buffer is four entries long and it passed m = 2, so only
        // the first `m` of them reached the library; the fixture keeps the whole
        // printed buffer and slices it here for that reason.
        zger(cpx!(1.0, -0.5), &x[..m], y, &mut a, m, n);
        for i in 0..m * n {
            let scale = want[i].norm().max(1.0);
            assert!(
                (a[i] - want[i]).norm() / scale < 1e-13,
                "zger [{i}]: got {} but the reference says {}",
                a[i],
                want[i]
            );
        }
    }

    /// `C := 2 A A^T - C` and `C := 2 A^T A - C`, lower triangle, `n = 3`, `k = 4`.
    #[test]
    fn dsyrk_matches_cblas() {
        let a: &[f64] = &[
            242.66666666666666,
            55.0,
            287.66666666666669,
            144.0,
            -254.33333333333334,
            284.0,
            105.0,
            78.666666666666671,
            -87.666666666666671,
            -48.333333333333336,
            -295.33333333333331,
            -227.33333333333334,
        ];

        let c0: &[f64] = &[
            56.833333333333336,
            13.0,
            11.666666666666666,
            120.16666666666667,
            110.5,
            110.66666666666667,
            24.666666666666668,
            -109.83333333333333,
            85.666666666666671,
        ];

        let want_n: &[f64] = &[
            185911.61111111109,
            -1498.7777777777831,
            224960.11111111112,
            120.16666666666667,
            322130.83333333331,
            7556.6666666666615,
            24.666666666666668,
            -109.83333333333333,
            445462.33333333331,
        ];

        let want_t: &[f64] = &[
            330743.61111111112,
            -9143.4444444444325,
            -283263.00000000006,
            120.16666666666667,
            324999.27777777775,
            -80758.0,
            24.666666666666668,
            -109.83333333333333,
            297761.88888888888,
        ];
        let (n, k) = (3usize, 4usize);
        for (transposed, want) in [(false, want_n), (true, want_t)] {
            let mut c = c0.to_vec();
            dsyrk(2.0, a, -1.0, &mut c, n, k, false, transposed);
            for j in 0..n {
                for i in 0..n {
                    let idx = i + j * n;
                    assert_close(c[idx], want[idx], &format!("dsyrk transposed={transposed} ({i},{j})"));
                    if i < j {
                        assert_eq!(c[idx], c0[idx], "the upper triangle must be left alone");
                    }
                }
            }
        }
    }

    /// `C := 2 A A^H - C` and `C := 2 A^H A - C`, lower triangle, `n = 2`, `k = 3`.
    #[test]
    fn zherk_matches_cblas() {
        let a: &[Complex64] = &[
            cpx!(-133.33333333333334, 47.0),
            cpx!(149.16666666666666, -45.833333333333336),
            cpx!(-10.5, -75.916666666666671),
            cpx!(143.33333333333334, -11.333333333333334),
            cpx!(-19.833333333333332, 17.25),
            cpx!(52.333333333333336, 47.5),
        ];

        let c0: &[Complex64] = &[
            cpx!(-96.166666666666671, 0.0),
            cpx!(-94.166666666666671, 120.16666666666667),
            cpx!(159.5, -136.16666666666666),
            cpx!(0.66666666666666663, 0.0),
        ];

        let want_n: &[Complex64] = &[
            cpx!(53198.75, 0.0),
            cpx!(-45718.305555555562, 16391.500000000004),
            cpx!(159.5, -136.16666666666666),
            cpx!(100037.94444444445, 0.0),
        ];

        let want_c: &[Complex64] = &[
            cpx!(100519.68055555558, 0.0),
            cpx!(-55002.666666666679, 54.305555555554136),
            cpx!(159.5, -136.16666666666666),
            cpx!(52717.013888888898, 0.0),
        ];
        let (n, k) = (2usize, 3usize);
        for (conj_transposed, want) in [(false, want_n), (true, want_c)] {
            let mut c = c0.to_vec();
            zherk(2.0, a, -1.0, &mut c, n, k, false, conj_transposed);
            for j in 0..n {
                for i in 0..n {
                    let idx = i + j * n;
                    let scale = want[idx].norm().max(1.0);
                    assert!(
                        (c[idx] - want[idx]).norm() / scale < 1e-13,
                        "zherk conj_transposed={conj_transposed} ({i},{j}): got {} but the reference says {}",
                        c[idx],
                        want[idx]
                    );
                    if i == j {
                        assert_eq!(c[idx].im, 0.0, "the Hermitian diagonal must be exactly real");
                    }
                    if i < j {
                        assert_eq!(c[idx], c0[idx], "the upper triangle must be left alone");
                    }
                }
            }
        }
    }
}
