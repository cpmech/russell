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
//! # What is here, and what is not
//!
//! Routed: `dgemm`, `dgemv`, `dgemv_trans`, `ddot`, `dnrm2`, `dasum`, `idamax`,
//! `dlange` and `zgemm`. `dgemv_trans` and `dasum` are included because the call
//! sites being routed need them -- `vec_mat_mul` and `Norm::One`, respectively
//! -- and splitting a file between this module and a local `extern "C"` block
//! would be worse than adding the entry.
//!
//! Everything else keeps its existing per-file dispatch, which is the convention
//! described in `AGENTS.md`. In particular this module does **not** cover the
//! transposed `dgemm`/`zgemm` of `mat_t_mat_mul`/`complex_mat_t_mat_mul`, nor
//! `dger`, `dsyrk`/`zherk`, `zgemv`, or the level-1 `daxpy`/`dcopy`/`dscal`
//! family: `native::blas` has no transposed `gemm` to mirror, and the level-1
//! routines are not what this dispatch is for. Those paths are unchanged.
//!
//! # Why the shim, and not CBLAS directly
//!
//! The shim is the crate's only C layer, and `build.rs` selects which library it
//! compiles against. Calling CBLAS from here would mean a second set of
//! declarations per call site and a second library assumption in the Rust code,
//! which is exactly what `interface_blas.c` exists to centralize.

use crate::Complex64;

#[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
use crate::internal::{NORM_EUC, NORM_INF, NORM_MAX, NORM_ONE};
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
}
