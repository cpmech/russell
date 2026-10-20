//! LAPACK driver entry points, dispatched to the native routines or to the
//! linked library.
//!
//! Every function here has the same signature as its counterpart in
//! [`crate::native::lapack`], so a call site names the *decomposition* rather
//! than the backend: the default build forwards to the native Rust routine, and
//! a build with an accelerator forwards to the C shim
//! (`c_code/interface_blas.c`), which calls LAPACK.
//!
//! # What is here, and what is not
//!
//! The five drivers the crate's public API leans on hardest, and nothing else:
//!
//! | Fortran | Here as | Call sites |
//! |---|---|---|
//! | `dgesv` | [`dgesv`] | [`crate::solve_lin_sys`] |
//! | `dpotrf` | [`dpotrf`] | [`crate::mat_cholesky`] |
//! | `dsyev` | [`dsyev`] | [`crate::mat_eigen_sym`] |
//! | `dgesvd` | [`dgesvd`] | [`crate::mat_svd`] |
//! | `dgeev` | [`dgeev`], [`dgeev_lr`], [`dgeev_eigenvalues`] | [`crate::mat_eigen`], [`crate::mat_eigen_lr`], [`crate::mat_eigenvalues`] |
//!
//! `dgeev` appears three times because that is how many different jobs the crate
//! asks of it: `dgeev` computes the right eigenvectors (`JOBVR = 'V'`),
//! [`dgeev_lr`] both sets (`JOBVL = JOBVR = 'V'`), and
//! [`dgeev_eigenvalues`] neither (`'N'`, `'N'`) -- which is not an optimization
//! to skip lightly, since `dtrevc3` and its back-transformation are most of
//! `dgeev`'s cost.
//!
//! Still native, and untouched by this module: `dgetrf`/`dgetri` (used by
//! `mat_inverse`), every complex routine (`zgesv`, `zgetrf`, `zgetri`, `zpotrf`,
//! `zheev`, `zgeev`, `zggev`, `zgesvd`), the generalized eigen-decomposition
//! (`dggev`/`zggev`), `dlange`, and `small_mat_svd`, which is a const-generic
//! entry over stack arrays rather than a driver -- routing it would hand LAPACK
//! the workspace allocation that the point of the `small_` API is to avoid. Their
//! C wrappers are already in the shim, so routing one of the others later is an
//! entry here plus a call-site edit, not new C code.
//!
//! # `info` is translated here, once
//!
//! LAPACK reports through an integer out-parameter: `info < 0` names an illegal
//! argument, `info > 0` means the routine could not finish (a zero pivot, a
//! leading minor that is not positive definite, or an iteration that gave up).
//! The native drivers never expose that integer -- they return an error at the
//! boundary, which is the crate's convention (see `AGENTS.md`) -- so the
//! accelerated path does the same, and produces the *same* [`LapackError`] the
//! native path would: `Singular` for a zero factor, `Convergence` for a failed
//! iteration, `BadShape` for an illegal argument.
//!
//! The mapping is shared rather than per call site so that a caller's
//! `match` on [`LapackError`] keeps working unchanged, and it prints the
//! diagnostic the call sites used to print (`U(3,3) is exactly zero`, and so
//! on) instead of dropping it.
//!
//! ## The one thing that is not identical: what counts as singular
//!
//! The *kinds* and the messages agree; the boundary does not, and it cannot
//! without a rank check neither backend performs. The native `dgetrf` treats a
//! pivot at or below [`crate::native::lapack::ZERO_DETERMINANT`] (1e-15) as zero,
//! while LAPACK's `dgetrf` -- and therefore `dgesv` and `dpotrf` -- reports
//! `info > 0` only for an *exactly* zero pivot or a leading minor that is not
//! positive definite. A matrix that is singular to 1e-16 but not exactly is
//! `Singular` on the native path and solved to working precision on the
//! accelerated one. That divergence predates this module -- the call sites
//! already had it between their two branches -- and closing it would mean
//! post-checking the factor on the accelerated path, which is a policy change
//! rather than a dispatch one.
//!
//! # Precedence and gating
//!
//! As in [`crate::base::blas_dispatch`]: `intel_mkl` wins when both features are
//! enabled, since `build.rs` decides which library defines the `c_*` symbols,
//! and the gates here only ask whether *an* accelerator is in use. A build with
//! neither feature compiles no C at all, which is also what keeps
//! `wasm32-unknown-unknown` building: the shim needs a libc the wasm target does
//! not have, and the feature gate -- not a `target_arch` test -- is what excludes
//! it.
//!
//! # Which entries reach the library
//!
//! All five, measured through the public entry points with
//! `cargo bench --bench backends_benchmark` (`lapack_*` groups) against MKL 2026.0
//! pinned to one thread. The columns are the same binary, same data, run once
//! plain and once with `--all-features`; `native` is the Rust driver those runs
//! reach, quoted as the control:
//!
//! | call site (driver) | n = 128 | n = 512 | routed |
//! |---|---|---|---|
//! | `solve_lin_sys` (`dgesv`) | 931 us -> 44.7 us (21x) | 101 ms -> 1.38 ms (74x) | yes |
//! | `mat_cholesky` (`dpotrf`) | 155 us -> 27.5 us (5.6x) | 14.8 ms -> 761 us (19x) | yes |
//! | `mat_eigen_sym` (`dsyev`) | 1.51 ms -> 673 us (2.2x) | 61.9 ms -> 21.7 ms (2.9x) | yes |
//! | `mat_svd` (`dgesvd`), n x n | 7.50 ms -> 2.50 ms (3.0x) | 625 ms -> 50.7 ms (12x) | yes |
//! | `mat_svd` (`dgesvd`), 512 x 256 | 72.6 ms -> 8.30 ms (8.8x) | -- | yes |
//! | `mat_eigen` (`dgeev`, vectors) | 6.64 ms -> 2.41 ms (2.8x) | 542 ms -> 72.4 ms (7.5x) | yes |
//! | `mat_eigenvalues` (`dgeev`, values) | 6.53 ms -> 1.42 ms (4.6x) | 560 ms -> 52.5 ms (11x) | yes |
//!
//! The native control rows agreed between the two runs to within 7% (mostly
//! 1-4%), so the columns are comparable. Nothing was left native: unlike the
//! level-1 BLAS routines, whose win depends on the size, these are O(n^3) and the
//! library wins at every size measured, which is why there is no `if n > ...`
//! anywhere in this module.
//!
//! The two smallest wins are `dsyev` (2.2-2.9x) and `dgeev` with vectors
//! (2.8-7.5x), and both are the same story: the native reductions and
//! back-substitutions are BLAS-2 at best, while LAPACK blocks the same algorithms
//! into BLAS-3. The largest is `dgesv` -- 74x at n = 512 -- because the native LU
//! is a scalar triple loop with no blocking at all.
//!
//! `mat_eigenvalues` is quoted separately from `mat_eigen` because it asks
//! `dgeev` for a different job: `'N'`/`'N'` skips `dtrevc3`, which is 28% of the
//! accelerated cost (52 ms against 72 ms at n = 512). The native driver always
//! computes the vectors and discards them, so natively the two cost the same --
//! one more reason the `'N'` entry exists rather than being folded into
//! [`dgeev`].

use crate::native::lapack::LapackError;

#[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
use crate::internal::dgeev_data;
#[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
use crate::internal::dgeev_data_lr;
#[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
use crate::internal::{C_FALSE, C_TRUE, CcBool, SVD_CODE_A};
#[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
use crate::to_i32;

#[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
unsafe extern "C" {
    // All of these are the shim's own entry points (see `c_code/interface_blas.c`
    // and its `interface.h`), which supply LAPACK's character arguments and their
    // hidden lengths. They are Fortran LAPACK calls, not CBLAS: the leading
    // dimensions, the workspace and `lwork` are the caller's, which is why every
    // entry below sizes a workspace instead of letting the shim do it.
    fn c_dgesv(
        n: *const i32,
        nrhs: *const i32,
        a: *mut f64,
        lda: *const i32,
        ipiv: *mut i32,
        b: *mut f64,
        ldb: *const i32,
        info: *mut i32,
    );
    fn c_dpotrf(upper: CcBool, n: *const i32, a: *mut f64, lda: *const i32, info: *mut i32);
    fn c_dsyev(
        calc_v: CcBool,
        upper: CcBool,
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        w: *mut f64,
        work: *mut f64,
        lwork: *const i32,
        info: *mut i32,
    );
    fn c_dgesvd(
        jobu_code: i32,
        jobvt_code: i32,
        m: *const i32,
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        s: *mut f64,
        u: *mut f64,
        ldu: *const i32,
        vt: *mut f64,
        ldvt: *const i32,
        work: *mut f64,
        lwork: *const i32,
        info: *mut i32,
    );
    fn c_dgeev(
        calc_vl: CcBool,
        calc_vr: CcBool,
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        wr: *mut f64,
        wi: *mut f64,
        vl: *mut f64,
        ldvl: *const i32,
        vr: *mut f64,
        ldvr: *const i32,
        work: *mut f64,
        lwork: *const i32,
        info: *mut i32,
    );
}

/// Maps LAPACK's `INFO` to the crate's error, printing the diagnostic the call
/// sites printed before they were centralized.
///
/// `INFO < 0` names an illegal argument, and it is `BadShape` here because that
/// is the only way a shape can reach LAPACK wrong: the entries in this module
/// check every length before calling, so in practice `info < 0` stays
/// unreachable and the message exists for the day one of those checks is wrong.
/// `INFO > 0` is the caller's error kind -- `Singular` for an exactly zero
/// factor, `Convergence` for an iteration that gave up.
#[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
fn lapack_error(routine: &str, info: i32, positive: LapackError, detail: impl FnOnce()) -> LapackError {
    if info < 0 {
        println!("LAPACK ERROR ({routine}): Argument #{} had an illegal value", -info);
        LapackError::BadShape
    } else {
        detail();
        positive
    }
}

/// `dgesv`: solve `A X = B`.
///
/// `a` is `n x n` column-major and is destroyed; `b` is the right-hand side on
/// entry and the solution on exit. Its length is the number of rows of `B`, as
/// in [`crate::native::lapack::dgesv`].
///
/// # Errors
///
/// [`LapackError::Singular`] if `A` is singular, [`LapackError::BadShape`] if the
/// shapes disagree.
pub fn dgesv(a: &mut [f64], b: &mut [f64]) -> Result<(), LapackError> {
    let n = b.len();
    if a.len() != n * n {
        return Err(LapackError::BadShape);
    }
    if n == 0 {
        return Ok(());
    }
    #[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
    {
        let n_i32 = to_i32(n);
        let nrhs = 1;
        let lda = n_i32;
        let ldb = n_i32;
        let mut ipiv = vec![0i32; n];
        let mut info = 0i32;
        // Safety: `a` has n*n entries and `b` n, both checked above, and the shim
        // is linked in this configuration.
        unsafe {
            c_dgesv(
                &n_i32,
                &nrhs,
                a.as_mut_ptr(),
                &lda,
                ipiv.as_mut_ptr(),
                b.as_mut_ptr(),
                &ldb,
                &mut info,
            );
        }
        if info != 0 {
            return Err(lapack_error("dgesv", info, LapackError::Singular, || {
                println!("LAPACK ERROR (dgesv): U({},{}) is exactly zero", info - 1, info - 1);
            }));
        }
        Ok(())
    }
    #[cfg(not(any(feature = "open_blas", feature = "intel_mkl")))]
    crate::native::lapack::dgesv(a, b)
}

/// `dpotrf`: Cholesky factorization `A = L L^T`, lower triangle, in place.
///
/// Only the lower triangle of `a` is read and only the lower triangle is
/// written, exactly as [`crate::native::lapack::dpotrf`] does. The upper factor
/// is the caller's business -- `mat_cholesky` mirrors and transposes -- because a
/// driver signature that took an `upper` flag would not match the native one, and
/// then the two backends would not be interchangeable.
///
/// # Errors
///
/// [`LapackError::Singular`] if `A` is not positive definite,
/// [`LapackError::BadShape`] on a shape mismatch.
pub fn dpotrf(a: &mut [f64], n: usize) -> Result<(), LapackError> {
    if a.len() != n * n {
        return Err(LapackError::BadShape);
    }
    if n == 0 {
        return Ok(());
    }
    #[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
    {
        let n_i32 = to_i32(n);
        let lda = n_i32;
        let mut info = 0i32;
        // Safety: `a` has n*n entries, checked above, and the shim is linked in
        // this configuration.
        unsafe { c_dpotrf(C_FALSE, &n_i32, a.as_mut_ptr(), &lda, &mut info) };
        if info != 0 {
            return Err(lapack_error("dpotrf", info, LapackError::Singular, || {
                println!(
                    "LAPACK ERROR (dpotrf): The leading minor of order {} is not positive definite",
                    info
                );
            }));
        }
        Ok(())
    }
    #[cfg(not(any(feature = "open_blas", feature = "intel_mkl")))]
    crate::native::lapack::dpotrf(a, n)
}

/// `dsyev`: eigenvalues and eigenvectors of a symmetric matrix, `A V = V diag(w)`.
///
/// `a` is `n x n` column-major and is destroyed; on exit its columns are the
/// eigenvectors. `w` receives the eigenvalues in ascending order. `upper` picks
/// the triangle that is *read*, and the other one is left alone.
///
/// # Errors
///
/// [`LapackError::Convergence`] if `dsteqr` exhausted its iteration limit,
/// [`LapackError::BadShape`] on a shape mismatch.
pub fn dsyev(a: &mut [f64], w: &mut [f64], n: usize, upper: bool) -> Result<(), LapackError> {
    if a.len() != n * n || w.len() != n {
        return Err(LapackError::BadShape);
    }
    if n == 0 {
        return Ok(());
    }
    #[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
    {
        let n_i32 = to_i32(n);
        let lda = n_i32;
        let c_upper = if upper { C_TRUE } else { C_FALSE };
        // `JOBZ = 'V'`: the native routine always computes the vectors, so the
        // dispatch does too -- a eigenvalues-only dsyev would be a different
        // function, not a faster one.
        let mut info = 0i32;
        // Workspace query: `lwork = -1` returns the optimal size in `work[0]`
        // without touching `a` or `w`, which is the only way to size this without
        // hard-coding LAPACK's minimum (3n - 1) and paying for the pessimal
        // blocking that implies.
        let mut work = vec![0.0f64; 1];
        let mut lwork = -1i32;
        // Safety: `a` has n*n entries and `w` n, both checked above; the query
        // writes only `work[0]`.
        unsafe {
            c_dsyev(
                C_TRUE,
                c_upper,
                &n_i32,
                a.as_mut_ptr(),
                &lda,
                w.as_mut_ptr(),
                work.as_mut_ptr(),
                &lwork,
                &mut info,
            );
        }
        if info != 0 {
            return Err(lapack_error("dsyev", info, LapackError::Convergence, || {
                println!(
                    "LAPACK ERROR (dsyev): {} off-diagonal elements of an intermediate tri-diagonal form did not converge to zero",
                    info - 1
                );
            }));
        }
        let need = work[0].max(1.0) as usize;
        let mut work = vec![0.0f64; need];
        lwork = need as i32;
        // Safety: as above, and `work` now holds the size the library asked for.
        unsafe {
            c_dsyev(
                C_TRUE,
                c_upper,
                &n_i32,
                a.as_mut_ptr(),
                &lda,
                w.as_mut_ptr(),
                work.as_mut_ptr(),
                &lwork,
                &mut info,
            );
        }
        if info != 0 {
            return Err(lapack_error("dsyev", info, LapackError::Convergence, || {
                println!(
                    "LAPACK ERROR (dsyev): {} off-diagonal elements of an intermediate tri-diagonal form did not converge to zero",
                    info - 1
                );
            }));
        }
        Ok(())
    }
    #[cfg(not(any(feature = "open_blas", feature = "intel_mkl")))]
    crate::native::lapack::sym::dsyev(a, w, n, upper)
}

/// `dgesvd`: singular value decomposition `A = U S V^T`.
///
/// `a` is `m x n` column-major and is destroyed; `s` has `min(m, n)` entries,
/// `u` is `m x m` and `vt` is `n x n`. Both sets of singular vectors are
/// computed (`JOBU = JOBVT = 'A'`), matching
/// [`crate::native::lapack::svd::dgesvd`], which has no way to ask for less.
///
/// # Errors
///
/// [`LapackError::Convergence`] if `dbdsqr` exhausted its iteration limit,
/// [`LapackError::BadShape`] on a shape mismatch.
pub fn dgesvd(
    a: &mut [f64],
    m: usize,
    n: usize,
    s: &mut [f64],
    u: &mut [f64],
    vt: &mut [f64],
) -> Result<(), LapackError> {
    if a.len() != m * n {
        return Err(LapackError::BadShape);
    }
    if m == 0 || n == 0 {
        return Ok(());
    }
    let minmn = m.min(n);
    if s.len() != minmn || u.len() != m * m || vt.len() != n * n {
        return Err(LapackError::BadShape);
    }
    #[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
    {
        let m_i32 = to_i32(m);
        let n_i32 = to_i32(n);
        let lda = m_i32;
        let ldu = m_i32;
        let ldvt = n_i32;
        let mut info = 0i32;
        // Workspace query first: `lwork = -1` returns the optimal size in
        // `work[0]` and computes nothing.
        let mut work = vec![0.0f64; 1];
        let mut lwork = -1i32;
        // Safety: every buffer was checked against `m` and `n` above; the query
        // writes only `work[0]`.
        unsafe {
            c_dgesvd(
                SVD_CODE_A,
                SVD_CODE_A,
                &m_i32,
                &n_i32,
                a.as_mut_ptr(),
                &lda,
                s.as_mut_ptr(),
                u.as_mut_ptr(),
                &ldu,
                vt.as_mut_ptr(),
                &ldvt,
                work.as_mut_ptr(),
                &lwork,
                &mut info,
            );
        }
        if info != 0 {
            return Err(lapack_error("dgesvd", info, LapackError::Convergence, || {
                println!(
                    "LAPACK ERROR (dgesvd): {} is the number of super-diagonals of an intermediate bi-diagonal form B which did not converge to zero",
                    info
                );
            }));
        }
        let need = work[0].max(1.0) as usize;
        let mut work = vec![0.0f64; need];
        lwork = need as i32;
        // Safety: as above, and `work` now holds the size the library asked for.
        unsafe {
            c_dgesvd(
                SVD_CODE_A,
                SVD_CODE_A,
                &m_i32,
                &n_i32,
                a.as_mut_ptr(),
                &lda,
                s.as_mut_ptr(),
                u.as_mut_ptr(),
                &ldu,
                vt.as_mut_ptr(),
                &ldvt,
                work.as_mut_ptr(),
                &lwork,
                &mut info,
            );
        }
        if info != 0 {
            return Err(lapack_error("dgesvd", info, LapackError::Convergence, || {
                println!(
                    "LAPACK ERROR (dgesvd): {} is the number of super-diagonals of an intermediate bi-diagonal form B which did not converge to zero",
                    info
                );
            }));
        }
        Ok(())
    }
    #[cfg(not(any(feature = "open_blas", feature = "intel_mkl")))]
    crate::native::lapack::svd::dgesvd(a, m, n, s, u, vt)
}

/// `dgeev` with the right eigenvectors: `A v = lambda v`.
///
/// `a` is `n x n` column-major and is destroyed. Returns the eigenvalues as a
/// `(real, imaginary)` pair of vectors, and writes the eigenvectors into
/// `v_real`/`v_imag` as `n x n` column-major matrices, already expanded out of
/// LAPACK's compact storage: a complex conjugate pair occupies two columns, with
/// the second the conjugate of the first. That expansion is
/// [`crate::internal::dgeev_data`], which the accelerated path applies to
/// LAPACK's output and which the native routine performs on its own, so callers
/// see one convention from either backend.
///
/// `JOBVL = 'N'`: the left eigenvectors are [`dgeev_lr`].
///
/// # Panics
///
/// Panics if the shapes are inconsistent, as the native routine does.
///
/// # Errors
///
/// [`LapackError::Convergence`] if `dlahqr` exhausted its iteration limit.
pub fn dgeev(
    v_real: &mut [f64],
    v_imag: &mut [f64],
    a: &mut [f64],
    n: usize,
) -> Result<(Vec<f64>, Vec<f64>), LapackError> {
    assert_eq!(a.len(), n * n, "dgeev: a must be n*n");
    assert_eq!(v_real.len(), n * n, "dgeev: v_real must be n*n");
    assert_eq!(v_imag.len(), n * n, "dgeev: v_imag must be n*n");
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    #[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
    {
        let n_i32 = to_i32(n);
        let lda = n_i32;
        let ldvl = 1i32;
        let ldvr = n_i32;
        let mut wr = vec![0.0f64; n];
        let mut wi = vec![0.0f64; n];
        let mut vl = vec![0.0f64; 1];
        // LAPACK's own storage: column j is the real part of v(j) and column j+1
        // the imaginary part, for a conjugate pair.
        let mut vr = vec![0.0f64; n * n];
        let mut info = 0i32;
        let mut work = vec![0.0f64; 1];
        let mut lwork = -1i32;
        // Safety: `a` has n*n entries and the outputs are sized above; the query
        // writes only `work[0]`.
        unsafe {
            c_dgeev(
                C_FALSE,
                C_TRUE,
                &n_i32,
                a.as_mut_ptr(),
                &lda,
                wr.as_mut_ptr(),
                wi.as_mut_ptr(),
                vl.as_mut_ptr(),
                &ldvl,
                vr.as_mut_ptr(),
                &ldvr,
                work.as_mut_ptr(),
                &lwork,
                &mut info,
            );
        }
        if info != 0 {
            return Err(lapack_error("dgeev", info, LapackError::Convergence, || {
                println!(
                    "LAPACK ERROR (dgeev): The QR algorithm failed. Elements {}+1:N of l_real and l_imag contain eigenvalues which have converged",
                    info - 1
                );
            }));
        }
        let need = work[0].max(1.0) as usize;
        let mut work = vec![0.0f64; need];
        lwork = need as i32;
        // Safety: as above, and `work` now holds the size the library asked for.
        unsafe {
            c_dgeev(
                C_FALSE,
                C_TRUE,
                &n_i32,
                a.as_mut_ptr(),
                &lda,
                wr.as_mut_ptr(),
                wi.as_mut_ptr(),
                vl.as_mut_ptr(),
                &ldvl,
                vr.as_mut_ptr(),
                &ldvr,
                work.as_mut_ptr(),
                &lwork,
                &mut info,
            );
        }
        if info != 0 {
            return Err(lapack_error("dgeev", info, LapackError::Convergence, || {
                println!(
                    "LAPACK ERROR (dgeev): The QR algorithm failed. Elements {}+1:N of l_real and l_imag contain eigenvalues which have converged",
                    info - 1
                );
            }));
        }
        // Unreachable in practice: the shapes were asserted above and LAPACK
        // returns conjugate pairs in order, so `dgeev_data`'s only failures are a
        // length mismatch or a trailing eigenvalue with a nonzero imaginary part.
        dgeev_data(v_real, v_imag, &wi, &vr).map_err(|_| LapackError::BadShape)?;
        Ok((wr, wi))
    }
    #[cfg(not(any(feature = "open_blas", feature = "intel_mkl")))]
    crate::native::lapack::eig::dgeev(v_real, v_imag, a, n)
}

/// `dgeev` with both sets of eigenvectors: `A v = lambda v` and `u^H A = lambda u^H`.
///
/// As [`dgeev`], with the left eigenvectors in `u_real`/`u_imag`
/// (see [`crate::native::lapack::eig::dgeev_lr`]).
///
/// # Panics
///
/// Panics if the shapes are inconsistent, as the native routine does.
///
/// # Errors
///
/// [`LapackError::Convergence`] if `dlahqr` exhausted its iteration limit.
pub fn dgeev_lr(
    u_real: &mut [f64],
    u_imag: &mut [f64],
    v_real: &mut [f64],
    v_imag: &mut [f64],
    a: &mut [f64],
    n: usize,
) -> Result<(Vec<f64>, Vec<f64>), LapackError> {
    assert_eq!(a.len(), n * n, "dgeev_lr: a must be n*n");
    assert_eq!(u_real.len(), n * n, "dgeev_lr: u_real must be n*n");
    assert_eq!(u_imag.len(), n * n, "dgeev_lr: u_imag must be n*n");
    assert_eq!(v_real.len(), n * n, "dgeev_lr: v_real must be n*n");
    assert_eq!(v_imag.len(), n * n, "dgeev_lr: v_imag must be n*n");
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    #[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
    {
        let n_i32 = to_i32(n);
        let lda = n_i32;
        let ldvl = n_i32;
        let ldvr = n_i32;
        let mut wr = vec![0.0f64; n];
        let mut wi = vec![0.0f64; n];
        let mut vl = vec![0.0f64; n * n];
        let mut vr = vec![0.0f64; n * n];
        let mut info = 0i32;
        let mut work = vec![0.0f64; 1];
        let mut lwork = -1i32;
        // Safety: every buffer is n*n or n, sized above; the query writes only
        // `work[0]`.
        unsafe {
            c_dgeev(
                C_TRUE,
                C_TRUE,
                &n_i32,
                a.as_mut_ptr(),
                &lda,
                wr.as_mut_ptr(),
                wi.as_mut_ptr(),
                vl.as_mut_ptr(),
                &ldvl,
                vr.as_mut_ptr(),
                &ldvr,
                work.as_mut_ptr(),
                &lwork,
                &mut info,
            );
        }
        if info != 0 {
            return Err(lapack_error("dgeev", info, LapackError::Convergence, || {
                println!(
                    "LAPACK ERROR (dgeev): The QR algorithm failed. Elements {}+1:N of l_real and l_imag contain eigenvalues which have converged",
                    info - 1
                );
            }));
        }
        let need = work[0].max(1.0) as usize;
        let mut work = vec![0.0f64; need];
        lwork = need as i32;
        // Safety: as above, and `work` now holds the size the library asked for.
        unsafe {
            c_dgeev(
                C_TRUE,
                C_TRUE,
                &n_i32,
                a.as_mut_ptr(),
                &lda,
                wr.as_mut_ptr(),
                wi.as_mut_ptr(),
                vl.as_mut_ptr(),
                &ldvl,
                vr.as_mut_ptr(),
                &ldvr,
                work.as_mut_ptr(),
                &lwork,
                &mut info,
            );
        }
        if info != 0 {
            return Err(lapack_error("dgeev", info, LapackError::Convergence, || {
                println!(
                    "LAPACK ERROR (dgeev): The QR algorithm failed. Elements {}+1:N of l_real and l_imag contain eigenvalues which have converged",
                    info - 1
                );
            }));
        }
        // Unreachable in practice, for the reason given in `dgeev`.
        dgeev_data_lr(u_real, u_imag, v_real, v_imag, &wi, &vl, &vr).map_err(|_| LapackError::BadShape)?;
        Ok((wr, wi))
    }
    #[cfg(not(any(feature = "open_blas", feature = "intel_mkl")))]
    crate::native::lapack::eig::dgeev_lr(u_real, u_imag, v_real, v_imag, a, n)
}

/// `dgeev` with no eigenvectors: the eigenvalues only (`JOBVL = JOBVR = 'N'`).
///
/// `wr` and `wi` receive the real and imaginary parts, `a` is `n x n` and is
/// destroyed. This exists as its own entry because skipping `dtrevc3` is most of
/// what makes it cheap, and the native path does the same -- [`crate::mat_eigenvalues`]
/// discards the vectors it is given.
///
/// # Panics
///
/// Panics if the shapes are inconsistent.
///
/// # Errors
///
/// [`LapackError::Convergence`] if `dlahqr` exhausted its iteration limit.
pub fn dgeev_eigenvalues(wr: &mut [f64], wi: &mut [f64], a: &mut [f64], n: usize) -> Result<(), LapackError> {
    assert_eq!(a.len(), n * n, "dgeev_eigenvalues: a must be n*n");
    assert_eq!(wr.len(), n, "dgeev_eigenvalues: wr must have n entries");
    assert_eq!(wi.len(), n, "dgeev_eigenvalues: wi must have n entries");
    if n == 0 {
        return Ok(());
    }
    #[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
    {
        let n_i32 = to_i32(n);
        let lda = n_i32;
        let ldvl = 1i32;
        let ldvr = 1i32;
        // LAPACK requires a valid pointer even when the vectors are not wanted;
        // one element and `ld* = 1` is what it documents.
        let mut vl = vec![0.0f64; 1];
        let mut vr = vec![0.0f64; 1];
        let mut info = 0i32;
        let mut work = vec![0.0f64; 1];
        let mut lwork = -1i32;
        // Safety: `a` has n*n entries and `wr`/`wi` n; the query writes only
        // `work[0]`.
        unsafe {
            c_dgeev(
                C_FALSE,
                C_FALSE,
                &n_i32,
                a.as_mut_ptr(),
                &lda,
                wr.as_mut_ptr(),
                wi.as_mut_ptr(),
                vl.as_mut_ptr(),
                &ldvl,
                vr.as_mut_ptr(),
                &ldvr,
                work.as_mut_ptr(),
                &lwork,
                &mut info,
            );
        }
        if info != 0 {
            return Err(lapack_error("dgeev", info, LapackError::Convergence, || {
                println!(
                    "LAPACK ERROR (dgeev): The QR algorithm failed. Elements {}+1:N of l_real and l_imag contain eigenvalues which have converged",
                    info - 1
                );
            }));
        }
        let need = work[0].max(1.0) as usize;
        let mut work = vec![0.0f64; need];
        lwork = need as i32;
        // Safety: as above, and `work` now holds the size the library asked for.
        unsafe {
            c_dgeev(
                C_FALSE,
                C_FALSE,
                &n_i32,
                a.as_mut_ptr(),
                &lda,
                wr.as_mut_ptr(),
                wi.as_mut_ptr(),
                vl.as_mut_ptr(),
                &ldvl,
                vr.as_mut_ptr(),
                &ldvr,
                work.as_mut_ptr(),
                &lwork,
                &mut info,
            );
        }
        if info != 0 {
            return Err(lapack_error("dgeev", info, LapackError::Convergence, || {
                println!(
                    "LAPACK ERROR (dgeev): The QR algorithm failed. Elements {}+1:N of l_real and l_imag contain eigenvalues which have converged",
                    info - 1
                );
            }));
        }
        Ok(())
    }
    #[cfg(not(any(feature = "open_blas", feature = "intel_mkl")))]
    {
        let mut v_real = vec![0.0; n * n];
        let mut v_imag = vec![0.0; n * n];
        let (wr_native, wi_native) = crate::native::lapack::eig::dgeev(&mut v_real, &mut v_imag, a, n)?;
        wr.copy_from_slice(&wr_native);
        wi.copy_from_slice(&wi_native);
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    /// A small, well-conditioned, non-symmetric matrix, so no symmetry can hide a
    /// transposed index.
    fn sample4() -> Vec<f64> {
        // Row-major [[4,1,2,1],[1,3,0,1],[2,0,5,2],[1,1,2,6]] as column-major.
        vec![
            4.0, 1.0, 2.0, 1.0, 1.0, 3.0, 0.0, 1.0, 2.0, 0.0, 5.0, 2.0, 1.0, 1.0, 2.0, 6.0,
        ]
    }

    /// The same matrix multiplied by a vector, through the BLAS dispatch, so the
    /// tests do not depend on a second implementation of the product.
    fn mat_vec(a: &[f64], x: &[f64], n: usize) -> Vec<f64> {
        let mut y = vec![0.0; n];
        crate::base::blas_dispatch::dgemv(1.0, a, x, 0.0, &mut y, n, n);
        y
    }

    #[test]
    fn dgesv_matches_the_native_driver() {
        let a = sample4();
        let x_true = [1.0, -2.0, 3.0, 5.0];
        let b = mat_vec(&a, &x_true, 4);

        let mut a_got = a.clone();
        let mut b_got = b.clone();
        dgesv(&mut a_got, &mut b_got).unwrap();

        let mut a_want = a.clone();
        let mut b_want = b.clone();
        crate::native::lapack::dgesv(&mut a_want, &mut b_want).unwrap();

        for i in 0..4 {
            assert!((b_got[i] - x_true[i]).abs() < 1e-12, "x[{i}] = {}", b_got[i]);
            assert!(
                (b_got[i] - b_want[i]).abs() < 1e-12,
                "x[{i}]: {} vs {}",
                b_got[i],
                b_want[i]
            );
        }
    }

    /// The error must be the one the native path reports, not merely an error:
    /// `solve_lin_sys` matches on the variant to choose its message.
    #[test]
    fn dgesv_reports_the_same_errors_as_the_native_driver() {
        // The second column is twice the first, so the matrix is singular. Each
        // call gets a fresh copy: `dgesv` destroys `a` and `b`.
        //
        // The first column is all zeros rather than proportional to another one,
        // so the zero pivot is exact under any arithmetic: a merely
        // *nearly* singular matrix is detected by the native path's
        // `ZERO_DETERMINANT` threshold and not by LAPACK, which tests for exactly
        // zero (see the note on `lapack_dispatch`).
        let singular = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0];
        let b0 = vec![1.0, 1.0, 1.0];
        let mut a = singular.clone();
        let mut b = b0.clone();
        let got = dgesv(&mut a, &mut b);
        assert_eq!(got, Err(LapackError::Singular));
        let mut a = singular.clone();
        let mut b = b0.clone();
        assert_eq!(got, crate::native::lapack::dgesv(&mut a, &mut b));

        let mut a = vec![1.0, 2.0, 3.0];
        assert_eq!(dgesv(&mut a, &mut [1.0, 2.0]), Err(LapackError::BadShape));

        let mut a: Vec<f64> = Vec::new();
        assert_eq!(dgesv(&mut a, &mut Vec::new()), Ok(()));
    }

    #[test]
    fn dpotrf_matches_the_native_driver() {
        // A = [[4,2],[2,3]] is symmetric positive definite.
        let mut got = vec![4.0, 2.0, 2.0, 3.0];
        dpotrf(&mut got, 2).unwrap();
        let mut want = vec![4.0, 2.0, 2.0, 3.0];
        crate::native::lapack::dpotrf(&mut want, 2).unwrap();
        for i in 0..4 {
            assert!(
                (got[i] - want[i]).abs() < 1e-14,
                "dpotrf [{i}]: {} vs {}",
                got[i],
                want[i]
            );
        }
        // The upper triangle keeps what it held, as LAPACK specifies.
        assert_eq!(got[2], 2.0);

        // A larger one, so the blocking differs: 20 x 20 diagonally dominant.
        let n = 20;
        let mut a = vec![0.0; n * n];
        for j in 0..n {
            for i in 0..n {
                a[i + j * n] = if i == j { 4.0 } else { 0.5 / ((i + j + 1) as f64) };
            }
        }
        let (mut got, mut want) = (a.clone(), a.clone());
        dpotrf(&mut got, n).unwrap();
        crate::native::lapack::dpotrf(&mut want, n).unwrap();
        for i in 0..n * n {
            assert!((got[i] - want[i]).abs() < 1e-13, "dpotrf n={n} [{i}]");
        }
    }

    #[test]
    fn dpotrf_reports_the_same_errors_as_the_native_driver() {
        // [[1,2],[2,1]] has a negative eigenvalue.
        let mut a = vec![1.0, 2.0, 2.0, 1.0];
        assert_eq!(dpotrf(&mut a, 2), Err(LapackError::Singular));
        let mut b = vec![1.0, 2.0, 2.0, 1.0];
        assert_eq!(
            dpotrf(&mut b, 2),
            crate::native::lapack::dpotrf(&mut [1.0, 2.0, 2.0, 1.0], 2)
        );

        let mut a = vec![1.0, 2.0, 3.0];
        assert_eq!(dpotrf(&mut a, 2), Err(LapackError::BadShape));
        assert_eq!(dpotrf(&mut Vec::new(), 0), Ok(()));
    }

    #[test]
    fn dsyev_matches_the_native_eigenvalues() {
        let n = 6;
        // Symmetric but not diagonal, and not tridiagonal either.
        let mut a = vec![0.0; n * n];
        for j in 0..n {
            for i in 0..n {
                a[i + j * n] = if i == j {
                    2.0 + (i as f64)
                } else {
                    (((i * 3 + j * 5) % 7) as f64) * 0.25
                };
            }
        }
        // Symmetrize: the routine reads one triangle, the other may be anything.
        for j in 0..n {
            for i in 0..j {
                a[i + j * n] = a[j + i * n];
            }
        }

        for upper in [false, true] {
            let mut a_got = a.clone();
            let mut w_got = vec![0.0; n];
            dsyev(&mut a_got, &mut w_got, n, upper).unwrap();
            let mut a_want = a.clone();
            let mut w_want = vec![0.0; n];
            crate::native::lapack::sym::dsyev(&mut a_want, &mut w_want, n, upper).unwrap();
            for i in 0..n {
                assert!(
                    (w_got[i] - w_want[i]).abs() < 1e-11,
                    "dsyev upper={upper} w[{i}]: {} vs {}",
                    w_got[i],
                    w_want[i]
                );
            }
            // Ascending order is part of the contract.
            assert!(w_got.windows(2).all(|p| p[0] <= p[1]), "dsyev: w must ascend");
            // And the decomposition must reconstruct A, which the eigenvalues
            // alone would not catch (a sign-flipped or transposed V would).
            for j in 0..n {
                for i in 0..n {
                    let mut s = 0.0;
                    for k in 0..n {
                        s += a_got[i + k * n] * w_got[k] * a_got[j + k * n];
                    }
                    assert!((s - a[i + j * n]).abs() < 1e-11, "dsyev upper={upper} A({i},{j})");
                }
            }
        }
    }

    #[test]
    fn dsyev_handles_a_bad_shape_and_the_degenerate_sizes() {
        let mut a = vec![2.0, 0.0, 0.0, 3.0];
        let mut w = vec![0.0];
        assert_eq!(dsyev(&mut a, &mut w, 2, false), Err(LapackError::BadShape));
        assert_eq!(dsyev(&mut Vec::new(), &mut Vec::new(), 0, false), Ok(()));
        // n = 1 is the case where LAPACK and the native routine each special-case
        // the answer: w[0] = a[0] and the single eigenvector is 1.
        let mut a = vec![7.0];
        let mut w = vec![0.0];
        dsyev(&mut a, &mut w, 1, false).unwrap();
        assert_eq!(w[0], 7.0);
        assert_eq!(a[0], 1.0);
    }

    #[test]
    fn dgesvd_matches_the_native_singular_values() {
        // Tall and wide, because the two branches of the native routine differ
        // (m >= n bidiagonalizes directly, m < n transposes the problem).
        for (m, n) in [(7usize, 4usize), (4usize, 7usize)] {
            let a: Vec<f64> = (0..m * n).map(|i| ((i * 7 % 19) as f64) - 9.0).collect();
            let minmn = m.min(n);
            let mut a_got = a.clone();
            let mut s_got = vec![0.0; minmn];
            let mut u_got = vec![0.0; m * m];
            let mut vt_got = vec![0.0; n * n];
            dgesvd(&mut a_got, m, n, &mut s_got, &mut u_got, &mut vt_got).unwrap();

            let mut a_want = a.clone();
            let mut s_want = vec![0.0; minmn];
            let mut u_want = vec![0.0; m * m];
            let mut vt_want = vec![0.0; n * n];
            crate::native::lapack::svd::dgesvd(&mut a_want, m, n, &mut s_want, &mut u_want, &mut vt_want).unwrap();

            for i in 0..minmn {
                assert!(
                    (s_got[i] - s_want[i]).abs() < 1e-11,
                    "dgesvd {m}x{n} s[{i}]: {} vs {}",
                    s_got[i],
                    s_want[i]
                );
            }
            // Descending order and non-negativity are part of the contract.
            assert!(s_got.windows(2).all(|p| p[0] >= p[1]), "dgesvd: s must descend");
            assert!(s_got.iter().all(|s| *s >= 0.0), "dgesvd: s must be non-negative");
            // And A = U S V^T must hold, which the values alone do not check.
            for j in 0..n {
                for i in 0..m {
                    let mut sum = 0.0;
                    for k in 0..minmn {
                        sum += u_got[i + k * m] * s_got[k] * vt_got[k + j * n];
                    }
                    assert!((sum - a[i + j * m]).abs() < 1e-10, "dgesvd {m}x{n} A({i},{j})");
                }
            }
        }
    }

    #[test]
    fn dgesvd_rejects_a_bad_shape() {
        let mut a = vec![0.0; 6];
        let mut s = vec![0.0; 2];
        let mut u = vec![0.0; 4];
        let mut vt = vec![0.0; 6];
        // `a` is checked first, as in the native routine: 6 entries is not 2*2.
        assert_eq!(
            dgesvd(&mut a, 2, 2, &mut s, &mut u, &mut vt),
            Err(LapackError::BadShape)
        );
        // A zero dimension is not an error, but nothing is written either.
        assert_eq!(dgesvd(&mut Vec::new(), 0, 3, &mut s, &mut u, &mut vt), Ok(()));
        let mut wrong_s = vec![0.0; 3];
        let mut a = vec![0.0; 6];
        assert_eq!(
            dgesvd(&mut a, 2, 3, &mut wrong_s, &mut u, &mut vt),
            Err(LapackError::BadShape)
        );
    }

    /// The eigenvalues, sorted, because the order of a Schur form is not part of
    /// either implementation's contract.
    fn sorted(values: &[(f64, f64)]) -> Vec<(f64, f64)> {
        let mut out = values.to_vec();
        out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.partial_cmp(&b.1).unwrap()));
        out
    }

    #[test]
    fn dgeev_matches_the_native_eigenvalues_and_reconstructs_them() {
        // A rotation-like block plus a real diagonal: one conjugate pair, so the
        // packed storage on the LAPACK side is exercised.
        #[rustfmt::skip]
        let a = vec![
            0.0, 1.0, 0.0, 0.0,
            -1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 3.0, 0.0,
            0.0, 0.0, 0.0, 5.0,
        ];
        let n = 4;
        let mut a_got = a.clone();
        let (mut v_real, mut v_imag) = (vec![0.0; n * n], vec![0.0; n * n]);
        let (wr_got, wi_got) = dgeev(&mut v_real, &mut v_imag, &mut a_got, n).unwrap();

        let got: Vec<(f64, f64)> = (0..n).map(|i| (wr_got[i], wi_got[i])).collect();
        let sorted_got = sorted(&got);
        let expected = sorted(&[(0.0, 1.0), (0.0, -1.0), (3.0, 0.0), (5.0, 0.0)]);
        for i in 0..n {
            assert!(
                (sorted_got[i].0 - expected[i].0).abs() < 1e-12 && (sorted_got[i].1 - expected[i].1).abs() < 1e-12,
                "dgeev l[{i}] = {:?} but expected {:?}",
                sorted_got[i],
                expected[i]
            );
        }

        // Each eigenvector satisfies A v = lambda v, with the conjugate pair
        // expanded into columns 0 and 1 by the dispatch (a packed-layout mistake
        // would leave column 1 equal to column 0 and break this).
        let orig = a.clone();
        for j in 0..n {
            let (re, im) = got[j];
            // Column j holds the real part; for a complex pair, column j+1 holds
            // the imaginary part and the next eigenvalue is its conjugate.
            let re_of = |i: usize| v_real[i + j * n];
            let im_of = |i: usize| v_imag[i + j * n];
            if im == 0.0 {
                let v: Vec<f64> = (0..n).map(re_of).collect();
                let av = mat_vec(&orig, &v, n);
                for i in 0..n {
                    assert!((av[i] - re * v[i]).abs() < 1e-12, "dgeev real column {j} row {i}");
                }
            } else if im > 0.0 {
                let vr: Vec<f64> = (0..n).map(re_of).collect();
                let vi: Vec<f64> = (0..n).map(im_of).collect();
                let avr = mat_vec(&orig, &vr, n);
                let avi = mat_vec(&orig, &vi, n);
                for i in 0..n {
                    assert!(
                        (avr[i] - (re * vr[i] - im * vi[i])).abs() < 1e-12,
                        "dgeev pair {j} re {i}"
                    );
                    assert!(
                        (avi[i] - (re * vi[i] + im * vr[i])).abs() < 1e-12,
                        "dgeev pair {j} im {i}"
                    );
                }
            }
        }
    }

    #[test]
    fn dgeev_lr_and_eigenvalues_agree_with_each_other() {
        let n = 5;
        let a: Vec<f64> = (0..n * n)
            .map(|i| {
                let (r, c) = (i % n, i / n);
                if r == c {
                    1.0 + (r as f64)
                } else {
                    (((r * 3 + c * 7) % 5) as f64) * 0.1
                }
            })
            .collect();

        let mut a_lr = a.clone();
        let (mut ur, mut ui) = (vec![0.0; n * n], vec![0.0; n * n]);
        let (mut vr, mut vi) = (vec![0.0; n * n], vec![0.0; n * n]);
        let (wr_lr, wi_lr) = dgeev_lr(&mut ur, &mut ui, &mut vr, &mut vi, &mut a_lr, n).unwrap();

        let mut a_ev = a.clone();
        let (mut wr_ev, mut wi_ev) = (vec![0.0; n], vec![0.0; n]);
        dgeev_eigenvalues(&mut wr_ev, &mut wi_ev, &mut a_ev, n).unwrap();

        for i in 0..n {
            assert!((wr_lr[i] - wr_ev[i]).abs() < 1e-12, "l_real[{i}]");
            assert!((wi_lr[i] - wi_ev[i]).abs() < 1e-12, "l_imag[{i}]");
        }

        // The left eigenvectors satisfy u^H A = lambda u^H, i.e. A^T conj(u) =
        // conj(lambda) conj(u).
        let at: Vec<f64> = (0..n * n).map(|i| a[(i % n) * n + i / n]).collect();
        for j in 0..n {
            let (re, im) = (wr_lr[j], wi_lr[j]);
            if im != 0.0 {
                continue; // covered by the pair below, and only the positive half is stored
            }
            let u: Vec<f64> = (0..n).map(|i| ur[i + j * n]).collect();
            let atu = mat_vec(&at, &u, n);
            for i in 0..n {
                assert!((atu[i] - re * u[i]).abs() < 1e-11, "dgeev_lr real column {j} row {i}");
            }
        }
    }
}
