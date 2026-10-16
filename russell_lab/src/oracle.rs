//! The reference LAPACK, for testing the native routines against.
//!
//! **Development only.** Enabled by the `oracle` feature, which compiles
//! `c_code/interface_blas.c` against a locally built LAPACK and links it. Not for
//! release: it needs a LAPACK build on the machine (see `build.rs`, and
//! `RUSSELL_ORACLE_ROOT` to point it elsewhere).
//!
//! # Declaration strategy: prefer the shim's wrappers
//!
//! The shim's `c_*` functions take `C_BOOL` instead of Fortran character
//! arguments and supply the hidden string length themselves. Declaring those is
//! much less error-prone than reproducing LAPACK's string ABI by hand, so this
//! module uses them wherever they exist and only reaches for a raw `LAPACK_*`
//! symbol when it does not.
//!
//! # Column-major
//!
//! Everything here is column-major, matching [`crate::Matrix`]. That is because
//! `lapack.h` binds directly to the Fortran routines. **LAPACKE would not work
//! here**: its `LAPACKE_*` entry points are row-major and would transpose every
//! matrix, so a comparison against them would be meaningless.
//!
//! # Who calls this
//!
//! Only `native::parity`, and only under `#[cfg(all(test, feature = "oracle"))]`.

#![allow(dead_code)]

use crate::Complex64;
use crate::internal::{C_TRUE, CcBool};

unsafe extern "C" {
    /// `dgesv`: solve `A X = B`. Takes `nrhs` right-hand sides; the shim's
    /// signature is the general one even though `russell_lab` only uses one.
    pub fn oracle_c_dgesv(
        n: *const i32,
        nrhs: *const i32,
        a: *mut f64,
        lda: *const i32,
        ipiv: *mut i32,
        b: *mut f64,
        ldb: *const i32,
        info: *mut i32,
    );

    /// `dgetrf`: LU factorisation with partial pivoting, `m x n`.
    pub fn oracle_c_dgetrf(m: *const i32, n: *const i32, a: *mut f64, lda: *const i32, ipiv: *mut i32, info: *mut i32);

    /// `dgetri`: inverse from a `dgetrf` factorisation.
    pub fn oracle_c_dgetri(
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        ipiv: *const i32,
        work: *mut f64,
        lwork: *const i32,
        info: *mut i32,
    );

    /// `dpotrf`: Cholesky factorisation. `upper` selects the triangle to write.
    pub fn oracle_c_dpotrf(upper: CcBool, n: *const i32, a: *mut f64, lda: *const i32, info: *mut i32);

    /// `zgetrf`: LU factorisation of a complex matrix.
    ///
    /// `COMPLEX64` in the C shim is `lapack_complex_double`, which is two
    /// `double`s in sequence, so `Complex64` (which is `#[repr(C)] { re, im }`)
    /// is layout-compatible and can be passed directly.
    pub fn oracle_c_zgetrf(
        m: *const i32,
        n: *const i32,
        a: *mut Complex64,
        lda: *const i32,
        ipiv: *mut i32,
        info: *mut i32,
    );

    /// `zgetri`: complex inverse from a `zgetrf` factorisation.
    pub fn oracle_c_zgetri(
        n: *const i32,
        a: *mut Complex64,
        lda: *const i32,
        ipiv: *const i32,
        work: *mut Complex64,
        lwork: *const i32,
        info: *mut i32,
    );

    /// `zgesv`: complex solve `A X = B`.
    pub fn oracle_c_zgesv(
        n: *const i32,
        nrhs: *const i32,
        a: *mut Complex64,
        lda: *const i32,
        ipiv: *mut i32,
        b: *mut Complex64,
        ldb: *const i32,
        info: *mut i32,
    );

    /// `zpotrf`: Hermitian positive-definite Cholesky.
    pub fn oracle_c_zpotrf(upper: CcBool, n: *const i32, a: *mut Complex64, lda: *const i32, info: *mut i32);

    /// `zheev`: eigenvalues (and optionally eigenvectors) of a Hermitian matrix.
    pub fn oracle_c_zheev(
        calc_v: CcBool,
        upper: CcBool,
        n: *const i32,
        a: *mut Complex64,
        lda: *const i32,
        w: *mut f64,
        work: *mut Complex64,
        lwork: *const i32,
        rwork: *mut f64,
        info: *mut i32,
    );

    /// `dsyev`: eigenvalues (and optionally eigenvectors) of a symmetric matrix.
    pub fn oracle_c_dsyev(
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

    /// `dgeev`: eigenvalues and eigenvectors of a general matrix.
    ///
    /// Eigenvalues come back as a real/imaginary pair because a real matrix can
    /// have complex conjugate pairs. Real eigenvectors are not returned for a
    /// complex pair, and `wi` must be inspected to know which case applies.
    pub fn oracle_c_dgeev(
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

    /// `dgesvd`: singular value decomposition, `m x n`.
    pub fn oracle_c_dgesvd(
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
}

/// The `jobu`/`jobvt` codes the shim's `c_dgesvd` expects, mirroring the Fortran
/// characters `A`, `S`, `O` and `N`.
pub mod svd_code {
    pub const ALL: i32 = 0; // 'A'
    pub const SKINNY: i32 = 1; // 'S'
    pub const OVERWRITE: i32 = 2; // 'O'
    pub const NONE: i32 = 3; // 'N'
}

/// Convenience wrappers that handle the workspace query, so the tests do not each
/// repeat it.
///
/// LAPACK's convention is to call with `lwork = -1` to be told the workspace size
/// in `work[0]`. Getting that wrong is a silent buffer overrun, so it is done once
/// here.
pub mod query {
    use super::*;
    // `use super::*` brings in this module's own items, not its imports, so the
    // constants have to be named again here.
    use crate::internal::{C_FALSE, C_TRUE};

    /// Runs `dgeev` with a correctly sized workspace.
    ///
    /// `a` is `n x n` column-major and is destroyed. `wr`/`wi` receive the
    /// eigenvalues, `vr` the right eigenvectors (if `want_v`).
    #[allow(clippy::too_many_arguments)]
    pub fn dgeev(a: &mut [f64], n: usize, wr: &mut [f64], wi: &mut [f64], vr: Option<&mut [f64]>) -> i32 {
        let n_i32 = n as i32;
        let ld = n_i32;
        let mut info = 0i32;
        let (want_vr, ldvr) = match &vr {
            Some(v) => (C_TRUE, n_i32),
            None => (C_FALSE, 1),
        };
        let vr_ptr = match vr {
            Some(v) => v.as_mut_ptr(),
            None => std::ptr::null_mut(),
        };

        // Workspace query.
        let mut work = vec![0.0f64; 1];
        let mut lwork = -1i32;
        unsafe {
            oracle_c_dgeev(
                C_FALSE,
                want_vr,
                &n_i32,
                a.as_mut_ptr(),
                &ld,
                wr.as_mut_ptr(),
                wi.as_mut_ptr(),
                std::ptr::null_mut(),
                &1i32,
                vr_ptr,
                &ldvr,
                work.as_mut_ptr(),
                &lwork,
                &mut info,
            );
        }
        if info != 0 {
            return info;
        }
        let need = work[0].max(1.0) as usize;
        let mut work = vec![0.0f64; need];
        lwork = need as i32;
        unsafe {
            oracle_c_dgeev(
                C_FALSE,
                want_vr,
                &n_i32,
                a.as_mut_ptr(),
                &ld,
                wr.as_mut_ptr(),
                wi.as_mut_ptr(),
                std::ptr::null_mut(),
                &1i32,
                vr_ptr,
                &ldvr,
                work.as_mut_ptr(),
                &lwork,
                &mut info,
            );
        }
        info
    }

    /// Runs `dsyev` in the lower triangle, with a correctly sized workspace.
    pub fn dsyev(a: &mut [f64], n: usize, w: &mut [f64], want_v: bool) -> i32 {
        let n_i32 = n as i32;
        let ld = n_i32;
        let mut info = 0i32;
        let jobz = if want_v { C_TRUE } else { C_FALSE };

        let mut work = vec![0.0f64; 1];
        let mut lwork = -1i32;
        unsafe {
            oracle_c_dsyev(
                jobz,
                C_FALSE,
                &n_i32,
                a.as_mut_ptr(),
                &ld,
                w.as_mut_ptr(),
                work.as_mut_ptr(),
                &lwork,
                &mut info,
            );
        }
        if info != 0 {
            return info;
        }
        let need = work[0].max(1.0) as usize;
        let mut work = vec![0.0f64; need];
        lwork = need as i32;
        unsafe {
            oracle_c_dsyev(
                jobz,
                C_FALSE,
                &n_i32,
                a.as_mut_ptr(),
                &ld,
                w.as_mut_ptr(),
                work.as_mut_ptr(),
                &lwork,
                &mut info,
            );
        }
        info
    }

    /// Runs `zheev` in the lower triangle, with a correctly sized workspace.
    pub fn zheev(a: &mut [Complex64], n: usize, w: &mut [f64], want_v: bool) -> i32 {
        let n_i32 = n as i32;
        let ld = n_i32;
        let mut info = 0i32;
        let jobz = if want_v { C_TRUE } else { C_FALSE };

        let mut work = vec![Complex64::new(0.0, 0.0); 1];
        let mut rwork = vec![0.0f64; (3 * n).max(1)];
        let mut lwork = -1i32;
        unsafe {
            oracle_c_zheev(
                jobz,
                C_FALSE,
                &n_i32,
                a.as_mut_ptr(),
                &ld,
                w.as_mut_ptr(),
                work.as_mut_ptr(),
                &lwork,
                rwork.as_mut_ptr(),
                &mut info,
            );
        }
        if info != 0 {
            return info;
        }
        let need = work[0].re.max(1.0) as usize;
        let mut work = vec![Complex64::new(0.0, 0.0); need];
        lwork = need as i32;
        unsafe {
            oracle_c_zheev(
                jobz,
                C_FALSE,
                &n_i32,
                a.as_mut_ptr(),
                &ld,
                w.as_mut_ptr(),
                work.as_mut_ptr(),
                &lwork,
                rwork.as_mut_ptr(),
                &mut info,
            );
        }
        info
    }

    /// Runs `dgeev`-style workspace handling for `dgesvd`, returning `s`.
    #[allow(clippy::too_many_arguments)]
    pub fn dgesvd(a: &mut [f64], m: usize, n: usize, s: &mut [f64], u: &mut [f64], vt: &mut [f64]) -> i32 {
        let (m_i32, n_i32) = (m as i32, n as i32);
        let min_mn = m.min(n);
        let mut info = 0i32;

        let mut work = vec![0.0f64; 1];
        let mut lwork = -1i32;
        unsafe {
            oracle_c_dgesvd(
                svd_code::ALL,
                svd_code::ALL,
                &m_i32,
                &n_i32,
                a.as_mut_ptr(),
                &m_i32,
                s.as_mut_ptr(),
                u.as_mut_ptr(),
                &m_i32,
                vt.as_mut_ptr(),
                &n_i32,
                work.as_mut_ptr(),
                &lwork,
                &mut info,
            );
        }
        if info != 0 {
            return info;
        }
        let need = work[0].max(1.0) as usize;
        let mut work = vec![0.0f64; need];
        lwork = need as i32;
        let _ = min_mn;
        unsafe {
            oracle_c_dgesvd(
                svd_code::ALL,
                svd_code::ALL,
                &m_i32,
                &n_i32,
                a.as_mut_ptr(),
                &m_i32,
                s.as_mut_ptr(),
                u.as_mut_ptr(),
                &m_i32,
                vt.as_mut_ptr(),
                &n_i32,
                work.as_mut_ptr(),
                &lwork,
                &mut info,
            );
        }
        info
    }
}
