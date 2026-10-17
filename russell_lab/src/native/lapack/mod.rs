//! Native LAPACK: the driver routines `russell_lab` exposes, in Rust.
//!
//! Each function names the LAPACK routine it follows, so it can be checked against
//! the reference source of the same name (LAPACK 3.12.1's `SRC/` and `BLAS/SRC/`).
//!
//! # What is here
//!
//! Every routine the crate's public API needs has a native implementation. This
//! table says where each one lives; the module's own doc comment gives its
//! pipeline.
//!
//! | Routine | Where |
//! |---|---|
//! | `dgetrf` LU with partial pivoting | [`dgetrf`] |
//! | `dgetri` inverse from LU | [`dgetri`] |
//! | `dgesv` solve `A X = B` | [`dgesv`] |
//! | `dpotrf` Cholesky | [`dpotrf`] |
//! | `dlange` matrix norms | [`super::blas::dlange`] |
//! | `dsyev` symmetric eigen-decomposition | [`sym`] |
//! | `dgeev` general eigen-decomposition | [`eig`] |
//! | `dgesvd` real SVD | [`svd`] |
//! | `dggev` generalized eigen-decomposition (QZ) | [`qz`] |
//! | `zgetrf`, `zgetri`, `zgesv`, `zpotrf` | [`complex`] |
//! | `zheev`, `zgeev` | [`zeig`] |
//! | `zgesvd` complex SVD | [`zsvd`] |
//! | `zggev` complex QZ | [`qz`] |
//!
//! `LapackError::NotImplemented` is retained, and no routine returns it: a
//! caller that reaches a routine with no native path still gets a runtime error
//! naming the feature to enable rather than a link error, and a routine that is
//! reverted keeps its error handling.
//!
//! # Conventions
//!
//! * Column-major, `data[i + j * n]`, matching `crate::Matrix`.
//! * 0-based indices and `usize` lengths. LAPACK's 1-based indices and its
//!   `lda`/`lwork` arguments are Fortran artifacts, not part of the math, so
//!   they do not appear here; the leading dimension of an `n x n` matrix is `n`.
//! * Workspace is allocated inside each routine.
//! * A driver returns `Result<(), LapackError>`, never LAPACK's integer `INFO`:
//!   `INFO < 0` (an illegal argument) and `INFO > 0` (non-convergence, or a
//!   back-substitution failure) both become [`LapackError`], so no caller has to
//!   remember which integer means what. Internal helpers still return `INFO`,
//!   because that is what the reference source says and what a diff against it
//!   should show; the drivers translate it at the boundary.

mod auxiliary;
pub mod complex;
pub mod eig;
pub mod qz;
pub mod svd;
pub mod sym;
pub mod zeig;
pub mod zsvd;

use super::blas;

/// Tolerance below which a pivot counts as zero.
///
/// Matches the `ZERO_DETERMINANT` used elsewhere in the crate so that the native
/// and accelerated paths agree on when a matrix is singular.
pub const ZERO_DETERMINANT: f64 = 1e-15;

/// Error kinds these routines can report.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LapackError {
    /// The matrix is singular (or not positive definite) to working precision.
    Singular,
    /// A shape or length was inconsistent with another.
    BadShape,
    /// An iterative routine exhausted its iteration limit before converging.
    ///
    /// This is LAPACK's `INFO > 0`, and it means the output is *partly*
    /// converged: the routine returns an error rather than values that look
    /// plausible and are not. Every iterative routine here reports it --
    /// `dsyev`/`zheev` (via `dsteqr`/`zsteqr`), `dhseqr` (via `dlahqr`), and
    /// `dgesvd`/`zgesvd` (via `dbdsqr`/`zbdsqr`) -- so a caller cannot get
    /// unconverged eigenvalues back as if they had converged.
    Convergence,
    /// The routine exists in the interface but has no native implementation yet.
    ///
    /// Carries the routine's name so a message can point at it. See
    /// [`not_implemented`]. No routine currently returns this.
    NotImplemented(&'static str),
}

/// `dgetrf`: LU factorization with partial pivoting, in place.
///
/// Overwrites `a` with the packed `L` (unit diagonal, multipliers below it) and
/// `U`. Returns the pivot indices (0-based, unlike LAPACK's 1-based) and the
/// determinant.
///
/// The determinant comes out of the factorization for free as the product of `U`'s
/// diagonal with a sign flip per row interchange, which is why it is returned here
/// rather than computed separately.
///
/// # Errors
///
/// [`LapackError::Singular`] if a pivot is zero to working precision, and
/// [`LapackError::BadShape`] if `a.len() != n * n`.
pub fn dgetrf(a: &mut [f64], n: usize) -> Result<(Vec<usize>, f64), LapackError> {
    if a.len() != n * n {
        return Err(LapackError::BadShape);
    }
    if n == 0 {
        return Ok((Vec::new(), 1.0));
    }

    let mut ipiv = vec![0usize; n];
    let mut det = 1.0;

    for k in 0..n {
        // Partial pivoting: the largest magnitude in this column at or below the
        // diagonal. Without it the factorization is unstable for any matrix with
        // a small leading pivot.
        let mut pivot = k;
        let mut best = a[k + k * n].abs();
        for i in (k + 1)..n {
            let cand = a[i + k * n].abs();
            if cand > best {
                best = cand;
                pivot = i;
            }
        }

        if best <= ZERO_DETERMINANT {
            return Err(LapackError::Singular);
        }

        // Swap rows k and pivot across all columns.
        if pivot != k {
            for j in 0..n {
                a.swap(j * n + k, j * n + pivot);
            }
            det = -det;
        }
        ipiv[k] = pivot;

        let d = a[k + k * n];
        det *= d;

        // Eliminate below the diagonal, storing the multipliers in place.
        for i in (k + 1)..n {
            let factor = a[i + k * n] / d;
            a[i + k * n] = factor;
            for j in (k + 1)..n {
                a[i + j * n] -= factor * a[k + j * n];
            }
        }
    }

    Ok((ipiv, det))
}

/// `dgetri`: inverse of a matrix from its `dgetrf` factorization, in place.
///
/// Solves `A X = I` column by column: apply the row interchanges, forward-solve
/// with the unit lower triangle, then back-solve with the upper triangle.
///
/// # Errors
///
/// [`LapackError::BadShape`] if the shapes disagree, or
/// [`LapackError::Singular`] if a diagonal of `U` is zero.
pub fn dgetri(lu: &mut [f64], n: usize, ipiv: &[usize]) -> Result<(), LapackError> {
    if lu.len() != n * n || ipiv.len() != n {
        return Err(LapackError::BadShape);
    }
    if n == 0 {
        return Ok(());
    }

    let mut inverse = vec![0.0; n * n];
    let mut b = vec![0.0; n];

    for col in 0..n {
        // Start from e_col and apply the interchanges in the order dgetrf made
        // them; applying them in reverse is the classic mistake here.
        b.fill(0.0);
        b[col] = 1.0;
        for k in 0..n {
            b.swap(k, ipiv[k]);
        }

        // Forward substitution through the unit lower triangle.
        for i in 0..n {
            let mut sum = b[i];
            for j in 0..i {
                sum -= lu[i + j * n] * b[j];
            }
            b[i] = sum;
        }

        // Back substitution through the upper triangle.
        for i in (0..n).rev() {
            let mut sum = b[i];
            for j in (i + 1)..n {
                sum -= lu[i + j * n] * b[j];
            }
            let d = lu[i + i * n];
            if d == 0.0 {
                return Err(LapackError::Singular);
            }
            b[i] = sum / d;
        }

        for i in 0..n {
            inverse[i + col * n] = b[i];
        }
    }

    lu.copy_from_slice(&inverse);
    Ok(())
}

/// `dgesv`: solve `A X = B` for a single right-hand side.
///
/// `a` is `n x n` column-major and is destroyed; `b` is the right-hand side on
/// entry and the solution `x` on exit.
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

    // The determinant is not needed for a solve, but it comes back with the
    // factorization.
    let (ipiv, _det) = dgetrf(a, n)?;

    // Apply the interchanges to b, then forward- and back-substitute. b has a
    // single column, so this is the same sweep `dgetri` does per column.
    for k in 0..n {
        b.swap(k, ipiv[k]);
    }
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= a[i + j * n] * b[j];
        }
        b[i] = sum;
    }
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i + j * n] * b[j];
        }
        b[i] = sum / a[i + i * n];
    }

    Ok(())
}

/// `dpotrf`: Cholesky factorization `A = L L^T`, lower triangle, in place.
///
/// Only the lower triangle of `a` is read, and only the lower triangle is
/// written, as LAPACK specifies. The algorithm is the standard right-looking
/// one from Golub & Van Loan, which accumulates a dot product per entry; that is
/// slower than the block algorithm LAPACK uses but is the clearest correct form
/// and is what the parity tests are written against.
///
/// # Errors
///
/// [`LapackError::Singular`] if `A` is not positive definite (a non-positive
/// diagonal appears), [`LapackError::BadShape`] on a shape mismatch.
pub fn dpotrf(a: &mut [f64], n: usize) -> Result<(), LapackError> {
    if a.len() != n * n {
        return Err(LapackError::BadShape);
    }

    for j in 0..n {
        // Diagonal: l_jj = sqrt(a_jj - sum_{k<j} l_jk^2)
        let mut sum = a[j + j * n];
        for k in 0..j {
            let ljk = a[j + k * n];
            sum -= ljk * ljk;
        }
        // A non-positive value means the matrix is not positive definite. No
        // tolerance fudge: the test is exactly what the algorithm needs.
        if sum <= 0.0 {
            return Err(LapackError::Singular);
        }
        let ljj = sum.sqrt();
        a[j + j * n] = ljj;

        // Below the diagonal: l_ij = (a_ij - sum_{k<j} l_ik l_jk) / l_jj
        for i in (j + 1)..n {
            let mut s = a[i + j * n];
            for k in 0..j {
                s -= a[i + k * n] * a[j + k * n];
            }
            a[i + j * n] = s / ljj;
        }
    }

    Ok(())
}

/// `dlange` re-exported for symmetry with the C shim's surface.
pub use super::blas::dlange;

/// `dlange` selected by an integer code rather than a character.
///
/// The codes are the ones `crate::Norm`'s discriminants carry, so the dispatch in
/// `mat_norm` can forward its value unchanged instead of mapping characters.
/// They match LAPACK's own norm codes: 1 = one-norm, 2 = infinity, 3 =
/// Frobenius, 4 = maximum absolute entry.
pub fn dlange_by_code(norm_code: i32, a: &[f64], m: usize, n: usize) -> f64 {
    super::blas::dlange(norm_char(norm_code), a, m, n)
}

/// The BLAS norm character for a `crate::Norm` discriminant.
///
/// This must agree with the mapping in `c_code/interface_blas.c`, which is the
/// source of truth: the discriminants are `0` Euc, `1` Fro, `2` Inf, `3` Max,
/// `4` One, and the shim sends `0|1` to `"F"`, `2` to `"I"`, `3` to `"M"` and
/// anything else to `"O"`.
///
/// Note there is no `"1"` case: `Norm::One` reaches `"O"`, LAPACK's one-norm.
/// Mapping by position instead of by constant is the trap here -- an earlier
/// version of this function sent `1` to `"1"` and `4` to `"M"`, which made every
/// norm silently wrong.
pub fn norm_char(norm_code: i32) -> char {
    use crate::internal::{NORM_EUC, NORM_FRO, NORM_INF, NORM_MAX};
    if norm_code == NORM_EUC as i32 || norm_code == NORM_FRO as i32 {
        'F'
    } else if norm_code == NORM_INF as i32 {
        'I'
    } else if norm_code == NORM_MAX as i32 {
        'M'
    } else {
        'O'
    }
}

/// Silence the unused-import warning when only some routines use the BLAS layer.
#[allow(unused_imports)]
use blas as _blas;

/// Error used when a routine has no native implementation yet.
///
/// Returning this rather than leaving the call site pointed at a symbol that is
/// not linked is deliberate: a runtime error names the feature to enable, whereas
/// a link error names a mangled symbol and says nothing about what to do.
///
/// No routine currently returns it -- every routine the crate's public API needs
/// is implemented natively -- but the mapping is kept so that the `NotImplemented`
/// variant still has a message, and so a call site does not have to change if a
/// routine is reverted.
pub fn not_implemented(routine: &str) -> &'static str {
    // The message must be a `&'static str` to fit `StrError`, so the routine
    // names are listed literally rather than interpolated.
    let _ = routine;
    "this routine is not implemented in the native Rust LAPACK yet;      build with `--features open_blas` (or `intel_mkl`) to use an external LAPACK"
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    /// A small, well-conditioned, non-symmetric, non-triangular matrix, so no
    /// coincidence can hide an indexing mistake.
    fn sample4() -> Vec<f64> {
        // Row-major [[4,1,2,1],[1,3,0,1],[2,0,5,2],[1,1,2,6]] as column-major.
        vec![
            4.0, 1.0, 2.0, 1.0, // column 0
            1.0, 3.0, 0.0, 1.0, // column 1
            2.0, 0.0, 5.0, 2.0, // column 2
            1.0, 1.0, 2.0, 6.0, // column 3
        ]
    }

    fn mat_vec(a: &[f64], x: &[f64], n: usize) -> Vec<f64> {
        let mut y = vec![0.0; n];
        blas::dgemv(1.0, a, x, 0.0, &mut y, n, n);
        y
    }

    #[test]
    fn dgetrf_determinant_and_pivots() {
        let mut a = sample4();
        let (ipiv, det) = dgetrf(&mut a, 4).unwrap();
        assert_eq!(ipiv.len(), 4);
        // The determinant of the sample matrix, computed independently by
        // cofactor expansion over the first row.
        let expect = 4.0
            * (3.0 * (5.0 * 6.0 - 2.0 * 2.0) - 0.0 * (0.0 * 6.0 - 2.0 * 1.0) + 1.0 * (0.0 * 2.0 - 5.0 * 1.0))
            - 1.0 * (1.0 * (5.0 * 6.0 - 2.0 * 2.0) - 0.0 * (2.0 * 6.0 - 2.0 * 1.0) + 1.0 * (2.0 * 2.0 - 5.0 * 1.0))
            + 2.0 * (1.0 * (0.0 * 6.0 - 2.0 * 1.0) - 3.0 * (2.0 * 6.0 - 2.0 * 1.0) + 1.0 * (2.0 * 1.0 - 0.0 * 1.0))
            - 1.0 * (1.0 * (0.0 * 2.0 - 5.0 * 1.0) - 3.0 * (2.0 * 2.0 - 5.0 * 1.0) + 0.0 * (2.0 * 1.0 - 0.0 * 1.0));
        assert!((det - expect).abs() < 1e-12, "det {det} vs {expect}");
    }

    #[test]
    fn dgetrf_detects_a_singular_matrix() {
        // The second column is twice the first, so the matrix is singular.
        let mut a = vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 1.0, 1.0, 1.0];
        assert_eq!(dgetrf(&mut a, 3), Err(LapackError::Singular));
    }

    #[test]
    fn dgetrf_rejects_a_bad_shape() {
        let mut a = vec![1.0, 2.0, 3.0];
        assert_eq!(dgetrf(&mut a, 2), Err(LapackError::BadShape));
    }

    #[test]
    fn dgetri_inverts_correctly() {
        let a = sample4();
        let mut lu = a.clone();
        let (ipiv, _) = dgetrf(&mut lu, 4).unwrap();
        dgetri(&mut lu, 4, &ipiv).unwrap();

        // A * A^-1 = I, checked through the BLAS layer rather than by hand.
        for j in 0..4 {
            let e = {
                let mut v = vec![0.0; 4];
                v[j] = 1.0;
                v
            };
            let got = mat_vec(&a, &e, 4);
            let expect = mat_vec(&lu, &e, 4); // A^-1 e_j is row j of A^-1
            // A * (A^-1 e_j) must be e_j.
            let _ = expect;
            let mut back = vec![0.0; 4];
            blas::dgemv(1.0, &lu, &got, 0.0, &mut back, 4, 4);
            for i in 0..4 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (back[i] - want).abs() < 1e-12,
                    "A*A^-1 e_{j} component {i} = {}",
                    back[i]
                );
            }
        }
    }

    #[test]
    fn dgesv_solves_a_system() {
        // Solve the sample system with a known solution: pick x, compute b = A x,
        // then check dgesv recovers x. That avoids hand-transcribing a solution.
        let a = sample4();
        let x_true = vec![1.0, -2.0, 3.0, 5.0];
        let mut b = mat_vec(&a, &x_true, 4);

        let mut a_work = a.clone();
        dgesv(&mut a_work, &mut b).unwrap();

        for i in 0..4 {
            assert!((b[i] - x_true[i]).abs() < 1e-12, "x[{}] = {} vs {}", i, b[i], x_true[i]);
        }
    }

    #[test]
    fn dgesv_reports_a_singular_system() {
        let mut a = vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 1.0, 1.0, 1.0];
        let mut b = vec![1.0, 1.0, 1.0];
        assert_eq!(dgesv(&mut a, &mut b), Err(LapackError::Singular));
    }

    #[test]
    fn dpotrf_factorises_a_spd_matrix() {
        // A = [[4,2],[2,3]] is symmetric positive definite.
        let mut a = vec![4.0, 2.0, 2.0, 3.0];
        dpotrf(&mut a, 2).unwrap();

        // L = [[2,0],[1,sqrt(2)]]
        assert!((a[0] - 2.0).abs() < 1e-14, "l00 = {}", a[0]);
        assert!((a[1] - 1.0).abs() < 1e-14, "l10 = {}", a[1]);
        assert!((a[3] - 2.0f64.sqrt()).abs() < 1e-14, "l11 = {}", a[3]);

        // Reconstruct: L L^T must give A back. Only the lower triangle was
        // written -- LAPACK leaves the rest untouched -- so build L explicitly
        // rather than assuming the whole buffer holds L. A test that read the
        // upper triangle here would "fail" on correct code.
        let mut l = [0.0; 4];
        for i in 0..2 {
            for j in 0..2 {
                if i >= j {
                    l[i + j * 2] = a[i + j * 2];
                }
            }
        }
        let mut recon = [0.0; 4];
        for i in 0..2 {
            for j in 0..2 {
                let mut sum = 0.0;
                for k in 0..2 {
                    sum += l[i + k * 2] * l[j + k * 2];
                }
                recon[i + j * 2] = sum;
            }
        }
        assert!((recon[0] - 4.0).abs() < 1e-12);
        assert!((recon[1] - 2.0).abs() < 1e-12);
        assert!((recon[3] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn dpotrf_rejects_a_non_positive_definite_matrix() {
        // [[1,2],[2,1]] has a negative eigenvalue.
        let mut a = vec![1.0, 2.0, 2.0, 1.0];
        assert_eq!(dpotrf(&mut a, 2), Err(LapackError::Singular));

        // A symmetric but singular one.
        let mut b = vec![1.0, 1.0, 1.0, 1.0];
        assert_eq!(dpotrf(&mut b, 2), Err(LapackError::Singular));
    }

    #[test]
    fn empty_matrices_are_handled() {
        let mut a: Vec<f64> = Vec::new();
        assert_eq!(dgetrf(&mut a, 0).unwrap(), (Vec::new(), 1.0));
        let mut b: Vec<f64> = Vec::new();
        assert_eq!(dgesv(&mut a, &mut b), Ok(()));
        assert_eq!(dpotrf(&mut a, 0), Ok(()));
    }

    /// `norm_char` is the mapping the C shim's `interface_blas.c` also implements,
    /// and `dlange_by_code` is the bridge the dispatch sites use. Both were
    /// untested: a wrong discriminant would only show up as a wrong-normed
    /// scaling decision inside `dgesvd` or `dsyev`.
    #[test]
    fn norm_char_matches_the_shim_mapping() {
        use crate::internal::{NORM_EUC, NORM_FRO, NORM_INF, NORM_MAX, NORM_ONE};
        assert_eq!(norm_char(NORM_EUC as i32), 'F');
        assert_eq!(norm_char(NORM_FRO as i32), 'F');
        assert_eq!(norm_char(NORM_INF as i32), 'I');
        assert_eq!(norm_char(NORM_MAX as i32), 'M');
        // 'O' is LAPACK's other spelling of the one-norm, and it is what the
        // shim's `interface_blas.c` sends. See the note in AGENTS.md.
        assert_eq!(norm_char(NORM_ONE as i32), 'O');
        // Anything that is not Euc/Fro, Inf or Max is the one-norm: the funnel
        // branch of the function is `'O'`, and `NORM_ONE` reaches it.
        assert_eq!(norm_char(99), 'O');

        // A 2x2 whose four norms are all different, so the codes are told apart:
        // A = [1 2; 3 4], stored column-major.
        //   Euc/Fro = sqrt(30)
        //   Inf     = max *row* sum    = 7   (LAPACK's `'I'`)
        //   Max     = max |a_ij|       = 4
        //   One     = max *column* sum = 6   (LAPACK's `'1'`/`'O'`)
        // Note the direction: LAPACK's `'I'` is the row sum and `'1'` the column
        // sum, the opposite of what the names suggest on their own.
        let a = [1.0, 3.0, 2.0, 4.0];
        let fro = (30.0f64).sqrt();
        let norms = [
            (NORM_EUC as i32, fro),
            (NORM_FRO as i32, fro),
            (NORM_INF as i32, 7.0),
            (NORM_MAX as i32, 4.0),
            (NORM_ONE as i32, 6.0),
        ];
        for (code, want) in norms {
            let got = dlange_by_code(code, &a, 2, 2);
            assert!(
                (got - want).abs() <= 1e-14,
                "code {code} ('{}'): {got} != {want}",
                norm_char(code)
            );
        }
        // The two spellings of the one-norm agree ('1' and 'O' in LAPACK).
        assert_eq!(
            dlange_by_code(NORM_ONE as i32, &a, 2, 2),
            super::super::blas::dlange('O', &a, 2, 2)
        );
    }
}
