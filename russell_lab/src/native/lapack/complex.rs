//! Native complex LAPACK: the `z*` routines `russell_lab` exposes.
//!
//! Each function names the LAPACK 3.12.1 routine it follows and is a
//! transliteration of that routine's *unblocked* form, with LAPACK's 1-based
//! indexing and `lda`/`lwork` arguments mechanically turned into 0-based indices
//! and `usize` lengths:
//!
//! | Here | Reference |
//! |---|---|
//! | [`zgetrf`] | `zgetf2` (the unblocked kernel of `zgetrf`) |
//! | [`zgetri`] | `zgetri`'s unblocked branch, over `ztrti2`'s upper inverse |
//! | [`zgesv`] | `zgetf2` + `zgetrs('N')` for a single right-hand side |
//! | [`zpotrf`] | `zpotf2`, both triangles |
//!
//! # Why transliterate rather than re-derive
//!
//! The complex routines differ from their real counterparts only in the
//! arithmetic, and that is exactly where the mistakes live: a missing conjugate
//! in the Cholesky update, or a transpose where a Hermitian transpose is
//! required, still produces a plausible-looking factor. Keeping the operations
//! in the reference's order makes the two comparable line by line, and the
//! parity tests in `native::parity` then compare the observable results
//! element-wise against the reference library.
//!
//! # Conventions
//!
//! * Column-major, `data[i + j * n]`, matching [`crate::ComplexMatrix`].
//! * `usize` lengths, 0-based indices, no workspace arguments.
//! * Only the triangle LAPACK says is read is read, and only the triangle it
//!   says is written is written. See [`zpotrf`] in particular: `'U'` *reads* the
//!   upper triangle alone, so a caller may legitimately pass zeros below the
//!   diagonal, and it *writes* only the upper triangle.

use super::{LapackError, ZERO_DETERMINANT};
use crate::Complex64;

/// The zero element, named so the arithmetic reads like the Fortran `ZERO`.
fn zero() -> Complex64 {
    Complex64::new(0.0, 0.0)
}

/// The one element, named so the arithmetic reads like the Fortran `ONE`.
fn one() -> Complex64 {
    Complex64::new(1.0, 0.0)
}

/// `IZAMAX`'s measure of a complex entry: `|Re| + |Im|`.
///
/// This is what LAPACK's `izamax` maximizes when choosing a pivot, so using the
/// modulus here would pick a different row on some matrices and diverge from the
/// reference for no benefit.
fn abs1(z: Complex64) -> f64 {
    z.re.abs() + z.im.abs()
}

/// `zgetrf`/`zgetf2`: LU factorization with partial pivoting, in place.
///
/// Overwrites `a` with the packed unit-lower `L` (multipliers below the
/// diagonal) and `U`, and returns LAPACK's pivot indices made 0-based: at step
/// `k` rows `k` and `ipiv[k]` were interchanged. The last entry is always `n - 1`
/// because the final column has nothing to pivot against.
///
/// # Errors
///
/// [`LapackError::Singular`] if the pivot is zero to working precision, and
/// [`LapackError::BadShape`] if `a.len() != n * n`.
pub fn zgetrf(a: &mut [Complex64], n: usize) -> Result<Vec<usize>, LapackError> {
    if a.len() != n * n {
        return Err(LapackError::BadShape);
    }
    if n == 0 {
        return Ok(Vec::new());
    }

    let mut ipiv = vec![0usize; n];

    for j in 0..n {
        // JP = J - 1 + IZAMAX(M-J+1, A(J,J), 1) in Fortran, i.e. the first row at
        // or below the diagonal with the largest |Re| + |Im|.
        let mut best = abs1(a[j + j * n]);
        let mut jp = j;
        for i in (j + 1)..n {
            let cand = abs1(a[i + j * n]);
            if cand > best {
                best = cand;
                jp = i;
            }
        }
        ipiv[j] = jp;

        // LAPACK tests `A(JP,J) /= 0`; the crate's tolerance matches the
        // accelerated path's notion of "singular" so both agree on which systems
        // can be solved.
        if best <= ZERO_DETERMINANT {
            return Err(LapackError::Singular);
        }

        // ZSWAP(N, A(J,1), LDA, A(JP,1), LDA): swap the *whole* rows j and jp,
        // not just the trailing submatrix.
        if jp != j {
            for k in 0..n {
                a.swap(k * n + j, k * n + jp);
            }
        }

        // ZRSCL(M-J, A(J,J), A(J+1,J), 1): scale the column below the pivot by
        // 1/A(J,J) *before* the rank-1 update, which is what leaves the
        // multipliers in place of the eliminated entries.
        let d = a[j + j * n];
        for i in (j + 1)..n {
            a[i + j * n] /= d;
        }

        // ZGERU(M-J, N-J, -ONE, A(J+1,J), 1, A(J,J+1), LDA, A(J+1,J+1), LDA).
        for k in (j + 1)..n {
            let ajk = a[j + k * n];
            for i in (j + 1)..n {
                a[i + k * n] -= a[i + j * n] * ajk;
            }
        }
    }

    Ok(ipiv)
}

/// `ztrti2('Upper', 'Non-unit')`: invert an upper triangular matrix in place.
///
/// The diagonal entry of the original is replaced by its reciprocal before the
/// column is formed, so the running product uses 1/U(j,j) exactly as the
/// reference does. Returns [`LapackError::Singular`] if a diagonal is zero.
fn ztrti2_upper(a: &mut [Complex64], n: usize) -> Result<(), LapackError> {
    for j in 0..n {
        // ZTRTRI's own singularity test, done up front there; here it is folded
        // into the only place a zero diagonal would be divided by.
        if a[j + j * n] == zero() {
            return Err(LapackError::Singular);
        }
        let inv = one() / a[j + j * n];
        a[j + j * n] = inv;
        let ajj = -inv;

        // ZTRMV('Upper','No transpose','N', J-1, A, LDA, A(1,J), 1), followed by
        // ZSCAL(J-1, AJJ, A(1,J), 1). The vector is staged so the in-place
        // triangular multiply cannot read a partly-updated column.
        let mut x = vec![zero(); j];
        for i in 0..j {
            let mut sum = zero();
            for k in i..j {
                sum += a[i + k * n] * a[k + j * n];
            }
            x[i] = sum;
        }
        for i in 0..j {
            a[i + j * n] = ajj * x[i];
        }
    }
    Ok(())
}

/// `zgetri`: inverse of a matrix from its [`zgetrf`] factorization, in place.
///
/// Follows the reference: invert `U`, solve `inv(A) * L = inv(U)` column by
/// column (the unblocked branch), then undo the row interchanges as column
/// swaps.
///
/// # Errors
///
/// [`LapackError::BadShape`] if the shapes disagree, or
/// [`LapackError::Singular`] if a diagonal of `U` is zero.
pub fn zgetri(lu: &mut [Complex64], n: usize, ipiv: &[usize]) -> Result<(), LapackError> {
    if lu.len() != n * n || ipiv.len() != n {
        return Err(LapackError::BadShape);
    }
    if n == 0 {
        return Ok(());
    }

    // Form inv(U). A zero diagonal means U was singular.
    ztrti2_upper(lu, n)?;

    // Solve inv(A)*L = inv(U) for inv(A): the unblocked branch of zgetri.
    let mut work = vec![zero(); n];
    let mut y = vec![zero(); n];
    for j in (0..n).rev() {
        // Copy the current column of L to WORK and replace it with zeros; the
        // gemv below must see a strictly-upper A.
        for i in (j + 1)..n {
            work[i] = lu[i + j * n];
            lu[i + j * n] = zero();
        }

        if j + 1 < n {
            // ZGEMV('N', N, N-J, -ONE, A(1,J+1), LDA, WORK(J+1), 1, ONE, A(1,J), 1).
            for i in 0..n {
                let mut sum = zero();
                for k in (j + 1)..n {
                    sum += lu[i + k * n] * work[k];
                }
                y[i] = sum;
            }
            for i in 0..n {
                lu[i + j * n] -= y[i];
            }
        }
    }

    // ZSWAP, column by column: J runs from N-1 down to 1 in Fortran, so the last
    // pivot is deliberately skipped (it is always N).
    for j in (0..n.saturating_sub(1)).rev() {
        let jp = ipiv[j];
        if jp != j {
            for i in 0..n {
                lu.swap(i + j * n, i + jp * n);
            }
        }
    }

    Ok(())
}

/// `zgesv`: solve `A X = B` for a single right-hand side.
///
/// `a` is `n x n` column-major and is destroyed; `b` is the right-hand side on
/// entry and the solution on exit. This is `zgetrf` followed by `zgetrs('N')`
/// with `nrhs == 1`: apply the interchanges to `b`, forward-solve with the unit
/// lower triangle, back-solve with the upper one.
///
/// # Errors
///
/// [`LapackError::Singular`] if `A` is singular, [`LapackError::BadShape`] if the
/// shapes disagree.
pub fn zgesv(a: &mut [Complex64], b: &mut [Complex64]) -> Result<(), LapackError> {
    let n = b.len();
    if a.len() != n * n {
        return Err(LapackError::BadShape);
    }
    if n == 0 {
        return Ok(());
    }

    let ipiv = zgetrf(a, n)?;

    // ZLASWP(NRHS=1, B, LDB, 1, N, IPIV, 1).
    for k in 0..n {
        b.swap(k, ipiv[k]);
    }

    // ZTRSM('Left','Lower','No transpose','Unit', ...).
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= a[i + j * n] * b[j];
        }
        b[i] = sum;
    }

    // ZTRSM('Left','Upper','No transpose','Non-unit', ...).
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i + j * n] * b[j];
        }
        b[i] = sum / a[i + i * n];
    }

    Ok(())
}

/// `zpotrf`/`zpotf2`: Cholesky factorization of a Hermitian positive-definite
/// matrix, in place.
///
/// With `upper == false` it finds the lower triangular `L` with `A = L Lᴴ`;
/// with `upper == true` it finds the upper triangular `U` with `A = Uᴴ U`.
///
/// # The triangle convention
///
/// As LAPACK specifies, only the requested triangle is read and only that
/// triangle is written:
///
/// * `upper == false` reads and writes `i >= j`; the strict upper triangle is
///   left exactly as it was found.
/// * `upper == true` reads and writes `i <= j`; the strict lower triangle is
///   left exactly as it was found, so a caller may pass zeros there without the
///   result depending on them.
///
/// This is why the upper factor is computed by its own branch rather than by
/// transposing a lower factor: mirroring the other triangle first would read
/// entries LAPACK says are not part of the input, and writing it would clobber
/// entries LAPACK says must survive.
///
/// # Errors
///
/// [`LapackError::Singular`] if `A` is not positive definite (a non-positive or
/// NaN diagonal appears), [`LapackError::BadShape`] on a shape mismatch. As in
/// `zpotf2`, the diagonal is overwritten with the offending value before the
/// error is returned.
pub fn zpotrf(a: &mut [Complex64], n: usize, upper: bool) -> Result<(), LapackError> {
    if a.len() != n * n {
        return Err(LapackError::BadShape);
    }

    if upper {
        // Compute the Cholesky factorization A = U**H * U.
        for j in 0..n {
            // AJJ = DBLE(A(J,J)) - DBLE(ZDOTC(J-1, A(1,J), 1, A(1,J), 1)); the
            // dot product is real and equals sum |A(i,j)|^2.
            let mut ajj = a[j + j * n].re;
            for i in 0..j {
                ajj -= a[i + j * n].norm_sqr();
            }
            if ajj <= 0.0 || ajj.is_nan() {
                a[j + j * n] = Complex64::new(ajj, 0.0);
                return Err(LapackError::Singular);
            }
            let ajj = ajj.sqrt();
            a[j + j * n] = Complex64::new(ajj, 0.0);

            if j + 1 < n {
                // ZLACGV conjugates column j, ZGEMV('Transpose') forms
                // A(J,J+1:N) - A(1:J-1,J+1:N)^T * conj(A(1:J-1,J)), and ZLACGV
                // restores the column. Composing the two gives the conjugated
                // product used below without ever writing the conjugated values.
                //
                // ZDSCAL(N-J, 1/AJJ, A(J,J+1), LDA) divides the row by the real
                // diagonal.
                for k in (j + 1)..n {
                    let mut sum = a[j + k * n];
                    for i in 0..j {
                        sum -= a[i + k * n] * a[i + j * n].conj();
                    }
                    a[j + k * n] = sum / ajj;
                }
            }
        }
    } else {
        // Compute the Cholesky factorization A = L*L**H.
        for j in 0..n {
            // AJJ = DBLE(A(J,J)) - sum |A(j,k)|^2 over the already-formed row.
            let mut ajj = a[j + j * n].re;
            for k in 0..j {
                ajj -= a[j + k * n].norm_sqr();
            }
            if ajj <= 0.0 || ajj.is_nan() {
                a[j + j * n] = Complex64::new(ajj, 0.0);
                return Err(LapackError::Singular);
            }
            let ajj = ajj.sqrt();
            a[j + j * n] = Complex64::new(ajj, 0.0);

            // ZLACGV on row j, ZGEMV('No transpose') forming
            // A(J+1:N,J) - A(J+1:N,1:J-1) * conj(A(J,1:J-1)), then ZLACGV back;
            // ZDSCAL divides the column by the real diagonal.
            for i in (j + 1)..n {
                let mut sum = a[i + j * n];
                for k in 0..j {
                    sum -= a[i + k * n] * a[j + k * n].conj();
                }
                a[i + j * n] = sum / ajj;
            }
        }
    }

    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpx;

    /// A small Hermitian positive-definite matrix with non-zero imaginary parts,
    /// so a conjugation mistake cannot cancel out. Column-major, `n = 3`.
    fn spd3() -> Vec<Complex64> {
        // Row-major: [[5, 1+i, 2i], [1-i, 6, 1-2i], [-2i, 1+2i, 7]]
        vec![
            cpx!(5.0, 0.0),
            cpx!(1.0, -1.0),
            cpx!(0.0, 2.0), // column 0
            cpx!(1.0, 1.0),
            cpx!(6.0, 0.0),
            cpx!(1.0, 2.0), // column 1
            cpx!(0.0, -2.0),
            cpx!(1.0, -2.0),
            cpx!(7.0, 0.0), // column 2
        ]
    }

    /// A complex matrix that is neither Hermitian nor triangular.
    fn general3() -> Vec<Complex64> {
        // Row-major: [[4+1i, 1-2i, 2], [1+3i, 3, 0-1i], [2-1i, 1+1i, 5+2i]]
        vec![
            cpx!(4.0, 1.0),
            cpx!(1.0, 3.0),
            cpx!(2.0, -1.0), // column 0
            cpx!(1.0, -2.0),
            cpx!(3.0, 0.0),
            cpx!(1.0, 1.0), // column 1
            cpx!(2.0, 0.0),
            cpx!(0.0, -1.0),
            cpx!(5.0, 2.0), // column 2
        ]
    }

    fn mat_vec(a: &[Complex64], x: &[Complex64], n: usize) -> Vec<Complex64> {
        let mut y = vec![cpx!(0.0, 0.0); n];
        for i in 0..n {
            for j in 0..n {
                y[i] += a[i + j * n] * x[j];
            }
        }
        y
    }

    /// `zgetrf` + `zgetri` must give the inverse: `A * A^-1 = I`.
    ///
    /// The product is accumulated in a different order from the factorization,
    /// so this is not the trivial identity a shared bug could fake.
    #[test]
    fn zgetrf_then_zgetri_inverts() {
        let a = general3();
        let n = 3;
        let mut lu = a.clone();
        let piv = zgetrf(&mut lu, n).unwrap();
        assert_eq!(piv.len(), n);
        assert_eq!(piv[n - 1], n - 1, "the last pivot index is always n-1");
        zgetri(&mut lu, n, &piv).unwrap();

        // lu is now A^-1: check A * A^-1 = I column by column.
        for j in 0..n {
            let mut e = vec![cpx!(0.0, 0.0); n];
            e[j] = cpx!(1.0, 0.0);
            let inv_col = mat_vec(&lu, &e, n); // A^-1 e_j is column j of A^-1
            let back = mat_vec(&a, &inv_col, n);
            for i in 0..n {
                let want = if i == j { cpx!(1.0, 0.0) } else { cpx!(0.0, 0.0) };
                assert!(
                    (back[i] - want).norm() < 1e-12,
                    "A*A^-1 column {j} row {i} = {:?}",
                    back[i]
                );
            }
        }
    }

    /// `zgesv` recovers a known solution.
    #[test]
    fn zgesv_solves_a_system() {
        let a = general3();
        let n = 3;
        let x_true = [cpx!(1.0, -2.0), cpx!(3.0, 0.5), cpx!(-1.0, 1.0)];
        let mut b = mat_vec(&a, &x_true, n);
        let mut a_work = a.clone();
        zgesv(&mut a_work, &mut b).unwrap();
        for i in 0..n {
            assert!(
                (b[i] - x_true[i]).norm() < 1e-12,
                "x[{i}] = {:?} vs {:?}",
                b[i],
                x_true[i]
            );
        }
    }

    /// `zgetrf` reports an exactly singular matrix instead of dividing by a zero
    /// pivot. Column 2 is `2 * column 0`, so the second column is dependent.
    #[test]
    fn zgetrf_detects_a_singular_matrix() {
        let mut a = vec![
            cpx!(1.0, 1.0),
            cpx!(3.0, 0.0), // column 0
            cpx!(2.0, 2.0),
            cpx!(6.0, 0.0), // column 1 = 2 * column 0
        ];
        assert_eq!(zgetrf(&mut a, 2), Err(LapackError::Singular));
    }

    /// A shape mismatch is reported, not read out of bounds.
    #[test]
    fn bad_shapes_are_rejected() {
        let mut a = vec![cpx!(1.0, 0.0); 3];
        assert_eq!(zgetrf(&mut a, 2), Err(LapackError::BadShape));
        let ipiv = [0usize; 1];
        assert_eq!(zgetri(&mut a, 2, &ipiv), Err(LapackError::BadShape));
        let mut b = vec![cpx!(1.0, 0.0); 2];
        assert_eq!(zgesv(&mut a, &mut b), Err(LapackError::BadShape));
        assert_eq!(zpotrf(&mut a, 2, false), Err(LapackError::BadShape));
    }

    /// `zpotrf` gives `A = L L^H` (lower) and `A = U^H U` (upper).
    ///
    /// Only the requested triangle is written, so the factor has to be extracted
    /// from that triangle rather than read from the whole buffer.
    #[test]
    fn zpotrf_factorises_both_triangles() {
        let a = spd3();
        let n = 3;

        let mut lower = a.clone();
        zpotrf(&mut lower, n, false).unwrap();
        for i in 0..n {
            for j in 0..=i {
                let mut sum = cpx!(0.0, 0.0);
                for k in 0..=j {
                    sum += lower[i + k * n] * lower[j + k * n].conj();
                }
                assert!(
                    (sum - a[i + j * n]).norm() < 1e-12,
                    "L L^H at ({i},{j}) = {:?} vs {:?}",
                    sum,
                    a[i + j * n]
                );
            }
        }

        let mut upper = a.clone();
        zpotrf(&mut upper, n, true).unwrap();
        for i in 0..n {
            for j in i..n {
                let mut sum = cpx!(0.0, 0.0);
                for k in 0..=i {
                    sum += upper[k + i * n].conj() * upper[k + j * n];
                }
                assert!(
                    (sum - a[i + j * n]).norm() < 1e-12,
                    "U^H U at ({i},{j}) = {:?} vs {:?}",
                    sum,
                    a[i + j * n]
                );
            }
        }
    }

    /// A Hermitian matrix with a negative eigenvalue is not positive definite and
    /// must be rejected rather than factored.
    #[test]
    fn zpotrf_rejects_a_non_positive_definite_matrix() {
        // [[1, 2], [2, 1]] has eigenvalues 3 and -1.
        let mut a = vec![cpx!(1.0, 0.0), cpx!(2.0, 0.0), cpx!(2.0, 0.0), cpx!(1.0, 0.0)];
        assert_eq!(zpotrf(&mut a, 2, false), Err(LapackError::Singular));
    }

    /// `n == 0` is a no-op everywhere, not a panic or an error.
    #[test]
    fn empty_matrices_are_handled() {
        let mut a: Vec<Complex64> = Vec::new();
        assert!(zgetrf(&mut a, 0).unwrap().is_empty());
        assert_eq!(zgetri(&mut a, 0, &[]), Ok(()));
        let mut b: Vec<Complex64> = Vec::new();
        assert_eq!(zgesv(&mut a, &mut b), Ok(()));
        assert_eq!(zpotrf(&mut a, 0, false), Ok(()));
    }
}
