//! Portable Rust kernels used when the `lapack` feature is off.
//!
//! # Why this exists
//!
//! On a native target the functions below take their fast path through the C
//! shim (`c_code/interface_blas.c`) into BLAS/LAPACK. Targets such as
//! `wasm32-unknown-unknown` have neither: there is no wasm libc for the shim to
//! compile against (`inttypes.h` does not exist) and no OpenBLAS to link. With
//! the `lapack` feature off the shim is not built, and the functions here
//! provide a dependency-free replacement.
//!
//! These are deliberately plain implementations, not tuned ones. The goal is
//! correctness and portability for the mesh and geometry workloads that need a
//! browser build, where matrices are small and the dominant cost is elsewhere.
//!
//! # What is covered
//!
//! A function lives here only if it is on a path that a `lapack`-less consumer
//! actually needs. Everything else that calls into BLAS/LAPACK is compiled out
//! entirely when `lapack` is off, so the crate still links.
//!
//! | Function | Fast path it replaces |
//! |---|---|
//! | [`vec_copy`] | `cblas_dcopy` |
//! | [`vec_scale`] | `cblas_dscal` |
//! | [`vec_update`] | `cblas_daxpy` |
//! | [`vec_norm`] | `cblas_dnrm2`, `cblas_dasum`, `cblas_idamax` |
//! | [`mat_vec_mul`] | `cblas_dgemv` |
//! | [`mat_mat_mul`] | `cblas_dgemm` |
//! | [`mat_t_mat_mul`] | `cblas_dgemm` (transposed) |
//! | [`mat_inverse`] | `dgetrf` + `dgetri` |
//!
//! Notably **not** covered, and therefore unavailable without `lapack`:
//! eigen-decompositions, SVD, pseudo-inverse, Cholesky, `solve_lin_sys`, and
//! the complex-valued routines. Each of those is gated at its definition.

// With `lapack` on, the fast paths are used and none of these kernels are
// called. They are still compiled, so that the feature combinations differ only
// in which implementation runs, and so that the tests below always exercise the
// portable code on a native target.
#![cfg_attr(any(feature = "oblas", feature = "mkl"), allow(dead_code))]

use crate::{Norm, StrError};

/// Portable version of [`crate::matrix::Matrix`] inversion helper.
///
/// Mirrors the tolerance used by the LAPACK path so that behaviour does not
/// change with the backend.
pub const ZERO_DETERMINANT: f64 = 1e-15;

/// `v := u`, replacing `cblas_dcopy`.
///
/// # Panics
///
/// Panics if the vectors have different dimensions. The public wrapper in
/// [`crate::vector`] performs this check and returns an error instead, so this
/// is only reached with already-validated arguments.
pub fn vec_copy_data(v: &mut [f64], u: &[f64]) {
    v.copy_from_slice(u);
}

/// `v := α ⋅ v`, replacing `cblas_dscal`.
pub fn vec_scale_data(v: &mut [f64], alpha: f64) {
    for x in v.iter_mut() {
        *x *= alpha;
    }
}

/// `v := α ⋅ u + v`, replacing `cblas_daxpy`.
pub fn vec_update_data(v: &mut [f64], alpha: f64, u: &[f64]) {
    for (vi, ui) in v.iter_mut().zip(u.iter()) {
        *vi += alpha * ui;
    }
}

/// Vector norm, replacing `cblas_dnrm2` / `cblas_dasum` / `cblas_idamax`.
///
/// `Norm::Euc` and `Norm::Fro` are the Euclidean norm, `Norm::Inf` and
/// `Norm::Max` are the largest absolute entry, and `Norm::One` is the sum of
/// absolute values — matching the documentation of [`Norm`] and the tests in
/// [`crate::vector::vec_norm`].
pub fn vec_norm_data(v: &[f64], kind: Norm) -> f64 {
    match kind {
        Norm::Euc | Norm::Fro => v.iter().fold(0.0, |acc, x| acc + x * x).sqrt(),
        Norm::One => v.iter().map(|x| x.abs()).sum(),
        Norm::Inf | Norm::Max => v.iter().fold(0.0, |acc, x| acc.max(x.abs())),
    }
}

/// `v := α ⋅ a ⋅ u`, replacing `cblas_dgemv`.
///
/// `a` is column-major with `m` rows and `n` columns, which is how
/// [`crate::matrix::Matrix`] stores its data.
pub fn mat_vec_mul_data(v: &mut [f64], alpha: f64, a: &[f64], m: usize, n: usize, u: &[f64]) {
    v.fill(0.0);
    for j in 0..n {
        let uj = u[j];
        if uj == 0.0 {
            continue;
        }
        let column = &a[j * m..(j + 1) * m];
        for i in 0..m {
            v[i] += alpha * column[i] * uj;
        }
    }
}

/// `c := α ⋅ a ⋅ b + β ⋅ c`, replacing `cblas_dgemm`.
///
/// Dimensions: `a` is `m×k`, `b` is `k×n`, `c` is `m×n`; all column-major.
///
/// The indexing is written out explicitly rather than with slices: with all
/// three dimensions distinct, an off-by-one stride is easy to introduce and
/// hard to see, and this is not a hot path for the workloads that need it.
pub fn mat_mat_mul_data(c: &mut [f64], alpha: f64, a: &[f64], b: &[f64], beta: f64, m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            // a[i][l] is at l*m + i (column-major, m rows);
            // b[l][j] is at j*k + l (column-major, k rows).
            let mut sum = 0.0;
            for l in 0..k {
                sum += a[l * m + i] * b[j * k + l];
            }
            let i_c = i + j * m;
            c[i_c] = if beta == 0.0 {
                alpha * sum
            } else {
                alpha * sum + beta * c[i_c]
            };
        }
    }
}

/// `c := α ⋅ aᵀ ⋅ b + β ⋅ c`, replacing `cblas_dgemm` with `CblasTrans`.
///
/// Dimensions: `a` is `k×m`, `b` is `k×n`, `c` is `m×n`; all column-major.
///
/// Implemented by materialising `aᵀ` as an `m×k` buffer and delegating to
/// [`mat_mat_mul_data`], so there is a single tested multiplication. The scratch
/// is `m*k` floats, negligible for the small matrices this portable path exists
/// for, and it removes a whole class of stride mistake.
pub fn mat_t_mat_mul_data(c: &mut [f64], alpha: f64, a: &[f64], b: &[f64], beta: f64, m: usize, n: usize, k: usize) {
    // `a` is k×m column-major: element (row, col) is at col*k + row, i.e.
    // a[row][col] = a[col*k + row].
    // `at` is m×k column-major: element (i, j) is at j*m + i.
    // Transposing means at[i][j] = a[j][i], so:
    //   at[j*m + i] = a[i*k + j]
    let mut at = vec![0.0; m * k];
    for i in 0..m {
        for j in 0..k {
            at[j * m + i] = a[i * k + j];
        }
    }
    mat_mat_mul_data(c, alpha, &at, b, beta, m, n, k);
}

/// In-place LU decomposition with partial pivoting, replacing `dgetrf`.
///
/// Returns `(det, ipiv)`, or an error if the matrix is singular. `ipiv` holds
/// 0-based pivots, unlike LAPACK's 1-based convention.
///
/// The observable behaviour matches the LAPACK path used by
/// [`crate::matrix::mat_inverse`]: the packed LU is written back into `a`, and
/// the determinant is the product of the U diagonal with a sign flip per row
/// interchange.
pub fn lu_decompose(a: &mut [f64], m: usize) -> Result<(f64, Vec<usize>), StrError> {
    let mut ipiv = vec![0usize; m];
    let mut det = 1.0;

    for k in 0..m {
        // Partial pivoting: find the largest magnitude entry in column k below
        // the diagonal.
        let mut pivot = k;
        let mut best = f64::abs(a[k * m + k]);
        for i in (k + 1)..m {
            let candidate = f64::abs(a[k * m + i]);
            if candidate > best {
                best = candidate;
                pivot = i;
            }
        }

        if best <= ZERO_DETERMINANT {
            return Err("cannot compute inverse due to zero determinant");
        }

        // Swap rows k and pivot across all columns.
        if pivot != k {
            for j in 0..m {
                a.swap(j * m + k, j * m + pivot);
            }
            det = -det;
        }

        ipiv[k] = pivot;
        let diagonal = a[k * m + k];
        det *= diagonal;

        // Eliminate below the diagonal, storing the multipliers in place.
        for i in (k + 1)..m {
            let factor = a[k * m + i] / diagonal;
            a[k * m + i] = factor;
            for j in (k + 1)..m {
                a[j * m + i] -= factor * a[j * m + k];
            }
        }
    }

    Ok((det, ipiv))
}

/// Solves `L⋅U⋅x = P⋅b` in place, given the packed LU factors and pivots from
/// [`lu_decompose`], replacing `dgetri`.
///
/// `a` is the packed LU (column-major, `m×m`); on return `a` holds the inverse.
pub fn lu_invert(a: &mut [f64], m: usize, ipiv: &[usize]) -> Result<(), StrError> {
    let mut inverse = vec![0.0; m * m];

    // One right-hand side per column of the identity.
    let mut b = vec![0.0; m];
    for column in 0..m {
        b.fill(0.0);
        b[column] = 1.0;

        // Apply the row interchanges in the order they were made.
        for k in 0..m {
            b.swap(k, ipiv[k]);
        }

        // Forward substitution through the unit lower-triangular L.
        for i in 0..m {
            let mut sum = b[i];
            for j in 0..i {
                sum -= a[j * m + i] * b[j];
            }
            b[i] = sum;
        }

        // Back substitution through the upper-triangular U.
        for i in (0..m).rev() {
            let mut sum = b[i];
            for j in (i + 1)..m {
                sum -= a[j * m + i] * b[j];
            }
            let diagonal = a[i * m + i];
            if diagonal == 0.0 {
                return Err("cannot compute inverse due to zero determinant");
            }
            b[i] = sum / diagonal;
        }

        for i in 0..m {
            inverse[column * m + i] = b[i];
        }
    }

    a.copy_from_slice(&inverse);
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Matrix, Vector, approx_eq};

    #[test]
    fn vec_kernels_work() {
        let u = vec![1.0, 2.0, 3.0];
        let mut v = vec![0.0; 3];
        vec_copy_data(&mut v, &u);
        assert_eq!(v, u);

        vec_scale_data(&mut v, 2.0);
        assert_eq!(v, vec![2.0, 4.0, 6.0]);

        vec_update_data(&mut v, 0.5, &u);
        assert_eq!(v, vec![2.5, 5.0, 7.5]);
    }

    #[test]
    fn vec_norm_matches_documented_semantics() {
        let v = vec![-3.0, 2.0, 1.0, 1.0, 1.0];
        assert_eq!(vec_norm_data(&v, Norm::Euc), 4.0);
        assert_eq!(vec_norm_data(&v, Norm::Fro), 4.0);
        assert_eq!(vec_norm_data(&v, Norm::Inf), 3.0);
        assert_eq!(vec_norm_data(&v, Norm::Max), 3.0);
        assert_eq!(vec_norm_data(&v, Norm::One), 8.0);

        let empty: Vec<f64> = Vec::new();
        assert_eq!(vec_norm_data(&empty, Norm::Euc), 0.0);
        assert_eq!(vec_norm_data(&empty, Norm::Inf), 0.0);
        assert_eq!(vec_norm_data(&empty, Norm::One), 0.0);
    }

    #[test]
    fn mat_vec_mul_matches_expected() {
        // Same example as the `mat_vec_mul` doc test: a is 4x3, column-major.
        let a = Matrix::from(&[[5.0, -2.0, 1.0], [-4.0, 0.0, 2.0], [15.0, -6.0, 0.0], [3.0, 5.0, 1.0]]);
        let u = Vector::from(&[1.0, 2.0, 3.0]);
        let mut v = Vector::new(4);
        mat_vec_mul_data(v.as_mut_data(), 0.5, a.as_data(), 4, 3, u.as_data());
        assert_eq!(v.as_data(), &[2.0, 1.0, 1.5, 8.0]);
    }

    #[test]
    fn mat_mat_mul_matches_expected() {
        // c := a·b with a m×k, b k×n. Use m=2, k=3, n=2 so that no two of the
        // three dimensions coincide: an index that confuses k with n (as an
        // earlier version of this kernel did) cannot pass by accident.
        let a = Matrix::from(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]); // 2x3
        let b = Matrix::from(&[[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]); // 3x2
        let mut c = Matrix::new(2, 2);

        // c := 1.0 * a * b + 0.0 * c
        mat_mat_mul_data(c.as_mut_data(), 1.0, a.as_data(), b.as_data(), 0.0, 2, 2, 3);
        // [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
        //  = [[58, 64], [139, 154]]
        for i in 0..2 {
            for j in 0..2 {
                let expected = (0..3).map(|l| a.get(i, l) * b.get(l, j)).sum::<f64>();
                assert_eq!(c.get(i, j), expected, "c[{i},{j}]");
            }
        }
        assert_eq!(c.get(0, 0), 58.0);
        assert_eq!(c.get(0, 1), 64.0);
        assert_eq!(c.get(1, 0), 139.0);
        assert_eq!(c.get(1, 1), 154.0);

        // c := a * b + 2 * c_old
        let previous = c.as_data().clone();
        mat_mat_mul_data(c.as_mut_data(), 1.0, a.as_data(), b.as_data(), 2.0, 2, 2, 3);
        for i in 0..4 {
            assert_eq!(c.as_data()[i], previous[i] + 2.0 * previous[i]);
        }
    }

    /// The portable kernels must agree with the BLAS path. `mat_t_mat_mul` and
    /// `mat_mat_mul` are backed by `cblas_dgemm` (available because this test
    /// only runs with `lapack` on), so they define the expected result.
    ///
    /// This replaces an earlier hand-derived element formula, which was wrong
    /// about the column-major strides while looking entirely plausible.
    #[test]
    fn portable_kernels_agree_with_blas_path() {
        use crate::{mat_mat_mul, mat_t_mat_mul};

        let a = Matrix::from(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]); // 3x2
        let b = Matrix::from(&[[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]); // 3x2
        let m = 2;
        let n = 2;
        let k = 3;

        // c := aᵀ·b   (a is k×m, b is k×n, c is m×n)
        let mut expected = Matrix::new(m, n);
        mat_t_mat_mul(&mut expected, 1.0, &a, &b, 0.0).unwrap();
        let mut got = Matrix::new(m, n);
        mat_t_mat_mul_data(got.as_mut_data(), 1.0, a.as_data(), b.as_data(), 0.0, m, n, k);
        for i in 0..m {
            for j in 0..n {
                approx_eq(got.get(i, j), expected.get(i, j), 1e-12);
            }
        }

        // Same for the non-transposed form, with a m×k and b k×n.
        let a2 = Matrix::from(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]); // 2x3
        let b2 = Matrix::from(&[[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]); // 3x2
        let mut expected2 = Matrix::new(2, 2);
        mat_mat_mul(&mut expected2, 1.0, &a2, &b2, 0.0).unwrap();
        let mut got2 = Matrix::new(2, 2);
        mat_mat_mul_data(got2.as_mut_data(), 1.0, a2.as_data(), b2.as_data(), 0.0, 2, 2, 3);
        for i in 0..2 {
            for j in 0..2 {
                approx_eq(got2.get(i, j), expected2.get(i, j), 1e-12);
            }
        }
    }

    #[test]
    fn mat_t_mat_mul_matches_expected() {
        // c := aᵀ·b with a k×m, b k×n, so c is m×n. Use k=3, m=2, n=2.
        let a = Matrix::from(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]); // 3x2
        let b = Matrix::from(&[[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]); // 3x2
        let mut c = Matrix::new(2, 2);

        mat_t_mat_mul_data(c.as_mut_data(), 1.0, a.as_data(), b.as_data(), 0.0, 2, 2, 3);
        // aᵀ = [[1,3,5],[2,4,6]]
        // [[1*7+3*9+5*11, 1*8+3*10+5*12], [2*7+4*9+6*11, 2*8+4*10+6*12]]
        //  = [[89, 98], [116, 128]]
        assert_eq!(c.get(0, 0), 89.0);
        assert_eq!(c.get(0, 1), 98.0);
        assert_eq!(c.get(1, 0), 116.0);
        assert_eq!(c.get(1, 1), 128.0);
    }

    /// The portable SVD must agree with LAPACK's on the singular values, which
    /// are unique. (The singular *vectors* are not unique, so they are checked
    /// by reconstruction in the test above instead.)

    #[test]
    fn lu_decompose_and_invert_work() {
        // 4x4 so that the small-matrix shortcuts in `mat_inverse` do not apply.
        let mut a = Matrix::from(&[
            [4.0, 1.0, 2.0, 1.0],
            [1.0, 3.0, 0.0, 1.0],
            [2.0, 0.0, 5.0, 2.0],
            [1.0, 1.0, 2.0, 6.0],
        ]);
        let original = a.as_data().clone();
        let (det, ipiv) = lu_decompose(a.as_mut_data(), 4).unwrap();
        assert!(det > 0.0);
        lu_invert(a.as_mut_data(), 4, &ipiv).unwrap();

        // a * a⁻¹ = I
        let mut product = Matrix::new(4, 4);
        mat_mat_mul_data(product.as_mut_data(), 1.0, &original, a.as_data(), 0.0, 4, 4, 4);
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                approx_eq(product.get(i, j), expected, 1e-12);
            }
        }
    }

    #[test]
    fn lu_decompose_detects_singular_matrix() {
        let mut a = Matrix::from(&[[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 1.0, 1.0]]);
        assert!(lu_decompose(a.as_mut_data(), 3).is_err());
    }
}
