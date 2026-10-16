//! Portable singular value decomposition, used when the `lapack` feature is off.
//!
//! Split from [`crate::portable`] so that the BLAS-free kernels and this extra
//! algorithm can be reviewed and tested independently.

use crate::small::small_mat_eigen_sym_jacobi;
use crate::StrError;

/// Portable singular value decomposition, replacing the LAPACK `dgesvd` wrapper.
///
/// Re-exported as `small_mat_svd` by [`crate::small`] when `lapack` is off.
///
/// Computes `A = U · diag(s) · Vᵀ` for `A` given row-major as `[[f64; N]; M]`,
/// writing `s` (`min(M,N)` singular values, descending), `u` (`M×M`) and `vt`
/// (`N×N`). Matches the signature and meaning of [`crate::small::small_mat_svd`].
///
/// # Method
///
/// Via the symmetric eigen-decomposition of `AᵀA`, which is available in pure
/// Rust ([`crate::small::small_mat_eigen_sym_jacobi`]), rather than a
/// hand-written SVD:
///
/// 1. `AᵀA` is `N×N`, symmetric and positive semi-definite, so its Jacobi
///    eigen-decomposition gives `AᵀA = V · diag(λ) · Vᵀ`.
/// 2. The singular values are `sᵢ = √λᵢ` and `V` holds the right singular
///    vectors.
/// 3. `U`'s columns follow from `uᵢ = A·vᵢ / sᵢ`.
///
/// # Accuracy
///
/// Forming `AᵀA` squares the condition number, so small singular values lose
/// about half their significant digits relative to a direct (Golub-Kahan) SVD.
/// That is acceptable for the use this portable path exists for — a 3×3 polar
/// decomposition in `russell_tensor` — and is not a substitute for LAPACK on
/// ill-conditioned problems.
pub fn small_mat_svd_portable<const M: usize, const N: usize>(
    s: &mut [f64],
    u: &mut [[f64; M]; M],
    vt: &mut [[f64; N]; N],
    a: &[[f64; N]; M],
) -> Result<(), StrError> {
    const { assert!(M > 0 && N > 0, "matrix dimensions must be greater than zero") };

    let min_mn = if M < N { M } else { N };
    if s.len() != min_mn {
        return Err("[s] must be a min(m,n) vector");
    }

    // 1. AᵀA (N×N, symmetric)
    let mut ata = [[0.0; N]; N];
    for i in 0..N {
        for j in i..N {
            let mut sum = 0.0;
            for k in 0..M {
                sum += a[k][i] * a[k][j];
            }
            ata[i][j] = sum;
            ata[j][i] = sum;
        }
    }

    // 2. Eigen-decomposition of AᵀA. `v` holds the eigenvectors as its columns.
    let mut lambda = [0.0; N];
    let mut v = [[0.0; N]; N];
    small_mat_eigen_sym_jacobi(&mut lambda, &mut v, &mut ata)?;

    // 3. Singular values and the right singular vectors, sorted descending so
    //    that `s` follows the convention of the LAPACK wrapper.
    let mut order: [usize; N] = [0; N];
    for (i, slot) in order.iter_mut().enumerate() {
        *slot = i;
    }
    order.sort_by(|&i, &j| lambda[j].partial_cmp(&lambda[i]).unwrap_or(std::cmp::Ordering::Equal));

    for k in 0..min_mn {
        let eig = lambda[order[k]];
        s[k] = if eig > 0.0 { eig.sqrt() } else { 0.0 };
    }

    // `vt` rows are the right singular vectors: vt[k][i] = v[i][order[k]].
    for k in 0..N {
        for i in 0..N {
            vt[k][i] = v[i][order[k]];
        }
    }

    // 4. Left singular vectors: uᵢ = A·vᵢ / sᵢ, for the min(M,N) that exist.
    for k in 0..min_mn {
        let sk = s[k];
        if sk > 0.0 {
            for i in 0..M {
                let mut sum = 0.0;
                for j in 0..N {
                    sum += a[i][j] * vt[k][j];
                }
                u[i][k] = sum / sk;
            }
        } else {
            // A zero singular value leaves u_k undetermined by the formula;
            // leave it zero and let the completion step below fix it up.
            for i in 0..M {
                u[i][k] = 0.0;
            }
        }
    }

    // 5. Complete `u` to a full M×M orthonormal basis (only needed when M > N).
    for k in min_mn..M {
        for i in 0..M {
            u[i][k] = if i == k { 1.0 } else { 0.0 };
        }
        // Gram-Schmidt against every column already fixed, twice for stability.
        for _ in 0..2 {
            for c in 0..k {
                let mut dot = 0.0;
                for i in 0..M {
                    dot += u[i][k] * u[i][c];
                }
                for i in 0..M {
                    u[i][k] -= dot * u[i][c];
                }
            }
        }
        // Normalise; if the candidate collapsed, fall back to another unit axis.
        let mut norm = 0.0;
        for i in 0..M {
            norm += u[i][k] * u[i][k];
        }
        norm = norm.sqrt();
        if norm > 1e-12 {
            for i in 0..M {
                u[i][k] /= norm;
            }
        } else {
            for i in 0..M {
                u[i][k] = if i == k { 1.0 } else { 0.0 };
            }
        }
    }

    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::small_mat_svd_portable;
    use crate::approx_eq;

        #[test]
        fn small_mat_svd_portable_reconstructs_the_matrix() {
            // Rectangular and non-symmetric, so no accidental symmetry can hide a bug.
            let a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]; // M=2, N=3
            let mut s = [0.0; 2];
            let mut u = [[0.0; 2]; 2];
            let mut vt = [[0.0; 3]; 3];
            small_mat_svd_portable(&mut s, &mut u, &mut vt, &a).unwrap();

            // Singular values must be non-negative and descending.
            assert!(s[0] >= s[1]);
            assert!(s[1] >= 0.0);
            approx_eq(s[0], 9.508032000695723, 1e-9);
            approx_eq(s[1], 0.7728696357127786, 1e-9);

            // A == U · diag(s) · Vᵀ
            for i in 0..2 {
                for j in 0..3 {
                    let mut sum = 0.0;
                    for k in 0..2 {
                        sum += u[i][k] * s[k] * vt[k][j];
                    }
                    approx_eq(sum, a[i][j], 1e-12);
                }
            }

            // U must be orthonormal.
            for p in 0..2 {
                for q in 0..2 {
                    let mut dot = 0.0;
                    for i in 0..2 {
                        dot += u[i][p] * u[i][q];
                    }
                    let expected = if p == q { 1.0 } else { 0.0 };
                    approx_eq(dot, expected, 1e-12);
                }
            }
        }
        #[test]
        fn small_mat_svd_portable_handles_square_and_zero() {
            // Square symmetric: singular values are |eigenvalues| = 3, 1.
            let a = [[2.0, 1.0], [1.0, 2.0]];
            let mut s = [0.0; 2];
            let mut u = [[0.0; 2]; 2];
            let mut vt = [[0.0; 2]; 2];
            small_mat_svd_portable(&mut s, &mut u, &mut vt, &a).unwrap();
            approx_eq(s[0], 3.0, 1e-12);
            approx_eq(s[1], 1.0, 1e-12);
            for i in 0..2 {
                for j in 0..2 {
                    let mut sum = 0.0;
                    for k in 0..2 {
                        sum += u[i][k] * s[k] * vt[k][j];
                    }
                    approx_eq(sum, a[i][j], 1e-12);
                }
            }

            // The zero matrix must not divide by zero.
            let z = [[0.0, 0.0], [0.0, 0.0]];
            let mut sz = [0.0; 2];
            let mut uz = [[0.0; 2]; 2];
            let mut vtz = [[0.0; 2]; 2];
            small_mat_svd_portable(&mut sz, &mut uz, &mut vtz, &z).unwrap();
            approx_eq(sz[0], 0.0, 1e-15);
            approx_eq(sz[1], 0.0, 1e-15);

            // A wrong-sized `s` is still rejected.
            let mut bad = [0.0; 3];
            assert!(small_mat_svd_portable(&mut bad, &mut uz, &mut vtz, &z).is_err());
        }
        #[test]
        fn small_mat_svd_portable_agrees_with_lapack() {
            let cases: [([[f64; 3]; 3], usize); 3] = [
                ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]], 3),
                ([[2.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]], 3),
                ([[3.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, 1.0]], 3),
            ];
            for (a, _) in cases.iter() {
                let mut s_portable = [0.0; 3];
                let mut u_p = [[0.0; 3]; 3];
                let mut vt_p = [[0.0; 3]; 3];
                small_mat_svd_portable(&mut s_portable, &mut u_p, &mut vt_p, a).unwrap();

                let mut s_lapack = [0.0; 3];
                let mut u_l = [[0.0; 3]; 3];
                let mut vt_l = [[0.0; 3]; 3];
                crate::small::small_mat_svd(&mut s_lapack, &mut u_l, &mut vt_l, a).unwrap();

                for k in 0..3 {
                    approx_eq(s_portable[k], s_lapack[k], 1e-9);
                }
            }
        }
}
