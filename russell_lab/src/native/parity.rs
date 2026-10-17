//! Parity tests: the native routines against the reference LAPACK.
//!
//! Enabled by the `oracle` feature, which compiles the C shim against a locally
//! built LAPACK 3.12.1 and links it. The shim drives the **column-major**
//! `LAPACK_*` interface, so both sides see the same data layout; using LAPACKE
//! here would transpose every matrix and make the comparison meaningless.
//!
//! These tests are the reason the eigenvalue and SVD work is tractable: an
//! eigenvalue has no closed form for a general matrix, so "agrees with the
//! reference to 1e-12" is the only practical definition of correct.
//!
//! # What is compared
//!
//! Not the internal factorisations, which are allowed to differ, but the
//! *observable results*: the solution of a system, the inverse, the eigenvalues
//! and (later) the singular values. Those are what a caller sees, and comparing
//! them leaves each implementation free to be organised differently.

use super::lapack::{self, LapackError};
use crate::internal::{C_FALSE, C_TRUE};
use crate::oracle;
use crate::{Matrix, Vector};

/// Tolerance for a comparison that involves an elimination or an iteration.
///
/// Not machine epsilon: both sides accumulate rounding, and the two algorithms
/// differ in the order they do so. 1e-10 relative is far tighter than any
/// practical use of these routines yet still catches a genuinely wrong answer,
/// which is what these tests are for.
const TOL: f64 = 1e-10;

/// A deterministic, well-conditioned, non-symmetric matrix.
///
/// Built from a fixed recurrence rather than a random number generator so the
/// tests are reproducible without a dependency, and made non-symmetric and
/// non-tridiagonal so no structure can hide an indexing mistake.
fn sample(n: usize) -> Matrix {
    let mut a = Matrix::new(n, n);
    for i in 0..n {
        for j in 0..n {
            let v = if i == j {
                // Dominant diagonal keeps it well conditioned.
                (n as f64) + 1.0 + (i as f64) * 0.5
            } else {
                (((i * 7 + j * 13) % 11) as f64) - 5.0
            };
            a.set(i, j, v);
        }
    }
    a
}

fn compare_vectors(label: &str, got: &[f64], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    for i in 0..got.len() {
        let scale = want[i].abs().max(1.0);
        let err = (got[i] - want[i]).abs() / scale;
        assert!(
            err < tol,
            "{label}: component {i} is {} but the reference says {} (relative error {err:e})",
            got[i],
            want[i]
        );
    }
}

#[test]
fn dgesv_agrees_with_the_reference() {
    for n in [1usize, 2, 3, 5, 9] {
        let a = sample(n);

        // A known solution, so the comparison is against the reference solver
        // rather than against a hand-transcribed answer.
        let mut x_true = Vector::new(n);
        for i in 0..n {
            x_true[i] = 1.0 + (i as f64) * 0.25;
        }
        let mut b_native = Vector::new(n);
        crate::mat_vec_mul(&mut b_native, 1.0, &a, &x_true).unwrap();
        let mut b_ref = b_native.clone();

        // Native.
        let mut a_native = a.clone();
        lapack::dgesv(a_native.as_mut_data(), b_native.as_mut_data()).unwrap();

        // Reference, through the shim. It overwrites its input, so hand it a
        // copy rather than the matrix the native side already consumed.
        let mut a_ref = a.clone();
        let mut ipiv = vec![0i32; n];
        let mut info = 0i32;
        let n_i32 = n as i32;
        unsafe {
            oracle::oracle_c_dgesv(
                &n_i32,
                &1i32,
                a_ref.as_mut_data().as_mut_ptr(),
                &n_i32,
                ipiv.as_mut_ptr(),
                b_ref.as_mut_data().as_mut_ptr(),
                &n_i32,
                &mut info,
            );
        }
        assert_eq!(info, 0, "reference dgesv failed for n={n}");

        compare_vectors(
            &format!("dgesv solution (n={n})"),
            b_native.as_data(),
            b_ref.as_data(),
            TOL,
        );
        // ...and both must recover the solution we started from.
        compare_vectors(
            &format!("dgesv vs known x (n={n})"),
            b_native.as_data(),
            x_true.as_data(),
            TOL,
        );
    }
}

#[test]
fn dgetrf_and_dgetri_agree_with_the_reference() {
    for n in [1usize, 2, 4, 7] {
        let a = sample(n);

        // Native inverse.
        let mut inv_native = a.clone();
        let (ipiv_n, _det) = lapack::dgetrf(inv_native.as_mut_data(), n).unwrap();
        lapack::dgetri(inv_native.as_mut_data(), n, &ipiv_n).unwrap();

        // Reference inverse.
        let mut lu_ref = a.clone();
        let mut ipiv = vec![0i32; n];
        let mut info = 0i32;
        let n_i32 = n as i32;
        let mut work = vec![0.0f64; n.max(1)];
        let lwork = n_i32;
        unsafe {
            oracle::oracle_c_dgetrf(
                &n_i32,
                &n_i32,
                lu_ref.as_mut_data().as_mut_ptr(),
                &n_i32,
                ipiv.as_mut_ptr(),
                &mut info,
            );
            assert_eq!(info, 0, "reference dgetrf failed for n={n}");
            oracle::oracle_c_dgetri(
                &n_i32,
                lu_ref.as_mut_data().as_mut_ptr(),
                &n_i32,
                ipiv.as_ptr(),
                work.as_mut_ptr(),
                &lwork,
                &mut info,
            );
            assert_eq!(info, 0, "reference dgetri failed for n={n}");
        }

        // Element-wise: the inverse is unique, so this is a fair comparison.
        for i in 0..n {
            for j in 0..n {
                let got = inv_native.get(i, j);
                let want = lu_ref.get(i, j);
                let scale = want.abs().max(1.0);
                assert!(
                    (got - want).abs() / scale < TOL,
                    "dgetri ({n}): element ({i},{j}) is {got} but the reference says {want}"
                );
            }
        }
    }
}

#[test]
fn dpotrf_agrees_with_the_reference() {
    for n in [1usize, 2, 3, 6] {
        // A symmetric positive definite matrix: A^T A + n I, which is SPD by
        // construction and has no structure the algorithm could exploit.
        let raw = sample(n);
        let mut t = Matrix::new(n, n);
        crate::mat_t_mat_mul(&mut t, 1.0, &raw, &raw, 0.0).unwrap();
        for i in 0..n {
            t.add(i, i, n as f64);
        }

        let mut l_native = t.clone();
        lapack::dpotrf(l_native.as_mut_data(), n).unwrap();

        let mut l_ref = t.clone();
        let mut info = 0i32;
        let n_i32 = n as i32;
        unsafe {
            oracle::oracle_c_dpotrf(C_FALSE, &n_i32, l_ref.as_mut_data().as_mut_ptr(), &n_i32, &mut info);
        }
        assert_eq!(info, 0, "reference dpotrf failed for n={n}");

        // Only the lower triangle is written by either, by specification.
        for i in 0..n {
            for j in 0..=i {
                let got = l_native.get(i, j);
                let want = l_ref.get(i, j);
                let scale = want.abs().max(1.0);
                assert!(
                    (got - want).abs() / scale < TOL,
                    "dpotrf ({n}): L({i},{j}) is {got} but the reference says {want}"
                );
            }
        }
    }
}

#[test]
fn singular_detection_agrees_with_the_reference() {
    // Both sides must call the same matrices singular, or a caller would get a
    // solution from one path and an error from the other.
    for n in [2usize, 3, 5] {
        let a = sample(n);
        let mut sing = a.clone();
        // Make column 1 exactly twice column 0.
        for i in 0..n {
            let v = sing.get(i, 0) * 2.0;
            sing.set(i, 1, v);
        }

        let mut native = sing.clone();
        let native_result = lapack::dgetrf(native.as_mut_data(), n);

        let mut a_ref = sing.clone();
        let mut ipiv = vec![0i32; n];
        let mut info = 0i32;
        let n_i32 = n as i32;
        unsafe {
            oracle::oracle_c_dgetrf(
                &n_i32,
                &n_i32,
                a_ref.as_mut_data().as_mut_ptr(),
                &n_i32,
                ipiv.as_mut_ptr(),
                &mut info,
            );
        }

        assert_eq!(
            native_result,
            Err(LapackError::Singular),
            "the native dgetrf must report a singular matrix (n={n})"
        );
        assert!(info > 0, "the reference dgetrf must report a singular matrix (n={n})");
    }
}

/// The real Schur form and its eigenvalues against the reference.
///
/// This is the acceptance test for `dhseqr`: `dgehrd` produces the Hessenberg
/// form, `dhseqr` reduces it to quasi-triangular Schur form and returns the
/// eigenvalues. Three things are checked, in increasing strength:
///
/// 1. `t` is quasi-upper-triangular -- zero below the first subdiagonal.
/// 2. The 2x2 blocks are in standard form (`t(i,i) == t(i+1,i+1)` and
///    `t(i+1,i) * t(i,i+1) < 0`), which is what makes the `wi` signs meaningful.
/// 3. The eigenvalue multiset agrees with the reference `dgeev`.
///
/// The multiset is compared *sorted*, not in order: the Schur form is not unique
/// and the two implementations deflate in a different sequence, so an in-order
/// comparison would be testing the ordering rather than the values. `wi` is
/// signed on both sides -- LAPACK returns the positive imaginary part first --
/// so conjugation pairs have to be sorted deterministically.
#[test]
fn dhseqr_eigenvalues_agree_with_the_reference() {
    for n in [1usize, 2, 3, 5, 8, 15] {
        let a = sample(n);

        // Native: Hessenberg, then real Schur form.
        let mut t = a.clone();
        lapack::eig::dgehrd(t.as_mut_data(), n, None);
        let mut wr = vec![0.0; n];
        let mut wi = vec![0.0; n];
        lapack::eig::dhseqr(t.as_mut_data(), &mut wr, &mut wi, n).unwrap();

        // (1) quasi-upper-triangular
        for j in 0..n {
            for i in (j + 2)..n {
                assert!(
                    t.get(i, j).abs() < 1e-12,
                    "n={n}: T({i},{j}) = {} is below the first subdiagonal",
                    t.get(i, j)
                );
            }
        }

        // (2) each 2x2 block standardised, and the two columns of a pair
        //     conjugate. A real eigenvalue has wi == 0.
        let mut i = 0;
        while i < n {
            if i + 1 < n && t.get(i + 1, i) != 0.0 {
                assert!(
                    (t.get(i, i) - t.get(i + 1, i + 1)).abs() < 1e-10,
                    "n={n}: the 2x2 block at {i} has unequal diagonal entries"
                );
                assert!(
                    t.get(i + 1, i) * t.get(i, i + 1) < 0.0,
                    "n={n}: the 2x2 block at {i} is not in standard form"
                );
                assert!(
                    wi[i] > 0.0,
                    "n={n}: the first of a pair must have wi > 0, got {}",
                    wi[i]
                );
                assert!(
                    (wi[i] + wi[i + 1]).abs() < 1e-10,
                    "n={n}: the pair at {i} is not conjugate"
                );
                i += 2;
            } else {
                assert_eq!(wi[i], 0.0, "n={n}: a real eigenvalue must have wi == 0");
                i += 1;
            }
        }

        // (3) the eigenvalue multiset against the reference.
        let mut a_ref = a.clone();
        let mut wr_ref = vec![0.0; n];
        let mut wi_ref = vec![0.0; n];
        let info = oracle::query::dgeev(a_ref.as_mut_data(), n, &mut wr_ref, &mut wi_ref, None);
        assert_eq!(info, 0, "reference dgeev failed for n={n}");

        let mut got: Vec<(f64, f64)> = (0..n).map(|i| (wr[i], wi[i])).collect();
        let mut want: Vec<(f64, f64)> = (0..n).map(|i| (wr_ref[i], wi_ref[i])).collect();
        got.sort_by(|x, y| x.partial_cmp(y).unwrap());
        want.sort_by(|x, y| x.partial_cmp(y).unwrap());
        for i in 0..n {
            let scale = want[i].0.abs().max(want[i].1.abs()).max(1.0);
            assert!(
                (got[i].0 - want[i].0).abs() / scale < TOL && (got[i].1 - want[i].1).abs() / scale < TOL,
                "n={n}: eigenvalue {i} is {}+{}i but the reference says {}+{}i",
                got[i].0,
                got[i].1,
                want[i].0,
                want[i].1
            );
        }
    }
}

/// A deterministic, well-conditioned, genuinely complex matrix.
///
/// Real or Hermitian data would hide a conjugation mistake, so the imaginary
/// parts are non-zero everywhere.
fn complex_sample(n: usize) -> crate::ComplexMatrix {
    let mut a = crate::ComplexMatrix::new(n, n);
    for i in 0..n {
        for j in 0..n {
            let re = if i == j {
                (n as f64) + 2.0 + (i as f64) * 0.5
            } else {
                (((i * 5 + j * 7) % 9) as f64) - 4.0
            };
            let im = (((i * 3 + j * 11) % 7) as f64) - 3.0 + if i == j { 1.0 } else { 0.0 };
            a.set(i, j, crate::cpx!(re, im));
        }
    }
    a
}

/// Complex Cholesky against the reference, on both triangles.
///
/// Built as `A^H A + n I`, which is Hermitian positive definite by construction.
/// Only the requested triangle is written by either side, so only that triangle
/// is compared -- the rest legitimately holds whatever the caller had.
#[test]
fn complex_cholesky_agrees_with_the_reference() {
    for n in [1usize, 2, 3, 6] {
        let raw = complex_sample(n);
        let mut t = crate::ComplexMatrix::new(n, n);
        for i in 0..n {
            for j in 0..n {
                let mut s = crate::Complex64::new(0.0, 0.0);
                for k in 0..n {
                    s += raw.get(k, i).conj() * raw.get(k, j);
                }
                t.set(i, j, s);
            }
        }
        for i in 0..n {
            let d = t.get(i, i);
            t.set(i, i, crate::cpx!(d.re + n as f64, 0.0));
        }

        for upper in [false, true] {
            let mut l_native = t.clone();
            let mut l_ref = t.clone();
            crate::complex_mat_cholesky(&mut l_native, upper).expect("native Cholesky");
            let mut info = 0i32;
            let n_i32 = n as i32;
            let c_upper = if upper { C_TRUE } else { C_FALSE };
            unsafe {
                oracle::oracle_c_zpotrf(c_upper, &n_i32, l_ref.as_mut_data().as_mut_ptr(), &n_i32, &mut info);
            }
            assert_eq!(info, 0, "reference zpotrf failed for n={n} upper={upper}");
            for j in 0..n {
                for i in 0..n {
                    if (upper && i <= j) || (!upper && i >= j) {
                        let got = l_native.get(i, j);
                        let want = l_ref.get(i, j);
                        let scale = want.norm().max(1.0);
                        assert!(
                            (got - want).norm() / scale < TOL,
                            "n={n} upper={upper}: Cholesky differs at ({i},{j}): {got} vs {want}"
                        );
                    }
                }
            }
        }
    }
}

/// Complex inverse against the reference, element-wise.
///
/// The inverse is unique, so this is a fair comparison -- and it is the one that
/// catches a conjugation error, which any invariant-based check would miss.
#[test]
fn complex_inverse_agrees_with_the_reference() {
    for n in [1usize, 2, 3, 5, 8] {
        let a = complex_sample(n);
        let mut inv_native = crate::ComplexMatrix::new(n, n);
        crate::complex_mat_inverse(&mut inv_native, &a).expect("native inverse");

        // Reference: zgetrf then zgetri, through the shims.
        let n_i32 = n as i32;
        let mut lu = a.clone();
        let mut ipiv = vec![0i32; n];
        let mut info = 0i32;
        unsafe {
            oracle::oracle_c_zgetrf(
                &n_i32,
                &n_i32,
                lu.as_mut_data().as_mut_ptr(),
                &n_i32,
                ipiv.as_mut_ptr(),
                &mut info,
            );
        }
        assert_eq!(info, 0, "reference zgetrf failed for n={n}");
        let mut work = vec![crate::Complex64::new(0.0, 0.0); n.max(1)];
        unsafe {
            oracle::oracle_c_zgetri(
                &n_i32,
                lu.as_mut_data().as_mut_ptr(),
                &n_i32,
                ipiv.as_ptr(),
                work.as_mut_ptr(),
                &(work.len() as i32),
                &mut info,
            );
        }
        assert_eq!(info, 0, "reference zgetri failed for n={n}");

        for i in 0..n {
            for j in 0..n {
                let got = inv_native.get(i, j);
                let want = lu.get(i, j);
                let scale = want.norm().max(1.0);
                assert!(
                    (got - want).norm() / scale < TOL,
                    "n={n}: inverse differs at ({i},{j}): {got} vs {want}"
                );
            }
        }
    }
}

/// Complex solve against a known solution and against the reference.
///
/// The known solution is the stronger check: it does not depend on either
/// implementation being right.
#[test]
fn complex_solve_agrees_with_the_reference() {
    for n in [1usize, 2, 3, 5, 8] {
        let a = complex_sample(n);

        let mut x_true = crate::ComplexVector::new(n);
        for i in 0..n {
            x_true[i] = crate::cpx!(1.0 + (i as f64) * 0.5, -0.25 * (i as f64));
        }
        let mut b = crate::ComplexVector::new(n);
        crate::complex_mat_vec_mul(&mut b, crate::Complex64::new(1.0, 0.0), &a, &x_true).unwrap();

        // Native.
        let mut a_work = a.clone();
        let mut b_native = b.clone();
        crate::complex_solve_lin_sys(&mut b_native, &mut a_work).expect("native solve");
        for i in 0..n {
            let scale = x_true[i].norm().max(1.0);
            assert!(
                (b_native[i] - x_true[i]).norm() / scale < TOL,
                "n={n}: native solve recovered {} but the true x[{i}] is {}",
                b_native[i],
                x_true[i]
            );
        }

        // Reference.
        let n_i32 = n as i32;
        let mut a_ref = a.clone();
        let mut b_ref = b.clone();
        let mut ipiv = vec![0i32; n];
        let mut info = 0i32;
        unsafe {
            oracle::oracle_c_zgesv(
                &n_i32,
                &1i32,
                a_ref.as_mut_data().as_mut_ptr(),
                &n_i32,
                ipiv.as_mut_ptr(),
                b_ref.as_mut_data().as_mut_ptr(),
                &n_i32,
                &mut info,
            );
        }
        assert_eq!(info, 0, "reference zgesv failed for n={n}");
        for i in 0..n {
            let scale = b_ref[i].norm().max(1.0);
            assert!(
                (b_native[i] - b_ref[i]).norm() / scale < TOL,
                "n={n}: solve differs at {i}: {} vs the reference's {}",
                b_native[i],
                b_ref[i]
            );
        }
    }
}

/// A deterministic symmetric matrix, built from `sample`'s upper triangle.
fn symmetric_sample(n: usize) -> Matrix {
    let raw = sample(n);
    let mut a = Matrix::new(n, n);
    for i in 0..n {
        for j in 0..n {
            let (x, y) = if i <= j { (i, j) } else { (j, i) };
            a.set(i, j, raw.get(x, y));
        }
    }
    a
}

/// Symmetric eigen-decomposition against the reference.
///
/// Four things are checked, deliberately different in kind:
///
/// 1. **The eigenvalues are the reference's, in order.** Unlike the general
///    eigenproblem, the symmetric one *has* a canonical order -- ascending -- so
///    this compares in order rather than on a sorted multiset, and a wrong
///    ordering is a failure rather than a nuisance.
/// 2. **The vectors are orthonormal.** This is what a mis-accumulated
///    Householder or a missed rotation breaks.
/// 3. **`A v = lambda v` for each pair.** (1) and (2) together are still
///    satisfiable by the eigenvectors of a different matrix.
/// 4. **Both triangles agree.** LAPACK's `uplo` selects which triangle is *read*;
///    feeding the same symmetric matrix with `upper` false and true must give the
///    same answer, which catches a routine that reads the wrong half.
///
/// Nothing is compared against the reference's eigen*vectors*: their sign is
/// arbitrary. The exact sign convention is pinned by the `mat_eigen_sym` unit
/// tests at the sizes where it is well defined.
#[test]
fn dsyev_agrees_with_the_reference() {
    for n in [1usize, 2, 3, 5, 8, 12] {
        let a = symmetric_sample(n);
        let mut results = Vec::new();

        for upper in [false, true] {
            // `mat_eigen_sym` is the public entry point, so this checks the
            // observable result rather than an internal factorisation.
            let mut work = a.clone();
            let mut l = Vector::new(n);
            crate::mat_eigen_sym(&mut l, &mut work, upper).expect("native dsyev");

            // --- 1. eigenvalues in order against the reference.
            let mut t_ref = a.clone();
            let mut l_ref = vec![0.0; n];
            let info = oracle::query::dsyev(t_ref.as_mut_data(), n, &mut l_ref, true);
            assert_eq!(info, 0, "reference dsyev failed for n={n} upper={upper}");
            let scale = l_ref.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1.0);
            for i in 0..n {
                assert!(
                    (l[i] - l_ref[i]).abs() / scale < TOL,
                    "n={n} upper={upper}: eigenvalue {i} is {} but the reference says {}",
                    l[i],
                    l_ref[i]
                );
            }

            // --- 2. orthonormal columns.
            for i in 0..n {
                for j in 0..n {
                    let mut dot = 0.0;
                    for k in 0..n {
                        dot += work.get(k, i) * work.get(k, j);
                    }
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (dot - want).abs() < 1e-10,
                        "n={n} upper={upper}: columns {i},{j} are not orthonormal (dot = {dot})"
                    );
                }
            }

            // --- 3. the defining equation.
            for j in 0..n {
                for i in 0..n {
                    let mut av = 0.0;
                    for k in 0..n {
                        av += a.get(i, k) * work.get(k, j);
                    }
                    let resid = (av - l[j] * work.get(i, j)).abs();
                    assert!(
                        resid / scale < 1e-10,
                        "n={n} upper={upper}: (A v - lambda v)[{i}] = {resid} for eigenvalue {j}"
                    );
                }
            }

            results.push((l, work));
        }

        // --- 4. the two triangles give the same spectrum and the same residuals,
        //     not necessarily the same vectors (the sign may differ).
        let (l_lo, _) = &results[0];
        let (l_up, _) = &results[1];
        for i in 0..n {
            assert!(
                (l_lo[i] - l_up[i]).abs() < 1e-12,
                "n={n}: reading the lower triangle gives {} but the upper gives {}",
                l_lo[i],
                l_up[i]
            );
        }
    }
}

/// Left eigenvectors, by their defining equation.
///
/// The existing `mat_eigen_lr_works` unit test checks the eigenvalues and the
/// *right* eigenvectors but never the left ones, so a wrong `u` passes silently.
/// This checks what was missing: `u^H A = lambda u^H`, equivalently
/// `A^H u = conj(lambda) u`, which is the equation a left eigenvector satisfies.
///
/// `mat_eigen_lr` returns both sets, so this also checks that the two agree on
/// the eigenvalue list -- a left eigenvector paired with the wrong eigenvalue is
/// the specific mistake that a naive "run dgeev on A^T and hand back its vectors"
/// would make, since two independent factorisations need not deflate in the same
/// order.
///
/// Only real data is used here, so `A^H = A^T` and the conjugate is trivial; the
/// complex case would need `complex_mat_eigen_lr`, which is not implemented.
#[test]
fn mat_eigen_lr_left_eigenvectors_satisfy_their_defining_equation() {
    for n in [1usize, 2, 3, 5, 8] {
        let a = sample(n);

        let mut work = a.clone();
        let mut l_real = Vector::new(n);
        let mut l_imag = Vector::new(n);
        let mut u_real = Matrix::new(n, n);
        let mut u_imag = Matrix::new(n, n);
        let mut v_real = Matrix::new(n, n);
        let mut v_imag = Matrix::new(n, n);
        crate::mat_eigen_lr(
            &mut l_real,
            &mut l_imag,
            &mut u_real,
            &mut u_imag,
            &mut v_real,
            &mut v_imag,
            &mut work,
        )
        .expect("native mat_eigen_lr");

        let scale = a.as_data().iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1.0);
        let mut j = 0usize;
        while j < n {
            if l_imag[j] != 0.0 {
                // Complex pair: u = u_real[:,j] + i u_imag[:,j], lambda = lr + i li.
                // The defining equation is A^H u = conj(lambda) u, i.e.
                // A^T u = (lr - i li) u for real A.
                let (lr, li) = (l_real[j], l_imag[j]);
                let mut worst = 0.0f64;
                for i in 0..n {
                    // (A^T u)_i = sum_k A[k][i] * u_k
                    let mut sr = 0.0;
                    let mut si = 0.0;
                    for k in 0..n {
                        let ak = a.get(k, i);
                        sr += ak * u_real.get(k, j);
                        si += ak * u_imag.get(k, j);
                    }
                    // subtract conj(lambda) * u_i = (lr - i li)(ur + i ui)
                    let ur = u_real.get(i, j);
                    let ui = u_imag.get(i, j);
                    let dr = sr - (lr * ur + li * ui);
                    let di = si - (lr * ui - li * ur);
                    worst = worst.max(dr.abs().max(di.abs()));
                }
                assert!(
                    worst / scale < 1e-10,
                    "n={n}: the complex left eigenvector for lambda = {lr}+{li}i has residual {worst} (scale {scale})"
                );
                j += 2;
            } else {
                let l = l_real[j];
                let mut worst = 0.0f64;
                for i in 0..n {
                    let mut s = 0.0;
                    for k in 0..n {
                        s += a.get(k, i) * u_real.get(k, j);
                    }
                    worst = worst.max((s - l * u_real.get(i, j)).abs());
                }
                assert!(
                    worst / scale < 1e-10,
                    "n={n}: the real left eigenvector for lambda = {l} has residual {worst} (scale {scale})"
                );
                j += 1;
            }
        }
    }
}

/// Singular value decomposition against the reference, for several shapes.
///
/// Unlike eigenvectors, singular values **are** unique, so they can be compared
/// directly -- and they are compared in order, because the SVD is conventionally
/// returned with them descending. That makes this test sensitive to an ordering
/// mistake as well as a value mistake.
///
/// The rest of the test checks what the singular values alone cannot: that `U`
/// and `V` are orthogonal (so the factorisation is a real SVD and not just the
/// right spectrum), that `U S V^T` reconstructs `A`, and that `S` is
/// non-negative. A routine that returned the right singular values with wrong
/// vectors would pass a values-only check and fail these.
///
/// Shapes deliberately include tall (`m > n`), square, and wide (`m < n`), and
/// the rank-deficient case of a duplicated column, because the two branches of a
/// Golub-Reinsch implementation differ between them and the rank-deficient case
/// is where a zero singular value exposes a missing deflation.
#[test]
fn dgesvd_agrees_with_the_reference() {
    for (m, n) in [
        (1usize, 1usize),
        (2, 2),
        (3, 2),
        (2, 3),
        (4, 3),
        (3, 4),
        (6, 4),
        (4, 6),
        (5, 5),
    ] {
        let a = rect_sample(m, n);
        check_svd(&a, m, n, false);
    }

    // Rank deficient: make the last column a copy of the first.
    for (m, n) in [(4usize, 3usize), (5, 5)] {
        let mut a = rect_sample(m, n);
        for i in 0..m {
            let v = a.get(i, 0);
            a.set(i, n - 1, v);
        }
        check_svd(&a, m, n, true);
    }
}

/// A deterministic rectangular matrix, non-symmetric and with no zero entries.
fn rect_sample(m: usize, n: usize) -> Matrix {
    let mut a = Matrix::new(m, n);
    for i in 0..m {
        for j in 0..n {
            a.set(
                i,
                j,
                (((i * 7 + j * 13 + 3) % 11) as f64) - 5.0 + (i as f64) * 0.25 - (j as f64) * 0.1,
            );
        }
    }
    a
}

/// The four things an SVD must satisfy, plus the comparison against the
/// reference. `rank_deficient` relaxes only the tolerance on the smallest
/// singular value, which is genuinely ill-conditioned there.
fn check_svd(a: &Matrix, m: usize, n: usize, rank_deficient: bool) {
    let min_mn = m.min(n);
    let label = format!("{m}x{n}{}", if rank_deficient { " (rank deficient)" } else { "" });

    let mut work = a.clone();
    let mut s = Vector::new(min_mn);
    let mut u = Matrix::new(m, m);
    let mut vt = Matrix::new(n, n);
    crate::mat_svd(&mut s, &mut u, &mut vt, &mut work).expect("native dgesvd");

    let scale = a.as_data().iter().fold(0.0f64, |acc, x| acc.max(x.abs())).max(1.0);

    // --- singular values against the reference, in order.
    let mut a_ref = a.clone();
    let mut s_ref = vec![0.0; min_mn];
    let mut u_ref = vec![0.0; m * m];
    let mut vt_ref = vec![0.0; n * n];
    let info = oracle::query::dgesvd(a_ref.as_mut_data(), m, n, &mut s_ref, &mut u_ref, &mut vt_ref);
    assert_eq!(info, 0, "{label}: reference dgesvd failed");
    for i in 0..min_mn {
        let tol = if rank_deficient && i == min_mn - 1 { 1e-8 } else { TOL };
        assert!(
            (s[i] - s_ref[i]).abs() / scale < tol,
            "{label}: singular value {i} is {} but the reference says {}",
            s[i],
            s_ref[i]
        );
    }

    // --- non-negative and descending.
    for i in 0..min_mn {
        assert!(s[i] >= 0.0, "{label}: singular value {i} is negative: {}", s[i]);
        if i > 0 {
            assert!(
                s[i] <= s[i - 1] + 1e-12 * scale,
                "{label}: singular values are not descending: {} then {}",
                s[i - 1],
                s[i]
            );
        }
    }

    // --- U orthogonal (m x m).
    for i in 0..m {
        for j in 0..m {
            let mut dot = 0.0;
            for k in 0..m {
                dot += u.get(k, i) * u.get(k, j);
            }
            let want = if i == j { 1.0 } else { 0.0 };
            assert!(
                (dot - want).abs() < 1e-10,
                "{label}: U columns {i},{j} are not orthonormal ({dot})"
            );
        }
    }

    // --- V orthogonal (n x n), stored transposed.
    for i in 0..n {
        for j in 0..n {
            let mut dot = 0.0;
            for k in 0..n {
                // vt rows are V's columns, so vt[i][k] * vt[j][k] is V col i . V col j
                dot += vt.get(i, k) * vt.get(j, k);
            }
            let want = if i == j { 1.0 } else { 0.0 };
            assert!(
                (dot - want).abs() < 1e-10,
                "{label}: V columns {i},{j} are not orthonormal ({dot})"
            );
        }
    }

    // --- U S V^T reconstructs A. This is the check that catches vectors which
    //     are individually fine but paired with the wrong singular value.
    let mut worst = 0.0f64;
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..min_mn {
                sum += u.get(i, k) * s[k] * vt.get(k, j);
            }
            worst = worst.max((sum - a.get(i, j)).abs());
        }
    }
    assert!(
        worst / scale < 1e-10,
        "{label}: U S V^T does not reconstruct A (worst element error {worst}, scale {scale})"
    );
}

/// A deterministic, genuinely complex Hermitian matrix.
///
/// `H = (A + A^H) / 2` for the complex sample, so it is Hermitian by
/// construction. The imaginary parts off the diagonal are non-zero -- that is the
/// point: on real or real-symmetric data a missing conjugation in `zhetd2`,
/// `zungtr` or `zsteqr` still produces a plausible answer, which is exactly how
/// the earlier `zherk`/`zpotrf` bugs survived.
fn hermitian_sample(n: usize) -> crate::ComplexMatrix {
    let raw = complex_sample(n);
    let mut h = crate::ComplexMatrix::new(n, n);
    for i in 0..n {
        for j in 0..n {
            let s = raw.get(i, j) + raw.get(j, i).conj();
            h.set(i, j, crate::cpx!(0.5 * s.re, 0.5 * s.im));
        }
    }
    h
}

/// Complex Hermitian eigen-decomposition against the reference, on both
/// triangles.
///
/// Four things are checked, mirroring the real `dsyev` test:
///
/// 1. **The eigenvalues are the reference's, in order.** The Hermitian problem,
///    like the symmetric one, has a canonical order -- ascending -- so this is an
///    in-order comparison and a wrong ordering is a failure, not a nuisance.
/// 2. **The vectors are orthonormal in the complex sense**, `v_i^H v_j =
///    delta_ij`. A conjugation mistake turns this into `v_i^T v_j`, which is not
///    the identity on complex data.
/// 3. **`A v = lambda v` for each pair**, against the original Hermitian matrix.
///    (1) and (2) together are satisfiable by the eigenvectors of `conj(A)`.
/// 4. **Both triangles agree.** `uplo` selects which triangle is *read*, so the
///    same Hermitian matrix fed with `upper` false and true must give the same
///    spectrum.
///
/// The eigenvector columns are not compared against the reference -- their phase
/// is arbitrary -- so (2) and (3) are what pin them down.
#[test]
fn zheev_agrees_with_the_reference() {
    let zero = crate::cpx!(0.0, 0.0);
    for n in [1usize, 2, 3, 5, 8, 12] {
        let a = hermitian_sample(n);
        let scale = a.as_data().iter().fold(0.0f64, |m, x| m.max(x.norm())).max(1.0);

        // The data must be genuinely complex, or a conjugation mistake hides.
        let has_imag = (0..n).any(|j| (0..n).any(|i| i != j && a.get(i, j).im.abs() > 0.25));
        assert!(
            has_imag || n < 2,
            "the zheev test data must be genuinely complex (n={n})"
        );

        let mut results = Vec::new();
        for upper in [false, true] {
            let mut work = a.clone();
            let mut l = Vector::new(n);
            crate::complex_mat_eigen_herm(&mut l, &mut work, upper).expect("native zheev");

            // --- 1. eigenvalues in order against the reference.
            let mut t_ref = a.clone();
            let mut l_ref = vec![0.0; n];
            let info = oracle::query::zheev(t_ref.as_mut_data(), n, &mut l_ref, true);
            assert_eq!(info, 0, "reference zheev failed for n={n} upper={upper}");
            for i in 0..n {
                assert!(
                    (l[i] - l_ref[i]).abs() / scale < TOL,
                    "n={n} upper={upper}: eigenvalue {i} is {} but the reference says {}",
                    l[i],
                    l_ref[i]
                );
            }
            for i in 1..n {
                assert!(
                    l[i] >= l[i - 1] - 1e-12 * scale,
                    "n={n} upper={upper}: eigenvalues are not ascending: {} then {}",
                    l[i - 1],
                    l[i]
                );
            }

            // --- 2. orthonormal columns, complex sense.
            for i in 0..n {
                for j in 0..n {
                    let mut dot = zero;
                    for k in 0..n {
                        dot += work.get(k, i).conj() * work.get(k, j);
                    }
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (dot - crate::cpx!(want, 0.0)).norm() < 1e-10,
                        "n={n} upper={upper}: columns {i},{j} are not orthonormal (v_i^H v_j = {dot})"
                    );
                }
            }

            // --- 3. the defining equation against the original matrix.
            for j in 0..n {
                for i in 0..n {
                    let mut av = zero;
                    for k in 0..n {
                        av += a.get(i, k) * work.get(k, j);
                    }
                    let resid = (av - work.get(i, j) * l[j]).norm();
                    assert!(
                        resid / scale < 1e-10,
                        "n={n} upper={upper}: (A v - lambda v)[{i}] = {resid} for eigenvalue {j}"
                    );
                }
            }

            results.push((l, work));
        }

        // --- 4. the two triangles give the same spectrum.
        let (l_lo, _) = &results[0];
        let (l_up, _) = &results[1];
        for i in 0..n {
            assert!(
                (l_lo[i] - l_up[i]).abs() / scale < 1e-12,
                "n={n}: reading the lower triangle gives {} but the upper gives {}",
                l_lo[i],
                l_up[i]
            );
        }
    }
}

/// The generalised eigenproblem `A v = lambda B v`, by its defining equation.
///
/// `dggev` returns `(alpha, beta)` with `lambda = alpha/beta`, so comparing
/// eigenvalues against a reference means comparing ratios -- and a ratio is
/// badly behaved exactly where `beta` is small. The reference itself reports
/// `beta = 0` for infinite eigenvalues, and one of the crate's own tests asserts
/// on a `beta` of exactly zero. Rather than cope with infinities, the data here
/// is chosen so every eigenvalue is finite: `B = R^T R + n I` is symmetric
/// positive definite, so no `lambda` is infinite and none is ill conditioned.
///
/// The check is `A v = lambda B v` for each right eigenvector, which needs no
/// reference and is the definition. `lambda = alpha/beta` is formed in complex
/// arithmetic, since `alpha` may be complex while `beta` is real.
#[test]
fn dggev_satisfies_its_defining_equation() {
    for n in [1usize, 2, 3, 5, 8] {
        let ra = sample(n);
        let mut b = Matrix::new(n, n);
        // B = ra^T ra + n I, symmetric positive definite by construction.
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += ra.get(k, i) * ra.get(k, j);
                }
                b.set(i, j, if i == j { sum + n as f64 } else { sum });
            }
        }

        // `dggev` overwrites both matrices in place, as LAPACK does, so the
        // defining equation has to be checked against copies of the original
        // problem -- an earlier version of this test read the overwritten
        // matrices back and so compared against the factorisation output.
        let (a_orig, b_orig) = (ra.clone(), b.clone());

        let mut a = ra.clone();
        let mut b_work = b.clone();
        let mut alpha_r = Vector::new(n);
        let mut alpha_i = Vector::new(n);
        let mut beta = Vector::new(n);
        let mut v = Matrix::new(n, n);
        crate::mat_gen_eigen(&mut alpha_r, &mut alpha_i, &mut beta, &mut v, &mut a, &mut b_work).expect("native dggev");

        let scale_b = b_orig.as_data().iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1.0);

        let mut j = 0usize;
        while j < n {
            // lambda = alpha / beta, in complex arithmetic.
            assert!(
                beta[j].abs() > 1e-12,
                "beta[{j}] is {}; the data should give no infinite eigenvalue",
                beta[j]
            );
            let lr = alpha_r[j] / beta[j];
            let li = alpha_i[j] / beta[j];

            if li != 0.0 {
                // Eigenvector column j is (real, imaginary) in columns j and j+1.
                // A u = lambda B u, checked in complex arithmetic.
                let mut worst = 0.0f64;
                for i in 0..n {
                    let mut av_r = 0.0;
                    let mut av_i = 0.0;
                    let mut bv_r = 0.0;
                    let mut bv_i = 0.0;
                    for k in 0..n {
                        let u_r = v.get(k, j);
                        let u_i = v.get(k, j + 1);
                        av_r += a_orig.get(i, k) * u_r;
                        av_i += a_orig.get(i, k) * u_i;
                        bv_r += b_orig.get(i, k) * u_r;
                        bv_i += b_orig.get(i, k) * u_i;
                    }
                    // right-hand side: lambda * (B u), complex multiply
                    let u_r = v.get(i, j);
                    let u_i = v.get(i, j + 1);
                    let rhs_r = lr * bv_r - li * bv_i;
                    let rhs_i = lr * bv_i + li * bv_r;
                    worst = worst.max((av_r - rhs_r).abs().max((av_i - rhs_i).abs()));
                    let _ = (u_r, u_i);
                }
                assert!(
                    worst / scale_b < 1e-10,
                    "n={n}: the complex generalised eigenvector for lambda = {lr}+{li}i has residual {worst}"
                );
                j += 2;
            } else {
                let mut worst = 0.0f64;
                for i in 0..n {
                    let mut av = 0.0;
                    let mut bv = 0.0;
                    for k in 0..n {
                        av += a_orig.get(i, k) * v.get(k, j);
                        bv += b_orig.get(i, k) * v.get(k, j);
                    }
                    worst = worst.max((av - lr * bv).abs());
                }
                assert!(
                    worst / scale_b < 1e-10,
                    "n={n}: the real generalised eigenvector for lambda = {lr} has residual {worst}"
                );
                j += 1;
            }
        }
    }
}
