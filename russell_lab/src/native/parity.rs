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
use crate::internal::C_FALSE;
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
        lapack::eig::dhseqr(t.as_mut_data(), &mut wr, &mut wi, n);

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
