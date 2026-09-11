//! Compares the accuracy of the available eigenvalue methods on the benchmark
//! cases described in the Habera-Zilian and Harari-Albocher papers.
//!
//! The benchmark paths are:
//!
//! * `D1 = diag(1, 1, 1 + δ)` — a double eigenvalue moving towards a triple
//!   eigenvalue (`J2 → 0` and `J3 → 0`)
//! * `D2 = diag(-1, 1, 1 + δ)` — a double eigenvalue (the discriminant `Δ → 0`
//!   while `J2` and `J3` stay finite)
//!
//! The matrices are built as `A = U ⋅ diag(d) ⋅ Uᵀ` with the orthogonal
//! transformation `U_symm` used in the papers. The prescribed eigenvalues
//! `d` are used as the reference.

use russell_tensor::{EigMethod, Spectral2, StrError, Tensor2};
use std::f64::consts::PI;

fn main() -> Result<(), StrError> {
    let u = u_symm();
    let deltas = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14];

    for (label, is_d1) in [
        ("D1: diag(1, 1, 1 + δ)  (double → triple eigenvalue)", true),
        ("D2: diag(-1, 1, 1 + δ) (double eigenvalue)", false),
    ] {
        println!("\n{}", label);
        println!(
            "{:>8}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
            "δ", "HZ", "HA22", "HA23", "Jacobi", "naive"
        );
        for &delta in &deltas {
            let d = if is_d1 {
                [1.0, 1.0, 1.0 + delta]
            } else {
                [-1.0, 1.0, 1.0 + delta]
            };
            let a = build_a(&u, &d);
            let tt = Tensor2::<6>::from_std_matrix(&a)?;
            let mut exact = d;
            exact.sort_by(|x, y| x.partial_cmp(y).unwrap());

            let mut errs = [0.0; 5];
            let methods = [
                EigMethod::AnalyticalHZ,
                EigMethod::AnalyticalHA22,
                EigMethod::AnalyticalHA23,
                EigMethod::Iterative,
            ];
            for (i, method) in methods.iter().enumerate() {
                let mut spec = Spectral2::new();
                spec.decompose_mx(&tt, *method)?;
                errs[i] = max_error(&spec.lam, &exact);
            }
            errs[4] = max_error(&naive_eigvals(&a), &exact);

            println!(
                "{:>8.0e}  {:>10.2e}  {:>10.2e}  {:>10.2e}  {:>10.2e}  {:>10.2e}",
                delta, errs[0], errs[1], errs[2], errs[3], errs[4]
            );
        }
    }
    Ok(())
}

/// Orthogonal transformation matrix from the papers
fn u_symm() -> [[f64; 3]; 3] {
    let r2 = f64::sqrt(2.0);
    [[1.0 / r2, -0.5, 0.5], [1.0 / r2, 0.5, -0.5], [0.0, 1.0 / r2, 1.0 / r2]]
}

/// Builds the symmetric matrix A = U ⋅ diag(d) ⋅ Uᵀ (and symmetrizes it)
fn build_a(u: &[[f64; 3]; 3], d: &[f64; 3]) -> [[f64; 3]; 3] {
    let mut a = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                a[i][j] += u[i][k] * d[k] * u[j][k];
            }
        }
    }
    for i in 0..3 {
        for j in (i + 1)..3 {
            let m = 0.5 * (a[i][j] + a[j][i]);
            a[i][j] = m;
            a[j][i] = m;
        }
    }
    a
}

/// Returns the maximum absolute difference between two (sorted) set of eigenvalues
fn max_error(w: &[f64; 3], exact: &[f64; 3]) -> f64 {
    let mut ws = *w;
    ws.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut e = 0.0;
    for i in 0..3 {
        let d = f64::abs(ws[i] - exact[i]);
        if d > e {
            e = d;
        }
    }
    e
}

/// Naive eigenvalue computation based on the (monomial) cubic formula
///
/// This is the unstable baseline analogous to `impl_naive.py` from the `eig3x3`
/// library: the deviatoric invariants and the discriminant are computed with the
/// naive monomial expressions, which suffer from catastrophic cancellation.
fn naive_eigvals(a: &[[f64; 3]; 3]) -> [f64; 3] {
    let i1 = a[0][0] + a[1][1] + a[2][2];
    let m = i1 / 3.0;
    let s = [
        [a[0][0] - m, a[0][1], a[0][2]],
        [a[1][0], a[1][1] - m, a[1][2]],
        [a[2][0], a[2][1], a[2][2] - m],
    ];
    // J2 = ½ tr(s²)
    let mut j2 = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            j2 += 0.5 * s[i][j] * s[j][i];
        }
    }
    // J3 = det(s)
    let j3 = s[0][0] * (s[1][1] * s[2][2] - s[1][2] * s[2][1]) - s[0][1] * (s[1][0] * s[2][2] - s[1][2] * s[2][0])
        + s[0][2] * (s[1][0] * s[2][1] - s[1][1] * s[2][0]);
    // discriminant (naive)
    let mut disc = 4.0 * j2 * j2 * j2 - 27.0 * j3 * j3;
    if disc < 0.0 {
        disc = 0.0;
    }
    let j2 = if j2 < 0.0 { 0.0 } else { j2 };
    let phi = f64::atan2(f64::sqrt(27.0 * disc), 27.0 * j3);
    let sqrt_3j2 = f64::sqrt(3.0 * j2);
    let mut w = [0.0; 3];
    for k in 0..3 {
        let angle = (phi + 2.0 * PI * ((k + 1) as f64)) / 3.0;
        w[k] = (i1 + 2.0 * sqrt_3j2 * f64::cos(angle)) / 3.0;
    }
    w
}
