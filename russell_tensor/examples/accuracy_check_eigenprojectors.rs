//! Compares the reconstruction error of the eigenprojectors returned by the four
//! eigenvalue methods available in `EigenProjsT2::calculate_mx`.
//!
//! The symmetric tensors are built as `A = U ⋅ diag(d) ⋅ Uᵀ` with the orthogonal
//! transformation `U_sym` used in `accuracy_check_eigenvalues`, for diagonal cases
//! that approach the coalescent (two nearly equal eigenvalues) limit:
//!
//! * `D1 = diag(1, 1 + δ, 1 + 2δ)` — approaching a triple eigenvalue
//! * `D2 = diag(-1, 1, 1 + δ)` — approaching a double eigenvalue (the pair `1`, `1 + δ`)
//!
//! For each δ and method, the spectral reconstruction `A_rec = Σ_k λ_k P_k` is formed
//! and the maximum absolute error `max_m |A_m − A_rec,m|` is reported, with all values
//! in Kelvin-Mandel components.

use russell_tensor::{EigenProjsT2, EigenValMethod, StrError, Tensor2};

/// Eigenvalue methods, in benchmark order
const METHODS: [(&str, EigenValMethod); 4] = [
    ("HZ", EigenValMethod::AnalyticalHZ),
    ("HA22", EigenValMethod::AnalyticalHA22),
    ("HA23", EigenValMethod::AnalyticalHA23),
    ("Jacobi", EigenValMethod::Iterative),
];

fn main() -> Result<(), StrError> {
    let q = q_sym();
    let deltas = [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14];

    for (label, is_d1) in [
        ("D1: diag(1, 1 + δ, 1 + 2δ)  (approaching a triple eigenvalue)", true),
        ("D2: diag(-1, 1, 1 + δ)      (approaching a double eigenvalue)", false),
    ] {
        println!("\n{}", label);
        print!("{:>8}", "δ");
        for (name, _) in METHODS {
            print!("  {:>10}", name);
        }
        println!();
        for &delta in &deltas {
            let d = if is_d1 {
                [1.0, 1.0 + delta, 1.0 + 2.0 * delta]
            } else {
                [-1.0, 1.0, 1.0 + delta]
            };
            let a = build_a(&q, &d);
            let aa = Tensor2::<6>::from_std_matrix(&a)?;
            print!("{:>8.0e}", delta);
            for (_, method) in METHODS {
                print!("  {:>10.2e}", reconstruction_error(&aa, method)?);
            }
            println!();
        }
    }
    Ok(())
}

/// Returns the maximum absolute reconstruction error `max_m |A_m − Σ_k λ_k P_k,m|`
fn reconstruction_error(aa: &Tensor2<6>, method: EigenValMethod) -> Result<f64, StrError> {
    let mut eig = EigenProjsT2::new();
    let mut ll = [0.0; 3];
    let mut projs = [Tensor2::<6>::new(), Tensor2::<6>::new(), Tensor2::<6>::new()];
    eig.calculate_mx(&mut ll, &mut projs, aa, method)?;
    let mut max = 0.0;
    for m in 0..6 {
        let rec = ll[0] * projs[0].get(m) + ll[1] * projs[1].get(m) + ll[2] * projs[2].get(m);
        max = f64::max(max, f64::abs(aa.get(m) - rec));
    }
    Ok(max)
}

/// Orthogonal transformation matrix from the papers
fn q_sym() -> [[f64; 3]; 3] {
    let r2 = f64::sqrt(2.0);
    [[1.0 / r2, -0.5, 0.5], [1.0 / r2, 0.5, -0.5], [0.0, 1.0 / r2, 1.0 / r2]]
}

/// Builds the symmetric matrix A = U ⋅ diag(d) ⋅ Uᵀ (and symmetrizes it)
fn build_a(q: &[[f64; 3]; 3], d: &[f64; 3]) -> [[f64; 3]; 3] {
    let mut a = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                a[i][j] += q[i][k] * d[k] * q[j][k];
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
