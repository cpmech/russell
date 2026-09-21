//! Compares the derivatives of the eigenprojectors computed with
//! `EigenProjDerivsT2::calc_with_inv` (Miehe's inverse-based approach) and
//! `EigenProjDerivsT2::calc_with_char_poly` (Panteghini's characteristic-polynomial
//! approach), using the Habera-Zilian eigenvalue method (the default).
//!
//! The symmetric tensors are built as `A = U ⋅ diag(d) ⋅ Uᵀ` with the orthogonal
//! transformation `U_sym` used in `accuracy_check_eigenvalues`, for diagonal cases
//! that approach the coalescent (two nearly equal eigenvalues) limit:
//!
//! * `D1 = diag(1, 1 + δ, 1 + 2δ)` — approaching a triple eigenvalue
//! * `D2 = diag(-1, 1, 1 + δ)` — approaching a double eigenvalue (the pair `1`, `1 + δ`)
//!
//! The two functions are mathematically equivalent; for each δ the example reports
//! the maximum absolute difference between the two derivative tensors and that
//! difference relative to the largest derivative component. Cases where a method
//! rejects the tensor (coalescent, non-invertible, small `d[i]`, ...) are reported
//! per method.

use russell_tensor::{EigenProjDerivsT2, EigenValMethod, StrError, Tensor2, Tensor4};

// Expected output
// D1: diag(1, 1 + δ, 1 + 2δ)  (approaching a triple eigenvalue)
//        δ      max|ΔdP|           rel  status
//     1e-1      2.08e-13      4.16e-14  ok
//     1e-2      5.73e-11      1.15e-12  ok
//     1e-3       7.78e-7       1.56e-9  ok
//     1e-4             -             -  calc_with_inv failed: |d[i]| is nearly zero
//     1e-5             -             -  calc_with_inv failed: |d[i]| is nearly zero
//     1e-6             -             -  calc_with_inv failed: |d[i]| is nearly zero
//     1e-8             -             -  both failed: Failed due to spherical state (all equal eigenvalues)
//    1e-10             -             -  both failed: Failed due to spherical state (all equal eigenvalues)
//    1e-12             -             -  both failed: Failed due to spherical state (all equal eigenvalues)
//    1e-14             -             -  both failed: Failed due to spherical state (all equal eigenvalues)
//
// D2: diag(-1, 1, 1 + δ)      (approaching a double eigenvalue)
//        δ      max|ΔdP|           rel  status
//     1e-1      1.42e-14      2.84e-15  ok
//     1e-2      3.06e-13      6.11e-15  ok
//     1e-3      8.35e-11      1.67e-13  ok
//     1e-4       2.77e-9      5.55e-13  ok
//     1e-5       8.33e-7      1.67e-11  ok
//     1e-6       5.55e-5      1.11e-10  ok
//     1e-8             -             -  both failed: Failed due to two repeated eigenvalues
//    1e-10             -             -  both failed: Failed due to two repeated eigenvalues
//    1e-12             -             -  both failed: Failed due to two repeated eigenvalues
//    1e-14             -             -  both failed: Failed due to two repeated eigenvalues

/// Outcome of running both derivative methods on one tensor
enum Outcome {
    /// Both methods succeeded: (max absolute diff, max diff / max|dP|)
    Both(f64, f64),
    /// Only `calc_with_char_poly` succeeded
    OnlyCharPoly(StrError),
    /// Only `calc_with_inv` succeeded
    OnlyInv(StrError),
    /// Both methods rejected the tensor (message from `calc_with_inv`)
    None(StrError),
}

fn main() -> Result<(), StrError> {
    let method = EigenValMethod::AnalyticalHZ;
    let u = u_sym();
    let deltas = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14];

    for (label, is_d1) in [
        ("D1: diag(1, 1 + δ, 1 + 2δ)  (approaching a triple eigenvalue)", true),
        ("D2: diag(-1, 1, 1 + δ)      (approaching a double eigenvalue)", false),
    ] {
        println!("\n{}", label);
        println!("{:>8}  {:>12}  {:>12}  status", "δ", "max|ΔdP|", "rel");
        for &delta in &deltas {
            let d = if is_d1 {
                [1.0, 1.0 + delta, 1.0 + 2.0 * delta]
            } else {
                [-1.0, 1.0, 1.0 + delta]
            };
            let a = build_a(&u, &d);
            let aa = Tensor2::<6>::from_std_matrix(&a)?;
            match compare(&aa, method) {
                Outcome::Both(abs, rel) => println!("{:>8.0e}  {:>12.2e}  {:>12.2e}  ok", delta, abs, rel),
                Outcome::OnlyCharPoly(msg) => {
                    println!(
                        "{:>8.0e}  {:>12}  {:>12}  calc_with_inv failed: {}",
                        delta, "-", "-", msg
                    )
                }
                Outcome::OnlyInv(msg) => {
                    println!(
                        "{:>8.0e}  {:>12}  {:>12}  calc_with_char_poly failed: {}",
                        delta, "-", "-", msg
                    )
                }
                Outcome::None(msg) => println!("{:>8.0e}  {:>12}  {:>12}  both failed: {}", delta, "-", "-", msg),
            }
        }
    }
    Ok(())
}

/// Runs both derivative methods and compares the results
fn compare(aa: &Tensor2<6>, method: EigenValMethod) -> Outcome {
    let mut calc = EigenProjDerivsT2::new();
    let mut ll = [0.0; 3];
    let mut projs = [Tensor2::<6>::new(), Tensor2::<6>::new(), Tensor2::<6>::new()];
    let mut d_inv = [Tensor4::<6>::new(), Tensor4::<6>::new(), Tensor4::<6>::new()];
    let mut d_cp = [Tensor4::<6>::new(), Tensor4::<6>::new(), Tensor4::<6>::new()];

    let r_inv = calc.calc_with_inv(&mut ll, &mut projs, &mut d_inv, aa, method);
    let r_cp = calc.calc_with_char_poly(&mut ll, &mut projs, &mut d_cp, aa, method);

    match (r_inv, r_cp) {
        (Ok(()), Ok(())) => {
            let (abs, rel) = differences(&d_inv, &d_cp);
            Outcome::Both(abs, rel)
        }
        (Err(_), Ok(())) => Outcome::OnlyCharPoly(r_inv.unwrap_err()),
        (Ok(()), Err(_)) => Outcome::OnlyInv(r_cp.unwrap_err()),
        (Err(e), Err(_)) => Outcome::None(e),
    }
}

/// Returns the maximum absolute difference and the same difference relative to the
/// largest component of the two derivative tensors
fn differences(d_inv: &[Tensor4<6>; 3], d_cp: &[Tensor4<6>; 3]) -> (f64, f64) {
    let mut max_abs = 0.0;
    let mut scale = 0.0;
    for k in 0..3 {
        for m in 0..6 {
            for n in 0..6 {
                let a = d_inv[k].get(m, n);
                let b = d_cp[k].get(m, n);
                max_abs = f64::max(max_abs, f64::abs(a - b));
                scale = f64::max(scale, f64::max(f64::abs(a), f64::abs(b)));
            }
        }
    }
    let rel = if scale > 0.0 { max_abs / scale } else { 0.0 };
    (max_abs, rel)
}

/// Orthogonal transformation matrix from the papers
fn u_sym() -> [[f64; 3]; 3] {
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
