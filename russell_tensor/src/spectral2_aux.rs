use crate::{IDENTITY2, SQRT_2, SQRT_3, SQRT_6};
use crate::{Spectral2, Tensor2};
use russell_lab::small_mat_mat_mul;

/// Rotates the eigenvalues to the principal values space
///
/// Returns `(λ_star_1, λ_star_2, λ_star_3)`
pub fn spectral2_octahedral(spc: &Spectral2) -> (f64, f64, f64) {
    let (s1, s2, s3) = (spc.lam[0], spc.lam[1], spc.lam[2]);
    let ls1 = (2.0 * s1 - s2 - s3) / SQRT_6;
    let ls2 = (s1 + s2 + s3) / SQRT_3;
    let ls3 = (s3 - s2) / SQRT_2;
    (ls1, ls2, ls3)
}

/// Returns the maximum absolute difference between two (small) matrices
fn max_diff(aa: &[[f64; 3]; 3], bb: &[[f64; 3]; 3]) -> f64 {
    let mut max = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            max = f64::max(max, f64::abs(aa[i][j] - bb[i][j]));
        }
    }
    max
}

/// Prints a projector-rule result with a check mark (green on success) or a cross (red on failure)
fn print_rule(name: &str, ok: bool, error: f64, tol: f64) {
    const RESET: &str = "\u{1b}[0m";
    const GREEN: &str = "\u{1b}[32m";
    const RED: &str = "\u{1b}[31m";
    if ok {
        println!("{GREEN}✓{RESET} {name:<14} (error = {error:.1e} < tol = {tol:.1e})");
    } else {
        println!("{RED}✗{RESET} {name:<14} (error = {error:.1e} > tol = {tol:.1e})");
    }
}

/// Checks the properties (rules) of the eigenprojectors and optionally prints the results
///
/// The rules are (with `i,j ∈ {1,...,K}`):
///
/// 1. **Idempotent**: `P[i] · P[i] = P[i]`
/// 2. **Orthogonal**: `P[i] · P[j] = 0` for `i ≠ j`
/// 3. **Complete**: `Σ_{i=0}^{K-1} P[i] = I`
///
/// where `K = kk` is the number of distinct eigenvalues.
///
/// Returns the number of failed rules.
///
/// # Panics
///
/// Any projector beyond `kk` must be exactly zero, otherwise a panic will occur.
pub fn check_projector_rules(
    proj: &[Tensor2<6>],
    kk: usize,
    tol_idempotent: f64,
    tol_orthogonal: f64,
    tol_complete: f64,
    verbose: bool,
) -> usize {
    // auxiliary arrays
    const ZERO_3X3_MAT: [[f64; 3]; 3] = [[0.0; 3]; 3];
    let mut aux = [[0.0; 3]; 3];
    let mut ppi = [[0.0; 3]; 3];
    let mut ppj = [[0.0; 3]; 3];
    let mut sum = [0.0; 6];

    // check rules #1 and #2
    let mut max_idem = 0.0;
    let mut max_orth = 0.0;
    for i in 0..kk {
        proj[i].to_std_matrix_slice(&mut ppi); // P[i] <- 3x3 matrix from KM vector
        for j in 0..kk {
            if i == j {
                // 1. Idempotent rule: P[i] . P[i] = P[i]
                small_mat_mat_mul(&mut aux, 1.0, &ppi, &ppi, 0.0, 3);
                max_idem = f64::max(max_idem, max_diff(&aux, &ppi));
            } else {
                // 2. Orthogonal rule: P[i] . P[j] = 0-matrix
                proj[j].to_std_matrix_slice(&mut ppj); // P[j] <- 3x3 matrix from KM vector
                small_mat_mat_mul(&mut aux, 1.0, &ppi, &ppj, 0.0, 3);
                max_orth = f64::max(max_orth, max_diff(&aux, &ZERO_3X3_MAT));
            }
        }
    }
    let ok_idem = max_idem < tol_idempotent;
    let ok_orth = max_orth < tol_orthogonal;

    // 3. Complete rule: Σ_{i=0}^{K-1} P[i] = I-matrix
    for i in 0..kk {
        for m in 0..6 {
            sum[m] += proj[i].get(m);
        }
    }
    let mut max_comp = 0.0;
    for m in 0..6 {
        max_comp = f64::max(max_comp, f64::abs(sum[m] - IDENTITY2[m]));
    }
    let ok_comp = max_comp < tol_complete;

    // results
    if verbose {
        print_rule("1. Idempotent", ok_idem, max_idem, tol_idempotent);
        print_rule("2. Orthogonal", ok_orth, max_orth, tol_orthogonal);
        print_rule("3. Complete", ok_comp, max_comp, tol_complete);
    }

    [ok_idem, ok_orth, ok_comp].iter().filter(|ok| !**ok).count()
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{check_projector_rules, spectral2_octahedral};
    use crate::{
        EigMethod, SQRT_3, SQRT_3_BY_2, Spectral2, Tensor2,
        spectral_eigenvals::{WorkspaceEigenvalues, eigenvalues_sym_tensor2},
        test_common::generate_eigen_problem,
    };
    use russell_lab::approx_eq;

    #[test]
    fn spectral_octahedral_works() {
        // the following data corresponds to p = 1 and q = 3
        #[rustfmt::skip]
        let principal_stresses_and_lode = [
            ( 3.0          ,  0.0          ,  0.0          ,  1.0 ),
            ( 0.0          ,  3.0          ,  0.0          ,  1.0 ),
            ( 0.0          ,  0.0          ,  3.0          ,  1.0 ),
            ( 1.0 + SQRT_3 ,  1.0 - SQRT_3 ,  1.0          ,  0.0 ),
            ( 1.0 + SQRT_3 ,  1.0          ,  1.0 - SQRT_3 ,  0.0 ),
            ( 1.0          ,  1.0 + SQRT_3 ,  1.0 - SQRT_3 ,  0.0 ),
            ( 1.0 - SQRT_3 ,  1.0 + SQRT_3 ,  1.0          ,  0.0 ),
            ( 1.0          ,  1.0 - SQRT_3 ,  1.0 + SQRT_3 ,  0.0 ),
            ( 1.0 - SQRT_3 ,  1.0          ,  1.0 + SQRT_3 ,  0.0 ),
            ( 2.0          , -1.0          ,  2.0          , -1.0 ),
            ( 2.0          ,  2.0          , -1.0          , -1.0 ),
            (-1.0          ,  2.0          ,  2.0          , -1.0 ),
        ];
        let mut spec = Spectral2::new();
        let mut tt = Tensor2::<6>::new();
        for (sigma_1, sigma_2, sigma_3, lode_correct) in &principal_stresses_and_lode {
            tt.set(0, *sigma_1);
            tt.set(1, *sigma_2);
            tt.set(2, *sigma_3);
            spec.decompose(&tt).unwrap();
            let (ls1, ls2, ls3) = spectral2_octahedral(&spec);
            let radius = f64::sqrt(ls3 * ls3 + ls1 * ls1);
            let distance = ls2;
            approx_eq(distance / SQRT_3, 1.0, 1e-15);
            approx_eq(radius * SQRT_3_BY_2, 3.0, 1e-15);
            if radius > 0.0 {
                let cos_theta = ls1 / radius;
                let lode = 4.0 * f64::powf(cos_theta, 3.0) - 3.0 * cos_theta;
                approx_eq(lode, *lode_correct, 1e-15);
            }
        }
    }

    const VERBOSE: bool = true;

    #[test]
    fn check_projector_rules_works() {
        // eigenvalues = {4.0, 2.0, 2.0}
        //
        // orthonormal eigenvectors
        // e1 = [1/√2,  1/√2, 0]
        // e2 = [-1/√2, 1/√2, 0]
        // e3 = [0,     0,    1]
        //
        // P1 (associated with λ1 = 4.0):
        #[rustfmt::skip]
        let p0 = Tensor2::<6>::from_std_matrix(&[
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.0, 0.0],
        ]).unwrap();
        // P2 (associated with λ = 2.0) = e2 ⊗ e2 + e3 ⊗ e3
        #[rustfmt::skip]
        let p1 = Tensor2::<6>::from_std_matrix(&[
            [0.5, -0.5, 0.0],
            [-0.5, 0.5, 0.0],
            [0.0, 0.0, 1.0],
        ]).unwrap();
        // P3 = 0 (must be zero)
        #[rustfmt::skip]
        let p2 = Tensor2::<6>::from_std_matrix(&[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]).unwrap();

        let proj = &[p0, p1, p2];
        let kk = 2; // number of distinct eigenprojectors
        let tol_idem = 1e-15;
        let tol_orth = 1e-15;
        let tol_comp = 1e-15;
        let n_failed = check_projector_rules(proj, kk, tol_idem, tol_orth, tol_comp, VERBOSE);
        if VERBOSE {
            println!("n_failed = {}", n_failed)
        }

        let mut ll = [0.0; 3];
        let (aa, e_ll, e_proj) = generate_eigen_problem(4.0, 2.0, 2.0);
        let mut work = WorkspaceEigenvalues::new();
        eigenvalues_sym_tensor2(&mut ll, &aa, EigMethod::AnalyticalHZ, &mut work).unwrap();
        println!("ll = {:?}", ll);
        println!("e_ll = {:?}", e_ll);
        println!("p0 =\n{}", e_proj[0].as_std_matrix());
        println!("p1 =\n{}", e_proj[1].as_std_matrix());
        println!("p2 =\n{}", e_proj[2].as_std_matrix());
        let n_failed = check_projector_rules(&e_proj, kk, tol_idem, tol_orth, tol_comp, VERBOSE);
        if VERBOSE {
            println!("n_failed = {}", n_failed)
        }
        let mut aa_reconstruct = Tensor2::<6>::new();
        for m in 0..6 {
            aa_reconstruct.vec[m] = e_ll[0] * e_proj[0].vec[m] + e_ll[1] * e_proj[1].vec[m];
        }
        println!("A = \n{}", aa.as_std_matrix());
        println!("A = \n{}", aa_reconstruct.as_std_matrix());
    }
}
