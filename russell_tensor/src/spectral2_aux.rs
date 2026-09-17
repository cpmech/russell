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

/// Checks the properties (rules) of a set of candidate eigenprojectors
///
/// The rules are (with `i,j ∈ {1,2,3}`):
///
/// 1. **Idempotent**: `P[i] · P[i] = P[i]`
/// 2. **Orthogonal**: `P[i] · P[j] = 0` for `i ≠ j`
/// 3. **Complete**: `Σ_i P[i] = I`
///
/// Note: All three projector slots are included in the summation.
/// Therefore, for repeated eigenvalues, one projector may be set to
/// zero or an eigenspace projector may be distributed among multiple
/// slots. This function is indifferent to the particular representation
/// adopted, provided the rules above are satisfied.
///
/// # Results
///
/// Returns `status` where 7111 means success. See description of codes below.
///
/// This function returns a number such as `9xyz` where `xyz` holds three
/// boolean flags with `1` indicating success. Thus, the set of satisfied
/// rules combinations are:
///
/// ```text
/// Idempotent  Orthogonal  Complete
/// ----------  ----------  --------
///    1           1          1        
///    1           1          0
///    1           0          1  << impossible
///    1           0          0
///    0           1          1  << impossible
///    0           1          0
///    0           0          1
///    0           0          0
/// ```
///
/// Note that the case 011 is mathematically impossible for a finite family of operators.
/// The code should never reports this combination, unless the tolerances are too loose.
pub fn check_projector_rules(
    proj: &[Tensor2<6>],
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
    for i in 0..3 {
        proj[i].to_std_matrix_slice(&mut ppi); // P[i] <- 3x3 matrix from KM vector
        for j in 0..3 {
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
    let idem = if max_idem < tol_idempotent { 1 } else { 0 };
    let orth = if max_orth < tol_orthogonal { 1 } else { 0 };

    // 3. Complete rule: Σ_{i=0}^{K-1} P[i] = I-matrix
    for i in 0..3 {
        for m in 0..6 {
            sum[m] += proj[i].get(m);
        }
    }
    let mut max_comp = 0.0;
    for m in 0..6 {
        max_comp = f64::max(max_comp, f64::abs(sum[m] - IDENTITY2[m]));
    }
    let comp = if max_comp < tol_complete { 1 } else { 0 };

    // results
    if verbose {
        print_rule("1. Idempotent", idem == 1, max_idem, tol_idempotent);
        print_rule("2. Orthogonal", orth == 1, max_orth, tol_orthogonal);
        print_rule("3. Complete", comp == 1, max_comp, tol_complete);
    }

    7000 + idem * 100 + orth * 10 + comp
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{check_projector_rules, spectral2_octahedral};
    use crate::testing::{generate_eigen_problem, reference_eigendyads};
    use crate::{SQRT_3, SQRT_3_BY_2, Spectral2, Tensor2};
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

    #[test]
    fn check_projector_rules_zero_failed_works() {
        const VERBOSE: bool = false;
        let tol_idem = 1e-15;
        let tol_orth = 1e-15;
        let tol_comp = 1e-15;
        let (names, data) = reference_eigendyads();
        for i in 0..names.len() {
            let dat = &data[i];
            if VERBOSE {
                println!("\n{}", "=".repeat(80));
                println!("{}", names[i]);
            }
            let (_, _, proj) = generate_eigen_problem(dat.ll[2], dat.ll[1], dat.ll[0]);
            let n_failed = check_projector_rules(&proj, tol_idem, tol_orth, tol_comp, VERBOSE);
            if VERBOSE {
                println!("n_failed = {}", n_failed)
            }
            assert_eq!(n_failed, 0);
        }

        // let mut aa_reconstruct = Tensor2::<6>::new();
        // for m in 0..6 {
        //     aa_reconstruct.vec[m] = e_ll[0] * e_proj[0].vec[m] + e_ll[1] * e_proj[1].vec[m];
        // }
        // println!("A = \n{}", aa.as_std_matrix());
        // println!("A = \n{}", aa_reconstruct.as_std_matrix());
    }

    #[test]
    fn check_projector_rules_failed_works() {
        const VERBOSE: bool = true;
        let tol_idem = 1e-15;
        let tol_orth = 1e-15;
        let tol_comp = 1e-15;

        // correct projectors
        let dyad0 = [
            [4.0 / 9.0, -4.0 / 9.0, 2.0 / 9.0],
            [-4.0 / 9.0, 4.0 / 9.0, -2.0 / 9.0],
            [2.0 / 9.0, -2.0 / 9.0, 1.0 / 9.0],
        ];
        // n1 ⊗ n1 associated with λ1 = 4
        let dyad1 = [
            [4.0 / 9.0, 2.0 / 9.0, -4.0 / 9.0],
            [2.0 / 9.0, 1.0 / 9.0, -2.0 / 9.0],
            [-4.0 / 9.0, -2.0 / 9.0, 4.0 / 9.0],
        ];
        // n2 ⊗ n2 associated with λ2 = 2
        let dyad2 = [
            [1.0 / 9.0, 2.0 / 9.0, 2.0 / 9.0],
            [2.0 / 9.0, 4.0 / 9.0, 4.0 / 9.0],
            [2.0 / 9.0, 4.0 / 9.0, 4.0 / 9.0],
        ];
        let p0 = Tensor2::<6>::from_std_matrix(&dyad0).unwrap();
        let p1 = Tensor2::<6>::from_std_matrix(&dyad1).unwrap();
        let p2 = Tensor2::<6>::from_std_matrix(&dyad2).unwrap();
        let zero = Tensor2::<6>::new();
        let wrong = Tensor2::<6>::from_std_matrix(&[[123.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).unwrap();

        let proj = [p0.clone(), p1.clone(), p2.clone()];
        let status = check_projector_rules(&proj, tol_idem, tol_orth, tol_comp, VERBOSE);
        println!("Case 111: {}\n", status);
        assert_eq!(status, 7111);

        let proj = [p0.clone(), p0.clone(), p2.clone()];
        let status = check_projector_rules(&proj, tol_idem, tol_orth, tol_comp, VERBOSE);
        println!("Case 100: {}\n", status);
        assert_eq!(status, 7100);

        let proj = [p0.clone(), p1.clone(), zero.clone()];
        let status = check_projector_rules(&proj, tol_idem, tol_orth, tol_comp, VERBOSE);
        println!("Case 110: {}\n", status);
        assert_eq!(status, 7110);

        let proj = [p0.clone(), p1.clone(), wrong.clone()];
        let status = check_projector_rules(&proj, tol_idem, tol_orth, tol_comp, VERBOSE);
        println!("Case 000: {}\n", status);
        assert_eq!(status, 7000);

        let mut q0 = p0.clone();
        q0.scale(2.0);
        let proj = [q0.clone(), p1.clone(), zero.clone()];
        let status = check_projector_rules(&proj, tol_idem, tol_orth, tol_comp, VERBOSE);
        println!("Case 010: {}\n", status);
        assert_eq!(status, 7010);

        let mut q1 = p1.clone();
        q1.update(-1.0, &p0); // q1 -= p0
        let proj = [q0.clone(), q1.clone(), p2.clone()];
        let status = check_projector_rules(&proj, tol_idem, tol_orth, tol_comp, VERBOSE);
        println!("Case 001: {}\n", status);
        assert_eq!(status, 7001);
    }
}
