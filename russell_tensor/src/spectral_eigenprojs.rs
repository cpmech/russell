use crate::StrError;
use crate::{EigenMethod, EigenValuesT2, Tensor2};
use crate::{IDENTITY2, SQRT_2};
use russell_lab::small_mat_eigen_sym_jacobi;

/// Tolerance to assume repeated eigenvalues
const TOL_REPEATED: f64 = 1e-8;

/// Holds indices for permutation by looping in 0..3
const INDICES: [usize; 5] = [0, 1, 2, 0, 1];

pub struct EigenProjsT2 {
    /// Eigenvalues struct
    eig: EigenValuesT2,

    /// Deviatoric tensor in Kelvin-Mandel components
    ss: [f64; 6],
}

impl EigenProjsT2 {
    /// Allocates a new instance
    pub fn new() -> Self {
        EigenProjsT2 {
            eig: EigenValuesT2::new(),
            ss: [0.0; 6],
        }
    }

    /// Calculates the eigenprojectors of a symmetric second order tensor (with method selection)
    pub fn calculate_mx(
        &mut self,
        ll: &mut [f64; 3],
        projs: &mut [Tensor2<6>; 3],
        aa: &Tensor2<6>,
        method: EigenMethod,
    ) -> Result<(), StrError> {
        // calculate eigenvalues and eigenprojectors using Jacobi Iterative method
        if method == EigenMethod::Iterative {
            // eigenvalues and eigenvectors
            let mut lam = [0.0, 0.0, 0.0];
            aa.to_std_matrix_slice(&mut self.eig.aa);
            small_mat_eigen_sym_jacobi(&mut lam, &mut self.eig.vv, &mut self.eig.aa)?;

            // get indices to sort eigenvalues in descending order
            let mut indices = [0, 1, 2];
            indices.sort_by(|&i, &j| lam[j].partial_cmp(&lam[i]).unwrap());

            // set the return variables to the sorted eigenvalues and eigenprojectors
            let qq = &self.eig.vv;
            for i in 0..3 {
                let j = indices[i];
                ll[i] = lam[j];
                projs[i].vec[0] = qq[0][j] * qq[0][j];
                projs[i].vec[1] = qq[1][j] * qq[1][j];
                projs[i].vec[2] = qq[2][j] * qq[2][j];
                projs[i].vec[3] = (qq[0][j] * qq[1][j] + qq[1][j] * qq[0][j]) / SQRT_2;
                projs[i].vec[4] = (qq[1][j] * qq[2][j] + qq[2][j] * qq[1][j]) / SQRT_2;
                projs[i].vec[5] = (qq[0][j] * qq[2][j] + qq[2][j] * qq[0][j]) / SQRT_2;
            }
            return Ok(());
        }

        // clear the eigenprojectors
        for m in 0..6 {
            projs[0].vec[m] = 0.0;
            projs[1].vec[m] = 0.0;
            projs[2].vec[m] = 0.0;
        }

        // calculate the eigenvalues (sorted in descending order)
        let spherical = self.eig.calculate_mx(ll, aa, method)?;

        // handle spherical case → P0=I, P1=0, P2=0
        if spherical {
            projs[0].vec[0] = 1.0;
            projs[0].vec[1] = 1.0;
            projs[0].vec[2] = 1.0;
            return Ok(());
        }

        // calculate differences between the SORTED eigenvalues
        let scale = ll[0].abs().max(ll[1].abs()).max(ll[2].abs()).max(1.0);
        let tol = TOL_REPEATED * scale;
        let d01 = f64::abs(ll[0] - ll[1]);
        let d12 = f64::abs(ll[1] - ll[2]);

        // handle all-distinct case. Use Sylvester's equation
        if d01 >= tol && d12 >= tol {
            // P[r] = f * (A - λ[s] I) . (A - λ[t] I)
            for i in 0..3 {
                let r = INDICES[i];
                let s = INDICES[i + 1];
                let t = INDICES[i + 2];
                let p = -ll[s];
                let q = -ll[t];
                let f = 1.0 / ((ll[r] - ll[s]) * (ll[r] - ll[t]));
                t2_plus_diag_product(projs[r].as_mut_data(), f, aa.as_data(), p, q);
            }
            return Ok(());
        }

        // handle two-repeated eigenvalues
        // P_hat=(A-λ1*I)/κ, P1=I-P_hat, P_other=0 with κ=λ_hat-λ1
        if d01 < tol {
            // λ0 ≈ λ1 > (λ2) → λ_hat = λ2
            let kappa = ll[2] - ll[1];
            for m in 0..6 {
                projs[2].vec[m] = (aa.vec[m] - ll[1] * IDENTITY2[m]) / kappa;
                projs[1].vec[m] = IDENTITY2[m] - projs[2].vec[m];
                projs[0].vec[m] = 0.0;
            }
        } else {
            // (λ0) > λ1 ≈ λ2 → λ_hat = λ0
            let kappa = ll[0] - ll[1];
            for m in 0..6 {
                projs[0].vec[m] = (aa.vec[m] - ll[1] * IDENTITY2[m]) / kappa;
                projs[1].vec[m] = IDENTITY2[m] - projs[0].vec[m];
                projs[2].vec[m] = 0.0;
            }
        }
        Ok(())
    }
}

/// Calculates alpha * (A + p I) . (A + q I)
#[inline]
pub(crate) fn t2_plus_diag_product(res: &mut [f64], alpha: f64, a: &[f64], p: f64, q: f64) {
    res[0] = alpha * (2.0 * (p + a[0]) * (q + a[0]) + a[3] * a[3] + a[5] * a[5]) / 2.0;
    res[1] = alpha * (2.0 * (p + a[1]) * (q + a[1]) + a[3] * a[3] + a[4] * a[4]) / 2.0;
    res[2] = alpha * (2.0 * (p + a[2]) * (q + a[2]) + a[4] * a[4] + a[5] * a[5]) / 2.0;
    res[3] = alpha * ((p + q + a[0] + a[1]) * a[3] + a[4] * a[5] / SQRT_2);
    res[4] = alpha * ((p + q + a[1] + a[2]) * a[4] + a[3] * a[5] / SQRT_2);
    res[5] = alpha * ((p + q + a[0] + a[2]) * a[5] + a[3] * a[4] / SQRT_2);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::EigenProjsT2;
    use crate::OK_EIGENPROJ_RULES;
    use crate::{EigenMethod, SamplesTensor2, Tensor2, eigenprojector_rules};
    use russell_lab::{approx_eq, sort3};

    fn check_compose(aa: &Tensor2<6>, ll: &[f64; 3], projs: &[Tensor2<6>], tol: f64) {
        // let mut aa_rec = Tensor2::<6>::new();
        for m in 0..6 {
            let aa_m = ll[0] * projs[0].vec[m] + ll[1] * projs[1].vec[m] + ll[2] * projs[2].vec[m];
            approx_eq(aa.vec[m], aa_m, tol);
            println!("diff = {}", aa.vec[m] - aa_m);
            // aa_rec.vec[m] = aa_m;
        }
        // println!("A (rec) =\n{}", aa_rec.as_std_matrix());
    }

    #[test]
    fn calculate_mx_works_with_samples() {
        const VERBOSE: bool = true;
        const TOL_VALS: f64 = 1e-13;
        const TOL_IDEM: f64 = 1e-14;
        const TOL_ORTH: f64 = 1e-14;
        const TOL_COMP: f64 = 1e-14;
        const TOL_SPECTRAL: f64 = 1e-14;
        let mut ll = [0.0; 3];
        let mut eig = EigenProjsT2::new();
        let mut projs = [Tensor2::<6>::new(), Tensor2::<6>::new(), Tensor2::<6>::new()];
        for method in [
            EigenMethod::AnalyticalHZ,
            EigenMethod::AnalyticalHA22,
            EigenMethod::AnalyticalHA23,
            EigenMethod::Iterative,
        ] {
            if VERBOSE {
                println!("\n{}", "=".repeat(80));
                println!("{:?}", method);
            }

            for sample in SamplesTensor2::all_symmetric() {
                // for sample in [SamplesTensor2::COAL_01] {
                if VERBOSE {
                    println!("\n{}", "-".repeat(80));
                    println!("{}", sample.desc);
                }

                // calculate the eigenvalues and eigenprojectors
                let aa = Tensor2::<6>::from_std_matrix(&sample.matrix).unwrap();
                eig.calculate_mx(&mut ll, &mut projs, &aa, method).unwrap();
                // println!("ll = {:?}", ll);

                // check the eigenvalues
                let sample_ll = sample.eigenvalues.unwrap();
                let mut expected_l0 = sample_ll[0];
                let mut expected_l1 = sample_ll[1];
                let mut expected_l2 = sample_ll[2];
                sort3(&mut expected_l2, &mut expected_l1, &mut expected_l0); // will sort: l2 < l1 < l0
                approx_eq(ll[0], expected_l0, TOL_VALS);
                approx_eq(ll[1], expected_l1, TOL_VALS);
                approx_eq(ll[2], expected_l2, TOL_VALS);

                // check whether the eigenprojectors satisfy the eigenprojector rules
                let status = eigenprojector_rules(&projs, TOL_IDEM, TOL_ORTH, TOL_COMP, VERBOSE);
                assert_eq!(status, OK_EIGENPROJ_RULES);

                // check the spectral composition
                println!("A = \n{}", aa.as_std_matrix());
                check_compose(&aa, &ll, &projs, TOL_SPECTRAL);
            }
        }
    }
}
