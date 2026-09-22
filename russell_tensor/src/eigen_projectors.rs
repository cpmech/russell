use crate::StrError;
use crate::{EigenValMethod, EigenValuesT2, Tensor2};
use crate::{IDENTITY2, SQRT_2};
use russell_lab::small_mat_eigen_sym_jacobi;

/// Assists in calculating the eigenprojectors of a symmetric second-order tensor
pub struct EigenProjsT2 {
    /// Structure to calculate the eigenvalues
    eig: EigenValuesT2,
}

impl EigenProjsT2 {
    /// Allocates a new instance
    pub fn new() -> Self {
        EigenProjsT2 {
            eig: EigenValuesT2::new(),
        }
    }

    /// Calculates the eigenprojectors of a symmetric second order tensor
    ///
    /// # Arguments
    ///
    /// `ll` -- (output) the eigenvalues
    /// `projs` -- (output) the eigenprojectors
    /// `aa` -- the tensor A
    ///
    /// The default method is [EigenValMethod::AnalyticalHZ]
    pub fn calculate(
        &mut self,
        ll: &mut [f64; 3],
        projs: &mut [Tensor2<6>; 3],
        aa: &Tensor2<6>,
    ) -> Result<(), StrError> {
        self.calculate_mx(ll, projs, aa, EigenValMethod::AnalyticalHZ)
    }

    /// Calculates the eigenprojectors of a symmetric second order tensor (with method selection)
    ///
    /// # Arguments
    ///
    /// `ll` -- (output) the eigenvalues
    /// `projs` -- (output) the eigenprojectors
    /// `aa` -- the tensor A
    pub fn calculate_mx(
        &mut self,
        ll: &mut [f64; 3],
        projs: &mut [Tensor2<6>; 3],
        aa: &Tensor2<6>,
        method: EigenValMethod,
    ) -> Result<(), StrError> {
        // calculate eigenvalues and eigenprojectors using Jacobi Iterative method
        if method == EigenValMethod::Iterative {
            // eigenvalues and eigenvectors
            let mut lam = [0.0, 0.0, 0.0];
            aa.to_std_matrix_array(&mut self.eig.aa);
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

        // calculate the eigenvalues (sorted in descending order)
        let _ = self.eig.calculate_mx(ll, aa, method)?;

        // Note: the tolerance used in eig.calculate_mx to detect the spherical case is
        // too tight and cannot be used to calculate the eigenprojectors. Therefore, a
        // new check for the spherical case is made here.

        // calculate differences between the SORTED eigenvalues
        let d01 = f64::abs(ll[0] - ll[1]); // = |(κ0+iso) - (κ1+iso)| = |κ0 - κ1|
        let d12 = f64::abs(ll[1] - ll[2]); // = |(κ1+iso) - (κ2+iso)| = |κ1 - κ2|

        // calculate a tolerance to detect coalescence
        let tol_diff = 10.0 * f64::EPSILON.sqrt();

        // calculate the eigenprojectors using Sylvester's formula (non spherical)
        if d01 <= tol_diff && d12 <= tol_diff {
            // all eigenvalues are equal λ0 ≈ λ1 ≈ λ2 → P0=I, P1=0, P2=0
            for m in 0..6 {
                projs[0].vec[m] = 0.0;
                projs[1].vec[m] = 0.0;
                projs[2].vec[m] = 0.0;
            }
            projs[0].vec[0] = 1.0;
            projs[0].vec[1] = 1.0;
            projs[0].vec[2] = 1.0;
        } else if d01 <= tol_diff {
            // coalescent eigenvalues λ0 ≈ λ1 > λ2
            self.eval_proj(&mut projs[2], ll[2], ll[0], ll[1], aa);
            for m in 0..6 {
                projs[0].vec[m] = 0.0;
                projs[1].vec[m] = IDENTITY2[m] - projs[2].vec[m];
            }
        } else if d12 <= tol_diff {
            // coalescent eigenvalues λ0 > λ1 ≈ λ2
            self.eval_proj(&mut projs[0], ll[0], ll[1], ll[2], aa);
            for m in 0..6 {
                projs[1].vec[m] = IDENTITY2[m] - projs[0].vec[m];
                projs[2].vec[m] = 0.0;
            }
        } else {
            // all distinct eigenvalues
            self.eval_proj(&mut projs[0], ll[0], ll[1], ll[2], aa);
            self.eval_proj(&mut projs[2], ll[2], ll[0], ll[1], aa);
            for m in 0..6 {
                projs[1].vec[m] = IDENTITY2[m] - projs[0].vec[m] - projs[2].vec[m];
            }
        }
        Ok(())
    }

    /// Evaluates projector P_i corresponding to a non-repeated eigenvalue l_i
    #[inline]
    fn eval_proj(&mut self, pp_i: &mut Tensor2<6>, li: f64, lj: f64, lk: f64, aa: &Tensor2<6>) {
        let f = 1.0 / ((li - lj) * (li - lk));
        pp_i.vec[0] = f * ((aa.vec[0] - lj) * (aa.vec[0] - lk) + (aa.vec[3] * aa.vec[3] + aa.vec[5] * aa.vec[5]) / 2.0);
        pp_i.vec[1] = f * ((aa.vec[1] - lj) * (aa.vec[1] - lk) + (aa.vec[3] * aa.vec[3] + aa.vec[4] * aa.vec[4]) / 2.0);
        pp_i.vec[2] = f * ((aa.vec[2] - lj) * (aa.vec[2] - lk) + (aa.vec[4] * aa.vec[4] + aa.vec[5] * aa.vec[5]) / 2.0);
        pp_i.vec[3] = f * ((aa.vec[0] + aa.vec[1] - lj - lk) * aa.vec[3] + aa.vec[4] * aa.vec[5] / SQRT_2);
        pp_i.vec[4] = f * ((aa.vec[1] + aa.vec[2] - lj - lk) * aa.vec[4] + aa.vec[5] * aa.vec[3] / SQRT_2);
        pp_i.vec[5] = f * ((aa.vec[2] + aa.vec[0] - lj - lk) * aa.vec[5] + aa.vec[3] * aa.vec[4] / SQRT_2);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::EigenProjsT2;
    use crate::OK_EIGENPROJ_RULES;
    use crate::testing::{HaberaZilian, generate_eigen_problem, generate_tensors2};
    use crate::{EigenValMethod, SamplesTensor2, Tensor2, eigenprojector_rules};
    use russell_lab::{approx_eq, sort3};

    fn check_reconstruct(aa: &Tensor2<6>, ll: &[f64; 3], projs: &[Tensor2<6>], tol: f64, verbose: bool) {
        if verbose {
            let mut aa_rec = Tensor2::<6>::new();
            for m in 0..6 {
                aa_rec.vec[m] = ll[0] * projs[0].vec[m] + ll[1] * projs[1].vec[m] + ll[2] * projs[2].vec[m];
            }
            println!(
                "trace(A) = {}, trace(A_rec) = {}, |trace(A) - trace(A_rec)| = {:.5e}",
                aa.trace(),
                aa_rec.trace(),
                f64::abs(aa.trace() - aa_rec.trace())
            );
            println!("A (reconstructed) =\n{}", aa_rec.as_std_matrix());
        }
        for m in 0..6 {
            let aa_m = ll[0] * projs[0].vec[m] + ll[1] * projs[1].vec[m] + ll[2] * projs[2].vec[m];
            approx_eq(aa.vec[m], aa_m, tol);
        }
    }

    #[test]
    fn calculate_mx_works_with_samples() {
        const VERBOSE: bool = false;
        const VERB_RECONSTRUCT: bool = false;
        const TOL_VALS: f64 = 1e-13;
        const TOL_IDEM: f64 = 1e-13;
        const TOL_ORTH: f64 = 1e-13;
        const TOL_COMP: f64 = 1e-13;
        const TOL_SPECTRAL: f64 = 1e-13;
        let mut ll = [0.0; 3];
        let mut eig = EigenProjsT2::new();
        let mut projs = [Tensor2::<6>::new(), Tensor2::<6>::new(), Tensor2::<6>::new()];
        for method in [
            EigenValMethod::AnalyticalHZ,
            EigenValMethod::AnalyticalHA22,
            EigenValMethod::AnalyticalHA23,
            EigenValMethod::Iterative,
        ] {
            if VERBOSE {
                println!("\n{}", "=".repeat(80));
                println!("{:?}", method);
            }
            for sample in SamplesTensor2::all_symmetric() {
                // for sample in [SamplesTensor2::TENSOR_O] {
                // for sample in [SamplesTensor2::TENSOR_X] {
                // for sample in [SamplesTensor2::COAL_01] {
                if VERBOSE {
                    println!("\n{}", "-".repeat(80));
                    println!("{}", sample.desc);
                }

                // calculate the eigenvalues and eigenprojectors
                let aa = Tensor2::<6>::from_std_matrix(&sample.matrix).unwrap();
                if VERBOSE {
                    println!("A = \n{}", aa.as_std_matrix());
                }
                eig.calculate_mx(&mut ll, &mut projs, &aa, method).unwrap();

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
                check_reconstruct(&aa, &ll, &projs, TOL_SPECTRAL, VERB_RECONSTRUCT);
            }
        }
    }

    #[test]
    fn general_tensors2_works() {
        const VERBOSE: bool = false;
        const VERB_RECONSTRUCT: bool = false;
        const TOL_IDEM: f64 = 1e-15;
        const TOL_COMP: f64 = 1e-15;
        const TOL_RECON: f64 = 1e-15;
        let mut ll = [0.0; 3];
        let mut eig = EigenProjsT2::new();
        let mut projs = [Tensor2::<6>::new(), Tensor2::<6>::new(), Tensor2::<6>::new()];
        let (tensors, _) = generate_tensors2();
        for method in [
            EigenValMethod::AnalyticalHZ,
            EigenValMethod::AnalyticalHA22,
            EigenValMethod::AnalyticalHA23,
            EigenValMethod::Iterative,
        ] {
            if VERBOSE {
                println!("\n{}", "=".repeat(80));
                println!("{:?}", method);
            }
            for k in 0..tensors.len() {
                // calculate the eigenvalues and eigenprojectors
                let aa = &tensors[k];
                if VERBOSE {
                    println!("Test # {}: A = \n{}", k, aa.as_std_matrix());
                }
                eig.calculate_mx(&mut ll, &mut projs, aa, method).unwrap();

                // check whether the eigenprojectors satisfy the eigenprojector rules
                let (mut tol_idem, mut tol_recon) = (TOL_IDEM, TOL_RECON);
                if k == 17 {
                    if method == EigenValMethod::AnalyticalHA22 {
                        tol_recon = 1e-14;
                    }
                }
                if k == 26 {
                    tol_recon = 1e-14;
                }
                if k == 61 || k == 66 {
                    tol_idem = 1e-14;
                    tol_recon = 1e-14;
                }
                if k == 78 || k == 79 || k == 80 {
                    tol_idem = 1e-12;
                }
                if k == 81 || k == 82 || k == 83 {
                    tol_idem = 1e-9;
                }
                if k == 84 || k == 85 || k == 86 {
                    tol_recon = 1e-9;
                    if method == EigenValMethod::AnalyticalHA22 || method == EigenValMethod::AnalyticalHA23 {
                        tol_recon = 1e-8;
                    }
                }
                if k == 87 || k == 88 || k == 89 {
                    tol_recon = 1e-12;
                    if method == EigenValMethod::AnalyticalHA22 || method == EigenValMethod::AnalyticalHA23 {
                        tol_recon = 1e-11;
                    }
                }
                if k == 90 {
                    if method == EigenValMethod::AnalyticalHA22 {
                        tol_recon = 1e-14;
                    }
                }
                if k == 91 {
                    tol_recon = 1e-14;
                }
                if k == 92 {
                    if method == EigenValMethod::AnalyticalHA22 {
                        tol_recon = 1e-14;
                    }
                }
                let status = eigenprojector_rules(&projs, tol_idem, tol_idem, TOL_COMP, VERBOSE);
                assert_eq!(status, OK_EIGENPROJ_RULES);

                // check the spectral composition
                check_reconstruct(aa, &ll, &projs, tol_recon, VERB_RECONSTRUCT);
            }
        }
    }

    #[test]
    fn habera_zilian_cases_work_works() {
        const VERBOSE: bool = false;
        const VERBOSE_PROJ: bool = false;
        const VERB_RECONSTRUCT: bool = false;
        const TOL_COMP: f64 = 1e-15; // this is always near machine eps
        let mut ll = [0.0; 3];
        let mut eig = EigenProjsT2::new();
        let mut projs = [Tensor2::<6>::new(), Tensor2::<6>::new(), Tensor2::<6>::new()];
        let hz = HaberaZilian::new();
        for method in [
            EigenValMethod::AnalyticalHZ,
            EigenValMethod::AnalyticalHA22,
            EigenValMethod::AnalyticalHA23,
            EigenValMethod::Iterative,
        ] {
            for name in hz.names {
                for &delta in &hz.deltas {
                    // if !(name == "single_lim_J3J2" && delta == 1e-12) { continue; }
                    // if !(name == "single_lim_J3J2" && delta == 1e-4) { continue; }
                    if VERBOSE {
                        println!("\n{}", "=".repeat(80));
                        println!("{:?}", method);
                    }
                    // calculate the eigenvalues and eigenprojectors
                    let aa = hz.tensor(name, delta);
                    if VERBOSE {
                        println!("{} : {:.2e} : A = \n{}", name, delta, aa.as_std_matrix());
                    }
                    eig.calculate_mx(&mut ll, &mut projs, &aa, method).unwrap();

                    // check whether the eigenprojectors satisfy the eigenprojector rules
                    let (mut tol_idem, tol_recon) = hz.tolerances_projectors(name, delta);
                    if VERBOSE_PROJ {
                        println!("P0 =\n{}", projs[0].as_std_matrix());
                        println!("P1 =\n{}", projs[1].as_std_matrix());
                        println!("P2 =\n{}", projs[2].as_std_matrix());
                    }
                    if method == EigenValMethod::AnalyticalHA22 {
                        tol_idem *= 10.0;
                    }
                    let status = eigenprojector_rules(&projs, tol_idem, tol_idem, TOL_COMP, VERBOSE);
                    assert_eq!(status, OK_EIGENPROJ_RULES);

                    // check the reconstructed matrix
                    check_reconstruct(&aa, &ll, &projs, tol_recon, VERB_RECONSTRUCT);
                }
            }
        }
    }

    #[test]
    fn calculate_mx_works_with_wide_range_of_values() {
        const VERBOSE: bool = false;
        const VERB_RECONSTRUCT: bool = false;
        const TOL_IDEM: f64 = 1e-8;
        const TOL_COMP: f64 = 1e-15;
        const TOL_RECON: f64 = 1e-8;
        let mut ll = [0.0; 3];
        let mut eig = EigenProjsT2::new();
        let mut projs = [Tensor2::<6>::new(), Tensor2::<6>::new(), Tensor2::<6>::new()];
        let alpha = [1.0, 100.0, 1e6];
        let kappa = [0.0, 1e-10, 1e-8, 1e-6, 1e-3, 0.5];
        for method in [
            EigenValMethod::AnalyticalHZ,
            EigenValMethod::AnalyticalHA22,
            EigenValMethod::AnalyticalHA23,
            EigenValMethod::Iterative,
        ] {
            if VERBOSE {
                println!("\n{}", "=".repeat(80));
                println!("{:?}", method);
            }
            for r in 0..alpha.len() {
                for s in 0..kappa.len() {
                    for t in 0..kappa.len() {
                        if VERBOSE {
                            println!("r = {}, s = {}, t = {}", r, s, t);
                        }
                        // generate eigen-problem
                        let l1 = alpha[r];
                        let l2 = alpha[r] + kappa[s];
                        let l3 = alpha[r] + kappa[t];
                        let (aa, _, _) = generate_eigen_problem(l1, l2, l3);

                        // perform spectral decomposition
                        eig.calculate_mx(&mut ll, &mut projs, &aa, method).unwrap();

                        // check whether the eigenprojectors satisfy the eigenprojector rules
                        let mut tol_idem = TOL_IDEM;
                        if r == 1 {
                            tol_idem = 1e-7;
                        }
                        if r == 2 {
                            tol_idem = 1e-3;
                        }
                        let status = eigenprojector_rules(&projs, tol_idem, tol_idem, TOL_COMP, VERBOSE);
                        assert_eq!(status, OK_EIGENPROJ_RULES);

                        // check the reconstructed matrix
                        check_reconstruct(&aa, &ll, &projs, TOL_RECON, VERB_RECONSTRUCT);
                    }
                }
            }
        }
    }
}
