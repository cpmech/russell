use crate::StrError;
use crate::{EigenMethod, EigenValuesT2, Tensor2};
use crate::{IDENTITY2, SQRT_2};
use russell_lab::small_mat_eigen_sym_jacobi;

pub struct EigenProjsT2 {
    eig: EigenValuesT2,
    bs: Tensor2<6>,
    bt: Tensor2<6>,
}

impl EigenProjsT2 {
    /// Allocates a new instance
    pub fn new() -> Self {
        EigenProjsT2 {
            eig: EigenValuesT2::new(),
            bs: Tensor2::new(),
            bt: Tensor2::new(),
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

        // calculate the eigenvalues (sorted in descending order)
        let spherical = self.eig.calculate_mx(ll, aa, method)?;

        // calculate differences between the SORTED eigenvalues
        let scale = aa.norm();
        let tol = 1e3 * f64::EPSILON * f64::EPSILON * scale * scale;
        let d01 = f64::abs(ll[0] - ll[1]);
        let d12 = f64::abs(ll[1] - ll[2]);
        let repeated01 = d01 < tol;
        let repeated12 = d12 < tol;
        let both_repeated = repeated01 && repeated12;

        // handle spherical case → P0=I, P1=0, P2=0
        if spherical || both_repeated {
            for m in 0..6 {
                projs[0].vec[m] = 0.0;
                projs[1].vec[m] = 0.0;
                projs[2].vec[m] = 0.0;
            }
            projs[0].vec[0] = 1.0;
            projs[0].vec[1] = 1.0;
            projs[0].vec[2] = 1.0;
            return Ok(());
        }

        // handle all-distinct case. Use Sylvester's equation
        let repeated = repeated01 || repeated12;
        if !repeated {
            // r s t → 0 1 2
            let (r, s, t) = (0, 1, 2);
            for m in 0..6 {
                self.bs.vec[m] = (aa.vec[m] - ll[s] * IDENTITY2[m]) / (ll[r] - ll[s]);
                self.bt.vec[m] = (aa.vec[m] - ll[t] * IDENTITY2[m]) / (ll[r] - ll[t]);
            }
            bs_times_bt(&mut projs[0], &self.bs, &self.bt);
            // r s t → 1 2 0
            let (r, s, t) = (1, 2, 0);
            for m in 0..6 {
                self.bs.vec[m] = (aa.vec[m] - ll[s] * IDENTITY2[m]) / (ll[r] - ll[s]);
                self.bt.vec[m] = (aa.vec[m] - ll[t] * IDENTITY2[m]) / (ll[r] - ll[t]);
            }
            bs_times_bt(&mut projs[1], &self.bs, &self.bt);
            // r s t → 2 0 1 (use completeness identity instead)
            // for m in 0..6 {
            //     projs[2].vec[m] = IDENTITY2[m] - projs[0].vec[m] - projs[1].vec[m];
            // }
            let (r, s, t) = (2, 0, 1);
            for m in 0..6 {
                self.bs.vec[m] = (aa.vec[m] - ll[s] * IDENTITY2[m]) / (ll[r] - ll[s]);
                self.bt.vec[m] = (aa.vec[m] - ll[t] * IDENTITY2[m]) / (ll[r] - ll[t]);
            }
            bs_times_bt(&mut projs[2], &self.bs, &self.bt);
            return Ok(());
        }

        // handle two-repeated eigenvalues
        // P_hat=(A-λ1*I)/κ, P1=I-P_hat, P_other=0 with κ=λ_hat-λ1
        if repeated01 {
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

#[inline]
#[rustfmt::skip]
fn bs_times_bt(pp: &mut Tensor2<6>, bs_ten: &Tensor2<6>, bt_ten: &Tensor2<6>) {
    const TWO_SQRT_2: f64 = 2.0 * SQRT_2;
    let bs = bs_ten.as_data();
    let bt = bt_ten.as_data();
    pp.vec[0] = (2.0 * bs[0] * bt[0] + bs[3] * bt[3] + bs[5] * bt[5]) / 2.0;
    pp.vec[1] = (2.0 * bs[1] * bt[1] + bs[3] * bt[3] + bs[4] * bt[4]) / 2.0;
    pp.vec[2] = (2.0 * bs[2] * bt[2] + bs[4] * bt[4] + bs[5] * bt[5]) / 2.0;
    pp.vec[3] = (SQRT_2 * bs[3] * (bt[0] + bt[1]) + SQRT_2 * bs[0] * bt[3] + SQRT_2 * bs[1] * bt[3] + bs[5] * bt[4] + bs[4] * bt[5]) / TWO_SQRT_2;
    pp.vec[4] = (SQRT_2 * bs[4] * (bt[1] + bt[2]) + bs[5] * bt[3] + SQRT_2 * bs[1] * bt[4] + SQRT_2 * bs[2] * bt[4] + bs[3] * bt[5]) / TWO_SQRT_2;
    pp.vec[5] = (SQRT_2 * bs[5] * (bt[0] + bt[2]) + bs[4] * bt[3] + bs[3] * bt[4] + SQRT_2 * bs[0] * bt[5] + SQRT_2 * bs[2] * bt[5]) / TWO_SQRT_2;
    let res_6 = (SQRT_2 * bs[3] * (bt[1] - bt[0]) + SQRT_2 * bs[0] * bt[3] - SQRT_2 * bs[1] * bt[3] + bs[5] * bt[4] - bs[4] * bt[5]) / TWO_SQRT_2;
    let res_7 = (SQRT_2 * bs[4] * (bt[2] - bt[1]) - bs[5] * bt[3] + SQRT_2 * bs[1] * bt[4] - SQRT_2 * bs[2] * bt[4] + bs[3] * bt[5]) / TWO_SQRT_2;
    let res_8 = (SQRT_2 * bs[5] * (bt[2] - bt[0]) - bs[4] * bt[3] + bs[3] * bt[4] + SQRT_2 * bs[0] * bt[5] - SQRT_2 * bs[2] * bt[5]) / TWO_SQRT_2;
    debug_assert!(f64::abs(res_6)<1e-15);
    debug_assert!(f64::abs(res_7)<1e-15);
    debug_assert!(f64::abs(res_8)<1e-15);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::EigenProjsT2;
    use crate::OK_EIGENPROJ_RULES;
    use crate::testing::{HaberaZilian, generate_tensors2};
    use crate::{EigenMethod, SamplesTensor2, Tensor2, eigenprojector_rules};
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
        const VERBOSE: bool = true;
        const VERB_RECONSTRUCT: bool = true;
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
        const VERBOSE: bool = true;
        const VERB_RECONSTRUCT: bool = false;
        const TOL_IDEM: f64 = 1e-15;
        const TOL_ORTH: f64 = 1e-15;
        const TOL_COMP: f64 = 1e-15;
        const TOL_SPEC: f64 = 1e-15;
        let mut ll = [0.0; 3];
        let mut eig = EigenProjsT2::new();
        let mut projs = [Tensor2::<6>::new(), Tensor2::<6>::new(), Tensor2::<6>::new()];
        let (tensors, _) = generate_tensors2();
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
            for k in 0..tensors.len() {
                // calculate the eigenvalues and eigenprojectors
                let aa = &tensors[k];
                if VERBOSE {
                    println!("Test # {}: A = \n{}", k, aa.as_std_matrix());
                }
                eig.calculate_mx(&mut ll, &mut projs, &aa, method).unwrap();

                // check whether the eigenprojectors satisfy the eigenprojector rules
                let (mut tol_idem, mut tol_orth, mut tol_comp, mut tol_spec) = (TOL_IDEM, TOL_ORTH, TOL_COMP, TOL_SPEC);
                if method == EigenMethod::AnalyticalHA22 {
                    tol_spec = 1e-14
                }
                if k == 78 || k == 79 {
                    tol_idem = 1e-12;
                    tol_orth = 1e-12;
                }
                if k == 80 {
                    tol_idem = 1e-13;
                    tol_orth = 1e-13;
                    tol_comp = 1e-13;
                    tol_spec = 1e-13;
                }
                if k == 81 || k == 82 || k == 83 || k == 84 || k == 85 || k == 86 {
                    tol_idem = 1e-9;
                    tol_orth = 1e-9;
                }
                if k == 87 || k == 88 || k == 89 {
                    tol_idem = 1e-12;
                    tol_orth = 1e-12;
                }
                if k == 83 && (method == EigenMethod::AnalyticalHA22 || method == EigenMethod::AnalyticalHA23) {
                    tol_comp = 1e-10;
                    tol_spec = 1e-10;
                }
                let status = eigenprojector_rules(&projs, tol_idem, tol_orth, tol_comp, VERBOSE);
                assert_eq!(status, OK_EIGENPROJ_RULES);

                // check the spectral composition
                check_reconstruct(&aa, &ll, &projs, tol_spec, VERB_RECONSTRUCT);
            }
        }
    }

    #[test]
    fn habera_zilian_cases_work_works() {
        const VERBOSE: bool = true;
        const VERBOSE_PROJ: bool = false;
        const VERB_RECONSTRUCT: bool = true;
        const TOL_IDEM: f64 = 1e-13;
        const TOL_ORTH: f64 = 1e-13;
        const TOL_COMP: f64 = 1e-15;
        const TOL_SPEC: f64 = 1e-13;
        let mut ll = [0.0; 3];
        let mut eig = EigenProjsT2::new();
        let mut projs = [Tensor2::<6>::new(), Tensor2::<6>::new(), Tensor2::<6>::new()];
        let hz = HaberaZilian::new();
        for method in [
            EigenMethod::AnalyticalHZ,
            // EigenMethod::AnalyticalHA22,
            // EigenMethod::AnalyticalHA23,
            // EigenMethod::Iterative,
        ] {
            for name in hz.names {
                for &delta in &hz.deltas {
                    // tricky problem // if !(name == "single_lim_J3J2" && delta == 1e-12) { continue; }
                    if !(name == "single_lim_disc_t" && delta == 1e-12) {
                        continue;
                    }
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
                    let (mut tol_idem, mut tol_orth, mut tol_comp, mut tol_spec) =
                        (TOL_IDEM, TOL_ORTH, TOL_COMP, TOL_SPEC);
                    if name == "single_lim_disc_t" && delta == 1e-12 {
                        tol_idem = 1e-3;
                        tol_orth = 1e-3;
                        tol_comp = 1e-3;
                    }
                    if name == "single_lim_disc_t" && delta == 1e-8 {
                        tol_idem = 1e-7;
                        tol_orth = 1e-7;
                        tol_comp = 1e-8;
                        tol_spec = 1e-8;
                    }
                    if (name == "single_lim_disc_t" || name == "single_lim_disc_n") && delta == 1e-6 {
                        tol_idem = 1e-9;
                        tol_orth = 1e-9;
                    }
                    if name == "single_lim_disc_n" && delta == 1e-8 {
                        tol_idem = 1e-8;
                        tol_orth = 1e-8;
                    }
                    if name == "single_lim_J3J2" && delta == 1e-12 {
                        tol_idem = 1e-3;
                        tol_orth = 1e-3;
                    }
                    if VERBOSE_PROJ {
                        println!("P0 =\n{}", projs[0].as_std_matrix());
                        println!("P1 =\n{}", projs[1].as_std_matrix());
                        println!("P2 =\n{}", projs[2].as_std_matrix());
                    }
                    let status = eigenprojector_rules(&projs, tol_idem, tol_orth, tol_comp, VERBOSE);
                    assert_eq!(status, OK_EIGENPROJ_RULES);

                    // check the reconstructed matrix
                    check_reconstruct(&aa, &ll, &projs, tol_spec, VERB_RECONSTRUCT);
                }
            }
        }
    }
}
