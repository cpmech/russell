use crate::{ADD, P_SYM, SET};
use crate::{EigenProjsT2, EigenValMethod, Tensor2, Tensor4, dsd_fn};
use crate::{StrError, deriv2_invariant_ii3};
use crate::{ssd_fn, t2_dyad_t2};

/// Auxiliary fourth-order tensor Q := Psym − I⊗I in Kelvin-Mandel components
const Q4: [[f64; 6]; 6] = [
    [0.0, -1.0, -1.0, 0.0, 0.0, 0.0],
    [-1.0, 0.0, -1.0, 0.0, 0.0, 0.0],
    [-1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
];

/// Assists in calculating the derivatives of the eigenprojectors
pub struct EigenProjDerivsT2 {
    /// Structure to assist in calculating the eigenvalues and eigenprojectors
    eig: EigenProjsT2,

    /// Inverse Tensor2 of the input matrix A
    ///
    ///
    /// ```text
    /// aa_inv = A⁻¹
    /// ```
    aa_inv: Tensor2<6>,

    /// Auxiliary Tensor4: ssd(A⁻¹)
    ///
    /// ```text
    ///             _
    /// Y := ½ (A⁻¹ ⊗ A⁻¹ + A⁻¹ ⊗ A⁻¹) = ssd(A⁻¹) / 2
    ///                         ‾
    /// ```
    yy: Tensor4<6>,

    /// Workspace: auxiliary set of Tensor4
    work: [Tensor4<6>; 3],

    /// Auxiliary Tensor4 `∂²I3/∂a²` (second derivative of the third invariant)
    d2_ii3: Tensor4<6>,

    /// Identity Tensor2
    ii_ten: Tensor2<6>,
}

impl EigenProjDerivsT2 {
    /// Allocates a new instance
    pub fn new() -> Self {
        EigenProjDerivsT2 {
            eig: EigenProjsT2::new(),
            aa_inv: Tensor2::new(),
            yy: Tensor4::new(),
            work: [Tensor4::new(), Tensor4::new(), Tensor4::new()],
            d2_ii3: Tensor4::new(),
            ii_ten: Tensor2::identity(),
        }
    }

    /// Calculates the derivatives of the eigenprojectors w.r.t. the A
    ///
    /// Note: This function is only available for tensor A with *all-distinct* eigenvalues.
    ///
    /// # Arguments
    ///
    /// `ll` -- (output) the eigenvalues
    /// `projs` -- (output) the eigenprojectors
    /// `dpp` -- (output) the derivatives of the eigenprojectors w.r.t. A
    /// `aa` -- the tensor A
    ///
    /// The default method is [EigenValMethod::AnalyticalHZ]
    ///
    /// The default derivative method is `characteristic polynomial` (i.e., no inverse needed).
    pub fn calculate(
        &mut self,
        ll: &mut [f64; 3],
        projs: &mut [Tensor2<6>; 3],
        dpp: &mut [Tensor4<6>; 3],
        aa: &Tensor2<6>,
    ) -> Result<(), StrError> {
        self.calculate_mx(ll, projs, dpp, aa, EigenValMethod::AnalyticalHZ, false)
    }

    /// Calculates the derivatives of the eigenprojectors w.r.t. the A (method selection)
    ///
    /// Note: This function is only available for tensor A with *all-distinct* eigenvalues.
    ///
    /// If `use_inverse = true`, the tensor must be also invertible and all eigenvalues must not be zero.
    ///
    /// # Arguments
    ///
    /// `ll` -- (output) the eigenvalues
    /// `projs` -- (output) the eigenprojectors
    /// `dpp` -- (output) the derivatives of the eigenprojectors w.r.t. A
    /// `aa` -- the tensor A
    /// `method` -- the method to calculate the eigenvalues
    /// `use_inverse` -- whether to use the inverse method or the characteristic polynomial method
    pub fn calculate_mx(
        &mut self,
        ll: &mut [f64; 3],
        projs: &mut [Tensor2<6>; 3],
        dpp: &mut [Tensor4<6>; 3],
        aa: &Tensor2<6>,
        method: EigenValMethod,
        use_inverse: bool,
    ) -> Result<(), StrError> {
        if use_inverse {
            self.calc_with_inv(ll, projs, dpp, aa, method)
        } else {
            self.calc_with_char_poly(ll, projs, dpp, aa, method)
        }
    }

    /// Calculates the derivatives of the eigenprojectors w.r.t. the A using the inverse of A
    ///
    /// Note: This function is only available for *invertible* tensor A with *all-distinct* and *non-zero* eigenvalues.
    ///
    /// For all-distinct eigenvalues, this function calculates:
    ///
    /// ```text
    /// dPi                      3
    /// ─── = ai Psym - bi Y4 +  Σ (cij - ai) Pj ⊗ Pj
    /// dA                      j=1
    ///
    /// ```
    ///
    /// where:
    ///
    /// ```text
    /// Y4 = ½ ssd(A⁻¹)
    ///
    ///      li        I3           I3
    /// ai = ──,  bi = ──,  cij = ──────
    ///      di        di         di lj²
    ///
    ///                      I3
    /// di = 2 li² - I1 li + ──
    ///                      li
    /// ```
    ///
    /// where `li` is the i-th eigenvalue.
    ///
    /// # References
    ///
    /// 1. Miehe C. (1998) Comparison of two algorithms for the computation of fourth-order
    ///    isotropic tensor functions. Computers & Structures, 66(1):37-43.
    ///    <https://doi.org/10.1016/S0045-7949(97)00073-4>
    pub fn calc_with_inv(
        &mut self,
        ll: &mut [f64; 3],
        projs: &mut [Tensor2<6>; 3],
        dpp: &mut [Tensor4<6>; 3],
        aa: &Tensor2<6>,
        method: EigenValMethod,
    ) -> Result<(), StrError> {
        // compute the eigenvalues and eigenprojectors
        self.eig.calculate_mx(ll, projs, aa, method)?;

        // calculate differences between the SORTED eigenvalues
        let d01 = f64::abs(ll[0] - ll[1]);
        let d12 = f64::abs(ll[1] - ll[2]);

        // calculate a tolerance to detect coalescence
        let tol_diff = 10.0 * f64::EPSILON.sqrt();

        // spherical case
        if d01 <= tol_diff && d12 <= tol_diff {
            return Err("Failed due to spherical state (all equal eigenvalues)");
        }

        // check for distinct eigenvalues (the status is up to date because the projectors are available)
        if d01 <= tol_diff || d12 <= tol_diff {
            return Err("Failed due to two repeated eigenvalues");
        }

        // check for zero valued eigenvalues
        for i in 0..3 {
            if f64::abs(ll[i]) < tol_diff {
                return Err("Failed due to a zero valued eigenvalue");
            }
        }

        // use a determinant tolerance relative to the magnitude of the tensor so that a
        // uniform scaling of A does not change whether it is deemed invertible
        let norm = aa.norm();
        let det_tol = 10.0 * f64::EPSILON * norm * norm * norm;

        // calculate A⁻¹, the inverse of A, and I3 = det(A)
        let det = aa.inverse(&mut self.aa_inv, det_tol);
        if det.is_none() {
            return Err("Failed due to non-invertible tensor");
        }
        let ii3 = det.unwrap();

        // calculate the auxiliary tensor Y = ssd(A⁻¹) / 2
        ssd_fn(&mut self.yy, SET, 0.5, &self.aa_inv);

        // allocate and calculate auxiliary tensors P[j] ⊗ P[j]
        t2_dyad_t2(&mut self.work[0], SET, 1.0, &projs[0], &projs[0]);
        t2_dyad_t2(&mut self.work[1], SET, 1.0, &projs[1], &projs[1]);
        t2_dyad_t2(&mut self.work[2], SET, 1.0, &projs[2], &projs[2]);

        // calculate auxiliary coefficients
        let mut d = [0.0; 3];
        let mut a = [0.0; 3];
        let mut b = [0.0; 3];
        let mut c = [[0.0; 3]; 3];
        let ii1 = aa.invariant_ii1();
        for i in 0..3 {
            d[i] = 2.0 * ll[i] * ll[i] - ii1 * ll[i] + ii3 / ll[i];
            if f64::abs(d[i]) < tol_diff {
                return Err("|d[i]| is nearly zero");
            }
            a[i] = ll[i] / d[i];
            b[i] = ii3 / d[i];
            for j in 0..3 {
                c[i][j] = ii3 / (d[i] * ll[j] * ll[j]);
            }
        }

        // Compute the derivatives
        for i in 0..3 {
            for m in 0..6 {
                for n in 0..6 {
                    let p = a[i] * P_SYM[m][n] - b[i] * self.yy.get(m, n);
                    let q0 = (c[i][0] - a[i]) * self.work[0].get(m, n);
                    let q1 = (c[i][1] - a[i]) * self.work[1].get(m, n);
                    let q2 = (c[i][2] - a[i]) * self.work[2].get(m, n);
                    dpp[i].set(m, n, p + q0 + q1 + q2);
                }
            }
        }
        Ok(())
    }

    /// Calculates the derivatives of the eigenprojectors w.r.t. the A using the characteristic polynomial
    ///
    /// Note: This function is only available for tensor A with *all-distinct* eigenvalues.
    ///
    /// For all-distinct eigenvalues, this function calculates:
    ///
    /// ```text
    /// dPk   ak           bk             1              lk     1
    /// ─── = ── Pk ⊗ Pk + ── dsd(Pk,I) + ── dsd(Pk,A) + ── Q + ── M
    ///  dA   gk           gk             gk             gk     gk
    /// ```
    ///
    /// where
    ///
    /// ```text
    /// dsd(Pk,I) = Pk ⊗ I + I ⊗ Pk
    /// dsd(Pk,A) = Pk ⊗ A + A ⊗ Pk
    /// ```
    ///
    /// # References
    ///
    /// 1. Panteghini A. (2024) A simple spectral representation of a second-order symmetric
    ///    tensor and its variation. European Journal of Mechanics - A/Solids, 104:105208.
    ///    <https://doi.org/10.1016/j.euromechsol.2023.105208>
    pub fn calc_with_char_poly(
        &mut self,
        ll: &mut [f64; 3],
        projs: &mut [Tensor2<6>; 3],
        dpp: &mut [Tensor4<6>; 3],
        aa: &Tensor2<6>,
        method: EigenValMethod,
    ) -> Result<(), StrError> {
        // compute the eigenvalues and eigenprojectors
        self.eig.calculate_mx(ll, projs, aa, method)?;

        // calculate differences between the SORTED eigenvalues
        let d01 = f64::abs(ll[0] - ll[1]);
        let d12 = f64::abs(ll[1] - ll[2]);

        // calculate a tolerance to detect coalescence
        let tol_diff = 10.0 * f64::EPSILON.sqrt();

        // spherical case
        if d01 <= tol_diff && d12 <= tol_diff {
            return Err("Failed due to spherical state (all equal eigenvalues)");
        }

        // check for distinct eigenvalues (the status is up to date because the projectors are available)
        if d01 <= tol_diff || d12 <= tol_diff {
            return Err("Failed due to two repeated eigenvalues");
        }

        // compute the invariants
        let ii1 = aa.invariant_ii1();
        let ii2 = aa.invariant_ii2();

        // compute ∂²I3a/∂A² (the second derivative of the third invariant)
        deriv2_invariant_ii3(&mut self.d2_ii3, aa);

        // calculate the derivatives of eigenprojectors (for all-distinct eigenvalues)
        for k in 0..3 {
            self.calc_deriv_non_rep(&mut dpp[k], ll[k], &projs[k], ii1, ii2, aa)?;
        }
        Ok(())
    }

    /// Calculates the derivative of the eigenprojector for the non-repeated eigenvalue (k)
    fn calc_deriv_non_rep(
        &mut self,
        dpp_k: &mut Tensor4<6>,
        lam_k: f64,
        pp_k: &Tensor2<6>,
        ii1: f64,
        ii2: f64,
        aa: &Tensor2<6>,
    ) -> Result<(), StrError> {
        // calculate gamma[k]
        let g = 3.0 * lam_k * lam_k - 2.0 * ii1 * lam_k + ii2;

        // calculate auxiliary coefficients
        let ag = (2.0 * ii1 - 6.0 * lam_k) / g;
        let bg = (2.0 * lam_k - ii1) / g;
        let og = 1.0 / g;

        // work[0] := (l Q + M) / g
        for m in 0..6 {
            for n in 0..6 {
                dpp_k.set(m, n, (lam_k * Q4[m][n] + self.d2_ii3.get(m, n)) / g);
            }
        }

        // work[0] += (a/g) * P[k] ⊗ P[k]
        t2_dyad_t2(dpp_k, ADD, ag, pp_k, pp_k);

        // work[0] += (b/g) * (P[k] ⊗ I + I ⊗ P[k])
        dsd_fn(dpp_k, ADD, bg, pp_k, &self.ii_ten);

        // work[0] += (1/g) * (P[k] ⊗ A + A ⊗ P[k])
        dsd_fn(dpp_k, ADD, og, pp_k, aa);
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::EigenProjDerivsT2;
    use crate::{EigenProjsT2, EigenValMethod, SamplesTensor2, StrError, Tensor2, Tensor4};
    use russell_lab::{deriv1_central5, mat_approx_eq};

    /// Holds arguments for numerical differentiation corresponding to [dP[i]/dA]ₘₙ
    struct ArgsNumDerivProj {
        method: EigenValMethod, // method to calculate the eigenvalues
        calc: EigenProjsT2,     // eigenprojectors calculator
        ll: [f64; 3],           // eigenvalues
        projs: [Tensor2<6>; 3], // eigenprojectors
        aa: Tensor2<6>,         // the input tensor
        k: usize,               // projector index
        m: usize,               // index of ∂P[i]ₘ/∂aₙ (matrix representation)
        n: usize,               // index of ∂P[i]ₘ/∂aₙ (matrix representation)
    }

    /// Returns a component (m) of the i-th eigenprojector for a variation of a component (n) of tensor A
    fn component_of_projector_kelvin(x: f64, args: &mut ArgsNumDerivProj) -> Result<f64, StrError> {
        let original = args.aa.get(args.n);
        args.aa.set(args.n, x);
        args.calc
            .calculate_mx(&mut args.ll, &mut args.projs, &args.aa, args.method)
            .unwrap();
        args.aa.set(args.n, original);
        Ok(args.projs[args.k].get(args.m))
    }

    // compare analytical derivatives with numerical derivatives
    fn compare_with_numerical(method: EigenValMethod, aa: Tensor2<6>, ana_deriv: &[Tensor4<6>; 3], tol: f64) {
        let mut args = ArgsNumDerivProj {
            method,
            calc: EigenProjsT2::new(),
            ll: [0.0; 3],
            projs: [Tensor2::<6>::new(), Tensor2::<6>::new(), Tensor2::<6>::new()],
            aa: aa.clone(),
            k: 0,
            m: 0,
            n: 0,
        };
        let mut num_deriv = Tensor4::<6>::new();
        for k in 0..3 {
            args.k = k;
            for m in 0..6 {
                args.m = m;
                for n in 0..6 {
                    args.n = n;
                    let x = args.aa.get(args.n);
                    let res = deriv1_central5(x, &mut args, component_of_projector_kelvin).unwrap();
                    num_deriv.set(m, n, res);
                }
            }
            mat_approx_eq(&ana_deriv[k].as_std_matrix(), &num_deriv.as_std_matrix(), tol);
        }
    }

    #[test]
    fn calc_with_inv_works_with_samples() {
        const VERBOSE: bool = false;
        const TOL_DPP: f64 = 1e-10;
        let mut ll = [0.0; 3];
        let mut calc = EigenProjDerivsT2::new();
        let mut projs = [Tensor2::<6>::new(), Tensor2::<6>::new(), Tensor2::<6>::new()];
        let mut ddp = [Tensor4::<6>::new(), Tensor4::<6>::new(), Tensor4::<6>::new()];
        let method = EigenValMethod::AnalyticalHZ;
        for sample in [
            SamplesTensor2::TENSOR_Y,
            SamplesTensor2::TENSOR_Z,
            SamplesTensor2::TENSOR_U,
            SamplesTensor2::TENSOR_S,
        ] {
            if VERBOSE {
                println!("\n{}", "-".repeat(80));
                println!("{}", sample.desc);
            }

            // calculate the eigenvalues, eigenprojectors, and derivatives of eigenprojectors
            let aa = Tensor2::<6>::from_std_matrix(&sample.matrix).unwrap();
            if VERBOSE {
                println!("A = \n{}", aa.as_std_matrix());
            }
            calc.calc_with_inv(&mut ll, &mut projs, &mut ddp, &aa, method).unwrap();

            // check the derivatives using numerical differentiation
            let mut tol_dpp = TOL_DPP;
            if sample.desc.contains("Tensor U") {
                tol_dpp = 1e-9;
            }
            compare_with_numerical(method, aa, &ddp, tol_dpp);
        }
    }

    #[test]
    fn calc_with_char_poly_works_with_samples() {
        const VERBOSE: bool = false;
        const TOL_DPP: f64 = 1e-10;
        let mut ll = [0.0; 3];
        let mut calc = EigenProjDerivsT2::new();
        let mut projs = [Tensor2::<6>::new(), Tensor2::<6>::new(), Tensor2::<6>::new()];
        let mut ddp = [Tensor4::<6>::new(), Tensor4::<6>::new(), Tensor4::<6>::new()];
        let method = EigenValMethod::AnalyticalHZ;
        for sample in [
            SamplesTensor2::TENSOR_X,
            SamplesTensor2::TENSOR_Y,
            SamplesTensor2::TENSOR_Z,
            SamplesTensor2::TENSOR_U,
            SamplesTensor2::TENSOR_S,
        ] {
            if VERBOSE {
                println!("\n{}", "-".repeat(80));
                println!("{}", sample.desc);
            }

            // calculate the eigenvalues, eigenprojectors, and derivatives of eigenprojectors
            let aa = Tensor2::<6>::from_std_matrix(&sample.matrix).unwrap();
            if VERBOSE {
                println!("A = \n{}", aa.as_std_matrix());
            }
            calc.calc_with_char_poly(&mut ll, &mut projs, &mut ddp, &aa, method)
                .unwrap();

            // check the derivatives using numerical differentiation
            let mut tol_dpp = TOL_DPP;
            if sample.desc.contains("Tensor U") {
                tol_dpp = 1e-9;
            }
            compare_with_numerical(method, aa, &ddp, tol_dpp);
        }
    }
}
