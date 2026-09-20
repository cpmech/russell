#![allow(unused)]

use crate::StrError;
use crate::{EigenProjsT2, EigenValMethod, Tensor2, Tensor4};
use crate::{P_SYM, SET};
use crate::{ssd_fn, t2_dyad_t2};

/// Assists in calculating the derivatives of the eigenprojectors
pub struct EigenProjDerivsT2 {
    /// Structure to assist in calculating the eigenvalues and eigenprojectors
    eig: EigenProjsT2,

    /// Inverse of the input matrix A
    ///
    ///
    /// ```text
    /// aa_inv = A⁻¹
    /// ```
    aa_inv: Tensor2<6>,

    /// Auxiliary tensor: ssd(A⁻¹)
    ///
    /// ```text
    ///             _
    /// Y := ½ (A⁻¹ ⊗ A⁻¹ + A⁻¹ ⊗ A⁻¹) = ssd(A⁻¹) / 2
    ///                         ‾
    /// ```
    yy: Tensor4<6>,

    /// Auxiliary set of tensors (empty by default)
    ///
    /// ```text
    /// P[j] ⊗ P[j]  (no sum on j)
    /// ```
    p_dy_p: [Tensor4<6>; 3],

    /// Auxiliary tensor: M := ∂²I3/∂a² (second derivative of the third invariant)
    ///
    /// Used in the computation of the derivatives of the eigenprojectors
    d2_ii3: Tensor4<6>,
}

impl EigenProjDerivsT2 {
    /// Allocates a new instance
    pub fn new() -> Self {
        EigenProjDerivsT2 {
            eig: EigenProjsT2::new(),
            aa_inv: Tensor2::new(),
            yy: Tensor4::new(),
            p_dy_p: [Tensor4::new(), Tensor4::new(), Tensor4::new()],
            d2_ii3: Tensor4::new(),
        }
    }

    /// Calculates the derivatives of the eigenprojectors w.r.t. the A (method selection)
    pub fn calculate_mx(
        &mut self,
        ll: &mut [f64; 3],
        projs: &mut [Tensor2<6>; 3],
        dpp: &mut [Tensor4<6>; 3],
        aa: &Tensor2<6>,
        method: EigenValMethod,
    ) -> Result<(), StrError> {
        Ok(())
    }

    /// Calculates the derivatives of the eigenprojectors w.r.t. the A (using the inverse)
    ///
    /// Note: This function is only available for *invertible* tensor A with *distinct* and *non-zero* eigenvalues.
    ///
    /// ```text
    /// dP[i]                         3
    /// ───── = a[i] Psym - b[i] Y +  Σ (c[i][j] - a[i]) P[j] ⊗ P[j]
    ///  dA                          j=1
    ///
    /// where Y = ½ ssd(A⁻¹) and the coefficients are listed below.
    /// ```
    ///
    /// # Input
    ///
    /// TODO
    ///
    /// # Output
    ///
    /// TODO
    ///
    /// # Notes
    ///
    /// The coefficients are:
    ///
    /// ```text
    ///        λ[i]          I3a                  I3a
    /// a[i] = ────,  b[i] = ────,  c[i][j] = ────────────
    ///        d[i]          d[i]             d[i] (λ[j])²
    ///
    ///                               I3a
    /// d[i] = 2 (λ[i])² - I1a λ[i] + ────
    ///                               λ[i]
    /// ```
    ///
    /// # References
    ///
    /// 1. Miehe C. (1993) Computation of isotropic tensor functions. Communications in
    ///    Numerical Methods in Engineering, 9(11):889-896. <https://doi.org/10.1002/cnm.1640091105>
    /// 2. Miehe C. (1998) Comparison of two algorithms for the computation of fourth-order
    ///    isotropic tensor functions. Computers & Structures, 66(1):37-43.
    ///    <https://doi.org/10.1016/S0045-7949(97)00073-4>
    pub fn calculate_with_inv(
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
        t2_dyad_t2(&mut self.p_dy_p[0], SET, 1.0, &projs[0], &projs[0]);
        t2_dyad_t2(&mut self.p_dy_p[1], SET, 1.0, &projs[1], &projs[1]);
        t2_dyad_t2(&mut self.p_dy_p[2], SET, 1.0, &projs[2], &projs[2]);

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
                    let q0 = (c[i][0] - a[i]) * self.p_dy_p[0].get(m, n);
                    let q1 = (c[i][1] - a[i]) * self.p_dy_p[1].get(m, n);
                    let q2 = (c[i][2] - a[i]) * self.p_dy_p[2].get(m, n);
                    dpp[i].set(m, n, p + q0 + q1 + q2);
                }
            }
        }
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
    fn calculate_with_inv_works_with_samples() {
        const VERBOSE: bool = false;
        const TOL_DDP: f64 = 1e-9;
        let mut ll = [0.0; 3];
        let mut calc = EigenProjDerivsT2::new();
        let mut projs = [Tensor2::<6>::new(), Tensor2::<6>::new(), Tensor2::<6>::new()];
        let mut ddp = [Tensor4::<6>::new(), Tensor4::<6>::new(), Tensor4::<6>::new()];
        for method in [
            EigenValMethod::AnalyticalHZ,
            // EigenMethod::AnalyticalHA22,
            // EigenMethod::AnalyticalHA23,
            // EigenMethod::Iterative,
        ] {
            if VERBOSE {
                println!("\n{}", "=".repeat(80));
                println!("{:?}", method);
            }
            for sample in [SamplesTensor2::TENSOR_U] {
                if VERBOSE {
                    println!("\n{}", "-".repeat(80));
                    println!("{}", sample.desc);
                }

                // calculate the eigenvalues, eigenprojectors, and derivatives of eigenprojectors
                let aa = Tensor2::<6>::from_std_matrix(&sample.matrix).unwrap();
                if VERBOSE {
                    println!("A = \n{}", aa.as_std_matrix());
                }
                calc.calculate_with_inv(&mut ll, &mut projs, &mut ddp, &aa, method)
                    .unwrap();

                // check the derivatives using numerical differentiation
                compare_with_numerical(method, aa, &ddp, TOL_DDP);
            }
        }
    }
}
