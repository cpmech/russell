use super::{P_SYMDEV, SET, SQRT_2};
use crate::{EigenMethod, EigenValuesT2};
use crate::{StrError, Tensor2, Tensor4};
use crate::{deriv2_invariant_ii3, t2_dyad_t2};
use russell_lab::small_mat_eigen_sym_jacobi;

/// Tolerance to assume coalescent eigenvalues
const TOL_COALESCE: f64 = 1e-8;

/// Tolerance to assume (nearly) coalescent eigenvalues during the computation of derivatives
/// of eigenprojectors
///
/// It is compared against |γ|, where γ = 3 λ² - 2 I1 λ + I2 = (λ - λj)(λ - λk)
const TOL_GAMMA: f64 = 1e-8;

/// Holds indices for permutation by looping in 0..3
const INDICES: [usize; 5] = [0, 1, 2, 0, 1];

/// Auxiliary fourth-order tensor Q := Psym − I⊗I in Kelvin-Mandel components
const Q4: [[f64; 6]; 6] = [
    [0.0, -1.0, -1.0, 0.0, 0.0, 0.0],
    [-1.0, 0.0, -1.0, 0.0, 0.0, 0.0],
    [-1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
];

/// Specifies the current status of the eigenvalues
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EigStatus {
    /// The eigenvalues have not been computed yet
    NotComputed,

    /// All eigenvalues are distinct
    Distinct,

    /// Coalescent eigenvalues λ0 ≈ λ1 > λ2
    Coalesce01,

    /// Coalescent eigenvalues λ0 > λ1 ≈ λ2
    Coalesce12,

    /// All eigenvalues are equal λ0 ≈ λ1 ≈ λ2
    Spherical,
}

/// Specifies the status of the computation of derivative of eigenprojectors
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EigDerivStatus {
    /// Derivative of eigenprojectors completed successfully
    Success,

    /// Failed due to coalescent eigenvalues
    FailDueToCoalescent,

    /// Failed due to (near) zero eigenvalue
    FailDueToZeroEigenvalue,

    /// Failed due to non-invertible input tensor
    FailDueToNonInvertible,
}

/// Holds the spectral representation of a symmetric second-order tensor
///
/// **Warning:** The user must take care of the consistency of the data in this struct.
///
/// Given the tensor `A`, the spectral representation with eigenvalues `λ[k]` and eigenprojectors `P[k]`
/// is given by the following formula:
///
/// ```text
///      3
/// A =  Σ  λ[k] * P[k]
///     k=1
/// ```
///
pub struct Spectral2 {
    /// Holds the eigenvalues (sorted in descending order)
    pub lam: [f64; 3],

    /// Holds the eigenprojectors associated with the (sorted) eigenvalues
    ///
    /// Set of 3 symmetric Tensor2
    pub proj: [Tensor2<6>; 3],

    /// Holds the status of the eigenvalues (see [EigStatus])
    ///
    /// A value other than [EigStatus::NotComputed] indicates that the eigenvalues are up to
    /// date; [EigStatus::NotComputed] means they have not been computed (or the last
    /// computation failed).
    pub status: EigStatus,

    /// Holds the derivatives of the eigenprojector w.r.t the defining tensor (only for distinct eigenvalues)
    ///
    /// Set of 3 minor-symmetric Tensor4 (empty by default)
    pub dpp: Vec<Tensor4<6>>,

    /// Auxiliary tensor: inverse of the defining/input tensor A
    ///
    /// ```text
    /// aa_inv = A⁻¹
    /// ```
    pub aa_inv: Tensor2<6>,

    //
    // --- internal data
    //
    /// Workspace for the calculation of eigenvalues
    eig: EigenValuesT2,

    /// Input tensor as a 3x3 matrix (for Jacobi method)
    aa_3x3: [[f64; 3]; 3],

    /// Matrix whose columns are the eigenvectors (for Jacobi method)
    vv_3x3: [[f64; 3]; 3],

    /// Auxiliary set of tensors (empty by default)
    ///
    /// ```text
    /// P[j] ⊗ P[j]  (no sum on j)
    /// ```
    p_dy_p: Vec<Tensor4<6>>,

    /// Auxiliary tensor: M := ∂²I3/∂a² (second derivative of the third invariant)
    ///
    /// Used in the computation of the derivatives of the eigenprojectors
    d2_ii3: Option<Tensor4<6>>,
}

impl Spectral2 {
    //
    // --- public functions ---
    //

    /// Returns a new instance
    pub fn new() -> Self {
        Spectral2 {
            // public
            lam: [0.0; 3],
            proj: [Tensor2::<6>::new(), Tensor2::<6>::new(), Tensor2::<6>::new()],
            status: EigStatus::NotComputed,
            dpp: Vec::new(),
            aa_inv: Tensor2::<6>::new(),
            // private
            eig: EigenValuesT2::new(),
            aa_3x3: [[0.0; 3]; 3],
            vv_3x3: [[0.0; 3]; 3],
            p_dy_p: Vec::new(),
            d2_ii3: None,
        }
    }

    /// Calculates the eigenvalues (but not the eigenprojectors) of a symmetric second-order tensor (using the default method)
    ///
    /// The output is saved in this struct with the eigenvalues being sorted in descending order.
    /// The status is saved in `status`.
    ///
    /// Default method: [EigMethod::AnalyticalHZ]
    pub fn calc_eigenvalues(&mut self, aa: &Tensor2<6>) -> Result<(), StrError> {
        self.calc_eigenvalues_mx(aa, EigenMethod::AnalyticalHZ)
    }

    /// Calculates the eigenvalues (but not the eigenprojectors) of a symmetric second-order tensor
    ///
    /// The output is saved in this struct with the eigenvalues being sorted in descending order.
    /// The status is saved in `status`.
    pub fn calc_eigenvalues_mx(&mut self, aa: &Tensor2<6>, method: EigenMethod) -> Result<(), StrError> {
        // indicate that the eigenvalues and projectors are not available
        self.status = EigStatus::NotComputed;

        // calculate the eigenvalues
        let spherical = self.eig.calculate_mx(&mut self.lam, aa, method)?;

        // save the status
        if spherical {
            self.status = EigStatus::Spherical;
        } else {
            self.status = self.classify();
        }
        Ok(())
    }

    /// Performs the spectral decomposition of a symmetric second-order tensor (using the default method)
    ///
    /// The output is saved in this struct with the eigenvalues/projectors being sorted in descending order.
    /// The status is saved in `status`.
    ///
    /// Default method: [EigMethod::AnalyticalHZ]
    #[inline]
    pub fn decompose(&mut self, aa: &Tensor2<6>) -> Result<(), StrError> {
        self.decompose_mx(aa, EigenMethod::AnalyticalHZ)
    }

    /// Performs the spectral decomposition of a symmetric second-order tensor (specifying the method)
    ///
    /// The output is saved in this struct with the eigenvalues/projectors being sorted in descending order.
    /// The status is saved in `status`.
    pub fn decompose_mx(&mut self, aa: &Tensor2<6>, method: EigenMethod) -> Result<(), StrError> {
        // indicate that the eigenvalues and projectors are not available
        self.status = EigStatus::NotComputed;

        // Jacobi iterative method: calculate the eigenvalues and eigenprojectors
        if method == EigenMethod::Iterative {
            self.decompose_jacobi(aa)?;
            return Ok(());
        }

        // compute the eigenvalues (this sets `status`)
        self.calc_eigenvalues_mx(aa, method)?;

        // handle a (numerically) spherical tensor: the eigenvalues are all equal (at the
        // rounding level) and the eigenprojectors are not unique, so use the identity split
        if self.status == EigStatus::Spherical && Self::is_spherical(aa) {
            for m in 0..6 {
                self.proj[0].vec[m] = 0.0;
                self.proj[1].vec[m] = 0.0;
                self.proj[2].vec[m] = 0.0;
            }
            self.proj[0].vec[0] = 1.0;
            self.proj[0].vec[1] = 1.0;
            self.proj[0].vec[2] = 1.0;
            return Ok(());
        }

        // compute the eigenprojectors
        if self.status == EigStatus::Distinct {
            // well-separated eigenvalues: use the Sylvester formula
            // P[r] = f * (A - λ[s] I) . (A - λ[t] I)
            for i in 0..3 {
                let r = INDICES[i];
                let s = INDICES[i + 1];
                let t = INDICES[i + 2];
                let p = -self.lam[s];
                let q = -self.lam[t];
                let f = 1.0 / ((self.lam[r] - self.lam[s]) * (self.lam[r] - self.lam[t]));
                t2_plus_diag_product(self.proj[r].as_mut_data(), f, aa.as_data(), p, q);
            }
        } else {
            // nearly coalescent eigenvalues: fall back to Jacobi
            self.decompose_jacobi(aa)?;
        }

        // done
        Ok(())
    }

    /// Calculates the derivatives of the eigenprojectors w.r.t. the defining tensor (Panteghini form)
    ///
    /// This function handles both the *distinct* and the *coalescent* (two equal) eigenvalue
    /// cases. It uses the implicit approach of Panteghini (2024), which does not require the
    /// inverse of the tensor, nor non-zero eigenvalues.
    ///
    /// For distinct eigenvalues:
    ///
    /// ```text
    /// dP[i]   A[i]               B[i]                      
    /// ───── = ──── P[i] ⊗ P[i] + ──── (P[i] ⊗ I + I ⊗ P[i])
    ///  dA     γ[i]               γ[i]                      
    ///
    ///          1                           λ[i]      1
    ///       + ──── (P[i] ⊗ A + A ⊗ P[i]) + ──── Q + ──── M
    ///         γ[i]                         γ[i]     γ[i]
    /// ```
    ///
    /// where
    ///
    /// ```text
    /// γ[i] = 3 (λ[i])² - 2 I1a λ[i] + I2a
    /// A[i] = 2 I1a - 6 λ[i]
    /// B[i] = 2 λ[i] - I1a
    /// Q := Psym - I ⊗ I
    /// M := ∂²I3a/∂A²
    /// ```
    ///
    /// For coalescent eigenvalues (with κ = λ_dist - λ_coal, where λ_dist is the distinct eigenvalue and
    /// `S = A - (I1a/3) I`):
    ///
    /// ```text
    /// dP_dist   1        3
    /// ─────── = ─ Psd - ──── S ⊗ S
    ///    da     κ       2 κ³
    ///
    /// dP_coal     dP_dist
    /// ─────── = - ───────
    ///    da          da
    /// ```
    ///
    /// where `P_dist` is the eigenprojector of the distinct eigenvalue and `P_coal = I - P_dist` is the
    /// eigenprojector of the (double) coalescent eigenvalue; `Psd` is the symmetric-deviatoric
    /// making projector. These projectors are non-unique (Panteghini's choice).
    ///
    /// The spectral decomposition is performed internally using the given [EigMethod].
    ///
    /// # Input
    ///
    /// `a` -- The symmetric tensor
    /// `method` -- The method to calculate the eigenvalues
    ///
    /// # Output
    ///
    /// Returns [EigDerivStatus::Success] if the eigenvalues are distinct or coalescent.
    ///
    /// # Errors
    ///
    /// Returns an error if `γ[i]` is (nearly) zero (nearly coalescent eigenvalues) or if the
    /// tensor is spherical (the derivative of the eigenprojectors is not defined).
    ///
    /// # References
    ///
    /// 1. Panteghini A. (2024) A simple spectral representation of a second-order symmetric
    ///    tensor and its variation. European Journal of Mechanics - A/Solids, 104:105208.
    ///    <https://doi.org/10.1016/j.euromechsol.2023.105208>
    pub fn deriv_eigenproj(&mut self, a: &Tensor2<6>, method: EigenMethod) -> Result<EigDerivStatus, StrError> {
        // compute the eigenvalues and eigenprojectors
        self.decompose_mx(a, method)?;

        // dispatch on the eigenvalue multiplicity
        match self.status {
            EigStatus::Distinct => {
                self.deriv_eigenproj_distinct(a)?;
                Ok(EigDerivStatus::Success)
            }
            EigStatus::Coalesce01 | EigStatus::Coalesce12 => {
                self.deriv_eigenproj_coalescent(a)?;
                Ok(EigDerivStatus::Success)
            }
            _ => Err("the derivative of the eigenprojectors is not defined for spherical tensors"),
        }
    }

    /// (internal) Calculates the derivatives of the eigenprojectors for distinct eigenvalues
    fn deriv_eigenproj_distinct(&mut self, a: &Tensor2<6>) -> Result<(), StrError> {
        // M = ∂²I3a/∂A² (the second derivative of the third invariant)
        if self.d2_ii3.is_none() {
            self.d2_ii3 = Some(Tensor4::<6>::new());
        }
        deriv2_invariant_ii3(self.d2_ii3.as_mut().unwrap(), a);
        let m4 = self.d2_ii3.as_ref().unwrap().clone();

        // identity tensor
        let ii = Tensor2::<6>::identity();

        // auxiliary tensors P[j] ⊗ P[j]
        if self.p_dy_p.len() != 3 {
            self.p_dy_p = vec![Tensor4::<6>::new(), Tensor4::<6>::new(), Tensor4::<6>::new()];
        }

        // allocate the output tensors
        if self.dpp.len() != 3 {
            self.dpp = vec![Tensor4::<6>::new(), Tensor4::<6>::new(), Tensor4::<6>::new()];
        }

        // scratch tensor for the mixed dyadic products
        let mut scratch = Tensor4::<6>::new();

        // compute the derivatives
        let ii1 = a.invariant_ii1();
        let ii2 = a.invariant_ii2();
        for i in 0..3 {
            let lam = self.lam[i];
            let gam = 3.0 * lam * lam - 2.0 * ii1 * lam + ii2;
            if f64::abs(gam) < TOL_GAMMA {
                return Err("|γ[i]| is nearly zero (nearly coalescent eigenvalues)");
            }
            let aa = 2.0 * ii1 - 6.0 * lam;
            let bb = 2.0 * lam - ii1;
            let p = self.proj[i].clone();

            // P[i] ⊗ P[i]
            t2_dyad_t2(&mut self.p_dy_p[i], SET, 1.0, &p, &p);

            // start with (λ Q + M) / γ
            let mut res = Tensor4::<6>::new();
            for m in 0..6 {
                for n in 0..6 {
                    res.set(m, n, (lam * Q4[m][n] + m4.get(m, n)) / gam);
                }
            }

            // (A[i]/γ) P ⊗ P
            for m in 0..6 {
                for n in 0..6 {
                    res.add(m, n, (aa / gam) * self.p_dy_p[i].get(m, n));
                }
            }

            // (B[i]/γ) (P ⊗ I + I ⊗ P)
            t2_dyad_t2(&mut scratch, SET, 1.0, &p, &ii);
            for m in 0..6 {
                for n in 0..6 {
                    res.add(m, n, (bb / gam) * scratch.get(m, n));
                }
            }
            t2_dyad_t2(&mut scratch, SET, 1.0, &ii, &p);
            for m in 0..6 {
                for n in 0..6 {
                    res.add(m, n, (bb / gam) * scratch.get(m, n));
                }
            }

            // (1/γ) (P ⊗ A + A ⊗ P)
            t2_dyad_t2(&mut scratch, SET, 1.0, &p, a);
            for m in 0..6 {
                for n in 0..6 {
                    res.add(m, n, scratch.get(m, n) / gam);
                }
            }
            t2_dyad_t2(&mut scratch, SET, 1.0, a, &p);
            for m in 0..6 {
                for n in 0..6 {
                    res.add(m, n, scratch.get(m, n) / gam);
                }
            }

            self.dpp[i] = res;
        }
        Ok(())
    }

    /// (internal) Returns the slot of the distinct eigenvalue for the coalescent case
    ///
    /// The eigenvalues remain sorted in descending order:
    ///
    /// ```text
    /// Coalesce01 (λ0 ≈ λ1 > λ2):  i_dist = 2
    /// Coalesce12 (λ0 > λ1 ≈ λ2):  i_dist = 0
    /// ```
    fn coalescent_dist_slot(&self) -> Result<usize, StrError> {
        match self.status {
            EigStatus::Coalesce01 => Ok(2), // λ0 ≈ λ1 > λ2
            EigStatus::Coalesce12 => Ok(0), // λ0 > λ1 ≈ λ2
            _ => Err("invalid status for the coalescent case"),
        }
    }

    /// (internal) Calculates the derivatives of the eigenprojectors for coalescent eigenvalues
    ///
    /// Follows Panteghini (2024), based on the proportionality between the deviatoric tensor
    /// `S = A - (I1a/3) I` and the eigenprojector of the distinct eigenvalue:
    ///
    /// ```text
    /// P_dist = (1/3) I + (1/κ) S
    /// N_II   = ½ (I - P_dist)          (the eigenbasis of the repeated eigenvalue)
    /// λ_II   = ½ (I1a - λ_dist)        (Panteghini's definition; NOT λ0 nor λ1)
    /// κ      = λ_dist - λ_II
    /// ```
    ///
    /// with `dP_dist/dA = (1/κ) Psd - (3/(2 κ³)) S ⊗ S` and `dN_II/dA = -½ dP_dist/dA`.
    fn deriv_eigenproj_coalescent(&mut self, a: &Tensor2<6>) -> Result<(), StrError> {
        let i_dist = self.coalescent_dist_slot()?;
        let lam_dist = self.lam[i_dist];
        let lam_coal = (a.invariant_ii1() - lam_dist) / 2.0;
        let kappa = lam_dist - lam_coal;
        if f64::abs(kappa) < TOL_GAMMA {
            return Err("|κ| is nearly zero (coalescent eigenvalues)");
        }
        let inv_kappa = 1.0 / kappa;

        // S = A - (I1a/3) I
        let mut ss = Tensor2::<6>::new();
        a.deviator(&mut ss);

        // compute and store the Panteghini projectors (consistent with `dpp` below):
        //   P_dist = (1/3) I + (1/κ) S  and  N_II = ½ (I - P_dist)
        let mut p_dist = Tensor2::<6>::new();
        for m in 0..6 {
            p_dist.vec[m] = ss.vec[m] * inv_kappa;
        }
        for m in 0..3 {
            p_dist.vec[m] += 1.0 / 3.0;
        }
        let mut n_ii = Tensor2::<6>::new();
        for m in 0..3 {
            n_ii.vec[m] = 0.5 * (1.0 - p_dist.vec[m]);
        }
        for m in 3..6 {
            n_ii.vec[m] = -0.5 * p_dist.vec[m];
        }
        self.proj[i_dist] = p_dist;
        self.lam[i_dist] = lam_dist;
        for k in 0..3 {
            if k != i_dist {
                self.proj[k] = n_ii.clone();
                self.lam[k] = lam_coal;
            }
        }

        // allocate the output tensors
        if self.dpp.len() != 3 {
            self.dpp = vec![Tensor4::<6>::new(), Tensor4::<6>::new(), Tensor4::<6>::new()];
        }

        // dP_dist/dA = (1/κ) Psd - (3/(2κ³)) S ⊗ S  and  dN_II/dA = -½ dP_dist/dA
        let f = 3.0 / (2.0 * kappa * kappa * kappa);
        for m in 0..6 {
            for n in 0..6 {
                let dd = inv_kappa * P_SYMDEV[m][n] - f * ss.vec[m] * ss.vec[n];
                self.dpp[i_dist].set(m, n, dd);
                for k in 0..3 {
                    if k != i_dist {
                        self.dpp[k].set(m, n, -0.5 * dd);
                    }
                }
            }
        }
        Ok(())
    }

    //
    // --- internal functions ---
    //

    /// (internal) Indicates whether the tensor is spherical at the rounding level
    ///
    /// The `J2` invariant is compared against `1000 * eps² * scale²`.
    #[inline]
    fn is_spherical(aa: &Tensor2<6>) -> bool {
        let scale = aa.norm();
        aa.invariant_jj2() <= 1e3 * f64::EPSILON * f64::EPSILON * scale * scale
    }

    /// (internal) Returns the scaled tolerance used to detect coalescent (or nearly coalescent)
    /// eigenvalues
    ///
    /// The tolerance is `TOL_COALESCE * max(|λ|, 1)`.
    #[inline]
    fn tol_coalesce(&self) -> f64 {
        let scale = self.lam[0].abs().max(self.lam[1].abs()).max(self.lam[2].abs()).max(1.0);
        TOL_COALESCE * scale
    }

    /// (internal) Classifies the eigenvalues (which must be sorted in descending order)
    #[inline]
    fn classify(&self) -> EigStatus {
        let tol = self.tol_coalesce();
        let d01 = f64::abs(self.lam[0] - self.lam[1]);
        let d12 = f64::abs(self.lam[1] - self.lam[2]);
        if d01 < tol && d12 < tol {
            EigStatus::Spherical
        } else if d01 < tol {
            EigStatus::Coalesce01
        } else if d12 < tol {
            EigStatus::Coalesce12
        } else {
            EigStatus::Distinct
        }
    }

    /// (internal) Performs the spectral decomposition using the Jacobi method
    ///
    /// The eigenvalues, eigenprojectors, and status are saved in this struct.
    fn decompose_jacobi(&mut self, aa: &Tensor2<6>) -> Result<(), StrError> {
        // eigenvalues and eigenvectors
        aa.to_std_matrix_slice(&mut self.aa_3x3);
        let mut lambda = [0.0; 3];
        small_mat_eigen_sym_jacobi(&mut lambda, &mut self.vv_3x3, &mut self.aa_3x3)?;

        // get indices to sort eigenvalues in descending order
        let mut indices = [0, 1, 2];
        indices.sort_by(|&i, &j| lambda[j].partial_cmp(&lambda[i]).unwrap());

        // store sorted eigenvalues and eigenprojectors
        for i in 0..3 {
            let j = indices[i];
            self.lam[i] = lambda[j];
            self.set_projector_from_vector(i, j);
        }
        self.status = self.classify();
        Ok(())
    }

    /// (internal) Sets proj[i] = v ⊗ v where v is the j-th column of vv_3x3
    #[inline]
    fn set_projector_from_vector(&mut self, i: usize, j: usize) {
        let pp = &mut self.proj[i].vec;
        let qq = &self.vv_3x3;
        pp[0] = qq[0][j] * qq[0][j];
        pp[1] = qq[1][j] * qq[1][j];
        pp[2] = qq[2][j] * qq[2][j];
        pp[3] = (qq[0][j] * qq[1][j] + qq[1][j] * qq[0][j]) / SQRT_2;
        pp[4] = (qq[1][j] * qq[2][j] + qq[2][j] * qq[1][j]) / SQRT_2;
        pp[5] = (qq[0][j] * qq[2][j] + qq[2][j] * qq[0][j]) / SQRT_2;
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
    use super::{EigenMethod, Spectral2};
    use crate::{EigDerivStatus, SampleTensor2, SamplesTensor2, StrError, Tensor2, Tensor4};
    use russell_lab::{deriv1_central5, mat_approx_eq};

    /// Holds arguments for numerical differentiation corresponding to [dP[i]/dA]ₘₙ
    struct ArgsNumDerivProj {
        spec: Spectral2, // spectral decomposition struct
        i: usize,        // projector index
        a: Tensor2<6>,   // temporary  tensor
        m: usize,        // index of ∂P[i]ₘ/∂aₙ (matrix representation)
        n: usize,        // index of ∂P[i]ₘ/∂aₙ (matrix representation)
    }

    /// Returns a component (m) of the i-th eigenprojector for a variation of a component (n) of tensor A
    fn component_of_projector_kelvin(x: f64, args: &mut ArgsNumDerivProj) -> Result<f64, StrError> {
        let original = args.a.get(args.n);
        args.a.set(args.n, x);
        args.spec.decompose(&args.a).unwrap();
        args.a.set(args.n, original);
        Ok(args.spec.proj[args.i].get(args.m))
    }

    /// Check the derivative of eigenprojectors using numerical differentiation
    fn check_ddp(sample: &SampleTensor2, tol: f64) {
        // analytical derivative (Panteghini form)
        let aa = Tensor2::<6>::from_std_matrix(&sample.matrix).unwrap();
        let mut spec = Spectral2::new();
        let status = spec.deriv_eigenproj(&aa, EigenMethod::AnalyticalHZ).unwrap();
        if status != EigDerivStatus::Success {
            panic!("failed to compute analytical derivative");
        }

        // analytical derivative (Miehe form)
        // let mut spec_miehe = Spectral2::new();
        // let status = spec_miehe .deriv_eigenproj_miehe(&aa, EigenMethod::AnalyticalHZ) .unwrap();
        // if status != EigDerivStatus::Success {
        //     panic!("failed to compute the Miehe derivative");
        // }

        // the two analytical forms must agree
        // for i in 0..3 {
        //     mat_approx_eq(&spec.dpp[i].as_std_matrix(), &spec_miehe.dpp[i].as_std_matrix(), tol);
        // }

        // allocate arguments for numerical differentiation
        let mut args = ArgsNumDerivProj {
            spec: Spectral2::new(),
            i: 0,
            a: aa.clone(),
            m: 0,
            n: 0,
        };

        // check using numerical derivatives
        for i in 0..3 {
            let mut num_deriv = Tensor4::<6>::new();
            args.i = i;
            for m in 0..6 {
                args.m = m;
                for n in 0..6 {
                    args.n = n;
                    let x = args.a.get(args.n);
                    let res = deriv1_central5(x, &mut args, component_of_projector_kelvin).unwrap();
                    num_deriv.set(m, n, res);
                }
            }
            mat_approx_eq(&spec.dpp[i].as_std_matrix(), &num_deriv.as_std_matrix(), tol);
        }
    }

    #[test]
    fn deriv_eigenproj_works() {
        check_ddp(&SamplesTensor2::TENSOR_U, 1e-9);
        check_ddp(&SamplesTensor2::TENSOR_S, 1e-10);
    }

    #[test]
    fn deriv_eigenproj_coalescent_works() {
        // the Panteghini form also handles the coalescent cases
        for sample in [&SamplesTensor2::COAL_01, &SamplesTensor2::COAL_12] {
            let aa = Tensor2::<6>::from_std_matrix(&sample.matrix).unwrap();
            let mut spec = Spectral2::new();
            let status = spec.deriv_eigenproj(&aa, EigenMethod::AnalyticalHZ).unwrap();
            assert_eq!(status, EigDerivStatus::Success);
        }
    }

    #[test]
    fn deriv_eigenproj_spherical_fails() {
        let aa = Tensor2::<6>::identity();
        let mut spec = Spectral2::new();
        let res = spec.deriv_eigenproj(&aa, EigenMethod::AnalyticalHZ);
        assert!(res.is_err());
    }

    #[test]
    fn deriv_eigenproj_scale_invariant() {
        // a uniformly scaled tensor with distinct, non-zero eigenvalues is still
        // invertible (the determinant tolerance must be relative to the magnitude)
        let scale = 1e-4;
        let aa = Tensor2::<6>::from_std_matrix(&[
            [4.0 * scale, 0.0, 0.0],
            [0.0, 2.0 * scale, 0.0],
            [0.0, 0.0, 1.0 * scale],
        ])
        .unwrap();
        let mut spec = Spectral2::new();
        let status = spec.deriv_eigenproj(&aa, EigenMethod::AnalyticalHZ).unwrap();
        assert_eq!(status, EigDerivStatus::Success);
    }
}
