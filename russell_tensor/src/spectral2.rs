use super::{P_SYM, SET, SQRT_2, SQRT_3};
use crate::{StrError, Tensor2, Tensor4, ssd_fn, t2_dyad_t2};
use russell_lab::{small_mat_eigen_sym_jacobi, sort3};

/// Tolerance to assume zero eigenvalue of the deviatoric matrix
const TOL_ZERO_DEV_LAMBDA: f64 = 1e-15;

/// Tolerance to assume coalescent eigenvalues
const TOL_COALESCE: f64 = 1e-8;

/// Tolerance to assume zero eigenvalue during the computation of derivatives of eigenprojectors
///
/// It must be ~ sqrt(EPSILON) because the code performs division by lambda^2
const TOL_LAMBDA: f64 = 1e-8;

/// Holds indices for permutation by looping in 0..3
const INDICES: [usize; 5] = [0, 1, 2, 0, 1];

/// Defines the method to calculate the eigenvalues
///
/// # References
///
/// 1. Habera M. and Zilian A. (2025) Numerically stable evaluation of closed-form
///    expressions for eigenvalues of 3×3 matrices. <https://arxiv.org/abs/2511.00292>
/// 2. Harari I. and Albocher U. (2022) Computation of eigenvalues of a real, symmetric 3x3 matrix
///    with particular reference to the pernicious case of two nearly equal eigenvalues. International
///    Journal for Numerical Methods in Engineering, 124:1089-1110. <https://doi.org/10.1002/nme.7153>
/// 3. Harari I. and Albocher U. (2023) Using the discriminant in a numerically stable symmetric
///    3×3 direct eigenvalue solver. International Journal for Numerical Methods in Engineering,
///    124:4473-4489. <https://doi.org/10.1002/nme.7311>
/// 4. Itskov M. (2019) Tensor Algebra and Tensor Analysis for Engineers With Applications to Continuum
///    Mechanics, Fifth Edition, Springer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EigMethod {
    /// Analytical eigenvalues using Habera-Zilian method
    ///
    /// * Uses Habera-Zilian (2025) to compute the eigenvalues
    /// * Then, uses either Sylvester formula (Itskov 2019) or Jacobi iterations to compute the eigenprojectors
    AnalyticalHZ,

    /// Analytical eigenvalues using Harari-Albocher method (2022)
    ///
    /// * Uses Harari-Albocher (2022) to compute the eigenvalues
    /// * Then, uses either Sylvester formula (Itskov 2019) or Jacobi iterations to compute the eigenprojectors
    AnalyticalHA22,

    /// Analytical eigenvalues using Harari-Albocher method (2023)
    ///
    /// * Uses Harari-Albocher (2023) to compute the eigenvalues
    /// * Then, uses either Sylvester formula (Itskov 2019) or Jacobi iterations to compute the eigenprojectors
    AnalyticalHA23,

    /// Jacobi iterations for eigenvalues and eigenprojectors (via eigenvectors)
    Iterative,
}

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
    /// Input tensor as a 3x3 matrix (for Jacobi method)
    aa_3x3: [[f64; 3]; 3],

    /// Matrix whose columns are the eigenvectors (for Jacobi method)
    vv_3x3: [[f64; 3]; 3],

    /// Auxiliary tensor: ssd(A⁻¹)
    ///
    /// ```text
    ///             _
    /// Y := ½ (A⁻¹ ⊗ A⁻¹ + A⁻¹ ⊗ A⁻¹) = ssd(A⁻¹) / 2
    ///                         ‾
    /// ```
    yy: Option<Tensor4<6>>,

    /// Auxiliary set of tensors (empty by default)
    ///
    /// ```text
    /// P[j] ⊗ P[j]  (no sum on j)
    /// ```
    p_dy_p: Vec<Tensor4<6>>,

    /// Auxiliary deviatoric tensor: S = A - (I1/3) I
    ///
    /// Used in the Harari-Albocher (2022) method
    ss: [f64; 6],

    /// Auxiliary tensor: T = S^2 - (2J2/3) I
    ///
    /// Used in the Harari-Albocher (2022) method
    tt: [f64; 6],
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
            aa_3x3: [[0.0; 3]; 3],
            vv_3x3: [[0.0; 3]; 3],
            yy: None,
            p_dy_p: Vec::new(),
            // private: auxiliary tensors
            ss: [0.0; 6],
            tt: [0.0; 6],
        }
    }

    /// Calculates the eigenvalues (but not the eigenprojectors) of a symmetric second-order tensor (using the default method)
    ///
    /// The output is saved in this struct with the eigenvalues being sorted in descending order.
    /// The status is saved in `status`.
    ///
    /// Default method: [EigMethod::AnalyticalHZ]
    pub fn calc_eigenvalues(&mut self, aa: &Tensor2<6>) -> Result<(), StrError> {
        self.calc_eigenvalues_mx(aa, EigMethod::AnalyticalHZ)
    }

    /// Calculates the eigenvalues (but not the eigenprojectors) of a symmetric second-order tensor
    ///
    /// The output is saved in this struct with the eigenvalues being sorted in descending order.
    /// The status is saved in `status`.
    pub fn calc_eigenvalues_mx(&mut self, aa: &Tensor2<6>, method: EigMethod) -> Result<(), StrError> {
        // indicate that the eigenvalues and projectors are not available
        self.status = EigStatus::NotComputed;

        // detect a (numerically) spherical tensor, i.e., J2 at the rounding level
        let ii1 = aa.invariant_ii1();
        let iso = ii1 / 3.0;
        let jj2 = aa.invariant_jj2();
        if Self::is_spherical(aa) {
            self.lam[0] = iso;
            self.lam[1] = iso;
            self.lam[2] = iso;
            self.status = EigStatus::Spherical;
            return Ok(());
        }

        // Jacobi iterative method: calculate the eigenvalues (ignores eigenvectors)
        if method == EigMethod::Iterative {
            // eigenvalues and eigenvectors (ignored)
            aa.to_std_matrix_slice(&mut self.aa_3x3);
            small_mat_eigen_sym_jacobi(&mut self.lam, &mut self.vv_3x3, &mut self.aa_3x3)?;
            self.sort_eigenvalues();
            self.status = self.classify();
            return Ok(());
        }

        // Analytical methods: calculate the eigenvalues for non-spherical cases
        match method {
            //
            // Habera M. and Zilian A. (2025)
            //
            EigMethod::AnalyticalHZ => {
                // auxiliary variables
                let d0 = aa.vec[0] - aa.vec[1];
                let d1 = aa.vec[0] - aa.vec[2];
                let d2 = aa.vec[1] - aa.vec[2];
                let w = aa.vec[3] / SQRT_2;
                let v = aa.vec[5] / SQRT_2;
                let u = aa.vec[4] / SQRT_2;
                let t1 = d1 + d2;
                let t2 = d0 - d2;
                let t3 = -d0 - d1;
                let jj3 = 2.0 * w * u * v + (w * w * t1 + v * v * t2 + u * u * t3) / 3.0 - t1 * t2 * t3 / 27.0;
                // calculate delta (discriminant)
                let alpha = d2;
                let beta = -d1;
                let gamma = d0;
                let terms = [
                    3.0 * f64::sqrt(3.0) * (v * w * alpha + u * (v * v - w * w)),
                    alpha * beta * gamma + alpha * u * u + beta * v * v + gamma * w * w,
                    2.0 * u * beta * gamma - v * w * (beta - gamma) + u * (2.0 * u * u - v * v - w * w),
                    2.0 * (v * alpha * gamma + u * w * (beta - gamma) + v * (v * v + w * w - 2.0 * u * u)),
                    2.0 * (w * alpha * beta + u * v * (beta - gamma) + w * (v * v + w * w - 2.0 * u * u)),
                ];
                let mut delta = 0.0;
                for term in terms {
                    delta = term.mul_add(term, delta);
                }
                // calculate the eigenvalues using closed-form
                let phi = f64::atan2(f64::sqrt(27.0 * delta), 27.0 * jj3);
                let amplitude = 2.0 * f64::sqrt(3.0 * jj2);
                let two_pi = 2.0 * std::f64::consts::PI;
                let angle0 = (phi + two_pi * 1.0) / 3.0;
                let angle1 = (phi + two_pi * 2.0) / 3.0;
                let angle2 = (phi + two_pi * 3.0) / 3.0;
                self.lam[0] = amplitude.mul_add(f64::cos(angle0), ii1) / 3.0;
                self.lam[1] = amplitude.mul_add(f64::cos(angle1), ii1) / 3.0;
                self.lam[2] = amplitude.mul_add(f64::cos(angle2), ii1) / 3.0;
            }
            //
            // Harari I. and Albocher U. (2022)
            //
            EigMethod::AnalyticalHA22 => {
                let sqrt_jj2 = f64::sqrt(jj2);
                let fac1 = 2.0 * jj2 / 3.0;
                let fac2 = sqrt_jj2 / SQRT_3;
                self.ss[0] = aa.vec[0] - ii1 / 3.0;
                self.ss[1] = aa.vec[1] - ii1 / 3.0;
                self.ss[2] = aa.vec[2] - ii1 / 3.0;
                self.ss[3] = aa.vec[3];
                self.ss[4] = aa.vec[4];
                self.ss[5] = aa.vec[5];
                let s = &self.ss;
                self.tt[0] = s[0] * s[0] + s[3] * s[3] / 2.0 + s[5] * s[5] / 2.0 - fac1;
                self.tt[1] = s[1] * s[1] + s[3] * s[3] / 2.0 + s[4] * s[4] / 2.0 - fac1;
                self.tt[2] = s[2] * s[2] + s[4] * s[4] / 2.0 + s[5] * s[5] / 2.0 - fac1;
                self.tt[3] = (s[0] + s[1]) * s[3] + s[4] * s[5] / SQRT_2;
                self.tt[4] = (s[1] + s[2]) * s[4] + s[3] * s[5] / SQRT_2;
                self.tt[5] = (s[0] + s[2]) * s[5] + s[3] * s[4] / SQRT_2;
                let num = sq_norm_diff(&self.tt, -fac2, &self.ss);
                let den = sq_norm_diff(&self.tt, fac2, &self.ss);
                // this is not d in Eq (70) of Ref #1; it is the newly defined d in Box 1 of Ref #1
                let d_box = f64::sqrt(num / den);
                let sj = f64::signum(1.0 - d_box);
                if sj * (1.0 - d_box) < TOL_ZERO_DEV_LAMBDA {
                    // deviatoric matrix has a zero eigenvalue
                    self.lam[0] = iso + sqrt_jj2;
                    self.lam[1] = iso;
                    self.lam[2] = iso - sqrt_jj2;
                } else {
                    // deviatoric matrix doesn't have zero eigenvalue
                    let dsj = if sj < 0.0 { 1.0 / d_box } else { d_box };
                    let alpha = 2.0 * f64::atan(dsj) / 3.0;
                    let cd = sj * fac2 * f64::cos(alpha);
                    let sd = sqrt_jj2 * f64::sin(alpha);
                    self.lam[0] = iso + 2.0 * cd;
                    self.lam[1] = iso - cd + sd;
                    self.lam[2] = iso - cd - sd;
                }
            }
            //
            // Harari I. and Albocher U. (2023)
            //
            EigMethod::AnalyticalHA23 => {
                const R1_2: f64 = SQRT_2 / 2.0; // 1/√2
                let a = &aa.vec;
                let d12 = a[0] - a[1];
                let d23 = a[1] - a[2];
                let d31 = a[2] - a[0];
                let s01 = a[3] * R1_2;
                let s12 = a[4] * R1_2;
                let s02 = a[5] * R1_2;
                let jj3 = aa.invariant_jj3();
                let sd = if jj3 >= 0.0 { 1.0 } else { -1.0 };
                // discriminant as a sum of seven squares (Equation 17)
                let hx = d12 * d23 * d31 + s01 * s01 * d12 + s12 * s12 * d23 + s02 * s02 * d31;
                let hy1 = s12 * (2.0 * s12 * s12 - s02 * s02 - s01 * s01 + 2.0 * d12 * d31) + s01 * s02 * (d12 - d31);
                let hy2 = s02 * (2.0 * s02 * s02 - s12 * s12 - s01 * s01 + 2.0 * d23 * d12) + s01 * s12 * (d23 - d12);
                let hy3 = s01 * (2.0 * s01 * s01 - s12 * s12 - s02 * s02 + 2.0 * d31 * d23) + s02 * s12 * (d31 - d23);
                let hz1 = s12 * (s02 * s02 - s01 * s01) + s01 * s02 * d23;
                let hz2 = s02 * (s01 * s01 - s12 * s12) + s12 * s01 * d31;
                let hz3 = s01 * (s12 * s12 - s02 * s02) + s02 * s12 * d12;
                let delta =
                    (hx * hx + hy1 * hy1 + hy2 * hy2 + hy3 * hy3 + 15.0 * (hz1 * hz1 + hz2 * hz2 + hz3 * hz3)).max(0.0);
                // mixed tangent angle (Equation 11)
                let sqrt_jj2 = f64::sqrt(jj2);
                let numerator = f64::sqrt(delta);
                let denominator = 2.0 * jj2 * sqrt_jj2 + 3.0 * SQRT_3 * sd * jj3;
                let alpha = (2.0 / 3.0) * f64::atan2(numerator, denominator);
                // deviatoric eigenvalues (Equations 12-14)
                let lambda1 = 2.0 * sd * f64::sqrt(jj2 / 3.0) * f64::cos(alpha);
                let lambda2 = sd * sqrt_jj2 * f64::sin(alpha) - lambda1 / 2.0;
                let lambda3 = -sd * sqrt_jj2 * f64::sin(alpha) - lambda1 / 2.0;
                self.lam[0] = iso + lambda1;
                self.lam[1] = iso + lambda2;
                self.lam[2] = iso + lambda3;
            }
            EigMethod::Iterative => unreachable!("handled above"),
        };

        // sort the eigenvalues in descending order
        self.sort_eigenvalues();

        // save the status
        self.status = self.classify();
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
        self.decompose_mx(aa, EigMethod::AnalyticalHZ)
    }

    /// Performs the spectral decomposition of a symmetric second-order tensor (specifying the method)
    ///
    /// The output is saved in this struct with the eigenvalues/projectors being sorted in descending order.
    /// The status is saved in `status`.
    pub fn decompose_mx(&mut self, aa: &Tensor2<6>, method: EigMethod) -> Result<(), StrError> {
        // indicate that the eigenvalues and projectors are not available
        self.status = EigStatus::NotComputed;

        // Jacobi iterative method: calculate the eigenvalues and eigenprojectors
        if method == EigMethod::Iterative {
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
            self.proj[1].vec[1] = 1.0;
            self.proj[2].vec[2] = 1.0;
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

    /// Calculates the derivatives of the eigenprojectors w.r.t. the defining tensor
    ///
    /// Note: This function is only available for *invertible* tensor A with *distinct*
    /// and *non-zero* eigenvalues.
    ///
    /// ```text
    /// dP[i]                         3
    /// ───── = a[i] Psym - b[i] Y +  Σ (c[i][j] - a[i]) P[j] ⊗ P[j]
    ///  dA                          j=1
    ///
    /// where Y = ½ ssd(A⁻¹) and the coefficients are listed below.
    /// ```
    ///
    /// The spectral decomposition is performed internally using the given [EigMethod].
    ///
    /// If the eigenvalues are distinct and non-zero, the results will be available in
    /// [Spectral2], including the inverse `A⁻¹` (in [Spectral2::aa_inv]) and the
    /// derivatives of the eigenprojectors in [Spectral2::dpp].
    ///
    /// # Input
    ///
    /// `aa` -- The symmetric tensor A
    /// `method` -- The method to calculate the eigenvalues
    ///
    /// # Output
    ///
    /// Returns [EigDerivStatus::Success] if:
    ///
    /// 1. The eigenvalues are distinct
    /// 2. All eigenvalues are non-zero
    /// 3. The tensor is invertible (determinant above a scale-relative tolerance)
    ///
    /// Otherwise, returns [EigDerivStatus::FailDueToCoalescent],
    /// [EigDerivStatus::FailDueToZeroEigenvalue], or [EigDerivStatus::FailDueToNonInvertible].
    ///
    /// # Errors
    ///
    /// Returns an error if a coefficient `d[i]` is (nearly) zero.
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
    pub fn deriv_eigenproj(&mut self, aa: &Tensor2<6>, method: EigMethod) -> Result<EigDerivStatus, StrError> {
        // compute the eigenvalues and eigenprojectors
        self.decompose_mx(aa, method)?;

        // check for distinct eigenvalues (the status is up to date because the projectors are available)
        if self.status != EigStatus::Distinct {
            // the derivative of eigenprojectors is only available for distinct eigenvalues
            return Ok(EigDerivStatus::FailDueToCoalescent);
        }

        // check for null eigenvalues
        for i in 0..3 {
            if f64::abs(self.lam[i]) < TOL_LAMBDA {
                // cannot compute the derivatives because an eigenvalue is nearly zero
                return Ok(EigDerivStatus::FailDueToZeroEigenvalue);
            }
        }

        // use a determinant tolerance relative to the magnitude of the tensor so that a
        // uniform scaling of A does not change whether it is deemed invertible
        let norm = aa.norm();
        let det_tol = TOL_LAMBDA * norm * norm * norm;

        // calculate A⁻¹, the inverse of A, and I3 = det(A)
        let det = aa.inverse(&mut self.aa_inv, det_tol);
        if det.is_none() {
            // cannot compute the derivatives because the tensor is not invertible
            return Ok(EigDerivStatus::FailDueToNonInvertible);
        }
        let ii3 = det.unwrap();

        // calculate the auxiliary tensor Y = ssd(A⁻¹) / 2
        if self.yy.is_none() {
            self.yy = Some(Tensor4::<6>::new());
        }
        let yy = self.yy.as_mut().unwrap();
        ssd_fn(yy, SET, 0.5, &self.aa_inv);

        // allocate and calculate auxiliary tensors P[j] ⊗ P[j]
        if self.p_dy_p.len() != 3 {
            self.p_dy_p = vec![Tensor4::<6>::new(), Tensor4::<6>::new(), Tensor4::<6>::new()];
        }
        t2_dyad_t2(&mut self.p_dy_p[0], SET, 1.0, &self.proj[0], &self.proj[0]);
        t2_dyad_t2(&mut self.p_dy_p[1], SET, 1.0, &self.proj[1], &self.proj[1]);
        t2_dyad_t2(&mut self.p_dy_p[2], SET, 1.0, &self.proj[2], &self.proj[2]);

        // calculate auxiliary coefficients
        let mut d = [0.0; 3];
        let mut a = [0.0; 3];
        let mut b = [0.0; 3];
        let mut c = [[0.0; 3]; 3];
        let ii1 = aa.invariant_ii1();
        for i in 0..3 {
            d[i] = 2.0 * self.lam[i] * self.lam[i] - ii1 * self.lam[i] + ii3 / self.lam[i];
            if f64::abs(d[i]) < TOL_LAMBDA {
                return Err("|d[i]| is nearly zero");
            }
            a[i] = self.lam[i] / d[i];
            b[i] = ii3 / d[i];
            for j in 0..3 {
                c[i][j] = ii3 / (d[i] * self.lam[j] * self.lam[j]);
            }
        }

        // Allocate output tensors
        if self.dpp.len() != 3 {
            self.dpp = vec![Tensor4::<6>::new(), Tensor4::<6>::new(), Tensor4::<6>::new()];
        }

        // Compute the derivatives
        for i in 0..3 {
            for m in 0..6 {
                for n in 0..6 {
                    let p = a[i] * P_SYM[m][n] - b[i] * yy.get(m, n);
                    let q0 = (c[i][0] - a[i]) * self.p_dy_p[0].get(m, n);
                    let q1 = (c[i][1] - a[i]) * self.p_dy_p[1].get(m, n);
                    let q2 = (c[i][2] - a[i]) * self.p_dy_p[2].get(m, n);
                    self.dpp[i].set(m, n, p + q0 + q1 + q2);
                }
            }
        }
        Ok(EigDerivStatus::Success)
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

    /// (internal) Sorts the eigenvalues in descending order
    #[inline]
    fn sort_eigenvalues(&mut self) {
        let mut l0 = self.lam[0];
        let mut l1 = self.lam[1];
        let mut l2 = self.lam[2];
        sort3(&mut l2, &mut l1, &mut l0); // will sort: l2 < l1 < l0
        self.lam[0] = l0;
        self.lam[1] = l1;
        self.lam[2] = l2;
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

/// Calculates ||a + alpha * b||^2
#[rustfmt::skip]
#[inline]
fn sq_norm_diff(a: &[f64], alpha: f64, b: &[f64]) -> f64 {
      (a[0] + alpha * b[0]) * (a[0] + alpha * b[0])
    + (a[1] + alpha * b[1]) * (a[1] + alpha * b[1])
    + (a[2] + alpha * b[2]) * (a[2] + alpha * b[2])
    + (a[3] + alpha * b[3]) * (a[3] + alpha * b[3])
    + (a[4] + alpha * b[4]) * (a[4] + alpha * b[4])
    + (a[5] + alpha * b[5]) * (a[5] + alpha * b[5])
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
    use super::{EigMethod, EigStatus, Spectral2, t2_plus_diag_product};
    use crate::{EigDerivStatus, SampleTensor2, SamplesTensor2, StrError, Tensor2, Tensor4};
    use crate::{IDENTITY2, SQRT_2, SQRT_3, SQRT_6};
    use russell_lab::{Matrix, approx_eq, array_approx_eq, deriv1_central5, mat_approx_eq, mat_mat_mul};

    #[cfg(feature = "heap")]
    use russell_lab::vec_approx_eq;

    //
    // --- test essential method --------------
    //

    #[test]
    fn t2_plus_diag_product_works() {
        let aa = Tensor2::<6>::from_std_matrix(&SamplesTensor2::TENSOR_U.matrix).unwrap();
        let mut res = [0.0; 6];
        let (alpha, p, q) = (0.5, -1.5, 2.0);
        t2_plus_diag_product(&mut res, alpha, aa.as_data(), p, q);
        let mut aa_plus_p_times_ii = [[0.0; 3]; 3];
        let mut aa_plus_q_times_ii = [[0.0; 3]; 3];
        let mut expected_mat = [[0.0; 3]; 3];
        let ii_mat = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let aa_mat = aa.as_std_matrix();
        for i in 0..3 {
            for j in 0..3 {
                aa_plus_p_times_ii[i][j] = aa_mat[(i, j)] + p * ii_mat[i][j];
                aa_plus_q_times_ii[i][j] = aa_mat[(i, j)] + q * ii_mat[i][j];
            }
        }
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    expected_mat[i][j] += alpha * aa_plus_p_times_ii[i][k] * aa_plus_q_times_ii[k][j];
                }
            }
        }
        let expected = Tensor2::<6>::from_std_matrix(&expected_mat).unwrap();
        array_approx_eq(expected.as_data(), &res, 1e-14);
    }

    //
    // --- auxiliary --------------------------
    //

    /// Performs similarity transformation (make sure to return a symmetric matrix)
    ///
    /// ```text
    /// A = Q . L . Q^T
    /// ```
    fn transform(aa: &mut [[f64; 3]; 3], ll: &[[f64; 3]; 3], qq: &[[f64; 3]; 3]) {
        for i in 0..3 {
            for j in 0..3 {
                aa[i][j] = 0.0;
                for k in 0..3 {
                    for l in 0..3 {
                        aa[i][j] += qq[i][k] * ll[k][l] * qq[j][l];
                    }
                }
            }
        }
        for i in 0..3 {
            for j in i..3 {
                aa[i][j] = aa[j][i]; // symmetrize
            }
        }
    }

    #[test]
    fn check_transform() {
        // Q rotates axes to octahedral system
        #[rustfmt::skip]
        let qq_3x3 = [
            [2.0 / SQRT_6, -1.0 / SQRT_6, -1.0 / SQRT_6],
            [1.0 / SQRT_3,  1.0 / SQRT_3,  1.0 / SQRT_3],
            [0.0,          -1.0 / SQRT_2,  1.0 / SQRT_2],
        ];
        let l1 = 1.0;
        let l2 = 2.0;
        let l3 = 3.0;
        let ll = [[l1, 0.0, 0.0], [0.0, l2, 0.0], [0.0, 0.0, l3]];
        let mut aa_3x3 = [[0.0; 3]; 3];
        // transform and check invariants
        transform(&mut aa_3x3, &ll, &qq_3x3);
        let aa = Tensor2::<6>::from_std_matrix(&aa_3x3).unwrap();
        approx_eq(aa.invariant_ii1(), l1 + l2 + l3, 1e-15);
        approx_eq(aa.invariant_ii2(), l1 * l2 + l2 * l3 + l3 * l1, 1e-14);
        approx_eq(aa.invariant_ii3(), l1 * l2 * l3, 1e-14);
        approx_eq(aa.norm(), f64::sqrt(l1 * l1 + l2 * l2 + l3 * l3), 1e-15);
        #[rustfmt::skip]
        let qqt_3x3 = [
            [ 2.0 / SQRT_6, 1.0 / SQRT_3,  0.0         ],
            [-1.0 / SQRT_6, 1.0 / SQRT_3, -1.0 / SQRT_2],
            [-1.0 / SQRT_6, 1.0 / SQRT_3,  1.0 / SQRT_2],
        ];
        // transform back and compare matrices
        let mut ll_3x3 = [[0.0; 3]; 3];
        transform(&mut ll_3x3, &aa_3x3, &qqt_3x3);
        mat_approx_eq(&Matrix::from(&ll_3x3), &ll, 1e-14);
    }

    /// Returns the Habera-Zilian test names
    fn hz_cases() -> [&'static str; 11] {
        [
            "single",
            "single_lim_J3",
            "single_lim_disc_t",
            "single_lim_disc_n",
            "single_lim_J3J2",
            "single_J3",
            "single_J3_lim_J2",
            "double",
            "double_lim_J3J2",
            "triple_J3",
            "d3",
        ]
    }

    /// Returns the Habera-Zilian deltas to generate the test matrix
    fn hz_deltas() -> [f64; 10] {
        [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 5.0, 500.0]
    }

    /// Returns the prescribed diagonal of the Habera-Zilian test tensor
    fn hz_diagonal(name: &str, delta: f64) -> [f64; 3] {
        const A: f64 = 1.0;
        match name {
            "single" => [-A / 4.0, (1.0 * A) / 4.0, (2.0 + 2.0 * delta) * A / 4.0],
            "single_lim_J3" => [(-1.0 - delta) * A / 4.0, 0.0, (1.0 + 2.0 * delta) * A / 4.0],
            "single_lim_disc_t" => [-A, 1.0 * A, (1.0 + delta) * A],
            "single_lim_disc_n" => [0.0, (2.0 - delta) * A / 2.0, (2.0 + delta) * A / 2.0],
            "single_lim_J3J2" => [(1.0 - delta) * A, 1.0 * A, (1.0 + 2.0 * delta) * A],
            "single_J3" => [(-1.0 - delta) * A / 2.0, 0.0, (1.0 + delta) * A / 2.0],
            "single_J3_lim_J2" => [(1.0 - delta) * A, 1.0 * A, (1.0 + delta) * A],
            "double" => [(-1.0 - delta) * A, 1.0 * A, 1.0 * A],
            "double_lim_J3J2" => [1.0 * A, 1.0 * A, (1.0 + delta) * A],
            "triple_J3" => [-delta, 0.0, delta],
            "d3" => [0.0, 1.0 * A, (2.0 + delta) * A],
            _ => panic!("unknown HZ test case: {}", name),
        }
    }

    /// Generates the Habera-Zilian test tensor
    fn hz_tensor(name: &str, delta: f64) -> Tensor2<6> {
        let d = hz_diagonal(name, delta);
        let qq_3x3 = [
            [1.0 / SQRT_2, -0.5, 0.5],
            [1.0 / SQRT_2, 0.5, -0.5],
            [0.0, 1.0 / SQRT_2, 1.0 / SQRT_2],
        ];
        let mut aa_3x3 = [[0.0; 3]; 3];
        let ll = [[d[0], 0.0, 0.0], [0.0, d[1], 0.0], [0.0, 0.0, d[2]]];
        transform(&mut aa_3x3, &ll, &qq_3x3);
        Tensor2::from_std_matrix(&aa_3x3).unwrap()
    }

    /// Sort eigenvalues and projectors in descending order
    fn sort_projectors(lambda: &mut [f64; 3], projectors: &mut [[[f64; 3]; 3]; 3]) {
        let mut indices = [0, 1, 2];
        indices.sort_by(|&i, &j| lambda[j].partial_cmp(&lambda[i]).unwrap());
        let sorted_lambda = [lambda[indices[0]], lambda[indices[1]], lambda[indices[2]]];
        let sorted_projectors = [projectors[indices[0]], projectors[indices[1]], projectors[indices[2]]];
        *lambda = sorted_lambda;
        *projectors = sorted_projectors;
    }

    #[test]
    fn check_sort() {
        let mut lambda = [1.0, 3.0, 2.0];
        let aaa = 123.0;
        let bbb = 456.0;
        let ccc = 789.0;
        let mut projectors = [
            [[aaa, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, bbb, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, ccc]],
        ];
        sort_projectors(&mut lambda, &mut projectors);
        assert_eq!(lambda[0], 3.0);
        assert_eq!(lambda[1], 2.0);
        assert_eq!(lambda[2], 1.0);
        assert_eq!(projectors[0][1][1], bbb);
        assert_eq!(projectors[1][2][2], ccc);
        assert_eq!(projectors[2][0][0], aaa);
    }

    /// Check the properties of eigenprojectors
    fn check_eigenprojectors(pp_all: &[Tensor2<6>], tol: f64) {
        // sum check: P0 + P1 + P2 = I
        let mut sum = [0.0; 6];
        for i in 0..3 {
            for m in 0..6 {
                sum[m] += pp_all[i].get(m);
            }
        }
        array_approx_eq(&sum, &IDENTITY2[..6], tol);

        // orthogonality check: P[i] . P[j] = δ[i,j] P[i]
        let zero = [[0.0; 3]; 3];
        let mut ppi_times_ppj = Matrix::new(3, 3);
        for i in 0..3 {
            let ppi = pp_all[i].as_std_matrix();
            for j in 0..3 {
                let ppj = pp_all[j].as_std_matrix();
                mat_mat_mul(&mut ppi_times_ppj, 1.0, &ppi, &ppj, 0.0).unwrap();
                if i == j {
                    mat_approx_eq(&ppi_times_ppj, &ppi, tol);
                } else {
                    mat_approx_eq(&ppi_times_ppj, &zero, tol);
                }
            }
        }
    }

    /// Generates eigen-problem (with checks using check_eigenprojectors)
    ///
    /// Returns `(aa, expected_lambda, expected_proj)` sorted in decreasing order by lambda
    fn generate_eigen_problem(l1: f64, l2: f64, l3: f64) -> ([[f64; 3]; 3], [f64; 3], [[[f64; 3]; 3]; 3]) {
        // Q rotates axes to octahedral system
        #[rustfmt::skip]
        let qq = [
            [2.0 / SQRT_6, -1.0 / SQRT_6, -1.0 / SQRT_6],
            [1.0 / SQRT_3,  1.0 / SQRT_3,  1.0 / SQRT_3],
            [0.0,          -1.0 / SQRT_2,  1.0 / SQRT_2],
        ];
        // A = Q . L . Q^T
        let mut aa = [[0.0; 3]; 3];
        transform(&mut aa, &[[l1, 0.0, 0.0], [0.0, l2, 0.0], [0.0, 0.0, l3]], &qq);
        // expected eigenvectors
        #[rustfmt::skip]
        let n0 = [
            2.0 / SQRT_6,
            1.0 / SQRT_3,
            0.0,
        ];
        #[rustfmt::skip]
        let n1 = [
            -1.0 / SQRT_6,
             1.0 / SQRT_3,
            -1.0 / SQRT_2,
        ];
        #[rustfmt::skip]
        let n2 = [
            -1.0 / SQRT_6,
             1.0 / SQRT_3,
             1.0 / SQRT_2,
        ];
        // expected eigenprojectors
        let pp0 = [
            [n0[0] * n0[0], n0[0] * n0[1], n0[0] * n0[2]],
            [n0[1] * n0[0], n0[1] * n0[1], n0[1] * n0[2]],
            [n0[2] * n0[0], n0[2] * n0[1], n0[2] * n0[2]],
        ];
        let pp1 = [
            [n1[0] * n1[0], n1[0] * n1[1], n1[0] * n1[2]],
            [n1[1] * n1[0], n1[1] * n1[1], n1[1] * n1[2]],
            [n1[2] * n1[0], n1[2] * n1[1], n1[2] * n1[2]],
        ];
        let pp2 = [
            [n2[0] * n2[0], n2[0] * n2[1], n2[0] * n2[2]],
            [n2[1] * n2[0], n2[1] * n2[1], n2[1] * n2[2]],
            [n2[2] * n2[0], n2[2] * n2[1], n2[2] * n2[2]],
        ];
        // sort eigen variables
        let mut expected_lambda = [l1, l2, l3];
        let mut expected_proj = [pp0, pp1, pp2];
        sort_projectors(&mut expected_lambda, &mut expected_proj);
        let e_projectors = [
            Tensor2::<6>::from_std_matrix(&expected_proj[0]).unwrap(),
            Tensor2::<6>::from_std_matrix(&expected_proj[1]).unwrap(),
            Tensor2::<6>::from_std_matrix(&expected_proj[2]).unwrap(),
        ];
        // check
        check_eigenprojectors(&e_projectors, 1e-15);
        // results
        (aa, expected_lambda, expected_proj)
    }

    /// Calculates A = Σ λ[k] * P[k]
    fn compose(aa: &mut Tensor2<6>, spc: &Spectral2) {
        aa.vec[0] = spc.lam[0] * spc.proj[0].vec[0] + spc.lam[1] * spc.proj[1].vec[0] + spc.lam[2] * spc.proj[2].vec[0];
        aa.vec[1] = spc.lam[0] * spc.proj[0].vec[1] + spc.lam[1] * spc.proj[1].vec[1] + spc.lam[2] * spc.proj[2].vec[1];
        aa.vec[2] = spc.lam[0] * spc.proj[0].vec[2] + spc.lam[1] * spc.proj[1].vec[2] + spc.lam[2] * spc.proj[2].vec[2];
        aa.vec[3] = spc.lam[0] * spc.proj[0].vec[3] + spc.lam[1] * spc.proj[1].vec[3] + spc.lam[2] * spc.proj[2].vec[3];
        aa.vec[4] = spc.lam[0] * spc.proj[0].vec[4] + spc.lam[1] * spc.proj[1].vec[4] + spc.lam[2] * spc.proj[2].vec[4];
        aa.vec[5] = spc.lam[0] * spc.proj[0].vec[5] + spc.lam[1] * spc.proj[1].vec[5] + spc.lam[2] * spc.proj[2].vec[5];
    }

    /// Check the solution to the eigen-problem on tensor A
    fn check_eigen_problem(aa: &Tensor2<6>, spec: &Spectral2, tol_proj: f64, tol_compose: f64) {
        // check eigenprojectors
        check_eigenprojectors(&spec.proj, tol_proj);

        // check composed matrix
        let mut bb = Tensor2::<6>::new();
        compose(&mut bb, spec);
        #[cfg(feature = "heap")]
        vec_approx_eq(&aa.vec, &bb.vec, tol_compose);
        #[cfg(not(feature = "heap"))]
        array_approx_eq(&aa.vec, &bb.vec, tol_compose);
    }

    /// Checks the eigen-problem by comparing with known values
    fn check(
        method: EigMethod,
        spec: &mut Spectral2,
        sample: &SampleTensor2,
        tol_lambda: f64,
        tol_proj: f64,
        tol_compose: f64,
    ) {
        // extract eigenvalues
        let correct_lambda = sample.eigenvalues.unwrap();

        // perform the spectral decomposition
        let aa = Tensor2::<6>::from_std_matrix(&sample.matrix).unwrap();
        spec.decompose_mx(&aa, method).unwrap();

        // output (for debugging)
        // println!("eigenvalues = {:?}", spec.lam);
        // println!("P0 =\n{:.15}", spec.proj[0].as_std_matrix());
        // println!("P1 =\n{:.15}", spec.proj[1].as_std_matrix());
        // println!("P2 =\n{:.15}", spec.proj[2].as_std_matrix());

        // compare eigenvalues
        array_approx_eq(&spec.lam, &correct_lambda, tol_lambda);

        // compare eigenprojectors
        // note: for spherical tensors, the eigenprojectors are not unique, so this check is skipped
        if spec.status != EigStatus::Spherical {
            let correct_projectors = sample.eigenprojectors.unwrap();
            let pp0 = spec.proj[0].as_std_matrix();
            let pp1 = spec.proj[1].as_std_matrix();
            let pp2 = spec.proj[2].as_std_matrix();
            let correct0 = Matrix::from(&correct_projectors[0]);
            let correct1 = Matrix::from(&correct_projectors[1]);
            let correct2 = Matrix::from(&correct_projectors[2]);
            mat_approx_eq(&correct0, &pp0, tol_proj);
            mat_approx_eq(&correct1, &pp1, tol_proj);
            mat_approx_eq(&correct2, &pp2, tol_proj);
        }

        // further checks
        check_eigen_problem(&aa, spec, tol_proj, tol_compose);
    }

    //
    // --- tests -------------------------------
    //

    #[test]
    fn decompose_and_compose_using_jacobi_work_with_samples() {
        let mut spec = Spectral2::new();
        let m = EigMethod::Iterative;
        check(m, &mut spec, &SamplesTensor2::TENSOR_O, 1e-15, 1e-15, 1e-15);
        check(m, &mut spec, &SamplesTensor2::TENSOR_I, 1e-15, 1e-15, 1e-15);
        check(m, &mut spec, &SamplesTensor2::TENSOR_X, 1e-15, 1e-15, 1e-15);
        check(m, &mut spec, &SamplesTensor2::TENSOR_Y, 1e-13, 1e-15, 1e-15);
        check(m, &mut spec, &SamplesTensor2::TENSOR_Z, 1e-14, 1e-15, 1e-15);
        check(m, &mut spec, &SamplesTensor2::TENSOR_U, 1e-13, 1e-15, 1e-14);
        check(m, &mut spec, &SamplesTensor2::TENSOR_S, 1e-13, 1e-14, 1e-14);
    }

    #[test]
    fn decompose_and_compose_using_harari_albocher22_work_with_samples() {
        let mut spec = Spectral2::new();
        let m = EigMethod::AnalyticalHA22;
        check(m, &mut spec, &SamplesTensor2::TENSOR_O, 1e-15, 1e-15, 1e-15);
        check(m, &mut spec, &SamplesTensor2::TENSOR_I, 1e-15, 1e-15, 1e-15);
        check(m, &mut spec, &SamplesTensor2::TENSOR_X, 1e-15, 1e-15, 1e-15);
        check(m, &mut spec, &SamplesTensor2::TENSOR_Y, 1e-13, 1e-15, 1e-14);
        check(m, &mut spec, &SamplesTensor2::TENSOR_Z, 1e-14, 1e-14, 1e-14);
        check(m, &mut spec, &SamplesTensor2::TENSOR_U, 1e-13, 1e-14, 1e-14);
        check(m, &mut spec, &SamplesTensor2::TENSOR_S, 1e-13, 1e-14, 1e-14);
    }

    #[test]
    fn decompose_and_compose_using_harari_albocher23_work_with_samples() {
        let mut spec = Spectral2::new();
        let m = EigMethod::AnalyticalHA23;
        check(m, &mut spec, &SamplesTensor2::TENSOR_O, 1e-15, 1e-15, 1e-15);
        check(m, &mut spec, &SamplesTensor2::TENSOR_I, 1e-15, 1e-15, 1e-15);
        check(m, &mut spec, &SamplesTensor2::TENSOR_X, 1e-15, 1e-15, 1e-15);
        check(m, &mut spec, &SamplesTensor2::TENSOR_Y, 1e-13, 1e-15, 1e-14);
        check(m, &mut spec, &SamplesTensor2::TENSOR_Z, 1e-14, 1e-14, 1e-14);
        check(m, &mut spec, &SamplesTensor2::TENSOR_U, 1e-13, 1e-14, 1e-14);
        check(m, &mut spec, &SamplesTensor2::TENSOR_S, 1e-13, 1e-14, 1e-14);
    }

    #[test]
    fn decompose_and_compose_using_habera_zilian_work_with_samples() {
        let mut spec = Spectral2::new();
        let m = EigMethod::AnalyticalHZ;
        check(m, &mut spec, &SamplesTensor2::TENSOR_O, 1e-15, 1e-15, 1e-15);
        check(m, &mut spec, &SamplesTensor2::TENSOR_I, 1e-15, 1e-15, 1e-15);
        check(m, &mut spec, &SamplesTensor2::TENSOR_X, 1e-14, 1e-15, 1e-15);
        check(m, &mut spec, &SamplesTensor2::TENSOR_Y, 1e-13, 1e-15, 1e-14);
        check(m, &mut spec, &SamplesTensor2::TENSOR_Z, 1e-14, 1e-14, 1e-14);
        check(m, &mut spec, &SamplesTensor2::TENSOR_U, 1e-13, 1e-14, 1e-14);
        check(m, &mut spec, &SamplesTensor2::TENSOR_S, 1e-13, 1e-14, 1e-14);
    }

    #[test]
    fn decompose_reconstruction_works() {
        // Checks the eigen-decomposition reconstruction `A = Σ λᵢ Pᵢ` for random-like,
        // two-nearly-equal, and triple-equal eigenvalues.
        let r2 = f64::sqrt(2.0);
        let r3 = f64::sqrt(3.0);
        let r6 = f64::sqrt(6.0);
        // orthogonal matrices
        #[rustfmt::skip]
        let rotations = [
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ],
            [
                [2.0 / r6, -1.0 / r6, -1.0 / r6],
                [1.0 / r3,  1.0 / r3,  1.0 / r3],
                [0.0,      -1.0 / r2,  1.0 / r2],
            ],
        ];
        // diagonals: random-like, two-nearly-equal, and triple-equal
        let mut diagonals: Vec<[f64; 3]> = vec![[3.0, 1.0, 2.0]];
        for eps in [1e-3, 1e-6, 1e-9, 1e-12, 1e-15] {
            diagonals.push([1.0, -0.5 + eps / 2.0, -0.5 - eps / 2.0]);
        }
        diagonals.push([1.0, 1.0, 1.0]);
        // run the test
        for method in [
            EigMethod::AnalyticalHZ,
            EigMethod::AnalyticalHA22,
            EigMethod::AnalyticalHA23,
            EigMethod::Iterative,
        ] {
            for d in &diagonals {
                for r in &rotations {
                    // A = R ⋅ diag(d) ⋅ Rᵀ
                    let mut a = [[0.0; 3]; 3];
                    for i in 0..3 {
                        for j in 0..3 {
                            for k in 0..3 {
                                a[i][j] += r[i][k] * d[k] * r[j][k];
                            }
                        }
                    }
                    // symmetrize (to remove the tiny rounding asymmetries)
                    for i in 0..3 {
                        for j in (i + 1)..3 {
                            let m = 0.5 * (a[i][j] + a[j][i]);
                            a[i][j] = m;
                            a[j][i] = m;
                        }
                    }
                    let tt = Tensor2::<6>::from_std_matrix(&a).unwrap();
                    let mut spec = Spectral2::new();
                    spec.decompose_mx(&tt, method).unwrap();
                    // check the eigenvalues
                    let mut w = spec.lam;
                    w.sort_by(|x, y| x.partial_cmp(y).unwrap());
                    let mut d_sorted = *d;
                    d_sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
                    for i in 0..3 {
                        approx_eq(w[i], d_sorted[i], 1e-12);
                    }
                    // check the reconstruction A = Σ λᵢ Pᵢ
                    let mut bb = Tensor2::<6>::new();
                    compose(&mut bb, &spec);
                    mat_approx_eq(&tt.as_std_matrix(), &bb.as_std_matrix(), 1e-10);
                }
            }
        }
    }

    #[test]
    fn decompose_coalesce_with_samples_works() {
        for method in [
            EigMethod::AnalyticalHZ,
            EigMethod::AnalyticalHA22,
            EigMethod::AnalyticalHA23,
            EigMethod::Iterative,
        ] {
            for sample in [&SamplesTensor2::COAL_01, &SamplesTensor2::COAL_12] {
                // perform the spectral decomposition
                let aa = Tensor2::<6>::from_std_matrix(&sample.matrix).unwrap();
                let mut spec = Spectral2::new();
                spec.decompose_mx(&aa, method).unwrap();

                // check
                let correct_lambda = sample.eigenvalues.unwrap();
                array_approx_eq(&spec.lam, &correct_lambda, 1e-15);
                check_eigenprojectors(&spec.proj, 1e-15);
                assert_ne!(spec.status, EigStatus::Distinct);
            }
        }
    }

    #[test]
    fn decompose_coalesce_works() {
        for method in [
            EigMethod::AnalyticalHZ,
            EigMethod::AnalyticalHA22,
            EigMethod::AnalyticalHA23,
            EigMethod::Iterative,
        ] {
            for (l1, l2, l3) in [
                // d01
                (1.0, 2.0, 2.0),
                (2.0, 1.0, 2.0),
                (2.0, 2.0, 1.0),
                // d12
                (2.0, 1.0, 1.0),
                (1.0, 2.0, 1.0),
                (1.0, 1.0, 2.0),
                // d01
                (-2.0, -1.0, -1.0),
                (-1.0, -2.0, -1.0),
                (-1.0, -1.0, -2.0),
                // d12
                (-1.0, -2.0, -2.0),
                (-2.0, -1.0, -2.0),
                (-2.0, -2.0, -1.0),
                // d01
                (0.0, 1.0, 1.0),
                (1.0, 0.0, 1.0),
                (1.0, 1.0, 0.0),
                // d12
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                // d01
                (-1.0, 0.0, 0.0),
                (0.0, -1.0, 0.0),
                (0.0, 0.0, -1.0),
                // d12
                (0.0, -1.0, -1.0),
                (-1.0, 0.0, -1.0),
                (-1.0, -1.0, 0.0),
            ] {
                // generate matrix
                let (aa_3x3, expected_lambda, expected_proj) = generate_eigen_problem(l1, l2, l3);

                // allocate matrix A and Spectral2
                let aa = Tensor2::<6>::from_std_matrix(&aa_3x3).unwrap();
                let mut spec = Spectral2::new();

                // perform spectral decomposition
                spec.decompose_mx(&aa, method).unwrap();
                // println!("A =\n{}", aa.as_std_matrix());
                // println!("lambda = {:?}", spec.lam);

                // check
                array_approx_eq(&spec.lam, &expected_lambda, 1e-15);
                check_eigenprojectors(&spec.proj, 1e-15);
                assert_ne!(spec.status, EigStatus::Distinct);
                let is_d01_case = f64::abs(spec.lam[0] - spec.lam[1]) < 1e-8;
                if is_d01_case {
                    // println!("d01");
                    let pp2_mat = spec.proj[2].as_std_matrix();
                    mat_approx_eq(&pp2_mat, &expected_proj[2], 1e-15); // d01
                } else {
                    // println!("d12");
                    let pp0_mat = spec.proj[0].as_std_matrix();
                    mat_approx_eq(&pp0_mat, &expected_proj[0], 1e-15); // d12
                }
            }
        }
    }

    #[test]
    fn hz_cases_work() {
        let mut spec = Spectral2::new();
        for method in [
            EigMethod::AnalyticalHZ,
            EigMethod::AnalyticalHA22,
            EigMethod::AnalyticalHA23,
            EigMethod::Iterative,
        ] {
            for name in hz_cases() {
                let (mut tol_proj, mut tol_compose) = (1e-13, 1e-13);
                if name == "single_lim_disc_t" {
                    tol_proj = 1e-7;
                    tol_compose = 1e-8;
                }
                if name == "single_lim_disc_n" {
                    tol_proj = 1e-7;
                    tol_compose = 1e-8;
                }
                if name == "single_lim_J3J2" {
                    tol_proj = 1e-7;
                    tol_compose = 1e-7;
                }
                if name == "single_J3_lim_J2" {
                    tol_proj = 1e-9;
                    tol_compose = 1e-9;
                }
                if name == "triple_J3" {
                    tol_compose = 1e-12;
                }
                for &delta in &hz_deltas() {
                    let aa = hz_tensor(name, delta);
                    spec.decompose_mx(&aa, method).unwrap();

                    // check the eigenvalues against the prescribed values (relative to the scale)
                    let mut correct = hz_diagonal(name, delta);
                    correct.sort_by(|x, y| y.partial_cmp(x).unwrap());
                    let tol_lambda = 1e-12 * correct[0].abs().max(correct[2].abs());
                    for i in 0..3 {
                        assert!(
                            f64::abs(spec.lam[i] - correct[i]) < tol_lambda,
                            "method = {:?}, case = {}, delta = {}: lam = {:?}, correct = {:?}",
                            method,
                            name,
                            delta,
                            spec.lam,
                            correct
                        );
                    }

                    // check the eigenprojectors and the reconstruction
                    check_eigen_problem(&aa, &spec, tol_proj, tol_compose);
                }
            }
        }
    }

    #[test]
    fn decompose_with_scales_and_coalescence_works() {
        // Test the solvers across scales and coalescence levels.
        // Only the eigenvalues are checked here, with a tolerance relative to the tensor
        // scale, because the eigenprojectors (computed by the Sylvester formula) are
        // ill-conditioned for coalescing eigenvalues.
        let alpha = [1.0, 100.0, 1e6];
        let kappa = [0.0, 1e-10, 1e-8, 1e-6, 1e-3, 0.5];
        for method in [
            EigMethod::AnalyticalHZ,
            EigMethod::AnalyticalHA22,
            EigMethod::AnalyticalHA23,
            EigMethod::Iterative,
        ] {
            for r in 0..alpha.len() {
                for s in 0..kappa.len() {
                    for t in 0..kappa.len() {
                        // generate eigen-problem
                        let l1 = alpha[r];
                        let l2 = alpha[r] + kappa[s];
                        let l3 = alpha[r] + kappa[t];
                        let (aa_3x3, expected_lambda, _) = generate_eigen_problem(l1, l2, l3);

                        // perform spectral decomposition
                        let aa = Tensor2::<6>::from_std_matrix(&aa_3x3).unwrap();
                        let mut spec = Spectral2::new();
                        spec.decompose_mx(&aa, method).unwrap();

                        // check the eigenvalues (tolerance relative to the tensor scale)
                        array_approx_eq(&spec.lam, &expected_lambda, 1e-13 * alpha[r]);

                        // check the eigenprojectors and the reconstruction
                        check_eigen_problem(&aa, &spec, 1e-9, 1e-8);
                    }
                }
            }
        }
    }

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
        // analytical derivative
        let aa = Tensor2::<6>::from_std_matrix(&sample.matrix).unwrap();
        let mut spec = Spectral2::new();
        let status = spec.deriv_eigenproj(&aa, EigMethod::AnalyticalHZ).unwrap();
        println!("Status = {:?}", status);
        if status != EigDerivStatus::Success {
            panic!("failed to compute analytical derivative");
        }
        // println!("dP0/dA =\n{}", spec.dpp[0].as_std_matrix());
        // println!("dP1/dA =\n{}", spec.dpp[1].as_std_matrix());
        // println!("dP2/dA =\n{}", spec.dpp[2].as_std_matrix());

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
            // println!("num_deriv = \n{}", num_deriv.as_std_matrix());
            mat_approx_eq(&spec.dpp[i].as_std_matrix(), &num_deriv.as_std_matrix(), tol);
        }
    }

    #[test]
    fn deriv_eigenproj_works() {
        check_ddp(&SamplesTensor2::TENSOR_U, 1e-9);
        check_ddp(&SamplesTensor2::TENSOR_S, 1e-10);
    }

    #[test]
    fn deriv_eigenproj_captures_problems() {
        // near zero eigenvalues
        let aa = Tensor2::<6>::from_std_matrix(&SamplesTensor2::TENSOR_X.matrix).unwrap();
        let mut spec = Spectral2::new();
        let status = spec.deriv_eigenproj(&aa, EigMethod::AnalyticalHZ).unwrap();
        assert_eq!(status, EigDerivStatus::FailDueToZeroEigenvalue);

        // coalescent 01 eigenvalues
        let aa = Tensor2::<6>::from_std_matrix(&SamplesTensor2::COAL_01.matrix).unwrap();
        let mut spec = Spectral2::new();
        let status = spec.deriv_eigenproj(&aa, EigMethod::AnalyticalHZ).unwrap();
        assert_eq!(spec.status, EigStatus::Coalesce01);
        assert_eq!(status, EigDerivStatus::FailDueToCoalescent);

        // coalescent 12 eigenvalues
        let aa = Tensor2::<6>::from_std_matrix(&SamplesTensor2::COAL_12.matrix).unwrap();
        let mut spec = Spectral2::new();
        let status = spec.deriv_eigenproj(&aa, EigMethod::AnalyticalHZ).unwrap();
        assert_eq!(spec.status, EigStatus::Coalesce12);
        assert_eq!(status, EigDerivStatus::FailDueToCoalescent);
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
        let status = spec.deriv_eigenproj(&aa, EigMethod::AnalyticalHZ).unwrap();
        assert_eq!(status, EigDerivStatus::Success);
    }
}
