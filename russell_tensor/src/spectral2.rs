use super::{P_SYM, SET, SQRT_2, SQRT_3, SQRT_6};
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
    HaberaZilian,

    /// Analytical eigenvalues using Harari-Albocher method (2022)
    ///
    /// * Uses Harari-Albocher (2022) to compute the eigenvalues
    /// * Then, uses either Sylvester formula (Itskov 2019) or Jacobi iterations to compute the eigenprojectors
    HarariAlbocher22,

    /// Analytical eigenvalues using Harari-Albocher method (2023)
    ///
    /// * Uses Harari-Albocher (2023) to compute the eigenvalues
    /// * Then, uses either Sylvester formula (Itskov 2019) or Jacobi iterations to compute the eigenprojectors
    HarariAlbocher23,

    /// Jacobi iterations for eigenvalues and eigenprojectors (via eigenvectors)
    Jacobi,
}

/// Specifies the current status of the eigenvalues
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EigStatus {
    /// All eigenvalues are distinct
    Distinct,

    /// Coalescent eigenvalues λ0 ≈ λ1 > λ2
    Coalesce01,

    /// Coalescent eigenvalues λ0 > λ1 ≈ λ2
    Coalesce12,

    /// All eigenvalues are equal λ0 ≈ λ1 ≈ λ2
    Spherical,
}

/// Holds the spectral representation of a symmetric second-order tensor
///
/// Given the tensor `A`, the spectral representation with eigenvalues `λ[k]` and eigenprojectors `P[k]`
/// is given by the following formula:
///
/// ```text
///      3
/// A =  Σ  λ[k] * P[k]
///     k=1
/// ```
pub struct Spectral2 {
    /// Holds the eigenvalues (sorted in descending order)
    pub lam: [f64; 3],

    /// Holds the eigenprojectors associated with the (sorted) eigenvalues
    ///
    /// Set of 3 symmetric Tensor2
    pub proj: [Tensor2<6>; 3],

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

    /// Auxiliary tensor: ssd(inverse(T))
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
    /// Used in the Analytical method
    ss: [f64; 6],

    /// Auxiliary tensor: T = S^2 - (2J2/3) I
    ///
    /// Used in the Analytical method
    tt: [f64; 6],
}

impl Spectral2 {
    /// Returns a new instance
    ///
    /// **Note:** Must call [Spectral2::compose] to calculate `lambda` and `projectors`.
    pub fn new() -> Self {
        Spectral2 {
            lam: [0.0; 3],
            proj: [Tensor2::<6>::new(), Tensor2::<6>::new(), Tensor2::<6>::new()],
            dpp: Vec::new(),
            aa_inv: Tensor2::<6>::new(),
            aa_3x3: [[0.0; 3]; 3],
            vv_3x3: [[0.0; 3]; 3],
            yy: None,
            p_dy_p: Vec::new(),
            // auxiliary tensors
            ss: [0.0; 6],
            tt: [0.0; 6],
        }
    }

    /// Performs the spectral decomposition of a symmetric second-order tensor (using the default method)
    ///
    /// The output is saved in this struct with the eigenvalues/projectors being sorted in descending order.
    #[inline]
    pub fn decompose(&mut self, aa: &Tensor2<6>) -> Result<EigStatus, StrError> {
        self.decompose_mx(aa, EigMethod::HaberaZilian)
    }

    /// Performs the spectral decomposition of a symmetric second-order tensor (specifying the method)
    ///
    /// The output is saved in this struct with the eigenvalues/projectors being sorted in descending order.
    pub fn decompose_mx(&mut self, aa: &Tensor2<6>, method: EigMethod) -> Result<EigStatus, StrError> {
        // clear previous eigenprojectors data
        for m in 0..6 {
            self.proj[0].vec[m] = 0.0;
            self.proj[1].vec[m] = 0.0;
            self.proj[2].vec[m] = 0.0;
        }

        // detect a (numerically) spherical tensor, i.e., J2 at the rounding level
        let ii1 = aa.invariant_ii1();
        let iso = ii1 / 3.0;
        let scale = aa.norm();
        let jj2 = aa.invariant_jj2();
        if jj2 <= 1e3 * f64::EPSILON * f64::EPSILON * scale * scale {
            self.lam[0] = iso;
            self.lam[1] = iso;
            self.lam[2] = iso;
            self.proj[0].vec[0] = 1.0;
            self.proj[1].vec[1] = 1.0;
            self.proj[2].vec[2] = 1.0;
            return Ok(EigStatus::Spherical);
        }

        // Jacobi iterative method: calculate the eigenvalues and eigenprojectors
        if method == EigMethod::Jacobi {
            return self.decompose_jacobi(aa);
        }

        // Analytical methods: calculate the eigenvalues for non-spherical cases
        let (mut l0, mut l1, mut l2) = match method {
            //
            // Habera M. and Zilian A. (2025)
            //
            EigMethod::HaberaZilian => {
                let mut mat = [[0.0; 3]; 3];
                aa.to_std_matrix_slice(&mut mat);
                let w = crate::habera_zilian::eigvalss(&mat);
                (w[0], w[1], w[2])
            }
            //
            // Harari I. and Albocher U. (2022)
            //
            EigMethod::HarariAlbocher22 => {
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
                    (iso + sqrt_jj2, iso, iso - sqrt_jj2)
                } else {
                    // deviatoric matrix doesn't have zero eigenvalue
                    let dsj = if sj < 0.0 { 1.0 / d_box } else { d_box };
                    let alpha = 2.0 * f64::atan(dsj) / 3.0;
                    let cd = sj * fac2 * f64::cos(alpha);
                    let sd = sqrt_jj2 * f64::sin(alpha);
                    (iso + 2.0 * cd, iso - cd + sd, iso - cd - sd)
                }
            }
            //
            // Harari I. and Albocher U. (2023)
            //
            EigMethod::HarariAlbocher23 => {
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
                (iso + lambda1, iso + lambda2, iso + lambda3)
            }
            EigMethod::Jacobi => unreachable!("handled above"),
        };

        // sort the eigenvalues in descending order
        sort3(&mut l2, &mut l1, &mut l0); // will sort: l2 < l1 < l0
        self.lam[0] = l0;
        self.lam[1] = l1;
        self.lam[2] = l2;

        // compute the eigenprojectors
        self.compute_projectors(aa)
    }

    /// (internal) Performs the spectral decomposition using the Jacobi method
    fn decompose_jacobi(&mut self, aa: &Tensor2<6>) -> Result<EigStatus, StrError> {
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
        Ok(self.classify())
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

    /// (internal) Compute the eigenvalues and eigenprojectors for the given tensor
    ///
    /// * Well-separated eigenvalues: the Sylvester formula is used to compute the eigenprojectors
    /// * Nearly coalescent eigenvalues: the Sylvester formula is ill-conditioned, so the Jacobi
    ///   method is used to compute both the eigenvalues and the eigenprojectors so that they are
    ///   consistent.
    fn compute_projectors(&mut self, aa: &Tensor2<6>) -> Result<EigStatus, StrError> {
        if self.all_distinct() {
            // well-separated eigenvalues: use the Sylvester formula
            self.compute_projectors_sylvester(aa)
        } else {
            // nearly coalescent eigenvalues: fall back to Jacobi
            self.decompose_jacobi(aa)
        }
    }

    /// (internal) Compute the eigenprojectors using the Sylvester formula
    ///
    /// The eigenvalues must be sorted in descending order and well-separated.
    fn compute_projectors_sylvester(&mut self, aa: &Tensor2<6>) -> Result<EigStatus, StrError> {
        // P[r] = f * (A - λ[s] I) . (A - λ[t] I)
        for i in 0..3 {
            let r = INDICES[i];
            let s = INDICES[i + 1];
            let t = INDICES[i + 2];
            let p = -self.lam[s];
            let q = -self.lam[t];
            let f = 1.0 / ((self.lam[r] - self.lam[s]) * (self.lam[r] - self.lam[t]));
            t2_plus_diag_product(self.proj[r].as_mut_data(), f, &aa.as_data(), p, q);
        }
        Ok(EigStatus::Distinct)
    }

    /// Indicates whether all eigenvalues are distinct
    ///
    /// The tolerance used to decide whether two eigenvalues are equal is scaled:
    /// `TOL_COALESCE * max(|λ|, 1)`.
    #[inline]
    pub fn all_distinct(&self) -> bool {
        let tol = self.tol_coalesce();
        let d01 = f64::abs(self.lam[0] - self.lam[1]);
        let d12 = f64::abs(self.lam[1] - self.lam[2]);
        !(d01 < tol || d12 < tol)
    }

    /// Composes a new tensor from the eigenprojectors and diagonal values (lambda)
    ///
    /// ```text
    /// ```
    #[rustfmt::skip]
    pub fn compose(&self, bb: &mut Tensor2<6>, d: &[f64; 3]) {
        bb.vec[0] = d[0] * self.proj[0].vec[0] + d[1] * self.proj[1].vec[0] + d[2] * self.proj[2].vec[0];
        bb.vec[1] = d[0] * self.proj[0].vec[1] + d[1] * self.proj[1].vec[1] + d[2] * self.proj[2].vec[1];
        bb.vec[2] = d[0] * self.proj[0].vec[2] + d[1] * self.proj[1].vec[2] + d[2] * self.proj[2].vec[2];
        bb.vec[3] = d[0] * self.proj[0].vec[3] + d[1] * self.proj[1].vec[3] + d[2] * self.proj[2].vec[3];
        bb.vec[4] = d[0] * self.proj[0].vec[4] + d[1] * self.proj[1].vec[4] + d[2] * self.proj[2].vec[4];
        bb.vec[5] = d[0] * self.proj[0].vec[5] + d[1] * self.proj[1].vec[5] + d[2] * self.proj[2].vec[5];
    }

    /// Calculates the octahedral basis on the principal values space
    ///
    /// Returns `(λ_star_1, λ_star_2, λ_star_3)`
    pub fn octahedral_basis(&self) -> (f64, f64, f64) {
        let (s1, s2, s3) = (self.lam[0], self.lam[1], self.lam[2]);
        let ls1 = (2.0 * s1 - s2 - s3) / SQRT_6;
        let ls2 = (s1 + s2 + s3) / SQRT_3;
        let ls3 = (s3 - s2) / SQRT_2;
        (ls1, ls2, ls3)
    }

    /// Calculates the derivatives of the eigenprojectors w.r.t. the defining tensor
    ///
    /// Note: this function will call [Spectral2::decompose()] to compute the eigenvalues and eigenprojectors.
    ///
    /// The results are available in [Spectral2], including the inverse of T.
    pub fn deriv_eigenproj(&mut self, tt: &Tensor2<6>, method: EigMethod) -> Result<(), StrError> {
        // Perform the spectral decomposition
        self.decompose_mx(tt, method)?;

        // Check for distinct eigenvalues
        if !self.all_distinct() {
            return Err("derivative of eigenprojectors is only available for distinct eigenvalues");
        }

        // Check for null eigenvalues
        for i in 0..3 {
            if f64::abs(self.lam[i]) < TOL_LAMBDA {
                return Err("|lambda| is nearly zero");
            }
            if f64::abs(self.lam[i] * self.lam[i]) < TOL_LAMBDA {
                return Err("|lambda*lambda| is nearly zero");
            }
        }

        // Calculate T⁻¹, the inverse of T, and I3 = det(T)
        let det = tt.inverse(&mut self.aa_inv, TOL_LAMBDA);
        if det.is_none() {
            return Err("|I3| is nearly zero");
        }
        let ii3 = det.unwrap();

        // Calculate the auxiliary tensor Y = ssd(T⁻¹) / 2
        if self.yy.is_none() {
            self.yy = Some(Tensor4::<6>::new());
        }
        let mut yy = self.yy.as_mut().unwrap();
        ssd_fn(&mut yy, SET, 0.5, &self.aa_inv);

        // Allocate and calculate auxiliary tensors P[j] ⊗ P[j]
        if self.p_dy_p.len() != 3 {
            self.p_dy_p = vec![Tensor4::<6>::new(), Tensor4::<6>::new(), Tensor4::<6>::new()];
        }
        t2_dyad_t2(&mut self.p_dy_p[0], SET, 1.0, &self.proj[0], &self.proj[0]);
        t2_dyad_t2(&mut self.p_dy_p[1], SET, 1.0, &self.proj[1], &self.proj[1]);
        t2_dyad_t2(&mut self.p_dy_p[2], SET, 1.0, &self.proj[2], &self.proj[2]);

        // Calculate auxiliary coefficients
        let mut d = [0.0; 3];
        let mut a = [0.0; 3];
        let mut b = [0.0; 3];
        let mut c = [[0.0; 3]; 3];
        let ii1 = tt.invariant_ii1();
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
        Ok(())
    }
}

/// Calculates ||a - alpha * b||^2
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
    use crate::{IDENTITY2, SQRT_2, SQRT_3, SQRT_3_BY_2, SQRT_6, SampleTensor2, SamplesTensor2, Tensor2};
    use russell_lab::{Matrix, approx_eq, array_approx_eq, mat_approx_eq, mat_mat_mul};

    #[cfg(feature = "heap")]
    use russell_lab::vec_approx_eq;

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

    /// Sort eigenvalues and projectors in descending order
    fn sort_projectors(lambda: &mut [f64; 3], projectors: &mut [[[f64; 3]; 3]; 3]) {
        let mut indices = [0, 1, 2];
        indices.sort_by(|&i, &j| lambda[j].partial_cmp(&lambda[i]).unwrap());
        let sorted_lambda = [lambda[indices[0]], lambda[indices[1]], lambda[indices[2]]];
        let sorted_projectors = [
            projectors[indices[0]].clone(),
            projectors[indices[1]].clone(),
            projectors[indices[2]].clone(),
        ];
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

    /// Check the the solution to the eigen-problem on tensor A
    fn check_eigen_problem(aa: &Tensor2<6>, spec: &Spectral2, tol_proj: f64, tol_compose: f64) {
        // check eigenprojectors
        check_eigenprojectors(&spec.proj, tol_proj);

        // check compose
        let mut bb = Tensor2::<6>::new();
        let d = &[spec.lam[0], spec.lam[1], spec.lam[2]];
        spec.compose(&mut bb, &d);
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
        let status = spec.decompose_mx(&aa, method).unwrap();

        // compare eigenvalues
        array_approx_eq(&spec.lam, &correct_lambda, tol_lambda);

        // compare eigenprojectors
        // note: for spherical tensors, the eigenprojectors are not unique, so this check is skipped
        if status != EigStatus::Spherical {
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
        let m = EigMethod::Jacobi;
        check(m, &mut spec, &SamplesTensor2::TENSOR_O, 1e-15, 1e-15, 1e-15);
        check(m, &mut spec, &SamplesTensor2::TENSOR_I, 1e-15, 1e-15, 1e-15);
        check(m, &mut spec, &SamplesTensor2::TENSOR_X, 1e-15, 1e-15, 1e-15);
        check(m, &mut spec, &SamplesTensor2::TENSOR_Y, 1e-13, 1e-15, 1e-15);
        check(m, &mut spec, &SamplesTensor2::TENSOR_Z, 1e-14, 1e-15, 1e-15);
        check(m, &mut spec, &SamplesTensor2::TENSOR_U, 1e-13, 1e-15, 1e-14);
        check(m, &mut spec, &SamplesTensor2::TENSOR_S, 1e-13, 1e-14, 1e-14);
    }

    #[test]
    fn decompose_and_compose_using_analytical_work_with_samples() {
        let mut spec = Spectral2::new();
        let m = EigMethod::HarariAlbocher22;
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
        let m = EigMethod::HarariAlbocher23;
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
        let m = EigMethod::HaberaZilian;
        check(m, &mut spec, &SamplesTensor2::TENSOR_O, 1e-15, 1e-15, 1e-15);
        check(m, &mut spec, &SamplesTensor2::TENSOR_I, 1e-15, 1e-15, 1e-15);
        check(m, &mut spec, &SamplesTensor2::TENSOR_X, 1e-14, 1e-15, 1e-15);
        check(m, &mut spec, &SamplesTensor2::TENSOR_Y, 1e-13, 1e-15, 1e-14);
        check(m, &mut spec, &SamplesTensor2::TENSOR_Z, 1e-14, 1e-14, 1e-14);
        check(m, &mut spec, &SamplesTensor2::TENSOR_U, 1e-13, 1e-14, 1e-14);
        check(m, &mut spec, &SamplesTensor2::TENSOR_S, 1e-13, 1e-14, 1e-14);
    }

    /// Computes the Kelvin-Mandel components of the projector v ⊗ v
    fn projector(v: &[f64; 3]) -> [f64; 6] {
        [
            v[0] * v[0],
            v[1] * v[1],
            v[2] * v[2],
            SQRT_2 * v[0] * v[1],
            SQRT_2 * v[1] * v[2],
            SQRT_2 * v[0] * v[2],
        ]
    }

    /// Tests ported from TFEL's `tests/Math/stensor/stensor_eigenvectors3.cxx`
    #[test]
    fn decompose_tfel_tests_work() {
        const R1_2: f64 = SQRT_2 / 2.0; // 1/√2
        let methods = [
            EigMethod::HaberaZilian,
            EigMethod::HarariAlbocher22,
            EigMethod::HarariAlbocher23,
            EigMethod::Jacobi,
        ];

        // ---------------------------------------------------------------------
        // Test 1: general case
        // TFEL tensor (order: T00, T11, T22, T01√2, T02√2, T12√2):
        //     {1.232, 2.5198, 0.234, 1.5634, 3.3425, 0.9765}
        // ---------------------------------------------------------------------
        #[rustfmt::skip]
        let matrix = [
            [1.232,         1.5634 * R1_2, 3.3425 * R1_2],
            [1.5634 * R1_2, 2.5198,        0.9765 * R1_2],
            [3.3425 * R1_2, 0.9765 * R1_2, 0.234        ],
        ];
        let tt = Tensor2::<6>::from_std_matrix(&matrix).unwrap();
        // TFEL eigenvectors (columns), ordered by eigenvalue 4.167..., 1.507..., -1.689...
        // (the ordering was verified against NumPy)
        let tfel_e = [
            [-0.6208263966073649, -0.6185290894233862, -0.4816599950303030],
            [0.4557421346177839, -0.7846718738721010, 0.4202251266738716],
            [-0.6378665158240716, 0.0413740968617118, 0.7690347795121735],
        ];
        let correct_lambda = [4.16709379934921, 1.50793773158270, -1.68923153093191];
        let correct_proj = [projector(&tfel_e[0]), projector(&tfel_e[1]), projector(&tfel_e[2])];
        for method in methods {
            let mut spec = Spectral2::new();
            let status = spec.decompose_mx(&tt, method).unwrap();
            assert_eq!(status, EigStatus::Distinct);
            array_approx_eq(&spec.lam, &correct_lambda, 1e-12);
            for r in 0..3 {
                array_approx_eq(spec.proj[r].as_data(), &correct_proj[r], 1e-12);
            }
        }

        // ---------------------------------------------------------------------
        // Test 2: identity tensor (spherical)
        // ---------------------------------------------------------------------
        let tt = Tensor2::<6>::identity();
        for method in methods {
            let mut spec = Spectral2::new();
            let status = spec.decompose_mx(&tt, method).unwrap();
            assert_eq!(status, EigStatus::Spherical);
            array_approx_eq(&spec.lam, &[1.0, 1.0, 1.0], 1e-15);
        }

        // ---------------------------------------------------------------------
        // Test 3: diagonal tensor with a repeated eigenvalue (coalescent)
        // ---------------------------------------------------------------------
        #[rustfmt::skip]
        let matrix = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ];
        let tt = Tensor2::<6>::from_std_matrix(&matrix).unwrap();
        let e3_proj = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]; // e3 ⊗ e3
        for method in methods {
            let mut spec = Spectral2::new();
            let status = spec.decompose_mx(&tt, method).unwrap();
            assert_eq!(status, EigStatus::Coalesce01);
            array_approx_eq(&spec.lam, &[1.0, 1.0, 0.0], 1e-15);
            // the distinct eigenvalue (zero) has the projector e3 ⊗ e3
            array_approx_eq(spec.proj[2].as_data(), &e3_proj, 1e-15);
        }
    }

    /// Tests inspired by jaxmat's `tests/tensors/test_linear_algebra.py`
    ///
    /// Checks the eigen-decomposition reconstruction `A = Σ λᵢ Pᵢ` for random-like,
    /// two-nearly-equal, and triple-equal eigenvalues.
    #[test]
    fn decompose_jaxmat_reconstruction_works() {
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
        // loop
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
                spec.decompose_mx(&tt, EigMethod::HaberaZilian).unwrap();
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
                spec.compose(&mut bb, &spec.lam);
                mat_approx_eq(&tt.as_std_matrix(), &bb.as_std_matrix(), 1e-12);
            }
        }
    }

    #[test]
    fn decompose_coalesce_works() {
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

            // perform spectral decomposition
            let mut aa = Tensor2::<6>::from_std_matrix(&aa_3x3).unwrap();
            let mut spec = Spectral2::new();
            spec.decompose_mx(&mut aa, EigMethod::HarariAlbocher22).unwrap();
            // println!("A =\n{}", aa.as_std_matrix());
            // println!("lambda = {:?}", spec.lam);

            // check
            array_approx_eq(&spec.lam, &expected_lambda, 1e-15);
            check_eigenprojectors(&spec.proj, 1e-15);
            assert_eq!(spec.all_distinct(), false);
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

    #[test]
    fn decompose_analytical1_with_scales_and_coalescence_works() {
        // Test the Harari-Albocher (2023) TgHSC eigenvalue solver across scales and coalescence
        // levels. Only the eigenvalues are checked here, with a tolerance relative to the tensor
        // scale, because the eigenprojectors (computed by the Sylvester formula) are
        // ill-conditioned for coalescing eigenvalues.
        let alpha = [1.0, 100.0, 1e6];
        let kappa = [0.0, 1e-10, 1e-8, 1e-6, 1e-3, 0.5];
        for r in 0..alpha.len() {
            for s in 0..kappa.len() {
                for t in 0..kappa.len() {
                    // generate eigen-problem
                    let l1 = alpha[r];
                    let l2 = alpha[r] + kappa[s];
                    let l3 = alpha[r] + kappa[t];
                    let (aa_3x3, expected_lambda, _) = generate_eigen_problem(l1, l2, l3);

                    // perform spectral decomposition
                    let mut aa = Tensor2::<6>::from_std_matrix(&aa_3x3).unwrap();
                    let mut spec = Spectral2::new();
                    spec.decompose_mx(&mut aa, EigMethod::HarariAlbocher23).unwrap();

                    // check the eigenvalues (tolerance relative to the tensor scale)
                    array_approx_eq(&spec.lam, &expected_lambda, 1e-13 * alpha[r]);

                    // check the eigenprojectors
                    // check_eigenprojectors(&spec.proj, 1e-15);
                    // if spec.all_distinct() {
                    //     mat_approx_eq(&pp0_mat, &expected_proj[0], 1e-15);
                    //     mat_approx_eq(&pp1_mat, &expected_proj[1], 1e-15);
                    //     mat_approx_eq(&pp2_mat, &expected_proj[2], 1e-15);
                    // }
                }
            }
        }
    }

    #[test]
    fn octahedral_basis_using_jacobi_method_works() {
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
            spec.decompose_mx(&tt, EigMethod::Jacobi).unwrap();
            let (ls1, ls2, ls3) = spec.octahedral_basis();
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
}
