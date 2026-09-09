#![allow(unused)]

use super::{IDENTITY2, P_SYM, SET, SQRT_2, SQRT_3, SQRT_6, TOL_J2};
use crate::{StrError, Tensor1, Tensor2, Tensor4, ssd_fn, t1_dyad_t1, t2_dyad_t2};
use russell_lab::{Matrix, Vector, approx_eq, mat_eigen_sym_jacobi, math::PI};

/// Tolerance to assume zero eigenvalue
///
/// It must be ~ sqrt(EPSILON) because the code performs division by lambda^2
const TOL_LAMBDA: f64 = 1e-8;

const TOL_ZERO_DEV_LAMBDA: f64 = 1e-15;
const TOL_COALESCE: f64 = 1e-8;

const TOL_DELTA: f64 = 1e-15;
const TOL_DI: f64 = 1e-15;
const TOL_J3: f64 = 1e-15;
const TOL_INVERSE: f64 = 1e-14;

/// Holds indices for permutation by looping in 0..3
const INDICES: [usize; 5] = [0, 1, 2, 0, 1];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EigMethod {
    /// Jacobi iterations
    Jacobi,

    /// Analytical
    Analytical,
}

/// Holds the spectral representation of a symmetric second-order tensor
pub struct Spectral2 {
    /// Holds the eigenvalues
    ///
    /// dim = 3
    pub lambda: Vector,

    /// Holds the eigenprojectors
    ///
    /// Set of 3 symmetric Tensor2
    pub proj: Vec<Tensor2<6>>,

    /// Holds the derivatives of the eigenprojectors w.r.t the defining tensor
    ///
    /// Set of 3 minor-symmetric Tensor4 (empty by default)
    pub dpp: Vec<Tensor4<6>>,

    /// Auxiliary tensor: inverse of T
    ///
    /// ```text
    /// T⁻¹
    /// ```
    pub inverse: Tensor2<6>,

    /// Auxiliary tensor: ssd(inverse(T))
    ///
    /// ```text
    ///             _
    /// Y := ½ (T⁻¹ ⊗ T⁻¹ + T⁻¹ ⊗ T⁻¹) = ssd(T⁻¹) / 2
    ///                         ‾
    /// ```
    yy: Option<Tensor4<6>>,

    /// Auxiliary set of tensors
    ///
    /// ```text
    /// P[j] ⊗ P[j]  (no sum on j)
    /// ```
    p_dy_p: Vec<Tensor4<6>>,

    //
    // --- auxiliary tensors
    //
    ss: [f64; 6], // S = A - (I1/3) I
    tt: [f64; 6], // T = S^2 - (2J2/3) I

    //
    // --- auxiliary coefficients
    //
    a: [f64; 3],
    b: [f64; 3],
    c: [[f64; 3]; 3],
    d: [f64; 3],
}

impl Spectral2 {
    /// Returns a new instance
    ///
    /// **Note:** Must call [Spectral2::compose] to calculate `lambda` and `projectors`.
    pub fn new() -> Self {
        Spectral2 {
            lambda: Vector::new(3),
            proj: vec![Tensor2::<6>::new(), Tensor2::<6>::new(), Tensor2::<6>::new()],
            dpp: Vec::new(),
            inverse: Tensor2::<6>::new(),
            yy: None,
            p_dy_p: Vec::new(),
            // auxiliary tensors
            ss: [0.0; 6],
            tt: [0.0; 6],
            // auxiliary coefficients
            a: [0.0; 3],
            b: [0.0; 3],
            c: [[0.0; 3]; 3],
            d: [0.0; 3],
        }
    }

    /// Performs the spectral decomposition of a symmetric second-order tensor
    ///
    /// # Results
    ///
    /// The results are available in [Spectral2::lambda] and [Spectral2::projectors].
    pub fn decompose(&mut self, aa: &Tensor2<6>, method: EigMethod) -> Result<(), StrError> {
        match method {
            EigMethod::Jacobi => {
                // eigenvalues and eigenvectors
                let mut a = aa.as_std_matrix();
                let mut v = Matrix::new(3, 3);
                mat_eigen_sym_jacobi(&mut self.lambda, &mut v, &mut a)?;

                // extract eigenvectors
                let u0 = Tensor1::from(&[v.get(0, 0), v.get(1, 0), v.get(2, 0)]);
                let u1 = Tensor1::from(&[v.get(0, 1), v.get(1, 1), v.get(2, 1)]);
                let u2 = Tensor1::from(&[v.get(0, 2), v.get(1, 2), v.get(2, 2)]);

                // compute eigenprojectors
                t1_dyad_t1(&mut self.proj[0], SET, 1.0, &u0, &u0).unwrap();
                t1_dyad_t1(&mut self.proj[1], SET, 1.0, &u1, &u1).unwrap();
                t1_dyad_t1(&mut self.proj[2], SET, 1.0, &u2, &u2).unwrap();
            }
            EigMethod::Analytical => {
                let ii1 = aa.invariant_ii1();
                let jj2 = aa.invariant_jj2();
                let shift = ii1 / 3.0; // shift
                if jj2 < TOL_J2 {
                    // spherical
                    self.lambda[0] = shift;
                    self.lambda[1] = shift;
                    self.lambda[2] = shift;
                    // Setting so that A = Σ_{k=1,2,3} λ[k] P[k] still works)
                    // P0
                    self.proj[0].vec[0] = 1.0;
                    self.proj[0].vec[1] = 0.0;
                    self.proj[0].vec[2] = 0.0;
                    self.proj[0].vec[3] = 0.0;
                    self.proj[0].vec[4] = 0.0;
                    self.proj[0].vec[5] = 0.0;
                    // P1
                    self.proj[1].vec[0] = 0.0;
                    self.proj[1].vec[1] = 1.0;
                    self.proj[1].vec[2] = 0.0;
                    self.proj[1].vec[3] = 0.0;
                    self.proj[1].vec[4] = 0.0;
                    self.proj[1].vec[5] = 0.0;
                    // P2
                    self.proj[2].vec[0] = 0.0;
                    self.proj[2].vec[1] = 0.0;
                    self.proj[2].vec[2] = 1.0;
                    self.proj[2].vec[3] = 0.0;
                    self.proj[2].vec[4] = 0.0;
                    self.proj[2].vec[5] = 0.0;
                } else {
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
                    let d_box = f64::sqrt(num / den); // this is not d in Eq (70), but the newly defined d in Box 1
                    let sj = f64::signum(1.0 - d_box);
                    if sj * (1.0 - d_box) < TOL_ZERO_DEV_LAMBDA {
                        // deviatoric matrix has a zero eigenvalue
                        self.lambda[0] = shift + sqrt_jj2;
                        self.lambda[1] = shift;
                        self.lambda[2] = shift - sqrt_jj2;
                    } else {
                        // deviatoric matrix doesn't have zero eigenvalue
                        let dsj = if sj < 0.0 { 1.0 / d_box } else { d_box };
                        let alpha = 2.0 * f64::atan(dsj) / 3.0;
                        println!("alpha = {}", alpha * 180.0 / PI);
                        let cd = sj * fac2 * f64::cos(alpha);
                        let sd = sqrt_jj2 * f64::sin(alpha);
                        self.lambda[0] = shift + 2.0 * cd;
                        self.lambda[1] = shift - cd + sd;
                        self.lambda[2] = shift - cd - sd;
                    }
                    let d0 = self.lambda[0] - self.lambda[1];
                    let d1 = self.lambda[1] - self.lambda[2];
                    let d2 = self.lambda[2] - self.lambda[0];
                    println!("d0 = {}", d0);
                    println!("d1 = {}", d1);
                    println!("d2 = {}", d2);
                    if f64::abs(d0) < TOL_COALESCE {
                        println!(" >>. d0");
                    } else if f64::abs(d1) < TOL_COALESCE {
                        println!(" >>. d1");
                    } else if f64::abs(d2) < TOL_COALESCE {
                        println!(" >>. d2");
                    } else {
                        println!(" !!!!!!!!!!!!!!!!!");
                        for i in 0..3 {
                            let r = INDICES[i];
                            let s = INDICES[i + 1];
                            let t = INDICES[i + 2];
                            let p = -self.lambda[s];
                            let q = -self.lambda[t];
                            let f = 1.0 / ((self.lambda[r] - self.lambda[s]) * (self.lambda[r] - self.lambda[t]));
                            // Set P[r] = f * (A - λ[s] I) . (A - λ[t] I)
                            t2_plus_diag_product(self.proj[r].as_mut_data(), f, &aa.as_data(), p, q);
                        }
                    }
                }
            }
        }
        Ok(())
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
        let (s1, s2, s3) = (self.lambda[0], self.lambda[1], self.lambda[2]);
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
        self.decompose(tt, method)?;

        // Check for null eigenvalues
        for i in 0..3 {
            if f64::abs(self.lambda[i]) < TOL_LAMBDA {
                return Err("|lambda| is nearly zero");
            }
            if f64::abs(self.lambda[i] * self.lambda[i]) < TOL_LAMBDA {
                return Err("|lambda*lambda| is nearly zero");
            }
        }

        // Calculate T⁻¹, the inverse of T, and I3 = det(T)
        let det = tt.inverse(&mut self.inverse, TOL_LAMBDA);
        if det.is_none() {
            return Err("|I3| is nearly zero");
        }
        let ii3 = det.unwrap();

        // Calculate the auxiliary tensor Y = ssd(T⁻¹) / 2
        if self.yy.is_none() {
            self.yy = Some(Tensor4::<6>::new());
        }
        let mut yy = self.yy.as_mut().unwrap();
        ssd_fn(&mut yy, SET, 0.5, &self.inverse);

        // Allocate and calculate auxiliary tensors P[j] ⊗ P[j]
        if self.p_dy_p.len() != 3 {
            self.p_dy_p = vec![Tensor4::<6>::new(), Tensor4::<6>::new(), Tensor4::<6>::new()];
        }
        t2_dyad_t2(&mut self.p_dy_p[0], SET, 1.0, &self.proj[0], &self.proj[0]);
        t2_dyad_t2(&mut self.p_dy_p[1], SET, 1.0, &self.proj[1], &self.proj[1]);
        t2_dyad_t2(&mut self.p_dy_p[2], SET, 1.0, &self.proj[2], &self.proj[2]);

        // Calculate auxiliary coefficients
        let ii1 = tt.invariant_ii1();
        for i in 0..3 {
            self.d[i] = 2.0 * self.lambda[i] * self.lambda[i] - ii1 * self.lambda[i] + ii3 / self.lambda[i];
            if f64::abs(self.d[i]) < TOL_LAMBDA {
                return Err("|d[i]| is nearly zero");
            }
            self.a[i] = self.lambda[i] / self.d[i];
            self.b[i] = ii3 / self.d[i];
            for j in 0..3 {
                self.c[i][j] = ii3 / (self.d[i] * self.lambda[j] * self.lambda[j]);
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
                    let p = self.a[i] * P_SYM[m][n] - self.b[i] * yy.get(m, n);
                    let q0 = (self.c[i][0] - self.a[i]) * self.p_dy_p[0].get(m, n);
                    let q1 = (self.c[i][1] - self.a[i]) * self.p_dy_p[1].get(m, n);
                    let q2 = (self.c[i][2] - self.a[i]) * self.p_dy_p[2].get(m, n);
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
    use super::{EigMethod, Spectral2, t2_plus_diag_product};
    use crate::{IDENTITY2, SQRT_2, SQRT_3, SQRT_3_BY_2, SQRT_6, SampleTensor2, SamplesTensor2, Tensor2};
    use russell_lab::{Matrix, Vector, approx_eq, array_approx_eq, mat_approx_eq, mat_mat_mul, vec_approx_eq};

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
                approx_eq(aa[i][j], aa[j][i], 1e-15);
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
    /// Returns `(expected_lambda, expected_projectors)` sorted in decreasing order by lambda
    fn generate_eigen_problem(l1: f64, l2: f64, l3: f64) -> ([f64; 3], [[[f64; 3]; 3]; 3]) {
        // Q rotates axes to octahedral system
        #[rustfmt::skip]
        let qq_3x3 = [
            [2.0 / SQRT_6, -1.0 / SQRT_6, -1.0 / SQRT_6],
            [1.0 / SQRT_3,  1.0 / SQRT_3,  1.0 / SQRT_3],
            [0.0,          -1.0 / SQRT_2,  1.0 / SQRT_2],
        ];
        // eigenvectors
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
        let pp0_3x3 = [
            [n0[0] * n0[0], n0[0] * n0[1], n0[0] * n0[2]],
            [n0[1] * n0[0], n0[1] * n0[1], n0[1] * n0[2]],
            [n0[2] * n0[0], n0[2] * n0[1], n0[2] * n0[2]],
        ];
        let pp1_3x3 = [
            [n1[0] * n1[0], n1[0] * n1[1], n1[0] * n1[2]],
            [n1[1] * n1[0], n1[1] * n1[1], n1[1] * n1[2]],
            [n1[2] * n1[0], n1[2] * n1[1], n1[2] * n1[2]],
        ];
        let pp2_3x3 = [
            [n2[0] * n2[0], n2[0] * n2[1], n2[0] * n2[2]],
            [n2[1] * n2[0], n2[1] * n2[1], n2[1] * n2[2]],
            [n2[2] * n2[0], n2[2] * n2[1], n2[2] * n2[2]],
        ];
        // check
        let mut expected_lambda = [l1, l2, l3];
        let mut expected_projectors = [pp0_3x3, pp1_3x3, pp2_3x3];
        sort_projectors(&mut expected_lambda, &mut expected_projectors);
        let e_projectors = [
            Tensor2::<6>::from_std_matrix(&expected_projectors[0]).unwrap(),
            Tensor2::<6>::from_std_matrix(&expected_projectors[1]).unwrap(),
            Tensor2::<6>::from_std_matrix(&expected_projectors[2]).unwrap(),
        ];
        check_eigenprojectors(&e_projectors, 1e-15);
        // output
        println!("pp0_3x3 = \n{}", Matrix::from(&pp0_3x3));
        println!("pp1_3x3 = \n{}", Matrix::from(&pp1_3x3));
        println!("pp2_3x3 = \n{}", Matrix::from(&pp2_3x3));
        (expected_lambda, expected_projectors)
    }

    /// Check the the solution to the eigen-problem on tensor A
    fn check_eigen_problem(aa: &Tensor2<6>, spec: &Spectral2, tol_proj: f64, tol_compose: f64) {
        // check eigenprojectors
        let pp0 = spec.proj[0].as_std_matrix();
        let pp1 = spec.proj[1].as_std_matrix();
        let pp2 = spec.proj[2].as_std_matrix();
        check_eigenprojectors(&spec.proj, tol_proj);

        // check compose
        let mut bb = Tensor2::<6>::new();
        let d = &[spec.lambda[0], spec.lambda[1], spec.lambda[2]];
        spec.compose(&mut bb, &d);
        vec_approx_eq(&aa.vec, &bb.vec, tol_compose);
    }

    /// Checks the eigen-problem by comparing with known values
    fn check_j(spec: &mut Spectral2, sample: &SampleTensor2, tol_lambda: f64, tol_proj: f64, tol_compose: f64) {
        // extract eigenvalues and projectors
        let correct_lambda = sample.eigenvalues.unwrap();
        let correct_projectors = sample.eigenprojectors.unwrap();

        // perform the spectral decomposition
        let aa = Tensor2::<6>::from_std_matrix(&sample.matrix).unwrap();
        spec.decompose(&aa, EigMethod::Jacobi).unwrap();

        // compare eigenvalues
        vec_approx_eq(&spec.lambda, &correct_lambda, tol_lambda);

        // compare eigenprojectors
        let pp0 = spec.proj[0].as_std_matrix();
        let pp1 = spec.proj[1].as_std_matrix();
        let pp2 = spec.proj[2].as_std_matrix();
        let correct0 = Matrix::from(&correct_projectors[0]);
        let correct1 = Matrix::from(&correct_projectors[1]);
        let correct2 = Matrix::from(&correct_projectors[2]);
        mat_approx_eq(&correct0, &pp0, tol_proj);
        mat_approx_eq(&correct1, &pp1, tol_proj);
        mat_approx_eq(&correct2, &pp2, tol_proj);

        // further checks
        check_eigen_problem(&aa, spec, tol_proj, tol_compose);
    }

    //
    // --- tests -------------------------------
    //

    #[test]
    fn decompose_and_compose_work_using_jacobi_method() {
        let mut spec = Spectral2::new();
        check_j(&mut spec, &SamplesTensor2::TENSOR_O, 1e-15, 1e-15, 1e-15);
        check_j(&mut spec, &SamplesTensor2::TENSOR_I, 1e-15, 1e-15, 1e-15);
        check_j(&mut spec, &SamplesTensor2::TENSOR_X, 1e-15, 1e-15, 1e-15);
        check_j(&mut spec, &SamplesTensor2::TENSOR_Y, 1e-13, 1e-15, 1e-15);
        check_j(&mut spec, &SamplesTensor2::TENSOR_Z, 1e-14, 1e-15, 1e-15);
        check_j(&mut spec, &SamplesTensor2::TENSOR_U, 1e-13, 1e-15, 1e-14);
        check_j(&mut spec, &SamplesTensor2::TENSOR_S, 1e-13, 1e-14, 1e-14);
        check_j(&mut spec, &SamplesTensor2::TENSOR_O, 1e-15, 1e-15, 1e-15);
        check_j(&mut spec, &SamplesTensor2::TENSOR_I, 1e-15, 1e-15, 1e-15);
        check_j(&mut spec, &SamplesTensor2::TENSOR_X, 1e-15, 1e-15, 1e-15);
        check_j(&mut spec, &SamplesTensor2::TENSOR_Y, 1e-13, 1e-15, 1e-15);
        check_j(&mut spec, &SamplesTensor2::TENSOR_Z, 1e-14, 1e-15, 1e-15);
    }

    #[test]
    fn octahedral_basis_works_using_jacobi_method() {
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
            spec.decompose(&tt, EigMethod::Jacobi).unwrap();
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

    #[test]
    fn deriv_eigenproj_works_1() {
        // setup 3x3 matrix
        let l1 = 1.0;
        let l2 = 2.0;
        let l3 = 3.0;
        let ll = [[l1, 0.0, 0.0], [0.0, l2, 0.0], [0.0, 0.0, l3]];
        let mut aa_3x3 = [[0.0; 3]; 3];
        // transform(&mut aa_3x3, &ll, &qq_3x3);

        // setup symmetric tensor
        let mut aa = Tensor2::<6>::from_std_matrix(&aa_3x3).unwrap();
        println!("{}", aa.as_std_matrix());

        // perform spectral decomposition
        let mut spec = Spectral2::new();
        spec.decompose(&mut aa, EigMethod::Analytical).unwrap();

        let pp0_mat = spec.proj[0].as_std_matrix();
        let pp1_mat = spec.proj[1].as_std_matrix();
        let pp2_mat = spec.proj[2].as_std_matrix();

        println!("{}", spec.lambda);
        println!("{}", pp0_mat);
        // println!("{}", pp1_mat);
        // println!("{}", pp2_mat);

        check_eigenprojectors(&spec.proj, 1e-15);

        // let tt = Tensor2::<6>::from_std_matrix(&SamplesTensor2::TENSOR_X.matrix).unwrap();
        // let tt = Tensor2::<6>::from_std_matrix(&SamplesTensor2::TENSOR_I.matrix).unwrap();
        // spec.deriv_eigenproj(&tt).unwrap();
        // let m = EigMethod::Analytical;
        // check2(m, &mut spec, &SamplesTensor2::TENSOR_O, 1e-15, 1e-15, 1e-15);
        // check2(m, &mut spec, &SamplesTensor2::TENSOR_I, 1e-15, 1e-15, 1e-15);
        // check2(m, &mut spec, &SamplesTensor2::TENSOR_X, 1e-15, 1e-15, 1e-15);
        // check2(m, &mut spec, &SamplesTensor2::TENSOR_U, 1e-13, 1e-15, 1e-14);
    }
}

/*
EigMethod::HA1 => {
    let ii1 = aa.invariant_ii1();
    let jj2 = aa.invariant_jj2();
    let h = ii1 / 3.0; // shift
    if jj2 < TOL_J2 {
        // spherical
        self.lambda[0] = h;
        self.lambda[1] = h;
        self.lambda[2] = h;
        // Setting so that A = Σ_{k=1,2,3} λ[k] P[k] still works)
        // P0
        self.projectors[0].vec[0] = 1.0;
        self.projectors[0].vec[1] = 0.0;
        self.projectors[0].vec[2] = 0.0;
        self.projectors[0].vec[3] = 0.0;
        self.projectors[0].vec[4] = 0.0;
        self.projectors[0].vec[5] = 0.0;
        // P1
        self.projectors[1].vec[0] = 0.0;
        self.projectors[1].vec[1] = 1.0;
        self.projectors[1].vec[2] = 0.0;
        self.projectors[1].vec[3] = 0.0;
        self.projectors[1].vec[4] = 0.0;
        self.projectors[1].vec[5] = 0.0;
        // P2
        self.projectors[2].vec[0] = 0.0;
        self.projectors[2].vec[1] = 0.0;
        self.projectors[2].vec[2] = 1.0;
        self.projectors[2].vec[3] = 0.0;
        self.projectors[2].vec[4] = 0.0;
        self.projectors[2].vec[5] = 0.0;
    } else {
        let sqrt_jj2 = f64::sqrt(jj2);
        let fac1 = 2.0 * jj2 / 3.0;
        let fac2 = sqrt_jj2 / SQRT_3;
        let a = &aa.vec;
        self.ss[0] = a[0] - ii1 / 3.0;
        self.ss[1] = a[1] - ii1 / 3.0;
        self.ss[2] = a[2] - ii1 / 3.0;
        self.ss[3] = a[3];
        self.ss[4] = a[4];
        self.ss[5] = a[5];
        let s = &self.ss;
        self.tt[0] = s[0] * s[0] + s[3] * s[3] / 2.0 + s[5] * s[5] / 2.0 - fac1;
        self.tt[1] = s[1] * s[1] + s[3] * s[3] / 2.0 + s[4] * s[4] / 2.0 - fac1;
        self.tt[2] = s[2] * s[2] + s[4] * s[4] / 2.0 + s[5] * s[5] / 2.0 - fac1;
        self.tt[3] = (s[0] + s[1]) * s[3] + s[4] * s[5] / SQRT_2;
        self.tt[4] = (s[1] + s[2]) * s[4] + s[3] * s[5] / SQRT_2;
        self.tt[5] = (s[0] + s[2]) * s[5] + s[3] * s[4] / SQRT_2;
        let num = sq_norm_diff(&self.tt, -fac2, &self.ss);
        let den = sq_norm_diff(&self.tt, fac2, &self.ss);
        let d_box = f64::sqrt(num / den); // this is not d in Eq (70), but the newly defined d in Box 1
        let sj = f64::signum(1.0 - d_box);
        if sj * (1.0 - d_box) < COALESCENCE_TOL {
            // singular
            self.lambda[0] = h + sqrt_jj2;
            self.lambda[1] = h;
            self.lambda[2] = h - sqrt_jj2;
        } else {
            // all distinct
            let dsj = if sj < 0.0 { 1.0 / d_box } else { d_box };
            let alpha = 2.0 * f64::atan(dsj) / 3.0;
            let cd = sj * fac2 * f64::cos(alpha);
            let sd = sqrt_jj2 * f64::sin(alpha);
            self.lambda[0] = h + 2.0 * cd;
            self.lambda[1] = h - cd + sd;
            self.lambda[2] = h - cd - sd;
            for k in 0..3 {}
        }
    }
}
*/

/*
/// Calculate auxiliary coefficients
pub fn auxiliary(&mut self, ii1: f64, ii3: f64) -> Result<(), StrError> {
    for i in 0..3 {
        self.d[i] = 2.0 * self.lambda[i] * self.lambda[i] - ii1 * self.lambda[i] + ii3 / self.lambda[i];
        if f64::abs(self.d[i]) < TOL_DI {
            return Err("|d[i]| is nearly zero");
        }
        self.a[i] = self.lambda[i] / self.d[i];
        self.b[i] = ii3 / self.d[i];
        for j in 0..3 {
            self.c[i][j] = ii3 / (self.d[i] * self.lambda[j] * self.lambda[j]);
        }
    }
    Ok(())
}
*/

/*
let result = aa.inverse(&mut self.inverse, TOL_INVERSE);
if result.is_none() {
    return Err("cannot invert tensor A");
}
let ii3 = result.unwrap();
for i in 0..3 {
    if f64::abs(self.lambda[i]) < TOL_LAMBDA {
        return Err("|lambda| is nearly zero");
    }
    let ll = self.lambda[i] * self.lambda[i];
    let di = 2.0 * ll - ii1 * self.lambda[i] + ii3 / self.lambda[i];
    if f64::abs(self.d[i]) < TOL_LAMBDA {
        return Err("|d[i]| is nearly zero");
    }
    self.a[i] = self.lambda[i] / self.d[i];
    for m in 0..6 {
        self.projectors[i].vec[m] = self.a[i]
            * (aa.vec[m] - (ii1 - self.lambda[i]) * IDENTITY2[m]
                + ii3 * self.inverse.vec[m] / self.lambda[i]);
    }
}
*/
