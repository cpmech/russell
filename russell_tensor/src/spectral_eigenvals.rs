use crate::StrError;
use crate::Tensor2;
use crate::{SQRT_2, SQRT_3};
use russell_lab::{small_mat_eigen_sym_jacobi, sort3};

/// Tolerance to assume zero eigenvalue of the deviatoric matrix
const TOL_ZERO_DEV_LAMBDA: f64 = 1e-15;

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
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EigenMethod {
    /// Analytical eigenvalues using Habera-Zilian method
    AnalyticalHZ,

    /// Analytical eigenvalues using Harari-Albocher method (2022)
    AnalyticalHA22,

    /// Analytical eigenvalues using Harari-Albocher method (2023)
    AnalyticalHA23,

    /// Jacobi iterations for eigenvalues (ignore eigenvectors)
    Iterative,
}

/// Holds the eigenvalues of a symmetric second-order tensor
pub struct EigenValuesT2 {
    /// Auxiliary deviatoric tensor: S = A - (I1/3) I
    ///
    /// Used in the Harari-Albocher (2022) method
    ss: [f64; 6],

    /// Auxiliary tensor: T = S^2 - (2J2/3) I
    ///
    /// Used in the Harari-Albocher (2022) method
    tt: [f64; 6],

    /// Input tensor as a 3x3 matrix (for Jacobi method)
    pub(crate) aa: [[f64; 3]; 3],

    /// Matrix whose columns are the eigenvectors (for Jacobi method)
    pub(crate) vv: [[f64; 3]; 3],
}

impl EigenValuesT2 {
    /// Allocates a new instance
    pub fn new() -> Self {
        EigenValuesT2 {
            ss: [0.0; 6],
            tt: [0.0; 6],
            aa: [[0.0; 3]; 3],
            vv: [[0.0; 3]; 3],
        }
    }

    /// Calculates the eigenvalues of a symmetric second order tensor
    ///
    /// The results are stored in `ll` in descending order.
    ///
    /// Returns `true` if spherical, `false` otherwise.
    ///
    /// In the spherical case, `ll := [λ, λ, λ]` where `λ = λ1 = λ2 = λ3 = trace(A)`.
    ///
    /// Uses the default method: [EigenMethod::AnalyticalHZ]
    #[inline]
    pub fn calculate(&mut self, ll: &mut [f64; 3], aa: &Tensor2<6>) -> Result<bool, StrError> {
        self.calculate_mx(ll, aa, EigenMethod::AnalyticalHZ)
    }

    /// Calculates the eigenvalues of a symmetric second order tensor (with method selection)
    ///
    /// The results are stored in `ll` in descending order.
    ///
    /// Returns `true` if spherical, `false` otherwise.
    ///
    /// In the spherical case, `ll := [λ, λ, λ]` where `λ = λ1 = λ2 = λ3 = trace(A)`.
    pub fn calculate_mx(&mut self, ll: &mut [f64; 3], aa: &Tensor2<6>, method: EigenMethod) -> Result<bool, StrError> {
        // detect a (numerically) spherical tensor, i.e., J2 at the rounding level
        let ii1 = aa.invariant_ii1();
        let jj2 = aa.invariant_jj2();
        let scale = aa.norm();
        let spherical = jj2 <= 1e3 * f64::EPSILON * f64::EPSILON * scale * scale;
        let iso = ii1 / 3.0;
        if spherical {
            ll[0] = iso;
            ll[1] = iso;
            ll[2] = iso;
            return Ok(true); // true => spherical
        }

        // calculate the eigenvalues for non-spherical cases
        match method {
            //
            // Habera M. and Zilian A. (2025)
            //
            EigenMethod::AnalyticalHZ => {
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
                ll[0] = amplitude.mul_add(f64::cos(angle0), ii1) / 3.0;
                ll[1] = amplitude.mul_add(f64::cos(angle1), ii1) / 3.0;
                ll[2] = amplitude.mul_add(f64::cos(angle2), ii1) / 3.0;
            }
            //
            // Harari I. and Albocher U. (2022)
            //
            EigenMethod::AnalyticalHA22 => {
                let sqrt_jj2 = f64::sqrt(jj2);
                let fac1 = 2.0 * jj2 / 3.0;
                let fac2 = sqrt_jj2 / SQRT_3;
                let ss = &mut self.ss;
                let tt = &mut self.tt;
                ss[0] = aa.vec[0] - ii1 / 3.0;
                ss[1] = aa.vec[1] - ii1 / 3.0;
                ss[2] = aa.vec[2] - ii1 / 3.0;
                ss[3] = aa.vec[3];
                ss[4] = aa.vec[4];
                ss[5] = aa.vec[5];
                tt[0] = ss[0] * ss[0] + ss[3] * ss[3] / 2.0 + ss[5] * ss[5] / 2.0 - fac1;
                tt[1] = ss[1] * ss[1] + ss[3] * ss[3] / 2.0 + ss[4] * ss[4] / 2.0 - fac1;
                tt[2] = ss[2] * ss[2] + ss[4] * ss[4] / 2.0 + ss[5] * ss[5] / 2.0 - fac1;
                tt[3] = (ss[0] + ss[1]) * ss[3] + ss[4] * ss[5] / SQRT_2;
                tt[4] = (ss[1] + ss[2]) * ss[4] + ss[3] * ss[5] / SQRT_2;
                tt[5] = (ss[0] + ss[2]) * ss[5] + ss[3] * ss[4] / SQRT_2;
                let num = sq_norm_diff(tt, -fac2, ss);
                let den = sq_norm_diff(tt, fac2, ss);
                // this is not d in Eq (70) of Ref #1; it is the newly defined d in Box 1 of Ref #1
                let d_box = f64::sqrt(num / den);
                let sj = f64::signum(1.0 - d_box);
                if sj * (1.0 - d_box) < TOL_ZERO_DEV_LAMBDA {
                    // deviatoric matrix has a zero eigenvalue
                    ll[0] = iso + sqrt_jj2;
                    ll[1] = iso;
                    ll[2] = iso - sqrt_jj2;
                } else {
                    // deviatoric matrix doesn't have zero eigenvalue
                    let dsj = if sj < 0.0 { 1.0 / d_box } else { d_box };
                    let alpha = 2.0 * f64::atan(dsj) / 3.0;
                    let cd = sj * fac2 * f64::cos(alpha);
                    let sd = sqrt_jj2 * f64::sin(alpha);
                    ll[0] = iso + 2.0 * cd;
                    ll[1] = iso - cd + sd;
                    ll[2] = iso - cd - sd;
                }
            }
            //
            // Harari I. and Albocher U. (2023)
            //
            EigenMethod::AnalyticalHA23 => {
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
                ll[0] = iso + lambda1;
                ll[1] = iso + lambda2;
                ll[2] = iso + lambda3;
            }
            //
            // Jacobi iterative method: calculate the eigenvalues (ignores eigenvectors)
            //
            EigenMethod::Iterative => {
                // eigenvalues and eigenvectors (ignored)
                aa.to_std_matrix_slice(&mut self.aa);
                small_mat_eigen_sym_jacobi(ll, &mut self.vv, &mut self.aa)?;
            }
        };

        // sort the eigenvalues in descending order
        let mut l0 = ll[0];
        let mut l1 = ll[1];
        let mut l2 = ll[2];
        sort3(&mut l2, &mut l1, &mut l0); // will sort: l2 < l1 < l0
        ll[0] = l0;
        ll[1] = l1;
        ll[2] = l2;

        // return false => non-spherical
        Ok(false)
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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::EigenValuesT2;
    use crate::testing::{HaberaZilian, generate_tensors2};
    use crate::{EigenMethod, SamplesTensor2, Tensor2};
    use russell_lab::{approx_eq, array_approx_eq, sort3};

    #[test]
    fn calculate_mx_works_with_samples() {
        const TOLERANCE: f64 = 1e-13;
        let mut ll = [0.0; 3];
        let mut eig = EigenValuesT2::new();
        for method in [
            EigenMethod::AnalyticalHZ,
            EigenMethod::AnalyticalHA22,
            EigenMethod::AnalyticalHA23,
            EigenMethod::Iterative,
        ] {
            for sample in SamplesTensor2::all_symmetric() {
                let sample_ll = sample.eigenvalues.unwrap();
                let mut expected_l0 = sample_ll[0];
                let mut expected_l1 = sample_ll[1];
                let mut expected_l2 = sample_ll[2];
                sort3(&mut expected_l2, &mut expected_l1, &mut expected_l0); // will sort: l2 < l1 < l0
                let aa = Tensor2::<6>::from_std_matrix(&sample.matrix).unwrap();
                eig.calculate_mx(&mut ll, &aa, method).unwrap();
                approx_eq(ll[0], expected_l0, TOLERANCE);
                approx_eq(ll[1], expected_l1, TOLERANCE);
                approx_eq(ll[2], expected_l2, TOLERANCE);
            }
        }
    }

    #[test]
    fn general_tensors2_works() {
        let mut ll = [0.0; 3];
        let mut eig = EigenValuesT2::new();
        let (tensors, eigenvalues) = generate_tensors2();
        for method in [
            EigenMethod::AnalyticalHZ,
            EigenMethod::AnalyticalHA22,
            EigenMethod::AnalyticalHA23,
            EigenMethod::Iterative,
        ] {
            for k in 0..tensors.len() {
                let aa = &tensors[k];
                let ll_ref = &eigenvalues[k];
                eig.calculate_mx(&mut ll, aa, method).unwrap();
                array_approx_eq(&ll, ll_ref, 1e-15);
            }
        }
    }

    #[test]
    fn habera_zilian_cases_work() {
        const VERBOSE: bool = true;
        const THRESHOLD: f64 = 1e-14;

        let mut ll = [0.0; 3];
        let mut eig = EigenValuesT2::new();
        let hz = HaberaZilian::new();
        for method in [
            EigenMethod::AnalyticalHZ,
            EigenMethod::AnalyticalHA22,
            EigenMethod::AnalyticalHA23,
            EigenMethod::Iterative,
        ] {
            for name in hz.names {
                for &delta in &hz.deltas {
                    // generate tensor A and calculate the eigenvalues
                    let aa = hz.tensor(name, delta);
                    eig.calculate_mx(&mut ll, &aa, method).unwrap();

                    // get the correct answer and calculate the relative tolerance
                    let mut correct = hz.diagonal(name, delta);
                    correct.sort_by(|x, y| y.partial_cmp(x).unwrap());
                    let rel_tol = 1e-12 * correct[0].abs().max(correct[2].abs());

                    // output
                    if VERBOSE {
                        // calculate the error = ||diff||_max
                        let mut error = f64::NEG_INFINITY;
                        for i in 0..3 {
                            let diff = f64::abs(ll[i] - correct[i]);
                            if diff < rel_tol {
                                error = diff;
                            }
                        }

                        // format the error in red if it exceeds the threshold
                        let m = format!("{:?}", method);
                        let error_str = if error > THRESHOLD {
                            format!("\u{1b}[31m{:>8.1e}\u{1b}[0m", error)
                        } else {
                            format!("{:>8.1e}", error)
                        };

                        // print the debugging message
                        println!(
                            "{:>14}{:>18}{:>8.1e}{:>11.4e}{:>11.4e}{:>11.4e}{:>8.1e}{}",
                            m, name, delta, ll[0], ll[1], ll[2], rel_tol, error_str,
                        );
                    }

                    // check the error using relative tolerance
                    for i in 0..3 {
                        assert!(f64::abs(ll[i] - correct[i]) < rel_tol);
                    }
                }
            }
        }
    }
}
