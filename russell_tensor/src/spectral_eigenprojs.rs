#![allow(unused)]

use crate::StrError;
use crate::{EigenMethod, EigenValuesT2, Tensor2};
use crate::{IDENTITY2, ONE_BY_3, SQRT_2};
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

        // handle two-repeated eigenvalues using Panteghini's method
        let l_ii = ll[1]; // λ_II
        let l_hat = if d01 < tol {
            ll[2] // λ_III
        } else {
            ll[0] // λ_I
        };
        aa.deviator_slice(&mut self.ss);
        let kappa = l_hat - l_ii;
        for m in 0..6 {
            projs[0].vec[m] = ONE_BY_3 * IDENTITY2[m] + (1.0 / kappa) * self.ss[m];
            projs[1].vec[m] = IDENTITY2[m] - projs[0].vec[m];
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
