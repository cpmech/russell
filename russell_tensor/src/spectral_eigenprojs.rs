#![allow(unused)]

use crate::SQRT_2;
use crate::StrError;
use crate::{EigenMethod, EigenValuesT2, Tensor2};
use russell_lab::small_mat_eigen_sym_jacobi;

/// Tolerance to assume repeated eigenvalues
const TOL_REPEATED: f64 = 1e-8;

// enum EigenStatusT2 {
// Repeat01,
// Repeat12,
// }

pub struct EigenProjsT2 {
    /// Eigenvalues struct
    eig: EigenValuesT2,
}

impl EigenProjsT2 {
    /// Allocates a new instance
    pub fn new() -> Self {
        EigenProjsT2 {
            eig: EigenValuesT2::new(),
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

        // handle two-repeated eigenvalues using Panteghini's method
        let maybe_index = sorted_eigenvalues_non_repeated_index(ll);

        Ok(())
    }
}

/// Returns the index of the non-repeated eigenvalue if there are TWO repeated eigenvalues (spherical case is ignored)
fn sorted_eigenvalues_non_repeated_index(ll: &[f64; 3]) -> Option<usize> {
    let scale = ll[0].abs().max(ll[1].abs()).max(ll[2].abs()).max(1.0);
    let tol = TOL_REPEATED * scale;
    let d01 = f64::abs(ll[0] - ll[1]);
    let d12 = f64::abs(ll[1] - ll[2]);
    if d01 < tol {
        Some(2) // λ[2] = λ_III is non-repeated
    } else if d12 < tol {
        Some(0) // λ[0] = λ_I is non-repeated
    } else {
        None // no repeated found; so the result is irrelevant
    }
}
