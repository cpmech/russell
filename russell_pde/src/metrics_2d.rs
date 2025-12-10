#![allow(unused)]

use crate::StrError;
use russell_lab::{mat_inverse, vec_inner, Matrix, Vector};

/// Calculates and stores the 2D metrics for a given mapping between reference and physical coordinates
///
/// # Definitions
///
/// * Reference coordinates: `ξ = (r, s)`
/// * Covariant base vectors: `gᵢ = ∂x/∂ξⁱ`
/// * Contravariant base vectors: `gⁱ = ∂ξⁱ/∂x`
/// * Covariant metric tensor: `gᵢⱼ = gᵢ ⋅ gⱼ`
/// * Covariant matrix: `[g] = [gᵢⱼ]`
/// * Contravariant matrix: `[G] = [g]⁻¹`
/// * Determinant of the covariant matrix: `g = det([g])`
/// * Calculation of contravariant base vectors: `gⁱ = gⁱʲ ⋅ gⱼ`
/// * Christoffel vectors: `Cᵢⱼ = ∂gᵢ/∂ξʲ`
/// * Christoffel symbols of the second kind: `Γᵏᵢⱼ = ∂gᵢ/∂ξʲ ⋅ gᵏ = Cᵢⱼ ⋅ gᵏ`
///
/// Note that the Einstein summation convention is used in the definitions above.
pub struct Metrics2d {
    /// Indicates that the base vectors and metrics do not vary with position
    homogeneous: bool,

    /// Covariant base vectors gᵢ
    ///
    /// dim = 2
    pub g_cov: Vec<Vector>,

    /// Contravariant base vectors gⁱ
    ///
    /// dim = 2
    pub g_ctr: Vec<Vector>,

    /// Matrix with the covariant metric tensor gᵢⱼ
    ///
    /// dim = 2 x 2
    pub g_mat: Matrix,

    /// Matrix with the contravariant metric tensor gⁱʲ
    ///
    /// dim = 2 x 2
    pub gg_mat: Matrix,

    /// Christoffel symbols of the second kind Γᵏᵢⱼ where the ij values equal the ji values
    ///
    /// The values can be obtained by calling `gamma[k].get(i, j)`
    ///
    /// Only available if `homogeneous` is `false`
    ///
    /// size = 2 x 2 x 2
    pub christoffel_second: Vec<Vec<Vec<f64>>>,
}

impl Metrics2d {
    /// Creates a new instance
    ///
    /// If the coordinates are homogeneous, set `homogeneous` to `true` to skip the calculation of Christoffel symbols.
    ///
    /// If the coordinates are non-homogeneous, the second derivatives must be provided when calling [Metrics2d::calculate()].
    pub fn new(homogeneous: bool) -> Self {
        let gamma = if homogeneous {
            Vec::new()
        } else {
            vec![vec![vec![0.0; 2]; 2]; 2]
        };
        Metrics2d {
            homogeneous,
            g_cov: vec![Vector::new(2), Vector::new(2)],
            g_ctr: vec![Vector::new(2), Vector::new(2)],
            g_mat: Matrix::new(2, 2),
            gg_mat: Matrix::new(2, 2),
            christoffel_second: gamma,
        }
    }

    /// Calculates the metrics at a given position and returns the determinant of the covariant matrix
    ///
    /// If the coordinates are non-homogeneous, the second derivatives must be provided.
    ///
    /// The covariant base vectors are given by:
    ///
    /// ```text
    /// g₁ = ∂x/∂r
    /// g₂ = ∂x/∂s
    /// ```
    ///
    /// The second derivatives must be provided if the coordinates are non-homogeneous. In this case,
    /// note that the Christoffel vectors `Cᵢⱼ = ∂gᵢ/∂ξʲ` are:
    ///
    /// ```text
    /// C₁₁ = ∂²x/∂r²
    /// C₂₂ = ∂²x/∂s²
    /// C₁₂ = ∂²x/∂r∂s = C₂₁
    /// ```
    pub fn calculate(
        &mut self,
        dx_dr: &Vector,
        dx_ds: &Vector,
        d2x_dr2: Option<&Vector>,
        d2x_ds2: Option<&Vector>,
        d2x_drs: Option<&Vector>,
    ) -> Result<f64, StrError> {
        // check dimensions
        if dx_dr.dim() != 2 {
            return Err("dx_dr must have dimension 2");
        }
        if dx_ds.dim() != 2 {
            return Err("dx_ds must have dimension 2");
        }

        // covariant base vectors and metrics
        for d in 0..2 {
            self.g_cov[0][d] = dx_dr[d];
            self.g_cov[1][d] = dx_ds[d];
        }

        // covariant matrix
        for i in 0..2 {
            for j in 0..2 {
                let mut g_ij = 0.0;
                for d in 0..2 {
                    g_ij += self.g_cov[i][d] * self.g_cov[j][d];
                }
                self.g_mat.set(i, j, g_ij);
            }
        }

        // contravariant matrix and determinant of the covariant matrix
        let g = mat_inverse(&mut self.gg_mat, &self.g_mat)?;

        // contravariant base vectors
        for d in 0..2 {
            for i in 0..2 {
                self.g_ctr[i][d] = 0.0;
                for j in 0..2 {
                    self.g_ctr[i][d] += self.gg_mat.get(i, j) * self.g_cov[j][d];
                }
            }
        }

        // Christoffel symbols of the second kind
        if !self.homogeneous {
            if let (Some(d2x_dr2), Some(d2x_ds2), Some(d2x_drs)) = (d2x_dr2, d2x_ds2, d2x_drs) {
                // Christoffel vectors
                let mut cc = &[
                    [d2x_dr2, d2x_drs], // C₁ⱼ
                    [d2x_drs, d2x_ds2], // C₂ⱼ
                ];

                // Christoffel symbols of the second kind: Γᵏᵢⱼ = Cᵢⱼ ⋅ gᵏ
                for k in 0..2 {
                    for j in 0..2 {
                        for i in 0..2 {
                            self.christoffel_second[k][i][j] = vec_inner(cc[i][j], &self.g_ctr[k]);
                        }
                    }
                }
            } else {
                return Err("second derivatives must be provided for non-homogeneous metrics");
            }
        }

        // return the determinant of the covariant matrix
        Ok(g)
    }

    /// Calculates the L-coefficient for the Laplacian operator
    ///
    /// Returns:
    ///
    /// ```text
    /// Lᵏ = Γᵏᵢⱼ gⁱʲ
    /// ```
    ///
    /// **Warning**: `homogeneous` must be true and [Metrics2d::calculate()] must be called before using this method.
    ///
    /// # Panics
    ///
    /// A panic will occur if `homogeneous` is false and the Christoffel symbols have not been calculated.
    pub fn ell_coefficient_for_laplacian(&self, k: usize) -> f64 {
        let mut ell = 0.0;
        for i in 0..2 {
            for j in 0..2 {
                ell += self.christoffel_second[k][i][j] * self.gg_mat.get(i, j);
            }
        }
        ell
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Metrics2d;
    use russell_lab::{approx_eq, mat_approx_eq, vec_approx_eq, Vector};

    #[test]
    fn calculate_works() {
        // Consider the mapping:
        //
        //     -1 ≤ r ≤ +1
        //     x(r) = xmin + (xmax - xmin) ⋅ (1 + r) / 2
        //     r(x) = -1 + 2⋅(x - xmin) / (xmax - xmin)
        //     dx/dr = (xmax - xmin) / 2
        //
        // On a rectangular domain, similar expressions apply for y(s) and s(y).

        // define derivatives
        let xmin = -6.0;
        let xmax = 6.0;
        let ymin = -3.0;
        let ymax = 3.0;
        let dx_dr = Vector::from(&[(xmax - xmin) / 2.0, 0.0]);
        let dx_ds = Vector::from(&[0.0, (ymax - ymin) / 2.0]);

        // calculate metrics
        let mut met = Metrics2d::new(true);
        let g = met.calculate(&dx_dr, &dx_ds, None, None, None).unwrap();

        // check [g] and [G] matrices
        println!("g = det([g]) = {}", g);
        println!("[g] =\n{}", met.g_mat);
        approx_eq(g, 36.0 * 9.0, 1e-15);
        mat_approx_eq(&met.g_mat, &[[36.0, 0.0], [0.0, 9.0]], 1e-15);
        mat_approx_eq(&met.gg_mat, &[[1.0 / 36.0, 0.0], [0.0, 1.0 / 9.0]], 1e-15);

        // check covariant and contravariant vectors
        assert_eq!(&met.g_cov[0].as_data(), &dx_dr.as_data());
        assert_eq!(&met.g_cov[1].as_data(), &dx_ds.as_data());
        vec_approx_eq(&met.g_ctr[0], &[1.0 / 6.0, 0.0], 1e-15);
        vec_approx_eq(&met.g_ctr[1], &[0.0, 1.0 / 3.0], 1e-15);
    }
}
