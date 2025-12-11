use crate::StrError;
use russell_lab::{mat_inverse, vec_inner, Matrix, Vector};

/// Calculates and stores the metrics coefficients for a given mapping between reference and physical coordinates
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
pub struct Metrics {
    /// Dimension
    ndim: usize,

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

    /// Zero vector for internal algorithms
    zero: Vector,
}

impl Metrics {
    /// Creates a new instance
    ///
    /// If the coordinates are homogeneous, set `homogeneous` to `true` to skip the calculation of Christoffel symbols.
    ///
    /// If the coordinates are non-homogeneous, the second derivatives must be provided when calling [Metrics2d::calculate()].
    pub fn new(ndim: usize, homogeneous: bool) -> Self {
        let gamma = if homogeneous {
            Vec::new()
        } else {
            vec![vec![vec![0.0; ndim]; ndim]; ndim]
        };
        Metrics {
            homogeneous,
            ndim,
            g_cov: vec![Vector::new(ndim), Vector::new(ndim)],
            g_ctr: vec![Vector::new(ndim), Vector::new(ndim)],
            g_mat: Matrix::new(ndim, ndim),
            gg_mat: Matrix::new(ndim, ndim),
            christoffel_second: gamma,
            zero: Vector::new(ndim),
        }
    }

    /// Calculates the metrics at a given position and returns the determinant of the covariant matrix (2D version)
    ///
    /// Returns the determinant of the covariant matrix.
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
    pub fn calculate_2d(
        &mut self,
        dx_dr: &Vector,
        dx_ds: &Vector,
        d2x_dr2: Option<&Vector>,
        d2x_ds2: Option<&Vector>,
        d2x_drs: Option<&Vector>,
    ) -> Result<f64, StrError> {
        if self.ndim != 2 {
            return Err("calculate_2d only works for ndim = 2");
        }
        if dx_dr.dim() != 2 {
            return Err("dx_dr must have dimension 2");
        }
        if dx_ds.dim() != 2 {
            return Err("dx_ds must have dimension 2");
        }
        if !self.homogeneous {
            if d2x_dr2.is_none() {
                return Err("d2x_dr2 must be provided for non-homogeneous metrics");
            }
            if d2x_ds2.is_none() {
                return Err("d2x_ds2 must be provided for non-homogeneous metrics");
            }
            if d2x_drs.is_none() {
                return Err("d2x_drs must be provided for non-homogeneous metrics");
            }
        }
        self.calculate(
            dx_dr,   // dx_dr
            dx_ds,   // dx_ds
            None,    // dx_dt
            d2x_dr2, // d2x_dr2
            d2x_ds2, // d2x_ds2
            None,    // d2x_dt2
            d2x_drs, // d2x_drs
            None,    // d2x_drt
            None,    // d2x_dst
        )
    }

    /// Calculates the metrics at a given position and returns the determinant of the covariant matrix (3D version)
    ///
    /// Returns the determinant of the covariant matrix.
    ///
    /// If the coordinates are non-homogeneous, the second derivatives must be provided.
    ///
    /// The covariant base vectors are given by:
    ///
    /// ```text
    /// g₁ = ∂x/∂r
    /// g₂ = ∂x/∂s
    /// g₃ = ∂x/∂t
    /// ```
    ///
    /// The second derivatives must be provided if the coordinates are non-homogeneous. In this case,
    /// note that the Christoffel vectors `Cᵢⱼ = ∂gᵢ/∂ξʲ` are:
    ///
    /// ```text
    /// C₁₁ = ∂²x/∂r²
    /// C₂₂ = ∂²x/∂s²
    /// C₃₃ = ∂²x/∂t²
    /// C₁₂ = ∂²x/∂r∂s = C₂₁
    /// C₁₃ = ∂²x/∂r∂t = C₃₁
    /// C₂₃ = ∂²x/∂s∂t = C₃₂
    /// ```
    pub fn calculate_3d(
        &mut self,
        dx_dr: &Vector,
        dx_ds: &Vector,
        dx_dt: &Vector,
        d2x_dr2: Option<&Vector>,
        d2x_ds2: Option<&Vector>,
        d2x_dt2: Option<&Vector>,
        d2x_drs: Option<&Vector>,
        d2x_drt: Option<&Vector>,
        d2x_dst: Option<&Vector>,
    ) -> Result<f64, StrError> {
        if self.ndim != 3 {
            return Err("calculate_3d only works for ndim = 3");
        }
        if dx_dr.dim() != 3 {
            return Err("dx_dr must have dimension 3");
        }
        if dx_ds.dim() != 3 {
            return Err("dx_ds must have dimension 3");
        }
        if dx_dt.dim() != 3 {
            return Err("dx_dt must have dimension 3");
        }
        if !self.homogeneous {
            if d2x_dr2.is_none() {
                return Err("d2x_dr2 must be provided for non-homogeneous metrics");
            }
            if d2x_ds2.is_none() {
                return Err("d2x_ds2 must be provided for non-homogeneous metrics");
            }
            if d2x_dt2.is_none() {
                return Err("d2x_dt2 must be provided for non-homogeneous metrics");
            }
            if d2x_drs.is_none() {
                return Err("d2x_drs must be provided for non-homogeneous metrics");
            }
            if d2x_drt.is_none() {
                return Err("d2x_drt must be provided for non-homogeneous metrics");
            }
            if d2x_dst.is_none() {
                return Err("d2x_dst must be provided for non-homogeneous metrics");
            }
        }
        self.calculate(
            dx_dr,
            dx_ds,
            Some(dx_dt),
            d2x_dr2,
            d2x_ds2,
            d2x_dt2,
            d2x_drs,
            d2x_drt,
            d2x_dst,
        )
    }

    /// Calculates the metrics at a given position and returns the determinant of the covariant matrix
    ///
    /// Returns the determinant of the covariant matrix.
    fn calculate(
        &mut self,
        dx_dr: &Vector,
        dx_ds: &Vector,
        dx_dt: Option<&Vector>,
        d2x_dr2: Option<&Vector>,
        d2x_ds2: Option<&Vector>,
        d2x_dt2: Option<&Vector>,
        d2x_drs: Option<&Vector>,
        d2x_drt: Option<&Vector>,
        d2x_dst: Option<&Vector>,
    ) -> Result<f64, StrError> {
        // covariant base vectors and metrics
        for d in 0..self.ndim {
            self.g_cov[0][d] = dx_dr[d];
            self.g_cov[1][d] = dx_ds[d];
        }
        if self.ndim == 3 {
            let dx_dt = dx_dt.unwrap();
            for d in 0..self.ndim {
                self.g_cov[2][d] = dx_dt[d];
            }
        }

        // covariant matrix
        for i in 0..self.ndim {
            for j in 0..self.ndim {
                let mut g_ij = 0.0;
                for d in 0..self.ndim {
                    g_ij += self.g_cov[i][d] * self.g_cov[j][d];
                }
                self.g_mat.set(i, j, g_ij);
            }
        }

        // contravariant matrix and determinant of the covariant matrix
        let g = mat_inverse(&mut self.gg_mat, &self.g_mat)?;

        // contravariant base vectors
        for d in 0..self.ndim {
            for i in 0..self.ndim {
                self.g_ctr[i][d] = 0.0;
                for j in 0..self.ndim {
                    self.g_ctr[i][d] += self.gg_mat.get(i, j) * self.g_cov[j][d];
                }
            }
        }

        // Christoffel symbols of the second kind
        if !self.homogeneous {
            let d2x_dr2 = d2x_dr2.unwrap();
            let d2x_ds2 = d2x_ds2.unwrap();
            let d2x_drs = d2x_drs.unwrap();

            // Christoffel vectors
            let cc = if self.ndim == 2 {
                &[
                    [d2x_dr2, d2x_drs, &self.zero],       // C₁ⱼ
                    [d2x_drs, d2x_ds2, &self.zero],       // C₂ⱼ
                    [&self.zero, &self.zero, &self.zero], // C₃ⱼ
                ]
            } else {
                let d2x_dt2 = d2x_dt2.unwrap();
                let d2x_drt = d2x_drt.unwrap();
                let d2x_dst = d2x_dst.unwrap();
                &[
                    [d2x_dr2, d2x_drs, d2x_drt], // C₁ⱼ
                    [d2x_drs, d2x_ds2, d2x_dst], // C₂ⱼ
                    [d2x_drt, d2x_dst, d2x_dt2], // C₃ⱼ
                ]
            };

            // Christoffel symbols of the second kind: Γᵏᵢⱼ = Cᵢⱼ ⋅ gᵏ
            for k in 0..self.ndim {
                for j in 0..self.ndim {
                    for i in 0..self.ndim {
                        self.christoffel_second[k][i][j] = vec_inner(cc[i][j], &self.g_ctr[k]);
                    }
                }
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
        for i in 0..self.ndim {
            for j in 0..self.ndim {
                ell += self.christoffel_second[k][i][j] * self.gg_mat.get(i, j);
            }
        }
        ell
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Metrics;
    use russell_lab::{approx_eq, mat_approx_eq, vec_approx_eq, Vector};

    #[test]
    fn calculate_works_2d_1() {
        // Consider the mapping:
        //
        // -1 ≤ r ≤ +1
        // x(r) = (xb + xa) / 2 + (xb - xa) r / 2
        // r(x) = (2x - xb - xa) / (xb - xa)
        // dx/dr = (xb - xa) / 2
        // dr/dx = 2 / (xb - xa)
        //
        // On a rectangular domain, similar expressions apply for y(s) and s(y).

        // define derivatives
        let (xa, xb) = (-6.0, 6.0);
        let (ya, yb) = (-3.0, 3.0);
        let dx_dr = Vector::from(&[(xb - xa) / 2.0, 0.0]);
        let dx_ds = Vector::from(&[0.0, (yb - ya) / 2.0]);

        // calculate metrics
        let mut met = Metrics::new(2, true);
        let g = met.calculate_2d(&dx_dr, &dx_ds, None, None, None).unwrap();

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

        // no Christoffel symbols for homogeneous metrics
        assert_eq!(met.homogeneous, true);
        assert_eq!(met.christoffel_second.len(), 0);
    }
}
