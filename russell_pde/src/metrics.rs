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
    /// The values can be obtained by calling `gamma[k][i][j]`
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
    /// If the coordinates are non-homogeneous, the second derivatives must be provided when calling `calculate`.
    pub fn new(ndim: usize, homogeneous: bool) -> Self {
        let gamma = if homogeneous {
            Vec::new()
        } else {
            vec![vec![vec![0.0; ndim]; ndim]; ndim]
        };
        Metrics {
            homogeneous,
            ndim,
            g_cov: vec![Vector::new(ndim); ndim],
            g_ctr: vec![Vector::new(ndim); ndim],
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
    /// **Warning**: `homogeneous` must be true and `calculate()` must be called before using this method.
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
    use crate::TransfiniteSamples;
    use russell_lab::math::PI;
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

    #[test]
    fn calculate_works_3d_1() {
        // Consider the mapping:
        //
        // -1 ≤ r ≤ +1
        // x(r) = (xb + xa) / 2 + (xb - xa) r / 2
        // r(x) = (2x - xb - xa) / (xb - xa)
        // dx/dr = (xb - xa) / 2
        // dr/dx = 2 / (xb - xa)
        //
        // On a rectangular domain, similar expressions apply for y(s), s(y), z(t), and t(z).

        // with:
        //  (xa, xb) = (-2.0, -1.0)
        //  (ya, yb) = (-2.0, 2.0)
        //  (za, zb) = (-2.0, 0.0)

        // define derivatives
        let dx0_dr = 0.5; // (xb - xa) / 2.0
        let dx1_ds = 2.0; // (yb - ya) / 2.0
        let dx2_dt = 1.0; // (zb - za) / 2.0
        let dx_dr = Vector::from(&[dx0_dr, 0.0, 0.0]);
        let dx_ds = Vector::from(&[0.0, dx1_ds, 0.0]);
        let dx_dt = Vector::from(&[0.0, 0.0, dx2_dt]);

        // calculate metrics
        let mut met = Metrics::new(3, true);
        let g = met
            .calculate_3d(&dx_dr, &dx_ds, &dx_dt, None, None, None, None, None, None)
            .unwrap();

        // check [g] and [G] matrices
        println!("g = det([g]) = {}", g);
        println!("[g] =\n{}", met.g_mat);
        mat_approx_eq(
            &met.g_mat,
            &[
                [0.25, 0.0, 0.0], // g0.g0 = 0.5*0.5
                [0.0, 4.0, 0.0],  // g1.g1 = 2.0*2.0
                [0.0, 0.0, 1.0],  // g2.g2 = 1.0*1.0
            ],
            1e-15,
        );
        approx_eq(g, 0.25 * 4.0 * 1.0, 1e-15);
        mat_approx_eq(
            &met.gg_mat,
            &[[1.0 / 0.25, 0.0, 0.0], [0.0, 1.0 / 4.0, 0.0], [0.0, 0.0, 1.0 / 1.0]],
            1e-15,
        );

        // check covariant and contravariant vectors
        assert_eq!(&met.g_cov[0].as_data(), &dx_dr.as_data());
        assert_eq!(&met.g_cov[1].as_data(), &dx_ds.as_data());
        assert_eq!(&met.g_cov[2].as_data(), &dx_dt.as_data());
        vec_approx_eq(&met.g_ctr[0], &[1.0 / 0.5, 0.0, 0.0], 1e-15);
        vec_approx_eq(&met.g_ctr[1], &[0.0, 1.0 / 2.0, 0.0], 1e-15);
        vec_approx_eq(&met.g_ctr[2], &[0.0, 0.0, 1.0 / 1.0], 1e-15);

        // no Christoffel symbols for homogeneous metrics
        assert_eq!(met.homogeneous, true);
        assert_eq!(met.christoffel_second.len(), 0);
    }

    #[test]
    fn calculate_works_cylindrical_coords() {
        // x1 = ρ cos(θ)
        // x2 = ρ sin(θ)
        // x3 = z
        // with r = ρ, s = θ, and t = z
        // dx/dr = dx/dρ = [cos(θ), sin(θ)] = g₁
        // dx/ds = dx/dθ = [-ρ sin(θ), ρ cos(θ)] = g₂
        // d²x/dr² = d²x/dρ² = [0.0, 0.0]
        // d²x/ds² = d²x/dθ² = [-ρ cos(θ), -ρ sin(θ)]
        // d²x/(dr ds) = (d/ds)(dx/dr) = [-sin(θ), cos(θ)]]

        // define derivatives at a given point
        let rho = 2.0;
        let theta = PI / 6.0;
        let ct = f64::cos(theta);
        let st = f64::sin(theta);
        let dx_dr = Vector::from(&[ct, st]);
        let dx_ds = Vector::from(&[-rho * st, rho * ct]);
        let d2x_dr2 = Vector::from(&[0.0, 0.0]);
        let d2x_ds2 = Vector::from(&[-rho * ct, -rho * st]);
        let d2x_drs = Vector::from(&[-st, ct]);

        // calculate metrics
        let mut met = Metrics::new(2, false);
        let g = met
            .calculate_2d(&dx_dr, &dx_ds, Some(&d2x_dr2), Some(&d2x_ds2), Some(&d2x_drs))
            .unwrap();

        // check covariant vectors
        vec_approx_eq(&met.g_cov[0], &[ct, st], 1e-15);
        vec_approx_eq(&met.g_cov[1], &[-rho * st, rho * ct], 1e-15);

        // check [g] matrix and its determinant
        assert_eq!(g, rho * rho);
        mat_approx_eq(
            &met.g_mat,
            &[
                [1.0, 0.0],       // g₁·g₁, 0.0
                [0.0, rho * rho], // 0.0, g₂·g₂
            ],
            1e-15,
        );

        // check [G] matrix
        mat_approx_eq(
            &met.gg_mat,
            &[
                [1.0, 0.0], //
                [0.0, 1.0 / (rho * rho)],
            ],
            1e-15,
        );

        // check contravariant vectors
        vec_approx_eq(&met.g_ctr[0], &[ct, st], 1e-15);
        vec_approx_eq(&met.g_ctr[1], &[-st / rho, ct / rho], 1e-15);

        // check Christoffel symbols of the second kind
        // k = 0
        approx_eq(met.christoffel_second[0][0][0], 0.0, 1e-15);
        approx_eq(met.christoffel_second[0][0][1], 0.0, 1e-15);
        approx_eq(met.christoffel_second[0][1][0], 0.0, 1e-15);
        approx_eq(met.christoffel_second[0][1][1], -rho, 1e-15);
        // k = 1
        approx_eq(met.christoffel_second[1][0][0], 0.0, 1e-15);
        approx_eq(met.christoffel_second[1][0][1], 1.0 / rho, 1e-15);
        approx_eq(met.christoffel_second[1][1][0], 1.0 / rho, 1e-15);
        approx_eq(met.christoffel_second[1][1][1], 0.0, 1e-15);
    }

    #[test]
    fn calculate_works_spherical_coords() {
        // x1 = ρ sin(θ) cos(ɑ)
        // x2 = ρ sin(θ) sin(ɑ)
        // x3 = ρ cos(θ)
        // with r = ρ, s = θ, and t = ɑ
        // dx/dr = dx/dρ = [sin(θ) cos(ɑ), sin(θ) sin(ɑ), cos(θ)] = g₁
        // dx/ds = dx/dθ = [ρ cos(θ) cos(ɑ), ρ cos(θ) sin(ɑ), -ρ sin(θ)] = g₂
        // dx/dt = dx/dɑ = [-ρ sin(θ) sin(ɑ), ρ sin(θ) cos(ɑ), 0.0] = g₃
        // d²x/dr² = d²x/dρ² = [0.0, 0.0, 0.0]
        // d²x/ds² = d²x/dθ² = [-ρ sin(θ) cos(ɑ), -ρ sin(θ) sin(ɑ), -ρ cos(θ)]
        // d²x/dt² = d²x/dɑ² = [-ρ sin(θ) cos(ɑ), -ρ sin(θ) sin(ɑ), 0.0]
        // d²x/(dr ds) = (d/ds)(dx/dr) = [cos(θ) cos(ɑ), cos(θ) sin(ɑ), -sin(θ)]
        // d²x/(dr dt) = (d/dt)(dx/dr) = [-sin(θ) sin(ɑ), sin(θ) cos(ɑ), 0.0]
        // d²x/(ds dt) = (d/dt)(dx/ds) = [-ρ cos(θ) sin(ɑ), ρ cos(θ) cos(ɑ), 0.0]

        // define derivatives at a given point
        let rho = 2.0;
        let theta = PI / 6.0;
        let alpha = PI / 4.0;
        let ct = f64::cos(theta);
        let st = f64::sin(theta);
        let ca = f64::cos(alpha);
        let sa = f64::sin(alpha);
        let dx_dr = Vector::from(&[st * ca, st * sa, ct]);
        let dx_ds = Vector::from(&[rho * ct * ca, rho * ct * sa, -rho * st]);
        let dx_dt = Vector::from(&[-rho * st * sa, rho * st * ca, 0.0]);
        let d2x_dr2 = Vector::from(&[0.0, 0.0, 0.0]);
        let d2x_ds2 = Vector::from(&[-rho * st * ca, -rho * st * sa, -rho * ct]);
        let d2x_dt2 = Vector::from(&[-rho * st * ca, -rho * st * sa, 0.0]);
        let d2x_drs = Vector::from(&[ct * ca, ct * sa, -st]);
        let d2x_drt = Vector::from(&[-st * sa, st * ca, 0.0]);
        let d2x_dst = Vector::from(&[-rho * ct * sa, rho * ct * ca, 0.0]);

        // calculate metrics
        let mut met = Metrics::new(3, false);
        let g = met
            .calculate_3d(
                &dx_dr,
                &dx_ds,
                &dx_dt,
                Some(&d2x_dr2),
                Some(&d2x_ds2),
                Some(&d2x_dt2),
                Some(&d2x_drs),
                Some(&d2x_drt),
                Some(&d2x_dst),
            )
            .unwrap();

        // check covariant vectors
        vec_approx_eq(&met.g_cov[0], &[st * ca, st * sa, ct], 1e-15);
        vec_approx_eq(&met.g_cov[1], &[rho * ct * ca, rho * ct * sa, -rho * st], 1e-15);
        vec_approx_eq(&met.g_cov[2], &[-rho * st * sa, rho * st * ca, 0.0], 1e-15);

        // check [g] matrix and its determinant
        let a = rho * rho;
        let b = a * st * st;
        approx_eq(g, a * b, 1e-15);
        mat_approx_eq(
            &met.g_mat,
            &[
                [1.0, 0.0, 0.0], // g₁·g₁, 0.0, 0.0
                [0.0, a, 0.0],   // 0.0, g₂·g₂, 0.0
                [0.0, 0.0, b],   // 0.0, 0.0, g₃·g₃
            ],
            1e-15,
        );

        // check [G] matrix
        mat_approx_eq(
            &met.gg_mat,
            &[
                [1.0, 0.0, 0.0],     //
                [0.0, 1.0 / a, 0.0], //
                [0.0, 0.0, 1.0 / b], //
            ],
            1e-15,
        );

        // check contravariant vectors
        vec_approx_eq(&met.g_ctr[0], &[st * ca, st * sa, ct], 1e-15);
        vec_approx_eq(&met.g_ctr[1], &[ct * ca / rho, ct * sa / rho, -st / rho], 1e-15);
        vec_approx_eq(&met.g_ctr[2], &[-sa / (rho * st), ca / (rho * st), 0.0], 1e-15);

        // check Christoffel symbols of the second kind
        // k = 0
        approx_eq(met.christoffel_second[0][0][0], 0.0, 1e-15);
        approx_eq(met.christoffel_second[0][0][1], 0.0, 1e-15);
        approx_eq(met.christoffel_second[0][0][2], 0.0, 1e-15);
        approx_eq(met.christoffel_second[0][1][0], 0.0, 1e-15);
        approx_eq(met.christoffel_second[0][1][1], -rho, 1e-15);
        approx_eq(met.christoffel_second[0][1][2], 0.0, 1e-15);
        approx_eq(met.christoffel_second[0][2][0], 0.0, 1e-15);
        approx_eq(met.christoffel_second[0][2][1], 0.0, 1e-15);
        approx_eq(met.christoffel_second[0][2][2], -rho * st * st, 1e-15);
        // k = 1
        approx_eq(met.christoffel_second[1][0][0], 0.0, 1e-15);
        approx_eq(met.christoffel_second[1][0][1], 1.0 / rho, 1e-15);
        approx_eq(met.christoffel_second[1][0][2], 0.0, 1e-15);
        approx_eq(met.christoffel_second[1][1][0], 1.0 / rho, 1e-15);
        approx_eq(met.christoffel_second[1][1][1], 0.0, 1e-15);
        approx_eq(met.christoffel_second[1][1][2], 0.0, 1e-15);
        approx_eq(met.christoffel_second[1][2][0], 0.0, 1e-15);
        approx_eq(met.christoffel_second[1][2][1], 0.0, 1e-15);
        approx_eq(met.christoffel_second[1][2][2], -st * ct, 1e-15);
        // k = 2
        approx_eq(met.christoffel_second[2][0][0], 0.0, 1e-15);
        approx_eq(met.christoffel_second[2][0][1], 0.0, 1e-15);
        approx_eq(met.christoffel_second[2][0][2], 1.0 / rho, 1e-15);
        approx_eq(met.christoffel_second[2][1][0], 0.0, 1e-15);
        approx_eq(met.christoffel_second[2][1][1], 0.0, 1e-15);
        approx_eq(met.christoffel_second[2][1][2], ct / st, 1e-15);
        approx_eq(met.christoffel_second[2][2][0], 1.0 / rho, 1e-15);
        approx_eq(met.christoffel_second[2][2][1], ct / st, 1e-15);
        approx_eq(met.christoffel_second[2][2][2], 0.0, 1e-15);
    }

    #[test]
    fn calculate_works_quarter_ring_2d() {
        // Quarter ring with a 2D transfinite mapping
        //
        // Cylindrical coordinates (ρ, θ):
        // a: inner radius
        // b: outer radius
        // x1(ρ(r),θ(s)) = ρ cos(θ)
        // x2(ρ(r),θ(s)) = ρ sin(θ)
        // ρ(r) = (b + a) / 2 + (b - a) r / 2
        // θ(s) = (π / 4) (1 + s)
        // dρ/dr = (b - a) / 2
        // dθ/ds = π / 4
        // dx/dr = dx/dρ dρ/dr = [cos(θ) dρ/dr, sin(θ) dρ/dr] = g₁
        // dx/ds = dx/dθ dθ/ds = [-ρ sin(θ) dθ/ds, ρ cos(θ) dθ/ds] = g₂
        let (a, b) = (1.0, 2.0);
        let mut map = TransfiniteSamples::quarter_ring_2d(a, b);
        let mut x = Vector::new(2);
        let mut dx_dr = Vector::new(2);
        let mut dx_ds = Vector::new(2);
        let mut d2x_dr2 = Vector::new(2);
        let mut d2x_ds2 = Vector::new(2);
        let mut d2x_drs = Vector::new(2);

        // allocate the metrics instance
        let mut met = Metrics::new(2, false);

        // auxiliary constants
        let p = (b - a) / 2.0; // dρ/dr
        let q = PI / 4.0; // dθ/ds

        // loop over sample points
        for r in [-1.0, -0.5, 0.0, 0.5, 1.0] {
            for s in [-1.0, -0.5, 0.0, 0.5, 1.0] {
                // calculate point and derivatives @ (r,s)
                map.point_and_derivs(
                    &mut x,
                    &mut dx_dr,
                    &mut dx_ds,
                    Some(&mut d2x_dr2),
                    Some(&mut d2x_ds2),
                    Some(&mut d2x_drs),
                    r,
                    s,
                );

                // calculate the base vectors and metrics
                let g = met
                    .calculate_2d(&dx_dr, &dx_ds, Some(&d2x_dr2), Some(&d2x_ds2), Some(&d2x_drs))
                    .unwrap();

                // check x and g_cov
                let rho = (b + a) / 2.0 + (b - a) * r / 2.0;
                let theta = (PI / 4.0) * (1.0 + s);
                let ct = f64::cos(theta);
                let st = f64::sin(theta);
                vec_approx_eq(&x, &[rho * ct, rho * st], 1e-15);
                vec_approx_eq(&met.g_cov[0], &[ct * p, st * p], 1e-15);
                vec_approx_eq(&met.g_cov[1], &[-rho * st * q, rho * ct * q], 1e-15);

                // check [g] and [G] matrices
                mat_approx_eq(
                    &met.g_mat,
                    &[
                        [p * p, 0.0],             // g₁·g₁, 0.0
                        [0.0, rho * rho * q * q], // 0.0, g₂·g₂
                    ],
                    1e-15,
                );
                approx_eq(g, p * p * rho * rho * q * q, 1e-15);
                mat_approx_eq(
                    &met.gg_mat,
                    &[
                        [1.0 / (p * p), 0.0],             //
                        [0.0, 1.0 / (rho * rho * q * q)], //
                    ],
                    1e-14,
                );

                // check g_ctr
                vec_approx_eq(&met.g_ctr[0], &[ct / p, st / p], 1e-15);
                vec_approx_eq(&met.g_ctr[1], &[-st / (rho * q), ct / (rho * q)], 1e-15);

                // check Christoffel symbols of the second kind
                // k = 0
                approx_eq(met.christoffel_second[0][0][0], 0.0, 1e-15);
                approx_eq(met.christoffel_second[0][0][1], 0.0, 1e-15);
                approx_eq(met.christoffel_second[0][1][0], 0.0, 1e-15);
                approx_eq(met.christoffel_second[0][1][1], -rho * q * q / p, 1e-15);
                // k = 1
                approx_eq(met.christoffel_second[1][0][0], 0.0, 1e-15);
                approx_eq(met.christoffel_second[1][0][1], p / rho, 1e-15);
                approx_eq(met.christoffel_second[1][1][0], p / rho, 1e-15);
                approx_eq(met.christoffel_second[1][1][1], 0.0, 1e-15);
            }
        }
    }

    #[test]
    fn calculate_works_quarter_ring_3d() {
        // Quarter ring with a 3D transfinite mapping
        //
        // Cylindrical coordinates on y-z (ρ, θ), extruded in x:
        // a: inner radius
        // b: outer radius
        // h: thickness in x
        // x1(r) = h / 2 + h r / 2
        // x2(ρ(s),θ(t)) = ρ cos(θ)
        // x3(ρ(s),θ(t)) = ρ sin(θ)
        // ρ(s) = (b + a) / 2 + (b - a) s / 2
        // θ(t) = (π / 4) (1 + t)
        // dρ/ds = (b - a) / 2
        // dθ/dt = π / 4
        // dx/dr = [h / 2, 0.0, 0.0] = g₃
        // dx/ds = dx/dρ dρ/ds = [0.0, cos(θ) dρ/ds, sin(θ) dρ/ds] = g₁
        // dx/dt = dx/dθ dθ/dt = [0.0, -ρ sin(θ) dθ/dt, ρ cos(θ) dθ/dt] = g₂
        let (a, b, h) = (1.0, 2.0, 3.0);
        let mut map = TransfiniteSamples::quarter_ring_3d(a, b, h);
        let mut x = Vector::new(3);
        let mut dx_dr = Vector::new(3);
        let mut dx_ds = Vector::new(3);
        let mut dx_dt = Vector::new(3);
        let mut d2x_dr2 = Vector::new(3);
        let mut d2x_ds2 = Vector::new(3);
        let mut d2x_dt2 = Vector::new(3);
        let mut d2x_drs = Vector::new(3);
        let mut d2x_drt = Vector::new(3);
        let mut d2x_dst = Vector::new(3);

        // allocate the metrics instance
        let mut met = Metrics::new(3, false);

        // auxiliary constants
        let p = (b - a) / 2.0; // dρ/ds
        let q = PI / 4.0; // dθ/dt

        // loop over sample points
        for t in [-1.0, -0.5, 0.0, 0.5, 1.0] {
            for r in [-1.0, -0.5, 0.0, 0.5, 1.0] {
                for s in [-1.0, -0.5, 0.0, 0.5, 1.0] {
                    // calculate point and derivatives @ (r,s)
                    map.point_and_derivs(
                        &mut x,
                        &mut dx_dr,
                        &mut dx_ds,
                        &mut dx_dt,
                        Some(&mut d2x_dr2),
                        Some(&mut d2x_ds2),
                        Some(&mut d2x_dt2),
                        Some(&mut d2x_drs),
                        Some(&mut d2x_drt),
                        Some(&mut d2x_dst),
                        r,
                        s,
                        t,
                    );

                    // calculate the base vectors and metrics
                    let g = met
                        .calculate_3d(
                            &dx_dr,
                            &dx_ds,
                            &dx_dt,
                            Some(&d2x_dr2),
                            Some(&d2x_ds2),
                            Some(&d2x_dt2),
                            Some(&d2x_drs),
                            Some(&d2x_drt),
                            Some(&d2x_dst),
                        )
                        .unwrap();

                    // check x and g_cov
                    let x1 = h / 2.0 + h * r / 2.0;
                    let rho = (b + a) / 2.0 + (b - a) * s / 2.0;
                    let theta = (PI / 4.0) * (1.0 + t);
                    let ct = f64::cos(theta);
                    let st = f64::sin(theta);
                    vec_approx_eq(&x, &[x1, rho * ct, rho * st], 1e-14);
                    vec_approx_eq(&met.g_cov[0], &[h / 2.0, 0.0, 0.0], 1e-15);
                    vec_approx_eq(&met.g_cov[1], &[0.0, ct * p, st * p], 1e-15);
                    vec_approx_eq(&met.g_cov[2], &[0.0, -rho * st * q, rho * ct * q], 1e-14);

                    // check [g] and [G] matrices
                    mat_approx_eq(
                        &met.g_mat,
                        &[
                            [h * h / 4.0, 0.0, 0.0],       // g₁·g₁, 0.0, 0.0
                            [0.0, p * p, 0.0],             // 0.0, g₂·g₂, 0.0
                            [0.0, 0.0, rho * rho * q * q], // 0.0, 0.0, g₃·g₃
                        ],
                        1e-14,
                    );
                    approx_eq(g, (h * h / 4.0) * p * p * rho * rho * q * q, 1e-14);
                    mat_approx_eq(
                        &met.gg_mat,
                        &[
                            [4.0 / (h * h), 0.0, 0.0],             //
                            [0.0, 1.0 / (p * p), 0.0],             //
                            [0.0, 0.0, 1.0 / (rho * rho * q * q)], //
                        ],
                        1e-14,
                    );

                    // check g_ctr
                    vec_approx_eq(&met.g_ctr[0], &[2.0 / h, 0.0, 0.0], 1e-14);
                    vec_approx_eq(&met.g_ctr[1], &[0.0, ct / p, st / p], 1e-14);
                    vec_approx_eq(
                        &met.g_ctr[2],
                        &[0.0, -rho * st / (rho * rho * q), rho * ct / (rho * rho * q)],
                        1e-15,
                    );

                    // check Christoffel symbols of the second kind
                    // k = 0
                    for i in 0..3 {
                        for j in 0..3 {
                            approx_eq(met.christoffel_second[0][i][j], 0.0, 1e-15);
                        }
                    }
                    // k = 1
                    for i in 0..3 {
                        for j in 0..3 {
                            if i == 2 && j == 2 {
                                approx_eq(met.christoffel_second[1][2][2], -rho * q * q / p, 1e-14);
                            } else {
                                approx_eq(met.christoffel_second[1][i][j], 0.0, 1e-15);
                            }
                        }
                    }
                    // k = 2
                    for i in 0..3 {
                        for j in 0..3 {
                            if (i == 1 && j == 2) || (i == 2 && j == 1) {
                                approx_eq(met.christoffel_second[2][i][j], p / rho, 1e-15);
                            } else {
                                approx_eq(met.christoffel_second[2][i][j], 0.0, 1e-15);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn calculate_2d_captures_errors() {
        let mut metrics = Metrics::new(3, false);
        let v2 = Vector::new(2);
        let v3 = Vector::new(3);
        assert_eq!(
            metrics.calculate_2d(&v2, &v2, None, None, None).err(),
            Some("calculate_2d only works for ndim = 2")
        );
        let mut metrics = Metrics::new(2, false);
        assert_eq!(
            metrics.calculate_2d(&v3, &v2, None, None, None).err(),
            Some("dx_dr must have dimension 2")
        );
        assert_eq!(
            metrics.calculate_2d(&v2, &v3, None, None, None).err(),
            Some("dx_ds must have dimension 2")
        );
        assert_eq!(
            metrics.calculate_2d(&v2, &v2, None, None, None).err(),
            Some("d2x_dr2 must be provided for non-homogeneous metrics")
        );
        assert_eq!(
            metrics.calculate_2d(&v2, &v2, Some(&v2), None, None).err(),
            Some("d2x_ds2 must be provided for non-homogeneous metrics")
        );
        assert_eq!(
            metrics.calculate_2d(&v2, &v2, Some(&v2), Some(&v2), None).err(),
            Some("d2x_drs must be provided for non-homogeneous metrics")
        );
    }

    #[test]
    fn calculate_3d_captures_errors() {
        let mut metrics = Metrics::new(2, false);
        let v2 = Vector::new(2);
        let v3 = Vector::new(3);
        assert_eq!(
            metrics
                .calculate_3d(&v3, &v3, &v3, None, None, None, None, None, None)
                .err(),
            Some("calculate_3d only works for ndim = 3")
        );
        let mut metrics = Metrics::new(3, false);
        assert_eq!(
            metrics
                .calculate_3d(&v2, &v3, &v3, None, None, None, None, None, None)
                .err(),
            Some("dx_dr must have dimension 3")
        );
        assert_eq!(
            metrics
                .calculate_3d(&v3, &v2, &v3, None, None, None, None, None, None)
                .err(),
            Some("dx_ds must have dimension 3")
        );
        assert_eq!(
            metrics
                .calculate_3d(&v3, &v3, &v2, None, None, None, None, None, None)
                .err(),
            Some("dx_dt must have dimension 3")
        );
        assert_eq!(
            metrics
                .calculate_3d(&v3, &v3, &v3, None, None, None, None, None, None)
                .err(),
            Some("d2x_dr2 must be provided for non-homogeneous metrics")
        );
        assert_eq!(
            metrics
                .calculate_3d(&v3, &v3, &v3, Some(&v3), None, None, None, None, None)
                .err(),
            Some("d2x_ds2 must be provided for non-homogeneous metrics")
        );
        assert_eq!(
            metrics
                .calculate_3d(&v3, &v3, &v3, Some(&v3), Some(&v3), None, None, None, None)
                .err(),
            Some("d2x_dt2 must be provided for non-homogeneous metrics")
        );
        assert_eq!(
            metrics
                .calculate_3d(&v3, &v3, &v3, Some(&v3), Some(&v3), Some(&v3), None, None, None)
                .err(),
            Some("d2x_drs must be provided for non-homogeneous metrics")
        );
        assert_eq!(
            metrics
                .calculate_3d(&v3, &v3, &v3, Some(&v3), Some(&v3), Some(&v3), Some(&v3), None, None)
                .err(),
            Some("d2x_drt must be provided for non-homogeneous metrics")
        );
        assert_eq!(
            metrics
                .calculate_3d(
                    &v3,
                    &v3,
                    &v3,
                    Some(&v3),
                    Some(&v3),
                    Some(&v3),
                    Some(&v3),
                    Some(&v3),
                    None
                )
                .err(),
            Some("d2x_dst must be provided for non-homogeneous metrics")
        );
    }
}
