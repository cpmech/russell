use crate::Tensor4;

/// Implements the linear elasticity equations for small-strain problems
pub struct LinElasticity {
    /// Young's modulus
    young: f64,

    /// Poisson's coefficient
    poisson: f64,

    /// Plane-stress flag
    plane_stress: bool,

    /// Elasticity modulus (on Mandel basis) such that σ = D : ε
    dd: Tensor4,
}

impl LinElasticity {
    /// Creates a new linear-elasticity structure
    ///
    /// # Input
    ///
    /// * `young` -- Young's modulus
    /// * `poisson` -- Poisson's coefficient
    /// * `two_dim` -- 2D instead of 3D
    /// * `plane_stress` -- if `two_dim == 2`, specifies a Plane-Stress problem.
    ///                     Note: if true, this flag automatically turns `two_dim` to true.
    pub fn new(young: f64, poisson: f64, two_dim: bool, plane_stress: bool) -> Self {
        let mut res = LinElasticity {
            young,
            poisson,
            plane_stress,
            dd: Tensor4::new(true, two_dim || plane_stress),
        };
        res.calc_modulus();
        res
    }

    /// Sets the Young's modulus and Poisson's coefficient
    pub fn set_young_poisson(&mut self, young: f64, poisson: f64) {
        self.young = young;
        self.poisson = poisson;
        self.calc_modulus();
    }

    /// Computes elasticity modulus
    fn calc_modulus(&mut self) {
        if self.plane_stress {
            let c = self.young / (1.0 - self.poisson * self.poisson);
            self.dd.mat[0][0] = c;
            self.dd.mat[0][1] = c * self.poisson;
            self.dd.mat[1][0] = c * self.poisson;
            self.dd.mat[1][1] = c;
            self.dd.mat[3][3] = c * (1.0 - self.poisson); // Mandel: multiply by 2, so 1/2 disappears
        } else {
            let c = self.young / ((1.0 + self.poisson) * (1.0 - 2.0 * self.poisson));
            self.dd.mat[0][0] = c * (1.0 - self.poisson);
            self.dd.mat[0][1] = c * self.poisson;
            self.dd.mat[0][2] = c * self.poisson;
            self.dd.mat[1][0] = c * self.poisson;
            self.dd.mat[1][1] = c * (1.0 - self.poisson);
            self.dd.mat[1][2] = c * self.poisson;
            self.dd.mat[2][0] = c * self.poisson;
            self.dd.mat[2][1] = c * self.poisson;
            self.dd.mat[2][2] = c * (1.0 - self.poisson);
            self.dd.mat[3][3] = c * (1.0 - 2.0 * self.poisson); // Mandel: multiply by 2, so 1/2 disappears
        }
        if self.dd.mat.dims().0 > 4 {
            self.dd.mat[4][4] = self.dd.mat[3][3];
            self.dd.mat[5][5] = self.dd.mat[3][3];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::LinElasticity;

    #[test]
    fn new_works() {
        let ela = LinElasticity::new(3000.0, 0.2, false, true);
        let out = ela.dd.to_matrix();
        assert_eq!(
            format!("{}", out),
            "┌                                              ┐\n\
             │ 3125  625    0    0    0    0    0    0    0 │\n\
             │  625 3125    0    0    0    0    0    0    0 │\n\
             │    0    0    0    0    0    0    0    0    0 │\n\
             │    0    0    0 1250    0    0 1250    0    0 │\n\
             │    0    0    0    0    0    0    0    0    0 │\n\
             │    0    0    0    0    0    0    0    0    0 │\n\
             │    0    0    0 1250    0    0 1250    0    0 │\n\
             │    0    0    0    0    0    0    0    0    0 │\n\
             │    0    0    0    0    0    0    0    0    0 │\n\
             └                                              ┘"
        );
    }
}
