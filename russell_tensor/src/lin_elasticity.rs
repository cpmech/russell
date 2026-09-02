use crate::{SET, StrError, Tensor2, Tensor4, t4_ddot_t2};

/// Implements the linear elasticity equations for small-strain problems
pub struct LinElasticity<const N: usize> {
    /// Holds the Young's modulus
    young: f64,

    /// Holds the Poisson's coefficient
    poisson: f64,

    /// Holds the plane-stress flag
    plane_stress: bool,

    /// Holds the elastic rigidity (stiffness) modulus
    ///
    /// The rigidity modulus `D` is such that:
    ///
    /// ```text
    /// σ = D : ε
    /// ```
    dd: Tensor4<N>,
}

impl<const N: usize> LinElasticity<N> {
    /// Allocates a new instance
    ///
    /// # Input
    ///
    /// * `young` -- Young's modulus
    /// * `poisson` -- Poisson's coefficient
    /// * `plane_stress` -- specifies a Plane-Stress problem; only valid if N = 4 (2D)
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_tensor::{LinElasticity, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     // Symmetric
    ///     let ela = LinElasticity::<6>::new(900.0, 0.25, false)?;
    ///     let dd = ela.stiffness().as_std_matrix();
    ///     assert_eq!(
    ///         format!("{}", dd),
    ///         "┌                                              ┐\n\
    ///          │ 1080  360  360    0    0    0    0    0    0 │\n\
    ///          │  360 1080  360    0    0    0    0    0    0 │\n\
    ///          │  360  360 1080    0    0    0    0    0    0 │\n\
    ///          │    0    0    0  360    0    0  360    0    0 │\n\
    ///          │    0    0    0    0  360    0    0  360    0 │\n\
    ///          │    0    0    0    0    0  360    0    0  360 │\n\
    ///          │    0    0    0  360    0    0  360    0    0 │\n\
    ///          │    0    0    0    0  360    0    0  360    0 │\n\
    ///          │    0    0    0    0    0  360    0    0  360 │\n\
    ///          └                                              ┘"
    ///     );
    ///
    ///     // 2D plane-strain
    ///     let ela = LinElasticity::<4>::new(900.0, 0.25, false)?;
    ///     let dd = ela.stiffness().as_std_matrix();
    ///     assert_eq!(
    ///         format!("{}", dd),
    ///         "┌                                              ┐\n\
    ///          │ 1080  360  360    0    0    0    0    0    0 │\n\
    ///          │  360 1080  360    0    0    0    0    0    0 │\n\
    ///          │  360  360 1080    0    0    0    0    0    0 │\n\
    ///          │    0    0    0  360    0    0  360    0    0 │\n\
    ///          │    0    0    0    0    0    0    0    0    0 │\n\
    ///          │    0    0    0    0    0    0    0    0    0 │\n\
    ///          │    0    0    0  360    0    0  360    0    0 │\n\
    ///          │    0    0    0    0    0    0    0    0    0 │\n\
    ///          │    0    0    0    0    0    0    0    0    0 │\n\
    ///          └                                              ┘"
    ///     );
    ///
    ///     // 2D plane-stress
    ///     let ela = LinElasticity::<4>::new(3000.0, 0.2, true)?;
    ///     let dd = ela.stiffness().as_std_matrix();
    ///     assert_eq!(
    ///         format!("{}", dd),
    ///         "┌                                              ┐\n\
    ///          │ 3125  625    0    0    0    0    0    0    0 │\n\
    ///          │  625 3125    0    0    0    0    0    0    0 │\n\
    ///          │    0    0    0    0    0    0    0    0    0 │\n\
    ///          │    0    0    0 1250    0    0 1250    0    0 │\n\
    ///          │    0    0    0    0    0    0    0    0    0 │\n\
    ///          │    0    0    0    0    0    0    0    0    0 │\n\
    ///          │    0    0    0 1250    0    0 1250    0    0 │\n\
    ///          │    0    0    0    0    0    0    0    0    0 │\n\
    ///          │    0    0    0    0    0    0    0    0    0 │\n\
    ///          └                                              ┘"
    ///     );
    ///     Ok(())
    /// }
    /// ```
    pub fn new(young: f64, poisson: f64, plane_stress: bool) -> Result<Self, StrError> {
        if plane_stress && N != 4 {
            return Err("plane-stress assumption is only available for N = 4 (2D)");
        }
        let mut res = LinElasticity {
            young,
            poisson,
            plane_stress,
            dd: Tensor4::new(),
        };
        res.calc_rigidity();
        Ok(res)
    }

    /// Sets the Young's modulus and Poisson's coefficient
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_tensor::{LinElasticity, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let plane_stress = true;
    ///     let mut ela = LinElasticity::<4>::new(3000.0, 0.2, plane_stress)?;
    ///     ela.set_young_poisson(6000.0, 0.2);
    ///     let dd = ela.stiffness().as_std_matrix();
    ///     assert_eq!(
    ///         format!("{}", dd),
    ///         "┌                                              ┐\n\
    ///          │ 6250 1250    0    0    0    0    0    0    0 │\n\
    ///          │ 1250 6250    0    0    0    0    0    0    0 │\n\
    ///          │    0    0    0    0    0    0    0    0    0 │\n\
    ///          │    0    0    0 2500    0    0 2500    0    0 │\n\
    ///          │    0    0    0    0    0    0    0    0    0 │\n\
    ///          │    0    0    0    0    0    0    0    0    0 │\n\
    ///          │    0    0    0 2500    0    0 2500    0    0 │\n\
    ///          │    0    0    0    0    0    0    0    0    0 │\n\
    ///          │    0    0    0    0    0    0    0    0    0 │\n\
    ///          └                                              ┘"
    ///     );
    ///     Ok(())
    /// }
    /// ```
    pub fn set_young_poisson(&mut self, young: f64, poisson: f64) {
        self.young = young;
        self.poisson = poisson;
        self.calc_rigidity();
    }

    /// Sets the bulk (K) and shear (G) moduli
    pub fn set_bulk_shear(&mut self, bulk: f64, shear: f64) {
        self.young = 9.0 * bulk * shear / (3.0 * bulk + shear);
        self.poisson = (3.0 * bulk - 2.0 * shear) / (6.0 * bulk + 2.0 * shear);
        self.calc_rigidity();
    }

    /// Returns the Young's modulus and Poisson's coefficient
    ///
    /// Returns `(young, poisson)`
    pub fn get_young_poisson(&self) -> (f64, f64) {
        (self.young, self.poisson)
    }

    /// Returns the bulk (K) and shear (G) moduli
    ///
    /// Returns `(bulk, shear)`
    pub fn get_bulk_shear(&self) -> (f64, f64) {
        (
            self.young / (3.0 * (1.0 - 2.0 * self.poisson)),
            self.young / (2.0 * (1.0 + self.poisson)),
        )
    }

    /// Returns an access to the elastic stiffness tensor
    ///
    /// The rigidity modulus `D` is such that:
    ///
    /// ```text
    /// σ = D : ε
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_tensor::{LinElasticity, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let ela = LinElasticity::<4>::new(3000.0, 0.2, true)?;
    ///     let out = ela.stiffness().as_std_matrix();
    ///     assert_eq!(
    ///         format!("{}", out),
    ///         "┌                                              ┐\n\
    ///          │ 3125  625    0    0    0    0    0    0    0 │\n\
    ///          │  625 3125    0    0    0    0    0    0    0 │\n\
    ///          │    0    0    0    0    0    0    0    0    0 │\n\
    ///          │    0    0    0 1250    0    0 1250    0    0 │\n\
    ///          │    0    0    0    0    0    0    0    0    0 │\n\
    ///          │    0    0    0    0    0    0    0    0    0 │\n\
    ///          │    0    0    0 1250    0    0 1250    0    0 │\n\
    ///          │    0    0    0    0    0    0    0    0    0 │\n\
    ///          │    0    0    0    0    0    0    0    0    0 │\n\
    ///          └                                              ┘"
    ///     );
    ///     Ok(())
    /// }
    /// ```
    pub fn stiffness(&self) -> &Tensor4<N> {
        &self.dd
    }

    /// Calculates stress from strain
    ///
    /// ```text
    /// σ = D : ε
    /// ```
    ///
    /// # Output
    ///
    /// * `stress` -- the stress tensor σ
    ///
    /// # Input
    ///
    /// * `strain` -- the strain tensor ε
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_tensor::{LinElasticity, StrError, Tensor2};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     // define the strain matrix => will cause sum of rows of D
    ///     let strain_matrix_3d = &[
    ///         [1.0, 1.0, 1.0],
    ///         [1.0, 1.0, 1.0],
    ///         [1.0, 1.0, 1.0]
    ///     ];
    ///     let strain_matrix_2d = &[
    ///         [1.0, 1.0, 0.0],
    ///         [1.0, 1.0, 0.0],
    ///         [0.0, 0.0, 1.0]
    ///     ];
    ///
    ///     // 3D
    ///     // sum of first 3 rows = 1800
    ///     // sum of other rows = 720
    ///     let ela = LinElasticity::<6>::new(900.0, 0.25, false)?;
    ///     let out = ela.stiffness().as_std_matrix();
    ///     assert_eq!(
    ///         format!("{}", out),
    ///         "┌                                              ┐\n\
    ///          │ 1080  360  360    0    0    0    0    0    0 │\n\
    ///          │  360 1080  360    0    0    0    0    0    0 │\n\
    ///          │  360  360 1080    0    0    0    0    0    0 │\n\
    ///          │    0    0    0  360    0    0  360    0    0 │\n\
    ///          │    0    0    0    0  360    0    0  360    0 │\n\
    ///          │    0    0    0    0    0  360    0    0  360 │\n\
    ///          │    0    0    0  360    0    0  360    0    0 │\n\
    ///          │    0    0    0    0  360    0    0  360    0 │\n\
    ///          │    0    0    0    0    0  360    0    0  360 │\n\
    ///          └                                              ┘"
    ///     );
    ///     let strain = Tensor2::<6>::from_std_matrix(strain_matrix_3d)?;
    ///     let mut stress = Tensor2::<6>::new();
    ///     ela.calc_stress(&mut stress, &strain);
    ///     let out = stress.as_std_matrix();
    ///     assert_eq!(
    ///         format!("{:.0}", out),
    ///         "┌                ┐\n\
    ///          │ 1800  720  720 │\n\
    ///          │  720 1800  720 │\n\
    ///          │  720  720 1800 │\n\
    ///          └                ┘"
    ///     );
    ///
    ///     // 2D plane-strain
    ///     // sum of first 3 rows = 1800
    ///     // sum of other rows = 720
    ///     let ela = LinElasticity::<4>::new(900.0, 0.25, false)?;
    ///     let out = ela.stiffness().as_std_matrix();
    ///     println!("{}", out);
    ///     assert_eq!(
    ///         format!("{}", out),
    ///         "┌                                              ┐\n\
    ///          │ 1080  360  360    0    0    0    0    0    0 │\n\
    ///          │  360 1080  360    0    0    0    0    0    0 │\n\
    ///          │  360  360 1080    0    0    0    0    0    0 │\n\
    ///          │    0    0    0  360    0    0  360    0    0 │\n\
    ///          │    0    0    0    0    0    0    0    0    0 │\n\
    ///          │    0    0    0    0    0    0    0    0    0 │\n\
    ///          │    0    0    0  360    0    0  360    0    0 │\n\
    ///          │    0    0    0    0    0    0    0    0    0 │\n\
    ///          │    0    0    0    0    0    0    0    0    0 │\n\
    ///          └                                              ┘"
    ///     );
    ///     let strain = Tensor2::<4>::from_std_matrix(strain_matrix_2d)?;
    ///     let mut stress = Tensor2::<4>::new();
    ///     ela.calc_stress(&mut stress, &strain);
    ///     let out = stress.as_std_matrix();
    ///     assert_eq!(
    ///         format!("{:.0}", out),
    ///         "┌                ┐\n\
    ///          │ 1800  720    0 │\n\
    ///          │  720 1800    0 │\n\
    ///          │    0    0 1800 │\n\
    ///          └                ┘"
    ///     );
    ///     Ok(())
    /// }
    /// ```
    pub fn calc_stress(&self, stress: &mut Tensor2<N>, strain: &Tensor2<N>) {
        t4_ddot_t2(stress, SET, 1.0, &self.dd, strain);
    }

    /// Calculates and sets the out-of-plane strain in the Plane-Stress case
    ///
    /// # Input
    ///
    /// * `stress` -- the stress tensor σ
    ///
    /// # Output
    ///
    /// * Returns the `εzz` (out-of-plane) component
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_tensor::{LinElasticity, StrError, Tensor2};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let young = 2500.0;
    ///     let poisson = 0.25;
    ///     let ela = LinElasticity::<4>::new(young, poisson, true)?;
    ///     let (sig_xx, sig_yy) = (2000.0, 1000.0);
    ///     let stress = Tensor2::<4>::from_std_matrix(&[
    ///         [sig_xx,     0.0, 0.0],
    ///         [   0.0,  sig_yy, 0.0],
    ///         [   0.0,     0.0, 0.0],
    ///     ])?;
    ///     let eps_zz = ela.out_of_plane_strain(&stress)?;
    ///     let eps_zz_correct = -(poisson / young) * (sig_xx + sig_yy);
    ///     assert_eq!(eps_zz, eps_zz);
    ///     Ok(())
    /// }
    /// ```
    pub fn out_of_plane_strain(&self, stress: &Tensor2<N>) -> Result<f64, StrError> {
        if !self.plane_stress {
            return Err("out-of-plane strain works with plane-stress only");
        }
        let eps_zz = -(stress.get(0) + stress.get(1)) * self.poisson / self.young;
        Ok(eps_zz)
    }

    /// Calculates the elastic compliance modulus
    ///
    /// **Note:** The compliance modulus is not available for plane-stress.
    ///
    /// The Compliance modulus `C` is such that:
    ///
    /// ```text
    /// ε = C : σ
    /// ```
    ///
    /// The compliance modulus is calculate as `C = D⁻¹`
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::mat_approx_eq;
    /// use russell_tensor::*;
    ///
    /// fn main() -> Result<(), StrError> {
    ///     // calculate C
    ///     let ela = LinElasticity::<6>::new(900.0, 0.25, false)?;
    ///     let mut cc = Tensor4::<6>::new();
    ///     ela.calc_compliance(&mut cc).unwrap();
    ///
    ///     // check
    ///     let (kk, gg) = ela.get_bulk_shear();
    ///     let psd = Tensor4::<6>::constant_pp_symdev();
    ///     let piso = Tensor4::<6>::constant_pp_iso();
    ///     let mut correct = Tensor4::<6>::new();
    ///     t4_add(&mut correct, 1.0 / (3.0 * kk), &piso, 1.0 / (2.0 * gg), &psd);
    ///     mat_approx_eq(&cc.as_std_matrix(), &correct.as_std_matrix(), 1e-15);
    ///     Ok(())
    /// }
    /// ```
    pub fn calc_compliance(&self, cc: &mut Tensor4<N>) -> Result<(), StrError> {
        if self.plane_stress {
            return Err("The compliance modulus is not available for plane-stress");
        }
        self.dd
            .inverse(cc)
            .map_err(|_| "cannot invert the rigidity modulus D")?;
        Ok(())
    }

    /// Calculates the rigidity modulus
    fn calc_rigidity(&mut self) {
        if self.plane_stress {
            let c = self.young / (1.0 - self.poisson * self.poisson);
            self.dd.set(0, 0, c);
            self.dd.set(0, 1, c * self.poisson);
            self.dd.set(1, 0, c * self.poisson);
            self.dd.set(1, 1, c);
            self.dd.set(3, 3, c * (1.0 - self.poisson)); // Rep: multiply by 2, so 1/2 disappears
        } else {
            let c = self.young / ((1.0 + self.poisson) * (1.0 - 2.0 * self.poisson));
            self.dd.set(0, 0, c * (1.0 - self.poisson));
            self.dd.set(0, 1, c * self.poisson);
            self.dd.set(0, 2, c * self.poisson);
            self.dd.set(1, 0, c * self.poisson);
            self.dd.set(1, 1, c * (1.0 - self.poisson));
            self.dd.set(1, 2, c * self.poisson);
            self.dd.set(2, 0, c * self.poisson);
            self.dd.set(2, 1, c * self.poisson);
            self.dd.set(2, 2, c * (1.0 - self.poisson));
            self.dd.set(3, 3, c * (1.0 - 2.0 * self.poisson)); // Rep: multiply by 2, so 1/2 disappears
        }
        if N > 4 {
            let g = self.dd.get(3, 3);
            self.dd.set(4, 4, g);
            self.dd.set(5, 5, g);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::LinElasticity;
    use crate::StrError;
    use crate::{Tensor2, Tensor4, t4_add};
    use russell_lab::{Matrix, approx_eq, mat_approx_eq};

    // Checks the symmetry of a square matrix
    fn check_symmetry(mat: &Matrix) -> Result<(), StrError> {
        let (nrow, ncol) = mat.dims();
        assert_eq!(nrow, ncol, "matrix is not square");
        for l in 0..nrow {
            for ll in (l + 1)..nrow {
                assert_eq!(mat.get(l, ll), mat.get(ll, l));
            }
        }
        Ok(())
    }

    #[test]
    fn new_works() {
        // plane-stress
        // from Bhatti page 511 (Young divided by 1000)
        let ela = LinElasticity::<4>::new(3000.0, 0.2, true).unwrap();
        let out = ela.stiffness().as_std_matrix();
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
        check_symmetry(&out).unwrap();

        // plane-strain
        // from Bhatti page 519
        let ela = LinElasticity::<4>::new(30000.0, 0.3, false).unwrap();
        let out = ela.stiffness().as_std_matrix();
        assert_eq!(
            format!("{:.1}", out),
            "┌                                                                         ┐\n\
             │ 40384.6 17307.7 17307.7     0.0     0.0     0.0     0.0     0.0     0.0 │\n\
             │ 17307.7 40384.6 17307.7     0.0     0.0     0.0     0.0     0.0     0.0 │\n\
             │ 17307.7 17307.7 40384.6     0.0     0.0     0.0     0.0     0.0     0.0 │\n\
             │     0.0     0.0     0.0 11538.5     0.0     0.0 11538.5     0.0     0.0 │\n\
             │     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0 │\n\
             │     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0 │\n\
             │     0.0     0.0     0.0 11538.5     0.0     0.0 11538.5     0.0     0.0 │\n\
             │     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0 │\n\
             │     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0 │\n\
             └                                                                         ┘"
        );
        check_symmetry(&out).unwrap();
    }

    #[test]
    fn set_get_parameters_works() {
        let mut ela = LinElasticity::<4>::new(3000.0, 0.2, true).unwrap();
        ela.set_young_poisson(6000.0, 0.2);
        assert_eq!(ela.stiffness().get(0, 0), 6250.0);

        let mut ela = LinElasticity::<6>::new(3000.0, 0.2, false).unwrap();
        ela.set_bulk_shear(1000.0, 600.0);
        assert_eq!(ela.young, 1500.0);
        assert_eq!(ela.poisson, 0.25);
        assert_eq!(ela.stiffness().get(0, 0), 1800.0);
        assert_eq!(ela.stiffness().get(0, 1), 600.0);
        let c = ela.young / ((1.0 + ela.poisson) * (1.0 - 2.0 * ela.poisson));
        assert_eq!(ela.stiffness().get(0, 0), (1.0 - ela.poisson) * c);
        assert_eq!(ela.stiffness().get(0, 1), ela.poisson * c);

        let mut ela = LinElasticity::<6>::new(3000.0, 0.2, false).unwrap();
        ela.set_young_poisson(1500.0, 0.25);
        assert_eq!(ela.get_young_poisson(), (1500.0, 0.25));
        assert_eq!(ela.get_bulk_shear(), (1000.0, 600.0));
        assert_eq!(ela.stiffness().get(0, 0), 1800.0);
        assert_eq!(ela.stiffness().get(0, 1), 600.0);
    }

    #[test]
    fn stiffness_works() {
        let ela = LinElasticity::<4>::new(3000.0, 0.2, true).unwrap();
        let dd = ela.stiffness();
        assert_eq!(dd.get(0, 0), 3125.0);
        check_symmetry(&dd.as_std_matrix()).unwrap();
    }

    #[test]
    fn calc_stress_works() {
        // plane-stress
        // from Bhatti page 514 (Young divided by 1000)
        let ela = LinElasticity::<4>::new(3000.0, 0.2, true).unwrap();
        #[rustfmt::skip]
        let strain = Tensor2::<4>::from_std_matrix(&[
            [-0.036760, 0.0667910,       0.0],
            [ 0.066791, 0.0164861,       0.0],
            [      0.0,       0.0, 0.0050847],
        ]).unwrap();
        let mut stress = Tensor2::<4>::new();
        ela.calc_stress(&mut stress, &strain);
        let out = stress.as_std_matrix();
        assert_eq!(
            format!("{:.3}", out),
            "┌                            ┐\n\
             │ -104.571  166.977    0.000 │\n\
             │  166.977   28.544    0.000 │\n\
             │    0.000    0.000    0.000 │\n\
             └                            ┘"
        );

        // plane-strain
        // from Bhatti page 523
        let ela = LinElasticity::<4>::new(30000.0, 0.3, false).unwrap();
        #[rustfmt::skip]
        let strain = Tensor2::<4>::from_std_matrix(&[
            [    3.6836e-6, -2.675290e-4, 0.0],
            [ -2.675290e-4,    3.6836e-6, 0.0],
            [          0.0,          0.0, 0.0],
        ]).unwrap();
        let mut stress = Tensor2::<4>::new();
        ela.calc_stress(&mut stress, &strain);
        let out = stress.as_std_matrix();
        assert_eq!(
            format!("{:.6}", out),
            "┌                               ┐\n\
             │  0.212515 -6.173746  0.000000 │\n\
             │ -6.173746  0.212515  0.000000 │\n\
             │  0.000000  0.000000  0.127509 │\n\
             └                               ┘"
        );

        // 3D
        // sum of first 3 rows = 1800
        // sum of other rows = 720
        let ela = LinElasticity::<6>::new(900.0, 0.25, false).unwrap();
        let out = ela.stiffness().as_std_matrix();
        assert_eq!(
            format!("{}", out),
            "┌                                              ┐\n\
             │ 1080  360  360    0    0    0    0    0    0 │\n\
             │  360 1080  360    0    0    0    0    0    0 │\n\
             │  360  360 1080    0    0    0    0    0    0 │\n\
             │    0    0    0  360    0    0  360    0    0 │\n\
             │    0    0    0    0  360    0    0  360    0 │\n\
             │    0    0    0    0    0  360    0    0  360 │\n\
             │    0    0    0  360    0    0  360    0    0 │\n\
             │    0    0    0    0  360    0    0  360    0 │\n\
             │    0    0    0    0    0  360    0    0  360 │\n\
             └                                              ┘"
        );
        #[rustfmt::skip]
        let strain = Tensor2::<6>::from_std_matrix(&[
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]).unwrap();
        let mut stress = Tensor2::<6>::new();
        ela.calc_stress(&mut stress, &strain);
        let out = stress.as_std_matrix();
        assert_eq!(
            format!("{:.0}", out),
            "┌                ┐\n\
             │ 1800  720  720 │\n\
             │  720 1800  720 │\n\
             │  720  720 1800 │\n\
             └                ┘"
        );
    }

    #[test]
    fn out_of_plane_strain_fails_on_wrong_input() {
        let ela = LinElasticity::<4>::new(900.0, 0.25, false).unwrap();
        #[rustfmt::skip]
        let stress = Tensor2::<4>::from_std_matrix(&[
            [100.0,   0.0, 0.0],
            [  0.0, 100.0, 0.0],
            [  0.0,   0.0, 0.0],
        ]).unwrap();
        let res = ela.out_of_plane_strain(&stress);
        assert_eq!(res.err(), Some("out-of-plane strain works with plane-stress only"));
    }

    #[test]
    fn out_of_plane_strain_works() {
        let ela = LinElasticity::<4>::new(3000.0, 0.2, true).unwrap();
        #[rustfmt::skip]
        let stress = Tensor2::from_std_matrix(&[
            [-104.571, 166.977, 0.0],
            [ 166.977,  28.544, 0.0],
            [   0.0,     0.0,   0.0],
        ]).unwrap();
        let eps_zz = ela.out_of_plane_strain(&stress).unwrap();
        approx_eq(eps_zz, 0.0050847, 1e-4);
    }

    #[test]
    fn calc_compliance_modulus_handles_errors() {
        let ela = LinElasticity::<4>::new(900.0, 0.25, true).unwrap(); // plane-stress
        let mut cc = Tensor4::<4>::new();
        assert_eq!(
            ela.calc_compliance(&mut cc).err(),
            Some("The compliance modulus is not available for plane-stress")
        );
        let ela = LinElasticity::<4>::new(0.0, 0.0, false).unwrap(); // zero values (indeterminate)
        assert_eq!(
            ela.calc_compliance(&mut cc).err(),
            Some("cannot invert the rigidity modulus D")
        );
    }

    #[test]
    fn compliance_modulus_works() {
        // calculate C
        let mut ela = LinElasticity::<6>::new(900.0, 0.25, false).unwrap();
        let mut cc = Tensor4::<6>::new();
        ela.calc_compliance(&mut cc).unwrap();

        // check
        let (kk, gg) = ela.get_bulk_shear();
        let psd = Tensor4::<6>::constant_pp_symdev();
        let piso = Tensor4::<6>::constant_pp_iso();
        let mut correct = Tensor4::<6>::new();
        t4_add(&mut correct, 1.0 / (3.0 * kk), &piso, 1.0 / (2.0 * gg), &psd);
        mat_approx_eq(&cc.as_std_matrix(), &correct.as_std_matrix(), 1e-15);

        // change parameters
        let (kk, gg) = (1.0 / 6.0, 1.0 / 4.0);
        ela.set_bulk_shear(kk, gg);
        ela.calc_compliance(&mut cc).unwrap();

        // check again
        t4_add(&mut correct, 1.0 / (3.0 * kk), &piso, 1.0 / (2.0 * gg), &psd);
        // println!("{}", cc.as_std_matrix());
        mat_approx_eq(&cc.as_std_matrix(), &correct.as_std_matrix(), 1e-15);
    }
}
