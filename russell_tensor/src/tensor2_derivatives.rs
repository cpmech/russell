use crate::{Mandel, StrError, Tensor2, ONE_BY_3, SQRT_3, SQRT_3_BY_2, SQRT_6, TWO_BY_3};
use russell_lab::vec_add;

impl Tensor2 {
    /// Computes the first derivative of the norm w.r.t. the defining tensor
    ///
    /// ```text
    /// d‖σ‖    σ
    /// ──── = ───
    ///  dσ    ‖σ‖
    /// ```
    ///
    /// # Output
    ///
    /// * This function returns `Some(‖σ‖)` if ‖σ‖ > 0 and the computation was successful
    /// * Otherwise, this function returns `None` and the derivative cannot be computed
    ///   because the norm is zero
    pub fn deriv1_norm(&self, d1: &mut Tensor2) -> Result<Option<f64>, StrError> {
        let n = self.norm();
        if n > 0.0 {
            d1.mirror(self)?;
            for i in 0..d1.vec.dim() {
                d1.vec[i] /= n;
            }
            Ok(Some(n))
        } else {
            Ok(None)
        }
    }

    /// Computes the first derivative of the J2 invariant w.r.t. the defining tensor
    ///
    /// ```text
    /// s = deviator(σ)
    ///
    /// dJ2            dJ2
    /// ─── = sᵀ  or   ─── = s (if σ is symmetric)
    ///  dσ             dσ
    /// ```
    pub fn deriv1_invariant_jj2(&self, d1: &mut Tensor2) -> Result<(), StrError> {
        self.deviator(d1)?;
        if self.vec.dim() > 6 {
            // transpose
            d1.vec[6] *= -1.0;
            d1.vec[7] *= -1.0;
            d1.vec[8] *= -1.0;
        }
        Ok(())
    }

    /// Computes the first derivative of the J3 invariant w.r.t. the defining tensor
    ///
    /// ```text
    /// s = deviator(σ)
    ///
    /// dJ3            2 J2
    /// ─── = (s·s)ᵀ - ──── I
    ///  dσ              3
    ///
    /// or
    ///
    /// dJ3         2 J2
    /// ─── = s·s - ──── I (if σ is symmetric)
    ///  dσ           3
    /// ```
    pub fn deriv1_invariant_jj3(&self, d1: &mut Tensor2, s: &mut Tensor2) -> Result<(), StrError> {
        self.deviator(s)?;
        s.squared(d1)?; // d1 := s·s
        let jj2 = self.invariant_jj2();
        d1.vec[0] -= TWO_BY_3 * jj2;
        d1.vec[1] -= TWO_BY_3 * jj2;
        d1.vec[2] -= TWO_BY_3 * jj2;
        if self.vec.dim() > 6 {
            // transpose
            d1.vec[6] *= -1.0;
            d1.vec[7] *= -1.0;
            d1.vec[8] *= -1.0;
        }
        Ok(())
    }

    /// Computes the first derivative of the mean pressure invariant w.r.t. the defining tensor
    ///
    /// ```text
    /// dσm   1
    /// ─── = ─ I
    ///  dσ   3
    /// ```
    pub fn deriv1_invariant_sigma_m(&self, d1: &mut Tensor2) -> Result<(), StrError> {
        if d1.vec.dim() != self.vec.dim() {
            return Err("tensors are incompatible");
        }
        d1.clear();
        d1.vec[0] = ONE_BY_3;
        d1.vec[1] = ONE_BY_3;
        d1.vec[2] = ONE_BY_3;
        Ok(())
    }

    /// Computes the first derivative of the deviatoric stress invariant (von Mises) w.r.t. the defining tensor
    ///
    /// ```text
    /// dσd   √3  s
    /// ─── = ── ───
    /// dσ    √2 ‖s‖
    /// ```
    ///
    /// # Panics
    ///
    /// This function only makes sense for **symmetric** tensors. Otherwise, it will panic.
    ///
    /// # Output
    ///
    /// * This function returns `Some(‖s‖)` if `‖s‖ > 0` and the computation was successful
    /// * Otherwise, this function returns `None` and the derivative cannot be computed
    ///   because the deviatoric stress invariant is zero
    pub fn deriv1_invariant_sigma_d(&self, d1: &mut Tensor2) -> Result<Option<f64>, StrError> {
        assert!(Mandel::is_symmetric(self.vec.dim()));
        let n = self.deviator_norm();
        if n > 0.0 {
            self.deviator(d1)?;
            for i in 0..d1.vec.dim() {
                d1.vec[i] *= SQRT_3_BY_2 / n;
            }
            Ok(Some(n))
        } else {
            Ok(None)
        }
    }

    /// Computes the first derivative of the Lode invariant
    ///
    /// ```text
    /// σ represents this tensor
    /// l is the Lode invariant
    ///
    /// s = dev(σ)
    ///
    /// dl       3 √3       dJ3      9 √3 J3    dJ2
    /// ── = ─────────────  ─── - ───────────── ───
    /// dσ   2 pow(J2,1.5)  dσ    4 pow(J2,2.5) dσ
    /// ```
    ///
    /// # Panics
    ///
    /// This function only makes sense for **symmetric** tensors. Otherwise, it will panic.
    ///
    /// # Returns
    ///
    /// If `J2 > tol_jj2`, returns `J2` and the derivative in `d1`. Otherwise, returns None.
    pub fn deriv1_invariant_lode(
        &self,
        d1: &mut Tensor2,
        aux: &mut Tensor2,
        tol_jj2: f64,
    ) -> Result<Option<f64>, StrError> {
        let ndim = d1.vec.dim();
        assert!(Mandel::is_symmetric(ndim));
        if d1.vec.dim() != ndim || aux.vec.dim() != ndim {
            return Err("tensors are incompatible");
        }
        let jj2 = self.invariant_jj2();
        if jj2 > tol_jj2 {
            self.deriv1_invariant_jj3(d1, aux)?; // d1 := dJ3/dσ
            self.deriv1_invariant_jj2(aux)?; // aux := dJ2/dσ
            let jj3 = self.invariant_jj3();
            let a = 1.5 * SQRT_3 / f64::powf(jj2, 1.5);
            let b = 2.25 * SQRT_3 * jj3 / f64::powf(jj2, 2.5);
            for i in 0..ndim {
                d1.vec[i] = a * d1.vec[i] - b * aux.vec[i];
            }
            return Ok(Some(jj2));
        }
        Ok(None)
    }

    /// Computes the first derivative of the Lode invariant (alternative method)
    ///
    /// ```text
    /// σ represents this tensor
    /// l is the Lode invariant
    ///
    /// s = dev(σ)
    ///
    /// ψ = dev(s⁻¹)
    ///
    /// m = 3 l / ‖s‖²
    ///
    /// dl
    /// ── = l ψ - m s
    /// dσ
    /// ```
    ///
    /// # Panics
    ///
    /// This function only makes sense for **symmetric** tensors. Otherwise, it will panic.
    ///
    /// # Output
    ///
    /// * `d1` -- d1 = dl/dσ is the first derivative of the Lode invariant
    /// * `s` -- s = dev(σ) is the deviator of σ (this tensor)
    /// * `si` -- si = inverse(s) is the inverse of the deviator tensor
    /// * `psi` -- psi = dev(inverse(s)) is the deviator of the inverse of the deviator tensor
    /// * `tolerance` -- is a tolerance to compute the determinant of the deviator tensor
    ///
    /// # Returns
    ///
    /// If the norm and determinant of `s` (deviator) are not null, returns the derivative in `d1`
    /// and the Lode invariant `l`. Otherwise, returns None.
    pub fn deriv1_invariant_lode_alt(
        &self,
        d1: &mut Tensor2,
        s: &mut Tensor2,
        si: &mut Tensor2,
        psi: &mut Tensor2,
        tolerance: f64,
    ) -> Result<Option<f64>, StrError> {
        assert!(Mandel::is_symmetric(self.vec.dim()));
        let n = self.deviator_norm();
        let nnn = n * n * n;
        if f64::abs(nnn) > 0.0 {
            self.deviator(s)?;
            if let Some(det) = s.inverse(si, tolerance)? {
                si.deviator(psi)?;
                let l = 3.0 * SQRT_6 * det / nnn;
                let m = 3.0 * l / (n * n);
                vec_add(&mut d1.vec, l, &psi.vec, -m, &s.vec)?;
                return Ok(Some(l));
            }
        }
        Ok(None)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Tensor2;
    use crate::{Mandel, SampleTensor2, SamplesTensor2, ONE_BY_3, SQRT_3_BY_2};
    use russell_chk::{approx_eq, deriv_central5, vec_approx_eq};
    use russell_lab::{mat_approx_eq, Matrix};

    // -- deriv1_norm ---------------------------------------------------------------------------------------

    // Holds arguments for numerical differentiation of a scalar f(σ) w.r.t. σₘ with m being the Mandel index
    struct ArgsNumDeriv1 {
        at_sigma: Tensor2,   // @ σ value
        temp_sigma: Tensor2, // temporary σ
        m: usize,            // index i of ∂f/∂σₘ
    }

    #[test]
    fn deriv1_norm_captures_errors() {
        let sigma = Tensor2::from_matrix(&SamplesTensor2::TENSOR_I.matrix, Mandel::General).unwrap();
        let mut d1 = Tensor2::new(Mandel::Symmetric);
        assert_eq!(sigma.deriv1_norm(&mut d1).err(), Some("tensors are incompatible"));
    }

    #[test]
    fn deriv1_norm_handles_indeterminate_case() {
        let sigma = Tensor2::from_matrix(&SamplesTensor2::TENSOR_O.matrix, Mandel::General).unwrap();
        let mut d1 = Tensor2::new(Mandel::General);
        assert_eq!(sigma.deriv1_norm(&mut d1), Ok(None));
    }

    // Computes ‖σ‖ for varying v_mandel := MandelComponent(σᵢⱼ)
    fn norm_given_sigma_mandel(v_mandel: f64, args: &mut ArgsNumDeriv1) -> f64 {
        args.temp_sigma.mirror(&args.at_sigma).unwrap();
        args.temp_sigma.vec[args.m] = v_mandel;
        args.temp_sigma.norm()
    }

    // Checks first the derivative of ‖σ‖ w.r.t. σ
    fn check_deriv1_norm(case: Mandel, sample: &SampleTensor2, tol: f64, tol_num: f64, verbose: bool) {
        // compare with correct solution
        let mat = sample.matrix;
        let norm = sample.norm;
        let correct = Matrix::from(&[
            [mat[0][0] / norm, mat[0][1] / norm, mat[0][2] / norm],
            [mat[1][0] / norm, mat[1][1] / norm, mat[1][2] / norm],
            [mat[2][0] / norm, mat[2][1] / norm, mat[2][2] / norm],
        ]);
        let sigma = Tensor2::from_matrix(&sample.matrix, case).unwrap();
        let mut ana_deriv = Tensor2::new(case);
        sigma.deriv1_norm(&mut ana_deriv).unwrap();
        if verbose {
            println!("analytical d‖σ‖/dσ =\n{}", ana_deriv.to_matrix());
            println!("correct d‖σ‖/dσ =\n{}", correct);
        }
        mat_approx_eq(&ana_deriv.to_matrix(), &correct, tol);

        // compare with numerical derivative
        let mut args = ArgsNumDeriv1 {
            at_sigma: Tensor2::from_matrix(&sample.matrix, case).unwrap(),
            temp_sigma: Tensor2::new(case),
            m: 0,
        };
        let mut num_deriv = Tensor2::new(case);
        for m in 0..ana_deriv.vec.dim() {
            args.m = m;
            let res = deriv_central5(args.at_sigma.vec[m], &mut args, norm_given_sigma_mandel);
            num_deriv.vec[m] = res;
        }
        if verbose {
            println!("numerical d‖σ‖/dσ =\n{}", num_deriv.to_matrix());
        }
        vec_approx_eq(ana_deriv.vec.as_data(), num_deriv.vec.as_data(), tol_num);
    }

    #[test]
    fn deriv1_norm_works() {
        check_deriv1_norm(Mandel::General, &SamplesTensor2::TENSOR_T, 1e-15, 1e-10, false);
        check_deriv1_norm(Mandel::Symmetric, &SamplesTensor2::TENSOR_S, 1e-15, 1e-10, false);
        check_deriv1_norm(Mandel::Symmetric2D, &SamplesTensor2::TENSOR_Z, 1e-15, 1e-10, false);
    }

    // -- deriv1_invariant_jj2 ------------------------------------------------------------------------------

    #[test]
    fn deriv1_invariant_jj2_captures_errors() {
        let sigma = Tensor2::from_matrix(&SamplesTensor2::TENSOR_I.matrix, Mandel::General).unwrap();
        let mut d1 = Tensor2::new(Mandel::Symmetric);
        assert_eq!(
            sigma.deriv1_invariant_jj2(&mut d1).err(),
            Some("tensors are incompatible")
        );
    }

    // Computes J2 for varying v_mandel := MandelComponent(σᵢⱼ)
    fn jj2_given_sigma_mandel(v_mandel: f64, args: &mut ArgsNumDeriv1) -> f64 {
        args.temp_sigma.mirror(&args.at_sigma).unwrap();
        args.temp_sigma.vec[args.m] = v_mandel;
        args.temp_sigma.invariant_jj2()
    }

    // Checks the first derivative of J2 w.r.t. σ
    fn check_deriv1_jj2(case: Mandel, sample: &SampleTensor2, tol_num: f64, verbose: bool) {
        // analytical derivative
        let sigma = Tensor2::from_matrix(&sample.matrix, case).unwrap();
        let mut ana_deriv = Tensor2::new(case);
        sigma.deriv1_invariant_jj2(&mut ana_deriv).unwrap();
        if verbose {
            println!("analytical dJ2/dσ =\n{}", ana_deriv.to_matrix());
        }

        // compare with numerical derivative
        let mut args = ArgsNumDeriv1 {
            at_sigma: Tensor2::from_matrix(&sample.matrix, case).unwrap(),
            temp_sigma: Tensor2::new(case),
            m: 0,
        };
        let mut num_deriv = Tensor2::new(case);
        for m in 0..ana_deriv.vec.dim() {
            args.m = m;
            let res = deriv_central5(args.at_sigma.vec[m], &mut args, jj2_given_sigma_mandel);
            num_deriv.vec[m] = res;
        }
        if verbose {
            println!("numerical dJ2/dσ =\n{}", num_deriv.to_matrix());
        }
        vec_approx_eq(ana_deriv.vec.as_data(), num_deriv.vec.as_data(), tol_num);
    }

    #[test]
    fn deriv1_invariant_jj2_works() {
        check_deriv1_jj2(Mandel::General, &SamplesTensor2::TENSOR_T, 1e-10, false);
        check_deriv1_jj2(Mandel::Symmetric, &SamplesTensor2::TENSOR_S, 1e-11, false);
        check_deriv1_jj2(Mandel::Symmetric2D, &SamplesTensor2::TENSOR_Z, 1e-12, false);
        check_deriv1_jj2(Mandel::Symmetric2D, &SamplesTensor2::TENSOR_O, 1e-15, false);
        check_deriv1_jj2(Mandel::Symmetric2D, &SamplesTensor2::TENSOR_I, 1e-12, false);
    }

    // -- deriv1_invariant_jj3 ------------------------------------------------------------------------------

    #[test]
    fn deriv1_invariant_jj3_captures_errors() {
        let sigma = Tensor2::from_matrix(&SamplesTensor2::TENSOR_I.matrix, Mandel::General).unwrap();
        let mut d1 = Tensor2::new(Mandel::Symmetric);
        let mut s = Tensor2::new(Mandel::General);
        assert_eq!(
            sigma.deriv1_invariant_jj3(&mut d1, &mut s).err(),
            Some("tensors are incompatible")
        );
        let mut d1 = Tensor2::new(Mandel::General);
        let mut s = Tensor2::new(Mandel::Symmetric);
        assert_eq!(
            sigma.deriv1_invariant_jj3(&mut d1, &mut s).err(),
            Some("tensors are incompatible")
        );
    }

    // Computes J3 for varying v_mandel := MandelComponent(σᵢⱼ)
    fn jj3_given_sigma_mandel(v_mandel: f64, args: &mut ArgsNumDeriv1) -> f64 {
        args.temp_sigma.mirror(&args.at_sigma).unwrap();
        args.temp_sigma.vec[args.m] = v_mandel;
        args.temp_sigma.invariant_jj3()
    }

    // Checks the first derivative of J3 w.r.t. σ
    fn check_deriv1_jj3(case: Mandel, sample: &SampleTensor2, tol_num: f64, verbose: bool) {
        // analytical derivative
        let sigma = Tensor2::from_matrix(&sample.matrix, case).unwrap();
        let mut s = Tensor2::new(case);
        let mut ana_deriv = Tensor2::new(case);
        sigma.deriv1_invariant_jj3(&mut ana_deriv, &mut s).unwrap();
        if verbose {
            println!("analytical dJ3/dσ =\n{}", ana_deriv.to_matrix());
        }

        // compare with numerical derivative
        let mut args = ArgsNumDeriv1 {
            at_sigma: Tensor2::from_matrix(&sample.matrix, case).unwrap(),
            temp_sigma: Tensor2::new(case),
            m: 0,
        };
        let mut num_deriv = Tensor2::new(case);
        for m in 0..ana_deriv.vec.dim() {
            args.m = m;
            let res = deriv_central5(args.at_sigma.vec[m], &mut args, jj3_given_sigma_mandel);
            num_deriv.vec[m] = res;
        }
        if verbose {
            println!("numerical dJ3/dσ =\n{}", num_deriv.to_matrix());
        }
        vec_approx_eq(ana_deriv.vec.as_data(), num_deriv.vec.as_data(), tol_num);
    }

    #[test]
    fn deriv1_invariant_jj3_works() {
        check_deriv1_jj3(Mandel::General, &SamplesTensor2::TENSOR_T, 1e-8, false);
        check_deriv1_jj3(Mandel::Symmetric, &SamplesTensor2::TENSOR_S, 1e-9, false);
        check_deriv1_jj3(Mandel::Symmetric2D, &SamplesTensor2::TENSOR_Z, 1e-10, false);
        check_deriv1_jj3(Mandel::Symmetric2D, &SamplesTensor2::TENSOR_O, 1e-15, false);
        check_deriv1_jj3(Mandel::Symmetric2D, &SamplesTensor2::TENSOR_I, 1e-15, false);
    }

    // -- deriv1_invariant_sigma_m --------------------------------------------------------------------------

    #[test]
    fn deriv1_invariant_sigma_m_captures_errors() {
        let sigma = Tensor2::from_matrix(&SamplesTensor2::TENSOR_I.matrix, Mandel::General).unwrap();
        let mut d1 = Tensor2::new(Mandel::Symmetric);
        assert_eq!(
            sigma.deriv1_invariant_sigma_m(&mut d1).err(),
            Some("tensors are incompatible")
        );
    }

    // Computes σm for varying v_mandel := MandelComponent(σᵢⱼ)
    fn sigma_m_given_sigma_mandel(v_mandel: f64, args: &mut ArgsNumDeriv1) -> f64 {
        args.temp_sigma.mirror(&args.at_sigma).unwrap();
        args.temp_sigma.vec[args.m] = v_mandel;
        args.temp_sigma.invariant_sigma_m()
    }

    // Checks the first derivative of σm w.r.t. σ
    fn check_deriv1_sigma_m(case: Mandel, sample: &SampleTensor2, tol: f64, tol_num: f64, verbose: bool) {
        // compare with correct solution
        let correct = Matrix::from(&[[ONE_BY_3, 0.0, 0.0], [0.0, ONE_BY_3, 0.0], [0.0, 0.0, ONE_BY_3]]);
        let sigma = Tensor2::from_matrix(&sample.matrix, case).unwrap();
        let mut ana_deriv = Tensor2::new(case);
        sigma.deriv1_invariant_sigma_m(&mut ana_deriv).unwrap();
        if verbose {
            println!("analytical dσm/dσ =\n{}", ana_deriv.to_matrix());
            println!("correct dσm/dσ =\n{}", correct);
        }
        mat_approx_eq(&ana_deriv.to_matrix(), &correct, tol);

        // compare with numerical derivative
        let mut args = ArgsNumDeriv1 {
            at_sigma: Tensor2::from_matrix(&sample.matrix, case).unwrap(),
            temp_sigma: Tensor2::new(case),
            m: 0,
        };
        let mut num_deriv = Tensor2::new(case);
        for m in 0..ana_deriv.vec.dim() {
            args.m = m;
            let res = deriv_central5(args.at_sigma.vec[m], &mut args, sigma_m_given_sigma_mandel);
            num_deriv.vec[m] = res;
        }
        if verbose {
            println!("numerical dσm/dσ =\n{}", num_deriv.to_matrix());
        }
        vec_approx_eq(ana_deriv.vec.as_data(), num_deriv.vec.as_data(), tol_num);
    }

    #[test]
    fn deriv1_sigma_m_works() {
        check_deriv1_sigma_m(Mandel::General, &SamplesTensor2::TENSOR_T, 1e-15, 1e-12, false);
        check_deriv1_sigma_m(Mandel::Symmetric, &SamplesTensor2::TENSOR_S, 1e-15, 1e-11, false);
        check_deriv1_sigma_m(Mandel::Symmetric2D, &SamplesTensor2::TENSOR_Z, 1e-15, 1e-12, false);
    }

    // -- deriv1_invariant_sigma_d --------------------------------------------------------------------------

    #[test]
    fn deriv1_invariant_sigma_d_captures_errors() {
        let sigma = Tensor2::from_matrix(&SamplesTensor2::TENSOR_S.matrix, Mandel::Symmetric).unwrap();
        let mut d1 = Tensor2::new(Mandel::Symmetric2D);
        assert_eq!(
            sigma.deriv1_invariant_sigma_d(&mut d1).err(),
            Some("tensors are incompatible")
        );
    }

    #[test]
    fn deriv1_invariant_sigma_d_handles_indeterminate_case() {
        let sigma = Tensor2::from_matrix(&SamplesTensor2::TENSOR_O.matrix, Mandel::Symmetric).unwrap();
        let mut d1 = Tensor2::new(Mandel::Symmetric);
        assert_eq!(sigma.deriv1_invariant_sigma_d(&mut d1), Ok(None));
    }

    // Computes σd for varying v_mandel := MandelComponent(σᵢⱼ)
    fn sigma_d_given_sigma_mandel(v_mandel: f64, args: &mut ArgsNumDeriv1) -> f64 {
        args.temp_sigma.mirror(&args.at_sigma).unwrap();
        args.temp_sigma.vec[args.m] = v_mandel;
        args.temp_sigma.invariant_sigma_d()
    }

    // Checks the first derivative of σd w.r.t. σ
    fn check_deriv1_sigma_d(case: Mandel, sample: &SampleTensor2, tol: f64, tol_num: f64, verbose: bool) {
        // compare with correct solution
        let mut correct = Matrix::from(&sample.deviator);
        for i in 0..3 {
            for j in 0..3 {
                correct.set(i, j, correct.get(i, j) * SQRT_3_BY_2 / sample.deviator_norm);
            }
        }
        let sigma = Tensor2::from_matrix(&sample.matrix, case).unwrap();
        let mut ana_deriv = Tensor2::new(case);
        sigma.deriv1_invariant_sigma_d(&mut ana_deriv).unwrap();
        if verbose {
            println!("analytical dσd/dσ =\n{}", ana_deriv.to_matrix());
            println!("correct dσd/dσ =\n{}", correct);
        }
        mat_approx_eq(&ana_deriv.to_matrix(), &correct, tol);

        // compare with numerical derivative
        let mut args = ArgsNumDeriv1 {
            at_sigma: Tensor2::from_matrix(&sample.matrix, case).unwrap(),
            temp_sigma: Tensor2::new(case),
            m: 0,
        };
        let mut num_deriv = Tensor2::new(case);
        for m in 0..ana_deriv.vec.dim() {
            args.m = m;
            let res = deriv_central5(args.at_sigma.vec[m], &mut args, sigma_d_given_sigma_mandel);
            num_deriv.vec[m] = res;
        }
        if verbose {
            println!("numerical dσd/dσ =\n{}", num_deriv.to_matrix());
        }
        vec_approx_eq(ana_deriv.vec.as_data(), num_deriv.vec.as_data(), tol_num);
    }

    #[test]
    fn deriv1_sigma_d_works() {
        check_deriv1_sigma_d(Mandel::Symmetric, &SamplesTensor2::TENSOR_U, 1e-15, 1e-10, false);
        check_deriv1_sigma_d(Mandel::Symmetric, &SamplesTensor2::TENSOR_S, 1e-15, 1e-10, false);
        check_deriv1_sigma_d(Mandel::Symmetric2D, &SamplesTensor2::TENSOR_X, 1e-15, 1e-11, false);
        check_deriv1_sigma_d(Mandel::Symmetric2D, &SamplesTensor2::TENSOR_Y, 1e-15, 1e-10, false);
        check_deriv1_sigma_d(Mandel::Symmetric2D, &SamplesTensor2::TENSOR_Z, 1e-15, 1e-11, false);
    }

    // -- deriv1_invariant_lode --------------------------------------------------------------------------

    #[test]
    fn deriv1_invariant_lode_captures_errors() {
        let sigma = Tensor2::from_matrix(&SamplesTensor2::TENSOR_I.matrix, Mandel::Symmetric).unwrap();
        let mut d1 = Tensor2::new(Mandel::Symmetric2D);
        let mut aux = Tensor2::new(Mandel::Symmetric);
        assert_eq!(
            sigma.deriv1_invariant_lode(&mut d1, &mut aux, 1e-10).err(),
            Some("tensors are incompatible")
        );
        let mut d1 = Tensor2::new(Mandel::Symmetric2D);
        let mut s = Tensor2::new(Mandel::Symmetric);
        assert_eq!(
            sigma.deriv1_invariant_lode(&mut d1, &mut s, 1e-10).err(),
            Some("tensors are incompatible")
        );
    }

    #[test]
    fn deriv1_invariant_lode_handles_indeterminate_case() {
        let sigma = Tensor2::from_matrix(&SamplesTensor2::TENSOR_O.matrix, Mandel::Symmetric).unwrap();
        let mut d1 = Tensor2::new(Mandel::Symmetric);
        let mut aux = Tensor2::new(Mandel::Symmetric);
        assert_eq!(sigma.deriv1_invariant_lode(&mut d1, &mut aux, 1e-10), Ok(None));
    }

    // Computes l for varying v_mandel := MandelComponent(σᵢⱼ)
    fn lode_given_sigma_mandel(v_mandel: f64, args: &mut ArgsNumDeriv1) -> f64 {
        args.temp_sigma.mirror(&args.at_sigma).unwrap();
        args.temp_sigma.vec[args.m] = v_mandel;
        match args.temp_sigma.invariant_lode(1e-10) {
            Some(l) => l,
            None => panic!("cannot compute numerical derivative for None l value"),
        }
    }

    // Checks the first derivative of l w.r.t. σ
    fn check_deriv1_lode(case: Mandel, sample: &SampleTensor2, tol_num: f64, verbose: bool) {
        // analytical derivative
        let sigma = Tensor2::from_matrix(&sample.matrix, case).unwrap();
        let mut aux = Tensor2::new(case);
        let mut ana_deriv = Tensor2::new(case);
        sigma.deriv1_invariant_lode(&mut ana_deriv, &mut aux, 1e-15).unwrap();
        if verbose {
            println!("analytical dl/dσ =\n{}", ana_deriv.to_matrix());
        }

        // analytical derivative using alternative method
        let mut ana_deriv_alt = Tensor2::new(case);
        let mut s = Tensor2::new(case);
        let mut si = Tensor2::new(case);
        let mut psi = Tensor2::new(case);
        if let Some(_) = sigma
            .deriv1_invariant_lode_alt(&mut ana_deriv_alt, &mut s, &mut si, &mut psi, 1e-10)
            .unwrap()
        {
            if verbose {
                println!("analytical (alt) dl/dσ =\n{}", ana_deriv_alt.to_matrix());
            }
        } else {
            panic!("Lode invariant is None");
        }
        vec_approx_eq(&ana_deriv.vec.as_data(), &ana_deriv_alt.vec.as_data(), 1e-15);

        // compare with numerical derivative
        let mut args = ArgsNumDeriv1 {
            at_sigma: Tensor2::from_matrix(&sample.matrix, case).unwrap(),
            temp_sigma: Tensor2::new(case),
            m: 0,
        };
        let mut num_deriv = Tensor2::new(case);
        for m in 0..ana_deriv.vec.dim() {
            args.m = m;
            let res = deriv_central5(args.at_sigma.vec[m], &mut args, lode_given_sigma_mandel);
            num_deriv.vec[m] = res;
        }
        if verbose {
            println!("numerical dl/dσ =\n{}", num_deriv.to_matrix());
        }
        vec_approx_eq(ana_deriv.vec.as_data(), num_deriv.vec.as_data(), tol_num);
    }

    #[test]
    fn deriv1_invariant_lode_works() {
        check_deriv1_lode(Mandel::Symmetric, &SamplesTensor2::TENSOR_U, 1e-10, false);
        check_deriv1_lode(Mandel::Symmetric, &SamplesTensor2::TENSOR_S, 1e-10, false);
        check_deriv1_lode(Mandel::Symmetric2D, &SamplesTensor2::TENSOR_X, 1e-10, false);
        check_deriv1_lode(Mandel::Symmetric2D, &SamplesTensor2::TENSOR_Y, 1e-10, false);
        check_deriv1_lode(Mandel::Symmetric2D, &SamplesTensor2::TENSOR_Z, 1e-10, false);
    }

    // -- deriv1_invariant_lode_alt ----------------------------------------------------------------------

    #[test]
    fn deriv1_invariant_lode_alt_works() {
        // α = 30
        let (l1, l2, l3) = (1.0, 0.0, 1.0);
        let tt = Tensor2::from_matrix(&[[l1, 0.0, 0.0], [0.0, l2, 0.0], [0.0, 0.0, l3]], Mandel::Symmetric2D).unwrap();
        let mut ll = Tensor2::new(Mandel::Symmetric2D);
        let mut s = Tensor2::new(Mandel::Symmetric2D);
        let mut si = Tensor2::new(Mandel::Symmetric2D);
        let mut psi = Tensor2::new(Mandel::Symmetric2D);
        if let Some(l) = tt
            .deriv1_invariant_lode_alt(&mut ll, &mut s, &mut si, &mut psi, 1e-10)
            .unwrap()
        {
            approx_eq(l, -1.0, 1e-15);
            println!("L =\n{}", ll.to_matrix());
        } else {
            panic!("Lode invariant is None");
        }
    }
}
