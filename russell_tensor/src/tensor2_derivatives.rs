use crate::{StrError, Tensor2, ONE_BY_3, SQRT_3, SQRT_3_BY_2, TWO_BY_3};

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
        if d1.vec.dim() != self.vec.dim() {
            return Err("tensors are incompatible");
        }
        let n = self.norm();
        if n > 0.0 {
            d1.mirror(self).unwrap();
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
        if d1.vec.dim() != self.vec.dim() {
            return Err("tensors are incompatible");
        }
        self.deviator(d1).unwrap();
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
        if d1.vec.dim() != self.vec.dim() {
            return Err("tensors are incompatible");
        }
        self.deviator(s).unwrap();
        s.squared(d1).unwrap(); // d1 := s·s
        let jj2 = self.invariant_jj2();
        d1.vec[0] -= TWO_BY_3 * jj2;
        d1.vec[1] -= TWO_BY_3 * jj2;
        d1.vec[2] -= TWO_BY_3 * jj2;
        if self.vec.dim() > 6 {
            // transpose d1=s·s to get (s·s)ᵀ
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
    /// # Output
    ///
    /// * This function returns `Some(‖s‖)` if `‖s‖ > 0` and the computation was successful
    /// * Otherwise, this function returns `None` and the derivative cannot be computed
    ///   because the deviatoric stress invariant is zero
    pub fn deriv1_invariant_sigma_d(&self, d1: &mut Tensor2) -> Result<Option<f64>, StrError> {
        if d1.vec.dim() != self.vec.dim() {
            return Err("tensors are incompatible");
        }
        let n = self.deviator_norm();
        if n > 0.0 {
            self.deviator(d1).unwrap();
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
    /// dl       3 √3      dJ3      9 √3 J3    dJ2
    /// ── = ───────────── ─── - ───────────── ───
    /// dσ   2 pow(J2,1.5) dσ    4 pow(J2,2.5) dσ
    /// ```
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
        if d1.vec.dim() != self.vec.dim() {
            return Err("tensors are incompatible");
        }
        let ndim = d1.vec.dim();
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
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[allow(unused)]
#[cfg(test)]
mod tests {
    use super::Tensor2;
    use crate::{Mandel, SampleTensor2, SamplesTensor2, IJ_TO_M, ONE_BY_3, SQRT_3_BY_2, TOL_J2};
    use russell_chk::{approx_eq, deriv_central5, vec_approx_eq};
    use russell_lab::{mat_approx_eq, Matrix};

    // Defines f(σ)
    #[derive(Clone, Copy)]
    enum F {
        Norm,
        J2,
        J3,
        SigmaM,
        SigmaD,
        Lode,
    }

    // computes the analytical derivative df(σ)/dσ
    fn analytical_deriv(fn_name: F, d1: &mut Tensor2, sigma: &Tensor2) {
        match fn_name {
            F::Norm => {
                sigma.deriv1_norm(d1).unwrap().unwrap();
            }
            F::J2 => {
                sigma.deriv1_invariant_jj2(d1).unwrap();
            }
            F::J3 => {
                let mut s = sigma.clone();
                sigma.deriv1_invariant_jj3(d1, &mut s).unwrap();
            }
            F::SigmaM => {
                sigma.deriv1_invariant_sigma_m(d1).unwrap();
            }
            F::SigmaD => {
                sigma.deriv1_invariant_sigma_d(d1).unwrap().unwrap();
            }
            F::Lode => {
                let mut aux = sigma.clone();
                sigma.deriv1_invariant_lode(d1, &mut aux, 1e-10).unwrap().unwrap();
            }
        };
    }

    // Holds arguments for numerical differentiation of a scalar f(σ) w.r.t. σᵢⱼ (standard components)
    struct ArgsNumDeriv {
        fn_name: F,        // name of f(σ)
        sigma_mat: Matrix, // @ σ (3x3 matrix form)
        sigma: Tensor2,    // temporary tensor with varying ij-components
        i: usize,          // index i of ∂f/∂σᵢⱼ
        j: usize,          // index j of ∂f/∂σᵢⱼ
    }

    // Holds arguments for numerical differentiation of a scalar f(σ) w.r.t. σₘ (Mandel components)
    struct ArgsNumDerivMandel {
        fn_name: F,     // name of f(σ)
        sigma: Tensor2, // @ σ, with varying m-components
        m: usize,       // index m of ∂f/∂σₘ
    }

    // computes f(σ) for varying components x = σᵢⱼ
    fn f_sigma(x: f64, args: &mut ArgsNumDeriv) -> f64 {
        let original = args.sigma_mat.get(args.i, args.j);
        args.sigma_mat.set(args.i, args.j, x);
        args.sigma.set_matrix(&args.sigma_mat).unwrap();
        let res = match args.fn_name {
            F::Norm => args.sigma.norm(),
            F::J2 => args.sigma.invariant_jj2(),
            F::J3 => args.sigma.invariant_jj3(),
            F::SigmaM => args.sigma.invariant_sigma_m(),
            F::SigmaD => args.sigma.invariant_sigma_d(),
            F::Lode => args.sigma.invariant_lode().unwrap(),
        };
        args.sigma_mat.set(args.i, args.j, original);
        res
    }

    // computes f(σ) for varying components x = σₘ
    fn f_sigma_mandel(x: f64, args: &mut ArgsNumDerivMandel) -> f64 {
        let original = args.sigma.vec[args.m];
        args.sigma.vec[args.m] = x;
        let res = match args.fn_name {
            F::Norm => args.sigma.norm(),
            F::J2 => args.sigma.invariant_jj2(),
            F::J3 => args.sigma.invariant_jj3(),
            F::SigmaM => args.sigma.invariant_sigma_m(),
            F::SigmaD => args.sigma.invariant_sigma_d(),
            F::Lode => args.sigma.invariant_lode().unwrap(),
        };
        args.sigma.vec[args.m] = original;
        res
    }

    // computes ∂f/∂σᵢⱼ and returns as a 3x3 matrix of (standard) components
    fn numerical_deriv(sigma: &Tensor2, fn_name: F) -> Matrix {
        let mut args = ArgsNumDeriv {
            fn_name,
            sigma_mat: sigma.to_matrix(),
            sigma: sigma.to_general(),
            i: 0,
            j: 0,
        };
        let mut num_deriv = Matrix::new(3, 3);
        for i in 0..3 {
            args.i = i;
            for j in 0..3 {
                args.j = j;
                let x = args.sigma_mat.get(i, j);
                let res = deriv_central5(x, &mut args, f_sigma);
                num_deriv.set(i, j, res);
            }
        }
        num_deriv
    }

    // computes ∂f/∂σₘ and returns as a 3x3 matrix of (standard) components
    fn numerical_deriv_mandel(sigma: &Tensor2, fn_name: F) -> Matrix {
        let mut args = ArgsNumDerivMandel {
            fn_name,
            sigma: sigma.clone(),
            m: 0,
        };
        let mut num_deriv = sigma.clone();
        for m in 0..sigma.vec.dim() {
            args.m = m;
            let x = args.sigma.vec[m];
            let res = deriv_central5(x, &mut args, f_sigma_mandel);
            num_deriv.vec[m] = res;
        }
        num_deriv.to_matrix()
    }

    // checks ∂f/∂σᵢⱼ
    fn check_deriv(fn_name: F, case: Mandel, sample: &SampleTensor2, tol: f64, verbose: bool) {
        let sigma = Tensor2::from_matrix(&sample.matrix, case).unwrap();
        let mut d1 = Tensor2::new(case);
        analytical_deriv(fn_name, &mut d1, &sigma);
        let ana = d1.to_matrix();
        let num = numerical_deriv(&sigma, fn_name);
        let num_mandel = numerical_deriv_mandel(&sigma, fn_name);
        if verbose {
            println!("analytical derivative:\n{}", ana);
            println!("numerical derivative:\n{}", num);
            println!("numerical derivative (Mandel):\n{}", num_mandel);
        }
        mat_approx_eq(&ana, &num, tol);
        mat_approx_eq(&ana, &num_mandel, tol);
    }

    #[test]
    fn deriv_norm_works() {
        let v = false;
        check_deriv(F::Norm, Mandel::General, &SamplesTensor2::TENSOR_T, 1e-10, v);
        check_deriv(F::Norm, Mandel::Symmetric, &SamplesTensor2::TENSOR_S, 1e-10, v);
        check_deriv(F::Norm, Mandel::Symmetric2D, &SamplesTensor2::TENSOR_Z, 1e-11, v);
    }

    #[test]
    fn deriv_invariant_jj2_works() {
        let v = false;
        check_deriv(F::J2, Mandel::General, &SamplesTensor2::TENSOR_T, 1e-10, v);
        check_deriv(F::J2, Mandel::Symmetric, &SamplesTensor2::TENSOR_S, 1e-10, v);
        check_deriv(F::J2, Mandel::Symmetric2D, &SamplesTensor2::TENSOR_Z, 1e-11, v);
        check_deriv(F::J2, Mandel::Symmetric2D, &SamplesTensor2::TENSOR_O, 1e-15, v);
        check_deriv(F::J2, Mandel::Symmetric2D, &SamplesTensor2::TENSOR_I, 1e-12, v);
    }

    #[test]
    fn deriv_invariant_jj3_works() {
        let v = false;
        check_deriv(F::J3, Mandel::General, &SamplesTensor2::TENSOR_T, 1e-8, v);
        check_deriv(F::J3, Mandel::Symmetric, &SamplesTensor2::TENSOR_S, 1e-9, v);
        check_deriv(F::J3, Mandel::Symmetric2D, &SamplesTensor2::TENSOR_Z, 1e-10, v);
        check_deriv(F::J3, Mandel::Symmetric2D, &SamplesTensor2::TENSOR_O, 1e-15, v);
        check_deriv(F::J3, Mandel::Symmetric2D, &SamplesTensor2::TENSOR_I, 1e-15, v);
    }

    #[test]
    fn deriv_sigma_m_works() {
        let v = false;
        check_deriv(F::SigmaM, Mandel::General, &SamplesTensor2::TENSOR_T, 1e-12, v);
        check_deriv(F::SigmaM, Mandel::Symmetric, &SamplesTensor2::TENSOR_S, 1e-11, v);
        check_deriv(F::SigmaM, Mandel::Symmetric2D, &SamplesTensor2::TENSOR_Z, 1e-12, v);
    }

    #[test]
    fn deriv_sigma_d_works() {
        let v = false;
        check_deriv(F::SigmaD, Mandel::Symmetric, &SamplesTensor2::TENSOR_U, 1e-10, v);
        check_deriv(F::SigmaD, Mandel::Symmetric, &SamplesTensor2::TENSOR_S, 1e-10, v);
        check_deriv(F::SigmaD, Mandel::Symmetric2D, &SamplesTensor2::TENSOR_X, 1e-11, v);
        check_deriv(F::SigmaD, Mandel::Symmetric2D, &SamplesTensor2::TENSOR_Y, 1e-10, v);
        check_deriv(F::SigmaD, Mandel::Symmetric2D, &SamplesTensor2::TENSOR_Z, 1e-10, v);
    }

    #[test]
    fn deriv_invariant_lode_works() {
        let v = false;
        check_deriv(F::Lode, Mandel::Symmetric, &SamplesTensor2::TENSOR_U, 1e-10, v);
        check_deriv(F::Lode, Mandel::Symmetric, &SamplesTensor2::TENSOR_S, 1e-10, v);
        check_deriv(F::Lode, Mandel::Symmetric2D, &SamplesTensor2::TENSOR_X, 1e-10, v);
        check_deriv(F::Lode, Mandel::Symmetric2D, &SamplesTensor2::TENSOR_Y, 1e-10, v);
        check_deriv(F::Lode, Mandel::Symmetric2D, &SamplesTensor2::TENSOR_Z, 1e-10, v);
    }

    // -- check errors -------------------------------------------------------------------------------

    #[test]
    fn check_errors() {
        let sigma = Tensor2::from_matrix(&SamplesTensor2::TENSOR_I.matrix, Mandel::General).unwrap();
        let mut s = sigma.clone();
        let mut aux = sigma.clone();
        let mut d1 = Tensor2::new(Mandel::Symmetric);
        assert_eq!(sigma.deriv1_norm(&mut d1).err(), Some("tensors are incompatible"));
        assert_eq!(
            sigma.deriv1_invariant_jj2(&mut d1).err(),
            Some("tensors are incompatible")
        );
        assert_eq!(
            sigma.deriv1_invariant_jj3(&mut d1, &mut s).err(),
            Some("tensors are incompatible")
        );
        assert_eq!(
            sigma.deriv1_invariant_sigma_m(&mut d1).err(),
            Some("tensors are incompatible")
        );
        assert_eq!(
            sigma.deriv1_invariant_sigma_d(&mut d1).err(),
            Some("tensors are incompatible")
        );
        assert_eq!(
            sigma.deriv1_invariant_lode(&mut d1, &mut aux, TOL_J2).err(),
            Some("tensors are incompatible")
        );
    }
}
