use crate::{Mandel, StrError, Tensor2, ONE_BY_3, SQRT_3, TOL_J2, TWO_BY_3};

/// Calculates the first derivative of the norm
pub struct FirstDerivNorm {}

impl FirstDerivNorm {
    /// Calculates the first derivative of the norm
    ///
    /// ```text
    /// d‖σ‖    σ
    /// ──── = ───
    ///  dσ    ‖σ‖
    /// ```
    ///
    /// # Results
    ///
    /// If `‖σ‖ > 0`, returns `‖σ‖` and `d1` is valid;
    /// Otherwise, returns `None` and `d1` is unchanged.
    pub fn calc(d1: &mut Tensor2, sigma: &Tensor2) -> Result<Option<f64>, StrError> {
        let dim = d1.vec.dim();
        if sigma.vec.dim() != dim {
            return Err("sigma and d1 tensors are incompatible");
        }
        let n = sigma.norm();
        if n > 0.0 {
            d1.mirror(sigma).unwrap();
            for i in 0..dim {
                d1.vec[i] /= n;
            }
            return Ok(Some(n));
        }
        Ok(None)
    }
}

/// Calculates the first derivative of the J2 invariant
pub struct FirstDerivJ2 {}

impl FirstDerivJ2 {
    /// Calculates the first derivative of the J2 invariant
    ///
    /// ```text
    /// s = deviator(σ)
    ///
    /// dJ2
    /// ─── = s
    ///  dσ
    ///
    /// (σ is symmetric)
    /// ```
    ///
    /// Note: `sigma` must be Symmetric or Symmetric2D.
    pub fn calc(d1: &mut Tensor2, sigma: &Tensor2) -> Result<(), StrError> {
        if sigma.case() == Mandel::General {
            return Err("sigma tensor must be Symmetric or Symmetric2D");
        }
        sigma.deviator(d1)
    }
}

/// Calculates the first derivative of the J3 invariant
pub struct FirstDerivJ3 {
    /// Deviator tensor (Symmetric or Symmetric2D)
    pub s: Tensor2,
}

impl FirstDerivJ3 {
    /// Returns a new instance
    ///
    /// Note: `case` must be Symmetric or Symmetric2D
    pub fn new(case: Mandel) -> Result<Self, StrError> {
        if case == Mandel::General {
            return Err("case must be Symmetric or Symmetric2D");
        }
        Ok(FirstDerivJ3 { s: Tensor2::new(case) })
    }

    /// Calculates the first derivative of the J3 invariant
    ///
    /// ```text
    /// s = deviator(σ)
    ///
    /// dJ3         2 J2
    /// ─── = s·s - ──── I
    ///  dσ           3
    ///
    /// (σ is symmetric)
    /// ```
    ///
    /// Note: `sigma` must be Symmetric or Symmetric2D.
    pub fn calc(&mut self, d1: &mut Tensor2, sigma: &Tensor2) -> Result<(), StrError> {
        let dim = self.s.vec.dim();
        if sigma.vec.dim() != dim {
            return Err("sigma tensor is incompatible");
        }
        if d1.vec.dim() != dim {
            return Err("d1 tensor is incompatible");
        }
        let jj2 = sigma.invariant_jj2();
        sigma.deviator(&mut self.s).unwrap();
        self.s.squared(d1).unwrap();
        d1.vec[0] -= TWO_BY_3 * jj2;
        d1.vec[1] -= TWO_BY_3 * jj2;
        d1.vec[2] -= TWO_BY_3 * jj2;
        Ok(())
    }
}

/// Calculates the first derivative of the mean stress
pub struct FirstDerivSigmaM {}

impl FirstDerivSigmaM {
    /// Calculates the first derivative of the deviatoric stress invariant (von Mises)
    ///
    /// ```text
    /// dσm   1
    /// ─── = ─ I
    ///  dσ   3
    /// ```
    pub fn calc(d1: &mut Tensor2, sigma: &Tensor2) -> Result<(), StrError> {
        let dim = d1.vec.dim();
        if sigma.vec.dim() != dim {
            return Err("sigma and d1 tensors are incompatible");
        }
        d1.vec[0] = ONE_BY_3;
        d1.vec[1] = ONE_BY_3;
        d1.vec[2] = ONE_BY_3;
        for i in 3..dim {
            d1.vec[i] = 0.0;
        }
        Ok(())
    }
}

/// Calculates the first derivative of the deviatoric stress invariant (von Mises)
pub struct FirstDerivSigmaD {}

impl FirstDerivSigmaD {
    /// Calculates the first derivative of the deviatoric stress invariant (von Mises)
    ///
    /// ```text
    /// s = deviator(σ)
    ///
    /// dσd        √3       dJ2
    /// ─── = ───────────── ───
    ///  dσ   2 pow(J2,0.5)  dσ
    ///
    /// (σ is symmetric)
    /// ```
    ///
    /// Note: `sigma` must be Symmetric or Symmetric2D.
    ///
    /// # Results
    ///
    /// If `J2 > TOL_J2`, returns `J2` and `result` is valid;
    /// Otherwise, returns `None` and `result` is invalid.
    pub fn calc(d1: &mut Tensor2, sigma: &Tensor2) -> Result<Option<f64>, StrError> {
        if sigma.case() == Mandel::General {
            return Err("sigma tensor must be Symmetric or Symmetric2D");
        }
        let dim = d1.vec.dim();
        if sigma.vec.dim() != dim {
            return Err("sigma and d1 tensors are incompatible");
        }
        let jj2 = sigma.invariant_jj2();
        if jj2 > TOL_J2 {
            let a = 0.5 * SQRT_3 / f64::powf(jj2, 0.5);
            FirstDerivJ2::calc(d1, sigma).unwrap();
            for i in 0..dim {
                d1.vec[i] *= a;
            }
            return Ok(Some(jj2));
        }
        Ok(None)
    }
}

/// Calculates the first derivative of the Lode invariant
pub struct FirstDerivLode {
    /// first derivative of J2: dJ2/dσ (Symmetric or Symmetric2D)
    pub d1_jj2: Tensor2,

    /// first derivative of J3: dJ3/dσ (Symmetric or Symmetric2D)
    pub d1_jj3: Tensor2,

    /// Auxiliary structure to calculate the first derivative of J3
    pub aux_jj3: FirstDerivJ3,
}

impl FirstDerivLode {
    /// Returns a new instance
    ///
    /// Note: `case` must be Symmetric or Symmetric2D
    pub fn new(case: Mandel) -> Result<Self, StrError> {
        if case == Mandel::General {
            return Err("case must be Symmetric or Symmetric2D");
        }
        Ok(FirstDerivLode {
            d1_jj2: Tensor2::new(case),
            d1_jj3: Tensor2::new(case),
            aux_jj3: FirstDerivJ3::new(case).unwrap(),
        })
    }

    /// Calculates the first derivative of the Lode invariant
    ///
    /// ```text
    /// dl     dJ3        dJ2
    /// ── = a ─── - b J3 ───
    /// dσ     dσ         dσ
    /// ```
    ///
    /// ```text
    ///         3 √3                9 √3
    /// a = ─────────────   b = ─────────────
    ///     2 pow(J2,1.5)       4 pow(J2,2.5)
    /// ```
    ///
    /// Note: `sigma` must be Symmetric or Symmetric2D.
    ///
    /// # Results
    ///
    /// If `J2 > TOL_J2`, returns `J2` and `d1` is valid;
    /// Otherwise, returns `None` and `d1` is unchanged.
    pub fn calc(&mut self, d1: &mut Tensor2, sigma: &Tensor2) -> Result<Option<f64>, StrError> {
        let dim = self.d1_jj2.vec.dim();
        if sigma.vec.dim() != dim {
            return Err("sigma tensor is incompatible");
        }
        if d1.vec.dim() != dim {
            return Err("d1 tensor is incompatible");
        }
        let jj2 = sigma.invariant_jj2();
        if jj2 > TOL_J2 {
            FirstDerivJ2::calc(&mut self.d1_jj2, sigma).unwrap();
            self.aux_jj3.calc(&mut self.d1_jj3, sigma).unwrap();
            let jj3 = sigma.invariant_jj3();
            let a = 1.5 * SQRT_3 / f64::powf(jj2, 1.5);
            let b = 2.25 * SQRT_3 / f64::powf(jj2, 2.5);
            for i in 0..dim {
                d1.vec[i] = a * self.d1_jj3.vec[i] - b * jj3 * self.d1_jj2.vec[i];
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
    use crate::{
        FirstDerivJ2, FirstDerivJ3, FirstDerivLode, FirstDerivNorm, FirstDerivSigmaD, FirstDerivSigmaM, Mandel,
        SampleTensor2, SamplesTensor2, IJ_TO_M, ONE_BY_3, SQRT_3_BY_2,
    };
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

    #[test]
    fn f_enum_clone_works() {
        let a = F::Norm;
        a.clone();
    }

    // computes the analytical derivative df(σ)/dσ
    fn analytical_deriv(fn_name: F, d1: &mut Tensor2, sigma: &Tensor2) {
        match fn_name {
            F::Norm => {
                FirstDerivNorm::calc(d1, sigma).unwrap().unwrap();
            }
            F::J2 => FirstDerivJ2::calc(d1, sigma).unwrap(),
            F::J3 => {
                let mut s = Tensor2::new(sigma.case());
                let mut aux = FirstDerivJ3::new(sigma.case()).unwrap();
                aux.calc(d1, sigma).unwrap();
            }
            F::SigmaM => FirstDerivSigmaM::calc(d1, sigma).unwrap(),
            F::SigmaD => {
                FirstDerivSigmaD::calc(d1, sigma).unwrap().unwrap();
            }
            F::Lode => {
                let mut aux = FirstDerivLode::new(sigma.case()).unwrap();
                aux.calc(d1, sigma).unwrap().unwrap();
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
        check_deriv(F::J2, Mandel::Symmetric, &SamplesTensor2::TENSOR_S, 1e-10, v);
        check_deriv(F::J2, Mandel::Symmetric2D, &SamplesTensor2::TENSOR_Z, 1e-11, v);
        check_deriv(F::J2, Mandel::Symmetric2D, &SamplesTensor2::TENSOR_O, 1e-15, v);
        check_deriv(F::J2, Mandel::Symmetric2D, &SamplesTensor2::TENSOR_I, 1e-12, v);
    }

    #[test]
    fn deriv_invariant_jj3_works() {
        let v = false;
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

    // -- check None and errors ----------------------------------------------------------------------

    #[test]
    fn check_for_none() {
        /*
        let sigma = Tensor2::from_matrix(&SamplesTensor2::TENSOR_O.matrix, Mandel::Symmetric).unwrap();
        let mut d1 = Tensor2::new(Mandel::Symmetric);
        let mut aux = Tensor2::new(Mandel::Symmetric);
        assert_eq!(sigma.deriv1_norm(&mut d1).unwrap(), None);
        assert_eq!(sigma.deriv1_invariant_sigma_d(&mut d1).unwrap(), None);
        assert_eq!(sigma.deriv1_invariant_lode(&mut d1, &mut aux).unwrap(), None);
        */
    }

    #[test]
    fn check_errors() {
        /*
        let sigma = Tensor2::from_matrix(&SamplesTensor2::TENSOR_I.matrix, Mandel::General).unwrap();
        let mut aux = sigma.clone();
        let mut d1 = Tensor2::new(Mandel::Symmetric);
        assert_eq!(sigma.deriv1_norm(&mut d1).err(), Some("tensors are incompatible"));
        assert_eq!(
            sigma.deriv1_invariant_jj2(&mut d1).err(),
            Some("tensors are incompatible")
        );
        assert_eq!(
            sigma.deriv1_invariant_jj3(&mut d1, &mut aux).err(),
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
            sigma.deriv1_invariant_lode(&mut d1, &mut aux).err(),
            Some("tensors are incompatible")
        );
        */
    }
}
