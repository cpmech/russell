use crate::{Mandel, StrError, Tensor2, ONE_BY_3, SQRT_3, TOL_J2, TWO_BY_3};
use russell_lab::vec_add;

pub trait DerivFirst {
    /// Allocates a new instance
    fn new(case: Mandel) -> Result<Self, StrError>
    where
        Self: Sized;

    /// Computes the derivative
    ///
    /// * Returns `None` when it is impossible to compute the derivative
    /// * The returned value depends on the specific case
    fn compute(&mut self, sigma: &Tensor2) -> Result<Option<f64>, StrError>;

    /// Returns the Mandel component of the derivative
    ///
    /// * This function may panic if `m` is out-of-bounds
    fn get(&self, m: usize) -> f64;
}

/// Computes the first derivative of the norm
pub struct Deriv1Norm {
    /// Holds the resulting derivative
    pub result: Tensor2,
}

impl DerivFirst for Deriv1Norm {
    /// Returns a new instance
    fn new(case: Mandel) -> Result<Self, StrError> {
        Ok(Deriv1Norm {
            result: Tensor2::new(case),
        })
    }

    /// Computes the first derivative of the norm
    ///
    /// ```text
    /// d‖σ‖    σ
    /// ──── = ───
    ///  dσ    ‖σ‖
    /// ```
    ///
    /// # Results
    ///
    /// If `‖σ‖ > 0`, returns `‖σ‖` and `result` is valid;
    /// Otherwise, returns `None` and `result` is invalid.
    fn compute(&mut self, sigma: &Tensor2) -> Result<Option<f64>, StrError> {
        let dim = self.result.vec.dim();
        if sigma.vec.dim() != dim {
            return Err("tensor 'sigma' is incompatible");
        }
        let n = sigma.norm();
        if n > 0.0 {
            self.result.mirror(sigma).unwrap();
            for i in 0..dim {
                self.result.vec[i] /= n;
            }
            return Ok(Some(n));
        }
        Ok(None)
    }

    /// Returns the Mandel component of the derivative
    ///
    /// * This function may panic if `m` is out-of-bounds
    fn get(&self, m: usize) -> f64 {
        self.result.vec[m]
    }
}

/// Computes the first derivative of the J2 invariant
pub struct Deriv1InvariantJ2 {
    /// Holds the resulting derivative (Symmetric or Symmetric2D)
    pub result: Tensor2,
}

impl Deriv1InvariantJ2 {
    /// Returns a new instance
    ///
    /// Note: `case` must be Symmetric or Symmetric2D
    pub fn new(case: Mandel) -> Result<Self, StrError> {
        if case == Mandel::General {
            return Err("case must be Symmetric or Symmetric2D");
        }
        Ok(Deriv1InvariantJ2 {
            result: Tensor2::new(case),
        })
    }

    /// Computes the first derivative of the J2 invariant
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
    pub fn compute(&mut self, sigma: &Tensor2) -> Result<(), StrError> {
        if sigma.vec.dim() != self.result.vec.dim() {
            return Err("tensor 'sigma' is incompatible");
        }
        sigma.deviator(&mut self.result)
    }
}

/// Computes the first derivative of the J3 invariant
pub struct Deriv1InvariantJ3 {
    /// Deviator tensor (Symmetric or Symmetric2D)
    pub s: Tensor2,

    /// Holds the resulting derivative (Symmetric or Symmetric2D)
    pub result: Tensor2,
}

impl Deriv1InvariantJ3 {
    /// Returns a new instance
    ///
    /// Note: `case` must be Symmetric or Symmetric2D
    pub fn new(case: Mandel) -> Result<Self, StrError> {
        if case == Mandel::General {
            return Err("case must be Symmetric or Symmetric2D");
        }
        Ok(Deriv1InvariantJ3 {
            s: Tensor2::new(case),
            result: Tensor2::new(case),
        })
    }

    /// Computes the first derivative of the J3 invariant
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
    pub fn compute(&mut self, sigma: &Tensor2) -> Result<(), StrError> {
        if sigma.vec.dim() != self.s.vec.dim() {
            return Err("tensor 'sigma' is incompatible");
        }
        let jj2 = sigma.invariant_jj2();
        sigma.deviator(&mut self.s).unwrap();
        self.s.squared(&mut self.result).unwrap();
        self.result.vec[0] -= TWO_BY_3 * jj2;
        self.result.vec[1] -= TWO_BY_3 * jj2;
        self.result.vec[2] -= TWO_BY_3 * jj2;
        Ok(())
    }
}

/// Computes the first derivative of the mean stress
pub struct Deriv1InvariantSigmaM {
    /// Holds the resulting derivative
    pub result: Tensor2,
}

impl Deriv1InvariantSigmaM {
    /// Returns a new instance
    pub fn new(case: Mandel) -> Self {
        Deriv1InvariantSigmaM {
            result: Tensor2::new(case),
        }
    }

    /// Computes the first derivative of the deviatoric stress invariant (von Mises)
    ///
    /// ```text
    /// dσm   1
    /// ─── = ─ I
    ///  dσ   3
    /// ```
    pub fn compute(&mut self, sigma: &Tensor2) -> Result<(), StrError> {
        let dim = self.result.vec.dim();
        if sigma.vec.dim() != dim {
            return Err("tensor 'sigma' is incompatible");
        }
        self.result.vec[0] = ONE_BY_3;
        self.result.vec[1] = ONE_BY_3;
        self.result.vec[2] = ONE_BY_3;
        for i in 3..dim {
            self.result.vec[i] = 0.0;
        }
        Ok(())
    }
}

/// Computes the first derivative of the deviatoric stress invariant (von Mises)
pub struct Deriv1InvariantSigmaD {
    /// Auxiliary structure to compute the derivative of J2
    pub aux: Deriv1InvariantJ2,

    /// Holds the resulting derivative (Symmetric or Symmetric2D)
    pub result: Tensor2,
}

impl Deriv1InvariantSigmaD {
    /// Returns a new instance
    ///
    /// Note: `case` must be Symmetric or Symmetric2D
    pub fn new(case: Mandel) -> Result<Self, StrError> {
        if case == Mandel::General {
            return Err("case must be Symmetric or Symmetric2D");
        }
        Ok(Deriv1InvariantSigmaD {
            aux: Deriv1InvariantJ2::new(case).unwrap(),
            result: Tensor2::new(case),
        })
    }

    /// Computes the first derivative of the deviatoric stress invariant (von Mises)
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
    /// # Results
    ///
    /// If `J2 > TOL_J2`, returns `J2` and `result` is valid;
    /// Otherwise, returns `None` and `result` is invalid.
    pub fn compute(&mut self, sigma: &Tensor2) -> Result<Option<f64>, StrError> {
        let dim = self.result.vec.dim();
        if sigma.vec.dim() != dim {
            return Err("tensor 'sigma' is incompatible");
        }
        let jj2 = sigma.invariant_jj2();
        if jj2 > TOL_J2 {
            let a = 0.5 * SQRT_3 / f64::powf(jj2, 0.5);
            self.aux.compute(sigma).unwrap();
            for i in 0..dim {
                self.result.vec[i] = a * self.aux.result.vec[i];
            }
            return Ok(Some(jj2));
        }
        Ok(None)
    }
}

/// Computes the first derivative of the Lode invariant
pub struct Deriv1InvariantLode {
    /// Auxiliary structure to compute the first derivative of J2
    pub aux_jj2: Deriv1InvariantJ2,

    /// Auxiliary structure to compute the first derivative of J3
    pub aux_jj3: Deriv1InvariantJ3,

    /// Holds the resulting derivative (Symmetric or Symmetric2D)
    pub result: Tensor2,
}

impl Deriv1InvariantLode {
    /// Returns a new instance
    ///
    /// Note: `case` must be Symmetric or Symmetric2D
    pub fn new(case: Mandel) -> Result<Self, StrError> {
        if case == Mandel::General {
            return Err("case must be Symmetric or Symmetric2D");
        }
        Ok(Deriv1InvariantLode {
            aux_jj2: Deriv1InvariantJ2::new(case).unwrap(),
            aux_jj3: Deriv1InvariantJ3::new(case).unwrap(),
            result: Tensor2::new(case),
        })
    }

    /// Computes the first derivative of the Lode invariant
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
    /// # Results
    ///
    /// If `J2 > TOL_J2`, returns `J2` and `result` is valid;
    /// Otherwise, returns `None` and `result` is invalid.
    pub fn compute(&mut self, sigma: &Tensor2) -> Result<Option<f64>, StrError> {
        if sigma.vec.dim() != self.result.vec.dim() {
            return Err("tensor 'sigma' is incompatible");
        }
        let jj2 = sigma.invariant_jj2();
        if jj2 > TOL_J2 {
            self.aux_jj2.compute(sigma).unwrap();
            self.aux_jj3.compute(sigma).unwrap();
            let jj3 = sigma.invariant_jj3();
            let a = 1.5 * SQRT_3 / f64::powf(jj2, 1.5);
            let b = 2.25 * SQRT_3 / f64::powf(jj2, 2.5);
            vec_add(
                &mut self.result.vec,
                a,
                &self.aux_jj3.result.vec,
                -b * jj3,
                &self.aux_jj2.result.vec,
            )
            .unwrap();
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
        Deriv1InvariantJ2, Deriv1InvariantJ3, Deriv1InvariantLode, Deriv1InvariantSigmaD, Deriv1InvariantSigmaM,
        Deriv1Norm, DerivFirst, Mandel, SampleTensor2, SamplesTensor2, IJ_TO_M, ONE_BY_3, SQRT_3_BY_2,
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
                let mut aux = Deriv1Norm::new(sigma.case()).unwrap();
                aux.compute(sigma).unwrap().unwrap();
                d1.mirror(&aux.result).unwrap();
            }
            F::J2 => {
                let mut aux = Deriv1InvariantJ2::new(sigma.case()).unwrap();
                aux.compute(sigma).unwrap();
                d1.mirror(&aux.result).unwrap();
            }
            F::J3 => {
                let mut aux = Deriv1InvariantJ3::new(sigma.case()).unwrap();
                aux.compute(sigma).unwrap();
                d1.mirror(&aux.result).unwrap();
            }
            F::SigmaM => {
                let mut aux = Deriv1InvariantSigmaM::new(sigma.case());
                aux.compute(sigma).unwrap();
                d1.mirror(&aux.result).unwrap();
            }
            F::SigmaD => {
                let mut aux = Deriv1InvariantSigmaD::new(sigma.case()).unwrap();
                aux.compute(sigma).unwrap();
                d1.mirror(&aux.result).unwrap();
            }
            F::Lode => {
                let mut aux = Deriv1InvariantLode::new(sigma.case()).unwrap();
                aux.compute(sigma).unwrap().unwrap();
                d1.mirror(&aux.result).unwrap();
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
