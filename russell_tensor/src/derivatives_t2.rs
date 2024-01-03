use crate::{Mandel, StrError, Tensor2, ONE_BY_3, SQRT_3, TOL_J2, TWO_BY_3};

/// Calculates the first derivative of the norm w.r.t. the defining Tensor2
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
pub fn deriv1_norm(d1: &mut Tensor2, sigma: &Tensor2) -> Result<Option<f64>, StrError> {
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

/// Calculates the first derivative of the J2 invariant w.r.t. the stress tensor
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
pub fn deriv1_invariant_jj2(d1: &mut Tensor2, sigma: &Tensor2) -> Result<(), StrError> {
    if sigma.mandel() == Mandel::General {
        return Err("sigma tensor must be Symmetric or Symmetric2D");
    }
    sigma.deviator(d1)
}

/// Calculates the first derivative of the J3 invariant w.r.t. the stress tensor
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
pub fn deriv1_invariant_jj3(d1: &mut Tensor2, s: &mut Tensor2, sigma: &Tensor2) -> Result<(), StrError> {
    if sigma.mandel() == Mandel::General {
        return Err("sigma tensor must be Symmetric or Symmetric2D");
    }
    let dim = sigma.vec.dim();
    if s.vec.dim() != dim {
        return Err("s tensor is incompatible");
    }
    if d1.vec.dim() != dim {
        return Err("d1 tensor is incompatible");
    }
    let jj2 = sigma.invariant_jj2();
    sigma.deviator(s).unwrap();
    s.squared(d1).unwrap();
    d1.vec[0] -= TWO_BY_3 * jj2;
    d1.vec[1] -= TWO_BY_3 * jj2;
    d1.vec[2] -= TWO_BY_3 * jj2;
    Ok(())
}

/// Calculates the first derivative of the mean stress invariant w.r.t. the stress tensor
///
/// ```text
/// dσm   1
/// ─── = ─ I
///  dσ   3
/// ```
pub fn deriv1_invariant_sigma_m(d1: &mut Tensor2, sigma: &Tensor2) -> Result<(), StrError> {
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

/// Calculates the first derivative of the deviatoric stress invariant (von Mises) w.r.t. the stress tensor
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
pub fn deriv1_invariant_sigma_d(d1: &mut Tensor2, sigma: &Tensor2) -> Result<Option<f64>, StrError> {
    if sigma.mandel() == Mandel::General {
        return Err("sigma tensor must be Symmetric or Symmetric2D");
    }
    let dim = sigma.vec.dim();
    if d1.vec.dim() != dim {
        return Err("d1 tensor is incompatible");
    }
    let jj2 = sigma.invariant_jj2();
    if jj2 > TOL_J2 {
        let a = 0.5 * SQRT_3 / f64::powf(jj2, 0.5);
        deriv1_invariant_jj2(d1, sigma).unwrap();
        for i in 0..dim {
            d1.vec[i] *= a;
        }
        return Ok(Some(jj2));
    }
    Ok(None)
}

/// Calculates the first derivative of the Lode invariant w.r.t. the stress tensor
///
/// ```text
/// dl     dJ3        dJ2
/// ── = a ─── - b J3 ───
/// dσ     dσ         dσ
///
/// or
///
/// dl     dJ3
/// ── = a ─── - b J3 s
/// dσ     dσ
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
pub fn deriv1_invariant_lode(d1: &mut Tensor2, s: &mut Tensor2, sigma: &Tensor2) -> Result<Option<f64>, StrError> {
    if sigma.mandel() == Mandel::General {
        return Err("sigma tensor must be Symmetric or Symmetric2D");
    }
    let dim = sigma.vec.dim();
    if s.vec.dim() != dim {
        return Err("s tensor is incompatible");
    }
    if d1.vec.dim() != dim {
        return Err("d1 tensor is incompatible");
    }
    let jj2 = sigma.invariant_jj2();
    if jj2 > TOL_J2 {
        deriv1_invariant_jj3(d1, s, sigma).unwrap(); // d1 := dJ3/dσ
        let jj3 = sigma.invariant_jj3();
        let a = 1.5 * SQRT_3 / f64::powf(jj2, 1.5);
        let b = 2.25 * SQRT_3 / f64::powf(jj2, 2.5);
        for i in 0..dim {
            d1.vec[i] = a * d1.vec[i] - b * jj3 * s.vec[i];
        }
        return Ok(Some(jj2));
    }
    Ok(None)
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Tensor2;
    use crate::{
        deriv1_invariant_jj2, deriv1_invariant_jj3, deriv1_invariant_lode, deriv1_invariant_sigma_d,
        deriv1_invariant_sigma_m, deriv1_norm, Mandel, SampleTensor2, SamplesTensor2,
    };
    use russell_lab::{deriv_central5, mat_approx_eq, Matrix};

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
        let _ = a.clone();
    }

    // computes the analytical derivative df(σ)/dσ
    fn analytical_deriv(fn_name: F, d1: &mut Tensor2, sigma: &Tensor2) {
        match fn_name {
            F::Norm => {
                deriv1_norm(d1, sigma).unwrap().unwrap();
            }
            F::J2 => deriv1_invariant_jj2(d1, sigma).unwrap(),
            F::J3 => {
                let mut s = Tensor2::new(sigma.mandel());
                deriv1_invariant_jj3(d1, &mut s, sigma).unwrap();
            }
            F::SigmaM => deriv1_invariant_sigma_m(d1, sigma).unwrap(),
            F::SigmaD => {
                deriv1_invariant_sigma_d(d1, sigma).unwrap().unwrap();
            }
            F::Lode => {
                let mut s = Tensor2::new(sigma.mandel());
                deriv1_invariant_lode(d1, &mut s, sigma).unwrap().unwrap();
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
    fn check_deriv(fn_name: F, mandel: Mandel, sample: &SampleTensor2, tol: f64, _verbose: bool) {
        let sigma = Tensor2::from_matrix(&sample.matrix, mandel).unwrap();
        let mut d1 = Tensor2::new(mandel);
        analytical_deriv(fn_name, &mut d1, &sigma);
        let ana = d1.to_matrix();
        let num = numerical_deriv(&sigma, fn_name);
        let num_mandel = numerical_deriv_mandel(&sigma, fn_name);
        /*
        if verbose {
            println!("analytical derivative:\n{}", ana);
            println!("numerical derivative:\n{}", num);
            println!("numerical derivative (Mandel):\n{}", num_mandel);
        }
        */
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
        let sigma = Tensor2::from_matrix(&SamplesTensor2::TENSOR_O.matrix, Mandel::Symmetric).unwrap();
        let mut d1 = Tensor2::new(Mandel::Symmetric);
        let mut s = Tensor2::new(Mandel::Symmetric);
        assert_eq!(deriv1_norm(&mut d1, &sigma).unwrap(), None);
        assert_eq!(deriv1_invariant_sigma_d(&mut d1, &sigma).unwrap(), None);
        assert_eq!(deriv1_invariant_lode(&mut d1, &mut s, &sigma).unwrap(), None);
    }

    #[test]
    fn check_errors() {
        let sigma_gen = Tensor2::new(Mandel::General);
        let sigma_sym = Tensor2::new(Mandel::Symmetric);
        let mut s_gen = Tensor2::new(Mandel::General);
        let mut s_sym = Tensor2::new(Mandel::Symmetric);
        let mut d1_gen = Tensor2::new(Mandel::General);
        assert_eq!(
            deriv1_norm(&mut d1_gen, &sigma_sym).err(),
            Some("sigma and d1 tensors are incompatible")
        );
        assert_eq!(
            deriv1_invariant_jj2(&mut d1_gen, &sigma_gen).err(),
            Some("sigma tensor must be Symmetric or Symmetric2D")
        );
        assert_eq!(
            deriv1_invariant_jj2(&mut d1_gen, &sigma_sym).err(),
            Some("tensors are incompatible")
        );
        assert_eq!(
            deriv1_invariant_jj3(&mut d1_gen, &mut s_gen, &sigma_gen).err(),
            Some("sigma tensor must be Symmetric or Symmetric2D")
        );
        assert_eq!(
            deriv1_invariant_jj3(&mut d1_gen, &mut s_gen, &sigma_sym).err(),
            Some("s tensor is incompatible")
        );
        assert_eq!(
            deriv1_invariant_jj3(&mut d1_gen, &mut s_sym, &sigma_sym).err(),
            Some("d1 tensor is incompatible")
        );
        assert_eq!(
            deriv1_invariant_sigma_m(&mut d1_gen, &sigma_sym).err(),
            Some("sigma and d1 tensors are incompatible")
        );
        assert_eq!(
            deriv1_invariant_sigma_d(&mut d1_gen, &sigma_gen).err(),
            Some("sigma tensor must be Symmetric or Symmetric2D")
        );
        assert_eq!(
            deriv1_invariant_sigma_d(&mut d1_gen, &sigma_sym).err(),
            Some("d1 tensor is incompatible")
        );
        assert_eq!(
            deriv1_invariant_lode(&mut d1_gen, &mut s_gen, &sigma_gen).err(),
            Some("sigma tensor must be Symmetric or Symmetric2D")
        );
        assert_eq!(
            deriv1_invariant_lode(&mut d1_gen, &mut s_gen, &sigma_sym).err(),
            Some("s tensor is incompatible")
        );
        assert_eq!(
            deriv1_invariant_lode(&mut d1_gen, &mut s_sym, &sigma_sym).err(),
            Some("d1 tensor is incompatible")
        );
    }
}
