use crate::{Tensor2, ONE_BY_3, SQRT_3, TOL_J2, TWO_BY_3};

#[allow(unused)]
use crate::Mandel; // for documentation

/// Calculates the first derivative of the norm w.r.t. the defining Tensor2
///
/// ```text
/// d‖σ‖    σ
/// ──── = ───
///  dσ    ‖σ‖
/// ```
///
/// # Output
///
/// If `‖σ‖ > 0`, returns `‖σ‖`; otherwise, returns `None`.
///
/// * `d1` -- a tensor to hold the resulting derivative; with the same [Mandel] as `sigma`
///
/// # Input
///
/// * `sigma` -- the tensor; with the same [Mandel] as `d1`
///
/// # Panics
///
/// A panic will occur if the tensors have different [Mandel].
pub fn deriv1_norm(d1: &mut Tensor2, sigma: &Tensor2) -> Option<f64> {
    assert_eq!(d1.mandel, sigma.mandel);
    let dim = d1.vec.dim();
    let n = sigma.norm();
    if n > 0.0 {
        d1.set_tensor(1.0, sigma);
        for i in 0..dim {
            d1.vec[i] /= n;
        }
        return Some(n);
    }
    None
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
/// # Output
///
/// * `d1` -- a tensor to hold the resulting derivative; with the same [Mandel] as `sigma`
///
/// # Input
///
/// * `sigma` -- the [Mandel::Symmetric] or [Mandel::Symmetric2D] tensor; with the same [Mandel] as `d1`
///
/// # Panics
///
/// 1. A panic will occur if `sigma` is not symmetric.
/// 2. A panic will occur if the tensors have different [Mandel].
pub fn deriv1_invariant_jj2(d1: &mut Tensor2, sigma: &Tensor2) {
    assert!(sigma.mandel.symmetric());
    assert_eq!(d1.mandel, sigma.mandel);
    sigma.deviator(d1);
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
/// # Output
///
/// * `d1` -- a tensor to hold the resulting derivative; with the same [Mandel] as `sigma`
/// * `s` -- the resulting deviator tensor; with the same [Mandel] as `sigma`
///
/// # Input
///
/// * `sigma` -- the [Mandel::Symmetric] or [Mandel::Symmetric2D] tensor; with the same [Mandel] as `d1` and `s`
///
/// # Panics
///
/// 1. A panic will occur if `sigma` is not symmetric.
/// 2. A panic will occur if the tensors have different [Mandel].
pub fn deriv1_invariant_jj3(d1: &mut Tensor2, s: &mut Tensor2, sigma: &Tensor2) {
    assert!(sigma.mandel.symmetric());
    assert_eq!(d1.mandel, sigma.mandel);
    assert_eq!(s.mandel, sigma.mandel);
    let jj2 = sigma.invariant_jj2();
    sigma.deviator(s);
    s.squared(d1);
    d1.vec[0] -= TWO_BY_3 * jj2;
    d1.vec[1] -= TWO_BY_3 * jj2;
    d1.vec[2] -= TWO_BY_3 * jj2;
}

/// Calculates the first derivative of the mean stress invariant w.r.t. the stress tensor
///
/// ```text
/// dσm   1
/// ─── = ─ I
///  dσ   3
/// ```
///
/// # Output
///
/// * `d1` -- a tensor to hold the resulting derivative; with the same [Mandel] as `sigma`
///
/// # Input
///
/// * `sigma` -- the tensor; with the same [Mandel] as `d1`
///
/// # Panics
///
/// A panic will occur if the tensors have different [Mandel].
pub fn deriv1_invariant_sigma_m(d1: &mut Tensor2, sigma: &Tensor2) {
    assert_eq!(d1.mandel, sigma.mandel);
    let dim = d1.vec.dim();
    d1.vec[0] = ONE_BY_3;
    d1.vec[1] = ONE_BY_3;
    d1.vec[2] = ONE_BY_3;
    for i in 3..dim {
        d1.vec[i] = 0.0;
    }
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
/// # Output
///
/// * If `J2 > TOL_J2`, returns `J2`; otherwise, returns `None`.
/// * `d1` -- a tensor to hold the resulting derivative; with the same [Mandel] as `sigma`
///
/// # Input
///
/// * `sigma` -- the [Mandel::Symmetric] or [Mandel::Symmetric2D] tensor; with the same [Mandel] as `d1`
///
/// # Panics
///
/// 1. A panic will occur if `sigma` is not symmetric.
/// 2. A panic will occur if the tensors have different [Mandel].
pub fn deriv1_invariant_sigma_d(d1: &mut Tensor2, sigma: &Tensor2) -> Option<f64> {
    assert!(sigma.mandel.symmetric());
    assert_eq!(d1.mandel, sigma.mandel);
    let dim = sigma.vec.dim();
    let jj2 = sigma.invariant_jj2();
    if jj2 > TOL_J2 {
        let a = 0.5 * SQRT_3 / f64::powf(jj2, 0.5);
        deriv1_invariant_jj2(d1, sigma);
        for i in 0..dim {
            d1.vec[i] *= a;
        }
        return Some(jj2);
    }
    None
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
/// # Output
///
/// * If `J2 > TOL_J2`, returns `J2`; otherwise, returns `None`.
/// * `d1` -- a tensor to hold the resulting derivative; with the same [Mandel] as `sigma`
/// * `s` -- the resulting deviator tensor; with the same [Mandel] as `sigma`
///
/// # Input
///
/// * `sigma` -- the [Mandel::Symmetric] or [Mandel::Symmetric2D] tensor; with the same [Mandel] as `d1`
///
/// # Panics
///
/// 1. A panic will occur if `sigma` is not symmetric.
/// 2. A panic will occur if the tensors have different [Mandel].
pub fn deriv1_invariant_lode(d1: &mut Tensor2, s: &mut Tensor2, sigma: &Tensor2) -> Option<f64> {
    assert!(sigma.mandel.symmetric());
    assert_eq!(d1.mandel, sigma.mandel);
    assert_eq!(s.mandel, sigma.mandel);
    let dim = sigma.vec.dim();
    let jj2 = sigma.invariant_jj2();
    if jj2 > TOL_J2 {
        deriv1_invariant_jj3(d1, s, sigma); // d1 := dJ3/dσ
        let jj3 = sigma.invariant_jj3();
        let a = 1.5 * SQRT_3 / f64::powf(jj2, 1.5);
        let b = 2.25 * SQRT_3 / f64::powf(jj2, 2.5);
        for i in 0..dim {
            d1.vec[i] = a * d1.vec[i] - b * jj3 * s.vec[i];
        }
        return Some(jj2);
    }
    None
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SampleTensor2, SamplesTensor2, StrError};
    use russell_lab::{deriv1_central5, mat_approx_eq, Matrix};

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
                deriv1_norm(d1, sigma).unwrap();
            }
            F::J2 => deriv1_invariant_jj2(d1, sigma),
            F::J3 => {
                let mut s = Tensor2::new(sigma.mandel);
                deriv1_invariant_jj3(d1, &mut s, sigma);
            }
            F::SigmaM => deriv1_invariant_sigma_m(d1, sigma),
            F::SigmaD => {
                deriv1_invariant_sigma_d(d1, sigma).unwrap();
            }
            F::Lode => {
                let mut s = Tensor2::new(sigma.mandel);
                deriv1_invariant_lode(d1, &mut s, sigma).unwrap();
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
    fn f_sigma(x: f64, args: &mut ArgsNumDeriv) -> Result<f64, StrError> {
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
        Ok(res)
    }

    // computes f(σ) for varying components x = σₘ
    fn f_sigma_mandel(x: f64, args: &mut ArgsNumDerivMandel) -> Result<f64, StrError> {
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
        Ok(res)
    }

    // computes ∂f/∂σᵢⱼ and returns as a 3x3 matrix of (standard) components
    fn numerical_deriv(sigma: &Tensor2, fn_name: F) -> Matrix {
        let mut args = ArgsNumDeriv {
            fn_name,
            sigma_mat: sigma.as_matrix(),
            sigma: sigma.as_general(),
            i: 0,
            j: 0,
        };
        let mut num_deriv = Matrix::new(3, 3);
        for i in 0..3 {
            args.i = i;
            for j in 0..3 {
                args.j = j;
                let x = args.sigma_mat.get(i, j);
                let res = deriv1_central5(x, &mut args, f_sigma).unwrap();
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
            let res = deriv1_central5(x, &mut args, f_sigma_mandel).unwrap();
            num_deriv.vec[m] = res;
        }
        num_deriv.as_matrix()
    }

    // checks ∂f/∂σᵢⱼ
    fn check_deriv(fn_name: F, mandel: Mandel, sample: &SampleTensor2, tol: f64, _verbose: bool) {
        let sigma = Tensor2::from_matrix(&sample.matrix, mandel).unwrap();
        let mut d1 = Tensor2::new(mandel);
        analytical_deriv(fn_name, &mut d1, &sigma);
        let ana = d1.as_matrix();
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

    #[test]
    fn check_for_none() {
        let sigma = Tensor2::from_matrix(&SamplesTensor2::TENSOR_O.matrix, Mandel::Symmetric).unwrap();
        let mut d1 = Tensor2::new(Mandel::Symmetric);
        let mut s = Tensor2::new(Mandel::Symmetric);
        assert_eq!(deriv1_norm(&mut d1, &sigma), None);
        assert_eq!(deriv1_invariant_sigma_d(&mut d1, &sigma), None);
        assert_eq!(deriv1_invariant_lode(&mut d1, &mut s, &sigma), None);
    }

    // check assertions -----------------------------------------------------------------------------

    #[test]
    #[should_panic]
    fn deriv1_norm_panics_on_different_mandel() {
        let mut d1_gen = Tensor2::new(Mandel::General);
        let sigma_sym = Tensor2::new(Mandel::Symmetric);
        deriv1_norm(&mut d1_gen, &sigma_sym);
    }

    #[test]
    #[should_panic(expected = "sigma.mandel.symmetric()")]
    fn deriv1_invariant_jj2_panics_on_on_sym() {
        let mut d1_gen = Tensor2::new(Mandel::General);
        let sigma_gen = Tensor2::new(Mandel::General);
        deriv1_invariant_jj2(&mut d1_gen, &sigma_gen);
    }

    #[test]
    #[should_panic]
    fn deriv1_invariant_jj2_panics_on_different_mandel() {
        let mut d1_gen = Tensor2::new(Mandel::General);
        let sigma_sym = Tensor2::new(Mandel::Symmetric);
        deriv1_invariant_jj2(&mut d1_gen, &sigma_sym);
    }

    #[test]
    #[should_panic(expected = "sigma.mandel.symmetric()")]
    fn deriv1_invariant_jj3_panics_on_non_sym() {
        let mut d1_gen = Tensor2::new(Mandel::General);
        let mut s_gen = Tensor2::new(Mandel::General);
        let sigma_gen = Tensor2::new(Mandel::General);
        deriv1_invariant_jj3(&mut d1_gen, &mut s_gen, &sigma_gen);
    }

    #[test]
    #[should_panic]
    fn deriv1_invariant_jj3_panics_on_different_mandel1() {
        let mut d1_gen = Tensor2::new(Mandel::General);
        let mut s_gen = Tensor2::new(Mandel::General);
        let sigma_sym = Tensor2::new(Mandel::Symmetric);
        deriv1_invariant_jj3(&mut d1_gen, &mut s_gen, &sigma_sym);
    }

    #[test]
    #[should_panic]
    fn deriv1_invariant_jj3_panics_on_different_mandel2() {
        let mut d1_gen = Tensor2::new(Mandel::General);
        let mut s_sym = Tensor2::new(Mandel::Symmetric);
        let sigma_sym = Tensor2::new(Mandel::Symmetric);
        deriv1_invariant_jj3(&mut d1_gen, &mut s_sym, &sigma_sym);
    }

    #[test]
    #[should_panic]
    fn deriv1_invariant_sigma_m_panics_on_different_mandel() {
        let mut d1_gen = Tensor2::new(Mandel::General);
        let sigma_sym = Tensor2::new(Mandel::Symmetric);
        deriv1_invariant_sigma_m(&mut d1_gen, &sigma_sym);
    }

    #[test]
    #[should_panic(expected = "sigma.mandel.symmetric()")]
    fn deriv1_invariant_sigma_d_panics_on_non_sym() {
        let mut d1_gen = Tensor2::new(Mandel::General);
        let sigma_gen = Tensor2::new(Mandel::General);
        deriv1_invariant_sigma_d(&mut d1_gen, &sigma_gen);
    }

    #[test]
    #[should_panic]
    fn deriv1_invariant_sigma_d_panics_on_different_mandel() {
        let mut d1_gen = Tensor2::new(Mandel::General);
        let sigma_sym = Tensor2::new(Mandel::Symmetric);
        deriv1_invariant_sigma_d(&mut d1_gen, &sigma_sym);
    }

    #[test]
    #[should_panic(expected = "sigma.mandel.symmetric()")]
    fn deriv1_invariant_lode_panics_on_non_sym() {
        let mut d1_gen = Tensor2::new(Mandel::General);
        let mut s_gen = Tensor2::new(Mandel::General);
        let sigma_gen = Tensor2::new(Mandel::General);
        deriv1_invariant_lode(&mut d1_gen, &mut s_gen, &sigma_gen);
    }

    #[test]
    #[should_panic]
    fn deriv1_invariant_lode_panics_on_different_mandel1() {
        let mut d1_gen = Tensor2::new(Mandel::General);
        let mut s_gen = Tensor2::new(Mandel::General);
        let sigma_sym = Tensor2::new(Mandel::Symmetric);
        deriv1_invariant_lode(&mut d1_gen, &mut s_gen, &sigma_sym);
    }

    #[test]
    #[should_panic]
    fn deriv1_invariant_lode_panics_on_different_mandel2() {
        let mut d1_gen = Tensor2::new(Mandel::General);
        let mut s_sym = Tensor2::new(Mandel::Symmetric);
        let sigma_sym = Tensor2::new(Mandel::Symmetric);
        deriv1_invariant_lode(&mut d1_gen, &mut s_sym, &sigma_sym);
    }
}
