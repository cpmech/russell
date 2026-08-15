use crate::{ONE_BY_3, SQRT_3, TOL_J2, TWO_BY_3, Tensor2};

#[allow(unused)]
use crate::Rep; // for documentation

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
/// * `d1` -- a tensor to hold the resulting derivative; with the same [Rep] as `sigma`
///
/// # Input
///
/// * `sigma` -- the tensor; with the same [Rep] as `d1`
///
/// # Panics
///
/// A panic will occur if the tensors have different [Rep].
pub fn deriv1_norm(d1: &mut Tensor2, sigma: &Tensor2) -> Option<f64> {
    assert_eq!(d1.rep, sigma.rep);
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
/// * `d1` -- a tensor to hold the resulting derivative; with the same [Rep] as `sigma`
///
/// # Input
///
/// * `sigma` -- the [Rep::Symmetric] or [Rep::Symmetric2D] tensor; with the same [Rep] as `d1`
///
/// # Panics
///
/// 1. A panic will occur if `sigma` is not symmetric.
/// 2. A panic will occur if the tensors have different [Rep].
#[inline]
pub fn deriv1_invariant_jj2(d1: &mut Tensor2, sigma: &Tensor2) {
    assert!(sigma.rep.symmetric());
    assert_eq!(d1.rep, sigma.rep);
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
/// * `d1` -- a tensor to hold the resulting derivative; with the same [Rep] as `sigma`
/// * `s` -- the resulting deviator tensor; with the same [Rep] as `sigma`
///
/// # Input
///
/// * `sigma` -- the [Rep::Symmetric] or [Rep::Symmetric2D] tensor; with the same [Rep] as `d1` and `s`
///
/// # Panics
///
/// 1. A panic will occur if `sigma` is not symmetric.
/// 2. A panic will occur if the tensors have different [Rep].
#[inline]
pub fn deriv1_invariant_jj3(d1: &mut Tensor2, s: &mut Tensor2, sigma: &Tensor2) {
    assert!(sigma.rep.symmetric());
    assert_eq!(d1.rep, sigma.rep);
    assert_eq!(s.rep, sigma.rep);
    let jj2 = sigma.invariant_jj2();
    sigma.deviator(s);
    s.squared(d1);
    d1.vec[0] -= TWO_BY_3 * jj2;
    d1.vec[1] -= TWO_BY_3 * jj2;
    d1.vec[2] -= TWO_BY_3 * jj2;
}

/// Calculates the first derivative of σs w.r.t. the stress tensor
///
/// ```text
/// dσs   1
/// ─── = ── I
/// dσ    √3
/// ```
///
/// # Output
///
/// * `d1` -- a tensor to hold the resulting derivative; with the same [Rep] as `sigma`
///
/// # Input
///
/// * `sigma` -- the tensor; with the same [Rep] as `d1`
///
/// # Panics
///
/// A panic will occur if the tensors have different [Rep].
pub fn deriv1_invariant_sigma_s(d1: &mut Tensor2, sigma: &Tensor2) {
    assert_eq!(d1.rep, sigma.rep);
    let dim = d1.vec.dim();
    d1.vec[0] = 1.0 / SQRT_3;
    d1.vec[1] = 1.0 / SQRT_3;
    d1.vec[2] = 1.0 / SQRT_3;
    for i in 3..dim {
        d1.vec[i] = 0.0;
    }
}

/// Calculates the first derivative of σt w.r.t. the stress tensor
///
/// ```text
/// s = deviator(σ)
///
/// dσt      1    dJ2
/// ─── = ─────── ───
/// dσ    √(2 J2)  dσ
///
/// (σ is symmetric)
/// ```
///
/// # Output
///
/// * If `J2 > TOL_J2`, returns `J2`; otherwise, returns `None`.
/// * `d1` -- a tensor to hold the resulting derivative; with the same [Rep] as `sigma`
///
/// # Input
///
/// * `sigma` -- the [Rep::Symmetric] or [Rep::Symmetric2D] tensor; with the same [Rep] as `d1`
///
/// # Panics
///
/// 1. A panic will occur if `sigma` is not symmetric.
/// 2. A panic will occur if the tensors have different [Rep].
pub fn deriv1_invariant_sigma_t(d1: &mut Tensor2, sigma: &Tensor2) -> Option<f64> {
    assert!(sigma.rep.symmetric());
    assert_eq!(d1.rep, sigma.rep);
    let dim = sigma.vec.dim();
    let jj2 = sigma.invariant_jj2();
    if jj2 > TOL_J2 {
        let a = 1.0 / f64::sqrt(2.0 * jj2);
        deriv1_invariant_jj2(d1, sigma);
        for i in 0..dim {
            d1.vec[i] *= a;
        }
        return Some(jj2);
    }
    None
}

/// Calculates the first derivative of p w.r.t. the stress tensor
///
/// ```text
/// dp   1
/// ── = ─ I
/// dσ   3
/// ```
///
/// # Output
///
/// * `d1` -- a tensor to hold the resulting derivative; with the same [Rep] as `sigma`
///
/// # Input
///
/// * `sigma` -- the tensor; with the same [Rep] as `d1`
///
/// # Panics
///
/// A panic will occur if the tensors have different [Rep].
pub fn deriv1_invariant_p(d1: &mut Tensor2, sigma: &Tensor2) {
    assert_eq!(d1.rep, sigma.rep);
    let dim = d1.vec.dim();
    d1.vec[0] = ONE_BY_3;
    d1.vec[1] = ONE_BY_3;
    d1.vec[2] = ONE_BY_3;
    for i in 3..dim {
        d1.vec[i] = 0.0;
    }
}

/// Calculates the first derivative of q (von Mises) w.r.t. the stress tensor
///
/// ```text
/// s = deviator(σ)
///
/// dq     √3  dJ2
/// ── = ───── ───
/// dσ   2 √J2  dσ
///
/// (σ is symmetric)
/// ```
///
/// # Output
///
/// * If `J2 > TOL_J2`, returns `J2`; otherwise, returns `None`.
/// * `d1` -- a tensor to hold the resulting derivative; with the same [Rep] as `sigma`
///
/// # Input
///
/// * `sigma` -- the [Rep::Symmetric] or [Rep::Symmetric2D] tensor; with the same [Rep] as `d1`
///
/// # Panics
///
/// 1. A panic will occur if `sigma` is not symmetric.
/// 2. A panic will occur if the tensors have different [Rep].
pub fn deriv1_invariant_q(d1: &mut Tensor2, sigma: &Tensor2) -> Option<f64> {
    assert!(sigma.rep.symmetric());
    assert_eq!(d1.rep, sigma.rep);
    let dim = sigma.vec.dim();
    let jj2 = sigma.invariant_jj2();
    if jj2 > TOL_J2 {
        let a = 0.5 * SQRT_3 / f64::sqrt(jj2);
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
/// * `d1` -- a tensor to hold the resulting derivative; with the same [Rep] as `sigma`
/// * `s` -- the resulting deviator tensor; with the same [Rep] as `sigma`
///
/// # Input
///
/// * `sigma` -- the [Rep::Symmetric] or [Rep::Symmetric2D] tensor; with the same [Rep] as `d1`
///
/// # Panics
///
/// 1. A panic will occur if `sigma` is not symmetric.
/// 2. A panic will occur if the tensors have different [Rep].
pub fn deriv1_invariant_lode(d1: &mut Tensor2, s: &mut Tensor2, sigma: &Tensor2) -> Option<f64> {
    assert!(sigma.rep.symmetric());
    assert_eq!(d1.rep, sigma.rep);
    assert_eq!(s.rep, sigma.rep);
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
    use russell_lab::{Matrix, deriv1_central5, mat_approx_eq};

    // Defines f(σ)
    #[derive(Clone, Copy)]
    enum F {
        Norm,
        J2,
        J3,
        SigmaS, // σs
        SigmaT, // σt
        P,
        Q,
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
                let mut s = Tensor2::new(sigma.rep);
                deriv1_invariant_jj3(d1, &mut s, sigma);
            }
            F::SigmaS => deriv1_invariant_sigma_s(d1, sigma),
            F::SigmaT => {
                deriv1_invariant_sigma_t(d1, sigma).unwrap();
            }
            F::P => deriv1_invariant_p(d1, sigma),
            F::Q => {
                deriv1_invariant_q(d1, sigma).unwrap();
            }
            F::Lode => {
                let mut s = Tensor2::new(sigma.rep);
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

    // Holds arguments for numerical differentiation of a scalar f(σ) w.r.t. σₘ (matrix representation)
    struct ArgsNumDerivM {
        fn_name: F,     // name of f(σ)
        sigma: Tensor2, // @ σ, with varying m-components
        m: usize,       // index m of ∂f/∂σₘ
    }

    // computes f(σ) for varying components x = σᵢⱼ
    fn f_sigma(x: f64, args: &mut ArgsNumDeriv) -> Result<f64, StrError> {
        let original = args.sigma_mat.get(args.i, args.j);
        args.sigma_mat.set(args.i, args.j, x);
        args.sigma.set_std_matrix(&args.sigma_mat).unwrap();
        let res = match args.fn_name {
            F::Norm => args.sigma.norm(),
            F::J2 => args.sigma.invariant_jj2(),
            F::J3 => args.sigma.invariant_jj3(),
            F::SigmaS => args.sigma.invariant_sigma_s(),
            F::SigmaT => args.sigma.invariant_sigma_t(),
            F::P => args.sigma.invariant_p(),
            F::Q => args.sigma.invariant_q(),
            F::Lode => args.sigma.invariant_lode().unwrap(),
        };
        args.sigma_mat.set(args.i, args.j, original);
        Ok(res)
    }

    // computes f(σ) for varying components x = σₘ
    fn f_sigma_mat(x: f64, args: &mut ArgsNumDerivM) -> Result<f64, StrError> {
        let original = args.sigma.vec[args.m];
        args.sigma.vec[args.m] = x;
        let res = match args.fn_name {
            F::Norm => args.sigma.norm(),
            F::J2 => args.sigma.invariant_jj2(),
            F::J3 => args.sigma.invariant_jj3(),
            F::SigmaS => args.sigma.invariant_sigma_s(),
            F::SigmaT => args.sigma.invariant_sigma_t(),
            F::P => args.sigma.invariant_p(),
            F::Q => args.sigma.invariant_q(),
            F::Lode => args.sigma.invariant_lode().unwrap(),
        };
        args.sigma.vec[args.m] = original;
        Ok(res)
    }

    // computes ∂f/∂σᵢⱼ and returns as a 3x3 matrix of (standard) components
    fn numerical_deriv(sigma: &Tensor2, fn_name: F) -> Matrix {
        let mut args = ArgsNumDeriv {
            fn_name,
            sigma_mat: sigma.as_std_matrix(),
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
    fn numerical_deriv_mat(sigma: &Tensor2, fn_name: F) -> Matrix {
        let mut args = ArgsNumDerivM {
            fn_name,
            sigma: sigma.clone(),
            m: 0,
        };
        let mut num_deriv = sigma.clone();
        for m in 0..sigma.vec.dim() {
            args.m = m;
            let x = args.sigma.vec[m];
            let res = deriv1_central5(x, &mut args, f_sigma_mat).unwrap();
            num_deriv.vec[m] = res;
        }
        num_deriv.as_std_matrix()
    }

    // checks ∂f/∂σᵢⱼ
    fn check_deriv(fn_name: F, rep: Rep, sample: &SampleTensor2, tol: f64, _verbose: bool) {
        let sigma = Tensor2::from_std_matrix(&sample.matrix, rep).unwrap();
        let mut d1 = Tensor2::new(rep);
        analytical_deriv(fn_name, &mut d1, &sigma);
        let ana = d1.as_std_matrix();
        let num = numerical_deriv(&sigma, fn_name);
        let num_mat = numerical_deriv_mat(&sigma, fn_name);
        /*
        if verbose {
            println!("analytical derivative:\n{}", ana);
            println!("numerical derivative:\n{}", num);
            println!("numerical derivative (Rep):\n{}", num_mat);
        }
        */
        mat_approx_eq(&ana, &num, tol);
        mat_approx_eq(&ana, &num_mat, tol);
    }

    #[test]
    fn deriv_norm_works() {
        let v = false;
        check_deriv(F::Norm, Rep::General, &SamplesTensor2::TENSOR_T, 1e-10, v);
        check_deriv(F::Norm, Rep::Symmetric, &SamplesTensor2::TENSOR_S, 1e-10, v);
        check_deriv(F::Norm, Rep::Symmetric2D, &SamplesTensor2::TENSOR_Z, 1e-11, v);
    }

    #[test]
    fn deriv_invariant_jj2_works() {
        let v = false;
        check_deriv(F::J2, Rep::Symmetric, &SamplesTensor2::TENSOR_S, 1e-10, v);
        check_deriv(F::J2, Rep::Symmetric2D, &SamplesTensor2::TENSOR_Z, 1e-11, v);
        check_deriv(F::J2, Rep::Symmetric2D, &SamplesTensor2::TENSOR_O, 1e-15, v);
        check_deriv(F::J2, Rep::Symmetric2D, &SamplesTensor2::TENSOR_I, 1e-12, v);
    }

    #[test]
    fn deriv_invariant_jj3_works() {
        let v = false;
        check_deriv(F::J3, Rep::Symmetric, &SamplesTensor2::TENSOR_S, 1e-9, v);
        check_deriv(F::J3, Rep::Symmetric2D, &SamplesTensor2::TENSOR_Z, 1e-10, v);
        check_deriv(F::J3, Rep::Symmetric2D, &SamplesTensor2::TENSOR_O, 1e-15, v);
        check_deriv(F::J3, Rep::Symmetric2D, &SamplesTensor2::TENSOR_I, 1e-15, v);
    }

    #[test]
    fn deriv_sigma_s_works() {
        let v = false;
        check_deriv(F::SigmaS, Rep::General, &SamplesTensor2::TENSOR_T, 1e-11, v);
        check_deriv(F::SigmaS, Rep::Symmetric, &SamplesTensor2::TENSOR_S, 1e-11, v);
        check_deriv(F::SigmaS, Rep::Symmetric2D, &SamplesTensor2::TENSOR_Z, 1e-11, v);
    }

    #[test]
    fn deriv_sigma_t_works() {
        let v = false;
        check_deriv(F::SigmaT, Rep::Symmetric, &SamplesTensor2::TENSOR_U, 1e-10, v);
        check_deriv(F::SigmaT, Rep::Symmetric, &SamplesTensor2::TENSOR_S, 1e-10, v);
        check_deriv(F::SigmaT, Rep::Symmetric2D, &SamplesTensor2::TENSOR_X, 1e-11, v);
        check_deriv(F::SigmaT, Rep::Symmetric2D, &SamplesTensor2::TENSOR_Y, 1e-10, v);
        check_deriv(F::SigmaT, Rep::Symmetric2D, &SamplesTensor2::TENSOR_Z, 1e-10, v);
    }

    #[test]
    fn deriv_p_works() {
        let v = false;
        check_deriv(F::P, Rep::General, &SamplesTensor2::TENSOR_T, 1e-12, v);
        check_deriv(F::P, Rep::Symmetric, &SamplesTensor2::TENSOR_S, 1e-11, v);
        check_deriv(F::P, Rep::Symmetric2D, &SamplesTensor2::TENSOR_Z, 1e-12, v);
    }

    #[test]
    fn deriv_q_works() {
        let v = false;
        check_deriv(F::Q, Rep::Symmetric, &SamplesTensor2::TENSOR_U, 1e-10, v);
        check_deriv(F::Q, Rep::Symmetric, &SamplesTensor2::TENSOR_S, 1e-10, v);
        check_deriv(F::Q, Rep::Symmetric2D, &SamplesTensor2::TENSOR_X, 1e-11, v);
        check_deriv(F::Q, Rep::Symmetric2D, &SamplesTensor2::TENSOR_Y, 1e-10, v);
        check_deriv(F::Q, Rep::Symmetric2D, &SamplesTensor2::TENSOR_Z, 1e-10, v);
    }

    #[test]
    fn deriv_invariant_lode_works() {
        let v = false;
        check_deriv(F::Lode, Rep::Symmetric, &SamplesTensor2::TENSOR_U, 1e-10, v);
        check_deriv(F::Lode, Rep::Symmetric, &SamplesTensor2::TENSOR_S, 1e-10, v);
        check_deriv(F::Lode, Rep::Symmetric2D, &SamplesTensor2::TENSOR_X, 1e-10, v);
        check_deriv(F::Lode, Rep::Symmetric2D, &SamplesTensor2::TENSOR_Y, 1e-10, v);
        check_deriv(F::Lode, Rep::Symmetric2D, &SamplesTensor2::TENSOR_Z, 1e-10, v);
    }

    #[test]
    fn check_for_none() {
        let sigma = Tensor2::from_std_matrix(&SamplesTensor2::TENSOR_O.matrix, Rep::Symmetric).unwrap();
        let mut d1 = Tensor2::new(Rep::Symmetric);
        let mut s = Tensor2::new(Rep::Symmetric);
        assert_eq!(deriv1_norm(&mut d1, &sigma), None);
        assert_eq!(deriv1_invariant_q(&mut d1, &sigma), None);
        assert_eq!(deriv1_invariant_lode(&mut d1, &mut s, &sigma), None);
    }

    // check assertions -----------------------------------------------------------------------------

    #[test]
    #[should_panic]
    fn deriv1_norm_panics_on_different_mat() {
        let mut d1_gen = Tensor2::new(Rep::General);
        let sigma_sym = Tensor2::new(Rep::Symmetric);
        deriv1_norm(&mut d1_gen, &sigma_sym);
    }

    #[test]
    #[should_panic(expected = "sigma.rep.symmetric()")]
    fn deriv1_invariant_jj2_panics_on_on_sym() {
        let mut d1_gen = Tensor2::new(Rep::General);
        let sigma_gen = Tensor2::new(Rep::General);
        deriv1_invariant_jj2(&mut d1_gen, &sigma_gen);
    }

    #[test]
    #[should_panic]
    fn deriv1_invariant_jj2_panics_on_different_mat() {
        let mut d1_gen = Tensor2::new(Rep::General);
        let sigma_sym = Tensor2::new(Rep::Symmetric);
        deriv1_invariant_jj2(&mut d1_gen, &sigma_sym);
    }

    #[test]
    #[should_panic(expected = "sigma.rep.symmetric()")]
    fn deriv1_invariant_jj3_panics_on_non_sym() {
        let mut d1_gen = Tensor2::new(Rep::General);
        let mut s_gen = Tensor2::new(Rep::General);
        let sigma_gen = Tensor2::new(Rep::General);
        deriv1_invariant_jj3(&mut d1_gen, &mut s_gen, &sigma_gen);
    }

    #[test]
    #[should_panic]
    fn deriv1_invariant_jj3_panics_on_different_mat() {
        let mut d1_gen = Tensor2::new(Rep::General);
        let mut s_gen = Tensor2::new(Rep::General);
        let sigma_sym = Tensor2::new(Rep::Symmetric);
        deriv1_invariant_jj3(&mut d1_gen, &mut s_gen, &sigma_sym);
    }

    #[test]
    #[should_panic]
    fn deriv1_invariant_jj3_panics_on_different_mat2() {
        let mut d1_gen = Tensor2::new(Rep::General);
        let mut s_sym = Tensor2::new(Rep::Symmetric);
        let sigma_sym = Tensor2::new(Rep::Symmetric);
        deriv1_invariant_jj3(&mut d1_gen, &mut s_sym, &sigma_sym);
    }

    #[test]
    #[should_panic]
    fn deriv1_invariant_sigma_m_panics_on_different_mat() {
        let mut d1_gen = Tensor2::new(Rep::General);
        let sigma_sym = Tensor2::new(Rep::Symmetric);
        deriv1_invariant_p(&mut d1_gen, &sigma_sym);
    }

    #[test]
    #[should_panic(expected = "sigma.rep.symmetric()")]
    fn deriv1_invariant_q_panics_on_non_sym() {
        let mut d1_gen = Tensor2::new(Rep::General);
        let sigma_gen = Tensor2::new(Rep::General);
        deriv1_invariant_q(&mut d1_gen, &sigma_gen);
    }

    #[test]
    #[should_panic]
    fn deriv1_invariant_q_panics_on_different_mat() {
        let mut d1_gen = Tensor2::new(Rep::General);
        let sigma_sym = Tensor2::new(Rep::Symmetric);
        deriv1_invariant_q(&mut d1_gen, &sigma_sym);
    }

    #[test]
    #[should_panic(expected = "sigma.rep.symmetric()")]
    fn deriv1_invariant_lode_panics_on_non_sym() {
        let mut d1_gen = Tensor2::new(Rep::General);
        let mut s_gen = Tensor2::new(Rep::General);
        let sigma_gen = Tensor2::new(Rep::General);
        deriv1_invariant_lode(&mut d1_gen, &mut s_gen, &sigma_gen);
    }

    #[test]
    #[should_panic]
    fn deriv1_invariant_lode_panics_on_different_red1() {
        let mut d1_gen = Tensor2::new(Rep::General);
        let mut s_gen = Tensor2::new(Rep::General);
        let sigma_sym = Tensor2::new(Rep::Symmetric);
        deriv1_invariant_lode(&mut d1_gen, &mut s_gen, &sigma_sym);
    }

    #[test]
    #[should_panic]
    fn deriv1_invariant_lode_panics_on_different_red2() {
        let mut d1_gen = Tensor2::new(Rep::General);
        let mut s_sym = Tensor2::new(Rep::Symmetric);
        let sigma_sym = Tensor2::new(Rep::Symmetric);
        deriv1_invariant_lode(&mut d1_gen, &mut s_sym, &sigma_sym);
    }
}
