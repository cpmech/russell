use crate::{ONE_BY_3, SQRT_2, SQRT_3, TOL_J2, TWO_BY_3};
use crate::{Tensor2, squared_tensor_slice};

/// Calculates the first derivative of the norm w.r.t. the defining Tensor2
///
/// ```text
/// d‖T‖    T
/// ──── = ───
///  dT    ‖T‖
/// ```
///
/// # Output
///
/// If `‖T‖ > 0`, returns `‖T‖`; otherwise, returns `None`.
///
/// * `d1` -- a tensor to hold the resulting derivative
///
/// # Input
///
/// * `tt` -- the `T` tensor
pub fn deriv1_norm<const N: usize>(d1: &mut Tensor2<N>, tt: &Tensor2<N>) -> Option<f64> {
    let nrm = tt.norm();
    if nrm > 0.0 {
        d1.set_tensor(1.0, tt);
        for m in 0..N {
            d1.vec[m] /= nrm;
        }
        return Some(nrm);
    }
    None
}

/// Calculates the first derivative of the I2 invariant w.r.t. a symmetric tensor
///
/// ```text
/// dI2
/// ─── = I1 I - a
///  da
///
/// (a is symmetric)
/// ```
///
/// # Output
///
/// * `d1` -- a tensor to hold the resulting derivative
///
/// # Input
///
/// * `a` -- the symmetric tensor
///
/// # Panics
///
/// A panic will occur if `a` is not symmetric, i.e., N = 9.
pub fn deriv1_invariant_ii2<const N: usize>(d1: &mut Tensor2<N>, a: &Tensor2<N>) {
    assert!(N != 9, "the tensor must be symmetric with N = 4 or N = 6");
    if N == 4 {
        d1.vec[0] = a.vec[1] + a.vec[2];
        d1.vec[1] = a.vec[2] + a.vec[0];
        d1.vec[2] = a.vec[0] + a.vec[1];
        d1.vec[3] = -a.vec[3];
    } else {
        d1.vec[0] = a.vec[1] + a.vec[2];
        d1.vec[1] = a.vec[2] + a.vec[0];
        d1.vec[2] = a.vec[0] + a.vec[1];
        d1.vec[3] = -a.vec[3];
        d1.vec[4] = -a.vec[4];
        d1.vec[5] = -a.vec[5];
    }
}

/// Calculates the first derivative of the I3 invariant w.r.t. its defining tensor
///
/// If `a` is symmetric:
///
/// ```text
/// dI3
/// ─── = a² - I1 a + I2 I
///  da
/// ```
///
/// Otherwise, for a general tensor `a`, we use the Levi-Civita `ϵ` form:
///
/// ```text
/// ∂I3/∂a_ij = ½ ϵ_ikl ϵ_jrs a_kr a_ls = (det(a) a⁻ᵀ)_ij
/// ```
///
/// # Output
///
/// * `d1` -- a tensor to hold the resulting derivative
///
/// # Input
///
/// * `a` -- the tensor
///
/// # Notes
///
/// For `N = 9` the result is the general cofactor, which is not symmetric (its
/// KM9 vector differs from the KM6 one).
pub fn deriv1_invariant_ii3<const N: usize>(d1: &mut Tensor2<N>, a: &Tensor2<N>) {
    if N == 4 {
        d1.vec[0] = a.vec[1] * a.vec[2];
        d1.vec[1] = a.vec[2] * a.vec[0];
        d1.vec[2] = a.vec[0] * a.vec[1] - a.vec[3] * a.vec[3] / 2.0;
        d1.vec[3] = -a.vec[2] * a.vec[3];
    } else if N == 6 {
        d1.vec[0] = a.vec[1] * a.vec[2] - a.vec[4] * a.vec[4] / 2.0;
        d1.vec[1] = a.vec[2] * a.vec[0] - a.vec[5] * a.vec[5] / 2.0;
        d1.vec[2] = a.vec[0] * a.vec[1] - a.vec[3] * a.vec[3] / 2.0;
        d1.vec[3] = -a.vec[2] * a.vec[3] + a.vec[4] * a.vec[5] / SQRT_2;
        d1.vec[4] = -a.vec[0] * a.vec[4] + a.vec[5] * a.vec[3] / SQRT_2;
        d1.vec[5] = -a.vec[1] * a.vec[5] + a.vec[3] * a.vec[4] / SQRT_2;
    } else {
        d1.vec[0] = a.vec[7] * a.vec[7] / 2.0 - a.vec[4] * a.vec[4] / 2.0 + a.vec[1] * a.vec[2];
        d1.vec[1] = a.vec[8] * a.vec[8] / 2.0 - a.vec[5] * a.vec[5] / 2.0 + a.vec[0] * a.vec[2];
        d1.vec[2] = a.vec[6] * a.vec[6] / 2.0 - a.vec[3] * a.vec[3] / 2.0 + a.vec[0] * a.vec[1];
        d1.vec[3] = -a.vec[2] * a.vec[3] + a.vec[4] * a.vec[5] / SQRT_2 - a.vec[7] * a.vec[8] / SQRT_2;
        d1.vec[4] = -a.vec[0] * a.vec[4] + a.vec[3] * a.vec[5] / SQRT_2 - a.vec[6] * a.vec[8] / SQRT_2;
        d1.vec[5] = -a.vec[1] * a.vec[5] + a.vec[3] * a.vec[4] / SQRT_2 + a.vec[6] * a.vec[7] / SQRT_2;
        d1.vec[6] = a.vec[2] * a.vec[6] + a.vec[5] * a.vec[7] / SQRT_2 - a.vec[4] * a.vec[8] / SQRT_2;
        d1.vec[7] = a.vec[0] * a.vec[7] + a.vec[5] * a.vec[6] / SQRT_2 - a.vec[3] * a.vec[8] / SQRT_2;
        d1.vec[8] = a.vec[1] * a.vec[8] - a.vec[3] * a.vec[7] / SQRT_2 - a.vec[4] * a.vec[6] / SQRT_2;
    }
}

/// Calculates the first derivative of the J2 invariant w.r.t. the symmetric tensor
///
/// ```text
/// s = deviator(a)
///
/// dJ2
/// ─── = s
///  da
///
/// (a is symmetric)
/// ```
///
/// # Output
///
/// * `d1` -- a tensor to hold the resulting derivative
///
/// # Input
///
/// * `a` -- the symmetric tensor, i.e., N = 4 or N = 6.
///
/// # Panics
///
/// A panic will occur if `a` is not symmetric, i.e., N = 9.
pub fn deriv1_invariant_jj2<const N: usize>(d1: &mut Tensor2<N>, a: &Tensor2<N>) {
    assert!(N != 9, "the tensor must be symmetric with N = 4 or N = 6");
    a.deviator(d1);
}

/// Calculates the first derivative of the J3 invariant w.r.t. the symmetric tensor
///
/// ```text
/// s = deviator(a)
///
/// dJ3         2 J2
/// ─── = s·s - ──── I
///  da           3
///
/// (a is symmetric)
/// ```
///
/// # Output
///
/// * `d1` -- a tensor to hold the resulting derivative
///
/// # Input
///
/// * `a` -- the symmetric tensor, i.e., N = 4 or N = 6.
///
/// # Panics
///
/// A panic will occur if `a` is not symmetric, i.e., N = 9.
pub fn deriv1_invariant_jj3<const N: usize>(d1: &mut Tensor2<N>, a: &Tensor2<N>) {
    assert!(N != 9, "the tensor must be symmetric with N = 4 or N = 6");
    let mut s = [0.0; 6];
    deriv1_invariant_jj3_slice(d1.as_mut_data(), &mut s, a);
}

/// Calculates the first derivative of the J3 invariant (crate-internal)
#[inline]
pub(crate) fn deriv1_invariant_jj3_slice<const N: usize>(d1: &mut [f64], s: &mut [f64], a: &Tensor2<N>) {
    let jj2 = a.invariant_jj2();
    a.deviator_slice(s);
    squared_tensor_slice::<N>(d1, s);
    d1[0] -= TWO_BY_3 * jj2;
    d1[1] -= TWO_BY_3 * jj2;
    d1[2] -= TWO_BY_3 * jj2;
}

/// Calculates the first derivative of d w.r.t. the symmetric tensor
///
/// ```text
/// dd   1
/// ── = ── I
/// da   √3
///
/// (a is symmetric)
/// ```
///
/// # Output
///
/// * `d1` -- a tensor to hold the resulting derivative
///
/// # Input
///
/// * `a` -- the symmetric tensor, i.e., N = 4 or N = 6
///   (it's not actually used here, but kept for consistency)
///
/// # Panics
///
/// A panic will occur if `a` is not symmetric, i.e., N = 9.
pub fn deriv1_invariant_d<const N: usize>(d1: &mut Tensor2<N>, _a: &Tensor2<N>) {
    assert!(N != 9, "the tensor must be symmetric with N = 4 or N = 6");
    d1.vec[0] = 1.0 / SQRT_3;
    d1.vec[1] = 1.0 / SQRT_3;
    d1.vec[2] = 1.0 / SQRT_3;
    for m in 3..N {
        d1.vec[m] = 0.0;
    }
}

/// Calculates the first derivative of r w.r.t. the symmetric tensor
///
/// ```text
/// s = deviator(a)
///
/// dr     dJ2
/// ── = A ───
/// da      da
///
/// (a is symmetric)
/// ```
///
/// ```text
///        1
/// A = ───────
///     √(2 J2)
/// ```
///
/// # Output
///
/// * If `J2 > TOL_J2`, returns `J2`; otherwise, returns `None`.
/// * `d1` -- a tensor to hold the resulting derivative
///
/// # Input
///
/// * `a` -- symmetric tensor, i.e., N = 4 or N = 6.
///
/// # Panics
///
/// A panic will occur if `a` is not symmetric, i.e., N = 9.
pub fn deriv1_invariant_r<const N: usize>(d1: &mut Tensor2<N>, a: &Tensor2<N>) -> Option<f64> {
    assert!(N != 9, "the tensor must be symmetric with N = 4 or N = 6");
    let jj2 = a.invariant_jj2();
    if jj2 > TOL_J2 {
        let aa = 1.0 / f64::sqrt(2.0 * jj2);
        deriv1_invariant_jj2(d1, a);
        for m in 0..N {
            d1.vec[m] *= aa;
        }
        return Some(jj2);
    }
    None
}

/// Calculates the first derivative of p w.r.t. the symmetric tensor
///
/// ```text
/// dp   1
/// ── = ─ I
/// da   3
///
/// (a is symmetric)
/// ```
///
/// # Output
///
/// * `d1` -- a tensor to hold the resulting derivative
///
/// # Input
///
/// * `a` -- symmetric tensor, i.e., N = 4 or N = 6.
///   (it's not actually used here, but kept for consistency)
///
/// # Panics
///
/// A panic will occur if `a` is not symmetric, i.e., N = 9.
pub fn deriv1_invariant_p<const N: usize>(d1: &mut Tensor2<N>, _a: &Tensor2<N>) {
    assert!(N != 9, "the tensor must be symmetric with N = 4 or N = 6");
    d1.vec[0] = ONE_BY_3;
    d1.vec[1] = ONE_BY_3;
    d1.vec[2] = ONE_BY_3;
    for m in 3..N {
        d1.vec[m] = 0.0;
    }
}

/// Calculates the first derivative of q (von Mises) w.r.t. the symmetric tensor
///
/// ```text
/// s = deviator(a)
///
/// dq     dJ2
/// ── = A ───
/// da     da
///
/// (a is symmetric)
/// ```
///
/// ```text
///       √3
/// A = ─────
///     2 √J2
/// ```
///
/// # Output
///
/// * If `J2 > TOL_J2`, returns `J2`; otherwise, returns `None`.
/// * `d1` -- a tensor to hold the resulting derivative; with the same dimension as `a`
///
/// # Input
///
/// * `a` -- symmetric tensor, i.e., N = 4 or N = 6.
///
/// # Panics
///
/// A panic will occur if `a` is not symmetric, i.e., N = 9.
pub fn deriv1_invariant_q<const N: usize>(d1: &mut Tensor2<N>, a: &Tensor2<N>) -> Option<f64> {
    assert!(N != 9, "the tensor must be symmetric with N = 4 or N = 6");
    let jj2 = a.invariant_jj2();
    if jj2 > TOL_J2 {
        let aa = 0.5 * SQRT_3 / f64::sqrt(jj2);
        deriv1_invariant_jj2(d1, a);
        for m in 0..N {
            d1.vec[m] *= aa;
        }
        return Some(jj2);
    }
    None
}

/// Calculates the first derivative of the Lode invariant w.r.t. the symmetric tensor
///
/// ```text
/// dl     dJ3        dJ2
/// ── = A ─── - B J3 ───
/// da     da         da
///
/// or
///
/// dl     dJ3
/// ── = A ─── - B J3 s
/// da     da
/// ```
///
/// ```text
///         3 √3                9 √3
/// A = ─────────────   B = ─────────────
///     2 pow(J2,1.5)       4 pow(J2,2.5)
/// ```
///
/// # Output
///
/// * If `J2 > TOL_J2`, returns `J2`; otherwise, returns `None`.
/// * `d1` -- a tensor to hold the resulting derivative
///
/// # Input
///
/// * `a` -- symmetric tensor, i.e., N = 4 or N = 6.
pub fn deriv1_invariant_lode<const N: usize>(d1: &mut Tensor2<N>, a: &Tensor2<N>) -> Option<f64> {
    assert!(N != 9, "the tensor must be symmetric with N = 4 or N = 6");
    let jj2 = a.invariant_jj2();
    let mut s = [0.0; 6];
    if jj2 > TOL_J2 {
        deriv1_invariant_jj3_slice(d1.as_mut_data(), &mut s, a); // d1 := dJ3/da
        let jj3 = a.invariant_jj3();
        let sqrt_j2 = f64::sqrt(jj2);
        let aa = 1.5 * SQRT_3 / (jj2 * sqrt_j2);
        let bb = 2.25 * SQRT_3 / (jj2 * jj2 * sqrt_j2);
        for m in 0..N {
            d1.vec[m] = aa * d1.vec[m] - bb * jj3 * s[m];
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
    use russell_lab::{Matrix, approx_eq, deriv1_central5, mat_approx_eq};

    // Defines f(a)
    #[derive(Clone, Copy)]
    enum F {
        Norm,
        I2,
        I3,
        J2,
        J3,
        D, // distance invariant
        R, // radius invariant
        P,
        Q,
        Lode,
    }

    #[test]
    #[allow(clippy::clone_on_copy)] // test the derived Clone implementation
    fn f_enum_clone_works() {
        let a = F::Norm;
        let _ = a.clone();
    }

    // computes the analytical derivative df(a)/da
    fn analytical_deriv<const N: usize>(fn_name: F, d1: &mut Tensor2<N>, a: &Tensor2<N>) {
        match fn_name {
            F::Norm => {
                deriv1_norm(d1, a).unwrap();
            }
            F::I2 => deriv1_invariant_ii2(d1, a),
            F::I3 => deriv1_invariant_ii3(d1, a),
            F::J2 => deriv1_invariant_jj2(d1, a),
            F::J3 => deriv1_invariant_jj3(d1, a),
            F::D => deriv1_invariant_d(d1, a),
            F::R => {
                deriv1_invariant_r(d1, a).unwrap();
            }
            F::P => deriv1_invariant_p(d1, a),
            F::Q => {
                deriv1_invariant_q(d1, a).unwrap();
            }
            F::Lode => {
                deriv1_invariant_lode(d1, a).unwrap();
            }
        };
    }

    // Holds arguments for numerical differentiation of a scalar f(a) w.r.t. aᵢⱼ (standard components)
    struct ArgsNumDeriv<const N: usize> {
        fn_name: F,    // name of f(a)
        a_mat: Matrix, // @ a (3x3 matrix form)
        a: Tensor2<N>, // temporary tensor with varying ij-components
        i: usize,      // index i of ∂f/∂aᵢⱼ
        j: usize,      // index j of ∂f/∂aᵢⱼ
    }

    // Holds arguments for numerical differentiation of a scalar f(a) w.r.t. aₘ (matrix representation)
    struct ArgsNumDerivM<const N: usize> {
        fn_name: F,    // name of f(a)
        a: Tensor2<N>, // @ a, with varying m-components
        m: usize,      // index m of ∂f/∂aₘ
    }

    // computes f(a) for varying components x = aᵢⱼ
    fn f_a<const N: usize>(x: f64, args: &mut ArgsNumDeriv<N>) -> Result<f64, StrError> {
        let original = args.a_mat.get(args.i, args.j);
        args.a_mat.set(args.i, args.j, x);
        args.a.set_std_matrix(&args.a_mat).unwrap();
        let res = match args.fn_name {
            F::Norm => args.a.norm(),
            F::I2 => args.a.invariant_ii2(),
            F::I3 => args.a.invariant_ii3(),
            F::J2 => args.a.invariant_jj2(),
            F::J3 => args.a.invariant_jj3(),
            F::D => args.a.invariant_d(),
            F::R => args.a.invariant_r(),
            F::P => args.a.invariant_p(),
            F::Q => args.a.invariant_q(),
            F::Lode => args.a.invariant_lode().unwrap(),
        };
        args.a_mat.set(args.i, args.j, original);
        Ok(res)
    }

    // computes f(a) for varying components x = aₘ
    fn f_a_mat<const N: usize>(x: f64, args: &mut ArgsNumDerivM<N>) -> Result<f64, StrError> {
        let original = args.a.vec[args.m];
        args.a.vec[args.m] = x;
        let res = match args.fn_name {
            F::Norm => args.a.norm(),
            F::I2 => args.a.invariant_ii2(),
            F::I3 => args.a.invariant_ii3(),
            F::J2 => args.a.invariant_jj2(),
            F::J3 => args.a.invariant_jj3(),
            F::D => args.a.invariant_d(),
            F::R => args.a.invariant_r(),
            F::P => args.a.invariant_p(),
            F::Q => args.a.invariant_q(),
            F::Lode => args.a.invariant_lode().unwrap(),
        };
        args.a.vec[args.m] = original;
        Ok(res)
    }

    // computes ∂f/∂aᵢⱼ and returns as a 3x3 matrix of (standard) components
    fn numerical_deriv<const N: usize>(a: &Tensor2<N>, fn_name: F) -> Matrix {
        let mut args = ArgsNumDeriv {
            fn_name,
            a_mat: a.as_std_matrix(),
            a: a.as_general(),
            i: 0,
            j: 0,
        };
        let mut num_deriv = Matrix::new(3, 3);
        for i in 0..3 {
            args.i = i;
            for j in 0..3 {
                args.j = j;
                let x = args.a_mat.get(i, j);
                let res = deriv1_central5(x, &mut args, f_a).unwrap();
                num_deriv.set(i, j, res);
            }
        }
        num_deriv
    }

    // computes ∂f/∂aₘ and returns as a 3x3 matrix of (standard) components
    fn numerical_deriv_mat<const N: usize>(a: &Tensor2<N>, fn_name: F) -> Matrix {
        let mut args = ArgsNumDerivM {
            fn_name,
            a: a.clone(),
            m: 0,
        };
        let mut num_deriv = a.clone();
        for m in 0..N {
            args.m = m;
            let x = args.a.vec[m];
            let res = deriv1_central5(x, &mut args, f_a_mat).unwrap();
            num_deriv.vec[m] = res;
        }
        num_deriv.as_std_matrix()
    }

    // checks ∂f/∂aᵢⱼ
    fn check_deriv<const N: usize>(fn_name: F, sample: &SampleTensor2, tol: f64, _verbose: bool) {
        let a = Tensor2::<N>::from_std_matrix(&sample.matrix).unwrap();
        let mut d1 = Tensor2::<N>::new();
        analytical_deriv(fn_name, &mut d1, &a);
        let ana = d1.as_std_matrix();
        let num = numerical_deriv(&a, fn_name);
        let num_mat = numerical_deriv_mat(&a, fn_name);
        // println!("analytical derivative:\n{}", ana);
        // println!("numerical derivative:\n{}", num);
        // println!("numerical derivative (matrix):\n{}", num_mat);
        mat_approx_eq(&ana, &num, tol);
        mat_approx_eq(&ana, &num_mat, tol);
    }

    #[test]
    fn deriv_norm_works() {
        let v = false;
        check_deriv::<9>(F::Norm, &SamplesTensor2::TENSOR_T, 1e-10, v);
        check_deriv::<6>(F::Norm, &SamplesTensor2::TENSOR_S, 1e-10, v);
        check_deriv::<4>(F::Norm, &SamplesTensor2::TENSOR_Z, 1e-11, v);
    }

    #[test]
    fn deriv_invariant_ii2_works() {
        let v = false;
        check_deriv::<6>(F::I2, &SamplesTensor2::TENSOR_S, 1e-11, v);
        check_deriv::<4>(F::I2, &SamplesTensor2::TENSOR_Z, 1e-11, v);
        check_deriv::<4>(F::I2, &SamplesTensor2::TENSOR_O, 1e-15, v);
        check_deriv::<4>(F::I2, &SamplesTensor2::TENSOR_I, 1e-12, v);
    }

    #[test]
    fn deriv_invariant_ii3_works() {
        let v = false;
        check_deriv::<9>(F::I3, &SamplesTensor2::TENSOR_T, 1e-9, v);
        check_deriv::<6>(F::I3, &SamplesTensor2::TENSOR_S, 1e-10, v);
        check_deriv::<4>(F::I3, &SamplesTensor2::TENSOR_Z, 1e-11, v);
        check_deriv::<4>(F::I3, &SamplesTensor2::TENSOR_O, 1e-15, v);
        check_deriv::<4>(F::I3, &SamplesTensor2::TENSOR_I, 1e-12, v);
    }

    #[test]
    fn deriv1_invariant_ii3_cofactor_works() {
        // general tensor: dI3/da_ij must equal det(a) * (a⁻¹)ᵀ
        let a = Tensor2::<9>::from_std_matrix(&SamplesTensor2::TENSOR_T.matrix).unwrap();
        let mut d1 = Tensor2::<9>::new();
        deriv1_invariant_ii3(&mut d1, &a);
        let det = a.invariant_ii3();
        let mut ai = Tensor2::<9>::new();
        a.inverse(&mut ai, 1e-15).unwrap();
        let mut ai_t = Tensor2::<9>::new();
        ai.transpose(&mut ai_t);
        for m in 0..9 {
            approx_eq(d1.vec[m], det * ai_t.vec[m], 1e-12);
        }
    }

    #[test]
    fn deriv_invariant_jj2_works() {
        let v = false;
        check_deriv::<6>(F::J2, &SamplesTensor2::TENSOR_S, 1e-10, v);
        check_deriv::<4>(F::J2, &SamplesTensor2::TENSOR_Z, 1e-11, v);
        check_deriv::<4>(F::J2, &SamplesTensor2::TENSOR_O, 1e-15, v);
        check_deriv::<4>(F::J2, &SamplesTensor2::TENSOR_I, 1e-12, v);
    }

    #[test]
    fn deriv_invariant_jj3_works() {
        let v = false;
        check_deriv::<6>(F::J3, &SamplesTensor2::TENSOR_S, 1e-9, v);
        check_deriv::<4>(F::J3, &SamplesTensor2::TENSOR_Z, 1e-10, v);
        check_deriv::<4>(F::J3, &SamplesTensor2::TENSOR_O, 1e-15, v);
        check_deriv::<4>(F::J3, &SamplesTensor2::TENSOR_I, 1e-15, v);
    }

    #[test]
    fn deriv_d_works() {
        let v = false;
        check_deriv::<6>(F::D, &SamplesTensor2::TENSOR_S, 1e-11, v);
        check_deriv::<4>(F::D, &SamplesTensor2::TENSOR_Z, 1e-11, v);
    }

    #[test]
    fn deriv_r_works() {
        let v = false;
        check_deriv::<6>(F::R, &SamplesTensor2::TENSOR_U, 1e-10, v);
        check_deriv::<6>(F::R, &SamplesTensor2::TENSOR_S, 1e-10, v);
        check_deriv::<4>(F::R, &SamplesTensor2::TENSOR_X, 1e-11, v);
        check_deriv::<4>(F::R, &SamplesTensor2::TENSOR_Y, 1e-10, v);
        check_deriv::<4>(F::R, &SamplesTensor2::TENSOR_Z, 1e-10, v);
    }

    #[test]
    fn deriv_p_works() {
        let v = false;
        check_deriv::<6>(F::P, &SamplesTensor2::TENSOR_S, 1e-11, v);
        check_deriv::<4>(F::P, &SamplesTensor2::TENSOR_Z, 1e-12, v);
    }

    #[test]
    fn deriv_q_works() {
        let v = false;
        check_deriv::<6>(F::Q, &SamplesTensor2::TENSOR_U, 1e-10, v);
        check_deriv::<6>(F::Q, &SamplesTensor2::TENSOR_S, 1e-10, v);
        check_deriv::<4>(F::Q, &SamplesTensor2::TENSOR_X, 1e-11, v);
        check_deriv::<4>(F::Q, &SamplesTensor2::TENSOR_Y, 1e-10, v);
        check_deriv::<4>(F::Q, &SamplesTensor2::TENSOR_Z, 1e-10, v);
    }

    #[test]
    fn deriv_invariant_lode_works() {
        let v = false;
        check_deriv::<6>(F::Lode, &SamplesTensor2::TENSOR_U, 1e-10, v);
        check_deriv::<6>(F::Lode, &SamplesTensor2::TENSOR_S, 1e-10, v);
        check_deriv::<4>(F::Lode, &SamplesTensor2::TENSOR_X, 1e-10, v);
        check_deriv::<4>(F::Lode, &SamplesTensor2::TENSOR_Y, 1e-10, v);
        check_deriv::<4>(F::Lode, &SamplesTensor2::TENSOR_Z, 1e-10, v);
    }

    #[test]
    fn check_for_none() {
        let a = Tensor2::<4>::from_std_matrix(&SamplesTensor2::TENSOR_O.matrix).unwrap();
        let mut d1 = Tensor2::<4>::new();
        assert_eq!(deriv1_norm(&mut d1, &a), None);
        assert_eq!(deriv1_invariant_q(&mut d1, &a), None);
        assert_eq!(deriv1_invariant_lode(&mut d1, &a), None);
    }

    // check assertions -----------------------------------------------------------------------------

    #[test]
    #[should_panic(expected = "the tensor must be symmetric with N = 4 or N = 6")]
    fn deriv1_invariant_jj2_panics_on_on_gen() {
        let mut d1_gen = Tensor2::<9>::new();
        let a_gen = Tensor2::<9>::new();
        deriv1_invariant_jj2(&mut d1_gen, &a_gen);
    }

    #[test]
    #[should_panic(expected = "the tensor must be symmetric with N = 4 or N = 6")]
    fn deriv1_invariant_jj3_panics_on_non_gen() {
        let mut d1_gen = Tensor2::<9>::new();
        let a_gen = Tensor2::<9>::new();
        deriv1_invariant_jj3(&mut d1_gen, &a_gen);
    }

    #[test]
    #[should_panic(expected = "the tensor must be symmetric with N = 4 or N = 6")]
    fn deriv1_invariant_q_panics_on_non_gen() {
        let mut d1_gen = Tensor2::<9>::new();
        let a_gen = Tensor2::<9>::new();
        deriv1_invariant_q(&mut d1_gen, &a_gen);
    }

    #[test]
    #[should_panic(expected = "the tensor must be symmetric with N = 4 or N = 6")]
    fn deriv1_invariant_lode_panics_on_non_gen() {
        let mut d1_gen = Tensor2::<9>::new();
        let a_gen = Tensor2::<9>::new();
        deriv1_invariant_lode(&mut d1_gen, &a_gen);
    }
}
