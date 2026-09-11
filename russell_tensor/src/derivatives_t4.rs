use crate::{ADD, IDENTITY2, P_SYMDEV, SET, SQRT_2, SQRT_3, TOL_J2, deriv1_invariant_jj3_slice};
use crate::{Tensor2, Tensor4};
use crate::{qsd_fn_slice, ssd_fn_slice, t2_odyad_t2_slice};

/// Calculates the derivative of the inverse tensor w.r.t. the defining Tensor2
///
/// ```text
/// dA⁻¹         _
/// ──── = - A⁻¹ ⊗ A⁻ᵀ
///  dA
/// ```
///
/// ```text
/// With Cartesian components:
///
/// ∂A⁻¹ᵢⱼ
/// ────── = - A⁻¹ᵢₖ A⁻ᵀⱼₗ
///  ∂Aₖₗ
/// ```
///
/// # Output
///
/// * `dai_da` -- the derivative of the inverse tensor
///
/// # Input
///
/// * `ai` -- the pre-computed inverse tensor
pub fn deriv_inverse_tensor<const N: usize>(dai_da: &mut Tensor4<9>, ai: &Tensor2<N>) {
    let mut at = [0.0; 9];
    ai.transpose_slice(&mut at);
    t2_odyad_t2_slice::<N>(dai_da, SET, -1.0, ai.as_data(), &at);
}

/// Calculates the derivative of the inverse tensor w.r.t. a symmetric Tensor2
///
/// ```text
/// dA⁻¹     1      _                 
/// ──── = - ─ (A⁻¹ ⊗ A⁻¹ + A⁻¹ ⊗ A⁻¹)
///  dA      2                  ‾     
///
///      = - 0.5 ssd(A⁻¹)
/// ```
///
/// ```text
/// With Cartesian components:
///
/// ∂A⁻¹ᵢⱼ     1
/// ────── = - ─ (A⁻¹ᵢₖ A⁻¹ⱼₗ + A⁻¹ᵢₗ A⁻¹ⱼₖ)
///  ∂Aₖₗ      2
/// ```
///
/// # Output
///
/// * `dai_da` -- the derivative of the inverse tensor
///
/// # Input
///
/// * `ai` -- the pre-computed inverse symmetric tensor with N = 4 or N = 6.
///
/// # Panics
///
/// A panic will occur if `ai` is not symmetric, i.e., N = 9.
pub fn deriv_inverse_tensor_sym<const N: usize>(dai_da: &mut Tensor4<6>, ai: &Tensor2<N>) {
    assert!(N != 9, "the inverse tensor must be symmetric with N = 4 or N = 6");
    ssd_fn_slice::<N>(dai_da, SET, -0.5, ai.as_data());
}

/// Calculates the derivative of the squared tensor w.r.t. a Tensor2
///
/// ```text
/// dA²     _       _
/// ─── = A ⊗ I + I ⊗ Aᵀ
/// dA
/// ```
///
/// ```text
/// With Cartesian components:
///
/// ∂A²ᵢⱼ
/// ───── = Aᵢₖ δⱼₗ + δᵢₖ Aₗⱼ
///  ∂Aₖₗ
/// ```
///
/// **Note:** No temporary tensors are allocated in this function.
///
/// # Output
///
/// * `da2_da` -- the derivative of the squared tensor
///
/// # Input
///
/// * `a` -- the second-order tensor
pub fn deriv_squared_tensor<const N: usize>(da2_da: &mut Tensor4<9>, a: &Tensor2<N>) {
    let a_data = a.as_data();
    let mut at = [0.0; 9];
    a.transpose_slice(&mut at);

    // da2_da := A ⊗̄ I + I ⊗̄ Aᵀ
    t2_odyad_t2_slice::<N>(da2_da, SET, 1.0, a_data, &IDENTITY2);
    t2_odyad_t2_slice::<N>(da2_da, ADD, 1.0, &IDENTITY2, &at);
}

/// Calculates the derivative of the squared tensor w.r.t. a symmetric Tensor2
///
/// ```text
/// dA²   1    _               _
/// ─── = ─ (A ⊗ I + A ⊗ I + I ⊗ A + I ⊗ A)
/// dA    2            ‾               ‾
///
///     = 0.5 qsd(A, I)
/// ```
///
/// ```text
/// With Cartesian components:
///
/// ∂A²ᵢⱼ   1
/// ───── = ─ (Aᵢₖ δⱼₗ + Aᵢₗ δⱼₖ + δᵢₖ Aⱼₗ + δᵢₗ Aⱼₖ)
///  ∂Aₖₗ   2
/// ```
///
/// # Output
///
/// * `da2_da` -- the derivative of the squared tensor
///
/// # Input
///
/// * `a` -- the second-order symmetric tensor with N = 4 or N = 6.
///
/// # Panics
///
/// A panic will occur if `a` is not symmetric, i.e., N = 9.
pub fn deriv_squared_tensor_sym<const N: usize>(da2_da: &mut Tensor4<6>, a: &Tensor2<N>) {
    assert!(N != 9, "the tensor must be symmetric with N = 4 or N = 6");
    qsd_fn_slice::<N>(da2_da, SET, 0.5, a.as_data(), &IDENTITY2);
}

/// Calculates the second derivative of the J2 invariant w.r.t. the stress tensor
///
/// ```text
///  d²J2
/// ─────── = Psymdev   (σ must be symmetric)
/// dσ ⊗ dσ
/// ```
///
/// # Output
///
/// * `d2` -- the second derivative of J2
///
/// # Input
///
/// * `sigma` -- the symmetric stress tensor, i.e., N = 4 or N = 6.
///   (it's not actually used here, but kept for consistency).
///
/// # Panics
///
/// A panic will occur if `sigma` is not symmetric, i.e., N = 9.
pub fn deriv2_invariant_jj2<const N: usize>(d2: &mut Tensor4<6>, _sigma: &Tensor2<N>) {
    assert!(N != 9, "the tensor must be symmetric with N = 4 or N = 6");
    d2.set_pp_symdev();
}

/// Calculates the second derivative of the J3 invariant w.r.t. the stress tensor
///
/// ```text
/// s := deviator(σ)
///
///  d²J3     1            2
/// ─────── = ─ qsd(s,I) − ─ (s ⊗ I + I ⊗ s)
/// dσ ⊗ dσ   2            3
///
/// (σ must be symmetric)
/// ```
///
/// # Output
///
/// * `d2` -- the second derivative of J3
///
/// # Input
///
/// * `sigma` -- the symmetric stress tensor, i.e., N = 4 or N = 6.
///
/// # Panics
///
/// A panic will occur if `sigma` is not symmetric, i.e., N = 9.
pub fn deriv2_invariant_jj3<const N: usize>(d2: &mut Tensor4<6>, sigma: &Tensor2<N>) {
    assert!(N != 9, "the stress tensor must be symmetric with N = 4 or N = 6");

    // deviator: s = dev(σ) (stack array)
    let mut s = [0.0; 6];
    sigma.deviator_slice(&mut s);

    // row 0
    d2.set(0, 0, 2.0 * s[0] / 3.0);
    d2.set(0, 1, -2.0 * (s[0] + s[1]) / 3.0);
    d2.set(0, 2, -2.0 * (s[0] + s[2]) / 3.0);
    d2.set(0, 3, s[3] / 3.0);
    d2.set(0, 4, -2.0 * s[4] / 3.0);
    d2.set(0, 5, s[5] / 3.0);

    // row 1
    d2.set(1, 0, -2.0 * (s[0] + s[1]) / 3.0);
    d2.set(1, 1, 2.0 * s[1] / 3.0);
    d2.set(1, 2, -2.0 * (s[1] + s[2]) / 3.0);
    d2.set(1, 3, s[3] / 3.0);
    d2.set(1, 4, s[4] / 3.0);
    d2.set(1, 5, -2.0 * s[5] / 3.0);

    // row 2
    d2.set(2, 0, -2.0 * (s[0] + s[2]) / 3.0);
    d2.set(2, 1, -2.0 * (s[1] + s[2]) / 3.0);
    d2.set(2, 2, 2.0 * s[2] / 3.0);
    d2.set(2, 3, -2.0 * s[3] / 3.0);
    d2.set(2, 4, s[4] / 3.0);
    d2.set(2, 5, s[5] / 3.0);

    // row 3
    d2.set(3, 0, s[3] / 3.0);
    d2.set(3, 1, s[3] / 3.0);
    d2.set(3, 2, -2.0 * s[3] / 3.0);
    d2.set(3, 3, s[0] + s[1]);
    d2.set(3, 4, s[5] / SQRT_2);
    d2.set(3, 5, s[4] / SQRT_2);

    // row 4
    d2.set(4, 0, -2.0 * s[4] / 3.0);
    d2.set(4, 1, s[4] / 3.0);
    d2.set(4, 2, s[4] / 3.0);
    d2.set(4, 3, s[5] / SQRT_2);
    d2.set(4, 4, s[1] + s[2]);
    d2.set(4, 5, s[3] / SQRT_2);

    // row 5
    d2.set(5, 0, s[5] / 3.0);
    d2.set(5, 1, -2.0 * s[5] / 3.0);
    d2.set(5, 2, s[5] / 3.0);
    d2.set(5, 3, s[4] / SQRT_2);
    d2.set(5, 4, s[3] / SQRT_2);
    d2.set(5, 5, s[0] + s[2]);
}

/// Calculates the second derivative of the σt w.r.t. the stress tensor
///
/// ```text
/// d²σt      d²J2      dJ2   dJ2
/// ───── = a ───── - b ─── ⊗ ───
/// dσ⊗dσ     dσ⊗dσ      dσ    dσ
///
/// (σ must be symmetric)
/// ```
///
/// ```text
///          √2                  √2
/// a = ─────────────   b = ─────────────
///     2 pow(J2,0.5)       4 pow(J2,1.5)
/// ```
///
/// # Output
///
/// * If `J2 > TOL_J2`, returns `J2`; otherwise, returns None.
/// * `d2` -- the second derivative of `σt`
///
/// # Input
///
/// * `sigma` -- the symmetric stress tensor, i.e., N = 4 or N = 6.
///
/// # Panics
///
/// A panic will occur if `sigma` is not symmetric, i.e., N = 9.
pub fn deriv2_invariant_sigma_t<const N: usize>(d2: &mut Tensor4<6>, sigma: &Tensor2<N>) -> Option<f64> {
    assert!(N != 9, "the stress tensor must be symmetric with N = 4 or N = 6");
    let jj2 = sigma.invariant_jj2();
    if jj2 > TOL_J2 {
        let sqrt_j2 = f64::sqrt(jj2);
        let a = 0.5 * SQRT_2 / sqrt_j2;
        let b = 0.25 * SQRT_2 / (jj2 * sqrt_j2);
        let mut d1_jj2 = [0.0; 6];
        sigma.deviator_slice(&mut d1_jj2);
        let d2_jj2 = &P_SYMDEV;
        for m in 0..6 {
            for n in 0..6 {
                d2.set(m, n, a * d2_jj2[m][n] - b * d1_jj2[m] * d1_jj2[n]);
            }
        }
        return Some(jj2);
    }
    None
}

/// Calculates the second derivative of the deviatoric invariant (von Mises) w.r.t. the stress tensor
///
/// ```text
///  d²q      d²J2      dJ2   dJ2
/// ───── = a ───── - b ─── ⊗ ───
/// dσ⊗dσ     dσ⊗dσ      dσ    dσ
///
/// (σ must be symmetric)
/// ```
///
/// ```text
///          √3                  √3
/// a = ─────────────   b = ─────────────
///     2 pow(J2,0.5)       4 pow(J2,1.5)
/// ```
///
/// # Output
///
/// * If `J2 > TOL_J2`, returns `J2`; otherwise, returns None.
/// * `d2` -- the second derivative of `q`
///
/// # Input
///
/// * `sigma` -- the symmetric stress tensor, i.e., N = 4 or N = 6.
///
/// # Panics
///
/// A panic will occur if `sigma` is not symmetric, i.e., N = 9.
pub fn deriv2_invariant_q<const N: usize>(d2: &mut Tensor4<6>, sigma: &Tensor2<N>) -> Option<f64> {
    assert!(N != 9, "the stress tensor must be symmetric with N = 4 or N = 6");
    let jj2 = sigma.invariant_jj2();
    if jj2 > TOL_J2 {
        let sqrt_j2 = f64::sqrt(jj2);
        let a = 0.5 * SQRT_3 / sqrt_j2;
        let b = 0.25 * SQRT_3 / (jj2 * sqrt_j2);
        let mut d1_jj2 = [0.0; 6];
        sigma.deviator_slice(&mut d1_jj2);
        let d2_jj2 = &P_SYMDEV;
        for m in 0..6 {
            for n in 0..6 {
                d2.set(m, n, a * d2_jj2[m][n] - b * d1_jj2[m] * d1_jj2[n]);
            }
        }
        return Some(jj2);
    }
    None
}

/// Sets a workspace with temporary variables to calculate the second derivative of the Lode angle
pub struct WorkspaceDeriv2Lode {
    pub d1_jj3: Tensor2<6>,
    pub d2_jj3: Tensor4<6>,
}

impl WorkspaceDeriv2Lode {
    /// Allocates a new instance
    pub fn new() -> Self {
        WorkspaceDeriv2Lode {
            d1_jj3: Tensor2::new(),
            d2_jj3: Tensor4::new(),
        }
    }
}

/// Calculates the second derivative of the Lode invariant w.r.t. the stress tensor
///
/// ```text
///  d²l      d²J3         d²J2      ⎛ dJ3   dJ2   dJ2   dJ3 ⎞        dJ2   dJ2
/// ───── = a ───── - b J3 ───── - b ⎜ ─── ⊗ ─── + ─── ⊗ ─── ⎟ + c J3 ─── ⊗ ───
/// dσ⊗dσ     dσ⊗dσ        dσ⊗dσ     ⎝  dσ    dσ    dσ    dσ ⎠         dσ    dσ
///
/// (σ must be symmetric)
/// ```
///
/// ```text
///         3 √3               9 √3                45 √3
/// a = ─────────────   b = ─────────────   c = ─────────────
///     2 pow(J2,1.5)       4 pow(J2,2.5)       8 pow(J2,3.5)
/// ```
///
/// # Output
///
/// * If `J2 > TOL_J2`, returns `J2`; otherwise, returns None.
/// * `work` -- auxiliary workspace
///
/// # Input
///
/// * `sigma` -- the symmetric stress tensor, i.e., N = 4 or N = 6.
///
/// # Panics
///
/// A panic will occur if `sigma` is not symmetric, i.e., N = 9.
pub fn deriv2_invariant_lode<const N: usize>(
    d2: &mut Tensor4<6>,
    work: &mut WorkspaceDeriv2Lode,
    sigma: &Tensor2<N>,
) -> Option<f64> {
    assert!(N != 9, "the stress tensor must be symmetric with N = 4 or N = 6");
    let jj2 = sigma.invariant_jj2();
    if jj2 > TOL_J2 {
        let jj3 = sigma.invariant_jj3();
        let sqrt_j2 = f64::sqrt(jj2);
        let a = 1.5 * SQRT_3 / (jj2 * sqrt_j2);
        let b = 2.25 * SQRT_3 / (jj2 * jj2 * sqrt_j2);
        let c = 5.625 * SQRT_3 / (jj2 * jj2 * jj2 * sqrt_j2);
        let mut s = [0.0; 6];
        deriv1_invariant_jj3_slice(&mut work.d1_jj3.as_mut_data(), &mut s, sigma);
        deriv2_invariant_jj3(&mut work.d2_jj3, sigma);
        let d1_jj2 = &s;
        let d2_jj2 = &P_SYMDEV;
        for m in 0..6 {
            for n in 0..6 {
                d2.set(
                    m,
                    n,
                    //   d²J3
                    // a ─────
                    //   dσ⊗dσ
                    a * work.d2_jj3.get(m, n)
                    //        d²J2  
                    // - b J3 ───── 
                    //        dσ⊗dσ 
                    - b * jj3 * d2_jj2[m][n]
                    //     ⎛ dJ3   dJ2   dJ2   dJ3 ⎞
                    // - b ⎜ ─── ⊗ ─── + ─── ⊗ ─── ⎟
                    //     ⎝  dσ    dσ    dσ    dσ ⎠
                    - b * (work.d1_jj3.get(m) * d1_jj2[n] + d1_jj2[m] * work.d1_jj3.get(n))
                    //         dJ2   dJ2
                    // + c J3 ─── ⊗ ───
                    //          dσ    dσ
                    // 
                    + c * jj3 * d1_jj2[m] * d1_jj2[n],
                );
            }
        }
        return Some(jj2);
    }
    None
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MN_TO_IJKL, SQRT_2, SamplesTensor2, StrError};
    use crate::{
        deriv1_invariant_jj2, deriv1_invariant_jj3, deriv1_invariant_lode, deriv1_invariant_q, deriv1_invariant_sigma_t,
    };
    use russell_lab::{Matrix, approx_eq, deriv1_central5, mat_approx_eq};

    // Returns the dim x dim Kelvin-Mandel sub matrix of a Tensor4 as a Matrix
    fn kelvin_matrix<const N: usize>(dd: &Tensor4<N>) -> Matrix {
        let mut m = Matrix::new(N, N);
        for i in 0..N {
            for j in 0..N {
                m.set(i, j, dd.get(i, j));
            }
        }
        m
    }

    // Holds arguments for numerical differentiation corresponding to ∂aiᵢⱼ/∂aₖₗ
    struct ArgsNumDerivInverse {
        data: Matrix,   // 3x3 matrix form (standard components)
        a: Tensor2<9>,  // temporary tensor (will use "general" for numerical derivative)
        ai: Tensor2<9>, // temporary inverse tensor
        i: usize,       // index j of ∂aiᵢⱼ/∂aₖₗ
        j: usize,       // index j of ∂aiᵢⱼ/∂aₖₗ
        k: usize,       // index j of ∂aiᵢⱼ/∂aₖₗ
        l: usize,       // index j of ∂aiᵢⱼ/∂aₖₗ
    }

    // Holds arguments for numerical differentiation corresponding to ∂aiₘ/∂aₙ (Kelvin-Mandel representation)
    struct ArgsNumDerivInverseKelvin {
        a: Tensor2<9>,  // temporary tensor (will use "general" for numerical derivative)
        ai: Tensor2<9>, // temporary inverse tensor
        m: usize,       // index of ∂aiₘ/∂aₙ (matrix representation)
        n: usize,       // index of ∂aiₘ/∂aₙ (matrix representation)
    }

    fn component_of_inverse(x: f64, args: &mut ArgsNumDerivInverse) -> Result<f64, StrError> {
        let original = args.data.get(args.k, args.l);
        args.data.set(args.k, args.l, x);
        args.a.set_std_matrix(&args.data).unwrap();
        args.a.inverse(&mut args.ai, 1e-10).unwrap();
        args.data.set(args.k, args.l, original);
        Ok(args.ai.get_std(args.i, args.j))
    }

    fn component_of_inverse_kelvin(x: f64, args: &mut ArgsNumDerivInverseKelvin) -> Result<f64, StrError> {
        let original = args.a.get(args.n);
        args.a.set(args.n, x);
        args.a.inverse(&mut args.ai, 1e-10).unwrap();
        args.a.set(args.n, original);
        Ok(args.ai.get(args.m))
    }

    fn numerical_deriv_inverse<const N: usize>(a: &Tensor2<N>) -> Matrix {
        let mut args = ArgsNumDerivInverse {
            data: a.as_std_matrix(),
            a: Tensor2::new(),
            ai: Tensor2::new(),
            i: 0,
            j: 0,
            k: 0,
            l: 0,
        };
        let mut num_deriv = Matrix::new(9, 9);
        for m in 0..9 {
            for n in 0..9 {
                (args.i, args.j, args.k, args.l) = MN_TO_IJKL[m][n];
                let x = args.data.get(args.k, args.l);
                let res = deriv1_central5(x, &mut args, component_of_inverse).unwrap();
                num_deriv.set(m, n, res);
            }
        }
        num_deriv
    }

    fn numerical_deriv_inverse_kelvin<const N: usize>(a: &Tensor2<N>) -> Matrix {
        let mut args = ArgsNumDerivInverseKelvin {
            a: a.as_general(),
            ai: Tensor2::new(),
            m: 0,
            n: 0,
        };
        let mut num_deriv = Tensor4::<9>::new();
        for m in 0..9 {
            args.m = m;
            for n in 0..9 {
                args.n = n;
                let x = args.a.get(args.n);
                let res = deriv1_central5(x, &mut args, component_of_inverse_kelvin).unwrap();
                num_deriv.set(m, n, res);
            }
        }
        num_deriv.as_std_matrix()
    }

    fn numerical_deriv_inverse_sym_kelvin<const N: usize>(a: &Tensor2<N>) -> Matrix {
        let mut args = ArgsNumDerivInverseKelvin {
            a: Tensor2::new(),
            ai: Tensor2::new(),
            m: 0,
            n: 0,
        };
        args.a.set_std_matrix(&a.as_std_matrix()).unwrap();
        let mut num_deriv = Tensor4::<6>::new();
        for m in 0..6 {
            args.m = m;
            for n in 0..6 {
                args.n = n;
                let x = args.a.get(args.n);
                let res = deriv1_central5(x, &mut args, component_of_inverse_kelvin).unwrap();
                num_deriv.set(m, n, res);
            }
        }
        num_deriv.as_std_matrix()
    }

    fn check_deriv_inverse<const N: usize>(a: &Tensor2<N>, tol: f64) {
        // compute inverse tensor
        let mut ai = Tensor2::<N>::new();
        a.inverse(&mut ai, 1e-10).unwrap();

        // compute analytical derivative
        let mut dd_ana = Tensor4::<9>::new();
        deriv_inverse_tensor(&mut dd_ana, &ai);

        // check using index expression
        let arr = dd_ana.as_std_array();
        let mat = ai.as_std_matrix();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        approx_eq(arr[i][j][k][l], -mat.get(i, k) * mat.get(l, j), 1e-14)
                    }
                }
            }
        }

        // check using numerical derivative
        let ana = dd_ana.as_std_matrix();
        let num = numerical_deriv_inverse(&a);
        let num_kel = numerical_deriv_inverse_kelvin(&a);
        mat_approx_eq(&ana, &num, tol);
        mat_approx_eq(&ana, &num_kel, tol);
    }

    fn check_deriv_inverse_sym<const N: usize>(a: &Tensor2<N>, tol: f64) {
        // compute inverse tensor
        let mut ai = Tensor2::<N>::new();
        a.inverse(&mut ai, 1e-10).unwrap();

        // compute analytical derivative
        let mut dd_ana = Tensor4::<6>::new();
        deriv_inverse_tensor_sym(&mut dd_ana, &ai);

        // check using index expression
        let arr = dd_ana.as_std_array();
        let mat = ai.as_std_matrix();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        approx_eq(
                            arr[i][j][k][l],
                            -0.5 * (mat.get(i, k) * mat.get(j, l) + mat.get(i, l) * mat.get(j, k)),
                            1e-14,
                        )
                    }
                }
            }
        }

        // check using numerical derivative
        let ana = dd_ana.as_std_matrix();
        let num = numerical_deriv_inverse_sym_kelvin(&a);
        mat_approx_eq(&ana, &num, tol);
    }

    #[test]
    fn deriv_inverse_tensor_works() {
        // general
        let s = &SamplesTensor2::TENSOR_T;
        let a = Tensor2::<9>::from_std_matrix(&s.matrix).unwrap();
        check_deriv_inverse(&a, 1e-11);

        // symmetric
        let s = &SamplesTensor2::TENSOR_U;
        let a = Tensor2::<6>::from_std_matrix(&s.matrix).unwrap();
        check_deriv_inverse(&a, 1e-7);

        // symmetric 2d
        let s = &SamplesTensor2::TENSOR_Y;
        let a = Tensor2::<4>::from_std_matrix(&s.matrix).unwrap();
        check_deriv_inverse(&a, 1e-12);
    }

    #[test]
    fn deriv_inverse_tensor_sym_works() {
        // symmetric
        let s = &SamplesTensor2::TENSOR_U;
        let a = Tensor2::<6>::from_std_matrix(&s.matrix).unwrap();
        check_deriv_inverse_sym(&a, 1e-7);

        // symmetric 2d
        let s = &SamplesTensor2::TENSOR_Y;
        let a = Tensor2::<4>::from_std_matrix(&s.matrix).unwrap();
        check_deriv_inverse_sym(&a, 1e-12);
    }

    // squared tensor ------------------------------------------------------------------------------

    // Holds arguments for numerical differentiation corresponding to ∂a²ᵢⱼ/∂aₖₗ
    struct ArgsNumDerivSquared {
        data: Matrix,   // 3x3 matrix form (standard components)
        a: Tensor2<9>,  // temporary tensor (will use "general" for numerical derivative)
        a2: Tensor2<9>, // temporary squared tensor
        i: usize,       // index j of ∂aiᵢⱼ/∂aₖₗ
        j: usize,       // index j of ∂aiᵢⱼ/∂aₖₗ
        k: usize,       // index j of ∂aiᵢⱼ/∂aₖₗ
        l: usize,       // index j of ∂aiᵢⱼ/∂aₖₗ
    }

    // Holds arguments for numerical differentiation corresponding to ∂a²ₘ/∂aₙ (Kelvin-Mandel representation)
    struct ArgsNumDerivSquaredKelvin {
        a: Tensor2<9>,  // temporary tensor (will use "general" for numerical derivative)
        a2: Tensor2<9>, // temporary squared tensor
        m: usize,       // index of ∂aiₘ/∂aₙ (matrix representation)
        n: usize,       // index of ∂aiₘ/∂aₙ (matrix representation)
    }

    fn component_of_squared(x: f64, args: &mut ArgsNumDerivSquared) -> Result<f64, StrError> {
        let original = args.data.get(args.k, args.l);
        args.data.set(args.k, args.l, x);
        args.a.set_std_matrix(&args.data).unwrap();
        args.a.squared(&mut args.a2);
        args.data.set(args.k, args.l, original);
        Ok(args.a2.get_std(args.i, args.j))
    }

    fn component_of_squared_kelvin(x: f64, args: &mut ArgsNumDerivSquaredKelvin) -> Result<f64, StrError> {
        let original = args.a.get(args.n);
        args.a.set(args.n, x);
        args.a.squared(&mut args.a2);
        args.a.set(args.n, original);
        Ok(args.a2.get(args.m))
    }

    fn numerical_deriv_squared<const N: usize>(a: &Tensor2<N>) -> Matrix {
        let mut args = ArgsNumDerivSquared {
            data: a.as_std_matrix(),
            a: Tensor2::new(),
            a2: Tensor2::new(),
            i: 0,
            j: 0,
            k: 0,
            l: 0,
        };
        let mut num_deriv = Matrix::new(9, 9);
        for m in 0..9 {
            for n in 0..9 {
                (args.i, args.j, args.k, args.l) = MN_TO_IJKL[m][n];
                let x = args.data.get(args.k, args.l);
                let res = deriv1_central5(x, &mut args, component_of_squared).unwrap();
                num_deriv.set(m, n, res);
            }
        }
        num_deriv
    }

    fn numerical_deriv_squared_kelvin<const N: usize>(a: &Tensor2<N>) -> Matrix {
        let mut args = ArgsNumDerivSquaredKelvin {
            a: a.as_general(),
            a2: Tensor2::new(),
            m: 0,
            n: 0,
        };
        let mut num_deriv = Tensor4::<9>::new();
        for m in 0..9 {
            args.m = m;
            for n in 0..9 {
                args.n = n;
                let x = args.a.get(args.n);
                let res = deriv1_central5(x, &mut args, component_of_squared_kelvin).unwrap();
                num_deriv.set(m, n, res);
            }
        }
        num_deriv.as_std_matrix()
    }

    fn numerical_deriv_squared_sym_kelvin<const N: usize>(a: &Tensor2<N>) -> Matrix {
        let mut args = ArgsNumDerivSquaredKelvin {
            a: Tensor2::new(),
            a2: Tensor2::new(),
            m: 0,
            n: 0,
        };
        args.a.set_std_matrix(&a.as_std_matrix()).unwrap();
        let mut num_deriv = Tensor4::<6>::new();
        for m in 0..6 {
            args.m = m;
            for n in 0..6 {
                args.n = n;
                let x = args.a.get(args.n);
                let res = deriv1_central5(x, &mut args, component_of_squared_kelvin).unwrap();
                num_deriv.set(m, n, res);
            }
        }
        num_deriv.as_std_matrix()
    }

    fn check_deriv_squared<const N: usize>(a: &Tensor2<N>, tol: f64) {
        // compute analytical derivative
        let mut dd_ana = Tensor4::<9>::new();
        deriv_squared_tensor(&mut dd_ana, &a);

        // check using index expression
        let arr = dd_ana.as_std_array();
        let mat = a.as_std_matrix();
        let del = Matrix::diagonal(&[1.0, 1.0, 1.0]);
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        approx_eq(
                            arr[i][j][k][l],
                            mat.get(i, k) * del.get(j, l) + del.get(i, k) * mat.get(l, j),
                            1e-15,
                        )
                    }
                }
            }
        }

        // check using numerical derivative
        let ana = dd_ana.as_std_matrix();
        let num = numerical_deriv_squared(&a);
        let num_kel = numerical_deriv_squared_kelvin(&a);
        mat_approx_eq(&ana, &num, tol);
        mat_approx_eq(&ana, &num_kel, tol);
    }

    fn check_deriv_squared_sym<const N: usize>(a: &Tensor2<N>, tol: f64) {
        // compute analytical derivative
        let mut dd_ana = Tensor4::<6>::new();
        deriv_squared_tensor_sym(&mut dd_ana, &a);

        // check using index expression
        let arr = dd_ana.as_std_array();
        let mat = a.as_std_matrix();
        let del = Matrix::diagonal(&[1.0, 1.0, 1.0]);
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        approx_eq(
                            arr[i][j][k][l],
                            0.5 * (mat.get(i, k) * del.get(j, l)
                                + mat.get(i, l) * del.get(j, k)
                                + del.get(i, k) * mat.get(j, l)
                                + del.get(i, l) * mat.get(j, k)),
                            1e-15,
                        )
                    }
                }
            }
        }

        // check using numerical derivative
        let ana = dd_ana.as_std_matrix();
        let num = numerical_deriv_squared_sym_kelvin(&a);
        mat_approx_eq(&ana, &num, tol);
    }

    #[test]
    fn deriv_squared_tensor_works() {
        // general
        let s = &SamplesTensor2::TENSOR_T;
        let a = Tensor2::<9>::from_std_matrix(&s.matrix).unwrap();
        check_deriv_squared(&a, 1e-10);

        // symmetric
        let s = &SamplesTensor2::TENSOR_U;
        let a = Tensor2::<6>::from_std_matrix(&s.matrix).unwrap();
        check_deriv_squared(&a, 1e-10);

        // symmetric 2d
        let s = &SamplesTensor2::TENSOR_Y;
        let a = Tensor2::<4>::from_std_matrix(&s.matrix).unwrap();
        check_deriv_squared(&a, 1e-10);
    }

    #[test]
    fn deriv_squared_tensor_sym_works() {
        // symmetric
        let s = &SamplesTensor2::TENSOR_U;
        let a = Tensor2::<6>::from_std_matrix(&s.matrix).unwrap();
        check_deriv_squared_sym(&a, 1e-10);

        // symmetric 2d
        let s = &SamplesTensor2::TENSOR_Y;
        let a = Tensor2::<4>::from_std_matrix(&s.matrix).unwrap();
        check_deriv_squared_sym(&a, 1e-10);
    }

    // second derivative of invariants -------------------------------------------------------------

    enum Invariant {
        J2,
        J3,
        SigmaT,
        Q,
        Lode,
    }

    // Holds arguments for numerical differentiation corresponding to [dInvariant²/dσ⊗dσ]ₘₙ (Kelvin-Mandel representation)
    struct ArgsNumDeriv2InvariantKelvin {
        inv: Invariant,    // option
        sigma: Tensor2<6>, // temporary tensor
        d1: Tensor2<6>,    // dInvariant/dσ (first derivative)
        m: usize,          // index of [dInvariant²/dσ⊗dσ]ₘₙ (matrix representation)
        n: usize,          // index of [dInvariant²/dσ⊗dσ]ₘₙ (matrix representation)
    }

    fn component_of_deriv1_inv_kelvin(x: f64, args: &mut ArgsNumDeriv2InvariantKelvin) -> Result<f64, StrError> {
        let original = args.sigma.get(args.n);
        args.sigma.set(args.n, x);
        match args.inv {
            Invariant::J2 => {
                deriv1_invariant_jj2(&mut args.d1, &args.sigma);
            }
            Invariant::J3 => {
                deriv1_invariant_jj3(&mut args.d1, &args.sigma);
            }
            Invariant::SigmaT => {
                deriv1_invariant_sigma_t(&mut args.d1, &args.sigma).unwrap();
            }
            Invariant::Q => {
                deriv1_invariant_q(&mut args.d1, &args.sigma).unwrap();
            }
            Invariant::Lode => {
                deriv1_invariant_lode(&mut args.d1, &args.sigma);
            }
        };
        args.sigma.set(args.n, original);
        Ok(args.d1.get(args.m))
    }

    fn numerical_deriv2_inv_sym_kelvin<const N: usize>(sigma: &Tensor2<N>, inv: Invariant) -> Matrix {
        let mut args = ArgsNumDeriv2InvariantKelvin {
            inv,
            sigma: Tensor2::new(),
            d1: Tensor2::new(),
            m: 0,
            n: 0,
        };
        args.sigma.set_std_matrix(&sigma.as_std_matrix()).unwrap();
        let mut num_deriv = Tensor4::<6>::new();
        for m in 0..6 {
            args.m = m;
            for n in 0..6 {
                args.n = n;
                let x = args.sigma.get(args.n);
                let res = deriv1_central5(x, &mut args, component_of_deriv1_inv_kelvin).unwrap();
                num_deriv.set(m, n, res);
            }
        }
        num_deriv.as_std_matrix()
    }

    fn check_deriv2_jj2<const N: usize>(sigma: &Tensor2<N>, tol: f64) {
        // compute analytical derivative
        let mut dd2_ana = Tensor4::<6>::new();
        deriv2_invariant_jj2(&mut dd2_ana, &sigma);

        // compare with Psymdev
        let pp_symdev = Tensor4::<6>::constant_pp_symdev();
        mat_approx_eq(&dd2_ana.as_std_matrix(), &pp_symdev.as_std_matrix(), 1e-15);

        // check using numerical derivative
        let ana = dd2_ana.as_std_matrix();
        let num = numerical_deriv2_inv_sym_kelvin(&sigma, Invariant::J2);
        // println!("{}", ana);
        // println!("{}", num);
        mat_approx_eq(&ana, &num, tol);
    }

    fn check_deriv2_jj3<const N: usize>(sigma: &Tensor2<N>, tol: f64) {
        // compute analytical derivative
        let mut dd2_ana = Tensor4::<6>::new();
        deriv2_invariant_jj3(&mut dd2_ana, &sigma);

        // check using numerical derivative
        let ana = dd2_ana.as_std_matrix();
        let num = numerical_deriv2_inv_sym_kelvin(&sigma, Invariant::J3);
        // println!("{}", ana);
        // println!("{}", num);
        mat_approx_eq(&ana, &num, tol);
    }

    fn check_deriv2_sigma_t<const N: usize>(sigma: &Tensor2<N>, tol: f64) {
        // compute analytical derivative
        let mut dd2_ana = Tensor4::<6>::new();
        deriv2_invariant_sigma_t(&mut dd2_ana, &sigma).unwrap();

        // check using numerical derivative
        let ana = dd2_ana.as_std_matrix();
        let num = numerical_deriv2_inv_sym_kelvin(&sigma, Invariant::SigmaT);
        // println!("{}", ana);
        // println!("{}", num);
        mat_approx_eq(&ana, &num, tol);
    }

    fn check_deriv2_q<const N: usize>(sigma: &Tensor2<N>, tol: f64) {
        // compute analytical derivative
        let mut dd2_ana = Tensor4::<6>::new();
        deriv2_invariant_q(&mut dd2_ana, &sigma).unwrap();

        // check using numerical derivative
        let ana = dd2_ana.as_std_matrix();
        let num = numerical_deriv2_inv_sym_kelvin(&sigma, Invariant::Q);
        // println!("{}", ana);
        // println!("{}", num);
        mat_approx_eq(&ana, &num, tol);
    }

    fn check_deriv2_lode<const N: usize>(sigma: &Tensor2<N>, tol: f64) {
        // compute analytical derivative
        let mut dd2_ana = Tensor4::<6>::new();
        let mut work = WorkspaceDeriv2Lode::new();
        deriv2_invariant_lode(&mut dd2_ana, &mut work, &sigma).unwrap();

        // check using numerical derivative
        let ana = dd2_ana.as_std_matrix();
        let num = numerical_deriv2_inv_sym_kelvin(&sigma, Invariant::Lode);
        // println!("{}", ana);
        // println!("{}", num);
        mat_approx_eq(&ana, &num, tol);
    }

    #[test]
    fn deriv2_invariant_jj2_works() {
        // symmetric
        let sigma = Tensor2::<6>::from_std_matrix(&SamplesTensor2::TENSOR_U.matrix).unwrap();
        check_deriv2_jj2(&sigma, 1e-11);

        // symmetric
        let sigma = Tensor2::<6>::from_std_matrix(&SamplesTensor2::TENSOR_S.matrix).unwrap();
        check_deriv2_jj2(&sigma, 1e-11);

        // symmetric 2d
        let sigma = Tensor2::<4>::from_std_matrix(&SamplesTensor2::TENSOR_X.matrix).unwrap();
        check_deriv2_jj2(&sigma, 1e-11);

        // symmetric 2d
        let sigma = Tensor2::<4>::from_std_matrix(&SamplesTensor2::TENSOR_Y.matrix).unwrap();
        check_deriv2_jj2(&sigma, 1e-11);

        // zero
        let sigma = Tensor2::<6>::from_std_matrix(&SamplesTensor2::TENSOR_O.matrix).unwrap();
        check_deriv2_jj2(&sigma, 1e-15);

        // one
        let sigma = Tensor2::<6>::from_std_matrix(&SamplesTensor2::TENSOR_I.matrix).unwrap();
        check_deriv2_jj2(&sigma, 1e-12);
    }

    #[test]
    fn deriv2_invariant_jj3_works() {
        // symmetric
        let sigma = Tensor2::<6>::from_std_matrix(&SamplesTensor2::TENSOR_U.matrix).unwrap();
        check_deriv2_jj3(&sigma, 1e-10);

        // symmetric
        let sigma = Tensor2::<6>::from_std_matrix(&SamplesTensor2::TENSOR_S.matrix).unwrap();
        check_deriv2_jj3(&sigma, 1e-10);

        // symmetric 2d
        let sigma = Tensor2::<4>::from_std_matrix(&SamplesTensor2::TENSOR_X.matrix).unwrap();
        check_deriv2_jj3(&sigma, 1e-10);

        // symmetric 2d
        let sigma = Tensor2::<4>::from_std_matrix(&SamplesTensor2::TENSOR_Y.matrix).unwrap();
        check_deriv2_jj3(&sigma, 1e-10);

        // zero
        let sigma = Tensor2::<6>::from_std_matrix(&SamplesTensor2::TENSOR_O.matrix).unwrap();
        check_deriv2_jj3(&sigma, 1e-15);

        // one
        let sigma = Tensor2::<6>::from_std_matrix(&SamplesTensor2::TENSOR_I.matrix).unwrap();
        check_deriv2_jj3(&sigma, 1e-13);
    }

    #[test]
    fn deriv2_invariant_sigma_t_returns_none() {
        // identity
        let sigma = Tensor2::<4>::from_std_matrix(&SamplesTensor2::TENSOR_I.matrix).unwrap();
        let mut d2 = Tensor4::<6>::new();
        assert_eq!(deriv2_invariant_sigma_t(&mut d2, &sigma), None);
    }

    #[test]
    fn deriv2_invariant_sigma_t_works() {
        // symmetric
        let sigma = Tensor2::<6>::from_std_matrix(&SamplesTensor2::TENSOR_U.matrix).unwrap();
        check_deriv2_sigma_t(&sigma, 1e-11);

        // symmetric
        let sigma = Tensor2::<6>::from_std_matrix(&SamplesTensor2::TENSOR_S.matrix).unwrap();
        check_deriv2_sigma_t(&sigma, 1e-11);

        // symmetric 2d
        let sigma = Tensor2::<4>::from_std_matrix(&SamplesTensor2::TENSOR_X.matrix).unwrap();
        check_deriv2_sigma_t(&sigma, 1e-11);

        // symmetric 2d
        let sigma = Tensor2::<4>::from_std_matrix(&SamplesTensor2::TENSOR_Y.matrix).unwrap();
        check_deriv2_sigma_t(&sigma, 1e-11);
    }

    #[test]
    fn deriv2_invariant_q_returns_none() {
        // identity
        let sigma = Tensor2::<4>::from_std_matrix(&SamplesTensor2::TENSOR_I.matrix).unwrap();
        let mut d2 = Tensor4::<6>::new();
        assert_eq!(deriv2_invariant_q(&mut d2, &sigma), None);
    }

    #[test]
    fn deriv2_invariant_q_works() {
        // symmetric
        let sigma = Tensor2::<6>::from_std_matrix(&SamplesTensor2::TENSOR_U.matrix).unwrap();
        check_deriv2_q(&sigma, 1e-11);

        // symmetric
        let sigma = Tensor2::<6>::from_std_matrix(&SamplesTensor2::TENSOR_S.matrix).unwrap();
        check_deriv2_q(&sigma, 1e-11);

        // symmetric 2d
        let sigma = Tensor2::<4>::from_std_matrix(&SamplesTensor2::TENSOR_X.matrix).unwrap();
        check_deriv2_q(&sigma, 1e-11);

        // symmetric 2d
        let sigma = Tensor2::<4>::from_std_matrix(&SamplesTensor2::TENSOR_Y.matrix).unwrap();
        check_deriv2_q(&sigma, 1e-11);
    }

    #[test]
    fn deriv2_invariant_lode_returns_none() {
        // identity
        let sigma = Tensor2::<4>::from_std_matrix(&SamplesTensor2::TENSOR_I.matrix).unwrap();
        let mut d2 = Tensor4::<6>::new();
        let mut work = WorkspaceDeriv2Lode::new();
        assert_eq!(deriv2_invariant_lode(&mut d2, &mut work, &sigma), None);
    }

    #[test]
    fn deriv2_invariant_lode_works() {
        // symmetric
        let sigma = Tensor2::<6>::from_std_matrix(&SamplesTensor2::TENSOR_U.matrix).unwrap();
        check_deriv2_lode(&sigma, 1e-10);

        // symmetric
        let sigma = Tensor2::<6>::from_std_matrix(&SamplesTensor2::TENSOR_S.matrix).unwrap();
        check_deriv2_lode(&sigma, 1e-11);

        // symmetric 2d
        let sigma = Tensor2::<4>::from_std_matrix(&SamplesTensor2::TENSOR_X.matrix).unwrap();
        check_deriv2_lode(&sigma, 1e-10);

        // symmetric 2d
        let sigma = Tensor2::<4>::from_std_matrix(&SamplesTensor2::TENSOR_Y.matrix).unwrap();
        check_deriv2_lode(&sigma, 1e-9);
    }

    #[test]
    fn example_second_deriv_jj3_lode() {
        let sigma = Tensor2::<6>::from_std_matrix(&SamplesTensor2::TENSOR_U.matrix).unwrap();
        let mut s = Tensor2::<6>::new();
        sigma.deviator(&mut s);
        let mut d2 = Tensor4::<6>::new();
        deriv2_invariant_jj3(&mut d2, &sigma);

        // println!("sigma =\n{:.1}", sigma.to_std_matrix());
        // println!("sigma_mat =\n{}", sigma.vec);
        // println!("s = \n{}", s.vec);
        // println!("d2 = \n{}", d2.mat);

        #[rustfmt::skip]
        let correct = [
            [-16.0/9.0        ,  14.0/9.0      ,   2.0/9.0       ,  2.0*SQRT_2/3.0 , -10.0*SQRT_2/3.0 , SQRT_2      ],
            [ 14.0/9.0        ,   2.0/9.0      , -16.0/9.0       ,  2.0*SQRT_2/3.0 , 5.0*SQRT_2/3.0   , -2.0*SQRT_2 ],
            [  2.0/9.0        , -16.0/9.0      ,  14.0/9.0       , -4.0*SQRT_2/3.0 , 5.0*SQRT_2/3.0   , SQRT_2      ],
            [  2.0*SQRT_2/3.0 , 2.0*SQRT_2/3.0 , -4.0*SQRT_2/3.0 , -7.0/3.0        , 3.0              , 5.0         ],
            [-10.0*SQRT_2/3.0 , 5.0*SQRT_2/3.0 ,  5.0*SQRT_2/3.0 ,  3.0            , 8.0/3.0          , 2.0         ],
            [      SQRT_2     ,-2.0*SQRT_2     ,      SQRT_2     ,  5.0            , 2.0              , -1.0/3.0    ],
        ];
        mat_approx_eq(&kelvin_matrix(&d2), &correct, 1e-15);

        let mut work = WorkspaceDeriv2Lode::new();
        deriv2_invariant_lode(&mut d2, &mut work, &sigma).unwrap();

        // println!("d2 = \n{}", d2.mat);

        #[rustfmt::skip]
        let correct = [
            [-0.039528347708134,  0.0237434792780289,   0.0157848684301052,  0.0136392037983506, -0.0354377940510052,  0.0131589501434791],
            [0.0237434792780289, -0.0200332341113984,  -0.00371024516663052, 0.00899921464051518, 0.0234105185455438, -0.0229302648906723],
            [0.0157848684301052, -0.00371024516663052, -0.0120746232634746, -0.0226384184388658,  0.0120272755054614,  0.00977131474719321],
            [0.0136392037983506,  0.00899921464051518, -0.0226384184388658, -0.0635034452012119,  0.0103061398245104,  0.0374455252630319],
            [-0.0354377940510052, 0.0234105185455438,   0.0120272755054614,  0.0103061398245104, -0.0308487598599826,  0.0128121444219201],
            [0.0131589501434791, -0.0229302648906723,   0.00977131474719321, 0.0374455252630319,  0.0128121444219201, -0.0345929640882181],
        ];
        mat_approx_eq(&kelvin_matrix(&d2), &correct, 1e-15);
    }
}
