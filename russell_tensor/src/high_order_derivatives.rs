use crate::{t2_odyad_t2, t2_ssd, StrError, Tensor2, Tensor4};

/// Calculates the derivative of the inverse tensor w.r.t. the defining Tensor2
///
/// ```text
/// dA⁻¹         _
/// ──── = - A⁻¹ ⊗ A⁻ᵀ
///  dA
/// ```
///
/// ```text
/// With orthonormal Cartesian components:
///
/// ∂A⁻¹ᵢⱼ
/// ────── = - A⁻¹ᵢₖ A⁻ᵀⱼₗ
///  ∂Aₖₗ
///
/// ```
///
/// ## Output
///
/// * `dai_da` -- the derivative of the inverse tensor
///
/// ## Input
///
/// * `ai` -- the pre-computed inverse tensor
/// * `a` -- the defining tensor
pub fn deriv_inverse_tensor(dai_da: &mut Tensor4, ai: &Tensor2) -> Result<(), StrError> {
    let mut ai_t = ai.clone();
    ai.transpose(&mut ai_t).unwrap();
    t2_odyad_t2(dai_da, -1.0, &ai, &ai_t)
}

/// Calculates the derivative of the inverse tensor w.r.t. the defining Tensor2 (symmetric)
///
/// ```text
/// dA⁻¹     1      _                      1
/// ──── = - ─ (A⁻¹ ⊗ A⁻¹ + A⁻¹ ⊗ A⁻¹) = - ─ ssd(A⁻¹, A⁻¹)
///  dA      2                  ‾          2
/// ```
///
/// ```text
/// With orthonormal Cartesian components:
///
/// ∂A⁻¹ᵢⱼ     1
/// ────── = - ─ (A⁻¹ᵢₖ A⁻¹ⱼₗ + A⁻¹ᵢₗ A⁻¹ⱼₖ)
///  ∂Aₖₗ      2
///
/// ```
///
/// ## Output
///
/// * `dai_da` -- the derivative of the inverse tensor
///
/// ## Input
///
/// * `ai` -- the pre-computed inverse tensor (symmetric)
pub fn deriv_inverse_tensor_sym(dai_da: &mut Tensor4, ai: &Tensor2) -> Result<(), StrError> {
    if !ai.symmetric() {
        return Err("tensor 'ai' must be Symmetric or Symmetric2D");
    }
    if dai_da.mat.dims().1 != 6 {
        return Err("tensor 'dai_da' must be Symmetric");
    }
    t2_ssd(dai_da, -0.5, ai)
}

pub fn deriv_square_tensor(_dd: &mut Tensor4, _aa: &Tensor2) -> Result<(), StrError> {
    Err("TODO")
}

pub fn deriv_square_tensor_sym(_dd: &mut Tensor4, _aa: &Tensor2) -> Result<(), StrError> {
    Err("TODO")
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{Tensor2, Tensor4};
    use crate::{deriv_inverse_tensor, deriv_inverse_tensor_sym, Mandel, SamplesTensor2, MN_TO_IJKL};
    use russell_chk::{approx_eq, deriv_central5};
    use russell_lab::{mat_approx_eq, Matrix};

    // Holds arguments for numerical differentiation corresponding to ∂aiᵢⱼ/∂aₖₗ
    struct ArgsNumDerivInverse {
        a_mat: Matrix, // temporary tensor (3x3 matrix form)
        a: Tensor2,    // temporary tensor
        ai: Tensor2,   // temporary inverse tensor
        i: usize,      // index j of ∂aiᵢⱼ/∂aₖₗ
        j: usize,      // index j of ∂aiᵢⱼ/∂aₖₗ
        k: usize,      // index j of ∂aiᵢⱼ/∂aₖₗ
        l: usize,      // index j of ∂aiᵢⱼ/∂aₖₗ
    }

    // Holds arguments for numerical differentiation corresponding to ∂aiₘ/∂aₙ (Mandel components)
    struct ArgsNumDerivInverseMandel {
        a: Tensor2,  // temporary tensor
        ai: Tensor2, // temporary inverse tensor
        m: usize,    // index of ∂aiₘ/∂aₙ (Mandel components)
        n: usize,    // index of ∂aiₘ/∂aₙ (Mandel components)
    }

    fn component_of_inverse(x: f64, args: &mut ArgsNumDerivInverse) -> f64 {
        let original = args.a_mat.get(args.k, args.l);
        args.a_mat.set(args.k, args.l, x);
        args.a.set_matrix(&args.a_mat).unwrap();
        args.a.inverse(&mut args.ai, 1e-10).unwrap().unwrap();
        args.a_mat.set(args.k, args.l, original);
        args.ai.get(args.i, args.j)
    }

    fn component_of_inverse_mandel(x: f64, args: &mut ArgsNumDerivInverseMandel) -> f64 {
        let original = args.a.vec[args.n];
        args.a.vec[args.n] = x;
        args.a.inverse(&mut args.ai, 1e-10).unwrap().unwrap();
        args.a.vec[args.n] = original;
        args.ai.vec[args.m]
    }

    fn numerical_deriv_inverse(a: &Tensor2) -> Matrix {
        let mut args = ArgsNumDerivInverse {
            a_mat: a.to_matrix(),
            a: Tensor2::new(Mandel::General),
            ai: Tensor2::new(Mandel::General),
            i: 0,
            j: 0,
            k: 0,
            l: 0,
        };
        let mut num_deriv = Matrix::new(9, 9);
        for m in 0..9 {
            for n in 0..9 {
                (args.i, args.j, args.k, args.l) = MN_TO_IJKL[m][n];
                let x = args.a_mat.get(args.k, args.l);
                let res = deriv_central5(x, &mut args, component_of_inverse);
                num_deriv.set(m, n, res);
            }
        }
        num_deriv
    }

    fn numerical_deriv_inverse_mandel(a: &Tensor2) -> Matrix {
        let mut args = ArgsNumDerivInverseMandel {
            a: a.to_general(),
            ai: Tensor2::new(Mandel::General),
            m: 0,
            n: 0,
        };
        let mut num_deriv = Tensor4::new(Mandel::General);
        for m in 0..9 {
            args.m = m;
            for n in 0..9 {
                args.n = n;
                let x = args.a.vec[args.n];
                let res = deriv_central5(x, &mut args, component_of_inverse_mandel);
                num_deriv.mat.set(m, n, res);
            }
        }
        num_deriv.to_matrix()
    }

    fn numerical_deriv_inverse_sym(a: &Tensor2) -> Matrix {
        let mut args = ArgsNumDerivInverseMandel {
            a: Tensor2::new(Mandel::Symmetric),
            ai: Tensor2::new(Mandel::Symmetric),
            m: 0,
            n: 0,
        };
        args.a.vec[0] = a.vec[0];
        args.a.vec[1] = a.vec[1];
        args.a.vec[2] = a.vec[2];
        args.a.vec[3] = a.vec[3];
        if a.vec.dim() > 4 {
            args.a.vec[4] = a.vec[4];
            args.a.vec[5] = a.vec[5];
        }
        let mut num_deriv = Tensor4::new(Mandel::Symmetric);
        for m in 0..6 {
            args.m = m;
            for n in 0..6 {
                args.n = n;
                let x = args.a.vec[args.n];
                let res = deriv_central5(x, &mut args, component_of_inverse_mandel);
                num_deriv.mat.set(m, n, res);
            }
        }
        num_deriv.to_matrix()
    }

    fn check_deriv_inverse_components(ai: &Tensor2, dai_da: &Tensor4, tol: f64) {
        let arr = dai_da.to_array();
        let mat = ai.to_matrix();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        approx_eq(arr[i][j][k][l], -mat.get(i, k) * mat.get(l, j), tol)
                    }
                }
            }
        }
    }

    fn check_deriv_inverse_sym_components(ai: &Tensor2, dai_da: &Tensor4, tol: f64) {
        let arr = dai_da.to_array();
        let mat = ai.to_matrix();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        approx_eq(
                            arr[i][j][k][l],
                            -0.5 * (mat.get(i, k) * mat.get(j, l) + mat.get(i, l) * mat.get(j, k)),
                            tol,
                        )
                    }
                }
            }
        }
    }

    #[test]
    fn deriv_inverse_tensor_works() {
        // general
        let sample = &SamplesTensor2::TENSOR_T;
        let a = Tensor2::from_matrix(&sample.matrix, Mandel::General).unwrap();
        let mut ai = Tensor2::new(Mandel::General);
        a.inverse(&mut ai, 1e-10).unwrap().unwrap();
        let mut dai_da = Tensor4::new(Mandel::General);
        deriv_inverse_tensor(&mut dai_da, &ai).unwrap();
        check_deriv_inverse_components(&ai, &dai_da, 1e-15);
        let dai_da_ana = dai_da.to_matrix();
        let dai_da_num = numerical_deriv_inverse(&a);
        let dai_da_num_mandel = numerical_deriv_inverse_mandel(&a);
        mat_approx_eq(&dai_da_ana, &dai_da_num, 1e-11);
        mat_approx_eq(&dai_da_num, &dai_da_num_mandel, 1e-12);

        // symmetric 3D
        let sample = &SamplesTensor2::TENSOR_U;
        let a = Tensor2::from_matrix(&sample.matrix, Mandel::Symmetric).unwrap();
        let mut ai = Tensor2::new(Mandel::Symmetric);
        a.inverse(&mut ai, 1e-10).unwrap().unwrap();
        let mut dai_da = Tensor4::new(Mandel::General);
        deriv_inverse_tensor(&mut dai_da, &ai).unwrap();
        check_deriv_inverse_components(&ai, &dai_da, 1e-14);
        let dai_da_ana = dai_da.to_matrix();
        let dai_da_num = numerical_deriv_inverse(&a);
        let dai_da_num_mandel = numerical_deriv_inverse_mandel(&a);
        mat_approx_eq(&dai_da_ana, &dai_da_num, 1e-7);
        mat_approx_eq(&dai_da_num, &dai_da_num_mandel, 1e-7);

        // symmetric 2D
        let sample = &SamplesTensor2::TENSOR_Y;
        let a = Tensor2::from_matrix(&sample.matrix, Mandel::Symmetric2D).unwrap();
        let mut ai = Tensor2::new(Mandel::Symmetric2D);
        a.inverse(&mut ai, 1e-10).unwrap().unwrap();
        let mut dai_da = Tensor4::new(Mandel::General);
        deriv_inverse_tensor(&mut dai_da, &ai).unwrap();
        check_deriv_inverse_components(&ai, &dai_da, 1e-15);
        let dai_da_ana = dai_da.to_matrix();
        let dai_da_num = numerical_deriv_inverse(&a);
        let dai_da_num_mandel = numerical_deriv_inverse_mandel(&a);
        mat_approx_eq(&dai_da_ana, &dai_da_num, 1e-12);
        mat_approx_eq(&dai_da_num, &dai_da_num_mandel, 1e-11);
    }

    #[test]
    fn deriv_inverse_tensor_sym_captures_errors() {
        let ai = Tensor2::new(Mandel::General);
        let mut dai_da = Tensor4::new(Mandel::Symmetric);
        assert_eq!(
            deriv_inverse_tensor_sym(&mut dai_da, &ai).err(),
            Some("tensor 'ai' must be Symmetric or Symmetric2D")
        );

        let ai = Tensor2::new(Mandel::Symmetric2D);
        let mut dai_da = Tensor4::new(Mandel::Symmetric2D);
        assert_eq!(
            deriv_inverse_tensor_sym(&mut dai_da, &ai).err(),
            Some("tensor 'dai_da' must be Symmetric")
        );
    }

    #[test]
    fn deriv_inverse_tensor_sym_works() {
        // symmetric 3D
        let sample = &SamplesTensor2::TENSOR_U;
        let a = Tensor2::from_matrix(&sample.matrix, Mandel::Symmetric).unwrap();
        let mut ai = Tensor2::new(Mandel::Symmetric);
        a.inverse(&mut ai, 1e-10).unwrap().unwrap();
        let mut dai_da = Tensor4::new(Mandel::Symmetric);
        deriv_inverse_tensor_sym(&mut dai_da, &ai).unwrap();
        check_deriv_inverse_sym_components(&ai, &dai_da, 1e-14);
        let dai_da_ana = dai_da.to_matrix();
        let dai_da_num = numerical_deriv_inverse_sym(&a);
        mat_approx_eq(&dai_da_ana, &dai_da_num, 1e-7);

        // symmetric 2D
        let sample = &SamplesTensor2::TENSOR_Y;
        let a = Tensor2::from_matrix(&sample.matrix, Mandel::Symmetric2D).unwrap();
        let mut ai = Tensor2::new(Mandel::Symmetric2D);
        a.inverse(&mut ai, 1e-10).unwrap().unwrap();
        let mut dai_da = Tensor4::new(Mandel::Symmetric);
        deriv_inverse_tensor_sym(&mut dai_da, &ai).unwrap();
        check_deriv_inverse_sym_components(&ai, &dai_da, 1e-15);
        let dai_da_ana = dai_da.to_matrix();
        let dai_da_num = numerical_deriv_inverse_sym(&a);
        mat_approx_eq(&dai_da_ana, &dai_da_num, 1e-12);
    }
}
