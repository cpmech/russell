#![allow(unused)]

use crate::{t2_odyad_t2, StrError, Tensor2, Tensor4};

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
pub fn deriv_inverse_tensor(dai_da: &mut Tensor4, ai: &Tensor2, a: &Tensor2) -> Result<(), StrError> {
    let mut ai_t = ai.clone();
    ai.transpose(&mut ai_t)?;
    t2_odyad_t2(dai_da, -1.0, &ai, &ai_t)
}

pub fn deriv_square_tensor(dd: &mut Tensor4, aa: &Tensor2) -> Result<(), StrError> {
    Err("TODO")
}

pub fn deriv_square_tensor_sym(dd: &mut Tensor4, aa: &Tensor2) -> Result<(), StrError> {
    Err("TODO")
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{Tensor2, Tensor4};
    use crate::{deriv_inverse_tensor, Mandel, SampleTensor2, SamplesTensor2, MN_TO_IJKL, ONE_BY_3, SQRT_3_BY_2};
    use russell_chk::{approx_eq, deriv_central5, vec_approx_eq};
    use russell_lab::{mat_approx_eq, Matrix};

    // Holds arguments for numerical differentiation corresponding to ∂aiᵢⱼ/∂aₖₗ
    struct ArgsNumDerivInverse {
        a_mat: [[f64; 3]; 3], // temporary tensor (3x3 matrix form)
        a: Tensor2,           // temporary tensor
        ai: Tensor2,          // temporary inverse tensor
        i: usize,             // index j of ∂aiᵢⱼ/∂aₖₗ
        j: usize,             // index j of ∂aiᵢⱼ/∂aₖₗ
        k: usize,             // index j of ∂aiᵢⱼ/∂aₖₗ
        l: usize,             // index j of ∂aiᵢⱼ/∂aₖₗ
    }

    fn component_of_inverse(a_kl: f64, args: &mut ArgsNumDerivInverse) -> f64 {
        let a_kl_original = args.a_mat[args.k][args.l];
        args.a_mat[args.k][args.l] = a_kl;
        args.a.set_matrix(&args.a_mat).unwrap();
        args.a.inverse(&mut args.ai, 1e-10).unwrap().unwrap();
        args.a_mat[args.k][args.l] = a_kl_original;
        args.ai.get(args.i, args.j)
    }

    fn numerical_deriv_inverse(a: &Tensor2) -> Matrix {
        let mut args = ArgsNumDerivInverse {
            a_mat: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            a: Tensor2::new(Mandel::General),
            ai: Tensor2::new(Mandel::General),
            i: 0,
            j: 0,
            k: 0,
            l: 0,
        };
        let a_mat = a.to_matrix();
        for i in 0..3 {
            for j in 0..3 {
                args.a_mat[i][j] = a_mat.get(i, j);
            }
        }
        let mut num_deriv = Matrix::new(9, 9);
        for m in 0..9 {
            for n in 0..9 {
                (args.i, args.j, args.k, args.l) = MN_TO_IJKL[m][n];
                let at_x = args.a_mat[args.k][args.l];
                let res = deriv_central5(at_x, &mut args, component_of_inverse);
                num_deriv.set(m, n, res);
            }
        }
        num_deriv
    }

    fn check_deriv_inverse_components(ai: &Tensor2, dai_da: &Tensor4, tol: f64) {
        let array = dai_da.to_array();
        let mat = ai.to_matrix();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        approx_eq(array[i][j][k][l], -mat.get(i, k) * mat.get(l, j), tol)
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
        deriv_inverse_tensor(&mut dai_da, &ai, &a).unwrap();
        check_deriv_inverse_components(&ai, &dai_da, 1e-15);
        let dai_da_ana = dai_da.to_matrix();
        let dai_da_num = numerical_deriv_inverse(&a);
        mat_approx_eq(&dai_da_ana, &dai_da_num, 1e-11);

        // symmetric 3D
        let sample = &SamplesTensor2::TENSOR_U;
        let a = Tensor2::from_matrix(&sample.matrix, Mandel::Symmetric).unwrap();
        let mut ai = Tensor2::new(Mandel::Symmetric);
        a.inverse(&mut ai, 1e-10).unwrap().unwrap();
        let mut dai_da = Tensor4::new(Mandel::General);
        deriv_inverse_tensor(&mut dai_da, &ai, &a).unwrap();
        check_deriv_inverse_components(&ai, &dai_da, 1e-14);
        let dai_da_ana = dai_da.to_matrix();
        let dai_da_num = numerical_deriv_inverse(&a);
        // println!("{:.9}", dai_da_ana);
        // println!("{:.9}", dai_da_num);
        mat_approx_eq(&dai_da_ana, &dai_da_num, 1e-7);

        // symmetric 2D
        let sample = &SamplesTensor2::TENSOR_Y;
        let a = Tensor2::from_matrix(&sample.matrix, Mandel::Symmetric2D).unwrap();
        let mut ai = Tensor2::new(Mandel::Symmetric2D);
        a.inverse(&mut ai, 1e-10).unwrap().unwrap();
        let mut dai_da = Tensor4::new(Mandel::General);
        deriv_inverse_tensor(&mut dai_da, &ai, &a).unwrap();
        check_deriv_inverse_components(&ai, &dai_da, 1e-15);
        let dai_da_ana = dai_da.to_matrix();
        let dai_da_num = numerical_deriv_inverse(&a);
        // println!("{:.9}", dai_da_ana);
        // println!("{:.9}", dai_da_num);
        mat_approx_eq(&dai_da_ana, &dai_da_num, 1e-12);
    }
}
