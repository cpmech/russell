#![allow(unused)]

use crate::{t2_odyad_t2, StrError, Tensor2, Tensor4};

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
    use crate::{deriv_inverse_tensor, Mandel, SampleTensor2, SamplesTensor2, ONE_BY_3, SQRT_3_BY_2};
    use russell_chk::{approx_eq, deriv_central5, vec_approx_eq};
    use russell_lab::{mat_approx_eq, Matrix};

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
    }
}
