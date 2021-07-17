use super::*;

/// Performs the double dot (ddot) operation between two Tensor2 (inner product)
///
/// s = a : b
///
/// # Arguments
///
/// * `a` - [input] A second-order tensor
/// * `b` - [input] A second-order tensor
///
pub fn t2_ddot_t2(a: &Tensor2, b: &Tensor2) -> f64 {
    #[rustfmt::skip]
    let mut res = a.comps_mandel[0] * b.comps_mandel[0]
                    + a.comps_mandel[1] * b.comps_mandel[1]
                    + a.comps_mandel[2] * b.comps_mandel[2]
                    + a.comps_mandel[3] * b.comps_mandel[3]
                    + a.comps_mandel[4] * b.comps_mandel[4]
                    + a.comps_mandel[5] * b.comps_mandel[5];
    #[rustfmt::skip]
    if !a.symmetric && !b.symmetric {
        res += a.comps_mandel[6] * b.comps_mandel[6]
             + a.comps_mandel[7] * b.comps_mandel[7]
             + a.comps_mandel[8] * b.comps_mandel[8];
        // NOTE: if any tensor is unsymmetric, there is no need to augment res
        //       because the extra three components are zero
    };
    res
}

/// Performs the single dot (ddot) operation between two Tensor2 (matrix multiplication)
///
/// c = a . b
///
/// # Warning
///
/// This function is not efficient.
///
pub fn t2_sdot_t2(a: &Tensor2, b: &Tensor2) -> Tensor2 {
    let all_symmetric = a.symmetric && b.symmetric;
    let ta = a.to_tensor();
    let tb = b.to_tensor();
    let mut tc = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                tc[i][j] += ta[i][k] * tb[k][j];
            }
        }
    }
    Tensor2::from_tensor(&tc, all_symmetric)
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use russell_chk::*;

    #[test]
    fn t2_ddot_t2_works() {
        #[rustfmt::skip]
        let a = Tensor2::from_tensor(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ], false);

        #[rustfmt::skip]
        let b = Tensor2::from_tensor(&[
            [9.0, 8.0, 7.0],
            [6.0, 5.0, 4.0],
            [3.0, 2.0, 1.0],
        ], false);
        let s = t2_ddot_t2(&a, &b);
        assert_eq!(s, 165.0);
    }

    #[test]
    fn t2_ddot_t2_both_symmetric_works() {
        #[rustfmt::skip]
        let a = Tensor2::from_tensor(&[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0, 5.0, 3.0],
        ], true);
        #[rustfmt::skip]
        let b = Tensor2::from_tensor(&[
            [3.0, 5.0, 6.0],
            [5.0, 2.0, 4.0],
            [6.0, 4.0, 1.0],
        ], true);
        let s = t2_ddot_t2(&a, &b);
        assert_approx_eq!(s, 162.0, 1e-13);
    }

    #[test]
    fn t2_ddot_t2_sym_with_unsymmetric_works() {
        #[rustfmt::skip]
        let a = Tensor2::from_tensor(&[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0, 5.0, 3.0],
        ], true);
        #[rustfmt::skip]
        let b = Tensor2::from_tensor(&[
            [9.0, 8.0, 7.0],
            [6.0, 5.0, 4.0],
            [3.0, 2.0, 1.0],
        ], false);
        let s = t2_ddot_t2(&a, &b);
        assert_approx_eq!(s, 168.0, 1e-13);
    }

    #[test]
    fn t2_sdot_t2_works() {
        #[rustfmt::skip]
        let a = Tensor2::from_tensor(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ], false);
        #[rustfmt::skip]
        let b = Tensor2::from_tensor(&[
            [9.0, 8.0, 7.0],
            [6.0, 5.0, 4.0],
            [3.0, 2.0, 1.0],
        ], false);
        let c = t2_sdot_t2(&a, &b);
        println!("{}", c);
        #[rustfmt::skip]
        let correct = Tensor2::from_tensor(&[
            [ 30.0,  24.0, 18.0],
            [ 84.0,  69.0, 54.0],
            [138.0, 114.0, 90.0],
        ], false);
        assert_vec_approx_eq!(c.comps_mandel, correct.comps_mandel, 1e-13);
    }
}
