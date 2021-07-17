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
/// # Arguments
///
/// * `c` - [result] A place to save a second-order tensor
/// * `a` - [input] A second-order tensor
/// * `b` - [input] A second-order tensor
///
/// # Notes
///
/// * If `a` and `b` are symmetric then `c` can be symmetric
/// * If either `a` or `b` are not symmetric then `c` must be non-symmetric
///
/// # Panics
///
/// Panics if `c` is symmetric and either `a` or `b` are not symmetric.
///
pub fn t2_sdot_t2(c: &mut Tensor2, a: &Tensor2, b: &Tensor2) {
    let all_symmetric = a.symmetric && b.symmetric;
    if c.symmetric && !all_symmetric {
        panic!("tensor c must not be symmetric when either a or b are not symmetric");
    }
    // TODO
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
}
