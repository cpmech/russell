use super::*;

/// Performs the double dot (ddot) operation between two Tensor2
///
/// c = a : b
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
pub fn t2_ddot_t2(c: &mut Tensor2, a: &Tensor2, b: &Tensor2) {
    let all_symmetric = a.symmetric && b.symmetric;
    if c.symmetric && !all_symmetric {
        panic!("tensor c must not be symmetric when either a or b are not symmetric");
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

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

        let mut c = Tensor2::new(false);

        // c = a : b
        t2_ddot_t2(&mut c, &a, &b);
    }
}
