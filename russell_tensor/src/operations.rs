use super::{Tensor2, Tensor4};
use crate::StrError;
use russell_lab::{mat_vec_mul, vec_mat_mul, Vector};

/// Performs the double-dot (ddot) operation between two Tensor2 (inner product)
///
/// ```text
/// s = a : b
/// ```
///
/// # Arguments
///
/// * `a` - A second-order tensor
/// * `b` - A second-order tensor
///
pub fn t2_ddot_t2(a: &Tensor2, b: &Tensor2) -> f64 {
    #[rustfmt::skip]
    let mut res = a.vec[0] * b.vec[0]
                    + a.vec[1] * b.vec[1]
                    + a.vec[2] * b.vec[2]
                    + a.vec[3] * b.vec[3]
                    + a.vec[4] * b.vec[4]
                    + a.vec[5] * b.vec[5];
    if a.vec.dim() == 9 && b.vec.dim() == 9 {
        // NOTE: Only if both tensors are unsymmetric we have to
        //       compute extra terms because, otherwise, the corresponding
        //       components are zero any way.
        res += a.vec[6] * b.vec[6] + a.vec[7] * b.vec[7] + a.vec[8] * b.vec[8];
    }
    res
}

/// Performs the single dot operation between two Tensor2 (matrix multiplication)
///
/// ```text
/// c = a · b
/// ```
///
/// # Warning
///
/// This function is not very efficient.
///
/// # Note
///
/// - Even if `a` and `b` are symmetric, the result `c` may not be symmetric
/// - Thus, the result is always set with symmetric = false
///
pub fn t2_dot_t2(a: &Tensor2, b: &Tensor2) -> Result<Tensor2, StrError> {
    let ta = a.to_matrix();
    let tb = b.to_matrix();
    let mut tc = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                tc[i][j] += ta[i][k] * tb[k][j];
            }
        }
    }
    Tensor2::from_matrix(&tc, false, false)
}

/// Performs the single dot operation between a Tensor2 and a vector
///
/// ```text
/// v = α a · u
/// ```
pub fn t2_dot_vec(v: &mut Vector, alpha: f64, a: &Tensor2, u: &Vector) -> Result<(), StrError> {
    if a.vec.dim() == 4 {
        if v.dim() != 2 || u.dim() != 2 {
            return Err("vectors must have dim = 2");
        }
        v[0] = alpha * (a.get(0, 0) * u[0] + a.get(0, 1) * u[1]);
        v[1] = alpha * (a.get(1, 0) * u[0] + a.get(1, 1) * u[1]);
    } else {
        if v.dim() != 3 || u.dim() != 3 {
            return Err("vectors must have dim = 3");
        }
        v[0] = alpha * (a.get(0, 0) * u[0] + a.get(0, 1) * u[1] + a.get(0, 2) * u[2]);
        v[1] = alpha * (a.get(1, 0) * u[0] + a.get(1, 1) * u[1] + a.get(1, 2) * u[2]);
        v[2] = alpha * (a.get(2, 0) * u[0] + a.get(2, 1) * u[1] + a.get(2, 2) * u[2]);
    }
    Ok(())
}

/// Performs the single dot operation between a vector and a Tensor2
///
/// ```text
/// v = α u · a
/// ```
pub fn vec_dot_t2(v: &mut Vector, alpha: f64, u: &Vector, a: &Tensor2) -> Result<(), StrError> {
    if a.vec.dim() == 4 {
        if v.dim() != 2 || u.dim() != 2 {
            return Err("vectors must have dim = 2");
        }
    } else {
        if v.dim() != 3 || u.dim() != 3 {
            return Err("vectors must have dim = 3");
        }
        v[0] = alpha * (u[0] * a.get(0, 0) + u[1] * a.get(1, 0) + u[2] * a.get(2, 0));
        v[1] = alpha * (u[0] * a.get(0, 1) + u[1] * a.get(1, 1) + u[2] * a.get(2, 1));
        v[2] = alpha * (u[0] * a.get(0, 2) + u[1] * a.get(1, 2) + u[2] * a.get(2, 2));
    }
    Ok(())
}

/// Performs the double-dot (ddot) operation between a Tensor4 and a Tensor2
///
/// ```text
/// b = α D : a
/// ```
pub fn t4_ddot_t2(b: &mut Tensor2, alpha: f64, dd: &Tensor4, a: &Tensor2) -> Result<(), StrError> {
    mat_vec_mul(&mut b.vec, alpha, &dd.mat, &a.vec)
}

/// Performs the double-dot (ddot) operation between a Tensor2 and a Tensor4
///
/// ```text
/// b = α a : D
/// ```
pub fn t2_ddot_t4(b: &mut Tensor2, alpha: f64, a: &Tensor2, dd: &Tensor4) -> Result<(), StrError> {
    vec_mat_mul(&mut b.vec, alpha, &a.vec, &dd.mat)
}

pub fn t2_dyad_t2() {
    // TODO
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{t2_ddot_t2, t2_dot_t2, Tensor2};
    use crate::StrError;
    use russell_chk::{assert_approx_eq, assert_vec_approx_eq};

    #[test]
    fn t2_ddot_t2_works() -> Result<(), StrError> {
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ], false, false)?;

        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [9.0, 8.0, 7.0],
            [6.0, 5.0, 4.0],
            [3.0, 2.0, 1.0],
        ], false, false)?;
        let s = t2_ddot_t2(&a, &b);
        assert_eq!(s, 165.0);
        Ok(())
    }

    #[test]
    fn t2_ddot_t2_both_symmetric_works() -> Result<(), StrError> {
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0, 5.0, 3.0],
        ], true, false)?;
        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [3.0, 5.0, 6.0],
            [5.0, 2.0, 4.0],
            [6.0, 4.0, 1.0],
        ], true, false)?;
        let s = t2_ddot_t2(&a, &b);
        assert_approx_eq!(s, 162.0, 1e-13);
        Ok(())
    }

    #[test]
    fn t2_ddot_t2_sym_with_unsymmetric_works() -> Result<(), StrError> {
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0, 5.0, 3.0],
        ], true, false)?;
        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [9.0, 8.0, 7.0],
            [6.0, 5.0, 4.0],
            [3.0, 2.0, 1.0],
        ], false, false)?;
        let s = t2_ddot_t2(&a, &b);
        assert_approx_eq!(s, 168.0, 1e-13);
        Ok(())
    }

    #[test]
    fn t2_sdot_t2_works() -> Result<(), StrError> {
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ], false, false)?;
        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [9.0, 8.0, 7.0],
            [6.0, 5.0, 4.0],
            [3.0, 2.0, 1.0],
        ], false, false)?;
        let c = t2_dot_t2(&a, &b)?;
        #[rustfmt::skip]
        let correct = Tensor2::from_matrix(&[
            [ 30.0,  24.0, 18.0],
            [ 84.0,  69.0, 54.0],
            [138.0, 114.0, 90.0],
        ], false, false)?;
        assert_vec_approx_eq!(c.vec.as_data(), correct.vec.as_data(), 1e-13);
        Ok(())
    }

    #[test]
    fn t2_sdot_t2_both_symmetric_works() -> Result<(), StrError> {
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0, 5.0, 3.0],
        ], true, false)?;
        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [3.0, 5.0, 6.0],
            [5.0, 2.0, 4.0],
            [6.0, 4.0, 1.0],
        ], true, false)?;
        let c = t2_dot_t2(&a, &b)?;
        #[rustfmt::skip]
        let correct = Tensor2::from_matrix(&[
            [59.0, 37.0, 28.0],
            [52.0, 44.0, 37.0],
            [61.0, 52.0, 59.0],
        ], false, false)?;
        assert_vec_approx_eq!(c.vec.as_data(), correct.vec.as_data(), 1e-13);
        Ok(())
    }
}
