use super::{Tensor2, Tensor4};
use crate::StrError;
use russell_lab::{inner, mat_mat_mul, mat_vec_mul, outer, vec_mat_mul, Vector};

/// Performs the double-dot (ddot) operation between two Tensor2 (inner product)
///
/// ```text
/// s = a : b
/// ```
///
/// Note: this function works with mixed symmetry types.
#[inline]
pub fn t2_ddot_t2(a: &Tensor2, b: &Tensor2) -> f64 {
    inner(&a.vec, &b.vec)
}

/// Performs the single dot operation between two Tensor2 (matrix multiplication)
///
/// ```text
/// c = a · b
/// ```
///
/// # Note
///
/// Even if `a` and `b` are symmetric, the result `c` may not be symmetric.
/// Thus, the result is always a general tensor.
///
/// This function works with mixed symmetry types.
///
/// # Warning
///
/// This function is not very efficient because we convert both tensors to
/// a full matrix representation first.
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
        v[0] = alpha * (u[0] * a.get(0, 0) + u[1] * a.get(1, 0));
        v[1] = alpha * (u[0] * a.get(0, 1) + u[1] * a.get(1, 1));
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

/// Performs the dyadic product between two Tensor2
///
/// ```text
/// D = α a ⊗ b
/// ```
///
/// Note: this function does NOT work with mixed symmetry types.
#[inline]
pub fn t2_dyad_t2(dd: &mut Tensor4, alpha: f64, a: &Tensor2, b: &Tensor2) -> Result<(), StrError> {
    outer(&mut dd.mat, alpha, &a.vec, &b.vec)
}

/// Performs the double-dot (ddot) operation between a Tensor4 and a Tensor2
///
/// ```text
/// b = α D : a
/// ```
///
/// Note: this function does NOT work with mixed symmetry types.
#[inline]
pub fn t4_ddot_t2(b: &mut Tensor2, alpha: f64, dd: &Tensor4, a: &Tensor2) -> Result<(), StrError> {
    mat_vec_mul(&mut b.vec, alpha, &dd.mat, &a.vec)
}

/// Performs the double-dot (ddot) operation between a Tensor2 and a Tensor4
///
/// ```text
/// b = α a : D
/// ```
///
/// Note: this function does NOT work with mixed symmetry types.
#[inline]
pub fn t2_ddot_t4(b: &mut Tensor2, alpha: f64, a: &Tensor2, dd: &Tensor4) -> Result<(), StrError> {
    vec_mat_mul(&mut b.vec, alpha, &a.vec, &dd.mat)
}

/// Performs the double-dot (ddot) operation between two Tensor4
///
/// ```text
/// E = α C : D
/// ```
///
/// Note: this function does NOT work with mixed symmetry types.
#[inline]
pub fn t4_ddot_t4(ee: &mut Tensor4, alpha: f64, cc: &Tensor4, dd: &Tensor4) -> Result<(), StrError> {
    mat_mat_mul(&mut ee.mat, alpha, &cc.mat, &dd.mat)
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{
        t2_ddot_t2, t2_ddot_t4, t2_dot_t2, t2_dot_vec, t2_dyad_t2, t4_ddot_t2, t4_ddot_t4, vec_dot_t2, Tensor2, Tensor4,
    };
    use crate::{Samples, StrError};
    use russell_chk::{assert_approx_eq, assert_vec_approx_eq};
    use russell_lab::Vector;

    #[test]
    fn t2_ddot_t2_works() -> Result<(), StrError> {
        // general : general
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

        // sym-3D : sym-3D
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

        // sym-3D : general
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

        // sym-2D : sym-2D
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ], true, true)?;
        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [3.0, 5.0, 0.0],
            [5.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ], true, true)?;
        let s = t2_ddot_t2(&a, &b);
        assert_approx_eq!(s, 50.0, 1e-13);

        // sym-2D : sym-3D
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ], true, true)?;
        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [3.0, 5.0, 6.0],
            [5.0, 2.0, 4.0],
            [6.0, 4.0, 1.0],
        ], true, false)?;
        let s = t2_ddot_t2(&a, &b);
        assert_approx_eq!(s, 50.0, 1e-13);
        Ok(())
    }

    #[test]
    fn t2_sdot_t2_works() -> Result<(), StrError> {
        // general . general
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

        // sym-3D . sym-3D
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

        // sym-3D . general
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 2.0, 3.0],
            [2.0, 5.0, 6.0],
            [3.0, 6.0, 9.0],
        ], true, false)?;
        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [9.0, 8.0, 7.0],
            [6.0, 5.0, 4.0],
            [3.0, 2.0, 1.0],
        ], false, false)?;
        let c = t2_dot_t2(&a, &b)?;
        #[rustfmt::skip]
        let correct = Tensor2::from_matrix(&[
            [30.0, 24.0, 18.0],
            [66.0, 53.0, 40.0],
            [90.0, 72.0, 54.0],
        ], false, false)?;
        assert_vec_approx_eq!(c.vec.as_data(), correct.vec.as_data(), 1e-13);

        // sym-3D . sym-2D
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 2.0, 3.0],
            [2.0, 5.0, 6.0],
            [3.0, 6.0, 9.0],
        ], true, false)?;
        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [9.0, 8.0, 0.0],
            [8.0, 5.0, 0.0],
            [0.0, 0.0, 1.0],
        ], false, true)?;
        let c = t2_dot_t2(&a, &b)?;
        #[rustfmt::skip]
        let correct = Tensor2::from_matrix(&[
            [25.0, 18.0, 3.0],
            [58.0, 41.0, 6.0],
            [75.0, 54.0, 9.0],
        ], false, false)?;
        assert_vec_approx_eq!(c.vec.as_data(), correct.vec.as_data(), 1e-13);
        Ok(())
    }

    #[test]
    fn t2_dot_vec_works() -> Result<(), StrError> {
        // general . vec
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ], false, false)?;
        let u = Vector::from(&[-2.0, -3.0, -4.0]);
        let mut v = Vector::new(3);
        t2_dot_vec(&mut v, 2.0, &a, &u)?;
        assert_vec_approx_eq!(v.as_data(), &[-40.0, -94.0, -148.0], 1e-13);

        // sym-3D . vec
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 2.0, 3.0],
            [2.0, 5.0, 6.0],
            [3.0, 6.0, 9.0],
        ], true, false)?;
        let u = Vector::from(&[-2.0, -3.0, -4.0]);
        let mut v = Vector::new(3);
        t2_dot_vec(&mut v, 2.0, &a, &u)?;
        assert_vec_approx_eq!(v.as_data(), &[-40.0, -86.0, -120.0], 1e-13);

        // sym-2D . vec
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 2.0, 0.0],
            [2.0, 5.0, 0.0],
            [0.0, 0.0, 9.0],
        ], true, true)?;
        let u = Vector::from(&[-2.0, -3.0]);
        let mut v = Vector::new(2);
        t2_dot_vec(&mut v, 2.0, &a, &u)?;
        assert_vec_approx_eq!(v.as_data(), &[-16.0, -38.0], 1e-13);
        Ok(())
    }

    #[test]
    fn vec_dot_t2_works() -> Result<(), StrError> {
        // general . vec
        let u = Vector::from(&[-2.0, -3.0, -4.0]);
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ], false, false)?;
        let mut v = Vector::new(3);
        vec_dot_t2(&mut v, 2.0, &u, &a)?;
        assert_vec_approx_eq!(v.as_data(), &[-84.0, -102.0, -120.0], 1e-13);

        // sym-3D . vec
        let u = Vector::from(&[-2.0, -3.0, -4.0]);
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 2.0, 3.0],
            [2.0, 5.0, 6.0],
            [3.0, 6.0, 9.0],
        ], true, false)?;
        let mut v = Vector::new(3);
        vec_dot_t2(&mut v, 2.0, &u, &a)?;
        assert_vec_approx_eq!(v.as_data(), &[-40.0, -86.0, -120.0], 1e-13);

        // sym-2D . vec
        let u = Vector::from(&[-2.0, -3.0]);
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 2.0, 0.0],
            [2.0, 5.0, 0.0],
            [0.0, 0.0, 9.0],
        ], true, true)?;
        let mut v = Vector::new(2);
        vec_dot_t2(&mut v, 2.0, &u, &a)?;
        assert_vec_approx_eq!(v.as_data(), &[-16.0, -38.0], 1e-13);
        Ok(())
    }

    #[test]
    fn t2_dyad_t2_works() -> Result<(), StrError> {
        // general dyad general
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ], false, false)?;
        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
        ], false, false)?;
        let mut dd = Tensor4::new(false, false);
        t2_dyad_t2(&mut dd, 2.0, &a, &b)?;
        let mat = dd.to_matrix();
        assert_eq!(
            format!("{:.1}", mat),
            "┌                                     ┐\n\
             │ 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 │\n\
             │ 5.0 5.0 5.0 5.0 5.0 5.0 5.0 5.0 5.0 │\n\
             │ 9.0 9.0 9.0 9.0 9.0 9.0 9.0 9.0 9.0 │\n\
             │ 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 │\n\
             │ 6.0 6.0 6.0 6.0 6.0 6.0 6.0 6.0 6.0 │\n\
             │ 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 │\n\
             │ 4.0 4.0 4.0 4.0 4.0 4.0 4.0 4.0 4.0 │\n\
             │ 8.0 8.0 8.0 8.0 8.0 8.0 8.0 8.0 8.0 │\n\
             │ 7.0 7.0 7.0 7.0 7.0 7.0 7.0 7.0 7.0 │\n\
             └                                     ┘"
        );

        // dym-3D dyad general
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 2.0, 3.0],
            [2.0, 5.0, 6.0],
            [3.0, 6.0, 9.0],
        ], true, false)?;
        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
        ], true, false)?;
        let mut dd = Tensor4::new(true, false);
        t2_dyad_t2(&mut dd, 2.0, &a, &b)?;
        let mat = dd.to_matrix();
        assert_eq!(
            format!("{:.1}", mat),
            "┌                                     ┐\n\
             │ 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 │\n\
             │ 5.0 5.0 5.0 5.0 5.0 5.0 5.0 5.0 5.0 │\n\
             │ 9.0 9.0 9.0 9.0 9.0 9.0 9.0 9.0 9.0 │\n\
             │ 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 │\n\
             │ 6.0 6.0 6.0 6.0 6.0 6.0 6.0 6.0 6.0 │\n\
             │ 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 │\n\
             │ 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 │\n\
             │ 6.0 6.0 6.0 6.0 6.0 6.0 6.0 6.0 6.0 │\n\
             │ 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 │\n\
             └                                     ┘"
        );

        // sym-2D dyad sym-2D
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 2.0, 0.0],
            [2.0, 5.0, 0.0],
            [0.0, 0.0, 9.0],
        ], true, true)?;
        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.0, 0.5],
        ], true, true)?;
        let mut dd = Tensor4::new(true, true);
        t2_dyad_t2(&mut dd, 2.0, &a, &b)?;
        let mat = dd.to_matrix();
        assert_eq!(
            format!("{:.1}", mat),
            "┌                                     ┐\n\
             │ 1.0 1.0 1.0 1.0 0.0 0.0 1.0 0.0 0.0 │\n\
             │ 5.0 5.0 5.0 5.0 0.0 0.0 5.0 0.0 0.0 │\n\
             │ 9.0 9.0 9.0 9.0 0.0 0.0 9.0 0.0 0.0 │\n\
             │ 2.0 2.0 2.0 2.0 0.0 0.0 2.0 0.0 0.0 │\n\
             │ 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 │\n\
             │ 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 │\n\
             │ 2.0 2.0 2.0 2.0 0.0 0.0 2.0 0.0 0.0 │\n\
             │ 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 │\n\
             │ 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 │\n\
             └                                     ┘"
        );
        Ok(())
    }

    #[test]
    fn t4_ddot_t2_works() -> Result<(), StrError> {
        let dd = Tensor4::from_matrix(&Samples::TENSOR4_SYM_2D_SAMPLE1_STD_MATRIX, true, true)?;
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [-1.0, -2.0,  0.0],
            [-2.0,  2.0,  0.0],
            [ 0.0,  0.0, -3.0]], true, true)?;
        let mut b = Tensor2::new(true, true);
        t4_ddot_t2(&mut b, 1.0, &dd, &a)?;
        let out = b.to_matrix();
        assert_eq!(
            format!("{:.1}", out),
            "┌                      ┐\n\
             │  -46.0 -154.0    0.0 │\n\
             │ -154.0  -64.0    0.0 │\n\
             │    0.0    0.0  -82.0 │\n\
             └                      ┘"
        );
        Ok(())
    }

    #[test]
    fn t2_ddot_t4_works() -> Result<(), StrError> {
        let dd = Tensor4::from_matrix(&Samples::TENSOR4_SYM_2D_SAMPLE1_STD_MATRIX, true, true)?;
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [-1.0, -2.0,  0.0],
            [-2.0,  2.0,  0.0],
            [ 0.0,  0.0, -3.0]], true, true)?;
        let mut b = Tensor2::new(true, true);
        t2_ddot_t4(&mut b, 1.0, &a, &dd)?;
        let out = b.to_matrix();
        assert_eq!(
            format!("{:.1}", out),
            "┌                      ┐\n\
             │  -90.0 -144.0    0.0 │\n\
             │ -144.0  -96.0    0.0 │\n\
             │    0.0    0.0 -102.0 │\n\
             └                      ┘"
        );
        Ok(())
    }

    #[test]
    fn t4_ddot_t4_works() -> Result<(), StrError> {
        let cc = Tensor4::from_matrix(&Samples::TENSOR4_SYM_2D_SAMPLE1_STD_MATRIX, true, true)?;
        let mut ee = Tensor4::new(true, true);
        t4_ddot_t4(&mut ee, 1.0, &cc, &cc)?;
        let out = ee.to_matrix();
        assert_eq!(
            format!("{:.1}", out),
            "┌                                                                ┐\n\
             │  410.0  436.0  462.0  644.0    0.0    0.0  644.0    0.0    0.0 │\n\
             │  560.0  601.0  642.0  929.0    0.0    0.0  929.0    0.0    0.0 │\n\
             │  710.0  766.0  822.0 1214.0    0.0    0.0 1214.0    0.0    0.0 │\n\
             │ 1310.0 1426.0 1542.0 2354.0    0.0    0.0 2354.0    0.0    0.0 │\n\
             │    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0 │\n\
             │    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0 │\n\
             │ 1310.0 1426.0 1542.0 2354.0    0.0    0.0 2354.0    0.0    0.0 │\n\
             │    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0 │\n\
             │    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0 │\n\
             └                                                                ┘"
        );
        Ok(())
    }
}
