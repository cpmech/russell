use super::{Tensor2, Tensor4};
use crate::{Mandel, StrError, SQRT_2};
use russell_lab::{mat_copy, mat_mat_mul, mat_vec_mul, vec_inner, vec_mat_mul, vec_outer, Vector};

/// Copies Tensor2
///
/// ```text
/// b := a
/// ```
///
/// # Example
///
/// ```
/// use russell_tensor::{copy_tensor2, Mandel, Tensor2, StrError};
///
/// fn main() -> Result<(), StrError> {
///     let a = Tensor2::from_matrix(&[
///         [1.0, 4.0, 6.0],
///         [7.0, 2.0, 5.0],
///         [9.0, 8.0, 3.0],
///     ], Mandel::General)?;
///
///     let mut b = Tensor2::new(Mandel::General);
///     copy_tensor2(&mut b, &a)?;
///
///     assert_eq!(
///         format!("{:.1}", b.to_matrix()),
///         "┌             ┐\n\
///          │ 1.0 4.0 6.0 │\n\
///          │ 7.0 2.0 5.0 │\n\
///          │ 9.0 8.0 3.0 │\n\
///          └             ┘"
///     );
///     Ok(())
/// }
/// ```
pub fn copy_tensor2(b: &mut Tensor2, a: &Tensor2) -> Result<(), StrError> {
    let n = a.vec.dim();
    if b.vec.dim() != n {
        return Err("second-order tensors are incompatible");
    }
    b.vec.as_mut_data().clone_from_slice(&a.vec.as_data());
    Ok(())
}

/// Copies Tensor4
///
/// ```text
/// E := D
/// ```
///
/// # Example
///
/// ```
/// use russell_tensor::{copy_tensor4, Mandel, Tensor4, StrError};
///
/// fn main() -> Result<(), StrError> {
///     let dd = Tensor4::from_matrix(&[
///         [  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0],
///         [ -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0],
///         [  2.0,  4.0,  6.0,  8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
///         [ 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
///         [  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
///         [  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
///         [ -2.0, -4.0, -6.0, -8.0,-10.0,-12.0,-14.0,-16.0,-18.0],
///         [  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
///         [  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
///     ], Mandel::General)?;
///
///     let mut ee = Tensor4::new(Mandel::General);
///     copy_tensor4(&mut ee, &dd)?;
///
///     assert_eq!(format!("{:.1}", ee.to_matrix()), format!("{:.1}", dd.to_matrix()));
///     Ok(())
/// }
/// ```
#[inline]
pub fn copy_tensor4(ee: &mut Tensor4, dd: &Tensor4) -> Result<(), StrError> {
    mat_copy(&mut ee.mat, &dd.mat)
}

/// Performs the double-dot (ddot) operation between two Tensor2 (inner product)
///
/// ```text
/// s = a : b
/// ```
///
/// Note: this function works with mixed symmetry types.
///
/// # Example
///
/// ```
/// use russell_chk::approx_eq;
/// use russell_tensor::{t2_ddot_t2, Mandel, Tensor2, StrError};
///
/// fn main() -> Result<(), StrError> {
///     let a = Tensor2::from_matrix(&[
///         [1.0,  1.0, 0.0],
///         [1.0, -1.0, 0.0],
///         [0.0,  0.0, 1.0],
///     ], Mandel::Symmetric2D)?;
///
///     let b = Tensor2::from_matrix(&[
///         [1.0,  2.0, 0.0],
///         [3.0, -1.0, 5.0],
///         [0.0,  4.0, 1.0],
///     ], Mandel::General)?;
///
///     let res = t2_ddot_t2(&a, &b);
///
///     approx_eq(res, 8.0, 1e-15);
///     Ok(())
/// }
/// ```
#[inline]
pub fn t2_ddot_t2(a: &Tensor2, b: &Tensor2) -> f64 {
    vec_inner(&a.vec, &b.vec)
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
///
/// # Example
///
/// ```
/// use russell_tensor::{t2_dot_t2, Mandel, Tensor2, StrError};
///
/// fn main() -> Result<(), StrError> {
///     let a = Tensor2::from_matrix(&[
///         [1.0,  1.0, 0.0],
///         [1.0, -1.0, 0.0],
///         [0.0,  0.0, 1.0],
///     ], Mandel::Symmetric)?;
///
///     let b = Tensor2::from_matrix(&[
///         [1.0,  2.0, 0.0],
///         [3.0, -1.0, 5.0],
///         [0.0,  4.0, 1.0],
///     ], Mandel::General)?;
///
///     let c = t2_dot_t2(&a, &b)?;
///     assert_eq!(
///         format!("{:.1}", c.to_matrix()),
///         "┌                ┐\n\
///          │  4.0  1.0  5.0 │\n\
///          │ -2.0  3.0 -5.0 │\n\
///          │  0.0  4.0  1.0 │\n\
///          └                ┘"
///     );
///     Ok(())
/// }
/// ```
pub fn t2_dot_t2(a: &Tensor2, b: &Tensor2) -> Result<Tensor2, StrError> {
    let ta = a.to_matrix();
    let tb = b.to_matrix();
    let mut tc = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                tc[i][j] += ta.get(i, k) * tb.get(k, j);
            }
        }
    }
    Tensor2::from_matrix(&tc, Mandel::General)
}

/// Performs the single dot operation between a Tensor2 and a vector
///
/// ```text
/// v = α a · u
/// ```
///
/// # Example
///
/// ```
/// use russell_lab::Vector;
/// use russell_tensor::{t2_dot_vec, Mandel, Tensor2, StrError};
///
/// fn main() -> Result<(), StrError> {
///     let a = Tensor2::from_matrix(&[
///         [1.0,  1.0, 0.0],
///         [1.0, -1.0, 0.0],
///         [0.0,  0.0, 1.0],
///     ], Mandel::Symmetric2D)?;
///
///     let u = Vector::from(&[1.0, 2.0]);
///     let mut v = Vector::new(2);
///     t2_dot_vec(&mut v, 2.0, &a, &u)?;
///
///     assert_eq!(
///         format!("{:.1}", v),
///         "┌      ┐\n\
///          │  6.0 │\n\
///          │ -2.0 │\n\
///          └      ┘"
///     );
///     Ok(())
/// }
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
///
/// # Example
///
/// ```
/// use russell_lab::Vector;
/// use russell_tensor::{vec_dot_t2, Mandel, Tensor2, StrError};
///
/// fn main() -> Result<(), StrError> {
///     let u = Vector::from(&[1.0, 2.0]);
///     let a = Tensor2::from_matrix(&[
///         [1.0,  1.0, 0.0],
///         [1.0, -1.0, 0.0],
///         [0.0,  0.0, 1.0],
///     ], Mandel::Symmetric2D)?;
///
///     let mut v = Vector::new(2);
///     vec_dot_t2(&mut v, 2.0, &u, &a)?;
///
///     assert_eq!(
///         format!("{:.1}", v),
///         "┌      ┐\n\
///          │  6.0 │\n\
///          │ -2.0 │\n\
///          └      ┘"
///     );
///     Ok(())
/// }
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

/// Performs the dyadic product between two vectors resulting in a second-order tensor
///
/// ```text
/// T = α u ⊗ v
/// ```
///
/// # Notes
///
/// * Note that, in general, the dyadic product between two vectors
///   may result in a **non-symmetric** second-order tensor. Therefore,
///   if the input tensor `T` is symmetric, an error may occur. Thus,
///   make sure that the you expect `u ⊗ v` to be symmetric when passing
///   a symmetric tensor `T`.
///
/// # Example
///
/// ```
/// use russell_lab::Vector;
/// use russell_tensor::{vec_dyad_vec, Mandel, Tensor2, StrError};
///
/// fn main() -> Result<(), StrError> {
///     let u = Vector::from(&[1.0, 1.0, 1.0]);
///     let v = Vector::from(&[2.0, 2.0, 2.0]);
///
///     let mut tt = Tensor2::new(Mandel::Symmetric);
///     vec_dyad_vec(&mut tt, 1.0, &u, &v)?;
///
///     assert_eq!(
///         format!("{:.1}", tt.to_matrix()),
///         "┌             ┐\n\
///          │ 2.0 2.0 2.0 │\n\
///          │ 2.0 2.0 2.0 │\n\
///          │ 2.0 2.0 2.0 │\n\
///          └             ┘"
///     );
///     Ok(())
/// }
/// ```
#[inline]
pub fn vec_dyad_vec(tt: &mut Tensor2, alpha: f64, u: &Vector, v: &Vector) -> Result<(), StrError> {
    if tt.two_dim() {
        // and symmetric
        if u.dim() != 2 || v.dim() != 2 {
            return Err("vectors must have dim = 2");
        }
        if (u[0] * v[1]) != (u[1] * v[0]) {
            return Err("dyadic product between u and v does not generate a symmetric tensor");
        }
        tt.vec[0] = alpha * u[0] * v[0];
        tt.vec[1] = alpha * u[1] * v[1];
        tt.vec[2] = 0.0;
        tt.vec[3] = alpha * (u[0] * v[1] + u[1] * v[0]) / SQRT_2;
    } else {
        if u.dim() != 3 || v.dim() != 3 {
            return Err("vectors must have dim = 3");
        }
        tt.vec[0] = alpha * u[0] * v[0];
        tt.vec[1] = alpha * u[1] * v[1];
        tt.vec[2] = alpha * u[2] * v[2];
        tt.vec[3] = alpha * (u[0] * v[1] + u[1] * v[0]) / SQRT_2;
        tt.vec[4] = alpha * (u[1] * v[2] + u[2] * v[1]) / SQRT_2;
        tt.vec[5] = alpha * (u[0] * v[2] + u[2] * v[0]) / SQRT_2;
        if tt.symmetric() {
            if (u[0] * v[1]) != (u[1] * v[0]) || (u[1] * v[2]) != (u[2] * v[1]) || (u[0] * v[2]) != (u[2] * v[0]) {
                return Err("dyadic product between u and v does not generate a symmetric tensor");
            }
        } else {
            tt.vec[6] = alpha * (u[0] * v[1] - u[1] * v[0]) / SQRT_2;
            tt.vec[7] = alpha * (u[1] * v[2] - u[2] * v[1]) / SQRT_2;
            tt.vec[8] = alpha * (u[0] * v[2] - u[2] * v[0]) / SQRT_2;
        }
    }
    Ok(())
}

/// Performs the dyadic product between two Tensor2 resulting in a Tensor4
///
/// ```text
/// D = α a ⊗ b
/// ```
///
/// ```text
/// With orthonormal Cartesian components:
///
/// Dᵢⱼₖₗ = α aᵢⱼ bₖₗ
/// ```
///
/// Note: this function does NOT work with mixed symmetry types.
///
/// # Example
///
/// ```
/// use russell_tensor::{t2_dyad_t2, Mandel, Tensor2, Tensor4, StrError};
///
/// fn main() -> Result<(), StrError> {
///     let a = Tensor2::from_matrix(&[
///         [ 1.0, 10.0, 0.0],
///         [-2.0, -1.0, 0.0],
///         [ 0.0,  0.0, 2.0],
///     ], Mandel::General)?;
///
///     let b = Tensor2::from_matrix(&[
///         [1.0, 4.0, 6.0],
///         [7.0, 2.0, 5.0],
///         [9.0, 8.0, 3.0],
///     ], Mandel::General)?;
///
///     let mut dd = Tensor4::new(Mandel::General);
///     t2_dyad_t2(&mut dd, 1.0, &a, &b)?;
///
///     assert_eq!(
///         format!("{:.1}", dd.to_matrix()),
///         "┌                                                       ┐\n\
///          │   1.0   2.0   3.0   4.0   5.0   6.0   7.0   8.0   9.0 │\n\
///          │  -1.0  -2.0  -3.0  -4.0  -5.0  -6.0  -7.0  -8.0  -9.0 │\n\
///          │   2.0   4.0   6.0   8.0  10.0  12.0  14.0  16.0  18.0 │\n\
///          │  10.0  20.0  30.0  40.0  50.0  60.0  70.0  80.0  90.0 │\n\
///          │   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 │\n\
///          │   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 │\n\
///          │  -2.0  -4.0  -6.0  -8.0 -10.0 -12.0 -14.0 -16.0 -18.0 │\n\
///          │   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 │\n\
///          │   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 │\n\
///          └                                                       ┘"
///     );
///     Ok(())
/// }
/// ```
#[inline]
pub fn t2_dyad_t2(dd: &mut Tensor4, alpha: f64, a: &Tensor2, b: &Tensor2) -> Result<(), StrError> {
    vec_outer(&mut dd.mat, alpha, &a.vec, &b.vec)
}

/// Performs the overbar dyadic product between two Tensor2 resulting in a (general) Tensor4
///
/// ```text
///         _
/// D = α A ⊗ B
/// ```
///
/// ```text
/// With orthonormal Cartesian components:
///
/// Dᵢⱼₖₗ = α Aᵢₖ Bⱼₗ
/// ```
///
/// **Important:** The result is **not** necessarily minor-symmetric; therefore `dd` must be General.
#[rustfmt::skip]
pub fn t2_odyad_t2(dd: &mut Tensor4, _alpha: f64, aa: &Tensor2, bb: &Tensor2) -> Result<(), StrError> {
    let a = &aa.vec;
    let b = &bb.vec;
    let tsq2 = 2.0 * SQRT_2;

    dd.mat.set(0,0, a[0]*b[0]);
    dd.mat.set(0,1, ((a[3] + a[6])*(b[3] + b[6]))/2.0);
    dd.mat.set(0,2, ((a[5] + a[8])*(b[5] + b[8]))/2.0);
    dd.mat.set(0,3, (a[3]*b[0] + a[6]*b[0] + a[0]*(b[3] + b[6]))/2.0);
    dd.mat.set(0,4, ((a[5] + a[8])*(b[3] + b[6]) + (a[3] + a[6])*(b[5] + b[8]))/tsq2);
    dd.mat.set(0,5, (a[5]*b[0] + a[8]*b[0] + a[0]*(b[5] + b[8]))/2.0);
    dd.mat.set(0,6, (-(a[3]*b[0]) - a[6]*b[0] + a[0]*(b[3] + b[6]))/2.0);
    dd.mat.set(0,7, (-((a[5] + a[8])*(b[3] + b[6])) + (a[3] + a[6])*(b[5] + b[8]))/tsq2);
    dd.mat.set(0,8, (-(a[5]*b[0]) - a[8]*b[0] + a[0]*(b[5] + b[8]))/2.0);

    dd.mat.set(1,0, ((a[3] - a[6])*(b[3] - b[6]))/2.0);
    dd.mat.set(1,1, a[1]*b[1]);
    dd.mat.set(1,2, ((a[4] + a[7])*(b[4] + b[7]))/2.0);
    dd.mat.set(1,3, (a[3]*b[1] - a[6]*b[1] + a[1]*(b[3] - b[6]))/2.0);
    dd.mat.set(1,4, (a[4]*b[1] + a[7]*b[1] + a[1]*(b[4] + b[7]))/2.0);
    dd.mat.set(1,5, ((a[4] + a[7])*(b[3] - b[6]) + (a[3] - a[6])*(b[4] + b[7]))/tsq2);
    dd.mat.set(1,6, (a[3]*b[1] - a[6]*b[1] + a[1]*(-b[3] + b[6]))/2.0);
    dd.mat.set(1,7, (-(a[4]*b[1]) - a[7]*b[1] + a[1]*(b[4] + b[7]))/2.0);
    dd.mat.set(1,8, (-((a[4] + a[7])*(b[3] - b[6])) + (a[3] - a[6])*(b[4] + b[7]))/tsq2);

    dd.mat.set(2,0, ((a[5] - a[8])*(b[5] - b[8]))/2.0);
    dd.mat.set(2,1, ((a[4] - a[7])*(b[4] - b[7]))/2.0);
    dd.mat.set(2,2, a[2]*b[2]);
    dd.mat.set(2,3, ((a[5] - a[8])*(b[4] - b[7]) + (a[4] - a[7])*(b[5] - b[8]))/tsq2);
    dd.mat.set(2,4, (a[4]*b[2] - a[7]*b[2] + a[2]*(b[4] - b[7]))/2.0);
    dd.mat.set(2,5, (a[5]*b[2] - a[8]*b[2] + a[2]*(b[5] - b[8]))/2.0);
    dd.mat.set(2,6, ((a[5] - a[8])*(b[4] - b[7]) - (a[4] - a[7])*(b[5] - b[8]))/tsq2);
    dd.mat.set(2,7, (a[4]*b[2] - a[7]*b[2] + a[2]*(-b[4] + b[7]))/2.0);
    dd.mat.set(2,8, (a[5]*b[2] - a[8]*b[2] + a[2]*(-b[5] + b[8]))/2.0);

    dd.mat.set(3,0, (a[3]*b[0] - a[6]*b[0] + a[0]*(b[3] - b[6]))/2.0);
    dd.mat.set(3,1, (a[3]*b[1] + a[6]*b[1] + a[1]*(b[3] + b[6]))/2.0);
    dd.mat.set(3,2, ((a[5] + a[8])*(b[4] + b[7]) + (a[4] + a[7])*(b[5] + b[8]))/tsq2);
    dd.mat.set(3,3, (a[1]*b[0] + a[0]*b[1] + a[3]*b[3] - a[6]*b[6])/2.0);
    dd.mat.set(3,4, (SQRT_2*(a[5] + a[8])*b[1] + (a[4] + a[7])*(b[3] + b[6]) + (a[3] + a[6])*(b[4] + b[7]) + SQRT_2*a[1]*(b[5] + b[8]))/4.0);
    dd.mat.set(3,5, (SQRT_2*(a[4] + a[7])*b[0] + (a[5] + a[8])*(b[3] - b[6]) + SQRT_2*a[0]*(b[4] + b[7]) + (a[3] - a[6])*(b[5] + b[8]))/4.0);
    dd.mat.set(3,6, (-(a[1]*b[0]) + a[0]*b[1] - a[6]*b[3] + a[3]*b[6])/2.0);
    dd.mat.set(3,7, (-(SQRT_2*(a[5] + a[8])*b[1]) - (a[4] + a[7])*(b[3] + b[6]) + (a[3] + a[6])*(b[4] + b[7]) + SQRT_2*a[1]*(b[5] + b[8]))/4.0);
    dd.mat.set(3,8, (-(SQRT_2*(a[4] + a[7])*b[0]) - (a[5] + a[8])*(b[3] - b[6]) + SQRT_2*a[0]*(b[4] + b[7]) + (a[3] - a[6])*(b[5] + b[8]))/4.0);

    dd.mat.set(4,0, ((a[5] - a[8])*(b[3] - b[6]) + (a[3] - a[6])*(b[5] - b[8]))/tsq2);
    dd.mat.set(4,1, (a[4]*b[1] - a[7]*b[1] + a[1]*(b[4] - b[7]))/2.0);
    dd.mat.set(4,2, (a[4]*b[2] + a[7]*b[2] + a[2]*(b[4] + b[7]))/2.0);
    dd.mat.set(4,3, (SQRT_2*(a[5] - a[8])*b[1] + (a[4] - a[7])*(b[3] - b[6]) + (a[3] - a[6])*(b[4] - b[7]) + SQRT_2*a[1]*(b[5] - b[8]))/4.0);
    dd.mat.set(4,4, (a[2]*b[1] + a[1]*b[2] + a[4]*b[4] - a[7]*b[7])/2.0);
    dd.mat.set(4,5, (SQRT_2*(a[3] - a[6])*b[2] + SQRT_2*a[2]*(b[3] - b[6]) + (a[5] - a[8])*(b[4] + b[7]) + (a[4] + a[7])*(b[5] - b[8]))/4.0);
    dd.mat.set(4,6, (SQRT_2*(a[5] - a[8])*b[1] - (a[4] - a[7])*(b[3] - b[6]) + (a[3] - a[6])*(b[4] - b[7]) - SQRT_2*a[1]*(b[5] - b[8]))/4.0);
    dd.mat.set(4,7, (-(a[2]*b[1]) + a[1]*b[2] - a[7]*b[4] + a[4]*b[7])/2.0);
    dd.mat.set(4,8, (SQRT_2*(a[3] - a[6])*b[2] - SQRT_2*a[2]*(b[3] - b[6]) + (a[5] - a[8])*(b[4] + b[7]) - (a[4] + a[7])*(b[5] - b[8]))/4.0);

    dd.mat.set(5,0, (a[5]*b[0] - a[8]*b[0] + a[0]*(b[5] - b[8]))/2.0);
    dd.mat.set(5,1, ((a[4] - a[7])*(b[3] + b[6]) + (a[3] + a[6])*(b[4] - b[7]))/tsq2);
    dd.mat.set(5,2, (a[5]*b[2] + a[8]*b[2] + a[2]*(b[5] + b[8]))/2.0);
    dd.mat.set(5,3, (SQRT_2*(a[4] - a[7])*b[0] + (a[5] - a[8])*(b[3] + b[6]) + SQRT_2*a[0]*(b[4] - b[7]) + (a[3] + a[6])*(b[5] - b[8]))/4.0);
    dd.mat.set(5,4, (SQRT_2*(a[3] + a[6])*b[2] + SQRT_2*a[2]*(b[3] + b[6]) + (a[5] + a[8])*(b[4] - b[7]) + (a[4] - a[7])*(b[5] + b[8]))/4.0);
    dd.mat.set(5,5, (a[2]*b[0] + a[0]*b[2] + a[5]*b[5] - a[8]*b[8])/2.0);
    dd.mat.set(5,6, (-(SQRT_2*(a[4] - a[7])*b[0]) + (a[5] - a[8])*(b[3] + b[6]) + SQRT_2*a[0]*(b[4] - b[7]) - (a[3] + a[6])*(b[5] - b[8]))/4.0);
    dd.mat.set(5,7, (SQRT_2*(a[3] + a[6])*b[2] - SQRT_2*a[2]*(b[3] + b[6]) - (a[5] + a[8])*(b[4] - b[7]) + (a[4] - a[7])*(b[5] + b[8]))/4.0);
    dd.mat.set(5,8, (-(a[2]*b[0]) + a[0]*b[2] - a[8]*b[5] + a[5]*b[8])/2.0);

    dd.mat.set(6,0, (-(a[3]*b[0]) + a[6]*b[0] + a[0]*(b[3] - b[6]))/2.0);
    dd.mat.set(6,1, (a[3]*b[1] + a[6]*b[1] - a[1]*(b[3] + b[6]))/2.0);
    dd.mat.set(6,2, ((a[5] + a[8])*(b[4] + b[7]) - (a[4] + a[7])*(b[5] + b[8]))/tsq2);
    dd.mat.set(6,3, (-(a[1]*b[0]) + a[0]*b[1] + a[6]*b[3] - a[3]*b[6])/2.0);
    dd.mat.set(6,4, (SQRT_2*(a[5] + a[8])*b[1] - (a[4] + a[7])*(b[3] + b[6]) + (a[3] + a[6])*(b[4] + b[7]) - SQRT_2*a[1]*(b[5] + b[8]))/4.0);
    dd.mat.set(6,5, (-(SQRT_2*(a[4] + a[7])*b[0]) + (a[5] + a[8])*(b[3] - b[6]) + SQRT_2*a[0]*(b[4] + b[7]) - (a[3] - a[6])*(b[5] + b[8]))/4.0);
    dd.mat.set(6,6, (a[1]*b[0] + a[0]*b[1] - a[3]*b[3] + a[6]*b[6])/2.0);
    dd.mat.set(6,7, (-(SQRT_2*(a[5] + a[8])*b[1]) + (a[4] + a[7])*(b[3] + b[6]) + (a[3] + a[6])*(b[4] + b[7]) - SQRT_2*a[1]*(b[5] + b[8]))/4.0);
    dd.mat.set(6,8, (SQRT_2*(a[4] + a[7])*b[0] - (a[5] + a[8])*(b[3] - b[6]) + SQRT_2*a[0]*(b[4] + b[7]) - (a[3] - a[6])*(b[5] + b[8]))/4.0);

    dd.mat.set(7,0, (-((a[5] - a[8])*(b[3] - b[6])) + (a[3] - a[6])*(b[5] - b[8]))/tsq2);
    dd.mat.set(7,1, (-(a[4]*b[1]) + a[7]*b[1] + a[1]*(b[4] - b[7]))/2.0);
    dd.mat.set(7,2, (a[4]*b[2] + a[7]*b[2] - a[2]*(b[4] + b[7]))/2.0);
    dd.mat.set(7,3, (-(SQRT_2*(a[5] - a[8])*b[1]) - (a[4] - a[7])*(b[3] - b[6]) + (a[3] - a[6])*(b[4] - b[7]) + SQRT_2*a[1]*(b[5] - b[8]))/4.0);
    dd.mat.set(7,4, (-(a[2]*b[1]) + a[1]*b[2] + a[7]*b[4] - a[4]*b[7])/2.0);
    dd.mat.set(7,5, (SQRT_2*(a[3] - a[6])*b[2] - SQRT_2*a[2]*(b[3] - b[6]) - (a[5] - a[8])*(b[4] + b[7]) + (a[4] + a[7])*(b[5] - b[8]))/4.0);
    dd.mat.set(7,6, (-(SQRT_2*(a[5] - a[8])*b[1]) + (a[4] - a[7])*(b[3] - b[6]) + (a[3] - a[6])*(b[4] - b[7]) - SQRT_2*a[1]*(b[5] - b[8]))/4.0);
    dd.mat.set(7,7, (a[2]*b[1] + a[1]*b[2] - a[4]*b[4] + a[7]*b[7])/2.0);
    dd.mat.set(7,8, (SQRT_2*(a[3] - a[6])*b[2] + SQRT_2*a[2]*(b[3] - b[6]) - (a[5] - a[8])*(b[4] + b[7]) - (a[4] + a[7])*(b[5] - b[8]))/4.0);

    dd.mat.set(8,0, (-(a[5]*b[0]) + a[8]*b[0] + a[0]*(b[5] - b[8]))/2.0);
    dd.mat.set(8,1, (-((a[4] - a[7])*(b[3] + b[6])) + (a[3] + a[6])*(b[4] - b[7]))/tsq2);
    dd.mat.set(8,2, (a[5]*b[2] + a[8]*b[2] - a[2]*(b[5] + b[8]))/2.0);
    dd.mat.set(8,3, (-(SQRT_2*(a[4] - a[7])*b[0]) - (a[5] - a[8])*(b[3] + b[6]) + SQRT_2*a[0]*(b[4] - b[7]) + (a[3] + a[6])*(b[5] - b[8]))/4.0);
    dd.mat.set(8,4, (SQRT_2*(a[3] + a[6])*b[2] - SQRT_2*a[2]*(b[3] + b[6]) + (a[5] + a[8])*(b[4] - b[7]) - (a[4] - a[7])*(b[5] + b[8]))/4.0);
    dd.mat.set(8,5, (-(a[2]*b[0]) + a[0]*b[2] + a[8]*b[5] - a[5]*b[8])/2.0);
    dd.mat.set(8,6, (SQRT_2*(a[4] - a[7])*b[0] - (a[5] - a[8])*(b[3] + b[6]) + SQRT_2*a[0]*(b[4] - b[7]) - (a[3] + a[6])*(b[5] - b[8]))/4.0);
    dd.mat.set(8,7, (SQRT_2*(a[3] + a[6])*b[2] + SQRT_2*a[2]*(b[3] + b[6]) - (a[5] + a[8])*(b[4] - b[7]) - (a[4] - a[7])*(b[5] + b[8]))/4.0);
    dd.mat.set(8,8, (a[2]*b[0] + a[0]*b[2] - a[5]*b[5] + a[8]*b[8])/2.0);
    Ok(())
}

/// Performs the underbar dyadic product between two Tensor2 resulting in a (general) Tensor4
///
/// ```text
/// D = α a ⊗ b
///         ‾
/// ```
///
/// ```text
/// With orthonormal Cartesian components:
///
/// Dᵢⱼₖₗ = α aᵢₗ bⱼₖ
/// ```
///
/// **Important:** The result is **not** necessarily minor-symmetric; therefore `dd` must be [Mandel::General].
#[inline]
pub fn t2_udyad_t2(_dd: &mut Tensor4, _alpha: f64, _a: &Tensor2, _b: &Tensor2) -> Result<(), StrError> {
    Err("TODO")
}

/// Performs the symmetric dyad operation with symmetric Tensor2 resulting in a minor-symmetric Tensor4
///
/// ```text
///          _
/// D = α (a ⊗ b + a ⊗ b)
///                  ‾
/// ```
///
/// ```text
/// With orthonormal Cartesian components:
///
/// Dᵢⱼₖₗ = α (aᵢₖ bⱼₗ + aᵢₗ bⱼₖ)
/// ```
///
/// Note: `a` and `b` must be [Mandel::Symmetric] or [Mandel::Symmetric2D].
///
/// The result is minor-symmetric; therefore `dd` is required to be [Mandel::Symmetric] or [Mandel::Symmetric2D].
#[inline]
pub fn t2_sym_dyad_t2(_dd: &mut Tensor4, _alpha: f64, _a: &Tensor2, _b: &Tensor2) -> Result<(), StrError> {
    Err("TODO")
}

/// Performs the double-dot (ddot) operation between a Tensor4 and a Tensor2
///
/// ```text
/// b = α D : a
/// ```
///
/// Note: this function does NOT work with mixed symmetry types.
///
/// # Example
///
/// ```
/// use russell_tensor::{t4_ddot_t2, Mandel, Tensor2, Tensor4, StrError};
///
/// fn main() -> Result<(), StrError> {
///     let dd = Tensor4::from_matrix(&[
///         [  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0],
///         [ -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0],
///         [  2.0,  4.0,  6.0,  8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
///         [ 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
///         [  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
///         [  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
///         [ -2.0, -4.0, -6.0, -8.0,-10.0,-12.0,-14.0,-16.0,-18.0],
///         [  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
///         [  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
///     ], Mandel::General)?;
///
///     let a = Tensor2::from_matrix(&[
///         [1.0, 4.0, 6.0],
///         [7.0, 2.0, 5.0],
///         [9.0, 8.0, 3.0],
///     ], Mandel::General)?;
///
///     let mut b = Tensor2::new(Mandel::General);
///     t4_ddot_t2(&mut b, 1.0, &dd, &a)?;
///
///     assert_eq!(
///         format!("{:.1}", b.to_matrix()),
///         "┌                      ┐\n\
///          │  285.0 2850.0    0.0 │\n\
///          │ -570.0 -285.0    0.0 │\n\
///          │    0.0    0.0  570.0 │\n\
///          └                      ┘"
///     );
///     Ok(())
/// }
/// ```
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
///
/// # Example
///
/// ```
/// use russell_tensor::{t2_ddot_t4, Mandel, Tensor2, Tensor4, StrError};
///
/// fn main() -> Result<(), StrError> {
///     let a = Tensor2::from_matrix(&[
///         [1.0, 4.0, 6.0],
///         [7.0, 2.0, 5.0],
///         [9.0, 8.0, 3.0],
///     ], Mandel::General)?;
///
///     let dd = Tensor4::from_matrix(&[
///         [  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0],
///         [ -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0],
///         [  2.0,  4.0,  6.0,  8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
///         [ 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
///         [  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
///         [  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
///         [ -2.0, -4.0, -6.0, -8.0,-10.0,-12.0,-14.0,-16.0,-18.0],
///         [  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
///         [  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
///     ], Mandel::General)?;
///
///     let mut b = Tensor2::new(Mandel::General);
///     t2_ddot_t4(&mut b, 1.0, &a, &dd)?;
///
///     assert_eq!(
///         format!("{:.1}", b.to_matrix()),
///         "┌                   ┐\n\
///          │  31.0 124.0 186.0 │\n\
///          │ 217.0  62.0 155.0 │\n\
///          │ 279.0 248.0  93.0 │\n\
///          └                   ┘"
///     );
///     Ok(())
/// }
/// ```
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
///
/// # Example
///
/// ```
/// use russell_chk::approx_eq;
/// use russell_tensor::{Mandel, t4_ddot_t4, StrError, Tensor4};
///
/// fn main() -> Result<(), StrError> {
///     let cc = Tensor4::from_matrix(
///         &[
///             [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
///             [1.0, 1.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
///             [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
///             [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
///             [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
///             [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
///             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
///             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
///             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
///         ],
///         Mandel::General,
///     )?;
///
///     let dd = Tensor4::from_matrix(
///         &[
///             [-1.0, 1.0 / 3.0, 5.0 / 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
///             [1.0, -2.0 / 3.0, -1.0 / 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
///             [0.0, 1.0 / 3.0, -1.0 / 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
///             [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
///             [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
///             [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
///             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
///             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
///             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
///         ],
///         Mandel::General,
///     )?;
///
///     let mut ee = Tensor4::new(Mandel::General);
///     t4_ddot_t4(&mut ee, 1.0, &cc, &dd)?;
///
///     let out = ee.to_matrix();
///     for i in 0..9 {
///         for j in 0..9 {
///             if i == j {
///                 approx_eq(out.get(i, j), 1.0, 1e-15);
///             } else {
///                 approx_eq(out.get(i, j), 0.0, 1e-15);
///             }
///         }
///     }
///     Ok(())
/// }
/// ```
#[inline]
pub fn t4_ddot_t4(ee: &mut Tensor4, alpha: f64, cc: &Tensor4, dd: &Tensor4) -> Result<(), StrError> {
    mat_mat_mul(&mut ee.mat, alpha, &cc.mat, &dd.mat)
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{
        copy_tensor2, copy_tensor4, t2_ddot_t2, t2_ddot_t4, t2_dot_t2, t2_dot_vec, t2_dyad_t2, t2_odyad_t2, t4_ddot_t2,
        t4_ddot_t4, vec_dot_t2, vec_dyad_vec, Tensor2, Tensor4,
    };
    use crate::{Mandel, SamplesTensor4, SQRT_2};
    use russell_chk::{approx_eq, vec_approx_eq};
    use russell_lab::Vector;

    #[test]
    fn copy_tensor2_fails_on_wrong_input() {
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ], Mandel::General).unwrap();
        let mut b = Tensor2::new(Mandel::Symmetric2D);
        assert_eq!(
            copy_tensor2(&mut b, &a).err(),
            Some("second-order tensors are incompatible")
        );
    }

    #[test]
    fn copy_tensor2_works() {
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ], Mandel::General).unwrap();
        let mut b = Tensor2::new(Mandel::General);
        copy_tensor2(&mut b, &a).unwrap();
        assert_eq!(
            format!("{:.1}", b.to_matrix()),
            "┌             ┐\n\
             │ 1.0 2.0 3.0 │\n\
             │ 4.0 5.0 6.0 │\n\
             │ 7.0 8.0 9.0 │\n\
             └             ┘"
        );
    }

    #[test]
    fn copy_tensor4_fails_on_wrong_input() {
        let dd = Tensor4::new(Mandel::Symmetric);
        let mut ee = Tensor4::new(Mandel::General);
        assert_eq!(copy_tensor4(&mut ee, &dd).err(), Some("matrices are incompatible"));
    }

    #[test]
    fn copy_tensor4_works() {
        let dd = Tensor4::from_matrix(
            &[
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
                [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                [6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0],
                [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                [6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0],
                [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            ],
            Mandel::General,
        )
        .unwrap();
        let mut ee = Tensor4::new(Mandel::General);
        copy_tensor4(&mut ee, &dd).unwrap();
        assert_eq!(format!("{:.1}", ee.to_matrix()), format!("{:.1}", dd.to_matrix()));
    }

    #[test]
    fn t2_ddot_t2_works() {
        // general : general
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ], Mandel::General).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [9.0, 8.0, 7.0],
            [6.0, 5.0, 4.0],
            [3.0, 2.0, 1.0],
        ], Mandel::General).unwrap();
        let s = t2_ddot_t2(&a, &b);
        assert_eq!(s, 165.0);

        // sym-3D : sym-3D
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0, 5.0, 3.0],
        ], Mandel::Symmetric).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [3.0, 5.0, 6.0],
            [5.0, 2.0, 4.0],
            [6.0, 4.0, 1.0],
        ], Mandel::Symmetric).unwrap();
        let s = t2_ddot_t2(&a, &b);
        approx_eq(s, 162.0, 1e-13);

        // sym-3D : general
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0, 5.0, 3.0],
        ], Mandel::Symmetric).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [9.0, 8.0, 7.0],
            [6.0, 5.0, 4.0],
            [3.0, 2.0, 1.0],
        ], Mandel::General).unwrap();
        let s = t2_ddot_t2(&a, &b);
        approx_eq(s, 168.0, 1e-13);

        // sym-2D : sym-2D
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ], Mandel::Symmetric2D).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [3.0, 5.0, 0.0],
            [5.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ], Mandel::Symmetric2D).unwrap();
        let s = t2_ddot_t2(&a, &b);
        approx_eq(s, 50.0, 1e-13);

        // sym-2D : sym-3D
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ], Mandel::Symmetric2D).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [3.0, 5.0, 6.0],
            [5.0, 2.0, 4.0],
            [6.0, 4.0, 1.0],
        ], Mandel::Symmetric).unwrap();
        let s = t2_ddot_t2(&a, &b);
        approx_eq(s, 50.0, 1e-13);
    }

    #[test]
    fn t2_sdot_t2_works() {
        // general . general
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ], Mandel::General).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [9.0, 8.0, 7.0],
            [6.0, 5.0, 4.0],
            [3.0, 2.0, 1.0],
        ], Mandel::General).unwrap();
        let c = t2_dot_t2(&a, &b).unwrap();
        #[rustfmt::skip]
        let correct = Tensor2::from_matrix(&[
            [ 30.0,  24.0, 18.0],
            [ 84.0,  69.0, 54.0],
            [138.0, 114.0, 90.0],
        ], Mandel::General).unwrap();
        vec_approx_eq(c.vec.as_data(), correct.vec.as_data(), 1e-13);

        // sym-3D . sym-3D
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0, 5.0, 3.0],
        ], Mandel::Symmetric).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [3.0, 5.0, 6.0],
            [5.0, 2.0, 4.0],
            [6.0, 4.0, 1.0],
        ], Mandel::Symmetric).unwrap();
        let c = t2_dot_t2(&a, &b).unwrap();
        #[rustfmt::skip]
        let correct = Tensor2::from_matrix(&[
            [59.0, 37.0, 28.0],
            [52.0, 44.0, 37.0],
            [61.0, 52.0, 59.0],
        ], Mandel::General).unwrap();
        vec_approx_eq(c.vec.as_data(), correct.vec.as_data(), 1e-13);

        // sym-3D . general
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 2.0, 3.0],
            [2.0, 5.0, 6.0],
            [3.0, 6.0, 9.0],
        ], Mandel::Symmetric).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [9.0, 8.0, 7.0],
            [6.0, 5.0, 4.0],
            [3.0, 2.0, 1.0],
        ], Mandel::General).unwrap();
        let c = t2_dot_t2(&a, &b).unwrap();
        #[rustfmt::skip]
        let correct = Tensor2::from_matrix(&[
            [30.0, 24.0, 18.0],
            [66.0, 53.0, 40.0],
            [90.0, 72.0, 54.0],
        ], Mandel::General).unwrap();
        vec_approx_eq(c.vec.as_data(), correct.vec.as_data(), 1e-13);

        // sym-3D . sym-2D
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 2.0, 3.0],
            [2.0, 5.0, 6.0],
            [3.0, 6.0, 9.0],
        ], Mandel::Symmetric).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [9.0, 8.0, 0.0],
            [8.0, 5.0, 0.0],
            [0.0, 0.0, 1.0],
        ], Mandel::Symmetric2D).unwrap();
        let c = t2_dot_t2(&a, &b).unwrap();
        #[rustfmt::skip]
        let correct = Tensor2::from_matrix(&[
            [25.0, 18.0, 3.0],
            [58.0, 41.0, 6.0],
            [75.0, 54.0, 9.0],
        ], Mandel::General).unwrap();
        vec_approx_eq(c.vec.as_data(), correct.vec.as_data(), 1e-13);
    }

    #[test]
    fn t2_dot_vec_fails_on_wrong_input() {
        let mut v = Vector::new(3);
        let a = Tensor2::new(Mandel::General);
        let u = Vector::new(4);
        let res = t2_dot_vec(&mut v, 1.0, &a, &u);
        assert_eq!(res.err(), Some("vectors must have dim = 3"));

        let a = Tensor2::new(Mandel::Symmetric2D);
        let res = t2_dot_vec(&mut v, 1.0, &a, &u);
        assert_eq!(res.err(), Some("vectors must have dim = 2"));
    }

    #[test]
    fn t2_dot_vec_works() {
        // general . vec
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ], Mandel::General).unwrap();
        let u = Vector::from(&[-2.0, -3.0, -4.0]);
        let mut v = Vector::new(3);
        t2_dot_vec(&mut v, 2.0, &a, &u).unwrap();
        vec_approx_eq(v.as_data(), &[-40.0, -94.0, -148.0], 1e-13);

        // sym-3D . vec
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 2.0, 3.0],
            [2.0, 5.0, 6.0],
            [3.0, 6.0, 9.0],
        ], Mandel::Symmetric).unwrap();
        let u = Vector::from(&[-2.0, -3.0, -4.0]);
        let mut v = Vector::new(3);
        t2_dot_vec(&mut v, 2.0, &a, &u).unwrap();
        vec_approx_eq(v.as_data(), &[-40.0, -86.0, -120.0], 1e-13);

        // sym-2D . vec
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 2.0, 0.0],
            [2.0, 5.0, 0.0],
            [0.0, 0.0, 9.0],
        ], Mandel::Symmetric2D).unwrap();
        let u = Vector::from(&[-2.0, -3.0]);
        let mut v = Vector::new(2);
        t2_dot_vec(&mut v, 2.0, &a, &u).unwrap();
        vec_approx_eq(v.as_data(), &[-16.0, -38.0], 1e-13);
    }

    #[test]
    fn vec_dot_t2_fails_on_wrong_input() {
        let mut v = Vector::new(3);
        let a = Tensor2::new(Mandel::General);
        let u = Vector::new(4);
        let res = vec_dot_t2(&mut v, 1.0, &u, &a);
        assert_eq!(res.err(), Some("vectors must have dim = 3"));

        let a = Tensor2::new(Mandel::Symmetric2D);
        let res = vec_dot_t2(&mut v, 1.0, &u, &a);
        assert_eq!(res.err(), Some("vectors must have dim = 2"));
    }

    #[test]
    fn vec_dot_t2_works() {
        // general . vec
        let u = Vector::from(&[-2.0, -3.0, -4.0]);
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ], Mandel::General).unwrap();
        let mut v = Vector::new(3);
        vec_dot_t2(&mut v, 2.0, &u, &a).unwrap();
        vec_approx_eq(v.as_data(), &[-84.0, -102.0, -120.0], 1e-13);

        // sym-3D . vec
        let u = Vector::from(&[-2.0, -3.0, -4.0]);
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 2.0, 3.0],
            [2.0, 5.0, 6.0],
            [3.0, 6.0, 9.0],
        ], Mandel::Symmetric).unwrap();
        let mut v = Vector::new(3);
        vec_dot_t2(&mut v, 2.0, &u, &a).unwrap();
        vec_approx_eq(v.as_data(), &[-40.0, -86.0, -120.0], 1e-13);

        // sym-2D . vec
        let u = Vector::from(&[-2.0, -3.0]);
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 2.0, 0.0],
            [2.0, 5.0, 0.0],
            [0.0, 0.0, 9.0],
        ], Mandel::Symmetric2D).unwrap();
        let mut v = Vector::new(2);
        vec_dot_t2(&mut v, 2.0, &u, &a).unwrap();
        vec_approx_eq(v.as_data(), &[-16.0, -38.0], 1e-13);
    }

    #[test]
    fn vec_dyad_vec_captures_errors() {
        // general
        const WRONG: f64 = 123.0;
        let u = Vector::from(&[-2.0, -3.0, -4.0, WRONG]);
        let v = Vector::from(&[4.0, 3.0, 2.0]);
        let mut tt = Tensor2::new(Mandel::General);
        assert_eq!(
            vec_dyad_vec(&mut tt, 1.0, &u, &v).err(),
            Some("vectors must have dim = 3")
        );
        let u = Vector::from(&[-2.0, -3.0, -4.0]);
        let v = Vector::from(&[4.0, 3.0, 2.0, WRONG]);
        assert_eq!(
            vec_dyad_vec(&mut tt, 1.0, &u, &v).err(),
            Some("vectors must have dim = 3")
        );

        // symmetric 3D
        let u = Vector::from(&[-2.0, -3.0, -4.0]);
        let v = Vector::from(&[4.0, 3.0, 2.0]);
        let mut tt = Tensor2::new(Mandel::Symmetric);
        assert_eq!(
            vec_dyad_vec(&mut tt, 1.0, &u, &v).err(),
            Some("dyadic product between u and v does not generate a symmetric tensor")
        );

        // symmetric 2D
        let u = Vector::from(&[-2.0, -3.0, WRONG]);
        let v = Vector::from(&[4.0, 3.0]);
        let mut tt = Tensor2::new(Mandel::Symmetric2D);
        assert_eq!(
            vec_dyad_vec(&mut tt, 1.0, &u, &v).err(),
            Some("vectors must have dim = 2")
        );
        let u = Vector::from(&[-2.0, -3.0]);
        let v = Vector::from(&[4.0, 3.0, WRONG]);
        assert_eq!(
            vec_dyad_vec(&mut tt, 1.0, &u, &v).err(),
            Some("vectors must have dim = 2")
        );
        let u = Vector::from(&[-2.0, -3.0]);
        let v = Vector::from(&[4.0, 3.0]);
        assert_eq!(
            vec_dyad_vec(&mut tt, 1.0, &u, &v).err(),
            Some("dyadic product between u and v does not generate a symmetric tensor")
        );
    }

    #[test]
    fn vec_dyad_vec_works() {
        // general
        let u = Vector::from(&[-2.0, -3.0, -4.0]);
        let v = Vector::from(&[4.0, 3.0, 2.0]);
        let mut tt = Tensor2::new(Mandel::General);
        vec_dyad_vec(&mut tt, 2.0, &u, &v).unwrap();
        let correct = &[
            -16.0,
            -18.0,
            -16.0,
            -18.0 * SQRT_2,
            -18.0 * SQRT_2,
            -20.0 * SQRT_2,
            6.0 * SQRT_2,
            6.0 * SQRT_2,
            12.0 * SQRT_2,
        ];
        vec_approx_eq(tt.vec.as_data(), correct, 1e-14);

        // symmetric 3D
        let u = Vector::from(&[-2.0, -3.0, -4.0]);
        let v = Vector::from(&[2.0, 3.0, 4.0]);
        let mut tt = Tensor2::new(Mandel::Symmetric);
        vec_dyad_vec(&mut tt, 2.0, &u, &v).unwrap();
        let correct = &[-8.0, -18.0, -32.0, -12.0 * SQRT_2, -24.0 * SQRT_2, -16.0 * SQRT_2];
        vec_approx_eq(tt.vec.as_data(), correct, 1e-14);

        // symmetric 2D
        let u = Vector::from(&[-2.0, -3.0]);
        let v = Vector::from(&[2.0, 3.0]);
        let mut tt = Tensor2::new(Mandel::Symmetric2D);
        vec_dyad_vec(&mut tt, 2.0, &u, &v).unwrap();
        let correct = &[-8.0, -18.0, 0.0, -12.0 * SQRT_2];
        vec_approx_eq(tt.vec.as_data(), correct, 1e-14);
    }

    #[test]
    fn t2_dyad_t2_works() {
        // general dyad general
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ], Mandel::General).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
        ], Mandel::General).unwrap();
        let mut dd = Tensor4::new(Mandel::General);
        t2_dyad_t2(&mut dd, 2.0, &a, &b).unwrap();
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

        // sym-3D dyad general
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 2.0, 3.0],
            [2.0, 5.0, 6.0],
            [3.0, 6.0, 9.0],
        ], Mandel::Symmetric).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
        ], Mandel::Symmetric).unwrap();
        let mut dd = Tensor4::new(Mandel::Symmetric);
        t2_dyad_t2(&mut dd, 2.0, &a, &b).unwrap();
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
        ], Mandel::Symmetric2D).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.0, 0.5],
        ], Mandel::Symmetric2D).unwrap();
        let mut dd = Tensor4::new(Mandel::Symmetric2D);
        t2_dyad_t2(&mut dd, 2.0, &a, &b).unwrap();
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
    }

    #[test]
    fn t2_odyad_t2_works() {
        // general odyad general
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ], Mandel::General).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [9.0, 8.0, 7.0],
            [6.0, 5.0, 4.0],
            [3.0, 2.0, 1.0],
        ], Mandel::General).unwrap();
        let mut dd = Tensor4::new(Mandel::General);
        t2_odyad_t2(&mut dd, 2.0, &a, &b).unwrap();
        let mat = dd.to_matrix();
        println!("{:.1}", mat);
        assert_eq!(
            format!("{:.1}", mat),
            "┌                                              ┐\n\
             │  9.0 16.0 21.0  8.0 14.0  7.0 18.0 24.0 27.0 │\n\
             │ 24.0 25.0 24.0 20.0 20.0 16.0 30.0 30.0 36.0 │\n\
             │ 21.0 16.0  9.0 14.0  8.0  7.0 24.0 18.0 27.0 │\n\
             │  6.0 10.0 12.0  5.0  8.0  4.0 12.0 15.0 18.0 │\n\
             │ 12.0 10.0  6.0  8.0  5.0  4.0 15.0 12.0 18.0 │\n\
             │  3.0  4.0  3.0  2.0  2.0  1.0  6.0  6.0  9.0 │\n\
             │ 36.0 40.0 42.0 32.0 35.0 28.0 45.0 48.0 54.0 │\n\
             │ 42.0 40.0 36.0 35.0 32.0 28.0 48.0 45.0 54.0 │\n\
             │ 63.0 64.0 63.0 56.0 56.0 49.0 72.0 72.0 81.0 │\n\
             └                                              ┘"
        );
    }

    #[test]
    fn t4_ddot_t2_works() {
        let dd = Tensor4::from_matrix(&SamplesTensor4::SYM_2D_SAMPLE1_STD_MATRIX, Mandel::Symmetric2D).unwrap();
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [-1.0, -2.0,  0.0],
            [-2.0,  2.0,  0.0],
            [ 0.0,  0.0, -3.0]], Mandel::Symmetric2D).unwrap();
        let mut b = Tensor2::new(Mandel::Symmetric2D);
        t4_ddot_t2(&mut b, 1.0, &dd, &a).unwrap();
        let out = b.to_matrix();
        assert_eq!(
            format!("{:.1}", out),
            "┌                      ┐\n\
             │  -46.0 -154.0    0.0 │\n\
             │ -154.0  -64.0    0.0 │\n\
             │    0.0    0.0  -82.0 │\n\
             └                      ┘"
        );
    }

    #[test]
    fn t2_ddot_t4_works() {
        let dd = Tensor4::from_matrix(&SamplesTensor4::SYM_2D_SAMPLE1_STD_MATRIX, Mandel::Symmetric2D).unwrap();
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [-1.0, -2.0,  0.0],
            [-2.0,  2.0,  0.0],
            [ 0.0,  0.0, -3.0]], Mandel::Symmetric2D).unwrap();
        let mut b = Tensor2::new(Mandel::Symmetric2D);
        t2_ddot_t4(&mut b, 1.0, &a, &dd).unwrap();
        let out = b.to_matrix();
        assert_eq!(
            format!("{:.1}", out),
            "┌                      ┐\n\
             │  -90.0 -144.0    0.0 │\n\
             │ -144.0  -96.0    0.0 │\n\
             │    0.0    0.0 -102.0 │\n\
             └                      ┘"
        );
    }

    #[test]
    fn t4_ddot_t4_works() {
        let cc = Tensor4::from_matrix(&SamplesTensor4::SYM_2D_SAMPLE1_STD_MATRIX, Mandel::Symmetric2D).unwrap();
        let mut ee = Tensor4::new(Mandel::Symmetric2D);
        t4_ddot_t4(&mut ee, 1.0, &cc, &cc).unwrap();
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
    }
}
