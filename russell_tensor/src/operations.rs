use super::{Tensor2, Tensor4};
use crate::{Mandel, SQRT_2};
use russell_lab::{mat_mat_mul, mat_vec_mul, mat_vec_mul_update, vec_inner, vec_mat_mul, vec_outer, vec_outer_update};
use russell_lab::{StrError, Vector};

/// Performs the double-dot (ddot) operation between two Tensor2 (inner product)
///
/// Computes:
///
/// ```text
/// s = a : b
/// ```
///
/// With orthonormal Cartesian components:
///
/// ```text
/// s = Σ Σ aᵢⱼ bᵢⱼ
///     i j
/// ```
///
/// Or, in Mandel basis:
///
/// ```text
/// s = Σ aₘ bₘ
///     m
/// ```
///
/// # Input
///
/// * `a` -- first tensor; with the same [Mandel] as `b`
/// * `b` -- second tensor; with the same [Mandel] as `a`
///
/// # Output
///
/// Returns the scalar result of `a : b`.
///
/// # Panics
///
/// A panic will occur `a` and `b` have different [Mandel]
///
/// # Examples
///
/// ```
/// use russell_lab::approx_eq;
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
///     let res = t2_ddot_t2(&a.as_general(), &b);
///
///     approx_eq(res, 8.0, 1e-15);
///     Ok(())
/// }
/// ```
pub fn t2_ddot_t2(a: &Tensor2, b: &Tensor2) -> f64 {
    assert_eq!(a.mandel, b.mandel);
    vec_inner(&a.vec, &b.vec)
}

/// Performs the single dot operation between two Tensor2 (matrix multiplication)
///
/// Computes:
///
/// ```text
/// c = a · b
/// ```
///
/// With orthonormal Cartesian components:
/// 
/// ```text
/// cᵢⱼ = Σ aᵢₖ bₖⱼ
///       k
/// ```
///
/// **Important:** Even if `a` and `b` are symmetric, the result `c`
/// may not be symmetric. Therefore, `c` must be a General tensor.
/// 
/// # Output
/// 
/// * `c` -- the resulting tensor; it must be [Mandel::General]
///
/// # Input
///
/// * `a` -- first tensor; with the same [Mandel] as `b`
/// * `b` -- second tensor; with the same [Mandel] as `a`
///
/// # Panics
///
/// 1. A panic will occur if `c` is not [Mandel::General]
/// 2. A panic will occur the `a` and `b` have different [Mandel]
///
/// # Examples
///
/// ```
/// use russell_tensor::{t2_dot_t2, Mandel, Tensor2, StrError};
///
/// fn main() -> Result<(), StrError> {
///     let a = Tensor2::from_matrix(&[
///         [1.0,  1.0, 0.0],
///         [1.0, -1.0, 0.0],
///         [0.0,  0.0, 1.0],
///     ], Mandel::General)?;
///
///     let b = Tensor2::from_matrix(&[
///         [1.0,  2.0, 0.0],
///         [3.0, -1.0, 5.0],
///         [0.0,  4.0, 1.0],
///     ], Mandel::General)?;
///
///     let mut c = Tensor2::new(Mandel::General);
///     t2_dot_t2(&mut c, &a, &b);
///     assert_eq!(
///         format!("{:.1}", c.as_matrix()),
///         "┌                ┐\n\
///          │  4.0  1.0  5.0 │\n\
///          │ -2.0  3.0 -5.0 │\n\
///          │  0.0  4.0  1.0 │\n\
///          └                ┘"
///     );
///     Ok(())
/// }
/// ```
#[rustfmt::skip]
pub fn t2_dot_t2(c: &mut Tensor2, a: &Tensor2, b: &Tensor2) {
    assert_eq!(c.mandel, Mandel::General);
    assert_eq!(b.mandel, a.mandel);
    let dim = a.vec.dim();
    let a = &a.vec;
    let b = &b.vec;
    let c = &mut c.vec;
    let tsq2 = 2.0 * SQRT_2;
    if dim == 4 {
        c[0] = a[0] * b[0] + (a[3] * b[3]) / 2.0;
        c[1] = a[1] * b[1] + (a[3] * b[3]) / 2.0;
        c[2] = a[2] * b[2];
        c[3] = (a[3] * (b[0] + b[1]) + (a[0] + a[1]) * b[3]) / 2.0;
        c[4] = 0.0;
        c[5] = 0.0;
        c[6] = (a[3] * (-b[0] + b[1]) + (a[0] - a[1]) * b[3]) / 2.0;
        c[7] = 0.0;
        c[8] = 0.0;
    } else if dim == 6 {
        c[0] = (2.0 * a[0] * b[0] + a[3] * b[3] + a[5] * b[5]) / 2.0;
        c[1] = (2.0 * a[1] * b[1] + a[3] * b[3] + a[4] * b[4]) / 2.0;
        c[2] = (2.0 * a[2] * b[2] + a[4] * b[4] + a[5] * b[5]) / 2.0;
        c[3] = (SQRT_2 * a[3] * (b[0] + b[1]) + SQRT_2 * a[0] * b[3] + SQRT_2 * a[1] * b[3] + a[5] * b[4] + a[4] * b[5]) / tsq2;
        c[4] = (SQRT_2 * a[4] * (b[1] + b[2]) + a[5] * b[3] + SQRT_2 * a[1] * b[4] + SQRT_2 * a[2] * b[4] + a[3] * b[5]) / tsq2;
        c[5] = (SQRT_2 * a[5] * (b[0] + b[2]) + a[4] * b[3] + a[3] * b[4] + SQRT_2 * a[0] * b[5] + SQRT_2 * a[2] * b[5]) / tsq2;
        c[6] = (SQRT_2 * a[3] * (-b[0] + b[1]) + SQRT_2 * a[0] * b[3] - SQRT_2 * a[1] * b[3] + a[5] * b[4] - a[4] * b[5]) / tsq2;
        c[7] = (SQRT_2 * a[4] * (-b[1] + b[2]) - a[5] * b[3] + SQRT_2 * a[1] * b[4] - SQRT_2 * a[2] * b[4] + a[3] * b[5]) / tsq2;
        c[8] = (SQRT_2 * a[5] * (-b[0] + b[2]) - a[4] * b[3] + a[3] * b[4] + SQRT_2 * a[0] * b[5] - SQRT_2 * a[2] * b[5]) / tsq2;
    } else {
        c[0] = (2.0 * a[0] * b[0] + (a[3] + a[6]) * (b[3] - b[6]) + (a[5] + a[8]) * (b[5] - b[8])) / 2.0;
        c[1] = (2.0 * a[1] * b[1] + (a[3] - a[6]) * (b[3] + b[6]) + (a[4] + a[7]) * (b[4] - b[7])) / 2.0;
        c[2] = (2.0 * a[2] * b[2] + (a[4] - a[7]) * (b[4] + b[7]) + (a[5] - a[8]) * (b[5] + b[8])) / 2.0;
        c[3] = (2.0 * (a[3] - a[6]) * b[0] + 2.0 * (a[3] + a[6]) * b[1] + 2.0 * a[1] * (b[3] - b[6]) + 2.0 * a[0] * (b[3] + b[6]) + SQRT_2 * (a[5] + a[8]) * (b[4] - b[7]) + SQRT_2 * (a[4] + a[7]) * (b[5] - b[8])) / 4.0;
        c[4] = (2.0 * (a[4] - a[7]) * b[1] + 2.0 * (a[4] + a[7]) * b[2] + SQRT_2 * (a[5] - a[8]) * (b[3] + b[6]) + 2.0 * a[2] * (b[4] - b[7]) + 2.0 * a[1] * (b[4] + b[7]) + SQRT_2 * (a[3] - a[6]) * (b[5] + b[8])) / 4.0;
        c[5] = (2.0 * (a[5] - a[8]) * b[0] + 2.0 * (a[5] + a[8]) * b[2] + SQRT_2 * (a[4] - a[7]) * (b[3] - b[6]) + SQRT_2 * (a[3] + a[6]) * (b[4] + b[7]) + 2.0 * a[2] * (b[5] - b[8]) + 2.0 * a[0] * (b[5] + b[8])) / 4.0;
        c[6] = (-2.0 * (a[3] - a[6]) * b[0] + 2.0 * (a[3] + a[6]) * b[1] - 2.0 * a[1] * (b[3] - b[6]) + 2.0 * a[0] * (b[3] + b[6]) + SQRT_2 * (a[5] + a[8]) * (b[4] - b[7]) - SQRT_2 * (a[4] + a[7]) * (b[5] - b[8])) / 4.0;
        c[7] = (-2.0 * (a[4] - a[7]) * b[1] + 2.0 * (a[4] + a[7]) * b[2] - SQRT_2 * (a[5] - a[8]) * (b[3] + b[6]) - 2.0 * a[2] * (b[4] - b[7]) + 2.0 * a[1] * (b[4] + b[7]) + SQRT_2 * (a[3] - a[6]) * (b[5] + b[8])) / 4.0;
        c[8] = (-2.0 * (a[5] - a[8]) * b[0] + 2.0 * (a[5] + a[8]) * b[2] - SQRT_2 * (a[4] - a[7]) * (b[3] - b[6]) + SQRT_2 * (a[3] + a[6]) * (b[4] + b[7]) - 2.0 * a[2] * (b[5] - b[8]) + 2.0 * a[0] * (b[5] + b[8])) / 4.0;
    }
}

/// Performs the single dot operation between a Tensor2 and a vector
///
/// Computes:
///
/// ```text
/// v = α a · u
/// ```
///
/// With orthonormal Cartesian components:
///
/// ```text
/// vᵢ = α Σ aᵢⱼ uⱼ
///        j
/// ```
///
/// # Output
///
/// * `v` -- the resulting vector; with `dim` compatible with the dimension of `a` (2D or 3D)
///
/// # Input
///
/// * `alpha` -- the `α` multiplier
/// * `a` -- the second-order tensor
/// * `u` -- the 2D or 3D vector; with `dim` compatible with the dimension of `a` (2D or 3D)
///
/// # Panics
///
/// 1. If `a` is 2D, a panic will occur if `u` or `v` are not `2D`
/// 2. If `a` is 3D, a panic will occur if `u` or `v` are not `3D`
///
/// # Examples
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
///     t2_dot_vec(&mut v, 2.0, &a, &u);
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
pub fn t2_dot_vec(v: &mut Vector, alpha: f64, a: &Tensor2, u: &Vector) {
    if a.vec.dim() == 4 {
        assert_eq!(v.dim(), 2);
        assert_eq!(u.dim(), 2);
        v[0] = alpha * (a.get(0, 0) * u[0] + a.get(0, 1) * u[1]);
        v[1] = alpha * (a.get(1, 0) * u[0] + a.get(1, 1) * u[1]);
    } else {
        assert_eq!(v.dim(), 3);
        assert_eq!(u.dim(), 3);
        v[0] = alpha * (a.get(0, 0) * u[0] + a.get(0, 1) * u[1] + a.get(0, 2) * u[2]);
        v[1] = alpha * (a.get(1, 0) * u[0] + a.get(1, 1) * u[1] + a.get(1, 2) * u[2]);
        v[2] = alpha * (a.get(2, 0) * u[0] + a.get(2, 1) * u[1] + a.get(2, 2) * u[2]);
    }
}

/// Performs the single dot operation between a vector and a Tensor2
///
/// Computes:
///
/// ```text
/// v = α u · a
/// ```
///
/// With orthonormal Cartesian components:
///
/// ```text
/// vⱼ = α Σ uᵢ aᵢⱼ
///        i
/// ```
///
/// # Output
///
/// * `v` -- the resulting vector; with `dim` compatible with the dimension of `a` (2D or 3D)
///
/// # Input
///
/// * `alpha` -- the `α` multiplier
/// * `u` -- the 2D or 3D vector; with `dim` compatible with the dimension of `a` (2D or 3D)
/// * `a` -- the second-order tensor
///
/// # Panics
///
/// 1. If `a` is 2D, a panic will occur if `u` or `v` are not `2D`
/// 2. If `a` is 3D, a panic will occur if `u` or `v` are not `3D`
///
/// # Examples
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
///     vec_dot_t2(&mut v, 2.0, &u, &a);
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
pub fn vec_dot_t2(v: &mut Vector, alpha: f64, u: &Vector, a: &Tensor2) {
    if a.vec.dim() == 4 {
        assert_eq!(v.dim(), 2);
        assert_eq!(u.dim(), 2);
        v[0] = alpha * (u[0] * a.get(0, 0) + u[1] * a.get(1, 0));
        v[1] = alpha * (u[0] * a.get(0, 1) + u[1] * a.get(1, 1));
    } else {
        assert_eq!(v.dim(), 3);
        assert_eq!(u.dim(), 3);
        v[0] = alpha * (u[0] * a.get(0, 0) + u[1] * a.get(1, 0) + u[2] * a.get(2, 0));
        v[1] = alpha * (u[0] * a.get(0, 1) + u[1] * a.get(1, 1) + u[2] * a.get(2, 1));
        v[2] = alpha * (u[0] * a.get(0, 2) + u[1] * a.get(1, 2) + u[2] * a.get(2, 2));
    }
}

/// Performs the dyadic product between two vectors resulting in a second-order tensor
///
/// Computes:
///
/// ```text
/// A = α u ⊗ v
/// ```
///
/// With orthonormal Cartesian components:
///
/// ```text
/// Aᵢⱼ = α uᵢ vⱼ
/// ```
///
/// **Important:** The dyadic product between two vectors may result in a **non-symmetric**
/// second-order tensor. Therefore, if the input tensor `A` is symmetric, an error may occur.
/// Thus, make sure that the you expect `u ⊗ v` to be symmetric when passing a symmetric tensor `A`.
///
/// # Output
///
/// * `A` -- the resulting second-order tensor
///
/// # Input
///
/// * `alpha` -- the `α` multiplier
/// * `u` -- the 2D or 3D vector; with `dim` compatible with the dimension of `T` (2D or 3D)
/// * `v` -- the 2D or 3D vector; with `dim` compatible with the dimension of `T` (2D or 3D)
///
/// # Panics
///
/// 1. If `A` is 2D, a panic will occur if `u` or `v` are not `2D`
/// 2. If `A` is 3D, a panic will occur if `u` or `v` are not `3D`
///
/// # Examples
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
///         format!("{:.1}", tt.as_matrix()),
///         "┌             ┐\n\
///          │ 2.0 2.0 2.0 │\n\
///          │ 2.0 2.0 2.0 │\n\
///          │ 2.0 2.0 2.0 │\n\
///          └             ┘"
///     );
///     Ok(())
/// }
/// ```
pub fn vec_dyad_vec(a: &mut Tensor2, alpha: f64, u: &Vector, v: &Vector) -> Result<(), StrError> {
    let dim = a.vec.dim();
    if dim == 4 {
        assert_eq!(v.dim(), 2);
        assert_eq!(u.dim(), 2);
        if (u[0] * v[1]) != (u[1] * v[0]) {
            return Err("dyadic product between u and v does not generate a symmetric tensor");
        }
        a.vec[0] = alpha * u[0] * v[0];
        a.vec[1] = alpha * u[1] * v[1];
        a.vec[2] = 0.0;
        a.vec[3] = alpha * (u[0] * v[1] + u[1] * v[0]) / SQRT_2;
    } else {
        assert_eq!(v.dim(), 3);
        assert_eq!(u.dim(), 3);
        a.vec[0] = alpha * u[0] * v[0];
        a.vec[1] = alpha * u[1] * v[1];
        a.vec[2] = alpha * u[2] * v[2];
        a.vec[3] = alpha * (u[0] * v[1] + u[1] * v[0]) / SQRT_2;
        a.vec[4] = alpha * (u[1] * v[2] + u[2] * v[1]) / SQRT_2;
        a.vec[5] = alpha * (u[0] * v[2] + u[2] * v[0]) / SQRT_2;
        if dim == 6 {
            if (u[0] * v[1]) != (u[1] * v[0]) || (u[1] * v[2]) != (u[2] * v[1]) || (u[0] * v[2]) != (u[2] * v[0]) {
                return Err("dyadic product between u and v does not generate a symmetric tensor");
            }
        } else {
            a.vec[6] = alpha * (u[0] * v[1] - u[1] * v[0]) / SQRT_2;
            a.vec[7] = alpha * (u[1] * v[2] - u[2] * v[1]) / SQRT_2;
            a.vec[8] = alpha * (u[0] * v[2] - u[2] * v[0]) / SQRT_2;
        }
    }
    Ok(())
}

/// Performs the dyadic product between two Tensor2 resulting a Tensor4
///
/// ```text
/// D = α a ⊗ b
/// ```
///
/// With orthonormal Cartesian components:
///
/// ```text
/// Dᵢⱼₖₗ = α aᵢⱼ bₖₗ
/// ```
///
/// Or, in Mandel basis:
///
/// ```text
/// Dₘₙ = α aₘ bₙ
/// ```
///
/// # Output
///
/// * `dd` -- the tensor `D`; with the same [Mandel] as `a` and `b`
///
/// # Input
///
/// * `a` -- first tensor; with the same [Mandel] as `b` and `dd`
/// * `b` -- second tensor; with the same [Mandel] as `a` and `dd`
///
/// # Panics
///
/// A panic will occur the tensors have different [Mandel]
///
/// # Examples
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
///     t2_dyad_t2(&mut dd, 1.0, &a, &b);
///
///     assert_eq!(
///         format!("{:.1}", dd.as_matrix()),
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
pub fn t2_dyad_t2(dd: &mut Tensor4, alpha: f64, a: &Tensor2, b: &Tensor2) {
    assert_eq!(a.mandel, dd.mandel);
    assert_eq!(b.mandel, dd.mandel);
    vec_outer(&mut dd.mat, alpha, &a.vec, &b.vec).unwrap();
}

/// Performs the dyadic product between two Tensor2 resulting in a Tensor4 (with update)
///
/// Computes:
///
/// ```text
/// D += α a ⊗ b
/// ```
///
/// With orthonormal Cartesian components:
///
/// ```text
/// Dᵢⱼₖₗ += α aᵢⱼ bₖₗ
/// ```
///
/// Or, in Mandel basis:
///
/// ```text
/// Dₘₙ += α aₘ bₙ
/// ```
///
/// # Output
///
/// * `dd` -- the tensor `D`; with the same [Mandel] as `a` and `b`
///
/// # Input
///
/// * `a` -- first tensor; with the same [Mandel] as `b` and `dd`
/// * `b` -- second tensor; with the same [Mandel] as `a` and `dd`
///
/// # Panics
///
/// A panic will occur the tensors have different [Mandel]
///
/// # Examples
///
/// ```
/// use russell_lab::Matrix;
/// use russell_tensor::{t2_dyad_t2_update, Mandel, StrError, Tensor2, Tensor4};
///
/// fn main() -> Result<(), StrError> {
///     #[rustfmt::skip]
///     let a = Tensor2::from_matrix(&[
///         [ 1.0, 10.0, 0.0],
///         [ 2.0,  1.0, 0.0],
///         [ 0.0,  0.0, 2.0],
///     ], Mandel::General)?;
///
///     #[rustfmt::skip]
///     let b = Tensor2::from_matrix(&[
///         [1.0, 4.0, 6.0],
///         [7.0, 2.0, 5.0],
///         [9.0, 8.0, 3.0],
///     ], Mandel::General)?;
///
///     let mat = Matrix::filled(9, 9, 0.5);
///     let mut dd = Tensor4::from_matrix(&mat, Mandel::General)?;
///     t2_dyad_t2_update(&mut dd, 1.0, &a, &b);
///
///     assert_eq!(
///         format!("{:.1}", dd.as_matrix()),
///         "┌                                              ┐\n\
///          │  1.5  2.5  3.5  4.5  5.5  6.5  7.5  8.5  9.5 │\n\
///          │  1.5  2.5  3.5  4.5  5.5  6.5  7.5  8.5  9.5 │\n\
///          │  2.5  4.5  6.5  8.5 10.5 12.5 14.5 16.5 18.5 │\n\
///          │ 10.5 20.5 30.5 40.5 50.5 60.5 70.5 80.5 90.5 │\n\
///          │  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5 │\n\
///          │  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5 │\n\
///          │  2.5  4.5  6.5  8.5 10.5 12.5 14.5 16.5 18.5 │\n\
///          │  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5 │\n\
///          │  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5 │\n\
///          └                                              ┘"
///     );
///     Ok(())
/// }
/// ```
pub fn t2_dyad_t2_update(dd: &mut Tensor4, alpha: f64, a: &Tensor2, b: &Tensor2) {
    assert_eq!(a.mandel, dd.mandel);
    assert_eq!(b.mandel, dd.mandel);
    vec_outer_update(&mut dd.mat, alpha, &a.vec, &b.vec).unwrap();
}

/// Performs the overbar dyadic product between two Tensor2 resulting in a (general) Tensor4
///
/// Computes:
///
/// ```text
///         _
/// D = s A ⊗ B
/// ```
/// 
/// With orthonormal Cartesian components:
///
/// ```text
/// Dᵢⱼₖₗ = s Aᵢₖ Bⱼₗ
/// ```
///
/// **Important:** The result is **not** necessarily minor-symmetric; therefore `D` must be General.
///
/// # Output
///
/// * `dd` -- the tensor `D`; it must be [Mandel::General]
///
/// # Input
///
/// * `a` -- first tensor; with the same [Mandel] as `b`
/// * `b` -- second tensor; with the same [Mandel] as `a`
///
/// # Panics
///
/// 1. A panic will occur if `dd` is not [Mandel::General]
/// 2. A panic will occur the `a` and `b` have different [Mandel]
#[rustfmt::skip]
pub fn t2_odyad_t2(dd: &mut Tensor4, s: f64, aa: &Tensor2, bb: &Tensor2) {
    assert_eq!(dd.mandel, Mandel::General);
    assert_eq!(bb.mandel, aa.mandel);
    let dim = aa.vec.dim();
    let a = &aa.vec;
    let b = &bb.vec;
    let tsq2 = 2.0 * SQRT_2;
    if dim == 4 {
        dd.mat.set(0,0, s*a[0]*b[0]);
        dd.mat.set(0,1, s*(a[3]*b[3])/2.0);
        dd.mat.set(0,2, 0.0);
        dd.mat.set(0,3, s*(a[3]*b[0] + a[0]*b[3])/2.0);
        dd.mat.set(0,4, 0.0);
        dd.mat.set(0,5, 0.0);
        dd.mat.set(0,6, s*(-(a[3]*b[0]) + a[0]*b[3])/2.0);
        dd.mat.set(0,7, 0.0);
        dd.mat.set(0,8, 0.0);

        dd.mat.set(1,0, s*(a[3]*b[3])/2.0);
        dd.mat.set(1,1, s*a[1]*b[1]);
        dd.mat.set(1,2, 0.0);
        dd.mat.set(1,3, s*(a[3]*b[1] + a[1]*b[3])/2.0);
        dd.mat.set(1,4, 0.0);
        dd.mat.set(1,5, 0.0);
        dd.mat.set(1,6, s*(a[3]*b[1] - a[1]*b[3])/2.0);
        dd.mat.set(1,7, 0.0);
        dd.mat.set(1,8, 0.0);

        dd.mat.set(2,0, 0.0);
        dd.mat.set(2,1, 0.0);
        dd.mat.set(2,2, s*a[2]*b[2]);
        dd.mat.set(2,3, 0.0);
        dd.mat.set(2,4, 0.0);
        dd.mat.set(2,5, 0.0);
        dd.mat.set(2,6, 0.0);
        dd.mat.set(2,7, 0.0);
        dd.mat.set(2,8, 0.0);

        dd.mat.set(3,0, s*(a[3]*b[0] + a[0]*b[3])/2.0);
        dd.mat.set(3,1, s*(a[3]*b[1] + a[1]*b[3])/2.0);
        dd.mat.set(3,2, 0.0);
        dd.mat.set(3,3, s*(a[1]*b[0] + a[0]*b[1] + a[3]*b[3])/2.0);
        dd.mat.set(3,4, 0.0);
        dd.mat.set(3,5, 0.0);
        dd.mat.set(3,6, s*(-(a[1]*b[0]) + a[0]*b[1])/2.0);
        dd.mat.set(3,7, 0.0);
        dd.mat.set(3,8, 0.0);

        dd.mat.set(4,0, 0.0);
        dd.mat.set(4,1, 0.0);
        dd.mat.set(4,2, 0.0);
        dd.mat.set(4,3, 0.0);
        dd.mat.set(4,4, s*(a[2]*b[1] + a[1]*b[2])/2.0);
        dd.mat.set(4,5, s*(a[3]*b[2] + a[2]*b[3])/tsq2);
        dd.mat.set(4,6, 0.0);
        dd.mat.set(4,7, s*(-(a[2]*b[1]) + a[1]*b[2])/2.0);
        dd.mat.set(4,8, s*(a[3]*b[2] - a[2]*b[3])/tsq2);

        dd.mat.set(5,0, 0.0);
        dd.mat.set(5,1, 0.0);
        dd.mat.set(5,2, 0.0);
        dd.mat.set(5,3, 0.0);
        dd.mat.set(5,4, s*(a[3]*b[2] + a[2]*b[3])/tsq2);
        dd.mat.set(5,5, s*(a[2]*b[0] + a[0]*b[2])/2.0);
        dd.mat.set(5,6, 0.0);
        dd.mat.set(5,7, s*(a[3]*b[2] - a[2]*b[3])/tsq2);
        dd.mat.set(5,8, s*(-(a[2]*b[0]) + a[0]*b[2])/2.0);

        dd.mat.set(6,0, s*(-(a[3]*b[0]) + a[0]*b[3])/2.0);
        dd.mat.set(6,1, s*(a[3]*b[1] - a[1]*b[3])/2.0);
        dd.mat.set(6,2, 0.0);
        dd.mat.set(6,3, s*(-(a[1]*b[0]) + a[0]*b[1])/2.0);
        dd.mat.set(6,4, 0.0);
        dd.mat.set(6,5, 0.0);
        dd.mat.set(6,6, s*(a[1]*b[0] + a[0]*b[1] - a[3]*b[3])/2.0);
        dd.mat.set(6,7, 0.0);
        dd.mat.set(6,8, 0.0);

        dd.mat.set(7,0, 0.0);
        dd.mat.set(7,1, 0.0);
        dd.mat.set(7,2, 0.0);
        dd.mat.set(7,3, 0.0);
        dd.mat.set(7,4, s*(-(a[2]*b[1]) + a[1]*b[2])/2.0);
        dd.mat.set(7,5, s*(a[3]*b[2] - a[2]*b[3])/tsq2);
        dd.mat.set(7,6, 0.0);
        dd.mat.set(7,7, s*(a[2]*b[1] + a[1]*b[2])/2.0);
        dd.mat.set(7,8, s*(a[3]*b[2] + a[2]*b[3])/tsq2);

        dd.mat.set(8,0, 0.0);
        dd.mat.set(8,1, 0.0);
        dd.mat.set(8,2, 0.0);
        dd.mat.set(8,3, 0.0);
        dd.mat.set(8,4, s*(a[3]*b[2] - a[2]*b[3])/tsq2);
        dd.mat.set(8,5, s*(-(a[2]*b[0]) + a[0]*b[2])/2.0);
        dd.mat.set(8,6, 0.0);
        dd.mat.set(8,7, s*(a[3]*b[2] + a[2]*b[3])/tsq2);
        dd.mat.set(8,8, s*(a[2]*b[0] + a[0]*b[2])/2.0);
    } else if dim == 6 {
        dd.mat.set(0,0, s*a[0]*b[0]);
        dd.mat.set(0,1, s*(a[3]*b[3])/2.0);
        dd.mat.set(0,2, s*(a[5]*b[5])/2.0);
        dd.mat.set(0,3, s*(a[3]*b[0] + a[0]*b[3])/2.0);
        dd.mat.set(0,4, s*(a[5]*b[3] + a[3]*b[5])/tsq2);
        dd.mat.set(0,5, s*(a[5]*b[0] + a[0]*b[5])/2.0);
        dd.mat.set(0,6, s*(-(a[3]*b[0]) + a[0]*b[3])/2.0);
        dd.mat.set(0,7, s*(-(a[5]*b[3]) + a[3]*b[5])/tsq2);
        dd.mat.set(0,8, s*(-(a[5]*b[0]) + a[0]*b[5])/2.0);
                        
        dd.mat.set(1,0, s*(a[3]*b[3])/2.0);
        dd.mat.set(1,1, s*a[1]*b[1]);
        dd.mat.set(1,2, s*(a[4]*b[4])/2.0);
        dd.mat.set(1,3, s*(a[3]*b[1] + a[1]*b[3])/2.0);
        dd.mat.set(1,4, s*(a[4]*b[1] + a[1]*b[4])/2.0);
        dd.mat.set(1,5, s*(a[4]*b[3] + a[3]*b[4])/tsq2);
        dd.mat.set(1,6, s*(a[3]*b[1] - a[1]*b[3])/2.0);
        dd.mat.set(1,7, s*(-(a[4]*b[1]) + a[1]*b[4])/2.0);
        dd.mat.set(1,8, s*(-(a[4]*b[3]) + a[3]*b[4])/tsq2);
                        
        dd.mat.set(2,0, s*(a[5]*b[5])/2.0);
        dd.mat.set(2,1, s*(a[4]*b[4])/2.0);
        dd.mat.set(2,2, s*a[2]*b[2]);
        dd.mat.set(2,3, s*(a[5]*b[4] + a[4]*b[5])/tsq2);
        dd.mat.set(2,4, s*(a[4]*b[2] + a[2]*b[4])/2.0);
        dd.mat.set(2,5, s*(a[5]*b[2] + a[2]*b[5])/2.0);
        dd.mat.set(2,6, s*(a[5]*b[4] - a[4]*b[5])/tsq2);
        dd.mat.set(2,7, s*(a[4]*b[2] - a[2]*b[4])/2.0);
        dd.mat.set(2,8, s*(a[5]*b[2] - a[2]*b[5])/2.0);
                        
        dd.mat.set(3,0, s*(a[3]*b[0] + a[0]*b[3])/2.0);
        dd.mat.set(3,1, s*(a[3]*b[1] + a[1]*b[3])/2.0);
        dd.mat.set(3,2, s*(a[5]*b[4] + a[4]*b[5])/tsq2);
        dd.mat.set(3,3, s*(a[1]*b[0] + a[0]*b[1] + a[3]*b[3])/2.0);
        dd.mat.set(3,4, s*(SQRT_2*a[5]*b[1] + a[4]*b[3] + a[3]*b[4] + SQRT_2*a[1]*b[5])/4.0);
        dd.mat.set(3,5, s*(SQRT_2*a[4]*b[0] + a[5]*b[3] + SQRT_2*a[0]*b[4] + a[3]*b[5])/4.0);
        dd.mat.set(3,6, s*(-(a[1]*b[0]) + a[0]*b[1])/2.0);
        dd.mat.set(3,7, s*(-(SQRT_2*a[5]*b[1]) - a[4]*b[3] + a[3]*b[4] + SQRT_2*a[1]*b[5])/4.0);
        dd.mat.set(3,8, s*(-(SQRT_2*a[4]*b[0]) - a[5]*b[3] + SQRT_2*a[0]*b[4] + a[3]*b[5])/4.0);
                        
        dd.mat.set(4,0, s*(a[5]*b[3] + a[3]*b[5])/tsq2);
        dd.mat.set(4,1, s*(a[4]*b[1] + a[1]*b[4])/2.0);
        dd.mat.set(4,2, s*(a[4]*b[2] + a[2]*b[4])/2.0);
        dd.mat.set(4,3, s*(SQRT_2*a[5]*b[1] + a[4]*b[3] + a[3]*b[4] + SQRT_2*a[1]*b[5])/4.0);
        dd.mat.set(4,4, s*(a[2]*b[1] + a[1]*b[2] + a[4]*b[4])/2.0);
        dd.mat.set(4,5, s*(SQRT_2*a[3]*b[2] + SQRT_2*a[2]*b[3] + a[5]*b[4] + a[4]*b[5])/4.0);
        dd.mat.set(4,6, s*(SQRT_2*a[5]*b[1] - a[4]*b[3] + a[3]*b[4] - SQRT_2*a[1]*b[5])/4.0);
        dd.mat.set(4,7, s*(-(a[2]*b[1]) + a[1]*b[2])/2.0);
        dd.mat.set(4,8, s*(SQRT_2*a[3]*b[2] - SQRT_2*a[2]*b[3] + a[5]*b[4] - a[4]*b[5])/4.0);
                        
        dd.mat.set(5,0, s*(a[5]*b[0] + a[0]*b[5])/2.0);
        dd.mat.set(5,1, s*(a[4]*b[3] + a[3]*b[4])/tsq2);
        dd.mat.set(5,2, s*(a[5]*b[2] + a[2]*b[5])/2.0);
        dd.mat.set(5,3, s*(SQRT_2*a[4]*b[0] + a[5]*b[3] + SQRT_2*a[0]*b[4] + a[3]*b[5])/4.0);
        dd.mat.set(5,4, s*(SQRT_2*a[3]*b[2] + SQRT_2*a[2]*b[3] + a[5]*b[4] + a[4]*b[5])/4.0);
        dd.mat.set(5,5, s*(a[2]*b[0] + a[0]*b[2] + a[5]*b[5])/2.0);
        dd.mat.set(5,6, s*(-(SQRT_2*a[4]*b[0]) + a[5]*b[3] + SQRT_2*a[0]*b[4] - a[3]*b[5])/4.0);
        dd.mat.set(5,7, s*(SQRT_2*a[3]*b[2] - SQRT_2*a[2]*b[3] - a[5]*b[4] + a[4]*b[5])/4.0);
        dd.mat.set(5,8, s*(-(a[2]*b[0]) + a[0]*b[2])/2.0);
                        
        dd.mat.set(6,0, s*(-(a[3]*b[0]) + a[0]*b[3])/2.0);
        dd.mat.set(6,1, s*(a[3]*b[1] - a[1]*b[3])/2.0);
        dd.mat.set(6,2, s*(a[5]*b[4] - a[4]*b[5])/tsq2);
        dd.mat.set(6,3, s*(-(a[1]*b[0]) + a[0]*b[1])/2.0);
        dd.mat.set(6,4, s*(SQRT_2*a[5]*b[1] - a[4]*b[3] + a[3]*b[4] - SQRT_2*a[1]*b[5])/4.0);
        dd.mat.set(6,5, s*(-(SQRT_2*a[4]*b[0]) + a[5]*b[3] + SQRT_2*a[0]*b[4] - a[3]*b[5])/4.0);
        dd.mat.set(6,6, s*(a[1]*b[0] + a[0]*b[1] - a[3]*b[3])/2.0);
        dd.mat.set(6,7, s*(-(SQRT_2*a[5]*b[1]) + a[4]*b[3] + a[3]*b[4] - SQRT_2*a[1]*b[5])/4.0);
        dd.mat.set(6,8, s*(SQRT_2*a[4]*b[0] - a[5]*b[3] + SQRT_2*a[0]*b[4] - a[3]*b[5])/4.0);
                        
        dd.mat.set(7,0, s*(-(a[5]*b[3]) + a[3]*b[5])/tsq2);
        dd.mat.set(7,1, s*(-(a[4]*b[1]) + a[1]*b[4])/2.0);
        dd.mat.set(7,2, s*(a[4]*b[2] - a[2]*b[4])/2.0);
        dd.mat.set(7,3, s*(-(SQRT_2*a[5]*b[1]) - a[4]*b[3] + a[3]*b[4] + SQRT_2*a[1]*b[5])/4.0);
        dd.mat.set(7,4, s*(-(a[2]*b[1]) + a[1]*b[2])/2.0);
        dd.mat.set(7,5, s*(SQRT_2*a[3]*b[2] - SQRT_2*a[2]*b[3] - a[5]*b[4] + a[4]*b[5])/4.0);
        dd.mat.set(7,6, s*(-(SQRT_2*a[5]*b[1]) + a[4]*b[3] + a[3]*b[4] - SQRT_2*a[1]*b[5])/4.0);
        dd.mat.set(7,7, s*(a[2]*b[1] + a[1]*b[2] - a[4]*b[4])/2.0);
        dd.mat.set(7,8, s*(SQRT_2*a[3]*b[2] + SQRT_2*a[2]*b[3] - a[5]*b[4] - a[4]*b[5])/4.0);
                        
        dd.mat.set(8,0, s*(-(a[5]*b[0]) + a[0]*b[5])/2.0);
        dd.mat.set(8,1, s*(-(a[4]*b[3]) + a[3]*b[4])/tsq2);
        dd.mat.set(8,2, s*(a[5]*b[2] - a[2]*b[5])/2.0);
        dd.mat.set(8,3, s*(-(SQRT_2*a[4]*b[0]) - a[5]*b[3] + SQRT_2*a[0]*b[4] + a[3]*b[5])/4.0);
        dd.mat.set(8,4, s*(SQRT_2*a[3]*b[2] - SQRT_2*a[2]*b[3] + a[5]*b[4] - a[4]*b[5])/4.0);
        dd.mat.set(8,5, s*(-(a[2]*b[0]) + a[0]*b[2])/2.0);
        dd.mat.set(8,6, s*(SQRT_2*a[4]*b[0] - a[5]*b[3] + SQRT_2*a[0]*b[4] - a[3]*b[5])/4.0);
        dd.mat.set(8,7, s*(SQRT_2*a[3]*b[2] + SQRT_2*a[2]*b[3] - a[5]*b[4] - a[4]*b[5])/4.0);
        dd.mat.set(8,8, s*(a[2]*b[0] + a[0]*b[2] - a[5]*b[5])/2.0);
    } else {
        dd.mat.set(0,0, s*a[0]*b[0]);
        dd.mat.set(0,1, s*((a[3] + a[6])*(b[3] + b[6]))/2.0);
        dd.mat.set(0,2, s*((a[5] + a[8])*(b[5] + b[8]))/2.0);
        dd.mat.set(0,3, s*(a[3]*b[0] + a[6]*b[0] + a[0]*(b[3] + b[6]))/2.0);
        dd.mat.set(0,4, s*((a[5] + a[8])*(b[3] + b[6]) + (a[3] + a[6])*(b[5] + b[8]))/tsq2);
        dd.mat.set(0,5, s*(a[5]*b[0] + a[8]*b[0] + a[0]*(b[5] + b[8]))/2.0);
        dd.mat.set(0,6, s*(-(a[3]*b[0]) - a[6]*b[0] + a[0]*(b[3] + b[6]))/2.0);
        dd.mat.set(0,7, s*(-((a[5] + a[8])*(b[3] + b[6])) + (a[3] + a[6])*(b[5] + b[8]))/tsq2);
        dd.mat.set(0,8, s*(-(a[5]*b[0]) - a[8]*b[0] + a[0]*(b[5] + b[8]))/2.0);

        dd.mat.set(1,0, s*((a[3] - a[6])*(b[3] - b[6]))/2.0);
        dd.mat.set(1,1, s*a[1]*b[1]);
        dd.mat.set(1,2, s*((a[4] + a[7])*(b[4] + b[7]))/2.0);
        dd.mat.set(1,3, s*(a[3]*b[1] - a[6]*b[1] + a[1]*(b[3] - b[6]))/2.0);
        dd.mat.set(1,4, s*(a[4]*b[1] + a[7]*b[1] + a[1]*(b[4] + b[7]))/2.0);
        dd.mat.set(1,5, s*((a[4] + a[7])*(b[3] - b[6]) + (a[3] - a[6])*(b[4] + b[7]))/tsq2);
        dd.mat.set(1,6, s*(a[3]*b[1] - a[6]*b[1] + a[1]*(-b[3] + b[6]))/2.0);
        dd.mat.set(1,7, s*(-(a[4]*b[1]) - a[7]*b[1] + a[1]*(b[4] + b[7]))/2.0);
        dd.mat.set(1,8, s*(-((a[4] + a[7])*(b[3] - b[6])) + (a[3] - a[6])*(b[4] + b[7]))/tsq2);

        dd.mat.set(2,0, s*((a[5] - a[8])*(b[5] - b[8]))/2.0);
        dd.mat.set(2,1, s*((a[4] - a[7])*(b[4] - b[7]))/2.0);
        dd.mat.set(2,2, s*a[2]*b[2]);
        dd.mat.set(2,3, s*((a[5] - a[8])*(b[4] - b[7]) + (a[4] - a[7])*(b[5] - b[8]))/tsq2);
        dd.mat.set(2,4, s*(a[4]*b[2] - a[7]*b[2] + a[2]*(b[4] - b[7]))/2.0);
        dd.mat.set(2,5, s*(a[5]*b[2] - a[8]*b[2] + a[2]*(b[5] - b[8]))/2.0);
        dd.mat.set(2,6, s*((a[5] - a[8])*(b[4] - b[7]) - (a[4] - a[7])*(b[5] - b[8]))/tsq2);
        dd.mat.set(2,7, s*(a[4]*b[2] - a[7]*b[2] + a[2]*(-b[4] + b[7]))/2.0);
        dd.mat.set(2,8, s*(a[5]*b[2] - a[8]*b[2] + a[2]*(-b[5] + b[8]))/2.0);

        dd.mat.set(3,0, s*(a[3]*b[0] - a[6]*b[0] + a[0]*(b[3] - b[6]))/2.0);
        dd.mat.set(3,1, s*(a[3]*b[1] + a[6]*b[1] + a[1]*(b[3] + b[6]))/2.0);
        dd.mat.set(3,2, s*((a[5] + a[8])*(b[4] + b[7]) + (a[4] + a[7])*(b[5] + b[8]))/tsq2);
        dd.mat.set(3,3, s*(a[1]*b[0] + a[0]*b[1] + a[3]*b[3] - a[6]*b[6])/2.0);
        dd.mat.set(3,4, s*(SQRT_2*(a[5] + a[8])*b[1] + (a[4] + a[7])*(b[3] + b[6]) + (a[3] + a[6])*(b[4] + b[7]) + SQRT_2*a[1]*(b[5] + b[8]))/4.0);
        dd.mat.set(3,5, s*(SQRT_2*(a[4] + a[7])*b[0] + (a[5] + a[8])*(b[3] - b[6]) + SQRT_2*a[0]*(b[4] + b[7]) + (a[3] - a[6])*(b[5] + b[8]))/4.0);
        dd.mat.set(3,6, s*(-(a[1]*b[0]) + a[0]*b[1] - a[6]*b[3] + a[3]*b[6])/2.0);
        dd.mat.set(3,7, s*(-(SQRT_2*(a[5] + a[8])*b[1]) - (a[4] + a[7])*(b[3] + b[6]) + (a[3] + a[6])*(b[4] + b[7]) + SQRT_2*a[1]*(b[5] + b[8]))/4.0);
        dd.mat.set(3,8, s*(-(SQRT_2*(a[4] + a[7])*b[0]) - (a[5] + a[8])*(b[3] - b[6]) + SQRT_2*a[0]*(b[4] + b[7]) + (a[3] - a[6])*(b[5] + b[8]))/4.0);

        dd.mat.set(4,0, s*((a[5] - a[8])*(b[3] - b[6]) + (a[3] - a[6])*(b[5] - b[8]))/tsq2);
        dd.mat.set(4,1, s*(a[4]*b[1] - a[7]*b[1] + a[1]*(b[4] - b[7]))/2.0);
        dd.mat.set(4,2, s*(a[4]*b[2] + a[7]*b[2] + a[2]*(b[4] + b[7]))/2.0);
        dd.mat.set(4,3, s*(SQRT_2*(a[5] - a[8])*b[1] + (a[4] - a[7])*(b[3] - b[6]) + (a[3] - a[6])*(b[4] - b[7]) + SQRT_2*a[1]*(b[5] - b[8]))/4.0);
        dd.mat.set(4,4, s*(a[2]*b[1] + a[1]*b[2] + a[4]*b[4] - a[7]*b[7])/2.0);
        dd.mat.set(4,5, s*(SQRT_2*(a[3] - a[6])*b[2] + SQRT_2*a[2]*(b[3] - b[6]) + (a[5] - a[8])*(b[4] + b[7]) + (a[4] + a[7])*(b[5] - b[8]))/4.0);
        dd.mat.set(4,6, s*(SQRT_2*(a[5] - a[8])*b[1] - (a[4] - a[7])*(b[3] - b[6]) + (a[3] - a[6])*(b[4] - b[7]) - SQRT_2*a[1]*(b[5] - b[8]))/4.0);
        dd.mat.set(4,7, s*(-(a[2]*b[1]) + a[1]*b[2] - a[7]*b[4] + a[4]*b[7])/2.0);
        dd.mat.set(4,8, s*(SQRT_2*(a[3] - a[6])*b[2] - SQRT_2*a[2]*(b[3] - b[6]) + (a[5] - a[8])*(b[4] + b[7]) - (a[4] + a[7])*(b[5] - b[8]))/4.0);

        dd.mat.set(5,0, s*(a[5]*b[0] - a[8]*b[0] + a[0]*(b[5] - b[8]))/2.0);
        dd.mat.set(5,1, s*((a[4] - a[7])*(b[3] + b[6]) + (a[3] + a[6])*(b[4] - b[7]))/tsq2);
        dd.mat.set(5,2, s*(a[5]*b[2] + a[8]*b[2] + a[2]*(b[5] + b[8]))/2.0);
        dd.mat.set(5,3, s*(SQRT_2*(a[4] - a[7])*b[0] + (a[5] - a[8])*(b[3] + b[6]) + SQRT_2*a[0]*(b[4] - b[7]) + (a[3] + a[6])*(b[5] - b[8]))/4.0);
        dd.mat.set(5,4, s*(SQRT_2*(a[3] + a[6])*b[2] + SQRT_2*a[2]*(b[3] + b[6]) + (a[5] + a[8])*(b[4] - b[7]) + (a[4] - a[7])*(b[5] + b[8]))/4.0);
        dd.mat.set(5,5, s*(a[2]*b[0] + a[0]*b[2] + a[5]*b[5] - a[8]*b[8])/2.0);
        dd.mat.set(5,6, s*(-(SQRT_2*(a[4] - a[7])*b[0]) + (a[5] - a[8])*(b[3] + b[6]) + SQRT_2*a[0]*(b[4] - b[7]) - (a[3] + a[6])*(b[5] - b[8]))/4.0);
        dd.mat.set(5,7, s*(SQRT_2*(a[3] + a[6])*b[2] - SQRT_2*a[2]*(b[3] + b[6]) - (a[5] + a[8])*(b[4] - b[7]) + (a[4] - a[7])*(b[5] + b[8]))/4.0);
        dd.mat.set(5,8, s*(-(a[2]*b[0]) + a[0]*b[2] - a[8]*b[5] + a[5]*b[8])/2.0);

        dd.mat.set(6,0, s*(-(a[3]*b[0]) + a[6]*b[0] + a[0]*(b[3] - b[6]))/2.0);
        dd.mat.set(6,1, s*(a[3]*b[1] + a[6]*b[1] - a[1]*(b[3] + b[6]))/2.0);
        dd.mat.set(6,2, s*((a[5] + a[8])*(b[4] + b[7]) - (a[4] + a[7])*(b[5] + b[8]))/tsq2);
        dd.mat.set(6,3, s*(-(a[1]*b[0]) + a[0]*b[1] + a[6]*b[3] - a[3]*b[6])/2.0);
        dd.mat.set(6,4, s*(SQRT_2*(a[5] + a[8])*b[1] - (a[4] + a[7])*(b[3] + b[6]) + (a[3] + a[6])*(b[4] + b[7]) - SQRT_2*a[1]*(b[5] + b[8]))/4.0);
        dd.mat.set(6,5, s*(-(SQRT_2*(a[4] + a[7])*b[0]) + (a[5] + a[8])*(b[3] - b[6]) + SQRT_2*a[0]*(b[4] + b[7]) - (a[3] - a[6])*(b[5] + b[8]))/4.0);
        dd.mat.set(6,6, s*(a[1]*b[0] + a[0]*b[1] - a[3]*b[3] + a[6]*b[6])/2.0);
        dd.mat.set(6,7, s*(-(SQRT_2*(a[5] + a[8])*b[1]) + (a[4] + a[7])*(b[3] + b[6]) + (a[3] + a[6])*(b[4] + b[7]) - SQRT_2*a[1]*(b[5] + b[8]))/4.0);
        dd.mat.set(6,8, s*(SQRT_2*(a[4] + a[7])*b[0] - (a[5] + a[8])*(b[3] - b[6]) + SQRT_2*a[0]*(b[4] + b[7]) - (a[3] - a[6])*(b[5] + b[8]))/4.0);

        dd.mat.set(7,0, s*(-((a[5] - a[8])*(b[3] - b[6])) + (a[3] - a[6])*(b[5] - b[8]))/tsq2);
        dd.mat.set(7,1, s*(-(a[4]*b[1]) + a[7]*b[1] + a[1]*(b[4] - b[7]))/2.0);
        dd.mat.set(7,2, s*(a[4]*b[2] + a[7]*b[2] - a[2]*(b[4] + b[7]))/2.0);
        dd.mat.set(7,3, s*(-(SQRT_2*(a[5] - a[8])*b[1]) - (a[4] - a[7])*(b[3] - b[6]) + (a[3] - a[6])*(b[4] - b[7]) + SQRT_2*a[1]*(b[5] - b[8]))/4.0);
        dd.mat.set(7,4, s*(-(a[2]*b[1]) + a[1]*b[2] + a[7]*b[4] - a[4]*b[7])/2.0);
        dd.mat.set(7,5, s*(SQRT_2*(a[3] - a[6])*b[2] - SQRT_2*a[2]*(b[3] - b[6]) - (a[5] - a[8])*(b[4] + b[7]) + (a[4] + a[7])*(b[5] - b[8]))/4.0);
        dd.mat.set(7,6, s*(-(SQRT_2*(a[5] - a[8])*b[1]) + (a[4] - a[7])*(b[3] - b[6]) + (a[3] - a[6])*(b[4] - b[7]) - SQRT_2*a[1]*(b[5] - b[8]))/4.0);
        dd.mat.set(7,7, s*(a[2]*b[1] + a[1]*b[2] - a[4]*b[4] + a[7]*b[7])/2.0);
        dd.mat.set(7,8, s*(SQRT_2*(a[3] - a[6])*b[2] + SQRT_2*a[2]*(b[3] - b[6]) - (a[5] - a[8])*(b[4] + b[7]) - (a[4] + a[7])*(b[5] - b[8]))/4.0);

        dd.mat.set(8,0, s*(-(a[5]*b[0]) + a[8]*b[0] + a[0]*(b[5] - b[8]))/2.0);
        dd.mat.set(8,1, s*(-((a[4] - a[7])*(b[3] + b[6])) + (a[3] + a[6])*(b[4] - b[7]))/tsq2);
        dd.mat.set(8,2, s*(a[5]*b[2] + a[8]*b[2] - a[2]*(b[5] + b[8]))/2.0);
        dd.mat.set(8,3, s*(-(SQRT_2*(a[4] - a[7])*b[0]) - (a[5] - a[8])*(b[3] + b[6]) + SQRT_2*a[0]*(b[4] - b[7]) + (a[3] + a[6])*(b[5] - b[8]))/4.0);
        dd.mat.set(8,4, s*(SQRT_2*(a[3] + a[6])*b[2] - SQRT_2*a[2]*(b[3] + b[6]) + (a[5] + a[8])*(b[4] - b[7]) - (a[4] - a[7])*(b[5] + b[8]))/4.0);
        dd.mat.set(8,5, s*(-(a[2]*b[0]) + a[0]*b[2] + a[8]*b[5] - a[5]*b[8])/2.0);
        dd.mat.set(8,6, s*(SQRT_2*(a[4] - a[7])*b[0] - (a[5] - a[8])*(b[3] + b[6]) + SQRT_2*a[0]*(b[4] - b[7]) - (a[3] + a[6])*(b[5] - b[8]))/4.0);
        dd.mat.set(8,7, s*(SQRT_2*(a[3] + a[6])*b[2] + SQRT_2*a[2]*(b[3] + b[6]) - (a[5] + a[8])*(b[4] - b[7]) - (a[4] - a[7])*(b[5] + b[8]))/4.0);
        dd.mat.set(8,8, s*(a[2]*b[0] + a[0]*b[2] - a[5]*b[5] + a[8]*b[8])/2.0);
    }
}

/// Performs the underbar dyadic product between two Tensor2 resulting in a (general) Tensor4
///
/// Computes:
///
/// ```text
/// D = s A ⊗ B
///         ‾
/// ```
/// 
/// With orthonormal Cartesian components:
///
/// ```text
/// Dᵢⱼₖₗ = s Aᵢₗ Bⱼₖ
/// ```
///
/// **Important:** The result is **not** necessarily minor-symmetric; therefore `D` must be General.
///
/// # Output
///
/// * `dd` -- the tensor `D`; it must be [Mandel::General]
///
/// # Input
///
/// * `a` -- first tensor; with the same [Mandel] as `b`
/// * `b` -- second tensor; with the same [Mandel] as `a`
///
/// # Panics
///
/// 1. A panic will occur if `dd` is not [Mandel::General]
/// 2. A panic will occur the `a` and `b` have different [Mandel]
#[rustfmt::skip]
pub fn t2_udyad_t2(dd: &mut Tensor4, s: f64, aa: &Tensor2, bb: &Tensor2) {
    assert_eq!(dd.mandel, Mandel::General);
    assert_eq!(bb.mandel, aa.mandel);
    let dim = aa.vec.dim();
    let a = &aa.vec;
    let b = &bb.vec;
    let tsq2 = 2.0 * SQRT_2;
    if dim == 4 {
        dd.mat.set(0,0, s*a[0]*b[0]);
        dd.mat.set(0,1, s*(a[3]*b[3])/2.0);
        dd.mat.set(0,2, 0.0);
        dd.mat.set(0,3, s*(a[3]*b[0] + a[0]*b[3])/2.0);
        dd.mat.set(0,4, 0.0);
        dd.mat.set(0,5, 0.0);
        dd.mat.set(0,6, s*(a[3]*b[0] - a[0]*b[3])/2.0);
        dd.mat.set(0,7, 0.0);
        dd.mat.set(0,8, 0.0);
                        
        dd.mat.set(1,0, s*(a[3]*b[3])/2.0);
        dd.mat.set(1,1, s*a[1]*b[1]);
        dd.mat.set(1,2, 0.0);
        dd.mat.set(1,3, s*(a[3]*b[1] + a[1]*b[3])/2.0);
        dd.mat.set(1,4, 0.0);
        dd.mat.set(1,5, 0.0);
        dd.mat.set(1,6, s*(-(a[3]*b[1]) + a[1]*b[3])/2.0);
        dd.mat.set(1,7, 0.0);
        dd.mat.set(1,8, 0.0);
                        
        dd.mat.set(2,0, 0.0);
        dd.mat.set(2,1, 0.0);
        dd.mat.set(2,2, s*a[2]*b[2]);
        dd.mat.set(2,3, 0.0);
        dd.mat.set(2,4, 0.0);
        dd.mat.set(2,5, 0.0);
        dd.mat.set(2,6, 0.0);
        dd.mat.set(2,7, 0.0);
        dd.mat.set(2,8, 0.0);
                        
        dd.mat.set(3,0, s*(a[3]*b[0] + a[0]*b[3])/2.0);
        dd.mat.set(3,1, s*(a[3]*b[1] + a[1]*b[3])/2.0);
        dd.mat.set(3,2, 0.0);
        dd.mat.set(3,3, s*(a[1]*b[0] + a[0]*b[1] + a[3]*b[3])/2.0);
        dd.mat.set(3,4, 0.0);
        dd.mat.set(3,5, 0.0);
        dd.mat.set(3,6, s*(a[1]*b[0] - a[0]*b[1])/2.0);
        dd.mat.set(3,7, 0.0);
        dd.mat.set(3,8, 0.0);
                        
        dd.mat.set(4,0, 0.0);
        dd.mat.set(4,1, 0.0);
        dd.mat.set(4,2, 0.0);
        dd.mat.set(4,3, 0.0);
        dd.mat.set(4,4, s*(a[2]*b[1] + a[1]*b[2])/2.0);
        dd.mat.set(4,5, s*(a[3]*b[2] + a[2]*b[3])/tsq2);
        dd.mat.set(4,6, 0.0);
        dd.mat.set(4,7, s*(a[2]*b[1] - a[1]*b[2])/2.0);
        dd.mat.set(4,8, s*(-(a[3]*b[2]) + a[2]*b[3])/tsq2);
                        
        dd.mat.set(5,0, 0.0);
        dd.mat.set(5,1, 0.0);
        dd.mat.set(5,2, 0.0);
        dd.mat.set(5,3, 0.0);
        dd.mat.set(5,4, s*(a[3]*b[2] + a[2]*b[3])/tsq2);
        dd.mat.set(5,5, s*(a[2]*b[0] + a[0]*b[2])/2.0);
        dd.mat.set(5,6, 0.0);
        dd.mat.set(5,7, s*(-(a[3]*b[2]) + a[2]*b[3])/tsq2);
        dd.mat.set(5,8, s*(a[2]*b[0] - a[0]*b[2])/2.0);
                        
        dd.mat.set(6,0, s*(-(a[3]*b[0]) + a[0]*b[3])/2.0);
        dd.mat.set(6,1, s*(a[3]*b[1] - a[1]*b[3])/2.0);
        dd.mat.set(6,2, 0.0);
        dd.mat.set(6,3, s*(-(a[1]*b[0]) + a[0]*b[1])/2.0);
        dd.mat.set(6,4, 0.0);
        dd.mat.set(6,5, 0.0);
        dd.mat.set(6,6, s*(-(a[1]*b[0]) - a[0]*b[1] + a[3]*b[3])/2.0);
        dd.mat.set(6,7, 0.0);
        dd.mat.set(6,8, 0.0);
                        
        dd.mat.set(7,0, 0.0);
        dd.mat.set(7,1, 0.0);
        dd.mat.set(7,2, 0.0);
        dd.mat.set(7,3, 0.0);
        dd.mat.set(7,4, s*(-(a[2]*b[1]) + a[1]*b[2])/2.0);
        dd.mat.set(7,5, s*(a[3]*b[2] - a[2]*b[3])/tsq2);
        dd.mat.set(7,6, 0.0);
        dd.mat.set(7,7, s*(-(a[2]*b[1]) - a[1]*b[2])/2.0);
        dd.mat.set(7,8, s*(-(a[3]*b[2] + a[2]*b[3])/tsq2));
                        
        dd.mat.set(8,0, 0.0);
        dd.mat.set(8,1, 0.0);
        dd.mat.set(8,2, 0.0);
        dd.mat.set(8,3, 0.0);
        dd.mat.set(8,4, s*(a[3]*b[2] - a[2]*b[3])/tsq2);
        dd.mat.set(8,5, s*(-(a[2]*b[0]) + a[0]*b[2])/2.0);
        dd.mat.set(8,6, 0.0);
        dd.mat.set(8,7, s*(-(a[3]*b[2] + a[2]*b[3])/tsq2));
        dd.mat.set(8,8, s*(-(a[2]*b[0]) - a[0]*b[2])/2.0);
    } else if dim == 6 {
        dd.mat.set(0,0, s*a[0]*b[0]);
        dd.mat.set(0,1, s*(a[3]*b[3])/2.0);
        dd.mat.set(0,2, s*(a[5]*b[5])/2.0);
        dd.mat.set(0,3, s*(a[3]*b[0] + a[0]*b[3])/2.0);
        dd.mat.set(0,4, s*(a[5]*b[3] + a[3]*b[5])/tsq2);
        dd.mat.set(0,5, s*(a[5]*b[0] + a[0]*b[5])/2.0);
        dd.mat.set(0,6, s*(a[3]*b[0] - a[0]*b[3])/2.0);
        dd.mat.set(0,7, s*(a[5]*b[3] - a[3]*b[5])/tsq2);
        dd.mat.set(0,8, s*(a[5]*b[0] - a[0]*b[5])/2.0);
                        
        dd.mat.set(1,0, s*(a[3]*b[3])/2.0);
        dd.mat.set(1,1, s*a[1]*b[1]);
        dd.mat.set(1,2, s*(a[4]*b[4])/2.0);
        dd.mat.set(1,3, s*(a[3]*b[1] + a[1]*b[3])/2.0);
        dd.mat.set(1,4, s*(a[4]*b[1] + a[1]*b[4])/2.0);
        dd.mat.set(1,5, s*(a[4]*b[3] + a[3]*b[4])/tsq2);
        dd.mat.set(1,6, s*(-(a[3]*b[1]) + a[1]*b[3])/2.0);
        dd.mat.set(1,7, s*(a[4]*b[1] - a[1]*b[4])/2.0);
        dd.mat.set(1,8, s*(a[4]*b[3] - a[3]*b[4])/tsq2);
                        
        dd.mat.set(2,0, s*(a[5]*b[5])/2.0);
        dd.mat.set(2,1, s*(a[4]*b[4])/2.0);
        dd.mat.set(2,2, s*a[2]*b[2]);
        dd.mat.set(2,3, s*(a[ 5]*b[4] + a[4]*b[5])/tsq2);
        dd.mat.set(2,4, s*(a[4]*b[2] + a[2]*b[4])/2.0);
        dd.mat.set(2,5, s*(a[5]*b[2] + a[2]*b[5])/2.0);
        dd.mat.set(2,6, s*(-(a[5]*b[4]) + a[4]*b[5])/tsq2);
        dd.mat.set(2,7, s*(-(a[4]*b[2]) + a[2]*b[4])/2.0);
        dd.mat.set(2,8, s*(-(a[5]*b[2]) + a[2]*b[5])/2.0);
                        
        dd.mat.set(3,0, s*(a[3]*b[0] + a[0]*b[3])/2.0);
        dd.mat.set(3,1, s*(a[3]*b[1] + a[1]*b[3])/2.0);
        dd.mat.set(3,2, s*(a[5]*b[4] + a[4]*b[5])/tsq2);
        dd.mat.set(3,3, s*(a[1]*b[0] + a[0]*b[1] + a[3]*b[3])/2.0);
        dd.mat.set(3,4, s*(SQRT_2*a[5]*b[1] + a[4]*b[3] + a[3]*b[4] + SQRT_2*a[1]*b[5])/4.0);
        dd.mat.set(3,5, s*(SQRT_2*a[4]*b[0] + a[5]*b[3] + SQRT_2*a[0]*b[4] + a[3]*b[5])/4.0);
        dd.mat.set(3,6, s*(a[1]*b[0] - a[0]*b[1])/2.0);
        dd.mat.set(3,7, s*(SQRT_2*a[5]*b[1] + a[4]*b[3] - a[3]*b[4] - SQRT_2*a[1]*b[5])/4.0);
        dd.mat.set(3,8, s*(SQRT_2*a[4]*b[0] + a[5]*b[3] - SQRT_2*a[0]*b[4] - a[3]*b[5])/4.0);
                        
        dd.mat.set(4,0, s*(a[5]*b[3] + a[3]*b[5])/tsq2);
        dd.mat.set(4,1, s*(a[4]*b[1] + a[1]*b[4])/2.0);
        dd.mat.set(4,2, s*(a[4]*b[2] + a[2]*b[4])/2.0);
        dd.mat.set(4,3, s*(SQRT_2*a[5]*b[1] + a[4]*b[3] + a[3]*b[4] + SQRT_2*a[1]*b[5])/4.0);
        dd.mat.set(4,4, s*(a[2]*b[1] + a[1]*b[2] + a[4]*b[4])/2.0);
        dd.mat.set(4,5, s*(SQRT_2*a[3]*b[2] + SQRT_2*a[2]*b[3] + a[5]*b[4] + a[4]*b[5])/4.0);
        dd.mat.set(4,6, s*(-(SQRT_2*a[5]*b[1]) + a[4]*b[3] - a[3]*b[4] + SQRT_2*a[1]*b[5])/4.0);
        dd.mat.set(4,7, s*(a[2]*b[1] - a[1]*b[2])/2.0);
        dd.mat.set(4,8, s*(-(SQRT_2*a[3]*b[2]) + SQRT_2*a[2]*b[3] - a[5]*b[4] + a[4]*b[5])/4.0);
                        
        dd.mat.set(5,0, s*(a[5]*b[0] + a[0]*b[5])/2.0);
        dd.mat.set(5,1, s*(a[4]*b[3] + a[3]*b[4])/tsq2);
        dd.mat.set(5,2, s*(a[5]*b[2] + a[2]*b[5])/2.0);
        dd.mat.set(5,3, s*(SQRT_2*a[4]*b[0] + a[5]*b[3] + SQRT_2*a[0]*b[4] + a[3]*b[5])/4.0);
        dd.mat.set(5,4, s*(SQRT_2*a[3]*b[2] + SQRT_2*a[2]*b[3] + a[5]*b[4] + a[4]*b[5])/4.0);
        dd.mat.set(5,5, s*(a[2]*b[0] + a[0]*b[2] + a[5]*b[5])/2.0);
        dd.mat.set(5,6, s*(SQRT_2*a[4]*b[0] - a[5]*b[3] - SQRT_2*a[0]*b[4] + a[3]*b[5])/4.0);
        dd.mat.set(5,7, s*(-(SQRT_2*a[3]*b[2]) + SQRT_2*a[2]*b[3] + a[5]*b[4] - a[4]*b[5])/4.0);
        dd.mat.set(5,8, s*(a[2]*b[0] - a[0]*b[2])/2.0);
                        
        dd.mat.set(6,0, s*(-(a[3]*b[0]) + a[0]*b[3])/2.0);
        dd.mat.set(6,1, s*(a[3]*b[1] - a[1]*b[3])/2.0);
        dd.mat.set(6,2, s*(a[5]*b[4] - a[4]*b[5])/tsq2);
        dd.mat.set(6,3, s*(-(a[1]*b[0]) + a[0]*b[1])/2.0);
        dd.mat.set(6,4, s*(SQRT_2*a[5]*b[1] - a[4]*b[3] + a[3]*b[4] - SQRT_2*a[1]*b[5])/4.0);
        dd.mat.set(6,5, s*(-(SQRT_2*a[4]*b[0]) + a[5]*b[3] + SQRT_2*a[0]*b[4] - a[3]*b[5])/4.0);
        dd.mat.set(6,6, s*(-(a[1]*b[0]) - a[0]*b[1] + a[3]*b[3])/2.0);
        dd.mat.set(6,7, s*(SQRT_2*a[5]*b[1] - a[4]*b[3] - a[3]*b[4] + SQRT_2*a[1]*b[5])/4.0);
        dd.mat.set(6,8, s*(-(SQRT_2*a[4]*b[0]) + a[5]*b[3] - SQRT_2*a[0]*b[4] + a[3]*b[5])/4.0);
                        
        dd.mat.set(7,0, s*(-(a[5]*b[3]) + a[3]*b[5])/tsq2);
        dd.mat.set(7,1, s*(-(a[4]*b[1]) + a[1]*b[4])/2.0);
        dd.mat.set(7,2, s*(a[4]*b[2] - a[2]*b[4])/2.0);
        dd.mat.set(7,3, s*(-(SQRT_2*a[5]*b[1]) - a[4]*b[3] + a[3]*b[4] + SQRT_2*a[1]*b[5])/4.0);
        dd.mat.set(7,4, s*(-(a[2]*b[1]) + a[1]*b[2])/2.0);
        dd.mat.set(7,5, s*(SQRT_2*a[3]*b[2] - SQRT_2*a[2]*b[3] - a[5]*b[4] + a[4]*b[5])/4.0);
        dd.mat.set(7,6, s*(SQRT_2*a[5]*b[1] - a[4]*b[3] - a[3]*b[4] + SQRT_2*a[1]*b[5])/4.0);
        dd.mat.set(7,7, s*(-(a[2]*b[1]) - a[1]*b[2] + a[4]*b[4])/2.0);
        dd.mat.set(7,8, s*(-(SQRT_2*a[3]*b[2]) - SQRT_2*a[2]*b[3] + a[5]*b[4] + a[4]*b[5])/4.0);
                        
        dd.mat.set(8,0, s*(-(a[5]*b[0]) + a[0]*b[5])/2.0);
        dd.mat.set(8,1, s*(-(a[4]*b[3]) + a[3]*b[4])/tsq2);
        dd.mat.set(8,2, s*(a[5]*b[2] - a[2]*b[5])/2.0);
        dd.mat.set(8,3, s*(-(SQRT_2*a[4]*b[0]) - a[5]*b[3] + SQRT_2*a[0]*b[4] + a[3]*b[5])/4.0);
        dd.mat.set(8,4, s*(SQRT_2*a[3]*b[2] - SQRT_2*a[2]*b[3] + a[5]*b[4] - a[4]*b[5])/4.0);
        dd.mat.set(8,5, s*(-(a[2]*b[0]) + a[0]*b[2])/2.0);
        dd.mat.set(8,6, s*(-(SQRT_2*a[4]*b[0]) + a[5]*b[3] - SQRT_2*a[0]*b[4] + a[3]*b[5])/4.0);
        dd.mat.set(8,7, s*(-(SQRT_2*a[3]*b[2]) - SQRT_2*a[2]*b[3] + a[5]*b[4] + a[4]*b[5])/4.0);
        dd.mat.set(8,8, s*(-(a[2]*b[0]) - a[0]*b[2] + a[5]*b[5])/2.0);
    } else {
        dd.mat.set(0,0, s*a[0]*b[0]);
        dd.mat.set(0,1, s*((a[3] + a[6])*(b[3] + b[6]))/2.0);
        dd.mat.set(0,2, s*((a[5] + a[8])*(b[5] + b[8]))/2.0);
        dd.mat.set(0,3, s*(a[3]*b[0] + a[6]*b[0] + a[0]*(b[3] + b[6]))/2.0);
        dd.mat.set(0,4, s*((a[5] + a[8])*(b[3] + b[6]) + (a[3] + a[6])*(b[5] + b[8]))/tsq2);
        dd.mat.set(0,5, s*(a[5]*b[0] + a[8]*b[0] + a[0]*(b[5] + b[8]))/2.0);
        dd.mat.set(0,6, s*(a[3]*b[0] + a[6]*b[0] - a[0]*(b[3] + b[6]))/2.0);
        dd.mat.set(0,7, s*((a[5] + a[8])*(b[3] + b[6]) - (a[3] + a[6])*(b[5] + b[8]))/tsq2);
        dd.mat.set(0,8, s*(a[5]*b[0] + a[8]*b[0] - a[0]*(b[5] + b[8]))/2.0);

        dd.mat.set(1,0, s*((a[3] - a[6])*(b[3] - b[6]))/2.0);
        dd.mat.set(1,1, s*a[1]*b[1]);
        dd.mat.set(1,2, s*((a[4] + a[7])*(b[4] + b[7]))/2.0);
        dd.mat.set(1,3, s*(a[3]*b[1] - a[6]*b[1] + a[1]*(b[3] - b[6]))/2.0);
        dd.mat.set(1,4, s*(a[4]*b[1] + a[7]*b[1] + a[1]*(b[4] + b[7]))/2.0);
        dd.mat.set(1,5, s*((a[4] + a[7])*(b[3] - b[6]) + (a[3] - a[6])*(b[4] + b[7]))/tsq2);
        dd.mat.set(1,6, s*(-(a[3]*b[1]) + a[6]*b[1] + a[1]*(b[3] - b[6]))/2.0);
        dd.mat.set(1,7, s*(a[4]*b[1] + a[7]*b[1] - a[1]*(b[4] + b[7]))/2.0);
        dd.mat.set(1,8, s*((a[4] + a[7])*(b[3] - b[6]) - (a[3] - a[6])*(b[4] + b[7]))/tsq2);

        dd.mat.set(2,0, s*((a[5] - a[8])*(b[5] - b[8]))/2.0);
        dd.mat.set(2,1, s*((a[4] - a[7])*(b[4] - b[7]))/2.0);
        dd.mat.set(2,2, s*a[2]*b[2]);
        dd.mat.set(2,3, s*((a[5] - a[8])*(b[4] - b[7]) + (a[4] - a[7])*(b[5] - b[8]))/tsq2);
        dd.mat.set(2,4, s*(a[4]*b[2] - a[7]*b[2] + a[2]*(b[4] - b[7]))/2.0);
        dd.mat.set(2,5, s*(a[5]*b[2] - a[8]*b[2] + a[2]*(b[5] - b[8]))/2.0);
        dd.mat.set(2,6, s*(-((a[5] - a[8])*(b[4] - b[7])) + (a[4] - a[7])*(b[5] - b[8]))/tsq2);
        dd.mat.set(2,7, s*(-(a[4]*b[2]) + a[7]*b[2] + a[2]*(b[4] - b[7]))/2.0);
        dd.mat.set(2,8, s*(-(a[5]*b[2]) + a[8]*b[2] + a[2]*(b[5] - b[8]))/2.0);

        dd.mat.set(3,0, s*(a[3]*b[0] - a[6]*b[0] + a[0]*(b[3] - b[6]))/2.0);
        dd.mat.set(3,1, s*(a[3]*b[1] + a[6]*b[1] + a[1]*(b[3] + b[6]))/2.0);
        dd.mat.set(3,2, s*((a[5] + a[8])*(b[4] + b[7]) + (a[4] + a[7])*(b[5] + b[8]))/tsq2);
        dd.mat.set(3,3, s*(a[1]*b[0] + a[0]*b[1] + a[3]*b[3] - a[6]*b[6])/2.0);
        dd.mat.set(3,4, s*(SQRT_2*(a[5] + a[8])*b[1] + (a[4] + a[7])*(b[3] + b[6]) + (a[3] + a[6])*(b[4] + b[7]) + SQRT_2*a[1]*(b[5] + b[8]))/4.0);
        dd.mat.set(3,5, s*(SQRT_2*(a[4] + a[7])*b[0] + (a[5] + a[8])*(b[3] - b[6]) + SQRT_2*a[0]*(b[4] + b[7]) + (a[3] - a[6])*(b[5] + b[8]))/4.0);
        dd.mat.set(3,6, s*(a[1]*b[0] - a[0]*b[1] + a[6]*b[3] - a[3]*b[6])/2.0);
        dd.mat.set(3,7, s*(SQRT_2*(a[5] + a[8])*b[1] + (a[4] + a[7])*(b[3] + b[6]) - (a[3] + a[6])*(b[4] + b[7]) - SQRT_2*a[1]*(b[5] + b[8]))/4.0);
        dd.mat.set(3,8, s*(SQRT_2*(a[4] + a[7])*b[0] + (a[5] + a[8])*(b[3] - b[6]) - SQRT_2*a[0]*(b[4] + b[7]) - (a[3] - a[6])*(b[5] + b[8]))/4.0);

        dd.mat.set(4,0, s*((a[5] - a[8])*(b[3] - b[6]) + (a[3] - a[6])*(b[5] - b[8]))/tsq2);
        dd.mat.set(4,1, s*(a[4]*b[1] - a[7]*b[1] + a[1]*(b[4] - b[7]))/2.0);
        dd.mat.set(4,2, s*(a[4]*b[2] + a[7]*b[2] + a[2]*(b[4] + b[7]))/2.0);
        dd.mat.set(4,3, s*(SQRT_2*(a[5] - a[8])*b[1] + (a[4] - a[7])*(b[3] - b[6]) + (a[3] - a[6])*(b[4] - b[7]) + SQRT_2*a[1]*(b[5] - b[8]))/4.0);
        dd.mat.set(4,4, s*(a[2]*b[1] + a[1]*b[2] + a[4]*b[4] - a[7]*b[7])/2.0);
        dd.mat.set(4,5, s*(SQRT_2*(a[3] - a[6])*b[2] + SQRT_2*a[2]*(b[3] - b[6]) + (a[5] - a[8])*(b[4] + b[7]) + (a[4] + a[7])*(b[5] - b[8]))/4.0);
        dd.mat.set(4,6, s*(-(SQRT_2*(a[5] - a[8])*b[1]) + (a[4] - a[7])*(b[3] - b[6]) - (a[3] - a[6])*(b[4] - b[7]) + SQRT_2*a[1]*(b[5] - b[8]))/4.0);
        dd.mat.set(4,7, s*(a[2]*b[1] - a[1]*b[2] + a[7]*b[4] - a[4]*b[7])/2.0);
        dd.mat.set(4,8, s*(-(SQRT_2*(a[3] - a[6])*b[2]) + SQRT_2*a[2]*(b[3] - b[6]) - (a[5] - a[8])*(b[4] + b[7]) + (a[4] + a[7])*(b[5] - b[8]))/4.0);

        dd.mat.set(5,0, s*(a[5]*b[0] - a[8]*b[0] + a[0]*(b[5] - b[8]))/2.0);
        dd.mat.set(5,1, s*((a[4] - a[7])*(b[3] + b[6]) + (a[3] + a[6])*(b[4] - b[7]))/tsq2);
        dd.mat.set(5,2, s*(a[5]*b[2] + a[8]*b[2] + a[2]*(b[5] + b[8]))/2.0);
        dd.mat.set(5,3, s*(SQRT_2*(a[4] - a[7])*b[0] + (a[5] - a[8])*(b[3] + b[6]) + SQRT_2*a[0]*(b[4] - b[7]) + (a[3] + a[6])*(b[5] - b[8]))/4.0);
        dd.mat.set(5,4, s*(SQRT_2*(a[3] + a[6])*b[2] + SQRT_2*a[2]*(b[3] + b[6]) + (a[5] + a[8])*(b[4] - b[7]) + (a[4] - a[7])*(b[5] + b[8]))/4.0);
        dd.mat.set(5,5, s*(a[2]*b[0] + a[0]*b[2] + a[5]*b[5] - a[8]*b[8])/2.0);
        dd.mat.set(5,6, s*(SQRT_2*(a[4] - a[7])*b[0] - (a[5] - a[8])*(b[3] + b[6]) - SQRT_2*a[0]*(b[4] - b[7]) + (a[3] + a[6])*(b[5] - b[8]))/4.0);
        dd.mat.set(5,7, s*(-(SQRT_2*(a[3] + a[6])*b[2]) + SQRT_2*a[2]*(b[3] + b[6]) + (a[5] + a[8])*(b[4] - b[7]) - (a[4] - a[7])*(b[5] + b[8]))/4.0);
        dd.mat.set(5,8, s*(a[2]*b[0] - a[0]*b[2] + a[8]*b[5] - a[5]*b[8])/2.0);

        dd.mat.set(6,0, s*(-(a[3]*b[0]) + a[6]*b[0] + a[0]*(b[3] - b[6]))/2.0);
        dd.mat.set(6,1, s*(a[3]*b[1] + a[6]*b[1] - a[1]*(b[3] + b[6]))/2.0);
        dd.mat.set(6,2, s*((a[5] + a[8])*(b[4] + b[7]) - (a[4] + a[7])*(b[5] + b[8]))/tsq2);
        dd.mat.set(6,3, s*(-(a[1]*b[0]) + a[0]*b[1] + a[6]*b[3] - a[3]*b[6])/2.0);
        dd.mat.set(6,4, s*(SQRT_2*(a[5] + a[8])*b[1] - (a[4] + a[7])*(b[3] + b[6]) + (a[3] + a[6])*(b[4] + b[7]) - SQRT_2*a[1]*(b[5] + b[8]))/4.0);
        dd.mat.set(6,5, s*(-(SQRT_2*(a[4] + a[7])*b[0]) + (a[5] + a[8])*(b[3] - b[6]) + SQRT_2*a[0]*(b[4] + b[7]) - (a[3] - a[6])*(b[5] + b[8]))/4.0);
        dd.mat.set(6,6, s*(-(a[1]*b[0]) - a[0]*b[1] + a[3]*b[3] - a[6]*b[6])/2.0);
        dd.mat.set(6,7, s*(SQRT_2*(a[5] + a[8])*b[1] - (a[4] + a[7])*(b[3] + b[6]) - (a[3] + a[6])*(b[4] + b[7]) + SQRT_2*a[1]*(b[5] + b[8]))/4.0);
        dd.mat.set(6,8, s*(-(SQRT_2*(a[4] + a[7])*b[0]) + (a[5] + a[8])*(b[3] - b[6]) - SQRT_2*a[0]*(b[4] + b[7]) + (a[3] - a[6])*(b[5] + b[8]))/4.0);

        dd.mat.set(7,0, s*(-((a[5] - a[8])*(b[3] - b[6])) + (a[3] - a[6])*(b[5] - b[8]))/tsq2);
        dd.mat.set(7,1, s*(-(a[4]*b[1]) + a[7]*b[1] + a[1]*(b[4] - b[7]))/2.0);
        dd.mat.set(7,2, s*(a[4]*b[2] + a[7]*b[2] - a[2]*(b[4] + b[7]))/2.0);
        dd.mat.set(7,3, s*(-(SQRT_2*(a[5] - a[8])*b[1]) - (a[4] - a[7])*(b[3] - b[6]) + (a[3] - a[6])*(b[4] - b[7]) + SQRT_2*a[1]*(b[5] - b[8]))/4.0);
        dd.mat.set(7,4, s*(-(a[2]*b[1]) + a[1]*b[2] + a[7]*b[4] - a[4]*b[7])/2.0);
        dd.mat.set(7,5, s*(SQRT_2*(a[3] - a[6])*b[2] - SQRT_2*a[2]*(b[3] - b[6]) - (a[5] - a[8])*(b[4] + b[7]) + (a[4] + a[7])*(b[5] - b[8]))/4.0);
        dd.mat.set(7,6, s*(SQRT_2*(a[5] - a[8])*b[1] - (a[4] - a[7])*(b[3] - b[6]) - (a[3] - a[6])*(b[4] - b[7]) + SQRT_2*a[1]*(b[5] - b[8]))/4.0);
        dd.mat.set(7,7, s*(-(a[2]*b[1]) - a[1]*b[2] + a[4]*b[4] - a[7]*b[7])/2.0);
        dd.mat.set(7,8, s*(-(SQRT_2*(a[3] - a[6])*b[2]) - SQRT_2*a[2]*(b[3] - b[6]) + (a[5] - a[8])*(b[4] + b[7]) + (a[4] + a[7])*(b[5] - b[8]))/4.0);

        dd.mat.set(8,0, s*(-(a[5]*b[0]) + a[8]*b[0] + a[0]*(b[5] - b[8]))/2.0);
        dd.mat.set(8,1, s*(-((a[4] - a[7])*(b[3] + b[6])) + (a[3] + a[6])*(b[4] - b[7]))/tsq2);
        dd.mat.set(8,2, s*(a[5]*b[2] + a[8]*b[2] - a[2]*(b[5] + b[8]))/2.0);
        dd.mat.set(8,3, s*(-(SQRT_2*(a[4] - a[7])*b[0]) - (a[5] - a[8])*(b[3] + b[6]) + SQRT_2*a[0]*(b[4] - b[7]) + (a[3] + a[6])*(b[5] - b[8]))/4.0);
        dd.mat.set(8,4, s*(SQRT_2*(a[3] + a[6])*b[2] - SQRT_2*a[2]*(b[3] + b[6]) + (a[5] + a[8])*(b[4] - b[7]) - (a[4] - a[7])*(b[5] + b[8]))/4.0);
        dd.mat.set(8,5, s*(-(a[2]*b[0]) + a[0]*b[2] + a[8]*b[5] - a[5]*b[8])/2.0);
        dd.mat.set(8,6, s*(-(SQRT_2*(a[4] - a[7])*b[0]) + (a[5] - a[8])*(b[3] + b[6]) - SQRT_2*a[0]*(b[4] - b[7]) + (a[3] + a[6])*(b[5] - b[8]))/4.0);
        dd.mat.set(8,7, s*(-(SQRT_2*(a[3] + a[6])*b[2]) - SQRT_2*a[2]*(b[3] + b[6]) + (a[5] + a[8])*(b[4] - b[7]) + (a[4] - a[7])*(b[5] + b[8]))/4.0);
        dd.mat.set(8,8, s*(-(a[2]*b[0]) - a[0]*b[2] + a[5]*b[5] - a[8]*b[8])/2.0);
    }
}

/// Performs the self-sum-dyadic (ssd) operation with a Tensor2 yielding a minor-symmetric Tensor4
///
/// Computes:
///
/// ```text
///          _
/// D = s (A ⊗ A + A ⊗ A)
///                  ‾
/// ```
/// 
/// With orthonormal Cartesian components:
///
/// ```text
/// Dᵢⱼₖₗ = s (Aᵢₖ Aⱼₗ + Aᵢₗ Aⱼₖ)
/// ```
/// 
/// **Important:** Even if `A` is Symmetric 2D, the result may not be expressed by a Symmetric 2D Tensor4.
///
/// # Output
///
/// * `dd` -- The resulting tensor (minor-symmetric); it must be [Mandel::Symmetric]
///
/// # Input
///
/// * `aa` -- Second-order tensor, symmetric or not.
/// 
/// # Panics
/// 
/// A panic will occur if `dd` is not [Mandel::Symmetric]
#[rustfmt::skip]
pub fn t2_ssd(dd: &mut Tensor4, s: f64, aa: &Tensor2) {
    assert_eq!(dd.mandel, Mandel::Symmetric);
    let dim = aa.vec.dim();
    let a = &aa.vec;
    if dim == 4 {
        dd.mat.set(0,0, s*(2.0*a[0]*a[0]));
        dd.mat.set(0,1, s*(a[3]*a[3]));
        dd.mat.set(0,2, 0.0);
        dd.mat.set(0,3, s*(2.0*a[0]*a[3]));
        dd.mat.set(0,4, 0.0);
        dd.mat.set(0,5, 0.0);

        dd.mat.set(1,0, s*(a[3]*a[3]));
        dd.mat.set(1,1, s*(2.0*a[1]*a[1]));
        dd.mat.set(1,2, 0.0);
        dd.mat.set(1,3, s*(2.0*a[1]*a[3]));
        dd.mat.set(1,4, 0.0);
        dd.mat.set(1,5, 0.0);

        dd.mat.set(2,0, 0.0);
        dd.mat.set(2,1, 0.0);
        dd.mat.set(2,2, s*(2.0*a[2]*a[2]));
        dd.mat.set(2,3, 0.0);
        dd.mat.set(2,4, 0.0);
        dd.mat.set(2,5, 0.0);

        dd.mat.set(3,0, s*(2.0*a[0]*a[3]));
        dd.mat.set(3,1, s*(2.0*a[1]*a[3]));
        dd.mat.set(3,2, 0.0);
        dd.mat.set(3,3, s*(2.0*a[0]*a[1] + a[3]*a[3]));
        dd.mat.set(3,4, 0.0);
        dd.mat.set(3,5, 0.0);

        dd.mat.set(4,0, 0.0);
        dd.mat.set(4,1, 0.0);
        dd.mat.set(4,2, 0.0);
        dd.mat.set(4,3, 0.0);
        dd.mat.set(4,4, s*(2.0*a[1]*a[2]));
        dd.mat.set(4,5, s*(SQRT_2*a[2]*a[3]));

        dd.mat.set(5,0, 0.0);
        dd.mat.set(5,1, 0.0);
        dd.mat.set(5,2, 0.0);
        dd.mat.set(5,3, 0.0);
        dd.mat.set(5,4, s*(SQRT_2*a[2]*a[3]));
        dd.mat.set(5,5, s*(2.0*a[0]*a[2]));
    } else if dim == 6 {
        dd.mat.set(0,0, s*(2.0*a[0]*a[0]));
        dd.mat.set(0,1, s*(a[3]*a[3]));
        dd.mat.set(0,2, s*(a[5]*a[5]));
        dd.mat.set(0,3, s*(2.0*a[0]*a[3]));
        dd.mat.set(0,4, s*(SQRT_2*a[3]*a[5]));
        dd.mat.set(0,5, s*(2.0*a[ 0]*a[5]));

        dd.mat.set(1,0, s*(a[3]*a[3]));
        dd.mat.set(1,1, s*(2.0*a[1]*a[1]));
        dd.mat.set(1,2, s*(a[4]*a[4]));
        dd.mat.set(1,3, s*(2.0*a[1]*a[3]));
        dd.mat.set(1,4, s*(2.0*a[1]*a[4]));
        dd.mat.set(1,5, s*(SQRT_2*a[3]*a[4]));

        dd.mat.set(2,0, s*(a[5]*a[5]));
        dd.mat.set(2,1, s*(a[4]*a[4]));
        dd.mat.set(2,2, s*(2.0*a[2]*a[2]));
        dd.mat.set(2,3, s*(SQRT_2*a[4]*a[ 5]));
        dd.mat.set(2,4, s*(2.0*a[2]*a[4]));
        dd.mat.set(2,5, s*(2.0*a[2]*a[5]));

        dd.mat.set(3,0, s*(2.0*a[0]*a[3]));
        dd.mat.set(3,1, s*(2.0*a[1]*a[3]));
        dd.mat.set(3,2, s*(SQRT_2*a[4]* a[5]));
        dd.mat.set(3,3, s*(2.0*a[0]*a[1] + a[3]*a[3]));
        dd.mat.set(3,4, s*(a[3]*a[4] + SQRT_2*a[1]*a[5]));
        dd.mat.set(3,5, s*(SQRT_2*a[0]*a[4] + a[3]*a[5]));

        dd.mat.set(4,0, s*(SQRT_2*a[3]*a[5]));
        dd.mat.set(4,1, s*(2.0*a[1]*a[4]));
        dd.mat.set(4,2, s*(2.0*a[2]*a[4]));
        dd.mat.set(4,3, s*(a[3]*a[4] + SQRT_2*a[1]*a[5]));
        dd.mat.set(4,4, s*(2.0*a[1]*a[2] + a[4]*a[4]));
        dd.mat.set(4,5, s*(SQRT_2*a[2]*a[3] + a[4]*a[5]));

        dd.mat.set(5,0, s*(2.0*a[0]*a[5]));
        dd.mat.set(5,1, s*(SQRT_2*a[3]*a[4]));
        dd.mat.set(5,2, s*(2.0*a[2]*a[5]));
        dd.mat.set(5,3, s*(SQRT_2*a[0]* a[4] + a[3]*a[5]));
        dd.mat.set(5,4, s*(SQRT_2*a[2]*a[3] + a[4]*a[5]));
        dd.mat.set(5,5, s*(2.0*a[0]*a[2] + a[5]*a[5]));
    } else {
        dd.mat.set(0,0, s*(2.0*a[0]*a[0]));
        dd.mat.set(0,1, s*((a[3] + a[6])*(a[3] + a[6])));
        dd.mat.set(0,2, s*((a[5] + a[8])*(a[5] + a[8])));
        dd.mat.set(0,3, s*(2.0*a[0]*(a[3] + a[6])));
        dd.mat.set(0,4, s*(SQRT_2*(a[3] + a[6])*(a[5] + a[8])));
        dd.mat.set(0,5, s*(2.0*a[0]*(a[5] + a[8])));

        dd.mat.set(1,0, s*((a[3] - a[6])*(a[3] - a[6])));
        dd.mat.set(1,1, s*(2.0*a[1]*a[1]));
        dd.mat.set(1,2, s*((a[4] + a[7])*(a[4] + a[7])));
        dd.mat.set(1,3, s*(2.0*a[1]*(a[3] - a[6])));
        dd.mat.set(1,4, s*(2.0*a[1]*(a[4] + a[7])));
        dd.mat.set(1,5, s*(SQRT_2*(a[3] - a[6])*(a[4] + a[7])));

        dd.mat.set(2,0, s*((a[5] - a[8])*(a[5] - a[8])));
        dd.mat.set(2,1, s*((a[4] - a[7])*(a[4] - a[7])));
        dd.mat.set(2,2, s*(2.0*a[2]*a[2]));
        dd.mat.set(2,3, s*(SQRT_2*(a[4] - a[7])*(a[5] - a[8])));
        dd.mat.set(2,4, s*(2.0*a[2]*(a[4] - a[7])));
        dd.mat.set(2,5, s*(2.0*a[2]*(a[5] - a[8])));

        dd.mat.set(3,0, s*(2.0*a[0]*(a[3] - a[6])));
        dd.mat.set(3,1, s*(2.0*a[1]*(a[3] + a[6])));
        dd.mat.set(3,2, s*(SQRT_2*(a[4] + a[7])*(a[5] + a[8])));
        dd.mat.set(3,3, s*(2.0*a[0]*a[1] + a[3]*a[3] - a[6]*a[6]));
        dd.mat.set(3,4, s*((a[3] + a[6])*(a[4] + a[7]) + SQRT_2*a[1]*(a[5] + a[8])));
        dd.mat.set(3,5, s*(SQRT_2*a[0]*(a[4] + a[7]) + (a[3] - a[6])*(a[5] + a[8])));

        dd.mat.set(4,0, s*(SQRT_2*(a[3] - a[6])*(a[5] - a[8])));
        dd.mat.set(4,1, s*(2.0*a[1]*(a[4] - a[7])));
        dd.mat.set(4,2, s*(2.0*a[2]*(a[4] + a[7])));
        dd.mat.set(4,3, s*((a[3] - a[6])*(a[4] - a[7]) + SQRT_2*a[1]*(a[5] - a[8])));
        dd.mat.set(4,4, s*(2.0*a[1]*a[2] + a[4]*a[4] - a[7]*a[7]));
        dd.mat.set(4,5, s*(SQRT_2*a[2]*(a[3] - a[6]) + (a[4] + a[7])*(a[5] - a[8])));

        dd.mat.set(5,0, s*(2.0*a[0]*(a[5] - a[8])));
        dd.mat.set(5,1, s*(SQRT_2*(a[3] + a[6])*(a[4] - a[7])));
        dd.mat.set(5,2, s*(2.0*a[2]*(a[5] + a[8])));
        dd.mat.set(5,3, s*(SQRT_2*a[0]*(a[4] - a[7]) + (a[3] + a[6])*(a[5] - a[8])));
        dd.mat.set(5,4, s*(SQRT_2*a[2]*(a[3] + a[6]) + (a[4] - a[7])*(a[5] + a[8])));
        dd.mat.set(5,5, s*(2.0*a[0]*a[2] + a[5]*a[5] - a[8]*a[8]));
    }
}

/// Performs the quad-sum-dyadic (qsd) operation with two Tensor2 yielding a minor-symmetric Tensor4
///
/// Computes:
///
/// ```text
///          _               _
/// D = s (A ⊗ B + A ⊗ B + B ⊗ A + B ⊗ A)
///                  ‾               ‾
/// ```
/// 
/// With orthonormal Cartesian components:
///
/// ```text
/// Dᵢⱼₖₗ = s (Aᵢₖ Bⱼₗ + Aᵢₗ Bⱼₖ + Bᵢₖ Aⱼₗ + Bᵢₗ Aⱼₖ)
/// ```
///
/// **Important:** Even if `A` and `B` are Symmetric 2D, the result may not be expressed by a Symmetric 2D Tensor4.
///
/// # Output
///
/// * `dd` -- The resulting tensor (minor-symmetric); it must be [Mandel::Symmetric]
///
/// # Input
///
/// * `aa` -- Second-order tensor, symmetric or not; with the same [Mandel] as `bb`
/// * `bb` -- Second-order tensor, symmetric or not; with the same [Mandel] as `aa`
/// 
/// # Panics
/// 
/// 1. A panic will occur if `dd` is not [Mandel::Symmetric]
/// 2. A panic will occur `aa` and `bb` have different [Mandel]
#[rustfmt::skip]
pub fn t2_qsd_t2(dd: &mut Tensor4, s: f64, aa: &Tensor2, bb: &Tensor2) {
    assert_eq!(dd.mandel, Mandel::Symmetric);
    assert_eq!(bb.mandel, aa.mandel);
    let dim = aa.vec.dim();
    let a = &aa.vec;
    let b = &bb.vec;
    if dim == 4 {
        dd.mat.set(0,0, s*(4.0*a[0]*b[0]));
        dd.mat.set(0,1, s*(2.0*a[3]*b[3]));
        dd.mat.set(0,2, 0.0);
        dd.mat.set(0,3, s*(2.0*(a[3]*b[0] + a[0]*b[3])));
        dd.mat.set(0,4, 0.0);
        dd.mat.set(0,5, 0.0);

        dd.mat.set(1,0, s*(2.0*a[3]*b[3]));
        dd.mat.set(1,1, s*(4.0*a[1]*b[1]));
        dd.mat.set(1,2, 0.0);
        dd.mat.set(1,3, s*(2.0*(a[3]*b[1] + a[1]*b[3])));
        dd.mat.set(1,4, 0.0);
        dd.mat.set(1,5, 0.0);

        dd.mat.set(2,0, 0.0);
        dd.mat.set(2,1, 0.0);
        dd.mat.set(2,2, s*(4.0*a[2]*b[2]));
        dd.mat.set(2,3, 0.0);
        dd.mat.set(2,4, 0.0);
        dd.mat.set(2,5, 0.0);

        dd.mat.set(3,0, s*(2.0*(a[3]*b[0] + a[0]*b[3])));
        dd.mat.set(3,1, s*(2.0*(a[3]*b[1] + a[1]*b[3])));
        dd.mat.set(3,2, 0.0);
        dd.mat.set(3,3, s*(2.0*(a[1]*b[0] + a[0]*b[1] + a[3]*b[3])));
        dd.mat.set(3,4, 0.0);
        dd.mat.set(3,5, 0.0);

        dd.mat.set(4,0, 0.0);
        dd.mat.set(4,1, 0.0);
        dd.mat.set(4,2, 0.0);
        dd.mat.set(4,3, 0.0);
        dd.mat.set(4,4, s*(2.0*(a[2]*b[1] + a[1]*b[2])));
        dd.mat.set(4,5, s*(SQRT_2*(a[3]*b[2] + a[2]*b[3])));

        dd.mat.set(5,0, 0.0);
        dd.mat.set(5,1, 0.0);
        dd.mat.set(5,2, 0.0);
        dd.mat.set(5,3, 0.0);
        dd.mat.set(5,4, s*(SQRT_2*(a[3]*b[2] + a[2]*b[3])));
        dd.mat.set(5,5, s*(2.0*(a[2]*b[0] + a[0]*b[2])));
    } else if dim == 6 {
        dd.mat.set(0,0, s*(4.0*a[0]*b[0]));
        dd.mat.set(0,1, s*(2.0*a[3]*b[3]));
        dd.mat.set(0,2, s*(2.0*a[5]*b[5]));
        dd.mat.set(0,3, s*(2.0*(a[3]*b[0] + a[0]*b[3])));
        dd.mat.set(0,4, s*(SQRT_2*(a[5]*b[3] + a[3]*b[5])));
        dd.mat.set(0,5, s*(2.0*(a[5]*b[0] + a[0]*b[5])));

        dd.mat.set(1,0, s*(2.0*a[3]*b[3]));
        dd.mat.set(1,1, s*(4.0*a[1]*b[1]));
        dd.mat.set(1,2, s*(2.0*a[4]*b[4]));
        dd.mat.set(1,3, s*(2.0*(a[3]*b[1] + a[1]*b[3])));
        dd.mat.set(1,4, s*(2.0*(a[4]*b[1] + a[1]*b[4])));
        dd.mat.set(1,5, s*(SQRT_2*(a[4]*b[3] + a[3]*b[4])));

        dd.mat.set(2,0, s*(2.0*a[5]*b[5]));
        dd.mat.set(2,1, s*(2.0*a[4]*b[4]));
        dd.mat.set(2,2, s*(4.0*a[2]*b[2]));
        dd.mat.set(2,3, s*(SQRT_2*(a[5]*b[4] + a[4]*b[5])));
        dd.mat.set(2,4, s*(2.0*(a[4]*b[2] + a[2]*b[4])));
        dd.mat.set(2,5, s*(2.0*(a[5]*b[2] + a[2]*b[5])));

        dd.mat.set(3,0, s*(2.0*(a[3]*b[0] + a[0]*b[3])));
        dd.mat.set(3,1, s*(2.0*(a[3]*b[1] + a[1]*b[3])));
        dd.mat.set(3,2, s*(SQRT_2*(a[5]*b[4] + a[4]*b[5])));
        dd.mat.set(3,3, s*(2.0*(a[1]*b[0] + a[0]*b[1] + a[3]*b[3])));
        dd.mat.set(3,4, s*(SQRT_2*a[5]*b[1] + a[4]*b[3] + a[3]*b[4] + SQRT_2*a[1]*b[5]));
        dd.mat.set(3,5, s*(SQRT_2*a[4]*b[0] + a[5]*b[3] + SQRT_2*a[0]*b[4] + a[3]*b[5]));

        dd.mat.set(4,0, s*(SQRT_2*(a[5]*b[3] + a[3]*b[5])));
        dd.mat.set(4,1, s*(2.0*(a[4]*b[1] + a[1]*b[4])));
        dd.mat.set(4,2, s*(2.0*(a[4]*b[2] + a[2]*b[4])));
        dd.mat.set(4,3, s*(SQRT_2*a[5]*b[1] + a[4]*b[3] + a[3]*b[4] + SQRT_2*a[1]*b[5]));
        dd.mat.set(4,4, s*(2.0*(a[2]*b[1] + a[1]*b[2] + a[4]*b[4])));
        dd.mat.set(4,5, s*(SQRT_2*a[3]*b[2] + SQRT_2*a[2]*b[3] + a[5]*b[4] + a[4]*b[5]));

        dd.mat.set(5,0, s*(2.0*(a[5]*b[0] + a[0]*b[5])));
        dd.mat.set(5,1, s*(SQRT_2*(a[4]*b[3] + a[3]*b[4])));
        dd.mat.set(5,2, s*(2.0*(a[5]*b[2] + a[2]*b[5])));
        dd.mat.set(5,3, s*(SQRT_2*a[4]*b[0] + a[5]*b[3] + SQRT_2*a[0]*b[4] + a[3]*b[5]));
        dd.mat.set(5,4, s*(SQRT_2*a[3]*b[2] + SQRT_2*a[2]*b[3] + a[5]*b[4] + a[4]*b[5]));
        dd.mat.set(5,5, s*(2.0*(a[2]*b[0] + a[0]*b[2] + a[5]*b[5])));
    } else {
        dd.mat.set(0,0, s*(4.0*a[0]*b[0]));
        dd.mat.set(0,1, s*(2.0*(a[3] + a[6])*(b[3] + b[6])));
        dd.mat.set(0,2, s*(2.0*(a[5] + a[8])*(b[5] + b[8])));
        dd.mat.set(0,3, s*(2.0*(a[3]*b[0] + a[6]*b[0] + a[0]*(b[3] + b[6]))));
        dd.mat.set(0,4, s*(SQRT_2*((a[5] + a[8])*(b[3] + b[6]) + (a[3] + a[6])*(b[5] + b[8]))));
        dd.mat.set(0,5, s*(2.0*(a[5]*b[0] + a[8]*b[0] + a[0]*(b[5] + b[8]))));

        dd.mat.set(1,0, s*(2.0*(a[3] - a[6])*(b[3] - b[6])));
        dd.mat.set(1,1, s*(4.0*a[1]*b[1]));
        dd.mat.set(1,2, s*(2.0*(a[4] + a[7])*(b[4] + b[7])));
        dd.mat.set(1,3, s*(2.0*(a[3]*b[1] - a[6]*b[1] + a[1]*(b[3] - b[6]))));
        dd.mat.set(1,4, s*(2.0*(a[4]*b[1] + a[7]*b[1] + a[1]*(b[4] + b[7]))));
        dd.mat.set(1,5, s*(SQRT_2*((a[4] + a[7])*(b[3] - b[6]) + (a[3] - a[6])*(b[4] + b[7]))));

        dd.mat.set(2,0, s*(2.0*(a[5] - a[8])*(b[5] - b[8])));
        dd.mat.set(2,1, s*(2.0*(a[4] - a[7])*(b[4] - b[7])));
        dd.mat.set(2,2, s*(4.0*a[2]*b[2]));
        dd.mat.set(2,3, s*(SQRT_2*((a[5] - a[8])*(b[4] - b[7]) + (a[4] - a[7])*(b[5] - b[8]))));
        dd.mat.set(2,4, s*(2.0*(a[4]*b[2] - a[7]*b[2] + a[2]*(b[4] - b[7]))));
        dd.mat.set(2,5, s*(2.0*(a[5]*b[2] - a[8]*b[2] + a[2]*(b[5] - b[8]))));

        dd.mat.set(3,0, s*(2.0*(a[3]*b[0] - a[6]*b[0] + a[0]*(b[3] - b[6]))));
        dd.mat.set(3,1, s*(2.0*(a[3]*b[1] + a[6]*b[1] + a[1]*(b[3] + b[6]))));
        dd.mat.set(3,2, s*(SQRT_2*((a[5] + a[8])*(b[4] + b[7]) + (a[4] + a[7])*(b[5] + b[8]))));
        dd.mat.set(3,3, s*(2.0*(a[1]*b[0] + a[0]*b[1] + a[3]*b[3] - a[6]*b[6])));
        dd.mat.set(3,4, s*(SQRT_2*(a[5] + a[8])*b[1] + (a[4] + a[7])*(b[3] + b[6]) + (a[3] + a[6])*(b[4] + b[7]) + SQRT_2*a[1]*(b[5] + b[8])));
        dd.mat.set(3,5, s*(SQRT_2*(a[4] + a[7])*b[0] + (a[5] + a[8])*(b[3] - b[6]) + SQRT_2*a[0]*(b[4] + b[7]) + (a[3] - a[6])*(b[5] + b[8])));

        dd.mat.set(4,0, s*(SQRT_2*((a[5] - a[8])*(b[3] - b[6]) + (a[3] - a[6])*(b[5] - b[8]))));
        dd.mat.set(4,1, s*(2.0*(a[4]*b[1] - a[7]*b[1] + a[1]*(b[4] - b[7]))));
        dd.mat.set(4,2, s*(2.0*(a[4]*b[2] + a[7]*b[2] + a[2]*(b[4] + b[7]))));
        dd.mat.set(4,3, s*(SQRT_2*(a[5] - a[8])*b[1] + (a[4] - a[7])*(b[3] - b[6]) + (a[3] - a[6])*(b[4] - b[7]) + SQRT_2*a[1]*(b[5] - b[8])));
        dd.mat.set(4,4, s*(2.0*(a[2]*b[1] + a[1]*b[2] + a[4]*b[4] - a[7]*b[7])));
        dd.mat.set(4,5, s*(SQRT_2*(a[3] - a[6])*b[2] + SQRT_2*a[2]*(b[3] - b[6]) + (a[5] - a[8])*(b[4] + b[7]) + (a[4] + a[7])*(b[5] - b[8])));

        dd.mat.set(5,0, s*(2.0*(a[5]*b[0] - a[8]*b[0] + a[0]*(b[5] - b[8]))));
        dd.mat.set(5,1, s*(SQRT_2*((a[4] - a[7])*(b[3] + b[6]) + (a[3] + a[6])*(b[4] - b[7]))));
        dd.mat.set(5,2, s*(2.0*(a[5]*b[2] + a[8]*b[2] + a[2]*(b[5] + b[8]))));
        dd.mat.set(5,3, s*(SQRT_2*(a[4] - a[7])*b[0] + (a[5] - a[8])*(b[3] + b[6]) + SQRT_2*a[0]*(b[4] - b[7]) + (a[3] + a[6])*(b[5] - b[8])));
        dd.mat.set(5,4, s*(SQRT_2*(a[3] + a[6])*b[2] + SQRT_2*a[2]*(b[3] + b[6]) + (a[5] + a[8])*(b[4] - b[7]) + (a[4] - a[7])*(b[5] + b[8])));
        dd.mat.set(5,5, s*(2.0*(a[2]*b[0] + a[0]*b[2] + a[5]*b[5] - a[8]*b[8])));
    }
}

/// Performs the double-dot (ddot) operation between a Tensor4 and a Tensor2
///
/// ```text
/// b = α D : a
/// ```
///
/// With orthonormal Cartesian components:
///
/// ```text
/// bᵢⱼ = α Σ Σ Dᵢⱼₖₗ aₖₗ
///         k l
/// ```
///
/// Or, in Mandel basis:
///
/// ```text
/// bₘ = α Σ Dₘₙ aₙ
///        n
/// ```
///
/// # Output
///
/// * `b` -- the resulting second-order tensor; with the same [Mandel] as `a` and `dd`
///
/// # Input
///
/// * `alpha` -- the scalar multiplier
/// * `dd` -- the fourth-order tensor; with the same [Mandel] as `a` and `b`
/// * `a` -- the input second-order tensor; with the same [Mandel] as `b` and `dd`
///
/// # Panics
///
/// A panic will occur the tensors have different [Mandel]
///
/// # Examples
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
///     t4_ddot_t2(&mut b, 1.0, &dd, &a);
///
///     assert_eq!(
///         format!("{:.1}", b.as_matrix()),
///         "┌                      ┐\n\
///          │  285.0 2850.0    0.0 │\n\
///          │ -570.0 -285.0    0.0 │\n\
///          │    0.0    0.0  570.0 │\n\
///          └                      ┘"
///     );
///     Ok(())
/// }
/// ```
pub fn t4_ddot_t2(b: &mut Tensor2, alpha: f64, dd: &Tensor4, a: &Tensor2) {
    assert_eq!(a.mandel, dd.mandel);
    assert_eq!(b.mandel, dd.mandel);
    mat_vec_mul(&mut b.vec, alpha, &dd.mat, &a.vec).unwrap();
}

/// Performs the double-dot (ddot) operation between a Tensor4 and a Tensor2 with update
///
/// Computes:
///
/// ```text
/// b = α D : a + β b
/// ```
///
/// With orthonormal Cartesian components:
///
/// ```text
/// bᵢⱼ = α Σ Σ Dᵢⱼₖₗ aₖₗ + β bᵢⱼ
///         k l
/// ```
///
/// Or, in Mandel basis:
///
/// ```text
/// bₘ = α Σ Dₘₙ aₙ + β bₘ
///        n
/// ```
///
/// # Output
///
/// * `b` -- the resulting second-order tensor; with the same [Mandel] as `a` and `dd`
///
/// # Input
///
/// * `alpha` -- the scalar multiplier
/// * `a` -- the input second-order tensor; with the same [Mandel] as `b` and `dd`
/// * `dd` -- the fourth-order tensor; with the same [Mandel] as `a` and `b`
/// * `beta` -- the other scalar multiplier
///
/// # Panics
///
/// A panic will occur the tensors have different [Mandel]
///
/// # Examples
///
/// ```
/// use russell_tensor::{t4_ddot_t2_update, Mandel, Tensor2, Tensor4, StrError};
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
///     let mut b = Tensor2::from_matrix(&[
///         [1.0, 0.0, 0.0],
///         [0.0, 1.0, 0.0],
///         [0.0, 0.0, 1.0],
///     ], Mandel::General)?;
///     t4_ddot_t2_update(&mut b, 1.0, &dd, &a, 1000.0);
///
///     assert_eq!(
///         format!("{:.1}", b.as_matrix()),
///         "┌                      ┐\n\
///          │ 1285.0 2850.0    0.0 │\n\
///          │ -570.0  715.0    0.0 │\n\
///          │    0.0    0.0 1570.0 │\n\
///          └                      ┘"
///     );
///     Ok(())
/// }
/// ```
pub fn t4_ddot_t2_update(b: &mut Tensor2, alpha: f64, dd: &Tensor4, a: &Tensor2, beta: f64) {
    assert_eq!(a.mandel, dd.mandel);
    assert_eq!(b.mandel, dd.mandel);
    mat_vec_mul_update(&mut b.vec, alpha, &dd.mat, &a.vec, beta).unwrap();
}

/// Performs the double-dot (ddot) operation between a Tensor2 and a Tensor4
///
/// Computes:
///
/// ```text
/// b = α a : D
/// ```
///
/// With orthonormal Cartesian components:
///
/// ```text
/// bₖₗ = α Σ Σ aᵢⱼ Dᵢⱼₖₗ
///         i j
/// ```
///
/// # Output
///
/// * `b` -- the resulting second-order tensor; with the same [Mandel] as `a` and `dd`
///
/// # Input
///
/// * `alpha` -- the scalar multiplier
/// * `a` -- the input second-order tensor; with the same [Mandel] as `b` and `dd`
/// * `dd` -- the fourth-order tensor; with the same [Mandel] as `a` and `b`
///
/// # Panics
///
/// A panic will occur the tensors have different [Mandel]
///
/// # Examples
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
///     t2_ddot_t4(&mut b, 1.0, &a, &dd);
///
///     assert_eq!(
///         format!("{:.1}", b.as_matrix()),
///         "┌                   ┐\n\
///          │  31.0 124.0 186.0 │\n\
///          │ 217.0  62.0 155.0 │\n\
///          │ 279.0 248.0  93.0 │\n\
///          └                   ┘"
///     );
///     Ok(())
/// }
/// ```
pub fn t2_ddot_t4(b: &mut Tensor2, alpha: f64, a: &Tensor2, dd: &Tensor4) {
    assert_eq!(a.mandel, dd.mandel);
    assert_eq!(b.mandel, dd.mandel);
    vec_mat_mul(&mut b.vec, alpha, &a.vec, &dd.mat).unwrap();
}

/// Performs the double-dot (ddot) operation between two Tensor4
///
/// Computes:
///
/// ```text
/// E = α C : D
/// ```
///
/// With orthonormal Cartesian components:
///
/// ```text
/// Eᵢⱼₖₗ = α Σ Σ Cᵢⱼₛₜ : Dₛₜₖₗ
///           s t
/// ```
///
/// Or, in Mandel basis:
///
/// ```text
/// Eₘₙ = α Σ Cₘₐ  Dₐₙ
///         m
/// ```
///
/// # Output
///
/// * `ee` -- the resulting fourth-order tensor; with the same [Mandel] as `cc` and `dd`
///
/// # Input
///
/// * `alpha` -- the scalar multiplier
/// * `a` -- the input second-order tensor; with the same [Mandel] as `b` and `dd`
/// * `dd` -- the fourth-order tensor; with the same [Mandel] as `a` and `b`
///
/// # Panics
///
/// A panic will occur the tensors have different [Mandel]
///
/// # Examples
///
/// ```
/// use russell_lab::approx_eq;
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
///     t4_ddot_t4(&mut ee, 1.0, &cc, &dd);
///
///     let out = ee.as_matrix();
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
pub fn t4_ddot_t4(ee: &mut Tensor4, alpha: f64, cc: &Tensor4, dd: &Tensor4) {
    assert_eq!(cc.mandel, dd.mandel);
    assert_eq!(ee.mandel, dd.mandel);
    mat_mat_mul(&mut ee.mat, alpha, &cc.mat, &dd.mat, 0.0).unwrap();
}

/// Performs the double-dot (ddot) operation between two Tensor4 with update
///
/// Computes:
///
/// ```text
/// E = α C : D + β E
/// ```
///
/// With orthonormal Cartesian components:
///
/// ```text
/// Eᵢⱼₖₗ = α (Σ Σ Cᵢⱼₛₜ : Dₛₜₖₗ) + β Eᵢⱼₖₗ
///            s t
/// ```
///
/// Or, in Mandel basis:
///
/// ```text
/// Eₘₙ = α (Σ Cₘₐ  Dₐₙ) + β Eₘₙ
///          m
/// ```
///
/// # Output
///
/// * `ee` -- the resulting fourth-order tensor; with the same [Mandel] as `cc` and `dd`
///
/// # Input
///
/// * `alpha` -- the scalar multiplier
/// * `a` -- the input second-order tensor; with the same [Mandel] as `b` and `dd`
/// * `dd` -- the fourth-order tensor; with the same [Mandel] as `a` and `b`
/// * `beta` -- the other scalar multiplier
///
/// # Panics
///
/// A panic will occur the tensors have different [Mandel]
pub fn t4_ddot_t4_update(ee: &mut Tensor4, alpha: f64, cc: &Tensor4, dd: &Tensor4, beta: f64) {
    assert_eq!(cc.mandel, dd.mandel);
    assert_eq!(ee.mandel, dd.mandel);
    mat_mat_mul(&mut ee.mat, alpha, &cc.mat, &dd.mat, beta).unwrap();
}

/// Computes Tensor2 double-dot Tensor4 double-dot Tensor2
///
/// Computes:
///
/// ```text
/// s = a : D : b
/// ```
///
/// With orthonormal Cartesian components:
///
/// ```text
/// s = Σ Σ Σ Σ aᵢⱼ Dᵢⱼₖₗ bₖₗ
///     i j k l
/// ```
///
/// Or, in Mandel basis:
///
/// ```text
/// s = Σ Σ aₘ Dₘₙ bₙ
///     m n
/// ```
///
/// Note: the Lagrange multiplier in Plasticity needs this operation.
///
/// # Input
///
/// * `a` -- the first second-order tensor; with the same [Mandel] as `b` and `dd`
/// * `dd` -- the fourth-order tensor; with the same [Mandel] as `a` and `b`
/// * `b` -- the second second-order tensor; with the same [Mandel] as `a` and `dd`
///
/// # Output
///
/// Returns the scalar results.
///
/// # Panics
///
/// A panic will occur the tensors have different [Mandel]
pub fn t2_ddot_t4_ddot_t2(a: &Tensor2, dd: &Tensor4, b: &Tensor2) -> f64 {
    assert_eq!(a.mandel, dd.mandel);
    assert_eq!(b.mandel, dd.mandel);
    let dim = a.vec.dim();
    let mut s = 0.0;
    for m in 0..dim {
        for n in 0..dim {
            s += a.vec[m] * dd.mat.get(m, n) * b.vec[n];
        }
    }
    s
}

/// Computes Tensor4 double-dot Tensor2 dyadic Tensor2 double-dot Tensor4
///
/// Computes:
///
/// ```text
/// E = α (D : a) ⊗ (b : D)
/// ```
///
/// With orthonormal Cartesian components:
///
/// ```text
/// Eᵢⱼₖₗ = α Σ Σ Σ Σ (Dᵢⱼₛₜ aₛₜ) (bₒₚ Dₒₚₖₗ)
///           s t o p
/// ```
///
/// Or, in Mandel basis:
///
/// ```text
/// Eₘₙ = α Σ Σ (Dₘₐ aₐ) (bₑ Dₑₙ)
///         a e
/// ```
///
/// Note: the elastoplastic modulus in Plasticity needs this operation.
///
/// # Output
///
/// * `ee` -- the resulting fourth-order tensor; with the same [Mandel] as the other tensors
///
/// # Input
///
/// * `alpha` -- the scalar multiplier
/// * `a` -- the first second-order tensor; with the same [Mandel] as the other tensors
/// * `b` -- the second second-order tensor; with the same [Mandel] as the other tensors
/// * `dd` -- the fourth-order tensor; with the same [Mandel] as the other tensors
///
/// # Panics
///
/// A panic will occur the tensors have different [Mandel]
pub fn t4_ddot_t2_dyad_t2_ddot_t4(ee: &mut Tensor4, alpha: f64, a: &Tensor2, b: &Tensor2, dd: &Tensor4) {
    assert_eq!(a.mandel, dd.mandel);
    assert_eq!(b.mandel, dd.mandel);
    assert_eq!(ee.mandel, dd.mandel);
    let dim = a.vec.dim();
    ee.mat.fill(0.0);
    for m in 0..dim {
        for n in 0..dim {
            for p in 0..dim {
                for q in 0..dim {
                    ee.mat
                        .add(m, n, alpha * dd.mat.get(m, p) * a.vec[p] * b.vec[q] * dd.mat.get(q, n));
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Mandel, SamplesTensor4, MN_TO_IJKL};
    use russell_lab::{approx_eq, mat_approx_eq, vec_approx_eq, Matrix};

    #[test]
    #[should_panic]
    fn t2_ddot_t2_panics_on_different_mandel() {
        let a = Tensor2::new(Mandel::Symmetric);
        let b = Tensor2::new(Mandel::General);
        t2_ddot_t2(&a, &b);
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
    }

    #[test]
    #[should_panic]
    fn t2_dot_t2_panics_on_non_general() {
        let a = Tensor2::new(Mandel::Symmetric);
        let b = Tensor2::new(Mandel::Symmetric);
        let mut c = Tensor2::new(Mandel::Symmetric); // wrong; it must be General
        t2_dot_t2(&mut c, &a, &b);
    }

    #[test]
    #[should_panic]
    fn t2_dot_t2_panics_on_different_mandel() {
        let a = Tensor2::new(Mandel::Symmetric);
        let b = Tensor2::new(Mandel::General); // wrong; it must be the same as `a`
        let mut c = Tensor2::new(Mandel::General);
        t2_dot_t2(&mut c, &a, &b);
    }

    #[test]
    fn t2_dot_t2_works() {
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
        let mut c = Tensor2::new(Mandel::General);
        t2_dot_t2(&mut c, &a, &b);
        #[rustfmt::skip]
        let correct = Tensor2::from_matrix(&[
            [ 30.0,  24.0, 18.0],
            [ 84.0,  69.0, 54.0],
            [138.0, 114.0, 90.0],
        ], Mandel::General).unwrap();
        vec_approx_eq(&c.vec, &correct.vec, 1e-13);

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
        let mut c = Tensor2::new(Mandel::General);
        t2_dot_t2(&mut c, &a, &b);
        #[rustfmt::skip]
        let correct = Tensor2::from_matrix(&[
            [59.0, 37.0, 28.0],
            [52.0, 44.0, 37.0],
            [61.0, 52.0, 59.0],
        ], Mandel::General).unwrap();
        vec_approx_eq(&c.vec, &correct.vec, 1e-13);

        // sym-2D . sym-2D
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
        let mut c = Tensor2::new(Mandel::General);
        t2_dot_t2(&mut c, &a, &b);
        #[rustfmt::skip]
        let correct = Tensor2::from_matrix(&[
            [23.0, 13.0, 0.0],
            [22.0, 24.0, 0.0],
            [ 0.0,  0.0, 3.0],
        ], Mandel::General).unwrap();
        vec_approx_eq(&c.vec, &correct.vec, 1e-13);
    }

    #[test]
    #[should_panic]
    fn t2_dot_vec_panics_on_non_2d_vector_v() {
        let mut v = Vector::new(3); // wrong; it must be 2
        let a = Tensor2::new(Mandel::Symmetric2D);
        let u = Vector::new(2);
        t2_dot_vec(&mut v, 1.0, &a, &u);
    }

    #[test]
    #[should_panic]
    fn t2_dot_vec_panics_on_non_2d_vector_u() {
        let mut v = Vector::new(2);
        let a = Tensor2::new(Mandel::Symmetric2D);
        let u = Vector::new(3); // wrong; it must be 2
        t2_dot_vec(&mut v, 1.0, &a, &u);
    }

    #[test]
    #[should_panic]
    fn t2_dot_vec_panics_on_non_3d_vector_v() {
        let mut v = Vector::new(2); // wrong; it must be 3
        let a = Tensor2::new(Mandel::General);
        let u = Vector::new(3);
        t2_dot_vec(&mut v, 1.0, &a, &u);
    }

    #[test]
    #[should_panic]
    fn t2_dot_vec_panics_on_non_3d_vector_u() {
        let mut v = Vector::new(3);
        let a = Tensor2::new(Mandel::General);
        let u = Vector::new(2); // wrong; it must be 3
        t2_dot_vec(&mut v, 1.0, &a, &u);
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
        t2_dot_vec(&mut v, 2.0, &a, &u);
        vec_approx_eq(&v, &[-40.0, -94.0, -148.0], 1e-13);

        // sym-3D . vec
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 2.0, 3.0],
            [2.0, 5.0, 6.0],
            [3.0, 6.0, 9.0],
        ], Mandel::Symmetric).unwrap();
        let u = Vector::from(&[-2.0, -3.0, -4.0]);
        let mut v = Vector::new(3);
        t2_dot_vec(&mut v, 2.0, &a, &u);
        vec_approx_eq(&v, &[-40.0, -86.0, -120.0], 1e-13);

        // sym-2D . vec
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 2.0, 0.0],
            [2.0, 5.0, 0.0],
            [0.0, 0.0, 9.0],
        ], Mandel::Symmetric2D).unwrap();
        let u = Vector::from(&[-2.0, -3.0]);
        let mut v = Vector::new(2);
        t2_dot_vec(&mut v, 2.0, &a, &u);
        vec_approx_eq(&v, &[-16.0, -38.0], 1e-13);
    }

    #[test]
    #[should_panic]
    fn vec_dot_t2_panics_on_non_2d_vector_v() {
        let mut v = Vector::new(3); // wrong; it must be 2
        let a = Tensor2::new(Mandel::Symmetric2D);
        let u = Vector::new(2);
        vec_dot_t2(&mut v, 1.0, &u, &a);
    }

    #[test]
    #[should_panic]
    fn vec_dot_t2_panics_on_non_2d_vector_u() {
        let mut v = Vector::new(2);
        let a = Tensor2::new(Mandel::Symmetric2D);
        let u = Vector::new(3); // wrong; it must be 2
        vec_dot_t2(&mut v, 1.0, &u, &a);
    }

    #[test]
    #[should_panic]
    fn vec_dot_t2_panics_on_non_3d_vector_v() {
        let mut v = Vector::new(2); // wrong; it must be 3
        let a = Tensor2::new(Mandel::General);
        let u = Vector::new(3);
        vec_dot_t2(&mut v, 1.0, &u, &a);
    }

    #[test]
    #[should_panic]
    fn vec_dot_t2_panics_on_non_3d_vector_u() {
        let mut v = Vector::new(3);
        let a = Tensor2::new(Mandel::General);
        let u = Vector::new(2); // wrong; it must be 3
        vec_dot_t2(&mut v, 1.0, &u, &a);
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
        vec_dot_t2(&mut v, 2.0, &u, &a);
        vec_approx_eq(&v, &[-84.0, -102.0, -120.0], 1e-13);

        // sym-3D . vec
        let u = Vector::from(&[-2.0, -3.0, -4.0]);
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 2.0, 3.0],
            [2.0, 5.0, 6.0],
            [3.0, 6.0, 9.0],
        ], Mandel::Symmetric).unwrap();
        let mut v = Vector::new(3);
        vec_dot_t2(&mut v, 2.0, &u, &a);
        vec_approx_eq(&v, &[-40.0, -86.0, -120.0], 1e-13);

        // sym-2D . vec
        let u = Vector::from(&[-2.0, -3.0]);
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 2.0, 0.0],
            [2.0, 5.0, 0.0],
            [0.0, 0.0, 9.0],
        ], Mandel::Symmetric2D).unwrap();
        let mut v = Vector::new(2);
        vec_dot_t2(&mut v, 2.0, &u, &a);
        vec_approx_eq(&v, &[-16.0, -38.0], 1e-13);
    }

    #[test]
    #[should_panic]
    fn vec_dyad_vec_panics_on_non_2d_vector_u() {
        let mut a = Tensor2::new(Mandel::Symmetric2D);
        let u = Vector::new(3); // wrong; it must be 2
        let v = Vector::new(2);
        let _ = vec_dyad_vec(&mut a, 1.0, &u, &v);
    }

    #[test]
    #[should_panic]
    fn vec_dyad_vec_panics_on_non_2d_vector_v() {
        let mut a = Tensor2::new(Mandel::Symmetric2D);
        let u = Vector::new(2);
        let v = Vector::new(3); // wrong; it must be 2
        let _ = vec_dyad_vec(&mut a, 1.0, &u, &v);
    }

    #[test]
    #[should_panic]
    fn vec_dyad_vec_panics_on_non_3d_vector_u() {
        let mut a = Tensor2::new(Mandel::General);
        let u = Vector::new(2); // wrong; it must be 3
        let v = Vector::new(3);
        let _ = vec_dyad_vec(&mut a, 1.0, &u, &v);
    }

    #[test]
    #[should_panic]
    fn vec_dyad_vec_panics_on_non_3d_vector_v() {
        let mut a = Tensor2::new(Mandel::General);
        let u = Vector::new(3);
        let v = Vector::new(2); // wrong; it must be 3
        let _ = vec_dyad_vec(&mut a, 1.0, &u, &v);
    }

    #[test]
    fn vec_dyad_vec_captures_errors() {
        // symmetric 2D
        let mut tt = Tensor2::new(Mandel::Symmetric2D);
        let u = Vector::from(&[-2.0, -3.0]);
        let v = Vector::from(&[4.0, 3.0]);
        assert_eq!(
            vec_dyad_vec(&mut tt, 1.0, &u, &v).err(),
            Some("dyadic product between u and v does not generate a symmetric tensor")
        );
        // symmetric 3D
        let u = Vector::from(&[-2.0, -3.0, -4.0]);
        let v = Vector::from(&[4.0, 3.0, 2.0]);
        let mut tt = Tensor2::new(Mandel::Symmetric);
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
        vec_approx_eq(&tt.vec, correct, 1e-14);

        // symmetric 3D
        let u = Vector::from(&[-2.0, -3.0, -4.0]);
        let v = Vector::from(&[2.0, 3.0, 4.0]);
        let mut tt = Tensor2::new(Mandel::Symmetric);
        vec_dyad_vec(&mut tt, 2.0, &u, &v).unwrap();
        let correct = &[-8.0, -18.0, -32.0, -12.0 * SQRT_2, -24.0 * SQRT_2, -16.0 * SQRT_2];
        vec_approx_eq(&tt.vec, correct, 1e-14);

        // symmetric 2D
        let u = Vector::from(&[-2.0, -3.0]);
        let v = Vector::from(&[2.0, 3.0]);
        let mut tt = Tensor2::new(Mandel::Symmetric2D);
        vec_dyad_vec(&mut tt, 2.0, &u, &v).unwrap();
        let correct = &[-8.0, -18.0, 0.0, -12.0 * SQRT_2];
        vec_approx_eq(&tt.vec, correct, 1e-14);
    }

    #[test]
    #[should_panic]
    fn t2_dyad_t2_panics_on_different_mandel1() {
        let a = Tensor2::new(Mandel::Symmetric2D);
        let b = Tensor2::new(Mandel::Symmetric); // wrong; it must be Symmetric2D
        let mut dd = Tensor4::new(Mandel::Symmetric2D);
        t2_dyad_t2(&mut dd, 1.0, &a, &b);
    }

    #[test]
    #[should_panic]
    fn t2_dyad_t2_panics_on_different_mandel2() {
        let a = Tensor2::new(Mandel::Symmetric2D);
        let b = Tensor2::new(Mandel::Symmetric2D);
        let mut dd = Tensor4::new(Mandel::Symmetric); // wrong; it must be Symmetric2D
        t2_dyad_t2(&mut dd, 1.0, &a, &b);
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
        t2_dyad_t2(&mut dd, 2.0, &a, &b);
        let mat = dd.as_matrix();
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
        t2_dyad_t2(&mut dd, 2.0, &a, &b);
        let mat = dd.as_matrix();
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
        t2_dyad_t2(&mut dd, 2.0, &a, &b);
        let mat = dd.as_matrix();
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
    #[should_panic]
    fn t2_dyad_t2_update_panics_on_different_mandel1() {
        let a = Tensor2::new(Mandel::Symmetric2D);
        let b = Tensor2::new(Mandel::Symmetric); // wrong; it must be Symmetric2D
        let mut dd = Tensor4::new(Mandel::Symmetric2D);
        t2_dyad_t2_update(&mut dd, 1.0, &a, &b);
    }

    #[test]
    #[should_panic]
    fn t2_dyad_t2_update_panics_on_different_mandel2() {
        let a = Tensor2::new(Mandel::Symmetric2D);
        let b = Tensor2::new(Mandel::Symmetric2D);
        let mut dd = Tensor4::new(Mandel::Symmetric); // wrong; it must be Symmetric2D
        t2_dyad_t2_update(&mut dd, 1.0, &a, &b);
    }

    #[test]
    fn t2_dyad_t2_update_works() {
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
        let mat = Matrix::filled(9, 9, 0.1);
        let mut dd = Tensor4::from_matrix(&mat, Mandel::General).unwrap();
        t2_dyad_t2_update(&mut dd, 2.0, &a, &b);
        let mat = dd.as_matrix();
        let correct = "┌                                     ┐\n\
                       │ 1.1 1.1 1.1 1.1 1.1 1.1 1.1 1.1 1.1 │\n\
                       │ 5.1 5.1 5.1 5.1 5.1 5.1 5.1 5.1 5.1 │\n\
                       │ 9.1 9.1 9.1 9.1 9.1 9.1 9.1 9.1 9.1 │\n\
                       │ 2.1 2.1 2.1 2.1 2.1 2.1 2.1 2.1 2.1 │\n\
                       │ 6.1 6.1 6.1 6.1 6.1 6.1 6.1 6.1 6.1 │\n\
                       │ 3.1 3.1 3.1 3.1 3.1 3.1 3.1 3.1 3.1 │\n\
                       │ 4.1 4.1 4.1 4.1 4.1 4.1 4.1 4.1 4.1 │\n\
                       │ 8.1 8.1 8.1 8.1 8.1 8.1 8.1 8.1 8.1 │\n\
                       │ 7.1 7.1 7.1 7.1 7.1 7.1 7.1 7.1 7.1 │\n\
                       └                                     ┘";
        assert_eq!(format!("{:.1}", mat), correct);
    }

    fn check_dyad(s: f64, a_ten: &Tensor2, b_ten: &Tensor2, dd_ten: &Tensor4, tol: f64) {
        let a = a_ten.as_matrix();
        let b = b_ten.as_matrix();
        let dd = dd_ten.as_matrix();
        let mut correct = Matrix::new(9, 9);
        for m in 0..9 {
            for n in 0..9 {
                let (i, j, k, l) = MN_TO_IJKL[m][n];
                correct.set(m, n, s * a.get(i, j) * b.get(k, l));
            }
        }
        mat_approx_eq(&dd, &correct, tol);
    }

    #[test]
    fn t2_dyad_t2_works_extra() {
        // general dyad general
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
        t2_dyad_t2(&mut dd, 2.0, &a, &b);
        let mat = dd.as_matrix();
        // println!("{:.1}", mat);
        let correct = Matrix::from(&[
            [18.0, 10.0, 2.0, 16.0, 8.0, 14.0, 12.0, 4.0, 6.0],
            [90.0, 50.0, 10.0, 80.0, 40.0, 70.0, 60.0, 20.0, 30.0],
            [162.0, 90.0, 18.0, 144.0, 72.0, 126.0, 108.0, 36.0, 54.0],
            [36.0, 20.0, 4.0, 32.0, 16.0, 28.0, 24.0, 8.0, 12.0],
            [108.0, 60.0, 12.0, 96.0, 48.0, 84.0, 72.0, 24.0, 36.0],
            [54.0, 30.0, 6.0, 48.0, 24.0, 42.0, 36.0, 12.0, 18.0],
            [72.0, 40.0, 8.0, 64.0, 32.0, 56.0, 48.0, 16.0, 24.0],
            [144.0, 80.0, 16.0, 128.0, 64.0, 112.0, 96.0, 32.0, 48.0],
            [126.0, 70.0, 14.0, 112.0, 56.0, 98.0, 84.0, 28.0, 42.0],
        ]);
        mat_approx_eq(&mat, &correct, 1e-13);
        check_dyad(2.0, &a, &b, &dd, 1e-13);

        // symmetric dyad symmetric
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
        let mut dd = Tensor4::new(Mandel::Symmetric);
        t2_dyad_t2(&mut dd, 2.0, &a, &b);
        let mat = dd.as_matrix();
        // println!("{:.1}", mat);
        let correct = Matrix::from(&[
            [6.0, 4.0, 2.0, 10.0, 8.0, 12.0, 10.0, 8.0, 12.0],
            [12.0, 8.0, 4.0, 20.0, 16.0, 24.0, 20.0, 16.0, 24.0],
            [18.0, 12.0, 6.0, 30.0, 24.0, 36.0, 30.0, 24.0, 36.0],
            [24.0, 16.0, 8.0, 40.0, 32.0, 48.0, 40.0, 32.0, 48.0],
            [30.0, 20.0, 10.0, 50.0, 40.0, 60.0, 50.0, 40.0, 60.0],
            [36.0, 24.0, 12.0, 60.0, 48.0, 72.0, 60.0, 48.0, 72.0],
            [24.0, 16.0, 8.0, 40.0, 32.0, 48.0, 40.0, 32.0, 48.0],
            [30.0, 20.0, 10.0, 50.0, 40.0, 60.0, 50.0, 40.0, 60.0],
            [36.0, 24.0, 12.0, 60.0, 48.0, 72.0, 60.0, 48.0, 72.0],
        ]);
        mat_approx_eq(&mat, &correct, 1e-13);
        check_dyad(2.0, &a, &b, &dd, 1e-13);

        // symmetric 2D dyad symmetric 2D
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ], Mandel::Symmetric2D).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [3.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ], Mandel::Symmetric2D).unwrap();
        let mut dd = Tensor4::new(Mandel::Symmetric2D);
        t2_dyad_t2(&mut dd, 2.0, &a, &b);
        let mat = dd.as_matrix();
        // println!("{:.1}", mat);
        let correct = Matrix::from(&[
            [6.0, 4.0, 2.0, 8.0, 0.0, 0.0, 8.0, 0.0, 0.0],
            [12.0, 8.0, 4.0, 16.0, 0.0, 0.0, 16.0, 0.0, 0.0],
            [18.0, 12.0, 6.0, 24.0, 0.0, 0.0, 24.0, 0.0, 0.0],
            [24.0, 16.0, 8.0, 32.0, 0.0, 0.0, 32.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [24.0, 16.0, 8.0, 32.0, 0.0, 0.0, 32.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]);
        mat_approx_eq(&mat, &correct, 1e-14);
        check_dyad(2.0, &a, &b, &dd, 1e-15);
    }

    #[test]
    #[should_panic]
    fn t2_odyad_t2_panics_on_non_general() {
        let a = Tensor2::new(Mandel::Symmetric2D);
        let b = Tensor2::new(Mandel::Symmetric2D);
        let mut dd = Tensor4::new(Mandel::Symmetric2D); // wrong; it must be General
        t2_odyad_t2(&mut dd, 1.0, &a, &b);
    }

    #[test]
    #[should_panic]
    fn t2_odyad_t2_panics_on_different_mandel() {
        let a = Tensor2::new(Mandel::Symmetric2D);
        let b = Tensor2::new(Mandel::Symmetric); // wrong; it must be the same as `a`
        let mut dd = Tensor4::new(Mandel::General);
        t2_odyad_t2(&mut dd, 1.0, &a, &b);
    }

    fn check_odyad(s: f64, a_ten: &Tensor2, b_ten: &Tensor2, dd_ten: &Tensor4, tol: f64) {
        let a = a_ten.as_matrix();
        let b = b_ten.as_matrix();
        let dd = dd_ten.as_matrix();
        let mut correct = Matrix::new(9, 9);
        for m in 0..9 {
            for n in 0..9 {
                let (i, j, k, l) = MN_TO_IJKL[m][n];
                correct.set(m, n, s * a.get(i, k) * b.get(j, l));
            }
        }
        mat_approx_eq(&dd, &correct, tol);
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
        t2_odyad_t2(&mut dd, 2.0, &a, &b);
        let mat = dd.as_matrix();
        let correct = Matrix::from(&[
            [18.0, 32.0, 42.0, 16.0, 28.0, 14.0, 36.0, 48.0, 54.0],
            [48.0, 50.0, 48.0, 40.0, 40.0, 32.0, 60.0, 60.0, 72.0],
            [42.0, 32.0, 18.0, 28.0, 16.0, 14.0, 48.0, 36.0, 54.0],
            [12.0, 20.0, 24.0, 10.0, 16.0, 8.0, 24.0, 30.0, 36.0],
            [24.0, 20.0, 12.0, 16.0, 10.0, 8.0, 30.0, 24.0, 36.0],
            [6.0, 8.0, 6.0, 4.0, 4.0, 2.0, 12.0, 12.0, 18.0],
            [72.0, 80.0, 84.0, 64.0, 70.0, 56.0, 90.0, 96.0, 108.0],
            [84.0, 80.0, 72.0, 70.0, 64.0, 56.0, 96.0, 90.0, 108.0],
            [126.0, 128.0, 126.0, 112.0, 112.0, 98.0, 144.0, 144.0, 162.0],
        ]);
        mat_approx_eq(&mat, &correct, 1e-13);
        check_odyad(2.0, &a, &b, &dd, 1e-13);

        // symmetric odyad symmetric
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
        let mut dd = Tensor4::new(Mandel::General);
        t2_odyad_t2(&mut dd, 2.0, &a, &b);
        let mat = dd.as_matrix();
        let correct = Matrix::from(&[
            [6.0, 40.0, 72.0, 10.0, 48.0, 12.0, 24.0, 60.0, 36.0],
            [40.0, 8.0, 40.0, 16.0, 16.0, 32.0, 20.0, 20.0, 50.0],
            [72.0, 40.0, 6.0, 48.0, 10.0, 12.0, 60.0, 24.0, 36.0],
            [10.0, 16.0, 48.0, 4.0, 32.0, 8.0, 40.0, 24.0, 60.0],
            [48.0, 16.0, 10.0, 32.0, 4.0, 8.0, 24.0, 40.0, 60.0],
            [12.0, 32.0, 12.0, 8.0, 8.0, 2.0, 48.0, 48.0, 72.0],
            [24.0, 20.0, 60.0, 40.0, 24.0, 48.0, 12.0, 50.0, 30.0],
            [60.0, 20.0, 24.0, 24.0, 40.0, 48.0, 50.0, 12.0, 30.0],
            [36.0, 50.0, 36.0, 60.0, 60.0, 72.0, 30.0, 30.0, 18.0],
        ]);
        mat_approx_eq(&mat, &correct, 1e-13);
        check_odyad(2.0, &a, &b, &dd, 1e-13);

        // symmetric 2D odyad symmetric 2D
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ], Mandel::Symmetric2D).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [3.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ], Mandel::Symmetric2D).unwrap();
        let mut dd = Tensor4::new(Mandel::General);
        t2_odyad_t2(&mut dd, 2.0, &a, &b);
        let mat = dd.as_matrix();
        // println!("{:.1}", mat);
        let correct = Matrix::from(&[
            [6.0, 32.0, 0.0, 8.0, 0.0, 0.0, 24.0, 0.0, 0.0],
            [32.0, 8.0, 0.0, 16.0, 0.0, 0.0, 16.0, 0.0, 0.0],
            [0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [8.0, 16.0, 0.0, 4.0, 0.0, 0.0, 32.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 4.0, 8.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 8.0, 2.0, 0.0, 0.0, 0.0],
            [24.0, 16.0, 0.0, 32.0, 0.0, 0.0, 12.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0, 24.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 24.0, 18.0],
        ]);
        mat_approx_eq(&mat, &correct, 1e-14);
        check_odyad(2.0, &a, &b, &dd, 1e-15);
    }

    #[test]
    #[should_panic]
    fn t2_udyad_t2_panics_on_non_general() {
        let a = Tensor2::new(Mandel::Symmetric2D);
        let b = Tensor2::new(Mandel::Symmetric2D);
        let mut dd = Tensor4::new(Mandel::Symmetric2D); // wrong; it must be General
        t2_udyad_t2(&mut dd, 1.0, &a, &b);
    }

    #[test]
    #[should_panic]
    fn t2_udyad_t2_panics_on_different_mandel() {
        let a = Tensor2::new(Mandel::Symmetric2D);
        let b = Tensor2::new(Mandel::Symmetric); // wrong; it must be the same as `a`
        let mut dd = Tensor4::new(Mandel::General);
        t2_udyad_t2(&mut dd, 1.0, &a, &b);
    }

    fn check_udyad(s: f64, a_ten: &Tensor2, b_ten: &Tensor2, dd_ten: &Tensor4, tol: f64) {
        let a = a_ten.as_matrix();
        let b = b_ten.as_matrix();
        let dd = dd_ten.as_matrix();
        let mut correct = Matrix::new(9, 9);
        for m in 0..9 {
            for n in 0..9 {
                let (i, j, k, l) = MN_TO_IJKL[m][n];
                correct.set(m, n, s * a.get(i, l) * b.get(j, k));
            }
        }
        mat_approx_eq(&dd, &correct, tol);
    }

    #[test]
    fn t2_udyad_t2_works() {
        // general udyad general
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
        t2_udyad_t2(&mut dd, 2.0, &a, &b);
        let mat = dd.as_matrix();
        let correct = Matrix::from(&[
            [18.0, 32.0, 42.0, 36.0, 48.0, 54.0, 16.0, 28.0, 14.0],
            [48.0, 50.0, 48.0, 60.0, 60.0, 72.0, 40.0, 40.0, 32.0],
            [42.0, 32.0, 18.0, 48.0, 36.0, 54.0, 28.0, 16.0, 14.0],
            [12.0, 20.0, 24.0, 24.0, 30.0, 36.0, 10.0, 16.0, 8.0],
            [24.0, 20.0, 12.0, 30.0, 24.0, 36.0, 16.0, 10.0, 8.0],
            [6.0, 8.0, 6.0, 12.0, 12.0, 18.0, 4.0, 4.0, 2.0],
            [72.0, 80.0, 84.0, 90.0, 96.0, 108.0, 64.0, 70.0, 56.0],
            [84.0, 80.0, 72.0, 96.0, 90.0, 108.0, 70.0, 64.0, 56.0],
            [126.0, 128.0, 126.0, 144.0, 144.0, 162.0, 112.0, 112.0, 98.0],
        ]);
        mat_approx_eq(&mat, &correct, 1e-13);
        check_udyad(2.0, &a, &b, &dd, 1e-13);

        // symmetric udyad symmetric
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
        let mut dd = Tensor4::new(Mandel::General);
        t2_udyad_t2(&mut dd, 2.0, &a, &b);
        let mat = dd.as_matrix();
        let correct = Matrix::from(&[
            [6.0, 40.0, 72.0, 24.0, 60.0, 36.0, 10.0, 48.0, 12.0],
            [40.0, 8.0, 40.0, 20.0, 20.0, 50.0, 16.0, 16.0, 32.0],
            [72.0, 40.0, 6.0, 60.0, 24.0, 36.0, 48.0, 10.0, 12.0],
            [10.0, 16.0, 48.0, 40.0, 24.0, 60.0, 4.0, 32.0, 8.0],
            [48.0, 16.0, 10.0, 24.0, 40.0, 60.0, 32.0, 4.0, 8.0],
            [12.0, 32.0, 12.0, 48.0, 48.0, 72.0, 8.0, 8.0, 2.0],
            [24.0, 20.0, 60.0, 12.0, 50.0, 30.0, 40.0, 24.0, 48.0],
            [60.0, 20.0, 24.0, 50.0, 12.0, 30.0, 24.0, 40.0, 48.0],
            [36.0, 50.0, 36.0, 30.0, 30.0, 18.0, 60.0, 60.0, 72.0],
        ]);
        mat_approx_eq(&mat, &correct, 1e-13);
        check_udyad(2.0, &a, &b, &dd, 1e-13);

        // symmetric 2D udyad symmetric 2D
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ], Mandel::Symmetric2D).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [3.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ], Mandel::Symmetric2D).unwrap();
        let mut dd = Tensor4::new(Mandel::General);
        t2_udyad_t2(&mut dd, 2.0, &a, &b);
        let mandel_mat = Matrix::from(&[
            [6.0, 32.0, 0.0, 16.0 * SQRT_2, 0.0, 0.0, 8.0 * SQRT_2, 0.0, 0.0],
            [32.0, 8.0, 0.0, 16.0 * SQRT_2, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [16.0 * SQRT_2, 16.0 * SQRT_2, 0.0, 40.0, 0.0, 0.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 8.0, 16.0, 0.0, 4.0, 8.0],
            [0.0, 0.0, 0.0, 0.0, 16.0, 10.0, 0.0, 8.0, 8.0],
            [-8.0 * SQRT_2, 0.0, 0.0, -4.0, 0.0, 0.0, 24.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -4.0, -8.0, 0.0, -8.0, -16.0],
            [0.0, 0.0, 0.0, 0.0, -8.0, -8.0, 0.0, -16.0, -10.0],
        ]);
        mat_approx_eq(&dd.mat, &mandel_mat, 1e-14);
        let mat = dd.as_matrix();
        let correct = Matrix::from(&[
            [6.0, 32.0, 0.0, 24.0, 0.0, 0.0, 8.0, 0.0, 0.0],
            [32.0, 8.0, 0.0, 16.0, 0.0, 0.0, 16.0, 0.0, 0.0],
            [0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [8.0, 16.0, 0.0, 32.0, 0.0, 0.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 8.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 2.0],
            [24.0, 16.0, 0.0, 12.0, 0.0, 0.0, 32.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 12.0, 24.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 24.0, 18.0, 0.0, 0.0, 0.0],
        ]);
        mat_approx_eq(&mat, &correct, 1e-14);
        check_udyad(2.0, &a, &b, &dd, 1e-15);
    }

    #[test]
    #[should_panic]
    fn t2_ssd_panics_on_non_sym() {
        let a = Tensor2::new(Mandel::Symmetric2D);
        let mut dd = Tensor4::new(Mandel::Symmetric2D); // wrong; it must be Symmetric
        t2_ssd(&mut dd, 1.0, &a);
    }

    fn check_ssd(s: f64, a_ten: &Tensor2, dd_ten: &Tensor4, tol: f64) {
        let a = a_ten.as_matrix();
        let dd = dd_ten.as_matrix();
        let mut correct = Matrix::new(9, 9);
        for m in 0..9 {
            for n in 0..9 {
                let (i, j, k, l) = MN_TO_IJKL[m][n];
                correct.set(m, n, s * (a.get(i, k) * a.get(j, l) + a.get(i, l) * a.get(j, k)));
            }
        }
        mat_approx_eq(&dd, &correct, tol);
    }

    #[test]
    fn t2_ssd_works() {
        // general
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ], Mandel::General).unwrap();
        let mut dd = Tensor4::new(Mandel::Symmetric);
        t2_ssd(&mut dd, 2.0, &a);
        let mat = dd.as_matrix();
        let correct = Matrix::from(&[
            [4.0, 16.0, 36.0, 8.0, 24.0, 12.0, 8.0, 24.0, 12.0],
            [64.0, 100.0, 144.0, 80.0, 120.0, 96.0, 80.0, 120.0, 96.0],
            [196.0, 256.0, 324.0, 224.0, 288.0, 252.0, 224.0, 288.0, 252.0],
            [16.0, 40.0, 72.0, 26.0, 54.0, 36.0, 26.0, 54.0, 36.0],
            [112.0, 160.0, 216.0, 134.0, 186.0, 156.0, 134.0, 186.0, 156.0],
            [28.0, 64.0, 108.0, 44.0, 84.0, 60.0, 44.0, 84.0, 60.0],
            [16.0, 40.0, 72.0, 26.0, 54.0, 36.0, 26.0, 54.0, 36.0],
            [112.0, 160.0, 216.0, 134.0, 186.0, 156.0, 134.0, 186.0, 156.0],
            [28.0, 64.0, 108.0, 44.0, 84.0, 60.0, 44.0, 84.0, 60.0],
        ]);
        mat_approx_eq(&mat, &correct, 1e-13);
        check_ssd(2.0, &a, &dd, 1e-13);

        // symmetric
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0, 5.0, 3.0],
        ], Mandel::Symmetric).unwrap();
        let mut dd = Tensor4::new(Mandel::Symmetric);
        t2_ssd(&mut dd, 2.0, &a);
        let mat = dd.as_matrix();
        let correct = Matrix::from(&[
            [4.0, 64.0, 144.0, 16.0, 96.0, 24.0, 16.0, 96.0, 24.0],
            [64.0, 16.0, 100.0, 32.0, 40.0, 80.0, 32.0, 40.0, 80.0],
            [144.0, 100.0, 36.0, 120.0, 60.0, 72.0, 120.0, 60.0, 72.0],
            [16.0, 32.0, 120.0, 36.0, 64.0, 58.0, 36.0, 64.0, 58.0],
            [96.0, 40.0, 60.0, 64.0, 62.0, 84.0, 64.0, 62.0, 84.0],
            [24.0, 80.0, 72.0, 58.0, 84.0, 78.0, 58.0, 84.0, 78.0],
            [16.0, 32.0, 120.0, 36.0, 64.0, 58.0, 36.0, 64.0, 58.0],
            [96.0, 40.0, 60.0, 64.0, 62.0, 84.0, 64.0, 62.0, 84.0],
            [24.0, 80.0, 72.0, 58.0, 84.0, 78.0, 58.0, 84.0, 78.0],
        ]);
        mat_approx_eq(&mat, &correct, 1e-13);
        check_ssd(2.0, &a, &dd, 1e-13);

        // symmetric 2D
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ], Mandel::Symmetric2D).unwrap();
        let mut dd = Tensor4::new(Mandel::Symmetric);
        t2_ssd(&mut dd, 2.0, &a);
        let mat = dd.as_matrix();
        let correct = Matrix::from(&[
            [4.0, 64.0, 0.0, 16.0, 0.0, 0.0, 16.0, 0.0, 0.0],
            [64.0, 16.0, 0.0, 32.0, 0.0, 0.0, 32.0, 0.0, 0.0],
            [0.0, 0.0, 36.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [16.0, 32.0, 0.0, 36.0, 0.0, 0.0, 36.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 12.0, 24.0, 0.0, 12.0, 24.0],
            [0.0, 0.0, 0.0, 0.0, 24.0, 6.0, 0.0, 24.0, 6.0],
            [16.0, 32.0, 0.0, 36.0, 0.0, 0.0, 36.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 12.0, 24.0, 0.0, 12.0, 24.0],
            [0.0, 0.0, 0.0, 0.0, 24.0, 6.0, 0.0, 24.0, 6.0],
        ]);
        mat_approx_eq(&mat, &correct, 1e-13);
        check_ssd(2.0, &a, &dd, 1e-14);
    }

    #[test]
    #[should_panic]
    fn t2_qsd_t2_panics_on_non_sym() {
        let a = Tensor2::new(Mandel::Symmetric2D);
        let b = Tensor2::new(Mandel::Symmetric2D);
        let mut dd = Tensor4::new(Mandel::Symmetric2D); // wrong; it must be Symmetric
        t2_qsd_t2(&mut dd, 1.0, &a, &b);
    }

    #[test]
    #[should_panic]
    fn t2_qsd_t2_panics_on_different_mandel() {
        let a = Tensor2::new(Mandel::Symmetric2D);
        let b = Tensor2::new(Mandel::Symmetric); // wrong; it must be the same as `a`
        let mut dd = Tensor4::new(Mandel::Symmetric);
        t2_qsd_t2(&mut dd, 1.0, &a, &b);
    }

    fn check_qsd(s: f64, a_ten: &Tensor2, b_ten: &Tensor2, dd_ten: &Tensor4, tol: f64) {
        let a = a_ten.as_matrix();
        let b = b_ten.as_matrix();
        let dd = dd_ten.as_matrix();
        let mut correct = Matrix::new(9, 9);
        for m in 0..9 {
            for n in 0..9 {
                let (i, j, k, l) = MN_TO_IJKL[m][n];
                correct.set(m, n, s * a.get(i, l) * b.get(j, k));
                correct.set(
                    m,
                    n,
                    s * (a.get(i, k) * b.get(j, l)
                        + a.get(i, l) * b.get(j, k)
                        + b.get(i, k) * a.get(j, l)
                        + b.get(i, l) * a.get(j, k)),
                );
            }
        }
        mat_approx_eq(&dd, &correct, tol);
    }

    #[test]
    fn t2_qsd_t2_works() {
        // general qsd general
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
        let mut dd = Tensor4::new(Mandel::Symmetric);
        t2_qsd_t2(&mut dd, 2.0, &a, &b);
        let mat = dd.as_matrix();
        let correct = Matrix::from(&[
            [72.0, 128.0, 168.0, 104.0, 152.0, 136.0, 104.0, 152.0, 136.0],
            [192.0, 200.0, 192.0, 200.0, 200.0, 208.0, 200.0, 200.0, 208.0],
            [168.0, 128.0, 72.0, 152.0, 104.0, 136.0, 152.0, 104.0, 136.0],
            [168.0, 200.0, 216.0, 188.0, 212.0, 208.0, 188.0, 212.0, 208.0],
            [216.0, 200.0, 168.0, 212.0, 188.0, 208.0, 212.0, 188.0, 208.0],
            [264.0, 272.0, 264.0, 272.0, 272.0, 280.0, 272.0, 272.0, 280.0],
            [168.0, 200.0, 216.0, 188.0, 212.0, 208.0, 188.0, 212.0, 208.0],
            [216.0, 200.0, 168.0, 212.0, 188.0, 208.0, 212.0, 188.0, 208.0],
            [264.0, 272.0, 264.0, 272.0, 272.0, 280.0, 272.0, 272.0, 280.0],
        ]);
        mat_approx_eq(&mat, &correct, 1e-13);
        check_qsd(2.0, &a, &b, &dd, 1e-13);

        // symmetric qsd symmetric
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
        let mut dd = Tensor4::new(Mandel::Symmetric);
        t2_qsd_t2(&mut dd, 2.0, &a, &b);
        let mat = dd.as_matrix();
        let correct = Matrix::from(&[
            [24.0, 160.0, 288.0, 68.0, 216.0, 96.0, 68.0, 216.0, 96.0],
            [160.0, 32.0, 160.0, 72.0, 72.0, 164.0, 72.0, 72.0, 164.0],
            [288.0, 160.0, 24.0, 216.0, 68.0, 96.0, 216.0, 68.0, 96.0],
            [68.0, 72.0, 216.0, 96.0, 130.0, 146.0, 96.0, 130.0, 146.0],
            [216.0, 72.0, 68.0, 130.0, 96.0, 146.0, 130.0, 96.0, 146.0],
            [96.0, 164.0, 96.0, 146.0, 146.0, 164.0, 146.0, 146.0, 164.0],
            [68.0, 72.0, 216.0, 96.0, 130.0, 146.0, 96.0, 130.0, 146.0],
            [216.0, 72.0, 68.0, 130.0, 96.0, 146.0, 130.0, 96.0, 146.0],
            [96.0, 164.0, 96.0, 146.0, 146.0, 164.0, 146.0, 146.0, 164.0],
        ]);
        mat_approx_eq(&mat, &correct, 1e-13);
        check_qsd(2.0, &a, &b, &dd, 1e-13);

        // symmetric 2D qsd symmetric 2D
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ], Mandel::Symmetric2D).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_matrix(&[
            [3.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ], Mandel::Symmetric2D).unwrap();
        let mut dd = Tensor4::new(Mandel::Symmetric);
        t2_qsd_t2(&mut dd, 2.0, &a, &b);
        let mat = dd.as_matrix();
        let correct = Matrix::from(&[
            [24.0, 128.0, 0.0, 64.0, 0.0, 0.0, 64.0, 0.0, 0.0],
            [128.0, 32.0, 0.0, 64.0, 0.0, 0.0, 64.0, 0.0, 0.0],
            [0.0, 0.0, 24.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [64.0, 64.0, 0.0, 80.0, 0.0, 0.0, 80.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 16.0, 32.0, 0.0, 16.0, 32.0],
            [0.0, 0.0, 0.0, 0.0, 32.0, 20.0, 0.0, 32.0, 20.0],
            [64.0, 64.0, 0.0, 80.0, 0.0, 0.0, 80.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 16.0, 32.0, 0.0, 16.0, 32.0],
            [0.0, 0.0, 0.0, 0.0, 32.0, 20.0, 0.0, 32.0, 20.0],
        ]);
        mat_approx_eq(&mat, &correct, 1e-13);
        check_qsd(2.0, &a, &b, &dd, 1e-14);
    }

    #[test]
    #[should_panic]
    fn t4_ddot_t2_panics_on_different_mandel1() {
        let a = Tensor2::new(Mandel::Symmetric); // wrong; it must be the same as `dd`
        let mut b = Tensor2::new(Mandel::Symmetric2D);
        let dd = Tensor4::new(Mandel::Symmetric2D);
        t4_ddot_t2(&mut b, 1.0, &dd, &a);
    }

    #[test]
    #[should_panic]
    fn t4_ddot_t2_panics_on_different_mandel2() {
        let a = Tensor2::new(Mandel::Symmetric2D);
        let mut b = Tensor2::new(Mandel::Symmetric); // wrong; it must be the same as `dd`
        let dd = Tensor4::new(Mandel::Symmetric2D);
        t4_ddot_t2(&mut b, 1.0, &dd, &a);
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
        t4_ddot_t2(&mut b, 1.0, &dd, &a);
        let out = b.as_matrix();
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
    #[should_panic]
    fn t4_ddot_t2_update_panics_on_different_mandel1() {
        let a = Tensor2::new(Mandel::Symmetric); // wrong; it must be the same as `dd`
        let mut b = Tensor2::new(Mandel::Symmetric2D);
        let dd = Tensor4::new(Mandel::Symmetric2D);
        t4_ddot_t2_update(&mut b, 1.0, &dd, &a, 1.0);
    }

    #[test]
    #[should_panic]
    fn t4_ddot_update_t2_panics_on_different_mandel2() {
        let a = Tensor2::new(Mandel::Symmetric2D);
        let mut b = Tensor2::new(Mandel::Symmetric); // wrong; it must be the same as `dd`
        let dd = Tensor4::new(Mandel::Symmetric2D);
        t4_ddot_t2_update(&mut b, 1.0, &dd, &a, 1.0);
    }

    #[test]
    fn t4_ddot_t2_update_works() {
        let dd = Tensor4::from_matrix(&SamplesTensor4::SYM_2D_SAMPLE1_STD_MATRIX, Mandel::Symmetric2D).unwrap();
        #[rustfmt::skip]
        let a = Tensor2::from_matrix(&[
            [-1.0, -2.0,  0.0],
            [-2.0,  2.0,  0.0],
            [ 0.0,  0.0, -3.0],
        ], Mandel::Symmetric2D).unwrap();
        #[rustfmt::skip]
        let mut b = Tensor2::from_matrix(&[
            [-1000.0, -1000.0,     0.0],
            [-1000.0, -1000.0,     0.0],
            [    0.0,     0.0, -1000.0],
        ], Mandel::Symmetric2D).unwrap();
        t4_ddot_t2_update(&mut b, 1.0, &dd, &a, 2.0);
        let out = b.as_matrix();
        assert_eq!(
            format!("{:.1}", out),
            "┌                         ┐\n\
             │ -2046.0 -2154.0     0.0 │\n\
             │ -2154.0 -2064.0     0.0 │\n\
             │     0.0     0.0 -2082.0 │\n\
             └                         ┘"
        );
    }

    #[test]
    #[should_panic]
    fn t2_ddot_t4_panics_on_different_mandel1() {
        let a = Tensor2::new(Mandel::Symmetric); // wrong; it must be the same as `dd`
        let mut b = Tensor2::new(Mandel::Symmetric2D);
        let dd = Tensor4::new(Mandel::Symmetric2D);
        t2_ddot_t4(&mut b, 1.0, &a, &dd);
    }

    #[test]
    #[should_panic]
    fn t2_ddot_t4_panics_on_different_mandel2() {
        let a = Tensor2::new(Mandel::Symmetric2D);
        let mut b = Tensor2::new(Mandel::Symmetric); // wrong; it must be the same as `dd`
        let dd = Tensor4::new(Mandel::Symmetric2D);
        t2_ddot_t4(&mut b, 1.0, &a, &dd);
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
        t2_ddot_t4(&mut b, 1.0, &a, &dd);
        let out = b.as_matrix();
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
    #[should_panic]
    fn t4_ddot_t4_panics_on_different_mandel1() {
        let cc = Tensor4::new(Mandel::Symmetric); // wrong; it must be the same as `dd`
        let dd = Tensor4::new(Mandel::Symmetric2D);
        let mut ee = Tensor4::new(Mandel::Symmetric2D);
        t4_ddot_t4(&mut ee, 1.0, &cc, &dd);
    }

    #[test]
    #[should_panic]
    fn t4_ddot_t4_panics_on_different_mandel2() {
        let cc = Tensor4::new(Mandel::Symmetric2D);
        let dd = Tensor4::new(Mandel::Symmetric); // wrong; it must be the same as `dd`
        let mut ee = Tensor4::new(Mandel::Symmetric2D);
        t4_ddot_t4(&mut ee, 1.0, &cc, &dd);
    }

    #[test]
    fn t4_ddot_t4_works() {
        let cc = Tensor4::from_matrix(&SamplesTensor4::SYM_2D_SAMPLE1_STD_MATRIX, Mandel::Symmetric2D).unwrap();
        let mut ee = Tensor4::new(Mandel::Symmetric2D);
        t4_ddot_t4(&mut ee, 2.0, &cc, &cc);
        let out = ee.as_matrix();
        assert_eq!(
            format!("{:.1}", out),
            "┌                                                                ┐\n\
             │  820.0  872.0  924.0 1288.0    0.0    0.0 1288.0    0.0    0.0 │\n\
             │ 1120.0 1202.0 1284.0 1858.0    0.0    0.0 1858.0    0.0    0.0 │\n\
             │ 1420.0 1532.0 1644.0 2428.0    0.0    0.0 2428.0    0.0    0.0 │\n\
             │ 2620.0 2852.0 3084.0 4708.0    0.0    0.0 4708.0    0.0    0.0 │\n\
             │    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0 │\n\
             │    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0 │\n\
             │ 2620.0 2852.0 3084.0 4708.0    0.0    0.0 4708.0    0.0    0.0 │\n\
             │    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0 │\n\
             │    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0 │\n\
             └                                                                ┘"
        );
    }

    #[test]
    #[should_panic]
    fn t4_ddot_t4_update_panics_on_different_mandel1() {
        let cc = Tensor4::new(Mandel::Symmetric); // wrong; it must be the same as `dd`
        let dd = Tensor4::new(Mandel::Symmetric2D);
        let mut ee = Tensor4::new(Mandel::Symmetric2D);
        t4_ddot_t4(&mut ee, 1.0, &cc, &dd);
    }

    #[test]
    #[should_panic]
    fn t4_ddot_t4_update_panics_on_different_mandel2() {
        let cc = Tensor4::new(Mandel::Symmetric2D);
        let dd = Tensor4::new(Mandel::Symmetric); // wrong; it must be the same as `dd`
        let mut ee = Tensor4::new(Mandel::Symmetric2D);
        t4_ddot_t4(&mut ee, 1.0, &cc, &dd);
    }

    #[test]
    fn t4_ddot_t4_update_works() {
        let cc = Tensor4::from_matrix(&SamplesTensor4::SYM_2D_SAMPLE1_STD_MATRIX, Mandel::Symmetric2D).unwrap();
        let mut mat = Matrix::new(9, 9);
        mat.set(0, 0, 0.1);
        mat.set(1, 1, 0.1);
        mat.set(2, 2, 0.1);
        let mut ee = Tensor4::from_matrix(&mat, Mandel::Symmetric2D).unwrap();
        t4_ddot_t4_update(&mut ee, 2.0, &cc, &cc, 2.0);
        let out = ee.as_matrix();
        assert_eq!(
            format!("{:.1}", out),
            "┌                                                                ┐\n\
             │  820.2  872.0  924.0 1288.0    0.0    0.0 1288.0    0.0    0.0 │\n\
             │ 1120.0 1202.2 1284.0 1858.0    0.0    0.0 1858.0    0.0    0.0 │\n\
             │ 1420.0 1532.0 1644.2 2428.0    0.0    0.0 2428.0    0.0    0.0 │\n\
             │ 2620.0 2852.0 3084.0 4708.0    0.0    0.0 4708.0    0.0    0.0 │\n\
             │    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0 │\n\
             │    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0 │\n\
             │ 2620.0 2852.0 3084.0 4708.0    0.0    0.0 4708.0    0.0    0.0 │\n\
             │    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0 │\n\
             │    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0 │\n\
             └                                                                ┘"
        );
    }

    #[test]
    #[should_panic]
    fn t2_ddot_t4_ddot_t2_panics_on_different_mandel1() {
        let a = Tensor2::new(Mandel::Symmetric); // wrong; it must be the same as `dd`
        let b = Tensor2::new(Mandel::Symmetric2D);
        let dd = Tensor4::new(Mandel::Symmetric2D);
        t2_ddot_t4_ddot_t2(&a, &dd, &b);
    }

    #[test]
    #[should_panic]
    fn t2_ddot_t4_ddot_t2_panics_on_different_mandel2() {
        let a = Tensor2::new(Mandel::Symmetric2D);
        let b = Tensor2::new(Mandel::Symmetric); // wrong; it must be the same as `dd`
        let dd = Tensor4::new(Mandel::Symmetric2D);
        t2_ddot_t4_ddot_t2(&a, &dd, &b);
    }

    #[test]
    fn t2_ddot_t4_ddot_t2_works() {
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
        let mat = Matrix::filled(9, 9, -1.0);
        let dd = Tensor4::from_matrix(&mat, Mandel::General).unwrap();
        let s = t2_ddot_t4_ddot_t2(&a, &dd, &b);
        approx_eq(s, -2025.0, 1e-15);
    }

    #[test]
    #[should_panic]
    fn t4_ddot_t2_dyad_t2_ddot_t4_panics_on_different_mandel1() {
        let a = Tensor2::new(Mandel::Symmetric); // wrong; it must be the same as `ee`
        let b = Tensor2::new(Mandel::Symmetric2D);
        let dd = Tensor4::new(Mandel::Symmetric2D);
        let mut ee = Tensor4::new(Mandel::Symmetric2D);
        t4_ddot_t2_dyad_t2_ddot_t4(&mut ee, 2.0, &a, &b, &dd);
    }

    #[test]
    #[should_panic]
    fn t4_ddot_t2_dyad_t2_ddot_t4_panics_on_different_mandel2() {
        let a = Tensor2::new(Mandel::Symmetric2D);
        let b = Tensor2::new(Mandel::Symmetric); // wrong; it must be the same as `ee`
        let dd = Tensor4::new(Mandel::Symmetric2D);
        let mut ee = Tensor4::new(Mandel::Symmetric2D);
        t4_ddot_t2_dyad_t2_ddot_t4(&mut ee, 2.0, &a, &b, &dd);
    }

    #[test]
    #[should_panic]
    fn t4_ddot_t2_dyad_t2_ddot_t4_panics_on_different_mandel3() {
        let a = Tensor2::new(Mandel::Symmetric2D);
        let b = Tensor2::new(Mandel::Symmetric2D);
        let dd = Tensor4::new(Mandel::Symmetric); // wrong; it must be the same as `ee`
        let mut ee = Tensor4::new(Mandel::Symmetric2D);
        t4_ddot_t2_dyad_t2_ddot_t4(&mut ee, 2.0, &a, &b, &dd);
    }

    #[test]
    fn t4_ddot_t2_dyad_t2_ddot_t4_works() {
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
        let mat = Matrix::filled(9, 9, -1.0);
        let dd = Tensor4::from_matrix(&mat, Mandel::General).unwrap();
        let mut ee = Tensor4::new(Mandel::General);
        t4_ddot_t2_dyad_t2_ddot_t4(&mut ee, 2.0, &a, &b, &dd);
        let correct = [
            [4050., 4050., 4050., 4050., 4050., 4050., 4050., 4050., 4050.],
            [4050., 4050., 4050., 4050., 4050., 4050., 4050., 4050., 4050.],
            [4050., 4050., 4050., 4050., 4050., 4050., 4050., 4050., 4050.],
            [4050., 4050., 4050., 4050., 4050., 4050., 4050., 4050., 4050.],
            [4050., 4050., 4050., 4050., 4050., 4050., 4050., 4050., 4050.],
            [4050., 4050., 4050., 4050., 4050., 4050., 4050., 4050., 4050.],
            [4050., 4050., 4050., 4050., 4050., 4050., 4050., 4050., 4050.],
            [4050., 4050., 4050., 4050., 4050., 4050., 4050., 4050., 4050.],
            [4050., 4050., 4050., 4050., 4050., 4050., 4050., 4050., 4050.],
        ];
        mat_approx_eq(&ee.as_matrix(), &correct, 1e-11);
    }
}
