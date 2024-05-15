use super::Tensor2;
use crate::{Mandel, SQRT_2};
use russell_lab::{vec_add, vec_inner};
use russell_lab::{StrError, Vector};

/// Adds two second-order tensors
///
/// ```text
/// c := α⋅a + β⋅b
/// ```
///
/// # Panics
///
/// A panic will occur the tensors have different [Mandel]
pub fn t2_add(c: &mut Tensor2, alpha: f64, a: &Tensor2, beta: f64, b: &Tensor2) {
    assert_eq!(b.mandel, a.mandel);
    assert_eq!(c.mandel, a.mandel);
    vec_add(&mut c.vec, alpha, &a.vec, beta, &b.vec).unwrap();
}

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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Mandel;
    use russell_lab::{approx_eq, mat_approx_eq, vec_approx_eq};

    #[test]
    #[should_panic]
    fn t2_add_panics_on_different_mandel1() {
        let a = Tensor2::new(Mandel::Symmetric2D);
        let b = Tensor2::new(Mandel::Symmetric); // wrong; it must be the same as `a`
        let mut c = Tensor2::new(Mandel::Symmetric2D);
        t2_add(&mut c, 2.0, &a, 3.0, &b);
    }

    #[test]
    #[should_panic]
    fn t2_add_panics_on_different_mandel2() {
        let a = Tensor2::new(Mandel::Symmetric2D);
        let b = Tensor2::new(Mandel::Symmetric2D);
        let mut c = Tensor2::new(Mandel::Symmetric); // wrong; it must be the same as `a`
        t2_add(&mut c, 2.0, &a, 3.0, &b);
    }

    #[test]
    fn t2_add_works() {
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
        let mut c = Tensor2::new(Mandel::Symmetric2D);
        t2_add(&mut c, 2.0, &a, 3.0, &b);
        #[rustfmt::skip]
        let correct = &[
            [11.0, 23.0, 0.0],
            [23.0, 10.0, 0.0],
            [ 0.0,  0.0, 9.0],
        ];
        mat_approx_eq(&c.as_matrix(), correct, 1e-14);
    }

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
}
