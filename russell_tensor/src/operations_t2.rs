use super::{Tensor1, Tensor2};
use crate::SQRT_2;
use russell_lab::StrError;

#[allow(unused)]
use crate::Rep; // for documentation

/// Adds two second-order tensors
///
/// ```text
/// c := α⋅a + β⋅b
/// ```
///
/// # Panics
///
/// A panic will occur if the tensors have different [Rep]
pub fn t2_add(c: &mut Tensor2, alpha: f64, a: &Tensor2, beta: f64, b: &Tensor2) {
    assert_eq!(b.rep(), a.rep());
    assert_eq!(c.rep(), a.rep());
    match a.dim() {
        4 => {
            c.vec[0] = alpha * a.vec[0] + beta * b.vec[0];
            c.vec[1] = alpha * a.vec[1] + beta * b.vec[1];
            c.vec[2] = alpha * a.vec[2] + beta * b.vec[2];
            c.vec[3] = alpha * a.vec[3] + beta * b.vec[3];
        }
        6 => {
            c.vec[0] = alpha * a.vec[0] + beta * b.vec[0];
            c.vec[1] = alpha * a.vec[1] + beta * b.vec[1];
            c.vec[2] = alpha * a.vec[2] + beta * b.vec[2];
            c.vec[3] = alpha * a.vec[3] + beta * b.vec[3];
            c.vec[4] = alpha * a.vec[4] + beta * b.vec[4];
            c.vec[5] = alpha * a.vec[5] + beta * b.vec[5];
        }
        _ => {
            c.vec[0] = alpha * a.vec[0] + beta * b.vec[0];
            c.vec[1] = alpha * a.vec[1] + beta * b.vec[1];
            c.vec[2] = alpha * a.vec[2] + beta * b.vec[2];
            c.vec[3] = alpha * a.vec[3] + beta * b.vec[3];
            c.vec[4] = alpha * a.vec[4] + beta * b.vec[4];
            c.vec[5] = alpha * a.vec[5] + beta * b.vec[5];
            c.vec[6] = alpha * a.vec[6] + beta * b.vec[6];
            c.vec[7] = alpha * a.vec[7] + beta * b.vec[7];
            c.vec[8] = alpha * a.vec[8] + beta * b.vec[8];
        }
    }
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
/// Or, in Kelvin basis:
///
/// ```text
/// s = Σ aₘ bₘ
///     m
/// ```
///
/// # Input
///
/// * `a` -- first tensor; with the same [Rep] as `b`
/// * `b` -- second tensor; with the same [Rep] as `a`
///
/// # Output
///
/// Returns the scalar result of `a : b`.
///
/// # Panics
///
/// A panic will occur if `a` and `b` have different [Rep]
///
/// # Examples
///
/// ```
/// use russell_lab::approx_eq;
/// use russell_tensor::{t2_ddot_t2, Rep, Tensor2, StrError};
///
/// fn main() -> Result<(), StrError> {
///     let a = Tensor2::from_std_matrix(&[
///         [1.0,  1.0, 0.0],
///         [1.0, -1.0, 0.0],
///         [0.0,  0.0, 1.0],
///     ], Rep::Symmetric2D)?;
///
///     let b = Tensor2::from_std_matrix(&[
///         [1.0,  2.0, 0.0],
///         [3.0, -1.0, 5.0],
///         [0.0,  4.0, 1.0],
///     ], Rep::General)?;
///
///     let res = t2_ddot_t2(&a.as_general(), &b);
///
///     approx_eq(res, 8.0, 1e-15);
///     Ok(())
/// }
/// ```
pub fn t2_ddot_t2(a: &Tensor2, b: &Tensor2) -> f64 {
    assert_eq!(a.rep(), b.rep());
    match a.dim() {
        4 => a.vec[0] * b.vec[0] + a.vec[1] * b.vec[1] + a.vec[2] * b.vec[2] + a.vec[3] * b.vec[3],
        6 => {
            a.vec[0] * b.vec[0]
                + a.vec[1] * b.vec[1]
                + a.vec[2] * b.vec[2]
                + a.vec[3] * b.vec[3]
                + a.vec[4] * b.vec[4]
                + a.vec[5] * b.vec[5]
        }
        _ => {
            a.vec[0] * b.vec[0]
                + a.vec[1] * b.vec[1]
                + a.vec[2] * b.vec[2]
                + a.vec[3] * b.vec[3]
                + a.vec[4] * b.vec[4]
                + a.vec[5] * b.vec[5]
                + a.vec[6] * b.vec[6]
                + a.vec[7] * b.vec[7]
                + a.vec[8] * b.vec[8]
        }
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
/// * `v` -- the resulting first-order tensor
///
/// # Input
///
/// * `alpha` -- the `α` multiplier
/// * `a` -- the second-order tensor
/// * `u` -- a 3D vector (first-order tensor)
///
/// # Examples
///
/// ```
/// use russell_tensor::{t2_dot_t1, Rep, Tensor1, Tensor2, StrError};
///
/// fn main() -> Result<(), StrError> {
///     let a = Tensor2::from_std_matrix(&[
///         [1.0,  1.0, 0.0],
///         [1.0, -1.0, 0.0],
///         [0.0,  0.0, 1.0],
///     ], Rep::Symmetric2D)?;
///
///     let u = Tensor1::from(&[1.0, 2.0, 0.0]);
///     let mut v = Tensor1::new();
///     t2_dot_t1(&mut v, 2.0, &a, &u);
///
///     assert_eq!(
///         format!("{:.1}", v),
///         "┌      ┐\n\
///          │  6.0 │\n\
///          │ -2.0 │\n\
///          │  0.0 │\n\
///          └      ┘"
///     );
///     Ok(())
/// }
/// ```
pub fn t2_dot_t1(v: &mut Tensor1, alpha: f64, a: &Tensor2, u: &Tensor1) {
    v.set(
        0,
        alpha * (a.get_std(0, 0) * u.get(0) + a.get_std(0, 1) * u.get(1) + a.get_std(0, 2) * u.get(2)),
    );
    v.set(
        1,
        alpha * (a.get_std(1, 0) * u.get(0) + a.get_std(1, 1) * u.get(1) + a.get_std(1, 2) * u.get(2)),
    );
    v.set(
        2,
        alpha * (a.get_std(2, 0) * u.get(0) + a.get_std(2, 1) * u.get(1) + a.get_std(2, 2) * u.get(2)),
    );
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
/// * `v` -- the resulting first-order tensor
///
/// # Input
///
/// * `alpha` -- the `α` multiplier
/// * `u` -- the first-order tensor (3D vector)
/// * `a` -- the second-order tensor
///
/// # Examples
///
/// ```
/// use russell_tensor::{t1_dot_t2, Rep, Tensor1, Tensor2, StrError};
///
/// fn main() -> Result<(), StrError> {
///     let u = Tensor1::from(&[1.0, 2.0, 0.0]);
///     let a = Tensor2::from_std_matrix(&[
///         [1.0,  1.0, 0.0],
///         [1.0, -1.0, 0.0],
///         [0.0,  0.0, 1.0],
///     ], Rep::Symmetric2D)?;
///
///     let mut v = Tensor1::new();
///     t1_dot_t2(&mut v, 2.0, &u, &a);
///
///     assert_eq!(
///         format!("{:.1}", v),
///         "┌      ┐\n\
///          │  6.0 │\n\
///          │ -2.0 │\n\
///          │  0.0 │\n\
///          └      ┘"
///     );
///     Ok(())
/// }
/// ```
pub fn t1_dot_t2(v: &mut Tensor1, alpha: f64, u: &Tensor1, a: &Tensor2) {
    v.set(
        0,
        alpha * (u.get(0) * a.get_std(0, 0) + u.get(1) * a.get_std(1, 0) + u.get(2) * a.get_std(2, 0)),
    );
    v.set(
        1,
        alpha * (u.get(0) * a.get_std(0, 1) + u.get(1) * a.get_std(1, 1) + u.get(2) * a.get_std(2, 1)),
    );
    v.set(
        2,
        alpha * (u.get(0) * a.get_std(0, 2) + u.get(1) * a.get_std(1, 2) + u.get(2) * a.get_std(2, 2)),
    );
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
/// * `u` -- the first-order tensor; 3D vector
/// * `v` -- the first-order tensor; 3D vector
///
/// # Examples
///
/// ```
/// use russell_tensor::{t1_dyad_t1, Rep, Tensor1, Tensor2, StrError};
///
/// fn main() -> Result<(), StrError> {
///     let u = Tensor1::from(&[1.0, 1.0, 1.0]);
///     let v = Tensor1::from(&[2.0, 2.0, 2.0]);
///
///     let mut tt = Tensor2::new(Rep::Symmetric);
///     t1_dyad_t1(&mut tt, 1.0, &u, &v)?;
///
///     assert_eq!(
///         format!("{:.1}", tt.as_std_matrix()),
///         "┌             ┐\n\
///          │ 2.0 2.0 2.0 │\n\
///          │ 2.0 2.0 2.0 │\n\
///          │ 2.0 2.0 2.0 │\n\
///          └             ┘"
///     );
///     Ok(())
/// }
/// ```
pub fn t1_dyad_t1(a: &mut Tensor2, alpha: f64, u: &Tensor1, v: &Tensor1) -> Result<(), StrError> {
    if a.dim() == 4 {
        if (u.get(0) * v.get(1)) != (u.get(1) * v.get(0)) {
            return Err("dyadic product between u and v does not generate a symmetric tensor");
        }
        a.vec[0] = alpha * u.get(0) * v.get(0);
        a.vec[1] = alpha * u.get(1) * v.get(1);
        a.vec[2] = 0.0;
        a.vec[3] = alpha * (u.get(0) * v.get(1) + u.get(1) * v.get(0)) / SQRT_2;
    } else {
        a.vec[0] = alpha * u.get(0) * v.get(0);
        a.vec[1] = alpha * u.get(1) * v.get(1);
        a.vec[2] = alpha * u.get(2) * v.get(2);
        a.vec[3] = alpha * (u.get(0) * v.get(1) + u.get(1) * v.get(0)) / SQRT_2;
        a.vec[4] = alpha * (u.get(1) * v.get(2) + u.get(2) * v.get(1)) / SQRT_2;
        a.vec[5] = alpha * (u.get(0) * v.get(2) + u.get(2) * v.get(0)) / SQRT_2;
        if a.dim() == 6 {
            if (u.get(0) * v.get(1)) != (u.get(1) * v.get(0))
                || (u.get(1) * v.get(2)) != (u.get(2) * v.get(1))
                || (u.get(0) * v.get(2)) != (u.get(2) * v.get(0))
            {
                return Err("dyadic product between u and v does not generate a symmetric tensor");
            }
        } else {
            a.vec[6] = alpha * (u.get(0) * v.get(1) - u.get(1) * v.get(0)) / SQRT_2;
            a.vec[7] = alpha * (u.get(1) * v.get(2) - u.get(2) * v.get(1)) / SQRT_2;
            a.vec[8] = alpha * (u.get(0) * v.get(2) - u.get(2) * v.get(0)) / SQRT_2;
        }
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor1;
    use russell_lab::{approx_eq, array_approx_eq, mat_approx_eq};

    fn kelvin_vector(tt: &Tensor2) -> Vec<f64> {
        let mut v = vec![0.0; tt.dim()];
        for m in 0..tt.dim() {
            v[m] = tt.get(m);
        }
        v
    }

    #[test]
    #[should_panic]
    fn t2_add_panics_on_different_rep1() {
        let a = Tensor2::new(Rep::Symmetric2D);
        let b = Tensor2::new(Rep::Symmetric); // wrong; it must be the same as `a`
        let mut c = Tensor2::new(Rep::Symmetric2D);
        t2_add(&mut c, 2.0, &a, 3.0, &b);
    }

    #[test]
    #[should_panic]
    fn t2_add_panics_on_different_rep2() {
        let a = Tensor2::new(Rep::Symmetric2D);
        let b = Tensor2::new(Rep::Symmetric2D);
        let mut c = Tensor2::new(Rep::Symmetric); // wrong; it must be the same as `a`
        t2_add(&mut c, 2.0, &a, 3.0, &b);
    }

    #[test]
    fn t2_add_works() {
        #[rustfmt::skip]
        let a = Tensor2::from_std_matrix(&[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ], Rep::Symmetric2D).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_std_matrix(&[
            [3.0, 5.0, 0.0],
            [5.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ], Rep::Symmetric2D).unwrap();
        let mut c = Tensor2::new(Rep::Symmetric2D);
        t2_add(&mut c, 2.0, &a, 3.0, &b);
        #[rustfmt::skip]
        let correct = &[
            [11.0, 23.0, 0.0],
            [23.0, 10.0, 0.0],
            [ 0.0,  0.0, 9.0],
        ];
        mat_approx_eq(&c.as_std_matrix(), correct, 1e-14);
    }

    #[test]
    #[should_panic]
    fn t2_ddot_t2_panics_on_different_rep() {
        let a = Tensor2::new(Rep::Symmetric);
        let b = Tensor2::new(Rep::General);
        t2_ddot_t2(&a, &b);
    }

    #[test]
    fn t2_ddot_t2_works() {
        // general : general
        #[rustfmt::skip]
        let a = Tensor2::from_std_matrix(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ], Rep::General).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_std_matrix(&[
            [9.0, 8.0, 7.0],
            [6.0, 5.0, 4.0],
            [3.0, 2.0, 1.0],
        ], Rep::General).unwrap();
        let s = t2_ddot_t2(&a, &b);
        assert_eq!(s, 165.0);

        // sym-3D : sym-3D
        #[rustfmt::skip]
        let a = Tensor2::from_std_matrix(&[
            [1.0, 4.0, 6.0],
            [4.0, 2.0, 5.0],
            [6.0, 5.0, 3.0],
        ], Rep::Symmetric).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_std_matrix(&[
            [3.0, 5.0, 6.0],
            [5.0, 2.0, 4.0],
            [6.0, 4.0, 1.0],
        ], Rep::Symmetric).unwrap();
        let s = t2_ddot_t2(&a, &b);
        approx_eq(s, 162.0, 1e-13);

        // sym-2D : sym-2D
        #[rustfmt::skip]
        let a = Tensor2::from_std_matrix(&[
            [1.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ], Rep::Symmetric2D).unwrap();
        #[rustfmt::skip]
        let b = Tensor2::from_std_matrix(&[
            [3.0, 5.0, 0.0],
            [5.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ], Rep::Symmetric2D).unwrap();
        let s = t2_ddot_t2(&a, &b);
        approx_eq(s, 50.0, 1e-13);
    }

    #[test]
    fn t2_dot_t1_works() {
        // general . vec
        #[rustfmt::skip]
        let a = Tensor2::from_std_matrix(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ], Rep::General).unwrap();
        let u = Tensor1::from(&[-2.0, -3.0, -4.0]);
        let mut v = Tensor1::new();
        t2_dot_t1(&mut v, 2.0, &a, &u);
        approx_eq(v.get(0), -40.0, 1e-13);
        approx_eq(v.get(1), -94.0, 1e-13);
        approx_eq(v.get(2), -148.0, 1e-13);

        // sym-3D . vec
        #[rustfmt::skip]
        let a = Tensor2::from_std_matrix(&[
            [1.0, 2.0, 3.0],
            [2.0, 5.0, 6.0],
            [3.0, 6.0, 9.0],
        ], Rep::Symmetric).unwrap();
        let u = Tensor1::from(&[-2.0, -3.0, -4.0]);
        let mut v = Tensor1::new();
        t2_dot_t1(&mut v, 2.0, &a, &u);
        approx_eq(v.get(0), -40.0, 1e-13);
        approx_eq(v.get(1), -86.0, 1e-13);
        approx_eq(v.get(2), -120.0, 1e-13);

        // sym-2D . vec
        #[rustfmt::skip]
        let a = Tensor2::from_std_matrix(&[
            [1.0, 2.0, 0.0],
            [2.0, 5.0, 0.0],
            [0.0, 0.0, 9.0],
        ], Rep::Symmetric2D).unwrap();
        let u = Tensor1::from(&[-2.0, -3.0, 0.0]);
        let mut v = Tensor1::new();
        t2_dot_t1(&mut v, 2.0, &a, &u);
        approx_eq(v.get(0), -16.0, 1e-13);
        approx_eq(v.get(1), -38.0, 1e-13);
        approx_eq(v.get(2), 0.0, 1e-13);
    }

    #[test]
    fn t1_dot_t2_works() {
        // vec . general
        let u = Tensor1::from(&[-2.0, -3.0, -4.0]);
        #[rustfmt::skip]
        let a = Tensor2::from_std_matrix(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ], Rep::General).unwrap();
        let mut v = Tensor1::new();
        t1_dot_t2(&mut v, 2.0, &u, &a);
        approx_eq(v.get(0), -84.0, 1e-13);
        approx_eq(v.get(1), -102.0, 1e-13);
        approx_eq(v.get(2), -120.0, 1e-13);

        // vec . sym-3D
        let u = Tensor1::from(&[-2.0, -3.0, -4.0]);
        #[rustfmt::skip]
        let a = Tensor2::from_std_matrix(&[
            [1.0, 2.0, 3.0],
            [2.0, 5.0, 6.0],
            [3.0, 6.0, 9.0],
        ], Rep::Symmetric).unwrap();
        let mut v = Tensor1::new();
        t1_dot_t2(&mut v, 2.0, &u, &a);
        approx_eq(v.get(0), -40.0, 1e-13);
        approx_eq(v.get(1), -86.0, 1e-13);
        approx_eq(v.get(2), -120.0, 1e-13);

        // vec . sym-2D
        let u = Tensor1::from(&[-2.0, -3.0, 0.0]);
        #[rustfmt::skip]
        let a = Tensor2::from_std_matrix(&[
            [1.0, 2.0, 0.0],
            [2.0, 5.0, 0.0],
            [0.0, 0.0, 9.0],
        ], Rep::Symmetric2D).unwrap();
        let mut v = Tensor1::new();
        t1_dot_t2(&mut v, 2.0, &u, &a);
        approx_eq(v.get(0), -16.0, 1e-13);
        approx_eq(v.get(1), -38.0, 1e-13);
        approx_eq(v.get(2), 0.0, 1e-13);
    }

    #[test]
    fn t1_dyad_t1_captures_errors() {
        // symmetric 2D
        let mut tt = Tensor2::new(Rep::Symmetric2D);
        let u = Tensor1::from(&[-2.0, -3.0, 0.0]);
        let v = Tensor1::from(&[4.0, 3.0, 0.0]);
        assert_eq!(
            t1_dyad_t1(&mut tt, 1.0, &u, &v).err(),
            Some("dyadic product between u and v does not generate a symmetric tensor")
        );
        // symmetric 3D
        let u = Tensor1::from(&[-2.0, -3.0, -4.0]);
        let v = Tensor1::from(&[4.0, 3.0, 2.0]);
        let mut tt = Tensor2::new(Rep::Symmetric);
        assert_eq!(
            t1_dyad_t1(&mut tt, 1.0, &u, &v).err(),
            Some("dyadic product between u and v does not generate a symmetric tensor")
        );
    }

    #[test]
    fn t1_dyad_t1_works() {
        // general
        let u = Tensor1::from(&[-2.0, -3.0, -4.0]);
        let v = Tensor1::from(&[4.0, 3.0, 2.0]);
        let mut tt = Tensor2::new(Rep::General);
        t1_dyad_t1(&mut tt, 2.0, &u, &v).unwrap();
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
        array_approx_eq(&kelvin_vector(&tt), correct, 1e-14);

        // symmetric 3D
        let u = Tensor1::from(&[-2.0, -3.0, -4.0]);
        let v = Tensor1::from(&[2.0, 3.0, 4.0]);
        let mut tt = Tensor2::new(Rep::Symmetric);
        t1_dyad_t1(&mut tt, 2.0, &u, &v).unwrap();
        let correct = &[-8.0, -18.0, -32.0, -12.0 * SQRT_2, -24.0 * SQRT_2, -16.0 * SQRT_2];
        array_approx_eq(&kelvin_vector(&tt), correct, 1e-14);

        // symmetric 2D
        let u = Tensor1::from(&[-2.0, -3.0, 0.0]);
        let v = Tensor1::from(&[2.0, 3.0, 0.0]);
        let mut tt = Tensor2::new(Rep::Symmetric2D);
        t1_dyad_t1(&mut tt, 2.0, &u, &v).unwrap();
        let correct = &[-8.0, -18.0, 0.0, -12.0 * SQRT_2];
        array_approx_eq(&kelvin_vector(&tt), correct, 1e-14);
    }
}
