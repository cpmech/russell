use super::Tensor2;
use crate::{Rep, SQRT_2};

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
/// * `c` -- the resulting tensor; it must be [Rep::General]
///
/// # Input
///
/// * `a` -- first tensor; with the same [Rep] as `b`
/// * `b` -- second tensor; with the same [Rep] as `a`
///
/// # Panics
///
/// 1. A panic will occur if `c` is not [Rep::General]
/// 2. A panic will occur if the `a` and `b` have different [Rep]
///
/// # Examples
///
/// ```
/// use russell_tensor::{t2_dot_t2, Rep, Tensor2, StrError};
///
/// fn main() -> Result<(), StrError> {
///     let a = Tensor2::from_std_matrix(&[
///         [1.0,  1.0, 0.0],
///         [1.0, -1.0, 0.0],
///         [0.0,  0.0, 1.0],
///     ], Rep::General)?;
///
///     let b = Tensor2::from_std_matrix(&[
///         [1.0,  2.0, 0.0],
///         [3.0, -1.0, 5.0],
///         [0.0,  4.0, 1.0],
///     ], Rep::General)?;
///
///     let mut c = Tensor2::new(Rep::General);
///     t2_dot_t2(&mut c, &a, &b);
///     assert_eq!(
///         format!("{:.1}", c.as_std_matrix()),
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
    assert_eq!(c.rep(), Rep::General);
    assert_eq!(b.rep(), a.rep());
    let dim = a.dim();
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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Rep, Tensor2};
    use russell_lab::mat_approx_eq;

    #[test]
    #[should_panic]
    fn t2_dot_t2_panics_on_non_general() {
        let a = Tensor2::new(Rep::Symmetric);
        let b = Tensor2::new(Rep::Symmetric);
        let mut c = Tensor2::new(Rep::Symmetric); // wrong; it must be General
        t2_dot_t2(&mut c, &a, &b);
    }

    #[test]
    #[should_panic]
    fn t2_dot_t2_panics_on_different_rep() {
        let a = Tensor2::new(Rep::Symmetric);
        let b = Tensor2::new(Rep::General); // wrong; it must be the same as `a`
        let mut c = Tensor2::new(Rep::General);
        t2_dot_t2(&mut c, &a, &b);
    }

    #[test]
    fn t2_dot_t2_works() {
        // general . general
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
        let mut c = Tensor2::new(Rep::General);
        t2_dot_t2(&mut c, &a, &b);
        #[rustfmt::skip]
        let correct = [
            [ 30.0,  24.0, 18.0],
            [ 84.0,  69.0, 54.0],
            [138.0, 114.0, 90.0],
        ];
        mat_approx_eq(&c.as_std_matrix(), &correct, 1e-13);

        // sym-3D . sym-3D
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
        let mut c = Tensor2::new(Rep::General);
        t2_dot_t2(&mut c, &a, &b);
        #[rustfmt::skip]
        let correct = [
            [59.0, 37.0, 28.0],
            [52.0, 44.0, 37.0],
            [61.0, 52.0, 59.0],
        ];
        mat_approx_eq(&c.as_std_matrix(), &correct, 1e-13);

        // sym-2D . sym-2D
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
        let mut c = Tensor2::new(Rep::General);
        t2_dot_t2(&mut c, &a, &b);
        #[rustfmt::skip]
        let correct = [
            [23.0, 13.0, 0.0],
            [22.0, 24.0, 0.0],
            [ 0.0,  0.0, 3.0],
        ];
        mat_approx_eq(&c.as_std_matrix(), &correct, 1e-13);
    }
}
