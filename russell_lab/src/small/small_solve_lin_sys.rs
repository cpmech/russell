use super::{SmallMatrix, SmallVector, num_recipes_gaussj_sol};
use crate::StrError;
use num_traits::Float;

/// Solves a linear system with a single right-hand side (real numbers)
///
/// For a square matrix `a`, find `x` such that:
///
/// ```text
///   a   ⋅  x  =  b
/// (N,N)   (N)   (N)
/// ```
///
/// However, the right-hand-side will hold the solution:
///
/// ```text
/// b := a⁻¹⋅b == x
/// ```
///
/// The solution is obtained via Gauss-Jordan elimination with full pivoting
/// (see [`num_recipes_gaussj_sol`]).
///
/// See also: [`crate::solve_lin_sys`] (the heap-allocated counterpart).
///
/// # Note
///
/// 1. The matrix `a` will be modified (replaced by its inverse)
/// 2. The right-hand-side `b` will contain the solution `x`
///
/// # Errors
///
/// Returns an error if the matrix is singular.
///
/// # Examples
///
/// ```
/// use russell_lab::{small_solve_lin_sys, StrError};
///
/// fn main() -> Result<(), StrError> {
///     // set matrix and right-hand side
///     let mut a: [[f64; 3]; 3] = [
///         [1.0,  3.0, -2.0],
///         [3.0,  5.0,  6.0],
///         [2.0,  4.0,  3.0],
///     ];
///     let mut b: [f64; 3] = [5.0, 7.0, 8.0];
///
///     // solve linear system b := a⁻¹⋅b
///     small_solve_lin_sys(&mut b, &mut a)?;
///
///     // check
///     let x_correct: [f64; 3] = [-15.0, 8.0, 2.0];
///     for i in 0..3 {
///         assert!((b[i] - x_correct[i]).abs() < 1e-14);
///     }
///     Ok(())
/// }
/// ```
pub fn small_solve_lin_sys<T, const N: usize>(
    b: &mut SmallVector<T, N>,
    a: &mut SmallMatrix<T, N>,
) -> Result<(), StrError>
where
    T: Float,
{
    // wrap the right-hand-side as a single-column matrix
    let mut bb = [[T::zero(); 1]; N];
    for i in 0..N {
        bb[i][0] = b[i];
    }

    // solve the linear system (a is replaced by its inverse; bb by the solution)
    num_recipes_gaussj_sol(a, &mut bb)?;

    // copy the solution back into the vector
    for i in 0..N {
        b[i] = bb[i][0];
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::small_solve_lin_sys;
    use crate::array_approx_eq;

    #[test]
    fn small_solve_lin_sys_0x0_works() {
        let mut a: [[f64; 0]; 0] = [];
        let mut b: [f64; 0] = [];
        small_solve_lin_sys(&mut b, &mut a).unwrap();
    }

    #[test]
    fn small_solve_lin_sys_works() {
        // 5×5 system (copied from solve_lin_sys)
        #[rustfmt::skip]
        let mut a: [[f64; 5]; 5] = [
            [2.0, 1.0, 1.0, 3.0, 2.0],
            [1.0, 2.0, 2.0, 1.0, 1.0],
            [1.0, 2.0, 9.0, 1.0, 5.0],
            [3.0, 1.0, 1.0, 7.0, 1.0],
            [2.0, 1.0, 5.0, 1.0, 8.0],
        ];
        let mut b: [f64; 5] = [-2.0, 4.0, 3.0, -5.0, 1.0];
        small_solve_lin_sys(&mut b, &mut a).unwrap();
        #[rustfmt::skip]
        let x_correct = [
            -629.0 / 98.0,
             237.0 / 49.0,
             -53.0 / 49.0,
              62.0 / 49.0,
              23.0 / 14.0,
        ];
        array_approx_eq(&b, &x_correct, 1e-13);
    }

    #[test]
    fn small_solve_lin_sys_1_works() {
        // example from https://numericalalgorithmsgroup.github.io/LAPACK_Examples/examples/doc/dgesv_example.html
        #[rustfmt::skip]
        let mut a: [[f64; 4]; 4] = [
            [ 1.80,  2.88,  2.05, -0.89],
            [ 5.25, -2.95, -0.95, -3.80],
            [ 1.58, -2.69, -2.90, -1.04],
            [-1.11, -0.66, -0.59,  0.80],
        ];
        let mut b: [f64; 4] = [9.52, 24.35, 0.77, -6.22];
        small_solve_lin_sys(&mut b, &mut a).unwrap();
        let x_correct = [1.0, -1.0, 3.0, -5.0];
        array_approx_eq(&b, &x_correct, 1e-13);
    }

    /// Checks the solution of a·x = b where a is filled with 1.0 on/below the
    /// diagonal and -1.0 above it, and b is filled with a constant (x = [c, 0, ...]).
    fn check_constant_rhs<const N: usize>() {
        const TARGET: f64 = 1234.0;
        let mut a = [[1.0; N]; N];
        for i in 0..N {
            for j in (i + 1)..N {
                a[i][j] = -1.0;
            }
        }
        let mut b = [TARGET; N];
        let a_copy = a;
        let b_copy = b;
        small_solve_lin_sys(&mut b, &mut a).unwrap();
        // the solution is x = [TARGET, 0, 0, ...]
        for i in 0..N {
            let correct = if i == 0 { TARGET } else { 0.0 };
            assert!((b[i] - correct).abs() < 1e-13, "b[{i}] = {}", b[i]);
        }
        // check that a_copy * x == b_copy (with x == b)
        for i in 0..N {
            let mut sum = 0.0;
            for k in 0..N {
                sum += a_copy[i][k] * b[k];
            }
            assert!((sum - b_copy[i]).abs() < 1e-13, "a*x[{i}] = {sum}");
        }
    }

    #[test]
    fn small_solve_lin_sys_1x1_works() {
        check_constant_rhs::<1>();
    }

    #[test]
    fn small_solve_lin_sys_5x5_constant_rhs_works() {
        check_constant_rhs::<5>();
    }

    #[test]
    fn small_solve_lin_sys_7x7_constant_rhs_works() {
        check_constant_rhs::<7>();
    }

    #[test]
    fn small_solve_lin_sys_12x12_constant_rhs_works() {
        check_constant_rhs::<12>();
    }

    #[test]
    fn small_solve_lin_sys_singular_handles_error() {
        let mut a: [[f64; 2]; 2] = [
            [0.0, 0.0], //
            [0.0, 1.0], //
        ];
        let mut b: [f64; 2] = [1.0, 1.0];
        assert_eq!(small_solve_lin_sys(&mut b, &mut a).err(), Some("matrix is singular"));
    }
}
