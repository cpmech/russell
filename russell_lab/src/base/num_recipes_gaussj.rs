use crate::StrError;

// Raw binding to the C function `num_recipes_gaussj` (Gauss-Jordan elimination
// with full pivoting from Numerical Recipes).
//
// Note: `a` is an (n×n) row-major matrix replaced by its inverse. `b` is an
// (n×m) row-major matrix replaced by the solutions; it may be NULL when `m = 0`.
unsafe extern "C" {
    #[link_name = "num_recipes_gaussj"]
    fn num_recipes_gaussj_c(a: *mut f64, n: i32, b: *mut f64, m: i32) -> i32;
}

/// Converts the C status code to a `Result`
fn status_to_result(status: i32) -> Result<(), StrError> {
    match status {
        0 => Ok(()),
        1 => Err("matrix is singular"),
        _ => Err("memory allocation failed"),
    }
}

/// Computes the inverse of a small square matrix using Gauss-Jordan elimination
/// with full pivoting
///
/// The inverse is computed **in place**: on output, `a` holds `a⁻¹`.
///
/// ```text
/// a := a⁻¹
/// ```
///
/// # Reference
///
/// Press, W.H., Teukolsky, S.A., Vetterling, W.T., and Flannery, B.P. (2007)
/// Numerical Recipes: The Art of Scientific Computing, 3rd Edition,
/// Cambridge University Press. (Section 2.1)
///
/// # Input
///
/// * `a` -- the (N,N) square matrix, symmetric or not; it is overwritten with
///   its inverse.
///
/// # Errors
///
/// Returns an error if the matrix is singular.
///
/// # Examples
///
/// ```
/// use russell_lab::{num_recipes_gaussj_inv, StrError};
///
/// fn main() -> Result<(), StrError> {
///     let a_original = [
///         [1.0, 2.0, 3.0],
///         [0.0, 4.0, 5.0],
///         [1.0, 0.0, 6.0],
///     ];
///     let mut a = a_original;
///     num_recipes_gaussj_inv(&mut a)?;
///     // check that a_original * a == identity
///     for i in 0..3 {
///         for j in 0..3 {
///             let mut sum = 0.0;
///             for k in 0..3 {
///                 sum += a_original[i][k] * a[k][j];
///             }
///             let correct = if i == j { 1.0 } else { 0.0 };
///             assert!((sum - correct).abs() < 1e-14);
///         }
///     }
///     Ok(())
/// }
/// ```
pub fn num_recipes_gaussj_inv<const N: usize>(a: &mut [[f64; N]; N]) -> Result<(), StrError> {
    let status = unsafe {
        num_recipes_gaussj_c(a.as_mut_ptr().cast::<f64>(), N as i32, std::ptr::null_mut(), 0)
    };
    status_to_result(status)
}

/// Solves the linear systems A·X = B using Gauss-Jordan elimination with full
/// pivoting
///
/// On output, `a` is replaced by its inverse `a⁻¹`, and `b` is replaced by the
/// corresponding solution vectors `X`.
///
/// ```text
/// a := a⁻¹
/// b := X,  where  X = a⁻¹·b
/// ```
///
/// # Reference
///
/// Press, W.H., Teukolsky, S.A., Vetterling, W.T., and Flannery, B.P. (2007)
/// Numerical Recipes: The Art of Scientific Computing, 3rd Edition,
/// Cambridge University Press. (Section 2.1)
///
/// # Input
///
/// * `a` -- the (N,N) square matrix, symmetric or not; it is overwritten with
///   its inverse.
/// * `b` -- the (N,M) right-hand side matrix; it is overwritten with the
///   solution matrix.
///
/// # Errors
///
/// Returns an error if the matrix is singular.
///
/// # Examples
///
/// ```
/// use russell_lab::{num_recipes_gaussj_sol, StrError};
///
/// fn main() -> Result<(), StrError> {
///     let a_original = [
///         [2.0, 1.0],
///         [1.0, 2.0],
///     ];
///     let b_original = [
///         [3.0],
///         [3.0],
///     ];
///     let mut a = a_original;
///     let mut b = b_original;
///     num_recipes_gaussj_sol(&mut a, &mut b)?;
///     // solution x = [1, 1]
///     assert!((b[0][0] - 1.0).abs() < 1e-14);
///     assert!((b[1][0] - 1.0).abs() < 1e-14);
///     Ok(())
/// }
/// ```
pub fn num_recipes_gaussj_sol<const N: usize, const M: usize>(
    a: &mut [[f64; N]; N],
    b: &mut [[f64; M]; N],
) -> Result<(), StrError> {
    let status = unsafe {
        num_recipes_gaussj_c(
            a.as_mut_ptr().cast::<f64>(),
            N as i32,
            b.as_mut_ptr().cast::<f64>(),
            M as i32,
        )
    };
    status_to_result(status)
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{num_recipes_gaussj_inv, num_recipes_gaussj_sol};

    /// Checks that `a * ai` is (approximately) the identity matrix
    fn check_inverse<const N: usize>(a: &[[f64; N]; N], ai: &[[f64; N]; N], tol: f64) {
        for i in 0..N {
            for j in 0..N {
                let mut sum = 0.0;
                for k in 0..N {
                    sum += a[i][k] * ai[k][j];
                }
                let correct = if i == j { 1.0 } else { 0.0 };
                assert!((sum - correct).abs() <= tol, "a*ai[{i}][{j}] = {sum}");
            }
        }
    }

    /// Checks that `a * x == b` (single right-hand side)
    fn check_solution<const N: usize>(a: &[[f64; N]; N], x: &[[f64; 1]; N], b: &[[f64; 1]; N], tol: f64) {
        for i in 0..N {
            let mut sum = 0.0;
            for k in 0..N {
                sum += a[i][k] * x[k][0];
            }
            assert!((sum - b[i][0]).abs() <= tol, "a*x[{i}] = {sum}");
        }
    }

    /// Checks the solution of A·x = b where A is filled with 1.0 on/below the
    /// diagonal and -1.0 above it, and b is filled with a constant (x = [c, 0, ...]).
    fn check_constant_rhs<const N: usize>() {
        const TARGET: f64 = 1234.0;
        let mut a = [[1.0; N]; N];
        for i in 0..N {
            for j in (i + 1)..N {
                a[i][j] = -1.0;
            }
        }
        let mut b = [[TARGET; 1]; N];
        let a_original = a;
        let b_original = b;
        num_recipes_gaussj_sol(&mut a, &mut b).unwrap();
        // the solution is x = [TARGET, 0, 0, ...]
        for i in 0..N {
            let correct = if i == 0 { TARGET } else { 0.0 };
            assert!((b[i][0] - correct).abs() < 1e-13, "b[{i}] = {}", b[i][0]);
        }
        check_solution(&a_original, &b, &b_original, 1e-13);
    }

    #[test]
    fn inv_1x1_works() {
        let mut a = [[2.0]];
        num_recipes_gaussj_inv(&mut a).unwrap();
        assert_eq!(a, [[0.5]]);
    }

    #[test]
    fn inv_3x3_works() {
        let data = [
            [1.0, 2.0, 3.0],
            [0.0, 4.0, 5.0],
            [1.0, 0.0, 6.0],
        ];
        let mut a = data;
        num_recipes_gaussj_inv(&mut a).unwrap();
        check_inverse(&data, &a, 1e-14);
    }

    #[test]
    fn inv_fails_on_singular() {
        let mut a = [
            [1.0, 2.0],
            [2.0, 4.0],
        ];
        assert_eq!(num_recipes_gaussj_inv(&mut a).err(), Some("matrix is singular"));
    }

    #[test]
    fn sol_2x2_works() {
        let a_original = [
            [2.0, 1.0],
            [1.0, 2.0],
        ];
        let b_original = [
            [3.0],
            [3.0],
        ];
        let mut a = a_original;
        let mut b = b_original;
        num_recipes_gaussj_sol(&mut a, &mut b).unwrap();
        // solution x = [1, 1]
        assert!((b[0][0] - 1.0).abs() < 1e-14);
        assert!((b[1][0] - 1.0).abs() < 1e-14);
        // a becomes the inverse
        check_inverse(&a_original, &a, 1e-14);
    }

    #[test]
    fn sol_3x3_multiple_rhs_works() {
        let a_original = [
            [1.0, 2.0, 3.0],
            [0.0, 4.0, 5.0],
            [1.0, 0.0, 6.0],
        ];
        let b_original = [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ];
        let mut a = a_original;
        let mut b = b_original;
        num_recipes_gaussj_sol(&mut a, &mut b).unwrap();
        // check that a * x == b
        for i in 0..3 {
            for j in 0..2 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += a_original[i][k] * b[k][j];
                }
                assert!((sum - b_original[i][j]).abs() < 1e-14, "a*x[{i}][{j}] = {sum}");
            }
        }
        // a becomes the inverse
        check_inverse(&a_original, &a, 1e-14);
    }

    #[test]
    fn sol_fails_on_singular() {
        let mut a = [
            [1.0, 2.0],
            [2.0, 4.0],
        ];
        let mut b = [
            [1.0],
            [2.0],
        ];
        assert_eq!(num_recipes_gaussj_sol(&mut a, &mut b).err(), Some("matrix is singular"));
    }

    #[test]
    fn sol_0x0_works() {
        let mut a = [[0.0; 0]; 0];
        let mut b = [[0.0; 1]; 0];
        num_recipes_gaussj_sol(&mut a, &mut b).unwrap();
    }

    #[test]
    fn sol_4x4_works() {
        // example from https://numericalalgorithmsgroup.github.io/LAPACK_Examples/examples/doc/dgesv_example.html
        let a_original = [
            [1.80, 2.88, 2.05, -0.89],
            [5.25, -2.95, -0.95, -3.80],
            [1.58, -2.69, -2.90, -1.04],
            [-1.11, -0.66, -0.59, 0.80],
        ];
        let b_original = [
            [9.52],
            [24.35],
            [0.77],
            [-6.22],
        ];
        let mut a = a_original;
        let mut b = b_original;
        num_recipes_gaussj_sol(&mut a, &mut b).unwrap();
        let x_correct = [
            [1.0],
            [-1.0],
            [3.0],
            [-5.0],
        ];
        for i in 0..4 {
            assert!((b[i][0] - x_correct[i][0]).abs() < 1e-13, "b[{i}] = {}", b[i][0]);
        }
        check_solution(&a_original, &b, &b_original, 1e-13);
    }

    #[test]
    fn sol_5x5_works() {
        let a_original = [
            [2.0, 1.0, 1.0, 3.0, 2.0],
            [1.0, 2.0, 2.0, 1.0, 1.0],
            [1.0, 2.0, 9.0, 1.0, 5.0],
            [3.0, 1.0, 1.0, 7.0, 1.0],
            [2.0, 1.0, 5.0, 1.0, 8.0],
        ];
        let b_original = [
            [-2.0],
            [4.0],
            [3.0],
            [-5.0],
            [1.0],
        ];
        let mut a = a_original;
        let mut b = b_original;
        num_recipes_gaussj_sol(&mut a, &mut b).unwrap();
        let x_correct = [
            [-629.0 / 98.0],
            [237.0 / 49.0],
            [-53.0 / 49.0],
            [62.0 / 49.0],
            [23.0 / 14.0],
        ];
        for i in 0..5 {
            assert!((b[i][0] - x_correct[i][0]).abs() < 1e-13, "b[{i}] = {}", b[i][0]);
        }
        check_solution(&a_original, &b, &b_original, 1e-13);
    }

    #[test]
    fn sol_1x1_works() {
        check_constant_rhs::<1>();
    }

    #[test]
    fn sol_5x5_constant_rhs_works() {
        check_constant_rhs::<5>();
    }

    #[test]
    fn sol_7x7_constant_rhs_works() {
        check_constant_rhs::<7>();
    }

    #[test]
    fn sol_12x12_constant_rhs_works() {
        check_constant_rhs::<12>();
    }

    #[test]
    fn sol_singular_handles_error() {
        let mut a = [
            [0.0, 0.0],
            [0.0, 1.0],
        ];
        let mut b = [
            [1.0],
            [1.0],
        ];
        assert_eq!(num_recipes_gaussj_sol(&mut a, &mut b).err(), Some("matrix is singular"));
    }
}
