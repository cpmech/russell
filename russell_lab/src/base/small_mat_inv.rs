use crate::{StrError, num_recipes_gaussj_inv};

/// Zero-determinant tolerance used to detect a singular matrix
const ZERO_DETERMINANT: f64 = 1e-15;

/// Computes the inverse of a small square matrix using Gauss-Jordan elimination
///
/// The inverse is computed **in place**: on output, `a` holds `a⁻¹`.
///
/// ```text
/// a := a⁻¹
/// ```
///
/// Two pivoting strategies are available:
///
/// * Partial (row) pivoting (the default) -- implemented in Rust. The algorithm
///   augments `a` with the identity matrix and reduces `[a | I]` to `[I | a⁻¹]`.
/// * Full pivoting (`full_pivot = true`) -- delegates to `num_recipes_gaussj_inv`,
///   which wraps the full-pivoting algorithm from Numerical Recipes (compiled C code).
///
/// # Input
///
/// * `a` -- the (N,N) square matrix, symmetric or not; it is overwritten with
///   its inverse.
/// * `full_pivot` -- if true, use full pivoting; otherwise, use partial pivoting.
///
/// # Errors
///
/// Returns an error if the matrix is singular.
///
/// # Examples
///
/// ```
/// use russell_lab::{small_mat_inv, StrError};
///
/// fn main() -> Result<(), StrError> {
///     let a_original = [
///         [1.0, 2.0, 3.0],
///         [0.0, 4.0, 5.0],
///         [1.0, 0.0, 6.0],
///     ];
///     let mut a = a_original;
///     small_mat_inv(&mut a, false)?;
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
pub fn small_mat_inv<const N: usize>(a: &mut [[f64; N]; N], full_pivot: bool) -> Result<(), StrError> {
    // Use full pivoting (Numerical Recipes) when requested
    if full_pivot {
        return num_recipes_gaussj_inv(a);
    }

    // The right-hand side starts as the identity matrix and accumulates the inverse.
    // `a` (the left-hand side) is reduced in place toward the identity.
    let mut ai = [[0.0; N]; N];
    for i in 0..N {
        ai[i][i] = 1.0;
    }

    // Gauss-Jordan elimination with partial (row) pivoting
    for k in 0..N {
        // Find the pivot: the largest |entry| in column k, over rows k..N
        let mut max_index = k;
        let mut max_value = a[k][k].abs();
        for i in (k + 1)..N {
            let value = a[i][k].abs();
            if value > max_value {
                max_value = value;
                max_index = i;
            }
        }

        // Check for singularity
        if max_value <= ZERO_DETERMINANT {
            return Err("matrix is singular");
        }

        // Swap rows if necessary (in both sides)
        if max_index != k {
            a.swap(k, max_index);
            ai.swap(k, max_index);
        }

        // Normalize the pivot row (in both sides), forcing the exact diagonal
        let pivot = a[k][k];
        for j in 0..N {
            a[k][j] /= pivot;
            ai[k][j] /= pivot;
        }
        a[k][k] = 1.0;

        // Eliminate the pivot column from all other rows (in both sides)
        for i in 0..N {
            if i != k {
                let factor = a[i][k];
                if factor != 0.0 {
                    for j in 0..N {
                        a[i][j] -= factor * a[k][j];
                        ai[i][j] -= factor * ai[k][j];
                    }
                    a[i][k] = 0.0;
                }
            }
        }
    }

    // Overwrite `a` with the computed inverse
    *a = ai;
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::small_mat_inv;

    /// Checks that the two (N,N) matrices are approximately equal
    fn check_matrix<const N: usize>(a: &[[f64; N]; N], b: &[[f64; N]; N], tol: f64) {
        for i in 0..N {
            for j in 0..N {
                assert!(
                    (a[i][j] - b[i][j]).abs() <= tol,
                    "mismatch at [{i}][{j}]: {} != {}",
                    a[i][j],
                    b[i][j]
                );
            }
        }
    }

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

    #[test]
    fn inverse_1x1_works() {
        let mut a = [[2.0]];
        small_mat_inv(&mut a, false).unwrap();
        assert_eq!(a, [[0.5]]);
        check_inverse(&[[2.0]], &a, 1e-15);
    }

    #[test]
    fn inverse_1x1_fails_on_zero_det() {
        let mut a = [[0.0]];
        assert_eq!(
            small_mat_inv(&mut a, false).err(),
            Some("matrix is singular")
        );
    }

    #[test]
    fn inverse_2x2_works() {
        #[rustfmt::skip]
        let data = [
            [1.0, 2.0],
            [3.0, 2.0],
        ];
        let mut a = data;
        small_mat_inv(&mut a, false).unwrap();
        #[rustfmt::skip]
        let ai_correct = [
            [-0.5, 0.5],
            [0.75, -0.25],
        ];
        check_matrix(&a, &ai_correct, 1e-15);
        check_inverse(&data, &a, 1e-15);
    }

    #[test]
    fn inverse_2x2_fails_on_zero_det() {
        #[rustfmt::skip]
        let mut a = [
            [   -1.0, 3.0/2.0],
            [2.0/3.0,    -1.0],
        ];
        assert_eq!(
            small_mat_inv(&mut a, false).err(),
            Some("matrix is singular")
        );
    }

    #[test]
    fn inverse_3x3_works() {
        #[rustfmt::skip]
        let data = [
            [1.0, 2.0, 3.0],
            [0.0, 4.0, 5.0],
            [1.0, 0.0, 6.0],
        ];
        let mut a = data;
        small_mat_inv(&mut a, false).unwrap();
        #[rustfmt::skip]
        let ai_correct = [
            [12.0/11.0, -6.0/11.0, -1.0/11.0],
            [ 2.5/11.0,  1.5/11.0, -2.5/11.0],
            [-2.0/11.0,  1.0/11.0,  2.0/11.0],
        ];
        check_matrix(&a, &ai_correct, 1e-15);
        check_inverse(&data, &a, 1e-15);
    }

    #[test]
    fn inverse_3x3_fails_on_zero_det() {
        #[rustfmt::skip]
        let mut a = [
            [1.0, 0.0, 3.0],
            [0.0, 0.0, 5.0],
            [1.0, 0.0, 6.0],
        ];
        assert_eq!(
            small_mat_inv(&mut a, false).err(),
            Some("matrix is singular")
        );
    }

    #[test]
    fn inverse_4x4_works() {
        #[rustfmt::skip]
        let data = [
            [ 3.0,  0.0,  2.0, -1.0],
            [ 1.0,  2.0,  0.0, -2.0],
            [ 4.0,  0.0,  6.0, -3.0],
            [ 5.0,  0.0,  2.0,  0.0],
        ];
        let mut a = data;
        small_mat_inv(&mut a, false).unwrap();
        #[rustfmt::skip]
        let ai_correct = [
            [ 0.6,  0.0, -0.2,  0.0],
            [-2.5,  0.5,  0.5,  1.0],
            [-1.5,  0.0,  0.5,  0.5],
            [-2.2,  0.0,  0.4,  1.0],
        ];
        check_matrix(&a, &ai_correct, 1e-15);
        check_inverse(&data, &a, 1e-15);
    }

    #[test]
    fn inverse_5x5_works() {
        #[rustfmt::skip]
        let data = [
            [12.0, 28.0, 22.0, 20.0,  8.0],
            [ 0.0,  3.0,  5.0, 17.0, 28.0],
            [56.0,  0.0, 23.0,  1.0,  0.0],
            [12.0, 29.0, 27.0, 10.0,  1.0],
            [ 9.0,  4.0, 13.0,  8.0, 22.0],
        ];
        let mut a = data;
        small_mat_inv(&mut a, false).unwrap();
        #[rustfmt::skip]
        let ai_correct = [
            [ 6.9128803717996279e-01, -7.4226114383340802e-01, -9.8756287260606410e-02, -6.9062496266472417e-01,  7.2471057693456553e-01],
            [ 1.5936129795342968e+00, -1.7482347881148397e+00, -2.8304321334273236e-01, -1.5600769405383470e+00,  1.7164430532490673e+00],
            [-1.6345384165063759e+00,  1.7495848317224429e+00,  2.7469205863729274e-01,  1.6325730875377857e+00, -1.7065745928961444e+00],
            [-1.1177465024312745e+00,  1.3261729250546601e+00,  2.1243473793622566e-01,  1.1258168958554866e+00, -1.3325766717243535e+00],
            [ 7.9976941733073770e-01, -8.9457712572131853e-01, -1.4770432850264653e-01, -8.0791149448632715e-01,  9.2990525800169743e-01],
        ];
        check_matrix(&a, &ai_correct, 1e-13);
        check_inverse(&data, &a, 1e-12);
    }

    #[test]
    fn inverse_6x6_works() {
        // NOTE: this matrix is nearly non-invertible; it originated from an FEM analysis
        #[rustfmt::skip]
        let data = [
            [ 3.46540497998689445e-05, -1.39368151175265866e-05, -1.39368151175265866e-05,  0.00000000000000000e+00, 7.15957288480514429e-23, -2.93617909908697186e+02],
            [-1.39368151175265866e-05,  3.46540497998689445e-05, -1.39368151175265866e-05,  0.00000000000000000e+00, 7.15957288480514429e-23, -2.93617909908697186e+02],
            [-1.39368151175265866e-05, -1.39368151175265866e-05,  3.46540497998689445e-05,  0.00000000000000000e+00, 7.15957288480514429e-23, -2.93617909908697186e+02],
            [ 0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00,  4.85908649173955311e-05, 0.00000000000000000e+00,  0.00000000000000000e+00],
            [ 3.13760264822604860e-18,  3.13760264822604860e-18,  3.13760264822604860e-18,  0.00000000000000000e+00, 1.00000000000000000e+00, -1.93012141894243434e+07],
            [ 0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00, -0.00000000000000000e+00, 0.00000000000000000e+00,  1.00000000000000000e+00],
        ];
        let mut a = data;
        small_mat_inv(&mut a, false).unwrap();
        #[rustfmt::skip]
        let ai_correct = &[
            [ 6.28811662297464645e+04,  4.23011662297464645e+04,  4.23011662297464645e+04, 0.00000000000000000e+00, -1.05591885817167332e-17, 4.33037966311565489e+07],
            [ 4.23011662297464645e+04,  6.28811662297464645e+04,  4.23011662297464645e+04, 0.00000000000000000e+00, -1.05591885817167332e-17, 4.33037966311565489e+07],
            [ 4.23011662297464645e+04,  4.23011662297464645e+04,  6.28811662297464645e+04, 0.00000000000000000e+00, -1.05591885817167348e-17, 4.33037966311565489e+07],
            [ 0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00, 2.05800000000000000e+04,  0.00000000000000000e+00, 0.00000000000000000e+00],
            [-4.62744616057000471e-13, -4.62744616057000471e-13, -4.62744616057000471e-13, 0.00000000000000000e+00,  1.00000000000000000e+00, 1.93012141894243434e+07],
            [ 0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00, 0.00000000000000000e+00,  0.00000000000000000e+00, 1.00000000000000000e+00],
        ];
        check_matrix(&a, &ai_correct, 1e-15);
        check_inverse(&data, &a, 1e-13);
    }

    #[test]
    fn inverse_full_pivot_works() {
        let data = [
            [1.0, 2.0, 3.0],
            [0.0, 4.0, 5.0],
            [1.0, 0.0, 6.0],
        ];
        let mut a = data;
        small_mat_inv(&mut a, true).unwrap();
        check_inverse(&data, &a, 1e-14);
    }
}
