use crate::{StrError, num_recipes_gaussj_inv};

/// Zero-determinant tolerance used to detect a singular matrix
const ZERO_DETERMINANT: f64 = 1e-15;

/// Computes the inverse of a small square matrix using Gauss-Jordan elimination
///
/// ```text
/// ai := a⁻¹
/// ```
///
/// Two pivoting strategies are available:
///
/// * Partial (row) pivoting (the default) -- implemented in Rust. The algorithm
///   uses analytical formulas for n ≤ 3 and augments `a` with the identity matrix
///   (reducing `[a | I]` to `[I | a⁻¹]`) for n ≥ 4.
/// * Full pivoting (`full_pivot = true`) -- delegates to `num_recipes_gaussj_inv`,
///   which wraps the full-pivoting algorithm from Numerical Recipes (compiled C code).
///   This is only applied for n ≥ 4, because the analytical solutions handle n ≤ 3.
///
/// # Input
///
/// * `ai` -- the (N,N) matrix that will hold the inverse
/// * `a` -- the (N,N) square matrix, symmetric or not
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
///     let a = [
///         [1.0, 2.0, 3.0],
///         [0.0, 4.0, 5.0],
///         [1.0, 0.0, 6.0],
///     ];
///     let mut ai = [[0.0; 3]; 3];
///     small_mat_inv(&mut ai, &a, false)?;
///     // check that a * ai == identity
///     for i in 0..3 {
///         for j in 0..3 {
///             let mut sum = 0.0;
///             for k in 0..3 {
///                 sum += a[i][k] * ai[k][j];
///             }
///             let correct = if i == j { 1.0 } else { 0.0 };
///             assert!((sum - correct).abs() < 1e-14);
///         }
///     }
///     Ok(())
/// }
/// ```
pub fn small_mat_inv<const N: usize>(
    ai: &mut [[f64; N]; N],
    a: &[[f64; N]; N],
    full_pivot: bool,
) -> Result<(), StrError> {
    // Analytical solution for a 1×1 matrix
    if N == 1 {
        let det = a[0][0];
        if det.abs() <= ZERO_DETERMINANT {
            return Err("matrix is singular");
        }
        ai[0][0] = 1.0 / det;
        return Ok(());
    }

    // Analytical solution for a 2×2 matrix
    if N == 2 {
        let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
        if det.abs() <= ZERO_DETERMINANT {
            return Err("matrix is singular");
        }
        ai[0][0] = a[1][1] / det;
        ai[0][1] = -a[0][1] / det;
        ai[1][0] = -a[1][0] / det;
        ai[1][1] = a[0][0] / det;
        return Ok(());
    }

    // Analytical solution for a 3×3 matrix
    if N == 3 {
        let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
        if det.abs() <= ZERO_DETERMINANT {
            return Err("matrix is singular");
        }
        ai[0][0] = (a[1][1] * a[2][2] - a[1][2] * a[2][1]) / det;
        ai[0][1] = (a[0][2] * a[2][1] - a[0][1] * a[2][2]) / det;
        ai[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) / det;
        ai[1][0] = (a[1][2] * a[2][0] - a[1][0] * a[2][2]) / det;
        ai[1][1] = (a[0][0] * a[2][2] - a[0][2] * a[2][0]) / det;
        ai[1][2] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) / det;
        ai[2][0] = (a[1][0] * a[2][1] - a[1][1] * a[2][0]) / det;
        ai[2][1] = (a[0][1] * a[2][0] - a[0][0] * a[2][1]) / det;
        ai[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) / det;
        return Ok(());
    }

    // Use full pivoting when requested. This delegates to `num_recipes_gaussj_inv`
    // (the Numerical Recipes algorithm, compiled C) and is only reached for n ≥ 4,
    // because the analytical solutions above handle n ≤ 3.
    if full_pivot {
        *ai = *a;
        return num_recipes_gaussj_inv(ai);
    }

    // Gauss-Jordan elimination with partial (row) pivoting (n ≥ 4)
    //
    // Copy `a` into a working matrix and set `ai` to the identity.
    let mut a_work = *a;
    for i in 0..N {
        for j in 0..N {
            ai[i][j] = 0.0;
        }
        ai[i][i] = 1.0;
    }

    for k in 0..N {
        // Find the pivot: the largest |entry| in column k, over rows k..N
        let mut max_index = k;
        let mut max_value = a_work[k][k].abs();
        for i in (k + 1)..N {
            let value = a_work[i][k].abs();
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
            a_work.swap(k, max_index);
            ai.swap(k, max_index);
        }

        // Normalize the pivot row (in both sides), forcing the exact diagonal
        let pivot = a_work[k][k];

        // Hint to the compiler that loops match N for SIMD auto-vectorization
        let row_a = &mut a_work[k];
        let row_ai = &mut ai[k];
        for j in 0..N {
            row_a[j] /= pivot;
            row_ai[j] /= pivot;
        }
        row_a[k] = 1.0;

        // Eliminate the pivot column from all other rows (in both sides)
        for i in 0..N {
            if i != k {
                let factor = a_work[i][k];
                if factor != 0.0 {
                    // Extracting slice references allows LLVM to eliminate bounds checks
                    // and safely apply target SIMD instructions (AVX2/AVX-512)
                    let (target_a, source_a) = if i < k {
                        let (left, right) = a_work.split_at_mut(k);
                        (&mut left[i], &right[0])
                    } else {
                        let (left, right) = a_work.split_at_mut(i);
                        (&mut right[0], &left[k])
                    };

                    let (target_ai, source_ai) = if i < k {
                        let (left, right) = ai.split_at_mut(k);
                        (&mut left[i], &right[0])
                    } else {
                        let (left, right) = ai.split_at_mut(i);
                        (&mut right[0], &left[k])
                    };

                    for j in 0..N {
                        target_a[j] -= factor * source_a[j];
                        target_ai[j] -= factor * source_ai[j];
                    }
                    target_a[k] = 0.0;
                }
            }
        }
    }

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
        let a = [[2.0]];
        let mut ai = [[0.0; 1]; 1];
        small_mat_inv(&mut ai, &a, false).unwrap();
        assert_eq!(ai, [[0.5]]);
        check_inverse(&a, &ai, 1e-15);
    }

    #[test]
    fn inverse_1x1_fails_on_zero_det() {
        let a = [[0.0]];
        let mut ai = [[0.0; 1]; 1];
        assert_eq!(small_mat_inv(&mut ai, &a, false).err(), Some("matrix is singular"));
    }

    #[test]
    fn inverse_2x2_works() {
        #[rustfmt::skip]
        let data = [
            [1.0, 2.0],
            [3.0, 2.0],
        ];
        let a = data;
        let mut ai = [[0.0; 2]; 2];
        small_mat_inv(&mut ai, &a, false).unwrap();
        #[rustfmt::skip]
        let ai_correct = [
            [-0.5, 0.5],
            [0.75, -0.25],
        ];
        check_matrix(&ai, &ai_correct, 1e-15);
        check_inverse(&data, &ai, 1e-15);
    }

    #[test]
    fn inverse_2x2_fails_on_zero_det() {
        #[rustfmt::skip]
        let a = [
            [   -1.0, 3.0/2.0],
            [2.0/3.0,    -1.0],
        ];
        let mut ai = [[0.0; 2]; 2];
        assert_eq!(small_mat_inv(&mut ai, &a, false).err(), Some("matrix is singular"));
    }

    #[test]
    fn inverse_3x3_works() {
        #[rustfmt::skip]
        let data = [
            [1.0, 2.0, 3.0],
            [0.0, 4.0, 5.0],
            [1.0, 0.0, 6.0],
        ];
        let a = data;
        let mut ai = [[0.0; 3]; 3];
        small_mat_inv(&mut ai, &a, false).unwrap();
        #[rustfmt::skip]
        let ai_correct = [
            [12.0/11.0, -6.0/11.0, -1.0/11.0],
            [ 2.5/11.0,  1.5/11.0, -2.5/11.0],
            [-2.0/11.0,  1.0/11.0,  2.0/11.0],
        ];
        check_matrix(&ai, &ai_correct, 1e-15);
        check_inverse(&data, &ai, 1e-15);
    }

    #[test]
    fn inverse_3x3_fails_on_zero_det() {
        #[rustfmt::skip]
        let a = [
            [1.0, 0.0, 3.0],
            [0.0, 0.0, 5.0],
            [1.0, 0.0, 6.0],
        ];
        let mut ai = [[0.0; 3]; 3];
        assert_eq!(small_mat_inv(&mut ai, &a, false).err(), Some("matrix is singular"));
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
        let a = data;
        let mut ai = [[0.0; 4]; 4];
        small_mat_inv(&mut ai, &a, false).unwrap();
        #[rustfmt::skip]
        let ai_correct = [
            [ 0.6,  0.0, -0.2,  0.0],
            [-2.5,  0.5,  0.5,  1.0],
            [-1.5,  0.0,  0.5,  0.5],
            [-2.2,  0.0,  0.4,  1.0],
        ];
        check_matrix(&ai, &ai_correct, 1e-15);
        check_inverse(&data, &ai, 1e-15);
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
        let a = data;
        let mut ai = [[0.0; 5]; 5];
        small_mat_inv(&mut ai, &a, false).unwrap();
        #[rustfmt::skip]
        let ai_correct = [
            [ 6.9128803717996279e-01, -7.4226114383340802e-01, -9.8756287260606410e-02, -6.9062496266472417e-01,  7.2471057693456553e-01],
            [ 1.5936129795342968e+00, -1.7482347881148397e+00, -2.8304321334273236e-01, -1.5600769405383470e+00,  1.7164430532490673e+00],
            [-1.6345384165063759e+00,  1.7495848317224429e+00,  2.7469205863729274e-01,  1.6325730875377857e+00, -1.7065745928961444e+00],
            [-1.1177465024312745e+00,  1.3261729250546601e+00,  2.1243473793622566e-01,  1.1258168958554866e+00, -1.3325766717243535e+00],
            [ 7.9976941733073770e-01, -8.9457712572131853e-01, -1.4770432850264653e-01, -8.0791149448632715e-01,  9.2990525800169743e-01],
        ];
        check_matrix(&ai, &ai_correct, 1e-13);
        check_inverse(&data, &ai, 1e-12);
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
        let a = data;
        let mut ai = [[0.0; 6]; 6];
        small_mat_inv(&mut ai, &a, false).unwrap();
        #[rustfmt::skip]
        let ai_correct = &[
            [ 6.28811662297464645e+04,  4.23011662297464645e+04,  4.23011662297464645e+04, 0.00000000000000000e+00, -1.05591885817167332e-17, 4.33037966311565489e+07],
            [ 4.23011662297464645e+04,  6.28811662297464645e+04,  4.23011662297464645e+04, 0.00000000000000000e+00, -1.05591885817167332e-17, 4.33037966311565489e+07],
            [ 4.23011662297464645e+04,  4.23011662297464645e+04,  6.28811662297464645e+04, 0.00000000000000000e+00, -1.05591885817167348e-17, 4.33037966311565489e+07],
            [ 0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00, 2.05800000000000000e+04,  0.00000000000000000e+00, 0.00000000000000000e+00],
            [-4.62744616057000471e-13, -4.62744616057000471e-13, -4.62744616057000471e-13, 0.00000000000000000e+00,  1.00000000000000000e+00, 1.93012141894243434e+07],
            [ 0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00, 0.00000000000000000e+00,  0.00000000000000000e+00, 1.00000000000000000e+00],
        ];
        check_matrix(&ai, &ai_correct, 1e-15);
        check_inverse(&data, &ai, 1e-13);
    }

    #[test]
    fn inverse_full_pivot_works() {
        let data = [[1.0, 2.0, 3.0], [0.0, 4.0, 5.0], [1.0, 0.0, 6.0]];
        let a = data;
        let mut ai = [[0.0; 3]; 3];
        small_mat_inv(&mut ai, &a, true).unwrap();
        check_inverse(&data, &ai, 1e-14);
    }

    // --- extra tests ---

    /// A simple, deterministic pseudo-random number generator for test generation.
    /// Avoids external dependencies like the `rand` crate.
    /// To generate random entries without external dependencies (like the rand crate),
    /// this code includes a minimal Linear Congruential Generator (LCG) so
    /// it can run immediately as a zero-dependency test.
    fn simple_rng(seed: &mut u32) -> f64 {
        *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        // Normalize to a range of [-1.0, 1.0]
        ((*seed as f64) / (u32::MAX as f64)) * 2.0 - 1.0
    }

    /// Macro to generate matrix inversion validation tests for any dimension `N`.
    macro_rules! generate_matrix_inv_test {
        ($test_name:ident, $size:expr) => {
            #[test]
            fn $test_name() {
                const N: usize = $size;
                let mut seed = 42u32; // Fixed seed for reproducible test runs

                // 1. Generate a random dense matrix
                let mut original = [[0.0; N]; N];
                for i in 0..N {
                    for j in 0..N {
                        original[i][j] = simple_rng(&mut seed);
                    }
                    // Ensure the matrix is diagonally dominant so it's invertible
                    original[i][i] += (N as f64) * 2.0;
                }

                // 2. Compute the inverse
                let mut inverted = [[0.0; N]; N];
                let result = small_mat_inv(&mut inverted, &original, false);

                assert!(result.is_ok(), "Matrix inversion failed for size {}", N);

                // 3. Verify Identity: [Original] * [Inverted] == [I]
                for i in 0..N {
                    for j in 0..N {
                        let mut sum = 0.0;
                        for k in 0..N {
                            sum += original[i][k] * inverted[k][j];
                        }

                        let expected = if i == j { 1.0 } else { 0.0 };
                        let difference = (sum - expected).abs();

                        assert!(
                            difference < 1e-11,
                            "Identity verification failed at [{}, {}] for size {}. Expected {}, got {}. Diff: {}",
                            i,
                            j,
                            N,
                            expected,
                            sum,
                            difference
                        );
                    }
                }
            }
        };
    }

    // Automatically expand and generate test cases for multiple dimensions
    generate_matrix_inv_test!(test_matrix_inverse_3x3, 3);
    generate_matrix_inv_test!(test_matrix_inverse_6x6, 6);
    generate_matrix_inv_test!(test_matrix_inverse_9x9, 9);
    generate_matrix_inv_test!(test_matrix_inverse_12x12, 12);
}
