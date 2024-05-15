use russell_lab::Matrix;

/// Defines a trait to handle 9x9 matrices
///
/// # Examples
///
/// ```
/// use russell_lab::Matrix;
/// use russell_tensor::{AsMatrix9x9, MN_TO_IJKL};
///
/// fn diagonal(mat: &dyn AsMatrix9x9) -> Vec<f64> {
///     let mut res = vec![0.0; 9];
///     for i in 0..9 {
///         res[i] = mat.at(i, i);
///     }
///     res
/// }
///
/// // heap-allocated matrix (vector of vectors)
/// // ┌                                              ┐
/// // │ 1111 1122 1133 1112 1123 1113 1121 1132 1131 │
/// // │ 2211 2222 2233 2212 2223 2213 2221 2232 2231 │
/// // │ 3311 3322 3333 3312 3323 3313 3321 3332 3331 │
/// // │ 1211 1222 1233 1212 1223 1213 1221 1232 1231 │
/// // │ 2311 2322 2333 2312 2323 2313 2321 2332 2331 │
/// // │ 1311 1322 1333 1312 1323 1313 1321 1332 1331 │
/// // │ 2111 2122 2133 2112 2123 2113 2121 2132 2131 │
/// // │ 3211 3222 3233 3212 3223 3213 3221 3232 3231 │
/// // │ 3111 3122 3133 3112 3123 3113 3121 3132 3131 │
/// // └                                              ┘
/// let mut mat = vec![vec![0.0; 9]; 9];
/// for m in 0..9 {
///     for n in 0..9 {
///         let (i, j, k, l) = MN_TO_IJKL[m][n];
///         mat[m][n] = (1000 * (i + 1) + 100 * (j + 1) + 10 * (k + 1) + (l + 1)) as f64;
///     }
/// }
/// assert_eq!(
///     diagonal(&mat),
///     &[1111.0, 2222.0, 3333.0, 1212.0, 2323.0, 1313.0, 2121.0, 3232.0, 3131.0]
/// );
/// ```
pub trait AsMatrix9x9 {
    /// Returns the value at (i,j) indices
    ///
    /// # Panics
    ///
    /// This function panics if the indices are out of range.
    fn at(&self, i: usize, j: usize) -> f64;
}

/// Defines a heap-allocated 9x9 matrix (vector of vectors)
///
/// # Panics
///
/// * The array must be 9x9; otherwise a panic will occur.
/// * The methods may panic if the array is empty.
impl AsMatrix9x9 for Vec<Vec<f64>> {
    fn at(&self, i: usize, j: usize) -> f64 {
        self[i][j]
    }
}

/// Defines a heap-allocated 9x9 matrix (slice of slices)
///
/// # Panics
///
/// * The array must be 9x9; otherwise a panic will occur.
/// * The methods may panic if the array is empty.
impl AsMatrix9x9 for &[&[f64]] {
    fn at(&self, i: usize, j: usize) -> f64 {
        self[i][j]
    }
}

/// Defines a stack-allocated (fixed-size) 9x9 matrix
///
/// # Panics
///
/// * The array must be 9x9; otherwise a panic will occur.
/// * The methods may panic if the array is empty.
impl AsMatrix9x9 for [[f64; 9]; 9] {
    fn at(&self, i: usize, j: usize) -> f64 {
        self[i][j]
    }
}

/// Defines a 9x9 matrix from russell_lab::Matrix
///
/// # Panics
///
/// * The matrix must be 9x9; otherwise a panic will occur.
/// * The methods may panic if the array is empty.
impl AsMatrix9x9 for Matrix {
    fn at(&self, i: usize, j: usize) -> f64 {
        self.get(i, j)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::AsMatrix9x9;
    use crate::MN_TO_IJKL;
    use russell_lab::Matrix;

    fn diagonal(mat: &dyn AsMatrix9x9) -> Vec<f64> {
        let mut res = vec![0.0; 9];
        for i in 0..9 {
            res[i] = mat.at(i, i);
        }
        res
    }

    #[test]
    fn as_matrix_9x9_works() {
        // heap-allocated matrix (vector of vectors)
        // ┌                                              ┐
        // │ 1111 1122 1133 1112 1123 1113 1121 1132 1131 │
        // │ 2211 2222 2233 2212 2223 2213 2221 2232 2231 │
        // │ 3311 3322 3333 3312 3323 3313 3321 3332 3331 │
        // │ 1211 1222 1233 1212 1223 1213 1221 1232 1231 │
        // │ 2311 2322 2333 2312 2323 2313 2321 2332 2331 │
        // │ 1311 1322 1333 1312 1323 1313 1321 1332 1331 │
        // │ 2111 2122 2133 2112 2123 2113 2121 2132 2131 │
        // │ 3211 3222 3233 3212 3223 3213 3221 3232 3231 │
        // │ 3111 3122 3133 3112 3123 3113 3121 3132 3131 │
        // └                                              ┘
        let mut mat = vec![vec![0.0; 9]; 9];
        for m in 0..9 {
            for n in 0..9 {
                let (i, j, k, l) = MN_TO_IJKL[m][n];
                mat[m][n] = (1000 * (i + 1) + 100 * (j + 1) + 10 * (k + 1) + (l + 1)) as f64;
            }
        }
        assert_eq!(
            diagonal(&mat),
            &[1111.0, 2222.0, 3333.0, 1212.0, 2323.0, 1313.0, 2121.0, 3232.0, 3131.0]
        );

        // heap-allocated 2D array (aka slice of slices)
        let ___ = 0.0;
        let mat: &[&[f64]] = &[
            &[1.0, ___, ___, ___, ___, ___, ___, ___, ___],
            &[___, 2.0, ___, ___, ___, ___, ___, ___, ___],
            &[___, ___, 3.0, ___, ___, ___, ___, ___, ___],
            &[___, ___, ___, 4.0, ___, ___, ___, ___, ___],
            &[___, ___, ___, ___, 5.0, ___, ___, ___, ___],
            &[___, ___, ___, ___, ___, 6.0, ___, ___, ___],
            &[___, ___, ___, ___, ___, ___, 7.0, ___, ___],
            &[___, ___, ___, ___, ___, ___, ___, 8.0, ___],
            &[___, ___, ___, ___, ___, ___, ___, ___, 9.0],
        ];
        assert_eq!(diagonal(&mat), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

        // stack-allocated (fixed-size) 2D array
        let mat = [
            [1.0, ___, ___, ___, ___, ___, ___, ___, ___],
            [___, 2.0, ___, ___, ___, ___, ___, ___, ___],
            [___, ___, 3.0, ___, ___, ___, ___, ___, ___],
            [___, ___, ___, 4.0, ___, ___, ___, ___, ___],
            [___, ___, ___, ___, 5.0, ___, ___, ___, ___],
            [___, ___, ___, ___, ___, 6.0, ___, ___, ___],
            [___, ___, ___, ___, ___, ___, 7.0, ___, ___],
            [___, ___, ___, ___, ___, ___, ___, 8.0, ___],
            [___, ___, ___, ___, ___, ___, ___, ___, 9.0],
        ];
        assert_eq!(diagonal(&mat), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

        // russell_lab::Matrix
        let diag = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0];
        let mat = Matrix::diagonal(&diag);
        assert_eq!(diagonal(&mat), &diag);
    }
}
