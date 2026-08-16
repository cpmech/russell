use russell_lab::Matrix;

/// Defines a trait to handle 9x3 matrices
///
/// # Examples
///
/// ```
/// use russell_lab::Matrix;
/// use russell_tensor::{AsMatrix9x3, MN_TO_IJK};
///
/// // heap-allocated matrix (vector of vectors)
/// // ┌             ┐
/// // │ 111 112 113 │
/// // │ 221 222 223 │
/// // │ 331 332 333 │
/// // │ 121 122 123 │
/// // │ 231 232 233 │
/// // │ 131 132 133 │
/// // │ 211 212 213 │
/// // │ 321 322 323 │
/// // │ 311 312 313 │
/// // └             ┘
/// let mut mat = vec![vec![0.0; 3]; 9];
/// for m in 0..9 {
///     for n in 0..3 {
///         let (i, j, k) = MN_TO_IJK[m][n];
///         mat[m][n] = (100 * (i + 1) + 10 * (j + 1) + (k + 1)) as f64;
///     }
/// }
/// assert_eq!(mat.at(0, 0), 111.0);
/// assert_eq!(mat.at(5, 1), 132.0);
/// assert_eq!(mat.at(8, 2), 313.0);
/// ```
pub trait AsMatrix9x3 {
    /// Returns the value at (i,j) indices
    ///
    /// # Panics
    ///
    /// This function panics if the indices are out of range.
    fn at(&self, i: usize, j: usize) -> f64;
}

/// Defines a heap-allocated 9x3 matrix (vector of vectors)
///
/// # Panics
///
/// * The array must be 9x3; otherwise a panic will occur.
/// * The methods may panic if the array is empty.
impl AsMatrix9x3 for Vec<Vec<f64>> {
    fn at(&self, i: usize, j: usize) -> f64 {
        self[i][j]
    }
}

/// Defines a heap-allocated 9x3 matrix (slice of slices)
///
/// # Panics
///
/// * The array must be 9x3; otherwise a panic will occur.
/// * The methods may panic if the array is empty.
impl AsMatrix9x3 for &[&[f64]] {
    fn at(&self, i: usize, j: usize) -> f64 {
        self[i][j]
    }
}

/// Defines a stack-allocated (fixed-size) 9x3 matrix
///
/// # Panics
///
/// * The array must be 9x3; otherwise a panic will occur.
/// * The methods may panic if the array is empty.
impl AsMatrix9x3 for [[f64; 3]; 9] {
    fn at(&self, i: usize, j: usize) -> f64 {
        self[i][j]
    }
}

/// Defines a 9x3 matrix from russell_lab::Matrix
///
/// # Panics
///
/// * The matrix must be 9x3; otherwise a panic will occur.
/// * The methods may panic if the array is empty.
impl AsMatrix9x3 for Matrix {
    fn at(&self, i: usize, j: usize) -> f64 {
        self.get(i, j)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::AsMatrix9x3;
    use crate::MN_TO_IJK;
    use russell_lab::Matrix;

    #[test]
    fn as_matrix_9x3_works() {
        // heap-allocated matrix (vector of vectors)
        // ┌             ┐
        // │ 111 112 113 │
        // │ 221 222 223 │
        // │ 331 332 333 │
        // │ 121 122 123 │
        // │ 231 232 233 │
        // │ 131 132 133 │
        // │ 211 212 213 │
        // │ 321 322 323 │
        // │ 311 312 313 │
        // └             ┘
        let mut mat = vec![vec![0.0; 3]; 9];
        for m in 0..9 {
            for n in 0..3 {
                let (i, j, k) = MN_TO_IJK[m][n];
                mat[m][n] = (100 * (i + 1) + 10 * (j + 1) + (k + 1)) as f64;
            }
        }
        assert_eq!(mat.at(0, 0), 111.0);
        assert_eq!(mat.at(5, 1), 132.0);
        assert_eq!(mat.at(8, 2), 313.0);

        // heap-allocated 2D array (aka slice of slices)
        let ___ = 0.0;
        let mat: &[&[f64]] = &[
            &[1.0, ___, ___],
            &[___, 2.0, ___],
            &[___, ___, 3.0],
            &[___, ___, ___],
            &[___, ___, ___],
            &[___, ___, ___],
            &[___, ___, ___],
            &[___, ___, ___],
            &[___, ___, ___],
        ];
        assert_eq!(mat.at(0, 0), 1.0);
        assert_eq!(mat.at(1, 1), 2.0);
        assert_eq!(mat.at(2, 2), 3.0);

        // stack-allocated (fixed-size) 2D array
        let mat = [
            [1.0, ___, ___],
            [___, 2.0, ___],
            [___, ___, 3.0],
            [___, ___, ___],
            [___, ___, ___],
            [___, ___, ___],
            [___, ___, ___],
            [___, ___, ___],
            [___, ___, ___],
        ];
        assert_eq!(mat.at(0, 0), 1.0);
        assert_eq!(mat.at(1, 1), 2.0);
        assert_eq!(mat.at(2, 2), 3.0);

        // russell_lab::Matrix
        let mat = Matrix::from(&[
            [1.1, 1.2, 1.3],
            [2.1, 2.2, 2.3],
            [3.1, 3.2, 3.3],
            [4.1, 4.2, 4.3],
            [5.1, 5.2, 5.3],
            [6.1, 6.2, 6.3],
            [7.1, 7.2, 7.3],
            [8.1, 8.2, 8.3],
            [9.1, 9.2, 9.3],
        ]);
        assert_eq!(mat.at(0, 0), 1.1);
        assert_eq!(mat.at(5, 1), 6.2);
        assert_eq!(mat.at(8, 2), 9.3);
    }
}
