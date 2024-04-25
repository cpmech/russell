use russell_lab::Matrix;

/// Defines a trait to handle 2D arrays
///
/// # Examples
///
/// ```
/// use russell_lab::Matrix;
/// use russell_tensor::AsMatrix3x3;
///
/// fn flatten(mat: &dyn AsMatrix3x3) -> Vec<f64> {
///     let mut res = vec![0.0; 9];
///     let mut m = 0;
///     for i in 0..3 {
///         for j in 0..3 {
///             res[m] = mat.at(i, j);
///             m += 1;
///         }
///     }
///     res
/// }
///
/// // heap-allocated matrix (vector of vectors)
/// let mat = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0], vec![7.0, 8.0, 9.0]];
/// assert_eq!(flatten(&mat), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
///
/// // heap-allocated 2D array (aka slice of slices)
/// let mat: &[&[f64]] = &[&[10.0, 20.0, 30.0], &[40.0, 50.0, 60.0], &[70.0, 80.0, 90.0]];
/// assert_eq!(flatten(&mat), &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]);
///
/// // stack-allocated (fixed-size) 2D array
/// let mat = [[100.0, 200.0, 300.0], [400.0, 500.0, 600.0], [700.0, 800.0, 900.0]];
/// assert_eq!(flatten(&mat), &[100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0]);
///
/// // russell_lab::Matrix
/// let mat = Matrix::from(&[[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0], [-7.0, -8.0, -9.0]]);
/// assert_eq!(flatten(&mat), &[-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0]);
/// ```
pub trait AsMatrix3x3 {
    /// Returns the value at (i,j) indices
    ///
    /// # Panics
    ///
    /// This function panics if the indices are out of range.
    fn at(&self, i: usize, j: usize) -> f64;
}

/// Defines a heap-allocated 3x3 matrix (vector of vectors)
///
/// # Panics
///
/// * The array must be 3x3; otherwise a panic will occur.
/// * The methods may panic if the array is empty.
impl AsMatrix3x3 for Vec<Vec<f64>> {
    fn at(&self, i: usize, j: usize) -> f64 {
        self[i][j]
    }
}

/// Defines a heap-allocated 3x3 matrix (slice of slices)
///
/// # Panics
///
/// * The array must be 3x3; otherwise a panic will occur.
/// * The methods may panic if the array is empty.
impl AsMatrix3x3 for &[&[f64]] {
    fn at(&self, i: usize, j: usize) -> f64 {
        self[i][j]
    }
}

/// Defines a stack-allocated (fixed-size) 3x3 matrix
///
/// # Panics
///
/// * The array must be 3x3; otherwise a panic will occur.
/// * The methods may panic if the array is empty.
impl AsMatrix3x3 for [[f64; 3]; 3] {
    fn at(&self, i: usize, j: usize) -> f64 {
        self[i][j]
    }
}

/// Defines a 3x3 matrix from russell_lab::Matrix
///
/// # Panics
///
/// * The matrix must be 3x3; otherwise a panic will occur.
/// * The methods may panic if the array is empty.
impl AsMatrix3x3 for Matrix {
    fn at(&self, i: usize, j: usize) -> f64 {
        self.get(i, j)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use russell_lab::Matrix;

    use super::AsMatrix3x3;

    fn flatten(mat: &dyn AsMatrix3x3) -> Vec<f64> {
        let mut res = vec![0.0; 9];
        let mut m = 0;
        for i in 0..3 {
            for j in 0..3 {
                res[m] = mat.at(i, j);
                m += 1;
            }
        }
        res
    }

    #[test]
    fn as_matrix_3x3_works() {
        // heap-allocated matrix (vector of vectors)
        let mat = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0], vec![7.0, 8.0, 9.0]];
        assert_eq!(flatten(&mat), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

        // heap-allocated 2D array (aka slice of slices)
        let mat: &[&[f64]] = &[&[10.0, 20.0, 30.0], &[40.0, 50.0, 60.0], &[70.0, 80.0, 90.0]];
        assert_eq!(flatten(&mat), &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]);

        // stack-allocated (fixed-size) 2D array
        let mat = [[100.0, 200.0, 300.0], [400.0, 500.0, 600.0], [700.0, 800.0, 900.0]];
        assert_eq!(
            flatten(&mat),
            &[100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0]
        );

        // russell_lab::Matrix
        let mat = Matrix::from(&[[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0], [-7.0, -8.0, -9.0]]);
        assert_eq!(flatten(&mat), &[-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0]);
    }
}
