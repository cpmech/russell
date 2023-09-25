use crate::{CooMatrix, Storage, Symmetry};

/// Holds some samples of small sparse matrices
pub struct Samples {}

impl Samples {
    /// Returns the matrix and its determinant
    ///
    /// Example from UMFPACK documentation
    ///
    /// ```text
    /// ┌                ┐
    /// │  2  3  0  0  0 │
    /// │  3  0  4  0  6 │
    /// │  0 -1 -3  2  0 │
    /// │  0  0  1  0  0 │
    /// │  0  4  2  0  1 │
    /// └                ┘
    /// ```
    ///
    /// With the right-hand side vector:
    ///
    /// ```text
    /// let rhs = Vector::from(&[8.0, 45.0, -3.0, 3.0, 19.0]);
    /// ```
    ///
    /// The solution of `A · x = rhs` is:
    ///
    /// ```text
    /// let x_correct = &[1.0, 2.0, 3.0, 4.0, 5.0];
    /// ```
    pub fn umfpack_sample1_unsymmetric(one_based: bool) -> (CooMatrix, f64) {
        let mut coo = CooMatrix::new(5, 5, 13, None, one_based).unwrap();
        coo.put(0, 0, 1.0).unwrap(); // << (0, 0, a00/2) duplicate
        coo.put(0, 0, 1.0).unwrap(); // << (0, 0, a00/2) duplicate
        coo.put(1, 0, 3.0).unwrap();
        coo.put(0, 1, 3.0).unwrap();
        coo.put(2, 1, -1.0).unwrap();
        coo.put(4, 1, 4.0).unwrap();
        coo.put(1, 2, 4.0).unwrap();
        coo.put(2, 2, -3.0).unwrap();
        coo.put(3, 2, 1.0).unwrap();
        coo.put(4, 2, 2.0).unwrap();
        coo.put(2, 3, 2.0).unwrap();
        coo.put(1, 4, 6.0).unwrap();
        coo.put(4, 4, 1.0).unwrap();
        (coo, 114.0)
    }

    /// Returns the matrix and its determinant
    ///
    /// Triplet with shuffled entries
    ///
    /// ```text
    ///  1  -1   .  -3   .
    /// -2   5   .   .   .
    ///  .   .   4   6   4
    /// -4   .   2   7   .
    ///  .   8   .   .  -5
    /// ```
    pub fn unsymmetric_5x5_with_shuffled_entries(one_based: bool) -> (CooMatrix, f64) {
        let mut coo = CooMatrix::new(5, 5, 13, None, one_based).unwrap();
        coo.put(2, 4, 4.0).unwrap();
        coo.put(4, 1, 8.0).unwrap();
        coo.put(0, 1, -1.0).unwrap();
        coo.put(2, 2, 4.0).unwrap();
        coo.put(4, 4, -5.0).unwrap();
        coo.put(3, 0, -4.0).unwrap();
        coo.put(0, 3, -3.0).unwrap();
        coo.put(2, 3, 6.0).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(1, 1, 5.0).unwrap();
        coo.put(3, 2, 2.0).unwrap();
        coo.put(1, 0, -2.0).unwrap();
        coo.put(3, 3, 7.0).unwrap();
        (coo, 1344.0)
    }

    /// Returns the matrix and its determinant
    ///
    /// Small triplet with shuffled entries
    ///
    /// ```text
    /// 1  2  .  .  .
    /// 3  4  .  .  .
    /// .  .  5  6  .
    /// .  .  7  8  .
    /// .  .  .  .  9
    /// ```
    pub fn block_unsym_5x5_with_shuffled_entries(one_based: bool) -> (CooMatrix, f64) {
        let mut coo = CooMatrix::new(5, 5, 9, None, one_based).unwrap();
        coo.put(4, 4, 9.0).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(1, 0, 3.0).unwrap();
        coo.put(2, 2, 5.0).unwrap();
        coo.put(2, 3, 6.0).unwrap();
        coo.put(0, 1, 2.0).unwrap();
        coo.put(3, 2, 7.0).unwrap();
        coo.put(1, 1, 4.0).unwrap();
        coo.put(3, 3, 8.0).unwrap();
        (coo, 36.0)
    }

    /// Returns the matrix and its determinant
    ///
    /// Small triplet with shuffled entries
    ///
    /// ```text
    /// 1  2  .  .  .
    /// 3  4  .  .  .
    /// .  .  5  6  .
    /// .  .  7  8  .
    /// .  .  .  .  9
    /// ```
    pub fn block_unsym_5x5_with_duplicates(one_based: bool) -> (CooMatrix, f64) {
        let mut coo = CooMatrix::new(5, 5, 11, None, one_based).unwrap();
        coo.put(4, 4, 9.0).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(1, 0, 3.0).unwrap();
        coo.put(2, 2, 5.0).unwrap();
        coo.put(2, 3, 3.0).unwrap(); // <<
        coo.put(0, 1, 2.0).unwrap();
        coo.put(3, 2, 7.0).unwrap();
        coo.put(1, 1, 2.0).unwrap(); // <<
        coo.put(3, 3, 8.0).unwrap();
        coo.put(2, 3, 3.0).unwrap(); // << duplicate
        coo.put(1, 1, 2.0).unwrap(); // << duplicate
        (coo, 36.0)
    }

    /// Returns the matrix and its determinant
    ///
    /// Example from Intel MKL documentation
    ///
    /// ```text
    /// ┌                               ┐
    /// │     9   1.5     6  0.75     3 │
    /// │   1.5   0.5     0     0     0 │
    /// │     6     0    12     0     0 │
    /// │  0.75     0     0 0.625     0 │
    /// │     3     0     0     0    16 │
    /// └                               ┘
    /// ```
    ///
    /// With the right-hand side vector:
    ///
    /// ```text
    /// let rhs = Vector::from(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    /// ```
    ///
    /// The solution of `A · x = rhs` is:
    ///
    /// ```text
    /// x_correct = vec![-979.0 / 3.0, 983.0, 1961.0 / 12.0, 398.0, 123.0 / 2.0];
    /// ```
    pub fn mkl_sample1_symmetric_lower(one_based: bool) -> (CooMatrix, f64) {
        let sym = Some(Symmetry::General(Storage::Lower));
        let mut coo = CooMatrix::new(5, 5, 9, sym, one_based).unwrap();
        coo.put(0, 0, 9.0).unwrap();
        coo.put(1, 1, 0.5).unwrap();
        coo.put(2, 2, 12.0).unwrap();
        coo.put(3, 3, 0.625).unwrap();
        coo.put(4, 4, 16.0).unwrap();
        coo.put(1, 0, 1.5).unwrap();
        coo.put(2, 0, 6.0).unwrap();
        coo.put(3, 0, 0.75).unwrap();
        coo.put(4, 0, 3.0).unwrap();
        (coo, 9.0 / 4.0)
    }

    /// Returns the matrix and its determinant
    ///
    /// Example from Intel MKL documentation
    ///
    /// ```text
    /// ┌                               ┐
    /// │     9   1.5     6  0.75     3 │
    /// │   1.5   0.5     0     0     0 │
    /// │     6     0    12     0     0 │
    /// │  0.75     0     0 0.625     0 │
    /// │     3     0     0     0    16 │
    /// └                               ┘
    /// ```
    ///
    /// With the right-hand side vector:
    ///
    /// ```text
    /// let rhs = Vector::from(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    /// ```
    ///
    /// The solution of `A · x = rhs` is:
    ///
    /// ```text
    /// x_correct = vec![-979.0 / 3.0, 983.0, 1961.0 / 12.0, 398.0, 123.0 / 2.0];
    /// ```
    pub fn mkl_sample1_positive_definite_lower(one_based: bool) -> (CooMatrix, f64) {
        let sym = Some(Symmetry::PositiveDefinite(Storage::Lower));
        let mut coo = CooMatrix::new(5, 5, 9, sym, one_based).unwrap();
        coo.put(0, 0, 9.0).unwrap();
        coo.put(1, 1, 0.5).unwrap();
        coo.put(2, 2, 12.0).unwrap();
        coo.put(3, 3, 0.625).unwrap();
        coo.put(4, 4, 16.0).unwrap();
        coo.put(1, 0, 1.5).unwrap();
        coo.put(2, 0, 6.0).unwrap();
        coo.put(3, 0, 0.75).unwrap();
        coo.put(4, 0, 3.0).unwrap();
        (coo, 9.0 / 4.0)
    }

    /// Returns the matrix and its determinant
    ///
    /// Example from Intel MKL documentation
    ///
    /// ```text
    /// ┌                               ┐
    /// │     9   1.5     6  0.75     3 │
    /// │   1.5   0.5     0     0     0 │
    /// │     6     0    12     0     0 │
    /// │  0.75     0     0 0.625     0 │
    /// │     3     0     0     0    16 │
    /// └                               ┘
    /// ```
    ///
    /// With the right-hand side vector:
    ///
    /// ```text
    /// let rhs = Vector::from(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    /// ```
    ///
    /// The solution of `A · x = rhs` is:
    ///
    /// ```text
    /// x_correct = vec![-979.0 / 3.0, 983.0, 1961.0 / 12.0, 398.0, 123.0 / 2.0];
    /// ```
    pub fn mkl_sample1_symmetric_upper(one_based: bool) -> (CooMatrix, f64) {
        let sym = Some(Symmetry::General(Storage::Upper));
        let mut coo = CooMatrix::new(5, 5, 9, sym, one_based).unwrap();
        coo.put(0, 0, 9.0).unwrap();
        coo.put(0, 1, 1.5).unwrap();
        coo.put(1, 1, 0.5).unwrap();
        coo.put(0, 2, 6.0).unwrap();
        coo.put(2, 2, 12.0).unwrap();
        coo.put(0, 3, 0.75).unwrap();
        coo.put(3, 3, 0.625).unwrap();
        coo.put(0, 4, 3.0).unwrap();
        coo.put(4, 4, 16.0).unwrap();
        (coo, 9.0 / 4.0)
    }

    /// Returns the matrix and its determinant
    ///
    /// Example from Intel MKL documentation
    ///
    /// ```text
    /// ┌                               ┐
    /// │     9   1.5     6  0.75     3 │
    /// │   1.5   0.5     0     0     0 │
    /// │     6     0    12     0     0 │
    /// │  0.75     0     0 0.625     0 │
    /// │     3     0     0     0    16 │
    /// └                               ┘
    /// ```
    ///
    /// With the right-hand side vector:
    ///
    /// ```text
    /// let rhs = Vector::from(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    /// ```
    ///
    /// The solution of `A · x = rhs` is:
    ///
    /// ```text
    /// x_correct = vec![-979.0 / 3.0, 983.0, 1961.0 / 12.0, 398.0, 123.0 / 2.0];
    /// ```
    pub fn mkl_sample1_positive_definite_upper(one_based: bool) -> (CooMatrix, f64) {
        let sym = Some(Symmetry::PositiveDefinite(Storage::Upper));
        let mut coo = CooMatrix::new(5, 5, 9, sym, one_based).unwrap();
        coo.put(0, 0, 9.0).unwrap();
        coo.put(0, 1, 1.5).unwrap();
        coo.put(1, 1, 0.5).unwrap();
        coo.put(0, 2, 6.0).unwrap();
        coo.put(2, 2, 12.0).unwrap();
        coo.put(0, 3, 0.75).unwrap();
        coo.put(3, 3, 0.625).unwrap();
        coo.put(0, 4, 3.0).unwrap();
        coo.put(4, 4, 16.0).unwrap();
        (coo, 9.0 / 4.0)
    }

    /// Returns the matrix and its determinant
    ///
    /// Example from Intel MKL documentation
    ///
    /// ```text
    /// ┌                               ┐
    /// │     9   1.5     6  0.75     3 │
    /// │   1.5   0.5     0     0     0 │
    /// │     6     0    12     0     0 │
    /// │  0.75     0     0 0.625     0 │
    /// │     3     0     0     0    16 │
    /// └                               ┘
    /// ```
    ///
    /// With the right-hand side vector:
    ///
    /// ```text
    /// let rhs = Vector::from(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    /// ```
    ///
    /// The solution of `A · x = rhs` is:
    ///
    /// ```text
    /// x_correct = vec![-979.0 / 3.0, 983.0, 1961.0 / 12.0, 398.0, 123.0 / 2.0];
    /// ```
    pub fn mkl_sample1_symmetric_full(one_based: bool) -> (CooMatrix, f64) {
        let sym = Some(Symmetry::General(Storage::Full));
        let mut coo = CooMatrix::new(5, 5, 13, sym, one_based).unwrap();
        coo.put(0, 0, 9.0).unwrap(); // 0
        coo.put(0, 1, 1.5).unwrap(); // 1
        coo.put(0, 2, 6.0).unwrap(); // 2
        coo.put(0, 3, 0.75).unwrap(); // 3
        coo.put(0, 4, 3.0).unwrap(); // 4
        coo.put(1, 0, 1.5).unwrap(); // 5
        coo.put(1, 1, 0.5).unwrap(); // 6
        coo.put(2, 0, 6.0).unwrap(); // 7
        coo.put(2, 2, 12.0).unwrap(); // 8
        coo.put(3, 0, 0.75).unwrap(); // 9
        coo.put(3, 3, 0.625).unwrap(); // 10
        coo.put(4, 0, 3.0).unwrap(); // 11
        coo.put(4, 4, 16.0).unwrap(); // 12
        (coo, 9.0 / 4.0)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Samples;
    use russell_chk::approx_eq;
    use russell_lab::{mat_inverse, Matrix};

    #[test]
    fn samples_are_correct() {
        // ----------------------------------------------------------------------------

        let (coo, correct_det) = Samples::umfpack_sample1_unsymmetric(false);
        let mat = coo.as_matrix();
        let correct = "┌                ┐\n\
                       │  2  3  0  0  0 │\n\
                       │  3  0  4  0  6 │\n\
                       │  0 -1 -3  2  0 │\n\
                       │  0  0  1  0  0 │\n\
                       │  0  4  2  0  1 │\n\
                       └                ┘";
        assert_eq!(format!("{}", mat), correct);
        let mut inv = Matrix::new(5, 5);
        let det = mat_inverse(&mut inv, &mat).unwrap();
        approx_eq(det, correct_det, 1e-15);

        let (coo, _) = Samples::umfpack_sample1_unsymmetric(true);
        let mat = coo.as_matrix();
        let correct = "┌                ┐\n\
                       │  2  3  0  0  0 │\n\
                       │  3  0  4  0  6 │\n\
                       │  0 -1 -3  2  0 │\n\
                       │  0  0  1  0  0 │\n\
                       │  0  4  2  0  1 │\n\
                       └                ┘";
        assert_eq!(format!("{}", mat), correct);

        // ----------------------------------------------------------------------------

        let (coo, correct_det) = Samples::unsymmetric_5x5_with_shuffled_entries(false);
        let mat = coo.as_matrix();
        let correct = "┌                ┐\n\
                       │  1 -1  0 -3  0 │\n\
                       │ -2  5  0  0  0 │\n\
                       │  0  0  4  6  4 │\n\
                       │ -4  0  2  7  0 │\n\
                       │  0  8  0  0 -5 │\n\
                       └                ┘";
        assert_eq!(format!("{}", mat), correct);
        let mut inv = Matrix::new(5, 5);
        let det = mat_inverse(&mut inv, &mat).unwrap();
        approx_eq(det, correct_det, 1e-15);

        // ----------------------------------------------------------------------------

        let correct = "┌           ┐\n\
                       │ 1 2 0 0 0 │\n\
                       │ 3 4 0 0 0 │\n\
                       │ 0 0 5 6 0 │\n\
                       │ 0 0 7 8 0 │\n\
                       │ 0 0 0 0 9 │\n\
                       └           ┘";
        for (coo, correct_det) in [
            Samples::block_unsym_5x5_with_shuffled_entries(false),
            Samples::block_unsym_5x5_with_shuffled_entries(true),
            Samples::block_unsym_5x5_with_duplicates(false),
            Samples::block_unsym_5x5_with_duplicates(true),
        ] {
            let mat = coo.as_matrix();
            assert_eq!(format!("{}", mat), correct);
            let mut inv = Matrix::new(5, 5);
            let det = mat_inverse(&mut inv, &mat).unwrap();
            approx_eq(det, correct_det, 1e-13);
        }

        // ----------------------------------------------------------------------------

        let correct = "┌                               ┐\n\
                       │     9   1.5     6  0.75     3 │\n\
                       │   1.5   0.5     0     0     0 │\n\
                       │     6     0    12     0     0 │\n\
                       │  0.75     0     0 0.625     0 │\n\
                       │     3     0     0     0    16 │\n\
                       └                               ┘";
        for (coo, correct_det) in [
            Samples::mkl_sample1_positive_definite_lower(false),
            Samples::mkl_sample1_positive_definite_lower(true),
            Samples::mkl_sample1_positive_definite_upper(false),
            Samples::mkl_sample1_positive_definite_upper(true),
            Samples::mkl_sample1_symmetric_lower(false),
            Samples::mkl_sample1_symmetric_lower(true),
            Samples::mkl_sample1_symmetric_upper(false),
            Samples::mkl_sample1_symmetric_upper(true),
            Samples::mkl_sample1_symmetric_full(false),
            Samples::mkl_sample1_symmetric_full(true),
        ] {
            let mat = coo.as_matrix();
            assert_eq!(format!("{}", mat), correct);
            let mut inv = Matrix::new(5, 5);
            let det = mat_inverse(&mut inv, &mat).unwrap();
            approx_eq(det, correct_det, 1e-14);
        }
    }
}
