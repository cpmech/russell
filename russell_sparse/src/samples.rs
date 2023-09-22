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
    pub fn umfpack_sample1_unsymmetric() -> (CooMatrix, f64) {
        let mut coo = CooMatrix::new(None, 5, 5, 13).unwrap();
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
    pub fn mkl_sample1_symmetric_lower() -> (CooMatrix, f64) {
        let sym = Some(Symmetry::General(Storage::Lower));
        let mut coo = CooMatrix::new(sym, 5, 5, 9).unwrap();
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
    pub fn mkl_sample1_positive_definite_lower() -> (CooMatrix, f64) {
        let sym = Some(Symmetry::PositiveDefinite(Storage::Lower));
        let mut coo = CooMatrix::new(sym, 5, 5, 9).unwrap();
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
    pub fn mkl_sample1_symmetric_upper() -> (CooMatrix, f64) {
        let sym = Some(Symmetry::General(Storage::Upper));
        let mut coo = CooMatrix::new(sym, 5, 5, 9).unwrap();
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
    pub fn mkl_sample1_positive_definite_upper() -> (CooMatrix, f64) {
        let sym = Some(Symmetry::PositiveDefinite(Storage::Upper));
        let mut coo = CooMatrix::new(sym, 5, 5, 9).unwrap();
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
    pub fn mkl_sample1_symmetric_full() -> (CooMatrix, f64) {
        let sym = Some(Symmetry::General(Storage::Full));
        let mut coo = CooMatrix::new(sym, 5, 5, 13).unwrap();
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
        let (coo, correct_det) = Samples::umfpack_sample1_unsymmetric();
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

        let correct = "┌                               ┐\n\
                       │     9   1.5     6  0.75     3 │\n\
                       │   1.5   0.5     0     0     0 │\n\
                       │     6     0    12     0     0 │\n\
                       │  0.75     0     0 0.625     0 │\n\
                       │     3     0     0     0    16 │\n\
                       └                               ┘";
        for (coo, correct_det) in [
            Samples::mkl_sample1_positive_definite_lower(),
            Samples::mkl_sample1_positive_definite_upper(),
            Samples::mkl_sample1_symmetric_lower(),
            Samples::mkl_sample1_symmetric_upper(),
            Samples::mkl_sample1_symmetric_full(),
        ] {
            let mat = coo.as_matrix();
            assert_eq!(format!("{}", mat), correct);
            let mut inv = Matrix::new(5, 5);
            let det = mat_inverse(&mut inv, &mat).unwrap();
            approx_eq(det, correct_det, 1e-14);
        }
    }
}
