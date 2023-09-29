use crate::{CooMatrix, CscMatrix, CsrMatrix, Storage, Symmetry};

const PLACEHOLDER: f64 = f64::MAX;

/// Holds some samples of small sparse matrices
pub struct Samples {}

impl Samples {
    /// Returns a (1 x 1) matrix
    ///
    /// Note: the last return value is not the determinant, but a PLACEHOLDER
    ///
    /// ```text
    /// ┌     ┐
    /// │ 123 │
    /// └     ┘
    /// ```
    pub fn tiny_1x1(one_based: bool) -> (CooMatrix, CscMatrix, CsrMatrix, f64) {
        let sym = None;
        let nrow = 1;
        let ncol = 1;
        let mut coo = CooMatrix::new(1, 1, 1, sym, one_based).unwrap();
        coo.put(0, 0, 123.0).unwrap();
        // CSC matrix
        let col_pointers = vec![0, 1];
        let row_indices = vec![0];
        let values = vec![123.0];
        let csc = CscMatrix::new(nrow, ncol, col_pointers, row_indices, values, sym).unwrap();
        // CSR matrix
        let row_pointers = vec![0, 1];
        let col_indices = vec![0];
        let values = vec![123.0];
        let csr = CsrMatrix::new(nrow, ncol, row_pointers, col_indices, values, sym).unwrap();
        (coo, csc, csr, 123.0)
    }

    /// Returns the COO, CSC, and CSR versions of the matrix and its determinant
    ///
    /// ```text
    ///  1  .  2
    ///  .  0  3   << the zero diagonal value is required for Intel DSS
    ///  4  5  6
    /// ```
    ///
    /// ```text
    /// det(a) = -15.0
    /// ```
    ///
    /// With the right-hand side vector:
    ///
    /// ```text
    /// let rhs = Vector::from(&[1.0, 1.0, 1.0]);
    /// ```
    ///
    /// The solution of `A · x = rhs` is:
    ///
    /// ```text
    /// let x_correct = &[3.0, 3.0, 15];
    /// ```
    pub fn unsymmetric_3x3(
        one_based: bool,
        shuffle_coo_entries: bool,
        duplicate_coo_entries: bool,
    ) -> (CooMatrix, CscMatrix, CsrMatrix, f64) {
        let sym = None;
        let nrow = 3;
        let ncol = 3;
        let max_nnz = 10; // more nnz than needed => OK
        let mut coo = CooMatrix::new(nrow, ncol, max_nnz, sym, one_based).unwrap();
        if shuffle_coo_entries {
            if duplicate_coo_entries {
                coo.put(0, 2, 2.0).unwrap();
                coo.put(2, 1, 5.0).unwrap();
                coo.put(1, 2, 3.0).unwrap();
                coo.put(2, 2, 2.0).unwrap(); // << duplicate
                coo.put(2, 0, 4.0).unwrap();
                coo.put(0, 0, 1.0).unwrap();
                coo.put(2, 2, 2.0).unwrap(); // << duplicate
                coo.put(1, 1, 0.0).unwrap(); // << needed for Intel DSS
                coo.put(2, 2, 2.0).unwrap(); // << duplicate
            } else {
                coo.put(2, 0, 4.0).unwrap();
                coo.put(0, 0, 1.0).unwrap();
                coo.put(2, 2, 6.0).unwrap();
                coo.put(0, 2, 2.0).unwrap();
                coo.put(1, 1, 0.0).unwrap(); // << needed for Intel DSS
                coo.put(2, 1, 5.0).unwrap();
                coo.put(1, 2, 3.0).unwrap();
            }
        } else {
            if duplicate_coo_entries {
                coo.put(0, 0, 1.0).unwrap();
                coo.put(0, 2, 2.0).unwrap();
                coo.put(1, 1, 0.0).unwrap(); // << needed for Intel DSS
                coo.put(1, 2, 3.0).unwrap();
                coo.put(2, 0, 4.0).unwrap();
                coo.put(2, 1, 5.0).unwrap();
                coo.put(2, 2, 2.0).unwrap(); // << duplicate
                coo.put(2, 2, 2.0).unwrap(); // << duplicate
                coo.put(2, 2, 2.0).unwrap(); // << duplicate
            } else {
                coo.put(0, 0, 1.0).unwrap();
                coo.put(0, 2, 2.0).unwrap();
                coo.put(1, 1, 0.0).unwrap(); // << needed for Intel DSS
                coo.put(1, 2, 3.0).unwrap();
                coo.put(2, 0, 4.0).unwrap();
                coo.put(2, 1, 5.0).unwrap();
                coo.put(2, 2, 6.0).unwrap();
            }
        }
        // CSC matrix
        let values = vec![
            1.0, 4.0, //      j=0 p=(0),1
            0.0, 5.0, //      j=1 p=(2),3
            2.0, 3.0, 6.0, // j=2 p=(4),5,6
        ]; //                     p=(7)
        let row_indices = vec![
            0, 2, //
            1, 2, //
            0, 1, 2, //
        ];
        let col_pointers = vec![0, 2, 4, 7];
        let csc = CscMatrix::new(nrow, ncol, col_pointers, row_indices, values, sym).unwrap();
        // CSR matrix
        let values = vec![
            1.0, 2.0, //      i=0 p=(0),1
            0.0, 3.0, //      i=1 p=(2),3
            4.0, 5.0, 6.0, // i=2 p=(4),5,6
        ]; //               p=(7)
        let col_indices = vec![
            0, 2, //
            1, 2, //
            0, 1, 2, //
        ];
        let row_pointers = vec![0, 2, 4, 7];
        let csr = CsrMatrix::new(nrow, ncol, row_pointers, col_indices, values, sym).unwrap();
        (coo, csc, csr, -15.0)
    }

    /// Returns the COO, CSC, and CSR versions of the matrix and its determinant
    ///
    /// Example from the [UMFPACK documentation](https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/UMFPACK/Doc/UMFPACK_QuickStart.pdf)
    ///
    /// ```text
    ///  2  3  .  .  .
    ///  3  .  4  .  6
    ///  . -1 -3  2  .
    ///  .  .  1  .  .
    ///  .  4  2  .  1
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
    pub fn umfpack_unsymmetric_5x5(one_based: bool) -> (CooMatrix, CscMatrix, CsrMatrix, f64) {
        let sym = None;
        let nrow = 5;
        let ncol = 5;
        let max_nnz = 13;
        let mut coo = CooMatrix::new(nrow, ncol, max_nnz, sym, one_based).unwrap();
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
        // CSC matrix
        let values = vec![
            2.0, 3.0, //            j=0, p=( 0),1
            3.0, -1.0, 4.0, //      j=1, p=( 2),3,4
            4.0, -3.0, 1.0, 2.0, // j=2, p=( 5),6,7,8
            2.0, //                 j=3, p=( 9)
            6.0, 1.0, //            j=4, p=(10),11
        ]; //                            p=(12)
        let row_indices = vec![
            0, 1, //
            0, 2, 4, //
            1, 2, 3, 4, //
            2, //
            1, 4, //
        ];
        let col_pointers = vec![0, 2, 5, 9, 10, 12];
        let csc = CscMatrix::new(nrow, ncol, col_pointers, row_indices, values, sym).unwrap();
        // CSR matrix
        let values = vec![
            2.0, 3.0, //        i=0, p=(0),1
            3.0, 4.0, 6.0, //   i=1, p=(2),3,4
            -1.0, -3.0, 2.0, // i=2, p=(5),6,7
            1.0, //             i=3, p=(8)
            4.0, 2.0, 1.0, //   i=4, p=(9),10,11
        ]; //                  p=(12)
        let col_indices = vec![
            0, 1, //
            0, 2, 4, //
            1, 2, 3, //
            2, //
            1, 2, 4, //
        ];
        let row_pointers = vec![0, 2, 5, 8, 9, 12];
        let csr = CsrMatrix::new(nrow, ncol, row_pointers, col_indices, values, sym).unwrap();
        (coo, csc, csr, 114.0)
    }

    /// Returns the COO, CSC, and CSR versions of the matrix and its determinant
    ///
    /// Triplet with shuffled entries (however, CSC and CSR have sorted entries).
    ///
    /// ```text
    ///  1  -1   .  -3   .
    /// -2   5   .   .   .
    ///  .   .   4   6   4
    /// -4   .   2   7   .
    ///  .   8   .   .  -5
    /// ```
    ///
    /// Reference:
    /// <https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-2/sparse-blas-csr-matrix-storage-format.html>
    pub fn mkl_unsymmetric_5x5(one_based: bool) -> (CooMatrix, CscMatrix, CsrMatrix, f64) {
        let sym = None;
        let nrow = 5;
        let ncol = 5;
        let mut coo = CooMatrix::new(nrow, ncol, 13, sym, one_based).unwrap();
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
        // CSC matrix
        let values = vec![
            1.0, -2.0, -4.0, // j=0, p=( 0),1,2
            -1.0, 5.0, 8.0, //  j=1, p=( 3),4,5
            4.0, 2.0, //        j=2, p=( 6),7
            -3.0, 6.0, 7.0, //  j=3, p=( 8),9,10
            4.0, -5.0, //       j=4, p=(11),12
        ]; //                        p=(13)
        let row_indices = vec![
            0, 1, 3, //
            0, 1, 4, //
            2, 3, //
            0, 2, 3, //
            2, 4, //
        ];
        let col_pointers = vec![0, 3, 6, 8, 11, 13];
        let csc = CscMatrix::new(nrow, ncol, col_pointers, row_indices, values, sym).unwrap();
        // CSR matrix
        let values = vec![
            1.0, -1.0, -3.0, // i=0, p=( 0),1,2
            -2.0, 5.0, //       i=1, p=( 3),4
            4.0, 6.0, 4.0, //   i=2, p=( 5),6,7
            -4.0, 2.0, 7.0, //  i=3, p=( 8),9,10
            8.0, -5.0, //       i=4, p=(11),12
        ]; //                        p=(13)
        let col_indices = vec![
            0, 1, 3, //
            0, 1, //
            2, 3, 4, //
            0, 2, 3, //
            1, 4, //
        ];
        let row_pointers = vec![0, 3, 5, 8, 11, 13];
        let csr = CsrMatrix::new(nrow, ncol, row_pointers, col_indices, values, sym).unwrap();
        (coo, csc, csr, 1344.0)
    }

    /// Returns the matrix and its determinant
    ///
    /// Triplet with shuffled entries (however, CSC and CSR have sorted entries).
    ///
    /// ```text
    /// 1  2  .  .  .
    /// 3  4  .  .  .
    /// .  .  5  6  .
    /// .  .  7  8  .
    /// .  .  .  .  9
    /// ```
    pub fn block_unsymmetric_5x5(
        one_based: bool,
        shuffle_coo_entries: bool,
        duplicate_coo_entries: bool,
    ) -> (CooMatrix, CscMatrix, CsrMatrix, f64) {
        let sym = None;
        let nrow = 5;
        let ncol = 5;
        let max_nnz = 11; // more nnz than needed => OK
        let mut coo = CooMatrix::new(nrow, ncol, max_nnz, sym, one_based).unwrap();
        if shuffle_coo_entries {
            if duplicate_coo_entries {
                coo.put(4, 4, 9.0).unwrap();
                coo.put(0, 0, 1.0).unwrap();
                coo.put(1, 0, 3.0).unwrap();
                coo.put(2, 2, 5.0).unwrap();
                coo.put(0, 1, 2.0).unwrap();
                coo.put(1, 1, 2.0).unwrap(); // << 1st put
                coo.put(3, 2, 7.0).unwrap();
                coo.put(2, 3, 3.0).unwrap(); // << 1st put
                coo.put(3, 3, 8.0).unwrap();
                coo.put(2, 3, 3.0).unwrap(); // << 2nd put => duplicate
                coo.put(1, 1, 2.0).unwrap(); // << 2nd put => duplicate
            } else {
                coo.put(4, 4, 9.0).unwrap();
                coo.put(0, 0, 1.0).unwrap();
                coo.put(1, 0, 3.0).unwrap();
                coo.put(2, 2, 5.0).unwrap();
                coo.put(2, 3, 6.0).unwrap();
                coo.put(0, 1, 2.0).unwrap();
                coo.put(3, 2, 7.0).unwrap();
                coo.put(1, 1, 4.0).unwrap();
                coo.put(3, 3, 8.0).unwrap();
            }
        } else {
            if duplicate_coo_entries {
                coo.put(0, 0, 1.0).unwrap();
                coo.put(0, 1, 2.0).unwrap();
                coo.put(1, 0, 3.0).unwrap();
                coo.put(1, 1, 2.0).unwrap(); // << 1st put
                coo.put(1, 1, 2.0).unwrap(); // << 2nd put => duplicate
                coo.put(2, 2, 5.0).unwrap();
                coo.put(2, 3, 3.0).unwrap(); // << 2nd put => duplicate
                coo.put(2, 3, 3.0).unwrap(); // << 1st put
                coo.put(3, 2, 7.0).unwrap();
                coo.put(3, 3, 8.0).unwrap();
                coo.put(4, 4, 9.0).unwrap();
            } else {
                coo.put(0, 0, 1.0).unwrap();
                coo.put(0, 1, 2.0).unwrap();
                coo.put(1, 0, 3.0).unwrap();
                coo.put(1, 1, 4.0).unwrap();
                coo.put(2, 2, 5.0).unwrap();
                coo.put(2, 3, 6.0).unwrap();
                coo.put(3, 2, 7.0).unwrap();
                coo.put(3, 3, 8.0).unwrap();
                coo.put(4, 4, 9.0).unwrap();
            }
        }
        // CSC matrix
        let values = vec![
            1.0, 3.0, // j=0, p=(0),1
            2.0, 4.0, // j=1, p=(2),3
            5.0, 7.0, // j=2, p=(4),5
            6.0, 8.0, // j=3, p=(6),7
            9.0, //      j=4, p=(8)
        ]; //                 p=(9)
        let row_indices = vec![
            0, 1, //
            0, 1, //
            2, 3, //
            2, 3, //
            4, //
        ];
        let col_pointers = vec![0, 2, 4, 6, 8, 9];
        let csc = CscMatrix::new(nrow, ncol, col_pointers, row_indices, values, sym).unwrap();
        // CSR matrix
        let values = vec![
            1.0, 2.0, // i=0, p=(0),1
            3.0, 4.0, // i=1, p=(2),3
            5.0, 6.0, // i=2, p=(4),5
            7.0, 8.0, // i=3, p=(6),7
            9.0, //      i=4, p=(8)
        ]; //                 p=(9)
        let col_indices = vec![
            0, 1, //
            0, 1, //
            2, 3, //
            2, 3, //
            4, //
        ];
        let row_pointers = vec![0, 2, 4, 6, 8, 9];
        let csr = CsrMatrix::new(nrow, ncol, row_pointers, col_indices, values, sym).unwrap();
        (coo, csc, csr, 36.0)
    }

    /// Returns the matrix and its determinant
    ///
    /// Example from Intel MKL documentation
    ///
    /// ```text
    ///     9   1.5     6  0.75     3
    ///   1.5   0.5     .     .     .
    ///     6     .    12     .     .
    ///  0.75     .     . 0.625     .
    ///     3     .     .     .    16
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
    pub fn mkl_positive_definite_5x5_lower(one_based: bool) -> (CooMatrix, CscMatrix, CsrMatrix, f64) {
        let sym = Some(Symmetry::PositiveDefinite(Storage::Lower));
        let nrow = 5;
        let ncol = 5;
        let mut coo = CooMatrix::new(nrow, ncol, 9, sym, one_based).unwrap();
        coo.put(0, 0, 9.0).unwrap();
        coo.put(1, 1, 0.5).unwrap();
        coo.put(2, 2, 12.0).unwrap();
        coo.put(3, 3, 0.625).unwrap();
        coo.put(4, 4, 16.0).unwrap();
        coo.put(1, 0, 1.5).unwrap();
        coo.put(2, 0, 6.0).unwrap();
        coo.put(3, 0, 0.75).unwrap();
        coo.put(4, 0, 3.0).unwrap();
        // CSC matrix
        let values = vec![
            9.0, 1.5, 6.0, 0.75, 3.0,   // j=0 p=(0),1,2,3,4
            0.5,   //                      j=1 p=(5)
            12.0,  //                      j=2 p=(6)
            0.625, //                      j=3 p=(7)
            16.0,  //                      j=4 p=(8)
        ]; //                                  p=(9)
        let row_indices = vec![
            0, 1, 2, 3, 4, //
            1, //
            2, //
            3, //
            4, //
        ];
        let col_pointers = vec![0, 5, 6, 7, 8, 9];
        let csc = CscMatrix::new(nrow, ncol, col_pointers, row_indices, values, sym).unwrap();
        // CSR matrix
        let values = vec![
            9.0, //         i=0 p=(0)
            1.5, 0.5, //    i=1 p=(1),2
            6.0, 12.0, //   i=2 p=(3),4
            0.75, 0.625, // i=3 p=(5),6
            3.0, 16.0, //   i=4 p=(7),8
        ]; //                   p=(9)
        let col_indices = vec![
            0, //
            0, 1, //
            0, 2, //
            0, 3, //
            0, 4, //
        ];
        let row_pointers = vec![0, 1, 3, 5, 7, 9];
        let csr = CsrMatrix::new(nrow, ncol, row_pointers, col_indices, values, sym).unwrap();
        (coo, csc, csr, 9.0 / 4.0)
    }

    /// Returns the matrix and its determinant
    ///
    /// Example from Intel MKL documentation
    ///
    /// ```text
    ///     9   1.5     6  0.75     3
    ///   1.5   0.5     .     .     .
    ///     6     .    12     .     .
    ///  0.75     .     . 0.625     .
    ///     3     .     .     .    16
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
    pub fn mkl_positive_definite_5x5_upper(one_based: bool) -> (CooMatrix, CscMatrix, CsrMatrix, f64) {
        let sym = Some(Symmetry::PositiveDefinite(Storage::Upper));
        let nrow = 5;
        let ncol = 5;
        let mut coo = CooMatrix::new(nrow, ncol, 9, sym, one_based).unwrap();
        coo.put(0, 0, 9.0).unwrap();
        coo.put(0, 1, 1.5).unwrap();
        coo.put(1, 1, 0.5).unwrap();
        coo.put(0, 2, 6.0).unwrap();
        coo.put(2, 2, 12.0).unwrap();
        coo.put(0, 3, 0.75).unwrap();
        coo.put(3, 3, 0.625).unwrap();
        coo.put(0, 4, 3.0).unwrap();
        coo.put(4, 4, 16.0).unwrap();
        // CSC matrix
        let values = vec![
            9.0, //         j=0 p=(0)
            1.5, 0.5, //    j=1 p=(1),2
            6.0, 12.0, //   j=2 p=(3),4
            0.75, 0.625, // j=3 p=(5),6
            3.0, 16.0, //   j=4 p=(7),8
        ]; //                   p=(9)
        let row_indices = vec![
            0, //
            0, 1, //
            0, 2, //
            0, 3, //
            0, 4,
        ];
        let col_pointers = vec![0, 1, 3, 5, 7, 9];
        let csc = CscMatrix::new(nrow, ncol, col_pointers, row_indices, values, sym).unwrap();
        // CSR matrix
        let values = vec![
            9.0, 1.5, 6.0, 0.75, 3.0,   // i=0 p=(0),1,2,3,4
            0.5,   //                      i=1 p=(5)
            12.0,  //                      i=2 p=(6)
            0.625, //                      i=3 p=(7)
            16.0,  //                      i=4 p=(8)
        ]; //                                  p=(9)
        let col_indices = vec![
            0, 1, 2, 3, 4, //
            1, //
            2, //
            3, //
            4, //
        ];
        let row_pointers = vec![0, 5, 6, 7, 8, 9];
        let csr = CsrMatrix::new(nrow, ncol, row_pointers, col_indices, values, sym).unwrap();
        (coo, csc, csr, 9.0 / 4.0)
    }

    /// Returns the matrix and its determinant
    ///
    /// Example from Intel MKL documentation
    ///
    /// ```text
    ///     9   1.5     6  0.75     3
    ///   1.5   0.5     .     .     .
    ///     6     .    12     .     .
    ///  0.75     .     . 0.625     .
    ///     3     .     .     .    16
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
    pub fn mkl_symmetric_5x5_lower(
        one_based: bool,
        shuffle_coo_entries: bool,
        duplicate_coo_entries: bool,
    ) -> (CooMatrix, CscMatrix, CsrMatrix, f64) {
        let sym = Some(Symmetry::General(Storage::Lower));
        let nrow = 5;
        let ncol = 5;
        let max_nnz = 13;
        let mut coo = CooMatrix::new(nrow, ncol, max_nnz, sym, one_based).unwrap();
        if shuffle_coo_entries {
            if duplicate_coo_entries {
                // diagonal
                coo.put(0, 0, 5.0).unwrap(); // << duplicate
                coo.put(0, 0, 4.0).unwrap(); // << duplicate
                coo.put(1, 1, 0.5).unwrap();
                coo.put(2, 2, 12.0).unwrap();
                coo.put(3, 3, 0.625).unwrap();
                coo.put(4, 4, 16.0).unwrap();
                // lower diagonal
                coo.put(1, 0, 1.5).unwrap();
                coo.put(2, 0, 4.0).unwrap(); // << duplicate
                coo.put(2, 0, 2.0).unwrap(); // << duplicate
                coo.put(3, 0, 0.75).unwrap();
                coo.put(4, 0, 3.0).unwrap();
            } else {
                // diagonal
                coo.put(0, 0, 9.0).unwrap();
                coo.put(1, 1, 0.5).unwrap();
                coo.put(2, 2, 12.0).unwrap();
                coo.put(3, 3, 0.625).unwrap();
                coo.put(4, 4, 16.0).unwrap();
                // lower diagonal
                coo.put(1, 0, 1.5).unwrap();
                coo.put(2, 0, 6.0).unwrap();
                coo.put(3, 0, 0.75).unwrap();
                coo.put(4, 0, 3.0).unwrap();
            }
        } else {
            if duplicate_coo_entries {
                coo.put(0, 0, 9.0).unwrap();
                coo.put(1, 0, 1.5).unwrap();
                coo.put(1, 1, 0.25).unwrap(); // << duplicate
                coo.put(1, 1, 0.25).unwrap(); // << duplicate
                coo.put(2, 0, 6.0).unwrap();
                coo.put(2, 2, 12.0).unwrap();
                coo.put(3, 0, 0.75).unwrap();
                coo.put(3, 3, 0.625).unwrap();
                coo.put(4, 0, 2.0).unwrap(); // << duplicate
                coo.put(4, 0, 1.0).unwrap(); // << duplicate
                coo.put(4, 4, 16.0).unwrap();
            } else {
                coo.put(0, 0, 9.0).unwrap();
                coo.put(1, 0, 1.5).unwrap();
                coo.put(1, 1, 0.5).unwrap();
                coo.put(2, 0, 6.0).unwrap();
                coo.put(2, 2, 12.0).unwrap();
                coo.put(3, 0, 0.75).unwrap();
                coo.put(3, 3, 0.625).unwrap();
                coo.put(4, 0, 3.0).unwrap();
                coo.put(4, 4, 16.0).unwrap();
            }
        }
        // CSC matrix
        let values = vec![
            9.0, 1.5, 6.0, 0.75, 3.0,   // j=0 p=(0),1,2,3,4
            0.5,   //                      j=1 p=(5)
            12.0,  //                      j=2 p=(6)
            0.625, //                      j=3 p=(7)
            16.0,  //                      j=4 p=(8)
        ]; //                                  p=(9)
        let row_indices = vec![
            0, 1, 2, 3, 4, //
            1, //
            2, //
            3, //
            4, //
        ];
        let col_pointers = vec![0, 5, 6, 7, 8, 9];
        let csc = CscMatrix::new(nrow, ncol, col_pointers, row_indices, values, sym).unwrap();
        // CSR matrix
        let values = vec![
            9.0, //         i=0 p=(0)
            1.5, 0.5, //    i=1 p=(1),2
            6.0, 12.0, //   i=2 p=(3),4
            0.75, 0.625, // i=3 p=(5),6
            3.0, 16.0, //   i=4 p=(7),8
        ]; //                   p=(9)
        let col_indices = vec![
            0, //
            0, 1, //
            0, 2, //
            0, 3, //
            0, 4, //
        ];
        let row_pointers = vec![0, 1, 3, 5, 7, 9];
        let csr = CsrMatrix::new(nrow, ncol, row_pointers, col_indices, values, sym).unwrap();
        (coo, csc, csr, 9.0 / 4.0)
    }

    /// Returns the matrix and its determinant
    ///
    /// Example from Intel MKL documentation
    ///
    /// ```text
    ///     9   1.5     6  0.75     3
    ///   1.5   0.5     .     .     .
    ///     6     .    12     .     .
    ///  0.75     .     . 0.625     .
    ///     3     .     .     .    16
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
    pub fn mkl_symmetric_5x5_upper(
        one_based: bool,
        shuffle_coo_entries: bool,
        duplicate_coo_entries: bool,
    ) -> (CooMatrix, CscMatrix, CsrMatrix, f64) {
        let sym = Some(Symmetry::General(Storage::Upper));
        let nrow = 5;
        let ncol = 5;
        let max_nnz = 15;
        let mut coo = CooMatrix::new(nrow, ncol, max_nnz, sym, one_based).unwrap();
        if shuffle_coo_entries {
            if duplicate_coo_entries {
                coo.put(0, 0, 6.0).unwrap(); // << duplicate
                coo.put(0, 0, 3.0).unwrap(); // << duplicate
                coo.put(0, 1, 1.5).unwrap();
                coo.put(1, 1, 0.5).unwrap();
                coo.put(0, 2, 5.0).unwrap(); // << duplicate
                coo.put(0, 2, 1.0).unwrap(); // << duplicate
                coo.put(2, 2, 12.0).unwrap();
                coo.put(0, 3, 0.75).unwrap();
                coo.put(3, 3, 0.625).unwrap();
                coo.put(0, 4, 3.0).unwrap();
                coo.put(4, 4, 16.0).unwrap();
            } else {
                coo.put(2, 2, 12.0).unwrap();
                coo.put(0, 0, 9.0).unwrap();
                coo.put(3, 3, 0.625).unwrap();
                coo.put(0, 1, 1.5).unwrap();
                coo.put(0, 2, 6.0).unwrap();
                coo.put(4, 4, 16.0).unwrap();
                coo.put(0, 3, 0.75).unwrap();
                coo.put(1, 1, 0.5).unwrap();
                coo.put(0, 4, 3.0).unwrap();
            }
        } else {
            if duplicate_coo_entries {
                coo.put(0, 0, 9.0).unwrap();
                coo.put(0, 1, 1.5).unwrap();
                coo.put(0, 2, 6.0).unwrap();
                coo.put(0, 3, 0.75).unwrap();
                coo.put(0, 4, 3.0).unwrap();
                coo.put(1, 1, 0.5).unwrap();
                coo.put(2, 2, 2.0).unwrap(); // << duplicate
                coo.put(2, 2, 2.0).unwrap(); // << duplicate
                coo.put(2, 2, 2.0).unwrap(); // << duplicate
                coo.put(2, 2, 2.0).unwrap(); // << duplicate
                coo.put(2, 2, 2.0).unwrap(); // << duplicate
                coo.put(2, 2, 2.0).unwrap(); // << duplicate
                coo.put(3, 3, 0.625).unwrap();
                coo.put(4, 4, 16.0).unwrap();
            } else {
                coo.put(0, 0, 9.0).unwrap();
                coo.put(0, 1, 1.5).unwrap();
                coo.put(0, 2, 6.0).unwrap();
                coo.put(0, 3, 0.75).unwrap();
                coo.put(0, 4, 3.0).unwrap();
                coo.put(1, 1, 0.5).unwrap();
                coo.put(2, 2, 12.0).unwrap();
                coo.put(3, 3, 0.625).unwrap();
                coo.put(4, 4, 16.0).unwrap();
            }
        }
        // CSC matrix
        let values = vec![
            9.0, //         j=0 p=(0)
            1.5, 0.5, //    j=1 p=(1),2
            6.0, 12.0, //   j=2 p=(3),4
            0.75, 0.625, // j=3 p=(5),6
            3.0, 16.0, //   j=4 p=(7),8
        ]; //                   p=(9)
        let row_indices = vec![
            0, //
            0, 1, //
            0, 2, //
            0, 3, //
            0, 4,
        ];
        let col_pointers = vec![0, 1, 3, 5, 7, 9];
        let csc = CscMatrix::new(nrow, ncol, col_pointers, row_indices, values, sym).unwrap();
        // CSR matrix
        let values = vec![
            9.0, 1.5, 6.0, 0.75, 3.0,   // i=0 p=(0),1,2,3,4
            0.5,   //                      i=1 p=(5)
            12.0,  //                      i=2 p=(6)
            0.625, //                      i=3 p=(7)
            16.0,  //                      i=4 p=(8)
        ]; //                                  p=(9)
        let col_indices = vec![
            0, 1, 2, 3, 4, //
            1, //
            2, //
            3, //
            4, //
        ];
        let row_pointers = vec![0, 5, 6, 7, 8, 9];
        let csr = CsrMatrix::new(nrow, ncol, row_pointers, col_indices, values, sym).unwrap();
        (coo, csc, csr, 9.0 / 4.0)
    }

    /// Returns the matrix and its determinant
    ///
    /// Example from Intel MKL documentation
    ///
    /// ```text
    ///     9   1.5     6  0.75     3
    ///   1.5   0.5     .     .     .
    ///     6     .    12     .     .
    ///  0.75     .     . 0.625     .
    ///     3     .     .     .    16
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
    pub fn mkl_symmetric_5x5_full(one_based: bool) -> (CooMatrix, CscMatrix, CsrMatrix, f64) {
        let sym = Some(Symmetry::General(Storage::Full));
        let nrow = 5;
        let ncol = 5;
        let max_nnz = 13;
        let mut coo = CooMatrix::new(nrow, ncol, max_nnz, sym, one_based).unwrap();
        coo.put(0, 0, 9.0).unwrap();
        coo.put(0, 1, 1.5).unwrap();
        coo.put(0, 2, 6.0).unwrap();
        coo.put(0, 3, 0.75).unwrap();
        coo.put(0, 4, 3.0).unwrap();
        coo.put(1, 0, 1.5).unwrap();
        coo.put(1, 1, 0.5).unwrap();
        coo.put(2, 0, 6.0).unwrap();
        coo.put(2, 2, 12.0).unwrap();
        coo.put(3, 0, 0.75).unwrap();
        coo.put(3, 3, 0.625).unwrap();
        coo.put(4, 0, 3.0).unwrap();
        coo.put(4, 4, 16.0).unwrap();
        // CSC matrix
        let values = vec![
            9.0, 1.5, 6.0, 0.75, 3.0, // j=0 p=(0),1,2,3,4
            1.5, 0.5, //                 j=1 p=(5),6
            6.0, 12.0, //                j=2 p=(7),8
            0.75, 0.625, //              j=3 p=(9),10
            3.0, 16.0, //                j=4 p=(11),12
        ]; //                                p=(13)
        let row_indices = vec![
            0, 1, 2, 3, 4, //
            0, 1, //
            0, 2, //
            0, 3, //
            0, 4,
        ];
        let col_pointers = vec![0, 5, 7, 9, 11, 13];
        let csc = CscMatrix::new(nrow, ncol, col_pointers, row_indices, values, sym).unwrap();
        // CSR matrix
        let values = vec![
            9.0, 1.5, 6.0, 0.75, 3.0, // i=0 p=(0),1,2,3,4
            1.5, 0.5, //                 i=1 p=(5),6
            6.0, 12.0, //                i=2 p=(7),8
            0.75, 0.625, //              i=3 p=(9),10
            3.0, 16.0, //                i=4 p=(11),12
        ]; //                                p=(13)
        let col_indices = vec![
            0, 1, 2, 3, 4, //
            0, 1, //
            0, 2, //
            0, 3, //
            0, 4,
        ];
        let row_pointers = vec![0, 5, 7, 9, 11, 13];
        let csr = CsrMatrix::new(nrow, ncol, row_pointers, col_indices, values, sym).unwrap();
        (coo, csc, csr, 9.0 / 4.0)
    }

    /// Returns a (1 x 2) rectangular matrix (COO, CSC, and CSR)
    ///
    /// Note: the last return value is not the determinant, but a PLACEHOLDER
    ///
    /// ```text
    /// ┌       ┐
    /// │ 10 20 │
    /// └       ┘
    /// ```
    pub fn rectangular_1x2(
        one_based: bool,
        shuffle_coo_entries: bool,
        duplicate_coo_entries: bool,
    ) -> (CooMatrix, CscMatrix, CsrMatrix, f64) {
        let sym = None;
        let nrow = 1;
        let ncol = 2;
        let max_nnz = 10;
        let mut coo = CooMatrix::new(nrow, ncol, max_nnz, sym, one_based).unwrap();
        if shuffle_coo_entries {
            if duplicate_coo_entries {
                coo.put(0, 1, 2.0).unwrap();
                coo.put(0, 1, 2.0).unwrap();
                coo.put(0, 1, 10.0).unwrap();
                coo.put(0, 1, 2.0).unwrap();
                coo.put(0, 1, 2.0).unwrap();
                coo.put(0, 1, 2.0).unwrap();
                coo.put(0, 0, 10.0).unwrap();
            } else {
                coo.put(0, 1, 20.0).unwrap();
                coo.put(0, 0, 10.0).unwrap();
            }
        } else {
            if duplicate_coo_entries {
                coo.put(0, 0, 10.0).unwrap();
                coo.put(0, 1, 10.0).unwrap();
                coo.put(0, 1, 2.0).unwrap();
                coo.put(0, 1, 2.0).unwrap();
                coo.put(0, 1, 2.0).unwrap();
                coo.put(0, 1, 2.0).unwrap();
                coo.put(0, 1, 2.0).unwrap();
            } else {
                coo.put(0, 0, 10.0).unwrap();
                coo.put(0, 1, 20.0).unwrap();
            }
        }
        // CSC
        let values = vec![
            10.0, // j=0, p=(0)
            20.0, // j=1, p=(1)
        ]; //             p=(2)
        let row_indices = vec![0, 0];
        let col_pointers = vec![0, 1, 2];
        let csc = CscMatrix::new(nrow, ncol, col_pointers, row_indices, values, sym).unwrap();
        // CSR
        let values = vec![
            10.0, 20.0, // i=0, p=(0),1
        ]; //                   p=(2)
        let col_indices = vec![0, 1];
        let row_pointers = vec![0, 2];
        let csr = CsrMatrix::new(nrow, ncol, row_pointers, col_indices, values, sym).unwrap();
        (coo, csc, csr, PLACEHOLDER)
    }

    /// Returns a (1 x 7) rectangular matrix (COO, CSC, and CSR)
    ///
    /// Note: the last return value is not the determinant, but a PLACEHOLDER
    ///
    /// ```text
    /// ┌               ┐
    /// │ 1 . 3 . 5 . 7 │
    /// └               ┘
    /// ```
    pub fn rectangular_1x7() -> (CooMatrix, CscMatrix, CsrMatrix, f64) {
        let sym = None;
        let nrow = 1;
        let ncol = 7;
        let mut coo = CooMatrix::new(nrow, ncol, 4, sym, false).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(0, 2, 3.0).unwrap();
        coo.put(0, 4, 5.0).unwrap();
        coo.put(0, 6, 7.0).unwrap();
        // CSC
        let values = vec![
            1.0, // j=0, p=(0)
            //      j=1, p=(1)
            3.0, // j=2, p=(1)
            //      j=3, p=(2)
            5.0, // j=4, p=(2)
            //      j=5, p=(3)
            7.0, // j=6, p=(3)
        ]; //            p=(4)
        let row_indices = vec![0, 0, 0, 0];
        let col_pointers = vec![0, 1, 1, 2, 2, 3, 3, 4];
        let csc = CscMatrix::new(nrow, ncol, col_pointers, row_indices, values, sym).unwrap();
        // CSR
        let values = vec![
            1.0, 3.0, 5.0, 7.0, // i=0, p=(0),1,2,3
        ]; //                           p=(4)
        let col_indices = vec![0, 2, 4, 6];
        let row_pointers = vec![0, 4];
        let csr = CsrMatrix::new(nrow, ncol, row_pointers, col_indices, values, sym).unwrap();
        (coo, csc, csr, PLACEHOLDER)
    }

    /// Returns a (7 x 1) rectangular matrix
    ///
    /// Note: the last return value is not the determinant, but a PLACEHOLDER
    ///
    /// ```text
    /// ┌   ┐
    /// │ . │
    /// │ 2 │
    /// │ . │
    /// │ 4 │
    /// │ . │
    /// │ 6 │
    /// │ . │
    /// └   ┘
    /// ```
    pub fn rectangular_7x1() -> (CooMatrix, CscMatrix, CsrMatrix, f64) {
        let sym = None;
        let nrow = 7;
        let ncol = 1;
        let mut coo = CooMatrix::new(nrow, ncol, 3, sym, false).unwrap();
        coo.put(1, 0, 2.0).unwrap();
        coo.put(3, 0, 4.0).unwrap();
        coo.put(5, 0, 6.0).unwrap();
        // CSC matrix
        let values = vec![
            2.0, 4.0, 6.0, // j=0, p=(0),1,2
        ]; //                      p=(3)
        let row_indices = vec![1, 3, 5];
        let col_pointers = vec![0, 3];
        let csc = CscMatrix::new(nrow, ncol, col_pointers, row_indices, values, sym).unwrap();
        // CSR matrix
        let values = vec![
            //      i=0, p=(0)
            2.0, // i=1, p=(0)
            //      i=2, p=(1)
            4.0, // i=3, p=(1)
            //      i=4, p=(2)
            6.0, // i=5, p=(2)
                 // i=6, p=(3)
        ]; //            p=(3)
        let col_indices = vec![0, 0, 0];
        let row_pointers = vec![0, 0, 1, 1, 2, 2, 3, 3];
        let csr = CsrMatrix::new(nrow, ncol, row_pointers, col_indices, values, sym).unwrap();
        (coo, csc, csr, PLACEHOLDER)
    }

    /// Returns a (3 x 4) rectangular matrix
    ///
    /// Note: the last return value is not the determinant, but a PLACEHOLDER
    ///
    /// ```text
    ///  5.0, -2.0, 0.0, 1.0,
    /// 10.0, -4.0, 0.0, 2.0,
    /// 15.0, -6.0, 0.0, 3.0,
    /// ```
    pub fn rectangular_3x4() -> (CooMatrix, CscMatrix, CsrMatrix, f64) {
        let sym = None;
        let nrow = 3;
        let ncol = 4;
        let mut coo = CooMatrix::new(nrow, ncol, 9, sym, false).unwrap();
        coo.put(0, 0, 5.0).unwrap();
        coo.put(1, 0, 10.0).unwrap();
        coo.put(2, 0, 15.0).unwrap();
        coo.put(0, 1, -2.0).unwrap();
        coo.put(1, 1, -4.0).unwrap();
        coo.put(2, 1, -6.0).unwrap();
        coo.put(0, 3, 1.0).unwrap();
        coo.put(1, 3, 2.0).unwrap();
        coo.put(2, 3, 3.0).unwrap();
        // CSC matrix
        let col_pointers = vec![0, 3, 6, 6, 9];
        let row_indices = vec![
            0, 1, 2, // j=0, p=(0),1,2
            0, 1, 2, // j=1, p=(3),4,5
            //          j=2, p=(6)
            0, 1, 2, // j=3, p=(6),7,8
        ]; //                  (9)
        let values = vec![
            5.0, 10.0, 15.0, //  j=0, p=(0),1,2
            -2.0, -4.0, -6.0, // j=1, p=(3),4,5
            //                   j=2, p=(6)
            1.0, 2.0, 3.0, //    j=3, p=(6),7,8
        ]; //                           (9)
        let csc = CscMatrix::new(nrow, ncol, col_pointers, row_indices, values, sym).unwrap();
        // CSR matrix
        let row_pointers = vec![0, 3, 6, 9];
        let col_indices = vec![
            0, 1, 3, // i=0, p=(0),1,2
            0, 1, 3, // i=1, p=(3),4,5
            0, 1, 3, // i=2, p=(6),7,8
        ]; //                  (9)
        let values = vec![
            5.0, -2.0, 1.0, //  i=0, p=(0),1,2
            10.0, -4.0, 2.0, // i=1, p=(3),4,5
            15.0, -6.0, 3.0, // i=2, p=(6),7,8
        ]; //                          (9)
        let csr = CsrMatrix::new(nrow, ncol, row_pointers, col_indices, values, sym).unwrap();
        (coo, csc, csr, PLACEHOLDER)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Samples;
    use crate::{CooMatrix, CscMatrix, CsrMatrix};
    use russell_chk::approx_eq;
    use russell_lab::{mat_approx_eq, mat_inverse, Matrix};

    /// Checks the samples
    ///
    /// ```text
    /// CSC
    /// nrow ≥ 1
    /// ncol ≥ 1
    /// col_pointers.len() == ncol + 1
    /// nnz = col_pointers[ncol] ≥ 1
    /// row_indices.len() == nnz
    /// values.len() == nnz
    /// ```
    ///
    /// ```text
    /// CSR
    /// nrow ≥ 1
    /// ncol ≥ 1
    /// row_pointers.len() == nrow + 1
    /// nnz = row_pointers[nrow] ≥ 1
    /// col_indices.len() == nnz
    /// values.len() == nnz
    /// ```
    fn check(coo: &CooMatrix, csc: &CscMatrix, csr: &CsrMatrix) {
        // COO
        let max_nnz = coo.max_nnz;
        assert!(coo.nrow >= 1);
        assert!(coo.ncol >= 1);
        assert!(coo.nnz >= 1);
        assert!(coo.nnz <= max_nnz);
        assert!(max_nnz >= 1);
        // CSC
        assert!(csc.nrow >= 1);
        assert!(csc.ncol >= 1);
        assert_eq!(csc.col_pointers.len(), csc.ncol + 1);
        assert!(csc.col_pointers[csc.ncol] >= 1);
        let nnz = csc.col_pointers[csc.ncol] as usize;
        assert!(nnz <= max_nnz);
        assert_eq!(csc.row_indices.len(), nnz);
        assert_eq!(csc.values.len(), nnz);
        // CSC vs COO
        assert_eq!(csc.symmetry, coo.symmetry);
        assert_eq!(csc.nrow, coo.nrow);
        assert_eq!(csc.ncol, coo.ncol);
        // CSR
        assert!(csr.nrow >= 1);
        assert!(csr.ncol >= 1);
        assert_eq!(csr.row_pointers.len(), csr.nrow + 1);
        assert!(csr.row_pointers[csr.nrow] >= 1);
        let nnz = csr.row_pointers[csr.nrow] as usize;
        assert!(nnz <= max_nnz);
        assert_eq!(csr.col_indices.len(), nnz);
        assert_eq!(csr.values.len(), nnz);
        // CSR vs COO
        assert_eq!(csr.symmetry, coo.symmetry);
        assert_eq!(csr.nrow, coo.nrow);
        assert_eq!(csr.ncol, coo.ncol);
    }

    #[test]
    fn samples_are_correct() {
        let correct = &[
            [123.0], //
        ];
        let a = Matrix::from(correct);
        let mut ai = Matrix::new(1, 1);
        let correct_det = mat_inverse(&mut ai, &a).unwrap();
        for (coo, csc, csr, det) in [
            Samples::tiny_1x1(false), //
            Samples::tiny_1x1(true),  //
        ] {
            let mat = coo.as_dense();
            approx_eq(det, correct_det, 1e-13);
            mat_approx_eq(&mat, correct, 1e-15);
            check(&coo, &csc, &csr);
        }

        // ----------------------------------------------------------------------------

        let correct = &[
            [1.0, 0.0, 2.0], //
            [0.0, 0.0, 3.0], //
            [4.0, 5.0, 6.0], //
        ];
        let a = Matrix::from(correct);
        let mut ai = Matrix::new(3, 3);
        let correct_det = mat_inverse(&mut ai, &a).unwrap();
        for (coo, csc, csr, det) in [
            Samples::unsymmetric_3x3(false, false, false),
            Samples::unsymmetric_3x3(false, true, false),
            Samples::unsymmetric_3x3(false, true, false),
            Samples::unsymmetric_3x3(false, true, true),
            Samples::unsymmetric_3x3(true, false, false),
            Samples::unsymmetric_3x3(true, true, false),
            Samples::unsymmetric_3x3(true, true, false),
            Samples::unsymmetric_3x3(true, true, true),
        ] {
            let mat = coo.as_dense();
            approx_eq(det, correct_det, 1e-13);
            mat_approx_eq(&mat, correct, 1e-15);
            check(&coo, &csc, &csr);
        }

        // ----------------------------------------------------------------------------

        let correct = &[
            [2.0, 3.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 4.0, 0.0, 6.0],
            [0.0, -1.0, -3.0, 2.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 4.0, 2.0, 0.0, 1.0],
        ];
        let a = Matrix::from(correct);
        let mut ai = Matrix::new(5, 5);
        let correct_det = mat_inverse(&mut ai, &a).unwrap();
        for (coo, csc, csr, det) in [
            Samples::umfpack_unsymmetric_5x5(false),
            Samples::umfpack_unsymmetric_5x5(true),
        ] {
            let mat = coo.as_dense();
            approx_eq(det, correct_det, 1e-13);
            mat_approx_eq(&mat, correct, 1e-15);
            check(&coo, &csc, &csr);
        }

        // ----------------------------------------------------------------------------

        let correct = &[
            [1.0, -1.0, 0.0, -3.0, 0.0],
            [-2.0, 5.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 6.0, 4.0],
            [-4.0, 0.0, 2.0, 7.0, 0.0],
            [0.0, 8.0, 0.0, 0.0, -5.0],
        ];
        let a = Matrix::from(correct);
        let mut ai = Matrix::new(5, 5);
        let correct_det = mat_inverse(&mut ai, &a).unwrap();
        for (coo, csc, csr, det) in [
            Samples::mkl_unsymmetric_5x5(false), //
        ] {
            let mat = coo.as_dense();
            approx_eq(det, correct_det, 1e-13);
            mat_approx_eq(&mat, correct, 1e-15);
            check(&coo, &csc, &csr);
        }

        // ----------------------------------------------------------------------------

        let correct = &[
            [1.0, 2.0, 0.0, 0.0, 0.0],
            [3.0, 4.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 6.0, 0.0],
            [0.0, 0.0, 7.0, 8.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 9.0],
        ];
        let a = Matrix::from(correct);
        let mut ai = Matrix::new(5, 5);
        let correct_det = mat_inverse(&mut ai, &a).unwrap();
        for (coo, csc, csr, det) in [
            Samples::block_unsymmetric_5x5(false, false, false),
            Samples::block_unsymmetric_5x5(false, true, false),
            Samples::block_unsymmetric_5x5(false, false, true),
            Samples::block_unsymmetric_5x5(false, true, true),
            Samples::block_unsymmetric_5x5(true, false, false),
            Samples::block_unsymmetric_5x5(true, true, false),
            Samples::block_unsymmetric_5x5(true, false, true),
            Samples::block_unsymmetric_5x5(true, true, true),
        ] {
            let mat = coo.as_dense();
            approx_eq(det, correct_det, 1e-13);
            mat_approx_eq(&mat, correct, 1e-15);
            check(&coo, &csc, &csr);
        }

        // ----------------------------------------------------------------------------

        let correct = &[
            [9.0, 1.5, 6.0, 0.75, 3.0],
            [1.5, 0.5, 0.0, 0.0, 0.0],
            [6.0, 0.0, 12.0, 0.0, 0.0],
            [0.75, 0.0, 0.0, 0.625, 0.0],
            [3.0, 0.0, 0.0, 0.0, 16.0],
        ];
        let a = Matrix::from(correct);
        let mut ai = Matrix::new(5, 5);
        let correct_det = mat_inverse(&mut ai, &a).unwrap();
        for (coo, csc, csr, det) in [
            Samples::mkl_positive_definite_5x5_lower(false),
            Samples::mkl_positive_definite_5x5_lower(true),
            Samples::mkl_positive_definite_5x5_upper(false),
            Samples::mkl_positive_definite_5x5_upper(true),
            Samples::mkl_symmetric_5x5_lower(false, false, false),
            Samples::mkl_symmetric_5x5_lower(false, true, false),
            Samples::mkl_symmetric_5x5_lower(false, false, true),
            Samples::mkl_symmetric_5x5_lower(false, true, true),
            Samples::mkl_symmetric_5x5_lower(true, false, false),
            Samples::mkl_symmetric_5x5_lower(true, true, false),
            Samples::mkl_symmetric_5x5_lower(true, false, true),
            Samples::mkl_symmetric_5x5_lower(true, true, true),
            Samples::mkl_symmetric_5x5_upper(false, false, false),
            Samples::mkl_symmetric_5x5_upper(false, true, false),
            Samples::mkl_symmetric_5x5_upper(false, false, true),
            Samples::mkl_symmetric_5x5_upper(false, true, true),
            Samples::mkl_symmetric_5x5_upper(true, false, false),
            Samples::mkl_symmetric_5x5_upper(true, true, false),
            Samples::mkl_symmetric_5x5_upper(true, false, true),
            Samples::mkl_symmetric_5x5_upper(true, true, true),
            Samples::mkl_symmetric_5x5_full(false),
            Samples::mkl_symmetric_5x5_full(true),
        ] {
            let mat = coo.as_dense();
            approx_eq(det, correct_det, 1e-13);
            mat_approx_eq(&mat, correct, 1e-15);
            check(&coo, &csc, &csr);
        }

        // ----------------------------------------------------------------------------

        let correct = &[[10.0, 20.0]];
        for (coo, csc, csr, _) in [
            Samples::rectangular_1x2(false, false, false),
            Samples::rectangular_1x2(false, true, false),
            Samples::rectangular_1x2(false, false, true),
            Samples::rectangular_1x2(false, true, true),
            Samples::rectangular_1x2(true, false, false),
            Samples::rectangular_1x2(true, true, false),
            Samples::rectangular_1x2(true, false, true),
            Samples::rectangular_1x2(true, true, true),
        ] {
            let mat = coo.as_dense();
            mat_approx_eq(&mat, correct, 1e-15);
            check(&coo, &csc, &csr);
        }

        // ----------------------------------------------------------------------------

        let correct = &[[1.0, 0.0, 3.0, 0.0, 5.0, 0.0, 7.0]];
        let (coo, csc, csr, _) = Samples::rectangular_1x7();
        let mat = coo.as_dense();
        mat_approx_eq(&mat, correct, 1e-15);
        check(&coo, &csc, &csr);

        // ----------------------------------------------------------------------------

        let correct = &[[0.0], [2.0], [0.0], [4.0], [0.0], [6.0], [0.0]];
        let (coo, csc, csr, _) = Samples::rectangular_7x1();
        let mat = coo.as_dense();
        mat_approx_eq(&mat, correct, 1e-15);
        check(&coo, &csc, &csr);

        // ----------------------------------------------------------------------------

        let correct = &[
            [5.0, -2.0, 0.0, 1.0],  //
            [10.0, -4.0, 0.0, 2.0], //
            [15.0, -6.0, 0.0, 3.0], //
        ];
        let (coo, csc, csr, _) = Samples::rectangular_3x4();
        let mat = coo.as_dense();
        mat_approx_eq(&mat, correct, 1e-15);
        check(&coo, &csc, &csr);
    }
}
