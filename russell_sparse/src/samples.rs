use crate::{CooMatrix, CscMatrix, CsrMatrix, Storage, Symmetry};

const PLACEHOLDER: f64 = f64::MAX;

/// Holds some samples of small sparse matrices
pub struct Samples {}

impl Samples {
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
        // CSC matrix
        let csc = CscMatrix {
            symmetry: None,
            nrow: 5,
            ncol: 5,
            values: vec![
                2.0, 3.0, //            j=0, p=( 0),1
                3.0, -1.0, 4.0, //      j=1, p=( 2),3,4
                4.0, -3.0, 1.0, 2.0, // j=2, p=( 5),6,7,8
                2.0, //                 j=3, p=( 9)
                6.0, 1.0, //            j=4, p=(10),11
            ], //                            p=(12)
            row_indices: vec![
                0, 1, //
                0, 2, 4, //
                1, 2, 3, 4, //
                2, //
                1, 4, //
            ],
            col_pointers: vec![0, 2, 5, 9, 10, 12],
        };
        // CSR matrix
        let csr = CsrMatrix {
            symmetry: None,
            nrow: 5,
            ncol: 5,
            values: vec![
                2.0, 3.0, //        i=0, p=(0),1
                3.0, 4.0, 6.0, //   i=1, p=(2),3,4
                -1.0, -3.0, 2.0, // i=2, p=(5),6,7
                1.0, //             i=3, p=(8)
                4.0, 2.0, 1.0, //   i=4, p=(9),10,11
                     //                  p=(12)
            ],
            col_indices: vec![
                0, 1, //
                0, 2, 4, //
                1, 2, 3, //
                2, //
                1, 2, 4, //
            ],
            row_pointers: vec![0, 2, 5, 8, 9, 12],
        };
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
        // CSC matrix
        let csc = CscMatrix {
            symmetry: None,
            nrow: 5,
            ncol: 5,
            values: vec![
                1.0, -2.0, -4.0, // j=0, p=( 0),1,2
                -1.0, 5.0, 8.0, //  j=1, p=( 3),4,5
                4.0, 2.0, //        j=2, p=( 6),7
                -3.0, 6.0, 7.0, //  j=3, p=( 8),9,10
                4.0, -5.0, //       j=4, p=(11),12
            ], //                        p=(13)
            row_indices: vec![
                0, 1, 3, //
                0, 1, 4, //
                2, 3, //
                0, 2, 3, //
                2, 4, //
            ],
            col_pointers: vec![0, 3, 6, 8, 11, 13],
        };
        // CSR matrix
        let csr = CsrMatrix {
            symmetry: None,
            nrow: 5,
            ncol: 5,
            values: vec![
                1.0, -1.0, -3.0, // i=0, p=( 0),1,2
                -2.0, 5.0, //       i=1, p=( 3),4
                4.0, 6.0, 4.0, //   i=2, p=( 5),6,7
                -4.0, 2.0, 7.0, //  i=3, p=( 8),9,10
                8.0, -5.0, //       i=4, p=(11),12
            ], //                        p=(13)
            col_indices: vec![
                0, 1, 3, //
                0, 1, //
                2, 3, 4, //
                0, 2, 3, //
                1, 4, //
            ],
            row_pointers: vec![0, 3, 5, 8, 11, 13],
        };
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
        let max = 11; // more nnz than needed => OK
        let mut coo = CooMatrix::new(5, 5, max, None, one_based).unwrap();
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
        let csc = CscMatrix {
            symmetry: None,
            nrow: 5,
            ncol: 5,
            values: vec![
                1.0, 3.0, // j=0, p=(0),1
                2.0, 4.0, // j=1, p=(2),3
                5.0, 7.0, // j=2, p=(4),5
                6.0, 8.0, // j=3, p=(6),7
                9.0, //      j=4, p=(8)
            ], //                 p=(9)
            row_indices: vec![
                0, 1, //
                0, 1, //
                2, 3, //
                2, 3, //
                4, //
            ],
            col_pointers: vec![0, 2, 4, 6, 8, 9],
        };
        // CSR matrix
        let csr = CsrMatrix {
            symmetry: None,
            nrow: 5,
            ncol: 5,
            values: vec![
                1.0, 2.0, // i=0, p=(0),1
                3.0, 4.0, // i=1, p=(2),3
                5.0, 6.0, // i=2, p=(4),5
                7.0, 8.0, // i=3, p=(6),7
                9.0, //      i=4, p=(8)
            ], //                 p=(9)
            col_indices: vec![
                0, 1, //
                0, 1, //
                2, 3, //
                2, 3, //
                4, //
            ],
            row_pointers: vec![0, 2, 4, 6, 8, 9],
        };
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
        // CSC matrix
        let csc = CscMatrix {
            symmetry: None,
            nrow: 5,
            ncol: 5,
            values: vec![
                9.0, 1.5, 6.0, 0.75, 3.0,   // j=0 p=(0),1,2,3,4
                0.5,   //                      j=1 p=(5)
                12.0,  //                      j=2 p=(6)
                0.625, //                      j=3 p=(7)
                16.0,  //                      j=4 p=(8)
            ], //                                  p=(9)
            row_indices: vec![
                0, 1, 2, 3, 4, //
                1, //
                2, //
                3, //
                4, //
            ],
            col_pointers: vec![0, 5, 6, 7, 8, 9],
        };
        // CSR matrix
        let csr = CsrMatrix {
            symmetry: None,
            nrow: 5,
            ncol: 5,
            values: vec![
                9.0, //         i=0 p=(0)
                1.5, 0.5, //    i=1 p=(1),2
                6.0, 12.0, //   i=2 p=(3),4
                0.75, 0.625, // i=3 p=(5),6
                3.0, 16.0, //   i=4 p=(7),8
            ], //                   p=(9)
            col_indices: vec![
                0, //
                0, 1, //
                0, 2, //
                0, 3, //
                0, 4, //
            ],
            row_pointers: vec![0, 1, 3, 5, 7, 9],
        };
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
        // CSC matrix
        let csc = CscMatrix {
            symmetry: None,
            nrow: 5,
            ncol: 5,
            values: vec![
                9.0, //         j=0 p=(0)
                1.5, 0.5, //    j=1 p=(1),2
                6.0, 12.0, //   j=2 p=(3),4
                0.75, 0.625, // j=3 p=(5),6
                3.0, 16.0, //   j=4 p=(7),8
            ], //                   p=(9)
            row_indices: vec![
                0, //
                0, 1, //
                0, 2, //
                0, 3, //
                0, 4,
            ],
            col_pointers: vec![0, 1, 3, 5, 7, 9],
        };
        // CSR matrix
        let csr = CsrMatrix {
            symmetry: None,
            nrow: 5,
            ncol: 5,
            values: vec![
                9.0, 1.5, 6.0, 0.75, 3.0,   // i=0 p=(0),1,2,3,4
                0.5,   //                      i=1 p=(5)
                12.0,  //                      i=2 p=(6)
                0.625, //                      i=3 p=(7)
                16.0,  //                      i=4 p=(8)
            ], //                                  p=(9)
            col_indices: vec![
                0, 1, 2, 3, 4, //
                1, //
                2, //
                3, //
                4, //
            ],
            row_pointers: vec![0, 5, 6, 7, 8, 9],
        };
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
        let mut coo = CooMatrix::new(5, 5, 20, sym, one_based).unwrap();
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
        let csc = CscMatrix {
            symmetry: None,
            nrow: 5,
            ncol: 5,
            values: vec![
                9.0, 1.5, 6.0, 0.75, 3.0,   // j=0 p=(0),1,2,3,4
                0.5,   //                      j=1 p=(5)
                12.0,  //                      j=2 p=(6)
                0.625, //                      j=3 p=(7)
                16.0,  //                      j=4 p=(8)
            ], //                                  p=(9)
            row_indices: vec![
                0, 1, 2, 3, 4, //
                1, //
                2, //
                3, //
                4, //
            ],
            col_pointers: vec![0, 5, 6, 7, 8, 9],
        };
        // CSR matrix
        let csr = CsrMatrix {
            symmetry: None,
            nrow: 5,
            ncol: 5,
            values: vec![
                9.0, //         i=0 p=(0)
                1.5, 0.5, //    i=1 p=(1),2
                6.0, 12.0, //   i=2 p=(3),4
                0.75, 0.625, // i=3 p=(5),6
                3.0, 16.0, //   i=4 p=(7),8
            ], //                   p=(9)
            col_indices: vec![
                0, //
                0, 1, //
                0, 2, //
                0, 3, //
                0, 4, //
            ],
            row_pointers: vec![0, 1, 3, 5, 7, 9],
        };
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
        let mut coo = CooMatrix::new(5, 5, 20, sym, one_based).unwrap();
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
        let csc = CscMatrix {
            symmetry: None,
            nrow: 5,
            ncol: 5,
            values: vec![
                9.0, //         j=0 p=(0)
                1.5, 0.5, //    j=1 p=(1),2
                6.0, 12.0, //   j=2 p=(3),4
                0.75, 0.625, // j=3 p=(5),6
                3.0, 16.0, //   j=4 p=(7),8
            ], //                   p=(9)
            row_indices: vec![
                0, //
                0, 1, //
                0, 2, //
                0, 3, //
                0, 4,
            ],
            col_pointers: vec![0, 1, 3, 5, 7, 9],
        };
        // CSR matrix
        let csr = CsrMatrix {
            symmetry: None,
            nrow: 5,
            ncol: 5,
            values: vec![
                9.0, 1.5, 6.0, 0.75, 3.0,   // i=0 p=(0),1,2,3,4
                0.5,   //                      i=1 p=(5)
                12.0,  //                      i=2 p=(6)
                0.625, //                      i=3 p=(7)
                16.0,  //                      i=4 p=(8)
            ], //                                  p=(9)
            col_indices: vec![
                0, 1, 2, 3, 4, //
                1, //
                2, //
                3, //
                4, //
            ],
            row_pointers: vec![0, 5, 6, 7, 8, 9],
        };
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
        let mut coo = CooMatrix::new(5, 5, 13, sym, one_based).unwrap();
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
        let csc = CscMatrix {
            symmetry: None,
            nrow: 5,
            ncol: 5,
            values: vec![
                9.0, 1.5, 6.0, 0.75, 3.0, // j=0 p=(0),1,2,3,4
                1.5, 0.5, //                 j=1 p=(5),6
                6.0, 12.0, //                j=2 p=(7),8
                0.75, 0.625, //              j=3 p=(9),10
                3.0, 16.0, //                j=4 p=(11),12
            ], //                                p=(13)
            row_indices: vec![
                0, 1, 2, 3, 4, //
                0, 1, //
                0, 2, //
                0, 3, //
                0, 4,
            ],
            col_pointers: vec![0, 5, 7, 9, 11, 13],
        };
        // CSR matrix
        let csr = CsrMatrix {
            symmetry: None,
            nrow: 5,
            ncol: 5,
            values: vec![
                9.0, 1.5, 6.0, 0.75, 3.0, // i=0 p=(0),1,2,3,4
                1.5, 0.5, //                 i=1 p=(5),6
                6.0, 12.0, //                i=2 p=(7),8
                0.75, 0.625, //              i=3 p=(9),10
                3.0, 16.0, //                i=4 p=(11),12
            ], //                                p=(13)
            col_indices: vec![
                0, 1, 2, 3, 4, //
                0, 1, //
                0, 2, //
                0, 3, //
                0, 4,
            ],
            row_pointers: vec![0, 5, 7, 9, 11, 13],
        };
        (coo, csc, csr, 9.0 / 4.0)
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
        let mut coo = CooMatrix::new(1, 7, 4, None, false).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(0, 2, 3.0).unwrap();
        coo.put(0, 4, 5.0).unwrap();
        coo.put(0, 6, 7.0).unwrap();
        let csc = CscMatrix {
            symmetry: None,
            nrow: 1,
            ncol: 7,
            values: vec![
                1.0, // j=0, p=(0)
                //      j=1, p=(1)
                3.0, // j=2, p=(1)
                //      j=3, p=(2)
                5.0, // j=4, p=(2)
                //      j=5, p=(3)
                7.0, // j=6, p=(3)
            ], //            p=(4)
            row_indices: vec![0, 0, 0, 0],
            col_pointers: vec![0, 1, 1, 2, 2, 3, 3, 4],
        };
        let csr = CsrMatrix {
            symmetry: None,
            nrow: 1,
            ncol: 7,
            values: vec![
                1.0, 3.0, 5.0, 7.0, // i=0, p=(0),1,2,3
            ], //                           p=(4)
            col_indices: vec![0, 2, 4, 6],
            row_pointers: vec![0, 4],
        };
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
        let mut coo = CooMatrix::new(7, 1, 3, None, false).unwrap();
        coo.put(1, 0, 2.0).unwrap();
        coo.put(3, 0, 4.0).unwrap();
        coo.put(5, 0, 6.0).unwrap();
        // CSC matrix
        let csc = CscMatrix {
            symmetry: None,
            nrow: 7,
            ncol: 1,
            values: vec![
                2.0, 4.0, 6.0, // j=0, p=(0),1,2
            ], //                      p=(3)
            row_indices: vec![1, 3, 5],
            col_pointers: vec![0, 3],
        };
        // CSR matrix
        let csr = CsrMatrix {
            symmetry: None,
            nrow: 7,
            ncol: 1,
            values: vec![
                //      i=0, p=(0)
                2.0, // i=1, p=(0)
                //      i=2, p=(1)
                4.0, // i=3, p=(1)
                //      i=4, p=(2)
                6.0, // i=5, p=(2)
                     // i=6, p=(3)
            ], //            p=(3)
            col_indices: vec![0, 0, 0],
            row_pointers: vec![0, 0, 1, 1, 2, 2, 3, 3],
        };
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
        let mut coo = CooMatrix::new(3, 4, 9, None, false).unwrap();
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
        let csc = CscMatrix {
            symmetry: None,
            nrow: 3,
            ncol: 4,
            col_pointers: vec![0, 3, 6, 6, 9],
            row_indices: vec![
                0, 1, 2, // j=0, p=(0),1,2
                0, 1, 2, // j=1, p=(3),4,5
                //          j=2, p=(6)
                0, 1, 2, // j=3, p=(6),7,8
            ], //                  (9)
            values: vec![
                5.0, 10.0, 15.0, //  j=0, p=(0),1,2
                -2.0, -4.0, -6.0, // j=1, p=(3),4,5
                //                   j=2, p=(6)
                1.0, 2.0, 3.0, //    j=3, p=(6),7,8
            ], //                           (9)
        };
        // CSR matrix
        let csr = CsrMatrix {
            symmetry: None,
            nrow: 3,
            ncol: 4,
            row_pointers: vec![0, 3, 6, 9],
            col_indices: vec![
                0, 1, 3, // i=0, p=(0),1,2
                0, 1, 3, // i=1, p=(3),4,5
                0, 1, 3, // i=2, p=(6),7,8
            ], //                  (9)
            values: vec![
                5.0, -2.0, 1.0, //  i=0, p=(0),1,2
                10.0, -4.0, 2.0, // i=1, p=(3),4,5
                15.0, -6.0, 3.0, // i=2, p=(6),7,8
            ], //                          (9)
        };
        (coo, csc, csr, PLACEHOLDER)
    }

    /// Returns a (1 x 1) matrix
    ///
    /// Note: the last return value is not the determinant, but a PLACEHOLDER
    ///
    /// ```text
    /// ┌     ┐
    /// │ 123 │
    /// └     ┘
    /// ```
    pub fn tiny_1x1() -> (CooMatrix, CscMatrix, CsrMatrix, f64) {
        let mut coo = CooMatrix::new(1, 1, 1, None, false).unwrap();
        coo.put(0, 0, 123.0).unwrap();
        // CSC matrix
        let csc = CscMatrix {
            symmetry: None,
            nrow: 1,
            ncol: 1,
            col_pointers: vec![0, 1],
            row_indices: vec![0],
            values: vec![123.0],
        };
        // CSR matrix
        let csr = CsrMatrix {
            symmetry: None,
            nrow: 1,
            ncol: 1,
            row_pointers: vec![0, 1],
            col_indices: vec![0],
            values: vec![123.0],
        };
        (coo, csc, csr, PLACEHOLDER)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Samples;
    use russell_chk::approx_eq;
    use russell_lab::{mat_approx_eq, mat_inverse, Matrix};

    #[test]
    fn samples_are_correct() {
        let correct = &[
            [2.0, 3.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 4.0, 0.0, 6.0],
            [0.0, -1.0, -3.0, 2.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 4.0, 2.0, 0.0, 1.0],
        ];
        for (coo, csc, csr, correct_det) in [
            Samples::umfpack_unsymmetric_5x5(false),
            Samples::umfpack_unsymmetric_5x5(true),
        ] {
            let mat = coo.as_matrix();
            mat_approx_eq(&mat, correct, 1e-15);
            csc.validate().unwrap();
            csr.validate().unwrap();
            let mut inv = Matrix::new(5, 5);
            let det = mat_inverse(&mut inv, &mat).unwrap();
            approx_eq(det, correct_det, 1e-15);
        }

        // ----------------------------------------------------------------------------

        let (coo, csc, csr, correct_det) = Samples::mkl_unsymmetric_5x5(false);
        let mat = coo.as_matrix();
        let correct = &[
            [1.0, -1.0, 0.0, -3.0, 0.0],
            [-2.0, 5.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 6.0, 4.0],
            [-4.0, 0.0, 2.0, 7.0, 0.0],
            [0.0, 8.0, 0.0, 0.0, -5.0],
        ];
        mat_approx_eq(&mat, correct, 1e-15);
        csc.validate().unwrap();
        csr.validate().unwrap();
        let mut inv = Matrix::new(5, 5);
        let det = mat_inverse(&mut inv, &mat).unwrap();
        approx_eq(det, correct_det, 1e-15);

        // ----------------------------------------------------------------------------

        let correct = &[
            [1.0, 2.0, 0.0, 0.0, 0.0],
            [3.0, 4.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 6.0, 0.0],
            [0.0, 0.0, 7.0, 8.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 9.0],
        ];
        for (coo, csc, csr, correct_det) in [
            Samples::block_unsymmetric_5x5(false, false, false),
            Samples::block_unsymmetric_5x5(false, true, false),
            Samples::block_unsymmetric_5x5(false, false, true),
            Samples::block_unsymmetric_5x5(false, true, true),
            Samples::block_unsymmetric_5x5(true, false, false),
            Samples::block_unsymmetric_5x5(true, true, false),
            Samples::block_unsymmetric_5x5(true, false, true),
            Samples::block_unsymmetric_5x5(true, true, true),
        ] {
            let mat = coo.as_matrix();
            mat_approx_eq(&mat, correct, 1e-15);
            csc.validate().unwrap();
            csr.validate().unwrap();
            let mut inv = Matrix::new(5, 5);
            let det = mat_inverse(&mut inv, &mat).unwrap();
            approx_eq(det, correct_det, 1e-13);
        }

        // ----------------------------------------------------------------------------

        let correct = &[
            [9.0, 1.5, 6.0, 0.75, 3.0],
            [1.5, 0.5, 0.0, 0.0, 0.0],
            [6.0, 0.0, 12.0, 0.0, 0.0],
            [0.75, 0.0, 0.0, 0.625, 0.0],
            [3.0, 0.0, 0.0, 0.0, 16.0],
        ];
        for (coo, csc, csr, correct_det) in [
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
            let mat = coo.as_matrix();
            mat_approx_eq(&mat, correct, 1e-15);
            csc.validate().unwrap();
            csr.validate().unwrap();
            let mut inv = Matrix::new(5, 5);
            let det = mat_inverse(&mut inv, &mat).unwrap();
            approx_eq(det, correct_det, 1e-14);
        }

        // ----------------------------------------------------------------------------

        let (coo, csc, csr, _) = Samples::rectangular_1x7();
        let mat = coo.as_matrix();
        let correct = &[[1.0, 0.0, 3.0, 0.0, 5.0, 0.0, 7.0]];
        mat_approx_eq(&mat, correct, 1e-15);
        csc.validate().unwrap();
        csr.validate().unwrap();

        // ----------------------------------------------------------------------------

        let (coo, csc, csr, _) = Samples::rectangular_7x1();
        let mat = coo.as_matrix();
        let correct = &[[0.0], [2.0], [0.0], [4.0], [0.0], [6.0], [0.0]];
        mat_approx_eq(&mat, correct, 1e-15);
        csc.validate().unwrap();
        csr.validate().unwrap();

        // ----------------------------------------------------------------------------

        let (coo, csc, csr, _) = Samples::rectangular_3x4();
        let mat = coo.as_matrix();
        let correct = &[
            [5.0, -2.0, 0.0, 1.0],  //
            [10.0, -4.0, 0.0, 2.0], //
            [15.0, -6.0, 0.0, 3.0], //
        ];
        mat_approx_eq(&mat, correct, 1e-15);
        csc.validate().unwrap();
        csr.validate().unwrap();

        // ----------------------------------------------------------------------------

        let (coo, csc, csr, _) = Samples::tiny_1x1();
        let mat = coo.as_matrix();
        let correct = &[[123.0]];
        mat_approx_eq(&mat, correct, 1e-15);
        csc.validate().unwrap();
        csr.validate().unwrap();
    }
}
