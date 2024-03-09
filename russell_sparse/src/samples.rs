use crate::{ComplexCooMatrix, ComplexCscMatrix, ComplexCsrMatrix};
use crate::{CooMatrix, CscMatrix, CsrMatrix, Sym};
use num_complex::Complex64;
use russell_lab::cpx;

const PLACEHOLDER: f64 = f64::MAX;

/// Holds some samples of small sparse matrices
pub struct Samples {}

impl Samples {
    /// Returns a (1 x 1) matrix
    ///
    /// ```text
    /// ┌     ┐
    /// │ 123 │
    /// └     ┘
    /// ```
    pub fn tiny_1x1() -> (CooMatrix, CscMatrix, CsrMatrix, f64) {
        let sym = Sym::No;
        let nrow = 1;
        let ncol = 1;
        let max_nnz = 1;
        let mut coo = CooMatrix::new(nrow, ncol, max_nnz, sym).unwrap();
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

    /// Returns a (1 x 1) matrix (complex version)
    ///
    /// ```text
    /// ┌       ┐
    /// │ 12+3i │
    /// └       ┘
    /// ```
    pub fn complex_tiny_1x1() -> (ComplexCooMatrix, ComplexCscMatrix, ComplexCsrMatrix, Complex64) {
        let sym = Sym::No;
        let nrow = 1;
        let ncol = 1;
        let max_nnz = 1;
        let mut coo = ComplexCooMatrix::new(nrow, ncol, max_nnz, sym).unwrap();
        coo.put(0, 0, cpx!(12.0, 3.0)).unwrap();
        // CSC matrix
        let col_pointers = vec![0, 1];
        let row_indices = vec![0];
        let values = vec![cpx!(12.0, 3.0)];
        let csc = ComplexCscMatrix::new(nrow, ncol, col_pointers, row_indices, values, sym).unwrap();
        // CSR matrix
        let row_pointers = vec![0, 1];
        let col_indices = vec![0];
        let values = vec![cpx!(12.0, 3.0)];
        let csr = ComplexCsrMatrix::new(nrow, ncol, row_pointers, col_indices, values, sym).unwrap();
        (coo, csc, csr, cpx!(12.0, 3.0))
    }

    /// Returns a (3 x 3) positive definite matrix
    ///
    /// ```text
    ///  2  -1              2     sym
    /// -1   2  -1    =>   -1   2
    ///     -1   2             -1   2
    /// ```
    pub fn positive_definite_3x3_lower() -> (CooMatrix, CscMatrix, CsrMatrix, f64) {
        let (nrow, ncol, nnz) = (3, 3, 6);
        let sym = Sym::YesLower;
        let mut coo = CooMatrix::new(nrow, ncol, nnz, sym).unwrap();
        coo.put(1, 0, -0.5).unwrap(); // duplicate
        coo.put(0, 0, 2.0).unwrap();
        coo.put(2, 2, 2.0).unwrap();
        coo.put(1, 0, -0.5).unwrap(); // duplicate
        coo.put(1, 1, 2.0).unwrap();
        coo.put(2, 1, -1.0).unwrap();
        // CSC matrix
        let col_pointers = vec![0, 2, 4, 5];
        let row_indices = vec![
            0, 1, // j=0, p=(0),1
            1, 2, // j=1, p=(2),3
            2, //    j=2, p=(4)
        ]; //               (5)
        let values = vec![
            2.0, -1.0, // j=0, p=(0),1
            2.0, -1.0, // j=1, p=(2),3
            2.0,  //      j=2, p=(4)
        ]; //                    (5)
        let csc = CscMatrix::new(nrow, ncol, col_pointers, row_indices, values, sym).unwrap();
        // CSR matrix
        let row_pointers = vec![0, 1, 3, 5];
        let col_indices = vec![
            0, //    i=0, p=(0)
            0, 1, // i=1, p=(1),2
            1, 2, // i=2, p=(3),4
        ]; //               (5)
        let values = vec![
            2.0, //       i=0, p=(0)
            -1.0, 2.0, // i=1, p=(1),2
            -1.0, 2.0, // i=2, p=(3),4
        ]; //                    (5)
        let csr = CsrMatrix::new(nrow, ncol, row_pointers, col_indices, values, sym).unwrap();
        (coo, csc, csr, 4.0)
    }

    /// Returns a complex symmetric (3 x 3) matrix (lower storage)
    ///
    /// ```text
    ///  2+1i  -1-1i                  2+1i          sym
    /// -1-1i   2+2i  -1+1i     =>   -1-1i   2+2i       
    ///        -1+1i   2-1i                 -1+1i   2-1i
    /// ```
    pub fn complex_symmetric_3x3_lower() -> (ComplexCooMatrix, ComplexCscMatrix, ComplexCsrMatrix, Complex64) {
        let (nrow, ncol, nnz) = (3, 3, 6);
        let sym = Sym::YesLower;
        let mut coo = ComplexCooMatrix::new(nrow, ncol, nnz, sym).unwrap();
        coo.put(1, 0, cpx!(-0.5, -0.5)).unwrap(); // duplicate
        coo.put(0, 0, cpx!(2.0, 1.0)).unwrap();
        coo.put(2, 2, cpx!(2.0, -1.0)).unwrap();
        coo.put(1, 0, cpx!(-0.5, -0.5)).unwrap(); // duplicate
        coo.put(1, 1, cpx!(2.0, 2.0)).unwrap();
        coo.put(2, 1, cpx!(-1.0, 1.0)).unwrap();
        // CSC matrix
        let col_pointers = vec![0, 2, 4, 5];
        let row_indices = vec![
            0, 1, // j=0, p=(0),1
            1, 2, // j=1, p=(2),3
            2, //    j=2, p=(4)
        ]; //               (5)
        #[rustfmt::skip]
        let values = vec![
            cpx!(2.0,  1.0), cpx!(-1.0, -1.0), // j=0, p=(0),1
            cpx!(2.0,  2.0), cpx!(-1.0,  1.0), // j=1, p=(2),3
            cpx!(2.0, -1.0), //                   j=2, p=(4)
        ]; //                                            (5)
        let csc = ComplexCscMatrix::new(nrow, ncol, col_pointers, row_indices, values, sym).unwrap();
        // CSR matrix
        let row_pointers = vec![0, 1, 3, 5];
        let col_indices = vec![
            0, //    i=0, p=(0)
            0, 1, // i=1, p=(1),2
            1, 2, // i=2, p=(3),4
        ]; //               (5)
        #[rustfmt::skip]
        let values = vec![
            cpx!( 2.0,  1.0), //                  i=0, p=(0)
            cpx!(-1.0, -1.0), cpx!(2.0,  2.0), // i=1, p=(1),2
            cpx!(-1.0,  1.0), cpx!(2.0, -1.0), // i=2, p=(3),4
        ]; //                                            (5)
        let csr = ComplexCsrMatrix::new(nrow, ncol, row_pointers, col_indices, values, sym).unwrap();
        (coo, csc, csr, cpx!(6.0, 10.0))
    }

    /// Returns a complex symmetric (3 x 3) matrix (full storage)
    ///
    /// ```text
    ///  2+1i  -1-1i      
    /// -1-1i   2+2i  -1+1i
    ///        -1+1i   2-1i
    /// ```
    pub fn complex_symmetric_3x3_full() -> (ComplexCooMatrix, ComplexCscMatrix, ComplexCsrMatrix, Complex64) {
        let (nrow, ncol, nnz) = (3, 3, 8);
        let sym = Sym::YesFull;
        let mut coo = ComplexCooMatrix::new(nrow, ncol, nnz, sym).unwrap();
        coo.put(1, 0, cpx!(-0.5, -0.5)).unwrap(); // duplicate
        coo.put(0, 0, cpx!(2.0, 1.0)).unwrap();
        coo.put(2, 2, cpx!(2.0, -1.0)).unwrap();
        coo.put(1, 0, cpx!(-0.5, -0.5)).unwrap(); // duplicate
        coo.put(1, 1, cpx!(2.0, 2.0)).unwrap();
        coo.put(2, 1, cpx!(-1.0, 1.0)).unwrap();
        coo.put(0, 1, cpx!(-1.0, -1.0)).unwrap();
        coo.put(1, 2, cpx!(-1.0, 1.0)).unwrap();
        // CSC matrix
        let col_pointers = vec![0, 2, 5, 7];
        let row_indices = vec![
            0, 1, //    j=0, p=(0),1
            0, 1, 2, // j=1, p=(2),3,4
            1, 2, //    j=2, p=(5),6
        ]; //                  (7)
        #[rustfmt::skip]
        let values = vec![
            cpx!( 2.0,  1.0), cpx!(-1.0, -1.0), //                  j=0, p=(0),1
            cpx!(-1.0, -1.0), cpx!( 2.0,  2.0), cpx!(-1.0, 1.0), // j=1, p=(2),3,4
            cpx!(-1.0,  1.0), cpx!( 2.0, -1.0), //                  j=2, p=(5),6
        ]; //                                                              (7)
        let csc = ComplexCscMatrix::new(nrow, ncol, col_pointers, row_indices, values, sym).unwrap();
        // CSR matrix
        let row_pointers = vec![0, 2, 5, 7];
        let col_indices = vec![
            0, 1, //    i=0, p=(0),1
            0, 1, 2, // i=1, p=(2),3,4
            1, 2, //    i=2, p=(5),6
        ]; //                  (7)
        #[rustfmt::skip]
        let values = vec![
            cpx!( 2.0,  1.0), cpx!(-1.0, -1.0), //                  i=0, p=(0),1
            cpx!(-1.0, -1.0), cpx!( 2.0,  2.0), cpx!(-1.0, 1.0), // i=1, p=(2),3,4
            cpx!(-1.0,  1.0), cpx!( 2.0, -1.0), //                  i=2, p=(5),6
        ]; //                                                              (7)
        let csr = ComplexCsrMatrix::new(nrow, ncol, row_pointers, col_indices, values, sym).unwrap();
        (coo, csc, csr, cpx!(6.0, 10.0))
    }

    /// Returns a lower symmetric 5 x 5 matrix
    ///
    /// ```text
    /// 2  1  1  3  2        2
    /// 1  2  2  1  1        1  2     sym
    /// 1  2  9  1  5   =>   1  2  9
    /// 3  1  1  7  1        3  1  1  7
    /// 2  1  5  1  8        2  1  5  1  8
    /// ```
    pub fn lower_symmetric_5x5() -> (CooMatrix, CscMatrix, CsrMatrix, f64) {
        let (nrow, ncol, nnz) = (5, 5, 18);
        let sym = Sym::YesLower;
        let mut coo = CooMatrix::new(nrow, ncol, nnz, sym).unwrap();
        coo.put(1, 1, 2.0).unwrap();
        coo.put(4, 2, 2.5).unwrap(); // duplicate
        coo.put(2, 2, 9.0).unwrap();
        coo.put(3, 3, 7.0).unwrap();
        coo.put(0, 0, 2.0).unwrap();
        coo.put(4, 4, 5.0).unwrap(); // duplicate
        coo.put(2, 0, 1.0).unwrap();
        coo.put(4, 4, 3.0).unwrap(); // duplicate
        coo.put(2, 1, 2.0).unwrap();
        coo.put(1, 0, 1.0).unwrap();
        coo.put(3, 0, 3.0).unwrap();
        coo.put(3, 2, 1.0).unwrap();
        coo.put(4, 0, 2.0).unwrap();
        coo.put(3, 1, 0.5).unwrap(); // duplicate
        coo.put(3, 1, 0.5).unwrap(); // duplicate
        coo.put(4, 1, 1.0).unwrap();
        coo.put(4, 2, 2.5).unwrap(); // duplicate
        coo.put(4, 3, 1.0).unwrap();
        // CSC matrix
        let col_pointers = vec![0, 5, 9, 12, 14, 15];
        let row_indices = vec![
            0, 1, 2, 3, 4, // j=0, p=(0),1,2,3,4
            1, 2, 3, 4, //    j=1, p=(5),6,7,8
            2, 3, 4, //       j=2, p=(9),10,11
            3, 4, //          j=3, p=(12),13
            4, //             j=4, p=(14)
        ]; //                        (15)
        let values = vec![
            2.0, 1.0, 1.0, 3.0, 2.0, // j=0, p=(0),1,2,3,4
            2.0, 2.0, 1.0, 1.0, //      j=1, p=(5),6,7,8
            9.0, 1.0, 5.0, //           j=2, p=(9),10,11
            7.0, 1.0, //                j=3, p=(12),13
            8.0, //                     j=4, p=(14)
        ]; //                                  (15)
        let csc = CscMatrix::new(nrow, ncol, col_pointers, row_indices, values, sym).unwrap();
        // CSR matrix
        let row_pointers = vec![0, 1, 3, 6, 10, 15];
        let col_indices = vec![
            0, //              i=0, p=(0)
            0, 1, //           i=1, p=(1),2
            0, 1, 2, //        i=2, p=(3),4,5
            0, 1, 2, 3, //     i=3, p=(6),7,8,9
            0, 1, 2, 3, 4, //  i=4, p=(10),11,12,13,14
        ]; //                         (15)
        let values = vec![
            2.0, //                      i=0, p=(0)
            1.0, 2.0, //                 i=1, p=(1),2
            1.0, 2.0, 9.0, //            i=2, p=(3),4,5
            3.0, 1.0, 1.0, 7.0, //       i=3, p=(6),7,8,9
            2.0, 1.0, 5.0, 1.0, 8.0, //  i=4, p=(10),11,12,13,14
        ]; //                                   (15)
        let csr = CsrMatrix::new(nrow, ncol, row_pointers, col_indices, values, sym).unwrap();
        (coo, csc, csr, 98.0)
    }

    /// Returns the COO, CSC, and CSR versions of the matrix and its determinant
    ///
    /// ```text
    ///  1  .  2
    ///  .  0  3
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
        shuffle_coo_entries: bool,
        duplicate_coo_entries: bool,
    ) -> (CooMatrix, CscMatrix, CsrMatrix, f64) {
        let sym = Sym::No;
        let nrow = 3;
        let ncol = 3;
        let max_nnz = 10; // more nnz than needed => OK
        let mut coo = CooMatrix::new(nrow, ncol, max_nnz, sym).unwrap();
        if shuffle_coo_entries {
            if duplicate_coo_entries {
                coo.put(0, 2, 2.0).unwrap();
                coo.put(2, 1, 5.0).unwrap();
                coo.put(1, 2, 3.0).unwrap();
                coo.put(2, 2, 2.0).unwrap(); // << duplicate
                coo.put(2, 0, 4.0).unwrap();
                coo.put(0, 0, 1.0).unwrap();
                coo.put(2, 2, 2.0).unwrap(); // << duplicate
                coo.put(1, 1, 0.0).unwrap(); // << notice that 0.0 may be specified
                coo.put(2, 2, 2.0).unwrap(); // << duplicate
            } else {
                coo.put(2, 0, 4.0).unwrap();
                coo.put(0, 0, 1.0).unwrap();
                coo.put(2, 2, 6.0).unwrap();
                coo.put(0, 2, 2.0).unwrap();
                coo.put(1, 1, 0.0).unwrap(); // << notice that 0.0 may be specified
                coo.put(2, 1, 5.0).unwrap();
                coo.put(1, 2, 3.0).unwrap();
            }
        } else {
            if duplicate_coo_entries {
                coo.put(0, 0, 1.0).unwrap();
                coo.put(0, 2, 2.0).unwrap();
                coo.put(1, 1, 0.0).unwrap(); // << notice that 0.0 may be specified
                coo.put(1, 2, 3.0).unwrap();
                coo.put(2, 0, 4.0).unwrap();
                coo.put(2, 1, 5.0).unwrap();
                coo.put(2, 2, 2.0).unwrap(); // << duplicate
                coo.put(2, 2, 2.0).unwrap(); // << duplicate
                coo.put(2, 2, 2.0).unwrap(); // << duplicate
            } else {
                coo.put(0, 0, 1.0).unwrap();
                coo.put(0, 2, 2.0).unwrap();
                coo.put(1, 1, 0.0).unwrap(); // << notice that 0.0 may be specified
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
        ]; //                     p=(7)
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
    pub fn umfpack_unsymmetric_5x5() -> (CooMatrix, CscMatrix, CsrMatrix, f64) {
        let sym = Sym::No;
        let nrow = 5;
        let ncol = 5;
        let max_nnz = 13;
        let mut coo = CooMatrix::new(nrow, ncol, max_nnz, sym).unwrap();
        coo.put(0, 0, 1.0).unwrap(); // << (0, 0, a00/2) duplicate
        coo.put(2, 1, -1.0).unwrap();
        coo.put(1, 0, 3.0).unwrap();
        coo.put(4, 1, 4.0).unwrap();
        coo.put(4, 4, 1.0).unwrap();
        coo.put(0, 1, 3.0).unwrap();
        coo.put(3, 2, 1.0).unwrap();
        coo.put(2, 2, -3.0).unwrap();
        coo.put(0, 0, 1.0).unwrap(); // << (0, 0, a00/2) duplicate
        coo.put(4, 2, 2.0).unwrap();
        coo.put(2, 3, 2.0).unwrap();
        coo.put(1, 4, 6.0).unwrap();
        coo.put(1, 2, 4.0).unwrap();
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
        ]; //                        p=(12)
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
    pub fn mkl_unsymmetric_5x5() -> (CooMatrix, CscMatrix, CsrMatrix, f64) {
        let sym = Sym::No;
        let nrow = 5;
        let ncol = 5;
        let mut coo = CooMatrix::new(nrow, ncol, 13, sym).unwrap();
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
        shuffle_coo_entries: bool,
        duplicate_coo_entries: bool,
    ) -> (CooMatrix, CscMatrix, CsrMatrix, f64) {
        let sym = Sym::No;
        let nrow = 5;
        let ncol = 5;
        let max_nnz = 11; // more nnz than needed => OK
        let mut coo = CooMatrix::new(nrow, ncol, max_nnz, sym).unwrap();
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
    pub fn mkl_positive_definite_5x5_lower() -> (CooMatrix, CscMatrix, CsrMatrix, f64) {
        let sym = Sym::YesLower;
        let nrow = 5;
        let ncol = 5;
        let mut coo = CooMatrix::new(nrow, ncol, 9, sym).unwrap();
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
    pub fn mkl_positive_definite_5x5_upper() -> (CooMatrix, CscMatrix, CsrMatrix, f64) {
        let sym = Sym::YesUpper;
        let nrow = 5;
        let ncol = 5;
        let mut coo = CooMatrix::new(nrow, ncol, 9, sym).unwrap();
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
        shuffle_coo_entries: bool,
        duplicate_coo_entries: bool,
    ) -> (CooMatrix, CscMatrix, CsrMatrix, f64) {
        let sym = Sym::YesLower;
        let nrow = 5;
        let ncol = 5;
        let max_nnz = 13;
        let mut coo = CooMatrix::new(nrow, ncol, max_nnz, sym).unwrap();
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
        shuffle_coo_entries: bool,
        duplicate_coo_entries: bool,
    ) -> (CooMatrix, CscMatrix, CsrMatrix, f64) {
        let sym = Sym::YesUpper;
        let nrow = 5;
        let ncol = 5;
        let max_nnz = 15;
        let mut coo = CooMatrix::new(nrow, ncol, max_nnz, sym).unwrap();
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
    pub fn mkl_symmetric_5x5_full() -> (CooMatrix, CscMatrix, CsrMatrix, f64) {
        let sym = Sym::YesFull;
        let nrow = 5;
        let ncol = 5;
        let max_nnz = 13;
        let mut coo = CooMatrix::new(nrow, ncol, max_nnz, sym).unwrap();
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
        shuffle_coo_entries: bool,
        duplicate_coo_entries: bool,
    ) -> (CooMatrix, CscMatrix, CsrMatrix, f64) {
        let sym = Sym::No;
        let nrow = 1;
        let ncol = 2;
        let max_nnz = 10;
        let mut coo = CooMatrix::new(nrow, ncol, max_nnz, sym).unwrap();
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
        let sym = Sym::No;
        let nrow = 1;
        let ncol = 7;
        let mut coo = CooMatrix::new(nrow, ncol, 4, sym).unwrap();
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
        let sym = Sym::No;
        let nrow = 7;
        let ncol = 1;
        let mut coo = CooMatrix::new(nrow, ncol, 3, sym).unwrap();
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
    ///   5  -2  .  1
    ///  10  -4  .  2
    ///  15  -6  .  3
    /// ```
    pub fn rectangular_3x4() -> (CooMatrix, CscMatrix, CsrMatrix, f64) {
        let sym = Sym::No;
        let nrow = 3;
        let ncol = 4;
        let mut coo = CooMatrix::new(nrow, ncol, 9, sym).unwrap();
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

    /// Returns a (4 x 3) complex rectangular matrix
    ///
    /// Note: the last return value is not the determinant, but a PLACEHOLDER
    ///
    /// ```text
    /// 4+4i    .     2+2i
    ///  .      1     3+3i
    ///  .     5+5i   1+1i
    ///  1      .      .  
    /// ```
    pub fn complex_rectangular_4x3() -> (ComplexCooMatrix, ComplexCscMatrix, ComplexCsrMatrix, f64) {
        let sym = Sym::No;
        let nrow = 4;
        let ncol = 3;
        let mut coo = ComplexCooMatrix::new(nrow, ncol, 7, sym).unwrap();
        coo.put(0, 0, cpx!(4.0, 4.0)).unwrap();
        coo.put(0, 2, cpx!(2.0, 2.0)).unwrap();
        coo.put(1, 1, cpx!(1.0, 0.0)).unwrap();
        coo.put(1, 2, cpx!(3.0, 3.0)).unwrap();
        coo.put(2, 1, cpx!(5.0, 5.0)).unwrap();
        coo.put(2, 2, cpx!(1.0, 1.0)).unwrap();
        coo.put(3, 0, cpx!(1.0, 0.0)).unwrap();
        // CSC matrix
        let col_pointers = vec![0, 2, 4, 7];
        let row_indices = vec![
            0, 3, //    j=0, p=(0),1
            1, 2, //    j=1, p=(2),3
            0, 1, 2, // j=2, p=(4),5,6
        ]; //                  (7)
        #[rustfmt::skip]
        let values = vec![
            cpx!(4.0,4.0), cpx!(1.0,0.0),                // j=0, p=(0),1
            cpx!(1.0,0.0), cpx!(5.0,5.0),                // j=1, p=(2),3
            cpx!(2.0,2.0), cpx!(3.0,3.0), cpx!(1.0,1.0), // j=2, p=(4),5,6
        ]; //                                                      (7)
        let csc = ComplexCscMatrix::new(nrow, ncol, col_pointers, row_indices, values, sym).unwrap();
        // CSR matrix
        let row_pointers = vec![0, 2, 4, 6, 7];
        let col_indices = vec![
            0, 2, // i=0, p=(0),1
            1, 2, // i=1, p=(2),3
            1, 2, // i=2, p=(4),5
            0, //    i=3, p=(6)
        ]; //               (7)
        #[rustfmt::skip]
        let values = vec![
            cpx!(4.0,4.0), cpx!(2.0,2.0), // i=0, p=(0),1
            cpx!(1.0,0.0), cpx!(3.0,3.0), // i=1, p=(2),3
            cpx!(5.0,5.0), cpx!(1.0,1.0), // i=2, p=(4),5
            cpx!(1.0,0.0),                // i=3, p=(6)
        ]; //                                       (7)
        let csr = ComplexCsrMatrix::new(nrow, ncol, row_pointers, col_indices, values, sym).unwrap();
        (coo, csc, csr, PLACEHOLDER)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Samples;
    use crate::{NumCooMatrix, NumCscMatrix, NumCsrMatrix};
    use num_complex::Complex64;
    use num_traits::{Num, NumCast};
    use russell_lab::{approx_eq, mat_approx_eq, mat_inverse, Matrix};
    use russell_lab::{complex_approx_eq, complex_mat_approx_eq, complex_mat_inverse, cpx, ComplexMatrix};
    use serde::de::DeserializeOwned;
    use serde::Serialize;
    use std::ops::{AddAssign, MulAssign};

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
    fn check<T>(coo: &NumCooMatrix<T>, csc: &NumCscMatrix<T>, csr: &NumCsrMatrix<T>)
    where
        T: AddAssign + MulAssign + Num + NumCast + Copy + DeserializeOwned + Serialize,
    {
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
        assert_eq!(csc.symmetric, coo.symmetric);
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
        assert_eq!(csr.symmetric, coo.symmetric);
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
        let (coo, csc, csr, det) = Samples::tiny_1x1();
        approx_eq(det, correct_det, 1e-15);
        mat_approx_eq(&coo.as_dense(), correct, 1e-15);
        mat_approx_eq(&csc.as_dense(), correct, 1e-15);
        mat_approx_eq(&csr.as_dense(), correct, 1e-15);
        check(&coo, &csc, &csr);

        // ----------------------------------------------------------------------------

        let correct = &[
            [cpx!(12.0, 3.0)], //
        ];
        let a = ComplexMatrix::from(correct);
        let mut ai = ComplexMatrix::new(1, 1);
        let correct_det = complex_mat_inverse(&mut ai, &a).unwrap();
        let (coo, csc, csr, det) = Samples::complex_tiny_1x1();
        complex_approx_eq(det, correct_det, 1e-15);
        complex_mat_approx_eq(&coo.as_dense(), correct, 1e-15);
        complex_mat_approx_eq(&csc.as_dense(), correct, 1e-15);
        complex_mat_approx_eq(&csr.as_dense(), correct, 1e-15);
        check(&coo, &csc, &csr);

        // ----------------------------------------------------------------------------

        let correct = &[
            [2.0, -1.0, 0.0],  //
            [-1.0, 2.0, -1.0], //
            [0.0, -1.0, 2.0],  //
        ];
        let a = Matrix::from(correct);
        let mut ai = Matrix::new(3, 3);
        let correct_det = mat_inverse(&mut ai, &a).unwrap();
        let (coo, csc, csr, det) = Samples::positive_definite_3x3_lower();
        approx_eq(det, correct_det, 1e-15);
        mat_approx_eq(&coo.as_dense(), correct, 1e-15);
        mat_approx_eq(&csc.as_dense(), correct, 1e-15);
        mat_approx_eq(&csr.as_dense(), correct, 1e-15);
        check(&coo, &csc, &csr);

        // ----------------------------------------------------------------------------

        #[rustfmt::skip]
        let correct = &[
            [cpx!( 2.0,  1.0), cpx!(-1.0, -1.0), cpx!( 0.0,  0.0)],
            [cpx!(-1.0, -1.0), cpx!( 2.0,  2.0), cpx!(-1.0,  1.0)],
            [cpx!( 0.0,  0.0), cpx!(-1.0,  1.0), cpx!( 2.0, -1.0)],
        ];
        let a = ComplexMatrix::from(correct);
        let mut ai = ComplexMatrix::new(3, 3);
        let correct_det = complex_mat_inverse(&mut ai, &a).unwrap();
        // lower
        let (coo, csc, csr, det) = Samples::complex_symmetric_3x3_lower();
        complex_approx_eq(det, correct_det, 1e-15);
        complex_mat_approx_eq(&coo.as_dense(), correct, 1e-15);
        complex_mat_approx_eq(&csc.as_dense(), correct, 1e-15);
        complex_mat_approx_eq(&csr.as_dense(), correct, 1e-15);
        check(&coo, &csc, &csr);
        // full
        let (coo, csc, csr, det) = Samples::complex_symmetric_3x3_full();
        complex_approx_eq(det, correct_det, 1e-15);
        complex_mat_approx_eq(&coo.as_dense(), correct, 1e-15);
        complex_mat_approx_eq(&csc.as_dense(), correct, 1e-15);
        complex_mat_approx_eq(&csr.as_dense(), correct, 1e-15);
        check(&coo, &csc, &csr);

        // ----------------------------------------------------------------------------

        let correct = &[
            [2.0, 1.0, 1.0, 3.0, 2.0],
            [1.0, 2.0, 2.0, 1.0, 1.0],
            [1.0, 2.0, 9.0, 1.0, 5.0],
            [3.0, 1.0, 1.0, 7.0, 1.0],
            [2.0, 1.0, 5.0, 1.0, 8.0],
        ];
        let a = Matrix::from(correct);
        let mut ai = Matrix::new(5, 5);
        let correct_det = mat_inverse(&mut ai, &a).unwrap();
        let (coo, csc, csr, det) = Samples::lower_symmetric_5x5();
        approx_eq(det, correct_det, 1e-12);
        mat_approx_eq(&coo.as_dense(), correct, 1e-15);
        mat_approx_eq(&csc.as_dense(), correct, 1e-15);
        mat_approx_eq(&csr.as_dense(), correct, 1e-15);
        check(&coo, &csc, &csr);

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
            Samples::unsymmetric_3x3(false, false),
            Samples::unsymmetric_3x3(false, true),
            Samples::unsymmetric_3x3(true, false),
            Samples::unsymmetric_3x3(true, true),
        ] {
            approx_eq(det, correct_det, 1e-13);
            mat_approx_eq(&coo.as_dense(), correct, 1e-15);
            mat_approx_eq(&csc.as_dense(), correct, 1e-15);
            mat_approx_eq(&csr.as_dense(), correct, 1e-15);
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
        let (coo, csc, csr, det) = Samples::umfpack_unsymmetric_5x5();
        approx_eq(det, correct_det, 1e-13);
        mat_approx_eq(&coo.as_dense(), correct, 1e-15);
        mat_approx_eq(&csc.as_dense(), correct, 1e-15);
        mat_approx_eq(&csr.as_dense(), correct, 1e-15);
        check(&coo, &csc, &csr);

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
        let (coo, csc, csr, det) = Samples::mkl_unsymmetric_5x5();
        approx_eq(det, correct_det, 1e-13);
        mat_approx_eq(&coo.as_dense(), correct, 1e-15);
        mat_approx_eq(&csc.as_dense(), correct, 1e-15);
        mat_approx_eq(&csr.as_dense(), correct, 1e-15);
        check(&coo, &csc, &csr);

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
            Samples::block_unsymmetric_5x5(false, false),
            Samples::block_unsymmetric_5x5(false, true),
            Samples::block_unsymmetric_5x5(true, false),
            Samples::block_unsymmetric_5x5(true, true),
        ] {
            approx_eq(det, correct_det, 1e-13);
            mat_approx_eq(&coo.as_dense(), correct, 1e-15);
            mat_approx_eq(&csc.as_dense(), correct, 1e-15);
            mat_approx_eq(&csr.as_dense(), correct, 1e-15);
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
            Samples::mkl_positive_definite_5x5_lower(),
            Samples::mkl_positive_definite_5x5_upper(),
            Samples::mkl_symmetric_5x5_lower(false, false),
            Samples::mkl_symmetric_5x5_lower(false, true),
            Samples::mkl_symmetric_5x5_lower(true, false),
            Samples::mkl_symmetric_5x5_lower(true, true),
            Samples::mkl_symmetric_5x5_upper(false, false),
            Samples::mkl_symmetric_5x5_upper(false, true),
            Samples::mkl_symmetric_5x5_upper(true, false),
            Samples::mkl_symmetric_5x5_upper(true, true),
            Samples::mkl_symmetric_5x5_full(),
        ] {
            approx_eq(det, correct_det, 1e-13);
            mat_approx_eq(&coo.as_dense(), correct, 1e-15);
            mat_approx_eq(&csc.as_dense(), correct, 1e-15);
            mat_approx_eq(&csr.as_dense(), correct, 1e-15);
            check(&coo, &csc, &csr);
        }

        // ----------------------------------------------------------------------------

        let correct = &[[10.0, 20.0]];
        for (coo, csc, csr, _) in [
            Samples::rectangular_1x2(false, false),
            Samples::rectangular_1x2(false, true),
            Samples::rectangular_1x2(true, false),
            Samples::rectangular_1x2(true, true),
        ] {
            mat_approx_eq(&coo.as_dense(), correct, 1e-15);
            mat_approx_eq(&csc.as_dense(), correct, 1e-15);
            mat_approx_eq(&csr.as_dense(), correct, 1e-15);
            check(&coo, &csc, &csr);
        }

        // ----------------------------------------------------------------------------

        let correct = &[[1.0, 0.0, 3.0, 0.0, 5.0, 0.0, 7.0]];
        let (coo, csc, csr, _) = Samples::rectangular_1x7();
        mat_approx_eq(&coo.as_dense(), correct, 1e-15);
        mat_approx_eq(&csc.as_dense(), correct, 1e-15);
        mat_approx_eq(&csr.as_dense(), correct, 1e-15);
        check(&coo, &csc, &csr);

        // ----------------------------------------------------------------------------

        let correct = &[[0.0], [2.0], [0.0], [4.0], [0.0], [6.0], [0.0]];
        let (coo, csc, csr, _) = Samples::rectangular_7x1();
        mat_approx_eq(&coo.as_dense(), correct, 1e-15);
        mat_approx_eq(&csc.as_dense(), correct, 1e-15);
        mat_approx_eq(&csr.as_dense(), correct, 1e-15);
        check(&coo, &csc, &csr);

        // ----------------------------------------------------------------------------

        let correct = &[
            [5.0, -2.0, 0.0, 1.0],  //
            [10.0, -4.0, 0.0, 2.0], //
            [15.0, -6.0, 0.0, 3.0], //
        ];
        let (coo, csc, csr, _) = Samples::rectangular_3x4();
        mat_approx_eq(&coo.as_dense(), correct, 1e-15);
        mat_approx_eq(&csc.as_dense(), correct, 1e-15);
        mat_approx_eq(&csr.as_dense(), correct, 1e-15);
        check(&coo, &csc, &csr);

        // ----------------------------------------------------------------------------

        let correct = &[
            [cpx!(4.0, 4.0), cpx!(0.0, 0.0), cpx!(2.0, 2.0)], //
            [cpx!(0.0, 0.0), cpx!(1.0, 0.0), cpx!(3.0, 3.0)], //
            [cpx!(0.0, 0.0), cpx!(5.0, 5.0), cpx!(1.0, 1.0)], //
            [cpx!(1.0, 0.0), cpx!(0.0, 0.0), cpx!(0.0, 0.0)], //
        ];
        let (coo, csc, csr, _) = Samples::complex_rectangular_4x3();
        complex_mat_approx_eq(&coo.as_dense(), correct, 1e-15);
        complex_mat_approx_eq(&csc.as_dense(), correct, 1e-15);
        complex_mat_approx_eq(&csr.as_dense(), correct, 1e-15);
        check(&coo, &csc, &csr);
    }
}
