use super::{handle_umfpack_error_code, to_i32, CooMatrix, Symmetry};
use crate::StrError;
use russell_lab::Matrix;

extern "C" {
    fn umfpack_coo_to_csc(
        col_pointers: *mut i32,
        row_indices: *mut i32,
        values: *mut f64,
        nrow: i32,
        ncol: i32,
        nnz: i32,
        indices_i: *const i32,
        indices_j: *const i32,
        values_aij: *const f64,
    ) -> i32;
}

/// Holds the arrays needed for a CSC (compressed sparse column) matrix
///
/// # Example (from UMFPACK QuickStart.pdf)
///
/// The sparse matrix is (dots indicate zero values);
///
/// ```text
///  2   3   .   .   .
///  3   .   4   .   6
///  .  -1  -3   2   .
///  .   .   1   .   .
///  .   4   2   .   1
/// ```
///
/// The values in compressed column order are (note the column indices `j` and pointers `p`):
///
/// ```text
///                                          p
///  2.0,  3.0,              j = 0, count =  0, 1,
///  3.0, -1.0,  4.0,        j = 1, count =  2, 3, 4,
///  4.0, -3.0,  1.0,  2.0,  j = 2, count =  5, 6, 7, 8,
///  2.0,                    j = 3, count =  9,
///  6.0,  1.0,              j = 4, count = 10, 11,
///                                         12
/// ```
///
/// The row indices are:
///
/// ```text
/// 0, 1,
/// 0, 2, 4,
/// 1, 2, 3, 4,
/// 2,
/// 1, 4,
/// ```
///
/// And the column pointers are (see the column indicated by `p` above):
///
/// ```text
/// 0, 2, 5, 9, 10, 12
/// ```
pub struct CscMatrix {
    /// Defines the symmetry and storage: lower-triangular, upper-triangular, full-matrix
    ///
    /// **Note:** `None` means unsymmetric matrix or unspecified symmetry,
    /// where the storage is automatically `Full`.
    pub symmetry: Option<Symmetry>,

    /// Holds the number of rows (must fit i32)
    pub nrow: usize,

    /// Holds the number of columns (must fit i32)
    pub ncol: usize,

    /// Defines the column pointers array with size = ncol + 1
    pub col_pointers: Vec<i32>,

    /// Defines the row indices array with size = nnz (number of non-zeros)
    pub row_indices: Vec<i32>,

    /// Defines the values array with size = nnz (number of non-zeros)
    pub values: Vec<f64>,
}

impl CscMatrix {
    /// Allocates an empty CscMatrix
    ///
    /// This function simply allocates the following arrays:
    ///
    /// * col_pointers: `vec![0; ncol + 1]`
    /// * row_indices: `vec![0; nnz]`
    /// * values: `vec![0.0; nnz]`
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_sparse::prelude::*;
    /// use russell_sparse::StrError;
    ///
    /// fn main() {
    ///     let csc = CscMatrix::new(None, 3, 3, 4);
    ///     assert_eq!(csc.symmetry, None);
    ///     assert_eq!(csc.nrow, 3);
    ///     assert_eq!(csc.ncol, 3);
    ///     assert_eq!(csc.col_pointers, &[0, 0, 0, 0]);
    ///     assert_eq!(csc.row_indices, &[0, 0, 0, 0]);
    ///     assert_eq!(csc.values, &[0.0, 0.0, 0.0, 0.0]);
    /// }
    /// ```
    pub fn new(symmetry: Option<Symmetry>, nrow: usize, ncol: usize, nnz: usize) -> Self {
        CscMatrix {
            symmetry,
            nrow,
            ncol,
            col_pointers: vec![0; ncol + 1],
            row_indices: vec![0; nnz],
            values: vec![0.0; nnz],
        }
    }

    /// Creates a new CscMatrix from a CooMatrix
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_sparse::prelude::*;
    /// use russell_sparse::StrError;
    ///
    /// fn main() -> Result<(), StrError> {
    ///     // allocate a square matrix and store as COO matrix
    ///     // ┌          ┐
    ///     // │  1  0  2 │
    ///     // │  0  0  3 │ << the diagonal 0 entry is optional,
    ///     // │  4  5  6 │    but should be saved for Intel DSS
    ///     // └          ┘
    ///     let (nrow, ncol, nnz) = (3, 3, 6);
    ///     let mut coo = CooMatrix::new(None, nrow, ncol, nnz)?;
    ///     coo.put(0, 0, 1.0)?;
    ///     coo.put(0, 2, 2.0)?;
    ///     coo.put(1, 2, 3.0)?;
    ///     coo.put(2, 0, 4.0)?;
    ///     coo.put(2, 1, 5.0)?;
    ///     coo.put(2, 2, 6.0)?;
    ///
    ///     // convert to CCR matrix
    ///     let csc = CscMatrix::from(&coo)?;
    ///     let correct_v = &[
    ///         //                               p
    ///         1.0, 4.0, //      j = 0, count = 0, 1
    ///         5.0, //           j = 1, count = 2
    ///         2.0, 3.0, 6.0, // j = 2, count = 3, 4, 5
    ///              //                  count = 6
    ///     ];
    ///     let correct_i = &[
    ///         //                         p
    ///         0, 2, //    j = 0, count = 0, 1
    ///         2, //       j = 1, count = 2
    ///         0, 1, 2, // j = 2, count = 3, 4, 5
    ///            //              count = 6
    ///     ];
    ///     let correct_p = &[0, 2, 3, 6];
    ///
    ///     // check
    ///     assert_eq!(&csc.col_pointers, correct_p);
    ///     assert_eq!(&csc.row_indices, correct_i);
    ///     assert_eq!(&csc.values, correct_v);
    ///     Ok(())
    /// }
    /// ```
    pub fn from(coo: &CooMatrix) -> Result<Self, StrError> {
        // check
        let nrow = coo.nrow;
        let ncol = coo.ncol;
        let nnz = coo.pos;
        if nrow < 1 || ncol < 1 || nnz < 1 {
            return Err("COO matrix must be at least (1 x 1) with 1 non-zero value");
        }

        // allocate arrays
        let mut csc = CscMatrix {
            symmetry: coo.symmetry,
            nrow: coo.nrow,
            ncol: coo.ncol,
            col_pointers: vec![0; ncol + 1],
            row_indices: vec![0; nnz],
            values: vec![0.0; nnz],
        };

        // call UMFPACK to convert COO to CSC
        unsafe {
            let status = umfpack_coo_to_csc(
                csc.col_pointers.as_mut_ptr(),
                csc.row_indices.as_mut_ptr(),
                csc.values.as_mut_ptr(),
                to_i32(coo.nrow)?,
                to_i32(coo.ncol)?,
                to_i32(coo.pos)?,
                coo.indices_i.as_ptr(),
                coo.indices_j.as_ptr(),
                coo.values_aij.as_ptr(),
            );
            if status != 0 {
                return Err(handle_umfpack_error_code(status));
            }
        }

        // reduce array sizes if duplicates have been eliminated
        let final_nnz = csc.col_pointers[csc.ncol] as usize;
        if final_nnz < coo.pos {
            csc.row_indices.resize(final_nnz, 0);
            csc.values.resize(final_nnz, 0.0);
        }

        // results
        Ok(csc)
    }

    /// Converts this CscMatrix to a dense matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_sparse::prelude::*;
    /// use russell_sparse::StrError;
    ///
    /// fn main() -> Result<(), StrError> {
    ///     // allocate a square matrix and store as CSC matrix
    ///     // ┌                ┐
    ///     // │  2  3  0  0  0 │
    ///     // │  3  0  4  0  6 │
    ///     // │  0 -1 -3  2  0 │
    ///     // │  0  0  1  0  0 │
    ///     // │  0  4  2  0  1 │
    ///     // └                ┘
    ///     let csc = CscMatrix {
    ///         symmetry: None,
    ///         nrow: 5,
    ///         ncol: 5,
    ///         col_pointers: vec![0, 2, 5, 9, 10, 12],
    ///         row_indices: vec![
    ///             //                             p
    ///             0, 1, //       j = 0, count =  0, 1,
    ///             0, 2, 4, //    j = 1, count =  2, 3, 4,
    ///             1, 2, 3, 4, // j = 2, count =  5, 6, 7, 8,
    ///             2, //          j = 3, count =  9,
    ///             1, 4, //       j = 4, count = 10, 11,
    ///                //                         12
    ///         ],
    ///         values: vec![
    ///             //                                      p
    ///             2.0, 3.0, //            j = 0, count =  0, 1,
    ///             3.0, -1.0, 4.0, //      j = 1, count =  2, 3, 4,
    ///             4.0, -3.0, 1.0, 2.0, // j = 2, count =  5, 6, 7, 8,
    ///             2.0, //                 j = 3, count =  9,
    ///             6.0, 1.0, //            j = 4, count = 10, 11,
    ///                  //                                12
    ///         ],
    ///     };
    ///
    ///     // covert to dense
    ///     let a = csc.as_matrix()?;
    ///     let correct = "┌                ┐\n\
    ///                    │  2  3  0  0  0 │\n\
    ///                    │  3  0  4  0  6 │\n\
    ///                    │  0 -1 -3  2  0 │\n\
    ///                    │  0  0  1  0  0 │\n\
    ///                    │  0  4  2  0  1 │\n\
    ///                    └                ┘";
    ///     assert_eq!(format!("{}", a), correct);
    ///     Ok(())
    /// }
    /// ```
    pub fn as_matrix(&self) -> Result<Matrix, StrError> {
        let mut a = Matrix::new(self.nrow, self.ncol);
        self.to_matrix(&mut a).unwrap();
        Ok(a)
    }

    /// Converts this CscMatrix to a dense matrix
    ///
    /// # Input
    ///
    /// * `a` -- where to store the dense matrix; must be (nrow, ncol)
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::Matrix;
    /// use russell_sparse::prelude::*;
    /// use russell_sparse::StrError;
    ///
    /// fn main() -> Result<(), StrError> {
    ///     // allocate a square matrix and store as CSC matrix
    ///     // ┌                ┐
    ///     // │  2  3  0  0  0 │
    ///     // │  3  0  4  0  6 │
    ///     // │  0 -1 -3  2  0 │
    ///     // │  0  0  1  0  0 │
    ///     // │  0  4  2  0  1 │
    ///     // └                ┘
    ///     let csc = CscMatrix {
    ///         symmetry: None,
    ///         nrow: 5,
    ///         ncol: 5,
    ///         col_pointers: vec![0, 2, 5, 9, 10, 12],
    ///         row_indices: vec![
    ///             //                             p
    ///             0, 1, //       j = 0, count =  0, 1,
    ///             0, 2, 4, //    j = 1, count =  2, 3, 4,
    ///             1, 2, 3, 4, // j = 2, count =  5, 6, 7, 8,
    ///             2, //          j = 3, count =  9,
    ///             1, 4, //       j = 4, count = 10, 11,
    ///                //                         12
    ///         ],
    ///         values: vec![
    ///             //                                      p
    ///             2.0, 3.0, //            j = 0, count =  0, 1,
    ///             3.0, -1.0, 4.0, //      j = 1, count =  2, 3, 4,
    ///             4.0, -3.0, 1.0, 2.0, // j = 2, count =  5, 6, 7, 8,
    ///             2.0, //                 j = 3, count =  9,
    ///             6.0, 1.0, //            j = 4, count = 10, 11,
    ///                  //                                12
    ///         ],
    ///     };
    ///
    ///     // covert to dense
    ///     let mut a = Matrix::new(5, 5);
    ///     csc.to_matrix(&mut a)?;
    ///     let correct = "┌                ┐\n\
    ///                    │  2  3  0  0  0 │\n\
    ///                    │  3  0  4  0  6 │\n\
    ///                    │  0 -1 -3  2  0 │\n\
    ///                    │  0  0  1  0  0 │\n\
    ///                    │  0  4  2  0  1 │\n\
    ///                    └                ┘";
    ///     assert_eq!(format!("{}", a), correct);
    ///     Ok(())
    /// }
    /// ```
    pub fn to_matrix(&self, a: &mut Matrix) -> Result<(), StrError> {
        let (m, n) = a.dims();
        if m != self.nrow || n != self.ncol {
            return Err("wrong matrix dimensions");
        }
        let mirror_required = match self.symmetry {
            Some(sym) => sym.triangular(),
            None => false,
        };
        a.fill(0.0);
        for j in 0..self.ncol {
            for p in self.col_pointers[j]..self.col_pointers[j + 1] {
                let i = self.row_indices[p as usize] as usize;
                a.add(i, j, self.values[p as usize]);
                if mirror_required && i != j {
                    a.add(j, i, self.values[p as usize]);
                }
            }
        }
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::CscMatrix;
    use crate::{CooMatrix, Storage, Symmetry};
    use russell_chk::vec_approx_eq;
    use russell_lab::Matrix;

    #[test]
    fn new_works() {
        let csc = CscMatrix::new(None, 3, 3, 4);
        assert_eq!(csc.symmetry, None);
        assert_eq!(csc.nrow, 3);
        assert_eq!(csc.ncol, 3);
        assert_eq!(csc.col_pointers, &[0, 0, 0, 0]);
        assert_eq!(csc.row_indices, &[0, 0, 0, 0]);
        assert_eq!(csc.values, &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn csc_matrix_first_triplet_with_shuffled_entries() {
        //  1  -1   .  -3   .
        // -2   5   .   .   .
        //  .   .   4   6   4
        // -4   .   2   7   .
        //  .   8   .   .  -5
        // first triplet with shuffled entries
        let mut coo = CooMatrix::new(None, 5, 5, 13).unwrap();
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
        let csc = CscMatrix::from(&coo).unwrap();
        // solution
        let correct_x = vec![
            //                                  p
            1.0, -2.0, -4.0, // j = 0, count =  0, 1, 2,
            -1.0, 5.0, 8.0, //  j = 1, count =  3, 4, 5,
            4.0, 2.0, //        j = 2, count =  6, 7,
            -3.0, 6.0, 7.0, //  j = 3, count =  8, 9, 10,
            4.0, -5.0, //       j = 4, count = 11, 12,
                  //                           13
        ];
        let correct_i = vec![
            //                          p
            0, 1, 3, // j = 0, count =  0, 1, 2,
            0, 1, 4, // j = 1, count =  3, 4, 5,
            2, 3, //    j = 2, count =  6, 7,
            0, 2, 3, // j = 3, count =  8, 9, 10,
            2, 4, //    j = 4, count = 11, 12,
               //                      13
        ];
        let correct_p = vec![0, 3, 6, 8, 11, 13];
        assert_eq!(&csc.col_pointers, &correct_p);
        assert_eq!(&csc.row_indices, &correct_i);
        vec_approx_eq(&csc.values, &correct_x, 1e-15);
    }

    #[test]
    fn csc_matrix_small_triplet_with_shuffled_entries() {
        // 1  2  .  .  .
        // 3  4  .  .  .
        // .  .  5  6  .
        // .  .  7  8  .
        // .  .  .  .  9
        // small triplet with shuffled entries
        let mut coo = CooMatrix::new(None, 5, 5, 9).unwrap();
        coo.put(4, 4, 9.0).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(1, 0, 3.0).unwrap();
        coo.put(2, 2, 5.0).unwrap();
        coo.put(2, 3, 6.0).unwrap();
        coo.put(0, 1, 2.0).unwrap();
        coo.put(3, 2, 7.0).unwrap();
        coo.put(1, 1, 4.0).unwrap();
        coo.put(3, 3, 8.0).unwrap();
        let csc = CscMatrix::from(&coo).unwrap();
        // solution
        let correct_x = vec![
            //                           p
            1.0, 3.0, // j = 0, count =  0, 1,
            2.0, 4.0, // j = 1, count =  2, 3,
            5.0, 7.0, // j = 2, count =  4, 5,
            6.0, 8.0, // j = 3, count =  6, 7,
            9.0, //      j = 4, count =  8,
                 //                      9
        ];
        let correct_i = vec![
            //                       p
            0, 1, // j = 0, count =  0, 1,
            0, 1, // j = 1, count =  2, 3,
            2, 3, // j = 2, count =  4, 5,
            2, 3, // j = 3, count =  6, 7,
            4, //    j = 4, count =  8,
               //                    9
        ];
        let correct_p = vec![0, 2, 4, 6, 8, 9];
        assert_eq!(&csc.col_pointers, &correct_p);
        assert_eq!(&csc.row_indices, &correct_i);
        vec_approx_eq(&csc.values, &correct_x, 1e-15);
    }

    #[test]
    fn csc_matrix_small_triplet_with_duplicates() {
        // 1  2  .  .  .
        // 3  4  .  .  .
        // .  .  5  6  .
        // .  .  7  8  .
        // .  .  .  .  9
        // with duplicates
        let mut coo = CooMatrix::new(None, 5, 5, 11).unwrap();
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
        let csc = CscMatrix::from(&coo).unwrap();
        // solution
        let correct_x = vec![
            //                           p
            1.0, 3.0, // j = 0, count =  0, 1,
            2.0, 4.0, // j = 1, count =  2, 3,
            5.0, 7.0, // j = 2, count =  4, 5,
            6.0, 8.0, // j = 3, count =  6, 7,
            9.0, //      j = 4, count =  8,
                 //                      9
        ];
        let correct_i = vec![
            //                       p
            0, 1, // j = 0, count =  0, 1,
            0, 1, // j = 1, count =  2, 3,
            2, 3, // j = 2, count =  4, 5,
            2, 3, // j = 3, count =  6, 7,
            4, //    j = 4, count =  8,
               //                    9
        ];
        let correct_p = vec![0, 2, 4, 6, 8, 9];
        // solution
        assert_eq!(&csc.col_pointers, &correct_p);
        assert_eq!(&csc.row_indices, &correct_i);
        vec_approx_eq(&csc.values, &correct_x, 1e-15);
    }

    #[test]
    fn csc_matrix_symmetric_with_all_values() {
        //  9.00  1.5   6.0  0.750   3.0
        //  1.50  0.5   0.0  0.000   0.0
        //  6.00  0.0  12.0  0.000   0.0
        //  0.75  0.0   0.0  0.625   0.0
        //  3.00  0.0   0.0  0.000  16.0
        // symmetric matrix, but all values provided
        let mut coo = CooMatrix::new(None, 5, 5, 13).unwrap();
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
        let csc = CscMatrix::from(&coo).unwrap();
        // solution
        let correct_x = vec![
            //                                     p
            9.0, 1.5, 6.0, 0.75, 3.0, // j=0 count=0,1,2,3,4
            1.5, 0.5, //                 j=1 count=5,6
            6.0, 12.0, //                j=2 count=7,8
            0.75, 0.625, //              j=3 count=9,10
            3.0, 16.0, //                j=4 count=11,12
                  //                               13
        ];
        let correct_i = vec![
            0, 1, 2, 3, 4, //
            0, 1, //
            0, 2, //
            0, 3, //
            0, 4,
        ];
        let correct_p = vec![0, 5, 7, 9, 11, 13];
        assert_eq!(&csc.col_pointers, &correct_p);
        assert_eq!(&csc.row_indices, &correct_i);
        vec_approx_eq(&csc.values, &correct_x, 1e-15);
    }

    #[test]
    fn csc_matrix_upper_triangular_with_ordered_entries() {
        //  9.00  1.5   6.0  0.750   3.0
        //  1.50  0.5   0.0  0.000   0.0
        //  6.00  0.0  12.0  0.000   0.0
        //  0.75  0.0   0.0  0.625   0.0
        //  3.00  0.0   0.0  0.000  16.0
        // upper triangular with ordered entries
        let sym = Some(Symmetry::General(Storage::Upper));
        let mut coo = CooMatrix::new(sym, 5, 5, 9).unwrap();
        coo.put(0, 0, 9.0).unwrap();
        coo.put(0, 1, 1.5).unwrap();
        coo.put(0, 2, 6.0).unwrap();
        coo.put(0, 3, 0.75).unwrap();
        coo.put(0, 4, 3.0).unwrap();
        coo.put(1, 1, 0.5).unwrap();
        coo.put(2, 2, 12.0).unwrap();
        coo.put(3, 3, 0.625).unwrap();
        coo.put(4, 4, 16.0).unwrap();
        let csc = CscMatrix::from(&coo).unwrap();
        // solution
        let correct_x = vec![
            //                        p
            9.0, //         j=0 count=0
            1.5, 0.5, //    j=1 count=1,2
            6.0, 12.0, //   j=2 count=3,4
            0.75, 0.625, // j=3 count=5,6
            3.0, 16.0, //   j=4 count=7,8
                  //                  9
        ];
        let correct_i = vec![
            0, //
            0, 1, //
            0, 2, //
            0, 3, //
            0, 4,
        ];
        let correct_p = vec![0, 1, 3, 5, 7, 9];
        assert_eq!(&csc.col_pointers, &correct_p);
        assert_eq!(&csc.row_indices, &correct_i);
        vec_approx_eq(&csc.values, &correct_x, 1e-15);
    }

    #[test]
    fn csc_matrix_upper_triangular_with_shuffled_entries() {
        //  9.00  1.5   6.0  0.750   3.0
        //  1.50  0.5   0.0  0.000   0.0
        //  6.00  0.0  12.0  0.000   0.0
        //  0.75  0.0   0.0  0.625   0.0
        //  3.00  0.0   0.0  0.000  16.0
        // upper triangular with shuffled entries
        let sym = Some(Symmetry::General(Storage::Upper));
        let mut coo = CooMatrix::new(sym, 5, 5, 9).unwrap();
        coo.put(2, 2, 12.0).unwrap();
        coo.put(0, 0, 9.0).unwrap();
        coo.put(3, 3, 0.625).unwrap();
        coo.put(0, 1, 1.5).unwrap();
        coo.put(0, 2, 6.0).unwrap();
        coo.put(4, 4, 16.0).unwrap();
        coo.put(0, 3, 0.75).unwrap();
        coo.put(1, 1, 0.5).unwrap();
        coo.put(0, 4, 3.0).unwrap();
        let csc = CscMatrix::from(&coo).unwrap();
        // solution
        let correct_x = vec![
            //                        p
            9.0, //         j=0 count=0
            1.5, 0.5, //    j=1 count=1,2
            6.0, 12.0, //   j=2 count=3,4
            0.75, 0.625, // j=3 count=5,6
            3.0, 16.0, //   j=4 count=7,8
                  //                  9
        ];
        let correct_i = vec![
            0, //
            0, 1, //
            0, 2, //
            0, 3, //
            0, 4,
        ];
        let correct_p = vec![0, 1, 3, 5, 7, 9];
        assert_eq!(&csc.col_pointers, &correct_p);
        assert_eq!(&csc.row_indices, &correct_i);
        vec_approx_eq(&csc.values, &correct_x, 1e-15);
    }

    #[test]
    fn csc_matrix_upper_triangular_with_diagonal_entries_first() {
        //  9.00  1.5   6.0  0.750   3.0
        //  1.50  0.5   0.0  0.000   0.0
        //  6.00  0.0  12.0  0.000   0.0
        //  0.75  0.0   0.0  0.625   0.0
        //  3.00  0.0   0.0  0.000  16.0
        // upper triangular with diagonal entries being set first
        let sym = Some(Symmetry::General(Storage::Upper));
        let mut coo = CooMatrix::new(sym, 5, 5, 9).unwrap();
        // diagonal
        coo.put(0, 0, 9.0).unwrap();
        coo.put(1, 1, 0.5).unwrap();
        coo.put(2, 2, 12.0).unwrap();
        coo.put(3, 3, 0.625).unwrap();
        coo.put(4, 4, 16.0).unwrap();
        // upper diagonal
        coo.put(0, 1, 1.5).unwrap();
        coo.put(0, 2, 6.0).unwrap();
        coo.put(0, 3, 0.75).unwrap();
        coo.put(0, 4, 3.0).unwrap();
        let csc = CscMatrix::from(&coo).unwrap();
        // solution
        let correct_x = vec![
            //                        p
            9.0, //         j=0 count=0
            1.5, 0.5, //    j=1 count=1,2
            6.0, 12.0, //   j=2 count=3,4
            0.75, 0.625, // j=3 count=5,6
            3.0, 16.0, //   j=4 count=7,8
                  //                  9
        ];
        let correct_i = vec![
            0, //
            0, 1, //
            0, 2, //
            0, 3, //
            0, 4,
        ];
        let correct_p = vec![0, 1, 3, 5, 7, 9];
        assert_eq!(&csc.col_pointers, &correct_p);
        assert_eq!(&csc.row_indices, &correct_i);
        vec_approx_eq(&csc.values, &correct_x, 1e-15);
    }

    #[test]
    fn csc_matrix_lower_triangular_with_ordered_entries() {
        //  9.00  1.5   6.0  0.750   3.0
        //  1.50  0.5   0.0  0.000   0.0
        //  6.00  0.0  12.0  0.000   0.0
        //  0.75  0.0   0.0  0.625   0.0
        //  3.00  0.0   0.0  0.000  16.0
        // lower diagonal with ordered entries
        let sym = Some(Symmetry::General(Storage::Lower));
        let mut coo = CooMatrix::new(sym, 5, 5, 9).unwrap();
        coo.put(0, 0, 9.0).unwrap();
        coo.put(1, 0, 1.5).unwrap();
        coo.put(1, 1, 0.5).unwrap();
        coo.put(2, 0, 6.0).unwrap();
        coo.put(2, 2, 12.0).unwrap();
        coo.put(3, 0, 0.75).unwrap();
        coo.put(3, 3, 0.625).unwrap();
        coo.put(4, 0, 3.0).unwrap();
        coo.put(4, 4, 16.0).unwrap();
        let csc = CscMatrix::from(&coo).unwrap();
        // solution
        let correct_x = vec![
            //                                       p
            9.0, 1.5, 6.0, 0.75, 3.0,   // j=0 count=0,1,2,3,4
            0.5,   //                      j=1 count=5
            12.0,  //                      j=2 count=6
            0.625, //                      j=3 count=7
            16.0,  //                      j=4 count=8
                   //                                9
        ];
        let correct_i = vec![
            0, 1, 2, 3, 4, //
            1, //
            2, //
            3, //
            4,
        ];
        let correct_p = vec![0, 5, 6, 7, 8, 9];
        assert_eq!(&csc.col_pointers, &correct_p);
        assert_eq!(&csc.row_indices, &correct_i);
        vec_approx_eq(&csc.values, &correct_x, 1e-15);
    }

    #[test]
    fn csc_matrix_lower_triangular_with_diagonal_first() {
        //  9.00  1.5   6.0  0.750   3.0
        //  1.50  0.5   0.0  0.000   0.0
        //  6.00  0.0  12.0  0.000   0.0
        //  0.75  0.0   0.0  0.625   0.0
        //  3.00  0.0   0.0  0.000  16.0
        // lower triangular with diagonal entries being set first
        let sym = Some(Symmetry::General(Storage::Lower));
        let mut coo = CooMatrix::new(sym, 5, 5, 9).unwrap();
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
        let csc = CscMatrix::from(&coo).unwrap();
        // solution
        let correct_x = vec![
            //                                       p
            9.0, 1.5, 6.0, 0.75, 3.0,   // j=0 count=0,1,2,3,4
            0.5,   //                      j=1 count=5
            12.0,  //                      j=2 count=6
            0.625, //                      j=3 count=7
            16.0,  //                      j=4 count=8
                   //                                9
        ];
        let correct_i = vec![
            0, 1, 2, 3, 4, //
            1, //
            2, //
            3, //
            4,
        ];
        let correct_p = vec![0, 5, 6, 7, 8, 9];
        assert_eq!(&csc.col_pointers, &correct_p);
        assert_eq!(&csc.row_indices, &correct_i);
        vec_approx_eq(&csc.values, &correct_x, 1e-15);
    }

    #[test]
    fn to_matrix_fails_on_wrong_dims() {
        let sym = Some(Symmetry::General(Storage::Upper));
        let csc = CscMatrix::new(sym, 2, 2, 3);
        let mut a_2x1 = Matrix::new(3, 1);
        let mut a_1x2 = Matrix::new(1, 3);
        assert_eq!(csc.to_matrix(&mut a_2x1), Err("wrong matrix dimensions"));
        assert_eq!(csc.to_matrix(&mut a_1x2), Err("wrong matrix dimensions"));
    }

    #[test]
    fn to_matrix_and_as_matrix_work() {
        let csc = CscMatrix {
            symmetry: None,
            nrow: 5,
            ncol: 5,
            col_pointers: vec![0, 2, 5, 9, 10, 12],
            row_indices: vec![
                //                             p
                0, 1, //       j = 0, count =  0, 1,
                0, 2, 4, //    j = 1, count =  2, 3, 4,
                1, 2, 3, 4, // j = 2, count =  5, 6, 7, 8,
                2, //          j = 3, count =  9,
                1, 4, //       j = 4, count = 10, 11,
                   //                         12
            ],
            values: vec![
                //                                      p
                2.0, 3.0, //            j = 0, count =  0, 1,
                3.0, -1.0, 4.0, //      j = 1, count =  2, 3, 4,
                4.0, -3.0, 1.0, 2.0, // j = 2, count =  5, 6, 7, 8,
                2.0, //                 j = 3, count =  9,
                6.0,
                1.0, //            j = 4, count = 10, 11,
                     //                                12
            ],
        };

        // covert to dense
        let mut a = Matrix::new(5, 5);
        csc.to_matrix(&mut a).unwrap();
        let correct = "┌                ┐\n\
                       │  2  3  0  0  0 │\n\
                       │  3  0  4  0  6 │\n\
                       │  0 -1 -3  2  0 │\n\
                       │  0  0  1  0  0 │\n\
                       │  0  4  2  0  1 │\n\
                       └                ┘";
        assert_eq!(format!("{}", a), correct);

        // use as_matrix
        let b = csc.as_matrix().unwrap();
        assert_eq!(format!("{}", b), correct);
    }

    #[test]
    fn to_matrix_upper_works() {
        let csc = CscMatrix {
            symmetry: Some(Symmetry::General(Storage::Upper)),
            nrow: 5,
            ncol: 5,
            col_pointers: vec![0, 1, 3, 5, 7, 9],
            row_indices: vec![0, 0, 1, 0, 2, 0, 3, 0, 4],
            values: vec![9.0, 1.5, 0.5, 6.0, 12.0, 0.75, 0.625, 3.0, 16.0],
        };
        let a = csc.as_matrix().unwrap();
        let correct = "┌                               ┐\n\
                       │     9   1.5     6  0.75     3 │\n\
                       │   1.5   0.5     0     0     0 │\n\
                       │     6     0    12     0     0 │\n\
                       │  0.75     0     0 0.625     0 │\n\
                       │     3     0     0     0    16 │\n\
                       └                               ┘";
        assert_eq!(format!("{}", a), correct);
    }

    #[test]
    fn to_matrix_lower_works() {
        let csc = CscMatrix {
            symmetry: Some(Symmetry::General(Storage::Lower)),
            nrow: 5,
            ncol: 5,
            col_pointers: vec![0, 5, 6, 7, 8, 9],
            row_indices: vec![0, 1, 2, 3, 4, 1, 2, 3, 4],
            values: vec![9.0, 1.5, 6.0, 0.75, 3.0, 0.5, 12.0, 0.625, 16.0],
        };
        let a = csc.as_matrix().unwrap();
        let correct = "┌                               ┐\n\
                       │     9   1.5     6  0.75     3 │\n\
                       │   1.5   0.5     0     0     0 │\n\
                       │     6     0    12     0     0 │\n\
                       │  0.75     0     0 0.625     0 │\n\
                       │     3     0     0     0    16 │\n\
                       └                               ┘";
        assert_eq!(format!("{}", a), correct);
    }
}
