use super::{handle_umfpack_error_code, to_i32, CooMatrix, CsrMatrix, Symmetry};
use crate::StrError;
use russell_lab::{Matrix, Vector};

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
    /// Checks the dimension of the arrays in the CSC matrix
    ///
    /// The following conditions must be satisfied:
    ///
    /// ```text
    /// nrow ≥ 1
    /// ncol ≥ 1
    /// nnz = col_pointers[ncol] ≥ 1
    /// col_pointers.len() == ncol + 1
    /// row_indices.len() == nnz
    /// values.len() == nnz
    /// ```
    pub fn check_dimensions(&self) -> Result<(), StrError> {
        if self.nrow < 1 {
            return Err("nrow must be ≥ 1");
        }
        if self.ncol < 1 {
            return Err("ncol must be ≥ 1");
        }
        if self.col_pointers.len() != self.ncol + 1 {
            return Err("col_pointers.len() must be = ncol + 1");
        }
        let nnz = self.col_pointers[self.ncol];
        if nnz < 1 {
            return Err("nnz must be ≥ 1");
        }
        if self.row_indices.len() != nnz as usize {
            return Err("row_indices.len() must be = nnz");
        }
        if self.values.len() != nnz as usize {
            return Err("values.len() must be = nnz");
        }
        Ok(())
    }

    /// Creates a new CSC matrix from a COO matrix
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
    ///     let mut coo = CooMatrix::new(nrow, ncol, nnz, None, false)?;
    ///     coo.put(0, 0, 1.0)?;
    ///     coo.put(0, 2, 2.0)?;
    ///     coo.put(1, 2, 3.0)?;
    ///     coo.put(2, 0, 4.0)?;
    ///     coo.put(2, 1, 5.0)?;
    ///     coo.put(2, 2, 6.0)?;
    ///
    ///     // convert to CCR matrix
    ///     let csc = CscMatrix::from_coo(&coo)?;
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
    pub fn from_coo(coo: &CooMatrix) -> Result<Self, StrError> {
        coo.check_dimensions_ready()?;
        let mut csc = CscMatrix {
            symmetry: coo.symmetry,
            nrow: coo.nrow,
            ncol: coo.ncol,
            col_pointers: vec![0; coo.ncol + 1],
            row_indices: vec![0; coo.pos],
            values: vec![0.0; coo.pos],
        };
        csc.update_from_coo(coo)?;
        Ok(csc)
    }

    /// Updates this CSC matrix from a COO matrix with a compatible structure
    ///
    /// **Note:** The COO matrix must match the symmetry, nrow, and ncol values.
    /// Also, the `pos` (nnz) value in the COO matrix must match `row_indices.len()`.
    pub fn update_from_coo(&mut self, coo: &CooMatrix) -> Result<(), StrError> {
        // check dimensions
        if coo.symmetry != self.symmetry {
            return Err("coo.symmetry must be equal to coo.symmetry");
        }
        if coo.nrow != self.nrow {
            return Err("coo.nrow must be equal to csc.nrow");
        }
        if coo.ncol != self.ncol {
            return Err("coo.ncol must be equal to csc.ncol");
        }
        if coo.pos != self.row_indices.len() {
            return Err("coo.pos must be equal to csc.row_indices.len()");
        }
        if coo.pos != self.values.len() {
            return Err("coo.pos must be equal to csc.values.len()");
        }

        // call UMFPACK to convert COO to CSC
        let status = if coo.one_based {
            // handle one-based indexing
            let mut indices_i = coo.indices_i.clone();
            let mut indices_j = coo.indices_j.clone();
            for k in 0..coo.pos {
                indices_i[k] -= 1;
                indices_j[k] -= 1;
            }
            unsafe {
                umfpack_coo_to_csc(
                    self.col_pointers.as_mut_ptr(),
                    self.row_indices.as_mut_ptr(),
                    self.values.as_mut_ptr(),
                    to_i32(coo.nrow)?,
                    to_i32(coo.ncol)?,
                    to_i32(coo.pos)?,
                    indices_i.as_ptr(),
                    indices_j.as_ptr(),
                    coo.values_aij.as_ptr(),
                )
            }
        } else {
            unsafe {
                umfpack_coo_to_csc(
                    self.col_pointers.as_mut_ptr(),
                    self.row_indices.as_mut_ptr(),
                    self.values.as_mut_ptr(),
                    to_i32(coo.nrow)?,
                    to_i32(coo.ncol)?,
                    to_i32(coo.pos)?,
                    coo.indices_i.as_ptr(),
                    coo.indices_j.as_ptr(),
                    coo.values_aij.as_ptr(),
                )
            }
        };
        if status != 0 {
            return Err(handle_umfpack_error_code(status));
        }

        // reduce array sizes if duplicates have been eliminated
        let final_nnz = self.col_pointers[self.ncol] as usize;
        if final_nnz < coo.pos {
            self.row_indices.resize(final_nnz, 0);
            self.values.resize(final_nnz, 0.0);
        }

        // results
        Ok(())
    }

    /// Creates a new CSC matrix from a CSR matrix
    pub fn from_csr(csr: &CsrMatrix) -> Result<Self, StrError> {
        // Based on the SciPy code (csr_tocsc) from here:
        //
        // https://github.com/scipy/scipy/blob/main/scipy/sparse/sparsetools/csr.h
        //
        // Notes:
        //
        // * Linear complexity: O(nnz(A) + max(nrow, ncol))
        // * Upgrading i32 to usize is OK (the opposite is not OK => use to_i32)

        // check and read in the dimensions
        csr.check_dimensions()?;
        let nrow = csr.nrow as usize;
        let ncol = csr.ncol as usize;
        let nnz = csr.row_pointers[nrow] as usize;

        // access the CSR data
        let ap = &csr.row_pointers;
        let aj = &csr.col_indices;
        let ax = &csr.values;

        // allocate the CSC arrays
        let mut csc = CscMatrix {
            symmetry: csr.symmetry,
            nrow: csr.nrow,
            ncol: csr.ncol,
            col_pointers: vec![0; ncol + 1],
            row_indices: vec![0; nnz],
            values: vec![0.0; nnz],
        };

        // access the CSC data
        let bp = &mut csc.col_pointers;
        let bi = &mut csc.row_indices;
        let bx = &mut csc.values;

        // compute the number of non-zero entries per column of A
        for k in 0..nnz {
            bp[aj[k] as usize] += 1;
        }

        // perform the cumulative sum of the nnz per row to get bp
        let mut sum = 0;
        for j in 0..ncol {
            let temp = bp[j];
            bp[j] = sum;
            sum += temp;
        }
        bp[ncol] = to_i32(nnz)?;

        // write aj and ax into bi and bx (will use bp as workspace)
        for i in 0..nrow {
            for p in ap[i]..ap[i + 1] {
                let j = aj[p as usize] as usize;
                let dest = bp[j] as usize;
                bi[dest] = to_i32(i)?;
                bx[dest] = ax[p as usize];
                bp[j] += 1;
            }
        }

        // fix bp
        let mut last = 0;
        for j in 0..(ncol + 1) {
            let temp = bp[j];
            bp[j] = last;
            last = temp;
        }

        // results
        Ok(csc)
    }

    /// Converts this CSC matrix to a dense matrix
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

    /// Converts this CSC matrix to a dense matrix
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
        self.check_dimensions()?;
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

    /// Performs the matrix-vector multiplication
    ///
    /// ```text
    ///  v  :=  α ⋅  a   ⋅  u
    /// (m)        (m,n)   (n)
    /// ```
    ///
    /// # Input
    ///
    /// * `u` -- Vector with dimension equal to the number of columns of the matrix
    ///
    /// # Output
    ///
    /// * `v` -- Vector with dimension equal to the number of rows of the matrix
    pub fn mat_vec_mul(&self, v: &mut Vector, alpha: f64, u: &Vector) -> Result<(), StrError> {
        self.check_dimensions()?;
        if u.dim() != self.ncol {
            return Err("u.ndim must equal ncol");
        }
        if v.dim() != self.nrow {
            return Err("v.ndim must equal nrow");
        }
        let mirror_required = match self.symmetry {
            Some(sym) => sym.triangular(),
            None => false,
        };
        v.fill(0.0);
        for j in 0..self.ncol {
            for p in self.col_pointers[j]..self.col_pointers[j + 1] {
                let i = self.row_indices[p as usize] as usize;
                let aij = self.values[p as usize];
                v[i] += alpha * aij * u[j];
                if mirror_required && i != j {
                    v[j] += alpha * aij * u[i];
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
    use crate::{CooMatrix, Samples, Storage, Symmetry};
    use russell_chk::vec_approx_eq;
    use russell_lab::{Matrix, Vector};

    #[test]
    fn from_coo_captures_errors() {
        let coo = CooMatrix::new(1, 1, 1, None, false).unwrap();
        assert_eq!(
            CscMatrix::from_coo(&coo).err(),
            Some("COO matrix: pos = nnz must be ≥ 1")
        );
    }

    #[test]
    fn from_coo_works() {
        for (coo, csc_correct, _, _) in [
            //  2  3  .  .  .
            //  3  .  4  .  6
            //  . -1 -3  2  .
            //  .  .  1  .  .
            //  .  4  2  .  1
            Samples::umfpack_unsymmetric_5x5(false),
            Samples::umfpack_unsymmetric_5x5(true),
            //  1  -1   .  -3   .
            // -2   5   .   .   .
            //  .   .   4   6   4
            // -4   .   2   7   .
            //  .   8   .   .  -5
            Samples::mkl_unsymmetric_5x5(false),
            Samples::mkl_unsymmetric_5x5(true),
            // 1  2  .  .  .
            // 3  4  .  .  .
            // .  .  5  6  .
            // .  .  7  8  .
            // .  .  .  .  9
            Samples::block_unsymmetric_5x5(false, false, false),
            Samples::block_unsymmetric_5x5(false, true, false),
            Samples::block_unsymmetric_5x5(false, false, true),
            Samples::block_unsymmetric_5x5(false, true, true),
            Samples::block_unsymmetric_5x5(true, false, false),
            Samples::block_unsymmetric_5x5(true, true, false),
            Samples::block_unsymmetric_5x5(true, false, true),
            Samples::block_unsymmetric_5x5(true, true, true),
            //     9   1.5     6  0.75     3
            //   1.5   0.5     .     .     .
            //     6     .    12     .     .
            //  0.75     .     . 0.625     .
            //     3     .     .     .    16
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
            // ┌               ┐
            // │ 1 . 3 . 5 . 7 │
            // └               ┘
            Samples::rectangular_1x7(),
            Samples::rectangular_7x1(),
            Samples::rectangular_3x4(),
            // ┌     ┐
            // │ 123 │
            // └     ┘
            Samples::tiny_1x1(),
        ] {
            let csc = CscMatrix::from_coo(&coo).unwrap();
            csc.check_dimensions().unwrap();
            assert_eq!(&csc.col_pointers, &csc_correct.col_pointers);
            assert_eq!(&csc.row_indices, &csc_correct.row_indices);
            vec_approx_eq(&csc.values, &csc_correct.values, 1e-15);
        }
    }

    #[test]
    fn from_csr_works() {
        const IGNORED: bool = false;
        for (_, csc_correct, csr, _) in [
            //  2  3  .  .  .
            //  3  .  4  .  6
            //  . -1 -3  2  .
            //  .  .  1  .  .
            //  .  4  2  .  1
            Samples::umfpack_unsymmetric_5x5(IGNORED),
            //  1  -1   .  -3   .
            // -2   5   .   .   .
            //  .   .   4   6   4
            // -4   .   2   7   .
            //  .   8   .   .  -5
            Samples::mkl_unsymmetric_5x5(IGNORED),
            // 1  2  .  .  .
            // 3  4  .  .  .
            // .  .  5  6  .
            // .  .  7  8  .
            // .  .  .  .  9
            Samples::block_unsymmetric_5x5(IGNORED, IGNORED, IGNORED),
            //     9   1.5     6  0.75     3
            //   1.5   0.5     .     .     .
            //     6     .    12     .     .
            //  0.75     .     . 0.625     .
            //     3     .     .     .    16
            Samples::mkl_positive_definite_5x5_lower(IGNORED),
            Samples::mkl_symmetric_5x5_lower(IGNORED, IGNORED, IGNORED),
            Samples::mkl_symmetric_5x5_upper(IGNORED, IGNORED, IGNORED),
            Samples::mkl_symmetric_5x5_full(IGNORED),
            // ┌               ┐
            // │ 1 . 3 . 5 . 7 │
            // └               ┘
            Samples::rectangular_1x7(),
            Samples::rectangular_7x1(),
            Samples::rectangular_3x4(),
            // ┌     ┐
            // │ 123 │
            // └     ┘
            Samples::tiny_1x1(),
        ] {
            let csc = CscMatrix::from_csr(&csr).unwrap();
            assert_eq!(&csc.col_pointers, &csc_correct.col_pointers);
            assert_eq!(&csc.row_indices, &csc_correct.row_indices);
            vec_approx_eq(&csc.values, &csc_correct.values, 1e-15);
        }
    }

    #[test]
    fn check_dimensions_works() {
        let mut csc = CscMatrix {
            symmetry: None,
            nrow: 0,
            ncol: 0,
            col_pointers: Vec::new(),
            row_indices: Vec::new(),
            values: Vec::new(),
        };
        assert_eq!(csc.check_dimensions().err(), Some("nrow must be ≥ 1"));
        csc.nrow = 2;
        csc.ncol = 0;
        assert_eq!(csc.check_dimensions().err(), Some("ncol must be ≥ 1"));
        csc.ncol = 4;
        assert_eq!(
            csc.check_dimensions().err(),
            Some("col_pointers.len() must be = ncol + 1")
        );
        csc.col_pointers = vec![0, 0, 0, 0, 0];
        assert_eq!(csc.check_dimensions().err(), Some("nnz must be ≥ 1"));
        csc.col_pointers = vec![0, 0, 0, 0, 1];
        assert_eq!(csc.check_dimensions().err(), Some("row_indices.len() must be = nnz"));
        csc.row_indices = vec![0];
        assert_eq!(csc.check_dimensions().err(), Some("values.len() must be = nnz"));
        csc.values = vec![0.0];
        assert_eq!(csc.check_dimensions().err(), None);
    }

    #[test]
    fn to_matrix_fails_on_wrong_dims() {
        // 10.0 20.0       << (1 x 2) matrix
        let csc = CscMatrix {
            symmetry: None,
            nrow: 1,
            ncol: 2,
            col_pointers: vec![0, 1, 2],
            row_indices: vec![0, 0],
            values: vec![10.0, 20.0],
        };
        let mut a_3x1 = Matrix::new(3, 1);
        let mut a_1x3 = Matrix::new(1, 3);
        assert_eq!(csc.to_matrix(&mut a_3x1), Err("wrong matrix dimensions"));
        assert_eq!(csc.to_matrix(&mut a_1x3), Err("wrong matrix dimensions"));
    }

    #[test]
    fn to_matrix_and_as_matrix_work() {
        // 10.0 20.0       << (1 x 2) matrix
        let csc = CscMatrix {
            symmetry: None,
            nrow: 1,
            ncol: 2,
            col_pointers: vec![0, 1, 2],
            row_indices: vec![0, 0],
            values: vec![10.0, 20.0],
        };
        let mut a = Matrix::new(1, 2);
        csc.to_matrix(&mut a).unwrap();
        let correct = "┌       ┐\n\
                       │ 10 20 │\n\
                       └       ┘";
        assert_eq!(format!("{}", a), correct);

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
        // call to_matrix again to make sure the matrix is filled with zeros before the sum
        csc.to_matrix(&mut a).unwrap();
        assert_eq!(format!("{}", a), correct);

        // use as_matrix
        let b = csc.as_matrix().unwrap();
        assert_eq!(format!("{}", b), correct);
    }

    #[test]
    fn as_matrix_upper_works() {
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
    fn as_matrix_lower_works() {
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

    #[test]
    fn mat_vec_mul_works() {
        //  5.0, -2.0, 0.0, 1.0,
        // 10.0, -4.0, 0.0, 2.0,
        // 15.0, -6.0, 0.0, 3.0,
        let (_, csc, _, _) = Samples::rectangular_3x4();
        let u = Vector::from(&[1.0, 3.0, 8.0, 5.0]);
        let mut v = Vector::new(csc.nrow);
        csc.mat_vec_mul(&mut v, 1.0, &u).unwrap();
        let correct = &[4.0, 8.0, 12.0];
        vec_approx_eq(v.as_data(), correct, 1e-15);
        // call mat_vec_mul again to make sure the vector is filled with zeros before the sum
        csc.mat_vec_mul(&mut v, 1.0, &u).unwrap();
        vec_approx_eq(v.as_data(), correct, 1e-15);
    }
}
