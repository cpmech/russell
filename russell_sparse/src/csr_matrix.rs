use super::{CooMatrix, Symmetry};
use crate::StrError;
use russell_lab::{Matrix, Vector};
use russell_openblas::to_i32;

/// Holds the arrays needed for a CSR (compressed sparse row) matrix
///
/// # Example
///
/// The sparse matrix is (dots indicate zero values);
///
/// ```text
///  1  -1   .  -3   .
/// -2   5   .   .   .
///  .   .   4   6   4
/// -4   .   2   7   .
///  .   8   .   .  -5
/// ```
///
/// The values in compressed row order are (note the row indices `i` and pointers `p`):
///
/// ```text
///                                   p
///  1.0, -1.0, -3.0,  i = 0, count = 0,  1,  2,
/// -2.0,  5.0,        i = 1, count = 3,  4,
///  4.0,  6.0,  4.0,  i = 2, count = 5,  6,  7,
/// -4.0,  2.0,  7.0,  i = 3, count = 8,  9,  10,
///  8.0, -5.0,        i = 4, count= 11, 12,
///                                  13
/// ```
///
/// The column indices are:
///
/// ```text
/// 0, 1, 3,
/// 0, 1,
/// 2, 3, 4,
/// 0, 2, 3,
/// 1, 4,
/// ```
///
/// And the row pointers are (see the column indicated by `p` above):
///
/// ```text
/// 0, 3, 5, 8, 11, 13
/// ```
pub struct CsrMatrix {
    /// Defines the symmetry and storage: lower-triangular, upper-triangular, full-matrix
    ///
    /// **Note:** `None` means unsymmetric matrix or unspecified symmetry,
    /// where the storage is automatically `Full`.
    pub symmetry: Option<Symmetry>,

    /// Holds the number of rows (must fit i32)
    pub nrow: usize,

    /// Holds the number of columns (must fit i32)
    pub ncol: usize,

    /// Defines the row pointers array with size = nrow + 1
    pub row_pointers: Vec<i32>,

    /// Defines the column indices array with size = nnz (number of non-zeros)
    pub col_indices: Vec<i32>,

    /// Defines the values array with size = nnz (number of non-zeros)
    pub values: Vec<f64>,
}

impl CsrMatrix {
    /// Validates the dimension of the arrays in the CSR matrix
    ///
    /// The following conditions must be satisfied:
    ///
    /// ```text
    /// nrow ≥ 1
    /// ncol ≥ 1
    /// nnz = row_pointers[nrow] ≥ 1
    /// row_pointers.len() == nrow + 1
    /// col_indices.len() == nnz
    /// values.len() == nnz
    /// ```
    pub fn validate(&self) -> Result<(), StrError> {
        if self.nrow < 1 {
            return Err("nrow must be ≥ 1");
        }
        if self.ncol < 1 {
            return Err("ncol must be ≥ 1");
        }
        if self.row_pointers.len() != self.nrow + 1 {
            return Err("row_pointers.len() must be = nrow + 1");
        }
        let nnz = self.row_pointers[self.nrow];
        if nnz < 1 {
            return Err("nnz = row_pointers[nrow] must be ≥ 1");
        }
        if self.col_indices.len() != nnz as usize {
            return Err("col_indices.len() must be = nnz");
        }
        if self.values.len() != nnz as usize {
            return Err("values.len() must be = nnz");
        }
        Ok(())
    }

    /// Creates a new CsrMatrix from a CooMatrix
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
    ///     let mut coo = CooMatrix::new(nrow, ncol, nnz, None)?;
    ///     coo.put(0, 0, 1.0)?;
    ///     coo.put(0, 2, 2.0)?;
    ///     coo.put(1, 2, 3.0)?;
    ///     coo.put(2, 0, 4.0)?;
    ///     coo.put(2, 1, 5.0)?;
    ///     coo.put(2, 2, 6.0)?;
    ///
    ///     // convert to CSR matrix
    ///     let csr = CsrMatrix::from(&coo)?;
    ///     let correct_v = &[
    ///         //                               p
    ///         1.0, 2.0, //      i = 0, count = 0, 1
    ///         3.0, //           i = 1, count = 2
    ///         4.0, 5.0, 6.0, // i = 2, count = 3, 4, 5
    ///              //                  count = 6
    ///     ];
    ///     let correct_j = &[
    ///         //                         p
    ///         0, 2, //    i = 0, count = 0, 1
    ///         2, //       i = 1, count = 2
    ///         0, 1, 2, // i = 2, count = 3, 4, 5
    ///            //              count = 6
    ///     ];
    ///     let correct_p = &[0, 2, 3, 6];
    ///
    ///     // check
    ///     assert_eq!(&csr.row_pointers, correct_p);
    ///     assert_eq!(&csr.col_indices, correct_j);
    ///     assert_eq!(&csr.values, correct_v);
    ///     Ok(())
    /// }
    /// ```
    pub fn from(coo: &CooMatrix) -> Result<Self, StrError> {
        // Based on the SciPy code from here:
        //
        // https://github.com/scipy/scipy/blob/main/scipy/sparse/sparsetools/coo.h
        //
        // Notes:
        //
        // * The row and column indices may be unordered
        // * Linear complexity: O(nnz(A) + max(nrow,ncol))

        // access triplet data
        let ai = &coo.indices_i;
        let aj = &coo.indices_j;
        let ax = &coo.values_aij;

        // allocate vectors
        let nrow = coo.nrow;
        let nnz = coo.pos;
        let mut csr = CsrMatrix {
            symmetry: coo.symmetry,
            nrow: coo.nrow,
            ncol: coo.ncol,
            row_pointers: vec![0; nrow + 1],
            col_indices: vec![0; nnz],
            values: vec![0.0; nnz],
        };
        let bp = &mut csr.row_pointers;
        let bj = &mut csr.col_indices;
        let bx = &mut csr.values;

        // compute number of non-zero entries per row of A
        for k in 0..nnz {
            bp[ai[k] as usize] += 1;
        }

        // perform the cumulative sum of the nnz per row to get bp[]
        let mut cum_sum = 0;
        for i in 0..nrow {
            let temp = bp[i];
            bp[i] = cum_sum;
            cum_sum += temp;
        }
        bp[nrow] = to_i32(nnz);

        // write aj and ax into bj and bx (will use bp as workspace)
        for k in 0..nnz {
            let row = ai[k];
            let dest = bp[row as usize];
            bj[dest as usize] = aj[k];
            bx[dest as usize] = ax[k];
            bp[row as usize] += 1;
        }

        // fix bp
        let mut last = 0;
        for i in 0..nrow {
            let temp = bp[i];
            bp[i] = last;
            last = temp;
        }

        // sort rows
        let mut temp: Vec<(i32, f64)> = Vec::new();
        for i in 0..nrow {
            let row_start = bp[i];
            let row_end = bp[i + 1];
            temp.resize((row_end - row_start) as usize, (0, 0.0));
            let mut n = 0;
            for jj in row_start..row_end {
                temp[n].0 = bj[jj as usize];
                temp[n].1 = bx[jj as usize];
                n += 1;
            }
            temp.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            n = 0;
            for jj in row_start..row_end {
                bj[jj as usize] = temp[n].0;
                bx[jj as usize] = temp[n].1;
                n += 1;
            }
        }

        // sum duplicates
        let final_nnz = csr_sum_duplicates(nrow, bp, bj, bx);
        bj.resize(final_nnz, 0);
        bx.resize(final_nnz, 0.0);

        // results
        Ok(csr)
    }

    /// Converts this CsrMatrix to a dense matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_sparse::prelude::*;
    /// use russell_sparse::StrError;
    ///
    /// fn main() -> Result<(), StrError> {
    ///     // allocate a square matrix and store as CSR matrix
    ///     // ┌                ┐
    ///     // │  2  3  0  0  0 │
    ///     // │  3  0  4  0  6 │
    ///     // │  0 -1 -3  2  0 │
    ///     // │  0  0  1  0  0 │
    ///     // │  0  4  2  0  1 │
    ///     // └                ┘
    ///     let csr = CsrMatrix {
    ///         symmetry: None,
    ///         nrow: 5,
    ///         ncol: 5,
    ///         row_pointers: vec![0, 2, 5, 8, 9, 12],
    ///         col_indices: vec![
    ///             //                         p
    ///             0, 1, //    i = 0, count = 0, 1
    ///             0, 2, 4, // i = 1, count = 2, 3, 4
    ///             1, 2, 3, // i = 2, count = 5, 6, 7
    ///             2, //       i = 3, count = 8
    ///             1, 2, 4, // i = 4, count = 9, 10, 11
    ///                //              count = 12
    ///         ],
    ///         values: vec![
    ///             //                                 p
    ///             2.0, 3.0, //        i = 0, count = 0, 1
    ///             3.0, 4.0, 6.0, //   i = 1, count = 2, 3, 4
    ///             -1.0, -3.0, 2.0, // i = 2, count = 5, 6, 7
    ///             1.0, //             i = 3, count = 8
    ///             4.0, 2.0, 1.0, //   i = 4, count = 9, 10, 11
    ///                  //                    count = 12
    ///         ],
    ///     };
    ///
    ///     // covert to dense
    ///     let a = csr.as_matrix()?;
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

    /// Converts this CsrMatrix to a dense matrix
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
    ///     // allocate a square matrix and store as CSR matrix
    ///     // ┌                ┐
    ///     // │  2  3  0  0  0 │
    ///     // │  3  0  4  0  6 │
    ///     // │  0 -1 -3  2  0 │
    ///     // │  0  0  1  0  0 │
    ///     // │  0  4  2  0  1 │
    ///     // └                ┘
    ///     let csr = CsrMatrix {
    ///         symmetry: None,
    ///         nrow: 5,
    ///         ncol: 5,
    ///         row_pointers: vec![0, 2, 5, 8, 9, 12],
    ///         col_indices: vec![
    ///             //                         p
    ///             0, 1, //    i = 0, count = 0, 1
    ///             0, 2, 4, // i = 1, count = 2, 3, 4
    ///             1, 2, 3, // i = 2, count = 5, 6, 7
    ///             2, //       i = 3, count = 8
    ///             1, 2, 4, // i = 4, count = 9, 10, 11
    ///                //              count = 12
    ///         ],
    ///         values: vec![
    ///             //                                 p
    ///             2.0, 3.0, //        i = 0, count = 0, 1
    ///             3.0, 4.0, 6.0, //   i = 1, count = 2, 3, 4
    ///             -1.0, -3.0, 2.0, // i = 2, count = 5, 6, 7
    ///             1.0, //             i = 3, count = 8
    ///             4.0, 2.0, 1.0, //   i = 4, count = 9, 10, 11
    ///                  //                    count = 12
    ///         ],
    ///     };
    ///
    ///     // covert to dense
    ///     let mut a = Matrix::new(5, 5);
    ///     csr.to_matrix(&mut a)?;
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
        self.validate()?;
        let (m, n) = a.dims();
        if m != self.nrow || n != self.ncol {
            return Err("wrong matrix dimensions");
        }
        let mirror_required = match self.symmetry {
            Some(sym) => sym.triangular(),
            None => false,
        };
        a.fill(0.0);
        for i in 0..self.nrow {
            for p in self.row_pointers[i]..self.row_pointers[i + 1] {
                let j = self.col_indices[p as usize] as usize;
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
        self.validate()?;
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
        for i in 0..self.nrow {
            for p in self.row_pointers[i]..self.row_pointers[i + 1] {
                let j = self.col_indices[p as usize] as usize;
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

/// brief Sums duplicate column entries in each row of a CSR matrix
///
/// Returns The final number of non-zeros (nnz) after duplicates have been handled
fn csr_sum_duplicates(nrow: usize, ap: &mut [i32], aj: &mut [i32], ax: &mut [f64]) -> usize {
    // Based on the SciPy code from here:
    //
    // https://github.com/scipy/scipy/blob/main/scipy/sparse/sparsetools/csr.h
    //
    // * ap[n_row+1] -- row pointer
    // * aj[nnz(A)]  -- column indices
    // * ax[nnz(A)]  -- non-zeros
    // * The column indices within each row must be sorted
    // * Explicit zeros are retained
    // * ap, aj, and ax will be modified in place

    let mut nnz: i32 = 0;
    let mut row_end = 0;
    for i in 0..nrow {
        let mut jj = row_end;
        row_end = ap[i + 1];
        while jj < row_end {
            let j = aj[jj as usize];
            let mut x = ax[jj as usize];
            jj += 1;
            while jj < row_end && aj[jj as usize] == j {
                x += ax[jj as usize];
                jj += 1;
            }
            aj[nnz as usize] = j;
            ax[nnz as usize] = x;
            nnz += 1;
        }
        ap[i + 1] = nnz;
    }
    nnz as usize
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::CsrMatrix;
    use crate::{CooMatrix, Storage, Symmetry};
    use russell_chk::vec_approx_eq;
    use russell_lab::Matrix;

    #[test]
    fn csr_matrix_first_triplet_with_shuffled_entries() {
        //  1  -1   .  -3   .
        // -2   5   .   .   .
        //  .   .   4   6   4
        // -4   .   2   7   .
        //  .   8   .   .  -5
        // first triplet with shuffled entries
        let mut coo = CooMatrix::new(5, 5, 13, None).unwrap();
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
        let csr = CsrMatrix::from(&coo).unwrap();
        // solution
        let correct_p = vec![0, 3, 5, 8, 11, 13];
        let correct_j = vec![
            0, 1, 3, /**/ 0, 1, /**/ 2, 3, 4, /**/ 0, 2, 3, /**/ 1, 4,
        ];
        let correct_x = vec![1.0, -1.0, -3.0, -2.0, 5.0, 4.0, 6.0, 4.0, -4.0, 2.0, 7.0, 8.0, -5.0];
        assert_eq!(&csr.row_pointers, &correct_p);
        assert_eq!(&csr.col_indices, &correct_j);
        vec_approx_eq(&csr.values, &correct_x, 1e-15);
    }

    #[test]
    fn csr_matrix_small_triplet_with_shuffled_entries() {
        // 1  2  .  .  .
        // 3  4  .  .  .
        // .  .  5  6  .
        // .  .  7  8  .
        // .  .  .  .  9
        // small triplet with shuffled entries
        let mut coo = CooMatrix::new(5, 5, 9, None).unwrap();
        coo.put(4, 4, 9.0).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(1, 0, 3.0).unwrap();
        coo.put(2, 2, 5.0).unwrap();
        coo.put(2, 3, 6.0).unwrap();
        coo.put(0, 1, 2.0).unwrap();
        coo.put(3, 2, 7.0).unwrap();
        coo.put(1, 1, 4.0).unwrap();
        coo.put(3, 3, 8.0).unwrap();
        let csr = CsrMatrix::from(&coo).unwrap();
        // solution
        let correct_p = vec![0, 2, 4, 6, 8, 9];
        let correct_j = vec![0, 1, 0, 1, 2, 3, 2, 3, 4];
        let correct_x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        assert_eq!(&csr.row_pointers, &correct_p);
        assert_eq!(&csr.col_indices, &correct_j);
        vec_approx_eq(&csr.values, &correct_x, 1e-15);
    }

    #[test]
    fn csr_matrix_small_triplet_with_duplicates() {
        // 1  2  .  .  .
        // 3  4  .  .  .
        // .  .  5  6  .
        // .  .  7  8  .
        // .  .  .  .  9
        // with duplicates
        let mut coo = CooMatrix::new(5, 5, 11, None).unwrap();
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
        let csr = CsrMatrix::from(&coo).unwrap();
        // solution
        let correct_p = vec![0, 2, 4, 6, 8, 9];
        let correct_j = vec![0, 1, 0, 1, 2, 3, 2, 3, 4];
        let correct_x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        assert_eq!(&csr.row_pointers, &correct_p);
        assert_eq!(&csr.col_indices, &correct_j);
        vec_approx_eq(&csr.values, &correct_x, 1e-15);
    }

    #[test]
    fn csr_matrix_upper_triangular_with_ordered_entries() {
        //  9.00  1.5   6.0  0.750   3.0
        //  1.50  0.5   0.0  0.000   0.0
        //  6.00  0.0  12.0  0.000   0.0
        //  0.75  0.0   0.0  0.625   0.0
        //  3.00  0.0   0.0  0.000  16.0
        // upper triangular with ordered entries
        let sym = Some(Symmetry::General(Storage::Upper));
        let mut coo = CooMatrix::new(5, 5, 9, sym).unwrap();
        coo.put(0, 0, 9.0).unwrap();
        coo.put(0, 1, 1.5).unwrap();
        coo.put(1, 1, 0.5).unwrap();
        coo.put(0, 2, 6.0).unwrap();
        coo.put(2, 2, 12.0).unwrap();
        coo.put(0, 3, 0.75).unwrap();
        coo.put(3, 3, 0.625).unwrap();
        coo.put(0, 4, 3.0).unwrap();
        coo.put(4, 4, 16.0).unwrap();
        let csr = CsrMatrix::from(&coo).unwrap();
        // solution
        let correct_p = vec![0, 5, 6, 7, 8, 9];
        let correct_j = vec![0, 1, 2, 3, 4, 1, 2, 3, 4];
        let correct_x = vec![9.0, 1.5, 6.0, 0.75, 3.0, 0.5, 12.0, 0.625, 16.0];
        assert_eq!(&csr.row_pointers, &correct_p);
        assert_eq!(&csr.col_indices, &correct_j);
        vec_approx_eq(&csr.values, &correct_x, 1e-15);
    }

    #[test]
    fn csr_matrix_upper_triangular_with_shuffled_entries() {
        //  9.00  1.5   6.0  0.750   3.0
        //  1.50  0.5   0.0  0.000   0.0
        //  6.00  0.0  12.0  0.000   0.0
        //  0.75  0.0   0.0  0.625   0.0
        //  3.00  0.0   0.0  0.000  16.0
        // upper triangular with shuffled entries
        let sym = Some(Symmetry::General(Storage::Upper));
        let mut coo = CooMatrix::new(5, 5, 9, sym).unwrap();
        coo.put(2, 2, 12.0).unwrap();
        coo.put(0, 0, 9.0).unwrap();
        coo.put(3, 3, 0.625).unwrap();
        coo.put(0, 1, 1.5).unwrap();
        coo.put(0, 2, 6.0).unwrap();
        coo.put(4, 4, 16.0).unwrap();
        coo.put(0, 3, 0.75).unwrap();
        coo.put(1, 1, 0.5).unwrap();
        coo.put(0, 4, 3.0).unwrap();
        let csr = CsrMatrix::from(&coo).unwrap();
        // solution
        let correct_p = vec![0, 5, 6, 7, 8, 9];
        let correct_j = vec![0, 1, 2, 3, 4, 1, 2, 3, 4];
        let correct_x = vec![9.0, 1.5, 6.0, 0.75, 3.0, 0.5, 12.0, 0.625, 16.0];
        assert_eq!(&csr.row_pointers, &correct_p);
        assert_eq!(&csr.col_indices, &correct_j);
        vec_approx_eq(&csr.values, &correct_x, 1e-15);
    }

    #[test]
    fn csr_matrix_upper_triangular_with_diagonal_entries_first() {
        //  9.00  1.5   6.0  0.750   3.0
        //  1.50  0.5   0.0  0.000   0.0
        //  6.00  0.0  12.0  0.000   0.0
        //  0.75  0.0   0.0  0.625   0.0
        //  3.00  0.0   0.0  0.000  16.0
        // upper triangular with diagonal entries being set first
        let sym = Some(Symmetry::General(Storage::Upper));
        let mut coo = CooMatrix::new(5, 5, 9, sym).unwrap();
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
        let csr = CsrMatrix::from(&coo).unwrap();
        // solution
        let correct_p = vec![0, 5, 6, 7, 8, 9];
        let correct_j = vec![0, 1, 2, 3, 4, 1, 2, 3, 4];
        let correct_x = vec![9.0, 1.5, 6.0, 0.75, 3.0, 0.5, 12.0, 0.625, 16.0];
        assert_eq!(&csr.row_pointers, &correct_p);
        assert_eq!(&csr.col_indices, &correct_j);
        vec_approx_eq(&csr.values, &correct_x, 1e-15);
    }

    #[test]
    fn csr_matrix_lower_triangular_with_ordered_entries() {
        //  9.00  1.5   6.0  0.750   3.0
        //  1.50  0.5   0.0  0.000   0.0
        //  6.00  0.0  12.0  0.000   0.0
        //  0.75  0.0   0.0  0.625   0.0
        //  3.00  0.0   0.0  0.000  16.0
        // lower diagonal with ordered entries
        let sym = Some(Symmetry::General(Storage::Lower));
        let mut coo = CooMatrix::new(5, 5, 9, sym).unwrap();
        coo.put(0, 0, 9.0).unwrap();
        coo.put(1, 0, 1.5).unwrap();
        coo.put(1, 1, 0.5).unwrap();
        coo.put(2, 0, 6.0).unwrap();
        coo.put(2, 2, 12.0).unwrap();
        coo.put(3, 0, 0.75).unwrap();
        coo.put(3, 3, 0.625).unwrap();
        coo.put(4, 0, 3.0).unwrap();
        coo.put(4, 4, 16.0).unwrap();
        let csr = CsrMatrix::from(&coo).unwrap();
        // solution
        let correct_p = vec![0, 1, 3, 5, 7, 9];
        let correct_j = vec![0, 0, 1, 0, 2, 0, 3, 0, 4];
        let correct_x = vec![9.0, 1.5, 0.5, 6.0, 12.0, 0.75, 0.625, 3.0, 16.0];
        assert_eq!(&csr.row_pointers, &correct_p);
        assert_eq!(&csr.col_indices, &correct_j);
        vec_approx_eq(&csr.values, &correct_x, 1e-15);
    }

    #[test]
    fn csr_matrix_lower_triangular_with_diagonal_first() {
        //  9.00  1.5   6.0  0.750   3.0
        //  1.50  0.5   0.0  0.000   0.0
        //  6.00  0.0  12.0  0.000   0.0
        //  0.75  0.0   0.0  0.625   0.0
        //  3.00  0.0   0.0  0.000  16.0
        // lower triangular with diagonal entries being set first
        let sym = Some(Symmetry::General(Storage::Lower));
        let mut coo = CooMatrix::new(5, 5, 9, sym).unwrap();
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
        let csr = CsrMatrix::from(&coo).unwrap();
        // solution
        let correct_p = vec![0, 1, 3, 5, 7, 9];
        let correct_j = vec![0, 0, 1, 0, 2, 0, 3, 0, 4];
        let correct_x = vec![9.0, 1.5, 0.5, 6.0, 12.0, 0.75, 0.625, 3.0, 16.0];
        assert_eq!(&csr.row_pointers, &correct_p);
        assert_eq!(&csr.col_indices, &correct_j);
        vec_approx_eq(&csr.values, &correct_x, 1e-15);
    }

    #[test]
    fn to_matrix_fails_on_wrong_dims() {
        // 10.0 20.0
        let csr = CsrMatrix {
            symmetry: None,
            nrow: 1,
            ncol: 2,
            row_pointers: vec![0, 2],
            col_indices: vec![0, 1],
            values: vec![10.0, 20.0],
        };
        let mut a_3x1 = Matrix::new(3, 1);
        let mut a_1x3 = Matrix::new(1, 3);
        assert_eq!(csr.to_matrix(&mut a_3x1), Err("wrong matrix dimensions"));
        assert_eq!(csr.to_matrix(&mut a_1x3), Err("wrong matrix dimensions"));
    }

    #[test]
    fn to_matrix_and_as_matrix_work() {
        // 10.0 20.0       << (1 x 2) matrix
        let csr = CsrMatrix {
            symmetry: None,
            nrow: 1,
            ncol: 2,
            row_pointers: vec![0, 2],
            col_indices: vec![0, 1],
            values: vec![10.0, 20.0],
        };
        let mut a = Matrix::new(1, 2);
        csr.to_matrix(&mut a).unwrap();
        let correct = "┌       ┐\n\
                       │ 10 20 │\n\
                       └       ┘";
        assert_eq!(format!("{}", a), correct);

        let csr = CsrMatrix {
            symmetry: None,
            nrow: 5,
            ncol: 5,
            row_pointers: vec![0, 2, 5, 8, 9, 12],
            col_indices: vec![
                //                         p
                0, 1, //    i = 0, count = 0, 1
                0, 2, 4, // i = 1, count = 2, 3, 4
                1, 2, 3, // i = 2, count = 5, 6, 7
                2, //       i = 3, count = 8
                1, 2, 4, // i = 4, count = 9, 10, 11
                   //              count = 12
            ],
            values: vec![
                //                                 p
                2.0, 3.0, //        i = 0, count = 0, 1
                3.0, 4.0, 6.0, //   i = 1, count = 2, 3, 4
                -1.0, -3.0, 2.0, // i = 2, count = 5, 6, 7
                1.0, //             i = 3, count = 8
                4.0, 2.0, 1.0, //   i = 4, count = 9, 10, 11
                     //                    count = 12
            ],
        };

        // covert to dense
        let mut a = Matrix::new(5, 5);
        csr.to_matrix(&mut a).unwrap();
        let correct = "┌                ┐\n\
                       │  2  3  0  0  0 │\n\
                       │  3  0  4  0  6 │\n\
                       │  0 -1 -3  2  0 │\n\
                       │  0  0  1  0  0 │\n\
                       │  0  4  2  0  1 │\n\
                       └                ┘";
        assert_eq!(format!("{}", a), correct);

        // use as_matrix
        let b = csr.as_matrix().unwrap();
        assert_eq!(format!("{}", b), correct);
    }

    #[test]
    fn to_matrix_upper_works() {
        let csr = CsrMatrix {
            symmetry: Some(Symmetry::General(Storage::Upper)),
            nrow: 5,
            ncol: 5,
            row_pointers: vec![0, 5, 6, 7, 8, 9],
            col_indices: vec![0, 1, 2, 3, 4, 1, 2, 3, 4],
            values: vec![9.0, 1.5, 6.0, 0.75, 3.0, 0.5, 12.0, 0.625, 16.0],
        };
        let a = csr.as_matrix().unwrap();
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
        let csr = CsrMatrix {
            symmetry: Some(Symmetry::General(Storage::Lower)),
            nrow: 5,
            ncol: 5,
            row_pointers: vec![0, 1, 3, 5, 7, 9],
            col_indices: vec![0, 0, 1, 0, 2, 0, 3, 0, 4],
            values: vec![9.0, 1.5, 0.5, 6.0, 12.0, 0.75, 0.625, 3.0, 16.0],
        };
        let a = csr.as_matrix().unwrap();
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
