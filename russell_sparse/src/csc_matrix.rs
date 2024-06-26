use super::{to_i32, NumCooMatrix, NumCsrMatrix, Sym};
use crate::StrError;
use num_traits::{Num, NumCast};
use russell_lab::{NumMatrix, NumVector};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::ops::{AddAssign, MulAssign};

/// Holds the arrays needed for a CSC (compressed sparse column) matrix
///
/// # Examples (from UMFPACK QuickStart.pdf)
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
///
/// **Note:** The number of non-zero values is `nnz = col_pointers[ncol]`
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NumCscMatrix<T>
where
    T: AddAssign + MulAssign + Num + NumCast + Copy + DeserializeOwned + Serialize,
{
    /// Indicates whether the matrix is symmetric or not. If symmetric, indicates the representation too.
    pub(crate) symmetric: Sym,

    /// Holds the number of rows (must fit i32)
    pub(crate) nrow: usize,

    /// Holds the number of columns (must fit i32)
    pub(crate) ncol: usize,

    /// Defines the column pointers array with size = ncol + 1
    ///
    /// ```text
    /// col_pointers.len() = ncol + 1
    /// nnz = col_pointers[ncol]
    /// ```
    pub(crate) col_pointers: Vec<i32>,

    /// Defines the row indices array with size = nnz_dup (number of non-zeros with duplicates)
    ///
    /// ```text
    /// nnz_dup ≥ nnz
    /// row_indices.len() = nnz_dup
    /// ```
    pub(crate) row_indices: Vec<i32>,

    /// Defines the values array with size = nnz_dup (number of non-zeros with duplicates)
    ///
    /// ```text
    /// nnz_dup ≥ nnz
    /// values.len() = nnz_dup
    /// ```
    #[serde(bound(deserialize = "Vec<T>: Deserialize<'de>"))]
    pub(crate) values: Vec<T>,

    /// Temporary row form (for COO to CSC conversion)
    #[serde(skip)]
    temp_rp: Vec<i32>,

    /// Temporary row form (for COO to CSC conversion)
    #[serde(skip)]
    temp_rj: Vec<i32>,

    /// Temporary row form (for COO to CSC conversion)
    #[serde(skip)]
    temp_rx: Vec<T>,

    /// Temporary row count (for COO to CSC conversion)
    #[serde(skip)]
    temp_rc: Vec<usize>,

    /// Temporary workspace (for COO to CSC conversion)
    #[serde(skip)]
    temp_w: Vec<i32>,
}

impl<T> NumCscMatrix<T>
where
    T: AddAssign + MulAssign + Num + NumCast + Copy + DeserializeOwned + Serialize,
{
    /// Creates a new CSC matrix from data arrays
    ///
    /// **Note:** The column pointers and row indices must be **sorted** in ascending order.
    ///
    /// # Input
    ///
    /// * `nrow` -- (≥ 1) number of rows
    /// * `ncol` -- (≥ 1) number of columns
    /// * `col_pointers` -- (len = ncol + 1) columns pointers with the last entry corresponding
    ///   to the number of non-zero values (sorted)
    /// * `row_indices` -- (len = nnz) row indices (sorted)
    /// * `values` -- the non-zero components of the matrix
    /// * `symmetric` -- indicates whether the matrix is symmetric or not.
    ///   If symmetric, indicates the representation too.
    ///
    /// The following conditions must be satisfied (nnz is the number of non-zeros
    /// and nnz_dup is the number of non-zeros with possible duplicates):
    ///
    /// ```text
    /// nrow ≥ 1
    /// ncol ≥ 1
    /// col_pointers.len() = ncol + 1
    /// row_indices.len() = nnz_dup
    /// values.len() = nnz_dup
    /// nnz = col_pointers[ncol] ≥ 1
    /// nnz_dup ≥ nnz
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_sparse::prelude::*;
    /// use russell_sparse::StrError;
    ///
    /// fn main() -> Result<(), StrError> {
    ///     // allocate a square matrix and store as CSC matrix
    ///     //  2  3  .  .  .
    ///     //  3  .  4  .  6
    ///     //  . -1 -3  2  .
    ///     //  .  .  1  .  .
    ///     //  .  4  2  .  1
    ///     let nrow = 5;
    ///     let ncol = 5;
    ///     let col_pointers = vec![0, 2, 5, 9, 10, 12];
    ///     let row_indices = vec![
    ///         //                             p
    ///         0, 1, //       j = 0, count =  0, 1,
    ///         0, 2, 4, //    j = 1, count =  2, 3, 4,
    ///         1, 2, 3, 4, // j = 2, count =  5, 6, 7, 8,
    ///         2, //          j = 3, count =  9,
    ///         1, 4, //       j = 4, count = 10, 11,
    ///            //                         12
    ///     ];
    ///     let values = vec![
    ///         //                                      p
    ///         2.0, 3.0, //            j = 0, count =  0, 1,
    ///         3.0, -1.0, 4.0, //      j = 1, count =  2, 3, 4,
    ///         4.0, -3.0, 1.0, 2.0, // j = 2, count =  5, 6, 7, 8,
    ///         2.0, //                 j = 3, count =  9,
    ///         6.0, 1.0, //            j = 4, count = 10, 11,
    ///              //                                12
    ///     ];
    ///     let csc = CscMatrix::new(nrow, ncol, col_pointers, row_indices, values, Sym::No)?;
    ///
    ///     // covert to dense
    ///     let a = csc.as_dense();
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
    pub fn new(
        nrow: usize,
        ncol: usize,
        col_pointers: Vec<i32>,
        row_indices: Vec<i32>,
        values: Vec<T>,
        symmetric: Sym,
    ) -> Result<Self, StrError> {
        if nrow < 1 {
            return Err("nrow must be ≥ 1");
        }
        if ncol < 1 {
            return Err("ncol must be ≥ 1");
        }
        if col_pointers.len() != ncol + 1 {
            return Err("col_pointers.len() must be = ncol + 1");
        }
        let nnz = col_pointers[ncol];
        if nnz < 1 {
            return Err("nnz = col_pointers[ncol] must be ≥ 1");
        }
        if row_indices.len() < nnz as usize {
            return Err("row_indices.len() must be ≥ nnz");
        }
        if values.len() < nnz as usize {
            return Err("values.len() must be ≥ nnz");
        }
        let m = to_i32(nrow);
        for j in 0..ncol {
            if col_pointers[j] < 0 {
                return Err("col pointers must be ≥ 0");
            }
            if col_pointers[j] > col_pointers[j + 1] {
                return Err("col pointers must be sorted in ascending order");
            }
            let start = col_pointers[j] as usize;
            let end = col_pointers[j + 1] as usize;
            for p in start..end {
                let i = row_indices[p];
                if i < 0 {
                    return Err("row indices must be ≥ 0");
                }
                if i >= m {
                    return Err("row indices must be < nrow");
                }
                if p > start {
                    if row_indices[p - 1] > row_indices[p] {
                        return Err("row indices must be sorted in ascending order (within their column)");
                    }
                }
            }
        }
        Ok(NumCscMatrix {
            symmetric,
            nrow,
            ncol,
            col_pointers,
            row_indices,
            values,
            temp_rp: Vec::new(),
            temp_rj: Vec::new(),
            temp_rx: Vec::new(),
            temp_rc: Vec::new(),
            temp_w: Vec::new(),
        })
    }

    /// Creates a new CSC matrix from a COO matrix
    ///
    /// **Note:** The final nnz may be smaller than the initial nnz because duplicates
    /// may have been summed up. The final nnz is available as `nnz = col_pointers[ncol]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_sparse::prelude::*;
    /// use russell_sparse::StrError;
    ///
    /// fn main() -> Result<(), StrError> {
    ///     // allocate a square matrix and store as COO matrix
    ///     //  2  3  .  .  .
    ///     //  3  .  4  .  6
    ///     //  . -1 -3  2  .
    ///     //  .  .  1  .  .
    ///     //  .  4  2  .  1
    ///     let (nrow, ncol, nnz) = (5, 5, 13);
    ///     let mut coo = CooMatrix::new(nrow, ncol, nnz, Sym::No)?;
    ///     coo.put(0, 0, 1.0)?; // << (0, 0, a00/2) duplicate
    ///     coo.put(0, 0, 1.0)?; // << (0, 0, a00/2) duplicate
    ///     coo.put(1, 0, 3.0)?;
    ///     coo.put(0, 1, 3.0)?;
    ///     coo.put(2, 1, -1.0)?;
    ///     coo.put(4, 1, 4.0)?;
    ///     coo.put(1, 2, 4.0)?;
    ///     coo.put(2, 2, -3.0)?;
    ///     coo.put(3, 2, 1.0)?;
    ///     coo.put(4, 2, 2.0)?;
    ///     coo.put(2, 3, 2.0)?;
    ///     coo.put(1, 4, 6.0)?;
    ///     coo.put(4, 4, 1.0)?;
    ///
    ///     // convert to CSC matrix
    ///     let csc = CscMatrix::from_coo(&coo)?;
    ///     let correct_pp = vec![0, 2, 5, 9, 10, 12];
    ///     let correct_ii = vec![
    ///         //                             p
    ///         0, 1, //       j = 0, count =  0, 1,
    ///         0, 2, 4, //    j = 1, count =  2, 3, 4,
    ///         1, 2, 3, 4, // j = 2, count =  5, 6, 7, 8,
    ///         2, //          j = 3, count =  9,
    ///         1, 4, //       j = 4, count = 10, 11,
    ///            //                         12
    ///     ];
    ///     let correct_vv = vec![
    ///         //                                      p
    ///         2.0, 3.0, //            j = 0, count =  0, 1,
    ///         3.0, -1.0, 4.0, //      j = 1, count =  2, 3, 4,
    ///         4.0, -3.0, 1.0, 2.0, // j = 2, count =  5, 6, 7, 8,
    ///         2.0, //                 j = 3, count =  9,
    ///         6.0, 1.0, //            j = 4, count = 10, 11,
    ///              //                                12
    ///     ];
    ///
    ///     // check
    ///     let pp = csc.get_col_pointers();
    ///     let ii = csc.get_row_indices();
    ///     let vv = csc.get_values();
    ///     let final_nnz = pp[nrow] as usize;
    ///     assert_eq!(final_nnz, 12);
    ///     assert_eq!(pp, correct_pp);
    ///     assert_eq!(&ii[0..final_nnz], correct_ii);
    ///     assert_eq!(&vv[0..final_nnz], correct_vv);
    ///     Ok(())
    /// }
    /// ```
    pub fn from_coo(coo: &NumCooMatrix<T>) -> Result<Self, StrError> {
        if coo.nnz < 1 {
            return Err("COO to CSC requires nnz > 0");
        }
        let mut csc = NumCscMatrix {
            symmetric: coo.symmetric,
            nrow: coo.nrow,
            ncol: coo.ncol,
            col_pointers: vec![0; coo.ncol + 1],
            row_indices: vec![0; coo.nnz],
            values: vec![T::zero(); coo.nnz],
            temp_rp: Vec::new(),
            temp_rj: Vec::new(),
            temp_rx: Vec::new(),
            temp_rc: Vec::new(),
            temp_w: Vec::new(),
        };
        csc.update_from_coo(coo).unwrap();
        Ok(csc)
    }

    /// Updates this CSC matrix from a COO matrix with a compatible structure
    ///
    /// **Note:** The COO matrix must match the symmetric type, nrow, and ncol values of the CSC matrix.
    /// Also, the `nnz` (may include duplicates) of the COO matrix must match `row_indices.len() = values.len()`.
    ///
    /// **Note:** The final nnz may be smaller than the initial nnz because duplicates
    /// may have been summed up. The final nnz is available as `nnz = col_pointers[ncol]`.
    pub fn update_from_coo(&mut self, coo: &NumCooMatrix<T>) -> Result<(), StrError> {
        // check dimensions
        if coo.symmetric != self.symmetric {
            return Err("coo.symmetric must be equal to csc.symmetric");
        }
        if coo.nrow != self.nrow {
            return Err("coo.nrow must be equal to csc.nrow");
        }
        if coo.ncol != self.ncol {
            return Err("coo.ncol must be equal to csc.ncol");
        }
        if coo.nnz != self.values.len() {
            return Err("coo.nnz must be equal to nnz(dup) = csc.row_indices.len() = csc.values.len()");
        }

        // Based on Prof Tim Davis' UMFPACK::UMF_triplet_map_x (umf_triplet.c)

        // constants
        let nrow = coo.nrow;
        let ncol = coo.ncol;
        let nnz = coo.nnz;
        let ndim = usize::max(nrow, ncol);

        // access the triplet data
        let ai = &coo.indices_i;
        let aj = &coo.indices_j;
        let ax = &coo.values;

        // access the CSC data
        let bp = &mut self.col_pointers;
        let bi = &mut self.row_indices;
        let bx = &mut self.values;

        // allocate workspaces and get an access to them
        if self.temp_w.len() == 0 {
            self.temp_rp = vec![0_i32; nrow + 1]; // temporary row form
            self.temp_rj = vec![0_i32; nnz]; // temporary row form
            self.temp_rx = vec![T::zero(); nnz]; // temporary row form
            self.temp_rc = vec![0_usize; nrow]; // temporary row count
            self.temp_w = vec![0_i32; ndim]; // temporary workspace
        } else {
            for i in 0..nrow {
                self.temp_w[i] = 0;
            }
        }
        let rp = &mut self.temp_rp;
        let rj = &mut self.temp_rj;
        let rx = &mut self.temp_rx;
        let rc = &mut self.temp_rc;
        let w = &mut self.temp_w;

        // count the entries in each row (also counting duplicates)
        // use w as workspace for row counts (including duplicates)
        for k in 0..nnz {
            let i = ai[k] as usize;
            w[i] += 1;
        }

        // compute the row pointers (save them in workspace)
        rp[0] = 0;
        for i in 0..nrow {
            rp[i + 1] = rp[i] + w[i];
            w[i] = rp[i];
        }

        // construct the row form (with unsorted values)
        for k in 0..nnz {
            let i = ai[k] as usize;
            let p = w[i] as usize;
            rj[p] = aj[k];
            rx[p] = ax[k];
            w[i] += 1; // w[i] is advanced to the start of row i+1
        }

        // sum duplicates. w[j] will hold the position in rj and rx of aij
        const EMPTY: i32 = -1;
        for j in 0..ncol {
            w[j] = EMPTY;
        }
        for i in 0..nrow {
            let p1 = rp[i] as usize;
            let p2 = rp[i + 1] as usize;
            let mut dest = p1;
            // w[j] < p1 for all columns j (note that rj and rx are stored in row oriented order)
            for p in p1..p2 {
                let j = rj[p] as usize;
                if w[j] >= p1 as i32 {
                    // j is already in row i, position pj
                    let pj = w[j] as usize;
                    let x = rx[p];
                    rx[pj] += x; // sum the entry
                } else {
                    // keep the entry
                    w[j] = dest as i32;
                    if dest != p {
                        // move is not needed
                        rj[dest] = j as i32;
                        rx[dest] = rx[p];
                    }
                    dest += 1;
                }
            }
            rc[i] = dest - p1;
        }

        // count the entries in each column
        for j in 0..ncol {
            w[j] = 0; // use the workspace for column counts
        }
        for i in 0..nrow {
            let p1 = rp[i] as usize;
            let p2 = p1 + rc[i];
            for p in p1..p2 {
                let j = rj[p] as usize;
                w[j] += 1;
            }
        }

        // create the column pointers
        bp[0] = 0;
        for j in 0..ncol {
            bp[j + 1] = bp[j] + w[j];
        }
        for j in 0..ncol {
            w[j] = bp[j];
        }

        // construct the column form
        for i in 0..nrow {
            let p1 = rp[i] as usize;
            let p2 = p1 + rc[i];
            for p in p1..p2 {
                let j = rj[p] as usize;
                let cp = w[j] as usize;
                bi[cp] = i as i32;
                bx[cp] = rx[p];
                w[j] += 1;
            }
        }
        Ok(())
    }

    /// Creates a new CSC matrix from a CSR matrix
    pub fn from_csr(csr: &NumCsrMatrix<T>) -> Result<Self, StrError> {
        // Based on the SciPy code (csr_tocsc) from here:
        //
        // https://github.com/scipy/scipy/blob/main/scipy/sparse/sparsetools/csr.h
        //
        // Notes:
        //
        // * Linear complexity: O(nnz(A) + max(nrow, ncol))
        // * Upgrading i32 to usize is OK (the opposite is not OK => use to_i32)

        // check and read in the dimensions
        let nrow = csr.nrow as usize;
        let ncol = csr.ncol as usize;
        let nnz = csr.row_pointers[nrow] as usize;

        // access the CSR data
        let ap = &csr.row_pointers;
        let aj = &csr.col_indices;
        let ax = &csr.values;

        // allocate the CSC arrays
        let mut csc = NumCscMatrix {
            symmetric: csr.symmetric,
            nrow: csr.nrow,
            ncol: csr.ncol,
            col_pointers: vec![0; ncol + 1],
            row_indices: vec![0; nnz],
            values: vec![T::zero(); nnz],
            temp_rp: Vec::new(),
            temp_rj: Vec::new(),
            temp_rx: Vec::new(),
            temp_rc: Vec::new(),
            temp_w: Vec::new(),
        };

        // access the CSC data
        let bp = &mut csc.col_pointers;
        let bi = &mut csc.row_indices;
        let bx = &mut csc.values;

        // compute the number of non-zero entries per column of A
        // bp.fill(0); // <<<< not needed because the array is zeroed already
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
        bp[ncol] = to_i32(nnz);

        // write aj and ax into bi and bx (will use bp as workspace)
        for i in 0..nrow {
            for p in ap[i]..ap[i + 1] {
                let j = aj[p as usize] as usize;
                let dest = bp[j] as usize;
                bi[dest] = to_i32(i);
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
    ///     let nrow = 5;
    ///     let ncol = 5;
    ///     let col_pointers = vec![0, 2, 5, 9, 10, 12];
    ///     let row_indices = vec![
    ///         //                             p
    ///         0, 1, //       j = 0, count =  0, 1,
    ///         0, 2, 4, //    j = 1, count =  2, 3, 4,
    ///         1, 2, 3, 4, // j = 2, count =  5, 6, 7, 8,
    ///         2, //          j = 3, count =  9,
    ///         1, 4, //       j = 4, count = 10, 11,
    ///            //                         12
    ///     ];
    ///     let values = vec![
    ///         //                                      p
    ///         2.0, 3.0, //            j = 0, count =  0, 1,
    ///         3.0, -1.0, 4.0, //      j = 1, count =  2, 3, 4,
    ///         4.0, -3.0, 1.0, 2.0, // j = 2, count =  5, 6, 7, 8,
    ///         2.0, //                 j = 3, count =  9,
    ///         6.0, 1.0, //            j = 4, count = 10, 11,
    ///              //                                12
    ///     ];
    ///     let csc = CscMatrix::new(nrow, ncol, col_pointers, row_indices, values, Sym::No)?;
    ///
    ///     // covert to dense
    ///     let a = csc.as_dense();
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
    pub fn as_dense(&self) -> NumMatrix<T> {
        let mut a = NumMatrix::new(self.nrow, self.ncol);
        self.to_dense(&mut a).unwrap();
        a
    }

    /// Converts this CSC matrix to a dense matrix
    ///
    /// # Input
    ///
    /// * `a` -- where to store the dense matrix; it must be (nrow, ncol)
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
    ///     let nrow = 5;
    ///     let ncol = 5;
    ///     let col_pointers = vec![0, 2, 5, 9, 10, 12];
    ///     let row_indices = vec![
    ///         //                             p
    ///         0, 1, //       j = 0, count =  0, 1,
    ///         0, 2, 4, //    j = 1, count =  2, 3, 4,
    ///         1, 2, 3, 4, // j = 2, count =  5, 6, 7, 8,
    ///         2, //          j = 3, count =  9,
    ///         1, 4, //       j = 4, count = 10, 11,
    ///            //                         12
    ///     ];
    ///     let values = vec![
    ///         //                                      p
    ///         2.0, 3.0, //            j = 0, count =  0, 1,
    ///         3.0, -1.0, 4.0, //      j = 1, count =  2, 3, 4,
    ///         4.0, -3.0, 1.0, 2.0, // j = 2, count =  5, 6, 7, 8,
    ///         2.0, //                 j = 3, count =  9,
    ///         6.0, 1.0, //            j = 4, count = 10, 11,
    ///              //                                12
    ///     ];
    ///     let csc = CscMatrix::new(nrow, ncol, col_pointers, row_indices, values, Sym::No)?;
    ///
    ///     // covert to dense
    ///     let a = csc.as_dense();
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
    pub fn to_dense(&self, a: &mut NumMatrix<T>) -> Result<(), StrError> {
        let (m, n) = a.dims();
        if m != self.nrow || n != self.ncol {
            return Err("wrong matrix dimensions");
        }
        let mirror_required = self.symmetric.triangular();
        a.fill(T::zero());
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
    pub fn mat_vec_mul(&self, v: &mut NumVector<T>, alpha: T, u: &NumVector<T>) -> Result<(), StrError> {
        if u.dim() != self.ncol {
            return Err("u vector is incompatible");
        }
        if v.dim() != self.nrow {
            return Err("v vector is incompatible");
        }
        let mirror_required = self.symmetric.triangular();
        v.fill(T::zero());
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

    /// Returns information about the dimensions and symmetric type
    ///
    /// Returns `(nrow, ncol, nnz, sym)`
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_sparse::prelude::*;
    /// use russell_sparse::StrError;
    ///
    /// fn main() -> Result<(), StrError> {
    ///     // ┌       ┐
    ///     // │ 10 20 │
    ///     // └       ┘
    ///     let col_pointers = vec![0, 1, 2];
    ///     let row_indices = vec![0, 0];
    ///     let values = vec![10.0, 20.0];
    ///     let csc = CscMatrix::new(1, 2,
    ///         col_pointers, row_indices, values, Sym::No)?;
    ///     let (nrow, ncol, nnz, sym) = csc.get_info();
    ///     assert_eq!(nrow, 1);
    ///     assert_eq!(ncol, 2);
    ///     assert_eq!(nnz, 2);
    ///     assert_eq!(sym, Sym::No);
    ///     let a = csc.as_dense();
    ///     let correct = "┌       ┐\n\
    ///                    │ 10 20 │\n\
    ///                    └       ┘";
    ///     assert_eq!(format!("{}", a), correct);
    ///     Ok(())
    /// }
    /// ```
    pub fn get_info(&self) -> (usize, usize, usize, Sym) {
        (
            self.nrow,
            self.ncol,
            self.col_pointers[self.ncol] as usize,
            self.symmetric,
        )
    }

    /// Get an access to the column pointers
    ///
    /// ```text
    /// col_pointers.len() = ncol + 1
    /// ```
    pub fn get_col_pointers(&self) -> &[i32] {
        &self.col_pointers
    }

    /// Get an access to the row indices
    ///
    /// ```text
    /// nnz = col_pointers[ncol]
    /// row_indices.len() == nnz
    /// ```
    pub fn get_row_indices(&self) -> &[i32] {
        let nnz = self.col_pointers[self.ncol] as usize;
        &self.row_indices[..nnz]
    }

    /// Get an access to the values
    ///
    /// ```text
    /// nnz = col_pointers[ncol]
    /// values.len() == nnz
    /// ```
    pub fn get_values(&self) -> &[T] {
        let nnz = self.col_pointers[self.ncol] as usize;
        &self.values[..nnz]
    }

    /// Get a mutable access to the values
    ///
    /// ```text
    /// nnz = col_pointers[ncol]
    /// values.len() == nnz
    /// ```
    ///
    /// Note: the values may be modified externally, but not the pointers or indices.
    pub fn get_values_mut(&mut self) -> &mut [T] {
        let nnz = self.col_pointers[self.ncol] as usize;
        &mut self.values[..nnz]
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::NumCscMatrix;
    use crate::{CooMatrix, Samples, Sym};
    use russell_lab::{
        array_approx_eq, complex_vec_approx_eq, cpx, vec_approx_eq, Complex64, ComplexVector, Matrix, Vector,
    };

    #[test]
    fn new_captures_errors() {
        assert_eq!(
            NumCscMatrix::<f64>::new(0, 1, vec![0], vec![], vec![], Sym::No).err(),
            Some("nrow must be ≥ 1")
        );
        assert_eq!(
            NumCscMatrix::<f64>::new(1, 0, vec![0], vec![], vec![], Sym::No).err(),
            Some("ncol must be ≥ 1")
        );
        assert_eq!(
            NumCscMatrix::<f64>::new(1, 1, vec![0], vec![], vec![], Sym::No).err(),
            Some("col_pointers.len() must be = ncol + 1")
        );
        assert_eq!(
            NumCscMatrix::<f64>::new(1, 1, vec![0, 0], vec![], vec![], Sym::No).err(),
            Some("nnz = col_pointers[ncol] must be ≥ 1")
        );
        assert_eq!(
            NumCscMatrix::<f64>::new(1, 1, vec![0, 1], vec![], vec![], Sym::No).err(),
            Some("row_indices.len() must be ≥ nnz")
        );
        assert_eq!(
            NumCscMatrix::<f64>::new(1, 1, vec![0, 1], vec![0], vec![], Sym::No).err(),
            Some("values.len() must be ≥ nnz")
        );
        assert_eq!(
            NumCscMatrix::<f64>::new(1, 1, vec![-1, 1], vec![0], vec![0.0], Sym::No).err(),
            Some("col pointers must be ≥ 0")
        );
        assert_eq!(
            NumCscMatrix::<f64>::new(1, 1, vec![2, 1], vec![0], vec![0.0], Sym::No).err(),
            Some("col pointers must be sorted in ascending order")
        );
        assert_eq!(
            NumCscMatrix::<f64>::new(1, 1, vec![0, 1], vec![-1], vec![0.0], Sym::No).err(),
            Some("row indices must be ≥ 0")
        );
        assert_eq!(
            NumCscMatrix::<f64>::new(1, 1, vec![0, 1], vec![2], vec![0.0], Sym::No).err(),
            Some("row indices must be < nrow")
        );
        // ┌    ┐
        // │ 10 │
        // │ 20 │
        // └    ┘
        let values = vec![
            10.0, 20.0, // j=0 p=(0),1
        ]; //                  p=(2)
        let row_indices = vec![1, 0]; // << incorrect, should be [0, 1]
        let col_pointers = vec![0, 2];
        assert_eq!(
            NumCscMatrix::<f64>::new(2, 1, col_pointers, row_indices, values, Sym::No).err(),
            Some("row indices must be sorted in ascending order (within their column)")
        );
    }

    #[test]
    fn new_works() {
        let (_, csc_correct, _, _) = Samples::rectangular_1x2(false, false);
        let csc = NumCscMatrix::<f64>::new(1, 2, vec![0, 1, 2], vec![0, 0], vec![10.0, 20.0], Sym::No).unwrap();
        assert_eq!(csc.symmetric, Sym::No);
        assert_eq!(csc.nrow, 1);
        assert_eq!(csc.ncol, 2);
        assert_eq!(&csc.col_pointers, &csc_correct.col_pointers);
        assert_eq!(&csc.row_indices, &csc_correct.row_indices);
        assert_eq!(&csc.values, &csc_correct.values);
    }

    #[test]
    fn from_coo_captures_errors() {
        let coo = CooMatrix::new(1, 1, 1, Sym::No).unwrap();
        assert_eq!(
            NumCscMatrix::<f64>::from_coo(&coo).err(),
            Some("COO to CSC requires nnz > 0")
        );
    }

    #[test]
    fn from_coo_works() {
        for (coo, csc_correct, _, _) in [
            // ┌     ┐
            // │ 123 │
            // └     ┘
            Samples::tiny_1x1(),
            //  1  .  2
            //  .  0  3
            //  4  5  6
            Samples::unsymmetric_3x3(false, false),
            Samples::unsymmetric_3x3(false, true),
            Samples::unsymmetric_3x3(true, false),
            Samples::unsymmetric_3x3(true, true),
            //  2  3  .  .  .
            //  3  .  4  .  6
            //  . -1 -3  2  .
            //  .  .  1  .  .
            //  .  4  2  .  1
            Samples::umfpack_unsymmetric_5x5(),
            //  1  -1   .  -3   .
            // -2   5   .   .   .
            //  .   .   4   6   4
            // -4   .   2   7   .
            //  .   8   .   .  -5
            Samples::mkl_unsymmetric_5x5(),
            // 1  2  .  .  .
            // 3  4  .  .  .
            // .  .  5  6  .
            // .  .  7  8  .
            // .  .  .  .  9
            Samples::block_unsymmetric_5x5(false, false),
            Samples::block_unsymmetric_5x5(false, true),
            Samples::block_unsymmetric_5x5(true, false),
            Samples::block_unsymmetric_5x5(true, true),
            //     9   1.5     6  0.75     3
            //   1.5   0.5     .     .     .
            //     6     .    12     .     .
            //  0.75     .     . 0.625     .
            //     3     .     .     .    16
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
            // ┌       ┐
            // │ 10 20 │
            // └       ┘
            Samples::rectangular_1x2(false, false),
            Samples::rectangular_1x2(false, true),
            Samples::rectangular_1x2(true, false),
            Samples::rectangular_1x2(true, true),
            // ┌               ┐
            // │ 1 . 3 . 5 . 7 │
            // └               ┘
            Samples::rectangular_1x7(),
            // ┌   ┐
            // │ . │
            // │ 2 │
            // │ . │
            // │ 4 │
            // │ . │
            // │ 6 │
            // │ . │
            // └   ┘
            Samples::rectangular_7x1(),
            //   5  -2  .  1
            //  10  -4  .  2
            //  15  -6  .  3
            Samples::rectangular_3x4(),
        ] {
            let csc = NumCscMatrix::<f64>::from_coo(&coo).unwrap();
            assert_eq!(&csc.col_pointers, &csc_correct.col_pointers);
            let nnz = csc.col_pointers[csc.ncol] as usize;
            assert_eq!(&csc.row_indices[0..nnz], &csc_correct.row_indices);
            array_approx_eq(&csc.values[0..nnz], &csc_correct.values, 1e-15);
        }
    }

    #[test]
    #[rustfmt::skip]
    fn update_from_coo_captures_errors() {
        let (coo, _, _, _) = Samples::rectangular_1x2(false, false, );
        let mut csc = NumCscMatrix::<f64>::from_coo(&coo).unwrap();
        let yes = Sym::YesLower;
        let no = Sym::No;
        assert_eq!(csc.update_from_coo(&CooMatrix { symmetric: yes,  nrow: 1, ncol: 2, nnz: 1, max_nnz: 1, indices_i: vec![0], indices_j: vec![0], values: vec![0.0] }).err(), Some("coo.symmetric must be equal to csc.symmetric"));
        assert_eq!(csc.update_from_coo(&CooMatrix { symmetric: no, nrow: 2, ncol: 2, nnz: 1, max_nnz: 1, indices_i: vec![0], indices_j: vec![0], values: vec![0.0] }).err(), Some("coo.nrow must be equal to csc.nrow"));
        assert_eq!(csc.update_from_coo(&CooMatrix { symmetric: no, nrow: 1, ncol: 1, nnz: 1, max_nnz: 1, indices_i: vec![0], indices_j: vec![0], values: vec![0.0] }).err(), Some("coo.ncol must be equal to csc.ncol"));
        assert_eq!(csc.update_from_coo(&CooMatrix { symmetric: no, nrow: 1, ncol: 2, nnz: 3, max_nnz: 3, indices_i: vec![0,0,0], indices_j: vec![0,0,0], values: vec![0.0,0.0,0.0] }).err(), Some("coo.nnz must be equal to nnz(dup) = csc.row_indices.len() = csc.values.len()"));
    }

    #[test]
    fn update_from_coo_again_works() {
        let (coo, csc_correct, _, _) = Samples::umfpack_unsymmetric_5x5();
        let mut csc = NumCscMatrix::<f64>::from_coo(&coo).unwrap();
        assert_eq!(&csc.col_pointers, &csc_correct.col_pointers);
        let nnz = csc.col_pointers[csc.ncol] as usize;
        assert_eq!(&csc.row_indices[0..nnz], &csc_correct.row_indices);
        array_approx_eq(&csc.values[0..nnz], &csc_correct.values, 1e-15);

        csc.update_from_coo(&coo).unwrap();
        assert_eq!(&csc.col_pointers, &csc_correct.col_pointers);
        let nnz = csc.col_pointers[csc.ncol] as usize;
        assert_eq!(&csc.row_indices[0..nnz], &csc_correct.row_indices);
        array_approx_eq(&csc.values[0..nnz], &csc_correct.values, 1e-15);
    }

    #[test]
    fn from_csr_works() {
        const IGNORED: bool = false;
        for (_, csc_correct, csr, _) in [
            // ┌     ┐
            // │ 123 │
            // └     ┘
            Samples::tiny_1x1(),
            //  1  .  2
            //  .  0  3
            //  4  5  6
            Samples::unsymmetric_3x3(IGNORED, IGNORED),
            //  2  3  .  .  .
            //  3  .  4  .  6
            //  . -1 -3  2  .
            //  .  .  1  .  .
            //  .  4  2  .  1
            Samples::umfpack_unsymmetric_5x5(),
            //  1  -1   .  -3   .
            // -2   5   .   .   .
            //  .   .   4   6   4
            // -4   .   2   7   .
            //  .   8   .   .  -5
            Samples::mkl_unsymmetric_5x5(),
            // 1  2  .  .  .
            // 3  4  .  .  .
            // .  .  5  6  .
            // .  .  7  8  .
            // .  .  .  .  9
            Samples::block_unsymmetric_5x5(IGNORED, IGNORED),
            //     9   1.5     6  0.75     3
            //   1.5   0.5     .     .     .
            //     6     .    12     .     .
            //  0.75     .     . 0.625     .
            //     3     .     .     .    16
            Samples::mkl_positive_definite_5x5_lower(),
            Samples::mkl_symmetric_5x5_lower(IGNORED, IGNORED),
            Samples::mkl_symmetric_5x5_upper(IGNORED, IGNORED),
            Samples::mkl_symmetric_5x5_full(),
            // ┌       ┐
            // │ 10 20 │
            // └       ┘
            Samples::rectangular_1x2(IGNORED, IGNORED),
            // ┌               ┐
            // │ 1 . 3 . 5 . 7 │
            // └               ┘
            Samples::rectangular_1x7(),
            // ┌   ┐
            // │ . │
            // │ 2 │
            // │ . │
            // │ 4 │
            // │ . │
            // │ 6 │
            // │ . │
            // └   ┘
            Samples::rectangular_7x1(),
            //   5  -2  .  1
            //  10  -4  .  2
            //  15  -6  .  3
            Samples::rectangular_3x4(),
        ] {
            let csc = NumCscMatrix::<f64>::from_csr(&csr).unwrap();
            assert_eq!(&csc.col_pointers, &csc_correct.col_pointers);
            assert_eq!(&csc.row_indices, &csc_correct.row_indices);
            array_approx_eq(&csc.values, &csc_correct.values, 1e-15);
        }
    }

    #[test]
    fn to_matrix_fails_on_wrong_dims() {
        let (_, csc, _, _) = Samples::rectangular_1x2(false, false);
        let mut a_3x1 = Matrix::new(3, 1);
        let mut a_1x3 = Matrix::new(1, 3);
        assert_eq!(csc.to_dense(&mut a_3x1), Err("wrong matrix dimensions"));
        assert_eq!(csc.to_dense(&mut a_1x3), Err("wrong matrix dimensions"));
    }

    #[test]
    fn to_matrix_and_as_matrix_work() {
        // 1 x 2 matrix
        let (_, csc, _, _) = Samples::rectangular_1x2(false, false);
        let mut a = Matrix::new(1, 2);
        csc.to_dense(&mut a).unwrap();
        let correct = "┌       ┐\n\
                       │ 10 20 │\n\
                       └       ┘";
        assert_eq!(format!("{}", a), correct);

        // 5 x 5 matrix
        let (_, csc, _, _) = Samples::umfpack_unsymmetric_5x5();
        let mut a = Matrix::new(5, 5);
        csc.to_dense(&mut a).unwrap();
        let correct = "┌                ┐\n\
                       │  2  3  0  0  0 │\n\
                       │  3  0  4  0  6 │\n\
                       │  0 -1 -3  2  0 │\n\
                       │  0  0  1  0  0 │\n\
                       │  0  4  2  0  1 │\n\
                       └                ┘";
        assert_eq!(format!("{}", a), correct);
        // call to_matrix again to make sure the matrix is filled with zeros before the sum
        csc.to_dense(&mut a).unwrap();
        assert_eq!(format!("{}", a), correct);

        // use as_matrix
        let b = csc.as_dense();
        assert_eq!(format!("{}", b), correct);
    }

    #[test]
    fn as_matrix_upper_works() {
        let (_, csc, _, _) = Samples::mkl_symmetric_5x5_upper(false, false);
        let a = csc.as_dense();
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
        let (_, csc, _, _) = Samples::mkl_symmetric_5x5_lower(false, false);
        let a = csc.as_dense();
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
    fn mat_vec_mul_captures_errors() {
        let (_, csc, _, _) = Samples::rectangular_3x4();
        let u = Vector::new(3);
        let mut v = Vector::new(csc.nrow);
        assert_eq!(csc.mat_vec_mul(&mut v, 2.0, &u).err(), Some("u vector is incompatible"));
        let u = Vector::new(4);
        let mut v = Vector::new(2);
        assert_eq!(csc.mat_vec_mul(&mut v, 2.0, &u).err(), Some("v vector is incompatible"));
    }

    #[test]
    fn mat_vec_mul_works() {
        //   5  -2  .  1
        //  10  -4  .  2
        //  15  -6  .  3
        let (_, csc, _, _) = Samples::rectangular_3x4();
        let u = Vector::from(&[1.0, 3.0, 8.0, 5.0]);
        let mut v = Vector::new(csc.nrow);
        csc.mat_vec_mul(&mut v, 2.0, &u).unwrap();
        let correct = &[8.0, 16.0, 24.0];
        vec_approx_eq(&v, correct, 1e-15);
        // call mat_vec_mul again to make sure the vector is filled with zeros before the sum
        csc.mat_vec_mul(&mut v, 2.0, &u).unwrap();
        vec_approx_eq(&v, correct, 1e-15);
    }

    #[test]
    fn mat_vec_mul_symmetric_lower_works() {
        let (_, csc, _, _) = Samples::mkl_symmetric_5x5_lower(false, false);
        let u = Vector::from(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut v = Vector::new(5);
        csc.mat_vec_mul(&mut v, 2.0, &u).unwrap();
        vec_approx_eq(&v, &[96.0, 5.0, 84.0, 6.5, 166.0], 1e-15);
        // another test
        let (_, csc, _, _) = Samples::lower_symmetric_5x5();
        let u = Vector::from(&[-629.0 / 98.0, 237.0 / 49.0, -53.0 / 49.0, 62.0 / 49.0, 23.0 / 14.0]);
        let mut v = Vector::new(5);
        csc.mat_vec_mul(&mut v, 2.0, &u).unwrap();
        vec_approx_eq(&v, &[-4.0, 8.0, 6.0, -10.0, 2.0], 1e-14);
    }

    #[test]
    fn mat_vec_mul_complex_works() {
        // 4+4i    .     2+2i
        //  .      1     3+3i
        //  .     5+5i   1+1i
        //  1      .      .
        let (_, csc, _, _) = Samples::complex_rectangular_4x3();
        let u = ComplexVector::from(&[cpx!(1.0, 1.0), cpx!(3.0, 1.0), cpx!(5.0, -1.0)]);
        let mut v = ComplexVector::new(csc.nrow);
        csc.mat_vec_mul(&mut v, cpx!(2.0, 4.0), &u).unwrap();
        let correct = &[
            cpx!(-40.0, 80.0),
            cpx!(-10.0, 110.0),
            cpx!(-64.0, 112.0),
            cpx!(-2.0, 6.0),
        ];
        complex_vec_approx_eq(&v, correct, 1e-15);
        // call mat_vec_mul again to make sure the vector is filled with zeros before the sum
        csc.mat_vec_mul(&mut v, cpx!(2.0, 4.0), &u).unwrap();
        complex_vec_approx_eq(&v, correct, 1e-15);
    }

    #[test]
    fn getters_are_correct() {
        let (_, csc, _, _) = Samples::rectangular_1x2(false, false);
        assert_eq!(csc.get_info(), (1, 2, 2, Sym::No));
        assert_eq!(csc.get_col_pointers(), &[0, 1, 2]);
        assert_eq!(csc.get_row_indices(), &[0, 0]);
        assert_eq!(csc.get_values(), &[10.0, 20.0]);
        // with duplicates
        let (coo, _, _, _) = Samples::rectangular_1x2(false, false);
        let csc = NumCscMatrix::<f64>::from_coo(&coo).unwrap();
        assert_eq!(csc.get_info(), (1, 2, 2, Sym::No));
        assert_eq!(csc.get_col_pointers(), &[0, 1, 2]);
        assert_eq!(csc.get_row_indices(), &[0, 0]);
        assert_eq!(csc.get_values(), &[10.0, 20.0]);
        // mutable
        let mut csc = NumCscMatrix::<f64> {
            symmetric: Sym::No,
            nrow: 1,
            ncol: 2,
            values: vec![10.0, 20.0],
            row_indices: vec![0, 0],
            col_pointers: vec![0, 1, 2],
            temp_rp: Vec::new(),
            temp_rj: Vec::new(),
            temp_rx: Vec::new(),
            temp_rc: Vec::new(),
            temp_w: Vec::new(),
        };
        let x = csc.get_values_mut();
        x.reverse();
        assert_eq!(csc.get_values(), &[20.0, 10.0]);
    }

    #[test]
    fn derive_methods_work() {
        let (coo, _, _, _) = Samples::umfpack_unsymmetric_5x5();
        let csc = NumCscMatrix::<f64>::from_coo(&coo).unwrap();
        let nrow = coo.nrow;
        let nnz = coo.nnz; // it must be COO nnz because of (removed) duplicates
        assert_eq!(csc.temp_rp.len(), nrow + 1);
        assert_eq!(csc.temp_rj.len(), nnz);
        assert_eq!(csc.temp_rx.len(), nnz);
        assert_eq!(csc.temp_rc.len(), nrow);
        assert_eq!(csc.temp_w.len(), nrow);
        let mut clone = csc.clone();
        clone.values[0] *= 2.0;
        assert_eq!(csc.values[0], 2.0);
        assert_eq!(clone.values[0], 4.0);
        assert!(format!("{:?}", csc).len() > 0);
        let json = serde_json::to_string(&csc).unwrap();
        assert_eq!(
            json,
            r#"{"symmetric":"No","nrow":5,"ncol":5,"col_pointers":[0,2,5,9,10,12],"row_indices":[0,1,0,2,4,1,2,3,4,2,1,4,0],"values":[2.0,3.0,3.0,-1.0,4.0,4.0,-3.0,1.0,2.0,2.0,6.0,1.0,0.0]}"#
        );
        let from_json: NumCscMatrix<f64> = serde_json::from_str(&json).unwrap();
        assert_eq!(from_json.symmetric, csc.symmetric);
        assert_eq!(from_json.nrow, csc.nrow);
        assert_eq!(from_json.ncol, csc.ncol);
        assert_eq!(from_json.col_pointers, csc.col_pointers);
        assert_eq!(from_json.row_indices, csc.row_indices);
        assert_eq!(from_json.values, csc.values);
        assert_eq!(from_json.temp_rp.len(), 0);
        assert_eq!(from_json.temp_rj.len(), 0);
        assert_eq!(from_json.temp_rx.len(), 0);
        assert_eq!(from_json.temp_rc.len(), 0);
        assert_eq!(from_json.temp_w.len(), 0);
    }
}
