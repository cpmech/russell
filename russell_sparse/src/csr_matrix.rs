use super::{to_i32, CooMatrix, CscMatrix, Symmetry};
use crate::StrError;
use russell_lab::{Matrix, Vector};
use serde::{Deserialize, Serialize};
use std::ffi::OsStr;
use std::fmt::Write;
use std::fs::{self, File};
use std::io::Write as IoWrite;
use std::path::Path;

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
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CsrMatrix {
    /// Defines the symmetry and storage: lower-triangular, upper-triangular, full-matrix
    pub(crate) symmetry: Symmetry,

    /// Holds the number of rows (must fit i32)
    pub(crate) nrow: usize,

    /// Holds the number of columns (must fit i32)
    pub(crate) ncol: usize,

    /// Defines the row pointers array with size = nrow + 1
    ///
    /// ```text
    /// row_pointers.len() = nrow + 1
    /// nnz = col_pointers[ncol]
    /// ```
    pub(crate) row_pointers: Vec<i32>,

    /// Defines the column indices array with size = nnz_dup (number of non-zeros with duplicates)
    ///
    /// ```text
    /// nnz_dup ≥ nnz
    /// col_indices.len() = nnz_dup
    /// ```
    pub(crate) col_indices: Vec<i32>,

    /// Defines the values array with size = nnz_dup (number of non-zeros with duplicates)
    ///
    /// ```text
    /// nnz_dup ≥ nnz
    /// values.len() = nnz_dup
    /// ```
    pub(crate) values: Vec<f64>,

    /// Temporary row form (for COO to CSR conversion)
    #[serde(skip)]
    temp_rp: Vec<i32>,

    /// Temporary row form (for COO to CSR conversion)
    #[serde(skip)]
    temp_rj: Vec<i32>,

    /// Temporary row form (for COO to CSR conversion)
    #[serde(skip)]
    temp_rx: Vec<f64>,

    /// Temporary row count (for COO to CSR conversion)
    #[serde(skip)]
    temp_rc: Vec<usize>,

    /// Temporary workspace (for COO to CSR conversion)
    #[serde(skip)]
    temp_w: Vec<i32>,
}

impl CsrMatrix {
    /// Creates a new CSR matrix from (sorted) data arrays
    ///
    /// **Note:** The row pointers and column indices must be **sorted** in ascending order.
    ///
    /// # Input
    ///
    /// * `nrow` -- (≥ 1) number of rows
    /// * `ncol` -- (≥ 1) number of columns
    /// * `row_pointers` -- (len = nrow + 1) row pointers with the last entry corresponding
    ///   to the number of non-zero values (sorted)
    /// * `col_indices` -- (len = nnz) column indices (sorted)
    /// * `values` -- the non-zero components of the matrix
    ///
    /// The following conditions must be satisfied (nnz is the number of non-zeros
    /// and nnz_dup is the number of non-zeros with possible duplicates):
    ///
    /// ```text
    /// nrow ≥ 1
    /// ncol ≥ 1
    /// row_pointers.len() = nrow + 1
    /// col_indices.len() = nnz_dup
    /// values.len() = nnz_dup
    /// nnz = row_pointers[nrow] ≥ 1
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
    ///     // allocate a square matrix and store as CSR matrix
    ///     //  2  3  .  .  .
    ///     //  3  .  4  .  6
    ///     //  . -1 -3  2  .
    ///     //  .  .  1  .  .
    ///     //  .  4  2  .  1
    ///     let nrow = 5;
    ///     let ncol = 5;
    ///     let row_pointers = vec![0, 2, 5, 8, 9, 12];
    ///     let col_indices = vec![
    ///         //                         p
    ///         0, 1, //    i = 0, count = 0, 1
    ///         0, 2, 4, // i = 1, count = 2, 3, 4
    ///         1, 2, 3, // i = 2, count = 5, 6, 7
    ///         2, //       i = 3, count = 8
    ///         1, 2, 4, // i = 4, count = 9, 10, 11
    ///            //              count = 12
    ///     ];
    ///     let values = vec![
    ///         //                                 p
    ///         2.0, 3.0, //        i = 0, count = 0, 1
    ///         3.0, 4.0, 6.0, //   i = 1, count = 2, 3, 4
    ///         -1.0, -3.0, 2.0, // i = 2, count = 5, 6, 7
    ///         1.0, //             i = 3, count = 8
    ///         4.0, 2.0, 1.0, //   i = 4, count = 9, 10, 11
    ///              //                    count = 12
    ///     ];
    ///     let symmetry = None;
    ///     let csr = CsrMatrix::new(nrow, ncol, row_pointers, col_indices, values, symmetry)?;
    ///
    ///     // covert to dense
    ///     let a = csr.as_dense();
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
        row_pointers: Vec<i32>,
        col_indices: Vec<i32>,
        values: Vec<f64>,
        symmetry: Option<Symmetry>,
    ) -> Result<Self, StrError> {
        if nrow < 1 {
            return Err("nrow must be ≥ 1");
        }
        if ncol < 1 {
            return Err("ncol must be ≥ 1");
        }
        if row_pointers.len() != nrow + 1 {
            return Err("row_pointers.len() must be = nrow + 1");
        }
        let nnz = row_pointers[nrow];
        if nnz < 1 {
            return Err("nnz = row_pointers[nrow] must be ≥ 1");
        }
        if col_indices.len() < nnz as usize {
            return Err("col_indices.len() must be ≥ nnz");
        }
        if values.len() < nnz as usize {
            return Err("values.len() must be ≥ nnz");
        }
        let n = to_i32(ncol);
        for i in 0..nrow {
            if row_pointers[i] < 0 {
                return Err("row pointers must be ≥ 0");
            }
            if row_pointers[i] > row_pointers[i + 1] {
                return Err("row pointers must be sorted in ascending order");
            }
            let start = row_pointers[i] as usize;
            let end = row_pointers[i + 1] as usize;
            for p in start..end {
                let j = col_indices[p];
                if j < 0 {
                    return Err("column indices must be ≥ 0");
                }
                if j >= n {
                    return Err("column indices must be < ncol");
                }
                if p > start {
                    if col_indices[p - 1] > col_indices[p] {
                        return Err("column indices must be sorted in ascending order (within their row)");
                    }
                }
            }
        }
        Ok(CsrMatrix {
            symmetry: if let Some(v) = symmetry { v } else { Symmetry::No },
            nrow,
            ncol,
            row_pointers,
            col_indices,
            values,
            temp_rp: Vec::new(),
            temp_rj: Vec::new(),
            temp_rx: Vec::new(),
            temp_rc: Vec::new(),
            temp_w: Vec::new(),
        })
    }

    /// Creates a new CSR matrix from a COO matrix
    ///
    /// **Note:** The final nnz may be smaller than the initial nnz because duplicates
    /// may have been summed up. The final nnz is available as `nnz = row_pointers[nrow]`.
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
    ///     let mut coo = CooMatrix::new(nrow, ncol, nnz, None, false)?;
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
    ///     // convert to CSR matrix
    ///     let csr = CsrMatrix::from_coo(&coo)?;
    ///     let correct_pp = &[0, 2, 5, 8, 9, 12];
    ///     let correct_jj = &[
    ///         //                         p
    ///         0, 1, //    i = 0, count = 0, 1
    ///         0, 2, 4, // i = 1, count = 2, 3, 4
    ///         1, 2, 3, // i = 2, count = 5, 6, 7
    ///         2, //       i = 3, count = 8
    ///         1, 2, 4, // i = 4, count = 9, 10, 11
    ///            //              count = 12
    ///     ];
    ///     let correct_vv = &[
    ///         //                                 p
    ///         2.0, 3.0, //        i = 0, count = 0, 1
    ///         3.0, 4.0, 6.0, //   i = 1, count = 2, 3, 4
    ///         -1.0, -3.0, 2.0, // i = 2, count = 5, 6, 7
    ///         1.0, //             i = 3, count = 8
    ///         4.0, 2.0, 1.0, //   i = 4, count = 9, 10, 11
    ///              //                    count = 12
    ///     ];
    ///
    ///     // check
    ///     let pp = csr.get_row_pointers();
    ///     let jj = csr.get_col_indices();
    ///     let vv = csr.get_values();
    ///     let final_nnz = pp[nrow] as usize;
    ///     assert_eq!(final_nnz, 12);
    ///     assert_eq!(pp, correct_pp);
    ///     assert_eq!(&jj[0..final_nnz], correct_jj);
    ///     assert_eq!(&vv[0..final_nnz], correct_vv);
    ///     Ok(())
    /// }
    /// ```
    pub fn from_coo(coo: &CooMatrix) -> Result<Self, StrError> {
        if coo.nnz < 1 {
            return Err("COO to CSR requires nnz > 0");
        }
        let mut csr = CsrMatrix {
            symmetry: coo.symmetry,
            nrow: coo.nrow,
            ncol: coo.ncol,
            row_pointers: vec![0; coo.nrow + 1],
            col_indices: vec![0; coo.nnz],
            values: vec![0.0; coo.nnz],
            temp_rp: Vec::new(),
            temp_rj: Vec::new(),
            temp_rx: Vec::new(),
            temp_rc: Vec::new(),
            temp_w: Vec::new(),
        };
        csr.update_from_coo(coo).unwrap();
        Ok(csr)
    }

    /// Updates this CSR matrix from a COO matrix with a compatible structure
    ///
    /// **Note:** The COO matrix must match the symmetry, nrow, and ncol values of the CSR matrix.
    /// Also, the `nnz` (may include duplicates) of the COO matrix must match `col_indices.len() = values.len()`.
    ///
    /// **Note:** The final nnz may be smaller than the initial nnz because duplicates
    /// may have been summed up. The final nnz is available as `nnz = row_pointers[nrow]`.
    pub fn update_from_coo(&mut self, coo: &CooMatrix) -> Result<(), StrError> {
        // check dimensions
        if coo.symmetry != self.symmetry {
            return Err("coo.symmetry must be equal to csr.symmetry");
        }
        if coo.nrow != self.nrow {
            return Err("coo.nrow must be equal to csr.nrow");
        }
        if coo.ncol != self.ncol {
            return Err("coo.ncol must be equal to csr.ncol");
        }
        if coo.nnz != self.values.len() {
            return Err("coo.nnz must be equal to nnz(dup) = self.col_indices.len() = csr.values.len()");
        }

        // Based on Prof Tim Davis' UMFPACK::UMF_triplet_map_x (umf_triplet.c)

        // constants
        let nrow = coo.nrow;
        let ncol = coo.ncol;
        let nnz = coo.nnz;
        let ndim = usize::max(nrow, ncol);
        let d = if coo.one_based { -1 } else { 0 };

        // access the triplet data
        let ai = &coo.indices_i;
        let aj = &coo.indices_j;
        let ax = &coo.values;

        // access the CSR data
        let bp = &mut self.row_pointers;
        let bj = &mut self.col_indices;
        let bx = &mut self.values;

        // allocate workspaces and get an access to them
        if self.temp_w.len() == 0 {
            self.temp_rp = vec![0_i32; nrow + 1]; // temporary row form
            self.temp_rj = vec![0_i32; nnz]; // temporary row form
            self.temp_rx = vec![0_f64; nnz]; // temporary row form
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
            let i = (ai[k] + d) as usize;
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
            let i = (ai[k] + d) as usize;
            let p = w[i] as usize;
            rj[p] = aj[k] + d;
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
                    rx[pj] += rx[p]; // sum the entry
                } else {
                    // keep the entry
                    w[j] = dest as i32;
                    if dest != p {
                        rj[dest] = j as i32;
                        rx[dest] = rx[p];
                    }
                    dest += 1;
                }
            }
            rc[i] = dest - p1;
        }

        // fix row pointers
        bp[0] = 0;
        for i in 0..nrow {
            bp[i + 1] = bp[i] + (rc[i] as i32);
        }

        // construct the row form (with sorted values)
        let mut k = 0;
        for i in 0..nrow {
            let p1 = rp[i] as usize;
            let p2 = p1 + rc[i];
            // TODO: find a way to improve this sorting step
            let mut columns: Vec<_> = (p1..p2).into_iter().map(|p| (rj[p], p)).collect();
            columns.sort();
            for (j, p) in columns {
                bj[k] = j;
                bx[k] = rx[p];
                k += 1;
            }
        }
        Ok(())
    }

    /// Creates a new CSR matrix from a CSC matrix
    pub fn from_csc(csc: &CscMatrix) -> Result<Self, StrError> {
        // Based on the SciPy code (csr_tocsc) from here:
        //
        // https://github.com/scipy/scipy/blob/main/scipy/sparse/sparsetools/csr.h
        //
        // Notes:
        //
        // * The same function csr_tocsc is used however with rows and columns swapped
        // * Linear complexity: O(nnz(A) + max(nrow, ncol))
        // * Upgrading i32 to usize is OK (the opposite is not OK => use to_i32)

        // check and read in the dimensions
        let ncol = csc.ncol as usize;
        let nrow = csc.nrow as usize;
        let nnz = csc.col_pointers[ncol] as usize;

        // access the CSC data
        let ap = &csc.col_pointers;
        let ai = &csc.row_indices;
        let ax = &csc.values;

        // allocate the CSR arrays
        let mut csr = CsrMatrix {
            symmetry: csc.symmetry,
            ncol: csc.ncol,
            nrow: csc.nrow,
            row_pointers: vec![0; nrow + 1],
            col_indices: vec![0; nnz],
            values: vec![0.0; nnz],
            temp_rp: Vec::new(),
            temp_rj: Vec::new(),
            temp_rx: Vec::new(),
            temp_rc: Vec::new(),
            temp_w: Vec::new(),
        };

        // access the CSR data
        let bp = &mut csr.row_pointers;
        let bj = &mut csr.col_indices;
        let bx = &mut csr.values;

        // compute the number of non-zero entries per row of A
        for k in 0..nnz {
            bp[ai[k] as usize] += 1;
        }

        // perform the cumulative sum of the nnz per column to get bp
        let mut sum = 0;
        for i in 0..nrow {
            let temp = bp[i];
            bp[i] = sum;
            sum += temp;
        }
        bp[nrow] = to_i32(nnz);

        // write ai and ax into bj and bx (will use bp as workspace)
        for j in 0..ncol {
            for p in ap[j]..ap[j + 1] {
                let i = ai[p as usize] as usize;
                let dest = bp[i] as usize;
                bj[dest] = to_i32(j);
                bx[dest] = ax[p as usize];
                bp[i] += 1;
            }
        }

        // fix bp
        let mut last = 0;
        for i in 0..(nrow + 1) {
            let temp = bp[i];
            bp[i] = last;
            last = temp;
        }

        // results
        Ok(csr)
    }

    /// Converts this CSR matrix to a dense matrix
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
    ///     let nrow = 5;
    ///     let ncol = 5;
    ///     let row_pointers = vec![0, 2, 5, 8, 9, 12];
    ///     let col_indices = vec![
    ///         //                         p
    ///         0, 1, //    i = 0, count = 0, 1
    ///         0, 2, 4, // i = 1, count = 2, 3, 4
    ///         1, 2, 3, // i = 2, count = 5, 6, 7
    ///         2, //       i = 3, count = 8
    ///         1, 2, 4, // i = 4, count = 9, 10, 11
    ///            //              count = 12
    ///     ];
    ///     let values = vec![
    ///         //                                 p
    ///         2.0, 3.0, //        i = 0, count = 0, 1
    ///         3.0, 4.0, 6.0, //   i = 1, count = 2, 3, 4
    ///         -1.0, -3.0, 2.0, // i = 2, count = 5, 6, 7
    ///         1.0, //             i = 3, count = 8
    ///         4.0, 2.0, 1.0, //   i = 4, count = 9, 10, 11
    ///              //                    count = 12
    ///     ];
    ///     let symmetry = None;
    ///     let csr = CsrMatrix::new(nrow, ncol,
    ///         row_pointers, col_indices, values, symmetry)?;
    ///
    ///     // covert to dense
    ///     let a = csr.as_dense();
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
    pub fn as_dense(&self) -> Matrix {
        let mut a = Matrix::new(self.nrow, self.ncol);
        self.to_dense(&mut a).unwrap();
        a
    }

    /// Converts this CSR matrix to a dense matrix
    ///
    /// # Input
    ///
    /// * `a` -- where to store the dense matrix; must be (nrow, ncol)
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
    ///     let nrow = 5;
    ///     let ncol = 5;
    ///     let row_pointers = vec![0, 2, 5, 8, 9, 12];
    ///     let col_indices = vec![
    ///         //                         p
    ///         0, 1, //    i = 0, count = 0, 1
    ///         0, 2, 4, // i = 1, count = 2, 3, 4
    ///         1, 2, 3, // i = 2, count = 5, 6, 7
    ///         2, //       i = 3, count = 8
    ///         1, 2, 4, // i = 4, count = 9, 10, 11
    ///            //              count = 12
    ///     ];
    ///     let values = vec![
    ///         //                                 p
    ///         2.0, 3.0, //        i = 0, count = 0, 1
    ///         3.0, 4.0, 6.0, //   i = 1, count = 2, 3, 4
    ///         -1.0, -3.0, 2.0, // i = 2, count = 5, 6, 7
    ///         1.0, //             i = 3, count = 8
    ///         4.0, 2.0, 1.0, //   i = 4, count = 9, 10, 11
    ///              //                    count = 12
    ///     ];
    ///     let symmetry = None;
    ///     let csr = CsrMatrix::new(nrow, ncol,
    ///         row_pointers, col_indices, values, symmetry)?;
    ///
    ///     // covert to dense
    ///     let a = csr.as_dense();
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
    pub fn to_dense(&self, a: &mut Matrix) -> Result<(), StrError> {
        let (m, n) = a.dims();
        if m != self.nrow || n != self.ncol {
            return Err("wrong matrix dimensions");
        }
        let mirror_required = self.symmetry.triangular();
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

    /// Writes a MatrixMarket file from a CooMatrix
    ///
    /// # Input
    ///
    /// * `full_path` -- may be a String, &str, or Path
    /// * `vismatrix` -- generate a SMAT file for Vismatrix instead of a MatrixMarket
    ///
    /// **Note:** The vismatrix format is is similar to the MatrixMarket format
    /// without the header, and the indices start at zero.
    ///
    /// # References
    ///
    /// * MatrixMarket: <https://math.nist.gov/MatrixMarket/formats.html>
    /// * Vismatrix: <https://github.com/cpmech/vismatrix>
    pub fn write_matrix_market<P>(&self, full_path: &P, vismatrix: bool) -> Result<(), StrError>
    where
        P: AsRef<OsStr> + ?Sized,
    {
        // output buffer
        let mut buffer = String::new();

        // handle one-based indexing
        let d = if vismatrix { 0 } else { 1 };

        // write header
        if !vismatrix {
            if self.symmetry == Symmetry::No {
                write!(&mut buffer, "%%MatrixMarket matrix coordinate real general\n").unwrap();
            } else {
                write!(&mut buffer, "%%MatrixMarket matrix coordinate real symmetric\n").unwrap();
            }
        }

        // write dimensions
        let nnz = self.row_pointers[self.nrow] as usize;
        write!(&mut buffer, "{} {} {}\n", self.nrow, self.ncol, nnz).unwrap();

        // write triplets
        for i in 0..self.nrow {
            for p in self.row_pointers[i]..self.row_pointers[i + 1] {
                let j = self.col_indices[p as usize] as usize;
                let aij = self.values[p as usize];
                write!(&mut buffer, "{} {} {:?}\n", i + d, j + d, aij).unwrap();
            }
        }

        // create directory
        let path = Path::new(full_path);
        if let Some(p) = path.parent() {
            fs::create_dir_all(p).map_err(|_| "cannot create directory")?;
        }

        // write file
        let mut file = File::create(path).map_err(|_| "cannot create file")?;
        file.write_all(buffer.as_bytes()).map_err(|_| "cannot write file")?;

        // force sync
        file.sync_all().map_err(|_| "cannot sync file")?;
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
        if u.dim() != self.ncol {
            return Err("u.ndim must equal ncol");
        }
        if v.dim() != self.nrow {
            return Err("v.ndim must equal nrow");
        }
        let mirror_required = self.symmetry.triangular();
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

    /// Returns information about the dimensions and symmetry type
    ///
    /// Returns `(nrow, ncol, nnz, symmetry)`
    ///
    /// # Example
    ///
    /// ```
    /// use russell_sparse::prelude::*;
    /// use russell_sparse::StrError;
    ///
    /// fn main() -> Result<(), StrError> {
    ///     // ┌       ┐
    ///     // │ 10 20 │
    ///     // └       ┘
    ///     let row_pointers = vec![0, 2];
    ///     let col_indices = vec![0, 1];
    ///     let values = vec![10.0, 20.0];
    ///     let csr = CsrMatrix::new(1, 2,
    ///         row_pointers, col_indices, values, None)?;
    ///     let (nrow, ncol, nnz, symmetry) = csr.get_info();
    ///     assert_eq!(nrow, 1);
    ///     assert_eq!(ncol, 2);
    ///     assert_eq!(nnz, 2);
    ///     assert_eq!(symmetry, Symmetry::No);
    ///     let a = csr.as_dense();
    ///     let correct = "┌       ┐\n\
    ///                    │ 10 20 │\n\
    ///                    └       ┘";
    ///     assert_eq!(format!("{}", a), correct);
    ///     Ok(())
    /// }
    /// ```
    pub fn get_info(&self) -> (usize, usize, usize, Symmetry) {
        (
            self.nrow,
            self.ncol,
            self.row_pointers[self.nrow] as usize,
            self.symmetry,
        )
    }

    /// Get an access to the row pointers
    pub fn get_row_pointers(&self) -> &[i32] {
        &self.row_pointers
    }

    /// Get an access to the column indices
    pub fn get_col_indices(&self) -> &[i32] {
        &self.col_indices
    }

    /// Get an access to the values
    pub fn get_values(&self) -> &[f64] {
        &self.values
    }

    /// Get a mutable access to the values
    pub fn get_values_mut(&mut self) -> &mut [f64] {
        &mut self.values
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::CsrMatrix;
    use crate::{CooMatrix, Samples, Storage, Symmetry};
    use russell_lab::{vec_approx_eq, Matrix, Vector};
    use std::fs;

    #[test]
    fn new_captures_errors() {
        assert_eq!(
            CsrMatrix::new(0, 1, vec![0], vec![], vec![], None).err(),
            Some("nrow must be ≥ 1")
        );
        assert_eq!(
            CsrMatrix::new(1, 0, vec![0], vec![], vec![], None).err(),
            Some("ncol must be ≥ 1")
        );
        assert_eq!(
            CsrMatrix::new(1, 1, vec![0], vec![], vec![], None).err(),
            Some("row_pointers.len() must be = nrow + 1")
        );
        assert_eq!(
            CsrMatrix::new(1, 1, vec![0, 0], vec![], vec![], None).err(),
            Some("nnz = row_pointers[nrow] must be ≥ 1")
        );
        assert_eq!(
            CsrMatrix::new(1, 1, vec![0, 1], vec![], vec![], None).err(),
            Some("col_indices.len() must be ≥ nnz")
        );
        assert_eq!(
            CsrMatrix::new(1, 1, vec![0, 1], vec![0], vec![], None).err(),
            Some("values.len() must be ≥ nnz")
        );
        assert_eq!(
            CsrMatrix::new(1, 1, vec![-1, 1], vec![0], vec![0.0], None).err(),
            Some("row pointers must be ≥ 0")
        );
        assert_eq!(
            CsrMatrix::new(1, 1, vec![2, 1], vec![0], vec![0.0], None).err(),
            Some("row pointers must be sorted in ascending order")
        );
        assert_eq!(
            CsrMatrix::new(1, 1, vec![0, 1], vec![-1], vec![0.0], None).err(),
            Some("column indices must be ≥ 0")
        );
        assert_eq!(
            CsrMatrix::new(1, 1, vec![0, 1], vec![2], vec![0.0], None).err(),
            Some("column indices must be < ncol")
        );
        // ┌       ┐
        // │ 10 20 │
        // └       ┘
        let values = vec![
            10.0, 20.0, // i=0, p=(0),1
        ]; //                   p=(2)
        let col_indices = vec![1, 0]; // << incorrect, should be [0, 1]
        let row_pointers = vec![0, 2];
        assert_eq!(
            CsrMatrix::new(1, 2, row_pointers, col_indices, values, None).err(),
            Some("column indices must be sorted in ascending order (within their row)")
        );
    }

    #[test]
    fn new_works() {
        let (_, _, csr_correct, _) = Samples::rectangular_1x2(false, false, false);
        let csr = CsrMatrix::new(1, 2, vec![0, 2], vec![0, 1], vec![10.0, 20.0], None).unwrap();
        assert_eq!(csr.symmetry, Symmetry::No);
        assert_eq!(csr.nrow, 1);
        assert_eq!(csr.ncol, 2);
        assert_eq!(&csr.row_pointers, &csr_correct.row_pointers);
        assert_eq!(&csr.col_indices, &csr_correct.col_indices);
        assert_eq!(&csr.values, &csr_correct.values);
    }

    #[test]
    fn from_coo_captures_errors() {
        let coo = CooMatrix::new(1, 1, 1, None, false).unwrap();
        assert_eq!(CsrMatrix::from_coo(&coo).err(), Some("COO to CSR requires nnz > 0"));
    }

    #[test]
    fn from_coo_works() {
        // 1  2  .  .  .
        // 3  4  .  .  .
        // .  .  5  6  .
        // .  .  7  8  .
        // .  .  .  .  9
        for (coo, _, csr_correct, _) in [
            // ┌     ┐
            // │ 123 │
            // └     ┘
            Samples::tiny_1x1(false),
            Samples::tiny_1x1(true),
            //  1  .  2
            //  .  0  3
            //  4  5  6
            Samples::unsymmetric_3x3(false, false, false),
            Samples::unsymmetric_3x3(false, true, false),
            Samples::unsymmetric_3x3(false, false, true),
            Samples::unsymmetric_3x3(false, true, true),
            Samples::unsymmetric_3x3(true, false, false),
            Samples::unsymmetric_3x3(true, true, false),
            Samples::unsymmetric_3x3(true, false, true),
            Samples::unsymmetric_3x3(true, true, true),
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
            // ┌       ┐
            // │ 10 20 │
            // └       ┘
            Samples::rectangular_1x2(false, false, false),
            Samples::rectangular_1x2(false, true, false),
            Samples::rectangular_1x2(false, false, true),
            Samples::rectangular_1x2(false, true, true),
            Samples::rectangular_1x2(true, false, false),
            Samples::rectangular_1x2(true, true, false),
            Samples::rectangular_1x2(true, false, true),
            Samples::rectangular_1x2(true, true, true),
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
            let csr = CsrMatrix::from_coo(&coo).unwrap();
            assert_eq!(&csr.row_pointers, &csr_correct.row_pointers);
            let nnz = csr.row_pointers[csr.nrow] as usize;
            assert_eq!(&csr.col_indices[0..nnz], &csr_correct.col_indices);
            vec_approx_eq(&csr.values[0..nnz], &csr_correct.values, 1e-15);
        }
    }

    #[test]
    fn debug_conversion() {
        let (coo, _, csr_correct, _) = Samples::umfpack_unsymmetric_5x5(false);
        let csr = CsrMatrix::from_coo(&coo).unwrap();
        assert_eq!(&csr.row_pointers, &csr_correct.row_pointers);
        let nnz = csr.row_pointers[csr.nrow] as usize;
        assert_eq!(&csr.col_indices[0..nnz], &csr_correct.col_indices);
        vec_approx_eq(&csr.values[0..nnz], &csr_correct.values, 1e-15);
    }

    #[test]
    #[rustfmt::skip]
    fn update_from_coo_captures_errors() {
        let (coo, _, _, _) = Samples::rectangular_1x2(false, false, false);
        let mut csr = CsrMatrix::from_coo(&coo).unwrap();
        let yes = Symmetry::General(Storage::Lower);
        let no = Symmetry::No;
        assert_eq!(csr.update_from_coo(&CooMatrix { symmetry: yes,  nrow: 1, ncol: 2, nnz: 1, max_nnz: 1, indices_i: vec![0], indices_j: vec![0], values: vec![0.0], one_based: false }).err(), Some("coo.symmetry must be equal to csr.symmetry"));
        assert_eq!(csr.update_from_coo(&CooMatrix { symmetry: no, nrow: 2, ncol: 2, nnz: 1, max_nnz: 1, indices_i: vec![0], indices_j: vec![0], values: vec![0.0], one_based: false }).err(), Some("coo.nrow must be equal to csr.nrow"));
        assert_eq!(csr.update_from_coo(&CooMatrix { symmetry: no, nrow: 1, ncol: 1, nnz: 1, max_nnz: 1, indices_i: vec![0], indices_j: vec![0], values: vec![0.0], one_based: false }).err(), Some("coo.ncol must be equal to csr.ncol"));
        assert_eq!(csr.update_from_coo(&CooMatrix { symmetry: no, nrow: 1, ncol: 2, nnz: 3, max_nnz: 3, indices_i: vec![0,0,0], indices_j: vec![0,0,0], values: vec![0.0,0.0,0.0], one_based: false }).err(), Some("coo.nnz must be equal to nnz(dup) = self.col_indices.len() = csr.values.len()"));
    }

    #[test]
    fn update_from_coo_again_works() {
        let (coo, _, csr_correct, _) = Samples::umfpack_unsymmetric_5x5(false);
        let mut csr = CsrMatrix::from_coo(&coo).unwrap();
        assert_eq!(&csr.row_pointers, &csr_correct.row_pointers);
        let nnz = csr.row_pointers[csr.nrow] as usize;
        assert_eq!(&csr.col_indices[0..nnz], &csr_correct.col_indices);
        vec_approx_eq(&csr.values[0..nnz], &csr_correct.values, 1e-15);

        csr.update_from_coo(&coo).unwrap();
        assert_eq!(&csr.row_pointers, &csr_correct.row_pointers);
        let nnz = csr.row_pointers[csr.nrow] as usize;
        assert_eq!(&csr.col_indices[0..nnz], &csr_correct.col_indices);
        vec_approx_eq(&csr.values[0..nnz], &csr_correct.values, 1e-15);
    }

    #[test]
    fn from_csc_works() {
        const IGNORED: bool = false;
        for (_, csc, csr_correct, _) in [
            // ┌     ┐
            // │ 123 │
            // └     ┘
            Samples::tiny_1x1(IGNORED),
            //  1  .  2
            //  .  0  3
            //  4  5  6
            Samples::unsymmetric_3x3(IGNORED, IGNORED, IGNORED),
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
            // ┌       ┐
            // │ 10 20 │
            // └       ┘
            Samples::rectangular_1x2(IGNORED, IGNORED, IGNORED),
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
            let csr = CsrMatrix::from_csc(&csc).unwrap();
            assert_eq!(&csr.row_pointers, &csr_correct.row_pointers);
            assert_eq!(&csr.col_indices, &csr_correct.col_indices);
            vec_approx_eq(&csr.values, &csr_correct.values, 1e-15);
        }
    }

    #[test]
    fn to_matrix_fails_on_wrong_dims() {
        let (_, _, csr, _) = Samples::rectangular_1x2(false, false, false);
        let mut a_3x1 = Matrix::new(3, 1);
        let mut a_1x3 = Matrix::new(1, 3);
        assert_eq!(csr.to_dense(&mut a_3x1), Err("wrong matrix dimensions"));
        assert_eq!(csr.to_dense(&mut a_1x3), Err("wrong matrix dimensions"));
    }

    #[test]
    fn to_matrix_and_as_matrix_work() {
        // 1 x 2 matrix
        let (_, _, csr, _) = Samples::rectangular_1x2(false, false, false);
        let mut a = Matrix::new(1, 2);
        csr.to_dense(&mut a).unwrap();
        let correct = "┌       ┐\n\
                       │ 10 20 │\n\
                       └       ┘";
        assert_eq!(format!("{}", a), correct);

        // 5 x 5 matrix
        let (_, _, csr, _) = Samples::umfpack_unsymmetric_5x5(false);
        let mut a = Matrix::new(5, 5);
        csr.to_dense(&mut a).unwrap();
        let correct = "┌                ┐\n\
                       │  2  3  0  0  0 │\n\
                       │  3  0  4  0  6 │\n\
                       │  0 -1 -3  2  0 │\n\
                       │  0  0  1  0  0 │\n\
                       │  0  4  2  0  1 │\n\
                       └                ┘";
        assert_eq!(format!("{}", a), correct);
        // call to_matrix again to make sure the matrix is filled with zeros before the sum
        csr.to_dense(&mut a).unwrap();
        assert_eq!(format!("{}", a), correct);

        // use as_matrix
        let b = csr.as_dense();
        assert_eq!(format!("{}", b), correct);
    }

    #[test]
    fn as_matrix_upper_works() {
        let (_, _, csr, _) = Samples::mkl_symmetric_5x5_upper(false, false, false);
        let a = csr.as_dense();
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
        let (_, _, csr, _) = Samples::mkl_symmetric_5x5_lower(false, false, false);
        let a = csr.as_dense();
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
        //   5  -2  .  1
        //  10  -4  .  2
        //  15  -6  .  3
        let (_, _, csr, _) = Samples::rectangular_3x4();
        let u = Vector::from(&[1.0, 3.0, 8.0, 5.0]);
        let mut v = Vector::new(csr.nrow);
        csr.mat_vec_mul(&mut v, 2.0, &u).unwrap();
        let correct = &[8.0, 16.0, 24.0];
        vec_approx_eq(v.as_data(), correct, 1e-15);
        // call mat_vec_mul again to make sure the vector is filled with zeros before the sum
        csr.mat_vec_mul(&mut v, 2.0, &u).unwrap();
        vec_approx_eq(v.as_data(), correct, 1e-15);
    }

    #[test]
    fn mat_vec_mul_symmetric_lower_works() {
        let (_, _, csr, _) = Samples::mkl_symmetric_5x5_lower(false, false, false);
        let u = Vector::from(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut v = Vector::new(5);
        csr.mat_vec_mul(&mut v, 2.0, &u).unwrap();
        vec_approx_eq(v.as_data(), &[96.0, 5.0, 84.0, 6.5, 166.0], 1e-15);
    }

    #[test]
    fn getters_are_correct() {
        let (_, _, csr, _) = Samples::rectangular_1x2(false, false, false);
        assert_eq!(csr.get_info(), (1, 2, 2, Symmetry::No));
        assert_eq!(csr.get_row_pointers(), &[0, 2]);
        assert_eq!(csr.get_col_indices(), &[0, 1]);
        assert_eq!(csr.get_values(), &[10.0, 20.0]);

        let mut csr = CsrMatrix {
            symmetry: Symmetry::No,
            nrow: 1,
            ncol: 2,
            values: vec![10.0, 20.0],
            col_indices: vec![0, 1],
            row_pointers: vec![0, 2],
            temp_rp: Vec::new(),
            temp_rj: Vec::new(),
            temp_rx: Vec::new(),
            temp_rc: Vec::new(),
            temp_w: Vec::new(),
        };
        let x = csr.get_values_mut();
        x.reverse();
        assert_eq!(csr.get_values(), &[20.0, 10.0]);
    }

    #[test]
    fn write_matrix_market_works() {
        //  2  3  .  .  .
        //  3  .  4  .  6
        //  . -1 -3  2  .
        //  .  .  1  .  .
        //  .  4  2  .  1
        let (_, _, csr, _) = Samples::umfpack_unsymmetric_5x5(false);
        let full_path = "/tmp/russell_sparse/test_write_matrix_market_csr.mtx";
        csr.write_matrix_market(full_path, false).unwrap();
        let contents = fs::read_to_string(full_path).map_err(|_| "cannot open file").unwrap();
        assert_eq!(
            contents,
            "%%MatrixMarket matrix coordinate real general\n\
             5 5 12\n\
             1 1 2.0\n\
             1 2 3.0\n\
             2 1 3.0\n\
             2 3 4.0\n\
             2 5 6.0\n\
             3 2 -1.0\n\
             3 3 -3.0\n\
             3 4 2.0\n\
             4 3 1.0\n\
             5 2 4.0\n\
             5 3 2.0\n\
             5 5 1.0\n"
        );
    }

    #[test]
    fn derive_methods_works() {
        let (coo, _, _, _) = Samples::umfpack_unsymmetric_5x5(false);
        let csr = CsrMatrix::from_coo(&coo).unwrap();
        let nrow = coo.nrow;
        let nnz = coo.nnz; // it must be COO nnz because of (removed) duplicates
        assert_eq!(csr.temp_rp.len(), nrow + 1);
        assert_eq!(csr.temp_rj.len(), nnz);
        assert_eq!(csr.temp_rx.len(), nnz);
        assert_eq!(csr.temp_rc.len(), nrow);
        assert_eq!(csr.temp_w.len(), nrow);
        let mut clone = csr.clone();
        clone.values[0] *= 2.0;
        assert_eq!(csr.values[0], 2.0);
        assert_eq!(clone.values[0], 4.0);
        assert!(format!("{:?}", csr).len() > 0);
        let json = serde_json::to_string(&csr).unwrap();
        assert_eq!(
            json,
            r#"{"symmetry":"No","nrow":5,"ncol":5,"row_pointers":[0,2,5,8,9,12],"col_indices":[0,1,0,2,4,1,2,3,2,1,2,4,0],"values":[2.0,3.0,3.0,4.0,6.0,-1.0,-3.0,2.0,1.0,4.0,2.0,1.0,0.0]}"#
        );
        let from_json: CsrMatrix = serde_json::from_str(&json).unwrap();
        assert_eq!(from_json.symmetry, csr.symmetry);
        assert_eq!(from_json.nrow, csr.nrow);
        assert_eq!(from_json.ncol, csr.ncol);
        assert_eq!(from_json.row_pointers, csr.row_pointers);
        assert_eq!(from_json.col_indices, csr.col_indices);
        assert_eq!(from_json.values, csr.values);
        assert_eq!(from_json.temp_rp.len(), 0);
        assert_eq!(from_json.temp_rj.len(), 0);
        assert_eq!(from_json.temp_rx.len(), 0);
        assert_eq!(from_json.temp_rc.len(), 0);
        assert_eq!(from_json.temp_w.len(), 0);
    }
}
