use super::{to_i32, CooMatrix, CsrMatrix, Symmetry};
use crate::StrError;
use russell_lab::{Matrix, Vector};
use std::ffi::OsStr;
use std::fmt::Write;
use std::fs::{self, File};
use std::io::Write as IoWrite;
use std::path::Path;

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
#[derive(Clone)]
pub struct CscMatrix {
    /// Defines the symmetry and storage: lower-triangular, upper-triangular, full-matrix
    pub(crate) symmetry: Symmetry,

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
    pub(crate) values: Vec<f64>,

    /// Temporary row form (for COO to CSC conversion)
    temp_rp: Vec<i32>,

    /// Temporary row form (for COO to CSC conversion)
    temp_rj: Vec<i32>,

    /// Temporary row form (for COO to CSC conversion)
    temp_rx: Vec<f64>,

    /// Temporary row count (for COO to CSC conversion)
    temp_rc: Vec<usize>,

    /// Temporary workspace (for COO to CSC conversion)
    temp_w: Vec<i32>,
}

impl CscMatrix {
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
    ///     let symmetry = None;
    ///     let csc = CscMatrix::new(nrow, ncol, col_pointers, row_indices, values, symmetry)?;
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
        values: Vec<f64>,
        symmetry: Option<Symmetry>,
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
        Ok(CscMatrix {
            symmetry: if let Some(v) = symmetry { v } else { Symmetry::No },
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
    pub fn from_coo(coo: &CooMatrix) -> Result<Self, StrError> {
        if coo.nnz < 1 {
            return Err("COO to CSC requires nnz > 0");
        }
        let mut csc = CscMatrix {
            symmetry: coo.symmetry,
            nrow: coo.nrow,
            ncol: coo.ncol,
            col_pointers: vec![0; coo.ncol + 1],
            row_indices: vec![0; coo.nnz],
            values: vec![0.0; coo.nnz],
            temp_rp: Vec::new(),
            temp_rj: Vec::new(),
            temp_rx: Vec::new(),
            temp_rc: Vec::new(),
            temp_w: Vec::new(),
        };
        csc.update_from_coo(coo)?;
        Ok(csc)
    }

    /// Updates this CSC matrix from a COO matrix with a compatible structure
    ///
    /// **Note:** The COO matrix must match the symmetry, nrow, and ncol values of the CSC matrix.
    /// Also, the `nnz` (may include duplicates) of the COO matrix must match `row_indices.len() = values.len()`.
    ///
    /// **Note:** The final nnz may be smaller than the initial nnz because duplicates
    /// may have been summed up. The final nnz is available as `nnz = col_pointers[ncol]`.
    pub fn update_from_coo(&mut self, coo: &CooMatrix) -> Result<(), StrError> {
        // check dimensions
        if coo.symmetry != self.symmetry {
            return Err("coo.symmetry must be equal to csc.symmetry");
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
        let d = if coo.one_based { -1 } else { 0 };

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
            rp[i + 1] = rp[i] + w[i] as i32;
            w[i] = rp[i];
        }

        // construct the row form
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
                        // move is not need
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
            bp[j + 1] = bp[j] + w[j] as i32;
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
    ///     let symmetry = None;
    ///     let csc = CscMatrix::new(nrow, ncol,
    ///         col_pointers, row_indices, values, symmetry)?;
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
    pub fn as_dense(&self) -> Matrix {
        let mut a = Matrix::new(self.nrow, self.ncol);
        self.to_dense(&mut a).unwrap();
        a
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
    ///     let symmetry = None;
    ///     let csc = CscMatrix::new(nrow, ncol,
    ///         col_pointers, row_indices, values, symmetry)?;
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
    pub fn to_dense(&self, a: &mut Matrix) -> Result<(), StrError> {
        let (m, n) = a.dims();
        if m != self.nrow || n != self.ncol {
            return Err("wrong matrix dimensions");
        }
        let mirror_required = self.symmetry.triangular();
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
        let nnz = self.col_pointers[self.ncol] as usize;
        write!(&mut buffer, "{} {} {}\n", self.nrow, self.ncol, nnz).unwrap();

        // write triplets
        for j in 0..self.ncol {
            for p in self.col_pointers[j]..self.col_pointers[j + 1] {
                let i = self.row_indices[p as usize] as usize;
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
    ///     let col_pointers = vec![0, 1, 2];
    ///     let row_indices = vec![0, 0];
    ///     let values = vec![10.0, 20.0];
    ///     let csc = CscMatrix::new(1, 2,
    ///         col_pointers, row_indices, values, None)?;
    ///     let (nrow, ncol, nnz, symmetry) = csc.get_info();
    ///     assert_eq!(nrow, 1);
    ///     assert_eq!(ncol, 2);
    ///     assert_eq!(nnz, 2);
    ///     assert_eq!(symmetry, Symmetry::No);
    ///     let a = csc.as_dense();
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
            self.col_pointers[self.ncol] as usize,
            self.symmetry,
        )
    }

    /// Get an access to the column pointers
    pub fn get_col_pointers(&self) -> &[i32] {
        &self.col_pointers
    }

    /// Get an access to the row indices
    pub fn get_row_indices(&self) -> &[i32] {
        &self.row_indices
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
    use super::CscMatrix;
    use crate::{CooMatrix, Samples, Storage, Symmetry};
    use russell_lab::{vec_approx_eq, Matrix, Vector};
    use std::fs;

    #[test]
    fn new_captures_errors() {
        assert_eq!(
            CscMatrix::new(0, 1, vec![0], vec![], vec![], None).err(),
            Some("nrow must be ≥ 1")
        );
        assert_eq!(
            CscMatrix::new(1, 0, vec![0], vec![], vec![], None).err(),
            Some("ncol must be ≥ 1")
        );
        assert_eq!(
            CscMatrix::new(1, 1, vec![0], vec![], vec![], None).err(),
            Some("col_pointers.len() must be = ncol + 1")
        );
        assert_eq!(
            CscMatrix::new(1, 1, vec![0, 0], vec![], vec![], None).err(),
            Some("nnz = col_pointers[ncol] must be ≥ 1")
        );
        assert_eq!(
            CscMatrix::new(1, 1, vec![0, 1], vec![], vec![], None).err(),
            Some("row_indices.len() must be ≥ nnz")
        );
        assert_eq!(
            CscMatrix::new(1, 1, vec![0, 1], vec![0], vec![], None).err(),
            Some("values.len() must be ≥ nnz")
        );
        assert_eq!(
            CscMatrix::new(1, 1, vec![-1, 1], vec![0], vec![0.0], None).err(),
            Some("col pointers must be ≥ 0")
        );
        assert_eq!(
            CscMatrix::new(1, 1, vec![2, 1], vec![0], vec![0.0], None).err(),
            Some("col pointers must be sorted in ascending order")
        );
        assert_eq!(
            CscMatrix::new(1, 1, vec![0, 1], vec![-1], vec![0.0], None).err(),
            Some("row indices must be ≥ 0")
        );
        assert_eq!(
            CscMatrix::new(1, 1, vec![0, 1], vec![2], vec![0.0], None).err(),
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
            CscMatrix::new(2, 1, col_pointers, row_indices, values, None).err(),
            Some("row indices must be sorted in ascending order (within their column)")
        );
    }

    #[test]
    fn new_works() {
        let (_, csc_correct, _, _) = Samples::rectangular_1x2(false, false, false);
        let csc = CscMatrix::new(1, 2, vec![0, 1, 2], vec![0, 0], vec![10.0, 20.0], None).unwrap();
        assert_eq!(csc.symmetry, Symmetry::No);
        assert_eq!(csc.nrow, 1);
        assert_eq!(csc.ncol, 2);
        assert_eq!(&csc.col_pointers, &csc_correct.col_pointers);
        assert_eq!(&csc.row_indices, &csc_correct.row_indices);
        assert_eq!(&csc.values, &csc_correct.values);
    }

    #[test]
    fn from_coo_captures_errors() {
        let coo = CooMatrix::new(1, 1, 1, None, false).unwrap();
        assert_eq!(CscMatrix::from_coo(&coo).err(), Some("COO to CSC requires nnz > 0"));
    }

    #[test]
    fn from_coo_works() {
        for (coo, csc_correct, _, _) in [
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
            // ┌               ┐
            // │ 1 . 3 . 5 . 7 │
            // └               ┘
            Samples::rectangular_1x7(),
            Samples::rectangular_7x1(),
            Samples::rectangular_3x4(),
        ] {
            let csc = CscMatrix::from_coo(&coo).unwrap();
            assert_eq!(&csc.col_pointers, &csc_correct.col_pointers);
            let nnz = csc.col_pointers[csc.ncol] as usize;
            assert_eq!(&csc.row_indices[0..nnz], &csc_correct.row_indices);
            vec_approx_eq(&csc.values[0..nnz], &csc_correct.values, 1e-15);
        }
    }

    #[test]
    #[rustfmt::skip]
    fn update_from_coo_captures_errors() {
        let (coo, _, _, _) = Samples::rectangular_1x2(false, false, false);
        let mut csc = CscMatrix::from_coo(&coo).unwrap();
        let yes = Symmetry::General(Storage::Lower);
        let no = Symmetry::No;
        assert_eq!(csc.update_from_coo(&CooMatrix { symmetry: yes,  nrow: 1, ncol: 2, nnz: 1, max_nnz: 1, indices_i: vec![0], indices_j: vec![0], values: vec![0.0], one_based: false }).err(), Some("coo.symmetry must be equal to csc.symmetry"));
        assert_eq!(csc.update_from_coo(&CooMatrix { symmetry: no, nrow: 2, ncol: 2, nnz: 1, max_nnz: 1, indices_i: vec![0], indices_j: vec![0], values: vec![0.0], one_based: false }).err(), Some("coo.nrow must be equal to csc.nrow"));
        assert_eq!(csc.update_from_coo(&CooMatrix { symmetry: no, nrow: 1, ncol: 1, nnz: 1, max_nnz: 1, indices_i: vec![0], indices_j: vec![0], values: vec![0.0], one_based: false }).err(), Some("coo.ncol must be equal to csc.ncol"));
        assert_eq!(csc.update_from_coo(&CooMatrix { symmetry: no, nrow: 1, ncol: 2, nnz: 3, max_nnz: 3, indices_i: vec![0,0,0], indices_j: vec![0,0,0], values: vec![0.0,0.0,0.0], one_based: false }).err(), Some("coo.nnz must be equal to nnz(dup) = csc.row_indices.len() = csc.values.len()"));
    }

    #[test]
    fn update_from_coo_again_works() {
        let (coo, csc_correct, _, _) = Samples::umfpack_unsymmetric_5x5(false);
        let mut csc = CscMatrix::from_coo(&coo).unwrap();
        assert_eq!(&csc.col_pointers, &csc_correct.col_pointers);
        let nnz = csc.col_pointers[csc.ncol] as usize;
        assert_eq!(&csc.row_indices[0..nnz], &csc_correct.row_indices);
        vec_approx_eq(&csc.values[0..nnz], &csc_correct.values, 1e-15);

        csc.update_from_coo(&coo).unwrap();
        assert_eq!(&csc.col_pointers, &csc_correct.col_pointers);
        let nnz = csc.col_pointers[csc.ncol] as usize;
        assert_eq!(&csc.row_indices[0..nnz], &csc_correct.row_indices);
        vec_approx_eq(&csc.values[0..nnz], &csc_correct.values, 1e-15);
    }

    #[test]
    fn from_csr_works() {
        const IGNORED: bool = false;
        for (_, csc_correct, csr, _) in [
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
            // ┌               ┐
            // │ 1 . 3 . 5 . 7 │
            // └               ┘
            Samples::rectangular_1x7(),
            Samples::rectangular_7x1(),
            Samples::rectangular_3x4(),
        ] {
            let csc = CscMatrix::from_csr(&csr).unwrap();
            assert_eq!(&csc.col_pointers, &csc_correct.col_pointers);
            assert_eq!(&csc.row_indices, &csc_correct.row_indices);
            vec_approx_eq(&csc.values, &csc_correct.values, 1e-15);
        }
    }

    #[test]
    fn to_matrix_fails_on_wrong_dims() {
        // 10.0 20.0       << (1 x 2) matrix
        let csc = CscMatrix {
            symmetry: Symmetry::No,
            nrow: 1,
            ncol: 2,
            col_pointers: vec![0, 1, 2],
            row_indices: vec![0, 0],
            values: vec![10.0, 20.0],
            temp_rp: Vec::new(),
            temp_rj: Vec::new(),
            temp_rx: Vec::new(),
            temp_rc: Vec::new(),
            temp_w: Vec::new(),
        };
        let mut a_3x1 = Matrix::new(3, 1);
        let mut a_1x3 = Matrix::new(1, 3);
        assert_eq!(csc.to_dense(&mut a_3x1), Err("wrong matrix dimensions"));
        assert_eq!(csc.to_dense(&mut a_1x3), Err("wrong matrix dimensions"));
    }

    #[test]
    fn to_matrix_and_as_matrix_work() {
        // 10.0 20.0       << (1 x 2) matrix
        let csc = CscMatrix {
            symmetry: Symmetry::No,
            nrow: 1,
            ncol: 2,
            col_pointers: vec![0, 1, 2],
            row_indices: vec![0, 0],
            values: vec![10.0, 20.0],
            temp_rp: Vec::new(),
            temp_rj: Vec::new(),
            temp_rx: Vec::new(),
            temp_rc: Vec::new(),
            temp_w: Vec::new(),
        };
        let mut a = Matrix::new(1, 2);
        csc.to_dense(&mut a).unwrap();
        let correct = "┌       ┐\n\
                       │ 10 20 │\n\
                       └       ┘";
        assert_eq!(format!("{}", a), correct);

        let csc = CscMatrix {
            symmetry: Symmetry::No,
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
            temp_rp: Vec::new(),
            temp_rj: Vec::new(),
            temp_rx: Vec::new(),
            temp_rc: Vec::new(),
            temp_w: Vec::new(),
        };

        // covert to dense
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
        let csc = CscMatrix {
            symmetry: Symmetry::General(Storage::Upper),
            nrow: 5,
            ncol: 5,
            col_pointers: vec![0, 1, 3, 5, 7, 9],
            row_indices: vec![0, 0, 1, 0, 2, 0, 3, 0, 4],
            values: vec![9.0, 1.5, 0.5, 6.0, 12.0, 0.75, 0.625, 3.0, 16.0],
            temp_rp: Vec::new(),
            temp_rj: Vec::new(),
            temp_rx: Vec::new(),
            temp_rc: Vec::new(),
            temp_w: Vec::new(),
        };
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
        let csc = CscMatrix {
            symmetry: Symmetry::General(Storage::Lower),
            nrow: 5,
            ncol: 5,
            col_pointers: vec![0, 5, 6, 7, 8, 9],
            row_indices: vec![0, 1, 2, 3, 4, 1, 2, 3, 4],
            values: vec![9.0, 1.5, 6.0, 0.75, 3.0, 0.5, 12.0, 0.625, 16.0],
            temp_rp: Vec::new(),
            temp_rj: Vec::new(),
            temp_rx: Vec::new(),
            temp_rc: Vec::new(),
            temp_w: Vec::new(),
        };
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
    fn mat_vec_mul_works() {
        //   5  -2  .  1
        //  10  -4  .  2
        //  15  -6  .  3
        let (_, csc, _, _) = Samples::rectangular_3x4();
        let u = Vector::from(&[1.0, 3.0, 8.0, 5.0]);
        let mut v = Vector::new(csc.nrow);
        csc.mat_vec_mul(&mut v, 2.0, &u).unwrap();
        let correct = &[8.0, 16.0, 24.0];
        vec_approx_eq(v.as_data(), correct, 1e-15);
        // call mat_vec_mul again to make sure the vector is filled with zeros before the sum
        csc.mat_vec_mul(&mut v, 2.0, &u).unwrap();
        vec_approx_eq(v.as_data(), correct, 1e-15);
    }

    #[test]
    fn mat_vec_mul_symmetric_lower_works() {
        let (_, csc, _, _) = Samples::mkl_symmetric_5x5_lower(false, false, false);
        let u = Vector::from(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut v = Vector::new(5);
        csc.mat_vec_mul(&mut v, 2.0, &u).unwrap();
        vec_approx_eq(v.as_data(), &[96.0, 5.0, 84.0, 6.5, 166.0], 1e-15);
    }

    #[test]
    fn getters_are_correct() {
        let (_, csc, _, _) = Samples::rectangular_1x2(false, false, false);
        assert_eq!(csc.get_info(), (1, 2, 2, Symmetry::No));
        assert_eq!(csc.get_col_pointers(), &[0, 1, 2]);
        assert_eq!(csc.get_row_indices(), &[0, 0]);
        assert_eq!(csc.get_values(), &[10.0, 20.0]);

        let mut csc = CscMatrix {
            symmetry: Symmetry::No,
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
    fn write_matrix_market_works() {
        //  2  3  .  .  .
        //  3  .  4  .  6
        //  . -1 -3  2  .
        //  .  .  1  .  .
        //  .  4  2  .  1
        let (_, csc, _, _) = Samples::umfpack_unsymmetric_5x5(false);
        let full_path = "/tmp/russell_sparse/test_write_matrix_market_csc.mtx";
        csc.write_matrix_market(full_path, false).unwrap();
        let contents = fs::read_to_string(full_path).map_err(|_| "cannot open file").unwrap();
        assert_eq!(
            contents,
            "%%MatrixMarket matrix coordinate real general\n\
             5 5 12\n\
             1 1 2.0\n\
             2 1 3.0\n\
             1 2 3.0\n\
             3 2 -1.0\n\
             5 2 4.0\n\
             2 3 4.0\n\
             3 3 -3.0\n\
             4 3 1.0\n\
             5 3 2.0\n\
             3 4 2.0\n\
             2 5 6.0\n\
             5 5 1.0\n"
        );
    }

    #[test]
    fn clone_works() {
        let (_, csc, _, _) = Samples::tiny_1x1(false);
        let mut clone = csc.clone();
        clone.values[0] *= 2.0;
        assert_eq!(csc.values[0], 123.0);
        assert_eq!(clone.values[0], 246.0);
    }
}
