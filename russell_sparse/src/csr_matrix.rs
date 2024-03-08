use super::{to_i32, NumCooMatrix, NumCscMatrix, Symmetry};
use crate::StrError;
use num_traits::{Num, NumCast};
use russell_lab::{NumMatrix, NumVector};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::ops::{AddAssign, MulAssign};

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
///
/// **Note:** The number of non-zero values is `nnz = row_pointers[nrow]`
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NumCsrMatrix<T>
where
    T: AddAssign + MulAssign + Num + NumCast + Copy + DeserializeOwned + Serialize,
{
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
    /// nnz = row_pointers[nrow]
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
    #[serde(bound(deserialize = "Vec<T>: Deserialize<'de>"))]
    pub(crate) values: Vec<T>,

    /// Temporary row form (for COO to CSR conversion)
    #[serde(skip)]
    temp_rp: Vec<i32>,

    /// Temporary row form (for COO to CSR conversion)
    #[serde(skip)]
    temp_rjx: Vec<(i32, T)>,

    /// Temporary row count (for COO to CSR conversion)
    #[serde(skip)]
    temp_rc: Vec<usize>,

    /// Temporary workspace (for COO to CSR conversion)
    #[serde(skip)]
    temp_w: Vec<i32>,
}

impl<T> NumCsrMatrix<T>
where
    T: AddAssign + MulAssign + Num + NumCast + Copy + DeserializeOwned + Serialize,
{
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
        values: Vec<T>,
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
        Ok(NumCsrMatrix {
            symmetry: if let Some(v) = symmetry { v } else { Symmetry::No },
            nrow,
            ncol,
            row_pointers,
            col_indices,
            values,
            temp_rp: Vec::new(),
            temp_rjx: Vec::new(),
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
    ///     let mut coo = CooMatrix::new(nrow, ncol, nnz, None)?;
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
    pub fn from_coo(coo: &NumCooMatrix<T>) -> Result<Self, StrError> {
        if coo.nnz < 1 {
            return Err("COO to CSR requires nnz > 0");
        }
        let mut csr = NumCsrMatrix {
            symmetry: coo.symmetry,
            nrow: coo.nrow,
            ncol: coo.ncol,
            row_pointers: vec![0; coo.nrow + 1],
            col_indices: vec![0; coo.nnz],
            values: vec![T::zero(); coo.nnz],
            temp_rp: Vec::new(),
            temp_rjx: Vec::new(),
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
    pub fn update_from_coo(&mut self, coo: &NumCooMatrix<T>) -> Result<(), StrError> {
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
            self.temp_rjx = vec![(0_i32, T::zero()); nnz]; // temporary row form
            self.temp_rc = vec![0_usize; nrow]; // temporary row count
            self.temp_w = vec![0_i32; ndim]; // temporary workspace
        } else {
            for i in 0..nrow {
                self.temp_w[i] = 0;
            }
        }
        let rp = &mut self.temp_rp;
        let rjx = &mut self.temp_rjx;
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
            rjx[p].0 = aj[k];
            rjx[p].1 = ax[k];
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
                let j = rjx[p].0 as usize;
                if w[j] >= p1 as i32 {
                    // j is already in row i, position pj
                    let pj = w[j] as usize;
                    let x = rjx[p].1;
                    rjx[pj].1 += x; // sum the entry
                } else {
                    // keep the entry
                    w[j] = dest as i32;
                    if dest != p {
                        rjx[dest].0 = j as i32;
                        rjx[dest].1 = rjx[p].1;
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
            rjx[p1..p2].sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            for (j, x) in &rjx[p1..p2] {
                bj[k] = *j;
                bx[k] = *x;
                k += 1;
            }
        }
        Ok(())
    }

    /// Creates a new CSR matrix from a CSC matrix
    pub fn from_csc(csc: &NumCscMatrix<T>) -> Result<Self, StrError> {
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
        let mut csr = NumCsrMatrix {
            symmetry: csc.symmetry,
            ncol: csc.ncol,
            nrow: csc.nrow,
            row_pointers: vec![0; nrow + 1],
            col_indices: vec![0; nnz],
            values: vec![T::zero(); nnz],
            temp_rp: Vec::new(),
            temp_rjx: Vec::new(),
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
    pub fn as_dense(&self) -> NumMatrix<T> {
        let mut a = NumMatrix::new(self.nrow, self.ncol);
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
    pub fn to_dense(&self, a: &mut NumMatrix<T>) -> Result<(), StrError> {
        let (m, n) = a.dims();
        if m != self.nrow || n != self.ncol {
            return Err("wrong matrix dimensions");
        }
        let mirror_required = self.symmetry.triangular();
        a.fill(T::zero());
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
    pub fn mat_vec_mul(&self, v: &mut NumVector<T>, alpha: T, u: &NumVector<T>) -> Result<(), StrError> {
        if u.dim() != self.ncol {
            return Err("u vector is incompatible");
        }
        if v.dim() != self.nrow {
            return Err("v vector is incompatible");
        }
        let mirror_required = self.symmetry.triangular();
        v.fill(T::zero());
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
    ///
    /// ```text
    /// row_pointers.len() = nrow + 1
    /// ```
    pub fn get_row_pointers(&self) -> &[i32] {
        &self.row_pointers
    }

    /// Get an access to the column indices
    ///
    /// ```text
    /// nnz = row_pointers[nrow]
    /// col_indices.len() == nnz
    /// ```
    pub fn get_col_indices(&self) -> &[i32] {
        let nnz = self.row_pointers[self.nrow] as usize;
        &self.col_indices[..nnz]
    }

    /// Get an access to the values
    ///
    /// ```text
    /// nnz = row_pointers[nrow]
    /// values.len() == nnz
    /// ```
    pub fn get_values(&self) -> &[T] {
        let nnz = self.row_pointers[self.nrow] as usize;
        &self.values[..nnz]
    }

    /// Get a mutable access to the values
    ///
    /// ```text
    /// nnz = row_pointers[nrow]
    /// values.len() == nnz
    /// ```
    ///
    /// Note: the values may be modified externally, but not the pointers or indices.
    pub fn get_values_mut(&mut self) -> &mut [T] {
        let nnz = self.row_pointers[self.nrow] as usize;
        &mut self.values[..nnz]
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::NumCsrMatrix;
    use crate::{CooMatrix, Samples, Storage, Symmetry};
    use num_complex::Complex64;
    use russell_lab::{complex_vec_approx_eq, cpx, vec_approx_eq, ComplexVector, Matrix, Vector};

    #[test]
    fn new_captures_errors() {
        assert_eq!(
            NumCsrMatrix::<f64>::new(0, 1, vec![0], vec![], vec![], None).err(),
            Some("nrow must be ≥ 1")
        );
        assert_eq!(
            NumCsrMatrix::<f64>::new(1, 0, vec![0], vec![], vec![], None).err(),
            Some("ncol must be ≥ 1")
        );
        assert_eq!(
            NumCsrMatrix::<f64>::new(1, 1, vec![0], vec![], vec![], None).err(),
            Some("row_pointers.len() must be = nrow + 1")
        );
        assert_eq!(
            NumCsrMatrix::<f64>::new(1, 1, vec![0, 0], vec![], vec![], None).err(),
            Some("nnz = row_pointers[nrow] must be ≥ 1")
        );
        assert_eq!(
            NumCsrMatrix::<f64>::new(1, 1, vec![0, 1], vec![], vec![], None).err(),
            Some("col_indices.len() must be ≥ nnz")
        );
        assert_eq!(
            NumCsrMatrix::<f64>::new(1, 1, vec![0, 1], vec![0], vec![], None).err(),
            Some("values.len() must be ≥ nnz")
        );
        assert_eq!(
            NumCsrMatrix::<f64>::new(1, 1, vec![-1, 1], vec![0], vec![0.0], None).err(),
            Some("row pointers must be ≥ 0")
        );
        assert_eq!(
            NumCsrMatrix::<f64>::new(1, 1, vec![2, 1], vec![0], vec![0.0], None).err(),
            Some("row pointers must be sorted in ascending order")
        );
        assert_eq!(
            NumCsrMatrix::<f64>::new(1, 1, vec![0, 1], vec![-1], vec![0.0], None).err(),
            Some("column indices must be ≥ 0")
        );
        assert_eq!(
            NumCsrMatrix::<f64>::new(1, 1, vec![0, 1], vec![2], vec![0.0], None).err(),
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
            NumCsrMatrix::<f64>::new(1, 2, row_pointers, col_indices, values, None).err(),
            Some("column indices must be sorted in ascending order (within their row)")
        );
    }

    #[test]
    fn new_works() {
        let (_, _, csr_correct, _) = Samples::rectangular_1x2(false, false);
        let csr = NumCsrMatrix::<f64>::new(1, 2, vec![0, 2], vec![0, 1], vec![10.0, 20.0], None).unwrap();
        assert_eq!(csr.symmetry, Symmetry::No);
        assert_eq!(csr.nrow, 1);
        assert_eq!(csr.ncol, 2);
        assert_eq!(&csr.row_pointers, &csr_correct.row_pointers);
        assert_eq!(&csr.col_indices, &csr_correct.col_indices);
        assert_eq!(&csr.values, &csr_correct.values);
    }

    #[test]
    fn from_coo_captures_errors() {
        let coo = CooMatrix::new(1, 1, 1, None).unwrap();
        assert_eq!(
            NumCsrMatrix::<f64>::from_coo(&coo).err(),
            Some("COO to CSR requires nnz > 0")
        );
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
            let csr = NumCsrMatrix::<f64>::from_coo(&coo).unwrap();
            assert_eq!(&csr.row_pointers, &csr_correct.row_pointers);
            let nnz = csr.row_pointers[csr.nrow] as usize;
            assert_eq!(&csr.col_indices[0..nnz], &csr_correct.col_indices);
            vec_approx_eq(&csr.values[0..nnz], &csr_correct.values, 1e-15);
        }
    }

    #[test]
    fn debug_conversion() {
        let (coo, _, csr_correct, _) = Samples::umfpack_unsymmetric_5x5();
        let csr = NumCsrMatrix::<f64>::from_coo(&coo).unwrap();
        assert_eq!(&csr.row_pointers, &csr_correct.row_pointers);
        let nnz = csr.row_pointers[csr.nrow] as usize;
        assert_eq!(&csr.col_indices[0..nnz], &csr_correct.col_indices);
        vec_approx_eq(&csr.values[0..nnz], &csr_correct.values, 1e-15);
    }

    #[test]
    #[rustfmt::skip]
    fn update_from_coo_captures_errors() {
        let (coo, _, _, _) = Samples::rectangular_1x2(false, false);
        let mut csr = NumCsrMatrix::<f64>::from_coo(&coo).unwrap();
        let yes = Symmetry::General(Storage::Lower);
        let no = Symmetry::No;
        assert_eq!(csr.update_from_coo(&CooMatrix { symmetry: yes,  nrow: 1, ncol: 2, nnz: 1, max_nnz: 1, indices_i: vec![0], indices_j: vec![0], values: vec![0.0] }).err(), Some("coo.symmetry must be equal to csr.symmetry"));
        assert_eq!(csr.update_from_coo(&CooMatrix { symmetry: no, nrow: 2, ncol: 2, nnz: 1, max_nnz: 1, indices_i: vec![0], indices_j: vec![0], values: vec![0.0] }).err(), Some("coo.nrow must be equal to csr.nrow"));
        assert_eq!(csr.update_from_coo(&CooMatrix { symmetry: no, nrow: 1, ncol: 1, nnz: 1, max_nnz: 1, indices_i: vec![0], indices_j: vec![0], values: vec![0.0] }).err(), Some("coo.ncol must be equal to csr.ncol"));
        assert_eq!(csr.update_from_coo(&CooMatrix { symmetry: no, nrow: 1, ncol: 2, nnz: 3, max_nnz: 3, indices_i: vec![0,0,0], indices_j: vec![0,0,0], values: vec![0.0,0.0,0.0] }).err(), Some("coo.nnz must be equal to nnz(dup) = self.col_indices.len() = csr.values.len()"));
    }

    #[test]
    fn update_from_coo_again_works() {
        let (coo, _, csr_correct, _) = Samples::umfpack_unsymmetric_5x5();
        let mut csr = NumCsrMatrix::<f64>::from_coo(&coo).unwrap();
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
            let csr = NumCsrMatrix::<f64>::from_csc(&csc).unwrap();
            assert_eq!(&csr.row_pointers, &csr_correct.row_pointers);
            assert_eq!(&csr.col_indices, &csr_correct.col_indices);
            vec_approx_eq(&csr.values, &csr_correct.values, 1e-15);
        }
    }

    #[test]
    fn to_matrix_fails_on_wrong_dims() {
        let (_, _, csr, _) = Samples::rectangular_1x2(false, false);
        let mut a_3x1 = Matrix::new(3, 1);
        let mut a_1x3 = Matrix::new(1, 3);
        assert_eq!(csr.to_dense(&mut a_3x1), Err("wrong matrix dimensions"));
        assert_eq!(csr.to_dense(&mut a_1x3), Err("wrong matrix dimensions"));
    }

    #[test]
    fn to_matrix_and_as_matrix_work() {
        // 1 x 2 matrix
        let (_, _, csr, _) = Samples::rectangular_1x2(false, false);
        let mut a = Matrix::new(1, 2);
        csr.to_dense(&mut a).unwrap();
        let correct = "┌       ┐\n\
                       │ 10 20 │\n\
                       └       ┘";
        assert_eq!(format!("{}", a), correct);

        // 5 x 5 matrix
        let (_, _, csr, _) = Samples::umfpack_unsymmetric_5x5();
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
        let (_, _, csr, _) = Samples::mkl_symmetric_5x5_upper(false, false);
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
        let (_, _, csr, _) = Samples::mkl_symmetric_5x5_lower(false, false);
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
    fn mat_vec_mul_captures_errors() {
        let (_, _, csr, _) = Samples::rectangular_3x4();
        let u = Vector::new(3);
        let mut v = Vector::new(csr.nrow);
        assert_eq!(csr.mat_vec_mul(&mut v, 2.0, &u).err(), Some("u vector is incompatible"));
        let u = Vector::new(4);
        let mut v = Vector::new(2);
        assert_eq!(csr.mat_vec_mul(&mut v, 2.0, &u).err(), Some("v vector is incompatible"));
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
        let (_, _, csr, _) = Samples::mkl_symmetric_5x5_lower(false, false);
        let u = Vector::from(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut v = Vector::new(5);
        csr.mat_vec_mul(&mut v, 2.0, &u).unwrap();
        vec_approx_eq(v.as_data(), &[96.0, 5.0, 84.0, 6.5, 166.0], 1e-15);
        // another test
        let (_, _, csr, _) = Samples::lower_symmetric_5x5();
        let u = Vector::from(&[-629.0 / 98.0, 237.0 / 49.0, -53.0 / 49.0, 62.0 / 49.0, 23.0 / 14.0]);
        let mut v = Vector::new(5);
        csr.mat_vec_mul(&mut v, 2.0, &u).unwrap();
        vec_approx_eq(v.as_data(), &[-4.0, 8.0, 6.0, -10.0, 2.0], 1e-14);
    }

    #[test]
    fn mat_vec_mul_complex_works() {
        // 4+4i    .     2+2i
        //  .      1     3+3i
        //  .     5+5i   1+1i
        //  1      .      .
        let (_, _, csr, _) = Samples::complex_rectangular_4x3();
        let u = ComplexVector::from(&[cpx!(1.0, 1.0), cpx!(3.0, 1.0), cpx!(5.0, -1.0)]);
        let mut v = ComplexVector::new(csr.nrow);
        csr.mat_vec_mul(&mut v, cpx!(2.0, 4.0), &u).unwrap();
        let correct = &[
            cpx!(-40.0, 80.0),
            cpx!(-10.0, 110.0),
            cpx!(-64.0, 112.0),
            cpx!(-2.0, 6.0),
        ];
        complex_vec_approx_eq(v.as_data(), correct, 1e-15);
        // call mat_vec_mul again to make sure the vector is filled with zeros before the sum
        csr.mat_vec_mul(&mut v, cpx!(2.0, 4.0), &u).unwrap();
        complex_vec_approx_eq(v.as_data(), correct, 1e-15);
    }

    #[test]
    fn getters_are_correct() {
        let (_, _, csr, _) = Samples::rectangular_1x2(false, false);
        assert_eq!(csr.get_info(), (1, 2, 2, Symmetry::No));
        assert_eq!(csr.get_row_pointers(), &[0, 2]);
        assert_eq!(csr.get_col_indices(), &[0, 1]);
        assert_eq!(csr.get_values(), &[10.0, 20.0]);
        // with duplicates
        let (coo, _, _, _) = Samples::rectangular_1x2(false, false);
        let csr = NumCsrMatrix::<f64>::from_coo(&coo).unwrap();
        assert_eq!(csr.get_info(), (1, 2, 2, Symmetry::No));
        assert_eq!(csr.get_row_pointers(), &[0, 2]);
        assert_eq!(csr.get_col_indices(), &[0, 1]);
        assert_eq!(csr.get_values(), &[10.0, 20.0]);
        // mutable
        let mut csr = NumCsrMatrix::<f64> {
            symmetry: Symmetry::No,
            nrow: 1,
            ncol: 2,
            values: vec![10.0, 20.0],
            col_indices: vec![0, 1],
            row_pointers: vec![0, 2],
            temp_rp: Vec::new(),
            temp_rjx: Vec::new(),
            temp_rc: Vec::new(),
            temp_w: Vec::new(),
        };
        let x = csr.get_values_mut();
        x.reverse();
        assert_eq!(csr.get_values(), &[20.0, 10.0]);
    }

    #[test]
    fn derive_methods_work() {
        let (coo, _, _, _) = Samples::umfpack_unsymmetric_5x5();
        let csr = NumCsrMatrix::<f64>::from_coo(&coo).unwrap();
        let nrow = coo.nrow;
        let nnz = coo.nnz; // it must be COO nnz because of (removed) duplicates
        assert_eq!(csr.temp_rp.len(), nrow + 1);
        assert_eq!(csr.temp_rjx.len(), nnz);
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
        let from_json: NumCsrMatrix<f64> = serde_json::from_str(&json).unwrap();
        assert_eq!(from_json.symmetry, csr.symmetry);
        assert_eq!(from_json.nrow, csr.nrow);
        assert_eq!(from_json.ncol, csr.ncol);
        assert_eq!(from_json.row_pointers, csr.row_pointers);
        assert_eq!(from_json.col_indices, csr.col_indices);
        assert_eq!(from_json.values, csr.values);
        assert_eq!(from_json.temp_rp.len(), 0);
        assert_eq!(from_json.temp_rjx.len(), 0);
        assert_eq!(from_json.temp_rc.len(), 0);
        assert_eq!(from_json.temp_w.len(), 0);
    }
}
