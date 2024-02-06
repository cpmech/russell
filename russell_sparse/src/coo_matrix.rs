use super::{Storage, Symmetry};
use crate::to_i32;
use crate::StrError;
use russell_lab::{Matrix, Vector};

/// Holds the row index, col index, and values of a matrix (also known as Triplet)
///
/// # Remarks
///
/// * Only the non-zero values are required
/// * Entries with repeated (i,j) indices are allowed
/// * Repeated (i,j) entries should be summed when solving a linear system
/// * The repeated (i,j) capability is of great convenience for Finite Element solvers
/// * A maximum number of entries must be decided prior to allocating a new COO matrix
/// * The maximum number of entries includes possible entries with repeated indices
#[derive(Clone)]
pub struct CooMatrix {
    /// Defines the symmetry and storage: lower-triangular, upper-triangular, full-matrix
    pub(crate) symmetry: Symmetry,

    /// Holds the number of rows (must fit i32)
    pub(crate) nrow: usize,

    /// Holds the number of columns (must fit i32)
    pub(crate) ncol: usize,

    /// Holds the current index/number of non-zeros, including duplicates (must fit i32)
    ///
    /// This will equal the number of non-zeros (nnz) after all items have been `put`.
    ///
    /// ```text
    /// nnz ≤ max_nnz
    /// ```
    pub(crate) nnz: usize,

    /// Defines the maximum allowed number of entries/non-zero values (must fit i32)
    ///
    /// This may be greater than the number of non-zeros (nnz)
    ///
    /// ```text
    /// max_nnz ≥ nnz
    /// ```
    pub(crate) max_nnz: usize,

    /// Holds the row indices i
    ///
    /// ```text
    /// indices_i.len() = max_nnz
    /// ```
    pub(crate) indices_i: Vec<i32>,

    /// Holds the column indices j
    ///
    /// ```text
    /// indices_j.len() = max_nnz
    /// ```
    pub(crate) indices_j: Vec<i32>,

    /// Holds the values aij
    ///
    /// ```text
    /// values.len() = max_nnz
    /// ```
    pub(crate) values: Vec<f64>,

    /// Defines the use of one-based indexing instead of zero-based (default)
    ///
    /// This option applies to indices_i and indices_j and enables the use of
    /// FORTRAN routines such as the ones implemented by the MUMPS solver.
    pub(crate) one_based: bool,
}

impl CooMatrix {
    /// Creates a new COO matrix representing a sparse matrix
    ///
    /// # Input
    ///
    /// * `nrow` -- (≥ 1) Is the number of rows of the sparse matrix (must be fit i32)
    /// * `ncol` -- (≥ 1) Is the number of columns of the sparse matrix (must be fit i32)
    /// * `max_nnz` -- (≥ 1) Maximum number of entries ≥ nnz (number of non-zeros),
    ///   including entries with repeated indices. (must be fit i32)
    /// * `symmetry` -- Defines the symmetry/storage, if any
    /// * `one_based` -- Use one-based indices; e.g., for MUMPS or other FORTRAN routines
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_sparse::prelude::*;
    /// use russell_sparse::StrError;
    ///
    /// fn main() -> Result<(), StrError> {
    ///     // allocate the coefficient matrix
    ///     //  2  3  .  .  .
    ///     //  3  .  4  .  6
    ///     //  . -1 -3  2  .
    ///     //  .  .  1  .  .
    ///     //  .  4  2  .  1
    ///     let mut coo = CooMatrix::new(5, 5, 13, None, true)?;
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
    ///     // covert to dense
    ///     let a = coo.as_dense();
    ///     let correct = "┌                ┐\n\
    ///                    │  2  3  0  0  0 │\n\
    ///                    │  3  0  4  0  6 │\n\
    ///                    │  0 -1 -3  2  0 │\n\
    ///                    │  0  0  1  0  0 │\n\
    ///                    │  0  4  2  0  1 │\n\
    ///                    └                ┘";
    ///     assert_eq!(format!("{}", a), correct);
    ///
    ///     // reset
    ///     coo.reset();
    ///
    ///     // covert to dense
    ///     let a = coo.as_dense();
    ///     let correct = "┌           ┐\n\
    ///                    │ 0 0 0 0 0 │\n\
    ///                    │ 0 0 0 0 0 │\n\
    ///                    │ 0 0 0 0 0 │\n\
    ///                    │ 0 0 0 0 0 │\n\
    ///                    │ 0 0 0 0 0 │\n\
    ///                    └           ┘";
    ///     assert_eq!(format!("{}", a), correct);
    ///
    ///     // put again doubled values
    ///     coo.put(0, 0, 2.0)?; // << duplicate
    ///     coo.put(0, 0, 2.0)?; // << duplicate
    ///     coo.put(1, 0, 6.0)?;
    ///     coo.put(0, 1, 6.0)?;
    ///     coo.put(2, 1, -2.0)?;
    ///     coo.put(4, 1, 8.0)?;
    ///     coo.put(1, 2, 8.0)?;
    ///     coo.put(2, 2, -6.0)?;
    ///     coo.put(3, 2, 2.0)?;
    ///     coo.put(4, 2, 4.0)?;
    ///     coo.put(2, 3, 4.0)?;
    ///     coo.put(1, 4, 12.0)?;
    ///     coo.put(4, 4, 2.0)?;
    ///
    ///     // covert to dense
    ///     let a = coo.as_dense();
    ///     let correct = "┌                ┐\n\
    ///                    │  4  6  0  0  0 │\n\
    ///                    │  6  0  8  0 12 │\n\
    ///                    │  0 -2 -6  4  0 │\n\
    ///                    │  0  0  2  0  0 │\n\
    ///                    │  0  8  4  0  2 │\n\
    ///                    └                ┘";
    ///     assert_eq!(format!("{}", a), correct);
    ///     Ok(())
    /// }
    /// ```
    pub fn new(
        nrow: usize,
        ncol: usize,
        max_nnz: usize,
        symmetry: Option<Symmetry>,
        one_based: bool,
    ) -> Result<Self, StrError> {
        if nrow < 1 {
            return Err("nrow must be ≥ 1");
        }
        if ncol < 1 {
            return Err("ncol must be ≥ 1");
        }
        if max_nnz < 1 {
            return Err("max_nnz must be ≥ 1");
        }
        Ok(CooMatrix {
            symmetry: if let Some(v) = symmetry { v } else { Symmetry::No },
            nrow,
            ncol,
            nnz: 0,
            max_nnz,
            indices_i: vec![0; max_nnz],
            indices_j: vec![0; max_nnz],
            values: vec![0.0; max_nnz],
            one_based,
        })
    }

    /// Creates a COO matrix from triplets: row indices, col indices, and non-zero values
    ///
    /// # Input
    ///
    /// * `nrow` -- (≥ 1) Is the number of rows of the sparse matrix (must be fit i32)
    /// * `ncol` -- (≥ 1) Is the number of columns of the sparse matrix (must be fit i32)
    /// * `row_indices` -- (len = nnz) Is the array of row indices
    /// * `col_indices` -- (len = nnz) Is the array of columns indices
    /// * `values` -- (len = nnz) Is the array of non-zero values
    /// * `symmetry` -- Defines the symmetry/storage, if any
    /// * `one_based` -- Use one-based indices; e.g., for MUMPS or other FORTRAN routines
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
    ///     let row_indices = vec![0, /*dup*/ 0, 1, 0, 2, 4, 1, 2, 3, 4, 2, 1, 4];
    ///     let col_indices = vec![0, /*dup*/ 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 4, 4];
    ///     let values = vec![
    ///         1.0, /*dup*/ 1.0, 3.0, 3.0, -1.0, 4.0, 4.0, -3.0, 1.0, 2.0, 2.0, 6.0, 1.0,
    ///     ];
    ///     let symmetry = None;
    ///     let coo = CooMatrix::from(nrow, ncol, row_indices, col_indices, values, symmetry, false)?;
    ///
    ///     // covert to dense
    ///     let a = coo.as_dense();
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
    pub fn from(
        nrow: usize,
        ncol: usize,
        row_indices: Vec<i32>,
        col_indices: Vec<i32>,
        values: Vec<f64>,
        symmetry: Option<Symmetry>,
        one_based: bool,
    ) -> Result<Self, StrError> {
        if nrow < 1 {
            return Err("nrow must be ≥ 1");
        }
        if ncol < 1 {
            return Err("ncol must be ≥ 1");
        }
        let nnz = row_indices.len();
        if nnz < 1 {
            return Err("nnz must be ≥ 1");
        }
        if col_indices.len() != nnz {
            return Err("col_indices.len() must be = nnz");
        }
        if values.len() != nnz {
            return Err("values.len() must be = nnz");
        }
        let d = if one_based { 1 } else { 0 };
        let m = to_i32(nrow);
        let n = to_i32(ncol);
        for k in 0..nnz {
            if row_indices[k] - d < 0 || row_indices[k] - d >= m {
                return Err("row index is out-of-range");
            }
            if col_indices[k] - d < 0 || col_indices[k] - d >= n {
                return Err("col index is out-of-range");
            }
        }
        Ok(CooMatrix {
            symmetry: if let Some(v) = symmetry { v } else { Symmetry::No },
            nrow,
            ncol,
            nnz,
            max_nnz: nnz,
            indices_i: row_indices,
            indices_j: col_indices,
            values,
            one_based,
        })
    }

    /// Puts a new entry and updates pos (may be duplicate)
    ///
    /// # Input
    ///
    /// * `i` -- row index (indices start at zero; zero-based)
    /// * `j` -- column index (indices start at zero; zero-based)
    /// * `aij` -- the value A(i,j)
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::vec_approx_eq;
    /// use russell_sparse::prelude::*;
    /// use russell_sparse::StrError;
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let (nrow, ncol, nnz) = (3, 3, 4);
    ///     let mut coo = CooMatrix::new(nrow, ncol, nnz, None, false)?;
    ///     coo.put(0, 0, 1.0)?;
    ///     coo.put(1, 1, 2.0)?;
    ///     coo.put(2, 2, 3.0)?;
    ///     coo.put(0, 1, 4.0)?;
    ///     let a = coo.as_dense();
    ///     let correct = "┌       ┐\n\
    ///                    │ 1 4 0 │\n\
    ///                    │ 0 2 0 │\n\
    ///                    │ 0 0 3 │\n\
    ///                    └       ┘";
    ///     assert_eq!(format!("{}", a), correct);
    ///     Ok(())
    /// }
    /// ```
    pub fn put(&mut self, i: usize, j: usize, aij: f64) -> Result<(), StrError> {
        // check range
        if i >= self.nrow {
            return Err("COO matrix: index of row is outside range");
        }
        if j >= self.ncol {
            return Err("COO matrix: index of column is outside range");
        }
        if self.nnz >= self.max_nnz {
            return Err("COO matrix: max number of items has been reached");
        }
        if self.symmetry != Symmetry::No {
            if self.symmetry.lower() {
                if j > i {
                    return Err("COO matrix: j > i is incorrect for lower triangular storage");
                }
            } else if self.symmetry.upper() {
                if j < i {
                    return Err("COO matrix: j < i is incorrect for upper triangular storage");
                }
            }
        }

        // insert a new entry
        let i_i32 = to_i32(i);
        let j_i32 = to_i32(j);
        let d = if self.one_based { 1 } else { 0 };
        self.indices_i[self.nnz] = i_i32 + d;
        self.indices_j[self.nnz] = j_i32 + d;
        self.values[self.nnz] = aij;
        self.nnz += 1;
        Ok(())
    }

    /// Resets the position of the current non-zero value
    ///
    /// This function allows using `put` all over again.
    ///
    /// # Example
    ///
    /// ```
    /// use russell_sparse::prelude::*;
    /// use russell_sparse::StrError;
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let (nrow, ncol, max_nnz) = (3, 3, 10);
    ///     let mut coo = CooMatrix::new(nrow, ncol, max_nnz, None, false)?;
    ///     coo.put(0, 0, 1.0)?;
    ///     coo.put(1, 1, 2.0)?;
    ///     coo.put(2, 2, 3.0)?;
    ///     coo.put(0, 1, 4.0)?;
    ///     let a = coo.as_dense();
    ///     let correct = "┌       ┐\n\
    ///                    │ 1 4 0 │\n\
    ///                    │ 0 2 0 │\n\
    ///                    │ 0 0 3 │\n\
    ///                    └       ┘";
    ///     coo.reset();
    ///     let (nrow, ncol, nnz, symmetry) = coo.get_info();
    ///     assert_eq!(nrow, 3);
    ///     assert_eq!(ncol, 3);
    ///     assert_eq!(nnz, 0);
    ///     assert_eq!(symmetry, Symmetry::No);
    ///     Ok(())
    /// }
    /// ```
    pub fn reset(&mut self) {
        self.nnz = 0;
    }

    /// Converts this COO matrix to a dense matrix
    ///
    /// ```
    /// use russell_sparse::prelude::*;
    /// use russell_sparse::StrError;
    ///
    /// fn main() -> Result<(), StrError> {
    ///     // define (4 x 4) sparse matrix with 6+1 non-zero values
    ///     // (with an extra ij-repeated entry)
    ///     let (nrow, ncol, max_nnz) = (4, 4, 10);
    ///     let mut coo = CooMatrix::new(nrow, ncol, max_nnz, None, false)?;
    ///     coo.put(0, 0, 0.5)?; // (0, 0, a00/2)  << duplicate
    ///     coo.put(0, 0, 0.5)?; // (0, 0, a00/2)  << duplicate
    ///     coo.put(0, 1, 2.0)?;
    ///     coo.put(1, 0, 3.0)?;
    ///     coo.put(1, 1, 4.0)?;
    ///     coo.put(2, 2, 5.0)?;
    ///     coo.put(3, 3, 6.0)?;
    ///
    ///     // convert to dense
    ///     let a = coo.as_dense();
    ///     let correct = "┌         ┐\n\
    ///                    │ 1 2 0 0 │\n\
    ///                    │ 3 4 0 0 │\n\
    ///                    │ 0 0 5 0 │\n\
    ///                    │ 0 0 0 6 │\n\
    ///                    └         ┘";
    ///     assert_eq!(format!("{}", a), correct);
    ///     Ok(())
    /// }
    /// ```
    pub fn as_dense(&self) -> Matrix {
        let mut a = Matrix::new(self.nrow, self.ncol);
        self.to_dense(&mut a).unwrap();
        a
    }

    /// Converts this COO matrix to a dense matrix
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
    ///     // define (4 x 4) sparse matrix with 6+1 non-zero values
    ///     // (with an extra ij-repeated entry)
    ///     let (nrow, ncol, max_nnz) = (4, 4, 10);
    ///     let mut coo = CooMatrix::new(nrow, ncol, max_nnz, None, false)?;
    ///     coo.put(0, 0, 0.5)?; // (0, 0, a00/2) << duplicate
    ///     coo.put(0, 0, 0.5)?; // (0, 0, a00/2) << duplicate
    ///     coo.put(0, 1, 2.0)?;
    ///     coo.put(1, 0, 3.0)?;
    ///     coo.put(1, 1, 4.0)?;
    ///     coo.put(2, 2, 5.0)?;
    ///     coo.put(3, 3, 6.0)?;
    ///
    ///     // convert to dense
    ///     let mut a = Matrix::new(nrow, ncol);
    ///     coo.to_dense(&mut a)?;
    ///     let correct = "┌         ┐\n\
    ///                    │ 1 2 0 0 │\n\
    ///                    │ 3 4 0 0 │\n\
    ///                    │ 0 0 5 0 │\n\
    ///                    │ 0 0 0 6 │\n\
    ///                    └         ┘";
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
        let d = if self.one_based { 1 } else { 0 };
        for p in 0..self.nnz {
            let i = (self.indices_i[p] - d) as usize;
            let j = (self.indices_j[p] - d) as usize;
            a.add(i, j, self.values[p]);
            if mirror_required && i != j {
                a.add(j, i, self.values[p]);
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
    ///
    /// # Note
    ///
    /// This method is not highly efficient but should useful in verifications.
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::{Matrix, Vector};
    /// use russell_sparse::prelude::*;
    /// use russell_sparse::StrError;
    ///
    /// fn main() -> Result<(), StrError> {
    ///     // set sparse matrix (3 x 3) with 6 non-zeros
    ///     let (nrow, ncol, max_nnz) = (3, 3, 6);
    ///     let mut coo = CooMatrix::new(nrow, ncol, max_nnz, None, false)?;
    ///     coo.put(0, 0, 1.0)?;
    ///     coo.put(1, 0, 2.0)?;
    ///     coo.put(1, 1, 3.0)?;
    ///     coo.put(2, 0, 4.0)?;
    ///     coo.put(2, 1, 5.0)?;
    ///     coo.put(2, 2, 6.0)?;
    ///
    ///     // check matrix
    ///     let a = coo.as_dense();
    ///     let correct_a = "┌       ┐\n\
    ///                      │ 1 0 0 │\n\
    ///                      │ 2 3 0 │\n\
    ///                      │ 4 5 6 │\n\
    ///                      └       ┘";
    ///     assert_eq!(format!("{}", a), correct_a);
    ///
    ///     // perform mat-vec-mul
    ///     let u = Vector::from(&[1.0, 1.0, 1.0]);
    ///     let mut v = Vector::new(nrow);
    ///     coo.mat_vec_mul(&mut v, 1.0, &u)?;
    ///
    ///     // check vector
    ///     let correct_v = "┌    ┐\n\
    ///                      │  1 │\n\
    ///                      │  5 │\n\
    ///                      │ 15 │\n\
    ///                      └    ┘";
    ///     assert_eq!(format!("{}", v), correct_v);
    ///     Ok(())
    /// }
    /// ```
    pub fn mat_vec_mul(&self, v: &mut Vector, alpha: f64, u: &Vector) -> Result<(), StrError> {
        if u.dim() != self.ncol {
            return Err("u.ndim must equal ncol");
        }
        if v.dim() != self.nrow {
            return Err("v.ndim must equal nrow");
        }
        let mirror_required = self.symmetry.triangular();
        v.fill(0.0);
        let d = if self.one_based { 1 } else { 0 };
        for p in 0..self.nnz {
            let i = (self.indices_i[p] - d) as usize;
            let j = (self.indices_j[p] - d) as usize;
            let aij = self.values[p];
            v[i] += alpha * aij * u[j];
            if mirror_required && i != j {
                v[j] += aij * u[i];
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
    ///     let coo = CooMatrix::new(1, 2, 3, None, false)?;
    ///     let (nrow, ncol, nnz, symmetry) = coo.get_info();
    ///     assert_eq!(nrow, 1);
    ///     assert_eq!(ncol, 2);
    ///     assert_eq!(nnz, 0);
    ///     assert_eq!(symmetry, Symmetry::No);
    ///     Ok(())
    /// }
    /// ```
    pub fn get_info(&self) -> (usize, usize, usize, Symmetry) {
        (self.nrow, self.ncol, self.nnz, self.symmetry)
    }

    /// Returns the storage corresponding to the symmetry type (if any)
    pub fn get_storage(&self) -> Storage {
        Symmetry::storage(self.symmetry)
    }

    /// Returns whether the symmetry flag corresponds to a symmetric matrix or not
    pub fn get_symmetric(&self) -> bool {
        self.symmetry != Symmetry::No
    }

    /// Get an access to the row indices
    pub fn get_row_indices(&self) -> &[i32] {
        &self.indices_i
    }

    /// Get an access to the column indices
    pub fn get_col_indices(&self) -> &[i32] {
        &self.indices_j
    }

    /// Get an access to the values
    pub fn get_values(&self) -> &[f64] {
        &self.values
    }

    /// Get a mutable access the values
    pub fn get_values_mut(&mut self) -> &mut [f64] {
        &mut self.values
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::CooMatrix;
    use crate::{Samples, Storage, Symmetry};
    use russell_lab::{vec_approx_eq, Matrix, Vector};

    #[test]
    fn new_captures_errors() {
        assert_eq!(CooMatrix::new(0, 1, 3, None, false).err(), Some("nrow must be ≥ 1"));
        assert_eq!(CooMatrix::new(1, 0, 3, None, false).err(), Some("ncol must be ≥ 1"));
        assert_eq!(CooMatrix::new(1, 1, 0, None, false).err(), Some("max_nnz must be ≥ 1"));
    }

    #[test]
    fn new_works() {
        let coo = CooMatrix::new(1, 1, 3, None, false).unwrap();
        assert_eq!(coo.symmetry, Symmetry::No);
        assert_eq!(coo.nrow, 1);
        assert_eq!(coo.ncol, 1);
        assert_eq!(coo.nnz, 0);
        assert_eq!(coo.max_nnz, 3);
        assert_eq!(coo.indices_i.len(), 3);
        assert_eq!(coo.indices_j.len(), 3);
        assert_eq!(coo.values.len(), 3);
    }

    #[test]
    #[rustfmt::skip]
    fn from_captures_errors(){
        assert_eq!(CooMatrix::from(0, 1, vec![ 0], vec![ 0], vec![0.0], None, false).err(), Some("nrow must be ≥ 1"));
        assert_eq!(CooMatrix::from(1, 0, vec![ 0], vec![ 0], vec![0.0], None, false).err(), Some("ncol must be ≥ 1"));
        assert_eq!(CooMatrix::from(1, 1, vec![  ], vec![ 0], vec![0.0], None, false).err(), Some("nnz must be ≥ 1"));
        assert_eq!(CooMatrix::from(1, 1, vec![ 0], vec![  ], vec![0.0], None, false).err(), Some("col_indices.len() must be = nnz"));
        assert_eq!(CooMatrix::from(1, 1, vec![ 0], vec![ 0], vec![   ], None, false).err(), Some("values.len() must be = nnz"));
        assert_eq!(CooMatrix::from(1, 1, vec![-1], vec![ 0], vec![0.0], None, false).err(), Some("row index is out-of-range"));
        assert_eq!(CooMatrix::from(1, 1, vec![ 1], vec![ 0], vec![0.0], None, false).err(), Some("row index is out-of-range"));
        assert_eq!(CooMatrix::from(1, 1, vec![ 0], vec![-1], vec![0.0], None, false).err(), Some("col index is out-of-range"));
        assert_eq!(CooMatrix::from(1, 1, vec![ 0], vec![ 1], vec![0.0], None, false).err(), Some("col index is out-of-range"));
        assert_eq!(CooMatrix::from(1, 1, vec![ 0], vec![ 1], vec![0.0], None, true).err(), Some("row index is out-of-range"));
        assert_eq!(CooMatrix::from(1, 1, vec![ 2], vec![ 1], vec![0.0], None, true).err(), Some("row index is out-of-range"));
        assert_eq!(CooMatrix::from(1, 1, vec![ 1], vec![ 0], vec![0.0], None, true).err(), Some("col index is out-of-range"));
        assert_eq!(CooMatrix::from(1, 1, vec![ 1], vec![ 2], vec![0.0], None, true).err(), Some("col index is out-of-range"));
    }

    #[test]
    fn from_works() {
        let coo = CooMatrix::from(1, 1, vec![1], vec![1], vec![123.0], None, true).unwrap();
        assert_eq!(coo.symmetry, Symmetry::No);
        assert_eq!(coo.nrow, 1);
        assert_eq!(coo.ncol, 1);
        assert_eq!(coo.nnz, 1);
        assert_eq!(coo.max_nnz, 1);
        assert_eq!(coo.indices_i, &[1]);
        assert_eq!(coo.indices_j, &[1]);
        assert_eq!(coo.values, &[123.0]);
    }

    #[test]
    fn get_info_works() {
        let coo = CooMatrix::new(1, 2, 10, None, false).unwrap();
        let (nrow, ncol, nnz, symmetry) = coo.get_info();
        assert_eq!(nrow, 1);
        assert_eq!(ncol, 2);
        assert_eq!(nnz, 0);
        assert_eq!(symmetry, Symmetry::No);
    }

    #[test]
    fn put_fails_on_wrong_values() {
        let mut coo = CooMatrix::new(1, 1, 1, None, false).unwrap();
        assert_eq!(
            coo.put(1, 0, 0.0).err(),
            Some("COO matrix: index of row is outside range")
        );
        assert_eq!(
            coo.put(0, 1, 0.0).err(),
            Some("COO matrix: index of column is outside range")
        );
        assert_eq!(coo.put(0, 0, 0.0).err(), None); // << will take all spots
        assert_eq!(
            coo.put(0, 0, 0.0).err(),
            Some("COO matrix: max number of items has been reached")
        );
        let sym = Some(Symmetry::General(Storage::Lower));
        let mut coo = CooMatrix::new(2, 2, 4, sym, false).unwrap();
        assert_eq!(
            coo.put(0, 1, 0.0).err(),
            Some("COO matrix: j > i is incorrect for lower triangular storage")
        );
        let sym = Some(Symmetry::General(Storage::Upper));
        let mut coo = CooMatrix::new(2, 2, 4, sym, false).unwrap();
        assert_eq!(
            coo.put(1, 0, 0.0).err(),
            Some("COO matrix: j < i is incorrect for upper triangular storage")
        );
    }

    #[test]
    fn put_works() {
        let mut coo = CooMatrix::new(3, 3, 5, None, false).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        assert_eq!(coo.nnz, 1);
        coo.put(0, 1, 2.0).unwrap();
        assert_eq!(coo.nnz, 2);
        coo.put(1, 0, 3.0).unwrap();
        assert_eq!(coo.nnz, 3);
        coo.put(1, 1, 4.0).unwrap();
        assert_eq!(coo.nnz, 4);
        coo.put(2, 2, 5.0).unwrap();
        assert_eq!(coo.nnz, 5);
    }

    #[test]
    fn reset_works() {
        let mut coo = CooMatrix::new(2, 2, 4, None, false).unwrap();
        assert_eq!(coo.nnz, 0);
        coo.put(0, 0, 1.0).unwrap();
        coo.put(0, 1, 4.0).unwrap();
        coo.put(1, 0, 2.0).unwrap();
        coo.put(1, 1, 3.0).unwrap();
        assert_eq!(coo.nnz, 4);
        coo.reset();
        assert_eq!(coo.nnz, 0);
    }

    #[test]
    fn to_dense_fails_on_wrong_dims() {
        let mut coo = CooMatrix::new(1, 1, 1, None, false).unwrap();
        coo.put(0, 0, 123.0).unwrap();
        let mut a_2x1 = Matrix::new(2, 1);
        let mut a_1x2 = Matrix::new(1, 2);
        assert_eq!(coo.to_dense(&mut a_2x1), Err("wrong matrix dimensions"));
        assert_eq!(coo.to_dense(&mut a_1x2), Err("wrong matrix dimensions"));
    }

    #[test]
    fn to_dense_works() {
        let mut coo = CooMatrix::new(3, 3, 5, None, false).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(0, 1, 2.0).unwrap();
        coo.put(1, 0, 3.0).unwrap();
        coo.put(1, 1, 4.0).unwrap();
        coo.put(2, 2, 5.0).unwrap();
        let mut a = Matrix::new(3, 3);
        coo.to_dense(&mut a).unwrap();
        assert_eq!(a.get(0, 0), 1.0);
        assert_eq!(a.get(0, 1), 2.0);
        assert_eq!(a.get(1, 0), 3.0);
        assert_eq!(a.get(1, 1), 4.0);
        assert_eq!(a.get(2, 2), 5.0);
        // call to_dense again to make sure the matrix is filled with zeros before the sum
        coo.to_dense(&mut a).unwrap();
        assert_eq!(a.get(0, 0), 1.0);
        assert_eq!(a.get(0, 1), 2.0);
        assert_eq!(a.get(1, 0), 3.0);
        assert_eq!(a.get(1, 1), 4.0);
        assert_eq!(a.get(2, 2), 5.0);
        // using as_dense
        let bb = coo.as_dense();
        assert_eq!(bb.get(0, 0), 1.0);
        assert_eq!(bb.get(1, 0), 3.0);
        // empty matrix
        let empty = CooMatrix::new(2, 2, 3, None, false).unwrap();
        let mat = empty.as_dense();
        assert_eq!(mat.as_data(), &[0.0, 0.0, 0.0, 0.0]);
        // single component matrix
        let mut single = CooMatrix::new(1, 1, 1, None, false).unwrap();
        single.put(0, 0, 123.0).unwrap();
        let mat = single.as_dense();
        assert_eq!(mat.as_data(), &[123.0]);
    }

    #[test]
    fn to_dense_with_duplicates_works() {
        // allocate a square matrix
        let (nrow, ncol, nnz) = (5, 5, 13);
        let mut coo = CooMatrix::new(nrow, ncol, nnz, None, false).unwrap();
        coo.put(0, 0, 1.0).unwrap(); // << (0, 0, a00/2)
        coo.put(0, 0, 1.0).unwrap(); // << (0, 0, a00/2)
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

        // print matrix
        let mut a = Matrix::new(nrow as usize, ncol as usize);
        coo.to_dense(&mut a).unwrap();
        let correct = "┌                ┐\n\
                       │  2  3  0  0  0 │\n\
                       │  3  0  4  0  6 │\n\
                       │  0 -1 -3  2  0 │\n\
                       │  0  0  1  0  0 │\n\
                       │  0  4  2  0  1 │\n\
                       └                ┘";
        assert_eq!(format!("{}", a), correct);
    }

    #[test]
    fn to_dense_symmetric_lower_works() {
        let sym = Some(Symmetry::General(Storage::Lower));
        let mut coo = CooMatrix::new(3, 3, 4, sym, false).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(1, 0, 2.0).unwrap();
        coo.put(1, 1, 3.0).unwrap();
        coo.put(2, 1, 4.0).unwrap();
        let mut a = Matrix::new(3, 3);
        coo.to_dense(&mut a).unwrap();
        let correct = "┌       ┐\n\
                       │ 1 2 0 │\n\
                       │ 2 3 4 │\n\
                       │ 0 4 0 │\n\
                       └       ┘";
        assert_eq!(format!("{}", a), correct);
    }

    #[test]
    fn to_dense_symmetric_upper_and_one_based_works() {
        let sym = Some(Symmetry::General(Storage::Upper));
        let mut coo = CooMatrix::new(3, 3, 4, sym, true).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(0, 1, 2.0).unwrap();
        coo.put(1, 1, 3.0).unwrap();
        coo.put(1, 2, 4.0).unwrap();
        let mut a = Matrix::new(3, 3);
        coo.to_dense(&mut a).unwrap();
        let correct = "┌       ┐\n\
                       │ 1 2 0 │\n\
                       │ 2 3 4 │\n\
                       │ 0 4 0 │\n\
                       └       ┘";
        assert_eq!(format!("{}", a), correct);
    }

    #[test]
    fn mat_vec_mul_fails_on_wrong_input() {
        let mut coo = CooMatrix::new(2, 2, 1, None, false).unwrap();
        coo.put(0, 0, 123.0).unwrap();
        let u = Vector::new(3);
        let mut v = Vector::new(coo.nrow);
        assert_eq!(coo.mat_vec_mul(&mut v, 1.0, &u).err(), Some("u.ndim must equal ncol"));
        let u = Vector::new(2);
        let mut v = Vector::new(1);
        assert_eq!(coo.mat_vec_mul(&mut v, 1.0, &u).err(), Some("v.ndim must equal nrow"));
    }

    #[test]
    fn mat_vec_mul_works() {
        //  1.0  2.0  3.0
        //  0.1  0.2  0.3
        // 10.0 20.0 30.0
        let mut coo = CooMatrix::new(3, 3, 9, None, false).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(0, 1, 2.0).unwrap();
        coo.put(0, 2, 3.0).unwrap();
        coo.put(1, 0, 0.1).unwrap();
        coo.put(1, 1, 0.2).unwrap();
        coo.put(1, 2, 0.3).unwrap();
        coo.put(2, 0, 10.0).unwrap();
        coo.put(2, 1, 20.0).unwrap();
        coo.put(2, 2, 30.0).unwrap();
        let u = Vector::from(&[0.1, 0.2, 0.3]);
        let mut v = Vector::new(coo.nrow);
        coo.mat_vec_mul(&mut v, 1.0, &u).unwrap();
        let correct_v = &[1.4, 0.14, 14.0];
        vec_approx_eq(v.as_data(), correct_v, 1e-15);

        // call mat_vec_mul again to make sure the vector is filled with zeros before the sum
        coo.mat_vec_mul(&mut v, 1.0, &u).unwrap();
        vec_approx_eq(v.as_data(), correct_v, 1e-15);

        // one-based indexing
        let mut coo = CooMatrix::new(3, 3, 9, None, true).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(0, 1, 2.0).unwrap();
        coo.put(0, 2, 3.0).unwrap();
        coo.put(1, 0, 0.1).unwrap();
        coo.put(1, 1, 0.2).unwrap();
        coo.put(1, 2, 0.3).unwrap();
        coo.put(2, 0, 10.0).unwrap();
        coo.put(2, 1, 20.0).unwrap();
        coo.put(2, 2, 30.0).unwrap();
        coo.mat_vec_mul(&mut v, 1.0, &u).unwrap();
        let correct_v = &[1.4, 0.14, 14.0];
        vec_approx_eq(v.as_data(), correct_v, 1e-15);

        // single component matrix
        let mut single = CooMatrix::new(1, 1, 1, None, false).unwrap();
        single.put(0, 0, 123.0).unwrap();
        let u = Vector::from(&[2.0]);
        let mut v = Vector::new(1);
        single.mat_vec_mul(&mut v, 1.0, &u).unwrap();
        assert_eq!(v.as_data(), &[246.0]);
    }

    #[test]
    fn mat_vec_mul_symmetric_lower_works() {
        // 2
        // 1  2     sym
        // 1  2  9
        // 3  1  1  7
        // 2  1  5  1  8
        let (nrow, ncol, nnz) = (5, 5, 15);
        let sym = Some(Symmetry::General(Storage::Lower));
        let mut coo = CooMatrix::new(nrow, ncol, nnz, sym, false).unwrap();
        coo.put(0, 0, 2.0).unwrap();
        coo.put(1, 1, 2.0).unwrap();
        coo.put(2, 2, 9.0).unwrap();
        coo.put(3, 3, 7.0).unwrap();
        coo.put(4, 4, 8.0).unwrap();

        coo.put(1, 0, 1.0).unwrap();

        coo.put(2, 0, 1.0).unwrap();
        coo.put(2, 1, 2.0).unwrap();

        coo.put(3, 0, 3.0).unwrap();
        coo.put(3, 1, 1.0).unwrap();
        coo.put(3, 2, 1.0).unwrap();

        coo.put(4, 0, 2.0).unwrap();
        coo.put(4, 1, 1.0).unwrap();
        coo.put(4, 2, 5.0).unwrap();
        coo.put(4, 3, 1.0).unwrap();
        let u = Vector::from(&[-629.0 / 98.0, 237.0 / 49.0, -53.0 / 49.0, 62.0 / 49.0, 23.0 / 14.0]);
        let mut v = Vector::new(coo.nrow);
        coo.mat_vec_mul(&mut v, 1.0, &u).unwrap();
        let correct_v = &[-2.0, 4.0, 3.0, -5.0, 1.0];
        vec_approx_eq(v.as_data(), correct_v, 1e-14);
    }

    #[test]
    fn mat_vec_mul_symmetric_full_works() {
        // 2  1  1  3  2
        // 1  2  2  1  1
        // 1  2  9  1  5
        // 3  1  1  7  1
        // 2  1  5  1  8
        let (nrow, ncol, nnz) = (5, 5, 25);
        let sym = Some(Symmetry::General(Storage::Full));
        let mut coo = CooMatrix::new(nrow, ncol, nnz, sym, false).unwrap();
        coo.put(0, 0, 2.0).unwrap();
        coo.put(1, 1, 2.0).unwrap();
        coo.put(2, 2, 9.0).unwrap();
        coo.put(3, 3, 7.0).unwrap();
        coo.put(4, 4, 8.0).unwrap();

        coo.put(1, 0, 1.0).unwrap();
        coo.put(0, 1, 1.0).unwrap();

        coo.put(2, 0, 1.0).unwrap();
        coo.put(0, 2, 1.0).unwrap();
        coo.put(2, 1, 2.0).unwrap();
        coo.put(1, 2, 2.0).unwrap();

        coo.put(3, 0, 3.0).unwrap();
        coo.put(0, 3, 3.0).unwrap();
        coo.put(3, 1, 1.0).unwrap();
        coo.put(1, 3, 1.0).unwrap();
        coo.put(3, 2, 1.0).unwrap();
        coo.put(2, 3, 1.0).unwrap();

        coo.put(4, 0, 2.0).unwrap();
        coo.put(0, 4, 2.0).unwrap();
        coo.put(4, 1, 1.0).unwrap();
        coo.put(1, 4, 1.0).unwrap();
        coo.put(4, 2, 5.0).unwrap();
        coo.put(2, 4, 5.0).unwrap();
        coo.put(4, 3, 1.0).unwrap();
        coo.put(3, 4, 1.0).unwrap();
        let u = Vector::from(&[-629.0 / 98.0, 237.0 / 49.0, -53.0 / 49.0, 62.0 / 49.0, 23.0 / 14.0]);
        let mut v = Vector::new(coo.nrow);
        coo.mat_vec_mul(&mut v, 1.0, &u).unwrap();
        let correct_v = &[-2.0, 4.0, 3.0, -5.0, 1.0];
        vec_approx_eq(v.as_data(), correct_v, 1e-14);
    }

    #[test]
    fn mat_vec_mul_positive_definite_works() {
        //  2  -1              2     ...
        // -1   2  -1    =>   -1   2
        //     -1   2             -1   2
        let (nrow, ncol, nnz) = (3, 3, 5);
        let sym = Some(Symmetry::PositiveDefinite(Storage::Lower));
        let mut coo = CooMatrix::new(nrow, ncol, nnz, sym, false).unwrap();
        coo.put(0, 0, 2.0).unwrap();
        coo.put(1, 1, 2.0).unwrap();
        coo.put(2, 2, 2.0).unwrap();
        coo.put(1, 0, -1.0).unwrap();
        coo.put(2, 1, -1.0).unwrap();
        let u = Vector::from(&[5.0, 8.0, 7.0]);
        let mut v = Vector::new(coo.nrow);
        coo.mat_vec_mul(&mut v, 1.0, &u).unwrap();
        let correct_v = &[2.0, 4.0, 6.0];
        vec_approx_eq(v.as_data(), correct_v, 1e-15);
    }

    #[test]
    fn getters_are_correct() {
        let (coo, _, _, _) = Samples::rectangular_1x2(false, false, false);
        assert_eq!(coo.get_info(), (1, 2, 2, Symmetry::No));
        assert_eq!(coo.get_storage(), Storage::Full);
        assert_eq!(coo.get_symmetric(), false);
        assert_eq!(coo.get_row_indices(), &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(coo.get_col_indices(), &[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(coo.get_values(), &[10.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        let sym = Some(Symmetry::new_general_full());
        let coo = CooMatrix::new(2, 2, 2, sym, false).unwrap();
        assert_eq!(coo.get_symmetric(), true);

        let mut coo = CooMatrix::new(2, 1, 2, None, false).unwrap();
        coo.put(0, 0, 123.0).unwrap();
        coo.put(1, 0, 456.0).unwrap();
        assert_eq!(coo.get_values_mut(), &[123.0, 456.0]);
        let x = coo.get_values_mut();
        x.reverse();
        assert_eq!(coo.get_values_mut(), &[456.0, 123.0]);
    }

    #[test]
    fn clone_works() {
        let (coo, _, _, _) = Samples::tiny_1x1(false);
        let mut clone = coo.clone();
        clone.values[0] *= 2.0;
        assert_eq!(coo.values[0], 123.0);
        assert_eq!(clone.values[0], 246.0);
    }
}
