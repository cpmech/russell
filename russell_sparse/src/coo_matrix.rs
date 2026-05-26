use super::Sym;
use crate::to_i32;
use crate::StrError;
use num_traits::{Num, NumCast};
use russell_lab::{NumMatrix, NumVector};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::ops::{AddAssign, MulAssign};

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
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NumCooMatrix<T>
where
    T: AddAssign + MulAssign + Num + NumCast + Copy + DeserializeOwned + Serialize,
{
    /// Indicates whether the matrix is symmetric or not. If symmetric, indicates the representation too.
    pub(crate) symmetric: Sym,

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
    #[serde(bound(deserialize = "Vec<T>: Deserialize<'de>"))]
    pub(crate) values: Vec<T>,
}

impl<T> NumCooMatrix<T>
where
    T: AddAssign + MulAssign + Num + NumCast + Copy + DeserializeOwned + Serialize,
{
    /// Creates a new COO matrix representing a sparse matrix
    ///
    /// # Input
    ///
    /// * `nrow` -- (≥ 1) Is the number of rows of the sparse matrix (must be fit i32)
    /// * `ncol` -- (≥ 1) Is the number of columns of the sparse matrix (must be fit i32)
    /// * `max_nnz` -- (≥ 1) Maximum number of entries ≥ nnz (number of non-zeros),
    ///   including entries with repeated indices. (must be fit i32)
    /// * `symmetric` -- indicates whether the matrix is symmetric or not.
    ///   If symmetric, indicates the representation too.
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
    ///     let mut coo = CooMatrix::new(5, 5, 13, Sym::No)?;
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
    pub fn new(nrow: usize, ncol: usize, max_nnz: usize, symmetric: Sym) -> Result<Self, StrError> {
        if nrow < 1 {
            return Err("nrow must be ≥ 1");
        }
        if ncol < 1 {
            return Err("ncol must be ≥ 1");
        }
        if max_nnz < 1 {
            return Err("max_nnz must be ≥ 1");
        }
        if symmetric != Sym::No && nrow != ncol {
            return Err("symmetric storage requires a square matrix");
        }
        Ok(NumCooMatrix {
            symmetric,
            nrow,
            ncol,
            nnz: 0,
            max_nnz,
            indices_i: vec![0; max_nnz],
            indices_j: vec![0; max_nnz],
            values: vec![T::zero(); max_nnz],
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
    /// * `symmetric` -- Indicates that this matrix is symmetric (with the storage type)
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
    ///     let coo = CooMatrix::from(nrow, ncol, row_indices, col_indices, values, Sym::No)?;
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
        values: Vec<T>,
        symmetric: Sym,
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
        let m = to_i32(nrow);
        let n = to_i32(ncol);
        for k in 0..nnz {
            if row_indices[k] < 0 || row_indices[k] >= m {
                return Err("row index is out-of-range");
            }
            if col_indices[k] < 0 || col_indices[k] >= n {
                return Err("col index is out-of-range");
            }
        }
        Ok(NumCooMatrix {
            symmetric,
            nrow,
            ncol,
            nnz,
            max_nnz: nnz,
            indices_i: row_indices,
            indices_j: col_indices,
            values,
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
    /// # Examples
    ///
    /// ```
    /// use russell_lab::vec_approx_eq;
    /// use russell_sparse::prelude::*;
    /// use russell_sparse::StrError;
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let (nrow, ncol, nnz) = (3, 3, 4);
    ///     let mut coo = CooMatrix::new(nrow, ncol, nnz, Sym::No)?;
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
    pub fn put(&mut self, i: usize, j: usize, aij: T) -> Result<(), StrError> {
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
        if self.symmetric == Sym::YesLower {
            if j > i {
                return Err("COO matrix: j > i is incorrect for lower triangular storage");
            }
        }
        if self.symmetric == Sym::YesUpper {
            if j < i {
                return Err("COO matrix: j < i is incorrect for upper triangular storage");
            }
        }

        // insert a new entry
        let i_i32 = to_i32(i);
        let j_i32 = to_i32(j);
        self.indices_i[self.nnz] = i_i32;
        self.indices_j[self.nnz] = j_i32;
        self.values[self.nnz] = aij;
        self.nnz += 1;
        Ok(())
    }

    /// Resets the position of the current non-zero value
    ///
    /// This function allows using `put` all over again.
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_sparse::prelude::*;
    /// use russell_sparse::StrError;
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let (nrow, ncol, max_nnz) = (3, 3, 10);
    ///     let mut coo = CooMatrix::new(nrow, ncol, max_nnz, Sym::No)?;
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
    ///     let (nrow, ncol, nnz, sym) = coo.get_info();
    ///     assert_eq!(nrow, 3);
    ///     assert_eq!(ncol, 3);
    ///     assert_eq!(nnz, 0);
    ///     assert_eq!(sym, Sym::No);
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
    ///     let mut coo = CooMatrix::new(nrow, ncol, max_nnz, Sym::No)?;
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
    pub fn as_dense(&self) -> NumMatrix<T> {
        let mut a = NumMatrix::new(self.nrow, self.ncol);
        self.to_dense(&mut a).unwrap();
        a
    }

    /// Converts this COO matrix to a dense matrix
    ///
    /// # Input
    ///
    /// * `a` -- where to store the dense matrix; it must be (nrow, ncol)
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
    ///     let mut coo = CooMatrix::new(nrow, ncol, max_nnz, Sym::No)?;
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
    pub fn to_dense(&self, a: &mut NumMatrix<T>) -> Result<(), StrError> {
        let (m, n) = a.dims();
        if m != self.nrow || n != self.ncol {
            return Err("wrong matrix dimensions");
        }
        let mirror_required = self.symmetric.triangular();
        a.fill(T::zero());
        for p in 0..self.nnz {
            let i = self.indices_i[p] as usize;
            let j = self.indices_j[p] as usize;
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
    /// * `u` -- Vector with dimension ≥ the number of columns of the matrix
    ///
    /// # Output
    ///
    /// * `v` -- Vector with dimension ≥ the number of rows of the matrix
    ///
    /// # Note
    ///
    /// This method is not highly efficient but should useful with small matrices or verifications.
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::{Matrix, Vector};
    /// use russell_sparse::prelude::*;
    /// use russell_sparse::StrError;
    ///
    /// fn main() -> Result<(), StrError> {
    ///     // set sparse matrix (3 x 3) with 6 non-zeros
    ///     let (nrow, ncol, max_nnz) = (3, 3, 6);
    ///     let mut coo = CooMatrix::new(nrow, ncol, max_nnz, Sym::No)?;
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
    pub fn mat_vec_mul(&self, v: &mut NumVector<T>, alpha: T, u: &NumVector<T>) -> Result<(), StrError> {
        if u.dim() < self.ncol {
            return Err("u.dim() must be ≥ the number of columns of the matrix");
        }
        if v.dim() < self.nrow {
            return Err("v.dim() must be ≥ the number of rows of the matrix");
        }
        let mirror_required = self.symmetric.triangular();
        v.fill(T::zero());
        for p in 0..self.nnz {
            let i = self.indices_i[p] as usize;
            let j = self.indices_j[p] as usize;
            let aij = self.values[p];
            v[i] += alpha * aij * u[j];
            if mirror_required && i != j {
                v[j] += alpha * aij * u[i];
            }
        }
        Ok(())
    }

    /// Performs the matrix-vector multiplication with update
    ///
    /// ```text
    ///  v  +=  α ⋅  a   ⋅  u
    /// (m)        (m,n)   (n)
    /// ```
    ///
    /// # Input
    ///
    /// * `u` -- Vector with dimension ≥ the number of columns of the matrix
    ///
    /// # Output
    ///
    /// * `v` -- Vector with dimension ≥ the number of rows of the matrix
    ///
    /// # Note
    ///
    /// This method is not highly efficient but should useful with small matrices or verifications.
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::{Matrix, Vector};
    /// use russell_sparse::prelude::*;
    /// use russell_sparse::StrError;
    ///
    /// fn main() -> Result<(), StrError> {
    ///     // set sparse matrix (3 x 3) with 6 non-zeros
    ///     let (nrow, ncol, max_nnz) = (3, 3, 6);
    ///     let mut coo = CooMatrix::new(nrow, ncol, max_nnz, Sym::No)?;
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
    ///     let mut v = Vector::from(&[1000.0, 2000.0, 3000.0]);
    ///     coo.mat_vec_mul_update(&mut v, 1.0, &u)?;
    ///
    ///     // check vector
    ///     let correct_v = "┌      ┐\n\
    ///                      │ 1001 │\n\
    ///                      │ 2005 │\n\
    ///                      │ 3015 │\n\
    ///                      └      ┘";
    ///     assert_eq!(format!("{}", v), correct_v);
    ///     Ok(())
    /// }
    /// ```
    pub fn mat_vec_mul_update(&self, v: &mut NumVector<T>, alpha: T, u: &NumVector<T>) -> Result<(), StrError> {
        if u.dim() < self.ncol {
            return Err("u.dim() must be ≥ the number of columns of the matrix");
        }
        if v.dim() < self.nrow {
            return Err("v.dim() must be ≥ the number of rows of the matrix");
        }
        let mirror_required = self.symmetric.triangular();
        for p in 0..self.nnz {
            let i = self.indices_i[p] as usize;
            let j = self.indices_j[p] as usize;
            let aij = self.values[p];
            v[i] += alpha * aij * u[j];
            if mirror_required && i != j {
                v[j] += alpha * aij * u[i];
            }
        }
        Ok(())
    }

    /// Performs the transpose(matrix)-vector multiplication
    ///
    /// ```text
    ///  v  :=  α ⋅  aᵀ ⋅  u
    /// (m)        (m,n)  (n)
    /// ```
    ///
    /// # Input
    ///
    /// * `u` -- Vector with dimension ≥ the number of rows of the matrix
    ///
    /// # Output
    ///
    /// * `v` -- Vector with dimension ≥ the number of columns of the matrix
    ///
    /// # Note
    ///
    /// This method is not highly efficient but should useful with small matrices or verifications.
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::{Matrix, Vector};
    /// use russell_sparse::prelude::*;
    /// use russell_sparse::StrError;
    ///
    /// fn main() -> Result<(), StrError> {
    ///     // set sparse matrix (3 x 2) with 5 non-zeros
    ///     let (nrow, ncol, max_nnz) = (3, 2, 5);
    ///     let mut coo = CooMatrix::new(nrow, ncol, max_nnz, Sym::No)?;
    ///     coo.put(0, 0, 1.0)?;
    ///     coo.put(1, 0, 2.0)?;
    ///     coo.put(1, 1, 3.0)?;
    ///     coo.put(2, 0, 4.0)?;
    ///     coo.put(2, 1, 5.0)?;
    ///
    ///     // check matrix
    ///     let a = coo.as_dense();
    ///     let correct_a = "┌     ┐\n\
    ///                      │ 1 0 │\n\
    ///                      │ 2 3 │\n\
    ///                      │ 4 5 │\n\
    ///                      └     ┘";
    ///     assert_eq!(format!("{}", a), correct_a);
    ///
    ///     // perform mat-t-vec-mul
    ///     let u = Vector::from(&[1.0, 1.0, 1.0]);
    ///     let mut v = Vector::new(ncol);
    ///     coo.mat_t_vec_mul(&mut v, 1.0, &u)?;
    ///
    ///     // check vector
    ///     let correct_v = "┌   ┐\n\
    ///                      │ 7 │\n\
    ///                      │ 8 │\n\
    ///                      └   ┘";
    ///     assert_eq!(format!("{}", v), correct_v);
    ///     Ok(())
    /// }
    /// ```
    pub fn mat_t_vec_mul(&self, v: &mut NumVector<T>, alpha: T, u: &NumVector<T>) -> Result<(), StrError> {
        if u.dim() < self.nrow {
            return Err("u.dim() must be ≥ the number of rows of the matrix");
        }
        if v.dim() < self.ncol {
            return Err("v.dim() must be ≥ the number of columns of the matrix");
        }
        let mirror_required = self.symmetric.triangular();
        v.fill(T::zero());
        for p in 0..self.nnz {
            let j = self.indices_i[p] as usize;
            let i = self.indices_j[p] as usize;
            let aji = self.values[p];
            v[i] += alpha * aji * u[j];
            if mirror_required && i != j {
                v[j] += alpha * aji * u[i];
            }
        }
        Ok(())
    }

    /// Assigns this matrix to the values of another matrix (scaled)
    ///
    /// Performs:
    ///
    /// ```text
    /// this = α · other
    /// ```
    ///
    /// **Warning:** make sure to allocate `max_nnz ≥ nnz(other)`.
    pub fn assign(&mut self, alpha: T, other: &NumCooMatrix<T>) -> Result<(), StrError> {
        if other.nrow != self.nrow {
            return Err("matrices must have the same nrow");
        }
        if other.ncol != self.ncol {
            return Err("matrices must have the same ncol");
        }
        if other.symmetric != self.symmetric {
            return Err("matrices must have the same symmetric type");
        }
        self.reset();
        for p in 0..other.nnz {
            let i = other.indices_i[p] as usize;
            let j = other.indices_j[p] as usize;
            self.put(i, j, alpha * other.values[p])?;
        }
        Ok(())
    }

    /// Puts the entries of another matrix into this matrix
    ///
    /// Effectively, performs:
    ///
    /// ```text
    /// this += α · other
    /// ```
    ///
    /// # Arguments
    ///
    /// * `alpha` -- scaling factor
    /// * `other` -- the other matrix to be added. It must be at most as large as `this`.
    ///
    /// # Requirements
    ///
    /// * `other.nrow ≤ this.nrow`
    /// * `other.ncol ≤ this.ncol`
    /// * `other.symmetric == this.symmetric`
    ///
    /// # Note
    ///
    /// * make sure to allocate `max_nnz ≥ nnz(this) + nnz(other)`.
    pub fn add(&mut self, alpha: T, other: &NumCooMatrix<T>) -> Result<(), StrError> {
        if other.nrow > self.nrow {
            return Err("other.nrow must be ≤ this.nrow");
        }
        if other.ncol > self.ncol {
            return Err("other.ncol must be ≤ this.ncol");
        }
        if other.symmetric != self.symmetric {
            return Err("matrices must have the same symmetric type");
        }
        for p in 0..other.nnz {
            let i = other.indices_i[p] as usize;
            let j = other.indices_j[p] as usize;
            self.put(i, j, alpha * other.values[p])?;
        }
        Ok(())
    }

    /// Puts a Lagrange block matrix into this matrix
    ///
    /// The resulting matrix will be:
    ///
    /// ```text
    ///          ┌                         ┐
    ///          │ ... ... ... b00 b10 ... │   ┌              ┐
    /// ncol(B)  │ ... ... ... b01 b11 ... │   │ ...   Bᵀ ... │
    ///    +     │ ... ... ... b02 b12 ... │ = │  B   ... ... │
    ///          │ b00 b01 b02 ... ... ... │   │ ...  ... ... │
    /// nrow(B)  │ b10 b11 b12 ... ... ... │   └              ┘
    ///          │ ... ... ... ... ... ... │
    ///          └                         ┘
    ///              ncol(B) + nrow(B)     ≤ ncol(A)
    /// ≤ nrow(A)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `bb` -- The block to be inserted. It must not be symmetric.
    ///
    /// # Requirements
    ///
    /// * This matrix must be allocated with enough space to accommodate the extra `2 nnz(B)`.
    /// * `ncol(B) + nrow(B) ≤ nrow(A)`
    /// * `ncol(B) + nrow(B) ≤ ncol(A)`
    pub fn put_lagrange_block(&mut self, bb: &NumCooMatrix<T>) -> Result<(), StrError> {
        if bb.symmetric != Sym::No {
            return Err("the Lagrange block must not be symmetric");
        }
        if bb.ncol + bb.nrow > self.nrow {
            return Err("ncol(B) + nrow(B) must be ≤ nrow(A)");
        }
        if bb.ncol + bb.nrow > self.ncol {
            return Err("ncol(B) + nrow(B) must be ≤ ncol(A)");
        }
        if self.symmetric == Sym::YesLower {
            for p in 0..bb.nnz {
                let i = bb.indices_i[p] as usize;
                let j = bb.indices_j[p] as usize;
                let x = bb.values[p];
                self.put(bb.ncol + i, j, x)?; // only needs to put B
            }
        } else if self.symmetric == Sym::YesUpper {
            for p in 0..bb.nnz {
                let i = bb.indices_i[p] as usize;
                let j = bb.indices_j[p] as usize;
                let x = bb.values[p];
                self.put(j, bb.ncol + i, x)?; // only needs to put Bᵀ
            }
        } else {
            for p in 0..bb.nnz {
                let i = bb.indices_i[p] as usize;
                let j = bb.indices_j[p] as usize;
                let x = bb.values[p];
                self.put(bb.ncol + i, j, x)?; // puts B
                self.put(j, bb.ncol + i, x)?; // puts Bᵀ
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
    ///     let coo = CooMatrix::new(1, 2, 3, Sym::No)?;
    ///     let (nrow, ncol, nnz, sym) = coo.get_info();
    ///     assert_eq!(nrow, 1);
    ///     assert_eq!(ncol, 2);
    ///     assert_eq!(nnz, 0);
    ///     assert_eq!(sym, Sym::No);
    ///     Ok(())
    /// }
    /// ```
    pub fn get_info(&self) -> (usize, usize, usize, Sym) {
        (self.nrow, self.ncol, self.nnz, self.symmetric)
    }

    /// Get an access to the row indices
    ///
    /// ```text
    /// row_indices.len() == nnz
    /// ```
    pub fn get_row_indices(&self) -> &[i32] {
        &self.indices_i[..self.nnz]
    }

    /// Get an access to the column indices
    ///
    /// ```text
    /// col_indices.len() == nnz
    /// ```
    pub fn get_col_indices(&self) -> &[i32] {
        &self.indices_j[..self.nnz]
    }

    /// Get an access to the values
    ///
    /// ```text
    /// values.len() == nnz
    /// ```
    pub fn get_values(&self) -> &[T] {
        &self.values[..self.nnz]
    }

    /// Get a mutable access the values
    ///
    /// ```text
    /// values.len() == nnz
    /// ```
    ///
    /// Note: the values may be modified externally, but not the indices.
    pub fn get_values_mut(&mut self) -> &mut [T] {
        &mut self.values[..self.nnz]
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::NumCooMatrix;
    use crate::{Samples, Sym};
    use russell_lab::{complex_vec_approx_eq, cpx, vec_approx_eq, ComplexVector, NumMatrix, NumVector};

    #[test]
    fn new_captures_errors() {
        assert_eq!(
            NumCooMatrix::<f32>::new(0, 1, 3, Sym::No).err(),
            Some("nrow must be ≥ 1")
        );
        assert_eq!(
            NumCooMatrix::<f32>::new(1, 0, 3, Sym::No).err(),
            Some("ncol must be ≥ 1")
        );
        assert_eq!(
            NumCooMatrix::<f32>::new(1, 1, 0, Sym::No).err(),
            Some("max_nnz must be ≥ 1")
        );
        assert_eq!(
            NumCooMatrix::<f32>::new(2, 1, 1, Sym::YesFull).err(),
            Some("symmetric storage requires a square matrix")
        );
        assert_eq!(
            NumCooMatrix::<f32>::new(2, 1, 1, Sym::YesLower).err(),
            Some("symmetric storage requires a square matrix")
        );
        assert_eq!(
            NumCooMatrix::<f32>::new(2, 1, 1, Sym::YesUpper).err(),
            Some("symmetric storage requires a square matrix")
        );
    }

    #[test]
    fn new_works() {
        let coo = NumCooMatrix::<f32>::new(1, 1, 3, Sym::No).unwrap();
        assert_eq!(coo.symmetric, Sym::No);
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
        assert_eq!(NumCooMatrix::<f32>::from(0, 1, vec![ 0], vec![ 0], vec![0.0], Sym::No).err(), Some("nrow must be ≥ 1"));
        assert_eq!(NumCooMatrix::<f32>::from(1, 0, vec![ 0], vec![ 0], vec![0.0], Sym::No).err(), Some("ncol must be ≥ 1"));
        assert_eq!(NumCooMatrix::<f32>::from(1, 1, vec![  ], vec![ 0], vec![0.0], Sym::No).err(), Some("nnz must be ≥ 1"));
        assert_eq!(NumCooMatrix::<f32>::from(1, 1, vec![ 0], vec![  ], vec![0.0], Sym::No).err(), Some("col_indices.len() must be = nnz"));
        assert_eq!(NumCooMatrix::<f32>::from(1, 1, vec![ 0], vec![ 0], vec![   ], Sym::No).err(), Some("values.len() must be = nnz"));
        assert_eq!(NumCooMatrix::<f32>::from(1, 1, vec![-1], vec![ 0], vec![0.0], Sym::No).err(), Some("row index is out-of-range"));
        assert_eq!(NumCooMatrix::<f32>::from(1, 1, vec![ 1], vec![ 0], vec![0.0], Sym::No).err(), Some("row index is out-of-range"));
        assert_eq!(NumCooMatrix::<f32>::from(1, 1, vec![ 0], vec![-1], vec![0.0], Sym::No).err(), Some("col index is out-of-range"));
        assert_eq!(NumCooMatrix::<f32>::from(1, 1, vec![ 0], vec![ 1], vec![0.0], Sym::No).err(), Some("col index is out-of-range"));
    }

    #[test]
    fn from_works() {
        let coo = NumCooMatrix::<f32>::from(1, 1, vec![0], vec![0], vec![123.0], Sym::No).unwrap();
        assert_eq!(coo.symmetric, Sym::No);
        assert_eq!(coo.nrow, 1);
        assert_eq!(coo.ncol, 1);
        assert_eq!(coo.nnz, 1);
        assert_eq!(coo.max_nnz, 1);
        assert_eq!(coo.indices_i, &[0]);
        assert_eq!(coo.indices_j, &[0]);
        assert_eq!(coo.values, &[123.0]);
        let coo = NumCooMatrix::<f32>::from(1, 1, vec![0], vec![0], vec![123.0], Sym::YesFull).unwrap();
        assert_eq!(coo.symmetric, Sym::YesFull);
    }

    #[test]
    fn get_info_works() {
        let coo = NumCooMatrix::<f32>::new(1, 2, 10, Sym::No).unwrap();
        let (nrow, ncol, nnz, sym) = coo.get_info();
        assert_eq!(nrow, 1);
        assert_eq!(ncol, 2);
        assert_eq!(nnz, 0);
        assert_eq!(sym, Sym::No);
    }

    #[test]
    fn put_fails_on_wrong_values() {
        let mut coo = NumCooMatrix::<i32>::new(1, 1, 1, Sym::No).unwrap();
        assert_eq!(
            coo.put(1, 0, 0).err(),
            Some("COO matrix: index of row is outside range")
        );
        assert_eq!(
            coo.put(0, 1, 0).err(),
            Some("COO matrix: index of column is outside range")
        );
        assert_eq!(coo.put(0, 0, 0).err(), None); // << will take all spots
        assert_eq!(
            coo.put(0, 0, 0).err(),
            Some("COO matrix: max number of items has been reached")
        );
        let mut coo = NumCooMatrix::<u8>::new(2, 2, 4, Sym::YesLower).unwrap();
        assert_eq!(
            coo.put(0, 1, 0).err(),
            Some("COO matrix: j > i is incorrect for lower triangular storage")
        );
        let mut coo = NumCooMatrix::<u8>::new(2, 2, 4, Sym::YesUpper).unwrap();
        assert_eq!(
            coo.put(1, 0, 0).err(),
            Some("COO matrix: j < i is incorrect for upper triangular storage")
        );
    }

    #[test]
    fn put_works() {
        let mut coo = NumCooMatrix::<f32>::new(3, 3, 5, Sym::No).unwrap();
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
        let mut coo = NumCooMatrix::<f64>::new(2, 2, 4, Sym::No).unwrap();
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
        let mut coo = NumCooMatrix::<f64>::new(1, 1, 1, Sym::No).unwrap();
        coo.put(0, 0, 123.0).unwrap();
        let mut a_2x1 = NumMatrix::<f64>::new(2, 1);
        let mut a_1x2 = NumMatrix::<f64>::new(1, 2);
        assert_eq!(coo.to_dense(&mut a_2x1), Err("wrong matrix dimensions"));
        assert_eq!(coo.to_dense(&mut a_1x2), Err("wrong matrix dimensions"));
    }

    #[test]
    fn to_dense_works() {
        let mut coo = NumCooMatrix::<f32>::new(3, 3, 5, Sym::No).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(0, 1, 2.0).unwrap();
        coo.put(1, 0, 3.0).unwrap();
        coo.put(1, 1, 4.0).unwrap();
        coo.put(2, 2, 5.0).unwrap();
        let mut a = NumMatrix::<f32>::new(3, 3);
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
        let empty = NumCooMatrix::<f64>::new(2, 2, 3, Sym::No).unwrap();
        let mat = empty.as_dense();
        assert_eq!(mat.as_data(), &[0.0, 0.0, 0.0, 0.0]);
        // single component matrix
        let mut single = NumCooMatrix::<f64>::new(1, 1, 1, Sym::No).unwrap();
        single.put(0, 0, 123.0).unwrap();
        let mat = single.as_dense();
        assert_eq!(mat.as_data(), &[123.0]);
    }

    #[test]
    fn to_dense_with_duplicates_works() {
        // allocate a square matrix
        let (nrow, ncol, nnz) = (5, 5, 13);
        let mut coo = NumCooMatrix::<f32>::new(nrow, ncol, nnz, Sym::No).unwrap();
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
        let mut a = NumMatrix::<f32>::new(nrow as usize, ncol as usize);
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
        let mut coo = NumCooMatrix::<f64>::new(3, 3, 4, Sym::YesLower).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(1, 0, 2.0).unwrap();
        coo.put(1, 1, 3.0).unwrap();
        coo.put(2, 1, 4.0).unwrap();
        let mut a = NumMatrix::<f64>::new(3, 3);
        coo.to_dense(&mut a).unwrap();
        let correct = "┌       ┐\n\
                       │ 1 2 0 │\n\
                       │ 2 3 4 │\n\
                       │ 0 4 0 │\n\
                       └       ┘";
        assert_eq!(format!("{}", a), correct);
    }

    #[test]
    fn to_dense_symmetric_upper_works() {
        let mut coo = NumCooMatrix::<f64>::new(3, 3, 4, Sym::YesUpper).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(0, 1, 2.0).unwrap();
        coo.put(1, 1, 3.0).unwrap();
        coo.put(1, 2, 4.0).unwrap();
        let mut a = NumMatrix::<f64>::new(3, 3);
        coo.to_dense(&mut a).unwrap();
        let correct = "┌       ┐\n\
                       │ 1 2 0 │\n\
                       │ 2 3 4 │\n\
                       │ 0 4 0 │\n\
                       └       ┘";
        assert_eq!(format!("{}", a), correct);
    }

    #[test]
    fn assign_capture_errors() {
        let nnz_a = 1;
        let nnz_b = 2; // wrong: must be ≤ nnz_a
        let mut a_1x2 = NumCooMatrix::<u32>::new(1, 2, nnz_a, Sym::No).unwrap();
        let b_2x1 = NumCooMatrix::<u32>::new(2, 1, nnz_b, Sym::No).unwrap();
        let b_1x3 = NumCooMatrix::<u32>::new(1, 3, nnz_b, Sym::No).unwrap();
        let mut b_1x2 = NumCooMatrix::<u32>::new(1, 2, nnz_b, Sym::No).unwrap();
        a_1x2.put(0, 0, 123).unwrap();
        b_1x2.put(0, 0, 456).unwrap();
        b_1x2.put(0, 1, 654).unwrap();
        assert_eq!(a_1x2.assign(2, &b_2x1).err(), Some("matrices must have the same nrow"));
        assert_eq!(a_1x2.assign(2, &b_1x3).err(), Some("matrices must have the same ncol"));
        assert_eq!(
            a_1x2.assign(2, &b_1x2).err(),
            Some("COO matrix: max number of items has been reached")
        );
        let mut a_2x2 = NumCooMatrix::<u32>::new(2, 2, nnz_a, Sym::YesLower).unwrap();
        let b_2x2 = NumCooMatrix::<u32>::new(2, 2, nnz_b, Sym::YesFull).unwrap();
        assert_eq!(
            a_2x2.assign(2, &b_2x2).err(),
            Some("matrices must have the same symmetric type")
        );
    }

    #[test]
    fn assign_works() {
        let nnz = 2;
        let mut a = NumCooMatrix::<f64>::new(3, 2, nnz, Sym::No).unwrap();
        let mut b = NumCooMatrix::<f64>::new(3, 2, nnz, Sym::No).unwrap();
        a.put(2, 1, 1000.0).unwrap();
        b.put(0, 0, 10.0).unwrap();
        b.put(2, 1, 20.0).unwrap();
        assert_eq!(
            format!("{}", a.as_dense()),
            "┌           ┐\n\
             │    0    0 │\n\
             │    0    0 │\n\
             │    0 1000 │\n\
             └           ┘"
        );
        a.assign(5.0, &b).unwrap();
        assert_eq!(
            format!("{}", a.as_dense()),
            "┌         ┐\n\
             │  50   0 │\n\
             │   0   0 │\n\
             │   0 100 │\n\
             └         ┘"
        );
    }

    #[test]
    fn add_capture_errors() {
        let nnz_a = 1;
        let nnz_b = 1;
        let mut a_1x2 = NumCooMatrix::<u32>::new(1, 2, nnz_a /* + nnz_b */, Sym::No).unwrap();
        let b_2x1 = NumCooMatrix::<u32>::new(2, 1, nnz_b, Sym::No).unwrap();
        let b_1x3 = NumCooMatrix::<u32>::new(1, 3, nnz_b, Sym::No).unwrap();
        let mut b_1x2 = NumCooMatrix::<u32>::new(1, 2, nnz_b, Sym::No).unwrap();
        a_1x2.put(0, 0, 123).unwrap();
        b_1x2.put(0, 0, 456).unwrap();
        assert_eq!(a_1x2.add(2, &b_2x1).err(), Some("other.nrow must be ≤ this.nrow"));
        assert_eq!(a_1x2.add(2, &b_1x3).err(), Some("other.ncol must be ≤ this.ncol"));
        assert_eq!(
            a_1x2.add(2, &b_1x2).err(),
            Some("COO matrix: max number of items has been reached")
        );
        let mut a_2x2 = NumCooMatrix::<u32>::new(2, 2, 1, Sym::YesLower).unwrap();
        let b_2x2 = NumCooMatrix::<u32>::new(2, 2, 1, Sym::YesFull).unwrap();
        assert_eq!(
            a_2x2.add(2, &b_2x2).err(),
            Some("matrices must have the same symmetric type")
        );
    }

    #[test]
    fn add_works() {
        let nnz_a = 1;
        let nnz_b = 2;
        let mut a = NumCooMatrix::<f64>::new(3, 2, nnz_a + nnz_b, Sym::No).unwrap();
        let mut b = NumCooMatrix::<f64>::new(3, 2, nnz_b, Sym::No).unwrap();
        a.put(2, 1, 1000.0).unwrap();
        b.put(0, 0, 10.0).unwrap();
        b.put(2, 1, 20.0).unwrap();
        assert_eq!(
            format!("{}", a.as_dense()),
            "┌           ┐\n\
             │    0    0 │\n\
             │    0    0 │\n\
             │    0 1000 │\n\
             └           ┘"
        );
        a.add(5.0, &b).unwrap();
        assert_eq!(
            format!("{}", a.as_dense()),
            "┌           ┐\n\
             │   50    0 │\n\
             │    0    0 │\n\
             │    0 1100 │\n\
             └           ┘"
        );
    }

    #[test]
    fn put_lagrange_block_capture_errors() {
        let bb = NumCooMatrix::<f64>::new(1, 1, 6, Sym::YesFull).unwrap();
        let mut aa = NumCooMatrix::<f64>::new(2, 3, 1, Sym::No).unwrap();
        assert_eq!(
            aa.put_lagrange_block(&bb).err(),
            Some("the Lagrange block must not be symmetric")
        );

        let bb = NumCooMatrix::<f64>::new(2, 1, 6, Sym::No).unwrap();

        let mut aa = NumCooMatrix::<f64>::new(2, 3, 1, Sym::No).unwrap();
        assert_eq!(
            aa.put_lagrange_block(&bb).err(),
            Some("ncol(B) + nrow(B) must be ≤ nrow(A)")
        );

        let mut aa = NumCooMatrix::<f64>::new(3, 2, 1, Sym::No).unwrap();
        assert_eq!(
            aa.put_lagrange_block(&bb).err(),
            Some("ncol(B) + nrow(B) must be ≤ ncol(A)")
        );

        let mut aa = NumCooMatrix::<f64>::new(2, 2, 1, Sym::No).unwrap();
        let mut bb = NumCooMatrix::<f64>::new(1, 1, 6, Sym::No).unwrap();
        aa.put(0, 0, 1000.0).unwrap();
        bb.put(0, 0, 10.0).unwrap();
        assert_eq!(
            aa.put_lagrange_block(&bb).err(),
            Some("COO matrix: max number of items has been reached")
        );
    }

    #[test]
    fn put_lagrange_block_works() {
        // definitions
        let (nrow_a, ncol_a) = (6, 6);
        let (nrow_b, ncol_b) = (2, 3);
        let nnz_k = nrow_a * ncol_a;
        let nnz_b = nrow_b * ncol_b;
        let nnz_a = nnz_k + 2 * nnz_b;
        let mut aa = NumCooMatrix::<f64>::new(nrow_a, ncol_a, nnz_a, Sym::No).unwrap();
        let mut bb = NumCooMatrix::<f64>::new(nrow_b, ncol_b, nnz_b, Sym::No).unwrap();

        // populate A
        for i in 0..nrow_a {
            for j in 0..ncol_a {
                aa.put(i, j, (10_000 * (i + 1) + 1_000 * (j + 1)) as f64).unwrap();
            }
        }
        assert_eq!(
            format!("{}", aa.as_dense()),
            "┌                                     ┐\n\
             │ 11000 12000 13000 14000 15000 16000 │\n\
             │ 21000 22000 23000 24000 25000 26000 │\n\
             │ 31000 32000 33000 34000 35000 36000 │\n\
             │ 41000 42000 43000 44000 45000 46000 │\n\
             │ 51000 52000 53000 54000 55000 56000 │\n\
             │ 61000 62000 63000 64000 65000 66000 │\n\
             └                                     ┘"
        );

        // populate B
        for i in 0..nrow_b {
            for j in 0..ncol_b {
                bb.put(i, j, (10 * (i + 1) + j + 1) as f64).unwrap();
            }
        }
        assert_eq!(
            format!("{}", bb.as_dense()),
            "┌          ┐\n\
             │ 11 12 13 │\n\
             │ 21 22 23 │\n\
             └          ┘"
        );

        // put B and Bᵀ into A
        aa.put_lagrange_block(&bb).unwrap();
        assert_eq!(
            format!("{}", aa.as_dense()),
            "┌                                     ┐\n\
             │ 11000 12000 13000 14011 15021 16000 │\n\
             │ 21000 22000 23000 24012 25022 26000 │\n\
             │ 31000 32000 33000 34013 35023 36000 │\n\
             │ 41011 42012 43013 44000 45000 46000 │\n\
             │ 51021 52022 53023 54000 55000 56000 │\n\
             │ 61000 62000 63000 64000 65000 66000 │\n\
             └                                     ┘"
        );
    }

    #[test]
    fn put_lagrange_block_sym_lower_works() {
        // definitions
        let na = 6; // square matrix
        let (nrow_b, ncol_b) = (2, 3);
        let nnz_k = na * (na + 1) / 2; // triangle including the diagonal
        let nnz_b = nrow_b * ncol_b;
        let nnz_a = nnz_k + 2 * nnz_b;
        let mut aa = NumCooMatrix::<f64>::new(na, na, nnz_a, Sym::YesLower).unwrap();
        let mut bb = NumCooMatrix::<f64>::new(nrow_b, ncol_b, nnz_b, Sym::No).unwrap();

        // populate A
        let mut count = 0;
        for i in 0..na {
            for j in 0..=i {
                aa.put(i, j, (10_000 * (i + 1) + 1_000 * (j + 1)) as f64).unwrap();
                count += 1;
            }
        }
        assert_eq!(aa.nnz, count);
        assert_eq!(
            format!("{}", aa.as_dense()),
            "┌                                     ┐\n\
             │ 11000 21000 31000 41000 51000 61000 │\n\
             │ 21000 22000 32000 42000 52000 62000 │\n\
             │ 31000 32000 33000 43000 53000 63000 │\n\
             │ 41000 42000 43000 44000 54000 64000 │\n\
             │ 51000 52000 53000 54000 55000 65000 │\n\
             │ 61000 62000 63000 64000 65000 66000 │\n\
             └                                     ┘"
        );

        // populate B
        for i in 0..nrow_b {
            for j in 0..ncol_b {
                bb.put(i, j, (10 * (i + 1) + j + 1) as f64).unwrap();
            }
        }
        assert_eq!(
            format!("{}", bb.as_dense()),
            "┌          ┐\n\
             │ 11 12 13 │\n\
             │ 21 22 23 │\n\
             └          ┘"
        );

        // put B and Bᵀ into A
        aa.put_lagrange_block(&bb).unwrap();
        assert_eq!(
            format!("{}", aa.as_dense()),
            "┌                                     ┐\n\
             │ 11000 21000 31000 41011 51021 61000 │\n\
             │ 21000 22000 32000 42012 52022 62000 │\n\
             │ 31000 32000 33000 43013 53023 63000 │\n\
             │ 41011 42012 43013 44000 54000 64000 │\n\
             │ 51021 52022 53023 54000 55000 65000 │\n\
             │ 61000 62000 63000 64000 65000 66000 │\n\
             └                                     ┘"
        );
    }

    #[test]
    fn put_lagrange_block_sym_upper_works() {
        // definitions
        let na = 6; // square matrix
        let (nrow_b, ncol_b) = (2, 3);
        let nnz_k = na * (na + 1) / 2; // triangle including the diagonal
        let nnz_b = nrow_b * ncol_b;
        let nnz_a = nnz_k + 2 * nnz_b;
        let mut aa = NumCooMatrix::<f64>::new(na, na, nnz_a, Sym::YesUpper).unwrap();
        let mut bb = NumCooMatrix::<f64>::new(nrow_b, ncol_b, nnz_b, Sym::No).unwrap();

        // populate A
        let mut count = 0;
        for i in 0..na {
            for j in i..na {
                aa.put(i, j, (10_000 * (i + 1) + 1_000 * (j + 1)) as f64).unwrap();
                count += 1;
            }
        }
        assert_eq!(aa.nnz, count);
        assert_eq!(
            format!("{}", aa.as_dense()),
            "┌                                     ┐\n\
             │ 11000 12000 13000 14000 15000 16000 │\n\
             │ 12000 22000 23000 24000 25000 26000 │\n\
             │ 13000 23000 33000 34000 35000 36000 │\n\
             │ 14000 24000 34000 44000 45000 46000 │\n\
             │ 15000 25000 35000 45000 55000 56000 │\n\
             │ 16000 26000 36000 46000 56000 66000 │\n\
             └                                     ┘"
        );

        // populate B
        for i in 0..nrow_b {
            for j in 0..ncol_b {
                bb.put(i, j, (10 * (i + 1) + j + 1) as f64).unwrap();
            }
        }
        assert_eq!(
            format!("{}", bb.as_dense()),
            "┌          ┐\n\
             │ 11 12 13 │\n\
             │ 21 22 23 │\n\
             └          ┘"
        );

        // put B and Bᵀ into A
        aa.put_lagrange_block(&bb).unwrap();
        assert_eq!(
            format!("{}", aa.as_dense()),
            "┌                                     ┐\n\
             │ 11000 12000 13000 14011 15021 16000 │\n\
             │ 12000 22000 23000 24012 25022 26000 │\n\
             │ 13000 23000 33000 34013 35023 36000 │\n\
             │ 14011 24012 34013 44000 45000 46000 │\n\
             │ 15021 25022 35023 45000 55000 56000 │\n\
             │ 16000 26000 36000 46000 56000 66000 │\n\
             └                                     ┘"
        );
    }

    #[test]
    fn put_lagrange_block_sym_full_works() {
        // definitions
        let na = 6; // square matrix
        let (nrow_b, ncol_b) = (2, 3);
        let nnz_k = na * na;
        let nnz_b = nrow_b * ncol_b;
        let nnz_a = nnz_k + 2 * nnz_b;
        let mut aa = NumCooMatrix::<f64>::new(na, na, nnz_a, Sym::YesFull).unwrap();
        let mut bb = NumCooMatrix::<f64>::new(nrow_b, ncol_b, nnz_b, Sym::No).unwrap();

        // populate A
        for i in 0..na {
            for j in i..na {
                let val = (10_000 * (i + 1) + 1_000 * (j + 1)) as f64;
                aa.put(i, j, val).unwrap();
                if i != j {
                    aa.put(j, i, val).unwrap();
                }
            }
        }
        assert_eq!(
            format!("{}", aa.as_dense()),
            "┌                                     ┐\n\
             │ 11000 12000 13000 14000 15000 16000 │\n\
             │ 12000 22000 23000 24000 25000 26000 │\n\
             │ 13000 23000 33000 34000 35000 36000 │\n\
             │ 14000 24000 34000 44000 45000 46000 │\n\
             │ 15000 25000 35000 45000 55000 56000 │\n\
             │ 16000 26000 36000 46000 56000 66000 │\n\
             └                                     ┘"
        );

        // populate B
        for i in 0..nrow_b {
            for j in 0..ncol_b {
                bb.put(i, j, (10 * (i + 1) + j + 1) as f64).unwrap();
            }
        }
        assert_eq!(
            format!("{}", bb.as_dense()),
            "┌          ┐\n\
             │ 11 12 13 │\n\
             │ 21 22 23 │\n\
             └          ┘"
        );

        // put B and Bᵀ into A
        aa.put_lagrange_block(&bb).unwrap();
        assert_eq!(
            format!("{}", aa.as_dense()),
            "┌                                     ┐\n\
             │ 11000 12000 13000 14011 15021 16000 │\n\
             │ 12000 22000 23000 24012 25022 26000 │\n\
             │ 13000 23000 33000 34013 35023 36000 │\n\
             │ 14011 24012 34013 44000 45000 46000 │\n\
             │ 15021 25022 35023 45000 55000 56000 │\n\
             │ 16000 26000 36000 46000 56000 66000 │\n\
             └                                     ┘"
        );
    }

    #[test]
    fn getters_are_correct() {
        let (coo, _, _, _) = Samples::rectangular_1x2(false, false);
        assert_eq!(coo.get_info(), (1, 2, 2, Sym::No));
        assert_eq!(coo.get_row_indices(), &[0, 0]);
        assert_eq!(coo.get_col_indices(), &[0, 1]);
        assert_eq!(coo.get_values(), &[10.0, 20.0]);

        let mut coo = NumCooMatrix::<f64>::new(2, 1, 2, Sym::No).unwrap();
        coo.put(0, 0, 123.0).unwrap();
        coo.put(1, 0, 456.0).unwrap();
        assert_eq!(coo.get_values_mut(), &[123.0, 456.0]);
        let x = coo.get_values_mut();
        x.reverse();
        assert_eq!(coo.get_values_mut(), &[456.0, 123.0]);
    }

    #[test]
    fn derive_methods_work() {
        let (coo, _, _, _) = Samples::tiny_1x1();
        let mut clone = coo.clone();
        clone.values[0] *= 2.0;
        assert_eq!(coo.values[0], 123.0);
        assert_eq!(clone.values[0], 246.0);
        assert!(format!("{:?}", coo).len() > 0);
        let json = serde_json::to_string(&coo).unwrap();
        assert_eq!(
            json,
            r#"{"symmetric":"No","nrow":1,"ncol":1,"nnz":1,"max_nnz":1,"indices_i":[0],"indices_j":[0],"values":[123.0]}"#
        );
        let from_json: NumCooMatrix<f64> = serde_json::from_str(&json).unwrap();
        assert_eq!(from_json.symmetric, coo.symmetric);
        assert_eq!(from_json.nrow, coo.nrow);
        assert_eq!(from_json.ncol, coo.ncol);
        assert_eq!(from_json.nnz, coo.nnz);
        assert_eq!(from_json.max_nnz, coo.max_nnz);
        assert_eq!(from_json.indices_i, coo.indices_i);
        assert_eq!(from_json.indices_j, coo.indices_j);
        assert_eq!(from_json.values, coo.values);
    }

    // --- mat_vec_mul (start) ---

    #[test]
    fn mat_vec_mul_captures_errors() {
        let mut coo = NumCooMatrix::<u8>::new(2, 2, 1, Sym::No).unwrap();
        coo.put(0, 0, 123).unwrap();
        let u = NumVector::<u8>::new(1);
        let mut v = NumVector::<u8>::new(coo.nrow);
        assert_eq!(
            coo.mat_vec_mul(&mut v, 1, &u).err(),
            Some("u.dim() must be ≥ the number of columns of the matrix")
        );
        let u = NumVector::<u8>::new(2);
        let mut v = NumVector::<u8>::new(1);
        assert_eq!(
            coo.mat_vec_mul(&mut v, 1, &u).err(),
            Some("v.dim() must be ≥ the number of rows of the matrix")
        );
    }

    #[test]
    fn mat_vec_mul_works() {
        //  1.0  2.0  3.0
        //  0.1  0.2  0.3
        // 10.0 20.0 30.0
        let mut coo = NumCooMatrix::<f64>::new(3, 3, 9, Sym::No).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(0, 1, 2.0).unwrap();
        coo.put(0, 2, 3.0).unwrap();
        coo.put(1, 0, 0.1).unwrap();
        coo.put(1, 1, 0.2).unwrap();
        coo.put(1, 2, 0.3).unwrap();
        coo.put(2, 0, 10.0).unwrap();
        coo.put(2, 1, 20.0).unwrap();
        coo.put(2, 2, 30.0).unwrap();
        let u = NumVector::<f64>::from(&[0.1, 0.2, 0.3]);
        let mut v = NumVector::<f64>::new(coo.nrow);
        coo.mat_vec_mul(&mut v, 2.0, &u).unwrap();
        let correct_v = &[2.8, 0.28, 28.0];
        vec_approx_eq(&v, correct_v, 1e-15);

        // call mat_vec_mul again to make sure the vector is filled with zeros before the sum
        coo.mat_vec_mul(&mut v, 2.0, &u).unwrap();
        vec_approx_eq(&v, correct_v, 1e-15);

        // single component matrix
        let mut single = NumCooMatrix::<f64>::new(1, 1, 1, Sym::No).unwrap();
        single.put(0, 0, 123.0).unwrap();
        let u = NumVector::from(&[2.0]);
        let mut v = NumVector::<f64>::new(1);
        single.mat_vec_mul(&mut v, 2.0, &u).unwrap();
        assert_eq!(v.as_data(), &[492.0]);
    }

    #[test]
    fn mat_vec_mul_symmetric_lower_works() {
        // 2
        // 1  2     sym
        // 1  2  9
        // 3  1  1  7
        // 2  1  5  1  8
        let (nrow, ncol, nnz) = (5, 5, 15);
        let mut coo = NumCooMatrix::<f64>::new(nrow, ncol, nnz, Sym::YesLower).unwrap();
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
        let u = NumVector::<f64>::from(&[-629.0 / 98.0, 237.0 / 49.0, -53.0 / 49.0, 62.0 / 49.0, 23.0 / 14.0]);
        let mut v = NumVector::<f64>::new(coo.nrow);
        coo.mat_vec_mul(&mut v, 2.0, &u).unwrap();
        let correct_v = &[-4.0, 8.0, 6.0, -10.0, 2.0];
        vec_approx_eq(&v, correct_v, 1e-14);
    }

    #[test]
    fn mat_vec_mul_symmetric_full_works() {
        // 2  1  1  3  2
        // 1  2  2  1  1
        // 1  2  9  1  5
        // 3  1  1  7  1
        // 2  1  5  1  8
        let (nrow, ncol, nnz) = (5, 5, 25);
        let mut coo = NumCooMatrix::<f64>::new(nrow, ncol, nnz, Sym::YesFull).unwrap();
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
        let u = NumVector::<f64>::from(&[-629.0 / 98.0, 237.0 / 49.0, -53.0 / 49.0, 62.0 / 49.0, 23.0 / 14.0]);
        let mut v = NumVector::<f64>::new(coo.nrow);
        coo.mat_vec_mul(&mut v, 2.0, &u).unwrap();
        let correct_v = &[-4.0, 8.0, 6.0, -10.0, 2.0];
        vec_approx_eq(&v, correct_v, 1e-14);
    }

    #[test]
    fn mat_vec_mul_positive_definite_works() {
        //  2  -1              2     ...
        // -1   2  -1    =>   -1   2
        //     -1   2             -1   2
        let (nrow, ncol, nnz) = (3, 3, 5);
        let mut coo = NumCooMatrix::<f64>::new(nrow, ncol, nnz, Sym::YesLower).unwrap();
        coo.put(0, 0, 2.0).unwrap();
        coo.put(1, 1, 2.0).unwrap();
        coo.put(2, 2, 2.0).unwrap();
        coo.put(1, 0, -1.0).unwrap();
        coo.put(2, 1, -1.0).unwrap();
        let u = NumVector::<f64>::from(&[5.0, 8.0, 7.0]);
        let mut v = NumVector::<f64>::new(coo.nrow);
        coo.mat_vec_mul(&mut v, 2.0, &u).unwrap();
        let correct_v = &[4.0, 8.0, 12.0];
        vec_approx_eq(&v, correct_v, 1e-15);
    }

    #[test]
    fn mat_vec_mul_rectangular_works() {
        //  5  -2  .  1
        // 10  -4  .  2
        // 15  -6  .  3
        let (coo, _, _, _) = Samples::rectangular_3x4();
        let u = NumVector::<f64>::from(&[1.0, 2.0, 3.0, -5.0]);
        let mut v = NumVector::<f64>::new(coo.nrow);
        coo.mat_vec_mul(&mut v, 2.0, &u).unwrap();
        let correct = &[-8.0, -16.0, -24.0];
        vec_approx_eq(&v, correct, 1e-15);
        // call mat_vec_mul again to make sure the vector is filled with zeros before the sum
        coo.mat_vec_mul(&mut v, 2.0, &u).unwrap();
        vec_approx_eq(&v, correct, 1e-15);
    }

    #[test]
    fn mat_vec_mul_complex_works() {
        // 4+4i    .     2+2i
        //  .      1     3+3i
        //  .     5+5i   1+1i
        //  1      .      .
        let (coo, _, _, _) = Samples::complex_rectangular_4x3();
        let u = ComplexVector::from(&[cpx!(1.0, 1.0), cpx!(3.0, 1.0), cpx!(5.0, -1.0)]);
        let mut v = ComplexVector::new(coo.nrow);
        coo.mat_vec_mul(&mut v, cpx!(2.0, 4.0), &u).unwrap();
        let correct = &[
            cpx!(-40.0, 80.0),
            cpx!(-10.0, 110.0),
            cpx!(-64.0, 112.0),
            cpx!(-2.0, 6.0),
        ];
        complex_vec_approx_eq(&v, correct, 1e-15);
        // call mat_vec_mul again to make sure the vector is filled with zeros before the sum
        coo.mat_vec_mul(&mut v, cpx!(2.0, 4.0), &u).unwrap();
        complex_vec_approx_eq(&v, correct, 1e-15);
    }

    // --- mat_vec_mul (end) ---

    // --- mat_vec_mul_update (start) ---

    #[test]
    fn mat_vec_mul_update_captures_errors() {
        let mut coo = NumCooMatrix::<u8>::new(2, 2, 1, Sym::No).unwrap();
        coo.put(0, 0, 123).unwrap();
        let u = NumVector::<u8>::new(1);
        let mut v = NumVector::<u8>::new(coo.nrow);
        assert_eq!(
            coo.mat_vec_mul_update(&mut v, 1, &u).err(),
            Some("u.dim() must be ≥ the number of columns of the matrix")
        );
        let u = NumVector::<u8>::new(2);
        let mut v = NumVector::<u8>::new(1);
        assert_eq!(
            coo.mat_vec_mul_update(&mut v, 1, &u).err(),
            Some("v.dim() must be ≥ the number of rows of the matrix")
        );
    }

    #[test]
    fn mat_vec_mul_update_works() {
        //  1.0  2.0  3.0
        //  0.1  0.2  0.3
        // 10.0 20.0 30.0
        let mut coo = NumCooMatrix::<f64>::new(3, 3, 9, Sym::No).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(0, 1, 2.0).unwrap();
        coo.put(0, 2, 3.0).unwrap();
        coo.put(1, 0, 0.1).unwrap();
        coo.put(1, 1, 0.2).unwrap();
        coo.put(1, 2, 0.3).unwrap();
        coo.put(2, 0, 10.0).unwrap();
        coo.put(2, 1, 20.0).unwrap();
        coo.put(2, 2, 30.0).unwrap();
        let u = NumVector::<f64>::from(&[0.1, 0.2, 0.3]);
        let mut v = NumVector::<f64>::new(coo.nrow);
        coo.mat_vec_mul_update(&mut v, 2.0, &u).unwrap();
        let correct_v = &[2.8, 0.28, 28.0];
        vec_approx_eq(&v, correct_v, 1e-15);

        // call mat_vec_mul_update again
        let mut v = NumVector::<f64>::from(&[1000.0, 2000.0, 3000.0]);
        coo.mat_vec_mul_update(&mut v, 2.0, &u).unwrap();
        let correct_v = &[1002.8, 2000.28, 3028.0];
        vec_approx_eq(&v, correct_v, 1e-15);

        // single component matrix
        let mut single = NumCooMatrix::<f64>::new(1, 1, 1, Sym::No).unwrap();
        single.put(0, 0, 123.0).unwrap();
        let u = NumVector::from(&[2.0]);
        let mut v = NumVector::<f64>::new(1);
        single.mat_vec_mul_update(&mut v, 2.0, &u).unwrap();
        assert_eq!(v.as_data(), &[492.0]);
    }

    #[test]
    fn mat_vec_mul_update_symmetric_lower_works() {
        // 2
        // 1  2     sym
        // 1  2  9
        // 3  1  1  7
        // 2  1  5  1  8
        let (nrow, ncol, nnz) = (5, 5, 15);
        let mut coo = NumCooMatrix::<f64>::new(nrow, ncol, nnz, Sym::YesLower).unwrap();
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
        let u = NumVector::<f64>::from(&[-629.0 / 98.0, 237.0 / 49.0, -53.0 / 49.0, 62.0 / 49.0, 23.0 / 14.0]);
        let mut v = NumVector::<f64>::from(&[1004.0, 2000.0, 3000.0, 4010.0, 5000.0]);
        coo.mat_vec_mul_update(&mut v, 2.0, &u).unwrap();
        let correct_v = &[1000.0, 2008.0, 3006.0, 4000.0, 5002.0];
        vec_approx_eq(&v, correct_v, 1e-12);
    }

    #[test]
    fn mat_vec_mul_update_symmetric_full_works() {
        // 2  1  1  3  2
        // 1  2  2  1  1
        // 1  2  9  1  5
        // 3  1  1  7  1
        // 2  1  5  1  8
        let (nrow, ncol, nnz) = (5, 5, 25);
        let mut coo = NumCooMatrix::<f64>::new(nrow, ncol, nnz, Sym::YesFull).unwrap();
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
        let u = NumVector::<f64>::from(&[-629.0 / 98.0, 237.0 / 49.0, -53.0 / 49.0, 62.0 / 49.0, 23.0 / 14.0]);
        let mut v = NumVector::<f64>::from(&[1004.0, 2000.0, 3000.0, 4010.0, 5000.0]);
        coo.mat_vec_mul_update(&mut v, 2.0, &u).unwrap();
        let correct_v = &[1000.0, 2008.0, 3006.0, 4000.0, 5002.0];
        vec_approx_eq(&v, correct_v, 1e-12);
    }

    #[test]
    fn mat_vec_mul_update_positive_definite_works() {
        //  2  -1              2     ...
        // -1   2  -1    =>   -1   2
        //     -1   2             -1   2
        let (nrow, ncol, nnz) = (3, 3, 5);
        let mut coo = NumCooMatrix::<f64>::new(nrow, ncol, nnz, Sym::YesLower).unwrap();
        coo.put(0, 0, 2.0).unwrap();
        coo.put(1, 1, 2.0).unwrap();
        coo.put(2, 2, 2.0).unwrap();
        coo.put(1, 0, -1.0).unwrap();
        coo.put(2, 1, -1.0).unwrap();
        let u = NumVector::<f64>::from(&[5.0, 8.0, 7.0]);
        let mut v = NumVector::<f64>::from(&[1000.0, 2000.0, 3000.0]);
        coo.mat_vec_mul_update(&mut v, 2.0, &u).unwrap();
        let correct_v = &[1004.0, 2008.0, 3012.0];
        vec_approx_eq(&v, correct_v, 1e-15);
    }

    #[test]
    fn mat_vec_mul_update_rectangular_works() {
        //  5  -2  .  1
        // 10  -4  .  2
        // 15  -6  .  3
        let (coo, _, _, _) = Samples::rectangular_3x4();
        let u = NumVector::<f64>::from(&[1.0, 2.0, 3.0, -5.0]);
        let mut v = NumVector::<f64>::new(coo.nrow);
        coo.mat_vec_mul_update(&mut v, 2.0, &u).unwrap();
        let correct = &[-8.0, -16.0, -24.0];
        vec_approx_eq(&v, correct, 1e-15);
        // call mat_vec_mul_update again
        coo.mat_vec_mul_update(&mut v, 2.0, &u).unwrap();
        let correct = &[-16.0, -32.0, -48.0];
        vec_approx_eq(&v, correct, 1e-15);
    }

    #[test]
    fn mat_vec_mul_update_complex_works() {
        // 4+4i    .     2+2i
        //  .      1     3+3i
        //  .     5+5i   1+1i
        //  1      .      .
        let (coo, _, _, _) = Samples::complex_rectangular_4x3();
        let u = ComplexVector::from(&[cpx!(1.0, 1.0), cpx!(3.0, 1.0), cpx!(5.0, -1.0)]);
        let mut v = ComplexVector::new(coo.nrow);
        coo.mat_vec_mul_update(&mut v, cpx!(2.0, 4.0), &u).unwrap();
        let correct = &[
            cpx!(-40.0, 80.0),
            cpx!(-10.0, 110.0),
            cpx!(-64.0, 112.0),
            cpx!(-2.0, 6.0),
        ];
        complex_vec_approx_eq(&v, correct, 1e-15);
        // call mat_vec_mul_update again
        coo.mat_vec_mul_update(&mut v, cpx!(2.0, 4.0), &u).unwrap();
        let correct = &[
            cpx!(-80.0, 160.0),
            cpx!(-20.0, 220.0),
            cpx!(-128.0, 224.0),
            cpx!(-4.0, 12.0),
        ];
        complex_vec_approx_eq(&v, correct, 1e-15);
    }

    // --- mat_vec_mul_update (end) ---

    // --- mat_t_vec_mul (start) ---

    #[test]
    fn mat_t_vec_mul_captures_errors() {
        // A is 2x3 and Aᵀ is 3x2
        // v_3x1 = Aᵀ_3x2 * u_2x1
        let mut coo = NumCooMatrix::<u8>::new(2, 3, 1, Sym::No).unwrap();
        coo.put(0, 0, 123).unwrap();
        let u = NumVector::<u8>::new(1); // wrong, must be ≥ nrow
        let mut v = NumVector::<u8>::new(3); // ok, must be ≥ ncol
        assert_eq!(
            coo.mat_t_vec_mul(&mut v, 1, &u).err(),
            Some("u.dim() must be ≥ the number of rows of the matrix")
        );
        let u = NumVector::<u8>::new(2); // ok, must be ≥ nrow
        let mut v = NumVector::<u8>::new(1); // wrong, must be ≥ ncol
        assert_eq!(
            coo.mat_t_vec_mul(&mut v, 1, &u).err(),
            Some("v.dim() must be ≥ the number of columns of the matrix")
        );
    }

    #[test]
    fn mat_t_vec_mul_works() {
        //  1.0  2.0  3.0
        //  0.1  0.2  0.3
        // 10.0 20.0 30.0
        let mut coo = NumCooMatrix::<f64>::new(3, 3, 9, Sym::No).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(0, 1, 2.0).unwrap();
        coo.put(0, 2, 3.0).unwrap();
        coo.put(1, 0, 0.1).unwrap();
        coo.put(1, 1, 0.2).unwrap();
        coo.put(1, 2, 0.3).unwrap();
        coo.put(2, 0, 10.0).unwrap();
        coo.put(2, 1, 20.0).unwrap();
        coo.put(2, 2, 30.0).unwrap();
        let u = NumVector::<f64>::from(&[0.1, 0.2, 0.3]);
        let mut v = NumVector::<f64>::new(coo.nrow);
        coo.mat_t_vec_mul(&mut v, 2.0, &u).unwrap();
        let correct_v = &[6.24, 12.48, 18.72];
        vec_approx_eq(&v, correct_v, 1e-15);

        // call mat_t_vec_mul again to make sure the vector is filled with zeros before the sum
        coo.mat_t_vec_mul(&mut v, 2.0, &u).unwrap();
        vec_approx_eq(&v, correct_v, 1e-15);

        // single component matrix
        let mut single = NumCooMatrix::<f64>::new(1, 1, 1, Sym::No).unwrap();
        single.put(0, 0, 123.0).unwrap();
        let u = NumVector::from(&[2.0]);
        let mut v = NumVector::<f64>::new(1);
        single.mat_t_vec_mul(&mut v, 2.0, &u).unwrap();
        assert_eq!(v.as_data(), &[492.0]);
    }

    #[test]
    fn mat_t_vec_mul_symmetric_lower_works() {
        // 2
        // 1  2     sym
        // 1  2  9
        // 3  1  1  7
        // 2  1  5  1  8
        let (nrow, ncol, nnz) = (5, 5, 15);
        let mut coo = NumCooMatrix::<f64>::new(nrow, ncol, nnz, Sym::YesLower).unwrap();
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
        let u = NumVector::<f64>::from(&[-629.0 / 98.0, 237.0 / 49.0, -53.0 / 49.0, 62.0 / 49.0, 23.0 / 14.0]);
        let mut v = NumVector::<f64>::new(coo.nrow);
        coo.mat_t_vec_mul(&mut v, 2.0, &u).unwrap();
        let correct_v = &[-4.0, 8.0, 6.0, -10.0, 2.0];
        vec_approx_eq(&v, correct_v, 1e-14);
    }

    #[test]
    fn mat_t_vec_mul_symmetric_full_works() {
        // 2  1  1  3  2
        // 1  2  2  1  1
        // 1  2  9  1  5
        // 3  1  1  7  1
        // 2  1  5  1  8
        let (nrow, ncol, nnz) = (5, 5, 25);
        let mut coo = NumCooMatrix::<f64>::new(nrow, ncol, nnz, Sym::YesFull).unwrap();
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
        let u = NumVector::<f64>::from(&[-629.0 / 98.0, 237.0 / 49.0, -53.0 / 49.0, 62.0 / 49.0, 23.0 / 14.0]);
        let mut v = NumVector::<f64>::new(coo.nrow);
        coo.mat_t_vec_mul(&mut v, 2.0, &u).unwrap();
        let correct_v = &[-4.0, 8.0, 6.0, -10.0, 2.0];
        vec_approx_eq(&v, correct_v, 1e-14);
    }

    #[test]
    fn mat_t_vec_mul_positive_definite_works() {
        //  2  -1              2     ...
        // -1   2  -1    =>   -1   2
        //     -1   2             -1   2
        let (nrow, ncol, nnz) = (3, 3, 5);
        let mut coo = NumCooMatrix::<f64>::new(nrow, ncol, nnz, Sym::YesLower).unwrap();
        coo.put(0, 0, 2.0).unwrap();
        coo.put(1, 1, 2.0).unwrap();
        coo.put(2, 2, 2.0).unwrap();
        coo.put(1, 0, -1.0).unwrap();
        coo.put(2, 1, -1.0).unwrap();
        let u = NumVector::<f64>::from(&[5.0, 8.0, 7.0]);
        let mut v = NumVector::<f64>::new(coo.nrow);
        coo.mat_t_vec_mul(&mut v, 2.0, &u).unwrap();
        let correct_v = &[4.0, 8.0, 12.0];
        vec_approx_eq(&v, correct_v, 1e-15);
    }

    #[test]
    fn mat_t_vec_mul_rectangular_works() {
        //  5  -2  .  1
        // 10  -4  .  2
        // 15  -6  .  3
        let (coo, _, _, _) = Samples::rectangular_3x4();
        let u = NumVector::<f64>::from(&[1.0, 2.0, 3.0]);
        let mut v = NumVector::<f64>::new(coo.ncol); // coo.ncol = 4, because we do Aᵀ * u
        coo.mat_t_vec_mul(&mut v, 2.0, &u).unwrap();
        let correct = &[140.0, -56.0, 0.0, 28.0];
        vec_approx_eq(&v, correct, 1e-15);
        // call mat_t_vec_mul again to make sure the vector is filled with zeros before the sum
        coo.mat_t_vec_mul(&mut v, 2.0, &u).unwrap();
        vec_approx_eq(&v, correct, 1e-15);
    }

    #[test]
    fn mat_t_vec_mul_complex_works() {
        // 4+4i    .     2+2i
        //  .      1     3+3i
        //  .     5+5i   1+1i
        //  1      .      .
        let (coo, _, _, _) = Samples::complex_rectangular_4x3();
        let u = ComplexVector::from(&[cpx!(1.0, -1.0), cpx!(2.0, 1.0), cpx!(5.0, -1.0), cpx!(3.0, 2.0)]);
        let mut v = ComplexVector::new(coo.ncol); // coo.ncol = 3, because we do Aᵀ * u
        coo.mat_t_vec_mul(&mut v, cpx!(2.0, -4.0), &u).unwrap();
        let correct = &[cpx!(30.0, -40.0), cpx!(148.0, -86.0), cpx!(78.0, -26.0)];
        complex_vec_approx_eq(&v, correct, 1e-15);
        // call mat_t_vec_mul again to make sure the vector is filled with zeros before the sum
        coo.mat_t_vec_mul(&mut v, cpx!(2.0, -4.0), &u).unwrap();
        complex_vec_approx_eq(&v, correct, 1e-15);
    }

    // --- mat_t_vec_mul (end) ---
}
