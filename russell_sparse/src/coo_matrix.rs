use super::Layout;
use crate::StrError;
use russell_lab::{Matrix, Vector};
use russell_openblas::to_i32;
use std::fmt;

/// Holds the row index, col index, and values of a matrix (also known as Triplet)
///
/// # Remarks
///
/// * Only the non-zero values are required
/// * Entries with repeated (i,j) indices are allowed
/// * Repeated (i,j) entries will have the aij values summed when solving a linear system
/// * The repeated (i,j) capability is of great convenience for Finite Element solvers
/// * A maximum number of entries must be decided prior to allocating a new Triplet
/// * The maximum number of entries includes possible entries with repeated indices
/// * See the `to_matrix` method for an example
pub struct CooMatrix {
    /// Defines the stored layout: lower-triangular, upper-triangular, full-matrix
    pub layout: Layout,

    /// Holds the number of rows (must fit i32)
    pub nrow: usize,

    /// Holds the number of columns (must fit i32)
    pub ncol: usize,

    /// Holds the current index (must fit i32)
    ///
    /// This will equal the number of non-zeros (nnz) after all items have been `put`.
    pub pos: usize,

    /// Defines the maximum allowed number of entries (must fit i32)
    ///
    /// This may be greater than the number of non-zeros (nnz)
    pub max: usize,

    /// Holds the row indices i
    pub indices_i: Vec<i32>,

    /// Holds the column indices j
    pub indices_j: Vec<i32>,

    /// Holds the values aij
    pub values_aij: Vec<f64>,
}

impl CooMatrix {
    /// Creates a new CooMatrix representing a sparse matrix
    ///
    /// # Input
    ///
    /// * `layout` -- Defines the layout of the associated matrix
    /// * `nrow` -- Is the number of rows of the sparse matrix (must be fit i32)
    /// * `ncol` -- Is the number of columns of the sparse matrix (must be fit i32)
    /// * `max` -- Maximum number of entries ≥ nnz (number of non-zeros),
    ///            including entries with repeated indices. (must be fit i32)
    ///
    /// # Example
    ///
    /// ```
    /// use russell_chk::vec_approx_eq;
    /// use russell_sparse::{CooMatrix, Layout, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let (nrow, ncol, nnz) = (3, 3, 4);
    ///     let coo = CooMatrix::new(Layout::Full, nrow, ncol, nnz)?;
    ///     assert_eq!(coo.layout, Layout::Full);
    ///     assert_eq!(coo.pos, 0);
    ///     assert_eq!(coo.max, 4);
    ///     assert_eq!(coo.indices_i, &[0, 0, 0, 0]);
    ///     assert_eq!(coo.indices_j, &[0, 0, 0, 0]);
    ///     vec_approx_eq(&coo.values_aij, &[0.0, 0.0, 0.0, 0.0], 1e-15);
    ///     Ok(())
    /// }
    /// ```
    pub fn new(layout: Layout, nrow: usize, ncol: usize, max: usize) -> Result<Self, StrError> {
        if nrow < 1 {
            return Err("nrow must be greater than zero");
        }
        if ncol < 1 {
            return Err("ncol must be greater than zero");
        }
        if max < 1 {
            return Err("max must be greater than zero");
        }
        Ok(CooMatrix {
            layout,
            nrow,
            ncol,
            pos: 0,
            max,
            indices_i: vec![0; max],
            indices_j: vec![0; max],
            values_aij: vec![0.0; max],
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
    /// use russell_chk::vec_approx_eq;
    /// use russell_sparse::{CooMatrix, Layout, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let (nrow, ncol, nnz) = (3, 3, 4);
    ///     let mut coo = CooMatrix::new(Layout::Full, nrow, ncol, nnz)?;
    ///     coo.put(0, 0, 1.0)?;
    ///     coo.put(1, 1, 2.0)?;
    ///     coo.put(2, 2, 3.0)?;
    ///     coo.put(0, 1, 4.0)?;
    ///     assert_eq!(coo.layout, Layout::Full);
    ///     assert_eq!(coo.pos, 4);
    ///     assert_eq!(coo.max, 4);
    ///     assert_eq!(coo.indices_i, &[0, 1, 2, 0]);
    ///     assert_eq!(coo.indices_j, &[0, 1, 2, 1]);
    ///     vec_approx_eq(&coo.values_aij, &[1.0, 2.0, 3.0, 4.0], 1e-15);
    ///     Ok(())
    /// }
    /// ```
    pub fn put(&mut self, i: usize, j: usize, aij: f64) -> Result<(), StrError> {
        // check range
        if i >= self.nrow {
            return Err("index of row is outside range");
        }
        if j >= self.ncol {
            return Err("index of column is outside range");
        }
        if self.pos >= self.max {
            return Err("max number of items has been exceeded");
        }
        if self.layout == Layout::Lower {
            if j > i {
                return Err("j > i is incorrect for lower triangular layout");
            }
        } else if self.layout == Layout::Upper {
            if j < i {
                return Err("j < i is incorrect for upper triangular layout");
            }
        }

        // insert a new entry
        let i_i32 = to_i32(i);
        let j_i32 = to_i32(j);
        self.indices_i[self.pos] = i_i32;
        self.indices_j[self.pos] = j_i32;
        self.values_aij[self.pos] = aij;
        self.pos += 1;
        Ok(())
    }

    /// Resets the position of the current non-zero value
    ///
    /// This function allows using `put` all over again.
    ///
    /// # Example
    ///
    /// ```
    /// use russell_sparse::{CooMatrix, Layout, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     let (nrow, ncol, nnz) = (3, 3, 4);
    ///     let mut coo = CooMatrix::new(Layout::Full, nrow, ncol, nnz)?;
    ///     coo.put(0, 0, 1.0)?;
    ///     coo.put(1, 1, 2.0)?;
    ///     coo.put(2, 2, 3.0)?;
    ///     coo.put(0, 1, 4.0)?;
    ///     assert_eq!(coo.pos, 4);
    ///     coo.reset();
    ///     assert_eq!(coo.pos, 0);
    ///     Ok(())
    /// }
    /// ```
    pub fn reset(&mut self) {
        self.pos = 0;
    }

    /// Converts the CooMatrix to a dense matrix and returns the matrix
    ///
    /// Note: this function calls [CooMatrix::to_matrix].
    ///
    /// ```
    /// use russell_sparse::{CooMatrix, Layout, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     // define (4 x 4) sparse matrix with 6+1 non-zero values
    ///     // (with an extra ij-repeated entry)
    ///     let (nrow, ncol, nnz) = (4, 4, 7);
    ///     let mut coo = CooMatrix::new(Layout::Full, nrow, ncol, nnz)?;
    ///     coo.put(0, 0, 0.5)?; // (0, 0, a00/2)
    ///     coo.put(0, 0, 0.5)?; // (0, 0, a00/2)
    ///     coo.put(0, 1, 2.0)?;
    ///     coo.put(1, 0, 3.0)?;
    ///     coo.put(1, 1, 4.0)?;
    ///     coo.put(2, 2, 5.0)?;
    ///     coo.put(3, 3, 6.0)?;
    ///
    ///     // convert to matrix
    ///     let a = coo.as_matrix();
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
    pub fn as_matrix(&self) -> Matrix {
        let mut a = Matrix::new(self.nrow, self.ncol);
        self.to_matrix(&mut a).unwrap();
        a
    }

    /// Converts the CooMatrix to a dense matrix, up to a limit
    ///
    /// Note: see the function [CooMatrix::as_matrix] that returns the Matrix.
    ///
    /// # Input
    ///
    /// `a` -- (nrow_max, ncol_max) matrix to hold the triplet data.
    ///  The output matrix may have fewer rows or fewer columns than the triplet data.
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::{Matrix};
    /// use russell_sparse::{CooMatrix, Layout, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     // define (4 x 4) sparse matrix with 6+1 non-zero values
    ///     // (with an extra ij-repeated entry)
    ///     let (nrow, ncol, nnz) = (4, 4, 7);
    ///     let mut coo = CooMatrix::new(Layout::Full, nrow, ncol, nnz)?;
    ///     coo.put(0, 0, 0.5)?; // (0, 0, a00/2)
    ///     coo.put(0, 0, 0.5)?; // (0, 0, a00/2)
    ///     coo.put(0, 1, 2.0)?;
    ///     coo.put(1, 0, 3.0)?;
    ///     coo.put(1, 1, 4.0)?;
    ///     coo.put(2, 2, 5.0)?;
    ///     coo.put(3, 3, 6.0)?;
    ///
    ///     // convert the first (3 x 3) values
    ///     let mut a = Matrix::new(3, 3);
    ///     coo.to_matrix(&mut a)?;
    ///     let correct = "┌       ┐\n\
    ///                    │ 1 2 0 │\n\
    ///                    │ 3 4 0 │\n\
    ///                    │ 0 0 5 │\n\
    ///                    └       ┘";
    ///     assert_eq!(format!("{}", a), correct);
    ///
    ///     // convert the first (4 x 4) values
    ///     let mut b = Matrix::new(4, 4);
    ///     coo.to_matrix(&mut b)?;
    ///     let correct = "┌         ┐\n\
    ///                    │ 1 2 0 0 │\n\
    ///                    │ 3 4 0 0 │\n\
    ///                    │ 0 0 5 0 │\n\
    ///                    │ 0 0 0 6 │\n\
    ///                    └         ┘";
    ///     assert_eq!(format!("{}", b), correct);
    ///     Ok(())
    /// }
    /// ```
    pub fn to_matrix(&self, a: &mut Matrix) -> Result<(), StrError> {
        let (m, n) = a.dims();
        if m > self.nrow || n > self.ncol {
            return Err("wrong matrix dimensions");
        }
        if self.layout != Layout::Full && m != n {
            return Err("the resulting matrix must be square when the layout is either lower of upper triangular");
        }
        let m_i32 = to_i32(m);
        let n_i32 = to_i32(n);
        a.fill(0.0);
        for p in 0..self.pos {
            if self.indices_i[p] < m_i32 && self.indices_j[p] < n_i32 {
                let (i, j) = (self.indices_i[p] as usize, self.indices_j[p] as usize);
                a.add(i, j, self.values_aij[p]);
                if self.layout != Layout::Full && i != j {
                    a.add(j, i, self.values_aij[p]);
                }
            }
        }
        Ok(())
    }

    /// Performs the matrix-vector multiplication
    ///
    /// ```text
    ///  v  :=   a   ⋅  u
    /// (m)    (m,n)   (n)
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
    /// use russell_sparse::{CooMatrix, Layout, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     // set sparse matrix (3 x 3) with 6 non-zeros
    ///     let (nrow, ncol, nnz) = (3, 3, 6);
    ///     let mut coo = CooMatrix::new(Layout::Full, nrow, ncol, nnz)?;
    ///     coo.put(0, 0, 1.0)?;
    ///     coo.put(1, 0, 2.0)?;
    ///     coo.put(1, 1, 3.0)?;
    ///     coo.put(2, 0, 4.0)?;
    ///     coo.put(2, 1, 5.0)?;
    ///     coo.put(2, 2, 6.0)?;
    ///
    ///     // check matrix
    ///     let mut a = Matrix::new(nrow, ncol);
    ///     coo.to_matrix(&mut a)?;
    ///     let correct_a = "┌       ┐\n\
    ///                      │ 1 0 0 │\n\
    ///                      │ 2 3 0 │\n\
    ///                      │ 4 5 6 │\n\
    ///                      └       ┘";
    ///     assert_eq!(format!("{}", a), correct_a);
    ///
    ///     // perform mat-vec-mul
    ///     let u = Vector::from(&[1.0, 1.0, 1.0]);
    ///     let v = coo.mat_vec_mul(&u)?;
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
    pub fn mat_vec_mul(&self, u: &Vector) -> Result<Vector, StrError> {
        if u.dim() != self.ncol {
            return Err("u.ndim must equal ncol");
        }
        let mut v = Vector::new(self.nrow);
        for p in 0..self.pos {
            let i = self.indices_i[p] as usize;
            let j = self.indices_j[p] as usize;
            let aij = self.values_aij[p];
            v[i] += aij * u[j];
            if self.layout != Layout::Full && i != j {
                v[j] += aij * u[i];
            }
        }
        Ok(v)
    }
}

impl fmt::Display for CooMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "\x20\x20\x20\x20\"neq\": {},\n\
             \x20\x20\x20\x20\"nnz_current\": {},\n\
             \x20\x20\x20\x20\"nnz_maximum\": {},\n",
            self.nrow, self.pos, self.max,
        )
        .unwrap();
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::CooMatrix;
    use crate::Layout;
    use russell_chk::vec_approx_eq;
    use russell_lab::{Matrix, Vector};

    #[test]
    fn new_fails_on_wrong_input() {
        assert_eq!(
            CooMatrix::new(Layout::Full, 0, 1, 3).err(),
            Some("nrow must be greater than zero")
        );
        assert_eq!(
            CooMatrix::new(Layout::Full, 1, 0, 3).err(),
            Some("ncol must be greater than zero")
        );
        assert_eq!(
            CooMatrix::new(Layout::Full, 1, 1, 0).err(),
            Some("max must be greater than zero")
        );
    }

    #[test]
    fn new_works() {
        let coo = CooMatrix::new(Layout::Full, 1, 1, 3).unwrap();
        assert_eq!(coo.layout, Layout::Full);
        assert_eq!(coo.nrow, 1);
        assert_eq!(coo.ncol, 1);
        assert_eq!(coo.pos, 0);
        assert_eq!(coo.max, 3);
        assert_eq!(coo.indices_i.len(), 3);
        assert_eq!(coo.indices_j.len(), 3);
        assert_eq!(coo.values_aij.len(), 3);
    }

    #[test]
    fn put_fails_on_wrong_values() {
        let mut coo = CooMatrix::new(Layout::Full, 1, 1, 1).unwrap();
        assert_eq!(coo.put(1, 0, 0.0).err(), Some("index of row is outside range"));
        assert_eq!(coo.put(0, 1, 0.0).err(), Some("index of column is outside range"));
        assert_eq!(coo.put(0, 0, 0.0).err(), None); // << will take all spots
        assert_eq!(coo.put(0, 0, 0.0).err(), Some("max number of items has been exceeded"));
        let mut coo = CooMatrix::new(Layout::Lower, 2, 2, 4).unwrap();
        assert_eq!(
            coo.put(0, 1, 0.0).err(),
            Some("j > i is incorrect for lower triangular layout")
        );
        let mut coo = CooMatrix::new(Layout::Upper, 2, 2, 4).unwrap();
        assert_eq!(
            coo.put(1, 0, 0.0).err(),
            Some("j < i is incorrect for upper triangular layout")
        );
    }

    #[test]
    fn put_works() {
        let mut coo = CooMatrix::new(Layout::Full, 3, 3, 5).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        assert_eq!(coo.pos, 1);
        coo.put(0, 1, 2.0).unwrap();
        assert_eq!(coo.pos, 2);
        coo.put(1, 0, 3.0).unwrap();
        assert_eq!(coo.pos, 3);
        coo.put(1, 1, 4.0).unwrap();
        assert_eq!(coo.pos, 4);
        coo.put(2, 2, 5.0).unwrap();
        assert_eq!(coo.pos, 5);
    }

    #[test]
    fn reset_works() {
        let mut coo = CooMatrix::new(Layout::Full, 2, 2, 4).unwrap();
        assert_eq!(coo.pos, 0);
        coo.put(0, 0, 1.0).unwrap();
        coo.put(0, 1, 4.0).unwrap();
        coo.put(1, 0, 2.0).unwrap();
        coo.put(1, 1, 3.0).unwrap();
        assert_eq!(coo.pos, 4);
        coo.reset();
        assert_eq!(coo.pos, 0);
    }

    #[test]
    fn to_matrix_fails_on_wrong_dims() {
        let coo = CooMatrix::new(Layout::Full, 1, 1, 1).unwrap();
        let mut a_2x1 = Matrix::new(2, 1);
        let mut a_1x2 = Matrix::new(1, 2);
        assert_eq!(coo.to_matrix(&mut a_2x1), Err("wrong matrix dimensions"));
        assert_eq!(coo.to_matrix(&mut a_1x2), Err("wrong matrix dimensions"));
    }

    #[test]
    fn to_matrix_works() {
        let mut coo = CooMatrix::new(Layout::Full, 3, 3, 5).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(0, 1, 2.0).unwrap();
        coo.put(1, 0, 3.0).unwrap();
        coo.put(1, 1, 4.0).unwrap();
        coo.put(2, 2, 5.0).unwrap();
        let mut a = Matrix::new(3, 3);
        coo.to_matrix(&mut a).unwrap();
        assert_eq!(a.get(0, 0), 1.0);
        assert_eq!(a.get(0, 1), 2.0);
        assert_eq!(a.get(1, 0), 3.0);
        assert_eq!(a.get(1, 1), 4.0);
        assert_eq!(a.get(2, 2), 5.0);
        let mut b = Matrix::new(2, 1);
        coo.to_matrix(&mut b).unwrap();
        assert_eq!(b.get(0, 0), 1.0);
        assert_eq!(b.get(1, 0), 3.0);
        // using as_matrix
        let bb = coo.as_matrix();
        assert_eq!(bb.get(0, 0), 1.0);
        assert_eq!(bb.get(1, 0), 3.0);
    }

    #[test]
    fn to_matrix_with_duplicates_works() {
        // allocate a square matrix
        let (nrow, ncol, nnz) = (5, 5, 13);
        let mut coo = CooMatrix::new(Layout::Full, nrow, ncol, nnz).unwrap();
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
        coo.to_matrix(&mut a).unwrap();
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
    fn to_matrix_symmetric_lower_works() {
        let mut coo = CooMatrix::new(Layout::Lower, 3, 3, 4).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(1, 0, 2.0).unwrap();
        coo.put(1, 1, 3.0).unwrap();
        coo.put(2, 1, 4.0).unwrap();
        let mut a = Matrix::new(3, 3);
        coo.to_matrix(&mut a).unwrap();
        let correct = "┌       ┐\n\
                       │ 1 2 0 │\n\
                       │ 2 3 4 │\n\
                       │ 0 4 0 │\n\
                       └       ┘";
        assert_eq!(format!("{}", a), correct);
        let mut b = Matrix::new(2, 1);
        assert_eq!(
            coo.to_matrix(&mut b).err(),
            Some("the resulting matrix must be square when the layout is either lower of upper triangular")
        );
    }

    #[test]
    fn to_matrix_symmetric_upper_works() {
        let mut coo = CooMatrix::new(Layout::Upper, 3, 3, 4).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(0, 1, 2.0).unwrap();
        coo.put(1, 1, 3.0).unwrap();
        coo.put(1, 2, 4.0).unwrap();
        let mut a = Matrix::new(3, 3);
        coo.to_matrix(&mut a).unwrap();
        let correct = "┌       ┐\n\
                       │ 1 2 0 │\n\
                       │ 2 3 4 │\n\
                       │ 0 4 0 │\n\
                       └       ┘";
        assert_eq!(format!("{}", a), correct);
        let mut b = Matrix::new(2, 1);
        assert_eq!(
            coo.to_matrix(&mut b).err(),
            Some("the resulting matrix must be square when the layout is either lower of upper triangular")
        );
    }

    #[test]
    fn mat_vec_mul_fails_on_wrong_input() {
        let coo = CooMatrix::new(Layout::Full, 2, 2, 1).unwrap();
        let u = Vector::new(3);
        assert_eq!(coo.mat_vec_mul(&u).err(), Some("u.ndim must equal ncol"));
    }

    #[test]
    fn mat_vec_mul_works() {
        //  1.0  2.0  3.0
        //  0.1  0.2  0.3
        // 10.0 20.0 30.0
        let mut coo = CooMatrix::new(Layout::Full, 3, 3, 9).unwrap();
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
        let correct_v = &[1.4, 0.14, 14.0];
        let v = coo.mat_vec_mul(&u).unwrap();
        vec_approx_eq(v.as_data(), correct_v, 1e-15);
    }

    #[test]
    fn mat_vec_mul_symmetric_lower_works() {
        // 2
        // 1  2     sym
        // 1  2  9
        // 3  1  1  7
        // 2  1  5  1  8
        let (nrow, ncol, nnz) = (5, 5, 15);
        let mut coo = CooMatrix::new(Layout::Lower, nrow, ncol, nnz).unwrap();
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
        let correct_v = &[-2.0, 4.0, 3.0, -5.0, 1.0];
        let v = coo.mat_vec_mul(&u).unwrap();
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
        let mut coo = CooMatrix::new(Layout::Full, nrow, ncol, nnz).unwrap();
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
        let correct_v = &[-2.0, 4.0, 3.0, -5.0, 1.0];
        let v = coo.mat_vec_mul(&u).unwrap();
        vec_approx_eq(v.as_data(), correct_v, 1e-14);
    }

    #[test]
    fn mat_vec_mul_positive_definite_works() {
        //  2  -1              2     ...
        // -1   2  -1    =>   -1   2
        //     -1   2             -1   2
        let (nrow, ncol, nnz) = (3, 3, 5);
        let mut coo = CooMatrix::new(Layout::Lower, nrow, ncol, nnz).unwrap();
        coo.put(0, 0, 2.0).unwrap();
        coo.put(1, 1, 2.0).unwrap();
        coo.put(2, 2, 2.0).unwrap();
        coo.put(1, 0, -1.0).unwrap();
        coo.put(2, 1, -1.0).unwrap();
        let u = Vector::from(&[5.0, 8.0, 7.0]);
        let correct_v = &[2.0, 4.0, 6.0];
        let v = coo.mat_vec_mul(&u).unwrap();
        vec_approx_eq(v.as_data(), correct_v, 1e-15);
    }
}
