use russell_lab::{Matrix, Vector};
use russell_openblas::to_i32;
use std::fmt;

/// Holds triples (i,j,aij) representing a sparse matrix
///
/// # Remarks
///
/// - Only the non-zero values are required
/// - Entries with repeated (i,j) indices are allowed
/// - Repeated (i,j) entries will have the aij values summed when solving a linear system
/// - The repeated (i,j) capability is of great convenience for Finite Element solvers
/// - A maximum number of entries must be decided prior to allocating a new Triplet
/// - The maximum number of entries includes possible entries with repeated indices
/// - See the `to_matrix` method for an example
pub struct SparseTriplet {
    pub(crate) nrow: usize,          // [i32] number of rows
    pub(crate) ncol: usize,          // [i32] number of columns
    pub(crate) pos: usize,           // [i32] current index => nnz in the end
    pub(crate) max: usize,           // [i32] max allowed number of entries (may be > nnz)
    pub(crate) symmetric: bool,      // general symmetric?, but WITHOUT one side of the diagonal
    pub(crate) indices_i: Vec<i32>,  // [nnz] indices i
    pub(crate) indices_j: Vec<i32>,  // [nnz] indices j
    pub(crate) values_aij: Vec<f64>, // [nnz] values aij
}

impl SparseTriplet {
    /// Creates a new SparseTriplet representing a sparse matrix
    ///
    /// ```text
    /// trip  :=  sparse(a)
    /// (max)    (nrow,ncol)
    /// ```
    ///
    /// # Input
    ///
    /// * `nrow` -- The number of rows of the sparse matrix
    /// * `ncol` -- The number of columns of the sparse matrix
    /// * `max` -- The maximum number fo non-zero values in the sparse matrix,
    ///            including entries with repeated indices
    /// * `symmetric` -- This Triplet represents a **general** symmetric matrix.
    ///                  In this case, one side of the diagonal **must** be ignored.
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), &'static str> {
    /// use russell_sparse::*;
    /// let trip = SparseTriplet::new(3, 3, 5, false)?;
    /// let correct: &str = "nrow      = 3\n\
    ///                      ncol      = 3\n\
    ///                      max       = 5\n\
    ///                      pos       = 0\n\
    ///                      symmetric = false\n";
    /// assert_eq!(format!("{}", trip), correct);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(nrow: usize, ncol: usize, max: usize, symmetric: bool) -> Result<Self, &'static str> {
        if nrow == 0 || ncol == 0 || max == 0 {
            return Err("nrow, ncol, and max must all be greater than zero");
        }
        Ok(SparseTriplet {
            nrow,
            ncol,
            pos: 0,
            max,
            symmetric,
            indices_i: vec![0; max],
            indices_j: vec![0; max],
            values_aij: vec![0.0; max],
        })
    }

    /// Puts the next triple (i,j,aij) into the Triplet
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), &'static str> {
    /// use russell_sparse::*;
    /// let mut trip = SparseTriplet::new(2, 2, 1, false)?;
    /// trip.put(0, 0, 1.0);
    /// let correct: &str = "nrow      = 2\n\
    ///                      ncol      = 2\n\
    ///                      max       = 1\n\
    ///                      pos       = 1 (FULL)\n\
    ///                      symmetric = false\n";
    /// assert_eq!(format!("{}", trip), correct);
    /// # Ok(())
    /// # }
    /// ```
    pub fn put(&mut self, i: usize, j: usize, aij: f64) {
        assert!(i < self.nrow);
        assert!(j < self.ncol);
        assert!(self.pos < self.max);
        let i_i32 = to_i32(i);
        let j_i32 = to_i32(j);
        self.indices_i[self.pos] = i_i32;
        self.indices_j[self.pos] = j_i32;
        self.values_aij[self.pos] = aij;
        self.pos += 1;
    }

    /// Returns the (nrow x ncol) dimensions of the matrix represented by this Triplet
    ///
    /// # Output
    ///
    /// * `nrow` -- number of rows
    /// * `ncol` -- number of columns
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), &'static str> {
    /// use russell_sparse::*;
    /// let trip = SparseTriplet::new(2, 2, 1, false)?;
    /// assert_eq!(trip.dims(), (2, 2));
    /// # Ok(())
    /// # }
    /// ```
    pub fn dims(&self) -> (usize, usize) {
        (self.nrow, self.ncol)
    }

    /// Returns the value of the symmetric flag
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), &'static str> {
    /// use russell_sparse::*;
    /// let trip = SparseTriplet::new(2, 2, 1, true)?;
    /// assert_eq!(trip.is_symmetric(), true);
    /// # Ok(())
    /// # }
    /// ```
    pub fn is_symmetric(&self) -> bool {
        self.symmetric
    }

    /// Converts the triples data to a matrix, up to a limit
    ///
    /// # Input
    ///
    /// `a` -- (nrow_max, ncol_max) matrix to hold the triples data. Thus, the matrix may have less rows or less columns than the triplet data
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), &'static str> {
    /// // import
    /// use russell_lab::*;
    /// use russell_sparse::*;
    ///
    /// // define (4 x 4) sparse matrix with 6+1 non-zero values
    /// // (with an extra ij-repeated entry)
    /// let mut trip = SparseTriplet::new(4, 4, 6+1, false)?;
    /// trip.put(0, 0, 0.5); // (0, 0, a00/2)
    /// trip.put(0, 0, 0.5); // (0, 0, a00/2)
    /// trip.put(0, 1, 2.0);
    /// trip.put(1, 0, 3.0);
    /// trip.put(1, 1, 4.0);
    /// trip.put(2, 2, 5.0);
    /// trip.put(3, 3, 6.0);
    ///
    /// // convert the first (3 x 3) values
    /// let mut a = Matrix::new(3, 3);
    /// trip.to_matrix(&mut a)?;
    /// let correct = "┌       ┐\n\
    ///                │ 1 2 0 │\n\
    ///                │ 3 4 0 │\n\
    ///                │ 0 0 5 │\n\
    ///                └       ┘";
    /// assert_eq!(format!("{}", a), correct);
    ///
    /// // convert the first (4 x 4) values
    /// let mut b = Matrix::new(4, 4);
    /// trip.to_matrix(&mut b)?;
    /// let correct = "┌         ┐\n\
    ///                │ 1 2 0 0 │\n\
    ///                │ 3 4 0 0 │\n\
    ///                │ 0 0 5 0 │\n\
    ///                │ 0 0 0 6 │\n\
    ///                └         ┘";
    /// assert_eq!(format!("{}", b), correct);
    /// # Ok(())
    /// # }
    /// ```
    pub fn to_matrix(&self, a: &mut Matrix) -> Result<(), &'static str> {
        let (m, n) = a.dims();
        if m > self.nrow || n > self.ncol {
            return Err("wrong matrix dimensions");
        }
        let m_i32 = to_i32(m);
        let n_i32 = to_i32(n);
        a.fill(0.0);
        for p in 0..self.pos {
            if self.indices_i[p] < m_i32 && self.indices_j[p] < n_i32 {
                a.plus_equal(
                    self.indices_i[p] as usize,
                    self.indices_j[p] as usize,
                    self.values_aij[p],
                );
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
    /// # Note
    ///
    /// This method is not highly efficient but should useful in verifications.
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), &'static str> {
    /// // import
    /// use russell_lab::*;
    /// use russell_sparse::*;
    ///
    /// // set sparse matrix (4 x 3) with 6 non-zeros
    /// let mut trip = SparseTriplet::new(4, 3, 6, false)?;
    /// trip.put(0, 0, 1.0);
    /// trip.put(1, 0, 2.0);
    /// trip.put(1, 1, 3.0);
    /// trip.put(2, 0, 4.0);
    /// trip.put(3, 0, 5.0);
    /// trip.put(3, 2, 6.0);
    ///
    /// // check matrix
    /// let (m, n) = trip.dims();
    /// let mut a = Matrix::new(m, n);
    /// trip.to_matrix(&mut a)?;
    /// let correct_a = "┌       ┐\n\
    ///                  │ 1 0 0 │\n\
    ///                  │ 2 3 0 │\n\
    ///                  │ 4 0 0 │\n\
    ///                  │ 5 0 6 │\n\
    ///                  └       ┘";
    /// assert_eq!(format!("{}", a), correct_a);
    ///
    /// // perform mat-vec-mul
    /// let u = Vector::from(&[1.0, 1.0, 1.0]);
    /// let v = trip.mat_vec_mul(&u)?;
    ///
    /// // check vector
    /// let correct_v = "┌    ┐\n\
    ///                  │  1 │\n\
    ///                  │  5 │\n\
    ///                  │  4 │\n\
    ///                  │ 11 │\n\
    ///                  └    ┘";
    /// assert_eq!(format!("{}", v), correct_v);
    /// # Ok(())
    /// # }
    /// ```
    pub fn mat_vec_mul(&self, u: &Vector) -> Result<Vector, &'static str> {
        if u.dim() != self.ncol {
            return Err("u.ndim must equal a.ncol");
        }
        let mut v = Vector::new(self.nrow);
        for p in 0..self.pos {
            let i = self.indices_i[p] as usize;
            let j = self.indices_j[p] as usize;
            let aij = self.values_aij[p];
            v.plus_equal(i, aij * u.get(j));
            if self.symmetric && i != j {
                v.plus_equal(j, aij * u.get(i));
            }
        }
        Ok(v)
    }
}

impl fmt::Display for SparseTriplet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pos = if self.pos == self.max {
            format!("{} (FULL)", self.pos)
        } else {
            format!("{}", self.pos)
        };
        write!(
            f,
            "nrow      = {}\n\
             ncol      = {}\n\
             max       = {}\n\
             pos       = {}\n\
             symmetric = {}\n",
            self.nrow, self.ncol, self.max, pos, self.symmetric
        )?;
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use russell_chk::assert_vec_approx_eq;

    use super::*;

    #[test]
    fn new_fails_on_wrong_dims() {
        assert_eq!(
            SparseTriplet::new(0, 3, 5, false).err(),
            Some("nrow, ncol, and max must all be greater than zero")
        );
        assert_eq!(
            SparseTriplet::new(3, 0, 5, false).err(),
            Some("nrow, ncol, and max must all be greater than zero")
        );
        assert_eq!(
            SparseTriplet::new(3, 3, 0, false).err(),
            Some("nrow, ncol, and max must all be greater than zero")
        );
    }

    #[test]
    fn new_works() -> Result<(), &'static str> {
        let trip = SparseTriplet::new(3, 3, 5, false)?;
        assert_eq!(trip.nrow, 3);
        assert_eq!(trip.ncol, 3);
        assert_eq!(trip.pos, 0);
        assert_eq!(trip.max, 5);
        assert_eq!(trip.symmetric, false);
        Ok(())
    }

    #[test]
    fn display_trait_works() -> Result<(), &'static str> {
        let mut trip = SparseTriplet::new(3, 3, 1, false)?;
        let correct_1: &str = "nrow      = 3\n\
                               ncol      = 3\n\
                               max       = 1\n\
                               pos       = 0\n\
                               symmetric = false\n";
        assert_eq!(format!("{}", trip), correct_1);
        trip.put(0, 0, 1.0);
        let correct_2: &str = "nrow      = 3\n\
                               ncol      = 3\n\
                               max       = 1\n\
                               pos       = 1 (FULL)\n\
                               symmetric = false\n";
        assert_eq!(format!("{}", trip), correct_2);
        Ok(())
    }

    #[test]
    #[should_panic]
    fn put_panics_on_wrong_values_1() {
        let mut trip = SparseTriplet::new(1, 1, 1, false).unwrap();
        trip.put(1, 0, 0.0);
    }

    #[test]
    #[should_panic]
    fn put_panics_on_wrong_values_2() {
        let mut trip = SparseTriplet::new(1, 1, 1, false).unwrap();
        trip.put(0, 1, 0.0);
    }

    #[test]
    #[should_panic]
    fn put_panics_on_wrong_values_3() {
        let mut trip = SparseTriplet::new(1, 1, 1, false).unwrap();
        trip.put(0, 0, 0.0); // << all spots occupied
        trip.put(0, 0, 0.0);
    }

    #[test]
    fn put_works() -> Result<(), &'static str> {
        let mut trip = SparseTriplet::new(3, 3, 5, false)?;
        trip.put(0, 0, 1.0);
        assert_eq!(trip.pos, 1);
        trip.put(0, 1, 2.0);
        assert_eq!(trip.pos, 2);
        trip.put(1, 0, 3.0);
        assert_eq!(trip.pos, 3);
        trip.put(1, 1, 4.0);
        assert_eq!(trip.pos, 4);
        trip.put(2, 2, 5.0);
        assert_eq!(trip.pos, 5);
        Ok(())
    }

    #[test]
    fn dims_works() -> Result<(), &'static str> {
        let trip = SparseTriplet::new(3, 2, 1, false)?;
        assert_eq!(trip.dims(), (3, 2));
        Ok(())
    }

    #[test]
    fn is_symmetric_works() -> Result<(), &'static str> {
        let trip = SparseTriplet::new(3, 2, 1, true)?;
        assert_eq!(trip.is_symmetric(), true);
        Ok(())
    }

    #[test]
    fn to_matrix_fails_on_wrong_dims() -> Result<(), &'static str> {
        let trip = SparseTriplet::new(1, 1, 1, false)?;
        let mut a_2x1 = Matrix::new(2, 1);
        let mut a_1x2 = Matrix::new(1, 2);
        assert_eq!(trip.to_matrix(&mut a_2x1), Err("wrong matrix dimensions"));
        assert_eq!(trip.to_matrix(&mut a_1x2), Err("wrong matrix dimensions"));
        Ok(())
    }

    #[test]
    fn to_matrix_works() -> Result<(), &'static str> {
        let mut trip = SparseTriplet::new(3, 3, 5, false)?;
        trip.put(0, 0, 1.0);
        trip.put(0, 1, 2.0);
        trip.put(1, 0, 3.0);
        trip.put(1, 1, 4.0);
        trip.put(2, 2, 5.0);
        let mut a = Matrix::new(3, 3);
        trip.to_matrix(&mut a)?;
        assert_eq!(a.get(0, 0), 1.0);
        assert_eq!(a.get(0, 1), 2.0);
        assert_eq!(a.get(1, 0), 3.0);
        assert_eq!(a.get(1, 1), 4.0);
        assert_eq!(a.get(2, 2), 5.0);
        let mut b = Matrix::new(2, 1);
        trip.to_matrix(&mut b)?;
        assert_eq!(b.get(0, 0), 1.0);
        assert_eq!(b.get(1, 0), 3.0);
        Ok(())
    }

    #[test]
    fn to_matrix_with_duplicates_works() -> Result<(), &'static str> {
        // allocate a square matrix
        let mut trip = SparseTriplet::new(5, 5, 13, false)?;
        trip.put(0, 0, 1.0); // << (0, 0, a00/2)
        trip.put(0, 0, 1.0); // << (0, 0, a00/2)
        trip.put(1, 0, 3.0);
        trip.put(0, 1, 3.0);
        trip.put(2, 1, -1.0);
        trip.put(4, 1, 4.0);
        trip.put(1, 2, 4.0);
        trip.put(2, 2, -3.0);
        trip.put(3, 2, 1.0);
        trip.put(4, 2, 2.0);
        trip.put(2, 3, 2.0);
        trip.put(1, 4, 6.0);
        trip.put(4, 4, 1.0);

        // print matrix
        let (m, n) = trip.dims();
        let mut a = Matrix::new(m, n);
        trip.to_matrix(&mut a)?;
        let correct = "┌                ┐\n\
                            │  2  3  0  0  0 │\n\
                            │  3  0  4  0  6 │\n\
                            │  0 -1 -3  2  0 │\n\
                            │  0  0  1  0  0 │\n\
                            │  0  4  2  0  1 │\n\
                            └                ┘";
        assert_eq!(format!("{}", a), correct);
        Ok(())
    }

    #[test]
    fn mat_vec_mul_fails_on_wrong_input() -> Result<(), &'static str> {
        let trip = SparseTriplet::new(2, 2, 1, false)?;
        let u = Vector::new(3);
        assert_eq!(trip.mat_vec_mul(&u).err(), Some("u.ndim must equal a.ncol"));
        Ok(())
    }

    #[test]
    fn mat_vec_mul_works() -> Result<(), &'static str> {
        //  1.0  2.0  3.0  4.0  5.0
        //  0.1  0.2  0.3  0.4  0.5
        // 10.0 20.0 30.0 40.0 50.0
        let mut trip = SparseTriplet::new(3, 5, 15, false)?;
        trip.put(0, 0, 1.0);
        trip.put(0, 1, 2.0);
        trip.put(0, 2, 3.0);
        trip.put(0, 3, 4.0);
        trip.put(0, 4, 5.0);
        trip.put(1, 0, 0.1);
        trip.put(1, 1, 0.2);
        trip.put(1, 2, 0.3);
        trip.put(1, 3, 0.4);
        trip.put(1, 4, 0.5);
        trip.put(2, 0, 10.0);
        trip.put(2, 1, 20.0);
        trip.put(2, 2, 30.0);
        trip.put(2, 3, 40.0);
        trip.put(2, 4, 50.0);
        let u = Vector::from(&[0.1, 0.2, 0.3, 0.4, 0.5]);
        let correct_v = &[5.5, 0.55, 55.0];
        let v = trip.mat_vec_mul(&u)?;
        assert_vec_approx_eq!(v.as_data(), correct_v, 1e-15);
        Ok(())
    }

    #[test]
    fn mat_vec_mul_symmetric_works() -> Result<(), &'static str> {
        // 2
        // 1  2     sym
        // 1  2  9
        // 3  1  1  7
        // 2  1  5  1  8
        let mut trip = SparseTriplet::new(5, 5, 15, true)?;
        trip.put(0, 0, 2.0);
        trip.put(1, 1, 2.0);
        trip.put(2, 2, 9.0);
        trip.put(3, 3, 7.0);
        trip.put(4, 4, 8.0);
        trip.put(1, 0, 1.0);
        trip.put(2, 0, 1.0);
        trip.put(2, 1, 2.0);
        trip.put(3, 0, 3.0);
        trip.put(3, 1, 1.0);
        trip.put(3, 2, 1.0);
        trip.put(4, 0, 2.0);
        trip.put(4, 1, 1.0);
        trip.put(4, 2, 5.0);
        trip.put(4, 3, 1.0);
        let u = Vector::from(&[-629.0 / 98.0, 237.0 / 49.0, -53.0 / 49.0, 62.0 / 49.0, 23.0 / 14.0]);
        let correct_v = &[-2.0, 4.0, 3.0, -5.0, 1.0];
        let v = trip.mat_vec_mul(&u)?;
        assert_vec_approx_eq!(v.as_data(), correct_v, 1e-14);
        Ok(())
    }
}
