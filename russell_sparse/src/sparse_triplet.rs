use russell_lab::*;
use std::convert::TryFrom;
use std::fmt;

use crate::C_HAS_ERROR;

#[repr(C)]
pub struct ExternalSparseTriplet {
    data: [u8; 0],
    marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

extern "C" {
    pub fn new_sparse_triplet(max: i32) -> *mut ExternalSparseTriplet;
    pub fn drop_sparse_triplet(trip: *mut ExternalSparseTriplet);
    pub fn sparse_triplet_set(
        trip: *mut ExternalSparseTriplet,
        pos: i32,
        i: i32,
        j: i32,
        x: f64,
    ) -> i32;
}

/// Holds triples (i,j,x) representing a sparse matrix
pub struct SparseTriplet {
    pub(crate) nrow: usize,     // [i32] number of rows
    pub(crate) ncol: usize,     // [i32] number of columns
    pub(crate) pos: usize,      // [i32] current index => nnz in the end
    pub(crate) max: usize,      // [i32] max allowed number of entries
    pub(crate) symmetric: bool, // symmetric matrix?, but WITHOUT both sides of the diagonal

    data: *mut ExternalSparseTriplet,
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
    /// `nrow` -- The number of rows of the sparse matrix
    /// `ncol` -- The number of columns of the sparse matrix
    /// `max` -- The maximum number fo non-zero values in the sparse matrix
    ///
    /// # Example
    /// ```
    /// # fn main() -> Result<(), &'static str> {
    /// use russell_sparse::*;
    /// let trip = SparseTriplet::new(3, 3, 5)?;
    /// let correct: &str = "=========================\n\
    ///                      SparseTriplet\n\
    ///                      -------------------------\n\
    ///                      nrow      = 3\n\
    ///                      ncol      = 3\n\
    ///                      max       = 5\n\
    ///                      pos       = 0\n\
    ///                      symmetric = false\n\
    ///                      =========================";
    /// assert_eq!(format!("{}", trip), correct);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(nrow: usize, ncol: usize, max: usize) -> Result<Self, &'static str> {
        if nrow == 0 || ncol == 0 || max == 0 {
            return Err("nrow, ncol, and max must all be greater than zero");
        }
        let max_i32 = i32::try_from(max).unwrap();
        unsafe {
            let data = new_sparse_triplet(max_i32);
            if data.is_null() {
                return Err("c-code failed to allocate SparseTriplet");
            }
            Ok(SparseTriplet {
                nrow,
                ncol,
                pos: 0,
                max,
                symmetric: false,
                data,
            })
        }
    }

    /// Puts the next triple (i,j,x) into the Triplet
    /// # Example
    /// ```
    /// # fn main() -> Result<(), &'static str> {
    /// use russell_sparse::*;
    /// let mut trip = SparseTriplet::new(2, 2, 1)?;
    /// trip.put(0, 0, 1.0)?;
    /// let correct: &str = "=========================\n\
    ///                      SparseTriplet\n\
    ///                      -------------------------\n\
    ///                      nrow      = 2\n\
    ///                      ncol      = 2\n\
    ///                      max       = 1\n\
    ///                      pos       = 1 (FULL)\n\
    ///                      symmetric = false\n\
    ///                      =========================";
    /// assert_eq!(format!("{}", trip), correct);
    /// # Ok(())
    /// # }
    /// ```
    pub fn put(&mut self, i: usize, j: usize, x: f64) -> Result<(), &'static str> {
        if i >= self.nrow {
            return Err("i index must be smaller than nrow");
        }
        if j >= self.ncol {
            return Err("j index must be smaller than ncol");
        }
        if self.pos >= self.max {
            return Err("max number of entries reached");
        }
        let i_i32 = i32::try_from(i).unwrap();
        let j_i32 = i32::try_from(j).unwrap();
        let pos_i32 = i32::try_from(self.pos).unwrap();
        unsafe {
            let res = sparse_triplet_set(self.data, pos_i32, i_i32, j_i32, x);
            if res == C_HAS_ERROR {
                return Err("c-code failed to put (i,j,x) triple");
            }
            self.pos += 1;
        }
        Ok(())
    }
}

impl Drop for SparseTriplet {
    /// Tells the c-code to release memory
    fn drop(&mut self) {
        unsafe {
            drop_sparse_triplet(self.data);
        }
    }
}

impl fmt::Display for SparseTriplet {
    /// Implements the Display trait
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let pos = if self.pos == self.max {
            format!("{} (FULL)", self.pos)
        } else {
            format!("{}", self.pos)
        };
        write!(
            f,
            "=========================\n\
             SparseTriplet\n\
             -------------------------\n\
             nrow      = {}\n\
             ncol      = {}\n\
             max       = {}\n\
             pos       = {}\n\
             symmetric = {}\n\
             =========================",
            self.nrow, self.ncol, self.max, pos, self.symmetric
        )?;
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_fails_on_wrong_dims() {
        assert_eq!(
            SparseTriplet::new(0, 3, 5).err(),
            Some("nrow, ncol, and max must all be greater than zero")
        );
        assert_eq!(
            SparseTriplet::new(3, 0, 5).err(),
            Some("nrow, ncol, and max must all be greater than zero")
        );
        assert_eq!(
            SparseTriplet::new(3, 3, 0).err(),
            Some("nrow, ncol, and max must all be greater than zero")
        );
    }

    #[test]
    fn new_works() -> Result<(), &'static str> {
        let trip = SparseTriplet::new(3, 3, 5)?;
        assert_eq!(trip.nrow, 3);
        assert_eq!(trip.ncol, 3);
        assert_eq!(trip.pos, 0);
        assert_eq!(trip.max, 5);
        assert_eq!(trip.symmetric, false);
        Ok(())
    }

    #[test]
    fn display_trait_works() -> Result<(), &'static str> {
        let trip = SparseTriplet::new(3, 3, 5)?;
        let correct: &str = "=========================\n\
                             SparseTriplet\n\
                             -------------------------\n\
                             nrow      = 3\n\
                             ncol      = 3\n\
                             max       = 5\n\
                             pos       = 0\n\
                             symmetric = false\n\
                             =========================";
        assert_eq!(format!("{}", trip), correct);
        Ok(())
    }

    #[test]
    fn put_fails_on_wrong_values() -> Result<(), &'static str> {
        let mut trip = SparseTriplet::new(1, 1, 1)?;
        assert_eq!(
            trip.put(1, 0, 0.0),
            Err("i index must be smaller than nrow")
        );
        assert_eq!(
            trip.put(0, 1, 0.0),
            Err("j index must be smaller than ncol")
        );
        trip.put(0, 0, 0.0)?; // << all spots occupied
        assert_eq!(trip.put(0, 0, 0.0), Err("max number of entries reached"));
        Ok(())
    }

    #[test]
    fn put_works() -> Result<(), &'static str> {
        let mut trip = SparseTriplet::new(3, 3, 5)?;
        trip.put(0, 0, 1.0)?;
        assert_eq!(trip.pos, 1);
        trip.put(0, 1, 2.0)?;
        assert_eq!(trip.pos, 2);
        trip.put(1, 0, 3.0)?;
        assert_eq!(trip.pos, 3);
        trip.put(1, 1, 4.0)?;
        assert_eq!(trip.pos, 4);
        trip.put(2, 2, 5.0)?;
        assert_eq!(trip.pos, 5);
        Ok(())
    }
}
