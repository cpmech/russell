use std::convert::TryFrom;
use std::fmt;

#[repr(C)]
pub struct ExternalSparseTriplet {
    data: [u8; 0],
    marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

extern "C" {
    pub fn new_sparse_triplet(m: i32, n: i32, max: i32) -> *mut ExternalSparseTriplet;
    pub fn drop_sparse_triplet(trip: *mut ExternalSparseTriplet);
    pub fn sparse_triplet_put(trip: *mut ExternalSparseTriplet, i: i32, j: i32, x: f64) -> i32;
    pub fn sparse_triplet_restart(trip: *mut ExternalSparseTriplet) -> i32;
}

#[allow(dead_code)]

/// Holds triples (i,j,x) representing a sparse matrix
pub struct SparseTriplet {
    pub(crate) nrow: usize,     // [i32] number of rows
    pub(crate) ncol: usize,     // [i32] number of columns
    pub(crate) pos: usize,      // [i32] current index => nnz in the end
    pub(crate) max: usize,      // [i32] max allowed number of entries
    pub(crate) one_based: bool, // indices (i; j) start with 1 instead of 0 (e.g. for MUMPS)
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
    ///                      one_based = false\n\
    ///                      symmetric = false\n\
    ///                      =========================";
    /// assert_eq!(format!("{}", trip), correct);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(nrow: usize, ncol: usize, max: usize) -> Result<Self, &'static str> {
        let m = i32::try_from(nrow).unwrap();
        let n = i32::try_from(ncol).unwrap();
        let max_i32 = i32::try_from(max).unwrap();
        unsafe {
            let data = new_sparse_triplet(m, n, max_i32);
            if data.is_null() {
                return Err("c-code failed to allocate SparseTriplet");
            }
            Ok(SparseTriplet {
                nrow,
                ncol,
                pos: 0,
                max,
                one_based: false,
                symmetric: false,
                data,
            })
        }
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
        write!(
            f,
            "=========================\n\
             SparseTriplet\n\
             -------------------------\n\
             nrow      = {}\n\
             ncol      = {}\n\
             max       = {}\n\
             pos       = {}\n\
             one_based = {}\n\
             symmetric = {}\n\
             =========================",
            self.nrow, self.ncol, self.max, self.pos, self.one_based, self.symmetric
        )?;
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    // use russell_chk::*;

    #[test]
    fn new_works() -> Result<(), &'static str> {
        let trip = SparseTriplet::new(3, 3, 5)?;
        assert_eq!(trip.nrow, 3);
        assert_eq!(trip.ncol, 3);
        assert_eq!(trip.pos, 0);
        assert_eq!(trip.max, 5);
        assert_eq!(trip.one_based, false);
        assert_eq!(trip.symmetric, false);
        Ok(())
    }

    #[test]
    fn info_works() -> Result<(), &'static str> {
        let trip = SparseTriplet::new(3, 3, 5)?;
        let correct: &str = "=========================\n\
                             SparseTriplet\n\
                             -------------------------\n\
                             nrow      = 3\n\
                             ncol      = 3\n\
                             max       = 5\n\
                             pos       = 0\n\
                             one_based = false\n\
                             symmetric = false\n\
                             =========================";
        assert_eq!(format!("{}", trip), correct);
        Ok(())
    }
}
