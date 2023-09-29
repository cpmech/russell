use super::SparseMatrix;
use crate::StrError;
use russell_lab::{vec_norm, vec_update, Norm, Stopwatch, Vector};

/// Verifies the linear system a ⋅ x = rhs
pub struct VerifyLinSys {
    pub max_abs_a: f64,      // max abs a
    pub max_abs_ax: f64,     // max abs a ⋅ x
    pub max_abs_diff: f64,   // max abs diff = a ⋅ x - rhs
    pub relative_error: f64, // max_abs_diff / (max_abs_a + 1)
    pub time_check: u128,    // elapsed time spent in the `new` method
}

impl VerifyLinSys {
    /// Creates a new verification dataset
    ///
    /// ```text
    /// diff : = |a ⋅ x - rhs|
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::{Matrix, Vector};
    /// use russell_sparse::prelude::*;
    /// use russell_sparse::StrError;
    ///
    /// fn main() -> Result<(), StrError> {
    ///     // set sparse matrix (3 x 3) with 4 non-zeros
    ///     let (nrow, ncol, nnz) = (3, 3, 4);
    ///     let mut coo = SparseMatrix::new_coo(nrow, ncol, nnz, None, false)?;
    ///     coo.put(0, 0, 1.0)?;
    ///     coo.put(0, 2, 4.0)?;
    ///     coo.put(1, 1, 2.0)?;
    ///     coo.put(2, 2, 3.0)?;
    ///
    ///     // check matrix
    ///     let mut a = Matrix::new(nrow, nrow);
    ///     coo.to_dense(&mut a)?;
    ///     let correct_a = "┌       ┐\n\
    ///                      │ 1 0 4 │\n\
    ///                      │ 0 2 0 │\n\
    ///                      │ 0 0 3 │\n\
    ///                      └       ┘";
    ///     assert_eq!(format!("{}", a), correct_a);
    ///
    ///     // verify lin-sys
    ///     let x = Vector::from(&[1.0, 1.0, 1.0]);
    ///     let rhs = Vector::from(&[5.0, 2.0, 3.0]);
    ///     let verify = VerifyLinSys::new(&coo, &x, &rhs)?;
    ///     assert_eq!(verify.max_abs_a, 4.0);
    ///     assert_eq!(verify.max_abs_ax, 5.0);
    ///     assert_eq!(verify.max_abs_diff, 0.0);
    ///     assert_eq!(verify.relative_error, 0.0);
    ///     assert!(verify.time_check > 0);
    ///     Ok(())
    /// }
    /// ```
    pub fn new(mat: &SparseMatrix, x: &Vector, rhs: &Vector) -> Result<Self, StrError> {
        let (nrow, ncol, _, _) = mat.get_info()?;
        if nrow != ncol {
            return Err("matrix must be square");
        }
        if x.dim() != nrow || rhs.dim() != nrow {
            return Err("vector dimensions are incompatible");
        }
        // start stopwatch
        let mut sw = Stopwatch::new("");

        // compute max_abs_a
        let max_abs_a = mat.get_max_abs_value()?;

        // compute max_abs_ax
        let mut ax = Vector::new(nrow);
        mat.mat_vec_mul(&mut ax, 1.0, &x)?;
        let max_abs_ax = vec_norm(&ax, Norm::Max);

        // compute max_abs_diff
        vec_update(&mut ax, -1.0, &rhs).unwrap(); // ax := ax - rhs
        let max_abs_diff = vec_norm(&ax, Norm::Max);

        // compute relative_error
        let relative_error = max_abs_diff / (max_abs_a + 1.0);

        // stop stopwatch
        let time_check = sw.stop();

        // results
        Ok(VerifyLinSys {
            max_abs_a,
            max_abs_ax,
            max_abs_diff,
            relative_error,
            time_check,
        })
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::VerifyLinSys;
    use crate::SparseMatrix;
    use russell_lab::Vector;

    #[test]
    fn new_fails_on_wrong_vectors() {
        let coo = SparseMatrix::new_coo(1, 1, 1, None, false).unwrap();
        let x = Vector::new(2);
        let rhs = Vector::new(3);
        let x_wrong = Vector::new(3);
        let rhs_wrong = Vector::new(2);
        assert_eq!(
            VerifyLinSys::new(&coo, &x_wrong, &rhs).err(),
            Some("vector dimensions are incompatible")
        );
        assert_eq!(
            VerifyLinSys::new(&coo, &x, &rhs_wrong).err(),
            Some("vector dimensions are incompatible")
        );
    }

    #[test]
    fn new_works() {
        // | 1  3 -2 |
        // | 3  5  6 |
        // | 2  4  3 |
        let mut coo = SparseMatrix::new_coo(3, 3, 9, None, false).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(0, 1, 3.0).unwrap();
        coo.put(0, 2, -2.0).unwrap();
        coo.put(1, 0, 3.0).unwrap();
        coo.put(1, 1, 5.0).unwrap();
        coo.put(1, 2, 6.0).unwrap();
        coo.put(2, 0, 2.0).unwrap();
        coo.put(2, 1, 4.0).unwrap();
        coo.put(2, 2, 3.0).unwrap();
        let x = Vector::from(&[-15.0, 8.0, 2.0]);
        let rhs = Vector::from(&[5.0, 7.0, 8.0]);
        let verify = VerifyLinSys::new(&coo, &x, &rhs).unwrap();
        assert_eq!(verify.max_abs_a, 6.0);
        assert_eq!(verify.max_abs_ax, 8.0);
        assert_eq!(verify.max_abs_diff, 0.0);
        assert_eq!(verify.relative_error, 0.0);
        assert!(verify.time_check > 0);
    }
}
