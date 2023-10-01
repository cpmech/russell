use super::SparseMatrix;
use crate::StrError;
use russell_lab::{vec_norm, vec_update, Norm, Vector};
use serde::{Deserialize, Serialize};

/// Verifies the linear system a ⋅ x = rhs
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct VerifyLinSys {
    pub max_abs_a: f64,      // max abs a
    pub max_abs_ax: f64,     // max abs a ⋅ x
    pub max_abs_diff: f64,   // max abs diff = a ⋅ x - rhs
    pub relative_error: f64, // max_abs_diff / (max_abs_a + 1)
}

impl VerifyLinSys {
    /// Creates a new verification dataset
    ///
    /// ```text
    /// diff : = | a  ⋅  x - rhs|
    ///          (m,n)  (n)  (m)
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
    ///     Ok(())
    /// }
    /// ```
    pub fn new(mat: &SparseMatrix, x: &Vector, rhs: &Vector) -> Result<Self, StrError> {
        let (nrow, ncol, _, _) = mat.get_info();
        if x.dim() != ncol {
            return Err("x.dim() must be equal to ncol");
        }
        if rhs.dim() != nrow {
            return Err("rhs.dim() must be equal to nrow");
        }

        // compute max_abs_a
        let max_abs_a = mat.get_max_abs_value();

        // compute max_abs_ax
        let mut ax = Vector::new(nrow);
        mat.mat_vec_mul(&mut ax, 1.0, &x).unwrap(); // unwrap bc already checked dims
        let max_abs_ax = vec_norm(&ax, Norm::Max);

        // compute max_abs_diff
        vec_update(&mut ax, -1.0, &rhs).unwrap(); // ax := ax - rhs
        let max_abs_diff = vec_norm(&ax, Norm::Max);

        // compute relative_error
        let relative_error = max_abs_diff / (max_abs_a + 1.0);

        // results
        Ok(VerifyLinSys {
            max_abs_a,
            max_abs_ax,
            max_abs_diff,
            relative_error,
        })
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::VerifyLinSys;
    use crate::{Samples, SparseMatrix};
    use russell_chk::approx_eq;
    use russell_lab::Vector;

    #[test]
    fn new_captures_errors() {
        let coo = SparseMatrix::new_coo(2, 1, 1, None, false).unwrap();
        let x = Vector::new(1);
        let rhs = Vector::new(2);
        assert_eq!(VerifyLinSys::new(&coo, &x, &rhs).err(), None);
        let x_wrong = Vector::new(2);
        let rhs_wrong = Vector::new(1);
        assert_eq!(
            VerifyLinSys::new(&coo, &x_wrong, &rhs).err(),
            Some("x.dim() must be equal to ncol")
        );
        assert_eq!(
            VerifyLinSys::new(&coo, &x, &rhs_wrong).err(),
            Some("rhs.dim() must be equal to nrow")
        );
    }

    #[test]
    fn new_works() {
        // 1  3 -2
        // 3  5  6
        // 2  4  3
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
    }

    #[test]
    fn new_rectangular_matrix_works() {
        //   5  -2  .  1
        //  10  -4  .  2
        //  15  -6  .  3
        let (coo, csc, csr, _) = Samples::rectangular_3x4();
        let x = Vector::from(&[1.0, 3.0, 8.0, 5.0]);

        let rhs = Vector::from(&[0.0, 0.0, 0.0]);
        let a_times_x = &[4.0, 8.0, 12.0];

        // COO
        let mat = SparseMatrix::from_coo(coo);
        let verify = VerifyLinSys::new(&mat, &x, &rhs).unwrap();
        assert_eq!(verify.max_abs_a, 15.0);
        assert_eq!(verify.max_abs_ax, 12.0);
        assert_eq!(verify.max_abs_diff, 12.0);
        approx_eq(verify.relative_error, 12.0 / (15.0 + 1.0), 1e-15);

        let verify = VerifyLinSys::new(&mat, &x, &Vector::from(a_times_x)).unwrap();
        assert_eq!(verify.max_abs_a, 15.0);
        assert_eq!(verify.max_abs_ax, 12.0);
        assert_eq!(verify.max_abs_diff, 0.0);
        approx_eq(verify.relative_error, 0.0, 1e-15);

        // CSC
        let mat = SparseMatrix::from_csc(csc);
        let verify = VerifyLinSys::new(&mat, &x, &rhs).unwrap();
        assert_eq!(verify.max_abs_a, 15.0);
        assert_eq!(verify.max_abs_ax, 12.0);
        assert_eq!(verify.max_abs_diff, 12.0);
        approx_eq(verify.relative_error, 12.0 / (15.0 + 1.0), 1e-15);

        // CSR
        let mat = SparseMatrix::from_csr(csr);
        let verify = VerifyLinSys::new(&mat, &x, &rhs).unwrap();
        assert_eq!(verify.max_abs_a, 15.0);
        assert_eq!(verify.max_abs_ax, 12.0);
        assert_eq!(verify.max_abs_diff, 12.0);
        approx_eq(verify.relative_error, 12.0 / (15.0 + 1.0), 1e-15);
    }
}
