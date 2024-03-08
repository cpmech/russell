use super::{ComplexSparseMatrix, SparseMatrix};
use crate::StrError;
use num_complex::Complex64;
use russell_lab::{complex_vec_norm, complex_vec_update, cpx, ComplexVector};
use russell_lab::{find_index_abs_max, vec_norm, vec_update, Norm, Vector};
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
    /// Computes verification data for a sparse system
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
    ///     let mut coo = SparseMatrix::new_coo(nrow, ncol, nnz, None)?;
    ///     coo.put(0, 0, 1.0)?;
    ///     coo.put(0, 2, 4.0)?;
    ///     coo.put(1, 1, 2.0)?;
    ///     coo.put(2, 2, 3.0)?;
    ///
    ///     // check matrix
    ///     let mut a = coo.as_dense();
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
    ///     let verify = VerifyLinSys::from(&coo, &x, &rhs)?;
    ///     assert_eq!(verify.max_abs_a, 4.0);
    ///     assert_eq!(verify.max_abs_ax, 5.0);
    ///     assert_eq!(verify.max_abs_diff, 0.0);
    ///     assert_eq!(verify.relative_error, 0.0);
    ///     Ok(())
    /// }
    /// ```
    pub fn from(mat: &SparseMatrix, x: &Vector, rhs: &Vector) -> Result<Self, StrError> {
        let (nrow, ncol, _, _) = mat.get_info();
        if x.dim() != ncol {
            return Err("x.dim() must be equal to ncol");
        }
        if rhs.dim() != nrow {
            return Err("rhs.dim() must be equal to nrow");
        }

        // compute max_abs_a
        let values = mat.get_values();
        if values.len() < 1 {
            return Err("matrix is empty");
        }
        let idx = find_index_abs_max(values);
        let max_abs_a = f64::abs(values[idx as usize]);

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

    /// Computes verification data for a complex sparse system
    ///
    /// ```text
    /// diff : = | a  ⋅  x - rhs|
    ///          (m,n)  (n)  (m)
    /// ```
    pub fn from_complex(mat: &ComplexSparseMatrix, x: &ComplexVector, rhs: &ComplexVector) -> Result<Self, StrError> {
        let (nrow, ncol, _, _) = mat.get_info();
        if x.dim() != ncol {
            return Err("x.dim() must be equal to ncol");
        }
        if rhs.dim() != nrow {
            return Err("rhs.dim() must be equal to nrow");
        }

        // compute max_abs_a
        let values = mat.get_values();
        if values.len() < 1 {
            return Err("matrix is empty");
        }
        let nnz = values.len();
        let mut max_abs_a = 0.0;
        for k in 0..nnz {
            let abs = values[k].norm();
            if abs > max_abs_a {
                max_abs_a = abs;
            }
        }

        // compute max_abs_ax
        let mut ax = ComplexVector::new(nrow);
        mat.mat_vec_mul(&mut ax, cpx!(1.0, 0.0), &x).unwrap(); // unwrap bc already checked dims
        let max_abs_ax = complex_vec_norm(&ax, Norm::Max);

        // compute max_abs_diff
        complex_vec_update(&mut ax, cpx!(-1.0, 0.0), &rhs).unwrap(); // ax := ax - rhs
        let max_abs_diff = complex_vec_norm(&ax, Norm::Max);

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
    use crate::{ComplexSparseMatrix, Samples, SparseMatrix};
    use num_complex::Complex64;
    use russell_lab::{approx_eq, cpx, ComplexVector, Vector};

    #[test]
    fn from_captures_errors() {
        // real
        let coo = SparseMatrix::new_coo(2, 1, 1, None).unwrap();
        let x = Vector::new(1);
        let rhs = Vector::new(2);
        assert_eq!(VerifyLinSys::from(&coo, &x, &rhs).err(), Some("matrix is empty"));
        let x_wrong = Vector::new(2);
        let rhs_wrong = Vector::new(1);
        assert_eq!(
            VerifyLinSys::from(&coo, &x_wrong, &rhs).err(),
            Some("x.dim() must be equal to ncol")
        );
        assert_eq!(
            VerifyLinSys::from(&coo, &x, &rhs_wrong).err(),
            Some("rhs.dim() must be equal to nrow")
        );
        // complex
        let coo = ComplexSparseMatrix::new_coo(2, 1, 1, None).unwrap();
        let x = ComplexVector::new(1);
        let rhs = ComplexVector::new(2);
        assert_eq!(
            VerifyLinSys::from_complex(&coo, &x, &rhs).err(),
            Some("matrix is empty")
        );
        let x_wrong = ComplexVector::new(2);
        let rhs_wrong = ComplexVector::new(1);
        assert_eq!(
            VerifyLinSys::from_complex(&coo, &x_wrong, &rhs).err(),
            Some("x.dim() must be equal to ncol")
        );
        assert_eq!(
            VerifyLinSys::from_complex(&coo, &x, &rhs_wrong).err(),
            Some("rhs.dim() must be equal to nrow")
        );
    }

    #[test]
    fn new_works() {
        // 1  3 -2
        // 3  5  6
        // 2  4  3
        let mut coo = SparseMatrix::new_coo(3, 3, 9, None).unwrap();
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
        let verify = VerifyLinSys::from(&coo, &x, &rhs).unwrap();
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
        let verify = VerifyLinSys::from(&mat, &x, &rhs).unwrap();
        assert_eq!(verify.max_abs_a, 15.0);
        assert_eq!(verify.max_abs_ax, 12.0);
        assert_eq!(verify.max_abs_diff, 12.0);
        approx_eq(verify.relative_error, 12.0 / (15.0 + 1.0), 1e-15);

        let verify = VerifyLinSys::from(&mat, &x, &Vector::from(a_times_x)).unwrap();
        assert_eq!(verify.max_abs_a, 15.0);
        assert_eq!(verify.max_abs_ax, 12.0);
        assert_eq!(verify.max_abs_diff, 0.0);
        approx_eq(verify.relative_error, 0.0, 1e-15);

        // CSC
        let mat = SparseMatrix::from_csc(csc);
        let verify = VerifyLinSys::from(&mat, &x, &rhs).unwrap();
        assert_eq!(verify.max_abs_a, 15.0);
        assert_eq!(verify.max_abs_ax, 12.0);
        assert_eq!(verify.max_abs_diff, 12.0);
        approx_eq(verify.relative_error, 12.0 / (15.0 + 1.0), 1e-15);

        // CSR
        let mat = SparseMatrix::from_csr(csr);
        let verify = VerifyLinSys::from(&mat, &x, &rhs).unwrap();
        assert_eq!(verify.max_abs_a, 15.0);
        assert_eq!(verify.max_abs_ax, 12.0);
        assert_eq!(verify.max_abs_diff, 12.0);
        approx_eq(verify.relative_error, 12.0 / (15.0 + 1.0), 1e-15);
    }

    #[test]
    fn new_complex_matrix_works() {
        // 4+4i    .     2+2i
        //  .      1     3+3i
        //  .     5+5i   1+1i
        //  1      .      .
        let (coo, _, _, _) = Samples::complex_rectangular_4x3();
        let mat = ComplexSparseMatrix::from_coo(coo);
        let x = ComplexVector::from(&[cpx!(1.0, 2.0), cpx!(2.0, -1.0), cpx!(0.0, 1.0)]);

        // zero error
        let rhs = ComplexVector::from(&[cpx!(-6.0, 14.0), cpx!(-1.0, 2.0), cpx!(14.0, 6.0), cpx!(1.0, 2.0)]);
        let verify = VerifyLinSys::from_complex(&mat, &x, &rhs).unwrap();
        approx_eq(verify.max_abs_a, 7.0710678118654755, 1e-15);
        approx_eq(verify.max_abs_ax, 15.231546211727817, 1e-15);
        approx_eq(verify.max_abs_diff, 0.0, 1e-15);
        approx_eq(verify.relative_error, 0.0, 1e-15);

        // with error
        let rhs = ComplexVector::from(&[cpx!(-6.0, 14.0), cpx!(-1.0, 2.0), cpx!(14.0, 6.0), cpx!(1.0, 0.0)]);
        let verify = VerifyLinSys::from_complex(&mat, &x, &rhs).unwrap();
        approx_eq(verify.max_abs_a, 7.0710678118654755, 1e-15);
        approx_eq(verify.max_abs_ax, 15.231546211727817, 1e-15);
        approx_eq(verify.max_abs_diff, 2.0, 1e-15);
        approx_eq(verify.relative_error, 2.0 / (7.0710678118654755 + 1.0), 1e-15);
    }
}
