use super::{NumCooMatrix, NumCscMatrix, NumCsrMatrix, Symmetry};
use crate::StrError;
use num_traits::{Num, NumCast};
use russell_lab::{NumMatrix, NumVector};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::ops::{AddAssign, MulAssign};

/// Unifies the sparse matrix representations by wrapping COO, CSC, and CSR structures
///
/// This structure is a wrapper around COO, CSC, or CSR matrices. For instance:
///
/// ```text
/// pub struct NumSparseMatrix<T> {
///     coo: Option<NumCooMatrix<T>>,
///     csc: Option<NumCscMatrix<T>>,
///     csr: Option<NumCsrMatrix<T>>,
/// }
/// ```
///
/// # Notes
///
/// 1. At least one of [NumCooMatrix], [NumCscMatrix], or [NumCsrMatrix] will be `Some`
/// 2. `(COO and CSC)` or `(COO and CSR)` pairs may be `Some` at the same time
/// 3. When getting data/information from the sparse matrix, the default priority is `CSC -> CSR -> COO`
/// 4. If needed, the CSC or CSR are automatically computed from COO
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NumSparseMatrix<T>
where
    T: AddAssign + MulAssign + Num + NumCast + Copy + DeserializeOwned + Serialize,
{
    // Holds the COO version
    #[serde(bound(deserialize = "NumCooMatrix<T>: Deserialize<'de>"))]
    coo: Option<NumCooMatrix<T>>,

    // Holds the CSC version (will not co-exist with CSR)
    #[serde(bound(deserialize = "NumCscMatrix<T>: Deserialize<'de>"))]
    csc: Option<NumCscMatrix<T>>,

    // Holds the CSR version (will not co-exist with CSC)
    #[serde(bound(deserialize = "NumCsrMatrix<T>: Deserialize<'de>"))]
    csr: Option<NumCsrMatrix<T>>,
}

impl<T> NumSparseMatrix<T>
where
    T: AddAssign + MulAssign + Num + NumCast + Copy + DeserializeOwned + Serialize,
{
    /// Allocates a new sparse matrix as COO to be later updated with put and reset methods
    ///
    /// **Note:** This is the most convenient structure for recurrent updates of the sparse
    /// matrix data; e.g. in finite element simulation codes. See the [NumCooMatrix::put] and
    /// [NumCooMatrix::reset] functions for more details.
    ///
    /// # Input
    ///
    /// * `nrow` -- (≥ 1) Is the number of rows of the sparse matrix (must be fit i32)
    /// * `ncol` -- (≥ 1) Is the number of columns of the sparse matrix (must be fit i32)
    /// * `max_nnz` -- (≥ 1) Maximum number of entries ≥ nnz (number of non-zeros),
    ///   including entries with repeated indices. (must be fit i32)
    /// * `symmetry` -- Defines the symmetry/storage, if any
    pub fn new_coo(nrow: usize, ncol: usize, max_nnz: usize, symmetry: Option<Symmetry>) -> Result<Self, StrError> {
        Ok(NumSparseMatrix {
            coo: Some(NumCooMatrix::new(nrow, ncol, max_nnz, symmetry)?),
            csc: None,
            csr: None,
        })
    }

    /// Allocates a new sparse matrix as CSC from the underlying arrays
    ///
    /// **Note:** The column pointers and row indices must be **sorted** in ascending order.
    ///
    /// # Input
    ///
    /// * `nrow` -- (≥ 1) number of rows
    /// * `ncol` -- (≥ 1) number of columns
    /// * `col_pointers` -- (len = ncol + 1) columns pointers with the last entry corresponding
    ///   to the number of non-zero values (sorted)
    /// * `row_indices` -- (len = nnz) row indices (sorted)
    /// * `values` -- the non-zero components of the matrix
    ///
    /// The following conditions must be satisfied (nnz is the number of non-zeros
    /// and nnz_dup is the number of non-zeros with possible duplicates):
    ///
    /// ```text
    /// nrow ≥ 1
    /// ncol ≥ 1
    /// col_pointers.len() = ncol + 1
    /// row_indices.len() = nnz_dup
    /// values.len() = nnz_dup
    /// nnz = col_pointers[ncol] ≥ 1
    /// nnz_dup ≥ nnz
    /// ```
    pub fn new_csc(
        nrow: usize,
        ncol: usize,
        col_pointers: Vec<i32>,
        row_indices: Vec<i32>,
        values: Vec<T>,
        symmetry: Option<Symmetry>,
    ) -> Result<Self, StrError> {
        Ok(NumSparseMatrix {
            coo: None,
            csc: Some(NumCscMatrix::new(
                nrow,
                ncol,
                col_pointers,
                row_indices,
                values,
                symmetry,
            )?),
            csr: None,
        })
    }

    /// Allocates a new sparse matrix as CSR from the underlying arrays
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
    pub fn new_csr(
        nrow: usize,
        ncol: usize,
        row_pointers: Vec<i32>,
        col_indices: Vec<i32>,
        values: Vec<T>,
        symmetry: Option<Symmetry>,
    ) -> Result<Self, StrError> {
        Ok(NumSparseMatrix {
            coo: None,
            csc: None,
            csr: Some(NumCsrMatrix::new(
                nrow,
                ncol,
                row_pointers,
                col_indices,
                values,
                symmetry,
            )?),
        })
    }

    /// Creates a new sparse matrix from COO (move occurs)
    pub fn from_coo(coo: NumCooMatrix<T>) -> Self {
        NumSparseMatrix {
            coo: Some(coo),
            csc: None,
            csr: None,
        }
    }

    /// Creates a new sparse matrix from CSC (move occurs)
    pub fn from_csc(csc: NumCscMatrix<T>) -> Self {
        NumSparseMatrix {
            coo: None,
            csc: Some(csc),
            csr: None,
        }
    }

    /// Creates a new sparse matrix from CSR (move occurs)
    pub fn from_csr(csr: NumCsrMatrix<T>) -> Self {
        NumSparseMatrix {
            coo: None,
            csc: None,
            csr: Some(csr),
        }
    }

    /// Returns information about the dimensions and symmetry type
    ///
    /// Returns `(nrow, ncol, nnz, symmetry)`
    ///
    /// **Priority**: CSC -> CSR -> COO
    pub fn get_info(&self) -> (usize, usize, usize, Symmetry) {
        match &self.csc {
            Some(csc) => csc.get_info(),
            None => match &self.csr {
                Some(csr) => csr.get_info(),
                None => self.coo.as_ref().unwrap().get_info(), // unwrap OK because at least one mat must be available
            },
        }
    }

    /// Get an access to the values
    ///
    /// **Priority**: CSC -> CSR -> COO
    pub fn get_values(&self) -> &[T] {
        match &self.csc {
            Some(csc) => csc.get_values(),
            None => match &self.csr {
                Some(csr) => csr.get_values(),
                None => self.coo.as_ref().unwrap().get_values(), // unwrap OK because at least one mat must be available
            },
        }
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
    /// **Priority**: CSC -> CSR -> COO
    pub fn mat_vec_mul(&self, v: &mut NumVector<T>, alpha: T, u: &NumVector<T>) -> Result<(), StrError> {
        match &self.csc {
            Some(csc) => csc.mat_vec_mul(v, alpha, u),
            None => match &self.csr {
                Some(csr) => csr.mat_vec_mul(v, alpha, u),
                None => self.coo.as_ref().unwrap().mat_vec_mul(v, alpha, u), // unwrap OK because at least one mat must be available
            },
        }
    }

    /// Converts the sparse matrix to dense format
    ///
    /// **Priority**: CSC -> CSR -> COO
    pub fn as_dense(&self) -> NumMatrix<T> {
        match &self.csc {
            Some(csc) => csc.as_dense(),
            None => match &self.csr {
                Some(csr) => csr.as_dense(),
                None => self.coo.as_ref().unwrap().as_dense(), // unwrap OK because at least one mat must be available
            },
        }
    }

    /// Converts the sparse matrix to dense format
    ///
    /// **Priority**: CSC -> CSR -> COO
    pub fn to_dense(&self, a: &mut NumMatrix<T>) -> Result<(), StrError> {
        match &self.csc {
            Some(csc) => csc.to_dense(a),
            None => match &self.csr {
                Some(csr) => csr.to_dense(a),
                None => self.coo.as_ref().unwrap().to_dense(a), // unwrap OK because at least one mat must be available
            },
        }
    }

    // COO ------------------------------------------------------------------------

    /// Puts a new entry and updates pos (may be duplicate)
    ///
    /// # Input
    ///
    /// * `i` -- row index (indices start at zero; zero-based)
    /// * `j` -- column index (indices start at zero; zero-based)
    /// * `aij` -- the value A(i,j)
    pub fn put(&mut self, i: usize, j: usize, aij: T) -> Result<(), StrError> {
        match &mut self.coo {
            Some(coo) => coo.put(i, j, aij),
            None => Err("COO matrix is not available to put items"),
        }
    }

    /// Resets the position of the current non-zero value
    ///
    /// This function allows using `put` all over again.
    pub fn reset(&mut self) -> Result<(), StrError> {
        match &mut self.coo {
            Some(coo) => {
                coo.reset();
                Ok(())
            }
            None => Err("COO matrix is not available to reset nnz counter"),
        }
    }

    /// Returns a read-only access to the COO matrix, if available
    pub fn get_coo(&self) -> Result<&NumCooMatrix<T>, StrError> {
        match &self.coo {
            Some(coo) => Ok(coo),
            None => Err("COO matrix is not available"),
        }
    }

    /// Returns a read-write access to the COO matrix, if available
    pub fn get_coo_mut(&mut self) -> Result<&mut NumCooMatrix<T>, StrError> {
        match &mut self.coo {
            Some(coo) => Ok(coo),
            None => Err("COO matrix is not available"),
        }
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
    pub fn assign(&mut self, alpha: T, other: &NumSparseMatrix<T>) -> Result<(), StrError> {
        match &mut self.coo {
            Some(coo) => coo.assign(alpha, other.get_coo()?),
            None => Err("COO matrix is not available to perform assignment"),
        }
    }

    /// Augments this matrix with the entries of another matrix (scaled)
    ///
    /// Effectively, performs:
    ///
    /// ```text
    /// this += α · other
    /// ```
    ///
    /// **Warning:** make sure to allocate `max_nnz ≥ nnz(this) + nnz(other)`.
    pub fn augment(&mut self, alpha: T, other: &NumSparseMatrix<T>) -> Result<(), StrError> {
        match &mut self.coo {
            Some(coo) => coo.augment(alpha, other.get_coo()?),
            None => Err("COO matrix is not available to augment"),
        }
    }

    // CSC ------------------------------------------------------------------------

    /// Returns a read-only access to the CSC matrix, if available
    pub fn get_csc(&self) -> Result<&NumCscMatrix<T>, StrError> {
        match &self.csc {
            Some(csc) => Ok(csc),
            None => Err("CSC matrix is not available"),
        }
    }

    /// Returns a read-write access to the CSC matrix, if available
    pub fn get_csc_mut(&mut self) -> Result<&mut NumCscMatrix<T>, StrError> {
        match &mut self.csc {
            Some(csc) => Ok(csc),
            None => Err("CSC matrix is not available"),
        }
    }

    /// Returns the CSC or creates a CSC from COO or updates the CSC from COO
    ///
    /// This function is convenient to update the COO recurrently and later
    /// automatically get the converted CSC matrix.
    ///
    /// **Priority**: COO -> CSC
    pub fn get_csc_or_from_coo(&mut self) -> Result<&NumCscMatrix<T>, StrError> {
        match &self.coo {
            Some(coo) => match &mut self.csc {
                Some(csc) => {
                    csc.update_from_coo(coo).unwrap(); // unwrap because csc cannot be wrong (created here)
                    Ok(self.csc.as_ref().unwrap())
                }
                None => {
                    self.csc = Some(NumCscMatrix::from_coo(coo)?);
                    Ok(self.csc.as_ref().unwrap())
                }
            },
            None => match &self.csc {
                Some(csc) => Ok(csc),
                None => Err("CSC is not available and COO matrix is not available to convert to CSC"),
            },
        }
    }

    // CSR ------------------------------------------------------------------------

    /// Returns a read-only access to the CSR matrix, if available
    pub fn get_csr(&self) -> Result<&NumCsrMatrix<T>, StrError> {
        match &self.csr {
            Some(csr) => Ok(csr),
            None => Err("CSR matrix is not available"),
        }
    }

    /// Returns a read-write access to the CSR matrix, if available
    pub fn get_csr_mut(&mut self) -> Result<&mut NumCsrMatrix<T>, StrError> {
        match &mut self.csr {
            Some(csr) => Ok(csr),
            None => Err("CSR matrix is not available"),
        }
    }

    /// Returns the CSR or creates a CSR from COO or updates the CSR from COO
    ///
    /// This function is convenient to update the COO recurrently and later
    /// automatically get the converted CSR matrix.
    ///
    /// **Priority**: COO -> CSR
    pub fn get_csr_or_from_coo(&mut self) -> Result<&NumCsrMatrix<T>, StrError> {
        match &self.coo {
            Some(coo) => match &mut self.csr {
                Some(csr) => {
                    csr.update_from_coo(coo).unwrap(); // unwrap because csr cannot be wrong (created here)
                    Ok(self.csr.as_ref().unwrap())
                }
                None => {
                    self.csr = Some(NumCsrMatrix::from_coo(coo)?);
                    Ok(self.csr.as_ref().unwrap())
                }
            },
            None => match &self.csr {
                Some(csr) => Ok(csr),
                None => Err("CSR is not available and COO matrix is not available to convert to CSR"),
            },
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::NumSparseMatrix;
    use crate::{Samples, Symmetry};
    use russell_lab::{vec_approx_eq, Matrix, Vector};

    #[test]
    fn new_functions_work() {
        // COO
        NumSparseMatrix::<f64>::new_coo(1, 1, 1, None).unwrap();
        assert_eq!(
            NumSparseMatrix::<f64>::new_coo(0, 1, 1, None).err(),
            Some("nrow must be ≥ 1")
        );
        // CSC
        NumSparseMatrix::<f64>::new_csc(1, 1, vec![0, 1], vec![0], vec![0.0], None).unwrap();
        assert_eq!(
            NumSparseMatrix::<f64>::new_csc(0, 1, vec![0, 1], vec![0], vec![0.0], None).err(),
            Some("nrow must be ≥ 1")
        );
        // CSR
        NumSparseMatrix::<f64>::new_csr(1, 1, vec![0, 1], vec![0], vec![0.0], None).unwrap();
        assert_eq!(
            NumSparseMatrix::<f64>::new_csr(0, 1, vec![0, 1], vec![0], vec![0.0], None).err(),
            Some("nrow must be ≥ 1")
        );
    }

    #[test]
    fn getters_work() {
        // test matrices
        let (coo, csc, csr, _) = Samples::rectangular_1x2(false, false);
        let mut a = Matrix::new(1, 2);
        let x = Vector::from(&[2.0, 1.0]);
        let mut wrong = Vector::new(2);
        // COO
        let coo_mat = NumSparseMatrix::<f64>::from_coo(coo);
        assert_eq!(coo_mat.get_info(), (1, 2, 2, Symmetry::No));
        assert_eq!(coo_mat.get_coo().unwrap().get_info(), (1, 2, 2, Symmetry::No));
        assert_eq!(coo_mat.get_csc().err(), Some("CSC matrix is not available"));
        assert_eq!(coo_mat.get_csr().err(), Some("CSR matrix is not available"));
        assert_eq!(coo_mat.get_values(), &[10.0, 20.0]);
        // CSC
        let csc_mat = NumSparseMatrix::<f64>::from_csc(csc);
        assert_eq!(csc_mat.get_info(), (1, 2, 2, Symmetry::No));
        assert_eq!(csc_mat.get_csc().unwrap().get_info(), (1, 2, 2, Symmetry::No));
        assert_eq!(csc_mat.get_coo().err(), Some("COO matrix is not available"));
        assert_eq!(csc_mat.get_csr().err(), Some("CSR matrix is not available"));
        assert_eq!(csc_mat.get_values(), &[10.0, 20.0]);
        // CSR
        let csr_mat = NumSparseMatrix::<f64>::from_csr(csr);
        assert_eq!(csr_mat.get_info(), (1, 2, 2, Symmetry::No));
        assert_eq!(csr_mat.get_csr().unwrap().get_info(), (1, 2, 2, Symmetry::No));
        assert_eq!(csr_mat.get_csc().err(), Some("CSC matrix is not available"));
        assert_eq!(csr_mat.get_coo().err(), Some("COO matrix is not available"));
        assert_eq!(csr_mat.get_values(), &[10.0, 20.0]);
        // COO, CSC, CSR
        let mut ax = Vector::new(1);
        for mat in [&coo_mat, &csc_mat, &csr_mat] {
            mat.mat_vec_mul(&mut ax, 2.0, &x).unwrap();
            vec_approx_eq(&ax.as_data(), &[80.0], 1e-15);
            assert_eq!(
                mat.mat_vec_mul(&mut wrong, 1.0, &x).err(),
                Some("v vector is incompatible")
            );
            mat.to_dense(&mut a).unwrap();
            assert_eq!(a.dims(), (1, 2));
            assert_eq!(a.get(0, 0), 10.0);
            assert_eq!(a.get(0, 1), 20.0);
            let aa = mat.as_dense();
            assert_eq!(aa.dims(), (1, 2));
            assert_eq!(aa.get(0, 0), 10.0);
            assert_eq!(aa.get(0, 1), 20.0);
        }
    }

    #[test]
    fn setters_work() {
        // test matrices
        let (coo, csc, csr, _) = Samples::rectangular_1x2(false, false);
        let mut other = NumSparseMatrix::<f64>::new_coo(1, 1, 1, None).unwrap();
        let mut wrong = NumSparseMatrix::<f64>::new_coo(1, 1, 3, None).unwrap();
        other.put(0, 0, 2.0).unwrap();
        wrong.put(0, 0, 1.0).unwrap();
        wrong.put(0, 0, 2.0).unwrap();
        wrong.put(0, 0, 3.0).unwrap();
        // COO
        let mut coo_mat = NumSparseMatrix::<f64>::from_coo(coo);
        assert_eq!(coo_mat.get_coo_mut().unwrap().get_info(), (1, 2, 2, Symmetry::No));
        assert_eq!(coo_mat.get_csc_mut().err(), Some("CSC matrix is not available"));
        assert_eq!(coo_mat.get_csr_mut().err(), Some("CSR matrix is not available"));
        let mut empty = NumSparseMatrix::<f64>::new_coo(1, 1, 1, None).unwrap();
        assert_eq!(empty.get_csc_or_from_coo().err(), Some("COO to CSC requires nnz > 0"));
        assert_eq!(empty.get_csr_or_from_coo().err(), Some("COO to CSR requires nnz > 0"));
        // CSC
        let mut csc_mat = NumSparseMatrix::<f64>::from_csc(csc);
        assert_eq!(csc_mat.get_csc_mut().unwrap().get_info(), (1, 2, 2, Symmetry::No));
        assert_eq!(csc_mat.get_coo_mut().err(), Some("COO matrix is not available"));
        assert_eq!(csc_mat.get_csr_mut().err(), Some("CSR matrix is not available"));
        assert_eq!(
            csc_mat.get_csc_or_from_coo().unwrap().get_info(),
            (1, 2, 2, Symmetry::No)
        );
        assert_eq!(
            csc_mat.get_csr_or_from_coo().err(),
            Some("CSR is not available and COO matrix is not available to convert to CSR")
        );
        assert_eq!(
            csc_mat.put(0, 0, 0.0).err(),
            Some("COO matrix is not available to put items")
        );
        assert_eq!(
            csc_mat.reset().err(),
            Some("COO matrix is not available to reset nnz counter")
        );
        assert_eq!(
            csc_mat.assign(4.0, &other).err(),
            Some("COO matrix is not available to perform assignment")
        );
        assert_eq!(
            csc_mat.augment(4.0, &other).err(),
            Some("COO matrix is not available to augment")
        );
        // CSR
        let mut csr_mat = NumSparseMatrix::<f64>::from_csr(csr);
        assert_eq!(csr_mat.get_csr_mut().unwrap().get_info(), (1, 2, 2, Symmetry::No));
        assert_eq!(csr_mat.get_csc_mut().err(), Some("CSC matrix is not available"));
        assert_eq!(csr_mat.get_coo_mut().err(), Some("COO matrix is not available"));
        assert_eq!(
            csr_mat.get_csr_or_from_coo().unwrap().get_info(),
            (1, 2, 2, Symmetry::No)
        );
        assert_eq!(
            csr_mat.get_csc_or_from_coo().err(),
            Some("CSC is not available and COO matrix is not available to convert to CSC")
        );
        assert_eq!(
            csr_mat.put(0, 0, 0.0).err(),
            Some("COO matrix is not available to put items")
        );
        assert_eq!(
            csr_mat.reset().err(),
            Some("COO matrix is not available to reset nnz counter")
        );
        assert_eq!(
            csr_mat.assign(4.0, &other).err(),
            Some("COO matrix is not available to perform assignment")
        );
        assert_eq!(
            csr_mat.augment(4.0, &other).err(),
            Some("COO matrix is not available to augment")
        );
        // COO
        let mut coo = NumSparseMatrix::<f64>::new_coo(2, 2, 1, None).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        assert_eq!(
            coo.put(1, 1, 2.0).err(),
            Some("COO matrix: max number of items has been reached")
        );
        coo.reset().unwrap();
        coo.put(1, 1, 2.0).unwrap();
        // COO (assign)
        let mut this = NumSparseMatrix::<f64>::new_coo(1, 1, 1, None).unwrap();
        this.put(0, 0, 8000.0).unwrap();
        this.assign(4.0, &other).unwrap();
        assert_eq!(
            format!("{}", this.as_dense()),
            "┌   ┐\n\
             │ 8 │\n\
             └   ┘"
        );
        assert_eq!(
            this.assign(2.0, &wrong).err(),
            Some("COO matrix: max number of items has been reached")
        );
        assert_eq!(this.assign(2.0, &csc_mat).err(), Some("COO matrix is not available"));
        // COO (augment)
        let mut this = NumSparseMatrix::<f64>::new_coo(1, 1, 1 + 1, None).unwrap();
        this.put(0, 0, 100.0).unwrap();
        this.augment(4.0, &other).unwrap();
        assert_eq!(
            format!("{}", this.as_dense()),
            "┌     ┐\n\
             │ 108 │\n\
             └     ┘"
        );
        assert_eq!(this.augment(2.0, &csc_mat).err(), Some("COO matrix is not available"));
    }

    #[test]
    fn get_csc_or_from_coo_works() {
        // ┌       ┐
        // │ 10 20 │
        // └       ┘
        let (coo, _, _, _) = Samples::rectangular_1x2(false, false);
        let mut mat = NumSparseMatrix::<f64>::from_coo(coo);
        let csc = mat.get_csc_or_from_coo().unwrap(); // will create a new csc
        assert_eq!(csc.get_values(), &[10.0, 20.0]);
        let coo_internal = mat.get_coo_mut().unwrap();
        let source = coo_internal.get_values_mut();
        source[0] = 30.0; // change a value
        let csc = mat.get_csc_or_from_coo().unwrap(); // will update existent csc
        assert_eq!(csc.get_values(), &[30.0, 20.0]);
    }

    #[test]
    fn get_csr_or_from_coo_works() {
        // ┌       ┐
        // │ 10 20 │
        // └       ┘
        let (coo, _, _, _) = Samples::rectangular_1x2(false, false);
        let mut mat = NumSparseMatrix::<f64>::from_coo(coo);
        let csr = mat.get_csr_or_from_coo().unwrap(); // will create a new csr
        assert_eq!(csr.get_values(), &[10.0, 20.0]);
        let coo_internal = mat.get_coo_mut().unwrap();
        let source = coo_internal.get_values_mut();
        source[0] = 30.0; // change a value
        let csr = mat.get_csr_or_from_coo().unwrap(); // will update existent csr
        assert_eq!(csr.get_values(), &[30.0, 20.0]);
    }

    #[test]
    fn derive_methods_work() {
        let (coo, _, _, _) = Samples::tiny_1x1();
        let (nrow, ncol, nnz, symmetry) = coo.get_info();
        let mat = NumSparseMatrix::<f64>::from_coo(coo);
        let mut clone = mat.clone();
        clone.get_coo_mut().unwrap().values[0] *= 2.0;
        assert_eq!(mat.get_coo().unwrap().values[0], 123.0);
        assert_eq!(clone.get_coo().unwrap().values[0], 246.0);
        assert!(format!("{:?}", mat).len() > 0);
        let json = serde_json::to_string(&mat).unwrap();
        assert_eq!(
            json,
            r#"{"coo":{"symmetry":"No","nrow":1,"ncol":1,"nnz":1,"max_nnz":1,"indices_i":[0],"indices_j":[0],"values":[123.0]},"csc":null,"csr":null}"#
        );
        let from_json: NumSparseMatrix<f64> = serde_json::from_str(&json).unwrap();
        let (json_nrow, json_ncol, json_nnz, json_symmetry) = from_json.get_coo().unwrap().get_info();
        assert_eq!(json_symmetry, symmetry);
        assert_eq!(json_nrow, nrow);
        assert_eq!(json_ncol, ncol);
        assert_eq!(json_nnz, nnz);
        assert!(from_json.csc.is_none());
        assert!(from_json.csr.is_none());
    }
}
