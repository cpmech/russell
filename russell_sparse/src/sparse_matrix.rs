use super::{to_i32, CooMatrix, CscMatrix, CsrMatrix, Symmetry};
use crate::StrError;
use russell_lab::{Matrix, Vector};
use russell_openblas::idamax;

pub struct SparseMatrix {
    coo: Option<CooMatrix>,
    csc: Option<CscMatrix>,
    csr: Option<CsrMatrix>,
}

impl SparseMatrix {
    pub fn new_coo(
        nrow: usize,
        ncol: usize,
        max_nnz: usize,
        symmetry: Option<Symmetry>,
        one_based: bool,
    ) -> Result<Self, StrError> {
        Ok(SparseMatrix {
            coo: Some(CooMatrix::new(nrow, ncol, max_nnz, symmetry, one_based)?),
            csc: None,
            csr: None,
        })
    }

    pub fn new_csc(
        nrow: usize,
        ncol: usize,
        col_pointers: Vec<i32>,
        row_indices: Vec<i32>,
        values: Vec<f64>,
        symmetry: Option<Symmetry>,
    ) -> Result<Self, StrError> {
        Ok(SparseMatrix {
            coo: None,
            csc: Some(CscMatrix::new(nrow, ncol, col_pointers, row_indices, values, symmetry)?),
            csr: None,
        })
    }

    pub fn new_csr(
        nrow: usize,
        ncol: usize,
        row_pointers: Vec<i32>,
        col_indices: Vec<i32>,
        values: Vec<f64>,
        symmetry: Option<Symmetry>,
    ) -> Result<Self, StrError> {
        Ok(SparseMatrix {
            coo: None,
            csc: None,
            csr: Some(CsrMatrix::new(nrow, ncol, row_pointers, col_indices, values, symmetry)?),
        })
    }

    pub fn from_coo(coo: CooMatrix) -> Self {
        SparseMatrix {
            coo: Some(coo),
            csc: None,
            csr: None,
        }
    }

    pub fn from_csc(csc: CscMatrix) -> Self {
        SparseMatrix {
            coo: None,
            csc: Some(csc),
            csr: None,
        }
    }

    pub fn from_csr(csr: CsrMatrix) -> Self {
        SparseMatrix {
            coo: None,
            csc: None,
            csr: Some(csr),
        }
    }

    /// Returns information about the dimensions and symmetry type
    ///
    /// Returns `(nrow, ncol, nnz, symmetry)`
    ///
    /// Priority: CSC -> CSR -> COO
    pub fn get_info(&self) -> (usize, usize, usize, Option<Symmetry>) {
        match &self.csc {
            Some(csc) => csc.get_info(),
            None => match &self.csr {
                Some(csr) => csr.get_info(),
                None => self.coo.as_ref().unwrap().get_info(), // unwrap OK because at least one mat must be available
            },
        }
    }

    /// Returns the maximum absolute value among all values
    ///
    /// Priority: CSC -> CSR -> COO
    pub fn get_max_abs_value(&self) -> f64 {
        let values = match &self.csc {
            Some(csc) => &csc.values,
            None => match &self.csr {
                Some(csr) => &csr.values,
                None => &self.coo.as_ref().unwrap().values, // unwrap OK because at least one mat must be available
            },
        };
        let n = to_i32(values.len());
        let idx = idamax(n, values, 1);
        f64::abs(values[idx as usize])
    }

    /// Performs the matrix-vector multiplication
    ///
    /// Priority: CSC -> CSR -> COO
    pub fn mat_vec_mul(&self, v: &mut Vector, alpha: f64, u: &Vector) -> Result<(), StrError> {
        match &self.csc {
            Some(csc) => csc.mat_vec_mul(v, alpha, u),
            None => match &self.csr {
                Some(csr) => csr.mat_vec_mul(v, alpha, u),
                None => self.coo.as_ref().unwrap().mat_vec_mul(v, alpha, u), // unwrap OK because at least one mat must be available
            },
        }
    }

    pub fn to_dense(&self, a: &mut Matrix) -> Result<(), StrError> {
        match &self.csc {
            Some(csc) => csc.to_dense(a),
            None => match &self.csr {
                Some(csr) => csr.to_dense(a),
                None => self.coo.as_ref().unwrap().to_dense(a), // unwrap OK because at least one mat must be available
            },
        }
    }

    // COO ------------------------------------------------------------------------

    pub fn put(&mut self, i: usize, j: usize, aij: f64) -> Result<(), StrError> {
        match &mut self.coo {
            Some(coo) => coo.put(i, j, aij),
            None => Err("COO matrix is not available to put items"),
        }
    }

    pub fn reset(&mut self) -> Result<(), StrError> {
        match &mut self.coo {
            Some(coo) => {
                coo.reset();
                Ok(())
            }
            None => Err("COO matrix is not available to reset nnz counter"),
        }
    }

    pub fn get_coo(&self) -> Result<&CooMatrix, StrError> {
        match &self.coo {
            Some(coo) => Ok(coo),
            None => Err("COO matrix is not available"),
        }
    }

    pub fn get_coo_mut(&mut self) -> Result<&mut CooMatrix, StrError> {
        match &mut self.coo {
            Some(coo) => Ok(coo),
            None => Err("COO matrix is not available"),
        }
    }

    // CSC ------------------------------------------------------------------------

    pub fn get_csc(&self) -> Result<&CscMatrix, StrError> {
        match &self.csc {
            Some(csc) => Ok(csc),
            None => Err("CSC matrix is not available"),
        }
    }

    pub fn get_csc_mut(&mut self) -> Result<&mut CscMatrix, StrError> {
        match &mut self.csc {
            Some(csc) => Ok(csc),
            None => Err("CSC matrix is not available"),
        }
    }

    /// Returns the CSC or creates a CSC from COO or updates the CSC from COO
    ///
    /// Priority: COO -> CSC
    pub fn get_csc_or_from_coo(&mut self) -> Result<&CscMatrix, StrError> {
        match &self.coo {
            Some(coo) => match &mut self.csc {
                Some(csc) => {
                    csc.update_from_coo(coo)?;
                    Ok(self.csc.as_ref().unwrap())
                }
                None => {
                    self.csc = Some(CscMatrix::from_coo(coo)?);
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

    pub fn get_csr(&self) -> Result<&CsrMatrix, StrError> {
        match &self.csr {
            Some(csr) => Ok(csr),
            None => Err("CSR matrix is not available"),
        }
    }

    pub fn get_csr_mut(&mut self) -> Result<&mut CsrMatrix, StrError> {
        match &mut self.csr {
            Some(csr) => Ok(csr),
            None => Err("CSR matrix is not available"),
        }
    }

    /// Returns the CSR or creates a CSR from COO or updates the CSR from COO
    ///
    /// Priority: COO -> CSR
    pub fn get_csr_or_from_coo(&mut self) -> Result<&CsrMatrix, StrError> {
        match &self.coo {
            Some(coo) => match &mut self.csr {
                Some(csr) => {
                    csr.update_from_coo(coo)?;
                    Ok(self.csr.as_ref().unwrap())
                }
                None => {
                    self.csr = Some(CsrMatrix::from_coo(coo)?);
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
