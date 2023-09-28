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

    /// Returns `(nrow, ncol, nnz, symmetry)`
    ///
    /// Priority: CSC -> CSR -> COO
    pub fn get_info(&self) -> (usize, usize, usize, Option<Symmetry>) {
        match &self.csc {
            Some(csc) => {
                let nnz = if csc.col_pointers.len() == csc.ncol + 1 {
                    csc.col_pointers[csc.ncol] as usize
                } else {
                    0
                };
                (csc.nrow, csc.ncol, nnz, csc.symmetry)
            }
            None => match &self.csr {
                Some(csr) => {
                    let nnz = if csr.row_pointers.len() == csr.nrow + 1 {
                        csr.row_pointers[csr.nrow] as usize
                    } else {
                        0
                    };
                    (csr.nrow, csr.ncol, nnz, csr.symmetry)
                }
                None => match &self.coo {
                    Some(coo) => (coo.nrow, coo.ncol, coo.nnz, coo.symmetry),
                    None => (0, 0, 0, None),
                },
            },
        }
    }

    /// Returns the maximum absolute value among all values
    ///
    /// Priority: CSC -> CSR -> COO
    pub fn get_max_abs_value(&self) -> Result<f64, StrError> {
        let values = match &self.csc {
            Some(csc) => &csc.values,
            None => match &self.csr {
                Some(csr) => &csr.values,
                None => match &self.coo {
                    Some(coo) => &coo.values,
                    None => return Err("no matrix is available"),
                },
            },
        };
        let n = to_i32(values.len())?;
        let idx = idamax(n, values, 1);
        if idx >= 0 && idx < n {
            Ok(f64::abs(values[idx as usize]))
        } else {
            Err("INTERNAL ERROR: cannot find max abs value")
        }
    }

    /// Performs the matrix-vector multiplication
    ///
    /// Priority: CSC -> CSR -> COO
    pub fn mat_vec_mul(&self, v: &mut Vector, alpha: f64, u: &Vector) -> Result<(), StrError> {
        match &self.csc {
            Some(csc) => csc.mat_vec_mul(v, alpha, u),
            None => match &self.csr {
                Some(csr) => csr.mat_vec_mul(v, alpha, u),
                None => match &self.coo {
                    Some(coo) => coo.mat_vec_mul(v, alpha, u),
                    None => Err("no matrix is available"),
                },
            },
        }
    }

    pub fn get_coo(&mut self) -> Result<&CooMatrix, StrError> {
        match &self.coo {
            Some(coo) => Ok(coo),
            None => Err("COO matrix is not available"),
        }
    }

    pub fn get_mut_coo(&mut self) -> Result<&mut CooMatrix, StrError> {
        match &mut self.coo {
            Some(coo) => Ok(coo),
            None => Err("COO matrix is not available"),
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

    pub fn to_dense(&self, a: &mut Matrix) -> Result<(), StrError> {
        match &self.csc {
            Some(csc) => csc.to_matrix(a),
            None => match &self.csr {
                Some(csr) => csr.to_matrix(a),
                None => match &self.coo {
                    Some(coo) => coo.to_matrix(a),
                    None => Err("no matrix is available"),
                },
            },
        }
    }

    pub fn csc_col_pointers(&self) -> Result<&Vec<i32>, StrError> {
        match &self.csc {
            Some(csc) => Ok(&csc.col_pointers),
            None => Err("CSC matrix is not available"),
        }
    }

    pub fn csc_row_indices(&self) -> Result<&Vec<i32>, StrError> {
        match &self.csc {
            Some(csc) => Ok(&csc.row_indices),
            None => Err("CSC matrix is not available"),
        }
    }

    pub fn csc_values(&self) -> Result<&Vec<f64>, StrError> {
        match &self.csc {
            Some(csc) => Ok(&csc.values),
            None => Err("CSC matrix is not available"),
        }
    }
}
