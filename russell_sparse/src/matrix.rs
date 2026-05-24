use crate::{CooMatrix, CsrMatrix, StrError};

#[derive(Clone, Debug)]
pub enum SparseMatrix {
    Coo(CooMatrix),
    Csr(CsrMatrix),
}

impl SparseMatrix {
    /// Returns a reference to the inner COO matrix, if the format matches
    pub fn as_coo(&self) -> Result<&CooMatrix, StrError> {
        match self {
            SparseMatrix::Coo(mat) => Ok(mat),
            _ => Err("wrong matrix format: COO required"),
        }
    }

    /// Returns a reference to the inner CSR matrix, if the format matches
    pub fn as_csr(&self) -> Result<&CsrMatrix, StrError> {
        match self {
            SparseMatrix::Csr(mat) => Ok(mat),
            _ => Err("wrong matrix format: CSR required"),
        }
    }
}

impl From<CooMatrix> for SparseMatrix {
    fn from(mat: CooMatrix) -> Self {
        SparseMatrix::Coo(mat)
    }
}

impl From<CsrMatrix> for SparseMatrix {
    fn from(mat: CsrMatrix) -> Self {
        SparseMatrix::Csr(mat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CooMatrix, CsrMatrix, Sym};

    #[test]
    fn sparse_matrix_enum_works() {
        let mut coo = CooMatrix::new(2, 2, 3, Sym::No).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(0, 1, 2.0).unwrap();
        coo.put(1, 1, 3.0).unwrap();

        let sparse_coo = SparseMatrix::from(coo.clone());
        assert!(sparse_coo.as_coo().is_ok());
        assert!(sparse_coo.as_csr().is_err());

        // Fixed: Use CsrMatrix::from_coo() instead of From trait
        let csr = CsrMatrix::from_coo(&coo).unwrap();
        let sparse_csr = SparseMatrix::from(csr);
        assert!(sparse_csr.as_csr().is_ok());
        assert!(sparse_csr.as_coo().is_err());
    }
}