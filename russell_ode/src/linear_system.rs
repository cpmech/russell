#![allow(unused)]

use crate::OdeParams;
use crate::StrError;
use russell_lab::Vector;
use russell_sparse::{LinSolver, SparseMatrix};

/// Holds variables to solve the linear system K · δy = r
pub(crate) struct LinearSystem<'a> {
    /// Total number of equations
    ndim: usize,

    /// Coefficient matrix K = h J - I
    pub kk: SparseMatrix,

    /// Unknowns vector (the solution of the linear system)
    pub dy: Vector,

    /// Right-hand size
    pub r: Vector,

    /// Linear solver
    pub solver: LinSolver<'a>,
}

impl<'a> LinearSystem<'a> {
    /// Allocates new instance
    pub fn new(params: &'a OdeParams, ndim: usize, nnz: usize) -> Self {
        let symmetry = None;
        let one_based = false;
        LinearSystem {
            ndim,
            kk: SparseMatrix::new_coo(ndim, ndim, nnz, symmetry, one_based).unwrap(),
            dy: Vector::new(ndim),
            r: Vector::new(ndim),
            solver: LinSolver::new(params.genie).unwrap(),
        }
    }

    pub fn compute_coefficient_matrix(&mut self) -> Result<(), StrError> {
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    #[test]
    fn new_handles_errors() {
        // TODO
    }
}
