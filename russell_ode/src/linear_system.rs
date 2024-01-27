use crate::OdeParams;
use russell_lab::Vector;
use russell_sparse::{LinSolver, SparseMatrix};

/// Holds variables to solve the linear system A · u = b
pub(crate) struct LinearSystem<'a> {
    /// Total number of equations
    ndim: usize,

    /// Number of nonzero values
    nnz: usize,

    /// Coefficient matrix
    pub aa: SparseMatrix,

    /// Unknowns vector (the solution of the linear system)
    pub u: Vector,

    /// Right-hand size
    pub b: Vector,

    /// Linear solver
    pub solver: LinSolver<'a>,
}

impl<'a> LinearSystem<'a> {
    /// Allocates new instance
    pub fn new(params: &'a OdeParams, ndim: usize) -> Self {
        let nnz = 0; // TODO
        let symmetry = None;
        let one_based = false;
        LinearSystem {
            ndim,
            nnz,
            aa: SparseMatrix::new_coo(ndim, ndim, nnz, symmetry, one_based).unwrap(),
            u: Vector::new(ndim),
            b: Vector::new(ndim),
            solver: LinSolver::new(params.genie).unwrap(),
        }
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
