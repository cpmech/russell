#[cfg(feature = "with_mumps")]
use super::ComplexSolverMUMPS;

use super::{ComplexSparseMatrix, Genie, LinSolParams, StatsLinSol};
use crate::StrError;
use crate::{ComplexSolverKLU, ComplexSolverUMFPACK};
use russell_lab::ComplexVector;

/// Defines a unified interface for complex linear system solvers
pub trait ComplexLinSolTrait {
    /// Performs the factorization (and analysis/initialization if needed)
    ///
    /// # Input
    ///
    /// * `mat` -- The sparse matrix (COO, CSC, or CSR).
    /// * `params` -- configuration parameters; None => use default
    ///
    /// # Notes
    ///
    /// 1. The structure of the matrix (nrow, ncol, nnz, sym) must be
    ///    exactly the same among multiple calls to `factorize`. The values may differ
    ///    from call to call, nonetheless.
    /// 2. The first call to `factorize` will define the structure which must be
    ///    kept the same for the next calls.
    /// 3. If the structure of the matrix needs to be changed, the solver must
    ///    be "dropped" and a new solver allocated.
    fn factorize(&mut self, mat: &mut ComplexSparseMatrix, params: Option<LinSolParams>) -> Result<(), StrError>;

    /// Computes the solution of the linear system
    ///
    /// Solves the linear system:
    ///
    /// ```text
    ///   A   · x = rhs
    /// (m,n)  (n)  (m)
    /// ```
    ///
    /// # Output
    ///
    /// * `x` -- the vector of unknown values with dimension equal to mat.ncol
    ///
    /// # Input
    ///
    /// * `mat` -- the coefficient matrix A.
    /// * `rhs` -- the right-hand side vector with know values an dimension equal to mat.ncol
    /// * `verbose` -- shows messages
    ///
    /// **Warning:** the matrix must be same one used in `factorize`.
    fn solve(
        &mut self,
        x: &mut ComplexVector,
        mat: &ComplexSparseMatrix,
        rhs: &ComplexVector,
        verbose: bool,
    ) -> Result<(), StrError>;

    /// Updates the stats structure (should be called after solve)
    fn update_stats(&self, stats: &mut StatsLinSol);

    /// Returns the nanoseconds spent on initialize
    fn get_ns_init(&self) -> u128;

    /// Returns the nanoseconds spent on factorize
    fn get_ns_fact(&self) -> u128;

    /// Returns the nanoseconds spent on solve
    fn get_ns_solve(&self) -> u128;
}

/// Unifies the access to linear system solvers
pub struct ComplexLinSolver<'a> {
    /// Holds the actual implementation
    pub actual: Box<dyn Send + ComplexLinSolTrait + 'a>,
}

impl<'a> ComplexLinSolver<'a> {
    /// Allocates a new instance
    ///
    /// # Input
    ///
    /// * `genie` -- the actual implementation that does all the magic
    pub fn new(genie: Genie) -> Result<Self, StrError> {
        #[cfg(feature = "with_mumps")]
        let actual: Box<dyn Send + ComplexLinSolTrait> = match genie {
            Genie::Klu => Box::new(ComplexSolverKLU::new()?),
            Genie::Mumps => Box::new(ComplexSolverMUMPS::new()?),
            Genie::Umfpack => Box::new(ComplexSolverUMFPACK::new()?),
        };
        #[cfg(not(feature = "with_mumps"))]
        let actual: Box<dyn Send + ComplexLinSolTrait> = match genie {
            Genie::Klu => Box::new(ComplexSolverKLU::new()?),
            Genie::Mumps => return Err("MUMPS solver is not available"),
            Genie::Umfpack => Box::new(ComplexSolverUMFPACK::new()?),
        };
        Ok(ComplexLinSolver { actual })
    }

    /// Computes the solution of a complex linear system
    ///
    /// Solves the linear system:
    ///
    /// ```text
    ///   A   · x = rhs
    /// (m,n)  (n)  (m)
    /// ```
    ///
    /// # Output
    ///
    /// * `x` -- the vector of unknown values with dimension equal to mat.ncol
    ///
    /// # Input
    ///
    /// * `genie` -- the actual implementation that does all the magic
    /// * `mat` -- the matrix representing the sparse coefficient matrix A (see Notes below)
    /// * `rhs` -- the right-hand side vector with know values an dimension equal to coo.nrow
    /// * `verbose` -- shows messages
    ///
    /// # Notes
    ///
    /// 1. For symmetric matrices, `MUMPS` requires [crate::Sym::YesLower]
    /// 2. For symmetric matrices, `UMFPACK` requires [crate::Sym::YesFull]
    /// 4. This function calls the actual implementation (genie) via the functions `factorize`, and `solve`.
    /// 5. This function is best for a **single-use**, whereas the actual
    ///    solver should be considered for a recurrent use (e.g., inside a loop).
    pub fn compute(
        genie: Genie,
        x: &mut ComplexVector,
        mat: &mut ComplexSparseMatrix,
        rhs: &ComplexVector,
        params: Option<LinSolParams>,
    ) -> Result<Self, StrError> {
        let mut solver = ComplexLinSolver::new(genie)?;
        solver.actual.factorize(mat, params)?;
        let verbose = if let Some(p) = params { p.verbose } else { false };
        solver.actual.solve(x, mat, rhs, verbose)?;
        Ok(solver)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::ComplexLinSolver;
    use crate::{ComplexSparseMatrix, Genie, Samples};
    use russell_lab::{complex_vec_approx_eq, cpx, Complex64, ComplexVector};

    #[cfg(feature = "with_mumps")]
    use serial_test::serial;

    #[test]
    fn complex_lin_solver_compute_works_klu() {
        let (coo, _, _, _) = Samples::complex_symmetric_3x3_full();
        let mut mat = ComplexSparseMatrix::from_coo(coo);
        let mut x = ComplexVector::new(3);
        let rhs = ComplexVector::from(&[cpx!(-3.0, 3.0), cpx!(2.0, -2.0), cpx!(9.0, 7.0)]);
        ComplexLinSolver::compute(Genie::Klu, &mut x, &mut mat, &rhs, None).unwrap();
        let x_correct = &[cpx!(1.0, 1.0), cpx!(2.0, -2.0), cpx!(3.0, 3.0)];
        complex_vec_approx_eq(&x, x_correct, 1e-15);
    }

    #[test]
    #[serial]
    #[cfg(feature = "with_mumps")]
    fn complex_lin_solver_compute_works_mumps() {
        let (coo, _, _, _) = Samples::complex_symmetric_3x3_lower();
        let mut mat = ComplexSparseMatrix::from_coo(coo);
        let mut x = ComplexVector::new(3);
        let rhs = ComplexVector::from(&[cpx!(-3.0, 3.0), cpx!(2.0, -2.0), cpx!(9.0, 7.0)]);
        ComplexLinSolver::compute(Genie::Mumps, &mut x, &mut mat, &rhs, None).unwrap();
        let x_correct = &[cpx!(1.0, 1.0), cpx!(2.0, -2.0), cpx!(3.0, 3.0)];
        complex_vec_approx_eq(&x, x_correct, 1e-15);
    }

    #[test]
    fn complex_lin_solver_compute_works_umfpack() {
        let (coo, _, _, _) = Samples::complex_symmetric_3x3_full();
        let mut mat = ComplexSparseMatrix::from_coo(coo);
        let mut x = ComplexVector::new(3);
        let rhs = ComplexVector::from(&[cpx!(-3.0, 3.0), cpx!(2.0, -2.0), cpx!(9.0, 7.0)]);
        ComplexLinSolver::compute(Genie::Umfpack, &mut x, &mut mat, &rhs, None).unwrap();
        let x_correct = &[cpx!(1.0, 1.0), cpx!(2.0, -2.0), cpx!(3.0, 3.0)];
        complex_vec_approx_eq(&x, x_correct, 1e-15);
    }
}
