#[cfg(feature = "with_mumps")]
use super::SolverMUMPS;

use super::{CooMatrix, Genie, LinSolParams, SolverKLU, SolverUMFPACK, StatsLinSol, SparseMatrix};
use crate::StrError;
use russell_lab::Vector;

/// Defines a unified interface for linear system solvers
pub trait LinSolTrait: Send { 
    /// For direct solvers: performs factorization
    /// For iterative solvers: builds the preconditioner
    fn setup(&mut self, mat: &SparseMatrix, params: Option<LinSolParams>) -> Result<(), StrError>;

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
    #[deprecated(since = "1.16.0", note = "Please use `setup()` instead for both direct and iterative solvers")]
    fn factorize(&mut self, mat: &CooMatrix, params: Option<LinSolParams>) -> Result<(), StrError>;

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
    fn solve(&mut self, x: &mut Vector, rhs: &Vector, verbose: bool) -> Result<(), StrError>;

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
pub struct LinSolver<'a> {
    /// Holds the actual implementation
    pub actual: Box<dyn Send + LinSolTrait + 'a>,
}

impl<'a> LinSolver<'a> {
    /// Allocates a new instance
    ///
    /// # Input
    ///
    /// * `genie` -- the actual implementation that does all the magic
    pub fn new(genie: Genie) -> Result<Self, StrError> {
        #[cfg(feature = "with_mumps")]
        let actual: Box<dyn Send + LinSolTrait> = match genie {
            Genie::Klu => Box::new(SolverKLU::new()?),
            Genie::Mumps => Box::new(SolverMUMPS::new()?),
            Genie::Umfpack => Box::new(SolverUMFPACK::new()?),
        };
        #[cfg(not(feature = "with_mumps"))]
        let actual: Box<dyn Send + LinSolTrait> = match genie {
            Genie::Klu => Box::new(SolverKLU::new()?),
            Genie::Mumps => return Err("MUMPS solver is not available"),
            Genie::Umfpack => Box::new(SolverUMFPACK::new()?),
        };
        Ok(LinSolver { actual })
    }

    /// Computes the solution of a linear system
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
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::{vec_approx_eq, Vector};
    /// use russell_sparse::prelude::*;
    /// use russell_sparse::StrError;
    /// use russell_sparse::SparseMatrix;
    ///
    /// fn main() -> Result<(), StrError> {
    ///     // constants
    ///     let ndim = 3; // number of rows = number of columns
    ///     let nnz = 5; // number of non-zero values
    ///
    ///     // allocate the coefficient matrix
    ///     let mut mat = CooMatrix::new(ndim, ndim, nnz, Sym::No)?;
    ///     mat.put(0, 0, 0.2)?;
    ///     mat.put(0, 1, 0.2)?;
    ///     mat.put(1, 0, 0.5)?;
    ///     mat.put(1, 1, -0.25)?;
    ///     mat.put(2, 2, 0.25)?;
    ///
    ///     // print matrix
    ///     let mut a = mat.as_dense();
    ///     let correct = "┌                   ┐\n\
    ///                    │   0.2   0.2     0 │\n\
    ///                    │   0.5 -0.25     0 │\n\
    ///                    │     0     0  0.25 │\n\
    ///                    └                   ┘";
    ///     assert_eq!(format!("{}", a), correct);
    ///
    ///     // allocate the right-hand side vector
    ///     let rhs = Vector::from(&[1.0, 1.0, 1.0]);
    ///
    ///     // calculate the solution
    ///     let mut x = Vector::new(ndim);
    ///     LinSolver::compute(Genie::Umfpack, &mut x, &SparseMatrix::from(mat), &rhs, None)?;
    ///     let correct = vec![3.0, 2.0, 4.0];
    ///     vec_approx_eq(&x, &correct, 1e-14);
    ///     Ok(())
    /// }
    /// ```
    pub fn compute(
        genie: Genie,
        x: &mut Vector,
        mat: &SparseMatrix,
        rhs: &Vector,
        params: Option<LinSolParams>,
    ) -> Result<Self, StrError> {
        let mut solver = LinSolver::new(genie)?;
        solver.actual.setup(mat, params)?;
        let verbose = if let Some(p) = params { p.verbose } else { false };
        solver.actual.solve(x, rhs, verbose)?;
        Ok(solver)
    }

    ///
    /// Please use the new `compute` function with `SparseMatrix` instead.
    #[deprecated(since = "1.16.0", note = "Please use `compute` with the unified `SparseMatrix` type instead")]
    pub fn compute_with_coo(
        genie: Genie,
        x: &mut Vector,
        mat: &CooMatrix,
        rhs: &Vector,
        params: Option<LinSolParams>,
    ) -> Result<Self, StrError> {
        let sparse_mat = SparseMatrix::from(mat.clone());
        Self::compute(genie, x, &sparse_mat, rhs, params)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::LinSolver;
    use crate::{Genie, Samples};
    use russell_lab::{vec_approx_eq, Vector};
    use super::SparseMatrix;

    #[cfg(feature = "with_mumps")]
    use serial_test::serial;

    #[test]
    fn lin_solver_compute_works_klu() {
        let (coo, _, _, _) = Samples::mkl_symmetric_5x5_full();
        let mut x = Vector::new(5);
        let rhs = Vector::from(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        LinSolver::compute(Genie::Klu, &mut x, &SparseMatrix::from(coo), &rhs, None).unwrap();
        let x_correct = vec![-979.0 / 3.0, 983.0, 1961.0 / 12.0, 398.0, 123.0 / 2.0];
        vec_approx_eq(&x, &x_correct, 1e-10);
    }

    #[test]
    #[serial]
    #[cfg(feature = "with_mumps")]
    fn lin_solver_compute_works_mumps() {
        let (coo, _, _, _) = Samples::mkl_symmetric_5x5_lower(true, false);
        let sparse = SparseMatrix::from(coo);
        let mut x = Vector::new(5);
        let rhs = Vector::from(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        LinSolver::compute(Genie::Mumps, &mut x, &sparse, &rhs, None).unwrap();
        let x_correct = vec![-979.0 / 3.0, 983.0, 1961.0 / 12.0, 398.0, 123.0 / 2.0];
        vec_approx_eq(&x, &x_correct, 1e-10);
    }

    #[test]
    fn lin_solver_compute_works_umfpack() {
        let (coo, _, _, _) = Samples::mkl_symmetric_5x5_full();
        let mut x = Vector::new(5);
        let rhs = Vector::from(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        LinSolver::compute(Genie::Umfpack, &mut x, &SparseMatrix::from(coo), &rhs, None).unwrap();
        let x_correct = vec![-979.0 / 3.0, 983.0, 1961.0 / 12.0, 398.0, 123.0 / 2.0];
        vec_approx_eq(&x, &x_correct, 1e-10);
    }

    #[test]
    #[cfg(feature = "with_mumps")]
    fn lin_solver_compute_with_coo_works() {
        let (coo, _, _, _) = Samples::mkl_symmetric_5x5_full();
        let mut x = Vector::new(5);
        let rhs = Vector::from(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        #[allow(deprecated)]
        LinSolver::compute_with_coo(Genie::Umfpack, &mut x, &coo, &rhs, None).unwrap();
        let x_correct = vec![-979.0 / 3.0, 983.0, 1961.0 / 12.0, 398.0, 123.0 / 2.0];
        vec_approx_eq(&x, &x_correct, 1e-10);
    }

#[cfg(test)]
    mod tests {
        use crate::{CooMatrix, LinSolTrait, Samples, SparseMatrix, Sym};
        use russell_lab::{Vector, vec_approx_eq};

        fn create_test_coo() -> CooMatrix {
            let (coo, _, _, _) = Samples::umfpack_unsymmetric_5x5();
            coo
        }

        fn get_rhs() -> Vector {
            Vector::from(&[8.0, 45.0, -3.0, 3.0, 19.0])
        }

        fn get_correct() -> Vec<f64> {
            vec![1.0, 2.0, 3.0, 4.0, 5.0]
        }

        #[test]
        #[cfg(feature = "with_mumps")]
        fn backward_compatibility_works() {
            let coo = create_test_coo();
            let mut umfpack = crate::SolverUMFPACK::new().unwrap();
            
            #[allow(deprecated)]
            umfpack.factorize(&coo, None).unwrap();
            
            let mut x = Vector::new(5);
            let b = get_rhs();
            umfpack.solve(&mut x, &b, false).unwrap();
            
            let correct = get_correct();
            vec_approx_eq(&x, &correct, 1e-14);
        }

        #[test]
        #[cfg(feature = "with_mumps")]
        fn new_setup_interface_works() {
            let coo = create_test_coo();
            let sparse = SparseMatrix::from(coo);
            let mut umfpack = crate::SolverUMFPACK::new().unwrap();
            
            umfpack.setup(&sparse, None).unwrap();
            
            let mut x = Vector::new(5);
            let b = get_rhs();
            umfpack.solve(&mut x, &b, false).unwrap();
            
            let correct = get_correct();
            vec_approx_eq(&x, &correct, 1e-14);
        }
    }
}
