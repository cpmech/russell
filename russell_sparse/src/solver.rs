use super::{CooMatrix, Genie, Ordering, Scaling, SolverMUMPS, SolverUMFPACK, Symmetry};
use crate::StrError;
use russell_lab::Vector;

/// Defines the configuration parameters for the sparse solver
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConfigSolver {
    /// Defines the symmetric permutation (ordering)
    pub ordering: Ordering,

    /// Defines the scaling strategy
    pub scaling: Scaling,

    /// Requests that the determinant be computed
    ///
    /// **Note:** The determinant will be available after `factorize`
    pub compute_determinant: bool,

    /// Sets the % increase in the estimated working space (MUMPS only)
    ///
    /// **Note:** The default (recommended) value is 100 (%)
    pub mumps_pct_inc_workspace: usize,

    /// Sets the max size of the working memory in mega bytes (MUMPS only)
    ///
    /// **Note:** Set this value to 0 for an automatic configuration
    pub mumps_max_work_memory: usize,

    /// Defines the number of OpenMP threads (MUMPS only)
    ///
    /// **Note:** Set this value to 0 to allow an automatic detection
    pub mumps_openmp_num_threads: usize,

    /// Enforces the unsymmetric strategy, even for symmetric matrices (not recommended; UMFPACK only)
    pub umfpack_enforce_unsymmetric_strategy: bool,
}

impl ConfigSolver {
    /// Allocates a new instance with default values
    pub fn new() -> Self {
        ConfigSolver {
            ordering: Ordering::Auto,
            scaling: Scaling::Auto,
            compute_determinant: false,
            mumps_pct_inc_workspace: 100,
            mumps_max_work_memory: 0,
            mumps_openmp_num_threads: 0,
            umfpack_enforce_unsymmetric_strategy: false,
        }
    }
}

/// Defines a unified interface for sparse solvers
pub trait SolverTrait {
    /// Initializes the C interface to the underlying solver (Genie)
    ///
    /// Initializes the solver for the linear system:
    ///
    /// ```text
    /// A · x = rhs
    /// ```
    ///
    /// # Input
    ///
    /// * `ndim` -- number of rows = number of columns of the coefficient matrix A
    /// * `nnz` -- number of non-zero values on the coefficient matrix A
    /// * `symmetry` -- symmetry (or lack of it) type of the coefficient matrix A
    /// * `config` -- configuration parameters; None => use default
    ///
    /// # Notes
    ///
    /// * For symmetric matrices, `MUMPS` requires that the symmetry/storage be Lower or Full.
    /// * For symmetric matrices, `UMFPACK` requires that the symmetry/storage be Full.
    fn initialize(
        &mut self,
        ndim: usize,
        nnz: usize,
        symmetry: Option<Symmetry>,
        config: Option<ConfigSolver>,
    ) -> Result<(), StrError>;

    /// Performs the factorization (and analysis)
    ///
    /// **Note::** Initialize must be called first. Also, the dimension and symmetry/storage
    /// of the CooMatrix must be the same as the ones provided by `initialize`.
    ///
    /// # Input
    ///
    /// * `coo` -- The **same** matrix provided to `initialize`
    /// * `verbose` -- shows messages
    fn factorize(&mut self, coo: &CooMatrix, verbose: bool) -> Result<(), StrError>;

    /// Computes the solution of the linear system
    ///
    /// Solves the linear system:
    ///
    /// ```text
    /// A · x = rhs
    /// ```
    ///
    /// # Input
    ///
    /// * `x` -- the vector of unknown values with dimension equal to coo.nrow
    /// * `rhs` -- the right-hand side vector with know values an dimension equal to coo.nrow
    /// * `verbose` -- shows messages
    fn solve(&mut self, x: &mut Vector, rhs: &Vector, verbose: bool) -> Result<(), StrError>;

    /// Returns the determinant
    ///
    /// Returns the three values `(mantissa, base, exponent)`, such that the determinant is calculated by:
    ///
    /// ```text
    /// determinant = mantissa · pow(base, exponent)
    /// ```
    ///
    /// **Note:** This is only available if compute_determinant was requested.
    fn get_determinant(&self) -> (f64, f64, f64);

    /// Returns the ordering effectively used by the solver
    fn get_effective_ordering(&self) -> String;

    /// Returns the scaling effectively used by the solver
    fn get_effective_scaling(&self) -> String;

    /// Returns the strategy (concerning symmetry) effectively used by the solver
    ///
    /// For example, returns whether the `symmetric strategy` was used or not (UMFPACK only)
    fn get_effective_strategy(&self) -> String;

    /// Returns the name of the underlying solver (Genie)
    fn get_name(&self) -> String;
}

/// Unifies the access to sparse solvers
pub struct Solver<'a> {
    /// Holds the actual implementation
    pub actual: Box<dyn SolverTrait + 'a>,
}

impl<'a> Solver<'a> {
    /// Allocates a new instance
    ///
    /// # Input
    ///
    /// * `genie` -- the actual implementation that does all the magic
    pub fn new(genie: Genie) -> Result<Self, StrError> {
        let actual: Box<dyn SolverTrait> = match genie {
            Genie::Mumps => Box::new(SolverMUMPS::new()?),
            Genie::Umfpack => Box::new(SolverUMFPACK::new()?),
        };
        Ok(Solver { actual })
    }

    /// Computes the solution of a linear system
    ///
    /// Solves the linear system:
    ///
    /// ```text
    /// A · x = rhs
    /// ```
    ///
    /// # Input
    ///
    /// * `genie` -- the actual implementation that does all the magic
    /// * `coo` -- the CooMatrix representing the sparse coefficient matrix (see Notes below)
    /// * `x` -- the vector of unknown values with dimension equal to coo.nrow
    /// * `rhs` -- the right-hand side vector with know values an dimension equal to coo.nrow
    /// * `verbose` -- shows messages
    ///
    /// # Notes
    ///
    /// 1. For symmetric matrices, `MUMPS` requires that the symmetry/storage be Lower or Full.
    /// 2. For symmetric matrices, `UMFPACK` requires that the symmetry/storage be Full.
    /// 3. This function calls the actual implementation (genie) via the functions
    ///    `initialize`, `factorize`, and `solve`.
    /// 4. This function is best for a **single-use** need, whereas the actual
    ///    solver should be considered for a recurrent use (e.g., inside a loop).
    /// 5. Also, use the individual implementations if options such as ordering or scaling
    ///    need to be configured.
    pub fn compute(
        genie: Genie,
        coo: &CooMatrix,
        x: &mut Vector,
        rhs: &Vector,
        verbose: bool,
    ) -> Result<Self, StrError> {
        if coo.ncol != coo.nrow {
            return Err("the matrix must be square");
        }
        let mut solver = Solver::new(genie)?;
        solver.actual.initialize(coo.nrow, coo.max, coo.symmetry, None)?;
        solver.actual.factorize(coo, verbose)?;
        solver.actual.solve(x, rhs, verbose)?;
        Ok(solver)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{ConfigSolver, Solver};
    use crate::{Genie, Ordering, Samples, Scaling};
    use russell_chk::vec_approx_eq;
    use russell_lab::Vector;

    #[test]
    fn config_solver_new_works() {
        let config = ConfigSolver::new();
        assert_eq!(config.ordering, Ordering::Auto);
        assert_eq!(config.scaling, Scaling::Auto);
        assert_eq!(config.compute_determinant, false);
        assert_eq!(config.mumps_pct_inc_workspace, 100);
        assert_eq!(config.mumps_max_work_memory, 0);
        assert_eq!(config.mumps_openmp_num_threads, 0);
        assert!(!config.umfpack_enforce_unsymmetric_strategy);
    }

    #[test]
    fn solver_compute_works() {
        let (coo, _) = Samples::mkl_sample1_symmetric_full(false);
        let mut x = Vector::new(5);
        let rhs = Vector::from(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        Solver::compute(Genie::Umfpack, &coo, &mut x, &rhs, false).unwrap();
        let x_correct = vec![-979.0 / 3.0, 983.0, 1961.0 / 12.0, 398.0, 123.0 / 2.0];
        vec_approx_eq(x.as_data(), &x_correct, 1e-10);
    }
}
