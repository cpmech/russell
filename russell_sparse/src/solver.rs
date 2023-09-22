use super::{CooMatrix, Genie, Ordering, Scaling, SolverMUMPS, SolverUMFPACK};
use crate::StrError;
use russell_lab::Vector;

/// Holds optional settings for the sparse solver
pub struct Settings {
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
}

impl Settings {
    /// Allocates a new instance with default values
    pub fn new() -> Self {
        Settings {
            ordering: Ordering::Auto,
            scaling: Scaling::Auto,
            compute_determinant: false,
            mumps_pct_inc_workspace: 100,
            mumps_max_work_memory: 0,
            mumps_openmp_num_threads: 0,
        }
    }
}

/// Defines a common interface for sparse solvers
pub trait SolverTrait {
    /// Configures the solver (before initialization)
    fn configure(&mut self, settings: Settings);

    /// Initializes the C interface to the underlying solver (Genie)
    ///
    /// # Input
    ///
    /// * `coo` -- the CooMatrix representing the sparse coefficient matrix.
    ///   Note that only symmetry/storage equal to Lower or Full are allowed by MUMPS.
    fn initialize(&mut self, coo: &CooMatrix) -> Result<(), StrError>;

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
    pub fn new(selection: Genie) -> Result<Self, StrError> {
        let actual: Box<dyn SolverTrait> = match selection {
            Genie::Mumps => Box::new(SolverMUMPS::new()?),
            Genie::Umfpack => Box::new(SolverUMFPACK::new()?),
        };
        Ok(Solver { actual })
    }
}
