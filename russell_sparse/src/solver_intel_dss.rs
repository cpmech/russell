#![allow(unused)]

use super::{to_i32, ConfigSolver, CooMatrix, SolverTrait, Symmetry};
use crate::{CsrMatrix, Storage, StrError};
use russell_lab::Vector;

const MKL_DSS_SUCCESS: i32 = 0;

/// Opaque struct holding a C-pointer to InterfaceIntelDSS
///
/// Reference: <https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs>
#[repr(C)]
struct InterfaceIntelDSS {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

extern "C" {
    fn solver_intel_dss_new() -> *mut InterfaceIntelDSS;
    fn solver_intel_dss_drop(solver: *mut InterfaceIntelDSS);
    fn solver_intel_dss_initialize(solver: *mut InterfaceIntelDSS, symmetric: i32, positive_definite: i32) -> i32;
    fn solver_intel_dss_factorize(
        solver: *mut InterfaceIntelDSS,
        ndim: i32,
        row_pointers: *const i32,
        col_indices: *const i32,
        values: *const f64,
    ) -> i32;
    fn solver_intel_dss_solve(solver: *mut InterfaceIntelDSS, x: *mut f64, rhs: *const f64) -> i32;
}

/// Wraps the IntelDSS solver for sparse linear systems
///
/// **Warning:** This solver may fail with large matrices (e.g., ATandT/pre2).
pub struct SolverIntelDSS {
    /// Holds a pointer to the C interface to IntelDSS
    solver: *mut InterfaceIntelDSS,

    /// Number of rows = number of columns
    ndim: i32,

    /// Symmetry option
    symmetry: Option<Symmetry>,

    /// Configuration parameters
    pub config: ConfigSolver,

    /// Indicates whether the C interface has been initialized or not
    initialized: bool,

    /// Indicates whether the sparse matrix has been factorized or not
    factorized: bool,
}

impl Drop for SolverIntelDSS {
    /// Tells the c-code to release memory
    fn drop(&mut self) {
        unsafe {
            solver_intel_dss_drop(self.solver);
        }
    }
}

impl SolverIntelDSS {
    /// Allocates a new instance
    ///
    /// # Examples
    ///
    /// See [SolverIntelDSS::solve]
    pub fn new() -> Result<Self, StrError> {
        if !cfg!(with_intel_dss) {
            return Err("This code has not been compiled with Intel DSS");
        }
        unsafe {
            let solver = solver_intel_dss_new();
            if solver.is_null() {
                return Err("c-code failed to allocate the IntelDSS solver");
            }
            Ok(SolverIntelDSS {
                solver,
                ndim: 0,
                symmetry: None,
                config: ConfigSolver::new(),
                initialized: false,
                factorized: false,
            })
        }
    }
}

impl SolverTrait for SolverIntelDSS {
    /// Initializes the C interface to IntelDSS
    ///
    /// # Input
    ///
    /// * `ndim` -- number of rows = number of columns of the coefficient matrix A
    /// * `_nnz` -- NOT used here
    /// * `_symmetry` -- NOT used here
    /// * `config` -- configuration parameters; None => use default
    ///
    /// # Examples
    ///
    /// See [SolverIntelDSS::solve]
    fn initialize(
        &mut self,
        ndim: usize,
        _nnz: usize,
        symmetry: Option<Symmetry>,
        config: Option<ConfigSolver>,
    ) -> Result<(), StrError> {
        if self.initialized {
            return Err("initialize can only be called once");
        }
        self.ndim = to_i32(ndim)?;
        if let Some(cfg) = config {
            self.config = cfg;
        }
        let (symmetric, positive_definite) = match symmetry {
            Some(sym) => match sym {
                Symmetry::General(storage) => {
                    if storage != Storage::Upper {
                        return Err("Intel DSS requires upper-triangular storage for symmetric matrices");
                    }
                    (1, 0)
                }
                Symmetry::PositiveDefinite(storage) => {
                    if storage != Storage::Upper {
                        return Err("Intel DSS requires upper-triangular storage for symmetric matrices");
                    }
                    (1, 1)
                }
            },
            None => (0, 0),
        };
        unsafe {
            let status = solver_intel_dss_initialize(self.solver, symmetric, positive_definite);
            if status != MKL_DSS_SUCCESS {
                return Err(handle_intel_dss_error_code(status));
            }
        }
        self.symmetry = symmetry;
        self.initialized = true;
        self.factorized = false;
        Ok(())
    }

    /// Performs the factorization (and analysis)
    ///
    /// **Note::** Initialize must be called first. Also, the dimension and symmetry/storage
    /// of the CooMatrix must be the same as the ones provided by `initialize`.
    ///
    /// # Input
    ///
    /// * `coo` -- The **same** matrix provided to `initialize`
    /// * `_verbose` -- shows messages (NOT USED)
    ///
    /// # Examples
    ///
    /// See [SolverIntelDSS::solve]
    fn factorize(&mut self, coo: &CooMatrix, _verbose: bool) -> Result<(), StrError> {
        self.factorized = false;
        if !self.initialized {
            return Err("the function initialize must be called before factorize");
        }
        if coo.nrow != self.ndim as usize || coo.ncol != self.ndim as usize {
            return Err("the dimension of the CooMatrix must be equal to ndim");
        }
        if coo.symmetry != self.symmetry {
            return Err("the CooMatrix symmetry option must be equal to the one provided to initialize");
        }
        let csr = CsrMatrix::from(coo)?;
        unsafe {
            let status = solver_intel_dss_factorize(
                self.solver,
                self.ndim,
                csr.row_pointers.as_ptr(),
                csr.col_indices.as_ptr(),
                csr.values.as_ptr(),
            );
            if status != MKL_DSS_SUCCESS {
                return Err(handle_intel_dss_error_code(status));
            }
        }
        self.factorized = true;
        Ok(())
    }

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
    /// * `_verbose` -- shows messages (NOT USED)
    ///
    /// # Examples
    ///
    /// TODO
    fn solve(&mut self, x: &mut Vector, rhs: &Vector, _verbose: bool) -> Result<(), StrError> {
        if !self.factorized {
            return Err("the function factorize must be called before solve");
        }
        if x.dim() != self.ndim as usize {
            return Err("the dimension of the vector of unknown values x is incorrect");
        }
        if rhs.dim() != self.ndim as usize {
            return Err("the dimension of the right-hand side vector is incorrect");
        }
        unsafe {
            let status = solver_intel_dss_solve(self.solver, x.as_mut_data().as_mut_ptr(), rhs.as_data().as_ptr());
            if status != MKL_DSS_SUCCESS {
                return Err(handle_intel_dss_error_code(status));
            }
        }
        Ok(())
    }

    /// Returns the determinant (NOT AVAILABLE)
    fn get_determinant(&self) -> (f64, f64, f64) {
        (0.0, 10.0, 0.0)
    }

    /// Returns the ordering effectively used by the solver (NOT AVAILABLE)
    fn get_effective_ordering(&self) -> String {
        "Unknown".to_string()
    }

    /// Returns the scaling effectively used by the solver (NOT AVAILABLE)
    fn get_effective_scaling(&self) -> String {
        "Unknown".to_string()
    }

    /// Returns the strategy (concerning symmetry) effectively used by the solver
    fn get_effective_strategy(&self) -> String {
        "Unknown".to_string()
    }

    /// Returns the name of this solver
    ///
    /// # Output
    ///
    /// * `IntelDSS` -- if the default system IntelDSS has been used
    /// * `IntelDSS-local` -- if the locally compiled IntelDSS has be used
    fn get_name(&self) -> String {
        if cfg!(with_intel_dss) {
            "IntelDSS".to_string()
        } else {
            "INTEL_DSS_IS_NOT_AVAILABLE".to_string()
        }
    }
}

/// Handles Intel DSS error code
pub(crate) fn handle_intel_dss_error_code(err: i32) -> StrError {
    match err {
        -1 => return "MKL_DSS_ZERO_PIVOT",
        -2 => return "MKL_DSS_OUT_OF_MEMORY",
        -3 => return "MKL_DSS_FAILURE",
        -4 => return "MKL_DSS_ROW_ERR",
        -5 => return "MKL_DSS_COL_ERR",
        -6 => return "MKL_DSS_TOO_FEW_VALUES",
        -7 => return "MKL_DSS_TOO_MANY_VALUES",
        -8 => return "MKL_DSS_NOT_SQUARE",
        -9 => return "MKL_DSS_STATE_ERR",
        -10 => return "MKL_DSS_INVALID_OPTION",
        -11 => return "MKL_DSS_OPTION_CONFLICT",
        -12 => return "MKL_DSS_MSG_LVL_ERR",
        -13 => return "MKL_DSS_TERM_LVL_ERR",
        -14 => return "MKL_DSS_STRUCTURE_ERR",
        -15 => return "MKL_DSS_REORDER_ERR",
        -16 => return "MKL_DSS_VALUES_ERR",
        17 => return "MKL_DSS_STATISTICS_INVALID_MATRIX",
        18 => return "MKL_DSS_STATISTICS_INVALID_STATE",
        19 => return "MKL_DSS_STATISTICS_INVALID_STRING",
        20 => return "MKL_DSS_REORDER1_ERR",
        21 => return "MKL_DSS_PREORDER_ERR",
        22 => return "MKL_DSS_DIAG_ERR",
        23 => return "MKL_DSS_I32BIT_ERR",
        24 => return "MKL_DSS_OOC_MEM_ERR",
        25 => return "MKL_DSS_OOC_OC_ERR",
        26 => return "MKL_DSS_OOC_RW_ERR",
        100000 => return "Error: c-code returned null pointer (IntelDSS)",
        200000 => return "Error: c-code failed to allocate memory (IntelDSS)",
        400000 => return "This code has not been compiled with Intel DSS",
        _ => return "Error: unknown error returned by c-code (IntelDSS)",
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[cfg(with_intel_dss)]
mod tests {
    use super::{handle_intel_dss_error_code, SolverIntelDSS};
    use crate::{ConfigSolver, CooMatrix, Ordering, Samples, Scaling, SolverTrait, Storage, Symmetry};
    use russell_chk::{approx_eq, vec_approx_eq};
    use russell_lab::Vector;

    #[test]
    fn new_and_drop_work() {
        // you may debug into the C-code to see that drop is working
        let solver = SolverIntelDSS::new().unwrap();
        assert!(!solver.initialized);
        assert!(!solver.factorized);
    }

    #[test]
    fn initialize_handles_errors_and_works() {
        // allocate a new solver
        let mut solver = SolverIntelDSS::new().unwrap();
        assert!(!solver.initialized);
        assert!(!solver.factorized);
    }
}
