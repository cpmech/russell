use super::{to_i32, ConfigSolver, CooMatrix, CscMatrix, CsrMatrix, SolverTrait};
use crate::StrError;
use russell_lab::Vector;

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
    fn solver_intel_dss_factorize(
        solver: *mut InterfaceIntelDSS,
        // output
        determinant_coefficient: *mut f64,
        determinant_exponent: *mut f64,
        // requests
        compute_determinant: i32,
        // matrix config
        general_symmetric: i32,
        positive_definite: i32,
        ndim: i32,
        // matrix
        row_pointers: *const i32,
        col_indices: *const i32,
        values: *const f64,
    ) -> i32;
    fn solver_intel_dss_solve(solver: *mut InterfaceIntelDSS, x: *mut f64, rhs: *const f64) -> i32;
}

/// Wraps the IntelDSS solver for sparse linear systems
///
/// **Warning:** This solver does not check whether the matrix is singular or not;
/// thus it may return **incorrect results** if a singular matrix is given to factorize.
///
/// **Warning:** This solver may fail with large matrices (e.g., ATandT/pre2) and
/// may return **incorrect results**.
pub struct SolverIntelDSS {
    /// Holds a pointer to the C interface to IntelDSS
    solver: *mut InterfaceIntelDSS,

    /// Indicates whether the sparse matrix has been factorized or not
    factorized: bool,

    /// Matrix dimension (to validate vectors in solve)
    ndim: usize,

    /// Holds the determinant coefficient: det = coefficient * pow(10, exponent)
    determinant_coefficient: f64,

    /// Holds the determinant exponent: det = coefficient * pow(10, exponent)
    determinant_exponent: f64,
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
    /// **Warning:** This solver does not check whether the matrix is singular or not;
    /// thus it may return **incorrect results** if a singular matrix is given to factorize.
    ///
    /// **Warning:** This solver may fail with large matrices (e.g., ATandT/pre2) and
    /// may return **incorrect results**.
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
                factorized: false,
                ndim: 0,
                determinant_coefficient: 0.0,
                determinant_exponent: 0.0,
            })
        }
    }
}

impl SolverTrait for SolverIntelDSS {
    /// Performs the factorization (and analysis) given a COO matrix
    ///
    /// # Input
    ///
    /// * `coo` -- The COO matrix
    /// * `params` -- configuration parameters; None => use default
    ///
    /// **Warning:** This solver does not check whether the matrix is singular or not;
    /// thus it may return **incorrect results** if a singular matrix is given to factorize.
    ///
    /// **Warning:** This solver may fail with large matrices (e.g., ATandT/pre2) and
    /// may return **incorrect results**.
    fn factorize_coo(&mut self, coo: &CooMatrix, params: Option<ConfigSolver>) -> Result<(), StrError> {
        // set flag
        self.factorized = false;

        // check the COO matrix
        if coo.one_based {
            return Err("the COO matrix must have zero-based indices as required by Intel DSS");
        }
        if coo.nrow != coo.ncol {
            return Err("the matrix must be square");
        }
        coo.check_dimensions_ready()?;

        // configuration parameters
        let cfg = if let Some(p) = params { p } else { ConfigSolver::new() };

        // requests
        let determinant = if cfg.compute_determinant { 1 } else { 0 };

        // extract the symmetry flags and check the storage type
        let (general_symmetric, positive_definite) = match coo.symmetry {
            Some(symmetry) => symmetry.status(false, true)?,
            None => (0, 0),
        };

        // convert COO to CSR
        let csr = CsrMatrix::from_coo(coo)?;
        csr.check_dimensions()?;

        // check the number of non-zero values
        let nnz = csr.row_pointers[csr.nrow];
        if (nnz as usize) < csr.nrow {
            return Err("for Intel DSS, nnz = row_pointers[nrow] must be ≥ nrow");
        }

        // matrix config
        let ndim = to_i32(csr.nrow)?;

        // call Intel DSS factorize
        unsafe {
            let status = solver_intel_dss_factorize(
                self.solver,
                // output
                &mut self.determinant_coefficient,
                &mut self.determinant_exponent,
                // requests
                determinant,
                // matrix config
                general_symmetric,
                positive_definite,
                ndim,
                // matrix
                csr.row_pointers.as_ptr(),
                csr.col_indices.as_ptr(),
                csr.values.as_ptr(),
            );
            if status != MKL_DSS_SUCCESS {
                return Err(handle_intel_dss_error_code(status));
            }
        }
        self.ndim = csr.nrow;
        self.factorized = true;
        Ok(())
    }

    /// Performs the factorization (and analysis) given a CSC matrix
    ///
    /// # Input
    ///
    /// * `csc` -- The CSC matrix
    /// * `params` -- configuration parameters; None => use default
    fn factorize_csc(&mut self, _csc: &CscMatrix, _params: Option<ConfigSolver>) -> Result<(), StrError> {
        return Err("TODO");
    }

    /// Performs the factorization (and analysis) given a CSR matrix
    ///
    /// # Input
    ///
    /// * `csr` -- The CSR matrix
    /// * `params` -- configuration parameters; None => use default
    fn factorize_csr(&mut self, _csr: &CsrMatrix, _params: Option<ConfigSolver>) -> Result<(), StrError> {
        return Err("TODO");
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

    /// Returns the determinant
    ///
    /// Returns the three values `(mantissa, 10.0, exponent)`, such that the determinant is calculated by:
    ///
    /// ```text
    /// determinant = mantissa · pow(10.0, exponent)
    /// ```
    ///
    /// **Note:** This is only available if compute_determinant was requested.
    fn get_determinant(&self) -> (f64, f64, f64) {
        (self.determinant_coefficient, 10.0, self.determinant_exponent)
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

const MKL_DSS_SUCCESS: i32 = 0;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[cfg(with_intel_dss)]
mod tests {
    use super::{handle_intel_dss_error_code, SolverIntelDSS};
    use crate::{ConfigSolver, CooMatrix, Samples, SolverTrait};
    use russell_chk::{approx_eq, vec_approx_eq};
    use russell_lab::Vector;

    #[test]
    fn new_and_drop_work() {
        // you may debug into the C-code to see that drop is working
        let solver = SolverIntelDSS::new().unwrap();
        assert!(!solver.factorized);
    }

    #[test]
    fn factorize_handles_errors() {
        let mut solver = SolverIntelDSS::new().unwrap();
        assert!(!solver.factorized);
        let (coo, _, _, _) = Samples::umfpack_unsymmetric_5x5(true);
        assert_eq!(
            solver.factorize_coo(&coo, None).err(),
            Some("the COO matrix must have zero-based indices as required by Intel DSS")
        );
        let (coo, _, _, _) = Samples::rectangular_1x7();
        assert_eq!(
            solver.factorize_coo(&coo, None).err(),
            Some("the matrix must be square")
        );
        let coo = CooMatrix::new(1, 1, 1, None, false).unwrap();
        assert_eq!(
            solver.factorize_coo(&coo, None).err(),
            Some("COO matrix: pos = nnz must be ≥ 1")
        );
        let (coo, _, _, _) = Samples::mkl_symmetric_5x5_lower(false, false, false);
        assert_eq!(
            solver.factorize_coo(&coo, None).err(),
            Some("if the matrix is general symmetric, the required storage is upper triangular")
        );
    }

    #[test]
    fn factorize_works() {
        let mut solver = SolverIntelDSS::new().unwrap();
        assert!(!solver.factorized);
        let (coo, _, _, _) = Samples::umfpack_unsymmetric_5x5(false);
        let mut params = ConfigSolver::new();

        params.compute_determinant = true;

        solver.factorize_coo(&coo, Some(params)).unwrap();
        assert!(solver.factorized);

        assert_eq!(solver.get_effective_ordering(), "Unknown");
        assert_eq!(solver.get_effective_scaling(), "Unknown");

        let (a, b, c) = solver.get_determinant();
        let det = a * f64::powf(b, c);
        approx_eq(det, 114.0, 1e-13);
    }

    #[test]
    fn solve_handles_errors() {
        let mut solver = SolverIntelDSS::new().unwrap();
        assert!(!solver.factorized);
        let mut x = Vector::new(1);
        let rhs = Vector::new(1);
        assert_eq!(
            solver.solve(&mut x, &rhs, false),
            Err("the function factorize must be called before solve")
        );
        let (coo, _, _, _) = Samples::umfpack_unsymmetric_5x5(false);
        solver.factorize_coo(&coo, None).unwrap();
        assert_eq!(
            solver.solve(&mut x, &rhs, false),
            Err("the dimension of the vector of unknown values x is incorrect")
        );
        let mut x = Vector::new(5);
        assert_eq!(
            solver.solve(&mut x, &rhs, false),
            Err("the dimension of the right-hand side vector is incorrect")
        );
    }

    #[test]
    fn solve_works() {
        let mut solver = SolverIntelDSS::new().unwrap();
        let (coo, _, _, _) = Samples::umfpack_unsymmetric_5x5(false);
        let mut x = Vector::new(5);
        let rhs = Vector::from(&[8.0, 45.0, -3.0, 3.0, 19.0]);
        let x_correct = &[1.0, 2.0, 3.0, 4.0, 5.0];
        solver.factorize_coo(&coo, None).unwrap();
        solver.solve(&mut x, &rhs, false).unwrap();
        vec_approx_eq(x.as_data(), x_correct, 1e-14);

        // calling solve again works
        let mut x_again = Vector::new(5);
        solver.solve(&mut x_again, &rhs, false).unwrap();
        vec_approx_eq(x_again.as_data(), x_correct, 1e-14);
    }

    #[test]
    fn handle_intel_dss_error_code_works() {
        let default = "Error: unknown error returned by c-code (IntelDSS)";
        for i in 1..17 {
            let res = handle_intel_dss_error_code(-i);
            assert!(res.len() > 0);
            assert_ne!(res, default);
        }
        for i in 17..27 {
            let res = handle_intel_dss_error_code(i);
            assert!(res.len() > 0);
            assert_ne!(res, default);
        }
        assert_eq!(
            handle_intel_dss_error_code(100000),
            "Error: c-code returned null pointer (IntelDSS)"
        );
        assert_eq!(
            handle_intel_dss_error_code(200000),
            "Error: c-code failed to allocate memory (IntelDSS)"
        );
        assert_eq!(
            handle_intel_dss_error_code(400000),
            "This code has not been compiled with Intel DSS"
        );
        assert_eq!(handle_intel_dss_error_code(123), default);
    }
}
