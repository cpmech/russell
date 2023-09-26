use super::{to_i32, ConfigSolver, CooMatrix, CscMatrix, Ordering, Scaling, SolverTrait};
use crate::StrError;
use russell_lab::Vector;

/// Opaque struct holding a C-pointer to InterfaceUMFPACK
///
/// Reference: <https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs>
#[repr(C)]
struct InterfaceUMFPACK {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

extern "C" {
    fn solver_umfpack_new() -> *mut InterfaceUMFPACK;
    fn solver_umfpack_drop(solver: *mut InterfaceUMFPACK);
    fn solver_umfpack_factorize(
        solver: *mut InterfaceUMFPACK,
        // output
        effective_strategy: *mut i32,
        effective_ordering: *mut i32,
        effective_scaling: *mut i32,
        determinant_coefficient: *mut f64,
        determinant_exponent: *mut f64,
        // input
        ordering: i32,
        scaling: i32,
        // requests
        compute_determinant: i32,
        verbose: i32,
        // matrix config
        enforce_unsymmetric_strategy: i32,
        ndim: i32,
        // matrix
        col_pointers: *const i32,
        row_indices: *const i32,
        values: *const f64,
    ) -> i32;
    fn solver_umfpack_solve(
        solver: *mut InterfaceUMFPACK,
        x: *mut f64,
        rhs: *const f64,
        col_pointers: *const i32,
        row_indices: *const i32,
        values: *const f64,
        verbose: i32,
    ) -> i32;
}

/// Wraps the UMFPACK solver for sparse linear systems
///
/// **Warning:** This solver may "run out of memory"
pub struct SolverUMFPACK {
    /// Holds a pointer to the C interface to UMFPACK
    solver: *mut InterfaceUMFPACK,

    /// Indicates whether the sparse matrix has been factorized or not
    factorized: bool,

    /// Matrix dimension (to validate vectors in solve)
    ndim: usize,

    /// Holds the used strategy (after factorize)
    effective_strategy: i32,

    /// Holds the used ordering (after factorize)
    effective_ordering: i32,

    /// Holds the used scaling (after factorize)
    effective_scaling: i32,

    /// Holds the determinant coefficient: det = coefficient * pow(10, exponent)
    determinant_coefficient: f64,

    /// Holds the determinant exponent: det = coefficient * pow(10, exponent)
    determinant_exponent: f64,

    /// Compressed sparse column (CSC) matrix
    ///
    /// **Note:** This is created only if a COO matrix is given to factorize
    csc_matrix: Option<CscMatrix>,
}

impl Drop for SolverUMFPACK {
    /// Tells the c-code to release memory
    fn drop(&mut self) {
        unsafe {
            solver_umfpack_drop(self.solver);
        }
    }
}

impl SolverUMFPACK {
    /// Allocates a new instance
    pub fn new() -> Result<Self, StrError> {
        unsafe {
            let solver = solver_umfpack_new();
            if solver.is_null() {
                return Err("c-code failed to allocate the UMFPACK solver");
            }
            Ok(SolverUMFPACK {
                solver,
                factorized: false,
                ndim: 0,
                effective_strategy: -1,
                effective_ordering: -1,
                effective_scaling: -1,
                determinant_coefficient: 0.0,
                determinant_exponent: 0.0,
                csc_matrix: None,
            })
        }
    }
}

impl SolverTrait for SolverUMFPACK {
    /// Performs the factorization (and analysis) given a COO matrix
    ///
    /// # Input
    ///
    /// * `coo` -- The COO matrix
    /// * `params` -- configuration parameters; None => use default
    fn factorize_coo(&mut self, coo: &CooMatrix, params: Option<ConfigSolver>) -> Result<(), StrError> {
        // set flag
        self.factorized = false;

        // check the COO matrix
        if coo.one_based {
            return Err("the COO matrix must have zero-based indices as required by UMFPACK");
        }
        if coo.nrow != coo.ncol {
            return Err("the matrix must be square");
        }
        coo.check_dimensions_ready()?;

        // configuration parameters
        let cfg = if let Some(p) = params { p } else { ConfigSolver::new() };

        // input parameters
        let ordering = match cfg.ordering {
            Ordering::Amd => UMFPACK_ORDERING_AMD,
            Ordering::Amf => UMFPACK_DEFAULT_ORDERING,
            Ordering::Auto => UMFPACK_DEFAULT_ORDERING,
            Ordering::Best => UMFPACK_ORDERING_BEST,
            Ordering::Cholmod => UMFPACK_ORDERING_CHOLMOD,
            Ordering::Metis => UMFPACK_ORDERING_METIS,
            Ordering::No => UMFPACK_ORDERING_NONE,
            Ordering::Pord => UMFPACK_DEFAULT_ORDERING,
            Ordering::Qamd => UMFPACK_DEFAULT_ORDERING,
            Ordering::Scotch => UMFPACK_DEFAULT_ORDERING,
        };
        let scaling = match cfg.scaling {
            Scaling::Auto => UMFPACK_DEFAULT_SCALE,
            Scaling::Column => UMFPACK_DEFAULT_SCALE,
            Scaling::Diagonal => UMFPACK_DEFAULT_SCALE,
            Scaling::Max => UMFPACK_SCALE_MAX,
            Scaling::No => UMFPACK_SCALE_NONE,
            Scaling::RowCol => UMFPACK_DEFAULT_SCALE,
            Scaling::RowColIter => UMFPACK_DEFAULT_SCALE,
            Scaling::RowColRig => UMFPACK_DEFAULT_SCALE,
            Scaling::Sum => UMFPACK_SCALE_SUM,
        };

        // requests
        let determinant = if cfg.compute_determinant { 1 } else { 0 };
        let verbose_mode = if cfg.verbose { 1 } else { 0 };
        let enforce_unsymmetric = if cfg.umfpack_enforce_unsymmetric_strategy { 1 } else { 0 };

        // check the storage type
        if let Some(symmetry) = coo.symmetry {
            if symmetry.triangular() {
                return Err("the matrix must not be triangular for UMFPACK");
            }
        }

        // convert COO to CSC
        let csc = CscMatrix::from_coo(coo)?;
        csc.check_dimensions()?;

        // matrix config
        let ndim = to_i32(csc.nrow)?;

        // call UMFPACK factorize
        unsafe {
            let status = solver_umfpack_factorize(
                self.solver,
                // output
                &mut self.effective_strategy,
                &mut self.effective_ordering,
                &mut self.effective_scaling,
                &mut self.determinant_coefficient,
                &mut self.determinant_exponent,
                // input
                ordering,
                scaling,
                // requests
                determinant,
                verbose_mode,
                // matrix config
                enforce_unsymmetric,
                ndim,
                // matrix
                csc.col_pointers.as_ptr(),
                csc.row_indices.as_ptr(),
                csc.values.as_ptr(),
            );
            if status != UMFPACK_SUCCESS {
                return Err(handle_umfpack_error_code(status));
            }
        }
        self.ndim = csc.nrow;
        self.csc_matrix = Some(csc);
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
    /// * `verbose` -- shows messages
    fn solve(&mut self, x: &mut Vector, rhs: &Vector, verbose: bool) -> Result<(), StrError> {
        if !self.factorized {
            return Err("the function factorize must be called before solve");
        }
        if x.dim() != self.ndim as usize {
            return Err("the dimension of the vector of unknown values x is incorrect");
        }
        if rhs.dim() != self.ndim as usize {
            return Err("the dimension of the right-hand side vector is incorrect");
        }
        let csc = match &self.csc_matrix {
            Some(c) => c,
            None => {
                return Err("the CSC matrix was not factorized yet");
            }
        };
        let verb = if verbose { 1 } else { 0 };
        unsafe {
            let status = solver_umfpack_solve(
                self.solver,
                x.as_mut_data().as_mut_ptr(),
                rhs.as_data().as_ptr(),
                csc.col_pointers.as_ptr(),
                csc.row_indices.as_ptr(),
                csc.values.as_ptr(),
                verb,
            );
            if status != UMFPACK_SUCCESS {
                return Err(handle_umfpack_error_code(status));
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

    /// Returns the ordering effectively used by the solver
    fn get_effective_ordering(&self) -> String {
        match self.effective_ordering {
            UMFPACK_ORDERING_CHOLMOD => "Cholmod".to_string(),
            UMFPACK_ORDERING_AMD => "Amd".to_string(),
            UMFPACK_ORDERING_METIS => "Metis".to_string(),
            UMFPACK_ORDERING_BEST => "Best".to_string(),
            UMFPACK_ORDERING_NONE => "No".to_string(),
            _ => "Unknown".to_string(),
        }
    }

    /// Returns the scaling effectively used by the solver
    fn get_effective_scaling(&self) -> String {
        match self.effective_scaling {
            UMFPACK_SCALE_NONE => "No".to_string(),
            UMFPACK_SCALE_SUM => "Sum".to_string(),
            UMFPACK_SCALE_MAX => "Max".to_string(),
            _ => "Unknown".to_string(),
        }
    }

    /// Returns the strategy (concerning symmetry) effectively used by the solver
    fn get_effective_strategy(&self) -> String {
        match self.effective_strategy {
            UMFPACK_STRATEGY_AUTO => "Auto".to_string(),
            UMFPACK_STRATEGY_UNSYMMETRIC => "Unsymmetric".to_string(),
            UMFPACK_STRATEGY_SYMMETRIC => "Symmetric".to_string(),
            _ => "Unknown".to_string(),
        }
    }

    /// Returns the name of this solver
    ///
    /// # Output
    ///
    /// * `UMFPACK` -- if the default system UMFPACK has been used
    /// * `UMFPACK-local` -- if the locally compiled UMFPACK has be used
    fn get_name(&self) -> String {
        if cfg!(local_umfpack) {
            "UMFPACK-local".to_string()
        } else {
            "UMFPACK".to_string()
        }
    }
}

/// Handles UMFPACK error code
pub(crate) fn handle_umfpack_error_code(err: i32) -> StrError {
    match err {
        1 => return "Error(1): Matrix is singular",
        2 => return "Error(2): The determinant is nonzero, but smaller than allowed",
        3 => return "Error(3): The determinant is larger than allowed",
        -1 => return "Error(-1): Not enough memory",
        -3 => return "Error(-3): Invalid numeric object",
        -4 => return "Error(-4): Invalid symbolic object",
        -5 => return "Error(-5): Argument missing",
        -6 => return "Error(-6): Nrow or ncol must be greater than zero",
        -8 => return "Error(-8): Invalid matrix",
        -11 => return "Error(-11): Different pattern",
        -13 => return "Error(-13): Invalid system",
        -15 => return "Error(-15): Invalid permutation",
        -17 => return "Error(-17): Failed to save/load file",
        -18 => return "Error(-18): Ordering method failed",
        -911 => return "Error(-911): An internal error has occurred",
        100000 => return "Error: c-code returned null pointer (UMFPACK)",
        200000 => return "Error: c-code failed to allocate memory (UMFPACK)",
        _ => return "Error: unknown error returned by c-code (UMFPACK)",
    }
}

const UMFPACK_SUCCESS: i32 = 0;

const UMFPACK_STRATEGY_AUTO: i32 = 0; // use symmetric or unsymmetric strategy
const UMFPACK_STRATEGY_UNSYMMETRIC: i32 = 1; // COLAMD(A), col-tree post-order, do not prefer diag
const UMFPACK_STRATEGY_SYMMETRIC: i32 = 3; // AMD(A+A'), no col-tree post-order, prefer diagonal

const UMFPACK_ORDERING_CHOLMOD: i32 = 0; // use CHOLMOD (AMD/COLAMD then METIS)
const UMFPACK_ORDERING_AMD: i32 = 1; // use AMD/COLAMD
const UMFPACK_ORDERING_METIS: i32 = 3; // use METIS
const UMFPACK_ORDERING_BEST: i32 = 4; // try many orderings, pick best
const UMFPACK_ORDERING_NONE: i32 = 5; // natural ordering
const UMFPACK_DEFAULT_ORDERING: i32 = UMFPACK_ORDERING_AMD;

const UMFPACK_SCALE_NONE: i32 = 0; // no scaling
const UMFPACK_SCALE_SUM: i32 = 1; // default: divide each row by sum (abs (row))
const UMFPACK_SCALE_MAX: i32 = 2; // divide each row by max (abs (row))
const UMFPACK_DEFAULT_SCALE: i32 = UMFPACK_SCALE_SUM;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{handle_umfpack_error_code, SolverUMFPACK};
    use crate::{ConfigSolver, CooMatrix, Ordering, Samples, Scaling, SolverTrait};
    use russell_chk::{approx_eq, vec_approx_eq};
    use russell_lab::Vector;

    #[test]
    fn new_and_drop_work() {
        // you may debug into the C-code to see that drop is working
        let solver = SolverUMFPACK::new().unwrap();
        assert!(!solver.factorized);
    }

    #[test]
    fn factorize_handles_errors() {
        let mut solver = SolverUMFPACK::new().unwrap();
        assert!(!solver.factorized);
        let (coo, _, _, _) = Samples::umfpack_unsymmetric_5x5(true);
        assert_eq!(
            solver.factorize_coo(&coo, None).err(),
            Some("the COO matrix must have zero-based indices as required by UMFPACK")
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
            Some("the matrix must not be triangular for UMFPACK")
        );
    }

    #[test]
    fn factorize_works() {
        let mut solver = SolverUMFPACK::new().unwrap();
        assert!(!solver.factorized);
        let (coo, _, _, _) = Samples::umfpack_unsymmetric_5x5(false);
        let mut params = ConfigSolver::new();

        params.compute_determinant = true;
        params.ordering = Ordering::Amd;
        params.scaling = Scaling::Sum;

        solver.factorize_coo(&coo, Some(params)).unwrap();
        assert!(solver.factorized);

        assert_eq!(solver.get_effective_ordering(), "Amd");
        assert_eq!(solver.get_effective_scaling(), "Sum");

        let (a, b, c) = solver.get_determinant();
        let det = a * f64::powf(b, c);
        approx_eq(det, 114.0, 1e-13);
    }

    #[test]
    fn factorize_fails_on_singular_matrix() {
        let mut solver = SolverUMFPACK::new().unwrap();
        let mut coo = CooMatrix::new(2, 2, 2, None, false).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(1, 1, 0.0).unwrap();
        assert_eq!(solver.factorize_coo(&coo, None), Err("Error(1): Matrix is singular"));
    }

    #[test]
    fn solve_handles_errors() {
        let mut solver = SolverUMFPACK::new().unwrap();
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
        let mut solver = SolverUMFPACK::new().unwrap();
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
    fn handle_umfpack_error_code_works() {
        let default = "Error: unknown error returned by c-code (UMFPACK)";
        for c in &[1, 2, 3, -1, -3, -4, -5, -6, -8, -11, -13, -15, -17, -18, -911] {
            let res = handle_umfpack_error_code(*c);
            assert!(res.len() > 0);
            assert_ne!(res, default);
        }
        assert_eq!(
            handle_umfpack_error_code(100000),
            "Error: c-code returned null pointer (UMFPACK)"
        );
        assert_eq!(
            handle_umfpack_error_code(200000),
            "Error: c-code failed to allocate memory (UMFPACK)"
        );
        assert_eq!(handle_umfpack_error_code(123), default);
    }
}
