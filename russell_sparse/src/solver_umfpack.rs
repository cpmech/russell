use super::{to_i32, ConfigSolver, CooMatrix, CscMatrix, Ordering, Scaling, SolverTrait, Symmetry};
use crate::StrError;
use russell_lab::Vector;

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
        ndim: i32,
        symmetry: i32,
        ordering: i32,
        scaling: i32,
        col_pointers: *const i32,
        row_indices: *const i32,
        values: *const f64,
        compute_determinant: i32,
        verbose: i32,
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
    fn solver_umfpack_get_strategy(solver: *const InterfaceUMFPACK) -> i32;
    fn solver_umfpack_get_ordering(solver: *const InterfaceUMFPACK) -> i32;
    fn solver_umfpack_get_scaling(solver: *const InterfaceUMFPACK) -> i32;
    fn solver_umfpack_get_det_mx(solver: *const InterfaceUMFPACK) -> f64;
    fn solver_umfpack_get_det_ex(solver: *const InterfaceUMFPACK) -> f64;
}

/// Wraps the UMFPACK solver for sparse linear systems
///
/// **Warning:** This solver may "run out of memory"
pub struct SolverUMFPACK {
    /// Holds a pointer to the C interface to UMFPACK
    solver: *mut InterfaceUMFPACK,

    /// Number of rows = number of columns
    ndim: i32,

    /// Configuration parameters
    pub config: ConfigSolver,

    /// Indicates whether the C interface has been initialized or not
    initialized: bool,

    /// Indicates whether the sparse matrix has been factorized or not
    factorized: bool,

    /// Compressed sparse column (CSC) matrix
    ///
    /// **Note:** This is created only if a COO matrix is given to factorize
    pub csc_matrix: Option<CscMatrix>,
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
    ///
    /// # Examples
    ///
    /// See [SolverUMFPACK::solve]
    pub fn new() -> Result<Self, StrError> {
        unsafe {
            let solver = solver_umfpack_new();
            if solver.is_null() {
                return Err("c-code failed to allocate the UMFPACK solver");
            }
            Ok(SolverUMFPACK {
                solver,
                ndim: 0,
                config: ConfigSolver::new(),
                initialized: false,
                factorized: false,
                csc_matrix: None,
            })
        }
    }
}

impl SolverTrait for SolverUMFPACK {
    /// Initializes the C interface to UMFPACK
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
    /// See [SolverUMFPACK::solve]
    fn initialize(
        &mut self,
        ndim: usize,
        _nnz: usize,
        _symmetry: Option<Symmetry>,
        config: Option<ConfigSolver>,
    ) -> Result<(), StrError> {
        if self.initialized {
            return Err("initialize can only be called once");
        }
        self.ndim = to_i32(ndim)?;
        if let Some(cfg) = config {
            self.config = cfg;
        }
        self.initialized = true;
        self.factorized = false;
        Ok(())
    }

    /// Performs the factorization (and analysis) given COO matrix
    ///
    /// **Note::** Initialize must be called first. Also, the dimension and symmetry/storage
    /// of the CooMatrix must be the same as the ones provided by `initialize`.
    ///
    /// # Input
    ///
    /// * `coo` -- The **same** matrix provided to `initialize`
    /// * `verbose` -- shows messages
    ///
    /// # Examples
    ///
    /// See [SolverUMFPACK::solve]
    fn factorize_coo(&mut self, coo: &CooMatrix, verbose: bool) -> Result<(), StrError> {
        self.factorized = false;
        if !self.initialized {
            return Err("the function initialize must be called before factorize");
        }
        if coo.nrow != self.ndim as usize || coo.ncol != self.ndim as usize {
            return Err("the dimension of the CooMatrix must be equal to ndim");
        }
        if let Some(sym) = coo.symmetry {
            if sym.triangular() {
                return Err("the CooMatrix cannot be triangular for UMFPACK");
            }
        }
        let csc = CscMatrix::from_coo(coo)?;
        csc.check_dimensions()?;
        let symmetry = if self.config.umfpack_enforce_unsymmetric_strategy {
            UMFPACK_STRATEGY_UNSYMMETRIC
        } else {
            UMFPACK_STRATEGY_AUTO
        };
        let ordering = match self.config.ordering {
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
        let scaling = match self.config.scaling {
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
        let determinant = if self.config.compute_determinant { 1 } else { 0 };
        let verb = if verbose { 1 } else { 0 };
        unsafe {
            let status = solver_umfpack_factorize(
                self.solver,
                self.ndim,
                symmetry,
                ordering,
                scaling,
                csc.col_pointers.as_ptr(),
                csc.row_indices.as_ptr(),
                csc.values.as_ptr(),
                determinant,
                verb,
            );
            if status != 0 {
                return Err(handle_umfpack_error_code(status));
            }
        }
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
    ///
    /// # Examples
    ///
    /// TODO
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
            if status != 0 {
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
        unsafe {
            let mx = solver_umfpack_get_det_mx(self.solver);
            let ex = solver_umfpack_get_det_ex(self.solver);
            (mx, 10.0, ex)
        }
    }

    /// Returns the ordering effectively used by the solver
    fn get_effective_ordering(&self) -> String {
        unsafe {
            let ordering = solver_umfpack_get_ordering(self.solver);
            match ordering {
                UMFPACK_ORDERING_CHOLMOD => "Cholmod".to_string(),
                UMFPACK_ORDERING_AMD => "Amd".to_string(),
                UMFPACK_ORDERING_METIS => "Metis".to_string(),
                UMFPACK_ORDERING_BEST => "Best".to_string(),
                UMFPACK_ORDERING_NONE => "No".to_string(),
                _ => "Unknown".to_string(),
            }
        }
    }

    /// Returns the scaling effectively used by the solver
    fn get_effective_scaling(&self) -> String {
        unsafe {
            let scaling = solver_umfpack_get_scaling(self.solver);
            match scaling {
                UMFPACK_SCALE_NONE => "No".to_string(),
                UMFPACK_SCALE_SUM => "Sum".to_string(),
                UMFPACK_SCALE_MAX => "Max".to_string(),
                _ => "Unknown".to_string(),
            }
        }
    }

    /// Returns the strategy (concerning symmetry) effectively used by the solver
    fn get_effective_strategy(&self) -> String {
        unsafe {
            let strategy = solver_umfpack_get_strategy(self.solver);
            match strategy {
                UMFPACK_STRATEGY_AUTO => "Auto".to_string(),
                UMFPACK_STRATEGY_UNSYMMETRIC => "Unsymmetric".to_string(),
                UMFPACK_STRATEGY_SYMMETRIC => "Symmetric".to_string(),
                _ => "Unknown".to_string(),
            }
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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{handle_umfpack_error_code, SolverUMFPACK};
    use crate::{ConfigSolver, CooMatrix, Ordering, Samples, Scaling, SolverTrait, Storage, Symmetry};
    use russell_chk::{approx_eq, vec_approx_eq};
    use russell_lab::Vector;

    #[test]
    fn new_and_drop_work() {
        // you may debug into the C-code to see that drop is working
        let solver = SolverUMFPACK::new().unwrap();
        assert!(!solver.initialized);
        assert!(!solver.factorized);
    }

    #[test]
    fn initialize_handles_errors_and_works() {
        // allocate a new solver
        let mut solver = SolverUMFPACK::new().unwrap();
        assert!(!solver.initialized);
        assert!(!solver.factorized);

        // TODO
        // initialize fails on incorrect storage
        // assert_eq!(
        //     solver
        //         .initialize(2, 2, Some(Symmetry::General(Storage::Lower)), None)
        //         .err(),
        //     Some("for UMFPACK, if the matrix is symmetric, the storage still must be full")
        // );
        // assert_eq!(
        //     solver
        //         .initialize(2, 2, Some(Symmetry::General(Storage::Upper)), None)
        //         .err(),
        //     Some("for UMFPACK, if the matrix is symmetric, the storage still must be full")
        // );

        // initialize works
        solver.initialize(2, 2, None, None).unwrap();
        assert!(solver.initialized);
    }

    #[test]
    fn factorize_handles_errors_and_works() {
        // allocate a new solver
        let mut solver = SolverUMFPACK::new().unwrap();
        assert!(!solver.initialized);
        assert!(!solver.factorized);

        // sample matrix
        let (coo, _, _, _) = Samples::umfpack_unsymmetric_5x5(false);

        // factorize requests initialize
        assert_eq!(
            solver.factorize_coo(&coo, false).err(),
            Some("the function initialize must be called before factorize")
        );

        // call initialize
        solver.initialize(coo.nrow, coo.pos, coo.symmetry, None).unwrap();

        // factorize fails on incompatible coo matrix
        let sym = Some(Symmetry::General(Storage::Lower));
        let mut coo_wrong_1 = CooMatrix::new(1, 5, 13, None, false).unwrap();
        let coo_wrong_2 = CooMatrix::new(5, 1, 13, None, false).unwrap();
        let coo_wrong_3 = CooMatrix::new(5, 5, 12, None, false).unwrap();
        let mut coo_wrong_4 = CooMatrix::new(5, 5, 13, sym, false).unwrap();
        for _ in 0..13 {
            coo_wrong_1.put(0, 0, 1.0).unwrap();
            coo_wrong_4.put(0, 0, 1.0).unwrap();
        }
        assert_eq!(
            solver.factorize_coo(&coo_wrong_1, false).err(),
            Some("the dimension of the CooMatrix must be equal to ndim")
        );
        assert_eq!(
            solver.factorize_coo(&coo_wrong_2, false).err(),
            Some("the dimension of the CooMatrix must be equal to ndim")
        );
        assert_eq!(
            solver.factorize_coo(&coo_wrong_3, false).err(),
            Some("COO matrix: pos = nnz must be ≥ 1")
        );
        assert_eq!(
            solver.factorize_coo(&coo_wrong_4, false).err(),
            Some("the CooMatrix cannot be triangular for UMFPACK")
        );

        // factorize works
        solver.factorize_coo(&coo, false).unwrap();
        assert!(solver.factorized);
    }

    #[test]
    fn factorize_fails_on_singular_matrix() {
        let mut solver = SolverUMFPACK::new().unwrap();
        let mut coo = CooMatrix::new(2, 2, 2, None, false).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(1, 1, 0.0).unwrap();
        solver.initialize(coo.nrow, coo.max, coo.symmetry, None).unwrap();
        assert_eq!(solver.factorize_coo(&coo, false), Err("Error(1): Matrix is singular"));
    }

    #[test]
    fn solve_handles_errors_and_works() {
        // allocate a new solver
        let mut solver = SolverUMFPACK::new().unwrap();
        assert!(!solver.initialized);
        assert!(!solver.factorized);

        // sample matrix
        let (coo, _, _, _) = Samples::umfpack_unsymmetric_5x5(false);

        // allocate x and rhs
        let mut x = Vector::new(5);
        let rhs = Vector::from(&[8.0, 45.0, -3.0, 3.0, 19.0]);
        let x_correct = &[1.0, 2.0, 3.0, 4.0, 5.0];

        // solve fails on non-factorized system
        assert_eq!(
            solver.solve(&mut x, &rhs, false),
            Err("the function factorize must be called before solve")
        );

        // call initialize and factorize
        solver.initialize(coo.nrow, coo.pos, coo.symmetry, None).unwrap();
        assert!(solver.initialized);
        solver.factorize_coo(&coo, false).unwrap();
        assert!(solver.factorized);

        // solve fails on wrong x and rhs vectors
        let mut x_wrong = Vector::new(3);
        let rhs_wrong = Vector::new(2);
        assert_eq!(
            solver.solve(&mut x_wrong, &rhs, false),
            Err("the dimension of the vector of unknown values x is incorrect")
        );
        assert_eq!(
            solver.solve(&mut x, &rhs_wrong, false),
            Err("the dimension of the right-hand side vector is incorrect")
        );

        // solve works
        solver.solve(&mut x, &rhs, false).unwrap();
        vec_approx_eq(x.as_data(), x_correct, 1e-14);

        // calling solve again works
        let mut x_again = Vector::new(5);
        solver.solve(&mut x_again, &rhs, false).unwrap();
        vec_approx_eq(x_again.as_data(), x_correct, 1e-14);
    }

    #[test]
    fn get_ordering_and_scaling_work() {
        let mut config = ConfigSolver::new();
        config.ordering = Ordering::Amd;
        config.scaling = Scaling::Sum;
        let mut solver = SolverUMFPACK::new().unwrap();
        let (coo, _, _, _) = Samples::umfpack_unsymmetric_5x5(false);
        solver
            .initialize(coo.nrow, coo.max, coo.symmetry, Some(config))
            .unwrap();
        // TODO assert_eq!(solver.get_effective_ordering(), "Amd");
        // TODO assert_eq!(solver.get_effective_scaling(), "Sum");
    }

    #[test]
    fn get_determinant_works() {
        let mut config = ConfigSolver::new();
        config.compute_determinant = true;
        let mut solver = SolverUMFPACK::new().unwrap();
        let (coo, _, _, _) = Samples::umfpack_unsymmetric_5x5(false);
        solver
            .initialize(coo.nrow, coo.max, coo.symmetry, Some(config))
            .unwrap();
        solver.factorize_coo(&coo, false).unwrap();

        let (a, b, c) = solver.get_determinant();
        let det = a * f64::powf(b, c);
        approx_eq(a, 1.14, 1e-15);
        approx_eq(c, 2.0, 1e-15);
        approx_eq(det, 114.0, 1e-13);
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
