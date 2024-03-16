use super::{LinSolParams, LinSolTrait, Ordering, Scaling, SparseMatrix, StatsLinSol, Sym};
use crate::auxiliary_and_constants::*;
use crate::StrError;
use russell_lab::{Stopwatch, Vector};

/// Opaque struct holding a C-pointer to InterfaceUMFPACK
///
/// Reference: <https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs>
#[repr(C)]
struct InterfaceUMFPACK {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

/// Enforce Send on the C structure
///
/// <https://stackoverflow.com/questions/50258359/can-a-struct-containing-a-raw-pointer-implement-send-and-be-ffi-safe>
unsafe impl Send for InterfaceUMFPACK {}

/// Enforce Send on the Rust structure
///
/// <https://stackoverflow.com/questions/50258359/can-a-struct-containing-a-raw-pointer-implement-send-and-be-ffi-safe>
unsafe impl Send for SolverUMFPACK {}

extern "C" {
    fn solver_umfpack_new() -> *mut InterfaceUMFPACK;
    fn solver_umfpack_drop(solver: *mut InterfaceUMFPACK);
    fn solver_umfpack_initialize(
        solver: *mut InterfaceUMFPACK,
        ordering: i32,
        scaling: i32,
        verbose: CcBool,
        enforce_unsymmetric_strategy: CcBool,
        ndim: i32,
        col_pointers: *const i32,
        row_indices: *const i32,
        values: *const f64,
    ) -> i32;
    fn solver_umfpack_factorize(
        solver: *mut InterfaceUMFPACK,
        effective_strategy: *mut i32,
        effective_ordering: *mut i32,
        effective_scaling: *mut i32,
        rcond_estimate: *mut f64,
        determinant_coefficient: *mut f64,
        determinant_exponent: *mut f64,
        compute_determinant: CcBool,
        verbose: CcBool,
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
        verbose: CcBool,
    ) -> i32;
}

/// Wraps the UMFPACK solver for sparse linear systems
///
/// **Warning:** This solver may "run out of memory" for very large matrices.
pub struct SolverUMFPACK {
    /// Holds a pointer to the C interface to UMFPACK
    solver: *mut InterfaceUMFPACK,

    /// Indicates whether the solver has been initialized or not (just once)
    initialized: bool,

    /// Indicates whether the sparse matrix has been factorized or not
    factorized: bool,

    /// Holds the symmetric flag saved in initialize
    initialized_sym: Sym,

    /// Holds the matrix dimension saved in initialize
    initialized_ndim: usize,

    /// Holds the number of non-zeros saved in initialize
    initialized_nnz: usize,

    /// Holds the used strategy (after factorize)
    effective_strategy: i32,

    /// Holds the used ordering (after factorize)
    effective_ordering: i32,

    /// Holds the used scaling (after factorize)
    effective_scaling: i32,

    /// Reciprocal condition number estimate (after factorize)
    rcond_estimate: f64,

    /// Holds the determinant coefficient (if requested)
    ///
    /// det = coefficient * pow(10, exponent)
    determinant_coefficient: f64,

    /// Holds the determinant exponent (if requested)
    ///
    /// det = coefficient * pow(10, exponent)
    determinant_exponent: f64,

    /// Stopwatch to measure computation times
    stopwatch: Stopwatch,

    /// Time spent on initialize in nanoseconds
    time_initialize_ns: u128,

    /// Time spent on factorize in nanoseconds
    time_factorize_ns: u128,

    /// Time spent on solve in nanoseconds
    time_solve_ns: u128,
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
                initialized: false,
                factorized: false,
                initialized_sym: Sym::No,
                initialized_ndim: 0,
                initialized_nnz: 0,
                effective_strategy: -1,
                effective_ordering: -1,
                effective_scaling: -1,
                rcond_estimate: 0.0,
                determinant_coefficient: 0.0,
                determinant_exponent: 0.0,
                stopwatch: Stopwatch::new(),
                time_initialize_ns: 0,
                time_factorize_ns: 0,
                time_solve_ns: 0,
            })
        }
    }
}

impl LinSolTrait for SolverUMFPACK {
    /// Performs the factorization (and analysis/initialization if needed)
    ///
    /// # Input
    ///
    /// * `mat` -- the coefficient matrix A (**COO** or **CSC**, but not CSR).
    ///   Also, the matrix must be square (`nrow = ncol`) and, if symmetric,
    ///   the symmetric flag must be [Sym::YesFull]
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
    /// 4. For symmetric matrices, `UMFPACK` requires [Sym::YesFull]
    fn factorize(&mut self, mat: &mut SparseMatrix, params: Option<LinSolParams>) -> Result<(), StrError> {
        // get CSC matrix
        // (or convert from COO if CSC is not available and COO is available)
        let csc = mat.get_csc_or_from_coo()?;

        // check
        if self.initialized {
            if csc.symmetric != self.initialized_sym {
                return Err("subsequent factorizations must use the same matrix (symmetric differs)");
            }
            if csc.nrow != self.initialized_ndim {
                return Err("subsequent factorizations must use the same matrix (ndim differs)");
            }
            if (csc.col_pointers[csc.ncol] as usize) != self.initialized_nnz {
                return Err("subsequent factorizations must use the same matrix (nnz differs)");
            }
        } else {
            if csc.nrow != csc.ncol {
                return Err("the matrix must be square");
            }
            if csc.symmetric == Sym::YesLower || csc.symmetric == Sym::YesUpper {
                return Err("UMFPACK requires Sym::YesFull for symmetric matrices");
            }
            self.initialized_sym = csc.symmetric;
            self.initialized_ndim = csc.nrow;
            self.initialized_nnz = csc.col_pointers[csc.ncol] as usize;
        }

        // parameters
        let par = if let Some(p) = params { p } else { LinSolParams::new() };

        // input parameters
        let ordering = umfpack_ordering(par.ordering);
        let scaling = umfpack_scaling(par.scaling);

        // requests
        let compute_determinant = if par.compute_determinant { 1 } else { 0 };
        let verbose = if par.verbose { 1 } else { 0 };

        // matrix config
        let enforce_unsym = if par.umfpack_enforce_unsymmetric_strategy { 1 } else { 0 };
        let ndim = to_i32(csc.nrow);

        // call initialize just once
        if !self.initialized {
            self.stopwatch.reset();
            unsafe {
                let status = solver_umfpack_initialize(
                    self.solver,
                    ordering,
                    scaling,
                    verbose,
                    enforce_unsym,
                    ndim,
                    csc.col_pointers.as_ptr(),
                    csc.row_indices.as_ptr(),
                    csc.values.as_ptr(),
                );
                if status != SUCCESSFUL_EXIT {
                    return Err(handle_umfpack_error_code(status));
                }
            }
            self.time_initialize_ns = self.stopwatch.stop();
            self.initialized = true;
        }

        // call factorize
        self.stopwatch.reset();
        unsafe {
            let status = solver_umfpack_factorize(
                self.solver,
                &mut self.effective_strategy,
                &mut self.effective_ordering,
                &mut self.effective_scaling,
                &mut self.rcond_estimate,
                &mut self.determinant_coefficient,
                &mut self.determinant_exponent,
                compute_determinant,
                verbose,
                csc.col_pointers.as_ptr(),
                csc.row_indices.as_ptr(),
                csc.values.as_ptr(),
            );
            if status != SUCCESSFUL_EXIT {
                return Err(handle_umfpack_error_code(status));
            }
        }
        self.time_factorize_ns = self.stopwatch.stop();

        // done
        self.factorized = true;
        Ok(())
    }

    /// Computes the solution of the linear system
    ///
    /// Solves the linear system:
    ///
    /// ```text
    ///   A   · x = rhs
    /// (m,m)  (m)  (m)
    /// ```
    ///
    /// # Output
    ///
    /// * `x` -- the vector of unknown values with dimension equal to mat.nrow
    ///
    /// # Input
    ///
    /// * `mat` -- the coefficient matrix A; must be square and, if symmetric, [Sym::YesFull].
    /// * `rhs` -- the right-hand side vector with know values an dimension equal to mat.nrow
    /// * `verbose` -- shows messages
    ///
    /// **Warning:** the matrix must be same one used in `factorize`.
    fn solve(&mut self, x: &mut Vector, mat: &SparseMatrix, rhs: &Vector, verbose: bool) -> Result<(), StrError> {
        // check
        if !self.factorized {
            return Err("the function factorize must be called before solve");
        }

        // access CSC matrix
        // (possibly already converted from COO, because factorize was (should have been) called)
        let csc = mat.get_csc()?;

        // check already factorized data
        let (nrow, ncol, nnz, sym) = csc.get_info();
        if sym != self.initialized_sym {
            return Err("solve must use the same matrix (symmetric differs)");
        }
        if nrow != self.initialized_ndim || ncol != self.initialized_ndim {
            return Err("solve must use the same matrix (ndim differs)");
        }
        if nnz != self.initialized_nnz {
            return Err("solve must use the same matrix (nnz differs)");
        }

        // check vectors
        if x.dim() != self.initialized_ndim {
            return Err("the dimension of the vector of unknown values x is incorrect");
        }
        if rhs.dim() != self.initialized_ndim {
            return Err("the dimension of the right-hand side vector is incorrect");
        }

        // call UMFPACK solve
        let verb = if verbose { 1 } else { 0 };
        self.stopwatch.reset();
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
            if status != SUCCESSFUL_EXIT {
                return Err(handle_umfpack_error_code(status));
            }
        }
        self.time_solve_ns = self.stopwatch.stop();

        // done
        Ok(())
    }

    /// Updates the stats structure (should be called after solve)
    fn update_stats(&self, stats: &mut StatsLinSol) {
        stats.main.solver = if cfg!(local_umfpack) {
            "UMFPACK-local".to_string()
        } else {
            "UMFPACK".to_string()
        };
        stats.determinant.mantissa_real = self.determinant_coefficient;
        stats.determinant.mantissa_imag = 0.0;
        stats.determinant.base = 10.0;
        stats.determinant.exponent = self.determinant_exponent;
        stats.output.umfpack_rcond_estimate = self.rcond_estimate;
        stats.output.effective_ordering = match self.effective_ordering {
            UMFPACK_ORDERING_CHOLMOD => "Cholmod".to_string(),
            UMFPACK_ORDERING_AMD => "Amd".to_string(),
            UMFPACK_ORDERING_METIS => "Metis".to_string(),
            UMFPACK_ORDERING_BEST => "Best".to_string(),
            UMFPACK_ORDERING_NONE => "No".to_string(),
            _ => "Unknown".to_string(),
        };
        stats.output.effective_scaling = match self.effective_scaling {
            UMFPACK_SCALE_NONE => "No".to_string(),
            UMFPACK_SCALE_SUM => "Sum".to_string(),
            UMFPACK_SCALE_MAX => "Max".to_string(),
            _ => "Unknown".to_string(),
        };
        stats.output.umfpack_strategy = match self.effective_strategy {
            UMFPACK_STRATEGY_AUTO => "Auto".to_string(),
            UMFPACK_STRATEGY_UNSYMMETRIC => "Unsymmetric".to_string(),
            UMFPACK_STRATEGY_SYMMETRIC => "Symmetric".to_string(),
            _ => "Unknown".to_string(),
        };
        stats.time_nanoseconds.initialize = self.time_initialize_ns;
        stats.time_nanoseconds.factorize = self.time_factorize_ns;
        stats.time_nanoseconds.solve = self.time_solve_ns;
    }
}

pub(crate) const UMFPACK_STRATEGY_AUTO: i32 = 0; // use symmetric or unsymmetric strategy
pub(crate) const UMFPACK_STRATEGY_UNSYMMETRIC: i32 = 1; // COLAMD(A), col-tree post-order, do not prefer diag
pub(crate) const UMFPACK_STRATEGY_SYMMETRIC: i32 = 3; // AMD(A+A'), no col-tree post-order, prefer diagonal

pub(crate) const UMFPACK_ORDERING_CHOLMOD: i32 = 0; // use CHOLMOD (AMD/COLAMD then METIS)
pub(crate) const UMFPACK_ORDERING_AMD: i32 = 1; // use AMD/COLAMD
pub(crate) const UMFPACK_ORDERING_METIS: i32 = 3; // use METIS
pub(crate) const UMFPACK_ORDERING_BEST: i32 = 4; // try many orderings, pick best
pub(crate) const UMFPACK_ORDERING_NONE: i32 = 5; // natural ordering
pub(crate) const UMFPACK_DEFAULT_ORDERING: i32 = UMFPACK_ORDERING_AMD;

pub(crate) const UMFPACK_SCALE_NONE: i32 = 0; // no scaling
pub(crate) const UMFPACK_SCALE_SUM: i32 = 1; // default: divide each row by sum (abs (row))
pub(crate) const UMFPACK_SCALE_MAX: i32 = 2; // divide each row by max (abs (row))
pub(crate) const UMFPACK_DEFAULT_SCALE: i32 = UMFPACK_SCALE_SUM;

/// Returns the UMFPACK ordering constant
pub(crate) fn umfpack_ordering(ordering: Ordering) -> i32 {
    match ordering {
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
    }
}

/// Returns the UMFPACK scaling constant
pub(crate) fn umfpack_scaling(scaling: Scaling) -> i32 {
    match scaling {
        Scaling::Auto => UMFPACK_DEFAULT_SCALE,
        Scaling::Column => UMFPACK_DEFAULT_SCALE,
        Scaling::Diagonal => UMFPACK_DEFAULT_SCALE,
        Scaling::Max => UMFPACK_SCALE_MAX,
        Scaling::No => UMFPACK_SCALE_NONE,
        Scaling::RowCol => UMFPACK_DEFAULT_SCALE,
        Scaling::RowColIter => UMFPACK_DEFAULT_SCALE,
        Scaling::RowColRig => UMFPACK_DEFAULT_SCALE,
        Scaling::Sum => UMFPACK_SCALE_SUM,
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
        ERROR_NULL_POINTER => return "UMFPACK failed due to NULL POINTER error",
        ERROR_MALLOC => return "UMFPACK failed due to MALLOC error",
        ERROR_VERSION => return "UMFPACK failed due to VERSION error",
        ERROR_NOT_AVAILABLE => return "UMFPACK is not AVAILABLE",
        ERROR_NEED_INITIALIZATION => return "UMFPACK failed because INITIALIZATION is needed",
        ERROR_NEED_FACTORIZATION => return "UMFPACK failed because FACTORIZATION is needed",
        ERROR_ALREADY_INITIALIZED => return "UMFPACK failed because INITIALIZATION has been completed already",
        _ => return "Error: unknown error returned by c-code (UMFPACK)",
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CooMatrix, Samples};
    use russell_lab::{approx_eq, vec_approx_eq};

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

        // COO to CSC errors
        let coo = CooMatrix::new(1, 1, 1, Sym::No).unwrap();
        let mut mat = SparseMatrix::from_coo(coo);
        assert_eq!(
            solver.factorize(&mut mat, None).err(),
            Some("COO to CSC requires nnz > 0")
        );

        // check CSC matrix
        let (coo, _, _, _) = Samples::rectangular_1x7();
        let mut mat = SparseMatrix::from_coo(coo);
        assert_eq!(
            solver.factorize(&mut mat, None).err(),
            Some("the matrix must be square")
        );
        let (coo, _, _, _) = Samples::mkl_symmetric_5x5_lower(false, false);
        let mut mat = SparseMatrix::from_coo(coo);
        assert_eq!(
            solver.factorize(&mut mat, None).err(),
            Some("UMFPACK requires Sym::YesFull for symmetric matrices")
        );

        // check already factorized data
        let mut coo = CooMatrix::new(2, 2, 2, Sym::No).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(1, 1, 2.0).unwrap();
        let mut mat = SparseMatrix::from_coo(coo);
        // ... factorize once => OK
        solver.factorize(&mut mat, None).unwrap();
        // ... change matrix (symmetric)
        let mut coo = CooMatrix::new(2, 2, 2, Sym::YesFull).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(1, 1, 2.0).unwrap();
        let mut mat = SparseMatrix::from_coo(coo);
        assert_eq!(
            solver.factorize(&mut mat, None).err(),
            Some("subsequent factorizations must use the same matrix (symmetric differs)")
        );
        // ... change matrix (ndim)
        let mut coo = CooMatrix::new(1, 1, 1, Sym::No).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        let mut mat = SparseMatrix::from_coo(coo);
        assert_eq!(
            solver.factorize(&mut mat, None).err(),
            Some("subsequent factorizations must use the same matrix (ndim differs)")
        );
        // ... change matrix (nnz)
        let mut coo = CooMatrix::new(2, 2, 1, Sym::No).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        let mut mat = SparseMatrix::from_coo(coo);
        assert_eq!(
            solver.factorize(&mut mat, None).err(),
            Some("subsequent factorizations must use the same matrix (nnz differs)")
        );
    }

    #[test]
    fn factorize_works() {
        let mut solver = SolverUMFPACK::new().unwrap();
        assert!(!solver.factorized);
        let (coo, _, _, _) = Samples::umfpack_unsymmetric_5x5();
        let mut mat = SparseMatrix::from_coo(coo);
        let mut params = LinSolParams::new();

        params.compute_determinant = true;
        params.ordering = Ordering::Amd;
        params.scaling = Scaling::Sum;

        solver.factorize(&mut mat, Some(params)).unwrap();
        assert!(solver.factorized);

        assert_eq!(solver.effective_ordering, UMFPACK_ORDERING_AMD);
        assert_eq!(solver.effective_scaling, UMFPACK_SCALE_SUM);

        let det = solver.determinant_coefficient * f64::powf(10.0, solver.determinant_exponent);
        approx_eq(det, 114.0, 1e-13);

        // calling factorize again works
        solver.factorize(&mut mat, Some(params)).unwrap();
        let det = solver.determinant_coefficient * f64::powf(10.0, solver.determinant_exponent);
        approx_eq(det, 114.0, 1e-13);
    }

    #[test]
    fn factorize_fails_on_singular_matrix() {
        let mut solver = SolverUMFPACK::new().unwrap();
        let mut coo = CooMatrix::new(2, 2, 2, Sym::No).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(1, 1, 0.0).unwrap();
        let mut mat = SparseMatrix::from_coo(coo);
        assert_eq!(solver.factorize(&mut mat, None), Err("Error(1): Matrix is singular"));
    }

    #[test]
    fn solve_handles_errors() {
        let mut coo = CooMatrix::new(2, 2, 2, Sym::No).unwrap();
        coo.put(0, 0, 123.0).unwrap();
        coo.put(1, 1, 456.0).unwrap();
        let mut mat = SparseMatrix::from_coo(coo);
        let mut solver = SolverUMFPACK::new().unwrap();
        assert!(!solver.factorized);
        let mut x = Vector::new(2);
        let rhs = Vector::new(2);
        assert_eq!(
            solver.solve(&mut x, &mut mat, &rhs, false),
            Err("the function factorize must be called before solve")
        );
        let mut x = Vector::new(1);
        solver.factorize(&mut mat, None).unwrap();
        assert_eq!(
            solver.solve(&mut x, &mut mat, &rhs, false),
            Err("the dimension of the vector of unknown values x is incorrect")
        );
        let mut x = Vector::new(2);
        let rhs = Vector::new(1);
        assert_eq!(
            solver.solve(&mut x, &mut mat, &rhs, false),
            Err("the dimension of the right-hand side vector is incorrect")
        );
        // wrong symmetric
        let rhs = Vector::new(2);
        let mut coo_wrong = CooMatrix::new(2, 2, 2, Sym::YesFull).unwrap();
        coo_wrong.put(0, 0, 123.0).unwrap();
        coo_wrong.put(1, 1, 456.0).unwrap();
        let mut mat_wrong = SparseMatrix::from_coo(coo_wrong);
        mat_wrong.get_csc_or_from_coo().unwrap(); // make sure to convert to CSC (because we're not calling factorize on this wrong matrix)
        assert_eq!(
            solver.solve(&mut x, &mut mat_wrong, &rhs, false),
            Err("solve must use the same matrix (symmetric differs)")
        );
        // wrong ndim
        let mut coo_wrong = CooMatrix::new(1, 1, 1, Sym::No).unwrap();
        coo_wrong.put(0, 0, 123.0).unwrap();
        let mut mat_wrong = SparseMatrix::from_coo(coo_wrong);
        mat_wrong.get_csc_or_from_coo().unwrap(); // make sure to convert to CSC (because we're not calling factorize on this wrong matrix)
        assert_eq!(
            solver.solve(&mut x, &mut mat_wrong, &rhs, false),
            Err("solve must use the same matrix (ndim differs)")
        );
        // wrong nnz
        let mut coo_wrong = CooMatrix::new(2, 2, 3, Sym::No).unwrap();
        coo_wrong.put(0, 0, 123.0).unwrap();
        coo_wrong.put(1, 1, 123.0).unwrap();
        coo_wrong.put(0, 1, 100.0).unwrap();
        let mut mat_wrong = SparseMatrix::from_coo(coo_wrong);
        mat_wrong.get_csc_or_from_coo().unwrap(); // make sure to convert to CSC (because we're not calling factorize on this wrong matrix)
        assert_eq!(
            solver.solve(&mut x, &mut mat_wrong, &rhs, false),
            Err("solve must use the same matrix (nnz differs)")
        );
    }

    #[test]
    fn solve_works() {
        let mut solver = SolverUMFPACK::new().unwrap();
        let (coo, _, _, _) = Samples::umfpack_unsymmetric_5x5();
        let mut mat = SparseMatrix::from_coo(coo);
        let mut x = Vector::new(5);
        let rhs = Vector::from(&[8.0, 45.0, -3.0, 3.0, 19.0]);
        let x_correct = &[1.0, 2.0, 3.0, 4.0, 5.0];
        solver.factorize(&mut mat, None).unwrap();
        solver.solve(&mut x, &mut mat, &rhs, false).unwrap();
        vec_approx_eq(x.as_data(), x_correct, 1e-14);

        // calling solve again works
        let mut x_again = Vector::new(5);
        solver.solve(&mut x_again, &mut mat, &rhs, false).unwrap();
        vec_approx_eq(x_again.as_data(), x_correct, 1e-14);

        // update stats
        let mut stats = StatsLinSol::new();
        solver.update_stats(&mut stats);
        assert_eq!(stats.output.effective_ordering, "Amd");
        assert_eq!(stats.output.effective_scaling, "Sum");
    }

    #[test]
    fn solve_works_symmetric() {
        let mut solver = SolverUMFPACK::new().unwrap();
        let (coo, _, _, _) = Samples::mkl_symmetric_5x5_full();
        let mut mat = SparseMatrix::from_coo(coo);
        let mut x = Vector::new(5);
        let rhs = Vector::from(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let x_correct = &[-979.0 / 3.0, 983.0, 1961.0 / 12.0, 398.0, 123.0 / 2.0];
        solver.factorize(&mut mat, None).unwrap();
        solver.solve(&mut x, &mut mat, &rhs, false).unwrap();
        vec_approx_eq(x.as_data(), x_correct, 1e-10);

        // calling solve again works
        let mut x_again = Vector::new(5);
        solver.solve(&mut x_again, &mut mat, &rhs, false).unwrap();
        vec_approx_eq(x_again.as_data(), x_correct, 1e-10);

        // update stats
        let mut stats = StatsLinSol::new();
        solver.update_stats(&mut stats);
        assert_eq!(stats.output.effective_ordering, "Amd");
        assert_eq!(stats.output.effective_scaling, "Sum");
    }

    #[test]
    fn ordering_and_scaling_works() {
        assert_eq!(umfpack_ordering(Ordering::Amd), UMFPACK_ORDERING_AMD);
        assert_eq!(umfpack_ordering(Ordering::Amf), UMFPACK_DEFAULT_ORDERING);
        assert_eq!(umfpack_ordering(Ordering::Auto), UMFPACK_DEFAULT_ORDERING);
        assert_eq!(umfpack_ordering(Ordering::Best), UMFPACK_ORDERING_BEST);
        assert_eq!(umfpack_ordering(Ordering::Cholmod), UMFPACK_ORDERING_CHOLMOD);
        assert_eq!(umfpack_ordering(Ordering::Metis), UMFPACK_ORDERING_METIS);
        assert_eq!(umfpack_ordering(Ordering::No), UMFPACK_ORDERING_NONE);
        assert_eq!(umfpack_ordering(Ordering::Pord), UMFPACK_DEFAULT_ORDERING);
        assert_eq!(umfpack_ordering(Ordering::Qamd), UMFPACK_DEFAULT_ORDERING);
        assert_eq!(umfpack_ordering(Ordering::Scotch), UMFPACK_DEFAULT_ORDERING);

        assert_eq!(umfpack_scaling(Scaling::Auto), UMFPACK_DEFAULT_SCALE);
        assert_eq!(umfpack_scaling(Scaling::Column), UMFPACK_DEFAULT_SCALE);
        assert_eq!(umfpack_scaling(Scaling::Diagonal), UMFPACK_DEFAULT_SCALE);
        assert_eq!(umfpack_scaling(Scaling::Max), UMFPACK_SCALE_MAX);
        assert_eq!(umfpack_scaling(Scaling::No), UMFPACK_SCALE_NONE);
        assert_eq!(umfpack_scaling(Scaling::RowCol), UMFPACK_DEFAULT_SCALE);
        assert_eq!(umfpack_scaling(Scaling::RowColIter), UMFPACK_DEFAULT_SCALE);
        assert_eq!(umfpack_scaling(Scaling::RowColRig), UMFPACK_DEFAULT_SCALE);
        assert_eq!(umfpack_scaling(Scaling::Sum), UMFPACK_SCALE_SUM);
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
            handle_umfpack_error_code(ERROR_NULL_POINTER),
            "UMFPACK failed due to NULL POINTER error"
        );
        assert_eq!(
            handle_umfpack_error_code(ERROR_MALLOC),
            "UMFPACK failed due to MALLOC error"
        );
        assert_eq!(
            handle_umfpack_error_code(ERROR_VERSION),
            "UMFPACK failed due to VERSION error"
        );
        assert_eq!(
            handle_umfpack_error_code(ERROR_NOT_AVAILABLE),
            "UMFPACK is not AVAILABLE"
        );
        assert_eq!(
            handle_umfpack_error_code(ERROR_NEED_INITIALIZATION),
            "UMFPACK failed because INITIALIZATION is needed"
        );
        assert_eq!(
            handle_umfpack_error_code(ERROR_NEED_FACTORIZATION),
            "UMFPACK failed because FACTORIZATION is needed"
        );
        assert_eq!(
            handle_umfpack_error_code(ERROR_ALREADY_INITIALIZED),
            "UMFPACK failed because INITIALIZATION has been completed already"
        );
        assert_eq!(handle_umfpack_error_code(123), default);
    }
}
