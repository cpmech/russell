use super::{LinSolParams, LinSolTrait, Ordering, Scaling, SparseMatrix, StatsLinSol, Sym};
use crate::constants::*;
use crate::StrError;
use russell_lab::{vec_copy, Stopwatch, Vector};

/// Opaque struct holding a C-pointer to InterfaceKLU
///
/// Reference: <https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs>
#[repr(C)]
struct InterfaceKLU {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

/// Enforce Send on the C structure
///
/// <https://stackoverflow.com/questions/50258359/can-a-struct-containing-a-raw-pointer-implement-send-and-be-ffi-safe>
unsafe impl Send for InterfaceKLU {}

/// Enforce Send on the Rust structure
///
/// <https://stackoverflow.com/questions/50258359/can-a-struct-containing-a-raw-pointer-implement-send-and-be-ffi-safe>
unsafe impl Send for SolverKLU {}

extern "C" {
    fn solver_klu_new() -> *mut InterfaceKLU;
    fn solver_klu_drop(solver: *mut InterfaceKLU);
    fn solver_klu_initialize(
        solver: *mut InterfaceKLU,
        ordering: i32,
        scaling: i32,
        ndim: i32,
        col_pointers: *const i32,
        row_indices: *const i32,
    ) -> i32;
    fn solver_klu_factorize(
        solver: *mut InterfaceKLU,
        effective_ordering: *mut i32,
        effective_scaling: *mut i32,
        cond_estimate: *mut f64,
        compute_cond: CcBool,
        col_pointers: *const i32,
        row_indices: *const i32,
        values: *const f64,
    ) -> i32;
    fn solver_klu_solve(solver: *mut InterfaceKLU, ndim: i32, in_rhs_out_x: *mut f64) -> i32;
}

/// Wraps the KLU solver for sparse linear systems
///
/// **Warning:** This solver may "run out of memory" for very large matrices.
pub struct SolverKLU {
    /// Holds a pointer to the C interface to KLU
    solver: *mut InterfaceKLU,

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

    /// Holds the used ordering (after factorize)
    effective_ordering: i32,

    /// Holds the used scaling (after factorize)
    effective_scaling: i32,

    /// Holds the 1-norm condition number estimate (after factorize)
    cond_estimate: f64,

    /// Stopwatch to measure computation times
    stopwatch: Stopwatch,

    /// Time spent on initialize in nanoseconds
    time_initialize_ns: u128,

    /// Time spent on factorize in nanoseconds
    time_factorize_ns: u128,

    /// Time spent on solve in nanoseconds
    time_solve_ns: u128,
}

impl Drop for SolverKLU {
    /// Tells the c-code to release memory
    fn drop(&mut self) {
        unsafe {
            solver_klu_drop(self.solver);
        }
    }
}

impl SolverKLU {
    /// Allocates a new instance
    pub fn new() -> Result<Self, StrError> {
        unsafe {
            let solver = solver_klu_new();
            if solver.is_null() {
                return Err("c-code failed to allocate the KLU solver");
            }
            Ok(SolverKLU {
                solver,
                initialized: false,
                factorized: false,
                initialized_sym: Sym::No,
                initialized_ndim: 0,
                initialized_nnz: 0,
                effective_ordering: -1,
                effective_scaling: -1,
                cond_estimate: 0.0,
                stopwatch: Stopwatch::new(),
                time_initialize_ns: 0,
                time_factorize_ns: 0,
                time_solve_ns: 0,
            })
        }
    }
}

impl LinSolTrait for SolverKLU {
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
    /// 4. For symmetric matrices, `KLU` requires [Sym::YesFull]
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
                return Err("KLU requires Sym::YesFull for symmetric matrices");
            }
            self.initialized_sym = csc.symmetric;
            self.initialized_ndim = csc.nrow;
            self.initialized_nnz = csc.col_pointers[csc.ncol] as usize;
        }

        // parameters
        let par = if let Some(p) = params { p } else { LinSolParams::new() };

        // input parameters
        let ordering = klu_ordering(par.ordering);
        let scaling = klu_scaling(par.scaling);

        // requests
        let compute_cond = if par.compute_condition_numbers { 1 } else { 0 };

        // matrix config
        let ndim = to_i32(csc.nrow);

        // call initialize just once
        if !self.initialized {
            self.stopwatch.reset();
            unsafe {
                let status = solver_klu_initialize(
                    self.solver,
                    ordering,
                    scaling,
                    ndim,
                    csc.col_pointers.as_ptr(),
                    csc.row_indices.as_ptr(),
                );
                if status != SUCCESSFUL_EXIT {
                    return Err(handle_klu_error_code(status));
                }
            }
            self.time_initialize_ns = self.stopwatch.stop();
            self.initialized = true;
        }

        // call factorize
        self.stopwatch.reset();
        unsafe {
            let status = solver_klu_factorize(
                self.solver,
                &mut self.effective_ordering,
                &mut self.effective_scaling,
                &mut self.cond_estimate,
                compute_cond,
                csc.col_pointers.as_ptr(),
                csc.row_indices.as_ptr(),
                csc.values.as_ptr(),
            );
            if status != SUCCESSFUL_EXIT {
                return Err(handle_klu_error_code(status));
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
    /// * `verbose` -- NOT AVAILABLE
    ///
    /// **Warning:** the matrix must be same one used in `factorize`.
    fn solve(&mut self, x: &mut Vector, mat: &SparseMatrix, rhs: &Vector, _verbose: bool) -> Result<(), StrError> {
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

        // call KLU solve
        let ndim = to_i32(self.initialized_ndim);
        vec_copy(x, rhs).unwrap();
        self.stopwatch.reset();
        unsafe {
            let status = solver_klu_solve(self.solver, ndim, x.as_mut_data().as_mut_ptr());
            if status != SUCCESSFUL_EXIT {
                return Err(handle_klu_error_code(status));
            }
        }
        self.time_solve_ns = self.stopwatch.stop();

        // done
        Ok(())
    }

    /// Updates the stats structure (should be called after solve)
    fn update_stats(&self, stats: &mut StatsLinSol) {
        stats.main.solver = if cfg!(feature = "local_suitesparse") {
            "KLU-local".to_string()
        } else {
            "KLU".to_string()
        };
        stats.output.umfpack_rcond_estimate = self.cond_estimate;
        stats.output.effective_ordering = match self.effective_ordering {
            KLU_ORDERING_AMD => "Amd".to_string(),
            KLU_ORDERING_COLAMD => "Colamd".to_string(),
            _ => "Unknown".to_string(),
        };
        stats.output.effective_scaling = match self.effective_scaling {
            KLU_SCALE_NONE => "No".to_string(),
            KLU_SCALE_SUM => "Sum".to_string(),
            KLU_SCALE_MAX => "Max".to_string(),
            _ => "Unknown".to_string(),
        };
        stats.time_nanoseconds.initialize = self.time_initialize_ns;
        stats.time_nanoseconds.factorize = self.time_factorize_ns;
        stats.time_nanoseconds.solve = self.time_solve_ns;
    }

    /// Returns the nanoseconds spent on initialize
    fn get_ns_init(&self) -> u128 {
        self.time_initialize_ns
    }

    /// Returns the nanoseconds spent on factorize
    fn get_ns_fact(&self) -> u128 {
        self.time_factorize_ns
    }

    /// Returns the nanoseconds spent on solve
    fn get_ns_solve(&self) -> u128 {
        self.time_solve_ns
    }
}

pub(crate) const KLU_ORDERING_AUTO: i32 = -10; // (code defined here) use defaults
pub(crate) const KLU_ORDERING_AMD: i32 = 0; // (from KLU manual) AMD
pub(crate) const KLU_ORDERING_COLAMD: i32 = 1; // (from KLU manual) COLAMD

pub(crate) const KLU_SCALE_AUTO: i32 = -10; // (code defined here) use defaults
pub(crate) const KLU_SCALE_NONE: i32 = 0; // (from KLU manual) no scaling
pub(crate) const KLU_SCALE_SUM: i32 = 1; // (from KLU manual) divide each row by sum (abs (row))
pub(crate) const KLU_SCALE_MAX: i32 = 2; // (from KLU manual) divide each row by max (abs (row))

/// Returns the KLU ordering constant
pub(crate) fn klu_ordering(ordering: Ordering) -> i32 {
    match ordering {
        Ordering::Amd => KLU_ORDERING_AMD,
        Ordering::Amf => KLU_ORDERING_AUTO,
        Ordering::Auto => KLU_ORDERING_AUTO,
        Ordering::Best => KLU_ORDERING_AUTO,
        Ordering::Cholmod => KLU_ORDERING_AUTO,
        Ordering::Colamd => KLU_ORDERING_COLAMD,
        Ordering::Metis => KLU_ORDERING_AUTO,
        Ordering::No => KLU_ORDERING_AUTO,
        Ordering::Pord => KLU_ORDERING_AUTO,
        Ordering::Qamd => KLU_ORDERING_AUTO,
        Ordering::Scotch => KLU_ORDERING_AUTO,
    }
}

/// Returns the KLU scaling constant
pub(crate) fn klu_scaling(scaling: Scaling) -> i32 {
    match scaling {
        Scaling::Auto => KLU_SCALE_AUTO,
        Scaling::Column => KLU_ORDERING_AUTO,
        Scaling::Diagonal => KLU_ORDERING_AUTO,
        Scaling::Max => KLU_SCALE_MAX,
        Scaling::No => KLU_SCALE_NONE,
        Scaling::RowCol => KLU_ORDERING_AUTO,
        Scaling::RowColIter => KLU_ORDERING_AUTO,
        Scaling::RowColRig => KLU_ORDERING_AUTO,
        Scaling::Sum => KLU_SCALE_SUM,
    }
}

/// Handles KLU error code
pub(crate) fn handle_klu_error_code(err: i32) -> StrError {
    match err {
        -9 => "klu_analyze failed",
        -8 => "klu_factor failed",
        -7 => "klu_condest failed",
        ERROR_NULL_POINTER => "KLU failed due to NULL POINTER error",
        ERROR_MALLOC => "KLU failed due to MALLOC error",
        ERROR_VERSION => "KLU failed due to VERSION error",
        ERROR_NOT_AVAILABLE => "KLU is not AVAILABLE",
        ERROR_NEED_INITIALIZATION => "KLU failed because INITIALIZATION is needed",
        ERROR_NEED_FACTORIZATION => "KLU failed because FACTORIZATION is needed",
        ERROR_ALREADY_INITIALIZED => "KLU failed because INITIALIZATION has been completed already",
        _ => "Error: unknown error returned by c-code (KLU)",
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CooMatrix, Samples};
    use russell_lab::vec_approx_eq;

    #[test]
    fn new_and_drop_work() {
        // you may debug into the C-code to see that drop is working
        let solver = SolverKLU::new().unwrap();
        assert!(!solver.factorized);
    }

    #[test]
    fn factorize_handles_errors() {
        let mut solver = SolverKLU::new().unwrap();
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
            Some("KLU requires Sym::YesFull for symmetric matrices")
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
        let mut solver = SolverKLU::new().unwrap();
        assert!(!solver.factorized);
        let (coo, _, _, _) = Samples::umfpack_unsymmetric_5x5();
        let mut mat = SparseMatrix::from_coo(coo);

        let mut params = LinSolParams::new();
        params.ordering = Ordering::Metis;
        params.scaling = Scaling::Sum;

        solver.factorize(&mut mat, Some(params)).unwrap();
        assert!(solver.factorized);
        assert_eq!(solver.effective_ordering, KLU_ORDERING_AMD);
        assert_eq!(solver.effective_scaling, KLU_SCALE_SUM);

        // calling factorize again works
        solver.factorize(&mut mat, Some(params)).unwrap();
    }

    #[test]
    fn factorize_fails_on_singular_matrix() {
        let mut solver = SolverKLU::new().unwrap();
        let mut coo = CooMatrix::new(2, 2, 2, Sym::No).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(1, 1, 0.0).unwrap();
        let mut mat = SparseMatrix::from_coo(coo);
        assert_eq!(solver.factorize(&mut mat, None), Err("klu_factor failed"));
    }

    #[test]
    fn solve_handles_errors() {
        let mut coo = CooMatrix::new(2, 2, 2, Sym::No).unwrap();
        coo.put(0, 0, 123.0).unwrap();
        coo.put(1, 1, 456.0).unwrap();
        let mut mat = SparseMatrix::from_coo(coo);
        let mut solver = SolverKLU::new().unwrap();
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
        let mut solver = SolverKLU::new().unwrap();
        let (coo, _, _, _) = Samples::umfpack_unsymmetric_5x5();
        let mut mat = SparseMatrix::from_coo(coo);
        let mut x = Vector::new(5);
        let rhs = Vector::from(&[8.0, 45.0, -3.0, 3.0, 19.0]);
        let x_correct = &[1.0, 2.0, 3.0, 4.0, 5.0];

        let mut params = LinSolParams::new();
        params.ordering = Ordering::Cholmod;
        params.scaling = Scaling::Max;

        solver.factorize(&mut mat, Some(params)).unwrap();
        solver.solve(&mut x, &mut mat, &rhs, false).unwrap();
        vec_approx_eq(&x, x_correct, 1e-14);

        // calling solve again works
        let mut x_again = Vector::new(5);
        solver.solve(&mut x_again, &mut mat, &rhs, false).unwrap();
        vec_approx_eq(&x_again, x_correct, 1e-14);

        // update stats
        let mut stats = StatsLinSol::new();
        solver.update_stats(&mut stats);
        assert_eq!(stats.output.effective_ordering, "Amd");
        assert_eq!(stats.output.effective_scaling, "Max");
    }

    #[test]
    fn solve_works_symmetric() {
        let mut solver = SolverKLU::new().unwrap();
        let (coo, _, _, _) = Samples::mkl_symmetric_5x5_full();
        let mut mat = SparseMatrix::from_coo(coo);
        let mut x = Vector::new(5);
        let rhs = Vector::from(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let x_correct = &[-979.0 / 3.0, 983.0, 1961.0 / 12.0, 398.0, 123.0 / 2.0];

        let mut params = LinSolParams::new();
        params.ordering = Ordering::Colamd;
        params.scaling = Scaling::No;

        solver.factorize(&mut mat, Some(params)).unwrap();
        solver.solve(&mut x, &mut mat, &rhs, false).unwrap();
        vec_approx_eq(&x, x_correct, 1e-10);

        // calling solve again works
        let mut x_again = Vector::new(5);
        solver.solve(&mut x_again, &mut mat, &rhs, false).unwrap();
        vec_approx_eq(&x_again, x_correct, 1e-10);

        // update stats
        let mut stats = StatsLinSol::new();
        solver.update_stats(&mut stats);
        assert_eq!(stats.output.effective_ordering, "Colamd");
        assert_eq!(stats.output.effective_scaling, "No");
    }

    #[test]
    fn ordering_and_scaling_works() {
        assert_eq!(klu_ordering(Ordering::Amd), KLU_ORDERING_AMD);
        assert_eq!(klu_ordering(Ordering::Amf), KLU_ORDERING_AUTO);
        assert_eq!(klu_ordering(Ordering::Auto), KLU_ORDERING_AUTO);
        assert_eq!(klu_ordering(Ordering::Best), KLU_ORDERING_AUTO);
        assert_eq!(klu_ordering(Ordering::Cholmod), KLU_ORDERING_AUTO);
        assert_eq!(klu_ordering(Ordering::Colamd), KLU_ORDERING_COLAMD);
        assert_eq!(klu_ordering(Ordering::Metis), KLU_ORDERING_AUTO);
        assert_eq!(klu_ordering(Ordering::No), KLU_ORDERING_AUTO);
        assert_eq!(klu_ordering(Ordering::Pord), KLU_ORDERING_AUTO);
        assert_eq!(klu_ordering(Ordering::Qamd), KLU_ORDERING_AUTO);
        assert_eq!(klu_ordering(Ordering::Scotch), KLU_ORDERING_AUTO);

        assert_eq!(klu_scaling(Scaling::Auto), KLU_SCALE_AUTO);
        assert_eq!(klu_scaling(Scaling::Column), KLU_SCALE_AUTO);
        assert_eq!(klu_scaling(Scaling::Diagonal), KLU_SCALE_AUTO);
        assert_eq!(klu_scaling(Scaling::Max), KLU_SCALE_MAX);
        assert_eq!(klu_scaling(Scaling::No), KLU_SCALE_NONE);
        assert_eq!(klu_scaling(Scaling::RowCol), KLU_SCALE_AUTO);
        assert_eq!(klu_scaling(Scaling::RowColIter), KLU_SCALE_AUTO);
        assert_eq!(klu_scaling(Scaling::RowColRig), KLU_SCALE_AUTO);
        assert_eq!(klu_scaling(Scaling::Sum), KLU_SCALE_SUM);
    }

    #[test]
    fn handle_klu_error_code_works() {
        let default = "Error: unknown error returned by c-code (KLU)";
        assert_eq!(handle_klu_error_code(-9), "klu_analyze failed");
        assert_eq!(handle_klu_error_code(-8), "klu_factor failed");
        assert_eq!(handle_klu_error_code(-7), "klu_condest failed");
        assert_eq!(
            handle_klu_error_code(ERROR_NULL_POINTER),
            "KLU failed due to NULL POINTER error"
        );
        assert_eq!(handle_klu_error_code(ERROR_MALLOC), "KLU failed due to MALLOC error");
        assert_eq!(handle_klu_error_code(ERROR_VERSION), "KLU failed due to VERSION error");
        assert_eq!(handle_klu_error_code(ERROR_NOT_AVAILABLE), "KLU is not AVAILABLE");
        assert_eq!(
            handle_klu_error_code(ERROR_NEED_INITIALIZATION),
            "KLU failed because INITIALIZATION is needed"
        );
        assert_eq!(
            handle_klu_error_code(ERROR_NEED_FACTORIZATION),
            "KLU failed because FACTORIZATION is needed"
        );
        assert_eq!(
            handle_klu_error_code(ERROR_ALREADY_INITIALIZED),
            "KLU failed because INITIALIZATION has been completed already"
        );
        assert_eq!(handle_klu_error_code(123), default);
    }
}
