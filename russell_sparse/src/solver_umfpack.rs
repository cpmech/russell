use super::{LinSolParams, LinSolTrait, Ordering, Scaling, SparseMatrix, StatsLinSol, Symmetry};
use crate::auxiliary_and_constants::{
    to_i32, CcBool, MALLOC_ERROR, NEED_FACTORIZATION, NULL_POINTER_ERROR, SUCCESSFUL_EXIT,
};
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
        rcond_estimate: *mut f64,
        determinant_coefficient: *mut f64,
        determinant_exponent: *mut f64,
        // input
        ordering: i32,
        scaling: i32,
        // requests
        compute_determinant: CcBool,
        verbose: CcBool,
        // matrix config
        enforce_unsymmetric_strategy: CcBool,
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
        verbose: CcBool,
    ) -> i32;
}

/// Wraps the UMFPACK solver for sparse linear systems
///
/// **Warning:** This solver may "run out of memory" for very large matrices.
pub struct SolverUMFPACK {
    /// Holds a pointer to the C interface to UMFPACK
    solver: *mut InterfaceUMFPACK,

    /// Indicates whether the sparse matrix has been factorized or not
    factorized: bool,

    /// Holds the symmetry type used in the first call to factorize
    factorized_symmetry: Option<Symmetry>,

    /// Holds the matrix dimension saved in the first call to factorize
    factorized_ndim: usize,

    /// Holds the number of non-zeros saved in the first call to factorize
    factorized_nnz: usize,

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
                factorized_symmetry: None,
                factorized_ndim: 0,
                factorized_nnz: 0,
                effective_strategy: -1,
                effective_ordering: -1,
                effective_scaling: -1,
                rcond_estimate: 0.0,
                determinant_coefficient: 0.0,
                determinant_exponent: 0.0,
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
    ///   the symmetry/storage must [crate::Storage::Full].
    /// * `params` -- configuration parameters; None => use default
    ///
    /// # Notes
    ///
    /// 1. The structure of the matrix (nrow, ncol, nnz, symmetry) must be
    ///    exactly the same among multiple calls to `factorize`. The values may differ
    ///    from call to call, nonetheless.
    /// 2. The first call to `factorize` will define the structure which must be
    ///    kept the same for the next calls.
    /// 3. If the structure of the matrix needs to be changed, the solver must
    ///    be "dropped" and a new solver allocated.
    /// 4. For symmetric matrices, `UMFPACK` requires that the symmetry/storage be [crate::Storage::Full].
    fn factorize(&mut self, mat: &mut SparseMatrix, params: Option<LinSolParams>) -> Result<(), StrError> {
        // get CSC matrix
        // (or convert from COO if CSC is not available and COO is available)
        let csc = mat.get_csc_or_from_coo()?;

        // check CSC matrix
        if csc.nrow != csc.ncol {
            return Err("the matrix must be square");
        }
        if let Some(symmetry) = csc.symmetry {
            if symmetry.triangular() {
                return Err("for UMFPACK, the matrix must not be triangular");
            }
        }

        // check already factorized data
        if self.factorized {
            if csc.symmetry != self.factorized_symmetry {
                return Err("subsequent factorizations must use the same matrix (symmetry differs)");
            }
            if csc.nrow != self.factorized_ndim {
                return Err("subsequent factorizations must use the same matrix (ndim differs)");
            }
            if (csc.col_pointers[csc.ncol] as usize) != self.factorized_nnz {
                return Err("subsequent factorizations must use the same matrix (nnz differs)");
            }
        } else {
            self.factorized_symmetry = csc.symmetry;
            self.factorized_ndim = csc.nrow;
            self.factorized_nnz = csc.col_pointers[csc.ncol] as usize;
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

        // call UMFPACK factorize
        unsafe {
            let status = solver_umfpack_factorize(
                self.solver,
                // output
                &mut self.effective_strategy,
                &mut self.effective_ordering,
                &mut self.effective_scaling,
                &mut self.rcond_estimate,
                &mut self.determinant_coefficient,
                &mut self.determinant_exponent,
                // input
                ordering,
                scaling,
                // requests
                compute_determinant,
                verbose,
                // matrix config
                enforce_unsym,
                ndim,
                // matrix
                csc.col_pointers.as_ptr(),
                csc.row_indices.as_ptr(),
                csc.values.as_ptr(),
            );
            if status != SUCCESSFUL_EXIT {
                return Err(handle_umfpack_error_code(status));
            }
        }

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
    /// * `mat` -- the coefficient matrix A; must be square and, if symmetric, [crate::Storage::Full].
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
        let (nrow, ncol, nnz, symmetry) = csc.get_info();
        if symmetry != self.factorized_symmetry {
            return Err("solve must use the same matrix (symmetry differs)");
        }
        if nrow != self.factorized_ndim || ncol != self.factorized_ndim {
            return Err("solve must use the same matrix (ndim differs)");
        }
        if nnz != self.factorized_nnz {
            return Err("solve must use the same matrix (nnz differs)");
        }

        // check vectors
        if x.dim() != self.factorized_ndim {
            return Err("the dimension of the vector of unknown values x is incorrect");
        }
        if rhs.dim() != self.factorized_ndim {
            return Err("the dimension of the right-hand side vector is incorrect");
        }

        // call UMFPACK solve
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
            if status != SUCCESSFUL_EXIT {
                return Err(handle_umfpack_error_code(status));
            }
        }

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
        stats.determinant.mantissa = self.determinant_coefficient;
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
    }
}

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

fn umfpack_ordering(ordering: Ordering) -> i32 {
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

fn umfpack_scaling(scaling: Scaling) -> i32 {
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
        NULL_POINTER_ERROR => return "Error: c-code returned null pointer (UMFPACK)",
        MALLOC_ERROR => return "Error: c-code failed to allocate memory (UMFPACK)",
        NEED_FACTORIZATION => return "INTERNAL ERROR: factorization must be completed before solve",
        _ => return "Error: unknown error returned by c-code (UMFPACK)",
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CooMatrix, LinSolParams, LinSolTrait, Ordering, Samples, Scaling, SparseMatrix};
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
        let (coo, _, _, _) = Samples::rectangular_1x7();
        let mut mat = SparseMatrix::from_coo(coo);
        assert_eq!(
            solver.factorize(&mut mat, None).err(),
            Some("the matrix must be square")
        );
        let coo = CooMatrix::new(1, 1, 1, None, false).unwrap();
        let mut mat = SparseMatrix::from_coo(coo);
        assert_eq!(
            solver.factorize(&mut mat, None).err(),
            Some("COO to CSC requires nnz > 0")
        );
        let (coo, _, _, _) = Samples::mkl_symmetric_5x5_lower(false, false, false);
        let mut mat = SparseMatrix::from_coo(coo);
        assert_eq!(
            solver.factorize(&mut mat, None).err(),
            Some("for UMFPACK, the matrix must not be triangular")
        );
    }

    #[test]
    fn factorize_works() {
        let mut solver = SolverUMFPACK::new().unwrap();
        assert!(!solver.factorized);
        let (coo, _, _, _) = Samples::umfpack_unsymmetric_5x5(false);
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
        let mut coo = CooMatrix::new(2, 2, 2, None, false).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(1, 1, 0.0).unwrap();
        let mut mat = SparseMatrix::from_coo(coo);
        assert_eq!(solver.factorize(&mut mat, None), Err("Error(1): Matrix is singular"));
    }

    #[test]
    fn solve_handles_errors() {
        let (coo, _, _, _) = Samples::tiny_1x1(false);
        let mut mat = SparseMatrix::from_coo(coo);
        let mut solver = SolverUMFPACK::new().unwrap();
        assert!(!solver.factorized);
        let mut x = Vector::new(2);
        let rhs = Vector::new(1);
        assert_eq!(
            solver.solve(&mut x, &mut mat, &rhs, false),
            Err("the function factorize must be called before solve")
        );
        solver.factorize(&mut mat, None).unwrap();
        assert_eq!(
            solver.solve(&mut x, &mut mat, &rhs, false),
            Err("the dimension of the vector of unknown values x is incorrect")
        );
        let mut x = Vector::new(1);
        let rhs = Vector::new(2);
        assert_eq!(
            solver.solve(&mut x, &mut mat, &rhs, false),
            Err("the dimension of the right-hand side vector is incorrect")
        );
    }

    #[test]
    fn solve_works() {
        let mut solver = SolverUMFPACK::new().unwrap();
        let (coo, _, _, _) = Samples::umfpack_unsymmetric_5x5(false);
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
    }

    #[test]
    fn get_ordering_and_scaling_works() {
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
