use super::{handle_umfpack_error_code, umfpack_ordering, umfpack_scaling};
use super::{ComplexCooMatrix, ComplexCscMatrix, ComplexLinSolTrait, LinSolParams, StatsLinSol, Sym};
use super::{
    UMFPACK_ORDERING_AMD, UMFPACK_ORDERING_BEST, UMFPACK_ORDERING_CHOLMOD, UMFPACK_ORDERING_METIS,
    UMFPACK_ORDERING_NONE, UMFPACK_SCALE_MAX, UMFPACK_SCALE_NONE, UMFPACK_SCALE_SUM, UMFPACK_STRATEGY_AUTO,
    UMFPACK_STRATEGY_SYMMETRIC, UMFPACK_STRATEGY_UNSYMMETRIC,
};
use crate::constants::*;
use crate::StrError;
use russell_lab::{Complex64, ComplexVector, Stopwatch};

/// Opaque struct holding a C-pointer to InterfaceComplexUMFPACK
///
/// Reference: <https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs>
#[repr(C)]
struct InterfaceComplexUMFPACK {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

/// Enforce Send on the C structure
///
/// <https://stackoverflow.com/questions/50258359/can-a-struct-containing-a-raw-pointer-implement-send-and-be-ffi-safe>
unsafe impl Send for InterfaceComplexUMFPACK {}

/// Enforce Send on the Rust structure
///
/// <https://stackoverflow.com/questions/50258359/can-a-struct-containing-a-raw-pointer-implement-send-and-be-ffi-safe>
unsafe impl Send for ComplexSolverUMFPACK {}

extern "C" {
    fn complex_solver_umfpack_new() -> *mut InterfaceComplexUMFPACK;
    fn complex_solver_umfpack_drop(solver: *mut InterfaceComplexUMFPACK);
    fn complex_solver_umfpack_initialize(
        solver: *mut InterfaceComplexUMFPACK,
        ordering: i32,
        scaling: i32,
        verbose: CcBool,
        enforce_unsymmetric_strategy: CcBool,
        ndim: i32,
        col_pointers: *const i32,
        row_indices: *const i32,
        values: *const Complex64,
    ) -> i32;
    fn complex_solver_umfpack_factorize(
        solver: *mut InterfaceComplexUMFPACK,
        effective_strategy: *mut i32,
        effective_ordering: *mut i32,
        effective_scaling: *mut i32,
        rcond_estimate: *mut f64,
        determinant_coefficient_real: *mut f64,
        determinant_coefficient_imag: *mut f64,
        determinant_exponent: *mut f64,
        compute_determinant: CcBool,
        verbose: CcBool,
        col_pointers: *const i32,
        row_indices: *const i32,
        values: *const Complex64,
    ) -> i32;
    fn complex_solver_umfpack_solve(
        solver: *mut InterfaceComplexUMFPACK,
        x: *mut Complex64,
        rhs: *const Complex64,
        col_pointers: *const i32,
        row_indices: *const i32,
        values: *const Complex64,
        verbose: CcBool,
    ) -> i32;
}

/// Wraps the UMFPACK solver for sparse linear systems
///
/// **Warning:** This solver may "run out of memory" for very large matrices.
pub struct ComplexSolverUMFPACK {
    /// Holds a pointer to the C interface to UMFPACK
    solver: *mut InterfaceComplexUMFPACK,

    /// Holds the CSC matrix used in factorize
    csc: Option<ComplexCscMatrix>,

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

    /// Holds the determinant coefficient (if requested) (real part)
    ///
    /// det = coefficient * pow(10, exponent)
    determinant_coefficient_real: f64,

    /// Holds the determinant coefficient (if requested) (imaginary part)
    ///
    /// det = coefficient * pow(10, exponent)
    determinant_coefficient_imag: f64,

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

impl Drop for ComplexSolverUMFPACK {
    /// Tells the c-code to release memory
    fn drop(&mut self) {
        unsafe {
            complex_solver_umfpack_drop(self.solver);
        }
    }
}

impl ComplexSolverUMFPACK {
    /// Allocates a new instance
    pub fn new() -> Result<Self, StrError> {
        unsafe {
            let solver = complex_solver_umfpack_new();
            if solver.is_null() {
                return Err("c-code failed to allocate the UMFPACK solver");
            }
            Ok(ComplexSolverUMFPACK {
                solver,
                csc: None,
                initialized: false,
                factorized: false,
                initialized_sym: Sym::No,
                initialized_ndim: 0,
                initialized_nnz: 0,
                effective_strategy: -1,
                effective_ordering: -1,
                effective_scaling: -1,
                rcond_estimate: 0.0,
                determinant_coefficient_real: 0.0,
                determinant_coefficient_imag: 0.0,
                determinant_exponent: 0.0,
                stopwatch: Stopwatch::new(),
                time_initialize_ns: 0,
                time_factorize_ns: 0,
                time_solve_ns: 0,
            })
        }
    }
}

impl ComplexLinSolTrait for ComplexSolverUMFPACK {
    /// Performs the factorization (and analysis/initialization if needed)
    ///
    /// # Input
    ///
    /// * `mat` -- the coefficient matrix A. The matrix must be square (`nrow = ncol`) and,
    ///   if symmetric, the symmetric flag must be [Sym::YesFull]
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
    /// 4. For symmetric matrices, `UMFPACK` requires [Sym::YesFull].
    fn factorize(&mut self, mat: &ComplexCooMatrix, params: Option<LinSolParams>) -> Result<(), StrError> {
        // convert from COO to CSC
        if self.initialized {
            if mat.symmetric != self.initialized_sym {
                return Err("subsequent factorizations must use the same matrix (symmetric differs)");
            }
            if mat.nrow != self.initialized_ndim {
                return Err("subsequent factorizations must use the same matrix (ndim differs)");
            }
            if mat.nnz != self.initialized_nnz {
                return Err("subsequent factorizations must use the same matrix (nnz differs)");
            }
            self.csc.as_mut().unwrap().update_from_coo(mat)?;
        } else {
            if mat.nrow != mat.ncol {
                return Err("the matrix must be square");
            }
            if mat.nnz < 1 {
                return Err("the COO matrix must have at least one non-zero value");
            }
            if mat.symmetric == Sym::YesLower || mat.symmetric == Sym::YesUpper {
                return Err("UMFPACK requires Sym::YesFull for symmetric matrices");
            }
            self.initialized_sym = mat.symmetric;
            self.initialized_ndim = mat.nrow;
            self.initialized_nnz = mat.nnz;
            self.csc = Some(ComplexCscMatrix::from_coo(mat)?);
        }
        let csc = self.csc.as_ref().unwrap();

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
                let status = complex_solver_umfpack_initialize(
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
            let status = complex_solver_umfpack_factorize(
                self.solver,
                &mut self.effective_strategy,
                &mut self.effective_ordering,
                &mut self.effective_scaling,
                &mut self.rcond_estimate,
                &mut self.determinant_coefficient_real,
                &mut self.determinant_coefficient_imag,
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
    /// * `mat` -- the coefficient matrix A; it must be square and, if symmetric, [Sym::YesFull].
    /// * `rhs` -- the right-hand side vector with know values an dimension equal to mat.nrow
    /// * `verbose` -- shows messages
    ///
    /// **Warning:** the matrix must be same one used in `factorize`.
    fn solve(&mut self, x: &mut ComplexVector, rhs: &ComplexVector, verbose: bool) -> Result<(), StrError> {
        // check
        if !self.factorized {
            return Err("the function factorize must be called before solve");
        }

        // access CSC matrix
        let csc = self.csc.as_ref().unwrap();

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
            let status = complex_solver_umfpack_solve(
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
        stats.main.solver = if cfg!(feature = "local_suitesparse") {
            "UMFPACK-local".to_string()
        } else {
            "UMFPACK".to_string()
        };
        stats.determinant.mantissa_real = self.determinant_coefficient_real;
        stats.determinant.mantissa_imag = self.determinant_coefficient_imag;
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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ComplexCooMatrix, Ordering, Samples, Scaling};
    use russell_lab::{complex_approx_eq, complex_vec_approx_eq, cpx};

    #[test]
    fn new_and_drop_work() {
        // you may debug into the C-code to see that drop is working
        let solver = ComplexSolverUMFPACK::new().unwrap();
        assert!(!solver.factorized);
    }

    #[test]
    fn factorize_handles_errors() {
        let mut solver = ComplexSolverUMFPACK::new().unwrap();
        assert!(!solver.factorized);

        // initial validation
        let (coo, _, _, _) = Samples::complex_rectangular_4x3();
        assert_eq!(solver.factorize(&coo, None).err(), Some("the matrix must be square"));
        let coo = ComplexCooMatrix::new(1, 1, 1, Sym::No).unwrap();
        assert_eq!(
            solver.factorize(&coo, None).err(),
            Some("the COO matrix must have at least one non-zero value")
        );
        let (coo, _, _, _) = Samples::complex_symmetric_3x3_lower();
        assert_eq!(
            solver.factorize(&coo, None).err(),
            Some("UMFPACK requires Sym::YesFull for symmetric matrices")
        );

        // check already factorized data
        let mut coo = ComplexCooMatrix::new(2, 2, 2, Sym::No).unwrap();
        coo.put(0, 0, cpx!(1.0, 0.0)).unwrap();
        coo.put(1, 1, cpx!(2.0, 0.0)).unwrap();
        // ... factorize once => OK
        solver.factorize(&coo, None).unwrap();
        // ... change matrix (symmetric)
        let mut coo = ComplexCooMatrix::new(2, 2, 2, Sym::YesFull).unwrap();
        coo.put(0, 0, cpx!(1.0, 0.0)).unwrap();
        coo.put(1, 1, cpx!(2.0, 0.0)).unwrap();
        assert_eq!(
            solver.factorize(&coo, None).err(),
            Some("subsequent factorizations must use the same matrix (symmetric differs)")
        );
        // ... change matrix (ndim)
        let mut coo = ComplexCooMatrix::new(1, 1, 1, Sym::No).unwrap();
        coo.put(0, 0, cpx!(1.0, 0.0)).unwrap();
        assert_eq!(
            solver.factorize(&coo, None).err(),
            Some("subsequent factorizations must use the same matrix (ndim differs)")
        );
        // ... change matrix (nnz)
        let mut coo = ComplexCooMatrix::new(2, 2, 1, Sym::No).unwrap();
        coo.put(0, 0, cpx!(1.0, 0.0)).unwrap();
        assert_eq!(
            solver.factorize(&coo, None).err(),
            Some("subsequent factorizations must use the same matrix (nnz differs)")
        );
    }

    #[test]
    fn factorize_works() {
        let mut solver = ComplexSolverUMFPACK::new().unwrap();
        assert!(!solver.factorized);
        let (coo, _, _, _) = Samples::complex_symmetric_3x3_full();
        let mut params = LinSolParams::new();

        params.compute_determinant = true;
        params.ordering = Ordering::Amd;
        params.scaling = Scaling::Sum;

        solver.factorize(&coo, Some(params)).unwrap();
        assert!(solver.factorized);

        assert_eq!(solver.effective_ordering, UMFPACK_ORDERING_AMD);
        assert_eq!(solver.effective_scaling, UMFPACK_SCALE_SUM);

        let m = cpx!(solver.determinant_coefficient_real, solver.determinant_coefficient_imag);
        let det = m * f64::powf(10.0, solver.determinant_exponent);
        complex_approx_eq(det, cpx!(6.0, 10.0), 1e-14);

        // calling factorize again works
        solver.factorize(&coo, Some(params)).unwrap();
        let m = cpx!(solver.determinant_coefficient_real, solver.determinant_coefficient_imag);
        let det = m * f64::powf(10.0, solver.determinant_exponent);
        complex_approx_eq(det, cpx!(6.0, 10.0), 1e-14);
    }

    #[test]
    fn factorize_fails_on_singular_matrix() {
        let mut solver = ComplexSolverUMFPACK::new().unwrap();
        let mut coo = ComplexCooMatrix::new(2, 2, 2, Sym::No).unwrap();
        coo.put(0, 0, cpx!(1.0, 0.0)).unwrap();
        coo.put(1, 1, cpx!(0.0, 0.0)).unwrap();
        assert_eq!(solver.factorize(&coo, None), Err("Error(1): Matrix is singular"));
    }

    #[test]
    fn solve_handles_errors() {
        let mut coo = ComplexCooMatrix::new(2, 2, 2, Sym::No).unwrap();
        coo.put(0, 0, cpx!(123.0, 1.0)).unwrap();
        coo.put(1, 1, cpx!(456.0, 2.0)).unwrap();
        let mut solver = ComplexSolverUMFPACK::new().unwrap();
        assert!(!solver.factorized);
        let mut x = ComplexVector::new(2);
        let rhs = ComplexVector::new(2);
        assert_eq!(
            solver.solve(&mut x, &rhs, false),
            Err("the function factorize must be called before solve")
        );
        let mut x = ComplexVector::new(1);
        solver.factorize(&coo, None).unwrap();
        assert_eq!(
            solver.solve(&mut x, &rhs, false),
            Err("the dimension of the vector of unknown values x is incorrect")
        );
        let mut x = ComplexVector::new(2);
        let rhs = ComplexVector::new(1);
        assert_eq!(
            solver.solve(&mut x, &rhs, false),
            Err("the dimension of the right-hand side vector is incorrect")
        );
    }

    #[test]
    fn solve_works() {
        let mut solver = ComplexSolverUMFPACK::new().unwrap();
        let (coo, _, _, _) = Samples::complex_symmetric_3x3_full();
        let mut x = ComplexVector::new(3);
        let rhs = ComplexVector::from(&[cpx!(-3.0, 3.0), cpx!(2.0, -2.0), cpx!(9.0, 7.0)]);
        let x_correct = &[cpx!(1.0, 1.0), cpx!(2.0, -2.0), cpx!(3.0, 3.0)];
        solver.factorize(&coo, None).unwrap();
        solver.solve(&mut x, &rhs, false).unwrap();
        complex_vec_approx_eq(&x, x_correct, 1e-14);

        // calling solve again works
        let mut x_again = ComplexVector::new(3);
        solver.solve(&mut x_again, &rhs, false).unwrap();
        complex_vec_approx_eq(&x_again, x_correct, 1e-14);

        // update stats
        let mut stats = StatsLinSol::new();
        solver.update_stats(&mut stats);
        assert_eq!(stats.output.effective_ordering, "Amd");
        assert_eq!(stats.output.effective_scaling, "Sum");
    }
}
