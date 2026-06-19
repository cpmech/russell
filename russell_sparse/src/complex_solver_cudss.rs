use super::{ComplexCooMatrix, ComplexCsrMatrix, ComplexLinSolTrait, LinSolParams, StatsLinSol, Sym};
use crate::StrError;
use crate::constants::*;
use crate::{
    CUDSS_MATCHING_ALG_AUTO, CUDSS_MATCHING_ALG_MAX_DIAG_COUNT, CUDSS_MATCHING_ALG_MAX_DIAG_PRODUCT,
    CUDSS_MATCHING_ALG_MAX_DIAG_SUM, CUDSS_MATCHING_ALG_MAX_MIN_DIAG, CUDSS_MATCHING_ALG_MAX_MIN_DIAG_ALT,
    CUDSS_MATCHING_ALG_NONE, CUDSS_PIVOT_AUTO, CUDSS_PIVOT_DIAGONAL, CUDSS_PIVOT_GLOBAL_COL, CUDSS_PIVOT_GLOBAL_ROW,
    CUDSS_PIVOT_LOCAL_BLOCK, CUDSS_PIVOT_NONE,
};
use crate::{cudss_matching, cudss_ordering, cudss_pivoting, handle_cudss_error_code};
use russell_lab::{Complex64, ComplexVector, Stopwatch};

/// Opaque struct holding a C-pointer to InterfaceComplexCUDSS
///
/// Reference: <https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs>
#[repr(C)]
struct InterfaceComplexCUDSS {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

/// Enforce Send on the C structure
///
/// <https://stackoverflow.com/questions/50258359/can-a-struct-containing-a-raw-pointer-implement-send-and-be-ffi-safe>
unsafe impl Send for InterfaceComplexCUDSS {}

/// Enforce Send on the Rust structure
///
/// <https://stackoverflow.com/questions/50258359/can-a-struct-containing-a-raw-pointer-implement-send-and-be-ffi-safe>
unsafe impl Send for ComplexSolverCUDSS {}

unsafe extern "C" {
    fn complex_solver_cudss_new() -> *mut InterfaceComplexCUDSS;
    fn complex_solver_cudss_drop(solver: *mut InterfaceComplexCUDSS);
    fn complex_solver_cudss_initialize(
        solver: *mut InterfaceComplexCUDSS,
        ordering: i32,
        matching: i32,
        pivoting: i32,
        pivot_epsilon: f64,
        refinement_nstep: i32,
        hybrid_memory: i32,
        verbose: CcBool,
        general_symmetric: CcBool,
        positive_definite: CcBool,
        ndim: i32,
        row_pointers: *const i32,
        col_indices: *const i32,
        values: *const Complex64,
    ) -> i32;
    fn complex_solver_cudss_factorize(
        solver: *mut InterfaceComplexCUDSS,
        effective_matching: *mut i32,
        effective_pivoting: *mut i32,
        verbose: CcBool,
        values: *const Complex64,
    ) -> i32;
    fn complex_solver_cudss_solve(
        solver: *mut InterfaceComplexCUDSS,
        x: *mut Complex64,
        rhs: *const Complex64,
        verbose: CcBool,
    ) -> i32;
}

/// Wraps the cuDSS solver for sparse linear systems
///
/// # Memory leak reports with Valgrind
///
/// Running `valgrind` (or `cargo valgrind`) will report memory leaks originating
/// from NVIDIA's CUDA and cuDSS libraries (`cuInit`, `cuDevicePrimaryCtxRetain`,
/// `cuLaunchKernel`, `cuLibraryLoadData`, `cudaStreamCreate`). Empirically, these
/// are allocations made by CUDA during device/library initialization that persist
/// for the lifetime of the process and are not freed until exit. No leaks originate
/// from our code (`interface_cudss.cu`); all allocations are properly freed in
/// `complex_solver_cudss_drop`.
///
/// # Singularity detection
///
/// cuDSS returns both **host-side** and **device-side** errors:
///
/// * **Host-side errors** are returned by `cudssExecute()` via `cudssStatus_t`
///   and are checked directly. If non-zero, the corresponding error code is returned.
///
/// * **Device-side errors** (e.g., non-positive minor in an SPD matrix) are only
///   available asynchronously via `cudssDataGet(handle, data, CUDSS_DATA_INFO, ...)`
///   after synchronizing the CUDA stream. These are checked after each phase.
///
/// **Important:** cuDSS does **not** report structural or numerical singularity as
/// an error. Instead, it replaces small diagonal entries (pivots smaller than
/// `pivot_epsilon`) with an appropriately signed epsilon value and continues the
/// factorization. The factorization therefore "succeeds" on singular matrices by
/// perturbing them slightly. To detect that pivoting occurred, query
/// `CUDSS_DATA_NPIVOTS` after factorization.
///
/// See: <https://docs.nvidia.com/cuda/cudss/types.html#cudssdataparam-t>
pub struct ComplexSolverCUDSS {
    /// Holds a pointer to the C interface to cuDSS
    solver: *mut InterfaceComplexCUDSS,

    /// Holds the CSR matrix used in factorize
    csr: Option<ComplexCsrMatrix>,

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

    /// Holds the used matching algorithm (retrieved after factorize)
    effective_matching: i32,

    /// Holds the used pivoting strategy (retrieved after factorize)
    effective_pivoting: i32,

    /// Stopwatch to measure computation times
    stopwatch: Stopwatch,

    /// Time spent on initialize in nanoseconds
    time_initialize_ns: u128,

    /// Time spent on factorize in nanoseconds
    time_factorize_ns: u128,

    /// Time spent on solve in nanoseconds
    time_solve_ns: u128,
}

impl Drop for ComplexSolverCUDSS {
    /// Tells the c-code to release memory
    fn drop(&mut self) {
        unsafe {
            complex_solver_cudss_drop(self.solver);
        }
    }
}

impl ComplexSolverCUDSS {
    /// Allocates a new instance
    pub fn new() -> Result<Self, StrError> {
        unsafe {
            let solver = complex_solver_cudss_new();
            if solver.is_null() {
                return Err("c-code failed to allocate the cuDSS solver");
            }
            Ok(ComplexSolverCUDSS {
                solver,
                csr: None,
                initialized: false,
                factorized: false,
                initialized_sym: Sym::No,
                initialized_ndim: 0,
                initialized_nnz: 0,
                effective_matching: 0,
                effective_pivoting: 0,
                stopwatch: Stopwatch::new(),
                time_initialize_ns: 0,
                time_factorize_ns: 0,
                time_solve_ns: 0,
            })
        }
    }
}

impl ComplexLinSolTrait for ComplexSolverCUDSS {
    /// Performs the factorization (and analysis/initialization if needed)
    ///
    /// # Input
    ///
    /// * `mat` -- the coefficient matrix A. The matrix must be square (`nrow = ncol`) and,
    ///   if symmetric, the symmetric flag must be [Sym::YesLower]
    /// * `params` -- configuration parameters; None => use default
    ///
    /// **Important:** `params` must be set to `None` when calling `factorize` again;
    /// e.g., when the values of the coefficient matrix change but the structure remains the same.
    /// This limitation is required because the first factorization performs both the *symbolic* and
    /// *numeric* factorizations, whereas the subsequent factorizations are only *numeric*. Thus,
    /// options such as *ordering* and *scaling* have no further impact after the symbolic factorization.
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
    /// 4. For symmetric matrices, `cuDSS` requires [Sym::YesLower]
    fn factorize(&mut self, mat: &ComplexCooMatrix, params: Option<LinSolParams>) -> Result<(), StrError> {
        // convert from COO to CSR
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
            if params.is_some() {
                return Err("subsequent factorizations must not change LinSolParams");
            }
            self.csr.as_mut().unwrap().update_from_coo(mat)?;
        } else {
            if mat.nrow != mat.ncol {
                return Err("the matrix must be square");
            }
            if mat.nnz < 1 {
                return Err("the COO matrix must have at least one non-zero value");
            }
            if mat.symmetric == Sym::YesFull || mat.symmetric == Sym::YesUpper {
                return Err("cuDSS requires Sym::YesLower for symmetric matrices");
            }
            self.initialized_sym = mat.symmetric;
            self.initialized_ndim = mat.nrow;
            self.initialized_nnz = mat.nnz;
            self.csr = Some(ComplexCsrMatrix::from_coo(mat)?);
        }
        let csr = self.csr.as_ref().unwrap();

        // parameters
        let par = if let Some(p) = params { p } else { LinSolParams::new() };

        // input parameters
        let ordering = cudss_ordering(par.ordering);
        let matching = cudss_matching(par.matching);
        let pivoting = cudss_pivoting(par.pivoting);

        // pivoting parameters
        let pivot_epsilon = match par.pivot_epsilon {
            Some(val) => val,
            None => -1.0, // tell cuDSS to use the default
        };
        let refinement_nstep = match par.refinement_nstep {
            Some(val) => val,
            None => -1, // tell cuDSS to use the default
        };

        // hybrid memory flag
        let hybrid_memory = if par.hybrid_memory { 1 } else { 0 };

        // requests
        let verbose = if par.verbose { 1 } else { 0 };

        // matrix config
        let general_symmetric = if mat.symmetric == Sym::YesLower { 1 } else { 0 };
        let positive_definite = if par.positive_definite { 1 } else { 0 };
        let ndim = to_i32(csr.nrow);

        // call initialize just once
        if !self.initialized {
            self.stopwatch.reset();
            unsafe {
                let status = complex_solver_cudss_initialize(
                    self.solver,
                    ordering,
                    matching,
                    pivoting,
                    pivot_epsilon,
                    refinement_nstep,
                    hybrid_memory,
                    verbose,
                    general_symmetric,
                    positive_definite,
                    ndim,
                    csr.row_pointers.as_ptr(),
                    csr.col_indices.as_ptr(),
                    csr.values.as_ptr(),
                );
                if status != SUCCESSFUL_EXIT {
                    return Err(handle_cudss_error_code(status));
                }
            }
            self.time_initialize_ns = self.stopwatch.stop();
            self.initialized = true;
        }

        // call factorize
        self.stopwatch.reset();
        unsafe {
            let status = complex_solver_cudss_factorize(
                self.solver,
                &mut self.effective_matching,
                &mut self.effective_pivoting,
                verbose,
                csr.values.as_ptr(),
            );
            if status != SUCCESSFUL_EXIT {
                return Err(handle_cudss_error_code(status));
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
    /// * `mat` -- the coefficient matrix A; it must be square and, if symmetric, [Sym::YesLower].
    /// * `rhs` -- the right-hand side vector with know values an dimension equal to mat.nrow
    /// * `verbose` -- shows messages
    ///
    /// **Warning:** the matrix must be same one used in `factorize`.
    fn solve(&mut self, x: &mut ComplexVector, rhs: &ComplexVector, verbose: bool) -> Result<(), StrError> {
        // check
        if !self.factorized {
            return Err("the function factorize must be called before solve");
        }

        // check vectors
        if x.dim() != self.initialized_ndim {
            return Err("the dimension of the vector of unknown values x is incorrect");
        }
        if rhs.dim() != self.initialized_ndim {
            return Err("the dimension of the right-hand side vector is incorrect");
        }

        // call cuDSS solve
        let verb = if verbose { 1 } else { 0 };
        self.stopwatch.reset();
        unsafe {
            let status =
                complex_solver_cudss_solve(self.solver, x.as_mut_data().as_mut_ptr(), rhs.as_data().as_ptr(), verb);
            if status != SUCCESSFUL_EXIT {
                return Err(handle_cudss_error_code(status));
            }
        }
        self.time_solve_ns = self.stopwatch.stop();

        // done
        Ok(())
    }

    /// Updates the stats structure (should be called after solve)
    fn update_stats(&self, stats: &mut StatsLinSol) {
        stats.main.solver = "cuDSS".to_string();
        stats.time_nanoseconds.initialize_array.push(self.time_initialize_ns);
        stats.time_nanoseconds.factorize_array.push(self.time_factorize_ns);
        stats.time_nanoseconds.solve_array.push(self.time_solve_ns);

        // set the retrieved effective (used) matching algorithm
        stats.output.effective_matching = match self.effective_matching {
            CUDSS_MATCHING_ALG_NONE => "None".to_string(),
            CUDSS_MATCHING_ALG_AUTO => "Auto".to_string(),
            CUDSS_MATCHING_ALG_MAX_DIAG_COUNT => "MaxDiagCount".to_string(),
            CUDSS_MATCHING_ALG_MAX_MIN_DIAG => "MaxMinDiag".to_string(),
            CUDSS_MATCHING_ALG_MAX_MIN_DIAG_ALT => "MaxMinDiagAlt".to_string(),
            CUDSS_MATCHING_ALG_MAX_DIAG_SUM => "MaxDiagSum".to_string(),
            CUDSS_MATCHING_ALG_MAX_DIAG_PRODUCT => "MaxDiagProduct".to_string(),
            _ => "Unknown".to_string(),
        };

        // set the retrieved effective (used) pivoting strategy
        stats.output.effective_pivoting = match self.effective_pivoting {
            CUDSS_PIVOT_AUTO => "Auto".to_string(),
            CUDSS_PIVOT_NONE => "None".to_string(),
            CUDSS_PIVOT_GLOBAL_COL => "GlobalCol".to_string(),
            CUDSS_PIVOT_GLOBAL_ROW => "GlobalRow".to_string(),
            CUDSS_PIVOT_DIAGONAL => "Diagonal".to_string(),
            CUDSS_PIVOT_LOCAL_BLOCK => "LocalBlock".to_string(),
            _ => "Unknown".to_string(),
        };
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
    use crate::{ComplexCooMatrix, Matching, Ordering, Samples};
    use russell_lab::{complex_approx_eq, complex_vec_approx_eq, cpx};

    /*
    We're not using "serial" here because:
    - Each ComplexSolverCUDSS gets its own CUDA stream and cuDSS handle
    - cuDSS isolates device memory per cudssData_t/cudssHandle_t
    - Test matrices are tiny (5×5) — no memory pressure
    - Other CUDA libraries (cuBLAS, cuSOLVER) handle concurrency fine
    */

    #[test]
    fn factorize_handles_errors() {
        // allocate a new solver
        let mut solver = ComplexSolverCUDSS::new().unwrap();
        assert!(!solver.factorized);

        // check COO matrix
        let (coo, _, _, _) = Samples::complex_rectangular_4x3();
        assert_eq!(solver.factorize(&coo, None).err(), Some("the matrix must be square"));
        let coo = ComplexCooMatrix::new(1, 1, 1, Sym::No).unwrap();
        assert_eq!(
            solver.factorize(&coo, None).err(),
            Some("the COO matrix must have at least one non-zero value")
        );
        let (coo, _, _, _) = Samples::complex_symmetric_3x3_full();
        assert_eq!(
            solver.factorize(&coo, None).err(),
            Some("cuDSS requires Sym::YesLower for symmetric matrices")
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
    fn factorize_and_solve_work_unsymmetric_default() {
        // allocate x and rhs
        let mut x = ComplexVector::new(5);
        let rhs = ComplexVector::from(&[8.0, 45.0, -3.0, 3.0, 19.0]);
        let x_correct = ComplexVector::from(&[
            cpx!(1.0, 0.0),
            cpx!(2.0, 0.0),
            cpx!(3.0, 0.0),
            cpx!(4.0, 0.0),
            cpx!(5.0, 0.0),
        ]);

        // allocate a new solver
        let mut solver = ComplexSolverCUDSS::new().unwrap();
        assert!(!solver.factorized);

        // sample matrix
        let (coo, _, _, _) = Samples::umfpack_complex_unsymmetric_5x5();

        // set params
        let mut params = LinSolParams::new();

        // factorize works
        solver.factorize(&coo, Some(params)).unwrap();
        assert!(solver.factorized);

        // solve works
        solver.solve(&mut x, &rhs, false).unwrap();
        // NOTE: cuDSS loses precision on matrices with zero diagonal entries (rows 1 and 3 here
        //       have a zero on the diagonal). The error at x[3] is ~1.2e-3, so we use 2e-3
        //       tolerance instead of 1e-12. This is a known cuDSS numerical limitation.
        complex_approx_eq(x[0], x_correct[0], 1e-12);
        complex_approx_eq(x[1], x_correct[1], 1e-12);
        complex_approx_eq(x[2], x_correct[2], 1e-12);
        complex_approx_eq(x[3], cpx!(4.001243780749064, 0.0), 1e-15); // << issue: it should be 4.0
        complex_approx_eq(x[4], x_correct[4], 1e-12);

        // update stats
        let mut stats = StatsLinSol::new();
        solver.update_stats(&mut stats);
        assert_eq!(stats.main.solver, "cuDSS");
        assert_eq!(stats.time_nanoseconds.initialize_array.len(), 1);
        assert_eq!(stats.time_nanoseconds.factorize_array.len(), 1);
        assert_eq!(stats.time_nanoseconds.solve_array.len(), 1);

        // calling solve again
        let mut x_again = ComplexVector::new(5);
        solver.solve(&mut x_again, &rhs, false).unwrap();
        complex_approx_eq(x_again[0], x_correct[0], 1e-12);
        complex_approx_eq(x_again[1], x_correct[1], 1e-12);
        complex_approx_eq(x_again[2], x_correct[2], 1e-12);
        complex_approx_eq(x[3], cpx!(4.001243780749064, 0.0), 1e-15); // << issue: it should be 4.0
        complex_approx_eq(x_again[4], x_correct[4], 1e-12);

        // calling factorize/solve again with COLAMD
        // NOTE: we cannot change the ordering method after the symbolic factorization
        params.ordering = Ordering::Colamd;
        assert_eq!(
            solver.factorize(&coo, Some(params)),
            Err("subsequent factorizations must not change LinSolParams")
        );
    }

    #[test]
    fn factorize_and_solve_work_unsymmetric_colamd() {
        // allocate x and rhs
        let mut x = ComplexVector::new(5);
        let rhs = ComplexVector::from(&[8.0, 45.0, -3.0, 3.0, 19.0]);
        let x_correct = ComplexVector::from(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        // allocate a new solver
        let mut solver = ComplexSolverCUDSS::new().unwrap();
        assert!(!solver.factorized);

        // sample matrix
        let (coo, _, _, _) = Samples::umfpack_complex_unsymmetric_5x5();

        // set params
        let mut params = LinSolParams::new();
        params.ordering = Ordering::Colamd;

        // factorize works
        solver.factorize(&coo, Some(params)).unwrap();
        assert!(solver.factorized);

        // solve works
        solver.solve(&mut x, &rhs, false).unwrap();
        complex_vec_approx_eq(&x, &x_correct, 1e-12);

        // calling solve again
        let mut x_again = ComplexVector::new(5);
        solver.solve(&mut x_again, &rhs, false).unwrap();
        complex_vec_approx_eq(&x_again, &x_correct, 1e-12);
    }

    #[test]
    fn factorize_and_solve_work_unsymmetric_with_matching() {
        // allocate x and rhs
        let mut x = ComplexVector::new(5);
        let rhs = ComplexVector::from(&[8.0, 45.0, -3.0, 3.0, 19.0]);
        let x_correct = ComplexVector::from(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        // allocate a new solver
        let mut solver = ComplexSolverCUDSS::new().unwrap();
        assert!(!solver.factorized);

        // sample matrix
        let (coo, _, _, _) = Samples::umfpack_complex_unsymmetric_5x5();

        // set params
        let mut params = LinSolParams::new();
        params.matching = Matching::Auto;

        // factorize works
        solver.factorize(&coo, Some(params)).unwrap();
        assert!(solver.factorized);

        // solve works
        solver.solve(&mut x, &rhs, false).unwrap();
        complex_vec_approx_eq(&x, &x_correct, 1e-12);

        // calling solve again
        let mut x_again = ComplexVector::new(5);
        solver.solve(&mut x_again, &rhs, false).unwrap();
        complex_vec_approx_eq(&x_again, &x_correct, 1e-12);
    }

    #[test]
    fn factorize_and_solve_work_unsymmetric_pivot_params() {
        // allocate x and rhs
        let mut x = ComplexVector::new(5);
        let rhs = ComplexVector::from(&[8.0, 45.0, -3.0, 3.0, 19.0]);
        let x_correct = ComplexVector::from(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        // allocate a new solver
        let mut solver = ComplexSolverCUDSS::new().unwrap();
        assert!(!solver.factorized);

        // sample matrix
        let (coo, _, _, _) = Samples::umfpack_complex_unsymmetric_5x5();

        // set params
        let mut params = LinSolParams::new();
        params.pivot_epsilon = Some(1e-12);
        params.refinement_nstep = Some(1);

        // factorize works
        solver.factorize(&coo, Some(params)).unwrap();
        assert!(solver.factorized);

        // solve works
        solver.solve(&mut x, &rhs, false).unwrap();
        complex_vec_approx_eq(&x, &x_correct, 1e-12);

        // calling solve again
        let mut x_again = ComplexVector::new(5);
        solver.solve(&mut x_again, &rhs, false).unwrap();
        complex_vec_approx_eq(&x_again, &x_correct, 1e-12);
    }

    #[test]
    fn factorize_and_solve_work_sym_psd() {
        // allocate x and rhs
        let mut x = ComplexVector::new(5);
        let rhs = ComplexVector::from(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let x_correct = ComplexVector::from(&[-979.0 / 3.0, 983.0, 1961.0 / 12.0, 398.0, 123.0 / 2.0]);

        // allocate a new solver
        let mut solver = ComplexSolverCUDSS::new().unwrap();
        assert!(!solver.factorized);

        // sample matrix: symmetric positive-definite, lower triangle
        let (coo, _, _, _) = Samples::mkl_complex_positive_definite_5x5_lower();

        // set params
        let mut params = LinSolParams::new();
        params.positive_definite = true;

        // factorize works
        solver.factorize(&coo, Some(params)).unwrap();
        assert!(solver.factorized);

        // solve works
        solver.solve(&mut x, &rhs, false).unwrap();
        complex_vec_approx_eq(&x, &x_correct, 1e-10);
    }

    #[test]
    fn cudss_simple_spd() {
        // Corresponds to c_code/cudss-examples/cudss_simple_complex.cpp
        //
        // Symmetric positive-definite matrix (lower triangle):
        //   [4, 0, 1, 0, 0]
        //   [0, 3, 2, 0, 0]
        //   [1, 2, 5, 0, 1]
        //   [0, 0, 0, 1, 0]
        //   [0, 0, 1, 0, 2]
        let mut coo = ComplexCooMatrix::new(5, 5, 8, Sym::YesLower).unwrap();
        coo.put(0, 0, cpx!(4.0, 0.0)).unwrap();
        coo.put(1, 1, cpx!(3.0, 0.0)).unwrap();
        coo.put(2, 0, cpx!(1.0, 0.0)).unwrap();
        coo.put(2, 1, cpx!(2.0, 0.0)).unwrap();
        coo.put(2, 2, cpx!(5.0, 0.0)).unwrap();
        coo.put(3, 3, cpx!(1.0, 0.0)).unwrap();
        coo.put(4, 2, cpx!(1.0, 0.0)).unwrap();
        coo.put(4, 4, cpx!(2.0, 0.0)).unwrap();

        let mut x = ComplexVector::new(5);
        let rhs = ComplexVector::from(&[7.0, 12.0, 25.0, 4.0, 13.0]);
        let x_correct = ComplexVector::from(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        let mut params = LinSolParams::new();
        params.positive_definite = true;

        let mut solver = ComplexSolverCUDSS::new().unwrap();
        solver.factorize(&coo, Some(params)).unwrap();
        solver.solve(&mut x, &rhs, false).unwrap();
        complex_vec_approx_eq(&x, &x_correct, 1e-10);
    }

    #[test]
    fn cudss_unsymmetric() {
        // Corresponds to c_code/cudss-examples/cudss_complex_unsymmetric.cpp
        //
        // General (unsymmetric) matrix:
        //   [5, 1, 0, 0, 3]
        //   [2, 6, 0, 4, 0]
        //   [0, 0, 7, 2, 0]
        //   [0, 1, 3, 8, 0]
        //   [4, 0, 0, 0, 9]
        let mut coo = ComplexCooMatrix::new(5, 5, 13, Sym::No).unwrap();
        coo.put(0, 0, cpx!(5.0, 0.0)).unwrap();
        coo.put(0, 1, cpx!(1.0, 0.0)).unwrap();
        coo.put(0, 4, cpx!(3.0, 0.0)).unwrap();
        coo.put(1, 0, cpx!(2.0, 0.0)).unwrap();
        coo.put(1, 1, cpx!(6.0, 0.0)).unwrap();
        coo.put(1, 3, cpx!(4.0, 0.0)).unwrap();
        coo.put(2, 2, cpx!(7.0, 0.0)).unwrap();
        coo.put(2, 3, cpx!(2.0, 0.0)).unwrap();
        coo.put(3, 1, cpx!(1.0, 0.0)).unwrap();
        coo.put(3, 2, cpx!(3.0, 0.0)).unwrap();
        coo.put(3, 3, cpx!(8.0, 0.0)).unwrap();
        coo.put(4, 0, cpx!(4.0, 0.0)).unwrap();
        coo.put(4, 4, cpx!(9.0, 0.0)).unwrap();

        let mut x = ComplexVector::new(5);
        let rhs = ComplexVector::from(&[22.0, 30.0, 29.0, 43.0, 49.0]);
        let x_correct = ComplexVector::from(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        let mut solver = ComplexSolverCUDSS::new().unwrap();
        solver.factorize(&coo, None).unwrap();
        solver.solve(&mut x, &rhs, false).unwrap();
        complex_vec_approx_eq(&x, &x_correct, 1e-10);
    }

    #[test]
    fn hybrid_memory_works() {
        // Symmetric positive-definite matrix (lower triangle):
        //   [4, 0, 1, 0, 0]
        //   [0, 3, 2, 0, 0]
        //   [1, 2, 5, 0, 1]
        //   [0, 0, 0, 1, 0]
        //   [0, 0, 1, 0, 2]
        let mut coo = ComplexCooMatrix::new(5, 5, 8, Sym::YesLower).unwrap();
        coo.put(0, 0, cpx!(4.0, 0.0)).unwrap();
        coo.put(1, 1, cpx!(3.0, 0.0)).unwrap();
        coo.put(2, 0, cpx!(1.0, 0.0)).unwrap();
        coo.put(2, 1, cpx!(2.0, 0.0)).unwrap();
        coo.put(2, 2, cpx!(5.0, 0.0)).unwrap();
        coo.put(3, 3, cpx!(1.0, 0.0)).unwrap();
        coo.put(4, 2, cpx!(1.0, 0.0)).unwrap();
        coo.put(4, 4, cpx!(2.0, 0.0)).unwrap();

        let mut x = ComplexVector::new(5);
        let rhs = ComplexVector::from(&[7.0, 12.0, 25.0, 4.0, 13.0]);
        let x_correct = ComplexVector::from(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        let mut params = LinSolParams::new();
        params.positive_definite = true;
        params.hybrid_memory = true;
        params.verbose = false;

        let mut solver = ComplexSolverCUDSS::new().unwrap();
        solver.factorize(&coo, Some(params)).unwrap();
        solver.solve(&mut x, &rhs, false).unwrap();
        complex_vec_approx_eq(&x, &x_correct, 1e-10);
    }

    #[test]
    fn solve_handles_errors() {
        let mut coo = ComplexCooMatrix::new(2, 2, 2, Sym::No).unwrap();
        coo.put(0, 0, cpx!(123.0, 0.0)).unwrap();
        coo.put(1, 1, cpx!(456.0, 0.0)).unwrap();
        let mut solver = ComplexSolverCUDSS::new().unwrap();
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
}
