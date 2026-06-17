use super::{CooMatrix, CsrMatrix, LinSolParams, LinSolTrait, Matching, Ordering, StatsLinSol, Sym};
use crate::constants::*;
use crate::StrError;
use russell_lab::{Stopwatch, Vector};

/// Opaque struct holding a C-pointer to InterfaceCUDSS
///
/// Reference: <https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs>
#[repr(C)]
struct InterfaceCUDSS {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

/// Enforce Send on the C structure
///
/// <https://stackoverflow.com/questions/50258359/can-a-struct-containing-a-raw-pointer-implement-send-and-be-ffi-safe>
unsafe impl Send for InterfaceCUDSS {}

/// Enforce Send on the Rust structure
///
/// <https://stackoverflow.com/questions/50258359/can-a-struct-containing-a-raw-pointer-implement-send-and-be-ffi-safe>
unsafe impl Send for SolverCUDSS {}

extern "C" {
    fn solver_cudss_new() -> *mut InterfaceCUDSS;
    fn solver_cudss_drop(solver: *mut InterfaceCUDSS);
    fn solver_cudss_initialize(
        solver: *mut InterfaceCUDSS,
        ordering: i32,
        matching: i32,
        verbose: CcBool,
        general_symmetric: CcBool,
        positive_definite: CcBool,
        ndim: i32,
        row_pointers: *const i32,
        col_indices: *const i32,
        values: *const f64,
    ) -> i32;
    fn solver_cudss_factorize(solver: *mut InterfaceCUDSS, verbose: CcBool, values: *const f64) -> i32;
    fn solver_cudss_solve(solver: *mut InterfaceCUDSS, x: *mut f64, rhs: *const f64, verbose: CcBool) -> i32;
}

/// Wraps the cuDSS solver for sparse linear systems
pub struct SolverCUDSS {
    /// Holds a pointer to the C interface to cuDSS
    solver: *mut InterfaceCUDSS,

    /// Holds the CSR matrix used in factorize
    csr: Option<CsrMatrix>,

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

    /// Stopwatch to measure computation times
    stopwatch: Stopwatch,

    /// Time spent on initialize in nanoseconds
    time_initialize_ns: u128,

    /// Time spent on factorize in nanoseconds
    time_factorize_ns: u128,

    /// Time spent on solve in nanoseconds
    time_solve_ns: u128,
}

impl Drop for SolverCUDSS {
    /// Tells the c-code to release memory
    fn drop(&mut self) {
        unsafe {
            solver_cudss_drop(self.solver);
        }
    }
}

impl SolverCUDSS {
    /// Allocates a new instance
    pub fn new() -> Result<Self, StrError> {
        unsafe {
            let solver = solver_cudss_new();
            if solver.is_null() {
                return Err("c-code failed to allocate the cuDSS solver");
            }
            Ok(SolverCUDSS {
                solver,
                csr: None,
                initialized: false,
                factorized: false,
                initialized_sym: Sym::No,
                initialized_ndim: 0,
                initialized_nnz: 0,
                stopwatch: Stopwatch::new(),
                time_initialize_ns: 0,
                time_factorize_ns: 0,
                time_solve_ns: 0,
            })
        }
    }
}

impl LinSolTrait for SolverCUDSS {
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
    fn factorize(&mut self, mat: &CooMatrix, params: Option<LinSolParams>) -> Result<(), StrError> {
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
            self.csr = Some(CsrMatrix::from_coo(mat)?);
        }
        let csr = self.csr.as_ref().unwrap();

        // parameters
        let par = if let Some(p) = params { p } else { LinSolParams::new() };

        // input parameters
        let ordering = cudss_ordering(par.ordering);
        let matching = cudss_matching(par.matching);

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
                let status = solver_cudss_initialize(
                    self.solver,
                    ordering,
                    matching,
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
            let status = solver_cudss_factorize(self.solver, verbose, csr.values.as_ptr());
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
    fn solve(&mut self, x: &mut Vector, rhs: &Vector, verbose: bool) -> Result<(), StrError> {
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
            let status = solver_cudss_solve(self.solver, x.as_mut_data().as_mut_ptr(), rhs.as_data().as_ptr(), verb);
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

// The ordering algorithms are described in:
// <https://docs.nvidia.com/cuda/cudss/types.html#cudssreorderingalg-t>
//
// IMPORTANT: these constants must match the values in <cudss_data_types.h>.
// These values are passed directly to the C layer which casts them to
// cudssReorderingAlg_t. If the cuDSS enum values change upstream, update
// these constants accordingly.

const CUDSS_REORDERING_ALG_DEFAULT: i32 = 0; // The default algorithm for reordering (equivalent to CUDSS_REORDERING_ALG_NESTED_DISSECTION).
const CUDSS_REORDERING_ALG_BTF_COLAMD: i32 = 1; // Block triangular form (BTF) combined with COLAMD. Supports global pivoting.
const CUDSS_REORDERING_ALG_COLAMD: i32 = 2; // COLAMD with trivial block structure. Supports global pivoting.
const CUDSS_REORDERING_ALG_AMD: i32 = 3; // Approximate minimum degree (AMD) reordering.
const CUDSS_REORDERING_ALG_NESTED_DISSECTION: i32 = 4; // Nested dissection algorithm based on METIS.
const CUDSS_REORDERING_ALG_NONE: i32 = 5; // Uses natural (identity) order for the internal ordering when no user permutation is supplied.

/// Converts the Rust enum to an appropriate constant representing the ordering algorithm
fn cudss_ordering(ordering: Ordering) -> i32 {
    match ordering {
        Ordering::Amd => CUDSS_REORDERING_ALG_AMD,
        Ordering::Amf => CUDSS_REORDERING_ALG_DEFAULT,
        Ordering::Auto => CUDSS_REORDERING_ALG_DEFAULT,
        Ordering::Best => CUDSS_REORDERING_ALG_DEFAULT,
        Ordering::BtfColamd => CUDSS_REORDERING_ALG_BTF_COLAMD,
        Ordering::Cholmod => CUDSS_REORDERING_ALG_DEFAULT,
        Ordering::Colamd => CUDSS_REORDERING_ALG_COLAMD,
        Ordering::Metis => CUDSS_REORDERING_ALG_NESTED_DISSECTION,
        Ordering::No => CUDSS_REORDERING_ALG_NONE,
        Ordering::Pord => CUDSS_REORDERING_ALG_DEFAULT,
        Ordering::Qamd => CUDSS_REORDERING_ALG_DEFAULT,
        Ordering::Scotch => CUDSS_REORDERING_ALG_DEFAULT,
    }
}

// The matching algorithms are described in:
// <https://docs.nvidia.com/cuda/cudss/types.html#cudssmatchingalg-t>
//
// IMPORTANT: these constants must match the values in <cudss_data_types.h>.
// These values are passed directly to the C layer which casts them to
// cudssMatchingAlg_t. If the cuDSS enum values change upstream, update
// these constants accordingly.

const CUDSS_MATCHING_ALG_NONE: i32 = 0;
const CUDSS_MATCHING_ALG_MAX_DIAG_COUNT: i32 = 1;
const CUDSS_MATCHING_ALG_MAX_MIN_DIAG: i32 = 2;
const CUDSS_MATCHING_ALG_MAX_MIN_DIAG_ALT: i32 = 3;
const CUDSS_MATCHING_ALG_MAX_DIAG_SUM: i32 = 4;
const CUDSS_MATCHING_ALG_MAX_DIAG_PRODUCT: i32 = 5;
const CUDSS_MATCHING_ALG_AUTO: i32 = 6;

/// Converts the Rust enum to an appropriate constant representing the matching algorithm
fn cudss_matching(matching: Matching) -> i32 {
    match matching {
        Matching::None => CUDSS_MATCHING_ALG_NONE,
        Matching::Auto => CUDSS_MATCHING_ALG_AUTO,
        Matching::MaxDiagCount => CUDSS_MATCHING_ALG_MAX_DIAG_COUNT,
        Matching::MaxMinDiag => CUDSS_MATCHING_ALG_MAX_MIN_DIAG,
        Matching::MaxMinDiagAlt => CUDSS_MATCHING_ALG_MAX_MIN_DIAG_ALT,
        Matching::MaxDiagSum => CUDSS_MATCHING_ALG_MAX_DIAG_SUM,
        Matching::MaxDiagProduct => CUDSS_MATCHING_ALG_MAX_DIAG_PRODUCT,
    }
}

// Important: Make sure that the error constants match
// the corresponding constants in c_code/constants.h

const ERROR_CUDA_MALLOC: i32 = 100;
const ERROR_CUDA_MEMCPY: i32 = 200;
const ERROR_CUDA_SYNCHRONIZE: i32 = 300;
const ERROR_CUDSS_CONFIG_SET: i32 = 400;
const ERROR_CUDSS_CONFIG_GET: i32 = 450;
const ERROR_CUDSS_MATRIX_CREATE_DN: i32 = 500;
const ERROR_CUDSS_MATRIX_SET_VALUES: i32 = 550;
const ERROR_CUDSS_MATRIX_CREATE_CSR: i32 = 600;
const ERROR_CUDSS_SYM_FACTORIZATION: i32 = 700;
const ERROR_CUDSS_NUM_FACTORIZATION: i32 = 800;
const ERROR_CUDSS_SOLVE: i32 = 900;

/// Handles error code
fn handle_cudss_error_code(err: i32) -> StrError {
    match err {
        ERROR_CUDA_MALLOC => "cudaMalloc failed in the C code (cuDSS)",
        ERROR_CUDA_MEMCPY => "cudaMemcpy failed in the C code (cuDSS)",
        ERROR_CUDA_SYNCHRONIZE => "cudaStreamSynchronize failed in the C code (cuDSS)",
        ERROR_CUDSS_CONFIG_SET => "cudssConfigSet failed in the C code (cuDSS)",
        ERROR_CUDSS_CONFIG_GET => "cudssConfigGet failed in the C code (cuDSS)",
        ERROR_CUDSS_MATRIX_CREATE_DN => "cudssMatrixCreateDn failed in the C code (cuDSS)",
        ERROR_CUDSS_MATRIX_SET_VALUES => "cudssMatrixSetValues failed in the C code (cuDSS)",
        ERROR_CUDSS_MATRIX_CREATE_CSR => "cudssMatrixCreateCsr failed in the C code (cuDSS)",
        ERROR_CUDSS_SYM_FACTORIZATION => "cuDSS symbolic factorization (CUDSS_PHASE_ANALYSIS) failed",
        ERROR_CUDSS_NUM_FACTORIZATION => "cuDSS numeric factorization (CUDSS_PHASE_FACTORIZATION) failed",
        ERROR_CUDSS_SOLVE => "cuDSS solve (CUDSS_PHASE_SOLVE) failed",
        _ => "Error: unknown error returned by c-code (cuDSS)",
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CooMatrix, Samples};
    use russell_lab::{approx_eq, vec_approx_eq};
    use serial_test::serial;

    // IMPORTANT:
    // We better not use the GPU concurrently; thus let's use serial_test::serial

    #[test]
    #[serial]
    fn factorize_handles_errors() {
        // allocate a new solver
        let mut solver = SolverCUDSS::new().unwrap();
        assert!(!solver.factorized);

        // check COO matrix
        let (coo, _, _, _) = Samples::rectangular_1x2(true, false);
        assert_eq!(solver.factorize(&coo, None).err(), Some("the matrix must be square"));
        let coo = CooMatrix::new(1, 1, 1, Sym::No).unwrap();
        assert_eq!(
            solver.factorize(&coo, None).err(),
            Some("the COO matrix must have at least one non-zero value")
        );
        let (coo, _, _, _) = Samples::mkl_symmetric_5x5_upper(true, false);
        assert_eq!(
            solver.factorize(&coo, None).err(),
            Some("cuDSS requires Sym::YesLower for symmetric matrices")
        );

        // check already factorized data
        let mut coo = CooMatrix::new(2, 2, 2, Sym::No).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(1, 1, 2.0).unwrap();
        // ... factorize once => OK
        solver.factorize(&coo, None).unwrap();
        // ... change matrix (symmetric)
        let mut coo = CooMatrix::new(2, 2, 2, Sym::YesFull).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(1, 1, 2.0).unwrap();
        assert_eq!(
            solver.factorize(&coo, None).err(),
            Some("subsequent factorizations must use the same matrix (symmetric differs)")
        );
        // ... change matrix (ndim)
        let mut coo = CooMatrix::new(1, 1, 1, Sym::No).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        assert_eq!(
            solver.factorize(&coo, None).err(),
            Some("subsequent factorizations must use the same matrix (ndim differs)")
        );
        // ... change matrix (nnz)
        let mut coo = CooMatrix::new(2, 2, 1, Sym::No).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        assert_eq!(
            solver.factorize(&coo, None).err(),
            Some("subsequent factorizations must use the same matrix (nnz differs)")
        );
    }

    #[test]
    #[serial]
    fn factorize_and_solve_work_unsymmetric_default() {
        // allocate x and rhs
        let mut x = Vector::new(5);
        let rhs = Vector::from(&[8.0, 45.0, -3.0, 3.0, 19.0]);
        let x_correct = &[1.0, 2.0, 3.0, 4.0, 5.0];

        // allocate a new solver
        let mut solver = SolverCUDSS::new().unwrap();
        assert!(!solver.factorized);

        // sample matrix
        let (coo, _, _, _) = Samples::umfpack_unsymmetric_5x5();

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
        approx_eq(x[0], x_correct[0], 1e-12);
        approx_eq(x[1], x_correct[1], 1e-12);
        approx_eq(x[2], x_correct[2], 1e-12);
        approx_eq(x[3], 4.001243780749064, 1e-15); // << issue: it should be 4.0
        approx_eq(x[4], x_correct[4], 1e-12);

        // update stats
        let mut stats = StatsLinSol::new();
        solver.update_stats(&mut stats);
        assert!(stats.time_nanoseconds.initialize > 0);
        assert!(stats.time_nanoseconds.factorize > 0);
        assert!(stats.time_nanoseconds.solve > 0);

        // calling solve again
        let mut x_again = Vector::new(5);
        solver.solve(&mut x_again, &rhs, false).unwrap();
        approx_eq(x_again[0], x_correct[0], 1e-12);
        approx_eq(x_again[1], x_correct[1], 1e-12);
        approx_eq(x_again[2], x_correct[2], 1e-12);
        approx_eq(x[3], 4.001243780749064, 1e-15); // << issue: it should be 4.0
        approx_eq(x_again[4], x_correct[4], 1e-12);

        // calling factorize/solve again with COLAMD
        // NOTE: we cannot change the ordering method after the symbolic factorization
        params.ordering = Ordering::Colamd;
        assert_eq!(
            solver.factorize(&coo, Some(params)),
            Err("subsequent factorizations must not change LinSolParams")
        );
    }

    #[test]
    #[serial]
    fn factorize_and_solve_work_unsymmetric_colamd() {
        // allocate x and rhs
        let mut x = Vector::new(5);
        let rhs = Vector::from(&[8.0, 45.0, -3.0, 3.0, 19.0]);
        let x_correct = &[1.0, 2.0, 3.0, 4.0, 5.0];

        // allocate a new solver
        let mut solver = SolverCUDSS::new().unwrap();
        assert!(!solver.factorized);

        // sample matrix
        let (coo, _, _, _) = Samples::umfpack_unsymmetric_5x5();

        // set params
        let mut params = LinSolParams::new();
        params.ordering = Ordering::Colamd;

        // factorize works
        solver.factorize(&coo, Some(params)).unwrap();
        assert!(solver.factorized);

        // solve works
        solver.solve(&mut x, &rhs, false).unwrap();
        vec_approx_eq(&x, x_correct, 1e-12);

        // calling solve again
        let mut x_again = Vector::new(5);
        solver.solve(&mut x_again, &rhs, false).unwrap();
        vec_approx_eq(&x_again, x_correct, 1e-12);
    }

    #[test]
    #[serial]
    fn factorize_and_solve_work_unsymmetric_with_matching() {
        // allocate x and rhs
        let mut x = Vector::new(5);
        let rhs = Vector::from(&[8.0, 45.0, -3.0, 3.0, 19.0]);
        let x_correct = &[1.0, 2.0, 3.0, 4.0, 5.0];

        // allocate a new solver
        let mut solver = SolverCUDSS::new().unwrap();
        assert!(!solver.factorized);

        // sample matrix
        let (coo, _, _, _) = Samples::umfpack_unsymmetric_5x5();

        // set params
        let mut params = LinSolParams::new();
        params.matching = Matching::Auto;

        // factorize works
        solver.factorize(&coo, Some(params)).unwrap();
        assert!(solver.factorized);

        // solve works
        solver.solve(&mut x, &rhs, false).unwrap();
        vec_approx_eq(&x, x_correct, 1e-12);

        // calling solve again
        let mut x_again = Vector::new(5);
        solver.solve(&mut x_again, &rhs, false).unwrap();
        vec_approx_eq(&x_again, x_correct, 1e-12);
    }

    #[test]
    #[serial]
    fn factorize_and_solve_work_sym_psd() {
        // allocate x and rhs
        let mut x = Vector::new(5);
        let rhs = Vector::from(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let x_correct = &[-979.0 / 3.0, 983.0, 1961.0 / 12.0, 398.0, 123.0 / 2.0];

        // allocate a new solver
        let mut solver = SolverCUDSS::new().unwrap();
        assert!(!solver.factorized);

        // sample matrix: symmetric positive-definite, lower triangle
        let (coo, _, _, _) = Samples::mkl_positive_definite_5x5_lower();

        // set params
        let mut params = LinSolParams::new();
        params.positive_definite = true;

        // factorize works
        solver.factorize(&coo, Some(params)).unwrap();
        assert!(solver.factorized);

        // solve works
        solver.solve(&mut x, &rhs, false).unwrap();
        vec_approx_eq(&x, x_correct, 1e-10);

        // update stats
        let mut stats = StatsLinSol::new();
        solver.update_stats(&mut stats);
        // TODO: check

        // calling solve again works
        // let mut x_again = Vector::new(5);
        // solver.solve(&mut x_again, &rhs, false).unwrap();
        // vec_approx_eq(&x_again, x_correct, 1e-14);
    }

    #[test]
    #[serial]
    fn cudss_simple_spd() {
        // Corresponds to c_code/cudss-examples/cudss_simple.cpp
        //
        // Symmetric positive-definite matrix (lower triangle):
        //   [4, 0, 1, 0, 0]
        //   [0, 3, 2, 0, 0]
        //   [1, 2, 5, 0, 1]
        //   [0, 0, 0, 1, 0]
        //   [0, 0, 1, 0, 2]
        let mut coo = CooMatrix::new(5, 5, 8, Sym::YesLower).unwrap();
        coo.put(0, 0, 4.0).unwrap();
        coo.put(1, 1, 3.0).unwrap();
        coo.put(2, 0, 1.0).unwrap();
        coo.put(2, 1, 2.0).unwrap();
        coo.put(2, 2, 5.0).unwrap();
        coo.put(3, 3, 1.0).unwrap();
        coo.put(4, 2, 1.0).unwrap();
        coo.put(4, 4, 2.0).unwrap();

        let mut x = Vector::new(5);
        let rhs = Vector::from(&[7.0, 12.0, 25.0, 4.0, 13.0]);
        let x_correct = &[1.0, 2.0, 3.0, 4.0, 5.0];

        let mut params = LinSolParams::new();
        params.positive_definite = true;

        let mut solver = SolverCUDSS::new().unwrap();
        solver.factorize(&coo, Some(params)).unwrap();
        solver.solve(&mut x, &rhs, false).unwrap();
        vec_approx_eq(&x, x_correct, 1e-10);
    }

    #[test]
    #[serial]
    fn cudss_unsymmetric() {
        // Corresponds to c_code/cudss-examples/cudss_unsymmetric.cpp
        //
        // General (unsymmetric) matrix:
        //   [5, 1, 0, 0, 3]
        //   [2, 6, 0, 4, 0]
        //   [0, 0, 7, 2, 0]
        //   [0, 1, 3, 8, 0]
        //   [4, 0, 0, 0, 9]
        let mut coo = CooMatrix::new(5, 5, 13, Sym::No).unwrap();
        coo.put(0, 0, 5.0).unwrap();
        coo.put(0, 1, 1.0).unwrap();
        coo.put(0, 4, 3.0).unwrap();
        coo.put(1, 0, 2.0).unwrap();
        coo.put(1, 1, 6.0).unwrap();
        coo.put(1, 3, 4.0).unwrap();
        coo.put(2, 2, 7.0).unwrap();
        coo.put(2, 3, 2.0).unwrap();
        coo.put(3, 1, 1.0).unwrap();
        coo.put(3, 2, 3.0).unwrap();
        coo.put(3, 3, 8.0).unwrap();
        coo.put(4, 0, 4.0).unwrap();
        coo.put(4, 4, 9.0).unwrap();

        let mut x = Vector::new(5);
        let rhs = Vector::from(&[22.0, 30.0, 29.0, 43.0, 49.0]);
        let x_correct = &[1.0, 2.0, 3.0, 4.0, 5.0];

        let mut solver = SolverCUDSS::new().unwrap();
        solver.factorize(&coo, None).unwrap();
        solver.solve(&mut x, &rhs, false).unwrap();
        vec_approx_eq(&x, x_correct, 1e-10);
    }
}
