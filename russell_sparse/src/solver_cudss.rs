#![allow(unused)]

use super::{CooMatrix, CsrMatrix, LinSolParams, LinSolTrait, Ordering, Scaling, StatsLinSol, Sym};
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
        verbose: CcBool,
        general_symmetric: CcBool,
        positive_definite: CcBool,
        ndim: i32,
        row_pointers: *const i32,
        col_indices: *const i32,
        values: *const f64,
    ) -> i32;
    fn solver_cudss_factorize(
        solver: *mut InterfaceCUDSS,
        verbose: CcBool,
        row_pointers: *const i32,
        col_indices: *const i32,
        values: *const f64,
    ) -> i32;
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
            let status = solver_cudss_factorize(
                self.solver,
                verbose,
                csr.row_pointers.as_ptr(),
                csr.col_indices.as_ptr(),
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

/// Handles error code
pub(crate) fn handle_cudss_error_code(err: i32) -> StrError {
    match err {
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
    fn factorize_and_solve_work() {
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
        vec_approx_eq(&x, x_correct, 1e-12);

        // update stats
        let mut stats = StatsLinSol::new();
        solver.update_stats(&mut stats);
        // TODO: check

        // calling solve again works
        // let mut x_again = Vector::new(5);
        // solver.solve(&mut x_again, &rhs, false).unwrap();
        // vec_approx_eq(&x_again, x_correct, 1e-14);
    }
}
