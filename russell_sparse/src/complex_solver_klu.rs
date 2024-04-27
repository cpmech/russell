use super::{handle_klu_error_code, klu_ordering, klu_scaling};
use super::{ComplexLinSolTrait, ComplexSparseMatrix, LinSolParams, StatsLinSol, Sym};
use super::{KLU_ORDERING_AMD, KLU_ORDERING_COLAMD, KLU_SCALE_MAX, KLU_SCALE_NONE, KLU_SCALE_SUM};
use crate::auxiliary_and_constants::*;
use crate::StrError;
use russell_lab::{complex_vec_copy, Complex64, ComplexVector, Stopwatch};

/// Opaque struct holding a C-pointer to InterfaceComplexKLU
///
/// Reference: <https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs>
#[repr(C)]
struct InterfaceComplexKLU {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

/// Enforce Send on the C structure
///
/// <https://stackoverflow.com/questions/50258359/can-a-struct-containing-a-raw-pointer-implement-send-and-be-ffi-safe>
unsafe impl Send for InterfaceComplexKLU {}

/// Enforce Send on the Rust structure
///
/// <https://stackoverflow.com/questions/50258359/can-a-struct-containing-a-raw-pointer-implement-send-and-be-ffi-safe>
unsafe impl Send for ComplexSolverKLU {}

extern "C" {
    fn complex_solver_klu_new() -> *mut InterfaceComplexKLU;
    fn complex_solver_klu_drop(solver: *mut InterfaceComplexKLU);
    fn complex_solver_klu_initialize(
        solver: *mut InterfaceComplexKLU,
        ordering: i32,
        scaling: i32,
        ndim: i32,
        col_pointers: *const i32,
        row_indices: *const i32,
    ) -> i32;
    fn complex_solver_klu_factorize(
        solver: *mut InterfaceComplexKLU,
        effective_ordering: *mut i32,
        effective_scaling: *mut i32,
        cond_estimate: *mut f64,
        compute_cond: CcBool,
        col_pointers: *const i32,
        row_indices: *const i32,
        values: *const Complex64,
    ) -> i32;
    fn complex_solver_klu_solve(solver: *mut InterfaceComplexKLU, ndim: i32, in_rhs_out_x: *mut Complex64) -> i32;
}

/// Wraps the KLU solver for sparse linear systems
///
/// **Warning:** This solver may "run out of memory" for very large matrices.
pub struct ComplexSolverKLU {
    /// Holds a pointer to the C interface to KLU
    solver: *mut InterfaceComplexKLU,

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

impl Drop for ComplexSolverKLU {
    /// Tells the c-code to release memory
    fn drop(&mut self) {
        unsafe {
            complex_solver_klu_drop(self.solver);
        }
    }
}

impl ComplexSolverKLU {
    /// Allocates a new instance
    pub fn new() -> Result<Self, StrError> {
        unsafe {
            let solver = complex_solver_klu_new();
            if solver.is_null() {
                return Err("c-code failed to allocate the KLU solver");
            }
            Ok(ComplexSolverKLU {
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

impl ComplexLinSolTrait for ComplexSolverKLU {
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
    fn factorize(&mut self, mat: &mut ComplexSparseMatrix, params: Option<LinSolParams>) -> Result<(), StrError> {
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
                let status = complex_solver_klu_initialize(
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
            let status = complex_solver_klu_factorize(
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
    fn solve(
        &mut self,
        x: &mut ComplexVector,
        mat: &ComplexSparseMatrix,
        rhs: &ComplexVector,
        _verbose: bool,
    ) -> Result<(), StrError> {
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
        complex_vec_copy(x, rhs).unwrap();
        self.stopwatch.reset();
        unsafe {
            let status = complex_solver_klu_solve(self.solver, ndim, x.as_mut_data().as_mut_ptr());
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
        stats.main.solver = if cfg!(local_umfpack) {
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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ComplexCooMatrix, Ordering, Samples, Scaling};
    use russell_lab::{complex_vec_approx_eq, cpx};

    #[test]
    fn new_and_drop_work() {
        // you may debug into the C-code to see that drop is working
        let solver = ComplexSolverKLU::new().unwrap();
        assert!(!solver.factorized);
    }

    #[test]
    fn factorize_handles_errors() {
        let mut solver = ComplexSolverKLU::new().unwrap();
        assert!(!solver.factorized);

        // COO to CSC errors
        let coo = ComplexCooMatrix::new(1, 1, 1, Sym::No).unwrap();
        let mut mat = ComplexSparseMatrix::from_coo(coo);
        assert_eq!(
            solver.factorize(&mut mat, None).err(),
            Some("COO to CSC requires nnz > 0")
        );

        // check CSC matrix
        let (coo, _, _, _) = Samples::complex_rectangular_4x3();
        let mut mat = ComplexSparseMatrix::from_coo(coo);
        assert_eq!(
            solver.factorize(&mut mat, None).err(),
            Some("the matrix must be square")
        );
        let (coo, _, _, _) = Samples::complex_symmetric_3x3_lower();
        let mut mat = ComplexSparseMatrix::from_coo(coo);
        assert_eq!(
            solver.factorize(&mut mat, None).err(),
            Some("KLU requires Sym::YesFull for symmetric matrices")
        );

        // check already factorized data
        let mut coo = ComplexCooMatrix::new(2, 2, 2, Sym::No).unwrap();
        coo.put(0, 0, cpx!(1.0, 0.0)).unwrap();
        coo.put(1, 1, cpx!(2.0, 0.0)).unwrap();
        let mut mat = ComplexSparseMatrix::from_coo(coo);
        // ... factorize once => OK
        solver.factorize(&mut mat, None).unwrap();
        // ... change matrix (symmetric)
        let mut coo = ComplexCooMatrix::new(2, 2, 2, Sym::YesFull).unwrap();
        coo.put(0, 0, cpx!(1.0, 0.0)).unwrap();
        coo.put(1, 1, cpx!(2.0, 0.0)).unwrap();
        let mut mat = ComplexSparseMatrix::from_coo(coo);
        assert_eq!(
            solver.factorize(&mut mat, None).err(),
            Some("subsequent factorizations must use the same matrix (symmetric differs)")
        );
        // ... change matrix (ndim)
        let mut coo = ComplexCooMatrix::new(1, 1, 1, Sym::No).unwrap();
        coo.put(0, 0, cpx!(1.0, 0.0)).unwrap();
        let mut mat = ComplexSparseMatrix::from_coo(coo);
        assert_eq!(
            solver.factorize(&mut mat, None).err(),
            Some("subsequent factorizations must use the same matrix (ndim differs)")
        );
        // ... change matrix (nnz)
        let mut coo = ComplexCooMatrix::new(2, 2, 1, Sym::No).unwrap();
        coo.put(0, 0, cpx!(1.0, 0.0)).unwrap();
        let mut mat = ComplexSparseMatrix::from_coo(coo);
        assert_eq!(
            solver.factorize(&mut mat, None).err(),
            Some("subsequent factorizations must use the same matrix (nnz differs)")
        );
    }

    #[test]
    fn factorize_works() {
        let mut solver = ComplexSolverKLU::new().unwrap();
        assert!(!solver.factorized);
        let (coo, _, _, _) = Samples::complex_symmetric_3x3_full();
        let mut mat = ComplexSparseMatrix::from_coo(coo);

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
        let mut solver = ComplexSolverKLU::new().unwrap();
        let mut coo = ComplexCooMatrix::new(2, 2, 2, Sym::No).unwrap();
        coo.put(0, 0, cpx!(1.0, 0.0)).unwrap();
        coo.put(1, 1, cpx!(0.0, 0.0)).unwrap();
        let mut mat = ComplexSparseMatrix::from_coo(coo);
        assert_eq!(solver.factorize(&mut mat, None), Err("klu_factor failed"));
    }

    #[test]
    fn solve_handles_errors() {
        let mut coo = ComplexCooMatrix::new(2, 2, 2, Sym::No).unwrap();
        coo.put(0, 0, cpx!(123.0, 1.0)).unwrap();
        coo.put(1, 1, cpx!(456.0, 2.0)).unwrap();
        let mut mat = ComplexSparseMatrix::from_coo(coo);
        let mut solver = ComplexSolverKLU::new().unwrap();
        assert!(!solver.factorized);
        let mut x = ComplexVector::new(2);
        let rhs = ComplexVector::new(2);
        assert_eq!(
            solver.solve(&mut x, &mut mat, &rhs, false),
            Err("the function factorize must be called before solve")
        );
        let mut x = ComplexVector::new(1);
        solver.factorize(&mut mat, None).unwrap();
        assert_eq!(
            solver.solve(&mut x, &mut mat, &rhs, false),
            Err("the dimension of the vector of unknown values x is incorrect")
        );
        let mut x = ComplexVector::new(2);
        let rhs = ComplexVector::new(1);
        assert_eq!(
            solver.solve(&mut x, &mut mat, &rhs, false),
            Err("the dimension of the right-hand side vector is incorrect")
        );
        // wrong symmetric
        let rhs = ComplexVector::new(2);
        let mut coo_wrong = ComplexCooMatrix::new(2, 2, 2, Sym::YesFull).unwrap();
        coo_wrong.put(0, 0, cpx!(123.0, 1.0)).unwrap();
        coo_wrong.put(1, 1, cpx!(456.0, 2.0)).unwrap();
        let mut mat_wrong = ComplexSparseMatrix::from_coo(coo_wrong);
        mat_wrong.get_csc_or_from_coo().unwrap(); // make sure to convert to CSC (because we're not calling factorize on this wrong matrix)
        assert_eq!(
            solver.solve(&mut x, &mut mat_wrong, &rhs, false),
            Err("solve must use the same matrix (symmetric differs)")
        );
        // wrong ndim
        let mut coo_wrong = ComplexCooMatrix::new(1, 1, 1, Sym::No).unwrap();
        coo_wrong.put(0, 0, cpx!(123.0, 1.0)).unwrap();
        let mut mat_wrong = ComplexSparseMatrix::from_coo(coo_wrong);
        mat_wrong.get_csc_or_from_coo().unwrap(); // make sure to convert to CSC (because we're not calling factorize on this wrong matrix)
        assert_eq!(
            solver.solve(&mut x, &mut mat_wrong, &rhs, false),
            Err("solve must use the same matrix (ndim differs)")
        );
        // wrong nnz
        let mut coo_wrong = ComplexCooMatrix::new(2, 2, 3, Sym::No).unwrap();
        coo_wrong.put(0, 0, cpx!(123.0, 1.0)).unwrap();
        coo_wrong.put(1, 1, cpx!(456.0, 2.0)).unwrap();
        coo_wrong.put(0, 1, cpx!(100.0, 1.0)).unwrap();
        let mut mat_wrong = ComplexSparseMatrix::from_coo(coo_wrong);
        mat_wrong.get_csc_or_from_coo().unwrap(); // make sure to convert to CSC (because we're not calling factorize on this wrong matrix)
        assert_eq!(
            solver.solve(&mut x, &mut mat_wrong, &rhs, false),
            Err("solve must use the same matrix (nnz differs)")
        );
    }

    #[test]
    fn solve_works() {
        let mut solver = ComplexSolverKLU::new().unwrap();
        let (coo, _, _, _) = Samples::complex_symmetric_3x3_full();
        let mut mat = ComplexSparseMatrix::from_coo(coo);
        let mut x = ComplexVector::new(3);
        let rhs = ComplexVector::from(&[cpx!(-3.0, 3.0), cpx!(2.0, -2.0), cpx!(9.0, 7.0)]);
        let x_correct = &[cpx!(1.0, 1.0), cpx!(2.0, -2.0), cpx!(3.0, 3.0)];

        let mut params = LinSolParams::new();
        params.ordering = Ordering::Cholmod;
        params.scaling = Scaling::Max;

        solver.factorize(&mut mat, Some(params)).unwrap();
        solver.solve(&mut x, &mut mat, &rhs, false).unwrap();
        complex_vec_approx_eq(&x, x_correct, 1e-14);

        // calling solve again works
        let mut x_again = ComplexVector::new(3);
        solver.solve(&mut x_again, &mut mat, &rhs, false).unwrap();
        complex_vec_approx_eq(&x_again, x_correct, 1e-14);

        // update stats
        let mut stats = StatsLinSol::new();
        solver.update_stats(&mut stats);
        assert_eq!(stats.output.effective_ordering, "Amd");
        assert_eq!(stats.output.effective_scaling, "Max");
    }
}
