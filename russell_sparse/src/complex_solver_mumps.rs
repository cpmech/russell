use super::{handle_mumps_error_code, mumps_ordering, mumps_scaling};
use super::{ComplexLinSolTrait, ComplexSparseMatrix, LinSolParams, StatsLinSol, Symmetry};
use super::{
    MUMPS_ORDERING_AMD, MUMPS_ORDERING_AMF, MUMPS_ORDERING_AUTO, MUMPS_ORDERING_METIS, MUMPS_ORDERING_PORD,
    MUMPS_ORDERING_QAMD, MUMPS_ORDERING_SCOTCH, MUMPS_SCALING_AUTO, MUMPS_SCALING_COLUMN, MUMPS_SCALING_DIAGONAL,
    MUMPS_SCALING_NO, MUMPS_SCALING_ROW_COL, MUMPS_SCALING_ROW_COL_ITER, MUMPS_SCALING_ROW_COL_RIG,
};
use crate::auxiliary_and_constants::*;
use crate::StrError;
use num_complex::Complex64;
use russell_lab::{complex_vec_copy, using_intel_mkl, ComplexVector, Stopwatch};

/// Opaque struct holding a C-pointer to InterfaceComplexMUMPS
///
/// Reference: <https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs>
#[repr(C)]
struct InterfaceComplexMUMPS {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

extern "C" {
    fn complex_solver_mumps_new() -> *mut InterfaceComplexMUMPS;
    fn complex_solver_mumps_drop(solver: *mut InterfaceComplexMUMPS);
    fn complex_solver_mumps_initialize(
        solver: *mut InterfaceComplexMUMPS,
        ordering: i32,
        scaling: i32,
        pct_inc_workspace: i32,
        max_work_memory: i32,
        openmp_num_threads: i32,
        verbose: CcBool,
        general_symmetric: CcBool,
        positive_definite: CcBool,
        ndim: i32,
        nnz: i32,
        indices_i: *const i32,
        indices_j: *const i32,
        values_aij: *const Complex64,
    ) -> i32;
    fn complex_solver_mumps_factorize(
        solver: *mut InterfaceComplexMUMPS,
        effective_ordering: *mut i32,
        effective_scaling: *mut i32,
        determinant_coefficient_real: *mut f64,
        determinant_coefficient_imag: *mut f64,
        determinant_exponent: *mut f64,
        compute_determinant: CcBool,
        verbose: CcBool,
    ) -> i32;
    fn complex_solver_mumps_solve(
        solver: *mut InterfaceComplexMUMPS,
        rhs: *mut Complex64,
        error_analysis_array_len_8: *mut f64,
        error_analysis_option: i32,
        verbose: CcBool,
    ) -> i32;
}

/// Wraps the MUMPS solver for (very large) sparse linear systems
///
/// **Warning:** This solver is **not** thread-safe, thus use only use in single-thread applications.
pub struct ComplexSolverMUMPS {
    /// Holds a pointer to the C interface to MUMPS
    solver: *mut InterfaceComplexMUMPS,

    /// Indicates whether the solver has been initialized or not (just once)
    initialized: bool,

    /// Indicates whether the sparse matrix has been factorized or not
    factorized: bool,

    /// Holds the symmetry type used in the initialize
    initialized_symmetry: Symmetry,

    /// Holds the matrix dimension saved in initialize
    initialized_ndim: usize,

    /// Holds the number of non-zeros saved in initialize
    initialized_nnz: usize,

    /// Holds the used ordering (after factorize)
    effective_ordering: i32,

    /// Holds the used scaling (after factorize)
    effective_scaling: i32,

    /// Holds the OpenMP number of threads passed down to MUMPS (ICNTL(16))
    effective_num_threads: i32,

    /// Holds the determinant coefficient (if requested) (real part)
    ///
    /// det = coefficient * pow(2, exponent)
    determinant_coefficient_real: f64,

    /// Holds the determinant coefficient (if requested) (imaginary part)
    ///
    /// det = coefficient * pow(2, exponent)
    determinant_coefficient_imag: f64,

    /// Holds the determinant exponent (if requested)
    ///
    /// det = coefficient * pow(2, exponent)
    determinant_exponent: f64,

    /// MUMPS code for error analysis (after solve)
    ///
    /// ICNTL(11): 0 (nothing), 1 (all; slow), 2 (just errors)
    error_analysis_option: i32,

    /// Holds the error analysis "stat" results
    error_analysis_array_len_8: Vec<f64>,

    /// Stopwatch to measure computation times
    stopwatch: Stopwatch,

    /// Time spent on initialize in nanoseconds
    time_initialize_ns: u128,

    /// Time spent on factorize in nanoseconds
    time_factorize_ns: u128,

    /// Time spent on solve in nanoseconds
    time_solve_ns: u128,
}

impl Drop for ComplexSolverMUMPS {
    /// Tells the c-code to release memory
    fn drop(&mut self) {
        unsafe {
            complex_solver_mumps_drop(self.solver);
        }
    }
}

impl ComplexSolverMUMPS {
    /// Allocates a new instance
    pub fn new() -> Result<Self, StrError> {
        unsafe {
            let solver = complex_solver_mumps_new();
            if solver.is_null() {
                return Err("c-code failed to allocate the MUMPS solver");
            }
            Ok(ComplexSolverMUMPS {
                solver,
                initialized: false,
                factorized: false,
                initialized_symmetry: Symmetry::No,
                initialized_ndim: 0,
                initialized_nnz: 0,
                effective_ordering: -1,
                effective_scaling: -1,
                effective_num_threads: 0,
                determinant_coefficient_real: 0.0,
                determinant_coefficient_imag: 0.0,
                determinant_exponent: 0.0,
                error_analysis_option: 0,
                error_analysis_array_len_8: vec![0.0; 8],
                stopwatch: Stopwatch::new(),
                time_initialize_ns: 0,
                time_factorize_ns: 0,
                time_solve_ns: 0,
            })
        }
    }
}

impl ComplexLinSolTrait for ComplexSolverMUMPS {
    /// Performs the factorization (and analysis/initialization if needed)
    ///
    /// # Input
    ///
    /// * `mat` -- the coefficient matrix A (one-base **COO** only, not CSC and not CSR).
    ///   Also, the matrix must be square (`nrow = ncol`) and, if symmetric,
    ///   the symmetry/storage must [crate::Storage::Lower].
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
    /// 4. For symmetric matrices, `MUMPS` requires that the symmetry/storage be [crate::Storage::Lower].
    /// 5. The COO matrix must be one-based.
    fn factorize(&mut self, mat: &mut ComplexSparseMatrix, params: Option<LinSolParams>) -> Result<(), StrError> {
        // get COO matrix
        let coo = mat.get_coo()?;

        // check the COO matrix
        if !coo.one_based {
            return Err("the COO matrix must have one-based (FORTRAN) indices as required by MUMPS");
        }
        if coo.nrow != coo.ncol {
            return Err("the COO matrix must be square");
        }
        if coo.nnz < 1 {
            return Err("the COO matrix must have at least one non-zero value");
        }

        // check already initialized data
        if self.initialized {
            if coo.symmetry != self.initialized_symmetry {
                return Err("subsequent factorizations must use the same matrix (symmetry differs)");
            }
            if coo.nrow != self.initialized_ndim {
                return Err("subsequent factorizations must use the same matrix (ndim differs)");
            }
            if coo.nnz != self.initialized_nnz {
                return Err("subsequent factorizations must use the same matrix (nnz differs)");
            }
        } else {
            self.initialized_symmetry = coo.symmetry;
            self.initialized_ndim = coo.nrow;
            self.initialized_nnz = coo.nnz;
        }

        // configuration parameters
        let par = if let Some(p) = params { p } else { LinSolParams::new() };

        // error analysis option
        self.error_analysis_option = if par.compute_condition_numbers {
            1 // all the statistics (very expensive) (page 40)
        } else if par.compute_error_estimates {
            2 // main statistics are computed (page 40)
        } else {
            0 // nothing
        };

        // input parameters
        let ordering = mumps_ordering(par.ordering);
        let scaling = mumps_scaling(par.scaling);
        let pct_inc_workspace = to_i32(par.mumps_pct_inc_workspace);
        let max_work_memory = to_i32(par.mumps_max_work_memory);
        self.effective_num_threads =
            if using_intel_mkl() || par.mumps_num_threads != 0 || par.mumps_override_prevent_nt_issue_with_openblas {
                to_i32(par.mumps_num_threads)
            } else {
                1 // avoid bug with OpenBLAS
            };

        // requests
        let compute_determinant = if par.compute_determinant { 1 } else { 0 };
        let verbose = if par.verbose { 1 } else { 0 };

        // extract the symmetry flags and check the storage type
        let (general_symmetric, positive_definite) = coo.symmetry.status(true, false)?;

        // matrix config
        let ndim = to_i32(coo.nrow);
        let nnz = to_i32(coo.nnz);

        // call initialize just once
        if !self.initialized {
            self.stopwatch.reset();
            unsafe {
                let status = complex_solver_mumps_initialize(
                    self.solver,
                    ordering,
                    scaling,
                    pct_inc_workspace,
                    max_work_memory,
                    self.effective_num_threads,
                    verbose,
                    general_symmetric,
                    positive_definite,
                    ndim,
                    nnz,
                    coo.indices_i.as_ptr(),
                    coo.indices_j.as_ptr(),
                    coo.values.as_ptr(),
                );
                if status != SUCCESSFUL_EXIT {
                    return Err(handle_mumps_error_code(status));
                }
            }
            self.time_initialize_ns = self.stopwatch.stop();
            self.initialized = true;
        }

        // call factorize
        self.stopwatch.reset();
        unsafe {
            let status = complex_solver_mumps_factorize(
                self.solver,
                &mut self.effective_ordering,
                &mut self.effective_scaling,
                &mut self.determinant_coefficient_real,
                &mut self.determinant_coefficient_imag,
                &mut self.determinant_exponent,
                compute_determinant,
                verbose,
            );
            if status != SUCCESSFUL_EXIT {
                return Err(handle_mumps_error_code(status));
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
    /// * `mat` -- the coefficient matrix A; must be square and, if symmetric, [crate::Storage::Lower].
    /// * `rhs` -- the right-hand side vector with know values an dimension equal to mat.nrow
    /// * `verbose` -- shows messages
    ///
    /// **Warning:** the matrix must be same one used in `factorize`.
    fn solve(
        &mut self,
        x: &mut ComplexVector,
        mat: &ComplexSparseMatrix,
        rhs: &ComplexVector,
        verbose: bool,
    ) -> Result<(), StrError> {
        // check
        if !self.factorized {
            return Err("the function factorize must be called before solve");
        }

        // access COO matrix
        let coo = mat.get_coo()?;

        // check already factorized data
        let (nrow, ncol, nnz, symmetry) = coo.get_info();
        if symmetry != self.initialized_symmetry {
            return Err("solve must use the same matrix (symmetry differs)");
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

        // call MUMPS solve
        complex_vec_copy(x, rhs).unwrap();
        let verb = if verbose { 1 } else { 0 };
        self.stopwatch.reset();
        unsafe {
            let status = complex_solver_mumps_solve(
                self.solver,
                x.as_mut_data().as_mut_ptr(),
                self.error_analysis_array_len_8.as_mut_ptr(),
                self.error_analysis_option,
                verb,
            );
            if status != SUCCESSFUL_EXIT {
                return Err(handle_mumps_error_code(status));
            }
        }
        self.time_solve_ns = self.stopwatch.stop();

        // done
        Ok(())
    }

    /// Updates the stats structure (should be called after solve)
    fn update_stats(&self, stats: &mut StatsLinSol) {
        stats.main.solver = if cfg!(local_mumps) {
            "MUMPS-local".to_string()
        } else {
            "MUMPS".to_string()
        };
        stats.determinant.mantissa_real = self.determinant_coefficient_real;
        stats.determinant.mantissa_imag = self.determinant_coefficient_imag;
        stats.determinant.base = 2.0;
        stats.determinant.exponent = self.determinant_exponent;
        stats.output.effective_ordering = match self.effective_ordering {
            MUMPS_ORDERING_AMD => "Amd".to_string(),
            MUMPS_ORDERING_AMF => "Amf".to_string(),
            MUMPS_ORDERING_AUTO => "Auto".to_string(),
            MUMPS_ORDERING_METIS => "Metis".to_string(),
            MUMPS_ORDERING_PORD => "Pord".to_string(),
            MUMPS_ORDERING_QAMD => "Qamd".to_string(),
            MUMPS_ORDERING_SCOTCH => "Scotch".to_string(),
            _ => "Unknown".to_string(),
        };
        stats.output.effective_scaling = match self.effective_scaling {
            MUMPS_SCALING_AUTO => "Auto".to_string(),
            MUMPS_SCALING_COLUMN => "Column".to_string(),
            MUMPS_SCALING_DIAGONAL => "Diagonal".to_string(),
            MUMPS_SCALING_NO => "No".to_string(),
            MUMPS_SCALING_ROW_COL => "RowCol".to_string(),
            MUMPS_SCALING_ROW_COL_ITER => "RowColIter".to_string(),
            MUMPS_SCALING_ROW_COL_RIG => "RowColRig".to_string(),
            -2 => "Scaling done during analysis".to_string(),
            _ => "Unknown".to_string(),
        };
        stats.output.effective_mumps_num_threads = self.effective_num_threads as usize;
        stats.mumps_stats.inf_norm_a = self.error_analysis_array_len_8[0];
        stats.mumps_stats.inf_norm_x = self.error_analysis_array_len_8[1];
        stats.mumps_stats.scaled_residual = self.error_analysis_array_len_8[2];
        stats.mumps_stats.backward_error_omega1 = self.error_analysis_array_len_8[3];
        stats.mumps_stats.backward_error_omega2 = self.error_analysis_array_len_8[4];
        stats.mumps_stats.normalized_delta_x = self.error_analysis_array_len_8[5];
        stats.mumps_stats.condition_number1 = self.error_analysis_array_len_8[6];
        stats.mumps_stats.condition_number2 = self.error_analysis_array_len_8[7];
        stats.time_nanoseconds.initialize = self.time_initialize_ns;
        stats.time_nanoseconds.factorize = self.time_factorize_ns;
        stats.time_nanoseconds.solve = self.time_solve_ns;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ComplexCooMatrix, ComplexSparseMatrix, LinSolParams, Ordering, Samples, Scaling, Storage, Symmetry};
    use num_complex::Complex64;
    use russell_lab::{complex_approx_eq, complex_vec_approx_eq, cpx, ComplexVector};
    use serial_test::serial;

    #[test]
    #[serial]
    fn complete_solution_cycle_works() {
        // IMPORTANT:
        // Since MUMPS is not thread-safe, we need to call all MUMPS functions
        // in a single test unit because the tests are run in parallel by default

        // allocate x and rhs
        let mut x = ComplexVector::new(3);
        let rhs = ComplexVector::from(&[cpx!(-3.0, 3.0), cpx!(2.0, -2.0), cpx!(9.0, 7.0)]);
        let x_correct = &[cpx!(1.0, 1.0), cpx!(2.0, -2.0), cpx!(3.0, 3.0)];

        // allocate a new solver
        let mut solver = ComplexSolverMUMPS::new().unwrap();
        assert!(!solver.factorized);

        // get COO matrix errors
        let (_, csc, _, _) = Samples::complex_symmetric_3x3_full();
        let mut mat = ComplexSparseMatrix::from_csc(csc);
        assert_eq!(
            solver.factorize(&mut mat, None).err(),
            Some("COO matrix is not available")
        );

        // check COO matrix
        let coo = ComplexCooMatrix::new(1, 1, 1, None, false).unwrap();
        let mut mat = ComplexSparseMatrix::from_coo(coo);
        assert_eq!(
            solver.factorize(&mut mat, None).err(),
            Some("the COO matrix must have one-based (FORTRAN) indices as required by MUMPS")
        );
        let mut coo = ComplexCooMatrix::new(2, 1, 1, None, true).unwrap();
        coo.put(0, 0, cpx!(4.0, 4.0)).unwrap();
        let mut mat = ComplexSparseMatrix::from_coo(coo);
        assert_eq!(
            solver.factorize(&mut mat, None).err(),
            Some("the COO matrix must be square")
        );
        let coo = ComplexCooMatrix::new(1, 1, 1, None, true).unwrap();
        let mut mat = ComplexSparseMatrix::from_coo(coo);
        assert_eq!(
            solver.factorize(&mut mat, None).err(),
            Some("the COO matrix must have at least one non-zero value")
        );
        let mut coo = ComplexCooMatrix::new(1, 1, 1, Some(Symmetry::General(Storage::Full)), true).unwrap();
        coo.put(0, 0, cpx!(4.0, 4.0)).unwrap();
        let mut mat = ComplexSparseMatrix::from_coo(coo);
        assert_eq!(
            solver.factorize(&mut mat, None).err(),
            Some("if the matrix is general symmetric, the required storage is lower triangular")
        );

        // check already factorized data
        let mut coo = ComplexCooMatrix::new(2, 2, 2, None, true).unwrap();
        coo.put(0, 0, cpx!(1.0, 0.0)).unwrap();
        coo.put(1, 1, cpx!(2.0, 0.0)).unwrap();
        let mut mat = ComplexSparseMatrix::from_coo(coo);
        // ... factorize once => OK
        solver.factorize(&mut mat, None).unwrap();
        // ... change matrix (symmetry)
        let mut coo = ComplexCooMatrix::new(2, 2, 2, Some(Symmetry::General(Storage::Full)), true).unwrap();
        coo.put(0, 0, cpx!(1.0, 0.0)).unwrap();
        coo.put(1, 1, cpx!(2.0, 0.0)).unwrap();
        let mut mat = ComplexSparseMatrix::from_coo(coo);
        assert_eq!(
            solver.factorize(&mut mat, None).err(),
            Some("subsequent factorizations must use the same matrix (symmetry differs)")
        );
        // ... change matrix (ndim)
        let mut coo = ComplexCooMatrix::new(1, 1, 1, None, true).unwrap();
        coo.put(0, 0, cpx!(1.0, 0.0)).unwrap();
        let mut mat = ComplexSparseMatrix::from_coo(coo);
        assert_eq!(
            solver.factorize(&mut mat, None).err(),
            Some("subsequent factorizations must use the same matrix (ndim differs)")
        );
        // ... change matrix (nnz)
        let mut coo = ComplexCooMatrix::new(2, 2, 1, None, true).unwrap();
        coo.put(0, 0, cpx!(1.0, 0.0)).unwrap();
        let mut mat = ComplexSparseMatrix::from_coo(coo);
        assert_eq!(
            solver.factorize(&mut mat, None).err(),
            Some("subsequent factorizations must use the same matrix (nnz differs)")
        );

        // allocate a new solver
        let mut solver = ComplexSolverMUMPS::new().unwrap();
        assert!(!solver.factorized);

        // sample matrix
        let (coo, _, _, _) = Samples::complex_symmetric_3x3_lower(true);
        let mut mat = ComplexSparseMatrix::from_coo(coo);

        // set params
        let mut params = LinSolParams::new();
        params.ordering = Ordering::Pord;
        params.scaling = Scaling::RowCol;
        params.compute_determinant = true;

        // solve fails on non-factorized system
        assert_eq!(
            solver.solve(&mut x, &mat, &rhs, false),
            Err("the function factorize must be called before solve")
        );

        // factorize works
        solver.factorize(&mut mat, Some(params)).unwrap();
        assert!(solver.factorized);

        // solve fails on wrong x and rhs vectors
        let mut x_wrong = ComplexVector::new(5);
        let rhs_wrong = ComplexVector::new(2);
        assert_eq!(
            solver.solve(&mut x_wrong, &mat, &rhs, false),
            Err("the dimension of the vector of unknown values x is incorrect")
        );
        assert_eq!(
            solver.solve(&mut x, &mat, &rhs_wrong, false),
            Err("the dimension of the right-hand side vector is incorrect")
        );

        // solve works
        solver.solve(&mut x, &mat, &rhs, false).unwrap();
        complex_vec_approx_eq(x.as_data(), x_correct, 1e-14);

        // check ordering and scaling
        assert_eq!(solver.effective_ordering, 4); // Pord
        assert_eq!(solver.effective_scaling, 0); // No, because we requested the determinant

        // check the determinant
        let m = cpx!(solver.determinant_coefficient_real, solver.determinant_coefficient_imag);
        let det = m * f64::powf(2.0, solver.determinant_exponent);
        complex_approx_eq(det, cpx!(6.0, 10.0), 1e-14);

        // update stats
        let mut stats = StatsLinSol::new();
        solver.update_stats(&mut stats);
        assert_eq!(stats.output.effective_ordering, "Pord");
        assert_eq!(stats.output.effective_scaling, "No");

        // calling solve again works
        let mut x_again = ComplexVector::new(3);
        solver.solve(&mut x_again, &mat, &rhs, false).unwrap();
        complex_vec_approx_eq(x_again.as_data(), x_correct, 1e-14);

        // factorize fails on singular matrix
        let mut mat_singular = ComplexSparseMatrix::new_coo(3, 3, 2, None, true).unwrap();
        mat_singular.put(0, 0, cpx!(1.0, 0.0)).unwrap();
        mat_singular.put(1, 1, cpx!(1.0, 0.0)).unwrap();
        let mut solver = ComplexSolverMUMPS::new().unwrap();
        assert_eq!(
            solver.factorize(&mut mat_singular, None),
            Err("Error(-10): numerically singular matrix")
        );

        // solve with positive-definite matrix works
        let sym = Some(Symmetry::PositiveDefinite(Storage::Lower));
        let nrow = 5;
        let ncol = 5;
        let mut coo_pd_lower = ComplexCooMatrix::new(nrow, ncol, 9, sym, true).unwrap();
        coo_pd_lower.put(0, 0, cpx!(9.0, 0.0)).unwrap();
        coo_pd_lower.put(1, 1, cpx!(0.5, 0.0)).unwrap();
        coo_pd_lower.put(2, 2, cpx!(12.0, 0.0)).unwrap();
        coo_pd_lower.put(3, 3, cpx!(0.625, 0.0)).unwrap();
        coo_pd_lower.put(4, 4, cpx!(16.0, 0.0)).unwrap();
        coo_pd_lower.put(1, 0, cpx!(1.5, 0.0)).unwrap();
        coo_pd_lower.put(2, 0, cpx!(6.0, 0.0)).unwrap();
        coo_pd_lower.put(3, 0, cpx!(0.75, 0.0)).unwrap();
        coo_pd_lower.put(4, 0, cpx!(3.0, 0.0)).unwrap();
        let mut mat_pd_lower = ComplexSparseMatrix::from_coo(coo_pd_lower);
        params.ordering = Ordering::Auto;
        params.scaling = Scaling::Auto;
        let mut solver = ComplexSolverMUMPS::new().unwrap();
        assert!(!solver.factorized);
        solver.factorize(&mut mat_pd_lower, Some(params)).unwrap();
        let mut x = ComplexVector::new(5);
        let rhs = ComplexVector::from(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        solver.solve(&mut x, &mat_pd_lower, &rhs, false).unwrap();
        let x_correct = &[
            cpx!(-979.0 / 3.0, 0.0),
            cpx!(983.0, 0.0),
            cpx!(1961.0 / 12.0, 0.0),
            cpx!(398.0, 0.0),
            cpx!(123.0 / 2.0, 0.0),
        ];
        complex_vec_approx_eq(x.as_data(), x_correct, 1e-13);

        // solve with different matrix fails
        assert_eq!(
            solver.solve(&mut x, &mat, &rhs, false).err(),
            Some("solve must use the same matrix (symmetry differs)")
        );
    }
}
