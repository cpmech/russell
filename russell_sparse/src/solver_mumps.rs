use super::{LinSolParams, LinSolTrait, Ordering, Scaling, SparseMatrix, StatsLinSol, Symmetry};
use crate::auxiliary_and_constants::*;
use crate::StrError;
use russell_lab::{using_intel_mkl, vec_copy, Stopwatch, Vector};
use serde::{Deserialize, Serialize};

/// Opaque struct holding a C-pointer to InterfaceMUMPS
///
/// Reference: <https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs>
#[repr(C)]
struct InterfaceMUMPS {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

extern "C" {
    fn solver_mumps_new() -> *mut InterfaceMUMPS;
    fn solver_mumps_drop(solver: *mut InterfaceMUMPS);
    fn solver_mumps_initialize(
        solver: *mut InterfaceMUMPS,
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
        values_aij: *const f64,
    ) -> i32;
    fn solver_mumps_factorize(
        solver: *mut InterfaceMUMPS,
        effective_ordering: *mut i32,
        effective_scaling: *mut i32,
        determinant_coefficient: *mut f64,
        determinant_exponent: *mut f64,
        compute_determinant: CcBool,
        verbose: CcBool,
    ) -> i32;
    fn solver_mumps_solve(
        solver: *mut InterfaceMUMPS,
        rhs: *mut f64,
        error_analysis_array_len_8: *mut f64,
        error_analysis_option: i32,
        verbose: CcBool,
    ) -> i32;
}

/// Holds the results of MUMPS error analysis ("stats")
///
/// See page 40 of MUMPS User's guide
///
/// MUMPS: computes the backward errors omega1 and omega2 (page 14):
///
/// ```text
///                                       |b - A · x_bar|ᵢ
/// omega1 = largest_scaled_residual_of ————————————————————
///                                     (|b| + |A| |x_bar|)ᵢ
///
///                                            |b - A · x_bar|ᵢ
/// omega2 = largest_scaled_residual_of ——————————————————————————————————
///                                     (|A| |x_approx|)ᵢ + ‖Aᵢ‖∞ ‖x_bar‖∞
///
/// where x_bar is the actual (approximate) solution returned by the linear solver
/// ```
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StatsLinSolMUMPS {
    /// Holds the infinite norm of the input matrix, RINFOG(4)
    pub inf_norm_a: f64,

    /// Holds the infinite norm of the computed solution, RINFOG(5)
    pub inf_norm_x: f64,

    /// Holds the scaled residual, RINFOG(6)
    pub scaled_residual: f64,

    /// Holds the backward error estimate omega1, RINFOG(7)
    pub backward_error_omega1: f64,

    /// Holds the backward error estimate omega2, RINFOG(8)
    pub backward_error_omega2: f64,

    /// Holds the normalized variation of the solution vector, RINFOG(9)
    ///
    /// Requires the full "stat" analysis.
    pub normalized_delta_x: f64,

    /// Holds the condition number1, RINFOG(10)
    ///
    /// Requires the full "stat" analysis.
    pub condition_number1: f64,

    /// Holds the condition number2, RINFOG(11)
    ///
    /// Requires the full "stat" analysis.
    pub condition_number2: f64,
}

/// Wraps the MUMPS solver for (very large) sparse linear systems
///
/// **Warning:** This solver is **not** thread-safe, thus use only use in single-thread applications.
pub struct SolverMUMPS {
    /// Holds a pointer to the C interface to MUMPS
    solver: *mut InterfaceMUMPS,

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

    /// Holds the determinant coefficient (if requested)
    ///
    /// det = coefficient * pow(2, exponent)
    determinant_coefficient: f64,

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

impl Drop for SolverMUMPS {
    /// Tells the c-code to release memory
    fn drop(&mut self) {
        unsafe {
            solver_mumps_drop(self.solver);
        }
    }
}

impl SolverMUMPS {
    /// Allocates a new instance
    pub fn new() -> Result<Self, StrError> {
        unsafe {
            let solver = solver_mumps_new();
            if solver.is_null() {
                return Err("c-code failed to allocate the MUMPS solver");
            }
            Ok(SolverMUMPS {
                solver,
                initialized: false,
                factorized: false,
                initialized_symmetry: Symmetry::No,
                initialized_ndim: 0,
                initialized_nnz: 0,
                effective_ordering: -1,
                effective_scaling: -1,
                effective_num_threads: 0,
                determinant_coefficient: 0.0,
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

impl LinSolTrait for SolverMUMPS {
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
    fn factorize(&mut self, mat: &mut SparseMatrix, params: Option<LinSolParams>) -> Result<(), StrError> {
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
                let status = solver_mumps_initialize(
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
            let status = solver_mumps_factorize(
                self.solver,
                &mut self.effective_ordering,
                &mut self.effective_scaling,
                &mut self.determinant_coefficient,
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
    fn solve(&mut self, x: &mut Vector, mat: &SparseMatrix, rhs: &Vector, verbose: bool) -> Result<(), StrError> {
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
        vec_copy(x, rhs).unwrap();
        let verb = if verbose { 1 } else { 0 };
        self.stopwatch.reset();
        unsafe {
            let status = solver_mumps_solve(
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
        stats.determinant.mantissa_real = self.determinant_coefficient;
        stats.determinant.mantissa_imag = 0.0;
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

const MUMPS_ORDERING_AMD: i32 = 0; // Amd (page 35)
const MUMPS_ORDERING_AMF: i32 = 2; // Amf (page 35)
const MUMPS_ORDERING_AUTO: i32 = 7; // Auto (page 36)
const MUMPS_ORDERING_METIS: i32 = 5; // Metis (page 35)
const MUMPS_ORDERING_PORD: i32 = 4; // Pord (page 35)
const MUMPS_ORDERING_QAMD: i32 = 6; // Qamd (page 35)
const MUMPS_ORDERING_SCOTCH: i32 = 3; // Scotch (page 35)

const MUMPS_SCALING_AUTO: i32 = 77; // Auto (page 33)
const MUMPS_SCALING_COLUMN: i32 = 3; // Column (page 33)
const MUMPS_SCALING_DIAGONAL: i32 = 1; // Diagonal (page 33)
const MUMPS_SCALING_NO: i32 = 0; // No (page 33)
const MUMPS_SCALING_ROW_COL: i32 = 4; // RowCol (page 33)
const MUMPS_SCALING_ROW_COL_ITER: i32 = 7; // RowColIter (page 33)
const MUMPS_SCALING_ROW_COL_RIG: i32 = 8; // RowColRig (page 33)

fn mumps_ordering(ordering: Ordering) -> i32 {
    match ordering {
        Ordering::Amd => MUMPS_ORDERING_AMD,       // Amd (page 35)
        Ordering::Amf => MUMPS_ORDERING_AMF,       // Amf (page 35)
        Ordering::Auto => MUMPS_ORDERING_AUTO,     // Auto (page 36)
        Ordering::Best => MUMPS_ORDERING_AUTO,     // Best => Auto (page 36)
        Ordering::Cholmod => MUMPS_ORDERING_AUTO,  // Cholmod => Auto (page 36)
        Ordering::Metis => MUMPS_ORDERING_METIS,   // Metis (page 35)
        Ordering::No => MUMPS_ORDERING_AUTO,       // No => Auto (page 36)
        Ordering::Pord => MUMPS_ORDERING_PORD,     // Pord (page 35)
        Ordering::Qamd => MUMPS_ORDERING_QAMD,     // Qamd (page 35)
        Ordering::Scotch => MUMPS_ORDERING_SCOTCH, // Scotch (page 35)
    }
}

fn mumps_scaling(scaling: Scaling) -> i32 {
    match scaling {
        Scaling::Auto => MUMPS_SCALING_AUTO,               // Auto (page 33)
        Scaling::Column => MUMPS_SCALING_COLUMN,           // Column (page 33)
        Scaling::Diagonal => MUMPS_SCALING_DIAGONAL,       // Diagonal (page 33)
        Scaling::Max => MUMPS_SCALING_AUTO,                // Max => Auto (page 33)
        Scaling::No => MUMPS_SCALING_NO,                   // No (page 33)
        Scaling::RowCol => MUMPS_SCALING_ROW_COL,          // RowCol (page 33)
        Scaling::RowColIter => MUMPS_SCALING_ROW_COL_ITER, // RowColIter (page 33)
        Scaling::RowColRig => MUMPS_SCALING_ROW_COL_RIG,   // RowColRig (page 33)
        Scaling::Sum => MUMPS_SCALING_AUTO,                // Sum => Auto (page 33)
    }
}

/// Handles error code
fn handle_mumps_error_code(err: i32) -> StrError {
    match err {
        -1 => "Error(-1): error on some processor",
        -2 => "Error(-2): nnz is out of range",
        -3 => "Error(-3): solver called with an invalid job value",
        -4 => "Error(-4): error in user-provided permutation array",
        -5 => "Error(-5): problem with real workspace allocation during analysis",
        -6 => "Error(-6): matrix is singular in structure",
        -7 => "Error(-7): problem with integer workspace allocation during analysis",
        -8 => "Error(-8): internal integer work array is too small for factorization",
        -9 => "Error(-9): internal real/complex work array is too small",
        -10 => "Error(-10): numerically singular matrix",
        -11 => "Error(-11): real/complex work array or lwk user is too small for solution",
        -12 => "Error(-12): real/complex work array is too small for iterative refinement",
        -13 => "Error(-13): problem with workspace allocation during factorization or solution",
        -14 => "Error(-14): integer work array is too small for solution",
        -15 => "Error(-15): integer work array is too small for iterative refinement and/or error analysis",
        -16 => "Error(-16): n is out of range",
        -17 => "Error(-17): internal send buffer is too small.",
        -18 => "Error(-18): blocking size for multiple rhs is too large",
        -19 => "Error(-19): maximum allowed size of working memory is too small for the factorization",
        -20 => "Error(-20): reception buffer is too small",
        -21 => "Error(-21): value of par=0 is not allowed",
        -22 => "Error(-22): problem with a pointer array provided by the user",
        -23 => "Error(-23): mpi was not initialized",
        -24 => "Error(-24): nelt is out of range",
        -25 => "Error(-25): problem with the initialization of BLACS",
        -26 => "Error(-26): lrhs is out of range",
        -27 => "Error(-27): nz rhs and irhs ptr(nrhs+1) do not match",
        -28 => "Error(-28): irhs ptr(1) is not equal to 1",
        -29 => "Error(-29): lsol loc is smaller than required",
        -30 => "Error(-30): Schur lld is out of range",
        -31 => "Error(-31): block cyclic symmetric Schur complement is required",
        -32 => "Error(-32): incompatible values of nrhs",
        -33 => "Error(-33): ICNTL(26) was asked for during solve/factorization phase",
        -34 => "Error(-34): lredrhs is out of range",
        -35 => "Error(-35): problem with the expansion phase",
        -36 => "Error(-36): incompatible values of ICNTL(25) and INFOG(28)",
        -37 => "Error(-37): value of ICNTL(25) is invalid",
        -38 => "Error(-38): parallel analysis requires PT-SCOTCH or ParMetis",
        -39 => "Error(-39): incompatible values for ICNTL(28), ICNTL(5) and/or ICNTL(19) and/or ICNTL(6)",
        -40 => "Error(-40): the matrix is not positive definite as assumed",
        -41 => "Error(-41): incompatible value of lwk user from factorization to solution",
        -42 => "Error(-42): incompatible ICNTL(32) value",
        -43 => "Error(-43): Incompatible values of ICNTL(32) and ICNTL(xx)",
        -44 => "Error(-44): the solve phase (JOB=3) cannot be performed",
        -45 => "Error(-45): nrhs less than 0",
        -46 => "Error(-46): nz rhs less than 0",
        -47 => "Error(-47): problem with entries of A-1",
        -48 => "Error(-48): A-1 incompatible values of ICNTL(30) and ICNTL(xx)",
        -49 => "Error(-49): size Schur has an incorrect value",
        -50 => "Error(-50): problem with fill-reducing ordering during analysis",
        -51 => "Error(-51): problem with external ordering (Metis/ParMetis, SCOTCH/PT-SCOTCH, PORD)",
        -52 => "Error(-52): problem with default Fortran integers",
        -53 => "Error(-53): inconsistent input data between two consecutive calls",
        -54 => "Error(-54): incompatible ICNTL(35)=0",
        -55 => "Error(-55): problem with solution and distributed right-hand side",
        -56 => "Error(-56): problem with solution and distributed right-hand side",
        -70 => "Error(-70): problem with the file to save the current instance",
        -71 => "Error(-71): problem with the creation of one of the files",
        -72 => "Error(-72): error while saving data",
        -73 => "Error(-73): problem with incompatible parameter of the current instance",
        -74 => "Error(-74): problem with output file",
        -75 => "Error(-75): error while restoring data",
        -76 => "Error(-76): error while deleting the files",
        -77 => "Error(-77): neither save dir nor the environment variable are defined.",
        -78 => "Error(-78): problem of workspace allocation during the restore step",
        -79 => "Error(-79): problem with the file unit used to open the save/restore file",
        -90 => "Error(-90): error in out-of-core management",
        -800 => "Error(-800): temporary error associated to the current release",
        1 => "Error(+1): index (in irn or jcn) is out of range",
        2 => "Error(+2): during error analysis the max-norm of the computed solution is close to zero",
        4 => "Error(+4): not used in current version",
        8 => "Error(+8): problem with the iterative refinement routine",
        ERROR_NULL_POINTER => return "MUMPS failed due to NULL POINTER error",
        ERROR_MALLOC => return "MUMPS failed due to MALLOC error",
        ERROR_VERSION => return "MUMPS failed due to VERSION error",
        ERROR_NOT_AVAILABLE => return "MUMPS is not AVAILABLE",
        ERROR_NEED_INITIALIZATION => return "MUMPS failed because INITIALIZATION is needed",
        ERROR_NEED_FACTORIZATION => return "MUMPS failed because FACTORIZATION is needed",
        ERROR_ALREADY_INITIALIZED => return "MUMPS failed because INITIALIZATION has been completed already",
        _ => return "Error: unknown error returned by c-code (MUMPS)",
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CooMatrix, LinSolParams, LinSolTrait, Ordering, Samples, Scaling, SparseMatrix, Storage, Symmetry};
    use russell_lab::{approx_eq, vec_approx_eq, Vector};

    #[test]
    fn complete_solution_cycle_works() {
        // IMPORTANT:
        // Since MUMPS is not thread-safe, we need to call all MUMPS functions
        // in a single test unit because the tests are run in parallel by default

        // allocate x and rhs
        let mut x = Vector::new(5);
        let rhs = Vector::from(&[8.0, 45.0, -3.0, 3.0, 19.0]);
        let x_correct = &[1.0, 2.0, 3.0, 4.0, 5.0];

        // allocate a new solver
        let mut solver = SolverMUMPS::new().unwrap();
        assert!(!solver.factorized);

        // get COO matrix errors
        let (_, csc, _, _) = Samples::tiny_1x1(true);
        let mut mat = SparseMatrix::from_csc(csc);
        assert_eq!(
            solver.factorize(&mut mat, None).err(),
            Some("COO matrix is not available")
        );

        // check COO matrix
        let coo = CooMatrix::new(1, 1, 1, None, false).unwrap();
        let mut mat = SparseMatrix::from_coo(coo);
        assert_eq!(
            solver.factorize(&mut mat, None).err(),
            Some("the COO matrix must have one-based (FORTRAN) indices as required by MUMPS")
        );
        let (coo, _, _, _) = Samples::rectangular_1x2(true, false, false);
        let mut mat = SparseMatrix::from_coo(coo);
        assert_eq!(
            solver.factorize(&mut mat, None).err(),
            Some("the COO matrix must be square")
        );
        let coo = CooMatrix::new(1, 1, 1, None, true).unwrap();
        let mut mat = SparseMatrix::from_coo(coo);
        assert_eq!(
            solver.factorize(&mut mat, None).err(),
            Some("the COO matrix must have at least one non-zero value")
        );
        let (coo, _, _, _) = Samples::mkl_symmetric_5x5_upper(true, false, false);
        let mut mat = SparseMatrix::from_coo(coo);
        assert_eq!(
            solver.factorize(&mut mat, None).err(),
            Some("if the matrix is general symmetric, the required storage is lower triangular")
        );

        // check already factorized data
        let mut coo = CooMatrix::new(2, 2, 2, None, true).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(1, 1, 2.0).unwrap();
        let mut mat = SparseMatrix::from_coo(coo);
        // ... factorize once => OK
        solver.factorize(&mut mat, None).unwrap();
        // ... change matrix (symmetry)
        let mut coo = CooMatrix::new(2, 2, 2, Some(Symmetry::General(Storage::Full)), true).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(1, 1, 2.0).unwrap();
        let mut mat = SparseMatrix::from_coo(coo);
        assert_eq!(
            solver.factorize(&mut mat, None).err(),
            Some("subsequent factorizations must use the same matrix (symmetry differs)")
        );
        // ... change matrix (ndim)
        let mut coo = CooMatrix::new(1, 1, 1, None, true).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        let mut mat = SparseMatrix::from_coo(coo);
        assert_eq!(
            solver.factorize(&mut mat, None).err(),
            Some("subsequent factorizations must use the same matrix (ndim differs)")
        );
        // ... change matrix (nnz)
        let mut coo = CooMatrix::new(2, 2, 1, None, true).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        let mut mat = SparseMatrix::from_coo(coo);
        assert_eq!(
            solver.factorize(&mut mat, None).err(),
            Some("subsequent factorizations must use the same matrix (nnz differs)")
        );

        // allocate a new solver
        let mut solver = SolverMUMPS::new().unwrap();
        assert!(!solver.factorized);

        // sample matrix
        let (coo, _, _, _) = Samples::umfpack_unsymmetric_5x5(true);
        let mut mat = SparseMatrix::from_coo(coo);

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
        let mut x_wrong = Vector::new(3);
        let rhs_wrong = Vector::new(2);
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
        vec_approx_eq(x.as_data(), x_correct, 1e-14);

        // check ordering and scaling
        assert_eq!(solver.effective_ordering, 4); // Pord
        assert_eq!(solver.effective_scaling, 0); // No, because we requested the determinant

        // check the determinant
        let d = solver.determinant_coefficient * f64::powf(2.0, solver.determinant_exponent);
        approx_eq(d, 114.0, 1e-13);

        // update stats
        let mut stats = StatsLinSol::new();
        solver.update_stats(&mut stats);
        assert_eq!(stats.output.effective_ordering, "Pord");
        assert_eq!(stats.output.effective_scaling, "No");

        // calling solve again works
        let mut x_again = Vector::new(5);
        solver.solve(&mut x_again, &mat, &rhs, false).unwrap();
        vec_approx_eq(x_again.as_data(), x_correct, 1e-14);

        // factorize fails on singular matrix
        let mut mat_singular = SparseMatrix::new_coo(5, 5, 2, None, true).unwrap();
        mat_singular.put(0, 0, 1.0).unwrap();
        mat_singular.put(4, 4, 1.0).unwrap();
        let mut solver = SolverMUMPS::new().unwrap();
        assert_eq!(
            solver.factorize(&mut mat_singular, None),
            Err("Error(-10): numerically singular matrix")
        );

        // solve with positive-definite matrix works
        let (coo_pd_lower, _, _, _) = Samples::mkl_positive_definite_5x5_lower(true);
        let mut mat_pd_lower = SparseMatrix::from_coo(coo_pd_lower);
        params.ordering = Ordering::Auto;
        params.scaling = Scaling::Auto;
        let mut solver = SolverMUMPS::new().unwrap();
        assert!(!solver.factorized);
        solver.factorize(&mut mat_pd_lower, Some(params)).unwrap();
        let mut x = Vector::new(5);
        let rhs = Vector::from(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        solver.solve(&mut x, &mat_pd_lower, &rhs, false).unwrap();
        let x_correct = &[-979.0 / 3.0, 983.0, 1961.0 / 12.0, 398.0, 123.0 / 2.0];
        vec_approx_eq(x.as_data(), x_correct, 1e-10);

        // solve with different matrix fails
        assert_eq!(
            solver.solve(&mut x, &mat, &rhs, false).err(),
            Some("solve must use the same matrix (symmetry differs)")
        );
    }

    #[test]
    fn ordering_and_scaling_works() {
        assert_eq!(mumps_ordering(Ordering::Amd), MUMPS_ORDERING_AMD);
        assert_eq!(mumps_ordering(Ordering::Amf), MUMPS_ORDERING_AMF);
        assert_eq!(mumps_ordering(Ordering::Auto), MUMPS_ORDERING_AUTO);
        assert_eq!(mumps_ordering(Ordering::Best), MUMPS_ORDERING_AUTO);
        assert_eq!(mumps_ordering(Ordering::Cholmod), MUMPS_ORDERING_AUTO);
        assert_eq!(mumps_ordering(Ordering::Metis), MUMPS_ORDERING_METIS);
        assert_eq!(mumps_ordering(Ordering::No), MUMPS_ORDERING_AUTO);
        assert_eq!(mumps_ordering(Ordering::Pord), MUMPS_ORDERING_PORD);
        assert_eq!(mumps_ordering(Ordering::Qamd), MUMPS_ORDERING_QAMD);
        assert_eq!(mumps_ordering(Ordering::Scotch), MUMPS_ORDERING_SCOTCH);

        assert_eq!(mumps_scaling(Scaling::Auto), MUMPS_SCALING_AUTO);
        assert_eq!(mumps_scaling(Scaling::Column), MUMPS_SCALING_COLUMN);
        assert_eq!(mumps_scaling(Scaling::Diagonal), MUMPS_SCALING_DIAGONAL);
        assert_eq!(mumps_scaling(Scaling::Max), MUMPS_SCALING_AUTO);
        assert_eq!(mumps_scaling(Scaling::No), MUMPS_SCALING_NO);
        assert_eq!(mumps_scaling(Scaling::RowCol), MUMPS_SCALING_ROW_COL);
        assert_eq!(mumps_scaling(Scaling::RowColIter), MUMPS_SCALING_ROW_COL_ITER);
        assert_eq!(mumps_scaling(Scaling::RowColRig), MUMPS_SCALING_ROW_COL_RIG);
        assert_eq!(mumps_scaling(Scaling::Sum), MUMPS_SCALING_AUTO);
    }

    #[test]
    fn handle_mumps_error_code_works() {
        let default = "Error: unknown error returned by c-code (MUMPS)";
        for c in 1..57 {
            let res = handle_mumps_error_code(-c);
            assert!(res.len() > 0);
            assert_ne!(res, default);
        }
        for c in 70..80 {
            let res = handle_mumps_error_code(-c);
            assert!(res.len() > 0);
            assert_ne!(res, default);
        }
        for c in &[-90, -800, 1, 2, 4, 8] {
            let res = handle_mumps_error_code(*c);
            assert!(res.len() > 0);
            assert_ne!(res, default);
        }
        assert_eq!(
            handle_mumps_error_code(ERROR_NULL_POINTER),
            "MUMPS failed due to NULL POINTER error"
        );
        assert_eq!(
            handle_mumps_error_code(ERROR_MALLOC),
            "MUMPS failed due to MALLOC error"
        );
        assert_eq!(
            handle_mumps_error_code(ERROR_VERSION),
            "MUMPS failed due to VERSION error"
        );
        assert_eq!(handle_mumps_error_code(ERROR_NOT_AVAILABLE), "MUMPS is not AVAILABLE");
        assert_eq!(
            handle_mumps_error_code(ERROR_NEED_INITIALIZATION),
            "MUMPS failed because INITIALIZATION is needed"
        );
        assert_eq!(
            handle_mumps_error_code(ERROR_NEED_FACTORIZATION),
            "MUMPS failed because FACTORIZATION is needed"
        );
        assert_eq!(
            handle_mumps_error_code(ERROR_ALREADY_INITIALIZED),
            "MUMPS failed because INITIALIZATION has been completed already"
        );
        assert_eq!(handle_mumps_error_code(123), default);
    }
}
