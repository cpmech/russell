use super::{LinSolParams, LinSolTrait, SparseMatrix, StatsLinSol, Symmetry};
use crate::{
    to_i32, CcBool, StrError, MALLOC_ERROR, NEED_FACTORIZATION, NOT_AVAILABLE, NULL_POINTER_ERROR, SUCCESSFUL_EXIT,
};
use russell_lab::Vector;

/// Opaque struct holding a C-pointer to InterfaceIntelDSS
///
/// Reference: <https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs>
#[repr(C)]
struct InterfaceIntelDSS {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

extern "C" {
    fn solver_intel_dss_new() -> *mut InterfaceIntelDSS;
    fn solver_intel_dss_drop(solver: *mut InterfaceIntelDSS);
    fn solver_intel_dss_factorize(
        solver: *mut InterfaceIntelDSS,
        // output
        determinant_coefficient: *mut f64,
        determinant_exponent: *mut f64,
        // requests
        compute_determinant: CcBool,
        // matrix config
        general_symmetric: CcBool,
        positive_definite: CcBool,
        ndim: i32,
        // matrix
        row_pointers: *const i32,
        col_indices: *const i32,
        values: *const f64,
    ) -> i32;
    fn solver_intel_dss_solve(solver: *mut InterfaceIntelDSS, x: *mut f64, rhs: *const f64) -> i32;
}

/// Wraps the IntelDSS solver for sparse linear systems
///
/// **Warning:** This solver does not check whether the matrix is singular or not;
/// thus it may return **incorrect results** if a singular matrix is given to factorize.
///
/// **Warning:** This solver may fail with large matrices (e.g., ATandT/pre2) and
/// may return **incorrect results**.
pub struct SolverIntelDSS {
    /// Holds a pointer to the C interface to IntelDSS
    solver: *mut InterfaceIntelDSS,

    /// Indicates whether the sparse matrix has been factorized or not
    factorized: bool,

    /// Holds the symmetry type used in the first call to factorize
    factorized_symmetry: Option<Symmetry>,

    /// Holds the matrix dimension saved in the first call to factorize
    factorized_ndim: usize,

    /// Holds the number of non-zeros saved in the first call to factorize
    factorized_nnz: usize,

    /// Holds the determinant coefficient: det = coefficient * pow(10, exponent)
    determinant_coefficient: f64,

    /// Holds the determinant exponent: det = coefficient * pow(10, exponent)
    determinant_exponent: f64,
}

impl Drop for SolverIntelDSS {
    /// Tells the c-code to release memory
    fn drop(&mut self) {
        unsafe {
            solver_intel_dss_drop(self.solver);
        }
    }
}

impl SolverIntelDSS {
    /// Allocates a new instance
    ///
    /// **Warning:** This solver does not check whether the matrix is singular or not;
    /// thus it may return **incorrect results** if a singular matrix is given to factorize.
    ///
    /// **Warning:** This solver may fail with large matrices (e.g., ATandT/pre2) and
    /// may return **incorrect results**.
    pub fn new() -> Result<Self, StrError> {
        if !cfg!(with_intel_dss) {
            return Err("This code has not been compiled with Intel DSS");
        }
        unsafe {
            let solver = solver_intel_dss_new();
            if solver.is_null() {
                return Err("c-code failed to allocate the IntelDSS solver");
            }
            Ok(SolverIntelDSS {
                solver,
                factorized: false,
                factorized_symmetry: None,
                factorized_ndim: 0,
                factorized_nnz: 0,
                determinant_coefficient: 0.0,
                determinant_exponent: 0.0,
            })
        }
    }
}

impl LinSolTrait for SolverIntelDSS {
    /// Performs the factorization (and analysis/initialization if needed)
    ///
    /// # Input
    ///
    /// * `mat` -- the coefficient matrix A (**COO** or **CSR**, but not CSC).
    ///   Also, the matrix must be square (`nrow = ncol`) and, if symmetric,
    ///   the symmetry/storage must [crate::Storage::Upper].
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
    /// 4. For symmetric matrices, `DSS` requires that the symmetry/storage be [crate::Storage::Upper].
    ///
    /// **Warning:** This solver does not check whether the matrix is singular or not;
    /// thus it may return **incorrect results** if a singular matrix is given to factorize.
    ///
    /// **Warning:** This solver may fail with large matrices (e.g., ATandT/pre2) and
    /// may return **incorrect results**.
    fn factorize(&mut self, mat: &mut SparseMatrix, params: Option<LinSolParams>) -> Result<(), StrError> {
        // get CSR matrix
        // (or convert from COO if CSR is not available and COO is available)
        let csr = mat.get_csr_or_from_coo()?;

        // check CSR matrix
        if csr.nrow != csr.ncol {
            return Err("the matrix must be square");
        }

        // check already factorized data
        if self.factorized {
            if csr.symmetry != self.factorized_symmetry {
                return Err("subsequent factorizations must use the same matrix (symmetry differs)");
            }
            if csr.nrow != self.factorized_ndim {
                return Err("subsequent factorizations must use the same matrix (ndim differs)");
            }
            if (csr.row_pointers[csr.nrow] as usize) != self.factorized_nnz {
                return Err("subsequent factorizations must use the same matrix (nnz differs)");
            }
        } else {
            self.factorized_symmetry = csr.symmetry;
            self.factorized_ndim = csr.nrow;
            self.factorized_nnz = csr.row_pointers[csr.nrow] as usize;
            if self.factorized_nnz < self.factorized_ndim {
                return Err("for Intel DSS, nnz = row_pointers[nrow] must be ≥ nrow");
            }
        }

        // configuration parameters
        let par = if let Some(p) = params { p } else { LinSolParams::new() };

        // requests
        let calc_det = if par.compute_determinant { 1 } else { 0 };

        // extract the symmetry flags and check the storage type
        let (general_symmetric, positive_definite) = match csr.symmetry {
            Some(symmetry) => symmetry.status(false, true)?,
            None => (0, 0),
        };

        // matrix config
        let ndim = to_i32(csr.nrow);

        // call Intel DSS factorize
        let nnz = self.factorized_nnz;
        unsafe {
            let status = solver_intel_dss_factorize(
                self.solver,
                // output
                &mut self.determinant_coefficient,
                &mut self.determinant_exponent,
                // requests
                calc_det,
                // matrix config
                general_symmetric,
                positive_definite,
                ndim,
                // matrix
                csr.row_pointers.as_ptr(),
                csr.col_indices[0..nnz].as_ptr(),
                csr.values[0..nnz].as_ptr(),
            );
            if status != SUCCESSFUL_EXIT {
                return Err(handle_intel_dss_error_code(status));
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
    /// * `mat` -- the coefficient matrix A; must be square and, if symmetric, [crate::Storage::Upper].
    /// * `rhs` -- the right-hand side vector with know values an dimension equal to mat.nrow
    /// * `_verbose` -- not used
    ///
    /// **Warning:** the matrix must be same one used in `factorize`.
    fn solve(&mut self, x: &mut Vector, mat: &SparseMatrix, rhs: &Vector, _verbose: bool) -> Result<(), StrError> {
        // check
        if !self.factorized {
            return Err("the function factorize must be called before solve");
        }

        // access CSR matrix
        // (possibly already converted from COO, because factorize was (should have been) called)
        let csr = mat.get_csr()?;

        // check already factorized data
        let (nrow, ncol, nnz, symmetry) = csr.get_info();
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

        // call Intel DSS solve
        unsafe {
            let status = solver_intel_dss_solve(self.solver, x.as_mut_data().as_mut_ptr(), rhs.as_data().as_ptr());
            if status != SUCCESSFUL_EXIT {
                return Err(handle_intel_dss_error_code(status));
            }
        }
        Ok(())
    }

    /// Updates the stats structure (should be called after solve)
    fn update_stats(&self, stats: &mut StatsLinSol) {
        stats.main.solver = if cfg!(with_intel_dss) {
            "IntelDSS".to_string()
        } else {
            "INTEL_DSS_IS_NOT_AVAILABLE".to_string()
        };
        stats.determinant.mantissa = self.determinant_coefficient;
        stats.determinant.base = 10.0;
        stats.determinant.exponent = self.determinant_exponent;
    }
}

/// Handles Intel DSS error code
pub(crate) fn handle_intel_dss_error_code(err: i32) -> StrError {
    match err {
        -1 => return "MKL_DSS_ZERO_PIVOT",
        -2 => return "MKL_DSS_OUT_OF_MEMORY",
        -3 => return "MKL_DSS_FAILURE",
        -4 => return "MKL_DSS_ROW_ERR",
        -5 => return "MKL_DSS_COL_ERR",
        -6 => return "MKL_DSS_TOO_FEW_VALUES",
        -7 => return "MKL_DSS_TOO_MANY_VALUES",
        -8 => return "MKL_DSS_NOT_SQUARE",
        -9 => return "MKL_DSS_STATE_ERR",
        -10 => return "MKL_DSS_INVALID_OPTION",
        -11 => return "MKL_DSS_OPTION_CONFLICT",
        -12 => return "MKL_DSS_MSG_LVL_ERR",
        -13 => return "MKL_DSS_TERM_LVL_ERR",
        -14 => return "MKL_DSS_STRUCTURE_ERR",
        -15 => return "MKL_DSS_REORDER_ERR",
        -16 => return "MKL_DSS_VALUES_ERR",
        17 => return "MKL_DSS_STATISTICS_INVALID_MATRIX",
        18 => return "MKL_DSS_STATISTICS_INVALID_STATE",
        19 => return "MKL_DSS_STATISTICS_INVALID_STRING",
        20 => return "MKL_DSS_REORDER1_ERR",
        21 => return "MKL_DSS_PREORDER_ERR",
        22 => return "MKL_DSS_DIAG_ERR",
        23 => return "MKL_DSS_I32BIT_ERR",
        24 => return "MKL_DSS_OOC_MEM_ERR",
        25 => return "MKL_DSS_OOC_OC_ERR",
        26 => return "MKL_DSS_OOC_RW_ERR",
        NULL_POINTER_ERROR => return "Error: c-code returned null pointer (IntelDSS)",
        MALLOC_ERROR => return "Error: c-code failed to allocate memory (IntelDSS)",
        NOT_AVAILABLE => return "This code has not been compiled with Intel DSS",
        NEED_FACTORIZATION => return "INTERNAL ERROR: factorization must be completed before solve",
        _ => return "Error: unknown error returned by c-code (IntelDSS)",
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[cfg(with_intel_dss)]
mod tests {
    use super::{handle_intel_dss_error_code, SolverIntelDSS};
    use crate::{CooMatrix, LinSolParams, LinSolTrait, Samples, SparseMatrix, StatsLinSol, Storage, Symmetry};
    use russell_lab::{approx_eq, vec_approx_eq, Vector};

    #[test]
    fn new_and_drop_work() {
        // you may debug into the C-code to see that drop is working
        let solver = SolverIntelDSS::new().unwrap();
        assert!(!solver.factorized);
    }

    #[test]
    fn factorize_handles_errors() {
        let mut solver = SolverIntelDSS::new().unwrap();
        assert!(!solver.factorized);

        // COO to CSR errors
        let coo = CooMatrix::new(1, 1, 1, None, false).unwrap();
        let mut mat = SparseMatrix::from_coo(coo);
        assert_eq!(
            solver.factorize(&mut mat, None).err(),
            Some("COO to CSR requires nnz > 0")
        );

        // check CSR matrix
        let (coo, _, _, _) = Samples::rectangular_1x7();
        let mut mat = SparseMatrix::from_coo(coo);
        assert_eq!(
            solver.factorize(&mut mat, None).err(),
            Some("the matrix must be square")
        );
        let (coo, _, _, _) = Samples::mkl_symmetric_5x5_lower(false, false, false);
        let mut mat = SparseMatrix::from_coo(coo);
        assert_eq!(
            solver.factorize(&mut mat, None).err(),
            Some("if the matrix is general symmetric, the required storage is upper triangular")
        );

        // check already factorized data
        let mut coo = CooMatrix::new(2, 2, 2, None, false).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(1, 1, 2.0).unwrap();
        let mut mat = SparseMatrix::from_coo(coo);
        // ... factorize once => OK
        solver.factorize(&mut mat, None).unwrap();
        // ... change matrix (symmetry)
        let mut coo = CooMatrix::new(2, 2, 2, Some(Symmetry::General(Storage::Full)), false).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(1, 1, 2.0).unwrap();
        let mut mat = SparseMatrix::from_coo(coo);
        assert_eq!(
            solver.factorize(&mut mat, None).err(),
            Some("subsequent factorizations must use the same matrix (symmetry differs)")
        );
        // ... change matrix (ndim)
        let mut coo = CooMatrix::new(1, 1, 1, None, false).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        let mut mat = SparseMatrix::from_coo(coo);
        assert_eq!(
            solver.factorize(&mut mat, None).err(),
            Some("subsequent factorizations must use the same matrix (ndim differs)")
        );
        // ... change matrix (nnz)
        let mut coo = CooMatrix::new(2, 2, 1, None, false).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        let mut mat = SparseMatrix::from_coo(coo);
        assert_eq!(
            solver.factorize(&mut mat, None).err(),
            Some("subsequent factorizations must use the same matrix (nnz differs)")
        );
    }

    #[test]
    fn factorize_works() {
        let mut solver = SolverIntelDSS::new().unwrap();
        assert!(!solver.factorized);
        let (coo, _, _, _) = Samples::umfpack_unsymmetric_5x5(false);
        let mut mat = SparseMatrix::from_coo(coo);
        let mut params = LinSolParams::new();

        params.compute_determinant = true;

        solver.factorize(&mut mat, Some(params)).unwrap();
        assert!(solver.factorized);
        let det = solver.determinant_coefficient * f64::powf(10.0, solver.determinant_exponent);
        approx_eq(det, 114.0, 1e-13);

        // calling factorize again works
        solver.factorize(&mut mat, Some(params)).unwrap();
        let det = solver.determinant_coefficient * f64::powf(10.0, solver.determinant_exponent);
        approx_eq(det, 114.0, 1e-13);
    }

    #[test]
    fn factorize_fails_on_singular_matrix() {
        let mut solver = SolverIntelDSS::new().unwrap();
        let mut coo = CooMatrix::new(2, 2, 2, None, false).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(1, 1, 0.0).unwrap();
        let mut mat = SparseMatrix::from_coo(coo);
        println!("Warning: Intel DSS does not detect singular matrices");
        assert_eq!(solver.factorize(&mut mat, None).err(), None);
    }

    #[test]
    fn solve_handles_errors() {
        let (coo, _, _, _) = Samples::tiny_1x1(false);
        let mut mat = SparseMatrix::from_coo(coo);
        let mut solver = SolverIntelDSS::new().unwrap();
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
        let mut solver = SolverIntelDSS::new().unwrap();
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

        // update stats
        let mut stats = StatsLinSol::new();
        solver.update_stats(&mut stats);
        assert_eq!(stats.output.effective_ordering, "Unknown");
        assert_eq!(stats.output.effective_scaling, "Unknown");
    }

    #[test]
    fn handle_intel_dss_error_code_works() {
        let default = "Error: unknown error returned by c-code (IntelDSS)";
        for i in 1..17 {
            let res = handle_intel_dss_error_code(-i);
            assert!(res.len() > 0);
            assert_ne!(res, default);
        }
        for i in 17..27 {
            let res = handle_intel_dss_error_code(i);
            assert!(res.len() > 0);
            assert_ne!(res, default);
        }
        assert_eq!(
            handle_intel_dss_error_code(100000),
            "Error: c-code returned null pointer (IntelDSS)"
        );
        assert_eq!(
            handle_intel_dss_error_code(200000),
            "Error: c-code failed to allocate memory (IntelDSS)"
        );
        assert_eq!(
            handle_intel_dss_error_code(400000),
            "This code has not been compiled with Intel DSS"
        );
        assert_eq!(handle_intel_dss_error_code(123), default);
    }
}
