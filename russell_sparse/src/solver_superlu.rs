use super::{to_i32, CcBool, LinSolParams, LinSolTrait, SparseMatrix, Symmetry};
use crate::{auxiliary_and_constants::SUCCESSFUL_EXIT, StrError};
use russell_lab::Vector;

/// Opaque struct holding a C-pointer to InterfaceSuperLU
///
/// Reference: <https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs>
#[repr(C)]
struct InterfaceSuperLU {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

extern "C" {
    fn solver_superlu_new() -> *mut InterfaceSuperLU;
    fn solver_superlu_drop(solver: *mut InterfaceSuperLU);
    fn solver_superlu_factorize(
        solver: *mut InterfaceSuperLU,
        // output
        condition_number: *mut f64,
        // input
        ordering: i32,
        scaling: CcBool,
        // matrix config
        ndim: i32,
        // matrix
        col_pointers: *const i32,
        row_indices: *const i32,
        values: *const f64,
    ) -> i32;
    fn solver_superlu_solve(solver: *mut InterfaceSuperLU, x: *mut f64, rhs: *const f64) -> i32;
}

/// Wraps the SuperLU solver for sparse linear systems
///
/// **Warning:** This solver has not been extensively tested here.
pub struct SolverSuperLU {
    /// Holds a pointer to the C interface to SuperLU
    solver: *mut InterfaceSuperLU,

    /// Indicates whether the sparse matrix has been factorized or not
    factorized: bool,

    /// Holds the symmetry type used in the first call to factorize
    factorized_symmetry: Option<Symmetry>,

    /// Holds the matrix dimension saved in the first call to factorize
    factorized_ndim: usize,

    /// Holds the number of non-zeros saved in the first call to factorize
    factorized_nnz: usize,

    /// Holds the reciprocal condition_number
    condition_number: f64,
}

impl Drop for SolverSuperLU {
    /// Tells the c-code to release memory
    fn drop(&mut self) {
        unsafe {
            solver_superlu_drop(self.solver);
        }
    }
}

impl SolverSuperLU {
    /// Allocates a new instance
    pub fn new() -> Result<Self, StrError> {
        unsafe {
            let solver = solver_superlu_new();
            if solver.is_null() {
                return Err("c-code failed to allocate the SuperLU solver");
            }
            Ok(SolverSuperLU {
                solver,
                factorized: false,
                factorized_symmetry: None,
                factorized_ndim: 0,
                factorized_nnz: 0,
                condition_number: 0.0,
            })
        }
    }
}

impl LinSolTrait for SolverSuperLU {
    /// Performs the factorization (and analysis/initialization if needed)
    ///
    /// # Input
    ///
    /// * `mat` -- the coefficient matrix A (**COO** or **CSC**, but not CSR).
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
    /// 4. For symmetric matrices, `SuperLU` (this implementation) requires that
    ///    the symmetry/storage be [crate::Storage::Lower].
    fn factorize(&mut self, mat: &mut SparseMatrix, params: Option<LinSolParams>) -> Result<(), StrError> {
        // get CSC matrix
        let csc = mat.get_csc_or_from_coo()?;

        // check CSC matrix
        if csc.nrow != csc.ncol {
            return Err("the matrix must be square");
        }
        csc.check_dimensions()?;
        if let Some(symmetry) = csc.symmetry {
            if symmetry.triangular() {
                return Err("for SuperLU, the matrix must not be triangular");
            }
        }

        // check already factorized data
        if self.factorized == true {
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
        let ordering = par.ordering as i32;
        let scaling = if par.superlu_scaling { 1 } else { 0 };

        // matrix config
        let ndim = to_i32(csc.nrow)?;

        // call SuperLU factorize
        unsafe {
            let status = solver_superlu_factorize(
                self.solver,
                // output
                &mut self.condition_number,
                // input
                ordering,
                scaling,
                // matrix config
                ndim,
                // matrix
                csc.col_pointers.as_ptr(),
                csc.row_indices.as_ptr(),
                csc.values.as_ptr(),
            );
            if status != SUCCESSFUL_EXIT {
                return Err(handle_superlu_error_code(status));
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
    /// * `_verbose` -- not used
    ///
    /// **Warning:** the matrix must be same one used in `factorize`.
    fn solve(&mut self, x: &mut Vector, mat: &SparseMatrix, rhs: &Vector, _verbose: bool) -> Result<(), StrError> {
        // check already factorized data
        if self.factorized == true {
            let (nrow, ncol, nnz, symmetry) = mat.get_info();
            if symmetry != self.factorized_symmetry {
                return Err("solve must use the same matrix (symmetry differs)");
            }
            if nrow != self.factorized_ndim || ncol != self.factorized_ndim {
                return Err("solve must use the same matrix (ndim differs)");
            }
            if nnz != self.factorized_nnz {
                return Err("solve must use the same matrix (nnz differs)");
            }
        } else {
            return Err("the function factorize must be called before solve");
        }

        // check vectors
        if x.dim() != self.factorized_ndim {
            return Err("the dimension of the vector of unknown values x is incorrect");
        }
        if rhs.dim() != self.factorized_ndim {
            return Err("the dimension of the right-hand side vector is incorrect");
        }

        // call SuperLU solve
        unsafe {
            let status = solver_superlu_solve(self.solver, x.as_mut_data().as_mut_ptr(), rhs.as_data().as_ptr());
            if status != SUCCESSFUL_EXIT {
                return Err(handle_superlu_error_code(status));
            }
        }

        // done
        Ok(())
    }

    /// Returns the determinant (NOT AVAILABLE)
    fn get_determinant(&self) -> (f64, f64, f64) {
        (0.0, 0.0, 0.0)
    }

    /// Returns the ordering effectively used by the solver (NOT AVAILABLE)
    fn get_effective_ordering(&self) -> String {
        "Unknown".to_string()
    }

    /// Returns the scaling effectively used by the solver (NOT AVAILABLE)
    fn get_effective_scaling(&self) -> String {
        "Unknown".to_string()
    }

    /// Returns the strategy (concerning symmetry) effectively used by the solver (NOT AVAILABLE)
    fn get_effective_strategy(&self) -> String {
        "Unknown".to_string()
    }

    /// Returns the name of this solver
    ///
    /// # Output
    ///
    /// * `SuperLU` -- if the default system SuperLU has been used
    /// * `SuperLU-local` -- if the locally compiled SuperLU has be used
    fn get_name(&self) -> String {
        if cfg!(local_superlu) {
            "SuperLU-local".to_string()
        } else {
            "SuperLU".to_string()
        }
    }
}

/// Handles SuperLU error code
pub(crate) fn handle_superlu_error_code(err: i32) -> StrError {
    match err {
        _ => return "Error: unknown error returned by c-code (SuperLU)",
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{handle_superlu_error_code, SolverSuperLU};
    use crate::{CooMatrix, LinSolParams, LinSolTrait, Ordering, Samples, Scaling, SparseMatrix};
    use russell_chk::{approx_eq, vec_approx_eq};
    use russell_lab::Vector;

    #[test]
    fn new_and_drop_work() {
        // you may debug into the C-code to see that drop is working
        let solver = SolverSuperLU::new().unwrap();
        assert!(!solver.factorized);
    }

    #[test]
    fn factorize_handles_errors() {
        let mut solver = SolverSuperLU::new().unwrap();
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
            Some("COO matrix: pos = nnz must be ≥ 1")
        );
        let (coo, _, _, _) = Samples::mkl_symmetric_5x5_lower(false, false, false);
        let mut mat = SparseMatrix::from_coo(coo);
        assert_eq!(
            solver.factorize(&mut mat, None).err(),
            Some("for SuperLU, the matrix must not be triangular")
        );
    }

    #[test]
    fn factorize_works() {
        let mut solver = SolverSuperLU::new().unwrap();
        assert!(!solver.factorized);
        let (coo, _, _, _) = Samples::umfpack_unsymmetric_5x5(false);
        let mut mat = SparseMatrix::from_coo(coo);
        let mut params = LinSolParams::new();

        params.compute_determinant = true;
        params.ordering = Ordering::Amd;
        params.scaling = Scaling::Sum;

        solver.factorize(&mut mat, Some(params)).unwrap();
        assert!(solver.factorized);

        assert_eq!(solver.get_effective_ordering(), "Amd");
        assert_eq!(solver.get_effective_scaling(), "Sum");

        let (a, b, c) = solver.get_determinant();
        let det = a * f64::powf(b, c);
        approx_eq(det, 114.0, 1e-13);

        // calling factorize again works
        solver.factorize(&mut mat, Some(params)).unwrap();
        let (a, b, c) = solver.get_determinant();
        let det = a * f64::powf(b, c);
        approx_eq(det, 114.0, 1e-13);
    }

    #[test]
    fn factorize_fails_on_singular_matrix() {
        let mut solver = SolverSuperLU::new().unwrap();
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
        let mut solver = SolverSuperLU::new().unwrap();
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
        let mut solver = SolverSuperLU::new().unwrap();
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
    fn handle_superlu_error_code_works() {
        let default = "Error: unknown error returned by c-code (SuperLU)";
        for c in &[1, 2, 3, -1, -3, -4, -5, -6, -8, -11, -13, -15, -17, -18, -911] {
            let res = handle_superlu_error_code(*c);
            assert!(res.len() > 0);
            assert_ne!(res, default);
        }
        assert_eq!(
            handle_superlu_error_code(100000),
            "Error: c-code returned null pointer (SuperLU)"
        );
        assert_eq!(
            handle_superlu_error_code(200000),
            "Error: c-code failed to allocate memory (SuperLU)"
        );
        assert_eq!(handle_superlu_error_code(123), default);
    }
}
