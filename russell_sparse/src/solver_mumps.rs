use super::{to_i32, LinSolParams, LinSolTrait, Ordering, Scaling, SparseMatrix, Symmetry};
use crate::StrError;
use russell_lab::{vec_copy, Vector};

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
    fn solver_mumps_factorize(
        solver: *mut InterfaceMUMPS,
        // output
        effective_ordering: *mut i32,
        effective_scaling: *mut i32,
        determinant_coefficient: *mut f64,
        determinant_exponent: *mut f64,
        // input
        ordering: i32,
        scaling: i32,
        pct_inc_workspace: i32,
        max_work_memory: i32,
        openmp_num_threads: i32,
        // requests
        compute_determinant: i32,
        verbose: i32,
        // matrix config
        general_symmetric: i32,
        positive_definite: i32,
        ndim: i32,
        nnz: i32,
        // matrix
        indices_i: *const i32,
        indices_j: *const i32,
        values_aij: *const f64,
    ) -> i32;
    fn solver_mumps_solve(solver: *mut InterfaceMUMPS, rhs: *mut f64, verbose: i32) -> i32;
}

/// Wraps the MUMPS solver for (very large) sparse linear systems
///
/// **Warning:** This solver is **not** thread-safe, thus use only use in single-thread applications.
pub struct SolverMUMPS {
    /// Holds a pointer to the C interface to MUMPS
    solver: *mut InterfaceMUMPS,

    /// Indicates whether the sparse matrix has been factorized or not
    factorized: bool,

    /// Holds the symmetry type used in the first call to factorize
    factorized_symmetry: Option<Symmetry>,

    /// Holds the matrix dimension saved in the first call to factorize
    factorized_ndim: usize,

    /// Holds the number of non-zeros saved in the first call to factorize
    factorized_nnz: usize,

    /// Holds the used ordering (after factorize)
    effective_ordering: i32,

    /// Holds the used scaling (after factorize)
    effective_scaling: i32,

    /// Holds the determinant coefficient: det = coefficient * pow(2, exponent)
    determinant_coefficient: f64,

    /// Holds the determinant exponent: det = coefficient * pow(2, exponent)
    determinant_exponent: f64,
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
                factorized: false,
                factorized_symmetry: None,
                factorized_ndim: 0,
                factorized_nnz: 0,
                effective_ordering: -1,
                effective_scaling: -1,
                determinant_coefficient: 0.0,
                determinant_exponent: 0.0,
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
            return Err("the matrix must be square");
        }
        coo.check_dimensions_ready()?;

        // check already factorized data
        if self.factorized == true {
            if coo.symmetry != self.factorized_symmetry {
                return Err("subsequent factorizations must use the same matrix (symmetry differs)");
            }
            if coo.nrow != self.factorized_ndim {
                return Err("subsequent factorizations must use the same matrix (ndim differs)");
            }
            if coo.nnz != self.factorized_nnz {
                return Err("subsequent factorizations must use the same matrix (nnz differs)");
            }
        } else {
            self.factorized_symmetry = coo.symmetry;
            self.factorized_ndim = coo.nrow;
            self.factorized_nnz = coo.nnz;
        }

        // configuration parameters
        let par = if let Some(p) = params { p } else { LinSolParams::new() };

        // input parameters
        let ordering = match par.ordering {
            Ordering::Amd => 0,     // Amd (page 35)
            Ordering::Amf => 2,     // Amf (page 35)
            Ordering::Auto => 7,    // Auto (page 36)
            Ordering::Best => 7,    // Best => Auto (page 36)
            Ordering::Cholmod => 7, // Cholmod => Auto (page 36)
            Ordering::Metis => 5,   // Metis (page 35)
            Ordering::No => 7,      // No => Auto (page 36)
            Ordering::Pord => 4,    // Pord (page 35)
            Ordering::Qamd => 6,    // Qamd (page 35)
            Ordering::Scotch => 3,  // Scotch (page 35)
        };
        let scaling = match par.scaling {
            Scaling::Auto => 77,      // Auto (page 33)
            Scaling::Column => 3,     // Column (page 33)
            Scaling::Diagonal => 1,   // Diagonal (page 33)
            Scaling::Max => 77,       // Max => Auto (page 33)
            Scaling::No => 0,         // No (page 33)
            Scaling::RowCol => 4,     // RowCol (page 33)
            Scaling::RowColIter => 7, // RowColIter (page 33)
            Scaling::RowColRig => 8,  // RowColRig (page 33)
            Scaling::Sum => 77,       // Sum => Auto (page 33)
        };
        let pct_inc_workspace = to_i32(par.mumps_pct_inc_workspace)?;
        let max_work_memory = to_i32(par.mumps_max_work_memory)?;
        let openmp_num_threads = to_i32(par.mumps_openmp_num_threads)?;

        // requests
        let calc_det = if par.compute_determinant { 1 } else { 0 };
        let verbose = if par.verbose { 1 } else { 0 };

        // extract the symmetry flags and check the storage type
        let (general_symmetric, positive_definite) = match coo.symmetry {
            Some(symmetry) => symmetry.status(true, false)?,
            None => (0, 0),
        };

        // matrix config
        let ndim = to_i32(coo.nrow)?;
        let nnz = to_i32(coo.nnz)?;

        // call MUMPS factorize
        unsafe {
            let status = solver_mumps_factorize(
                self.solver,
                // output
                &mut self.effective_ordering,
                &mut self.effective_scaling,
                &mut self.determinant_coefficient,
                &mut self.determinant_exponent,
                // input
                ordering,
                scaling,
                pct_inc_workspace,
                max_work_memory,
                openmp_num_threads,
                // requests
                calc_det,
                verbose,
                // matrix config
                general_symmetric,
                positive_definite,
                ndim,
                nnz,
                // matrix
                coo.indices_i.as_ptr(),
                coo.indices_j.as_ptr(),
                coo.values.as_ptr(),
            );
            if status != MUMPS_SUCCESS {
                return Err(handle_mumps_error_code(status));
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
    /// * `mat` -- the coefficient matrix A; must be square and, if symmetric, [crate::Storage::Lower].
    /// * `rhs` -- the right-hand side vector with know values an dimension equal to mat.nrow
    /// * `verbose` -- shows messages
    ///
    /// **Warning:** the matrix must be same one used in `factorize`.
    fn solve(&mut self, x: &mut Vector, mat: &SparseMatrix, rhs: &Vector, verbose: bool) -> Result<(), StrError> {
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

        // call MUMPS solve
        vec_copy(x, rhs).unwrap();
        let verb = if verbose { 1 } else { 0 };
        unsafe {
            let status = solver_mumps_solve(self.solver, x.as_mut_data().as_mut_ptr(), verb);
            if status != MUMPS_SUCCESS {
                return Err(handle_mumps_error_code(status));
            }
        }
        Ok(())
    }

    /// Returns the determinant
    ///
    /// Returns the three values `(mantissa, 2.0, exponent)`, such that the determinant is calculated by:
    ///
    /// ```text
    /// determinant = mantissa · pow(2.0, exponent)
    /// ```
    ///
    /// **Note:** This is only available if compute_determinant was requested.
    fn get_determinant(&self) -> (f64, f64, f64) {
        (self.determinant_coefficient, 2.0, self.determinant_exponent)
    }

    /// Returns the ordering effectively used by the solver
    fn get_effective_ordering(&self) -> String {
        match self.effective_ordering {
            0 => "Amd".to_string(),
            2 => "Amf".to_string(),
            7 => "Auto".to_string(),
            5 => "Metis".to_string(),
            4 => "Pord".to_string(),
            6 => "Qamd".to_string(),
            3 => "Scotch".to_string(),
            _ => "Unknown".to_string(),
        }
    }

    /// Returns the scaling effectively used by the solver
    fn get_effective_scaling(&self) -> String {
        match self.effective_scaling {
            77 => "Auto".to_string(),
            3 => "Column".to_string(),
            1 => "Diagonal".to_string(),
            0 => "No".to_string(),
            4 => "RowCol".to_string(),
            7 => "RowColIter".to_string(),
            8 => "RowColRig".to_string(),
            _ => "Unknown".to_string(),
        }
    }

    /// Returns the strategy (concerning symmetry) effectively used by the solver
    fn get_effective_strategy(&self) -> String {
        "Unknown".to_string()
    }

    /// Returns the name of this solver
    ///
    /// # Output
    ///
    /// * `MUMPS` -- if the default system MUMPS has been used
    /// * `MUMPS-local` -- if the locally compiled MUMPS has be used
    fn get_name(&self) -> String {
        if cfg!(local_mumps) {
            "MUMPS-local".to_string()
        } else {
            "MUMPS".to_string()
        }
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
        100000 => return "Error: c-code returned null pointer (MUMPS)",
        200000 => return "Error: c-code failed to allocate memory (MUMPS)",
        _ => return "Error: unknown error returned by c-code (MUMPS)",
    }
}

const MUMPS_SUCCESS: i32 = 0;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{handle_mumps_error_code, SolverMUMPS};
    use crate::{LinSolParams, LinSolTrait, Ordering, Samples, Scaling, SparseMatrix};
    use russell_chk::{approx_eq, vec_approx_eq};
    use russell_lab::Vector;

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
        let ordering = solver.get_effective_ordering();
        let scaling = solver.get_effective_scaling();
        assert_eq!(ordering, "Pord");
        assert_eq!(scaling, "No"); // because we requested the determinant

        // check the determinant
        let (a, b, c) = solver.get_determinant();
        let d = a * f64::powf(b, c);
        approx_eq(a, 57.0 / 64.0, 1e-15);
        approx_eq(c, 7.0, 1e-15);
        approx_eq(d, 114.0, 1e-13);

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
            handle_mumps_error_code(100000),
            "Error: c-code returned null pointer (MUMPS)"
        );
        assert_eq!(
            handle_mumps_error_code(200000),
            "Error: c-code failed to allocate memory (MUMPS)"
        );
        assert_eq!(handle_mumps_error_code(123), default);
    }
}
