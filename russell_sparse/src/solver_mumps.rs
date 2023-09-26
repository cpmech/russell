use super::{to_i32, ConfigSolver, CooMatrix, Ordering, Scaling, SolverTrait, Storage, Symmetry};
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
    fn solver_mumps_initialize(
        solver: *mut InterfaceMUMPS,
        n: i32,
        nnz: i32,
        symmetry: i32,
        ordering: i32,
        scaling: i32,
        pct_inc_workspace: i32,
        max_work_memory: i32,
        openmp_num_threads: i32,
        compute_determinant: i32,
    ) -> i32;
    fn solver_mumps_factorize(
        solver: *mut InterfaceMUMPS,
        indices_i: *const i32,
        indices_j: *const i32,
        values_aij: *const f64,
        verbose: i32,
    ) -> i32;
    fn solver_mumps_solve(solver: *mut InterfaceMUMPS, rhs: *mut f64, verbose: i32) -> i32;
    fn solver_mumps_get_ordering(solver: *const InterfaceMUMPS) -> i32;
    fn solver_mumps_get_scaling(solver: *const InterfaceMUMPS) -> i32;
    fn solver_mumps_get_det_coef_a(solver: *const InterfaceMUMPS) -> f64;
    fn solver_mumps_get_det_exp_c(solver: *const InterfaceMUMPS) -> f64;
}

/// Wraps the MUMPS solver for (large) sparse linear systems
///
/// **Warning:** This solver is **not** thread-safe, thus use only use in single-thread applications.
pub struct SolverMUMPS {
    /// Holds a pointer to the C interface to MUMPS
    solver: *mut InterfaceMUMPS,

    /// Symmetry option set by initialize (for consistency checking)
    symmetry: Option<Symmetry>,

    /// Number of rows set by initialize (for consistency checking)
    nrow: i32,

    /// Number of non-zero values set by initialize (for consistency checking)
    nnz: i32,

    /// Indicates whether the C interface has been initialized or not
    initialized: bool,

    /// Indicates whether the sparse matrix has been factorized or not
    factorized: bool,
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
                symmetry: None,
                nrow: 0,
                nnz: 0,
                initialized: false,
                factorized: false,
            })
        }
    }
}

impl SolverTrait for SolverMUMPS {
    /// Initializes the C interface to MUMPS
    ///
    /// # Input
    ///
    /// * `ndim` -- number of rows = number of columns of the coefficient matrix A
    /// * `nnz` -- number of non-zero values on the coefficient matrix A
    /// * `symmetry` -- symmetry (or lack of it) type of the coefficient matrix A
    ///   Note that only symmetry/storage equal to Lower or Full are allowed by MUMPS.
    /// * `config` -- configuration parameters; None => use default
    fn initialize(
        &mut self,
        ndim: usize,
        nnz: usize,
        symmetry: Option<Symmetry>,
        config: Option<ConfigSolver>,
    ) -> Result<(), StrError> {
        let cfg = if let Some(c) = config { c } else { ConfigSolver::new() };
        let sym_i32 = match symmetry {
            Some(sym) => match sym {
                Symmetry::General(storage) => {
                    if storage != Storage::Lower {
                        return Err("if the matrix is symmetric, the storage must be lower triangular");
                    }
                    2 // general symmetric (page 27)
                }
                Symmetry::PositiveDefinite(storage) => {
                    if storage != Storage::Lower {
                        return Err("if the matrix is positive-definite, the storage must be lower triangular");
                    }
                    1 // symmetric positive-definite (page 27)
                }
            },
            None => 0, // unsymmetric (page 27)
        };
        self.symmetry = symmetry;
        self.nrow = to_i32(ndim)?;
        self.nnz = to_i32(nnz)?;
        self.initialized = false;
        self.factorized = false;
        let ordering = match cfg.ordering {
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
        let scaling = match cfg.scaling {
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
        let pct_i32 = to_i32(cfg.mumps_pct_inc_workspace)?;
        let mem_i32 = to_i32(cfg.mumps_max_work_memory)?;
        let nt_i32 = to_i32(cfg.mumps_openmp_num_threads)?;
        let det_i32 = if cfg.compute_determinant { 1 } else { 0 };
        unsafe {
            let status = solver_mumps_initialize(
                self.solver,
                self.nrow,
                self.nnz,
                sym_i32,
                ordering,
                scaling,
                pct_i32,
                mem_i32,
                nt_i32,
                det_i32,
            );
            if status != 0 {
                return Err(handle_mumps_error_code(status));
            }
        }
        self.initialized = true;
        Ok(())
    }

    /// Performs the factorization (and analysis) given COO matrix
    ///
    /// **Note::** Initialize must be called first. Also, the dimension and symmetry/storage
    /// of the CooMatrix must be the same as the ones provided by `initialize`.
    ///
    /// # Input
    ///
    /// * `coo` -- The **same** matrix provided to `initialize`
    /// * `verbose` -- shows messages
    fn factorize_coo(&mut self, coo: &CooMatrix, verbose: bool) -> Result<(), StrError> {
        self.factorized = false;
        if !self.initialized {
            return Err("the function initialize must be called before factorize");
        }
        if !coo.one_based {
            return Err("the COO matrix must have one-based (FORTRAN) indices as required by MUMPS");
        }
        if coo.nrow != self.nrow as usize || coo.ncol != self.nrow as usize {
            return Err("the dimension of the CooMatrix must be the same as the one provided to initialize");
        }
        if coo.pos != self.nnz as usize {
            return Err("the number of non-zero values must be the same as the one provided to initialize");
        }
        if coo.symmetry != self.symmetry {
            return Err("the symmetry/storage of the CooMatrix must be the same as the one used in initialize");
        }
        let verb = if verbose { 1 } else { 0 };
        unsafe {
            let status = solver_mumps_factorize(
                self.solver,
                coo.indices_i.as_ptr(),
                coo.indices_j.as_ptr(),
                coo.values_aij.as_ptr(),
                verb,
            );
            if status != 0 {
                return Err(handle_mumps_error_code(status));
            }
        }
        self.factorized = true;
        Ok(())
    }

    /// Computes the solution of the linear system
    ///
    /// Solves the linear system:
    ///
    /// ```text
    /// A · x = rhs
    /// ```
    ///
    /// # Input
    ///
    /// * `x` -- the vector of unknown values with dimension equal to coo.nrow
    /// * `rhs` -- the right-hand side vector with know values an dimension equal to coo.nrow
    /// * `verbose` -- shows messages
    fn solve(&mut self, x: &mut Vector, rhs: &Vector, verbose: bool) -> Result<(), StrError> {
        if !self.factorized {
            return Err("the function factorize must be called before solve");
        }
        if x.dim() != self.nrow as usize {
            return Err("the dimension of the vector of unknown values x is incorrect");
        }
        if rhs.dim() != self.nrow as usize {
            return Err("the dimension of the right-hand side vector is incorrect");
        }
        let verb = if verbose { 1 } else { 0 };
        vec_copy(x, rhs).unwrap();
        unsafe {
            let status = solver_mumps_solve(self.solver, x.as_mut_data().as_mut_ptr(), verb);
            if status != 0 {
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
        unsafe {
            let a = solver_mumps_get_det_coef_a(self.solver);
            let c = solver_mumps_get_det_exp_c(self.solver);
            (a, 2.0, c)
        }
    }

    /// Returns the ordering effectively used by the solver
    fn get_effective_ordering(&self) -> String {
        unsafe {
            let ordering = solver_mumps_get_ordering(self.solver);
            match ordering {
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
    }

    /// Returns the scaling effectively used by the solver
    fn get_effective_scaling(&self) -> String {
        unsafe {
            let scaling = solver_mumps_get_scaling(self.solver);
            match scaling {
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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{handle_mumps_error_code, SolverMUMPS};
    use crate::{ConfigSolver, CooMatrix, Ordering, Samples, Scaling, SolverTrait, Storage, Symmetry};
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
        assert!(!solver.initialized);
        assert!(!solver.factorized);

        // initialize fails on incorrect symmetry/storage
        assert_eq!(
            solver
                .initialize(1, 1, Some(Symmetry::General(Storage::Upper)), None)
                .err(),
            Some("if the matrix is symmetric, the storage must be lower triangular")
        );
        assert_eq!(
            solver
                .initialize(1, 1, Some(Symmetry::PositiveDefinite(Storage::Upper)), None)
                .err(),
            Some("if the matrix is positive-definite, the storage must be lower triangular")
        );

        // sample matrix
        let (coo, _, _, _) = Samples::umfpack_unsymmetric_5x5(true);

        // factorize requests initialize
        assert_eq!(
            solver.factorize_coo(&coo, false).err(),
            Some("the function initialize must be called before factorize")
        );

        // set params
        let mut config = ConfigSolver::new();
        config.ordering = Ordering::Pord;
        config.scaling = Scaling::RowCol;
        config.compute_determinant = true;

        // initialize works
        solver
            .initialize(coo.nrow, coo.pos, coo.symmetry, Some(config))
            .unwrap();
        assert!(solver.initialized);

        // factorize fails on incompatible coo matrix
        let sym = Some(Symmetry::General(Storage::Lower));
        let mut coo_wrong_1 = CooMatrix::new(1, 5, 13, None, true).unwrap();
        let coo_wrong_2 = CooMatrix::new(5, 1, 13, None, true).unwrap();
        let coo_wrong_3 = CooMatrix::new(5, 5, 12, None, true).unwrap();
        let mut coo_wrong_4 = CooMatrix::new(5, 5, 13, sym, true).unwrap();
        for _ in 0..13 {
            coo_wrong_1.put(0, 0, 1.0).unwrap();
            coo_wrong_4.put(0, 0, 1.0).unwrap();
        }
        assert_eq!(
            solver.factorize_coo(&coo_wrong_1, false).err(),
            Some("the dimension of the CooMatrix must be the same as the one provided to initialize")
        );
        assert_eq!(
            solver.factorize_coo(&coo_wrong_2, false).err(),
            Some("the dimension of the CooMatrix must be the same as the one provided to initialize")
        );
        assert_eq!(
            solver.factorize_coo(&coo_wrong_3, false).err(),
            Some("the number of non-zero values must be the same as the one provided to initialize")
        );
        assert_eq!(
            solver.factorize_coo(&coo_wrong_4, false).err(),
            Some("the symmetry/storage of the CooMatrix must be the same as the one used in initialize")
        );

        // solve fails on non-factorized system
        assert_eq!(
            solver.solve(&mut x, &rhs, false),
            Err("the function factorize must be called before solve")
        );

        // factorize works
        solver.factorize_coo(&coo, false).unwrap();
        assert!(solver.factorized);

        // solve fails on wrong x and rhs vectors
        let mut x_wrong = Vector::new(3);
        let rhs_wrong = Vector::new(2);
        assert_eq!(
            solver.solve(&mut x_wrong, &rhs, false),
            Err("the dimension of the vector of unknown values x is incorrect")
        );
        assert_eq!(
            solver.solve(&mut x, &rhs_wrong, false),
            Err("the dimension of the right-hand side vector is incorrect")
        );

        // solve works
        solver.solve(&mut x, &rhs, false).unwrap();
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
        solver.solve(&mut x_again, &rhs, false).unwrap();
        vec_approx_eq(x_again.as_data(), x_correct, 1e-14);

        // factorize fails on singular matrix
        let mut coo_singular = CooMatrix::new(5, 5, 2, None, true).unwrap();
        coo_singular.put(0, 0, 1.0).unwrap();
        coo_singular.put(4, 4, 1.0).unwrap();
        let mut solver = SolverMUMPS::new().unwrap();
        solver
            .initialize(coo_singular.nrow, coo_singular.pos, coo_singular.symmetry, Some(config))
            .unwrap();
        assert_eq!(
            solver.factorize_coo(&coo_singular, false),
            Err("Error(-10): numerically singular matrix")
        );

        // solve with positive-definite matrix works
        let (coo_pd_lower, _, _, _) = Samples::mkl_sample1_positive_definite_lower(true);
        config.ordering = Ordering::Auto;
        config.scaling = Scaling::Auto;
        let mut solver = SolverMUMPS::new().unwrap();
        assert!(!solver.initialized);
        assert!(!solver.factorized);
        solver
            .initialize(coo_pd_lower.nrow, coo_pd_lower.pos, coo_pd_lower.symmetry, Some(config))
            .unwrap();
        solver.factorize_coo(&coo_pd_lower, false).unwrap();
        let mut x = Vector::new(5);
        let rhs = Vector::from(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        solver.solve(&mut x, &rhs, false).unwrap();
        let x_correct = &[-979.0 / 3.0, 983.0, 1961.0 / 12.0, 398.0, 123.0 / 2.0];
        vec_approx_eq(x.as_data(), x_correct, 1e-10);
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
