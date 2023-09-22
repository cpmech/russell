use super::{to_i32, CooMatrix, Layout, Ordering, Scaling};
use crate::StrError;
use russell_lab::{vec_copy, Vector};

// Representing opaque struct
// Reference: https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs
#[repr(C)]
pub(crate) struct InterfaceMUMPS {
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

    /// Layout set by initialize (for consistency checking)
    layout: Layout,

    /// Number of rows set by initialize (for consistency checking)
    nrow: i32,

    /// Number of non-zero values set by initialize (for consistency checking)
    nnz: i32,

    /// Indicates whether the C interface has been initialized or not
    initialized: bool,

    /// Indicates whether the sparse matrix has been factorized or not
    factorized: bool,

    /// Defines the symmetric permutation (ordering)
    pub ordering: Ordering,

    /// Defines the scaling strategy
    pub scaling: Scaling,

    /// Sets the % increase in the estimated working space
    ///
    /// **Note:** The default (recommended) value is 100 (%)
    pub pct_inc_workspace: usize,

    /// Sets the max size of the working memory in mega bytes
    ///
    /// **Note:** Set this value to 0 for an automatic configuration
    pub max_work_memory: usize,

    /// Defines the number of OpenMP threads
    ///
    /// **Note:** Set this value to 0 to allow an automatic detection
    pub openmp_num_threads: usize,

    /// Requests that the determinant be computed
    ///
    /// **Note:** The determinant will be available after `factorize`
    pub compute_determinant: bool,
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
                layout: Layout::Full,
                nrow: 0,
                nnz: 0,
                initialized: false,
                factorized: false,
                ordering: Ordering::Auto,
                scaling: Scaling::Auto,
                pct_inc_workspace: 100,
                max_work_memory: 0,
                openmp_num_threads: 0,
                compute_determinant: false,
            })
        }
    }

    /// Initializes the C interface to MUMPS
    ///
    /// # Input
    ///
    /// * `coo` -- the CooMatrix representing the sparse coefficient matrix
    ///   Note that only coo.Layout equal to Lower or Full are allowed by MUMPS.
    /// * `positive_definite` -- Wether the solver should treat the coefficient matrix as positive-definite or not.
    ///   Note that this will only be considered if coo.layout is Lower
    pub fn initialize(&mut self, coo: &CooMatrix, positive_definite: bool) -> Result<(), StrError> {
        if coo.layout == Layout::Upper {
            return Err("if the matrix is symmetric, the layout must be lower triangular");
        }
        if positive_definite && coo.layout != Layout::Lower {
            return Err("if positive definite is true, the layout must be lower triangular");
        }
        self.nrow = to_i32(coo.nrow)?;
        self.nnz = to_i32(coo.pos)?;
        self.initialized = false;
        self.factorized = false;
        let symmetry = if coo.layout == Layout::Lower {
            if positive_definite {
                1 // symmetric positive-definite (page 27)
            } else {
                2 // general symmetric (page 27)
            }
        } else {
            0 // unsymmetric (page 27)
        };
        let ordering = match self.ordering {
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
        let scaling = match self.scaling {
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
        let pct_i32 = to_i32(self.pct_inc_workspace)?;
        let mem_i32 = to_i32(self.max_work_memory)?;
        let nt_i32 = to_i32(self.openmp_num_threads)?;
        let det_i32 = if self.compute_determinant { 1 } else { 0 };
        unsafe {
            let status = solver_mumps_initialize(
                self.solver,
                self.nrow,
                self.nnz,
                symmetry,
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

    /// Performs the factorization (and analysis)
    ///
    /// **Note::** Initialize must be called first. Also, the dimension and layout of the
    /// CooMatrix must be the same as the ones provided by `initialize`.
    ///
    /// # Input
    ///
    /// * `coo` -- The **same** matrix provided to `initialize`
    pub fn factorize(&mut self, coo: &CooMatrix, verbose: bool) -> Result<(), StrError> {
        self.factorized = false;
        if !self.initialized {
            return Err("the function initialize must be called before factorize");
        }
        if coo.nrow != self.nrow as usize || coo.ncol != self.nrow as usize {
            return Err("the dimension of the CooMatrix must be the same as the one provided to initialize");
        }
        if coo.pos != self.nnz as usize {
            return Err("the number of non-zero values must be the same as the one provided to initialize");
        }
        if coo.layout != self.layout {
            return Err("the layout of the CooMatrix must be the same as the one used in initialize");
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
    pub fn solve(&mut self, x: &mut Vector, rhs: &Vector, verbose: bool) -> Result<(), StrError> {
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

    /// Returns the ordering effectively used by the solver
    pub fn get_effective_ordering(&self) -> String {
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
    pub fn get_effective_scaling(&self) -> String {
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

    /// Returns the coefficient and exponent of the determinant
    ///
    /// **Note:** This is only available if compute_determinant was requested.
    ///
    /// Returns `(a, c)`, such that
    ///
    /// ```text
    /// determinant = a * 2^c
    /// ```
    pub fn get_determinant(&self) -> (f64, f64) {
        if self.compute_determinant {
            unsafe {
                let a = solver_mumps_get_det_coef_a(self.solver);
                let c = solver_mumps_get_det_exp_c(self.solver);
                (a, c)
            }
        } else {
            (0.0, 0.0)
        }
    }

    /// Returns the name of this solver
    ///
    /// # Output
    ///
    /// * `MUMPS` -- if the default system MUMPS has been used
    /// * `MUMPS-local` -- if the locally compiled MUMPS has be used
    pub fn get_name() -> String {
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
    use crate::{CooMatrix, Layout, Ordering, Scaling};
    use russell_chk::{approx_eq, vec_approx_eq};
    use russell_lab::{mat_inverse, Matrix, Vector};

    #[test]
    fn complete_solution_cycle_works() {
        // IMPORTANT:
        // Since MUMPS is not thread-safe, we need to call all MUMPS functions
        // in a single test unit because the tests are run in parallel by default

        // allocate x and rhs
        let mut x = Vector::new(5);
        let rhs = Vector::from(&[8.0, 45.0, -3.0, 3.0, 19.0]);
        let x_correct = &[1.0, 2.0, 3.0, 4.0, 5.0];

        // allocate the CooMatrix
        let mut coo = CooMatrix::new(Layout::Full, 5, 5, 13).unwrap();
        coo.put(0, 0, 1.0).unwrap(); // << (0, 0, a00/2) duplicate
        coo.put(0, 0, 1.0).unwrap(); // << (0, 0, a00/2) duplicate
        coo.put(1, 0, 3.0).unwrap();
        coo.put(0, 1, 3.0).unwrap();
        coo.put(2, 1, -1.0).unwrap();
        coo.put(4, 1, 4.0).unwrap();
        coo.put(1, 2, 4.0).unwrap();
        coo.put(2, 2, -3.0).unwrap();
        coo.put(3, 2, 1.0).unwrap();
        coo.put(4, 2, 2.0).unwrap();
        coo.put(2, 3, 2.0).unwrap();
        coo.put(1, 4, 6.0).unwrap();
        coo.put(4, 4, 1.0).unwrap();

        // check the determinant of the CooMatrix
        let mat = coo.as_matrix();
        let mut inv = Matrix::new(5, 5);
        let det = mat_inverse(&mut inv, &mat).unwrap();
        approx_eq(det, 114.0, 1e-15);

        // upper triangular
        let mut coo_upper = CooMatrix::new(Layout::Upper, 5, 5, 9).unwrap();
        coo_upper.put(0, 0, 9.0).unwrap();
        coo_upper.put(0, 1, 1.5).unwrap();
        coo_upper.put(1, 1, 0.5).unwrap();
        coo_upper.put(0, 2, 6.0).unwrap();
        coo_upper.put(2, 2, 12.0).unwrap();
        coo_upper.put(0, 3, 0.75).unwrap();
        coo_upper.put(3, 3, 0.625).unwrap();
        coo_upper.put(0, 4, 3.0).unwrap();
        coo_upper.put(4, 4, 16.0).unwrap();

        // allocate a new solver
        let mut solver = SolverMUMPS::new().unwrap();
        solver.ordering = Ordering::Pord;
        solver.scaling = Scaling::RowCol;
        solver.compute_determinant = true;
        assert!(!solver.initialized);
        assert!(!solver.factorized);

        // initialize fails on incorrect layout
        assert_eq!(
            solver.initialize(&coo_upper, false).err(),
            Some("if the matrix is symmetric, the layout must be lower triangular")
        );
        assert_eq!(
            solver.initialize(&coo, true).err(),
            Some("if positive definite is true, the layout must be lower triangular")
        );

        // factorize requests initialize
        assert_eq!(
            solver.factorize(&coo, false).err(),
            Some("the function initialize must be called before factorize")
        );

        // initialize works
        solver.initialize(&coo, false).unwrap();
        assert!(solver.initialized);

        // factorize fails on incompatible coo matrix
        let mut coo_wrong_1 = CooMatrix::new(Layout::Full, 1, 5, 13).unwrap();
        let coo_wrong_2 = CooMatrix::new(Layout::Full, 5, 1, 13).unwrap();
        let coo_wrong_3 = CooMatrix::new(Layout::Full, 5, 5, 12).unwrap();
        let mut coo_wrong_4 = CooMatrix::new(Layout::Lower, 5, 5, 13).unwrap();
        for _ in 0..13 {
            coo_wrong_1.put(0, 0, 1.0).unwrap();
            coo_wrong_4.put(0, 0, 1.0).unwrap();
        }
        assert_eq!(
            solver.factorize(&coo_wrong_1, false).err(),
            Some("the dimension of the CooMatrix must be the same as the one provided to initialize")
        );
        assert_eq!(
            solver.factorize(&coo_wrong_2, false).err(),
            Some("the dimension of the CooMatrix must be the same as the one provided to initialize")
        );
        assert_eq!(
            solver.factorize(&coo_wrong_3, false).err(),
            Some("the number of non-zero values must be the same as the one provided to initialize")
        );
        assert_eq!(
            solver.factorize(&coo_wrong_4, false).err(),
            Some("the layout of the CooMatrix must be the same as the one used in initialize")
        );

        // solve fails on non-factorized system
        assert_eq!(
            solver.solve(&mut x, &rhs, false),
            Err("the function factorize must be called before solve")
        );

        // factorize works
        solver.factorize(&coo, false).unwrap();
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
        vec_approx_eq(x.as_data(), x_correct, 1e-15);

        // check ordering and scaling
        let ordering = solver.get_effective_ordering();
        let scaling = solver.get_effective_scaling();
        assert_eq!(ordering, "Pord");
        assert_eq!(scaling, "No"); // because we requested the determinant

        // check the determinant
        let (a, c) = solver.get_determinant();
        let d = a * f64::powf(2.0, c);
        approx_eq(a, 57.0 / 64.0, 1e-15);
        approx_eq(c, 7.0, 1e-15);
        approx_eq(d, 114.0, 1e-15);

        // calling solve again works
        let mut x_again = Vector::new(5);
        solver.solve(&mut x_again, &rhs, false).unwrap();
        vec_approx_eq(x_again.as_data(), x_correct, 1e-15);

        // factorize fails on singular matrix
        let mut coo_singular = CooMatrix::new(Layout::Full, 5, 5, 2).unwrap();
        coo_singular.put(0, 0, 1.0).unwrap();
        coo_singular.put(4, 4, 1.0).unwrap();
        let mut solver = SolverMUMPS::new().unwrap();
        solver.initialize(&coo_singular, false).unwrap();
        assert_eq!(
            solver.factorize(&coo_singular, false),
            Err("Error(-10): numerically singular matrix")
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
