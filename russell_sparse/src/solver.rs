use super::*;
use russell_lab::*;
use std::fmt;

#[repr(C)]
pub(crate) struct ExtSolver {
    data: [u8; 0],
    marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

extern "C" {
    // MMP
    fn new_solver_mmp(symmetry: i32) -> *mut ExtSolver;
    fn drop_solver_mmp(solver: *mut ExtSolver);
    fn solver_mmp_initialize(
        solver: *mut ExtSolver,
        n: i32,
        nnz: i32,
        indices_i: *const i32,
        indices_j: *const i32,
        values_a: *const f64,
        ordering: i32,
        scaling: i32,
        pct_inc_workspace: i32,
        max_work_memory: i32,
        openmp_num_threads: i32,
        verbose: i32,
    ) -> i32;
    fn solver_mmp_factorize(solver: *mut ExtSolver, verbose: i32) -> i32;
    fn solver_mmp_solve(solver: *mut ExtSolver, rhs: *mut f64, verbose: i32) -> i32;

    // UMF
    fn new_solver_umf(symmetry: i32) -> *mut ExtSolver;
    fn drop_solver_umf(solver: *mut ExtSolver);
    fn solver_umf_initialize(
        solver: *mut ExtSolver,
        n: i32,
        nnz: i32,
        indices_i: *const i32,
        indices_j: *const i32,
        values_a: *const f64,
        ordering: i32,
        scaling: i32,
        verbose: i32,
    ) -> i32;
    fn solver_umf_factorize(solver: *mut ExtSolver, verbose: i32) -> i32;
    fn solver_umf_solve(solver: *mut ExtSolver, x: *mut f64, rhs: *const f64, verbose: i32) -> i32;
}

/// Implements a sparse Solver
pub struct Solver {
    config: ConfigSolver,   // configuration
    done_initialize: bool,  // initialization completed
    done_factorize: bool,   // factorization completed
    ndim: usize,            // number of equations == nrow(a) where a*x=rhs
    solver: *mut ExtSolver, // data allocated by the c-code
}

impl Solver {
    /// Creates a new solver
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), &'static str> {
    /// use russell_sparse::*;
    /// let config = ConfigSolver::new();
    /// let solver = Solver::new(config)?;
    /// let correct: &str = "solver_kind        = UMF\n\
    ///                      symmetry           = No\n\
    ///                      ordering           = Auto\n\
    ///                      scaling            = Auto\n\
    ///                      done_initialize    = false\n\
    ///                      done_factorize     = false\n\
    ///                      ndim               = 0\n";
    /// assert_eq!(format!("{}", solver), correct);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(config: ConfigSolver) -> Result<Self, &'static str> {
        unsafe {
            let solver = match config.solver_kind {
                EnumSolverKind::Mmp => new_solver_mmp(config.symmetry),
                EnumSolverKind::Umf => new_solver_umf(config.symmetry),
            };
            if solver.is_null() {
                return Err("c-code failed to allocate solver");
            }
            Ok(Solver {
                config,
                done_initialize: false,
                done_factorize: false,
                ndim: 0,
                solver,
            })
        }
    }

    /// Initializes the solver
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), &'static str> {
    /// use russell_sparse::*;
    /// let config = ConfigSolver::new();
    /// let mut solver = Solver::new(config)?;
    /// let mut trip = SparseTriplet::new(2, 2, 2, false)?;
    /// trip.put(0, 0, 1.0);
    /// trip.put(1, 1, 1.0);
    /// solver.initialize(&trip, false)?;
    /// let correct: &str = "solver_kind        = UMF\n\
    ///                      symmetry           = No\n\
    ///                      ordering           = Auto\n\
    ///                      scaling            = Auto\n\
    ///                      done_initialize    = true\n\
    ///                      done_factorize     = false\n\
    ///                      ndim               = 2\n";
    /// assert_eq!(format!("{}", solver), correct);
    /// # Ok(())
    /// # }
    /// ```
    pub fn initialize(&mut self, trip: &SparseTriplet, verbose: bool) -> Result<(), &'static str> {
        if trip.nrow != trip.ncol {
            return Err("the matrix represented by the triplet must be square");
        }
        let n = to_i32(trip.nrow);
        let nnz = to_i32(trip.pos);
        let verb: i32 = if verbose { 1 } else { 0 };
        unsafe {
            match self.config.solver_kind {
                EnumSolverKind::Mmp => {
                    if self.done_initialize {
                        drop_solver_mmp(self.solver);
                        self.solver = new_solver_mmp(self.config.symmetry);
                        if self.solver.is_null() {
                            return Err("c-code failed to reallocate solver");
                        }
                    }
                    let res = solver_mmp_initialize(
                        self.solver,
                        n,
                        nnz,
                        trip.indices_i.as_ptr(),
                        trip.indices_j.as_ptr(),
                        trip.values_a.as_ptr(),
                        self.config.ordering,
                        self.config.scaling,
                        self.config.pct_inc_workspace,
                        self.config.max_work_memory,
                        self.config.openmp_num_threads,
                        verb,
                    );
                    if res != 0 {
                        return Err(self.handle_mmp_error_code(res));
                    }
                }
                EnumSolverKind::Umf => {
                    if self.done_initialize {
                        drop_solver_umf(self.solver);
                        self.solver = new_solver_umf(self.config.symmetry);
                        if self.solver.is_null() {
                            return Err("c-code failed to reallocate solver");
                        }
                    }
                    let res = solver_umf_initialize(
                        self.solver,
                        n,
                        nnz,
                        trip.indices_i.as_ptr(),
                        trip.indices_j.as_ptr(),
                        trip.values_a.as_ptr(),
                        self.config.ordering,
                        self.config.scaling,
                        verb,
                    );
                    if res != 0 {
                        return Err(self.handle_umf_error_code(res));
                    }
                }
            }
            self.done_initialize = true;
            self.ndim = trip.nrow;
        }
        Ok(())
    }

    /// Performs the factorization
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), &'static str> {
    /// use russell_sparse::*;
    /// let config = ConfigSolver::new();
    /// let mut solver = Solver::new(config)?;
    /// let mut trip = SparseTriplet::new(2, 2, 2, false)?;
    /// trip.put(0, 0, 1.0);
    /// trip.put(1, 1, 1.0);
    /// solver.initialize(&trip, false)?;
    /// solver.factorize(false)?;
    /// let correct: &str = "solver_kind        = UMF\n\
    ///                      symmetry           = No\n\
    ///                      ordering           = Auto\n\
    ///                      scaling            = Auto\n\
    ///                      done_initialize    = true\n\
    ///                      done_factorize     = true\n\
    ///                      ndim               = 2\n";
    /// assert_eq!(format!("{}", solver), correct);
    /// # Ok(())
    /// # }
    /// ```
    pub fn factorize(&mut self, verbose: bool) -> Result<(), &'static str> {
        if !self.done_initialize {
            return Err("initialization must be done before factorization");
        }
        let verb: i32 = if verbose { 1 } else { 0 };
        unsafe {
            match self.config.solver_kind {
                EnumSolverKind::Mmp => {
                    let res = solver_mmp_factorize(self.solver, verb);
                    if res != 0 {
                        return Err(self.handle_mmp_error_code(res));
                    }
                }
                EnumSolverKind::Umf => {
                    let res = solver_umf_factorize(self.solver, verb);
                    if res != 0 {
                        return Err(self.handle_umf_error_code(res));
                    }
                }
            }
            self.done_factorize = true;
        }
        Ok(())
    }

    /// Computes the solution
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), &'static str> {
    /// use russell_lab::*;
    /// use russell_sparse::*;
    ///
    /// // allocate a square matrix
    /// let mut trip = SparseTriplet::new(5, 5, 13, false)?;
    /// trip.put(0, 0, 1.0); // << (0, 0, a00/2)
    /// trip.put(0, 0, 1.0); // << (0, 0, a00/2)
    /// trip.put(1, 0, 3.0);
    /// trip.put(0, 1, 3.0);
    /// trip.put(2, 1, -1.0);
    /// trip.put(4, 1, 4.0);
    /// trip.put(1, 2, 4.0);
    /// trip.put(2, 2, -3.0);
    /// trip.put(3, 2, 1.0);
    /// trip.put(4, 2, 2.0);
    /// trip.put(2, 3, 2.0);
    /// trip.put(1, 4, 6.0);
    /// trip.put(4, 4, 1.0);
    ///
    /// // print matrix
    /// let (m, n) = trip.dims();
    /// let mut a = Matrix::new(m, n);
    /// trip.to_matrix(&mut a)?;
    /// let correct = "┌                ┐\n\
    ///                │  2  3  0  0  0 │\n\
    ///                │  3  0  4  0  6 │\n\
    ///                │  0 -1 -3  2  0 │\n\
    ///                │  0  0  1  0  0 │\n\
    ///                │  0  4  2  0  1 │\n\
    ///                └                ┘";
    /// assert_eq!(format!("{}", a), correct);
    ///
    /// // allocate x and rhs
    /// let mut x = Vector::new(5);
    /// let rhs = Vector::from(&[8.0, 45.0, -3.0, 3.0, 19.0]);
    /// let x_correct = &[1.0, 2.0, 3.0, 4.0, 5.0];
    ///
    /// // initialize, factorize, and solve
    /// let config = ConfigSolver::new();
    /// let mut solver = Solver::new(config)?;
    /// solver.initialize(&trip, false)?;
    /// solver.factorize(false)?;
    /// solver.solve(&mut x, &rhs, false)?;
    /// let correct = "┌          ┐\n\
    ///                │ 1.000000 │\n\
    ///                │ 2.000000 │\n\
    ///                │ 3.000000 │\n\
    ///                │ 4.000000 │\n\
    ///                │ 5.000000 │\n\
    ///                └          ┘";
    /// assert_eq!(format!("{:.6}", x), correct);
    /// # Ok(())
    /// # }
    /// ```
    pub fn solve(&mut self, x: &mut Vector, rhs: &Vector, verbose: bool) -> Result<(), &'static str> {
        if !self.done_factorize {
            return Err("factorization must be done before solution");
        }
        if x.dim() != self.ndim || rhs.dim() != self.ndim {
            return Err("x.ndim() and rhs.ndim() must equal the number of equations");
        }
        let verb: i32 = if verbose { 1 } else { 0 };
        unsafe {
            match self.config.solver_kind {
                EnumSolverKind::Mmp => {
                    copy_vector(x, rhs)?;
                    let res = solver_mmp_solve(self.solver, x.as_mut_data().as_mut_ptr(), verb);
                    if res != 0 {
                        return Err(self.handle_mmp_error_code(res));
                    }
                }
                EnumSolverKind::Umf => {
                    let res = solver_umf_solve(self.solver, x.as_mut_data().as_mut_ptr(), rhs.as_data().as_ptr(), verb);
                    if res != 0 {
                        return Err(self.handle_umf_error_code(res));
                    }
                }
            }
        }
        Ok(())
    }

    /// Handles error code
    fn handle_mmp_error_code(&self, err: i32) -> &'static str {
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
            100000 => return "Error: c-code returned null pointer (MMP)",
            200000 => return "Error: c-code failed to allocate memory (MMP)",
            _ => return "Error: unknown error returned by c-code (MMP)",
        }
    }

    /// Handles UMF error code
    fn handle_umf_error_code(&self, err: i32) -> &'static str {
        match err {
            1 => return "Error(1): Matrix is singular",
            2 => return "Error(2): The determinant is nonzero, but smaller than allowed",
            3 => return "Error(3): The determinant is larger than allowed",
            -1 => return "Error(-1): Not enough memory",
            -3 => return "Error(-3): Invalid numeric object",
            -4 => return "Error(-4): Invalid symbolic object",
            -5 => return "Error(-5): Argument missing",
            -6 => return "Error(-6): Nrow or ncol must be greater than zero",
            -8 => return "Error(-8): Invalid matrix",
            -11 => return "Error(-11): Different pattern",
            -13 => return "Error(-13): Invalid system",
            -15 => return "Error(-15): Invalid permutation",
            -17 => return "Error(-17): Failed to save/load file",
            -18 => return "Error(-18): Ordering method failed",
            -911 => return "Error(-911): An internal error has occurred",
            100000 => return "Error: c-code returned null pointer (UMF)",
            200000 => return "Error: c-code failed to allocate memory (UMF)",
            _ => return "Error: unknown error returned by c-code (UMF)",
        }
    }
}

impl Drop for Solver {
    /// Tells the c-code to release memory
    fn drop(&mut self) {
        unsafe {
            match self.config.solver_kind {
                EnumSolverKind::Mmp => drop_solver_mmp(self.solver),
                EnumSolverKind::Umf => drop_solver_umf(self.solver),
            }
        }
    }
}

impl fmt::Display for Solver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}\
             done_initialize    = {}\n\
             done_factorize     = {}\n\
             ndim               = {}\n",
            self.config, self.done_initialize, self.done_factorize, self.ndim,
        )?;
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use russell_chk::*;

    #[test]
    fn new_works() -> Result<(), &'static str> {
        let config = ConfigSolver::new();
        let solver = Solver::new(config)?;
        assert_eq!(solver.solver.is_null(), false);
        Ok(())
    }

    #[test]
    fn display_trait_works() -> Result<(), &'static str> {
        let config = ConfigSolver::new();
        let solver = Solver::new(config)?;
        let correct: &str = "solver_kind        = UMF\n\
                             symmetry           = No\n\
                             ordering           = Auto\n\
                             scaling            = Auto\n\
                             done_initialize    = false\n\
                             done_factorize     = false\n\
                             ndim               = 0\n";
        assert_eq!(format!("{}", solver), correct);
        Ok(())
    }

    #[test]
    fn initialize_fails_on_rect_matrix() -> Result<(), &'static str> {
        let config = ConfigSolver::new();
        let mut solver = Solver::new(config)?;
        let trip_rect = SparseTriplet::new(3, 2, 1, false)?;
        assert_eq!(
            solver.initialize(&trip_rect, false),
            Err("the matrix represented by the triplet must be square")
        );
        Ok(())
    }

    #[test]
    fn initialize_works() -> Result<(), &'static str> {
        let config = ConfigSolver::new();
        let mut solver = Solver::new(config)?;
        let mut trip = SparseTriplet::new(2, 2, 2, false)?;
        trip.put(0, 0, 1.0);
        trip.put(1, 1, 1.0);
        solver.initialize(&trip, false)?;
        assert!(solver.done_initialize);
        Ok(())
    }

    #[test]
    fn factorize_fails_on_non_initialized() -> Result<(), &'static str> {
        let config = ConfigSolver::new();
        let mut solver = Solver::new(config)?;
        assert_eq!(
            solver.factorize(false),
            Err("initialization must be done before factorization")
        );
        Ok(())
    }

    #[test]
    fn factorize_fails_on_singular_matrix() -> Result<(), &'static str> {
        let config = ConfigSolver::new();
        let mut solver = Solver::new(config)?;
        let mut trip = SparseTriplet::new(2, 2, 2, false)?;
        trip.put(0, 0, 1.0);
        trip.put(1, 1, 0.0);
        solver.initialize(&trip, false)?;
        assert_eq!(solver.factorize(false), Err("Error(1): Matrix is singular"));
        Ok(())
    }

    #[test]
    fn factorize_works() -> Result<(), &'static str> {
        let config = ConfigSolver::new();
        let mut solver = Solver::new(config)?;
        let mut trip = SparseTriplet::new(2, 2, 2, false)?;
        trip.put(0, 0, 1.0);
        trip.put(1, 1, 1.0);
        solver.initialize(&trip, false)?;
        solver.factorize(false)?;
        assert!(solver.done_factorize);
        Ok(())
    }

    #[test]
    fn solve_fails_on_non_factorized() -> Result<(), &'static str> {
        let config = ConfigSolver::new();
        let mut solver = Solver::new(config)?;
        let mut trip = SparseTriplet::new(2, 2, 2, false)?;
        trip.put(0, 0, 1.0);
        trip.put(1, 1, 1.0);
        solver.initialize(&trip, false)?;
        let mut x = Vector::new(2);
        let rhs = Vector::from(&[1.0, 1.0]);
        assert_eq!(
            solver.solve(&mut x, &rhs, false),
            Err("factorization must be done before solution")
        );
        Ok(())
    }

    #[test]
    fn solve_fails_on_wrong_vectors() -> Result<(), &'static str> {
        let config = ConfigSolver::new();
        let mut solver = Solver::new(config)?;
        let mut trip = SparseTriplet::new(2, 2, 2, false)?;
        trip.put(0, 0, 1.0);
        trip.put(1, 1, 1.0);
        solver.initialize(&trip, false)?;
        solver.factorize(false)?;
        let mut x = Vector::new(2);
        let rhs = Vector::from(&[1.0, 1.0]);
        let mut x_wrong = Vector::new(1);
        let rhs_wrong = Vector::from(&[1.0]);
        assert_eq!(
            solver.solve(&mut x_wrong, &rhs, false),
            Err("x.ndim() and rhs.ndim() must equal the number of equations")
        );
        assert_eq!(
            solver.solve(&mut x, &rhs_wrong, false),
            Err("x.ndim() and rhs.ndim() must equal the number of equations")
        );
        Ok(())
    }

    #[test]
    fn solve_works() -> Result<(), &'static str> {
        let config = ConfigSolver::new();
        let mut solver = Solver::new(config)?;

        // allocate a square matrix
        let mut trip = SparseTriplet::new(5, 5, 13, false)?;
        trip.put(0, 0, 1.0); // << (0, 0, a00/2)
        trip.put(0, 0, 1.0); // << (0, 0, a00/2)
        trip.put(1, 0, 3.0);
        trip.put(0, 1, 3.0);
        trip.put(2, 1, -1.0);
        trip.put(4, 1, 4.0);
        trip.put(1, 2, 4.0);
        trip.put(2, 2, -3.0);
        trip.put(3, 2, 1.0);
        trip.put(4, 2, 2.0);
        trip.put(2, 3, 2.0);
        trip.put(1, 4, 6.0);
        trip.put(4, 4, 1.0);

        // allocate x and rhs
        let mut x = Vector::new(5);
        let rhs = Vector::from(&[8.0, 45.0, -3.0, 3.0, 19.0]);
        let x_correct = &[1.0, 2.0, 3.0, 4.0, 5.0];

        // initialize, factorize, and solve
        solver.initialize(&trip, false)?;
        solver.factorize(false)?;
        solver.solve(&mut x, &rhs, false)?;

        // check
        assert_vec_approx_eq!(x.as_data(), x_correct, 1e-14);
        Ok(())
    }

    #[test]
    fn reinitialize_works() -> Result<(), &'static str> {
        let config = ConfigSolver::new();
        let mut solver = Solver::new(config)?;
        let mut trip = SparseTriplet::new(2, 2, 2, false)?;
        trip.put(0, 0, 1.0);
        trip.put(1, 1, 1.0);
        solver.initialize(&trip, false)?;
        solver.initialize(&trip, false)?;
        Ok(())
    }

    // This function tests many behaviors of the MMP solver.
    // All of these calls must be in a single function because the
    // MMP solver is NOT thread-safe.
    #[test]
    fn solver_mmp_behaves_as_expected() -> Result<(), &'static str> {
        // allocate a new solver
        let mut config = ConfigSolver::new();
        config.set_solver_kind(EnumSolverKind::Mmp);
        let mut solver = Solver::new(config)?;

        // initialize fails on rectangular matrix
        let trip_rect = SparseTriplet::new(3, 2, 1, false)?;
        assert_eq!(
            solver.initialize(&trip_rect, false),
            Err("the matrix represented by the triplet must be square")
        );

        // factorize fails on non-initialized solver
        assert_eq!(
            solver.factorize(false),
            Err("initialization must be done before factorization")
        );

        // allocate a square matrix
        let mut trip = SparseTriplet::new(5, 5, 13, false)?;
        trip.put(0, 0, 1.0); // << (0, 0, a00/2)
        trip.put(0, 0, 1.0); // << (0, 0, a00/2)
        trip.put(1, 0, 3.0);
        trip.put(0, 1, 3.0);
        trip.put(2, 1, -1.0);
        trip.put(4, 1, 4.0);
        trip.put(1, 2, 4.0);
        trip.put(2, 2, -3.0);
        trip.put(3, 2, 1.0);
        trip.put(4, 2, 2.0);
        trip.put(2, 3, 2.0);
        trip.put(1, 4, 6.0);
        trip.put(4, 4, 1.0);

        // allocate x and rhs
        let mut x = Vector::new(5);
        let rhs = Vector::from(&[8.0, 45.0, -3.0, 3.0, 19.0]);
        let x_correct = &[1.0, 2.0, 3.0, 4.0, 5.0];

        // initialize works
        solver.initialize(&trip, false)?;
        assert!(solver.done_initialize);

        // solve fails on non-factorized system
        assert_eq!(
            solver.solve(&mut x, &rhs, false),
            Err("factorization must be done before solution")
        );

        // factorize works
        solver.factorize(false)?;
        assert!(solver.done_factorize);

        // solve fails on wrong x vector
        let mut x_wrong = Vector::new(3);
        assert_eq!(
            solver.solve(&mut x_wrong, &rhs, false),
            Err("x.ndim() and rhs.ndim() must equal the number of equations")
        );

        // solve fails on wrong rhs vector
        let rhs_wrong = Vector::from(&[1.0]);
        assert_eq!(
            solver.solve(&mut x, &rhs_wrong, false),
            Err("x.ndim() and rhs.ndim() must equal the number of equations")
        );

        // solve works
        solver.solve(&mut x, &rhs, false)?;
        assert_vec_approx_eq!(x.as_data(), x_correct, 1e-14);

        // calling solve again works
        let mut x_again = Vector::new(5);
        solver.solve(&mut x_again, &rhs, false)?;
        assert_vec_approx_eq!(x_again.as_data(), x_correct, 1e-14);

        // factorize fails on singular matrix
        let mut trip_singular = SparseTriplet::new(5, 5, 2, false)?;
        trip_singular.put(0, 0, 1.0);
        trip_singular.put(4, 4, 1.0);
        solver.initialize(&trip_singular, false)?;
        assert_eq!(solver.factorize(false), Err("Error(-10): numerically singular matrix"));

        // done
        Ok(())
    }

    #[test]
    fn handle_mmp_error_code_works() -> Result<(), &'static str> {
        let default = "Error: unknown error returned by c-code (MMP)";
        let mut config = ConfigSolver::new();
        config.set_solver_kind(EnumSolverKind::Mmp);
        let solver = Solver::new(config)?;
        for c in 1..57 {
            let res = solver.handle_mmp_error_code(-c);
            assert!(res.len() > 0);
            assert_ne!(res, default);
        }
        for c in 70..80 {
            let res = solver.handle_mmp_error_code(-c);
            assert!(res.len() > 0);
            assert_ne!(res, default);
        }
        for c in &[-90, -800, 1, 2, 4, 8] {
            let res = solver.handle_mmp_error_code(*c);
            assert!(res.len() > 0);
            assert_ne!(res, default);
        }
        assert_eq!(
            solver.handle_mmp_error_code(100000),
            "Error: c-code returned null pointer (MMP)"
        );
        assert_eq!(
            solver.handle_mmp_error_code(200000),
            "Error: c-code failed to allocate memory (MMP)"
        );
        assert_eq!(solver.handle_mmp_error_code(123), default);
        Ok(())
    }

    #[test]
    fn handle_umf_error_code_works() -> Result<(), &'static str> {
        let default = "Error: unknown error returned by c-code (UMF)";
        let config = ConfigSolver::new();
        let solver = Solver::new(config)?;
        for c in &[1, 2, 3, -1, -3, -4, -5, -6, -8, -11, -13, -15, -17, -18, -911] {
            let res = solver.handle_umf_error_code(*c);
            assert!(res.len() > 0);
            assert_ne!(res, default);
        }
        assert_eq!(
            solver.handle_umf_error_code(100000),
            "Error: c-code returned null pointer (UMF)"
        );
        assert_eq!(
            solver.handle_umf_error_code(200000),
            "Error: c-code failed to allocate memory (UMF)"
        );
        assert_eq!(solver.handle_umf_error_code(123), default);
        Ok(())
    }
}
