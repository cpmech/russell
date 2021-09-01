use super::*;
use russell_lab::*;
use std::fmt;

#[repr(C)]
pub(crate) struct ExtSolverMMP {
    data: [u8; 0],
    marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

extern "C" {
    fn new_solver_mmp(symmetry: i32, verbose: i32) -> *mut ExtSolverMMP;
    fn drop_solver_mmp(solver: *mut ExtSolverMMP);
    fn solver_mmp_initialize(
        solver: *mut ExtSolverMMP,
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
    fn solver_mmp_factorize(solver: *mut ExtSolverMMP, verbose: i32) -> i32;
    fn solver_mmp_solve(solver: *mut ExtSolverMMP, rhs: *mut f64, verbose: i32) -> i32;
}

/// Implements the French (Mu-M-P) Solver
///
/// # Warning
///
/// This solver cannot be used in multiple threads, because
/// the Fortran implementation of Mu-M-P-S is not thread safe.
pub struct SolverMMP {
    ordering: i32,             // symmetric permutation (ordering). ICNTL(7)
    scaling: i32,              // scaling strategy. ICNTL(8)
    pct_inc_workspace: i32,    // % increase in the estimated working space. ICNTL(14)
    max_work_memory: i32,      // max size of the working memory in mega bytes. ICNTL(23)
    openmp_num_threads: i32,   // number of OpenMP threads. ICNTL(16)
    done_initialize: bool,     // initialization completed
    done_factorize: bool,      // factorization completed
    ndim: usize,               // number of equations == nrow(a) where a*x=rhs
    solver: *mut ExtSolverMMP, // data allocated by the c-code
}

impl SolverMMP {
    /// Creates a new solver
    ///
    /// # Input
    ///
    /// `symmetry` -- Tells wether the system matrix is unsymmetric, positive-definite symmetric, or general symmetric
    /// `verbose` -- Prints messages on the default terminal (we cannot control where)
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), &'static str> {
    /// use russell_sparse::*;
    /// let solver = SolverMMP::new(EnumSymmetry::No, true)?;
    /// let correct: &str = "===========================\n\
    ///                      SolverMMP\n\
    ///                      ---------------------------\n\
    ///                      ordering           = 5\n\
    ///                      scaling            = 77\n\
    ///                      pct_inc_workspace  = 100\n\
    ///                      max_work_memory    = 0\n\
    ///                      openmp_num_threads = 1\n\
    ///                      done_initialize    = false\n\
    ///                      done_factorize     = false\n\
    ///                      ===========================";
    /// assert_eq!(format!("{}", solver), correct);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(symmetry: EnumSymmetry, verbose: bool) -> Result<Self, &'static str> {
        let sym = code_symmetry(symmetry);
        let verb: i32 = if verbose { 1 } else { 0 };
        unsafe {
            let solver = new_solver_mmp(sym, verb);
            if solver.is_null() {
                return Err("c-code failed to allocate SolverMMP");
            }
            Ok(SolverMMP {
                ordering: code_ordering(EnumOrdering::Metis),
                scaling: code_scaling(EnumScaling::Auto),
                pct_inc_workspace: 100,
                max_work_memory: 0, // auto
                openmp_num_threads: 1,
                done_initialize: false,
                done_factorize: false,
                ndim: 0,
                solver,
            })
        }
    }

    /// Returns the name of this solver
    /// ```
    /// # fn main() -> Result<(), &'static str> {
    /// use russell_sparse::*;
    /// let solver = SolverMMP::new(EnumSymmetry::No, true)?;
    /// assert_eq!(solver.name(), "SolverMMP");
    /// # Ok(())
    /// # }
    /// ```
    pub fn name(&self) -> &'static str {
        "SolverMMP"
    }

    /// Sets the method to compute a symmetric permutation (ordering)
    pub fn set_ordering(&mut self, selection: EnumOrdering) {
        self.ordering = code_ordering(selection);
    }

    /// Sets the scaling strategy
    pub fn set_scaling(&mut self, selection: EnumScaling) {
        self.scaling = code_scaling(selection);
    }

    /// Sets the percentage increase in the estimated working space
    pub fn set_pct_inc_workspace(&mut self, value: usize) {
        self.pct_inc_workspace = to_i32(value);
    }

    /// Sets the maximum size of the working memory in mega bytes
    pub fn set_max_work_memory(&mut self, value: usize) {
        self.max_work_memory = to_i32(value);
    }

    /// Sets the number of OpenMP threads
    pub fn set_openmp_num_threads(&mut self, value: usize) {
        self.openmp_num_threads = to_i32(value);
    }

    /// Initializes the solver
    pub fn initialize(&mut self, trip: &SparseTriplet, verbose: bool) -> Result<(), &'static str> {
        if trip.nrow != trip.ncol {
            return Err("the matrix represented by the triplet must be square");
        }
        let n = to_i32(trip.nrow);
        let nnz = to_i32(trip.pos);
        let verb: i32 = if verbose { 1 } else { 0 };
        unsafe {
            let res = solver_mmp_initialize(
                self.solver,
                n,
                nnz,
                trip.indices_i.as_ptr(),
                trip.indices_j.as_ptr(),
                trip.values_a.as_ptr(),
                self.ordering,
                self.scaling,
                self.pct_inc_workspace,
                self.max_work_memory,
                self.openmp_num_threads,
                verb,
            );
            if res != 0 {
                return Err(self.handle_error_code(res));
            }
            self.done_initialize = true;
            self.ndim = trip.nrow;
        }
        Ok(())
    }

    /// Performs the factorization
    pub fn factorize(&mut self, verbose: bool) -> Result<(), &'static str> {
        if !self.done_initialize {
            return Err("initialization must be done before factorization");
        }
        let verb: i32 = if verbose { 1 } else { 0 };
        unsafe {
            let res = solver_mmp_factorize(self.solver, verb);
            if res != 0 {
                return Err(self.handle_error_code(res));
            }
            self.done_factorize = true;
        }
        Ok(())
    }

    /// Computes the solution
    pub fn solve(
        &mut self,
        x: &mut Vector,
        rhs: &Vector,
        verbose: bool,
    ) -> Result<(), &'static str> {
        if !self.done_factorize {
            return Err("factorization must be done before solution");
        }
        if x.dim() != self.ndim || rhs.dim() != self.ndim {
            return Err("x.ndim() and rhs.ndim() must equal the number of equations");
        }
        copy_vector(x, rhs)?;
        let verb: i32 = if verbose { 1 } else { 0 };
        unsafe {
            let res = solver_mmp_solve(self.solver, x.as_mut_data().as_mut_ptr(), verb);
            if res != 0 {
                return Err(self.handle_error_code(res));
            }
        }
        Ok(())
    }

    /// Handles error code
    fn handle_error_code(&self, err: i32) -> &'static str {
        match err {
            -1  => "Error(-1): error on some processor",
            -2  => "Error(-2): nnz is out of range",
            -3  => "Error(-3): solver called with an invalid job value",
            -4  => "Error(-4): error in user-provided permutation array",
            -5  => "Error(-5): problem with real workspace allocation during analysis",
            -6  => "Error(-6): matrix is singular in structure",
            -7  => "Error(-7): problem with integer workspace allocation during analysis",
            -8  => "Error(-8): internal integer work array is too small for factorization",
            -9  => "Error(-9): internal real/complex work array is too small",
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
            100000 => return "Error: c-code returned null pointer",
            _ => return "Error: unknown error returned by SolverMMP (c-code)",
        }
    }
}

impl Drop for SolverMMP {
    /// Tells the c-code to release memory
    fn drop(&mut self) {
        unsafe {
            drop_solver_mmp(self.solver);
        }
    }
}

impl fmt::Display for SolverMMP {
    /// Display some information about this solver
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "===========================\n\
            SolverMMP\n\
            ---------------------------\n\
            ordering           = {}\n\
            scaling            = {}\n\
            pct_inc_workspace  = {}\n\
            max_work_memory    = {}\n\
            openmp_num_threads = {}\n\
            done_initialize    = {}\n\
            done_factorize     = {}\n\
            ===========================",
            self.ordering,
            self.scaling,
            self.pct_inc_workspace,
            self.max_work_memory,
            self.openmp_num_threads,
            self.done_initialize,
            self.done_factorize,
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
        let solver = SolverMMP::new(EnumSymmetry::No, true)?;
        assert_eq!(solver.solver.is_null(), false);
        Ok(())
    }

    #[test]
    fn name_works() -> Result<(), &'static str> {
        let solver = SolverMMP::new(EnumSymmetry::No, true)?;
        assert_eq!(solver.name(), "SolverMMP");
        Ok(())
    }

    #[test]
    fn set_ordering() -> Result<(), &'static str> {
        let mut solver = SolverMMP::new(EnumSymmetry::No, true)?;
        solver.set_ordering(EnumOrdering::Amf);
        assert_eq!(solver.ordering, 2);
        Ok(())
    }

    #[test]
    fn set_scaling_works() -> Result<(), &'static str> {
        let mut solver = SolverMMP::new(EnumSymmetry::No, true)?;
        solver.set_scaling(EnumScaling::RowCol);
        assert_eq!(solver.scaling, 4);
        Ok(())
    }

    #[test]
    fn set_pct_inc_workspace_works() -> Result<(), &'static str> {
        let mut solver = SolverMMP::new(EnumSymmetry::No, true)?;
        solver.set_pct_inc_workspace(15);
        assert_eq!(solver.pct_inc_workspace, 15);
        Ok(())
    }

    #[test]
    fn set_max_work_memory_works() -> Result<(), &'static str> {
        let mut solver = SolverMMP::new(EnumSymmetry::No, true)?;
        solver.set_max_work_memory(500);
        assert_eq!(solver.max_work_memory, 500);
        Ok(())
    }

    #[test]
    fn set_openmp_num_threads_works() -> Result<(), &'static str> {
        let mut solver = SolverMMP::new(EnumSymmetry::No, true)?;
        solver.set_openmp_num_threads(3);
        assert_eq!(solver.openmp_num_threads, 3);
        Ok(())
    }

    #[test]
    fn display_trait_works() -> Result<(), &'static str> {
        let solver = SolverMMP::new(EnumSymmetry::No, true)?;
        let correct: &str = "===========================\n\
                             SolverMMP\n\
                             ---------------------------\n\
                             ordering           = 5\n\
                             scaling            = 77\n\
                             pct_inc_workspace  = 100\n\
                             max_work_memory    = 0\n\
                             openmp_num_threads = 1\n\
                             done_initialize    = false\n\
                             done_factorize     = false\n\
                             ===========================";
        assert_eq!(format!("{}", solver), correct);
        Ok(())
    }

    /*
    #[test]
    fn initialize_fails_on_non_square_matrix() -> Result<(), &'static str> {
        let trip = SparseTriplet::new(3, 2, 1)?;
        let mut solver = SolverMMP::new(EnumSymmetry::No, false)?;
        assert_eq!(
            solver.initialize(&trip, false),
            Err("the matrix represented by the triplet must be square")
        );
        Ok(())
    }

    #[test]
    fn initialize_works() -> Result<(), &'static str> {
        let mut trip = SparseTriplet::new(2, 2, 2)?;
        trip.put(0, 0, 1.0);
        trip.put(1, 1, 1.0);
        let mut solver = SolverMMP::new(EnumSymmetry::No, false)?;
        solver.initialize(&trip, false)?;
        assert!(solver.done_initialize);
        Ok(())
    }

    #[test]
    fn factorize_fails_on_non_initialized() -> Result<(), &'static str> {
        let mut solver = SolverMMP::new(EnumSymmetry::No, false)?;
        assert_eq!(
            solver.factorize(false),
            Err("initialization must be done before factorization")
        );
        Ok(())
    }

    #[test]
    fn initialize_and_factorize_works() -> Result<(), &'static str> {
        let mut trip = SparseTriplet::new(2, 2, 2)?;
        trip.put(0, 0, 1.0);
        trip.put(1, 1, 1.0);
        let mut solver = SolverMMP::new(EnumSymmetry::No, false)?;
        solver.initialize(&trip, false)?;
        assert!(solver.done_initialize);
        solver.factorize(false)?;
        assert!(solver.done_factorize);
        Ok(())
    }

    #[test]
    fn solve_fails_on_wrong_input() -> Result<(), &'static str> {
        let mut trip = SparseTriplet::new(2, 2, 2)?;
        trip.put(0, 0, 1.0);
        trip.put(1, 1, 1.0);
        let mut solver = SolverMMP::new(EnumSymmetry::No, false)?;
        let mut x = Vector::new(2);
        let rhs = Vector::from(&[1.0, 2.0]);
        assert_eq!(
            solver.solve(&mut x, &rhs, false),
            Err("factorization must be done before solution")
        );
        solver.initialize(&trip, false)?;
        solver.factorize(false)?;
        let mut x_wrong = Vector::new(3);
        assert_eq!(
            solver.solve(&mut x_wrong, &rhs, false),
            Err("x.ndim() and rhs.ndim() must equal the number of equations")
        );
        let rhs_wrong = Vector::from(&[1.0]);
        assert_eq!(
            solver.solve(&mut x, &rhs_wrong, false),
            Err("x.ndim() and rhs.ndim() must equal the number of equations")
        );
        Ok(())
    }

    #[test]
    fn factorize_fails_on_singular_matrix() -> Result<(), &'static str> {
        let mut trip = SparseTriplet::new(2, 2, 2)?;
        trip.put(0, 0, 1.0);
        trip.put(1, 1, 0.0);
        let mut solver = SolverMMP::new(EnumSymmetry::No, false)?;
        solver.initialize(&trip, false)?;
        assert_eq!(
            solver.factorize(false),
            Err("Error(-10): numerically singular matrix")
        );
        Ok(())
    }
    */

    #[test]
    fn initialize_factorize_and_solve_works() -> Result<(), &'static str> {
        let mut trip = SparseTriplet::new(5, 5, 13)?;
        trip.put(0, 0, 1.0); // << duplicated
        trip.put(0, 0, 1.0); // << duplicated
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

        let mut solver = SolverMMP::new(EnumSymmetry::No, false)?;
        solver.initialize(&trip, false)?;
        assert!(solver.done_initialize);

        solver.factorize(false)?;
        assert!(solver.done_factorize);

        let mut x = Vector::new(5);
        let rhs = Vector::from(&[8.0, 45.0, -3.0, 3.0, 19.0]);
        let x_correct = &[1.0, 2.0, 3.0, 4.0, 5.0];

        solver.solve(&mut x, &rhs, false)?;
        assert_vec_approx_eq!(x.as_data(), x_correct, 1e-14);

        let mut x_again = Vector::new(5);
        solver.solve(&mut x_again, &rhs, false)?;
        assert_vec_approx_eq!(x_again.as_data(), x_correct, 1e-14);

        Ok(())
    }

    #[test]
    fn handle_error_code_works() -> Result<(), &'static str> {
        let default = "Error: unknown error returned by SolverMMP (c-code)";
        let solver = SolverMMP::new(EnumSymmetry::No, false)?;
        for c in 1..57 {
            let res = solver.handle_error_code(-c);
            assert!(res.len() > 0);
            assert_ne!(res, default);
        }
        for c in 70..80 {
            let res = solver.handle_error_code(-c);
            assert!(res.len() > 0);
            assert_ne!(res, default);
        }
        for c in &[-90, -800, 1, 2, 4, 8] {
            let res = solver.handle_error_code(*c);
            assert!(res.len() > 0);
            assert_ne!(res, default);
        }
        assert_eq!(solver.handle_error_code(123), default);
        assert_eq!(
            solver.handle_error_code(100000),
            "Error: c-code returned null pointer"
        );
        Ok(())
    }
}
