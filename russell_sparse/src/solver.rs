use super::{
    code_symmetry_mmp, code_symmetry_umf, str_enum_ordering, str_enum_scaling, str_mmp_ordering, str_mmp_scaling,
    str_umf_ordering, str_umf_scaling, ConfigSolver, CooMatrix, LinSolKind,
};
use crate::{StrError, Symmetry};
use russell_lab::{format_nanoseconds, vec_copy, Stopwatch, Vector};
use russell_openblas::to_i32;
use std::fmt;

#[repr(C)]
pub(crate) struct ExtSolver {
    data: [u8; 0],
    marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

extern "C" {
    // MMP
    fn new_solver_mmp() -> *mut ExtSolver;
    fn drop_solver_mmp(solver: *mut ExtSolver);
    fn solver_mmp_initialize(
        solver: *mut ExtSolver,
        n: i32,
        nnz: i32,
        symmetry: i32,
        ordering: i32,
        scaling: i32,
        pct_inc_workspace: i32,
        max_work_memory: i32,
        openmp_num_threads: i32,
    ) -> i32;
    fn solver_mmp_factorize(
        solver: *mut ExtSolver,
        indices_i: *const i32,
        indices_j: *const i32,
        values_aij: *const f64,
        verbose: i32,
    ) -> i32;
    fn solver_mmp_solve(solver: *mut ExtSolver, rhs: *mut f64, verbose: i32) -> i32;
    fn solver_mmp_used_ordering(solver: *const ExtSolver) -> i32;
    fn solver_mmp_used_scaling(solver: *const ExtSolver) -> i32;

    // UMF
    fn new_solver_umf() -> *mut ExtSolver;
    fn drop_solver_umf(solver: *mut ExtSolver);
    fn solver_umf_initialize(
        solver: *mut ExtSolver,
        n: i32,
        nnz: i32,
        symmetry: i32,
        ordering: i32,
        scaling: i32,
        verbose: i32,
    ) -> i32;
    fn solver_umf_factorize(
        solver: *mut ExtSolver,
        indices_i: *const i32,
        indices_j: *const i32,
        values_aij: *const f64,
        verbose: i32,
    ) -> i32;
    fn solver_umf_solve(solver: *mut ExtSolver, x: *mut f64, rhs: *const f64, verbose: i32) -> i32;
    fn solver_umf_used_ordering(solver: *const ExtSolver) -> i32;
    fn solver_umf_used_scaling(solver: *const ExtSolver) -> i32;
}

/// Implements a sparse linear solver
///
/// For a general sparse and square matrix `a` (symmetric, non-symmetric)
/// find `x` such that:
///
/// ```text
///   a   ⋅  x  =  rhs
/// (m,m)   (m)    (m)
/// ```
pub struct Solver {
    kind: LinSolKind,            // solver kind
    verbose: i32,                // verbose mode
    done_factorize: bool,        // factorization completed
    neq: usize,                  // number of equations == nrow(a) where a*x=rhs
    solver: *mut ExtSolver,      // data allocated by the c-code
    stopwatch: Stopwatch,        // stopwatch to measure elapsed time
    time_fact: u128,             // elapsed time during factorize
    time_solve: u128,            // elapsed time during solve
    used_ordering: &'static str, // used ordering strategy
    used_scaling: &'static str,  // used scaling strategy
}

impl Solver {
    /// Creates a new solver
    pub fn new(config: ConfigSolver, neq: usize, nnz: usize, symmetry: Option<Symmetry>) -> Result<Self, StrError> {
        let n = to_i32(neq);
        let nnz = to_i32(nnz);
        unsafe {
            let solver = match config.lin_sol_kind {
                LinSolKind::Mmp => new_solver_mmp(),
                LinSolKind::Umf => new_solver_umf(),
            };
            if solver.is_null() {
                return Err("c-code failed to allocate solver");
            }
            match config.lin_sol_kind {
                LinSolKind::Mmp => {
                    let res = solver_mmp_initialize(
                        solver,
                        n,
                        nnz,
                        code_symmetry_mmp(symmetry)?,
                        config.ordering,
                        config.scaling,
                        config.pct_inc_workspace,
                        config.max_work_memory,
                        config.openmp_num_threads,
                    );
                    if res != 0 {
                        drop_solver_mmp(solver);
                        return Err(Solver::handle_mmp_error_code(res));
                    }
                }
                LinSolKind::Umf => {
                    let res = solver_umf_initialize(
                        solver,
                        n,
                        nnz,
                        code_symmetry_umf(symmetry)?,
                        config.ordering,
                        config.scaling,
                        config.verbose,
                    );
                    if res != 0 {
                        drop_solver_umf(solver);
                        return Err(Solver::handle_umf_error_code(res));
                    }
                }
            }
            Ok(Solver {
                kind: config.lin_sol_kind,
                verbose: config.verbose,
                done_factorize: false,
                neq,
                solver,
                stopwatch: Stopwatch::new(""),
                time_fact: 0,
                time_solve: 0,
                used_ordering: str_enum_ordering(config.ordering),
                used_scaling: str_enum_scaling(config.scaling),
            })
        }
    }

    /// Performs the factorization
    pub fn factorize(&mut self, trip: &CooMatrix) -> Result<(), StrError> {
        if trip.neq != self.neq {
            return Err("cannot factorize because the triplet has incompatible number of equations");
        }
        self.stopwatch.reset();
        unsafe {
            match self.kind {
                LinSolKind::Mmp => {
                    let res = solver_mmp_factorize(
                        self.solver,
                        trip.indices_i.as_ptr(),
                        trip.indices_j.as_ptr(),
                        trip.values_aij.as_ptr(),
                        self.verbose,
                    );
                    if res != 0 {
                        return Err(Solver::handle_mmp_error_code(res));
                    }
                    let ord = solver_mmp_used_ordering(self.solver);
                    let sca = solver_mmp_used_scaling(self.solver);
                    self.used_ordering = str_mmp_ordering(ord);
                    self.used_scaling = str_mmp_scaling(sca);
                }
                LinSolKind::Umf => {
                    let res = solver_umf_factorize(
                        self.solver,
                        trip.indices_i.as_ptr(),
                        trip.indices_j.as_ptr(),
                        trip.values_aij.as_ptr(),
                        self.verbose,
                    );
                    if res != 0 {
                        return Err(Solver::handle_umf_error_code(res));
                    }
                    let ord = solver_umf_used_ordering(self.solver);
                    let sca = solver_umf_used_scaling(self.solver);
                    self.used_ordering = str_umf_ordering(ord);
                    self.used_scaling = str_umf_scaling(sca);
                }
            }
        }
        self.done_factorize = true;
        self.time_fact = self.stopwatch.stop();
        Ok(())
    }

    /// Computes the solution
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::{Matrix, Vector};
    /// use russell_sparse::{ConfigSolver, CooMatrix, Solver, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     // allocate a square matrix
    ///     let (neq, nnz) = (5, 13);
    ///     let mut trip = CooMatrix::new(neq, nnz)?;
    ///     trip.put(0, 0, 1.0)?; // << (0, 0, a00/2)
    ///     trip.put(0, 0, 1.0)?; // << (0, 0, a00/2)
    ///     trip.put(1, 0, 3.0)?;
    ///     trip.put(0, 1, 3.0)?;
    ///     trip.put(2, 1, -1.0)?;
    ///     trip.put(4, 1, 4.0)?;
    ///     trip.put(1, 2, 4.0)?;
    ///     trip.put(2, 2, -3.0)?;
    ///     trip.put(3, 2, 1.0)?;
    ///     trip.put(4, 2, 2.0)?;
    ///     trip.put(2, 3, 2.0)?;
    ///     trip.put(1, 4, 6.0)?;
    ///     trip.put(4, 4, 1.0)?;
    ///
    ///     // print matrix
    ///     let mut a = Matrix::new(neq, neq);
    ///     trip.to_matrix(&mut a)?;
    ///     let correct = "┌                ┐\n\
    ///                    │  2  3  0  0  0 │\n\
    ///                    │  3  0  4  0  6 │\n\
    ///                    │  0 -1 -3  2  0 │\n\
    ///                    │  0  0  1  0  0 │\n\
    ///                    │  0  4  2  0  1 │\n\
    ///                    └                ┘";
    ///     assert_eq!(format!("{}", a), correct);
    ///
    ///     // allocate x and rhs
    ///     let mut x = Vector::new(neq);
    ///     let rhs = Vector::from(&[8.0, 45.0, -3.0, 3.0, 19.0]);
    ///
    ///     // initialize, factorize, and solve
    ///     let config = ConfigSolver::new();
    ///     let mut solver = Solver::new(config, neq, nnz, None)?;
    ///     solver.factorize(&trip)?;
    ///     solver.solve(&mut x, &rhs)?;
    ///     let correct = "┌          ┐\n\
    ///                    │ 1.000000 │\n\
    ///                    │ 2.000000 │\n\
    ///                    │ 3.000000 │\n\
    ///                    │ 4.000000 │\n\
    ///                    │ 5.000000 │\n\
    ///                    └          ┘";
    ///     assert_eq!(format!("{:.6}", x), correct);
    ///     Ok(())
    /// }
    /// ```
    pub fn solve(&mut self, x: &mut Vector, rhs: &Vector) -> Result<(), StrError> {
        if !self.done_factorize {
            return Err("factorization must be done before calling solve");
        }
        if x.dim() != self.neq || rhs.dim() != self.neq {
            return Err("x.ndim() and rhs.ndim() must equal the number of equations");
        }
        self.stopwatch.reset();
        unsafe {
            match self.kind {
                LinSolKind::Mmp => {
                    vec_copy(x, rhs)?;
                    let res = solver_mmp_solve(self.solver, x.as_mut_data().as_mut_ptr(), self.verbose);
                    if res != 0 {
                        return Err(Solver::handle_mmp_error_code(res));
                    }
                }
                LinSolKind::Umf => {
                    let res = solver_umf_solve(
                        self.solver,
                        x.as_mut_data().as_mut_ptr(),
                        rhs.as_data().as_ptr(),
                        self.verbose,
                    );
                    if res != 0 {
                        return Err(Solver::handle_umf_error_code(res));
                    }
                }
            }
        }
        self.time_solve = self.stopwatch.stop();
        Ok(())
    }

    /// Computes a new solution
    ///
    /// ```text
    ///   a   ⋅  x  =  rhs
    /// (m,m)   (m)    (m)
    /// ```
    ///
    /// # Output
    ///
    /// `(solver, x)` -- the solver and the solution vector
    ///
    /// # Note
    ///
    /// The solver will be initialized, the matrix will be factorized, and
    /// the solution will be calculated. These steps correspond to calling
    /// `initialize`, `factorize`, and `solve`, one after another. Thus,
    /// you may re-compute solutions with the already factorized matrix
    /// by calling `solve` again.
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::{Matrix, Vector};
    /// use russell_sparse::{ConfigSolver, Solver, CooMatrix, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     // allocate a square matrix
    ///     let (neq, nnz) = (3, 5);
    ///     let mut trip = CooMatrix::new(neq, nnz)?;
    ///     trip.put(0, 0, 0.2)?;
    ///     trip.put(0, 1, 0.2)?;
    ///     trip.put(1, 0, 0.5)?;
    ///     trip.put(1, 1, -0.25)?;
    ///     trip.put(2, 2, 0.25)?;
    ///
    ///     // print matrix
    ///     let mut a = Matrix::new(neq, neq);
    ///     trip.to_matrix(&mut a)?;
    ///     let correct = "┌                   ┐\n\
    ///                    │   0.2   0.2     0 │\n\
    ///                    │   0.5 -0.25     0 │\n\
    ///                    │     0     0  0.25 │\n\
    ///                    └                   ┘";
    ///     assert_eq!(format!("{}", a), correct);
    ///
    ///     // allocate rhs
    ///     let rhs1 = Vector::from(&[1.0, 1.0, 1.0]);
    ///     let rhs2 = Vector::from(&[2.0, 2.0, 2.0]);
    ///
    ///     // calculate solution
    ///     let config = ConfigSolver::new();
    ///     let (mut solver, x1) = Solver::compute(config, &trip, &rhs1)?;
    ///     let correct1 = "┌   ┐\n\
    ///                     │ 3 │\n\
    ///                     │ 2 │\n\
    ///                     │ 4 │\n\
    ///                     └   ┘";
    ///     assert_eq!(format!("{}", x1), correct1);
    ///
    ///     // solve again
    ///     let mut x2 = Vector::new(neq);
    ///     solver.solve(&mut x2, &rhs2)?;
    ///     let correct2 = "┌   ┐\n\
    ///                     │ 6 │\n\
    ///                     │ 4 │\n\
    ///                     │ 8 │\n\
    ///                     └   ┘";
    ///     assert_eq!(format!("{}", x2), correct2);
    ///     Ok(())
    /// }
    /// ```
    pub fn compute(config: ConfigSolver, trip: &CooMatrix, rhs: &Vector) -> Result<(Self, Vector), StrError> {
        let mut solver = Solver::new(config, trip.neq, trip.pos, None)?;
        let mut x = Vector::new(trip.neq());
        solver.factorize(&trip)?;
        solver.solve(&mut x, &rhs)?;
        Ok((solver, x))
    }

    /// Returns the elapsed times
    ///
    /// # Output
    ///
    /// * `(time_fact, time_solve)` -- elapsed times during factorize and solve, respectively
    pub fn get_elapsed_times(&self) -> (u128, u128) {
        (self.time_fact, self.time_solve)
    }

    /// Handles error code
    fn handle_mmp_error_code(err: i32) -> StrError {
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
    fn handle_umf_error_code(err: i32) -> StrError {
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
            match self.kind {
                LinSolKind::Mmp => drop_solver_mmp(self.solver),
                LinSolKind::Umf => drop_solver_umf(self.solver),
            }
        }
    }
}

impl fmt::Display for Solver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let time_total = self.time_fact + self.time_solve;
        write!(
            f,
            "\x20\x20\x20\x20\"usedOrdering\": \"{}\",\n\
             \x20\x20\x20\x20\"usedScaling\": \"{}\",\n\
             \x20\x20\x20\x20\"doneFactorize\": {},\n\
             \x20\x20\x20\x20\"neq\": {},\n\
             \x20\x20\x20\x20\"timeFactNs\": {},\n\
             \x20\x20\x20\x20\"timeSolveNs\": {},\n\
             \x20\x20\x20\x20\"timeTotalNs\": {},\n\
             \x20\x20\x20\x20\"timeFactStr\": \"{}\",\n\
             \x20\x20\x20\x20\"timeSolveStr\": \"{}\",\n\
             \x20\x20\x20\x20\"timeTotalStr\": \"{}\"",
            self.used_ordering,
            self.used_scaling,
            self.done_factorize,
            self.neq,
            self.time_fact,
            self.time_solve,
            time_total,
            format_nanoseconds(self.time_fact),
            format_nanoseconds(self.time_solve),
            format_nanoseconds(time_total)
        )?;
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{ConfigSolver, CooMatrix, LinSolKind, Solver};
    use russell_chk::vec_approx_eq;
    use russell_lab::Vector;

    #[test]
    fn new_works() {
        let config = ConfigSolver::new();
        let (neq, nnz) = (2, 2);
        let solver = Solver::new(config, neq, nnz, None).unwrap();
        assert_eq!(solver.done_factorize, false);
        assert_eq!(solver.neq, 2);
    }

    #[test]
    fn factorize_fails_on_incompatible_triplet() {
        let config = ConfigSolver::new();
        let mut solver = Solver::new(config, 1, 1, None).unwrap();
        let trip = CooMatrix::new(2, 2).unwrap();
        assert_eq!(
            solver.factorize(&trip).err(),
            Some("cannot factorize because the triplet has incompatible number of equations")
        );
    }

    #[test]
    fn factorize_fails_on_singular_matrix() {
        let config = ConfigSolver::new();
        let (neq, nnz) = (2, 2);
        let mut solver = Solver::new(config, neq, nnz, None).unwrap();
        let mut trip = CooMatrix::new(neq, nnz).unwrap();
        trip.put(0, 0, 1.0).unwrap();
        trip.put(1, 1, 0.0).unwrap();
        assert_eq!(solver.factorize(&trip), Err("Error(1): Matrix is singular"));
    }

    #[test]
    fn factorize_works() {
        let config = ConfigSolver::new();
        let (neq, nnz) = (2, 2);
        let mut solver = Solver::new(config, neq, nnz, None).unwrap();
        let mut trip = CooMatrix::new(neq, nnz).unwrap();
        trip.put(0, 0, 1.0).unwrap();
        trip.put(1, 1, 1.0).unwrap();
        solver.factorize(&trip).unwrap();
        assert!(solver.done_factorize);
    }

    #[test]
    fn solve_fails_on_non_factorized() {
        let config = ConfigSolver::new();
        let (neq, nnz) = (2, 2);
        let mut solver = Solver::new(config, neq, nnz, None).unwrap();
        let mut trip = CooMatrix::new(neq, nnz).unwrap();
        trip.put(0, 0, 1.0).unwrap();
        trip.put(1, 1, 1.0).unwrap();
        let mut x = Vector::new(neq);
        let rhs = Vector::from(&[1.0, 1.0]);
        assert_eq!(
            solver.solve(&mut x, &rhs),
            Err("factorization must be done before calling solve")
        );
    }

    #[test]
    fn solve_fails_on_wrong_vectors() {
        let config = ConfigSolver::new();
        let (neq, nnz) = (2, 2);
        let mut solver = Solver::new(config, neq, nnz, None).unwrap();
        let mut trip = CooMatrix::new(neq, nnz).unwrap();
        trip.put(0, 0, 1.0).unwrap();
        trip.put(1, 1, 1.0).unwrap();
        solver.factorize(&trip).unwrap();
        let mut x = Vector::new(2);
        let rhs = Vector::from(&[1.0, 1.0]);
        let mut x_wrong = Vector::new(1);
        let rhs_wrong = Vector::from(&[1.0]);
        assert_eq!(
            solver.solve(&mut x_wrong, &rhs),
            Err("x.ndim() and rhs.ndim() must equal the number of equations")
        );
        assert_eq!(
            solver.solve(&mut x, &rhs_wrong),
            Err("x.ndim() and rhs.ndim() must equal the number of equations")
        );
    }

    #[test]
    fn solve_works() {
        let config = ConfigSolver::new();
        let (neq, nnz) = (5, 13);
        let mut solver = Solver::new(config, neq, nnz, None).unwrap();

        // allocate a square matrix
        let mut trip = CooMatrix::new(neq, nnz).unwrap();
        trip.put(0, 0, 1.0).unwrap(); // << (0, 0, a00/2)
        trip.put(0, 0, 1.0).unwrap(); // << (0, 0, a00/2)
        trip.put(1, 0, 3.0).unwrap();
        trip.put(0, 1, 3.0).unwrap();
        trip.put(2, 1, -1.0).unwrap();
        trip.put(4, 1, 4.0).unwrap();
        trip.put(1, 2, 4.0).unwrap();
        trip.put(2, 2, -3.0).unwrap();
        trip.put(3, 2, 1.0).unwrap();
        trip.put(4, 2, 2.0).unwrap();
        trip.put(2, 3, 2.0).unwrap();
        trip.put(1, 4, 6.0).unwrap();
        trip.put(4, 4, 1.0).unwrap();

        // allocate x and rhs
        let mut x = Vector::new(5);
        let rhs = Vector::from(&[8.0, 45.0, -3.0, 3.0, 19.0]);
        let x_correct = &[1.0, 2.0, 3.0, 4.0, 5.0];

        // initialize, factorize, and solve
        solver.factorize(&trip).unwrap();
        solver.solve(&mut x, &rhs).unwrap();

        // check
        vec_approx_eq(x.as_data(), x_correct, 1e-14);
    }

    // This function tests many behaviors of the MMP solver.
    // All of these calls must be in a single function because the
    // MMP solver is NOT thread-safe.
    #[test]
    fn solver_mmp_behaves_as_expected() {
        // allocate a new solver
        let mut config = ConfigSolver::new();
        let (neq, nnz) = (5, 13);
        config.lin_sol_kind(LinSolKind::Mmp);
        let mut solver = Solver::new(config, neq, nnz, None).unwrap();

        // factorize fails on incompatible triplet
        let mut trip_wrong = CooMatrix::new(1, 1).unwrap();
        trip_wrong.put(0, 0, 1.0).unwrap();
        assert_eq!(
            solver.factorize(&trip_wrong).err(),
            Some("cannot factorize because the triplet has incompatible number of equations")
        );

        // allocate a square matrix
        let mut trip = CooMatrix::new(neq, nnz).unwrap();
        trip.put(0, 0, 1.0).unwrap(); // << (0, 0, a00/2)
        trip.put(0, 0, 1.0).unwrap(); // << (0, 0, a00/2)
        trip.put(1, 0, 3.0).unwrap();
        trip.put(0, 1, 3.0).unwrap();
        trip.put(2, 1, -1.0).unwrap();
        trip.put(4, 1, 4.0).unwrap();
        trip.put(1, 2, 4.0).unwrap();
        trip.put(2, 2, -3.0).unwrap();
        trip.put(3, 2, 1.0).unwrap();
        trip.put(4, 2, 2.0).unwrap();
        trip.put(2, 3, 2.0).unwrap();
        trip.put(1, 4, 6.0).unwrap();
        trip.put(4, 4, 1.0).unwrap();

        // allocate x and rhs
        let mut x = Vector::new(5);
        let rhs = Vector::from(&[8.0, 45.0, -3.0, 3.0, 19.0]);
        let x_correct = &[1.0, 2.0, 3.0, 4.0, 5.0];

        // solve fails on non-factorized system
        assert_eq!(
            solver.solve(&mut x, &rhs),
            Err("factorization must be done before calling solve")
        );

        // factorize works
        solver.factorize(&trip).unwrap();
        assert!(solver.done_factorize);

        // solve fails on wrong x vector
        let mut x_wrong = Vector::new(3);
        assert_eq!(
            solver.solve(&mut x_wrong, &rhs),
            Err("x.ndim() and rhs.ndim() must equal the number of equations")
        );

        // solve fails on wrong rhs vector
        let rhs_wrong = Vector::from(&[1.0]);
        assert_eq!(
            solver.solve(&mut x, &rhs_wrong),
            Err("x.ndim() and rhs.ndim() must equal the number of equations")
        );

        // solve works
        solver.solve(&mut x, &rhs).unwrap();
        vec_approx_eq(x.as_data(), x_correct, 1e-14);

        // calling solve again works
        let mut x_again = Vector::new(5);
        solver.solve(&mut x_again, &rhs).unwrap();
        vec_approx_eq(x_again.as_data(), x_correct, 1e-14);

        // factorize fails on singular matrix
        let mut trip_singular = CooMatrix::new(5, 2).unwrap();
        trip_singular.put(0, 0, 1.0).unwrap();
        trip_singular.put(4, 4, 1.0).unwrap();
        let mut solver = Solver::new(config, 5, 2, None).unwrap();
        assert_eq!(
            solver.factorize(&trip_singular),
            Err("Error(-10): numerically singular matrix")
        );
    }

    #[test]
    fn compute_works() {
        let (neq, nnz) = (3, 6);
        let mut trip = CooMatrix::new(neq, nnz).unwrap();
        trip.put(0, 0, 1.0).unwrap();
        trip.put(0, 1, 1.0).unwrap();
        trip.put(1, 0, 2.0).unwrap();
        trip.put(1, 1, 1.0).unwrap();
        trip.put(1, 2, 1.0).unwrap();
        trip.put(2, 2, 1.0).unwrap();
        let rhs1 = Vector::from(&[1.0, 2.0, 3.0]);
        let rhs2 = Vector::from(&[2.0, 4.0, 6.0]);
        let config = ConfigSolver::new();
        let (mut solver, x1) = Solver::compute(config, &trip, &rhs1).unwrap();
        vec_approx_eq(x1.as_data(), &[-2.0, 3.0, 3.0], 1e-15);
        let mut x2 = Vector::new(neq);
        solver.solve(&mut x2, &rhs2).unwrap();
    }

    #[test]
    fn get_elapsed_times_works() {
        let config = ConfigSolver::new();
        let (neq, nnz) = (2, 2);
        let solver = Solver::new(config, neq, nnz, None).unwrap();
        let times = solver.get_elapsed_times();
        assert_eq!(times, (0, 0));
    }

    #[test]
    fn handle_mmp_error_code_works() {
        let default = "Error: unknown error returned by c-code (MMP)";
        let mut config = ConfigSolver::new();
        config.lin_sol_kind(LinSolKind::Mmp);
        for c in 1..57 {
            let res = Solver::handle_mmp_error_code(-c);
            assert!(res.len() > 0);
            assert_ne!(res, default);
        }
        for c in 70..80 {
            let res = Solver::handle_mmp_error_code(-c);
            assert!(res.len() > 0);
            assert_ne!(res, default);
        }
        for c in &[-90, -800, 1, 2, 4, 8] {
            let res = Solver::handle_mmp_error_code(*c);
            assert!(res.len() > 0);
            assert_ne!(res, default);
        }
        assert_eq!(
            Solver::handle_mmp_error_code(100000),
            "Error: c-code returned null pointer (MMP)"
        );
        assert_eq!(
            Solver::handle_mmp_error_code(200000),
            "Error: c-code failed to allocate memory (MMP)"
        );
        assert_eq!(Solver::handle_mmp_error_code(123), default);
    }

    #[test]
    fn handle_umf_error_code_works() {
        let default = "Error: unknown error returned by c-code (UMF)";
        for c in &[1, 2, 3, -1, -3, -4, -5, -6, -8, -11, -13, -15, -17, -18, -911] {
            let res = Solver::handle_umf_error_code(*c);
            assert!(res.len() > 0);
            assert_ne!(res, default);
        }
        assert_eq!(
            Solver::handle_umf_error_code(100000),
            "Error: c-code returned null pointer (UMF)"
        );
        assert_eq!(
            Solver::handle_umf_error_code(200000),
            "Error: c-code failed to allocate memory (UMF)"
        );
        assert_eq!(Solver::handle_umf_error_code(123), default);
    }

    #[test]
    fn display_trait_works() {
        let config = ConfigSolver::new();
        let (neq, nnz) = (2, 2);
        let solver = Solver::new(config, neq, nnz, None).unwrap();
        let b: &str = "\x20\x20\x20\x20\"usedOrdering\": \"Auto\",\n\
                       \x20\x20\x20\x20\"usedScaling\": \"Auto\",\n\
                       \x20\x20\x20\x20\"doneFactorize\": false,\n\
                       \x20\x20\x20\x20\"neq\": 2,\n\
                       \x20\x20\x20\x20\"timeFactNs\": 0,\n\
                       \x20\x20\x20\x20\"timeSolveNs\": 0,\n\
                       \x20\x20\x20\x20\"timeTotalNs\": 0,\n\
                       \x20\x20\x20\x20\"timeFactStr\": \"0ns\",\n\
                       \x20\x20\x20\x20\"timeSolveStr\": \"0ns\",\n\
                       \x20\x20\x20\x20\"timeTotalStr\": \"0ns\"";
        assert_eq!(format!("{}", solver), b);
    }
}
