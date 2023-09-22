use super::{
    code_symmetry_umfpack, str_enum_ordering, str_enum_scaling, str_umfpack_ordering, str_umfpack_scaling, to_i32,
    ConfigSolver, CooMatrix, LinSolKind,
};
use crate::{StrError, Symmetry};
use russell_lab::{Stopwatch, Vector};

#[repr(C)]
pub(crate) struct ExtSolver {
    data: [u8; 0],
    marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

extern "C" {
    // UMFPACK
    fn solver_umfpack_new() -> *mut ExtSolver;
    fn solver_umfpack_drop(solver: *mut ExtSolver);
    fn solver_umfpack_initialize(
        solver: *mut ExtSolver,
        n: i32,
        nnz: i32,
        symmetry: i32,
        ordering: i32,
        scaling: i32,
        compute_determinant: i32,
    ) -> i32;
    fn solver_umfpack_factorize(
        solver: *mut ExtSolver,
        indices_i: *const i32,
        indices_j: *const i32,
        values_aij: *const f64,
        verbose: i32,
    ) -> i32;
    fn solver_umfpack_solve(solver: *mut ExtSolver, x: *mut f64, rhs: *const f64, verbose: i32) -> i32;
    fn solver_umfpack_get_ordering(solver: *const ExtSolver) -> i32;
    fn solver_umfpack_get_scaling(solver: *const ExtSolver) -> i32;
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
    nrow: usize,                 // number of equations == nrow(a) where a*x=rhs
    solver: *mut ExtSolver,      // data allocated by the c-code
    stopwatch: Stopwatch,        // stopwatch to measure elapsed time
    time_fact: u128,             // elapsed time during factorize
    time_solve: u128,            // elapsed time during solve
    used_ordering: &'static str, // used ordering strategy
    used_scaling: &'static str,  // used scaling strategy
}

impl Solver {
    /// Creates a new solver
    pub fn new(config: ConfigSolver, nrow: usize, nnz: usize, symmetry: Option<Symmetry>) -> Result<Self, StrError> {
        let n = to_i32(nrow)?;
        let nnz = to_i32(nnz)?;
        let compute_determinant = 0;
        unsafe {
            let solver = match config.lin_sol_kind {
                LinSolKind::Umfpack => solver_umfpack_new(),
            };
            if solver.is_null() {
                return Err("c-code failed to allocate solver");
            }
            match config.lin_sol_kind {
                LinSolKind::Umfpack => {
                    let res = solver_umfpack_initialize(
                        solver,
                        n,
                        nnz,
                        code_symmetry_umfpack(symmetry)?,
                        config.ordering,
                        config.scaling,
                        compute_determinant,
                    );
                    if res != 0 {
                        solver_umfpack_drop(solver);
                        return Err(Solver::handle_umfpack_error_code(res));
                    }
                }
            }
            Ok(Solver {
                kind: config.lin_sol_kind,
                verbose: config.verbose,
                done_factorize: false,
                nrow,
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
    pub fn factorize(&mut self, coo: &CooMatrix) -> Result<(), StrError> {
        if coo.nrow != coo.ncol {
            return Err("cannot factorize because the CooMatrix is not square");
        }
        if coo.nrow != self.nrow {
            return Err("cannot factorize because the CooMatrix has incompatible number of rows");
        }
        self.stopwatch.reset();
        unsafe {
            match self.kind {
                LinSolKind::Umfpack => {
                    let res = solver_umfpack_factorize(
                        self.solver,
                        coo.indices_i.as_ptr(),
                        coo.indices_j.as_ptr(),
                        coo.values_aij.as_ptr(),
                        self.verbose,
                    );
                    if res != 0 {
                        return Err(Solver::handle_umfpack_error_code(res));
                    }
                    let ord = solver_umfpack_get_ordering(self.solver);
                    let sca = solver_umfpack_get_scaling(self.solver);
                    self.used_ordering = str_umfpack_ordering(ord);
                    self.used_scaling = str_umfpack_scaling(sca);
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
    /// use russell_sparse::{ConfigSolver, CooMatrix, Layout, Solver, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     // allocate a square matrix
    ///     let (nrow, ncol, nnz) = (5, 5, 13);
    ///     let mut coo = CooMatrix::new(Layout::Full, nrow, ncol, nnz)?;
    ///     coo.put(0, 0, 1.0)?; // << (0, 0, a00/2)
    ///     coo.put(0, 0, 1.0)?; // << (0, 0, a00/2)
    ///     coo.put(1, 0, 3.0)?;
    ///     coo.put(0, 1, 3.0)?;
    ///     coo.put(2, 1, -1.0)?;
    ///     coo.put(4, 1, 4.0)?;
    ///     coo.put(1, 2, 4.0)?;
    ///     coo.put(2, 2, -3.0)?;
    ///     coo.put(3, 2, 1.0)?;
    ///     coo.put(4, 2, 2.0)?;
    ///     coo.put(2, 3, 2.0)?;
    ///     coo.put(1, 4, 6.0)?;
    ///     coo.put(4, 4, 1.0)?;
    ///
    ///     // print matrix
    ///     let mut a = Matrix::new(nrow, nrow);
    ///     coo.to_matrix(&mut a)?;
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
    ///     let mut x = Vector::new(nrow);
    ///     let rhs = Vector::from(&[8.0, 45.0, -3.0, 3.0, 19.0]);
    ///
    ///     // initialize, factorize, and solve
    ///     let config = ConfigSolver::new();
    ///     let mut solver = Solver::new(config, nrow, nnz, None)?;
    ///     solver.factorize(&coo)?;
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
        if x.dim() != self.nrow || rhs.dim() != self.nrow {
            return Err("x.ndim() and rhs.ndim() must equal the number of equations");
        }
        self.stopwatch.reset();
        unsafe {
            match self.kind {
                LinSolKind::Umfpack => {
                    let res = solver_umfpack_solve(
                        self.solver,
                        x.as_mut_data().as_mut_ptr(),
                        rhs.as_data().as_ptr(),
                        self.verbose,
                    );
                    if res != 0 {
                        return Err(Solver::handle_umfpack_error_code(res));
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
    /// use russell_sparse::{ConfigSolver, CooMatrix, Layout, Solver, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     // allocate a square matrix
    ///     let (nrow, ncol, nnz) = (3, 3, 5);
    ///     let mut coo = CooMatrix::new(Layout::Full, nrow, ncol, nnz)?;
    ///     coo.put(0, 0, 0.2)?;
    ///     coo.put(0, 1, 0.2)?;
    ///     coo.put(1, 0, 0.5)?;
    ///     coo.put(1, 1, -0.25)?;
    ///     coo.put(2, 2, 0.25)?;
    ///
    ///     // print matrix
    ///     let mut a = Matrix::new(nrow, nrow);
    ///     coo.to_matrix(&mut a)?;
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
    ///     let (mut solver, x1) = Solver::compute(config, &coo, &rhs1)?;
    ///     let correct1 = "┌   ┐\n\
    ///                     │ 3 │\n\
    ///                     │ 2 │\n\
    ///                     │ 4 │\n\
    ///                     └   ┘";
    ///     assert_eq!(format!("{}", x1), correct1);
    ///
    ///     // solve again
    ///     let mut x2 = Vector::new(nrow);
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
    pub fn compute(config: ConfigSolver, coo: &CooMatrix, rhs: &Vector) -> Result<(Self, Vector), StrError> {
        if coo.nrow != coo.ncol {
            return Err("CooMatrix must be symmetric");
        }
        let mut solver = Solver::new(config, coo.nrow, coo.pos, None)?;
        let mut x = Vector::new(coo.nrow);
        solver.factorize(&coo)?;
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

    /// Returns the ordering effectively used by the solver
    pub fn get_effective_ordering(&self) -> String {
        self.used_ordering.to_string()
    }

    /// Returns the scaling effectively used by the solver
    pub fn get_effective_scaling(&self) -> String {
        self.used_scaling.to_string()
    }

    /// Handles UMFPACK error code
    fn handle_umfpack_error_code(err: i32) -> StrError {
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
            100000 => return "Error: c-code returned null pointer (UMFPACK)",
            200000 => return "Error: c-code failed to allocate memory (UMFPACK)",
            _ => return "Error: unknown error returned by c-code (UMFPACK)",
        }
    }
}

impl Drop for Solver {
    /// Tells the c-code to release memory
    fn drop(&mut self) {
        unsafe {
            match self.kind {
                LinSolKind::Umfpack => solver_umfpack_drop(self.solver),
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{ConfigSolver, CooMatrix, Solver};
    use crate::Layout;
    use russell_chk::vec_approx_eq;
    use russell_lab::Vector;

    #[test]
    fn new_works() {
        let config = ConfigSolver::new();
        let (nrow, nnz) = (2, 2);
        let solver = Solver::new(config, nrow, nnz, None).unwrap();
        assert_eq!(solver.done_factorize, false);
        assert_eq!(solver.nrow, 2);
    }

    #[test]
    fn factorize_fails_on_incompatible_triplet() {
        let config = ConfigSolver::new();
        let mut solver = Solver::new(config, 1, 1, None).unwrap();
        let coo = CooMatrix::new(Layout::Full, 2, 2, 2).unwrap();
        assert_eq!(
            solver.factorize(&coo).err(),
            Some("cannot factorize because the CooMatrix has incompatible number of rows")
        );
    }

    #[test]
    fn factorize_fails_on_singular_matrix() {
        let config = ConfigSolver::new();
        let (nrow, ncol, nnz) = (2, 2, 2);
        let mut solver = Solver::new(config, nrow, nnz, None).unwrap();
        let mut coo = CooMatrix::new(Layout::Full, nrow, ncol, nnz).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(1, 1, 0.0).unwrap();
        assert_eq!(solver.factorize(&coo), Err("Error(1): Matrix is singular"));
    }

    #[test]
    fn factorize_works() {
        let config = ConfigSolver::new();
        let (nrow, ncol, nnz) = (2, 2, 2);
        let mut solver = Solver::new(config, nrow, nnz, None).unwrap();
        let mut coo = CooMatrix::new(Layout::Full, nrow, ncol, nnz).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(1, 1, 1.0).unwrap();
        solver.factorize(&coo).unwrap();
        assert!(solver.done_factorize);
    }

    #[test]
    fn solve_fails_on_non_factorized() {
        let config = ConfigSolver::new();
        let (nrow, ncol, nnz) = (2, 2, 2);
        let mut solver = Solver::new(config, nrow, nnz, None).unwrap();
        let mut coo = CooMatrix::new(Layout::Full, nrow, ncol, nnz).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(1, 1, 1.0).unwrap();
        let mut x = Vector::new(nrow);
        let rhs = Vector::from(&[1.0, 1.0]);
        assert_eq!(
            solver.solve(&mut x, &rhs),
            Err("factorization must be done before calling solve")
        );
    }

    #[test]
    fn solve_fails_on_wrong_vectors() {
        let config = ConfigSolver::new();
        let (nrow, ncol, nnz) = (2, 2, 2);
        let mut solver = Solver::new(config, nrow, nnz, None).unwrap();
        let mut coo = CooMatrix::new(Layout::Full, nrow, ncol, nnz).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(1, 1, 1.0).unwrap();
        solver.factorize(&coo).unwrap();
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
        let (nrow, ncol, nnz) = (5, 5, 13);
        let mut solver = Solver::new(config, nrow, nnz, None).unwrap();

        // allocate a square matrix
        let mut coo = CooMatrix::new(Layout::Full, nrow, ncol, nnz).unwrap();
        coo.put(0, 0, 1.0).unwrap(); // << (0, 0, a00/2)
        coo.put(0, 0, 1.0).unwrap(); // << (0, 0, a00/2)
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

        // allocate x and rhs
        let mut x = Vector::new(5);
        let rhs = Vector::from(&[8.0, 45.0, -3.0, 3.0, 19.0]);
        let x_correct = &[1.0, 2.0, 3.0, 4.0, 5.0];

        // initialize, factorize, and solve
        solver.factorize(&coo).unwrap();
        solver.solve(&mut x, &rhs).unwrap();

        // check
        vec_approx_eq(x.as_data(), x_correct, 1e-14);
    }

    #[test]
    fn compute_works() {
        let (nrow, ncol, nnz) = (3, 3, 6);
        let mut coo = CooMatrix::new(Layout::Full, nrow, ncol, nnz).unwrap();
        coo.put(0, 0, 1.0).unwrap();
        coo.put(0, 1, 1.0).unwrap();
        coo.put(1, 0, 2.0).unwrap();
        coo.put(1, 1, 1.0).unwrap();
        coo.put(1, 2, 1.0).unwrap();
        coo.put(2, 2, 1.0).unwrap();
        let rhs1 = Vector::from(&[1.0, 2.0, 3.0]);
        let rhs2 = Vector::from(&[2.0, 4.0, 6.0]);
        let config = ConfigSolver::new();
        let (mut solver, x1) = Solver::compute(config, &coo, &rhs1).unwrap();
        vec_approx_eq(x1.as_data(), &[-2.0, 3.0, 3.0], 1e-15);
        let mut x2 = Vector::new(nrow);
        solver.solve(&mut x2, &rhs2).unwrap();
    }

    #[test]
    fn get_elapsed_times_works() {
        let config = ConfigSolver::new();
        let (nrow, nnz) = (2, 2);
        let solver = Solver::new(config, nrow, nnz, None).unwrap();
        let times = solver.get_elapsed_times();
        assert_eq!(times, (0, 0));
    }
}
