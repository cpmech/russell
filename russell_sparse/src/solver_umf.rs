use super::*;
use russell_lab::*;
use std::fmt;

#[repr(C)]
pub(crate) struct ExtSolverUMF {
    data: [u8; 0],
    marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

extern "C" {
    fn new_solver_umf(symmetry: i32) -> *mut ExtSolverUMF;
    fn drop_solver_umf(solver: *mut ExtSolverUMF);
    fn solver_umf_initialize(
        solver: *mut ExtSolverUMF,
        n: i32,
        nnz: i32,
        indices_i: *const i32,
        indices_j: *const i32,
        values_a: *const f64,
        ordering: i32,
        scaling: i32,
    ) -> i32;
    fn solver_umf_factorize(solver: *mut ExtSolverUMF, verbose: i32) -> i32;
    fn solver_umf_solve(solver: *mut ExtSolverUMF, x: *mut f64, rhs: *const f64, verbose: i32) -> i32;
}

/// Implements Tim Davis' UMFPACK Solver
pub struct SolverUMF {
    config: ConfigSolver,      // configuration
    done_initialize: bool,     // initialization completed
    done_factorize: bool,      // factorization completed
    ndim: usize,               // number of equations == nrow(a) where a*x=rhs
    solver: *mut ExtSolverUMF, // data allocated by the c-code
}

impl SolverUMF {
    /// Creates a new solver
    ///
    /// # Input
    ///
    /// `symmetric` -- Tells wether the system matrix is general symmetric or not
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), &'static str> {
    /// use russell_sparse::*;
    /// let config = ConfigSolver::new();
    /// let solver = SolverUMF::new(config)?;
    /// let correct: &str = "solver_kind     = UMF\n\
    ///                      done_initialize = false\n\
    ///                      done_factorize  = false\n\
    ///                      ndim            = 0\n";
    /// assert_eq!(format!("{}", solver), correct);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(config: ConfigSolver) -> Result<Self, &'static str> {
        unsafe {
            let solver = new_solver_umf(config.symmetry);
            if solver.is_null() {
                return Err("c-code failed to allocate SolverUMF");
            }
            Ok(SolverUMF {
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
    /// let mut solver = SolverUMF::new(config)?;
    /// let mut trip = SparseTriplet::new(2, 2, 2, false)?;
    /// trip.put(0, 0, 1.0);
    /// trip.put(1, 1, 1.0);
    /// solver.initialize(&trip)?;
    /// let correct: &str = "solver_kind     = UMF\n\
    ///                      done_initialize = true\n\
    ///                      done_factorize  = false\n\
    ///                      ndim            = 2\n";
    /// assert_eq!(format!("{}", solver), correct);
    /// # Ok(())
    /// # }
    /// ```
    pub fn initialize(&mut self, trip: &SparseTriplet) -> Result<(), &'static str> {
        if trip.nrow != trip.ncol {
            return Err("the matrix represented by the triplet must be square");
        }
        let n = to_i32(trip.nrow);
        let nnz = to_i32(trip.pos);
        unsafe {
            if self.done_initialize {
                drop_solver_umf(self.solver);
                let solver = new_solver_umf(self.config.symmetry);
                if solver.is_null() {
                    return Err("c-code failed to allocate SolverUMF");
                }
                self.solver = solver;
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
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), &'static str> {
    /// use russell_sparse::*;
    /// let config = ConfigSolver::new();
    /// let mut solver = SolverUMF::new(config)?;
    /// let mut trip = SparseTriplet::new(2, 2, 2, false)?;
    /// trip.put(0, 0, 1.0);
    /// trip.put(1, 1, 1.0);
    /// solver.initialize(&trip)?;
    /// solver.factorize()?;
    /// let correct: &str = "solver_kind     = UMF\n\
    ///                      done_initialize = true\n\
    ///                      done_factorize  = true\n\
    ///                      ndim            = 2\n";
    /// assert_eq!(format!("{}", solver), correct);
    /// # Ok(())
    /// # }
    /// ```
    pub fn factorize(&mut self) -> Result<(), &'static str> {
        if !self.done_initialize {
            return Err("initialization must be done before factorization");
        }
        unsafe {
            let res = solver_umf_factorize(self.solver, self.config.verbose);
            if res != 0 {
                return Err(self.handle_error_code(res));
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
    /// trip.put(0, 0, 1.0); // << duplicated
    /// trip.put(0, 0, 1.0); // << duplicated
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
    /// let mut solver = SolverUMF::new(config)?;
    /// solver.initialize(&trip)?;
    /// solver.factorize()?;
    /// solver.solve(&mut x, &rhs)?;
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
    pub fn solve(&mut self, x: &mut Vector, rhs: &Vector) -> Result<(), &'static str> {
        if !self.done_factorize {
            return Err("factorization must be done before solution");
        }
        if x.dim() != self.ndim || rhs.dim() != self.ndim {
            return Err("x.ndim() and rhs.ndim() must equal the number of equations");
        }
        unsafe {
            let res = solver_umf_solve(
                self.solver,
                x.as_mut_data().as_mut_ptr(),
                rhs.as_data().as_ptr(),
                self.config.verbose,
            );
            if res != 0 {
                return Err(self.handle_error_code(res));
            }
        }
        Ok(())
    }

    /// Handles error code
    fn handle_error_code(&self, err: i32) -> &'static str {
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
            100000 => return "Error: c-code returned null pointer",
            200000 => return "Error: c-code failed to allocate memory",
            _ => return "Error: unknown error returned by SolverUMF (c-code)",
        }
    }
}

impl Drop for SolverUMF {
    /// Tells the c-code to release memory
    fn drop(&mut self) {
        unsafe {
            drop_solver_umf(self.solver);
        }
    }
}

impl fmt::Display for SolverUMF {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "solver_kind     = UMF\n\
             done_initialize = {}\n\
             done_factorize  = {}\n\
             ndim            = {}\n",
            self.done_initialize, self.done_factorize, self.ndim,
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
        let solver = SolverUMF::new(config)?;
        assert_eq!(solver.solver.is_null(), false);
        Ok(())
    }

    #[test]
    fn display_trait_works() -> Result<(), &'static str> {
        let config = ConfigSolver::new();
        let solver = SolverUMF::new(config)?;
        let correct: &str = "solver_kind     = UMF\n\
                             done_initialize = false\n\
                             done_factorize  = false\n\
                             ndim            = 0\n";
        assert_eq!(format!("{}", solver), correct);
        Ok(())
    }

    #[test]
    fn initialize_fails_on_rect_matrix() -> Result<(), &'static str> {
        let config = ConfigSolver::new();
        let mut solver = SolverUMF::new(config)?;
        let trip_rect = SparseTriplet::new(3, 2, 1, false)?;
        assert_eq!(
            solver.initialize(&trip_rect),
            Err("the matrix represented by the triplet must be square")
        );
        Ok(())
    }

    #[test]
    fn initialize_works() -> Result<(), &'static str> {
        let config = ConfigSolver::new();
        let mut solver = SolverUMF::new(config)?;
        let mut trip = SparseTriplet::new(2, 2, 2, false)?;
        trip.put(0, 0, 1.0);
        trip.put(1, 1, 1.0);
        solver.initialize(&trip)?;
        assert!(solver.done_initialize);
        Ok(())
    }

    #[test]
    fn factorize_fails_on_non_initialized() -> Result<(), &'static str> {
        let config = ConfigSolver::new();
        let mut solver = SolverUMF::new(config)?;
        assert_eq!(
            solver.factorize(),
            Err("initialization must be done before factorization")
        );
        Ok(())
    }

    #[test]
    fn factorize_fails_on_singular_matrix() -> Result<(), &'static str> {
        let config = ConfigSolver::new();
        let mut solver = SolverUMF::new(config)?;
        let mut trip = SparseTriplet::new(2, 2, 2, false)?;
        trip.put(0, 0, 1.0);
        trip.put(1, 1, 0.0);
        solver.initialize(&trip)?;
        assert_eq!(solver.factorize(), Err("Error(1): Matrix is singular"));
        Ok(())
    }

    #[test]
    fn factorize_works() -> Result<(), &'static str> {
        let config = ConfigSolver::new();
        let mut solver = SolverUMF::new(config)?;
        let mut trip = SparseTriplet::new(2, 2, 2, false)?;
        trip.put(0, 0, 1.0);
        trip.put(1, 1, 1.0);
        solver.initialize(&trip)?;
        solver.factorize()?;
        assert!(solver.done_factorize);
        Ok(())
    }

    #[test]
    fn solve_fails_on_non_factorized() -> Result<(), &'static str> {
        let config = ConfigSolver::new();
        let mut solver = SolverUMF::new(config)?;
        let mut trip = SparseTriplet::new(2, 2, 2, false)?;
        trip.put(0, 0, 1.0);
        trip.put(1, 1, 1.0);
        solver.initialize(&trip)?;
        let mut x = Vector::new(2);
        let rhs = Vector::from(&[1.0, 1.0]);
        assert_eq!(
            solver.solve(&mut x, &rhs),
            Err("factorization must be done before solution")
        );
        Ok(())
    }

    #[test]
    fn solve_fails_on_wrong_vectors() -> Result<(), &'static str> {
        let config = ConfigSolver::new();
        let mut solver = SolverUMF::new(config)?;
        let mut trip = SparseTriplet::new(2, 2, 2, false)?;
        trip.put(0, 0, 1.0);
        trip.put(1, 1, 1.0);
        solver.initialize(&trip)?;
        solver.factorize()?;
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
        Ok(())
    }

    #[test]
    fn solve_works() -> Result<(), &'static str> {
        let config = ConfigSolver::new();
        let mut solver = SolverUMF::new(config)?;

        // allocate a square matrix
        let mut trip = SparseTriplet::new(5, 5, 13, false)?;
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

        // allocate x and rhs
        let mut x = Vector::new(5);
        let rhs = Vector::from(&[8.0, 45.0, -3.0, 3.0, 19.0]);
        let x_correct = &[1.0, 2.0, 3.0, 4.0, 5.0];

        // initialize, factorize, and solve
        solver.initialize(&trip)?;
        solver.factorize()?;
        solver.solve(&mut x, &rhs)?;

        // check
        assert_vec_approx_eq!(x.as_data(), x_correct, 1e-14);
        Ok(())
    }

    #[test]
    fn reinitialize_works() -> Result<(), &'static str> {
        let config = ConfigSolver::new();
        let mut solver = SolverUMF::new(config)?;
        let mut trip = SparseTriplet::new(2, 2, 2, false)?;
        trip.put(0, 0, 1.0);
        trip.put(1, 1, 1.0);
        solver.initialize(&trip)?;
        solver.initialize(&trip)?;
        Ok(())
    }

    #[test]
    fn handle_error_code_works() -> Result<(), &'static str> {
        let default = "Error: unknown error returned by SolverUMF (c-code)";
        let config = ConfigSolver::new();
        let solver = SolverUMF::new(config)?;
        for c in &[1, 2, 3, -1, -3, -4, -5, -6, -8, -11, -13, -15, -17, -18, -911] {
            let res = solver.handle_error_code(*c);
            assert!(res.len() > 0);
            assert_ne!(res, default);
        }
        assert_eq!(solver.handle_error_code(100000), "Error: c-code returned null pointer");
        assert_eq!(
            solver.handle_error_code(200000),
            "Error: c-code failed to allocate memory"
        );
        assert_eq!(solver.handle_error_code(123), default);
        Ok(())
    }
}
