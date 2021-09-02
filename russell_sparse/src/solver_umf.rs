use super::*;
use russell_lab::*;
use std::fmt;

#[repr(C)]
pub(crate) struct ExtSolverUMF {
    data: [u8; 0],
    marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

extern "C" {
    fn new_solver_umf(symmetric: i32) -> *mut ExtSolverUMF;
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
    fn solver_umf_solve(
        solver: *mut ExtSolverUMF,
        x: *mut f64,
        rhs: *const f64,
        verbose: i32,
    ) -> i32;
}

/// Implements Tim Davis' UMFPACK Solver
pub struct SolverUMF {
    symmetric: i32,            // symmetric flag (0 or 1)
    ordering: i32,             // symmetric permutation (ordering)
    scaling: i32,              // scaling strategy
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
    /// let solver = SolverUMF::new(false)?;
    /// let correct: &str = "==============================\n\
    ///                      SolverUMF\n\
    ///                      ------------------------------\n\
    ///                      symmetric          = false\n\
    ///                      ordering           = Default\n\
    ///                      scaling            = Default\n\
    ///                      done_initialize    = false\n\
    ///                      done_factorize     = false\n\
    ///                      ==============================";
    /// assert_eq!(format!("{}", solver), correct);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(symmetric: bool) -> Result<Self, &'static str> {
        let sym: i32 = if symmetric { 1 } else { 0 };
        unsafe {
            let solver = new_solver_umf(sym);
            if solver.is_null() {
                return Err("c-code failed to allocate SolverUMF");
            }
            Ok(SolverUMF {
                symmetric: sym,
                ordering: EnumUmfOrdering::Default as i32,
                scaling: EnumUmfScaling::Default as i32,
                done_initialize: false,
                done_factorize: false,
                ndim: 0,
                solver,
            })
        }
    }

    /// Sets the method to compute a symmetric permutation (ordering)
    pub fn set_ordering(&mut self, selection: EnumUmfOrdering) {
        self.ordering = selection as i32;
    }

    /// Sets the scaling strategy
    pub fn set_scaling(&mut self, selection: EnumUmfScaling) {
        self.scaling = selection as i32;
    }

    /// Initializes the solver
    pub fn initialize(&mut self, trip: &SparseTriplet) -> Result<(), &'static str> {
        if trip.nrow != trip.ncol {
            return Err("the matrix represented by the triplet must be square");
        }
        let n = to_i32(trip.nrow);
        let nnz = to_i32(trip.pos);
        unsafe {
            if self.done_initialize {
                drop_solver_umf(self.solver);
                let solver = new_solver_umf(self.symmetric);
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
                self.ordering,
                self.scaling,
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
            let res = solver_umf_factorize(self.solver, verb);
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
        let verb: i32 = if verbose { 1 } else { 0 };
        unsafe {
            let res = solver_umf_solve(
                self.solver,
                x.as_mut_data().as_mut_ptr(),
                rhs.as_data().as_ptr(),
                verb,
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
    /// Display some information about this solver
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "==============================\n\
            SolverUMF\n\
            ------------------------------\n\
            symmetric          = {}\n\
            ordering           = {}\n\
            scaling            = {}\n\
            done_initialize    = {}\n\
            done_factorize     = {}\n\
            ==============================",
            if self.symmetric == 1 { "true" } else { "false" },
            str_umf_ordering(self.ordering),
            str_umf_scaling(self.scaling),
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
        let solver = SolverUMF::new(false)?;
        assert_eq!(solver.solver.is_null(), false);
        Ok(())
    }

    #[test]
    fn set_ordering() -> Result<(), &'static str> {
        let mut solver = SolverUMF::new(false)?;
        solver.set_ordering(EnumUmfOrdering::Metis);
        assert_eq!(solver.ordering, 4);
        Ok(())
    }

    #[test]
    fn set_scaling_works() -> Result<(), &'static str> {
        let mut solver = SolverUMF::new(false)?;
        solver.set_scaling(EnumUmfScaling::Max);
        assert_eq!(solver.scaling, 1);
        Ok(())
    }

    #[test]
    fn display_trait_works() -> Result<(), &'static str> {
        let solver = SolverUMF::new(false)?;
        let correct: &str = "==============================\n\
                             SolverUMF\n\
                             ------------------------------\n\
                             symmetric          = false\n\
                             ordering           = Default\n\
                             scaling            = Default\n\
                             done_initialize    = false\n\
                             done_factorize     = false\n\
                             ==============================";
        assert_eq!(format!("{}", solver), correct);
        Ok(())
    }

    #[test]
    fn initialize_fails_on_rect_matrix() -> Result<(), &'static str> {
        let mut solver = SolverUMF::new(false)?;
        let trip_rect = SparseTriplet::new(3, 2, 1, false)?;
        assert_eq!(
            solver.initialize(&trip_rect),
            Err("the matrix represented by the triplet must be square")
        );
        Ok(())
    }

    #[test]
    fn initialize_works() -> Result<(), &'static str> {
        let mut solver = SolverUMF::new(false)?;
        let mut trip = SparseTriplet::new(2, 2, 2, false)?;
        trip.put(0, 0, 1.0);
        trip.put(1, 1, 1.0);
        solver.initialize(&trip)?;
        assert!(solver.done_initialize);
        Ok(())
    }

    #[test]
    fn factorize_fails_on_non_initialized() -> Result<(), &'static str> {
        let mut solver = SolverUMF::new(false)?;
        assert_eq!(
            solver.factorize(false),
            Err("initialization must be done before factorization")
        );
        Ok(())
    }

    #[test]
    fn factorize_fails_on_singular_matrix() -> Result<(), &'static str> {
        let mut solver = SolverUMF::new(false)?;
        let mut trip = SparseTriplet::new(2, 2, 2, false)?;
        trip.put(0, 0, 1.0);
        trip.put(1, 1, 0.0);
        solver.initialize(&trip)?;
        assert_eq!(solver.factorize(false), Err("Error(1): Matrix is singular"));
        Ok(())
    }

    #[test]
    fn factorize_works() -> Result<(), &'static str> {
        let mut solver = SolverUMF::new(false)?;
        let mut trip = SparseTriplet::new(2, 2, 2, false)?;
        trip.put(0, 0, 1.0);
        trip.put(1, 1, 1.0);
        solver.initialize(&trip)?;
        solver.factorize(false)?;
        assert!(solver.done_factorize);
        Ok(())
    }

    #[test]
    fn solve_fails_on_non_factorized() -> Result<(), &'static str> {
        let mut solver = SolverUMF::new(false)?;
        let mut trip = SparseTriplet::new(2, 2, 2, false)?;
        trip.put(0, 0, 1.0);
        trip.put(1, 1, 1.0);
        solver.initialize(&trip)?;
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
        let mut solver = SolverUMF::new(false)?;
        let mut trip = SparseTriplet::new(2, 2, 2, false)?;
        trip.put(0, 0, 1.0);
        trip.put(1, 1, 1.0);
        solver.initialize(&trip)?;
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
        let mut solver = SolverUMF::new(false)?;

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
        solver.factorize(false)?;
        solver.solve(&mut x, &rhs, false)?;

        // check
        assert_vec_approx_eq!(x.as_data(), x_correct, 1e-14);
        Ok(())
    }

    #[test]
    fn reinitialize_works() -> Result<(), &'static str> {
        let mut solver = SolverUMF::new(false)?;
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
        let solver = SolverUMF::new(false)?;
        for c in &[
            1, 2, 3, -1, -3, -4, -5, -6, -8, -11, -13, -15, -17, -18, -911,
        ] {
            let res = solver.handle_error_code(*c);
            assert!(res.len() > 0);
            assert_ne!(res, default);
        }
        assert_eq!(
            solver.handle_error_code(100000),
            "Error: c-code returned null pointer"
        );
        assert_eq!(
            solver.handle_error_code(200000),
            "Error: c-code failed to allocate memory"
        );
        assert_eq!(solver.handle_error_code(123), default);
        Ok(())
    }
}
