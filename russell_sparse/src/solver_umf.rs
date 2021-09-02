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
        verbose: i32,
    ) -> i32;
    fn solver_umf_factorize(solver: *mut ExtSolverUMF, verbose: i32) -> i32;
    fn solver_umf_solve(solver: *mut ExtSolverUMF, rhs: *mut f64, verbose: i32) -> i32;
}

/// Implements Davis' UMFPACK Solver
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
    pub fn new(symmetric: bool) -> Result<Self, &'static str> {
        let sym: i32 = if symmetric { 1 } else { 0 };
        unsafe {
            let solver = new_solver_umf(sym);
            if solver.is_null() {
                return Err("c-code failed to allocate SolverUMF");
            }
            Ok(SolverUMF {
                symmetric: sym,
                ordering: code_ordering(EnumOrdering::Auto),
                scaling: code_scaling(EnumScaling::Auto),
                done_initialize: false,
                done_factorize: false,
                ndim: 0,
                solver,
            })
        }
    }

    /// Sets the method to compute a symmetric permutation (ordering)
    pub fn set_ordering(&mut self, selection: EnumOrdering) {
        self.ordering = code_ordering(selection);
    }

    /// Sets the scaling strategy
    pub fn set_scaling(&mut self, selection: EnumScaling) {
        self.scaling = code_scaling(selection);
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
        copy_vector(x, rhs)?;
        let verb: i32 = if verbose { 1 } else { 0 };
        unsafe {
            let res = solver_umf_solve(self.solver, x.as_mut_data().as_mut_ptr(), verb);
            if res != 0 {
                return Err(self.handle_error_code(res));
            }
        }
        Ok(())
    }

    /// Handles error code
    fn handle_error_code(&self, err: i32) -> &'static str {
        match err {
            _ => "TODO",
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
            "===========================\n\
            SolverUMF\n\
            ---------------------------\n\
            ordering           = {}\n\
            scaling            = {}\n\
            done_initialize    = {}\n\
            done_factorize     = {}\n\
            ===========================",
            self.ordering, self.scaling, self.done_initialize, self.done_factorize,
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
}
