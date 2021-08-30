use super::*;
use std::fmt;

#[repr(C)]
pub(crate) struct ExternalSolverMumps {
    data: [u8; 0],
    marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

extern "C" {
    fn new_solver_mumps(symmetry: i32, verbose: i32) -> *mut ExternalSolverMumps;
    fn drop_solver_mumps(solver: *mut ExternalSolverMumps);
    fn solver_mumps_analyze(
        solver: *mut ExternalSolverMumps,
        trip: *mut ExternalSparseTriplet,
        ordering: i32,
        scaling: i32,
        pct_inc_workspace: i32,
        max_work_memory: i32,
        openmp_num_threads: i32,
        verbose: i32,
    ) -> i32;
}

/// Implements the sparse solver called MUMPS (not the infection! but MUltifrontal Massively Parallel sparse direct Solver)
pub struct SolverMumps {
    solver: *mut ExternalSolverMumps,
}

impl SolverMumps {
    /// Creates a new sparse solver called MUMPS (not the infection!)
    ///
    /// # Input
    ///
    /// `symmetry` -- Tells wether the system matrix is unsymmetric, positive-definite symmetric, or general symmetric
    /// `verbose` -- Prints MUMPS messages on the default terminal (we cannot control where)
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), &'static str> {
    /// use russell_sparse::*;
    /// let solver = SolverMumps::new(EnumMumpsSymmetry::No, true)?;
    /// let correct: &str = "=========================\n\
    ///                      SolverMumps\n\
    ///                      -------------------------\n\
    ///                      name = MUMPS\n\
    ///                      =========================";
    /// assert_eq!(format!("{}", solver), correct);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(symmetry: EnumMumpsSymmetry, verbose: bool) -> Result<Self, &'static str> {
        let verb: i32 = if verbose { 1 } else { 0 };
        let sym = enum_mumps_symmetry(symmetry);
        unsafe {
            let solver = new_solver_mumps(sym, verb);
            if solver.is_null() {
                return Err("c-code failed to allocate SolverMumps");
            }
            Ok(SolverMumps { solver })
        }
    }

    /// Returns the name of this solver
    /// ```
    /// # fn main() -> Result<(), &'static str> {
    /// use russell_sparse::*;
    /// let solver = SolverMumps::new(EnumMumpsSymmetry::No, true)?;
    /// assert_eq!(solver.name(), "MUMPS");
    /// # Ok(())
    /// # }
    /// ```
    pub fn name(&self) -> &'static str {
        "MUMPS"
    }
}

impl Drop for SolverMumps {
    /// Tells the c-code to release memory
    fn drop(&mut self) {
        unsafe {
            drop_solver_mumps(self.solver);
        }
    }
}

impl fmt::Display for SolverMumps {
    /// Implements the Display trait
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "=========================\n\
             SolverMumps\n\
             -------------------------\n\
             name = {}\n\
             =========================",
            self.name()
        )?;
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    // use russell_chk::*;

    #[test]
    fn new_works() -> Result<(), &'static str> {
        let solver = SolverMumps::new(EnumMumpsSymmetry::No, true)?;
        assert_eq!(solver.solver.is_null(), false);
        Ok(())
    }

    #[test]
    fn display_trait_works() -> Result<(), &'static str> {
        let solver = SolverMumps::new(EnumMumpsSymmetry::No, true)?;
        let correct: &str = "=========================\n\
                             SolverMumps\n\
                             -------------------------\n\
                             name = MUMPS\n\
                             =========================";
        assert_eq!(format!("{}", solver), correct);
        Ok(())
    }
}
