use super::*;
use russell_lab::*;
use std::fmt;

#[repr(C)]
pub(crate) struct ExternalSolverMumps {
    data: [u8; 0],
    marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

extern "C" {
    fn new_solver_mumps(symmetry: i32, verbose: i32) -> *mut ExternalSolverMumps;
    fn drop_solver_mumps(solver: *mut ExternalSolverMumps);
    fn solver_mumps_initialize(
        solver: *mut ExternalSolverMumps,
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
    fn solver_mumps_factorize(solver: *mut ExternalSolverMumps, verbose: i32) -> i32;
    fn solver_mumps_solve(
        solver: *mut ExternalSolverMumps,
        n: i32,
        x: *mut f64,
        rhs: *const f64,
        verbose: i32,
    ) -> i32;
}

/// Implements the sparse solver called MUMPS (not the infection!)
pub struct SolverMumps {
    ordering: i32,           // symmetric permutation (ordering). ICNTL(7)
    scaling: i32,            // scaling strategy. ICNTL(8)
    pct_inc_workspace: i32,  // percentage increase in the estimated working space. ICNTL(14)
    max_work_memory: i32,    // maximum size of the working memory in mega bytes. ICNTL(23)
    openmp_num_threads: i32, // number of OpenMP threads. ICNTL(16)

    done_initialize: bool, // initialization completed
    done_factorize: bool,  // factorization completed

    // solver holds the c-side data that needs to be allocated by the c-code
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
    /// let correct: &str = "==========================\n\
    ///                      SolverMumps\n\
    ///                      --------------------------\n\
    ///                      name               = MUMPS\n\
    ///                      ordering           = 5\n\
    ///                      scaling            = 77\n\
    ///                      pct_inc_workspace  = 100\n\
    ///                      max_work_memory    = 0\n\
    ///                      openmp_num_threads = 1\n\
    ///                      done_analyze       = false\n\
    ///                      done_factorize     = false\n\
    ///                      ==========================";
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
            Ok(SolverMumps {
                ordering: enum_mumps_ordering(EnumMumpsOrdering::Metis),
                scaling: enum_mumps_scaling(EnumMumpsScaling::Auto),
                pct_inc_workspace: 100,
                max_work_memory: 0, // auto
                openmp_num_threads: 1,
                done_initialize: false,
                done_factorize: false,
                solver,
            })
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

    /// Sets the method to compute a symmetric permutation (ordering)
    pub fn set_ordering(&mut self, selection: EnumMumpsOrdering) {
        self.ordering = enum_mumps_ordering(selection);
    }

    /// Sets the scaling strategy
    pub fn set_scaling(&mut self, selection: EnumMumpsScaling) {
        self.scaling = enum_mumps_scaling(selection);
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
        let verb: i32 = if verbose { 1 } else { 0 };
        let n = to_i32(trip.nrow);
        let nnz = to_i32(trip.pos);
        unsafe {
            let merr = solver_mumps_initialize(
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
            if merr != 0 {
                return Err(self.handle_error_code(merr));
            }
            self.done_initialize = true;
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
            let merr = solver_mumps_factorize(self.solver, verb);
            if merr != 0 {
                return Err(self.handle_error_code(merr));
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
        let n = to_i32(x.dim());
        let verb: i32 = if verbose { 1 } else { 0 };
        unsafe {
            let merr = solver_mumps_solve(
                self.solver,
                n,
                x.as_mut_data().as_mut_ptr(),
                rhs.as_data().as_ptr(),
                verb,
            );
            if merr != 0 {
                return Err(self.handle_error_code(merr));
            }
        }
        Ok(())
    }

    /// Handles MUMPS error code
    fn handle_error_code(&self, merr: i32) -> &'static str {
        match merr {
            -6 => return "ERROR (-6) The matrix is singular in structure",
            -9 => return "ERROR (-9) The main internal real/complex work-array is too small",
            -10 => return "ERROR (-10) The matrix is numerically singular",
            -13 => return "ERROR (-13) There is a problem with workspace allocation",
            -19 => return "ERROR (-19) The maximum allowed size of working memory is too small",
            _ => return "ERROR: Some error occurred with MUMPS solver",
        }
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
            "==========================\n\
            SolverMumps\n\
            --------------------------\n\
            name               = {}\n\
            ordering           = {}\n\
            scaling            = {}\n\
            pct_inc_workspace  = {}\n\
            max_work_memory    = {}\n\
            openmp_num_threads = {}\n\
            done_initialize    = {}\n\
            done_factorize     = {}\n\
            ==========================",
            self.name(),
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
        let solver = SolverMumps::new(EnumMumpsSymmetry::No, true)?;
        assert_eq!(solver.solver.is_null(), false);
        Ok(())
    }

    #[test]
    fn name_works() -> Result<(), &'static str> {
        let solver = SolverMumps::new(EnumMumpsSymmetry::No, true)?;
        assert_eq!(solver.name(), "MUMPS");
        Ok(())
    }

    #[test]
    fn set_ordering() -> Result<(), &'static str> {
        let mut solver = SolverMumps::new(EnumMumpsSymmetry::No, true)?;
        solver.set_ordering(EnumMumpsOrdering::Amf);
        assert_eq!(solver.ordering, 2);
        Ok(())
    }

    #[test]
    fn set_scaling_works() -> Result<(), &'static str> {
        let mut solver = SolverMumps::new(EnumMumpsSymmetry::No, true)?;
        solver.set_scaling(EnumMumpsScaling::RowCol);
        assert_eq!(solver.scaling, 4);
        Ok(())
    }

    #[test]
    fn set_pct_inc_workspace_works() -> Result<(), &'static str> {
        let mut solver = SolverMumps::new(EnumMumpsSymmetry::No, true)?;
        solver.set_pct_inc_workspace(15);
        assert_eq!(solver.pct_inc_workspace, 15);
        Ok(())
    }

    #[test]
    fn set_max_work_memory_works() -> Result<(), &'static str> {
        let mut solver = SolverMumps::new(EnumMumpsSymmetry::No, true)?;
        solver.set_max_work_memory(500);
        assert_eq!(solver.max_work_memory, 500);
        Ok(())
    }

    #[test]
    fn set_openmp_num_threads_works() -> Result<(), &'static str> {
        let mut solver = SolverMumps::new(EnumMumpsSymmetry::No, true)?;
        solver.set_openmp_num_threads(3);
        assert_eq!(solver.openmp_num_threads, 3);
        Ok(())
    }

    #[test]
    fn display_trait_works() -> Result<(), &'static str> {
        let solver = SolverMumps::new(EnumMumpsSymmetry::No, true)?;
        let correct: &str = "==========================\n\
                             SolverMumps\n\
                             --------------------------\n\
                             name               = MUMPS\n\
                             ordering           = 5\n\
                             scaling            = 77\n\
                             pct_inc_workspace  = 100\n\
                             max_work_memory    = 0\n\
                             openmp_num_threads = 1\n\
                             done_initialize    = false\n\
                             done_factorize     = false\n\
                             ==========================";
        assert_eq!(format!("{}", solver), correct);
        Ok(())
    }

    #[test]
    fn analyze_factorize_and_solve_works() -> Result<(), &'static str> {
        let mut solver = SolverMumps::new(EnumMumpsSymmetry::No, true)?;
        let mut trip = SparseTriplet::new(5, 5, 13)?;
        trip.put(0, 0, 1.0)?; // << duplicated
        trip.put(0, 0, 1.0)?; // << duplicated
        trip.put(1, 0, 3.0)?;
        trip.put(0, 1, 3.0)?;
        trip.put(2, 1, -1.0)?;
        trip.put(4, 1, 4.0)?;
        trip.put(1, 2, 4.0)?;
        trip.put(2, 2, -3.0)?;
        trip.put(3, 2, 1.0)?;
        trip.put(4, 2, 2.0)?;
        trip.put(2, 3, 2.0)?;
        trip.put(1, 4, 6.0)?;
        trip.put(4, 4, 1.0)?;
        let mut x = Vector::new(5);
        let rhs = Vector::from(&[8.0, 45.0, -3.0, 3.0, 19.0]);
        solver.initialize(&trip, false)?;
        assert_eq!(solver.done_initialize, true);
        solver.factorize(false)?;
        assert_eq!(solver.done_factorize, true);
        solver.solve(&mut x, &rhs, false)?;
        assert_vec_approx_eq!(x.as_data(), &[1.0, 2.0, 3.0, 4.0, 5.0], 1e-15);
        Ok(())
    }
}
