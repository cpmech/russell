use super::*;
use std::fmt;

pub struct ConfigSolver {
    pub(crate) symmetry: i32,           // symmetry code
    pub(crate) ordering: i32,           // symmetric permutation (ordering)
    pub(crate) scaling: i32,            // scaling strategy
    pub(crate) pct_inc_workspace: i32,  // % increase in the estimated working space (MMP-only)
    pub(crate) max_work_memory: i32,    // max size of the working memory in mega bytes (MMP-only)
    pub(crate) openmp_num_threads: i32, // number of OpenMP threads (MMP-only)
    pub(crate) verbose: i32,            // show messages, when available or possible
}

impl ConfigSolver {
    /// Returns a default configuration
    pub fn new() -> Self {
        ConfigSolver {
            symmetry: EnumSymmetry::No as i32,
            ordering: EnumOrdering::Auto as i32,
            scaling: EnumScaling::Auto as i32,
            pct_inc_workspace: 100, // (MMP-only)
            max_work_memory: 0,     // (MMP-only) 0 => Auto
            openmp_num_threads: 1,  // (MMP-only)
            verbose: 0,
        }
    }

    /// Sets the method to compute a symmetric permutation (ordering)
    ///
    /// # Example
    ///
    /// ```
    /// use russell_sparse::*;
    /// let mut config = ConfigSolver::new();
    /// config.set_ordering(EnumOrdering::Metis);
    /// let correct: &str = "symmetry           = No\n\
    ///                      ordering           = Metis\n\
    ///                      scaling            = Auto\n\
    ///                      pct_inc_workspace  = 100\n\
    ///                      max_work_memory    = 0\n\
    ///                      openmp_num_threads = 1\n\
    ///                      verbose            = false\n";
    /// assert_eq!(format!("{}", config), correct);
    /// ```
    pub fn set_ordering(&mut self, selection: EnumOrdering) {
        self.ordering = selection as i32;
    }

    /// Sets the scaling strategy
    ///
    /// # Example
    ///
    /// ```
    /// use russell_sparse::*;
    /// let mut config = ConfigSolver::new();
    /// config.set_scaling(EnumScaling::No);
    /// let correct: &str = "symmetry           = No\n\
    ///                      ordering           = Auto\n\
    ///                      scaling            = No\n\
    ///                      pct_inc_workspace  = 100\n\
    ///                      max_work_memory    = 0\n\
    ///                      openmp_num_threads = 1\n\
    ///                      verbose            = false\n";
    /// assert_eq!(format!("{}", config), correct);
    /// ```
    pub fn set_scaling(&mut self, selection: EnumScaling) {
        self.scaling = selection as i32;
    }

    /// Sets the percentage increase in the estimated working space (MMP-only)
    ///
    /// # Example
    ///
    /// ```
    /// use russell_sparse::*;
    /// let mut config = ConfigSolver::new();
    /// config.set_pct_inc_workspace(25);
    /// let correct: &str = "symmetry           = No\n\
    ///                      ordering           = Auto\n\
    ///                      scaling            = Auto\n\
    ///                      pct_inc_workspace  = 25\n\
    ///                      max_work_memory    = 0\n\
    ///                      openmp_num_threads = 1\n\
    ///                      verbose            = false\n";
    /// assert_eq!(format!("{}", config), correct);
    /// ```
    pub fn set_pct_inc_workspace(&mut self, value: usize) {
        self.pct_inc_workspace = to_i32(value);
    }

    /// Sets the maximum size of the working memory in mega bytes (MMP-only)
    ///
    /// # Example
    ///
    /// ```
    /// use russell_sparse::*;
    /// let mut config = ConfigSolver::new();
    /// config.set_max_work_memory(1234);
    /// let correct: &str = "symmetry           = No\n\
    ///                      ordering           = Auto\n\
    ///                      scaling            = Auto\n\
    ///                      pct_inc_workspace  = 100\n\
    ///                      max_work_memory    = 1234\n\
    ///                      openmp_num_threads = 1\n\
    ///                      verbose            = false\n";
    /// assert_eq!(format!("{}", config), correct);
    /// ```
    pub fn set_max_work_memory(&mut self, value: usize) {
        self.max_work_memory = to_i32(value);
    }

    /// Sets the number of OpenMP threads (MMP-only)
    ///
    /// # Example
    ///
    /// ```
    /// use russell_sparse::*;
    /// let mut config = ConfigSolver::new();
    /// config.set_openmp_num_threads(4);
    /// let correct: &str = "symmetry           = No\n\
    ///                      ordering           = Auto\n\
    ///                      scaling            = Auto\n\
    ///                      pct_inc_workspace  = 100\n\
    ///                      max_work_memory    = 0\n\
    ///                      openmp_num_threads = 4\n\
    ///                      verbose            = false\n";
    /// assert_eq!(format!("{}", config), correct);
    /// ```
    pub fn set_openmp_num_threads(&mut self, value: usize) {
        self.openmp_num_threads = to_i32(value);
    }

    /// Sets verbose mode to show messages when available or possible
    ///
    /// # Example
    ///
    /// ```
    /// use russell_sparse::*;
    /// let mut config = ConfigSolver::new();
    /// config.set_verbose(true);
    /// let correct: &str = "symmetry           = No\n\
    ///                      ordering           = Auto\n\
    ///                      scaling            = Auto\n\
    ///                      pct_inc_workspace  = 100\n\
    ///                      max_work_memory    = 0\n\
    ///                      openmp_num_threads = 1\n\
    ///                      verbose            = true\n";
    /// assert_eq!(format!("{}", config), correct);
    /// ```
    pub fn set_verbose(&mut self, verbose: bool) {
        self.verbose = if verbose { 1 } else { 0 };
    }
}

impl fmt::Display for ConfigSolver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "symmetry           = {}\n\
             ordering           = {}\n\
             scaling            = {}\n\
             pct_inc_workspace  = {}\n\
             max_work_memory    = {}\n\
             openmp_num_threads = {}\n\
             verbose            = {}\n",
            str_enum_symmetry(self.symmetry),
            str_enum_ordering(self.ordering),
            str_enum_scaling(self.scaling),
            self.pct_inc_workspace,
            self.max_work_memory,
            self.openmp_num_threads,
            if self.verbose == 1 { true } else { false },
        )?;
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_works() {
        let config = ConfigSolver::new();
        assert_eq!(config.symmetry, EnumSymmetry::No as i32);
        assert_eq!(config.ordering, EnumOrdering::Auto as i32);
        assert_eq!(config.scaling, EnumScaling::Auto as i32);
        assert_eq!(config.pct_inc_workspace, 100);
        assert_eq!(config.max_work_memory, 0);
        assert_eq!(config.openmp_num_threads, 1);
    }
}
