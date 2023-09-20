use super::{str_enum_ordering, str_enum_scaling, LinSolKind, Ordering, Scaling};
use russell_openblas::to_i32;

/// Holds configuration options for the sparse Solver
#[derive(Copy, Clone, Debug)]
pub struct ConfigSolver {
    pub(crate) lin_sol_kind: LinSolKind, // linear solver kind
    pub(crate) ordering: i32,            // symmetric permutation (ordering)
    pub(crate) scaling: i32,             // scaling strategy
    pub(crate) pct_inc_workspace: i32,   // % increase in the estimated working space (MMP-only)
    pub(crate) max_work_memory: i32,     // max size of the working memory in mega bytes (MMP-only)
    pub(crate) openmp_num_threads: i32,  // number of OpenMP threads (MMP-only)
    pub(crate) verbose: i32,             // show lower-level messages
}

impl ConfigSolver {
    /// Returns a default configuration
    pub fn new() -> Self {
        ConfigSolver {
            lin_sol_kind: LinSolKind::Umf,
            ordering: Ordering::Auto as i32,
            scaling: Scaling::Auto as i32,
            pct_inc_workspace: 100, // (MMP-only)
            max_work_memory: 0,     // (MMP-only) 0 => Auto
            openmp_num_threads: 1,  // (MMP-only)
            verbose: 0,
        }
    }

    /// Sets the linear solver kind
    pub fn lin_sol_kind(&mut self, kind: LinSolKind) -> &mut Self {
        self.lin_sol_kind = kind;
        self
    }

    /// Sets the method to compute a symmetric permutation (ordering)
    pub fn ordering(&mut self, selection: Ordering) -> &mut Self {
        self.ordering = selection as i32;
        self
    }

    /// Sets the scaling strategy
    pub fn scaling(&mut self, selection: Scaling) -> &mut Self {
        self.scaling = selection as i32;
        self
    }

    /// Sets the percentage increase in the estimated working space (MMP-only)
    pub fn pct_inc_workspace(&mut self, value: usize) -> &mut Self {
        self.pct_inc_workspace = to_i32(value);
        self
    }

    /// Sets the maximum size of the working memory in mega bytes (MMP-only)
    pub fn max_work_memory(&mut self, value: usize) -> &mut Self {
        self.max_work_memory = to_i32(value);
        self
    }

    /// Sets the number of OpenMP threads (MMP-only)
    pub fn openmp_num_threads(&mut self, value: usize) -> &mut Self {
        self.openmp_num_threads = to_i32(value);
        self
    }

    /// Sets option to show lower-level messages
    pub fn verbose(&mut self) -> &mut Self {
        self.verbose = 1;
        self
    }

    /// Returns a string representation of the ordering option
    pub fn str_ordering(&self) -> String {
        str_enum_ordering(self.ordering).to_string()
    }

    /// Returns a string representation of the scaling option
    pub fn str_scaling(&self) -> String {
        str_enum_scaling(self.scaling).to_string()
    }

    /// Returns the name of the solver
    pub fn str_solver(&self) -> String {
        match self.lin_sol_kind {
            LinSolKind::Mmp => {
                if cfg!(local_mmp) {
                    "MUMPS-local".to_string()
                } else {
                    "MUMPS".to_string()
                }
            }
            LinSolKind::Umf => "UMFPACK".to_string(),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{ConfigSolver, LinSolKind, Ordering, Scaling};

    #[test]
    fn clone_copy_and_debug_work() {
        let correct = "ConfigSolver { lin_sol_kind: Umf, ordering: 2, scaling: 0, pct_inc_workspace: 100, max_work_memory: 0, openmp_num_threads: 1, verbose: 0 }";
        let config = ConfigSolver::new();
        let copy = config;
        let clone = config.clone();
        assert_eq!(format!("{:?}", config), correct);
        assert_eq!(format!("{:?}", copy), correct);
        assert_eq!(format!("{:?}", clone), correct);
    }

    #[test]
    fn new_works() {
        let config = ConfigSolver::new();
        assert_eq!(config.ordering, Ordering::Auto as i32);
        assert_eq!(config.scaling, Scaling::Auto as i32);
        assert_eq!(config.pct_inc_workspace, 100);
        assert_eq!(config.max_work_memory, 0);
        assert_eq!(config.openmp_num_threads, 1);
        assert_eq!(config.verbose, 0);
    }

    #[test]
    fn set_solver_works() {
        let mut config = ConfigSolver::new();
        for name in [LinSolKind::Mmp, LinSolKind::Umf] {
            config.lin_sol_kind(name);
            match config.lin_sol_kind {
                LinSolKind::Mmp => assert!(true),
                LinSolKind::Umf => assert!(true),
            }
        }
    }

    #[test]
    fn set_ordering_works() {
        let mut config = ConfigSolver::new();
        config.ordering(Ordering::Metis);
        assert_eq!(config.ordering, Ordering::Metis as i32);
    }

    #[test]
    fn set_scaling_works() {
        let mut config = ConfigSolver::new();
        config.scaling(Scaling::No);
        assert_eq!(config.scaling, Scaling::No as i32);
    }

    #[test]
    fn set_pct_inc_workspace_works() {
        let mut config = ConfigSolver::new();
        config.pct_inc_workspace(25);
        assert_eq!(config.pct_inc_workspace, 25);
    }

    #[test]
    fn set_max_work_memory_works() {
        let mut config = ConfigSolver::new();
        config.max_work_memory(1234);
        assert_eq!(config.max_work_memory, 1234);
    }

    #[test]
    fn set_openmp_num_threads_works() {
        let mut config = ConfigSolver::new();
        config.openmp_num_threads(2);
        assert_eq!(config.openmp_num_threads, 2);
    }

    #[test]
    fn set_verbose_works() {
        let mut config = ConfigSolver::new();
        config.verbose();
        assert_eq!(config.verbose, 1);
    }

    #[test]
    fn string_methods_work() {
        let config = ConfigSolver::new();
        assert_eq!(config.str_ordering(), "Auto");
        assert_eq!(config.str_scaling(), "Auto");
    }
}
