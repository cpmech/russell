use super::{str_enum_ordering, str_enum_scaling, LinSol, Ordering, Scaling};
use russell_openblas::to_i32;
use std::fmt;

/// Holds configuration options for the sparse Solver
pub struct ConfigSolver {
    pub(crate) name: LinSol,            // linear solver name
    pub(crate) ordering: i32,           // symmetric permutation (ordering)
    pub(crate) scaling: i32,            // scaling strategy
    pub(crate) pct_inc_workspace: i32,  // % increase in the estimated working space (MMP-only)
    pub(crate) max_work_memory: i32,    // max size of the working memory in mega bytes (MMP-only)
    pub(crate) openmp_num_threads: i32, // number of OpenMP threads (MMP-only)
}

impl ConfigSolver {
    /// Returns a default configuration
    pub fn new() -> Self {
        ConfigSolver {
            name: LinSol::Umf,
            ordering: Ordering::Auto as i32,
            scaling: Scaling::Auto as i32,
            pct_inc_workspace: 100, // (MMP-only)
            max_work_memory: 0,     // (MMP-only) 0 => Auto
            openmp_num_threads: 1,  // (MMP-only)
        }
    }

    /// Sets the linear solver
    pub fn set_solver(&mut self, name: LinSol) -> &mut Self {
        self.name = name;
        self
    }

    /// Sets the method to compute a symmetric permutation (ordering)
    pub fn set_ordering(&mut self, selection: Ordering) -> &mut Self {
        self.ordering = selection as i32;
        self
    }

    /// Sets the scaling strategy
    pub fn set_scaling(&mut self, selection: Scaling) -> &mut Self {
        self.scaling = selection as i32;
        self
    }

    /// Sets the percentage increase in the estimated working space (MMP-only)
    pub fn set_pct_inc_workspace(&mut self, value: usize) -> &mut Self {
        self.pct_inc_workspace = to_i32(value);
        self
    }

    /// Sets the maximum size of the working memory in mega bytes (MMP-only)
    pub fn set_max_work_memory(&mut self, value: usize) -> &mut Self {
        self.max_work_memory = to_i32(value);
        self
    }

    /// Sets the number of OpenMP threads (MMP-only)
    pub fn set_openmp_num_threads(&mut self, value: usize) -> &mut Self {
        self.openmp_num_threads = to_i32(value);
        self
    }
}

impl fmt::Display for ConfigSolver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self.name {
            LinSol::Mmp => {
                if cfg!(local_mmp) {
                    "MMP-local"
                } else {
                    "MMP"
                }
            }
            LinSol::Umf => "UMF",
        };
        write!(
            f,
            "\x20\x20\x20\x20\"name\": \"{}\",\n\
             \x20\x20\x20\x20\"ordering\": \"{}\",\n\
             \x20\x20\x20\x20\"scaling\": \"{}\",\n\
             \x20\x20\x20\x20\"pctIncWorkspace\": {},\n\
             \x20\x20\x20\x20\"maxWorkMemory\": {},\n\
             \x20\x20\x20\x20\"openmpNumThreads\": {}",
            name,
            str_enum_ordering(self.ordering),
            str_enum_scaling(self.scaling),
            self.pct_inc_workspace,
            self.max_work_memory,
            self.openmp_num_threads,
        )
        .unwrap();
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{ConfigSolver, LinSol, Ordering, Scaling};

    #[test]
    fn new_works() {
        let config = ConfigSolver::new();
        assert_eq!(config.ordering, Ordering::Auto as i32);
        assert_eq!(config.scaling, Scaling::Auto as i32);
        assert_eq!(config.pct_inc_workspace, 100);
        assert_eq!(config.max_work_memory, 0);
        assert_eq!(config.openmp_num_threads, 1);
    }

    #[test]
    fn set_solver_works() {
        let mut config = ConfigSolver::new();
        for name in [LinSol::Mmp, LinSol::Umf] {
            config.set_solver(name);
            match config.name {
                LinSol::Mmp => assert!(true),
                LinSol::Umf => assert!(true),
            }
        }
    }

    #[test]
    fn set_ordering_works() {
        let mut config = ConfigSolver::new();
        config.set_ordering(Ordering::Metis);
        assert_eq!(config.ordering, Ordering::Metis as i32);
    }

    #[test]
    fn set_scaling_works() {
        let mut config = ConfigSolver::new();
        config.set_scaling(Scaling::No);
        assert_eq!(config.scaling, Scaling::No as i32);
    }

    #[test]
    fn set_pct_inc_workspace_works() {
        let mut config = ConfigSolver::new();
        config.set_pct_inc_workspace(25);
        assert_eq!(config.pct_inc_workspace, 25);
    }

    #[test]
    fn set_max_work_memory_works() {
        let mut config = ConfigSolver::new();
        config.set_max_work_memory(1234);
        assert_eq!(config.max_work_memory, 1234);
    }

    #[test]
    fn set_openmp_num_threads_works() {
        let mut config = ConfigSolver::new();
        config.set_openmp_num_threads(2);
        assert_eq!(config.openmp_num_threads, 2);
    }

    #[test]
    fn display_trait_works() {
        let config1 = ConfigSolver::new();
        let correct1: &str = "\x20\x20\x20\x20\"name\": \"UMF\",\n\
                              \x20\x20\x20\x20\"ordering\": \"Auto\",\n\
                              \x20\x20\x20\x20\"scaling\": \"Auto\",\n\
                              \x20\x20\x20\x20\"pctIncWorkspace\": 100,\n\
                              \x20\x20\x20\x20\"maxWorkMemory\": 0,\n\
                              \x20\x20\x20\x20\"openmpNumThreads\": 1";
        assert_eq!(format!("{}", config1), correct1);
        let mut config2 = ConfigSolver::new();
        config2.set_solver(LinSol::Mmp);
        let correct2: &str = if cfg!(local_mmp) {
            "\x20\x20\x20\x20\"name\": \"MMP-local\",\n\
             \x20\x20\x20\x20\"ordering\": \"Auto\",\n\
             \x20\x20\x20\x20\"scaling\": \"Auto\",\n\
             \x20\x20\x20\x20\"pctIncWorkspace\": 100,\n\
             \x20\x20\x20\x20\"maxWorkMemory\": 0,\n\
             \x20\x20\x20\x20\"openmpNumThreads\": 1"
        } else {
            "\x20\x20\x20\x20\"name\": \"MMP\",\n\
             \x20\x20\x20\x20\"ordering\": \"Auto\",\n\
             \x20\x20\x20\x20\"scaling\": \"Auto\",\n\
             \x20\x20\x20\x20\"pctIncWorkspace\": 100,\n\
             \x20\x20\x20\x20\"maxWorkMemory\": 0,\n\
             \x20\x20\x20\x20\"openmpNumThreads\": 1"
        };
        assert_eq!(format!("{}", config2), correct2);
    }
}
