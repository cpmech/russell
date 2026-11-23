//! External sparse-solver library versions and sparse-benchmark CPU plan.

use russell_lab::{CpuPlan, format_cpu_list, get_cpu_plan_with_topology_overrides};

// Versions of external sparse solver libraries installed via the project's bash scripts
const MUMPS_SCRIPT_VERSION: &str = "5.9.2";
const SUITESPARSE_VERSION: &str = "latest (from GitHub)";

/// Returns version information for the external sparse solver libraries.
///
/// Reports where each backend came from, plus the version that the project's
/// installation scripts download:
///
/// - **UMFPACK** — present when the `umfpack` feature is enabled (opt-in);
///   the line says whether it is the package-manager library (`system`) or the
///   local install under `RUSSELL_LOCAL_PREFIX` that took precedence, with the
///   directory it was linked from; `[not compiled in]` otherwise
/// - **MUMPS** — version downloaded by `zscripts/*-compile-mumps.bash`; annotated
///   `[not compiled in]` when the `mumps` feature is disabled
/// - **SuiteSparse** — cloned from head of the upstream GitHub repository
pub fn get_library_versions() -> String {
    let mut info = String::new();

    // UMFPACK (optional; the build script records which one, if any)
    let umfpack = option_env!("RUSSELL_SPARSE_UMFPACK").unwrap_or("[not compiled in]");
    info.push_str(&format!("UMFPACK: {umfpack}\n"));

    // MUMPS
    if cfg!(feature = "mumps") {
        info.push_str(&format!("MUMPS: {}\n", MUMPS_SCRIPT_VERSION));
    } else {
        info.push_str(&format!("MUMPS: {} [not compiled in]\n", MUMPS_SCRIPT_VERSION));
    }

    // SuiteSparse (used when the umfpack feature is enabled)
    info.push_str(&format!("SuiteSparse: {}", SUITESPARSE_VERSION));

    info
}

/// CPU topology and thread plan tailored for the sparse-solver benchmarks
///
/// The topology comes from [`russell_lab::get_cpu_plan_with_topology_overrides`];
/// `mumps_threads` defaults to [`CpuPlan::physical_threads`] and `blas_threads`
/// defaults to [`CpuPlan::p_threads`], with optional `MUMPS_THREADS` and
/// `BLAS_THREADS` environment overrides.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SparseCpuPlan {
    /// Logical P-core CPUs, in ascending order
    pub p_cores: Vec<usize>,
    /// Logical E-core CPUs, in ascending order
    pub e_cores: Vec<usize>,
    /// Number of physical P-cores
    pub p_physical: usize,
    /// Number of physical E-cores
    pub e_physical: usize,
    /// Threads for MUMPS: one per physical core, P and E together
    pub mumps_threads: usize,
    /// Threads for the BLAS work of Faer and UMFPACK: one per physical P-core
    pub blas_threads: usize,
    /// Logical P-core CPUs the P-core runs should be pinned to; empty means no pinning
    pub affinity: Vec<usize>,
}

impl SparseCpuPlan {
    /// Formats the plan as shell `KEY=VALUE` assignments
    ///
    /// The keys match `run_solve_matrix_market.bash`: `P_CORES`, `E_CORES`,
    /// `P_LOGICAL`, `E_LOGICAL`, `P_PHYSICAL`, `E_PHYSICAL`, `MUMPS_THREADS`,
    /// `BLAS_THREADS`, and `CPU_AFFINITY`. An absent list is printed as an
    /// empty value, and `CPU_AFFINITY=none` means no pinning.
    pub fn to_shell(&self) -> String {
        let affinity = if self.affinity.is_empty() {
            "none".to_string()
        } else {
            format_cpu_list(&self.affinity)
        };
        format!(
            "P_CORES={}\n\
             E_CORES={}\n\
             P_LOGICAL={}\n\
             E_LOGICAL={}\n\
             P_PHYSICAL={}\n\
             E_PHYSICAL={}\n\
             MUMPS_THREADS={}\n\
             BLAS_THREADS={}\n\
             CPU_AFFINITY={}\n",
            format_cpu_list(&self.p_cores),
            format_cpu_list(&self.e_cores),
            self.p_cores.len(),
            self.e_cores.len(),
            self.p_physical,
            self.e_physical,
            self.mumps_threads,
            self.blas_threads,
            affinity,
        )
    }
}

/// Returns the sparse-benchmark CPU plan with `P_CORES`, `E_CORES`,
/// `MUMPS_THREADS`, and `BLAS_THREADS` environment overrides
///
/// Starts from [`russell_lab::get_cpu_plan_with_topology_overrides`], initializes
/// `mumps_threads = plan.physical_threads` and `blas_threads = plan.p_threads`,
/// and applies positive `MUMPS_THREADS` and `BLAS_THREADS` overrides. Empty,
/// zero, or invalid values keep the topology-derived count.
pub fn get_sparse_cpu_plan() -> SparseCpuPlan {
    let plan = get_cpu_plan_with_topology_overrides();
    sparse_cpu_plan_with_overrides(&plan, |name| std::env::var(name).ok())
}

/// Builds a [`SparseCpuPlan`] from a [`CpuPlan`] and an environment-variable lookup
fn sparse_cpu_plan_with_overrides(plan: &CpuPlan, get_env: impl Fn(&str) -> Option<String>) -> SparseCpuPlan {
    let mut sparse = SparseCpuPlan {
        p_cores: plan.p_cores.clone(),
        e_cores: plan.e_cores.clone(),
        p_physical: plan.p_physical,
        e_physical: plan.e_physical,
        mumps_threads: plan.physical_threads,
        blas_threads: plan.p_threads,
        affinity: plan.affinity.clone(),
    };
    if let Some(n) = parse_env_threads(get_env("MUMPS_THREADS").as_deref()) {
        sparse.mumps_threads = n;
    }
    if let Some(n) = parse_env_threads(get_env("BLAS_THREADS").as_deref()) {
        sparse.blas_threads = n;
    }
    sparse
}

/// Parses a positive thread count from an environment variable value
fn parse_env_threads(value: Option<&str>) -> Option<usize> {
    let n: usize = value?.trim().parse().ok()?;
    (n > 0).then_some(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_library_versions_is_non_empty() {
        let info = get_library_versions();
        assert!(!info.is_empty());
    }

    #[test]
    fn get_library_versions_contains_all_libraries() {
        let info = get_library_versions();
        assert!(info.contains("MUMPS:"), "Should contain MUMPS version");
        assert!(info.contains("SuiteSparse:"), "Should contain SuiteSparse version");
    }

    #[test]
    fn get_library_versions_contains_expected_version_numbers() {
        let info = get_library_versions();
        println!("{}", info);
        assert!(info.contains(MUMPS_SCRIPT_VERSION), "Should contain MUMPS version");
        assert!(info.contains(SUITESPARSE_VERSION), "Should contain SuiteSparse version");
    }

    /// Guards the source selection: when the prefix contains UMFPACK the line
    /// must say `local`; on Windows a vcpkg install is the next source and says
    /// `vcpkg`; otherwise the platform library must have been used.
    #[cfg(feature = "umfpack")]
    #[test]
    fn umfpack_source_follows_the_prefix() {
        let source = option_env!("RUSSELL_SPARSE_UMFPACK").unwrap_or("[not compiled in]");
        let prefix = std::env::var("RUSSELL_LOCAL_PREFIX").unwrap_or_default();
        let trimmed = prefix.trim().trim_end_matches(['/', '\\']);
        let local = !trimmed.is_empty() && std::path::Path::new(trimmed).join("include/umfpack/umfpack.h").is_file();
        if local {
            assert!(source.starts_with("local"), "expected a local UMFPACK, got: {source}");
        } else if source.starts_with("vcpkg") {
            // Windows, where vcpkg sits between the local prefix and MSYS2.
            if !cfg!(target_os = "windows") {
                panic!("a vcpkg UMFPACK outside Windows: {source}");
            }
        } else {
            assert!(source.starts_with("system"), "expected a system UMFPACK, got: {source}");
        }
    }

    /// Guards the build-script wiring: the UMFPACK line must name the source the
    /// features selected, because two SuiteSparse installations can be present and
    /// the wrong one is otherwise invisible.
    #[test]
    fn get_library_versions_reports_the_umfpack_source() {
        let info = get_library_versions();
        let line = info.lines().find(|line| line.starts_with("UMFPACK: ")).expect("the UMFPACK line");
        let source = option_env!("RUSSELL_SPARSE_UMFPACK").unwrap_or("[not compiled in]");
        assert!(line.contains(source), "expected {source:?} in: {line}");
    }

    #[test]
    fn sparse_cpu_plan_defaults_and_overrides_work() {
        let base = CpuPlan {
            p_cores: vec![0, 1, 2, 3],
            e_cores: vec![4, 5],
            p_physical: 2,
            e_physical: 2,
            physical_threads: 4,
            p_threads: 2,
            affinity: vec![0, 1, 2, 3],
        };

        let defaulted = sparse_cpu_plan_with_overrides(&base, |name| match name {
            "PHYSICAL_THREADS" => Some("99".to_string()),
            "P_THREADS" => Some("77".to_string()),
            _ => None,
        });
        assert_eq!(defaulted.p_cores, vec![0, 1, 2, 3]);
        assert_eq!(defaulted.e_cores, vec![4, 5]);
        assert_eq!(defaulted.p_physical, 2);
        assert_eq!(defaulted.e_physical, 2);
        assert_eq!(defaulted.mumps_threads, 4);
        assert_eq!(defaulted.blas_threads, 2);
        assert_eq!(defaulted.affinity, vec![0, 1, 2, 3]);

        let overridden = sparse_cpu_plan_with_overrides(&base, |name| match name {
            "MUMPS_THREADS" => Some("12".to_string()),
            "BLAS_THREADS" => Some("6".to_string()),
            _ => None,
        });
        assert_eq!(overridden.mumps_threads, 12);
        assert_eq!(overridden.blas_threads, 6);

        let ignored = sparse_cpu_plan_with_overrides(&base, |name| match name {
            "MUMPS_THREADS" => Some("0".to_string()),
            "BLAS_THREADS" => Some("nope".to_string()),
            _ => None,
        });
        assert_eq!(ignored.mumps_threads, 4);
        assert_eq!(ignored.blas_threads, 2);

        let empty = sparse_cpu_plan_with_overrides(&base, |name| match name {
            "MUMPS_THREADS" => Some("  ".to_string()),
            "BLAS_THREADS" => Some(String::new()),
            _ => None,
        });
        assert_eq!(empty.mumps_threads, 4);
        assert_eq!(empty.blas_threads, 2);
    }

    #[test]
    fn sparse_cpu_plan_shell_has_the_benchmark_keys() {
        let hybrid = SparseCpuPlan {
            p_cores: vec![0, 1],
            e_cores: vec![2, 3],
            p_physical: 2,
            e_physical: 2,
            mumps_threads: 4,
            blas_threads: 2,
            affinity: vec![0, 1],
        };
        let text = hybrid.to_shell();
        assert!(text.contains("P_CORES=0-1\n"));
        assert!(text.contains("E_CORES=2-3\n"));
        assert!(text.contains("P_LOGICAL=2\n"));
        assert!(text.contains("E_LOGICAL=2\n"));
        assert!(text.contains("P_PHYSICAL=2\n"));
        assert!(text.contains("E_PHYSICAL=2\n"));
        assert!(text.contains("MUMPS_THREADS=4\n"));
        assert!(text.contains("BLAS_THREADS=2\n"));
        assert!(text.contains("CPU_AFFINITY=0-1\n"));

        let non_hybrid = SparseCpuPlan {
            p_cores: Vec::new(),
            e_cores: Vec::new(),
            p_physical: 0,
            e_physical: 0,
            mumps_threads: 8,
            blas_threads: 8,
            affinity: Vec::new(),
        };
        let text = non_hybrid.to_shell();
        assert!(text.contains("P_CORES=\n"));
        assert!(text.contains("E_CORES=\n"));
        assert!(text.contains("MUMPS_THREADS=8\n"));
        assert!(text.contains("BLAS_THREADS=8\n"));
        assert!(text.contains("CPU_AFFINITY=none\n"));
    }

    #[test]
    fn get_sparse_cpu_plan_returns_positive_thread_counts() {
        let plan = get_sparse_cpu_plan();
        assert!(plan.mumps_threads >= 1);
        assert!(plan.blas_threads >= 1);
    }
}
