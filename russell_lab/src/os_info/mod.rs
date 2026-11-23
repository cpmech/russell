//! Operating-system, hardware, and CPU-topology information.
//!
//! The report is built from procfs and from standard tools (`uname`, `nvidia-smi`,
//! `lspci`, and `lscpu`); each section is omitted when its source is absent. Linux
//! is the only platform with a full report at the moment.
//!
//! The CPU plan detects hybrid P/E cores from the Linux sysfs interface and
//! computes physical-core and P-core thread counts along with an affinity list.

use std::collections::BTreeSet;
use std::path::Path;

#[cfg(target_os = "linux")]
use std::process::Command;

/// Upper bound on accepted logical CPU indices to prevent runaway allocations
const MAX_CPU_ID: usize = 65_535;

/// CPU topology and the thread counts derived from it
///
/// The detection follows the Linux hybrid-CPU interface:
///
/// * `/sys/devices/cpu_core/cpus` lists the logical performance cores (P-cores);
/// * `/sys/devices/cpu_atom/cpus` lists the logical efficiency cores (E-cores).
///
/// A machine without those files is treated as non-hybrid: every available logical
/// CPU is used, and no affinity list is produced.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CpuPlan {
    /// Logical P-core CPUs, in ascending order
    pub p_cores: Vec<usize>,
    /// Logical E-core CPUs, in ascending order
    pub e_cores: Vec<usize>,
    /// Number of physical P-cores
    pub p_physical: usize,
    /// Number of physical E-cores
    pub e_physical: usize,
    /// Threads for workloads using all physical cores, P and E together
    pub physical_threads: usize,
    /// Threads for workloads using physical P-cores only
    pub p_threads: usize,
}

impl CpuPlan {
    /// Returns the logical P-core CPUs that P-core runs should be pinned to; empty means no pinning
    pub fn affinity(&self) -> &[usize] {
        &self.p_cores
    }

    /// Returns whether the plan has a hybrid P/E topology
    pub fn is_hybrid(&self) -> bool {
        !self.p_cores.is_empty() || !self.e_cores.is_empty()
    }

    /// Formats the plan as shell `KEY=VALUE` assignments
    ///
    /// The emitted keys are `P_CORES`, `E_CORES`, `P_LOGICAL`, `E_LOGICAL`,
    /// `P_PHYSICAL`, `E_PHYSICAL`, `PHYSICAL_THREADS`, `P_THREADS`, and
    /// `CPU_AFFINITY`. An absent list is printed as an empty value, and
    /// `CPU_AFFINITY=none` means no pinning.
    pub fn to_shell(&self) -> String {
        let affinity = if self.affinity().is_empty() {
            "none".to_string()
        } else {
            format_cpu_list(self.affinity())
        };
        format!(
            "P_CORES={}\n\
             E_CORES={}\n\
             P_LOGICAL={}\n\
             E_LOGICAL={}\n\
             P_PHYSICAL={}\n\
             E_PHYSICAL={}\n\
             PHYSICAL_THREADS={}\n\
             P_THREADS={}\n\
             CPU_AFFINITY={}\n",
            format_cpu_list(&self.p_cores),
            format_cpu_list(&self.e_cores),
            self.p_cores.len(),
            self.e_cores.len(),
            self.p_physical,
            self.e_physical,
            self.physical_threads,
            self.p_threads,
            affinity,
        )
    }

    /// Formats the plan for the human-readable OS report
    pub fn summary(&self) -> String {
        let p = format_cpu_list(&self.p_cores);
        let e = format_cpu_list(&self.e_cores);
        let affinity = if self.affinity().is_empty() {
            "none".to_string()
        } else {
            format_cpu_list(self.affinity())
        };
        format!(
            "hybrid: {}\n\
             P cores: {} ({} logical, {} physical)\n\
             E cores: {} ({} logical, {} physical)\n\
             physical threads: {}\n\
             P threads: {}\n\
             CPU affinity: {}",
            if self.is_hybrid() { "yes" } else { "no" },
            if p.is_empty() { "none" } else { p.as_str() },
            self.p_cores.len(),
            self.p_physical,
            if e.is_empty() { "none" } else { e.as_str() },
            self.e_cores.len(),
            self.e_physical,
            self.physical_threads,
            self.p_threads,
            affinity,
        )
    }

    /// Builds the non-hybrid fallback: every available logical CPU, no pinning
    fn fallback() -> Self {
        let threads = available_threads();
        Self {
            p_cores: Vec::new(),
            e_cores: Vec::new(),
            p_physical: 0,
            e_physical: 0,
            physical_threads: threads,
            p_threads: threads,
        }
    }

    /// Builds a plan from the logical P/E lists
    ///
    /// `cpu_root` is the `/sys/devices/system/cpu` directory on Linux; `None`
    /// treats every logical CPU as a physical core, which is what the tests and
    /// the non-Linux fallback need.
    fn from_lists(mut p_cores: Vec<usize>, mut e_cores: Vec<usize>, cpu_root: Option<&Path>) -> Self {
        p_cores.sort_unstable();
        p_cores.dedup();
        e_cores.sort_unstable();
        e_cores.dedup();
        if p_cores.is_empty() && e_cores.is_empty() {
            return Self::fallback();
        }
        let p_physical = cpu_root.map_or(p_cores.len(), |root| physical_count(root, &p_cores));
        let e_physical = cpu_root.map_or(e_cores.len(), |root| physical_count(root, &e_cores));
        let physical_threads = match p_physical + e_physical {
            0 => available_threads(),
            n => n,
        };
        let p_threads = match p_physical {
            0 => available_threads(),
            n => n,
        };
        Self {
            p_cores,
            e_cores,
            p_physical,
            e_physical,
            physical_threads,
            p_threads,
        }
    }
}

/// Returns the detected P/E core lists and the Linux sysfs CPU root when available
fn detected_topology() -> (Vec<usize>, Vec<usize>, Option<&'static Path>) {
    #[cfg(target_os = "linux")]
    {
        let allowed = allowed_cpu_list();
        let p = intersect(read_cpu_list(Path::new("/sys/devices/cpu_core/cpus")), &allowed);
        let e = intersect(read_cpu_list(Path::new("/sys/devices/cpu_atom/cpus")), &allowed);
        (p, e, Some(Path::new("/sys/devices/system/cpu")))
    }
    #[cfg(not(target_os = "linux"))]
    {
        (Vec::new(), Vec::new(), None)
    }
}

/// Returns the CPU topology and thread plan of the current machine
///
/// On Linux the P/E lists come from sysfs and are intersected with the CPUs the
/// calling thread is allowed to run on (`Cpus_allowed_list` in
/// `/proc/thread-self/status`, falling back to `/proc/self/status`). Everywhere
/// else, and on a non-hybrid Linux machine, the plan falls back to all available
/// logical CPUs with no pinning.
pub fn get_cpu_plan() -> CpuPlan {
    let (p_cores, e_cores, cpu_root) = detected_topology();
    CpuPlan::from_lists(p_cores, e_cores, cpu_root)
}

/// Returns the CPU plan with `P_CORES` and `E_CORES` environment overrides
///
/// Only topology overrides (`P_CORES` and `E_CORES`) are read from the
/// environment; thread-count variables are ignored. An empty or invalid
/// `P_CORES` or `E_CORES` keeps the detected value. Explicit `P_CORES` and
/// `E_CORES` overrides are used as given and are not intersected with
/// `Cpus_allowed_list`.
pub fn get_cpu_plan_with_overrides() -> CpuPlan {
    let (detected_p, detected_e, cpu_root) = detected_topology();
    cpu_plan_with_overrides(detected_p, detected_e, cpu_root, |name| std::env::var(name).ok())
}

/// Builds a CPU plan from detected lists and `P_CORES`/`E_CORES` overrides
fn cpu_plan_with_overrides(
    detected_p: Vec<usize>,
    detected_e: Vec<usize>,
    cpu_root: Option<&Path>,
    get_env: impl Fn(&str) -> Option<String>,
) -> CpuPlan {
    let p_cores = parse_env_cpu_list(get_env("P_CORES").as_deref()).unwrap_or(detected_p);
    let e_cores = parse_env_cpu_list(get_env("E_CORES").as_deref()).unwrap_or(detected_e);
    CpuPlan::from_lists(p_cores, e_cores, cpu_root)
}

/// Gathers operating-system and hardware information from the current Linux machine
///
/// Collects the following sections, each prefixed with a `--- NAME ---` header:
///
/// - **OS** — Linux distribution name/version from `/etc/os-release` and kernel
///   release via `uname -r`
/// - **GPU** — NVIDIA GPU details from `nvidia-smi` (name, driver, memory),
///   falling back to `lspci` for non-NVIDIA or when `nvidia-smi` is unavailable
/// - **CPU** — CPU model, architecture, cores/threads, and cache sizes via
///   `lscpu`, with a fallback to `/proc/cpuinfo`
/// - **CPU topology** — hybrid P/E cores and the benchmark thread plan
/// - **Memory** — total, free, available, and swap from `/proc/meminfo`
///
/// The OS, GPU, CPU, and memory sections are omitted if the underlying tool or
/// procfs file is absent; the CPU topology section always reports the plan, using
/// the non-hybrid fallback when sysfs has no P/E lists.
#[cfg(target_os = "linux")]
fn get_os_info_linux() -> String {
    let mut info = String::new();

    // OS info
    let mut os_lines: Vec<String> = Vec::new();
    if let Ok(s) = std::fs::read_to_string("/etc/os-release") {
        for line in s.lines() {
            let line = line.trim();
            if line.starts_with("NAME") || line.starts_with("VERSION_ID") {
                os_lines.push(line.to_string());
            }
        }
    }
    if let Ok(output) = Command::new("uname").arg("-r").output() {
        if let Ok(s) = String::from_utf8(output.stdout) {
            os_lines.push(format!("KERNEL={}", s.trim()));
        }
    }
    if !os_lines.is_empty() {
        info.push_str(&format!("--- OS ---\n{}", os_lines.join("\n")));
    }

    // GPU info
    let mut gpu_info = String::new();
    if let Ok(output) = Command::new("nvidia-smi")
        .args(["--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
        .output()
    {
        if let Ok(s) = String::from_utf8(output.stdout) {
            let s = s.trim();
            if !s.is_empty() {
                for (i, line) in s.lines().enumerate() {
                    gpu_info.push_str(&format!("GPU[{}]: {}\n", i, line.trim()));
                }
            }
        }
    }
    if gpu_info.is_empty() {
        if let Ok(output) = Command::new("sh")
            .arg("-c")
            .arg("lspci 2>/dev/null | grep -iE 'vga|3d|display' | head -4")
            .output()
        {
            if let Ok(s) = String::from_utf8(output.stdout) {
                let s = s.trim();
                if !s.is_empty() {
                    gpu_info = s.to_string();
                }
            }
        }
    }
    if !gpu_info.is_empty() {
        info.push_str(&format!("\n\n--- GPU ---\n{}", gpu_info.trim_end()));
    }

    // CPU info
    if let Ok(s) = Command::new("sh")
        .arg("-c")
        .arg("lscpu 2>/dev/null | grep -E 'Model name|Architecture|Socket|Core|Thread|CPU\\(s\\)|MHz|BogoMIPS|L1|L2|L3' | head -20")
        .output()
    {
        if let Ok(s) = String::from_utf8(s.stdout) {
            let s = s.trim();
            if !s.is_empty() {
                let pairs: Vec<(&str, &str)> = s
                    .lines()
                    .filter_map(|line| line.split_once(':').map(|(k, v)| (k.trim(), v.trim())))
                    .collect();
                let max_key_len = pairs.iter().map(|(k, _)| k.len()).max().unwrap_or(0);
                let cleaned: Vec<String> = pairs
                    .iter()
                    .map(|(k, v)| format!("{}{}: {}", k, " ".repeat(max_key_len - k.len()), v))
                    .collect();
                info.push_str(&format!("\n\n--- CPU ---\n{}", cleaned.join("\n")));
            }
        }
    }
    // Fallback to /proc/cpuinfo
    if !info.contains("Model name") {
        if let Ok(s) = std::fs::read_to_string("/proc/cpuinfo") {
            let mut cpu_lines: Vec<&str> = Vec::new();
            for line in s.lines() {
                let line = line.trim();
                if line.starts_with("model name") || line.starts_with("processor") || line.starts_with("cpu cores") {
                    cpu_lines.push(line);
                }
            }
            if !cpu_lines.is_empty() {
                info.push_str(&format!("\n\n--- CPU ---\n{}", cpu_lines.join("\n")));
            }
        }
    }

    // CPU topology and the benchmark thread plan
    info.push_str(&format!("\n\n--- CPU topology ---\n{}", get_cpu_plan().summary()));

    // Memory
    if let Ok(s) = std::fs::read_to_string("/proc/meminfo") {
        let mut mem_lines: Vec<&str> = Vec::new();
        for line in s.lines() {
            let line = line.trim();
            if line.starts_with("MemTotal") || line.starts_with("MemFree") || line.starts_with("MemAvailable") || line.starts_with("SwapTotal") {
                mem_lines.push(line);
            }
        }
        if !mem_lines.is_empty() {
            info.push_str(&format!("\n\n--- Memory ---\n{}", mem_lines.join("\n")));
        }
    }

    info
}

/// Gathers operating-system and hardware information for the current platform
///
/// On Linux, this collects the OS, GPU, CPU topology, and memory information.
/// On other platforms, it returns `"NOT AVAILABLE"` because the information
/// gathering is only implemented for Linux at the moment.
#[cfg(target_os = "linux")]
pub fn get_os_info() -> String {
    get_os_info_linux()
}

/// Gathers operating-system and hardware information for the current platform
///
/// Returns `"NOT AVAILABLE"` — not yet implemented for this platform.
#[cfg(not(target_os = "linux"))]
pub fn get_os_info() -> String {
    "NOT AVAILABLE".to_string()
}

/// Pins the calling thread to the given logical CPUs
///
/// On Linux this calls `sched_setaffinity` with a dynamically sized bitmask, so
/// CPU indices above `CPU_SETSIZE` are allowed. The affinity is inherited by
/// threads created afterward, so call this before starting OpenMP, MKL, OpenBLAS,
/// or any worker pool.
///
/// An empty list is rejected. On non-Linux platforms the function returns an
/// informative error because there is no portable affinity interface.
pub fn set_cpu_affinity(cpus: &[usize]) -> Result<(), String> {
    if cpus.is_empty() {
        return Err("cannot set CPU affinity to an empty CPU list".to_string());
    }
    if let Some(&cpu) = cpus.iter().find(|&&cpu| cpu > MAX_CPU_ID) {
        return Err(format!("CPU index {cpu} exceeds maximum supported index {MAX_CPU_ID}"));
    }
    #[cfg(target_os = "linux")]
    {
        let bits_per_word = usize::BITS as usize;
        let max_cpu = *cpus.iter().max().unwrap();
        let words = max_cpu / bits_per_word + 1;
        let mut mask = vec![0usize; words];
        for &cpu in cpus {
            mask[cpu / bits_per_word] |= 1usize << (cpu % bits_per_word);
        }
        unsafe extern "C" {
            fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const usize) -> i32;
        }
        let size = words * std::mem::size_of::<usize>();
        let status = unsafe { sched_setaffinity(0, size, mask.as_ptr()) };
        if status != 0 {
            return Err(format!(
                "sched_setaffinity failed for CPUs {}: {}",
                format_cpu_list(cpus),
                std::io::Error::last_os_error()
            ));
        }
        Ok(())
    }
    #[cfg(not(target_os = "linux"))]
    {
        Err("CPU affinity is only supported on Linux".to_string())
    }
}

/// Returns the number of logical CPUs available to the process
fn available_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

/// Counts the physical cores in one logical-CPU list
///
/// Each CPU's `thread_siblings_list` identifies its SMT group; the first member of
/// every group is inserted into a set, so a P-core with two threads counts once.
/// If any sibling list is unreadable or malformed, the logical count is used
/// instead, which is the same fallback the shell script had.
fn physical_count(cpu_root: &Path, cpus: &[usize]) -> usize {
    if cpus.is_empty() {
        return 0;
    }
    let mut cores = BTreeSet::new();
    for &cpu in cpus {
        let path = cpu_root.join(format!("cpu{cpu}/topology/thread_siblings_list"));
        let Ok(text) = std::fs::read_to_string(path) else {
            return cpus.len();
        };
        let Some(siblings) = parse_cpu_list(&text) else {
            return cpus.len();
        };
        let Some(&first) = siblings.first() else {
            return cpus.len();
        };
        cores.insert(first);
    }
    cores.len()
}

/// Parses a Linux CPU list such as `0-15` or `0-7,16-23`
///
/// Returns `None` when the input is malformed. An empty or whitespace-only string
/// returns an empty list.
pub fn parse_cpu_list(text: &str) -> Option<Vec<usize>> {
    let mut cpus = Vec::new();
    for part in text.trim().split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        match part.split_once('-') {
            Some((first, last)) => {
                let first: usize = first.trim().parse().ok()?;
                let last: usize = last.trim().parse().ok()?;
                if first > last || last > MAX_CPU_ID {
                    return None;
                }
                cpus.extend(first..=last);
            }
            None => {
                let cpu: usize = part.parse().ok()?;
                if cpu > MAX_CPU_ID {
                    return None;
                }
                cpus.push(cpu);
            }
        }
    }
    cpus.sort_unstable();
    cpus.dedup();
    Some(cpus)
}

/// Formats a CPU list, compressing consecutive runs into ranges
///
/// For example, `[0, 1, 2, 3, 7, 9, 10]` becomes `0-3,7,9-10`. An empty list
/// becomes an empty string.
pub fn format_cpu_list(cpus: &[usize]) -> String {
    if cpus.is_empty() {
        return String::new();
    }
    let mut cpus = cpus.to_vec();
    cpus.sort_unstable();
    cpus.dedup();
    let mut parts = Vec::new();
    let mut start = cpus[0];
    let mut previous = start;
    for &cpu in &cpus[1..] {
        if cpu == previous + 1 {
            previous = cpu;
            continue;
        }
        parts.push(format_range(start, previous));
        start = cpu;
        previous = cpu;
    }
    parts.push(format_range(start, previous));
    parts.join(",")
}

/// Formats one run of a CPU list
fn format_range(first: usize, last: usize) -> String {
    if first == last { first.to_string() } else { format!("{first}-{last}") }
}

/// Reads a Linux CPU list from sysfs, returning an empty list when it is absent
#[cfg(target_os = "linux")]
fn read_cpu_list(path: &Path) -> Vec<usize> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|text| parse_cpu_list(&text))
        .unwrap_or_default()
}

/// Reads the CPUs the calling thread is allowed to run on
#[cfg(target_os = "linux")]
fn allowed_cpu_list() -> Option<Vec<usize>> {
    // `thread-self` is the calling thread; `self` is the thread-group leader and is
    // the fallback on kernels older than 3.17.
    for path in ["/proc/thread-self/status", "/proc/self/status"] {
        let Ok(status) = std::fs::read_to_string(path) else {
            continue;
        };
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("Cpus_allowed_list:") {
                return parse_cpu_list(rest);
            }
        }
    }
    None
}

/// Keeps only the CPUs present in the allowed set
#[cfg(target_os = "linux")]
fn intersect(mut cpus: Vec<usize>, allowed: &Option<Vec<usize>>) -> Vec<usize> {
    if let Some(allowed) = allowed {
        cpus.retain(|cpu| allowed.contains(cpu));
    }
    cpus
}

/// Parses a CPU list from an environment variable value, ignoring empty/invalid values
fn parse_env_cpu_list(value: Option<&str>) -> Option<Vec<usize>> {
    let value = value?;
    if value.trim().is_empty() {
        return None;
    }
    parse_cpu_list(value).filter(|cpus| !cpus.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_os = "linux")]
    fn get_os_info_linux_is_non_empty() {
        let info = get_os_info_linux();
        assert!(!info.is_empty(), "Linux OS info should not be empty");
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn get_os_info_linux_contains_expected_sections() {
        let info = get_os_info_linux();
        println!("{}", info);
        assert!(info.contains("--- OS ---"), "Should contain OS section");
        assert!(info.contains("NAME="), "Should contain NAME");
        assert!(info.contains("KERNEL="), "Should contain KERNEL");
        // assert!(info.contains("--- GPU ---"), "Should contain GPU section"); // doesn't work on GitHub Actions
        assert!(info.contains("--- CPU ---"), "Should contain CPU section");
        assert!(info.contains("--- CPU topology ---"), "Should contain CPU topology section");
        assert!(info.contains("--- Memory ---"), "Should contain Memory section");
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn get_os_info_linux_memory_contains_keys() {
        let info = get_os_info_linux();
        assert!(info.contains("MemTotal:"), "Should report total memory");
        assert!(info.contains("MemAvailable:"), "Should report available memory");
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn get_os_info_linux_os_contains_name() {
        let info = get_os_info_linux();
        assert!(info.contains("NAME"), "Should contain OS name");
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn get_os_info_is_non_empty_on_linux() {
        let info = get_os_info();
        assert!(!info.is_empty(), "Linux OS info should not be empty");
    }

    #[test]
    #[cfg(not(target_os = "linux"))]
    fn get_os_info_returns_not_available_on_non_linux() {
        assert_eq!(get_os_info(), "NOT AVAILABLE");
    }

    #[test]
    fn parse_cpu_list_reads_ranges_and_singles() {
        assert_eq!(parse_cpu_list("0-3,7,9-10"), Some(vec![0, 1, 2, 3, 7, 9, 10]));
        assert_eq!(parse_cpu_list(" 4 "), Some(vec![4]));
        assert_eq!(parse_cpu_list(""), Some(vec![]));
    }

    #[test]
    fn parse_cpu_list_rejects_malformed_input() {
        assert_eq!(parse_cpu_list("3-1"), None);
        assert_eq!(parse_cpu_list("x"), None);
        assert_eq!(parse_cpu_list("1-"), None);
        assert_eq!(parse_cpu_list("65536"), None);
        assert_eq!(parse_cpu_list("0-65536"), None);
    }

    #[test]
    fn format_cpu_list_compresses_runs() {
        assert_eq!(format_cpu_list(&[0, 1, 2, 3, 7, 9, 10]), "0-3,7,9-10");
        assert_eq!(format_cpu_list(&[5]), "5");
        assert_eq!(format_cpu_list(&[]), "");
    }

    #[test]
    fn cpu_plan_uses_the_p_and_e_lists() {
        let plan = CpuPlan::from_lists(vec![0, 1, 2, 3], vec![4, 5, 6, 7], None);
        assert!(plan.is_hybrid());
        assert_eq!(plan.p_physical, 4);
        assert_eq!(plan.e_physical, 4);
        assert_eq!(plan.physical_threads, 8);
        assert_eq!(plan.p_threads, 4);
        assert_eq!(plan.affinity(), &[0, 1, 2, 3]);
    }

    #[test]
    fn cpu_plan_falls_back_on_an_empty_topology() {
        let plan = CpuPlan::from_lists(Vec::new(), Vec::new(), None);
        let threads = available_threads();
        assert!(!plan.is_hybrid());
        assert_eq!(plan.p_physical, 0);
        assert_eq!(plan.e_physical, 0);
        assert_eq!(plan.physical_threads, threads);
        assert_eq!(plan.p_threads, threads);
        assert!(plan.affinity().is_empty());
    }

    #[test]
    fn cpu_plan_counts_physical_cores_from_the_sibling_lists() {
        let root = std::env::temp_dir().join(format!("russell_lab_os_info_siblings_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        for (cpu, siblings) in [(0, "0-1"), (1, "0-1"), (2, "2-3"), (3, "2-3")] {
            let dir = root.join(format!("cpu{cpu}/topology"));
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("thread_siblings_list"), siblings).unwrap();
        }
        assert_eq!(physical_count(&root, &[0, 1, 2, 3]), 2);
        assert_eq!(physical_count(&root, &[0, 2]), 2);
        assert_eq!(physical_count(&root, &[]), 0);
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn physical_count_falls_back_on_missing_or_malformed_siblings() {
        let root = std::env::temp_dir().join(format!("russell_lab_os_info_fallback_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);

        // Missing sibling file
        assert_eq!(physical_count(&root, &[0, 1]), 2);

        // Malformed sibling file
        let dir0 = root.join("cpu0/topology");
        std::fs::create_dir_all(&dir0).unwrap();
        std::fs::write(dir0.join("thread_siblings_list"), "bad").unwrap();
        assert_eq!(physical_count(&root, &[0, 1]), 2);

        // Empty sibling file
        std::fs::write(dir0.join("thread_siblings_list"), "   ").unwrap();
        assert_eq!(physical_count(&root, &[0, 1]), 2);

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn cpu_plan_handles_e_core_only_and_deduplication() {
        let plan = CpuPlan::from_lists(vec![], vec![19, 16, 17, 16, 18], None);
        assert!(plan.is_hybrid());
        assert!(plan.p_cores.is_empty());
        assert_eq!(plan.e_cores, vec![16, 17, 18, 19]);
        assert_eq!(plan.p_physical, 0);
        assert_eq!(plan.e_physical, 4);
        assert_eq!(plan.physical_threads, 4);
        assert_eq!(plan.p_threads, available_threads());
        assert!(plan.affinity().is_empty());
        assert!(plan.to_shell().contains("CPU_AFFINITY=none\n"));
    }

    #[test]
    fn cpu_plan_summary_formats_hybrid_and_non_hybrid() {
        let hybrid = CpuPlan::from_lists(vec![0, 1], vec![2, 3], None);
        let text = hybrid.summary();
        assert!(text.contains("hybrid: yes"));
        assert!(text.contains("P cores: 0-1 (2 logical, 2 physical)"));
        assert!(text.contains("E cores: 2-3 (2 logical, 2 physical)"));
        assert!(text.contains("physical threads: 4"));
        assert!(text.contains("P threads: 2"));
        assert!(text.contains("CPU affinity: 0-1"));

        let fallback = CpuPlan::from_lists(Vec::new(), Vec::new(), None);
        let text = fallback.summary();
        let threads = available_threads();
        assert!(text.contains("hybrid: no"));
        assert!(text.contains("P cores: none (0 logical, 0 physical)"));
        assert!(text.contains("E cores: none (0 logical, 0 physical)"));
        assert!(text.contains(&format!("physical threads: {threads}")));
        assert!(text.contains(&format!("P threads: {threads}")));
        assert!(text.contains("CPU affinity: none"));
    }

    #[test]
    fn cpu_plan_overrides_apply_topology_and_ignore_invalid_or_thread_env_vars() {
        let plan = cpu_plan_with_overrides(vec![0, 1, 2, 3], vec![4, 5], None, |name| match name {
            "P_CORES" => Some("0-1".to_string()),
            "E_CORES" => Some("8-11".to_string()),
            "PHYSICAL_THREADS" => Some("12".to_string()),
            "P_THREADS" => Some("4".to_string()),
            "MUMPS_THREADS" => Some("16".to_string()),
            "BLAS_THREADS" => Some("6".to_string()),
            _ => None,
        });
        assert_eq!(plan.p_cores, vec![0, 1]);
        assert_eq!(plan.e_cores, vec![8, 9, 10, 11]);
        assert_eq!(plan.p_physical, 2);
        assert_eq!(plan.e_physical, 4);
        assert_eq!(plan.physical_threads, 6);
        assert_eq!(plan.p_threads, 2);
        assert_eq!(plan.affinity(), &[0, 1]);

        // Empty, comma-only, or malformed topology values keep the detected plan
        let plan = cpu_plan_with_overrides(vec![0, 1], vec![2, 3], None, |name| match name {
            "P_CORES" => Some(" , ".to_string()),
            "E_CORES" => Some("invalid".to_string()),
            _ => None,
        });
        assert_eq!(plan.p_cores, vec![0, 1]);
        assert_eq!(plan.e_cores, vec![2, 3]);
        assert_eq!(plan.physical_threads, 4);
        assert_eq!(plan.p_threads, 2);
        assert_eq!(plan.affinity(), &[0, 1]);

        let runtime_plan = get_cpu_plan_with_overrides();
        assert!(runtime_plan.physical_threads >= 1);
        assert!(runtime_plan.p_threads >= 1);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn intersect_filters_against_allowed_cpus() {
        assert_eq!(intersect(vec![0, 1, 2, 3], &Some(vec![1, 3, 5])), vec![1, 3]);
        assert_eq!(intersect(vec![0, 1, 2, 3], &Some(vec![4, 5])), Vec::<usize>::new());
        assert_eq!(intersect(vec![0, 1, 2, 3], &None), vec![0, 1, 2, 3]);
    }

    #[test]
    fn set_cpu_affinity_rejects_an_empty_list() {
        assert!(set_cpu_affinity(&[]).is_err());
    }

    #[test]
    fn set_cpu_affinity_rejects_huge_cpu_id() {
        assert!(set_cpu_affinity(&[MAX_CPU_ID + 1]).is_err());
    }

    #[test]
    #[cfg(not(target_os = "linux"))]
    fn set_cpu_affinity_reports_the_platform() {
        let error = set_cpu_affinity(&[0]).unwrap_err();
        assert!(error.contains("only supported on Linux"));
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn set_cpu_affinity_round_trips_the_allowed_mask() {
        if !Path::new("/proc/thread-self/status").exists() {
            return;
        }
        let allowed = allowed_cpu_list().expect("Cpus_allowed_list");
        if allowed.is_empty() {
            return;
        }
        let first = vec![allowed[0]];
        struct Restore(Vec<usize>);
        impl Drop for Restore {
            fn drop(&mut self) {
                let _ = set_cpu_affinity(&self.0);
            }
        }
        let _restore = Restore(allowed.clone());
        set_cpu_affinity(&first).unwrap();
        assert_eq!(allowed_cpu_list(), Some(first));
    }

    #[test]
    fn cpu_plan_shell_has_the_expected_keys() {
        let plan = CpuPlan::from_lists(vec![0, 1], vec![2, 3], None);
        let text = plan.to_shell();
        assert!(text.contains("P_CORES=0-1\n"));
        assert!(text.contains("E_CORES=2-3\n"));
        assert!(text.contains("P_LOGICAL=2\n"));
        assert!(text.contains("E_LOGICAL=2\n"));
        assert!(text.contains("P_PHYSICAL=2\n"));
        assert!(text.contains("E_PHYSICAL=2\n"));
        assert!(text.contains("PHYSICAL_THREADS=4\n"));
        assert!(text.contains("P_THREADS=2\n"));
        assert!(text.contains("CPU_AFFINITY=0-1\n"));
        assert!(!text.contains("MUMPS_THREADS="));
        assert!(!text.contains("BLAS_THREADS="));
    }
}
