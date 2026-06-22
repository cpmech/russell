use std::process::Command;

/// Returns the system, including CPU and GPU information, on Linux platforms
pub fn get_system_info_linux() -> String {
    let mut info = String::new();

    // Kernel and hostname
    if let Ok(output) = Command::new("uname").arg("-r").output() {
        if let Ok(s) = String::from_utf8(output.stdout) {
            info.push_str(&format!("Kernel: {}", s.trim()));
        }
    }
    if let Ok(output) = Command::new("hostname").output() {
        if let Ok(s) = String::from_utf8(output.stdout) {
            info.push_str(&format!("\nHostname: {}", s.trim()));
        }
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
                info.push_str(&format!("\n\n--- CPU ---\n{}", s));
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

    // Memory
    if let Ok(s) = std::fs::read_to_string("/proc/meminfo") {
        let mut mem_lines: Vec<&str> = Vec::new();
        for line in s.lines() {
            let line = line.trim();
            if line.starts_with("MemTotal")
                || line.starts_with("MemFree")
                || line.starts_with("MemAvailable")
                || line.starts_with("SwapTotal")
            {
                mem_lines.push(line);
            }
        }
        if !mem_lines.is_empty() {
            info.push_str(&format!("\n\n--- Memory ---\n{}", mem_lines.join("\n")));
        }
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

    // OS release
    if let Ok(s) = std::fs::read_to_string("/etc/os-release") {
        let mut os_lines: Vec<&str> = Vec::new();
        for line in s.lines() {
            let line = line.trim();
            if line.starts_with("PRETTY_NAME") || line.starts_with("NAME") || line.starts_with("VERSION_ID") {
                os_lines.push(line);
            }
        }
        if !os_lines.is_empty() {
            info.push_str(&format!("\n\n--- OS ---\n{}", os_lines.join("\n")));
        }
    }

    info
}

pub fn get_system_info_windows() -> String {
    "NOT AVAILABLE".to_string()
}

pub fn get_system_info_macos() -> String {
    "NOT AVAILABLE".to_string()
}

// Versions of external sparse solver libraries installed via the project's bash scripts
const CUDSS_SCRIPT_VERSION: &str = "0.8.0.10";
const CUDA_SCRIPT_VERSION: &str = "13";
const MUMPS_SCRIPT_VERSION: &str = "5.9.0";
const SUITESPARSE_VERSION: &str = "latest (from GitHub)";

/// Returns versions of the external sparse solver libraries used by this crate
pub fn get_library_versions() -> String {
    let mut info = String::new();

    // cuDSS
    if cfg!(feature = "cudss") {
        info.push_str(&format!(
            "cuDSS: {} (CUDA {})\n",
            CUDSS_SCRIPT_VERSION, CUDA_SCRIPT_VERSION
        ));
    } else {
        info.push_str(&format!(
            "cuDSS: {} (CUDA {}) [not compiled in]\n",
            CUDSS_SCRIPT_VERSION, CUDA_SCRIPT_VERSION
        ));
    }

    // MUMPS
    if cfg!(feature = "local_sparse") {
        info.push_str(&format!("MUMPS: {}\n", MUMPS_SCRIPT_VERSION));
    } else {
        info.push_str(&format!("MUMPS: {} [not compiled in]\n", MUMPS_SCRIPT_VERSION));
    }

    // SuiteSparse (always compiled in, either from system or from script)
    info.push_str(&format!("SuiteSparse: {}", SUITESPARSE_VERSION));

    info
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_system_info_windows_returns_not_available() {
        assert_eq!(get_system_info_windows(), "NOT AVAILABLE");
    }

    #[test]
    fn get_system_info_macos_returns_not_available() {
        assert_eq!(get_system_info_macos(), "NOT AVAILABLE");
    }

    #[test]
    fn get_system_info_linux_is_non_empty() {
        let info = get_system_info_linux();
        assert!(!info.is_empty(), "Linux system info should not be empty");
    }

    #[test]
    fn get_system_info_linux_contains_expected_sections() {
        let info = get_system_info_linux();
        println!("{}", info);
        assert!(info.contains("Kernel:"), "Should contain kernel version");
        assert!(info.contains("Hostname:"), "Should contain hostname");
        assert!(info.contains("--- CPU ---"), "Should contain CPU section");
        assert!(info.contains("--- Memory ---"), "Should contain Memory section");
        assert!(info.contains("--- OS ---"), "Should contain OS section");
    }

    #[test]
    fn get_system_info_linux_memory_contains_keys() {
        let info = get_system_info_linux();
        assert!(info.contains("MemTotal:"), "Should report total memory");
        assert!(info.contains("MemAvailable:"), "Should report available memory");
    }

    #[test]
    fn get_system_info_linux_os_contains_pretty_name() {
        let info = get_system_info_linux();
        assert!(info.contains("PRETTY_NAME"), "Should contain OS pretty name");
    }

    #[test]
    fn get_library_versions_is_non_empty() {
        let info = get_library_versions();
        assert!(!info.is_empty());
    }

    #[test]
    fn get_library_versions_contains_all_libraries() {
        let info = get_library_versions();
        assert!(info.contains("cuDSS:"), "Should contain cuDSS version");
        assert!(info.contains("MUMPS:"), "Should contain MUMPS version");
        assert!(info.contains("SuiteSparse:"), "Should contain SuiteSparse version");
    }

    #[test]
    fn get_library_versions_contains_expected_version_numbers() {
        let info = get_library_versions();
        println!("{}", info);
        assert!(info.contains(CUDSS_SCRIPT_VERSION), "Should contain cuDSS version");
        assert!(info.contains(MUMPS_SCRIPT_VERSION), "Should contain MUMPS version");
        assert!(info.contains(SUITESPARSE_VERSION), "Should contain SuiteSparse version");
    }
}
