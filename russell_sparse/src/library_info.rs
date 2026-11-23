//! Versions of the external sparse-solver libraries.

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
}
