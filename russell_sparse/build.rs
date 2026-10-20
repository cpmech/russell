//! This build script compiles the C interfaces to the sparse solvers and emits
//! the corresponding link directives.
//!
//! The interface files are small C shims that wrap the external solver libraries
//! (UMFPACK and MUMPS) so that `russell_sparse` can call them through `extern "C"`.
//!
//! UMFPACK is required in every configuration, so the features never choose
//! *whether* there is a direct solver -- only where it comes from, and whether
//! MUMPS joins it:
//!
//! * (no feature)      UMFPACK from the package manager, located with pkg-config;
//! * `local_umfpack`   UMFPACK from a local install, under `RUSSELL_LOCAL_PREFIX`;
//! * `mumps`           MUMPS from that same local prefix, alongside either UMFPACK.
//!
//! The two UMFPACK sources exclude each other by construction: `/usr/local` (or
//! wherever the local install lives) is never searched when the system UMFPACK is
//! wanted, and pkg-config is never consulted when the local one is. A build
//! therefore cannot mix a package-manager SuiteSparse with a locally compiled one
//! -- they typically differ in version but share the `libumfpack.so.6` soname, so
//! a mix would not even be visible at run time.
//!
//! The local solvers live in a fixed layout, and `RUSSELL_LOCAL_PREFIX` must be
//! set (and exported) when either feature is enabled:
//!
//! ```text
//! $RUSSELL_LOCAL_PREFIX/include/umfpack     $RUSSELL_LOCAL_PREFIX/lib/umfpack
//! $RUSSELL_LOCAL_PREFIX/include/mumps       $RUSSELL_LOCAL_PREFIX/lib/mumps
//! ```
//!
//! Windows and Unix compile the *same* source list ([`shim_files`]); only the
//! include roots and the library names differ -- shared libraries on Unix, and
//! the statically linked `_cpmech` MUMPS on Windows.

#[cfg(any(feature = "local_umfpack", feature = "mumps"))]
use std::path::PathBuf;

/// The C shim files compiled in every configuration.
///
/// UMFPACK is always present; MUMPS joins with its feature. Keeping this in one
/// place is what stops the Windows and Unix builds from drifting apart.
fn shim_files() -> Vec<&'static str> {
    #[cfg_attr(not(feature = "mumps"), allow(unused_mut))]
    let mut files = vec!["c_code/interface_complex_umfpack.c", "c_code/interface_umfpack.c"];
    #[cfg(feature = "mumps")]
    files.extend(["c_code/interface_complex_mumps.c", "c_code/interface_mumps.c"]);
    files
}

fn main() {
    // The prefix decides which installation is linked, so changing it must rebuild.
    println!("cargo:rerun-if-env-changed=RUSSELL_LOCAL_PREFIX");
    println!("cargo:rerun-if-env-changed=MSYS2_PREFIX");

    #[cfg(target_os = "windows")]
    windows::compile();

    #[cfg(not(target_os = "windows"))]
    unix::compile();
}

// ----------------------------------------------------------------------------
// Local installs: $RUSSELL_LOCAL_PREFIX/{include,lib}/<solver>
// ----------------------------------------------------------------------------

/// Reads `RUSSELL_LOCAL_PREFIX`, dropping trailing separators and whitespace.
///
/// The variable is required by the local features, and an unset or empty value
/// stops the build with the expected layout instead of guessing a prefix: the
/// system and the local installations must never be confused, and a guess is how
/// they would be.
#[cfg(any(feature = "local_umfpack", feature = "mumps"))]
fn local_prefix() -> PathBuf {
    let raw = std::env::var("RUSSELL_LOCAL_PREFIX").unwrap_or_default();
    let trimmed = raw.trim().trim_end_matches(['/', '\\']);
    if trimmed.is_empty() {
        panic!(
            "russell_sparse: RUSSELL_LOCAL_PREFIX is not set, and a locally compiled solver was \
             requested. Export it as the root of the local install, laid out as\n\
             \x20 $RUSSELL_LOCAL_PREFIX/include/umfpack   $RUSSELL_LOCAL_PREFIX/lib/umfpack\n\
             \x20 $RUSSELL_LOCAL_PREFIX/include/mumps     $RUSSELL_LOCAL_PREFIX/lib/mumps"
        );
    }
    PathBuf::from(trimmed)
}

/// Resolves `$RUSSELL_LOCAL_PREFIX/{include,lib}/<solver>`, checking the header
/// the shim includes is there.
///
/// A wrong prefix fails here, naming the file it looked for, rather than later in
/// the C compiler or, worse, at link time against some other installation.
#[cfg(any(feature = "local_umfpack", feature = "mumps"))]
fn local_dirs(solver: &str, header: &str) -> (PathBuf, PathBuf) {
    let prefix = local_prefix();
    let include = prefix.join("include").join(solver);
    let lib = prefix.join("lib").join(solver);
    let header_path = include.join(header);
    if !header_path.is_file() {
        panic!(
            "russell_sparse: {header} not found at {}\n\
             The local install must be laid out as $RUSSELL_LOCAL_PREFIX/include/{solver} and \
             $RUSSELL_LOCAL_PREFIX/lib/{solver}; RUSSELL_LOCAL_PREFIX is currently {}.",
            header_path.display(),
            prefix.display()
        );
    }
    (include, lib)
}

/// Tells the crate which UMFPACK was linked, for `get_library_versions`.
fn emit_umfpack_source(source: &str) {
    println!("cargo:rustc-env=RUSSELL_SPARSE_UMFPACK={source}");
}

// ----------------------------------------------------------------------------
// Windows (MSYS2)
// ----------------------------------------------------------------------------

/// Windows (MSYS2): the system UMFPACK and MUMPS come from `MSYS2_PREFIX`, the
/// local ones from `RUSSELL_LOCAL_PREFIX`.
#[cfg(target_os = "windows")]
mod windows {
    use super::shim_files;
    #[cfg(not(feature = "local_umfpack"))]
    use std::path::PathBuf;

    /// Compiles and links the shims.
    pub fn compile() {
        let mut build = cc::Build::new();
        for file in shim_files() {
            build.file(file);
        }

        // UMFPACK: the system one, unless the local install was asked for.
        #[cfg(not(feature = "local_umfpack"))]
        {
            let prefix = msys2_prefix();
            build.include(prefix.join("include").join("suitesparse"));
            super::emit_link_search(&prefix.join("lib"));
            super::emit_umfpack_source("system (MSYS2)");
        }
        #[cfg(feature = "local_umfpack")]
        {
            let (include, lib) = super::local_dirs("umfpack", "umfpack.h");
            build.include(&include);
            super::emit_link_search(&lib);
            super::emit_rpath(&lib);
            super::emit_umfpack_source(&format!("local ({})", lib.display()));
        }

        // MUMPS, compiled locally with the `_cpmech` suffix: headers and shared
        // libraries here, static ones at link time.
        #[cfg(feature = "mumps")]
        {
            let (include, lib) = super::local_dirs("mumps", "dmumps_c.h");
            build.include(&include);
            super::emit_link_search(&lib);
            super::emit_rpath(&lib);
        }

        build.compile("c_code");

        println!("cargo:rustc-link-lib=dylib=umfpack");

        // The Windows MUMPS build is static and drags in its Fortran and METIS
        // dependencies; on Unix they are resolved by the shared libraries.
        #[cfg(feature = "mumps")]
        {
            println!("cargo:rustc-link-lib=static=dmumps_cpmech");
            println!("cargo:rustc-link-lib=static=zmumps_cpmech");
            println!("cargo:rustc-link-lib=static=mumps_common_cpmech");
            println!("cargo:rustc-link-lib=static=mpiseq_cpmech");
            println!("cargo:rustc-link-lib=static=pord_cpmech");
            println!("cargo:rustc-link-lib=dylib=gfortran");
            println!("cargo:rustc-link-lib=dylib=gomp");
            println!("cargo:rustc-link-lib=static=metis");
        }
    }

    /// The MSYS2 installation root, where the system UMFPACK lives.
    #[cfg(not(feature = "local_umfpack"))]
    fn msys2_prefix() -> PathBuf {
        let raw = std::env::var("MSYS2_PREFIX").expect("MSYS2_PREFIX environment variable not set");
        PathBuf::from(raw)
    }
}

// ----------------------------------------------------------------------------
// Unix (Linux/macOS)
// ----------------------------------------------------------------------------

/// Unix (Linux/macOS), where the system libraries are shared and located through
/// pkg-config, and the local ones through `RUSSELL_LOCAL_PREFIX`.
#[cfg(not(target_os = "windows"))]
mod unix {
    use super::shim_files;

    /// Include directories of a package-manager or Homebrew SuiteSparse, used only
    /// when pkg-config cannot answer. `$RUSSELL_LOCAL_PREFIX` is deliberately
    /// absent: the local install is the `local_umfpack` feature's business, and
    /// searching it here is how the two get mixed.
    #[cfg(not(feature = "local_umfpack"))]
    const SYSTEM_INCLUDE_DIRS: &[&str] = &["/opt/homebrew/include/suitesparse", "/usr/include/suitesparse"];

    /// Library directories of a package-manager or Homebrew SuiteSparse (see
    /// [`SYSTEM_INCLUDE_DIRS`]).
    #[cfg(not(feature = "local_umfpack"))]
    const SYSTEM_LIB_DIRS: &[&str] = &["/opt/homebrew/lib", "/usr/lib/x86_64-linux-gnu", "/usr/lib", "/usr/lib64"];

    /// Compiles and links the shims.
    pub fn compile() {
        let mut build = cc::Build::new();
        for file in shim_files() {
            build.file(file);
        }

        #[cfg(not(feature = "local_umfpack"))]
        umfpack_system(&mut build);
        #[cfg(feature = "local_umfpack")]
        umfpack_local(&mut build);

        #[cfg(feature = "mumps")]
        mumps_local(&mut build);

        build.compile("c_code");
    }

    /// UMFPACK from the package manager: pkg-config when it answers, the directory
    /// list above otherwise, and a hard failure if neither does.
    #[cfg(not(feature = "local_umfpack"))]
    fn umfpack_system(build: &mut cc::Build) {
        if let Some(lib) = probe_umfpack() {
            for include in &lib.include_paths {
                build.include(include);
            }
            super::emit_umfpack_source("system (pkg-config)");
            return;
        }

        let include_dirs = existing_dirs(SYSTEM_INCLUDE_DIRS);
        let lib_dirs = existing_dirs(SYSTEM_LIB_DIRS);
        let header_found = include_dirs.iter().any(|dir| std::path::Path::new(dir).join("umfpack.h").is_file());
        if !header_found || lib_dirs.is_empty() {
            panic!(
                "russell_sparse: UMFPACK was not found.\n\
                 Install it with `sudo apt install libsuitesparse-dev` (Debian/Ubuntu) or \
                 `sudo pacman -S suitesparse` (Arch), or enable the `local_umfpack` feature and set \
                 RUSSELL_LOCAL_PREFIX to a local install laid out as \
                 $RUSSELL_LOCAL_PREFIX/include/umfpack and $RUSSELL_LOCAL_PREFIX/lib/umfpack."
            );
        }
        build.includes(&include_dirs);
        for dir in &lib_dirs {
            super::emit_link_search(std::path::Path::new(dir));
        }
        println!("cargo:rustc-link-lib=dylib=umfpack");
        super::emit_umfpack_source(&format!("system ({})", include_dirs.join(", ")));
    }

    /// UMFPACK from `$RUSSELL_LOCAL_PREFIX`; pkg-config is not consulted.
    #[cfg(feature = "local_umfpack")]
    fn umfpack_local(build: &mut cc::Build) {
        let (include, lib) = super::local_dirs("umfpack", "umfpack.h");
        build.include(&include);
        super::emit_link_search(&lib);
        super::emit_rpath(&lib);
        println!("cargo:rustc-link-lib=dylib=umfpack");
        super::emit_umfpack_source(&format!("local ({})", lib.display()));
    }

    /// MUMPS from `$RUSSELL_LOCAL_PREFIX`, shared, so its dependencies are already
    /// satisfied by the libraries' own `DT_NEEDED` entries.
    #[cfg(feature = "mumps")]
    fn mumps_local(build: &mut cc::Build) {
        let (include, lib) = super::local_dirs("mumps", "dmumps_c.h");
        build.include(&include);
        super::emit_link_search(&lib);
        super::emit_rpath(&lib);
        println!("cargo:rustc-link-lib=dylib=dmumps_cpmech");
        println!("cargo:rustc-link-lib=dylib=zmumps_cpmech");
    }

    /// Asks `pkg-config` for UMFPACK, which pulls in AMD, CHOLMOD and friends via
    /// `Requires.private`.
    ///
    /// Both spellings are tried because SuiteSparse is inconsistent about it:
    /// Debian, Arch and upstream's own `make install` write `UMFPACK.pc`, while
    /// some setups use lowercase. Returns `None` if pkg-config is missing or does
    /// not know UMFPACK under either name.
    #[cfg(not(feature = "local_umfpack"))]
    fn probe_umfpack() -> Option<pkg_config::Library> {
        for name in ["UMFPACK", "umfpack"] {
            match pkg_config::Config::new().probe(name) {
                Ok(lib) => return Some(lib),
                // The library isn't known under this name; try the other spelling.
                Err(pkg_config::Error::ProbeFailure { .. }) | Err(pkg_config::Error::Failure { .. }) => continue,
                // pkg-config is missing, or cross-compiling without a sysroot:
                // retrying with a different name would not help, and falling back
                // silently would hide a broken toolchain.
                Err(error) => {
                    println!(
                        "cargo:warning=russell_sparse: pkg-config failed ({error}); \
                         falling back to the built-in directory list for UMFPACK"
                    );
                    break;
                }
            }
        }
        None
    }

    /// Keeps only the directories that actually exist.
    ///
    /// Apart from reducing command-line noise, this avoids entries such as
    /// `/usr/lib` shadowing what a Nix devShell provides.
    #[cfg(not(feature = "local_umfpack"))]
    fn existing_dirs(dirs: &[&str]) -> Vec<String> {
        dirs.iter()
            .filter(|dir| std::path::Path::new(dir).is_dir())
            .map(|dir| dir.to_string())
            .collect()
    }
}

/// Emits a link search directory.
fn emit_link_search(dir: &std::path::Path) {
    println!("cargo:rustc-link-search=native={}", dir.display());
}

/// Records the directory in the run path of this package's own binaries, tests and
/// examples.
///
/// A local SuiteSparse and the package-manager one share the `libumfpack.so.6`
/// soname, so without this the dynamic loader picks whichever copy happens to be in
/// its cache -- which on a machine with both installed is not necessarily the one
/// that was linked. The flags are scoped to this package's targets, so downstream
/// crates are unaffected: they get the requirement documented instead (the library
/// directory must be reachable at run time).
#[cfg(any(feature = "local_umfpack", feature = "mumps"))]
fn emit_rpath(dir: &std::path::Path) {
    let flag = format!("-Wl,-rpath,{}", dir.display());
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    // Cargo rejects the instruction for a target kind the package does not have, so
    // only the kinds that exist are announced; the plain instruction covers the
    // library's own unit-test binary, which the per-kind ones do not reach.
    println!("cargo:rustc-link-arg={flag}");
    for (kind, subdir) in [("bins", "src/bin"), ("tests", "tests"), ("examples", "examples"), ("benches", "benches")] {
        if manifest.join(subdir).is_dir() {
            println!("cargo:rustc-link-arg-{kind}={flag}");
        }
    }
}
