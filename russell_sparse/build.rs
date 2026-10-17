//! This build script compiles the C interfaces to the sparse solvers and emits
//! the corresponding link directives.
//!
//! The interface files are small C shims that wrap the external solver libraries
//! (UMFPACK and MUMPS) so that `russell_sparse` can call them through `extern "C"`.
//!
//! One feature controls what is compiled and linked:
//!
//! * `local_sparse` -- enable MUMPS (in addition to UMFPACK). MUMPS must be
//!   compiled and installed locally; its libraries are assumed to be reachable
//!   through the fallback directories below.
//!
//! UMFPACK is always compiled and linked, regardless of the features.
//!
//! Windows and Unix compile the *same* source list ([`shim_files`]); only the
//! include directories and the library names differ, so adding a solver means
//! touching one list and two link sections, not two copies of the whole build.

/// The C shim files compiled in every configuration.
///
/// UMFPACK is always present; MUMPS joins with its feature. Keeping this in one
/// place is what stops the Windows and Unix builds from drifting apart.
fn shim_files() -> Vec<&'static str> {
    #[cfg_attr(not(feature = "local_sparse"), allow(unused_mut))]
    let mut files = vec!["c_code/interface_complex_umfpack.c", "c_code/interface_umfpack.c"];
    #[cfg(feature = "local_sparse")]
    files.extend(["c_code/interface_complex_mumps.c", "c_code/interface_mumps.c"]);
    files
}

fn main() {
    #[cfg(target_os = "windows")]
    windows::compile();

    #[cfg(not(target_os = "windows"))]
    unix::compile();
}

// ----------------------------------------------------------------------------
// Windows (MSYS2)
// ----------------------------------------------------------------------------

/// Windows (MSYS2), where everything lives under `MSYS2_PREFIX`.
#[cfg(target_os = "windows")]
mod windows {
    use super::shim_files;

    /// Compiles and links the shims; `MSYS2_PREFIX` must point at the MSYS2 root.
    pub fn compile() {
        let prefix = std::env::var("MSYS2_PREFIX").expect("MSYS2_PREFIX environment variable not set");

        let mut build = cc::Build::new();
        for file in shim_files() {
            build.file(file);
        }
        build.include(format!("{prefix}/include/suitesparse"));
        #[cfg(feature = "local_sparse")]
        build.include(format!("{prefix}/include/mumps"));
        build.compile("c_code");

        println!("cargo:rustc-link-search=native={prefix}/lib");
        println!("cargo:rustc-link-lib=dylib=umfpack");

        // MUMPS, compiled locally with the `_cpmech` suffix, is static here and
        // drags in its Fortran and METIS dependencies.
        #[cfg(feature = "local_sparse")]
        {
            println!("cargo:rustc-link-search=native={prefix}/lib/mumps");
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
}

// ----------------------------------------------------------------------------
// Unix (Linux/macOS)
// ----------------------------------------------------------------------------

/// Unix (Linux/macOS), where the libraries are shared and found through
/// `pkg-config` or the fallback directories below.
#[cfg(not(target_os = "windows"))]
mod unix {
    use super::shim_files;

    /// Fallback include directories, used when pkg-config knows nothing about the
    /// libraries (locally compiled SuiteSparse/MUMPS and Homebrew ship no `.pc`
    /// files). Nonexistent directories are dropped by [`existing_dirs`].
    const INCLUDE_DIRS: &[&str] = &[
        "/opt/homebrew/include/suitesparse",
        "/usr/include/suitesparse",
        "/usr/local/include/mumps",
        "/usr/local/include/suitesparse",
    ];

    /// Fallback library directories (see [`INCLUDE_DIRS`]).
    const LIB_DIRS: &[&str] = &[
        "/opt/homebrew/lib",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib",
        "/usr/lib64",
        "/usr/local/lib/mumps",
        "/usr/local/lib/suitesparse",
    ];

    /// Compiles and links the shims.
    pub fn compile() {
        let inc_dirs = existing_dirs(INCLUDE_DIRS);
        let lib_dirs = existing_dirs(LIB_DIRS);

        // pkg-config is authoritative when it resolves, because it reports the real
        // include and link paths (distro packages, Homebrew, Nix devShells) instead
        // of the list this script has to guess. A locally compiled SuiteSparse
        // normally ships no `.pc` file, so the fallback takes over there.
        //
        // The answer depends only on what is installed. It used to be suppressed
        // whenever another solver feature was enabled, which threw away a working
        // pkg-config answer for an unrelated reason.
        let umfpack = probe_umfpack();

        let mut build = cc::Build::new();
        for file in shim_files() {
            build.file(file);
        }
        match &umfpack {
            Some(lib) => {
                for inc in &lib.include_paths {
                    build.include(inc);
                }
            }
            None => {
                build.includes(&inc_dirs);
            }
        }
        build.compile("c_code");

        // When pkg-config found UMFPACK it already emitted the link directives;
        // otherwise the fallback list has to provide them.
        if umfpack.is_none() {
            for dir in &lib_dirs {
                println!("cargo:rustc-link-search=native={dir}");
            }
            println!("cargo:rustc-link-lib=dylib=umfpack");
        }

        // MUMPS, compiled locally, is shared on Unix and needs no extra flags: the
        // fallback list above already carries its library directory.
        #[cfg(feature = "local_sparse")]
        {
            println!("cargo:rustc-link-lib=dylib=dmumps_cpmech");
            println!("cargo:rustc-link-lib=dylib=zmumps_cpmech");
        }
    }

    /// Asks `pkg-config` for UMFPACK, which pulls in AMD, CHOLMOD and friends via
    /// `Requires.private`.
    ///
    /// Both spellings are tried because SuiteSparse is inconsistent about it:
    /// Debian, Arch and upstream's own `make install` write `UMFPACK.pc`, while
    /// some setups use lowercase. Returns `None` if pkg-config is missing or does
    /// not know UMFPACK under either name.
    fn probe_umfpack() -> Option<pkg_config::Library> {
        for name in ["UMFPACK", "umfpack"] {
            match pkg_config::Config::new().probe(name) {
                Ok(lib) => return Some(lib),
                // The library isn't known under this name; try the other spelling.
                Err(pkg_config::Error::ProbeFailure { .. }) | Err(pkg_config::Error::Failure { .. }) => continue,
                // pkg-config is missing, or cross-compiling without a sysroot:
                // retrying with a different name would not help, and silently
                // falling back would hide a broken toolchain.
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
    fn existing_dirs(dirs: &[&str]) -> Vec<String> {
        dirs.iter()
            .filter(|dir| std::path::Path::new(dir).is_dir())
            .map(|dir| dir.to_string())
            .collect()
    }
}
