//! This build script compiles the C interfaces to the sparse solvers and emits
//! the corresponding link directives.
//!
//! The interface files are small C shims that wrap the external solver libraries
//! (UMFPACK and MUMPS) so that `russell_sparse` can call them through `extern "C"`.
//!
//! UMFPACK is optional and enabled with the `umfpack` feature. When it is
//! enabled, the build looks for a local installation first and only then for the
//! system one:
//!
//! * `$RUSSELL_LOCAL_PREFIX/include/umfpack` and `$RUSSELL_LOCAL_PREFIX/lib/umfpack`
//!   are used when `RUSSELL_LOCAL_PREFIX` is set and both directories exist;
//! * otherwise pkg-config (Unix) or `MSYS2_PREFIX` (Windows) is used, and a hard
//!   error is raised if the system library is missing too.
//!
//! A local installation therefore takes precedence over the system one. The two
//! sources exclude each other within a build: pkg-config is not consulted when
//! the local install was found, and the local prefix is not searched on the
//! system fallback path. They typically differ in version but share the
//! `libumfpack.so.6` soname, so a mix would not even be visible at run time.
//!
//! MUMPS is independent (`mumps` feature), always comes from
//! `RUSSELL_LOCAL_PREFIX`, and may be enabled with or without UMFPACK. Its
//! layout is fixed:
//!
//! ```text
//! $RUSSELL_LOCAL_PREFIX/include/umfpack     $RUSSELL_LOCAL_PREFIX/lib/umfpack
//! $RUSSELL_LOCAL_PREFIX/include/mumps       $RUSSELL_LOCAL_PREFIX/lib/mumps
//! ```
//!
//! Windows and Unix compile the *same* source list ([`shim_files`]); only the
//! include roots and the library names differ -- shared libraries on Unix, and
//! the statically linked `_cpmech` MUMPS on Windows.

#[cfg(any(feature = "umfpack", feature = "mumps"))]
use std::path::PathBuf;

/// The C shim files compiled for the enabled solvers.
///
/// UMFPACK joins with its feature (opt-in); MUMPS with its own. Keeping
/// this in one place is what stops the Windows and Unix builds from drifting
/// apart.
fn shim_files() -> Vec<&'static str> {
    #[cfg_attr(not(any(feature = "umfpack", feature = "mumps")), allow(unused_mut))]
    let mut files = Vec::new();
    #[cfg(feature = "umfpack")]
    files.extend(["c_code/interface_complex_umfpack.c", "c_code/interface_umfpack.c"]);
    #[cfg(feature = "mumps")]
    files.extend(["c_code/interface_complex_mumps.c", "c_code/interface_mumps.c"]);
    files
}

fn main() {
    // The prefix decides which installation is linked, so changing it must rebuild.
    println!("cargo:rerun-if-env-changed=RUSSELL_LOCAL_PREFIX");
    println!("cargo:rerun-if-env-changed=MSYS2_PREFIX");

    // Mirror the dense-crate notice: with neither external solver enabled the
    // Native backend is the only one, and that choice should be visible in the
    // build log.
    #[cfg(not(any(feature = "umfpack", feature = "mumps")))]
    println!("cargo:warning=russell_sparse: using the native Rust sparse solver (no external library)");

    #[cfg(target_os = "windows")]
    windows::compile();

    #[cfg(not(target_os = "windows"))]
    unix::compile();
}

// ----------------------------------------------------------------------------
// Local installs: $RUSSELL_LOCAL_PREFIX/{include,lib}/<solver>
// ----------------------------------------------------------------------------

/// Reads `RUSSELL_LOCAL_PREFIX`, returning `None` when unset or empty.
///
/// For UMFPACK an unset prefix simply means "use the system library"; MUMPS
/// requires the variable (see [`local_prefix`]).
#[cfg(any(feature = "umfpack", feature = "mumps"))]
fn local_prefix_opt() -> Option<PathBuf> {
    let raw = std::env::var("RUSSELL_LOCAL_PREFIX").ok()?;
    let trimmed = raw.trim().trim_end_matches(['/', '\\']);
    if trimmed.is_empty() { None } else { Some(PathBuf::from(trimmed)) }
}

/// Reads `RUSSELL_LOCAL_PREFIX`, stopping the build when it is unset.
///
/// MUMPS is compiled locally only, so the variable is required with `mumps`.
#[cfg(feature = "mumps")]
fn local_prefix() -> PathBuf {
    match local_prefix_opt() {
        Some(prefix) => prefix,
        None => panic!(
            "russell_sparse: RUSSELL_LOCAL_PREFIX is not set, and MUMPS was requested. Export it as the \
             root of the local install, laid out as\n\
             \x20 $RUSSELL_LOCAL_PREFIX/include/mumps   $RUSSELL_LOCAL_PREFIX/lib/mumps"
        ),
    }
}

/// Resolves `$RUSSELL_LOCAL_PREFIX/{include,lib}/<solver>`, checking the header
/// the shim includes is there.
///
/// A wrong prefix fails here, naming the file it looked for, rather than later in
/// the C compiler or, worse, at link time against some other installation.
#[cfg(feature = "mumps")]
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

/// Tries the local UMFPACK installation, returning `None` when it is not there.
///
/// The header is what selects the installation: a prefix that exists but does
/// not contain UMFPACK must fall through to the system library, not fail.
#[cfg(feature = "umfpack")]
fn try_local_dirs(solver: &str, header: &str) -> Option<(PathBuf, PathBuf)> {
    let prefix = local_prefix_opt()?;
    let include = prefix.join("include").join(solver);
    let lib = prefix.join("lib").join(solver);
    if include.join(header).is_file() && lib.is_dir() {
        Some((include, lib))
    } else {
        None
    }
}

/// Tries the vcpkg installation of UMFPACK, returning `None` when there is none.
///
/// vcpkg is the standard way for the MSVC toolchain to get UMFPACK. The root is
/// `VCPKG_ROOT`, or `VCPKG_INSTALLATION_ROOT` (which the GitHub Windows runners
/// export); the triplet comes from `RUSSELL_VCPKG_TRIPLET`, then
/// `VCPKG_DEFAULT_TRIPLET`, then the target architecture's dynamic triplet.
///
/// Only the dynamic triplet is accepted. Its `umfpack.lib` import library is all
/// the link needs; a static install renames the libraries (`umfpack_static.lib`)
/// and would also require the BLAS and its runtime to be linked by hand.
#[cfg(all(target_os = "windows", feature = "umfpack"))]
fn try_vcpkg_dirs(header: &str) -> Option<(PathBuf, PathBuf, String)> {
    let triplet = std::env::var("RUSSELL_VCPKG_TRIPLET")
        .or_else(|_| std::env::var("VCPKG_DEFAULT_TRIPLET"))
        .unwrap_or_else(|_| match std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() {
            Ok("aarch64") => "arm64-windows".to_string(),
            Ok("x86") | Ok("i686") => "x86-windows".to_string(),
            _ => "x64-windows".to_string(),
        });
    // Every candidate is tried, not just the first that is set: a machine can
    // have VCPKG_ROOT pointing at the vcpkg bundled with Visual Studio while the
    // install lives in another clone, which is exactly what the GitHub Windows
    // runners do.
    for var in ["VCPKG_ROOT", "VCPKG_INSTALLATION_ROOT"] {
        let Ok(root) = std::env::var(var) else {
            continue;
        };
        let installed = PathBuf::from(root).join("installed").join(&triplet);
        let include = installed.join("include").join("suitesparse");
        let lib = installed.join("lib");
        if include.join(header).is_file() && lib.join("umfpack.lib").is_file() {
            return Some((include, lib, triplet));
        }
    }
    None
}

/// Tells the crate which UMFPACK was linked, for `get_library_versions`.
#[cfg(feature = "umfpack")]
fn emit_umfpack_source(source: &str) {
    println!("cargo:rustc-env=RUSSELL_SPARSE_UMFPACK={source}");
}

// ----------------------------------------------------------------------------
// Windows (MSVC and MSYS2)
// ----------------------------------------------------------------------------

/// Windows, where two toolchains are supported and the UMFPACK sources differ
/// between them:
///
/// * a local install under `RUSSELL_LOCAL_PREFIX` (either toolchain);
/// * a vcpkg install (`VCPKG_ROOT` or `VCPKG_INSTALLATION_ROOT` plus a triplet),
///   which is how MSVC gets UMFPACK;
/// * MSYS2 (`MSYS2_PREFIX`), which is how the GNU toolchain does.
///
/// MUMPS is only available with the GNU toolchain: its Windows build is made of
/// MinGW objects and gfortran, which MSVC cannot link.
#[cfg(target_os = "windows")]
mod windows {
    use super::shim_files;
    #[cfg(feature = "umfpack")]
    use std::path::PathBuf;

    /// Compiles and links the shims.
    pub fn compile() {
        let files = shim_files();
        if files.is_empty() {
            return;
        }
        let mut build = cc::Build::new();
        for file in files {
            build.file(file);
        }

        // UMFPACK: the local install first, then vcpkg, then MSYS2.
        #[cfg(feature = "umfpack")]
        {
            let found = find_umfpack();
            build.include(&found.include);
            super::emit_link_search(&found.lib);
            super::emit_rpath(&found.lib);
            super::emit_umfpack_source(&found.label);
        }

        // MUMPS, compiled locally with the `_cpmech` suffix: headers and shared
        // libraries here, static ones at link time. The Windows build is MinGW
        // objects plus gfortran, so only the GNU toolchain can link it.
        #[cfg(feature = "mumps")]
        {
            assert!(
                super::target_env() != "msvc",
                "russell_sparse: MUMPS on Windows needs the GNU toolchain (MSYS2); build with \
                 `x86_64-pc-windows-gnu`, or disable the `mumps` feature"
            );
            let (include, lib) = super::local_dirs("mumps", "dmumps_c.h");
            build.include(&include);
            super::emit_link_search(&lib);
            super::emit_rpath(&lib);
        }

        build.compile("c_code");

        // UMFPACK comes after the shims that call it. The MSYS2 library is an
        // import archive, and GNU ld pulls an archive member only for a symbol
        // that is already undefined, so `-lumfpack` emitted before `-lc_code`
        // resolves nothing and every `umfpack_*` reference is reported missing.
        #[cfg(feature = "umfpack")]
        println!("cargo:rustc-link-lib=dylib=umfpack");

        // The same ordering applies to MUMPS, whose Windows libraries are static
        // archives: they have to follow the shims that call them.
        #[cfg(feature = "mumps")]
        {
            println!("cargo:rustc-link-lib=static=dmumps_cpmech");
            println!("cargo:rustc-link-lib=static=zmumps_cpmech");
            println!("cargo:rustc-link-lib=static=mumps_common_cpmech");
            println!("cargo:rustc-link-lib=static=mpiseq_cpmech");
            println!("cargo:rustc-link-lib=static=pord_cpmech");
            // MSYS2 ships an import library for METIS, not a static one.
            println!("cargo:rustc-link-lib=dylib=metis");
            // The Windows MUMPS libraries are static, so their BLAS calls (and
            // the Fortran runtime) have to be linked here; the Linux shared
            // objects carry them in DT_NEEDED instead.
            println!("cargo:rustc-link-lib=dylib=openblas");
            println!("cargo:rustc-link-lib=dylib=gfortran");
            println!("cargo:rustc-link-lib=dylib=gomp");
        }
    }

    /// Where the UMFPACK headers and import library were found.
    #[cfg(feature = "umfpack")]
    struct Umfpack {
        include: PathBuf,
        lib: PathBuf,
        label: String,
    }

    /// Selects the UMFPACK source: local prefix, then vcpkg, then MSYS2.
    #[cfg(feature = "umfpack")]
    fn find_umfpack() -> Umfpack {
        if let Some((include, lib)) = super::try_local_dirs("umfpack", "umfpack.h") {
            return Umfpack {
                include,
                lib: lib.clone(),
                label: format!("local ({})", lib.display()),
            };
        }
        if super::local_prefix_opt().is_some() {
            println!(
                "cargo:warning=russell_sparse: RUSSELL_LOCAL_PREFIX is set but does not contain \
                 UMFPACK; looking for a vcpkg or MSYS2 install"
            );
        }
        if let Some((include, lib, triplet)) = super::try_vcpkg_dirs("umfpack.h") {
            return Umfpack {
                include,
                lib,
                label: format!("vcpkg ({triplet})"),
            };
        }
        if super::target_env() == "msvc" {
            panic!(
                "russell_sparse: UMFPACK was not found for MSVC.\n\
                 Install it with vcpkg (`vcpkg install suitesparse-umfpack[core]:x64-windows`) and set \
                 VCPKG_ROOT (or VCPKG_INSTALLATION_ROOT), or set RUSSELL_LOCAL_PREFIX to an install laid \
                 out as $RUSSELL_LOCAL_PREFIX/include/umfpack and $RUSSELL_LOCAL_PREFIX/lib/umfpack. \
                 Alternatively, leave the `umfpack` feature disabled to build without UMFPACK."
            );
        }
        let raw = match std::env::var("MSYS2_PREFIX") {
            Ok(raw) => raw,
            Err(_) => panic!(
                "russell_sparse: UMFPACK was not found.\n\
                 Set MSYS2_PREFIX to the MSYS2 root that carries the SuiteSparse package, set \
                 VCPKG_ROOT for a vcpkg install, set RUSSELL_LOCAL_PREFIX to a local install, or leave \
                 the `umfpack` feature disabled."
            ),
        };
        let prefix = PathBuf::from(raw);
        Umfpack {
            include: prefix.join("include").join("suitesparse"),
            lib: prefix.join("lib"),
            label: "system (MSYS2)".to_string(),
        }
    }
}

/// The target ABI on Windows: `"msvc"` or `"gnu"`.
#[cfg(target_os = "windows")]
fn target_env() -> String {
    std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default()
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
    /// when pkg-config cannot answer.
    #[cfg(feature = "umfpack")]
    const SYSTEM_INCLUDE_DIRS: &[&str] = &["/opt/homebrew/include/suitesparse", "/usr/include/suitesparse"];

    /// Library directories of a package-manager or Homebrew SuiteSparse (see
    /// [`SYSTEM_INCLUDE_DIRS`]).
    #[cfg(feature = "umfpack")]
    const SYSTEM_LIB_DIRS: &[&str] = &["/opt/homebrew/lib", "/usr/lib/x86_64-linux-gnu", "/usr/lib", "/usr/lib64"];

    /// Compiles and links the shims.
    pub fn compile() {
        let files = shim_files();
        if files.is_empty() {
            return;
        }
        let mut build = cc::Build::new();
        for file in files {
            build.file(file);
        }

        #[cfg(feature = "umfpack")]
        umfpack(&mut build);

        #[cfg(feature = "mumps")]
        mumps_local(&mut build);

        build.compile("c_code");
    }

    /// UMFPACK: the local installation first, then the system one.
    #[cfg(feature = "umfpack")]
    fn umfpack(build: &mut cc::Build) {
        if let Some((include, lib)) = super::try_local_dirs("umfpack", "umfpack.h") {
            build.include(&include);
            super::emit_link_search(&lib);
            super::emit_rpath(&lib);
            println!("cargo:rustc-link-lib=dylib=umfpack");
            super::emit_umfpack_source(&format!("local ({})", lib.display()));
            return;
        }
        if super::local_prefix_opt().is_some() {
            println!(
                "cargo:warning=russell_sparse: RUSSELL_LOCAL_PREFIX is set but does not contain UMFPACK; \
                 falling back to the system library"
            );
        }
        umfpack_system(build);
    }

    /// UMFPACK from the package manager: pkg-config when it answers, the directory
    /// list above otherwise, and a hard failure if neither does.
    #[cfg(feature = "umfpack")]
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
                 `sudo pacman -S suitesparse` (Arch), or set RUSSELL_LOCAL_PREFIX to a local install \
                 laid out as $RUSSELL_LOCAL_PREFIX/include/umfpack and \
                 $RUSSELL_LOCAL_PREFIX/lib/umfpack. Alternatively, leave the `umfpack` feature \
                 disabled to build without UMFPACK."
            );
        }
        build.includes(&include_dirs);
        for dir in &lib_dirs {
            super::emit_link_search(std::path::Path::new(dir));
        }
        println!("cargo:rustc-link-lib=dylib=umfpack");
        super::emit_umfpack_source(&format!("system ({})", include_dirs.join(", ")));
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
    #[cfg(feature = "umfpack")]
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
    #[cfg(feature = "umfpack")]
    fn existing_dirs(dirs: &[&str]) -> Vec<String> {
        dirs.iter()
            .filter(|dir| std::path::Path::new(dir).is_dir())
            .map(|dir| dir.to_string())
            .collect()
    }
}

/// Emits a link search directory.
#[cfg(any(feature = "umfpack", feature = "mumps"))]
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
#[cfg(any(feature = "umfpack", feature = "mumps"))]
fn emit_rpath(dir: &std::path::Path) {
    // MSVC's link.exe has no rpath and rejects -Wl arguments. Windows resolves
    // the DLLs through PATH there, which the installation instructions say.
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        return;
    }
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
