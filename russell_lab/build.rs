//! Build script for `russell_lab`.
//!
//! The default configuration needs nothing from here: BLAS/LAPACK are the native
//! Rust routines in `src/native`, so no C code is compiled and nothing is linked.
//! That is also the only configuration available on targets with no C toolchain,
//! chiefly `wasm32-unknown-unknown` (there is no wasm libc for the shim to compile
//! against -- `inttypes.h` is missing).
//!
//! Two optional accelerators replace those routines with an external library,
//! through the C shim `c_code/interface_blas.c`. They provide the same symbols,
//! so only one library is ever linked: `intel_mkl` takes precedence, and
//! `open_blas` is used only when it is enabled and `intel_mkl` is not. Enabling
//! both is therefore MKL, not an error, and the same rule is written on every
//! gate in the crate as
//! `all(feature = "open_blas", not(feature = "intel_mkl"))`:
//!
//! * `open_blas` -- OpenBLAS, resolved through `pkg-config` with a fallback to
//!   hard-coded paths (mainly Homebrew, where the `.pc` files are keg-only);
//! * `intel_mkl` -- Intel oneAPI MKL, linked statically from the oneAPI tree.
//!
//! A third, development-only mode builds a prefixed copy of the shim against a
//! locally built reference LAPACK so the native routines can be checked for
//! parity: `dev_lapack`, in its own module below. It is independent of the
//! accelerators and may be combined with one of them.
//!
//! Each mode sits behind `#[cfg(feature = ...)]` on both its module and its call
//! site, so a configuration does not compile the code paths it does not use.

/// The C shim that exposes BLAS/LAPACK to the crate.
#[cfg(any(feature = "open_blas", feature = "intel_mkl", feature = "dev_lapack"))]
const SHIM: &str = "c_code/interface_blas.c";

fn main() {
    // The shim's prototypes and the vendor includes live in `c_code/interface.h`,
    // which `cc` does not track on its own: without this line an edit to the
    // header leaves a stale object behind and the change only shows up after a
    // `touch` of the source. (The build script has no `include_bytes!`, so this is
    // the only way cargo can know.)
    #[cfg(any(feature = "open_blas", feature = "intel_mkl", feature = "dev_lapack"))]
    println!("cargo:rerun-if-changed=c_code/interface.h");

    // Both accelerators define the same `c_*` entry points, so exactly one is
    // compiled: MKL when `intel_mkl` is on, OpenBLAS only when it is the only
    // accelerator. Saying so in the log is worthwhile, because a build with both
    // features is legal and the choice is otherwise invisible.
    #[cfg(all(feature = "open_blas", feature = "intel_mkl"))]
    println!("cargo:warning=russell_lab: both `open_blas` and `intel_mkl` are enabled -- linking MKL");

    // The reference build exists so the *native* routines can be tested against
    // it; it is independent of the accelerators, so it comes first.
    #[cfg(feature = "dev_lapack")]
    dev_lapack::compile();

    #[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
    accelerator::compile();

    #[cfg(not(any(feature = "open_blas", feature = "intel_mkl")))]
    println!("cargo:warning=russell_lab: using the native Rust BLAS/LAPACK (no external library)");
}

/// The external BLAS/LAPACK accelerators; MKL wins if both are enabled.
#[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
mod accelerator {
    use super::SHIM;

    /// Intel oneAPI MKL, linked statically.
    ///
    /// The version defaults to the `latest` symlink of the installed tree and can
    /// be pinned with `MKL_VERSION` (e.g. `2026.0`).
    #[cfg(feature = "intel_mkl")]
    pub fn compile() {
        const DEFAULT_VERSION: &str = "latest";
        let version = std::env::var("MKL_VERSION").unwrap_or_else(|_| DEFAULT_VERSION.to_string());
        let mkl = format!("/opt/intel/oneapi/mkl/{version}");
        let compiler = format!("/opt/intel/oneapi/compiler/{version}");

        cc::Build::new()
            .file(SHIM)
            .include(format!("{mkl}/include"))
            .define("USE_INTEL_MKL", None)
            .compile("c_code_interface_blas");

        // MKL's own library directory, then the compiler's (it carries the OpenMP
        // runtime `libiomp5`), then the system libraries MKL needs. Order matters.
        println!("cargo:rustc-link-search=native={mkl}/lib/intel64");
        println!("cargo:rustc-link-search=native={compiler}/lib");
        for lib in [
            "static=mkl_intel_lp64",
            "static=mkl_intel_thread",
            "static=mkl_core",
            "pthread",
            "m",
            "dl",
            "iomp5",
        ] {
            println!("cargo:rustc-link-lib={lib}");
        }
    }

    /// OpenBLAS, which carries LAPACK too.
    #[cfg(all(feature = "open_blas", not(feature = "intel_mkl")))]
    pub fn compile() {
        open_blas::compile();
    }

    #[cfg(all(feature = "open_blas", not(feature = "intel_mkl")))]
    mod open_blas {
        use super::SHIM;
        use std::path::PathBuf;

        /// Include directories used when `pkg-config` knows nothing.
        const FALLBACK_INCLUDES: &[&str] = &[
            "/usr/include/openblas",
            "/opt/homebrew/opt/lapack/include",
            "/opt/homebrew/opt/openblas/include",
            "/usr/local/opt/lapack/include",
            "/usr/local/opt/openblas/include",
        ];

        /// Library directories used when `pkg-config` knows nothing.
        const FALLBACK_LIBS: &[&str] = &[
            "/opt/homebrew/opt/lapack/lib",
            "/opt/homebrew/opt/openblas/lib",
            "/usr/local/opt/lapack/lib",
            "/usr/local/opt/openblas/lib",
        ];

        /// On Windows (MSYS2) everything lives under `MSYS2_PREFIX`.
        #[cfg(target_os = "windows")]
        pub fn compile() {
            let prefix = std::env::var("MSYS2_PREFIX").expect("MSYS2_PREFIX environment variable not set");
            cc::Build::new()
                .file(SHIM)
                .include(format!("{prefix}/include/openblas"))
                .compile("c_code_interface_blas");
            println!("cargo:rustc-link-search=native={prefix}/lib");
            println!("cargo:rustc-link-lib=dylib=openblas");
        }

        #[cfg(not(target_os = "windows"))]
        pub fn compile() {
            let mut build = cc::Build::new();
            build.file(SHIM);
            match resolve() {
                Some(found) => {
                    for dir in &found.includes {
                        build.include(dir);
                    }
                    build.compile("c_code_interface_blas");
                    emit_link_search(found.links.iter().map(|dir| dir.display().to_string()));
                    emit_link_lib(&found.libs);
                }
                None => {
                    build.includes(&existing_dirs(FALLBACK_INCLUDES));
                    build.compile("c_code_interface_blas");
                    emit_link_search(&existing_dirs(FALLBACK_LIBS));
                    emit_link_lib(["openblas", "lapack"]);
                }
            }
        }

        /// What `pkg-config` reported, once the headers have been checked.
        struct Resolved {
            includes: Vec<PathBuf>,
            links: Vec<PathBuf>,
            libs: Vec<String>,
        }

        /// Finds BLAS/LAPACK, preferring the module that carries both headers.
        ///
        /// `openblas` is one module with OpenBLAS plus LAPACK; the neutral
        /// `cblas` + `lapack` pair is what Nix provides.
        fn resolve() -> Option<Resolved> {
            from_pkg_config(&["openblas"]).or_else(|| from_pkg_config(&["cblas", "lapack"]))
        }

        /// Probes every module in `names`; all of them must resolve.
        ///
        /// Requiring the whole group matters for `cblas` + `lapack`: accepting
        /// `cblas` alone would compile and then fail at link time for want of
        /// `lapack`, which is a worse error than falling back.
        ///
        /// A successful `pkg-config` run is not enough on its own, hence the
        /// header check. The `lapack` module means different things on different
        /// distros: on RHEL/Rocky it is netlib LAPACK, which registers `lapack.pc`
        /// but ships no `lapack.h`, so the compile would fail instead of falling
        /// back.
        fn from_pkg_config(names: &[&str]) -> Option<Resolved> {
            let mut found = Vec::with_capacity(names.len());
            for name in names {
                found.push(pkg_config::Config::new().cargo_metadata(false).probe(name).ok()?);
            }
            let resolved = Resolved {
                includes: found.iter().flat_map(|lib| lib.include_paths.clone()).collect(),
                links: found.iter().flat_map(|lib| lib.link_paths.clone()).collect(),
                libs: found.iter().flat_map(|lib| lib.libs.clone()).collect(),
            };
            headers_available(&resolved.includes).then_some(resolved)
        }

        /// Checks that both headers used by the shim can actually be found.
        ///
        /// The implicit directories count too: Debian/Ubuntu put `lapack.h` in
        /// `/usr/include`, where no `-I` is needed.
        fn headers_available(include_paths: &[PathBuf]) -> bool {
            let implicit = [PathBuf::from("/usr/include"), PathBuf::from("/usr/local/include")];
            ["cblas.h", "lapack.h"].iter().all(|header| {
                include_paths
                    .iter()
                    .chain(&implicit)
                    .any(|dir| dir.join(header).is_file())
            })
        }

        /// Keeps only the directories that exist, so the fallback never emits
        /// `-L` flags pointing at nothing (Homebrew is under `/opt/homebrew` on
        /// Apple silicon and `/usr/local` on Intel).
        fn existing_dirs(dirs: &[&str]) -> Vec<String> {
            dirs.iter()
                .filter(|dir| std::path::Path::new(dir).is_dir())
                .map(|dir| dir.to_string())
                .collect()
        }

        fn emit_link_search(dirs: impl IntoIterator<Item = impl AsRef<str>>) {
            for dir in dirs {
                println!("cargo:rustc-link-search=native={}", dir.as_ref());
            }
        }

        fn emit_link_lib(libs: impl IntoIterator<Item = impl AsRef<str>>) {
            for lib in libs {
                println!("cargo:rustc-link-lib=dylib={}", lib.as_ref());
            }
        }
    }
}

/// A locally built reference LAPACK, for testing the native routines against it.
#[cfg(feature = "dev_lapack")]
mod dev_lapack {
    use super::SHIM;
    use std::path::{Path, PathBuf};

    /// Compiles a prefixed copy of the shim against the reference LAPACK.
    ///
    /// The shim is the crate's own, already known to drive the **column-major**
    /// `LAPACK_*` interface correctly: `lapack.h` binds straight to the Fortran
    /// routines, unlike `LAPACKE_*`, which is row-major and would silently
    /// transpose every matrix. One source is what keeps this reference
    /// trustworthy; the copy compiled here is derived from it mechanically (see
    /// [`prefixed_shim`]) so it cannot drift from the shim the crate ships with.
    ///
    /// # Layout expected
    ///
    /// A CMake build directory of LAPACK 3.12.1 with `include/` and `lib/` holding
    /// `lapack.h`, `cblas.h`, `liblapack.so` and `libcblas.so`. Point
    /// `RUSSELL_DEV_LAPACK_ROOT` at it; there is deliberately no default, since
    /// recording one would put a developer's directory layout into released source.
    pub fn compile() {
        let root = std::env::var("RUSSELL_DEV_LAPACK_ROOT").unwrap_or_else(|_| {
            panic!(
                "russell_lab: the `dev_lapack` feature needs RUSSELL_DEV_LAPACK_ROOT set to the CMake \
                 build directory of a LAPACK 3.12.1 build (the one containing include/ and lib/)"
            )
        });
        let include = Path::new(&root).join("include");
        let lib = Path::new(&root).join("lib");

        if !include.join("lapack.h").is_file() || !include.join("cblas.h").is_file() {
            panic!(
                "russell_lab: the `dev_lapack` feature needs a LAPACK build for its headers, \
                 looked for {}/lapack.h and {}/cblas.h. \
                 Set RUSSELL_DEV_LAPACK_ROOT to the CMake build directory of a LAPACK 3.12.1 build{}",
                include.display(),
                include.display(),
                build_dir_hint(&root)
            );
        }
        if !lib.join("libcblas.so").is_file() {
            panic!(
                "russell_lab: the `dev_lapack` feature needs {}/libcblas.so for the `cblas_*` symbols; \
                 set RUSSELL_DEV_LAPACK_ROOT to the CMake build directory of a LAPACK 3.12.1 build",
                lib.display()
            );
        }
        if !lib.join("liblapack.so").is_file() {
            panic!(
                "russell_lab: the `dev_lapack` feature needs {}/liblapack.so; \
                 set RUSSELL_DEV_LAPACK_ROOT to the CMake build directory of a LAPACK 3.12.1 build",
                lib.display()
            );
        }

        // A second copy of the shim can be linked alongside the normal one only if
        // its entry points have different names, so this build derives one with
        // each `c_*` renamed to `dev_lapack_c_*`.
        let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("russell_lab: OUT_DIR must be set by cargo"));
        let shim = std::fs::read_to_string(SHIM).expect("russell_lab: cannot read c_code/interface_blas.c");
        let prefixed_path = out_dir.join("interface_blas_dev_lapack.c");
        std::fs::write(&prefixed_path, prefixed_shim(&shim))
            .expect("russell_lab: cannot write the prefixed shim into OUT_DIR");
        cc::Build::new()
            .file(&prefixed_path)
            // The generated file sits in OUT_DIR, so the shim's own `constants.h`
            // has to be found through an explicit include directory.
            .include("c_code")
            .include(&include)
            .compile("dev_lapack_interface_blas");

        println!("cargo:rustc-link-search=native={}", lib.display());
        // Order matters for a static link and does no harm for a shared one: the
        // shim needs LAPACK, which needs BLAS. CBLAS is separate: the shim wraps
        // only LAPACK, so the `cblas_*` symbols it calls come from here.
        for lib_name in ["lapack", "cblas", "blas"] {
            println!("cargo:rustc-link-lib=dylib={lib_name}");
        }
        // The reference is Fortran, so its runtime is needed.
        println!("cargo:rustc-link-lib=dylib=gfortran");
        println!("cargo:rerun-if-env-changed=RUSSELL_DEV_LAPACK_ROOT");
        // The prefixed copy is derived from this file, so watching it is enough.
        println!("cargo:rerun-if-changed={SHIM}");
        println!("cargo:warning=russell_lab: `dev_lapack` feature ON -- tests compare against {root}");
    }

    /// Lists the subdirectories of `root` that look like a LAPACK CMake build, so
    /// a `RUSSELL_DEV_LAPACK_ROOT` pointing one level too high gets a useful error
    /// instead of a bare "file not found".
    ///
    /// The directories are only *reported*, never picked: a LAPACK source tree can
    /// hold several builds (`build-gfortran`, `-novec`, `-O0`, `build-ifx`) whose
    /// numerics differ, and choosing one would silently test the native routines
    /// against the wrong reference.
    fn build_dir_hint(root: &str) -> String {
        let Ok(entries) = std::fs::read_dir(root) else {
            return String::new();
        };
        let mut candidates: Vec<String> = entries
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.join("include/lapack.h").is_file() && path.join("lib/liblapack.so").is_file())
            .map(|path| format!("\n  {}", path.display()))
            .collect();
        if candidates.is_empty() {
            return String::new();
        }
        candidates.sort();
        format!("\nBuild directories found under that path:{}", candidates.concat())
    }

    /// Returns the shim with every `c_*` entry point renamed to `dev_lapack_c_*`.
    ///
    /// The rename covers identifiers only, but it is applied to the whole source
    /// because the shim calls none of its own entry points (each wrapper goes
    /// straight to a `FN_*` macro), so every `c_` token in the file is a
    /// definition. A non-identifier character before the token guards against
    /// matching inside a longer name such as `cblas_*`.
    fn prefixed_shim(source: &str) -> String {
        // The entry points are already `c_*`, so this yields `dev_lapack_c_*`.
        const PREFIX: &str = "dev_lapack_";
        let is_ident = |c: char| c.is_ascii_alphanumeric() || c == '_';
        let chars: Vec<char> = source.chars().collect();
        let mut out = String::with_capacity(source.len() + 4096);
        out.push_str("// Generated from interface_blas.c by build.rs -- do not edit.\n");
        out.push_str("// Entry points carry the `dev_lapack_c_` prefix so this copy can be\n");
        out.push_str("// linked alongside the normal shim without duplicate symbols.\n\n");
        let mut i = 0;
        while i < chars.len() {
            let starts_entry_point =
                chars[i] == 'c' && i + 1 < chars.len() && chars[i + 1] == '_' && (i == 0 || !is_ident(chars[i - 1]));
            if starts_entry_point {
                let start = i;
                while i < chars.len() && is_ident(chars[i]) {
                    i += 1;
                }
                out.push_str(PREFIX);
                out.extend(&chars[start..i]);
            } else {
                out.push(chars[i]);
                i += 1;
            }
        }
        out
    }
}
