//! Build script for `russell_lab`.
//!
//! The default configuration needs nothing from here: BLAS/LAPACK come from the
//! `algebra_linear` crate, pure Rust, so no C code is compiled and nothing is
//! linked. That is also the only configuration available on targets with no C
//! toolchain, chiefly `wasm32-unknown-unknown` (there is no wasm libc for the shim
//! to compile against -- `inttypes.h` is missing).
//!
//! Two optional accelerators replace those kernels with an external library,
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
//! Each mode sits behind `#[cfg(feature = ...)]` on both its module and its call
//! site, so a configuration does not compile the code paths it does not use.

/// The C shim that exposes BLAS/LAPACK to the crate.
#[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
const SHIM: &str = "c_code/interface_blas.c";

fn main() {
    // The shim's prototypes and the vendor includes live in `c_code/interface.h`,
    // which `cc` does not track on its own: without this line an edit to the
    // header leaves a stale object behind and the change only shows up after a
    // `touch` of the source. (The build script has no `include_bytes!`, so this is
    // the only way cargo can know.)
    #[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
    println!("cargo:rerun-if-changed=c_code/interface.h");

    // Both accelerators define the same `c_*` entry points, so exactly one is
    // compiled: MKL when `intel_mkl` is on, OpenBLAS only when it is the only
    // accelerator. Saying so in the log is worthwhile, because a build with both
    // features is legal and the choice is otherwise invisible.
    #[cfg(all(feature = "open_blas", feature = "intel_mkl"))]
    println!("cargo:warning=russell_lab: both `open_blas` and `intel_mkl` are enabled -- linking MKL");

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
