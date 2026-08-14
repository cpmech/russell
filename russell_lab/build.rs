//! This build script compiles and links `c_code/interface_blas.c`, the C shim
//! that exposes BLAS/LAPACK to `russell_lab`.
//!
//! Two backends are supported, selected at compile time via the `intel_mkl`
//! feature:
//!
//! * `intel_mkl` -- link against Intel oneAPI MKL (statically).
//! * OpenBLAS (default) -- resolved via pkg-config when possible, with a
//!   fallback to hardcoded paths (mainly for Homebrew, where the `.pc` files
//!   are keg-only).

/// Default MKL version used when the `MKL_VERSION` environment variable is unset.
#[cfg(feature = "intel_mkl")]
const MKL_VERSION: &str = "latest";

/// Compiles and links against Intel oneAPI MKL.
#[cfg(feature = "intel_mkl")]
fn compile_blas() {
    // Allow overriding the MKL/compiler version via the environment (e.g., "2024.2.0").
    let version = std::env::var("MKL_VERSION").unwrap_or_else(|_| MKL_VERSION.to_string());
    let mkl_root = format!("/opt/intel/oneapi/mkl/{}", version);
    let compiler_root = format!("/opt/intel/oneapi/compiler/{}", version);

    cc::Build::new()
        .file("c_code/interface_blas.c")
        .include(format!("{}/include", mkl_root))
        .define("USE_INTEL_MKL", None)
        .compile("c_code_interface_blas");

    // MKL's own library dir, the compiler's library dir (for the OpenMP runtime
    // `libiomp5`), and a few system libraries.
    println!("cargo:rustc-link-search=native={}/lib/intel64", mkl_root);
    println!("cargo:rustc-link-search=native={}/lib", compiler_root);
    println!("cargo:rustc-link-lib=static=mkl_intel_lp64");
    println!("cargo:rustc-link-lib=static=mkl_intel_thread");
    println!("cargo:rustc-link-lib=static=mkl_core");
    println!("cargo:rustc-link-lib=pthread");
    println!("cargo:rustc-link-lib=m");
    println!("cargo:rustc-link-lib=dl");
    println!("cargo:rustc-link-lib=iomp5");
}

/// Compiles and links against OpenBLAS (and LAPACK).
#[cfg(not(feature = "intel_mkl"))]
fn compile_blas() {
    #[cfg(target_os = "windows")]
    {
        // On Windows (MSYS2), everything lives under `MSYS2_PREFIX`.
        let prefix = std::env::var("MSYS2_PREFIX").expect("MSYS2_PREFIX environment variable not set");
        cc::Build::new()
            .file("c_code/interface_blas.c")
            .include(format!("{}/include/openblas", prefix))
            .compile("c_code_interface_blas");
        println!("cargo:rustc-link-search=native={}/lib", prefix);
        println!("cargo:rustc-link-lib=dylib=openblas");
    }

    #[cfg(not(target_os = "windows"))]
    {
        let mut build = cc::Build::new();
        build.file("c_code/interface_blas.c");

        match probe_blas_lapack() {
            // pkg-config knows the answer: use its include/link information.
            Some(found) => {
                for inc in &found.include_paths {
                    build.include(inc);
                }
                build.compile("c_code_interface_blas");
                for dir in &found.link_paths {
                    println!("cargo:rustc-link-search=native={}", dir.display());
                }
                for lib in &found.libs {
                    println!("cargo:rustc-link-lib=dylib={}", lib);
                }
            }
            // Fallback for setups pkg-config knows nothing about (mainly
            // Homebrew, where openblas/lapack are keg-only and their `.pc`
            // files are off the default PKG_CONFIG_PATH).
            None => {
                let inc_dirs = existing_dirs(&[
                    "/usr/include/openblas",
                    "/opt/homebrew/opt/lapack/include",
                    "/opt/homebrew/opt/openblas/include",
                    "/usr/local/opt/lapack/include",
                    "/usr/local/opt/openblas/include",
                ]);
                let lib_dirs = existing_dirs(&[
                    "/opt/homebrew/opt/lapack/lib",
                    "/opt/homebrew/opt/openblas/lib",
                    "/usr/local/opt/lapack/lib",
                    "/usr/local/opt/openblas/lib",
                ]);
                build.includes(&inc_dirs);
                build.compile("c_code_interface_blas");
                for d in &lib_dirs {
                    println!("cargo:rustc-link-search=native={}", d);
                }
                println!("cargo:rustc-link-lib=dylib=openblas");
                println!("cargo:rustc-link-lib=dylib=lapack");
            }
        }
    }
}

/// Keeps only the directories that actually exist, so the fallback does not
/// emit `-I`/`-L` flags pointing at non-existent directories (e.g., on a Mac
/// where Homebrew lives under `/opt/homebrew` rather than `/usr/local`).
#[cfg(all(not(target_os = "windows"), not(feature = "intel_mkl")))]
fn existing_dirs(dirs: &[&str]) -> Vec<String> {
    dirs.iter()
        .filter(|d| std::path::Path::new(d).is_dir())
        .map(|d| d.to_string())
        .collect()
}

/// The include/link information needed to build and link `interface_blas.c`.
#[cfg(all(not(target_os = "windows"), not(feature = "intel_mkl")))]
struct BlasLapack {
    include_paths: Vec<std::path::PathBuf>,
    link_paths: Vec<std::path::PathBuf>,
    libs: Vec<String>,
}

/// Finds BLAS/LAPACK via `pkg-config`. Returns `None` if there is no usable
/// answer, so the caller can fall back to hardcoded paths.
///
/// Tries `openblas` first (one module with both headers), then the neutral
/// `cblas` + `lapack` pair that Nix uses.
///
/// A successful `pkg-config` run is not enough: we also check that the headers
/// are really there. The `lapack` module means different things on different
/// distros — on RHEL/Rocky it is netlib LAPACK, which registers `lapack.pc`
/// but ships no `lapack.h`, so we would end up with no usable `-I` and fail
/// to compile instead of falling back.
#[cfg(all(not(target_os = "windows"), not(feature = "intel_mkl")))]
fn probe_blas_lapack() -> Option<BlasLapack> {
    // Only emit the cargo directives once we know the answer is usable.
    let probe = |name: &str| {
        pkg_config::Config::new()
            .cargo_metadata(false)
            .probe(name)
            .ok()
    };

    let candidates: Vec<Vec<pkg_config::Library>> = vec![
        probe("openblas").into_iter().collect(),
        probe("cblas")
            .into_iter()
            .chain(probe("lapack"))
            .collect::<Vec<_>>(),
    ];

    for libraries in candidates {
        if libraries.is_empty() {
            continue;
        }
        let found = BlasLapack {
            include_paths: libraries.iter().flat_map(|l| l.include_paths.clone()).collect(),
            link_paths: libraries.iter().flat_map(|l| l.link_paths.clone()).collect(),
            libs: libraries.iter().flat_map(|l| l.libs.clone()).collect(),
        };
        if headers_available(&found.include_paths) {
            return Some(found);
        }
    }
    None
}

/// Checks that both headers used by `interface_blas.c` can actually be found.
///
/// The default include dirs count too: Debian/Ubuntu put `lapack.h` directly
/// in `/usr/include`, where no `-I` is needed.
#[cfg(all(not(target_os = "windows"), not(feature = "intel_mkl")))]
fn headers_available(include_paths: &[std::path::PathBuf]) -> bool {
    let implicit = [
        std::path::PathBuf::from("/usr/include"),
        std::path::PathBuf::from("/usr/local/include"),
    ];
    ["cblas.h", "lapack.h"].iter().all(|header| {
        include_paths
            .iter()
            .chain(&implicit)
            .any(|dir| dir.join(header).is_file())
    })
}

fn main() {
    compile_blas();
}
