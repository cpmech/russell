//! This build script compiles the C (and CUDA) interfaces to the sparse
//! solvers and emits the corresponding link directives.
//!
//! The interface files are small C/CUDA shims that wrap the external solver
//! libraries (UMFPACK, MUMPS, and NVIDIA cuDSS) so that `russell_sparse` can
//! call them through `extern "C"`.
//!
//! Two features control what is compiled and linked:
//!
//! * `local_sparse` -- enable MUMPS (in addition to UMFPACK). MUMPS must be
//!   compiled and installed locally; its libraries are assumed to be reachable
//!   through the fallback directories below.
//! * `cudss` -- enable the NVIDIA cuDSS solver. This compiles the `.cu` files
//!   with nvcc and links against `cudart` and `cudss`. cuDSS is only built on
//!   non-Windows platforms (see [`compile_unix`]).
//!
//! UMFPACK is always compiled and linked, regardless of the features.

// ----------------------------------------------------------------------------
// Feature-independent helpers
// ----------------------------------------------------------------------------

/// Returns the C++ compiler to use when compiling CUDA sources.
///
/// Resolution order:
/// 1. `GCC_VERSION` environment variable (e.g., "15")
/// 2. The major version reported by `gcc -dumpversion`
/// 3. `g++` (if the detected version is ≤ 15) or `g++-15` (if it is > 15)
///
/// The `g++-15` fallback works around nvcc not supporting the newest GCC.
#[cfg(feature = "cudss")]
fn detect_cxx() -> String {
    let version: u32 = if let Ok(v) = std::env::var("GCC_VERSION") {
        v.parse().unwrap_or(0)
    } else {
        let output = std::process::Command::new("gcc")
            .arg("-dumpversion")
            .output()
            .ok()
            .filter(|o| o.status.success());
        if let Some(output) = output {
            let ver = String::from_utf8_lossy(&output.stdout);
            // "gcc -dumpversion" may report "14.2.1"; keep only the major version.
            ver.trim().split('.').next().unwrap_or("0").parse().unwrap_or(0)
        } else {
            0
        }
    };
    if version == 0 || version <= 15 {
        "g++".to_string()
    } else {
        "g++-15".to_string()
    }
}

/// Returns the CUDA compute architecture string (e.g., "sm_89").
///
/// Resolution order:
/// 1. `CUDSS_CUDA_ARCH` environment variable (e.g., "sm_90")
/// 2. The compute capability reported by `nvidia-smi` ("9.0" → "sm_90")
/// 3. "sm_89" (Ada Lovelace / RTX 40-series)
#[cfg(feature = "cudss")]
fn detect_cuda_arch() -> String {
    if let Ok(arch) = std::env::var("CUDSS_CUDA_ARCH") {
        if !arch.is_empty() {
            return arch;
        }
    }
    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
    {
        if output.status.success() {
            let cap = String::from_utf8_lossy(&output.stdout);
            let cap = cap.trim();
            if !cap.is_empty() {
                return format!("sm_{}", cap.replace('.', ""));
            }
        }
    }
    "sm_89".to_string()
}

/// Asks `pkg-config` for UMFPACK, which pulls in AMD, CHOLMOD and friends via
/// `Requires.private`.
///
/// Both spellings are tried because SuiteSparse is inconsistent about it:
/// Debian, Arch and upstream's own `make install` write `UMFPACK.pc`, while
/// some setups use lowercase. Returns `None` if pkg-config is missing or does
/// not know UMFPACK under either name.
#[cfg(all(not(target_os = "windows"), not(feature = "cudss"), not(feature = "local_sparse")))]
fn probe_umfpack() -> Option<pkg_config::Library> {
    for name in ["UMFPACK", "umfpack"] {
        match pkg_config::Config::new().probe(name) {
            Ok(lib) => return Some(lib),
            // The library isn't known under this name; try the other spelling.
            Err(pkg_config::Error::ProbeFailure { .. }) | Err(pkg_config::Error::Failure { .. }) => continue,
            // pkg-config is missing, or cross-compiling without a sysroot:
            // retrying with a different name would not help.
            Err(_) => break,
        }
    }
    None
}

/// Keeps only the directories that actually exist.
///
/// Apart from reducing command-line noise, this avoids entries such as
/// `/usr/lib` shadowing what a Nix devShell provides.
#[cfg(not(target_os = "windows"))]
fn existing_dirs(dirs: &[&str]) -> Vec<String> {
    dirs.iter()
        .filter(|d| std::path::Path::new(d).is_dir())
        .map(|d| d.to_string())
        .collect()
}

// ----------------------------------------------------------------------------
// Windows (MSYS2)
// ----------------------------------------------------------------------------

/// Compiles the sparse interfaces on Windows (MSYS2).
///
/// The `MSYS2_PREFIX` environment variable must point at the MSYS2 root.
///
/// **Note:** the `cudss` feature has no effect on Windows -- cuDSS is only
/// built on non-Windows platforms.
#[cfg(target_os = "windows")]
fn compile_windows() {
    let prefix = std::env::var("MSYS2_PREFIX").expect("MSYS2_PREFIX environment variable not set");

    let mut build = cc::Build::new();

    // --- source files -----------------------------------------------------
    // UMFPACK is always compiled; MUMPS only with `local_sparse`.
    build.file("c_code/interface_complex_umfpack.c");
    build.file("c_code/interface_umfpack.c");
    #[cfg(feature = "local_sparse")]
    {
        build.file("c_code/interface_complex_mumps.c");
        build.file("c_code/interface_mumps.c");
    }

    // --- include directories ---------------------------------------------
    build.include(format!("{}/include/suitesparse", prefix));
    #[cfg(feature = "local_sparse")]
    {
        build.include(format!("{}/include/mumps", prefix));
    }

    // --- compile ----------------------------------------------------------
    build.compile("c_code");

    // --- link -------------------------------------------------------------
    println!("cargo:rustc-link-search=native={}/lib", prefix);
    println!("cargo:rustc-link-lib=dylib=umfpack");
    #[cfg(feature = "local_sparse")]
    {
        println!("cargo:rustc-link-search=native={}/lib/mumps", prefix);
        // MUMPS (compiled locally with the `_cpmech` suffix) and its dependencies.
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

// ----------------------------------------------------------------------------
// Unix (Linux/macOS)
// ----------------------------------------------------------------------------

/// Fallback include directories, used when pkg-config knows nothing about the
/// libraries (locally compiled SuiteSparse/MUMPS, Homebrew, and cuDSS all ship
/// no `.pc` files). Nonexistent directories are dropped by [`existing_dirs`].
#[cfg(not(target_os = "windows"))]
const INCLUDE_DIRS: &[&str] = &[
    "/opt/homebrew/include/suitesparse",
    "/usr/include/suitesparse",
    "/usr/local/include/mumps",
    "/usr/local/include/suitesparse",
    "/usr/local/cuda/include",
    "/opt/cuda/include",
    "/opt/libcudss/include",
];

/// Fallback library directories (see [`INCLUDE_DIRS`]).
#[cfg(not(target_os = "windows"))]
const LIB_DIRS: &[&str] = &[
    "/opt/homebrew/lib",
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib",
    "/usr/lib64",
    "/usr/local/lib/mumps",
    "/usr/local/lib/suitesparse",
    "/usr/local/cuda/lib64",
    "/opt/cuda/lib64",
    "/opt/libcudss/lib",
];

/// Compiles the sparse interfaces on non-Windows platforms.
#[cfg(not(target_os = "windows"))]
fn compile_unix() {
    let inc_dirs = existing_dirs(INCLUDE_DIRS);
    let lib_dirs = existing_dirs(LIB_DIRS);

    // In the default setup ("Option 1" in README.md: UMFPACK from the package
    // manager), pkg-config is authoritative. With `local_sparse` or `cudss`,
    // the libraries are compiled/installed outside pkg-config's view, so the
    // fallback directories are always used.
    #[cfg(all(not(feature = "cudss"), not(feature = "local_sparse")))]
    let umfpack = probe_umfpack();
    #[cfg(any(feature = "cudss", feature = "local_sparse"))]
    let umfpack: Option<pkg_config::Library> = None;

    let mut build = cc::Build::new();

    // --- source files -----------------------------------------------------
    // UMFPACK: always compiled.
    build.file("c_code/interface_complex_umfpack.c");
    build.file("c_code/interface_umfpack.c");
    // MUMPS: only with `local_sparse`.
    #[cfg(feature = "local_sparse")]
    {
        build.file("c_code/interface_complex_mumps.c");
        build.file("c_code/interface_mumps.c");
    }
    // cuDSS: only with `cudss` (requires nvcc and a compatible C++ compiler).
    #[cfg(feature = "cudss")]
    {
        let arch = detect_cuda_arch();
        let cxx = detect_cxx();
        // The cc crate drives nvcc, which needs a host C++ compiler it supports.
        unsafe {
            std::env::set_var("CXX", &cxx);
        }
        build
            .cuda(true)
            .cudart("static")
            .flag(&format!("-arch={}", arch))
            .file("c_code/interface_complex_cudss.cu")
            .file("c_code/interface_cudss.cu");
    }

    // --- include directories ---------------------------------------------
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

    // --- compile ----------------------------------------------------------
    build.compile("c_code");

    // --- link -------------------------------------------------------------
    // When pkg-config found UMFPACK it already emitted the link directives;
    // otherwise (fallback, or with `local_sparse`/`cudss`) emit them manually.
    if umfpack.is_none() {
        for d in &lib_dirs {
            println!("cargo:rustc-link-search=native={}", d);
        }
        println!("cargo:rustc-link-lib=dylib=umfpack");
    }
    // MUMPS.
    #[cfg(feature = "local_sparse")]
    {
        println!("cargo:rustc-link-lib=dylib=dmumps_cpmech");
        println!("cargo:rustc-link-lib=dylib=zmumps_cpmech");
    }
    // cuDSS.
    #[cfg(feature = "cudss")]
    {
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cudss");
    }
}

// ----------------------------------------------------------------------------
// Entry point
// ----------------------------------------------------------------------------

fn main() {
    #[cfg(target_os = "windows")]
    compile_windows();

    #[cfg(not(target_os = "windows"))]
    compile_unix();
}
