fn main() {
    #[cfg(target_os = "windows")]
    {
        let msys2_prefix = std::env::var("MSYS2_PREFIX").unwrap();

        // cudss && local_sparse
        #[cfg(all(feature = "cudss", feature = "local_sparse"))]
        {
            cc::Build::new()
                .file("c_code/interface_complex_umfpack.c")
                .file("c_code/interface_complex_mumps.c")
                .file("c_code/interface_umfpack.c")
                .file("c_code/interface_mumps.c")
                .include(&format!("{}/include/mumps", msys2_prefix))
                .include(&format!("{}/include/suitesparse", msys2_prefix))
                .compile("c_code");
            println!("cargo:rustc-link-search=native={}/lib/mumps", msys2_prefix);
            println!("cargo:rustc-link-search=native={}/lib", msys2_prefix);
            println!("cargo:rustc-link-lib=dylib=umfpack");
            println!("cargo:rustc-link-lib=static=dmumps_cpmech");
            println!("cargo:rustc-link-lib=static=zmumps_cpmech");
            println!("cargo:rustc-link-lib=static=mumps_common_cpmech");
            println!("cargo:rustc-link-lib=static=mpiseq_cpmech");
            println!("cargo:rustc-link-lib=static=pord_cpmech");
            println!("cargo:rustc-link-lib=dylib=gfortran");
            println!("cargo:rustc-link-lib=dylib=gomp");
            println!("cargo:rustc-link-lib=static=metis");
        }

        // not(cudss) && local_sparse
        #[cfg(all(not(feature = "cudss"), feature = "local_sparse"))]
        {
            cc::Build::new()
                .file("c_code/interface_complex_umfpack.c")
                .file("c_code/interface_complex_mumps.c")
                .file("c_code/interface_umfpack.c")
                .file("c_code/interface_mumps.c")
                .include(&format!("{}/include/mumps", msys2_prefix))
                .include(&format!("{}/include/suitesparse", msys2_prefix))
                .compile("c_code");
            println!("cargo:rustc-link-search=native={}/lib/mumps", msys2_prefix);
            println!("cargo:rustc-link-search=native={}/lib", msys2_prefix);
            println!("cargo:rustc-link-lib=dylib=umfpack");
            println!("cargo:rustc-link-lib=static=dmumps_cpmech");
            println!("cargo:rustc-link-lib=static=zmumps_cpmech");
            println!("cargo:rustc-link-lib=static=mumps_common_cpmech");
            println!("cargo:rustc-link-lib=static=mpiseq_cpmech");
            println!("cargo:rustc-link-lib=static=pord_cpmech");
            println!("cargo:rustc-link-lib=dylib=gfortran");
            println!("cargo:rustc-link-lib=dylib=gomp");
            println!("cargo:rustc-link-lib=static=metis");
        }

        // cudss && not(local_sparse)
        #[cfg(all(feature = "cudss", not(feature = "local_sparse")))]
        {
            cc::Build::new()
                .file("c_code/interface_complex_umfpack.c")
                .file("c_code/interface_umfpack.c")
                .include(&format!("{}/include/suitesparse", msys2_prefix))
                .compile("c_code");
            println!("cargo:rustc-link-search=native={}/lib", msys2_prefix);
            println!("cargo:rustc-link-lib=dylib=umfpack");
        }

        // not(cudss) && not(local_sparse)
        #[cfg(all(not(feature = "cudss"), not(feature = "local_sparse")))]
        {
            cc::Build::new()
                .file("c_code/interface_complex_umfpack.c")
                .file("c_code/interface_umfpack.c")
                .include(&format!("{}/include/suitesparse", msys2_prefix))
                .compile("c_code");
            println!("cargo:rustc-link-search=native={}/lib", msys2_prefix);
            println!("cargo:rustc-link-lib=dylib=umfpack");
        }
    }

    #[cfg(not(target_os = "windows"))]
    {
        // Fallback search directories, used only when pkg-config cannot locate
        // the required libraries (e.g., locally compiled SuiteSparse/MUMPS via
        // zscripts/*-compile-*.bash, Homebrew, or the proprietary cuDSS
        // distribution — none of which ship .pc files). Non-existing
        // directories are filtered out so that they never end up as bogus
        // `-I`/`-L` flags on the compiler/linker command line; besides being
        // noisy, stale entries like `/usr/lib` can shadow the libraries that
        // an active Nix devShell injects via NIX_CFLAGS_COMPILE/NIX_LDFLAGS.
        let inc_dirs = existing_dirs(&[
            "/opt/homebrew/include/suitesparse",
            "/usr/include/suitesparse",
            "/usr/local/include/mumps",
            "/usr/local/include/suitesparse",
            "/usr/local/cuda/include",
            "/opt/cuda/include",
            "/opt/libcudss/include",
        ]);

        let lib_dirs = existing_dirs(&[
            "/opt/homebrew/lib",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib",
            "/usr/lib64",
            "/usr/local/lib/mumps",
            "/usr/local/lib/suitesparse",
            "/usr/local/cuda/lib64",
            "/opt/cuda/lib64",
            "/opt/libcudss/lib",
        ]);

        // cudss && local_sparse
        #[cfg(all(feature = "cudss", feature = "local_sparse"))]
        {
            let arch = detect_cuda_arch();
            let cxx = detect_cxx();
            unsafe {
                std::env::set_var("CXX", &cxx);
            }
            cc::Build::new()
                .cuda(true)
                .cudart("static")
                .flag(&format!("-arch={}", arch))
                .file("c_code/interface_complex_cudss.cu")
                .file("c_code/interface_complex_umfpack.c")
                .file("c_code/interface_complex_mumps.c")
                .file("c_code/interface_cudss.cu")
                .file("c_code/interface_umfpack.c")
                .file("c_code/interface_mumps.c")
                .includes(&inc_dirs)
                .compile("c_code");
            for d in &lib_dirs {
                println!("cargo:rustc-link-search=native={}", *d);
            }
            println!("cargo:rustc-link-lib=cudart");
            println!("cargo:rustc-link-lib=cudss");
            println!("cargo:rustc-link-lib=dylib=umfpack");
            println!("cargo:rustc-link-lib=dylib=dmumps_cpmech");
            println!("cargo:rustc-link-lib=dylib=zmumps_cpmech");
        }

        // not(cudss) && local_sparse
        #[cfg(all(not(feature = "cudss"), feature = "local_sparse"))]
        {
            cc::Build::new()
                .file("c_code/interface_complex_umfpack.c")
                .file("c_code/interface_complex_mumps.c")
                .file("c_code/interface_umfpack.c")
                .file("c_code/interface_mumps.c")
                .includes(&inc_dirs)
                .compile("c_code");
            for d in &lib_dirs {
                println!("cargo:rustc-link-search=native={}", *d);
            }
            println!("cargo:rustc-link-lib=dylib=umfpack");
            println!("cargo:rustc-link-lib=dylib=dmumps_cpmech");
            println!("cargo:rustc-link-lib=dylib=zmumps_cpmech");
        }

        // cudss && not(local_sparse)
        #[cfg(all(feature = "cudss", not(feature = "local_sparse")))]
        {
            let arch = detect_cuda_arch();
            let cxx = detect_cxx();
            unsafe {
                std::env::set_var("CXX", &cxx);
            }
            cc::Build::new()
                .cuda(true)
                .cudart("static")
                .flag(&format!("-arch={}", arch))
                .file("c_code/interface_complex_cudss.cu")
                .file("c_code/interface_complex_umfpack.c")
                .file("c_code/interface_cudss.cu")
                .file("c_code/interface_umfpack.c")
                .includes(&inc_dirs)
                .compile("c_code");
            for d in &lib_dirs {
                println!("cargo:rustc-link-search=native={}", *d);
            }
            println!("cargo:rustc-link-lib=cudart");
            println!("cargo:rustc-link-lib=cudss");
            println!("cargo:rustc-link-lib=dylib=umfpack");
        }

        // not(cudss) && not(local_sparse)
        //
        // This is "Option 1" in README.md: UMFPACK (and its SuiteSparse
        // dependencies) coming from the package manager. Query pkg-config
        // first, since that is the mechanism package managers (apt, pacman,
        // dnf, Nix, ...) use to advertise correct include/lib paths; only
        // fall back to the hardcoded search paths above when pkg-config is
        // unavailable or does not know about UMFPACK (e.g., nixpkgs'
        // `suitesparse` package currently ships no .pc files at all).
        #[cfg(all(not(feature = "cudss"), not(feature = "local_sparse")))]
        {
            let umfpack = probe_umfpack();

            let mut build = cc::Build::new();
            build
                .file("c_code/interface_complex_umfpack.c")
                .file("c_code/interface_umfpack.c");

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

            match umfpack {
                Some(lib) => {
                    // pkg-config already emitted the correct
                    // cargo:rustc-link-search/cargo:rustc-link-lib directives
                    // (see pkg_config::Config::probe).
                    let _ = lib;
                }
                None => {
                    for d in &lib_dirs {
                        println!("cargo:rustc-link-search=native={}", *d);
                    }
                    println!("cargo:rustc-link-lib=dylib=umfpack");
                }
            }
        }
    }
}

/// Returns the list of directories (from `dirs`) that actually exist on disk.
///
/// This is used to build the fallback include/lib search paths without
/// polluting the compiler/linker command line with directories that don't
/// exist, which could otherwise mask errors or interact poorly with
/// environment-injected flags (e.g., Nix's NIX_CFLAGS_COMPILE/NIX_LDFLAGS).
#[cfg(not(target_os = "windows"))]
fn existing_dirs(dirs: &[&str]) -> Vec<String> {
    dirs.iter()
        .filter(|d| std::path::Path::new(d).is_dir())
        .map(|d| d.to_string())
        .collect()
}

/// Probes `pkg-config` for the UMFPACK library (and transitively AMD, CHOLMOD,
/// SuiteSparse_config, etc., via `Requires.private`).
///
/// SuiteSparse's pkg-config files are inconsistently cased across
/// distributions/build systems (Debian, Arch, and a plain `make install` of
/// upstream SuiteSparse all use `UMFPACK.pc`, while some vendored setups use
/// lowercase `umfpack.pc`), so both spellings are tried. Returns `None` if
/// pkg-config itself is missing or if it cannot find UMFPACK under either
/// name (e.g., nixpkgs' `suitesparse` package ships no .pc files).
#[cfg(all(not(target_os = "windows"), not(feature = "cudss"), not(feature = "local_sparse")))]
fn probe_umfpack() -> Option<pkg_config::Library> {
    for name in ["UMFPACK", "umfpack"] {
        match pkg_config::Config::new().probe(name) {
            Ok(lib) => return Some(lib),
            // The library just isn't known under this name; try the other spelling.
            Err(pkg_config::Error::ProbeFailure { .. }) | Err(pkg_config::Error::Failure { .. }) => continue,
            // pkg-config itself is missing, or cross-compiling without a sysroot:
            // retrying with a different library name would not help.
            Err(_) => break,
        }
    }
    None
}

/// Returns the CXX compiler to use for cuDSS compilation.
///
/// Resolution order:
/// 1. If `GCC_VERSION` env var is set, use `g++-{version}`
/// 2. Auto-detect via `gcc -dumpversion`
/// 3. If the detected version > 15, fall back to `g++-15`
/// 4. Otherwise (version ≤ 15), use the system `g++`
#[cfg(feature = "cudss")]
fn detect_cxx() -> String {
    let version: u32 = if let Ok(ver_str) = std::env::var("GCC_VERSION") {
        ver_str.parse().unwrap_or(0)
    } else {
        let output = std::process::Command::new("gcc")
            .arg("-dumpversion")
            .output()
            .ok()
            .and_then(|o| if o.status.success() { Some(o) } else { None });
        if let Some(output) = output {
            let ver_str = String::from_utf8_lossy(&output.stdout);
            let ver_str = ver_str.trim();
            // gcc -dumpversion may return "14.2.1" — take the major version
            ver_str.split('.').next().unwrap_or("0").parse().unwrap_or(0)
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
/// 2. Auto-detected from `nvidia-smi` (maps "9.0" → "sm_90")
/// 3. Defaults to "sm_89" (Ada Lovelace / RTX 40-series)
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
                let sm = cap.replace('.', "");
                return format!("sm_{}", sm);
            }
        }
    }
    "sm_89".to_string()
}
