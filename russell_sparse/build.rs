fn main() {
    #[cfg(target_os = "windows")]
    {
        let msys2_prefix = std::env::var("MSYS2_PREFIX").unwrap();

        // cudss && local_sparse
        #[cfg(all(feature = "cudss", feature = "local_sparse"))]
        {
            cc::Build::new()
                .file("c_code/interface_complex_klu.c")
                .file("c_code/interface_complex_umfpack.c")
                .file("c_code/interface_complex_mumps.c")
                .file("c_code/interface_klu.c")
                .file("c_code/interface_umfpack.c")
                .file("c_code/interface_mumps.c")
                .include(&format!("{}/include/mumps", msys2_prefix))
                .include(&format!("{}/include/suitesparse", msys2_prefix))
                .compile("c_code");
            println!("cargo:rustc-link-search=native={}/lib/mumps", msys2_prefix);
            println!("cargo:rustc-link-search=native={}/lib", msys2_prefix);
            println!("cargo:rustc-link-lib=dylib=klu");
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
                .file("c_code/interface_complex_klu.c")
                .file("c_code/interface_complex_umfpack.c")
                .file("c_code/interface_complex_mumps.c")
                .file("c_code/interface_klu.c")
                .file("c_code/interface_umfpack.c")
                .file("c_code/interface_mumps.c")
                .include(&format!("{}/include/mumps", msys2_prefix))
                .include(&format!("{}/include/suitesparse", msys2_prefix))
                .compile("c_code");
            println!("cargo:rustc-link-search=native={}/lib/mumps", msys2_prefix);
            println!("cargo:rustc-link-search=native={}/lib", msys2_prefix);
            println!("cargo:rustc-link-lib=dylib=klu");
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
                .file("c_code/interface_complex_klu.c")
                .file("c_code/interface_complex_umfpack.c")
                .file("c_code/interface_klu.c")
                .file("c_code/interface_umfpack.c")
                .include(&format!("{}/include/suitesparse", msys2_prefix))
                .compile("c_code");
            println!("cargo:rustc-link-search=native={}/lib", msys2_prefix);
            println!("cargo:rustc-link-lib=dylib=klu");
            println!("cargo:rustc-link-lib=dylib=umfpack");
        }

        // not(cudss) && not(local_sparse)
        #[cfg(all(not(feature = "cudss"), not(feature = "local_sparse")))]
        {
            cc::Build::new()
                .file("c_code/interface_complex_klu.c")
                .file("c_code/interface_complex_umfpack.c")
                .file("c_code/interface_klu.c")
                .file("c_code/interface_umfpack.c")
                .include(&format!("{}/include/suitesparse", msys2_prefix))
                .compile("c_code");
            println!("cargo:rustc-link-search=native={}/lib", msys2_prefix);
            println!("cargo:rustc-link-lib=dylib=klu");
            println!("cargo:rustc-link-lib=dylib=umfpack");
        }
    }

    #[cfg(not(target_os = "windows"))]
    {
        let inc_dirs = vec![
            "/opt/homebrew/include/suitesparse",
            "/usr/include/suitesparse",
            "/usr/local/include/mumps",
            "/usr/local/include/suitesparse",
            "/opt/cuda/include",
            "/opt/libcudss/include",
        ];

        let lib_dirs = vec![
            "/opt/homebrew/lib",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib",
            "/usr/lib64",
            "/usr/local/lib/mumps",
            "/usr/local/lib/suitesparse",
        ];

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
                .file("c_code/interface_complex_klu.c")
                .file("c_code/interface_complex_umfpack.c")
                .file("c_code/interface_complex_mumps.c")
                .file("c_code/interface_cudss.cu")
                .file("c_code/interface_klu.c")
                .file("c_code/interface_umfpack.c")
                .file("c_code/interface_mumps.c")
                .includes(&inc_dirs)
                .compile("c_code");
            for d in &lib_dirs {
                println!("cargo:rustc-link-search=native={}", *d);
            }
            println!("cargo:rustc-link-search=native=/opt/cuda/lib64");
            println!("cargo:rustc-link-search=native=/opt/libcudss/lib");
            println!("cargo:rustc-link-lib=cudart");
            println!("cargo:rustc-link-lib=cudss");
            println!("cargo:rustc-link-lib=dylib=klu");
            println!("cargo:rustc-link-lib=dylib=umfpack");
            println!("cargo:rustc-link-lib=dylib=dmumps_cpmech");
            println!("cargo:rustc-link-lib=dylib=zmumps_cpmech");
        }

        // not(cudss) && local_sparse
        #[cfg(all(not(feature = "cudss"), feature = "local_sparse"))]
        {
            cc::Build::new()
                .file("c_code/interface_complex_klu.c")
                .file("c_code/interface_complex_umfpack.c")
                .file("c_code/interface_complex_mumps.c")
                .file("c_code/interface_klu.c")
                .file("c_code/interface_umfpack.c")
                .file("c_code/interface_mumps.c")
                .includes(&inc_dirs)
                .compile("c_code");
            for d in &lib_dirs {
                println!("cargo:rustc-link-search=native={}", *d);
            }
            println!("cargo:rustc-link-lib=dylib=klu");
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
                .file("c_code/interface_complex_klu.c")
                .file("c_code/interface_complex_umfpack.c")
                .file("c_code/interface_cudss.cu")
                .file("c_code/interface_klu.c")
                .file("c_code/interface_umfpack.c")
                .includes(&inc_dirs)
                .compile("c_code");
            for d in &lib_dirs {
                println!("cargo:rustc-link-search=native={}", *d);
            }
            println!("cargo:rustc-link-search=native=/opt/cuda/lib64");
            println!("cargo:rustc-link-search=native=/opt/libcudss/lib");
            println!("cargo:rustc-link-lib=cudart");
            println!("cargo:rustc-link-lib=cudss");
            println!("cargo:rustc-link-lib=dylib=klu");
            println!("cargo:rustc-link-lib=dylib=umfpack");
        }

        // not(cudss) && not(local_sparse)
        #[cfg(all(not(feature = "cudss"), not(feature = "local_sparse")))]
        {
            cc::Build::new()
                .file("c_code/interface_complex_klu.c")
                .file("c_code/interface_complex_umfpack.c")
                .file("c_code/interface_klu.c")
                .file("c_code/interface_umfpack.c")
                .includes(&inc_dirs)
                .compile("c_code");
            for d in &lib_dirs {
                println!("cargo:rustc-link-search=native={}", *d);
            }
            println!("cargo:rustc-link-lib=dylib=klu");
            println!("cargo:rustc-link-lib=dylib=umfpack");
        }
    }
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
