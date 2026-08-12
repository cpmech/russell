#[cfg(feature = "intel_mkl")]
const MKL_VERSION: &str = "latest";

// Intel MKL
#[cfg(feature = "intel_mkl")]
fn compile_blas() {
    let mkl_version = std::env::var("MKL_VERSION").unwrap_or_else(|_| MKL_VERSION.to_string());
    let mkl_root = format!("/opt/intel/oneapi/mkl/{}", mkl_version);
    let iomp_root = format!("/opt/intel/oneapi/compiler/{}", mkl_version);
    cc::Build::new()
        .file("c_code/interface_blas.c")
        .include(format!("{}/include", mkl_root))
        .define("USE_INTEL_MKL", None)
        .compile("c_code_interface_blas");
    println!("cargo:rustc-link-search=native={}/lib/intel64", mkl_root);
    println!("cargo:rustc-link-search=native={}/lib", iomp_root);
    println!("cargo:rustc-link-lib=static=mkl_intel_lp64");
    println!("cargo:rustc-link-lib=static=mkl_intel_thread");
    println!("cargo:rustc-link-lib=static=mkl_core");
    println!("cargo:rustc-link-lib=pthread");
    println!("cargo:rustc-link-lib=m");
    println!("cargo:rustc-link-lib=dl");
    println!("cargo:rustc-link-lib=iomp5");
}

// OpenBLAS
#[cfg(not(feature = "intel_mkl"))]
fn compile_blas() {
    #[cfg(target_os = "windows")]
    {
        let msys2_prefix = std::env::var("MSYS2_PREFIX").expect("MSYS2_PREFIX environment variable not set");
        let include_path = format!("{}/include/openblas", msys2_prefix);
        let lib_path = format!("{}/lib", msys2_prefix);

        cc::Build::new()
            .file("c_code/interface_blas.c")
            .include(&include_path)
            .compile("c_code_interface_blas");

        println!("cargo:rustc-link-search=native={}", lib_path);
        println!("cargo:rustc-link-lib=dylib=openblas");
    }

    #[cfg(not(target_os = "windows"))]
    {
        // Try pkg-config first (e.g., Nix's `blas`/`lapack`/`cblas` outputs,
        // or a distribution's OpenBLAS package that registers "openblas",
        // "cblas", "lapack" — naming is unfortunately not standardized
        // across distributions/build systems). Only fall back to the
        // hardcoded search paths below (used by Homebrew and some manual
        // installs) when pkg-config cannot find them.
        //
        // NOTE: this is a minimal fix, just enough to let pkg-config-based
        // environments (such as a Nix devShell) resolve the *same* BLAS/
        // LAPACK implementation that russell_sparse's SuiteSparse links
        // against (mixing an ILP64 OpenBLAS build with an LP64 one is a
        // silent ABI mismatch that segfaults at runtime).
        let cblas = pkg_config::Config::new().probe("cblas");
        let lapack = pkg_config::Config::new().probe("lapack");

        let mut build = cc::Build::new();
        build.file("c_code/interface_blas.c");

        match (&cblas, &lapack) {
            (Ok(cblas), Ok(lapack)) => {
                for inc in cblas.include_paths.iter().chain(&lapack.include_paths) {
                    build.include(inc);
                }
                build.compile("c_code_interface_blas");
                // pkg-config already emitted the correct
                // cargo:rustc-link-search/cargo:rustc-link-lib directives.
            }
            _ => {
                build.includes(&[
                    "/usr/include/openblas",
                    "/opt/homebrew/opt/lapack/include",
                    "/opt/homebrew/opt/openblas/include",
                    "/usr/local/opt/lapack/include",
                    "/usr/local/opt/openblas/include",
                ]);
                build.compile("c_code_interface_blas");
                for d in &[
                    "/opt/homebrew/opt/lapack/lib",
                    "/opt/homebrew/opt/openblas/lib",
                    "/usr/local/opt/lapack/lib",
                    "/usr/local/opt/openblas/lib",
                ] {
                    println!("cargo:rustc-link-search=native={}", *d);
                }
                println!("cargo:rustc-link-lib=dylib=openblas");
                println!("cargo:rustc-link-lib=dylib=lapack");
            }
        }
    }
}

fn main() {
    compile_blas();
}
