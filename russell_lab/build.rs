#[cfg(feature = "intel_mkl")]
const MKL_VERSION: &str = "2023.2.0";

// Intel MKL
#[cfg(feature = "intel_mkl")]
fn compile_blas() {
    cc::Build::new()
        .file("c_code/interface_blas.c")
        .include(format!("/opt/intel/oneapi/mkl/{}/include", MKL_VERSION))
        .define("USE_INTEL_MKL", None)
        .compile("c_code_interface_blas");
    println!(
        "cargo:rustc-link-search=native=/opt/intel/oneapi/mkl/{}/lib/intel64",
        MKL_VERSION
    );
    println!(
        "cargo:rustc-link-search=native=/opt/intel/oneapi/compiler/{}/linux/compiler/lib/intel64_lin",
        MKL_VERSION
    );
    println!("cargo:rustc-link-lib=mkl_intel_lp64");
    println!("cargo:rustc-link-lib=mkl_intel_thread");
    println!("cargo:rustc-link-lib=mkl_core");
    println!("cargo:rustc-link-lib=pthread");
    println!("cargo:rustc-link-lib=m");
    println!("cargo:rustc-link-lib=dl");
    println!("cargo:rustc-link-lib=iomp5");
}

// OpenBLAS
#[cfg(not(feature = "intel_mkl"))]
fn compile_blas() {
    cc::Build::new()
        .file("c_code/interface_blas.c")
        .includes(&[
            "/usr/include/openblas",              // Arch
            "/opt/homebrew/opt/lapack/include",   // macOS
            "/opt/homebrew/opt/openblas/include", // macOS
            "/usr/local/opt/lapack/include",      // macOS
            "/usr/local/opt/openblas/include",    // macOS
        ])
        .compile("c_code_interface_blas");
    for d in &[
        "/opt/homebrew/opt/lapack/lib",   // macOS
        "/opt/homebrew/opt/openblas/lib", // macOS
        "/usr/local/opt/lapack/lib",      // macOS
        "/usr/local/opt/openblas/lib",    // macOS
    ] {
        println!("cargo:rustc-link-search=native={}", *d);
    }
    println!("cargo:rustc-link-lib=dylib=openblas");
    println!("cargo:rustc-link-lib=dylib=lapack");
}

fn main() {
    compile_blas();
}
