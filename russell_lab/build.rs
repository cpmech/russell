#[cfg(feature = "intel_mkl")]
const MKL_VERSION: &str = "2023.2.0";

#[cfg(feature = "intel_mkl")]
fn handle_intel_mkl() {
    // Intel MKL
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
    println!("cargo:rustc-cfg=use_intel_mkl");
}

#[cfg(not(feature = "intel_mkl"))]
fn handle_intel_mkl() {
    // OpenBLAS
    cc::Build::new()
        .file("c_code/interface_blas.c")
        .include("/usr/include/openblas") // for archlinux
        .compile("c_code_interface_blas");
    println!("cargo:rustc-link-lib=dylib=openblas");
    println!("cargo:rustc-link-lib=dylib=lapack");
}

fn main() {
    // math functions
    cc::Build::new().file("c_code/math_functions.c").compile("c_code");
    // BLAS functions
    handle_intel_mkl();
}
