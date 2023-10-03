use std::env;

fn main() {
    // math functions
    cc::Build::new().file("c_code/math_functions.c").compile("c_code");

    // option
    let use_intel_mkl = match env::var("RUSSELL_OPENBLAS_USE_INTEL_MKL") {
        Ok(v) => v == "1" || v.to_lowercase() == "true",
        Err(_) => false,
    };

    if use_intel_mkl {
        // Intel MKL
        cc::Build::new()
            .file("c_code/interface_blas.c")
            .include("/opt/intel/oneapi/mkl/latest/include")
            .define("USE_INTEL_MKL", None)
            .compile("c_code_interface_blas");
        println!("cargo:rustc-link-search=native=/opt/intel/oneapi/mkl/latest/lib/intel64");
        println!("cargo:rustc-link-search=native=/opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin");
        println!("cargo:rustc-link-lib=mkl_intel_lp64");
        println!("cargo:rustc-link-lib=mkl_intel_thread");
        println!("cargo:rustc-link-lib=mkl_core");
        println!("cargo:rustc-link-lib=pthread");
        println!("cargo:rustc-link-lib=m");
        println!("cargo:rustc-link-lib=dl");
        println!("cargo:rustc-link-lib=iomp5");
        println!("cargo:rustc-cfg=use_intel_mkl");
    } else {
        // OpenBLAS
        cc::Build::new()
            .file("c_code/interface_blas.c")
            .include("/opt/intel/oneapi/mkl/latest/include")
            .compile("c_code_interface_blas");
        println!("cargo:rustc-link-lib=dylib=openblas");
        println!("cargo:rustc-link-lib=dylib=lapacke");
    }
}
