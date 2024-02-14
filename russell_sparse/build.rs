#[cfg(feature = "intel_mkl")]
const MKL_VERSION: &str = "2023.2.0";

#[cfg(feature = "local_libs")]
fn handle_local_libs() {
    // local MUMPS
    cc::Build::new()
        .file("c_code/interface_mumps.c")
        .include("/usr/local/include/mumps")
        .compile("c_code_interface_mumps");
    println!("cargo:rustc-link-search=native=/usr/local/lib/mumps");
    println!("cargo:rustc-link-lib=dylib=dmumps_cpmech");
    println!("cargo:rustc-cfg=local_mumps");
    // local UMFPACK
    cc::Build::new()
        .file("c_code/interface_complex_umfpack.c")
        .file("c_code/interface_umfpack.c")
        .include("/usr/local/include/umfpack")
        .compile("c_code_interface_umfpack");
    println!("cargo:rustc-link-search=native=/usr/local/lib/umfpack");
    println!("cargo:rustc-link-lib=dylib=umfpack");
    println!("cargo:rustc-cfg=local_umfpack");
}

#[cfg(not(feature = "local_libs"))]
fn handle_local_libs() {
    // MUMPS
    cc::Build::new()
        .file("c_code/interface_mumps.c")
        .compile("c_code_interface_mumps");
    println!("cargo:rustc-link-lib=dylib=dmumps_seq");
    // UMFPACK
    cc::Build::new()
        .file("c_code/interface_complex_umfpack.c")
        .file("c_code/interface_umfpack.c")
        .include("/usr/include/suitesparse")
        .compile("c_code_interface_umfpack");
    println!("cargo:rustc-link-lib=dylib=umfpack");
}

#[cfg(feature = "intel_mkl")]
fn handle_intel_mkl() {
    // Find the link libs with: pkg-config --libs mkl-dynamic-lp64-iomp
    cc::Build::new()
        .file("c_code/interface_intel_dss.c")
        .include(format!("/opt/intel/oneapi/mkl/{}/include", MKL_VERSION))
        .define("WITH_INTEL_DSS", None)
        .compile("c_code_interface_intel_dss");
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
    println!("cargo:rustc-cfg=with_intel_dss");
}

#[cfg(not(feature = "intel_mkl"))]
fn handle_intel_mkl() {
    cc::Build::new()
        .file("c_code/interface_intel_dss.c")
        .compile("c_code_interface_intel_dss");
}

fn main() {
    handle_local_libs();
    handle_intel_mkl();
}
