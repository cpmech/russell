#[cfg(feature = "intel_mkl")]
use std::env;

#[cfg(feature = "intel_mkl")]
fn validate_intel_setvars_completed() {
    let intel_setvars_completed = match env::var("SETVARS_COMPLETED") {
        Ok(v) => v == "1",
        Err(_) => false,
    };
    if !intel_setvars_completed {
        panic!("\n\nBUILD ERROR: Intel setvars.sh need to be sourced first.\nYou must execute the following command (just once):\nsource /opt/intel/oneapi/setvars.sh\n\n")
    }
}

#[cfg(not(feature = "intel_mkl"))] // OpenBLAS + OpenMPI
fn validate_intel_setvars_completed() {}

#[cfg(feature = "intel_mkl")]
fn get_mpi_include_dirs() -> Vec<String> {
    vec!["/opt/intel/oneapi/mpi/latest/include/".to_string()]
}

#[cfg(not(feature = "intel_mkl"))] // OpenBLAS + OpenMPI
fn get_mpi_include_dirs() -> Vec<String> {
    vec!["/usr/lib/x86_64-linux-gnu/openmpi/include".to_string()]
}

#[cfg(feature = "intel_mkl")]
fn get_mpi_link_dirs() -> Vec<String> {
    vec!["/opt/intel/oneapi/mpi/latest/lib/".to_string()]
}

#[cfg(not(feature = "intel_mkl"))] // OpenBLAS + OpenMPI
fn get_mpi_link_dirs() -> Vec<String> {
    vec!["/usr/lib/x86_64-linux-gnu/openmpi".to_string()]
}

#[cfg(feature = "local_libs")]
fn compile_libs() {
    // local MUMPS
    cc::Build::new()
        .file("c_code/interface_complex_mumps.c")
        .file("c_code/interface_mumps.c")
        .includes(get_mpi_include_dirs())
        .include("/usr/local/include/mumps")
        .compile("c_code_interface_mumps");
    for d in get_mpi_link_dirs() {
        println!("cargo:rustc-link-search=native={d}");
    }
    println!("cargo:rustc-link-search=native=/usr/local/lib/mumps");
    println!("cargo:rustc-link-lib=dylib=dmumps_cpmech");
    println!("cargo:rustc-link-lib=dylib=zmumps_cpmech");
    println!("cargo:rustc-link-lib=dylib=mpi");
    println!("cargo:rustc-cfg=local_mumps");
    // local UMFPACK and KLU
    cc::Build::new()
        .file("c_code/interface_complex_umfpack.c")
        .file("c_code/interface_complex_klu.c")
        .file("c_code/interface_umfpack.c")
        .file("c_code/interface_klu.c")
        .include("/usr/local/include/umfpack")
        .compile("c_code_interface_umfpack");
    println!("cargo:rustc-link-search=native=/usr/local/lib/umfpack");
    println!("cargo:rustc-link-lib=dylib=umfpack");
    println!("cargo:rustc-link-lib=dylib=klu");
    println!("cargo:rustc-cfg=local_umfpack");
}

#[cfg(not(feature = "local_libs"))] // Libraries from the distribution
fn compile_libs() {
    // MUMPS
    cc::Build::new()
        .file("c_code/interface_complex_mumps.c")
        .file("c_code/interface_mumps.c")
        .includes(get_mpi_include_dirs())
        .include("/usr/include/mumps")
        .compile("c_code_interface_mumps");
    for d in get_mpi_link_dirs() {
        println!("cargo:rustc-link-search=native={d}");
    }
    println!("cargo:rustc-link-lib=dylib=dmumps");
    println!("cargo:rustc-link-lib=dylib=zmumps");
    println!("cargo:rustc-link-lib=dylib=mpi");
    // UMFPACK and KLU
    cc::Build::new()
        .file("c_code/interface_complex_umfpack.c")
        .file("c_code/interface_complex_klu.c")
        .file("c_code/interface_umfpack.c")
        .file("c_code/interface_klu.c")
        .include("/usr/include/suitesparse")
        .compile("c_code_interface_umfpack");
    println!("cargo:rustc-link-lib=dylib=umfpack");
    println!("cargo:rustc-link-lib=dylib=klu");
}

fn main() {
    validate_intel_setvars_completed();
    compile_libs();
}
