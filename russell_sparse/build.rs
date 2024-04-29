#[cfg(feature = "local_libs")]
fn compile_libs() {
    // local MUMPS
    cc::Build::new()
        .file("c_code/interface_complex_mumps.c")
        .file("c_code/interface_mumps.c")
        .include("/usr/local/include/mumps")
        .include("/usr/lib/x86_64-linux-gnu/openmpi/include")
        .compile("c_code_interface_mumps");
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

#[cfg(not(feature = "local_libs"))]
fn compile_libs() {
    // MUMPS
    cc::Build::new()
        .file("c_code/interface_complex_mumps.c")
        .file("c_code/interface_mumps.c")
        .include("/usr/include/mumps")
        .include("/usr/lib/x86_64-linux-gnu/openmpi/include")
        .compile("c_code_interface_mumps");
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
    compile_libs();
}
