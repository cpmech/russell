use std::env;

fn main() {
    // compile MUMPS interface
    let use_local_mumps = match env::var("USE_LOCAL_MUMPS") {
        Ok(v) => v == "1" || v.to_lowercase() == "true",
        Err(_) => false,
    };
    if use_local_mumps {
        cc::Build::new()
            .file("c_code/interface_mumps.c")
            .include("/usr/local/include/mumps")
            .compile("c_code_interface_mumps");
        println!("cargo:rustc-link-search=native=/usr/local/lib/mumps");
        println!("cargo:rustc-link-lib=dylib=dmumps_cpmech");
        println!("cargo:rustc-cfg=local_mumps");
    } else {
        cc::Build::new()
            .file("c_code/interface_mumps.c")
            .compile("c_code_interface_mumps");
        println!("cargo:rustc-link-lib=dylib=dmumps_seq");
    }

    // compile UMFPACK interface
    let use_local_umfpack = match env::var("USE_LOCAL_UMFPACK") {
        Ok(v) => v == "1" || v.to_lowercase() == "true",
        Err(_) => false,
    };
    if use_local_umfpack {
        cc::Build::new()
            .file("c_code/interface_umfpack.c")
            .include("/usr/local/include/umfpack")
            .compile("c_code_interface_umfpack");
        println!("cargo:rustc-link-search=native=/usr/local/lib/umfpack");
        println!("cargo:rustc-link-lib=dylib=umfpack");
        println!("cargo:rustc-cfg=local_umfpack");
    } else {
        cc::Build::new()
            .file("c_code/interface_umfpack.c")
            .include("/usr/include/suitesparse")
            .compile("c_code_interface_umfpack");
        println!("cargo:rustc-link-lib=dylib=umfpack");
    }
}
