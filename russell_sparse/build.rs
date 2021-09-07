use std::env;

fn main() {
    let use_local_mumps = match env::var("USE_LOCAL_MUMPS") {
        Ok(v) => v == "1" || v.to_lowercase() == "true",
        Err(_) => false,
    };

    if use_local_mumps {
        cc::Build::new()
            .file("c_code/main.c")
            .include("/usr/include/suitesparse")
            .include("/usr/local/include/mumps")
            .compile("c_code_main");

        println!("cargo:rustc-link-search=native=/usr/local/lib/mumps");
        println!("cargo:rustc-link-lib=dylib=dmumps_open_seq_omp");
        println!("cargo:rustc-link-lib=dylib=umfpack");
    } else {
        cc::Build::new()
            .file("c_code/main.c")
            .include("/usr/include/suitesparse")
            .compile("c_code_main");

        println!("cargo:rustc-link-lib=dylib=dmumps_seq");
        println!("cargo:rustc-link-lib=dylib=umfpack");
    }
}
