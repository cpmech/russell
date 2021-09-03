// paths
const INC_DIR1: &str = "/usr/local/include/mumps";
const LIB_DIR1: &str = "/usr/local/lib/mumps";
const INC_DIR2: &str = "/usr/include/suitesparse";

// libraries
const LIB_DMUMPS: &str = "dmumps_open_seq_omp";
const LIB_UMFPACK: &str = "umfpack";

fn main() {
    cc::Build::new()
        .file("c_code/main.c")
        .include(INC_DIR1)
        .include(INC_DIR2)
        .compile("c_code_main");

    println!("cargo:rustc-link-search=native={}", LIB_DIR1);
    println!("cargo:rustc-link-lib=dylib={}", LIB_DMUMPS);
    println!("cargo:rustc-link-lib=dylib={}", LIB_UMFPACK);
}
