// paths
const INC_DIR: &str = "/usr/local/include/mumps";
const LIB_DIR: &str = "/usr/local/lib/mumps";

// libraries
const LIB_DMUMPS: &str = "dmumps_open_seq_omp";

fn main() {
    cc::Build::new()
        .file("c_code/main.c")
        .include(INC_DIR)
        .compile("c_code_main");

    println!("cargo:rustc-link-search=native={}", LIB_DIR);
    println!("cargo:rustc-link-lib=dylib={}", LIB_DMUMPS);
    // println!("cargo:rustc-link-lib=dylib=openblas");
    // println!("cargo:rustc-link-lib=dylib=lapacke");
}
