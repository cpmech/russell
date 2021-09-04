// paths
const INC_UMFPACK: &str = "/usr/include/suitesparse";

// libraries
const LIB_DMUMPS: &str = "dmumps_seq";
const LIB_UMFPACK: &str = "umfpack";

fn main() {
    cc::Build::new()
        .file("c_code/main.c")
        .include(INC_UMFPACK)
        .compile("c_code_main");

    println!("cargo:rustc-link-lib=dylib={}", LIB_DMUMPS);
    println!("cargo:rustc-link-lib=dylib={}", LIB_UMFPACK);
}
