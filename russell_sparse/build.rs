#![allow(unused_mut)]

fn main() {
    let mut libs = vec!["klu", "umfpack"];

    let mut lib_dirs = vec![
        "/usr/lib/x86_64-linux-gnu/", // Debian
        "/usr/lib/",                  // Arch
        "/usr/lib64/",                // Rocky
    ];

    let mut inc_dirs = vec!["/usr/include/suitesparse/"];

    #[cfg(feature = "local_suitesparse")]
    {
        lib_dirs = vec!["/usr/local/lib/umfpack"];
        inc_dirs = vec!["/usr/local/include/umfpack"];
    }

    cc::Build::new()
        .file("c_code/interface_complex_klu.c")
        .file("c_code/interface_complex_umfpack.c")
        .file("c_code/interface_klu.c")
        .file("c_code/interface_umfpack.c")
        .includes(&inc_dirs)
        .compile("c_code_suitesparse");

    for d in &lib_dirs {
        println!("cargo:rustc-link-search=native={}", *d);
    }
    for l in &libs {
        println!("cargo:rustc-link-lib=dylib={}", *l);
    }
}
