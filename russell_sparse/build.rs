use std::env;

fn main() {
    // compile the MUMPS interface
    let use_local_mumps = match env::var("RUSSELL_SPARSE_USE_LOCAL_MUMPS") {
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

    // compile the UMFPACK interface
    let use_local_umfpack = match env::var("RUSSELL_SPARSE_USE_LOCAL_UMFPACK") {
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

    // compile the Intel DSS interface
    let with_intel_dss = match env::var("RUSSELL_SPARSE_WITH_INTEL_DSS") {
        Ok(v) => v == "1" || v.to_lowercase() == "true",
        Err(_) => false,
    };
    if with_intel_dss {
        // Find the link libs with
        // pkg-config --libs mkl-dynamic-lp64-iomp
        cc::Build::new()
            .file("c_code/interface_intel_dss.c")
            .include("/opt/intel/oneapi/mkl/latest/include")
            .define("WITH_INTEL_DSS", None)
            .compile("c_code_interface_intel_dss");
        println!("cargo:rustc-link-search=native=/opt/intel/oneapi/mkl/latest/lib/intel64");
        println!("cargo:rustc-link-search=native=/opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin");
        println!("cargo:rustc-link-lib=mkl_intel_lp64");
        println!("cargo:rustc-link-lib=mkl_intel_thread");
        println!("cargo:rustc-link-lib=mkl_core");
        println!("cargo:rustc-link-lib=pthread");
        println!("cargo:rustc-link-lib=m");
        println!("cargo:rustc-link-lib=dl");
        println!("cargo:rustc-link-lib=iomp5");
        println!("cargo:rustc-cfg=with_intel_dss");
    } else {
        cc::Build::new()
            .file("c_code/interface_intel_dss.c")
            .compile("c_code_interface_intel_dss");
    }
}
