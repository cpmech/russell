fn main() {
    #[cfg(target_os = "windows")]
    {
        let msys2_prefix = std::env::var("MSYS2_PREFIX").unwrap();

        #[cfg(not(feature = "local_sparse"))]
        {
            cc::Build::new()
                .file("c_code/interface_complex_klu.c")
                .file("c_code/interface_complex_umfpack.c")
                .file("c_code/interface_klu.c")
                .file("c_code/interface_umfpack.c")
                .include(&format!("{}/include/suitesparse", msys2_prefix))
                .compile("c_code");
            println!("cargo:rustc-link-search=native={}/lib", msys2_prefix);
            println!("cargo:rustc-link-lib=dylib=klu");
            println!("cargo:rustc-link-lib=dylib=umfpack");
        }

        #[cfg(feature = "local_sparse")]
        {
            cc::Build::new()
                .file("c_code/interface_complex_klu.c")
                .file("c_code/interface_complex_umfpack.c")
                .file("c_code/interface_complex_mumps.c")
                .file("c_code/interface_klu.c")
                .file("c_code/interface_umfpack.c")
                .file("c_code/interface_mumps.c")
                .include(&format!("{}/include/mumps", msys2_prefix))
                .include(&format!("{}/include/suitesparse", msys2_prefix))
                .compile("c_code");
            println!("cargo:rustc-link-search=native={}/lib/mumps", msys2_prefix);
            println!("cargo:rustc-link-search=native={}/lib", msys2_prefix);
            println!("cargo:rustc-link-lib=dylib=klu");
            println!("cargo:rustc-link-lib=dylib=umfpack");
            println!("cargo:rustc-link-lib=static=dmumps_cpmech");
            println!("cargo:rustc-link-lib=static=zmumps_cpmech");
            println!("cargo:rustc-link-lib=static=mumps_common_cpmech");
            println!("cargo:rustc-link-lib=static=mpiseq_cpmech");
            println!("cargo:rustc-link-lib=static=pord_cpmech");
            println!("cargo:rustc-link-lib=dylib=gfortran");
            println!("cargo:rustc-link-lib=dylib=gomp");
            println!("cargo:rustc-link-lib=static=metis");
        }
    }

    #[cfg(not(target_os = "windows"))]
    {
        let inc_dirs = vec![
            "/opt/homebrew/include/suitesparse",
            "/usr/include/suitesparse",
            "/usr/local/include/mumps",
            "/usr/local/include/suitesparse",
        ];

        let lib_dirs = vec![
            "/opt/homebrew/lib",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib",
            "/usr/lib64",
            "/usr/local/lib/mumps",
            "/usr/local/lib/suitesparse",
        ];

        #[cfg(not(feature = "local_sparse"))]
        {
            cc::Build::new()
                .file("c_code/interface_complex_klu.c")
                .file("c_code/interface_complex_umfpack.c")
                .file("c_code/interface_klu.c")
                .file("c_code/interface_umfpack.c")
                .includes(&inc_dirs)
                .compile("c_code");
            for d in &lib_dirs {
                println!("cargo:rustc-link-search=native={}", *d);
            }
            println!("cargo:rustc-link-lib=dylib=klu");
            println!("cargo:rustc-link-lib=dylib=umfpack");
        }

        #[cfg(feature = "local_sparse")]
        {
            cc::Build::new()
                .file("c_code/interface_complex_klu.c")
                .file("c_code/interface_complex_umfpack.c")
                .file("c_code/interface_complex_mumps.c")
                .file("c_code/interface_klu.c")
                .file("c_code/interface_umfpack.c")
                .file("c_code/interface_mumps.c")
                .includes(&inc_dirs)
                .compile("c_code");
            for d in &lib_dirs {
                println!("cargo:rustc-link-search=native={}", *d);
            }
            println!("cargo:rustc-link-lib=dylib=klu");
            println!("cargo:rustc-link-lib=dylib=umfpack");
            println!("cargo:rustc-link-lib=dylib=dmumps_cpmech");
            println!("cargo:rustc-link-lib=dylib=zmumps_cpmech");
        }
    }
}
