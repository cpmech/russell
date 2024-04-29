// validate intel source command --------------------------------------------------------------------------

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

#[cfg(not(feature = "intel_mkl"))]
fn validate_intel_setvars_completed() {}

// information --------------------------------------------------------------------------------------------

// (default) Returns the directories and libraries
// returns `(inc_dirs, lib_dirs, libs)`
#[cfg(not(feature = "local_libs"))]
#[cfg(not(feature = "intel_mkl"))]
fn get_information() -> (Vec<&'static str>, Vec<&'static str>, Vec<&'static str>) {
    (
        // inc_dirs
        vec![
            "/usr/lib/x86_64-linux-gnu/openmpi/include", // Ubuntu
            "/usr/include/openmpi-x86_64",               // Rocky
            "/usr/include",                              // Ubuntu
            "/usr/include/MUMPS",                        // Rocky
            "/usr/include/suitesparse",                  // Ubuntu + Rocky
        ],
        // lib_dirs
        vec![
            "/usr/lib/x86_64-linux-gnu/openmpi/lib", // Ubuntu
            "/usr/lib64/openmpi/lib",                // Rocky
        ],
        // libs
        vec![
            "mpi",     //
            "dmumps",  //
            "zmumps",  //
            "umfpack", //
            "klu",     //
        ],
    )
}

// (local_libs) Returns the directories and libraries
// returns `(inc_dirs, lib_dirs, libs)`
#[cfg(feature = "local_libs")]
#[cfg(not(feature = "intel_mkl"))]
fn get_information() -> (Vec<&'static str>, Vec<&'static str>, Vec<&'static str>) {
    (
        // inc_dirs
        vec![
            "/usr/lib/x86_64-linux-gnu/openmpi/include", // Ubuntu
            "/usr/local/include/mumps",                  // Ubuntu
            "/usr/local/include/umfpack",                // Ubuntu
        ],
        // lib_dirs
        vec![
            "/usr/lib/x86_64-linux-gnu/openmpi/lib", // Ubuntu
            "/usr/local/lib/mumps",                  // Ubuntu
            "/usr/local/lib/umfpack",                // Ubuntu
        ],
        // libs
        vec![
            "mpi",           //
            "dmumps_cpmech", //
            "zmumps_cpmech", //
            "umfpack",       //
            "klu",           //
        ],
    )
}

// (intel_mkl) Returns the directories and libraries
// returns `(inc_dirs, lib_dirs, libs)`
#[cfg(feature = "intel_mkl")]
fn get_information() -> (Vec<&'static str>, Vec<&'static str>, Vec<&'static str>) {
    (
        // inc_dirs
        vec![
            "/opt/intel/oneapi/mpi/latest/include", // Ubuntu
            "/usr/local/include/mumps",             // Ubuntu
            "/usr/local/include/umfpack",           // Ubuntu
        ],
        // lib_dirs
        vec![
            "/opt/intel/oneapi/mpi/latest/lib", // Ubuntu
            "/usr/local/lib/mumps",             // Ubuntu
            "/usr/local/lib/umfpack",           // Ubuntu
        ],
        // libs
        vec![
            "mpi",           //
            "dmumps_cpmech", //
            "zmumps_cpmech", //
            "umfpack",       //
            "klu",           //
        ],
    )
}

// main ---------------------------------------------------------------------------------------------------

fn main() {
    // validate intel setvars
    validate_intel_setvars_completed();

    // information
    let (inc_dirs, lib_dirs, libs) = get_information();

    // MUMPS
    cc::Build::new()
        .file("c_code/interface_complex_mumps.c")
        .file("c_code/interface_mumps.c")
        .includes(&inc_dirs)
        .compile("c_code_interface_mumps");

    // UMFPACK and KLU
    cc::Build::new()
        .file("c_code/interface_complex_umfpack.c")
        .file("c_code/interface_complex_klu.c")
        .file("c_code/interface_umfpack.c")
        .file("c_code/interface_klu.c")
        .includes(&inc_dirs)
        .compile("c_code_interface_umfpack");

    // libraries
    for d in &lib_dirs {
        println!("cargo:rustc-link-search=native={}", *d);
    }
    for l in &libs {
        println!("cargo:rustc-link-lib=dylib={}", *l);
    }
}
