use russell_lab::*;

// The output looks like this:
//
// Using Intel MKL  = false
// BLAS num threads = 24
// BLAS num threads = 2
//
// or, if running with:
//
// ```bash
// cargo run --example base_auxiliary_blas --features intel_mk
// ```
//
// Then, the output looks like this:
//
// Using Intel MKL  = true
// BLAS num threads = 24
// BLAS num threads = 2

fn main() -> Result<(), StrError> {
    println!("Using Intel MKL  = {}", using_intel_mkl());
    println!("BLAS num threads = {}", get_num_threads());
    set_num_threads(2);
    println!("BLAS num threads = {}", get_num_threads());
    Ok(())
}
