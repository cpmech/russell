//! Auxiliary BLAS queries: the thread count, and which backend is in use.
//!
//! This is the one place where the native and accelerated paths differ in
//! *interface* rather than only in implementation, because thread control and
//! "which library am I using" only mean something when a library is involved:
//!
//! * with `open_blas` or `intel_mkl`, the C shim answers all three questions;
//! * with the native routines there is no library, so `using_intel_mkl()` is a
//!   compile-time `false`, the thread count is fixed at one, and setting it is a
//!   no-op rather than an error — the native kernels are single-threaded, and
//!   silently ignoring the request is friendlier than failing a program that set
//!   it for the accelerated build.

#[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
use crate::to_i32;

// `CcBool` and `C_TRUE` back `using_intel_mkl`, which only the MKL shim provides.
#[cfg(feature = "intel_mkl")]
use crate::internal::{C_TRUE, CcBool};

#[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
unsafe extern "C" {
    #[cfg(feature = "intel_mkl")]
    fn c_using_intel_mkl() -> CcBool;
    fn c_set_num_threads(n: i32);
    fn c_get_num_threads() -> i32;
}

/// Returns whether the code was compiled with Intel MKL or not
///
/// With the native Rust routines this is always `false`, because there is no
/// library to ask.
///
/// # Examples
///
/// ```
/// use russell_lab::using_intel_mkl;
///
/// println!("{}", using_intel_mkl());
/// ```
pub fn using_intel_mkl() -> bool {
    #[cfg(feature = "intel_mkl")]
    {
        // Safety: the shim is linked in this configuration.
        unsafe { c_using_intel_mkl() == C_TRUE }
    }
    #[cfg(not(feature = "intel_mkl"))]
    {
        false
    }
}

/// Sets the number of threads allowed for the BLAS routines
///
/// With the native Rust routines the kernels are single-threaded, so this is
/// accepted and ignored rather than reported as an error: a caller that sets it
/// for the accelerated build should still link and run here.
///
/// # Examples
///
/// ```
/// use russell_lab::set_num_threads;
///
/// set_num_threads(2);
/// ```
pub fn set_num_threads(n: usize) {
    #[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
    {
        let n_i32 = to_i32(n);
        // Safety: the shim is linked in this configuration.
        unsafe {
            c_set_num_threads(n_i32);
        }
    }
    #[cfg(not(any(feature = "open_blas", feature = "intel_mkl")))]
    {
        let _ = n;
    }
}

/// Gets the number of threads available to the BLAS routines
///
/// With the native Rust routines this is always 1.
///
/// # Examples
///
/// ```
/// use russell_lab::get_num_threads;
///
/// println!("{}", get_num_threads());
/// ```
pub fn get_num_threads() -> usize {
    #[cfg(any(feature = "open_blas", feature = "intel_mkl"))]
    {
        // Safety: the shim is linked in this configuration.
        unsafe { c_get_num_threads() as usize }
    }
    #[cfg(not(any(feature = "open_blas", feature = "intel_mkl")))]
    {
        1
    }
}

/// (idamax) Finds the index of the first element having maximum absolute value
///
/// The index returned is 0-based, unlike BLAS's own `idamax`, which is 1-based.
///
/// # Examples
///
/// ```
/// use russell_lab::find_index_abs_max;
///
/// let u = &[1.0, 22.0, 0.0, 8.0, 3.0];
/// assert_eq!(find_index_abs_max(u), 1);
/// ```
pub fn find_index_abs_max(x: &[f64]) -> usize {
    crate::base::blas_dispatch::idamax(x)
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{find_index_abs_max, get_num_threads, set_num_threads, using_intel_mkl};

    #[test]
    fn using_intel_mkl_works() {
        #[cfg(feature = "intel_mkl")]
        assert!(using_intel_mkl());
        #[cfg(not(feature = "intel_mkl"))]
        assert!(!using_intel_mkl());
    }

    #[test]
    fn find_index_abs_max_works() {
        assert_eq!(find_index_abs_max(&[1.0, 22.0, 0.0, 8.0, 3.0]), 1);
        // The first maximum wins, as in BLAS's idamax.
        assert_eq!(find_index_abs_max(&[5.0, -5.0, 1.0]), 0);
        assert_eq!(find_index_abs_max(&[-1.0, -9.0, 3.0]), 1);
        // An empty slice must not panic.
        assert_eq!(find_index_abs_max(&[]), 0);
    }

    #[test]
    fn num_threads_round_trip() {
        // With the native routines this is fixed at 1 and setting it is a no-op;
        // with an accelerator the shim reports and stores it. Neither must panic.
        let before = get_num_threads();
        set_num_threads(1);
        assert!(get_num_threads() >= 1);
        set_num_threads(before);
    }
}
