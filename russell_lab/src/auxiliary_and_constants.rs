extern "C" {
    fn c_using_intel_mkl() -> CcBool;
    fn c_set_num_threads(n: i32);
    fn c_get_num_threads() -> i32;
}

/// Defines the vector size to decide when to use the native Rust code or BLAS
pub(crate) const MAX_DIM_FOR_NATIVE_BLAS: usize = 16;

// -------------------------------------------------------------------------------------------
// IMPORTANT: The constants below must match the corresponding C-code constants in constants.h

// Represents the type of boolean flags interchanged with the C-code
pub(crate) type CcBool = i32;

// Boolean flags
pub(crate) const C_TRUE: i32 = 1;
pub(crate) const C_FALSE: i32 = 0;

// Norm codes
pub(crate) const NORM_EUC: isize = 0;
pub(crate) const NORM_FRO: isize = 1;
pub(crate) const NORM_INF: isize = 2;
pub(crate) const NORM_MAX: isize = 3;
pub(crate) const NORM_ONE: isize = 4;
// -------------------------------------------------------------------------------------------

/// Converts usize to i32
///
/// # Panics
///
/// Will panic if usize is too large to be an i32
#[inline]
pub(crate) fn to_i32(num: usize) -> i32 {
    i32::try_from(num).unwrap()
}

/// Returns whether the code was compiled with Intel MKL or not
pub fn using_intel_mkl() -> bool {
    unsafe { c_using_intel_mkl() == C_TRUE }
}

/// Sets the number of threads allowed for the BLAS routines
pub fn set_num_threads(n: usize) {
    let n_i32 = to_i32(n);
    unsafe {
        c_set_num_threads(n_i32);
    }
}

/// Gets the number of threads available to the BLAS routines
pub fn get_num_threads() -> usize {
    unsafe { c_get_num_threads() as usize }
}

// From: /usr/include/x86_64-linux-gnu/cblas.h
// From: /opt/intel/oneapi/mkl/latest/include/mkl_cblas.h
pub(crate) const CBLAS_COL_MAJOR: i32 = 102;
pub(crate) const CBLAS_NO_TRANS: i32 = 111;
pub(crate) const CBLAS_TRANS: i32 = 112;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{get_num_threads, set_num_threads, using_intel_mkl};

    #[test]
    fn using_intel_mkl_works() {
        if cfg!(use_intel_mkl) {
            assert!(using_intel_mkl());
        } else {
            assert!(!using_intel_mkl());
        }
    }

    #[test]
    fn set_num_threads_and_get_num_threads_work() {
        assert!(get_num_threads() > 2);
        set_num_threads(1);
        assert_eq!(get_num_threads(), 1);
    }
}
