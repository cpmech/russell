use crate::internal::{to_i32, CcBool, C_TRUE};

extern "C" {
    fn c_using_intel_mkl() -> CcBool;
    fn c_set_num_threads(n: i32);
    fn c_get_num_threads() -> i32;

    // Finds the index of the maximum absolute value
    // <https://www.netlib.org/lapack/explore-html/dd/de0/idamax_8f.html>
    fn cblas_idamax(n: i32, x: *const f64, incx: i32) -> i32;
}

/// Returns whether the code was compiled with Intel MKL or not
///
/// # Examples
///
/// ```
/// use russell_lab::using_intel_mkl;
///
/// println!("{}", using_intel_mkl());
/// ```
pub fn using_intel_mkl() -> bool {
    unsafe { c_using_intel_mkl() == C_TRUE }
}

/// Sets the number of threads allowed for the BLAS routines
///
/// # Examples
///
/// ```
/// use russell_lab::set_num_threads;
///
/// set_num_threads(2);
/// ```
pub fn set_num_threads(n: usize) {
    let n_i32 = to_i32(n);
    unsafe {
        c_set_num_threads(n_i32);
    }
}

/// Gets the number of threads available to the BLAS routines
///
/// # Examples
///
/// ```
/// use russell_lab::get_num_threads;
///
/// println!("{}", get_num_threads());
/// ```
pub fn get_num_threads() -> usize {
    unsafe { c_get_num_threads() as usize }
}

/// (idamax) Finds the index of the first element having maximum absolute value
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
    let n = to_i32(x.len());
    unsafe {
        let i = cblas_idamax(n, x.as_ptr(), 1);
        assert!(i >= 0);
        i as usize
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{find_index_abs_max, get_num_threads, set_num_threads, using_intel_mkl};

    #[test]
    fn using_intel_mkl_works() {
        if cfg!(feature = "intel_mkl") {
            assert!(using_intel_mkl());
        } else {
            assert!(!using_intel_mkl());
        }
    }

    #[test]
    fn set_num_threads_and_get_num_threads_work() {
        assert!(get_num_threads() > 0);
        set_num_threads(1);
        assert_eq!(get_num_threads(), 1);
    }

    #[test]
    fn find_index_abs_max_works() {
        let x = &[1.0, -2.0, -8.0, 3.0];
        assert_eq!(find_index_abs_max(x), 2);
    }
}
