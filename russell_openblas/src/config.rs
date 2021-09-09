extern "C" {
    fn openblas_set_num_threads(num_threads: i32);
    fn openblas_get_num_threads() -> i32;
}

/// Sets the number of threads
#[inline]
pub fn set_num_threads(num_threads: i32) {
    unsafe {
        openblas_set_num_threads(num_threads);
    }
}

/// Gets the number of threads
#[inline]
pub fn get_num_threads() -> i32 {
    unsafe { openblas_get_num_threads() }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_get_num_threads_work() {
        set_num_threads(2);
        assert_eq!(get_num_threads(), 2);
    }
}
