extern "C" {
    fn cblas_ddot(n: i32, x: *const f64, incx: i32, y: *const f64, incy: i32) -> f64;
}

/// Calculates the dot product of two vectors.
///
/// Uses unrolled loops for increments equal to one.
///
/// See: <http://www.netlib.org/lapack/explore-html/d5/df6/ddot_8f.html>
pub fn ddot(n: i32, x: &[f64], incx: i32, y: &[f64], incy: i32) -> f64 {
    unsafe { cblas_ddot(n, x.as_ptr(), incx, y.as_ptr(), incy) }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ddot_works() {
        let x = [20.0, 10.0, 30.0, 123.0, 123.0];
        let y = [-15.0, -5.0, -24.0, 666.0, 666.0, 666.0];
        let (n, incx, incy) = (3, 1, 1);
        assert_eq!(ddot(n, &x, incx, &y, incy), -1070.0);
    }
}
