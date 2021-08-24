#[rustfmt::skip]
extern "C" {
    fn cblas_ddot(n: i32, x: *const f64, incx: i32, y: *const f64, incy: i32) -> f64;
    fn cblas_dcopy(n: i32, x: *const f64, incx: i32, y: *mut f64, incy:i32);
    fn cblas_dscal(n: i32, alpha: f64, x: *const f64, incx: i32);
    fn cblas_daxpy(n: i32, alpha: f64, x: *const f64, incx: i32, y: *const f64, incy: i32);
}

/// Calculates the dot product of two vectors.
///
/// ```text
/// x dot y
/// ```
///
/// # Note
///
/// Uses unrolled loops for increments equal to one.
///
/// # Reference
///
/// <http://www.netlib.org/lapack/explore-html/d5/df6/ddot_8f.html>
///
#[inline]
pub fn ddot(n: i32, x: &[f64], incx: i32, y: &[f64], incy: i32) -> f64 {
    unsafe { cblas_ddot(n, x.as_ptr(), incx, y.as_ptr(), incy) }
}

/// Copies a vector into another.
///
/// ```text
/// y := x
/// ```
///
/// # Note
///
/// Uses unrolled loops for increment equal to 1.
///
/// # Reference
///
/// <https://www.netlib.org/lapack/explore-html/da/d6c/dcopy_8f.html>
///
#[inline]
pub fn dcopy(n: i32, x: &[f64], incx: i32, y: &mut [f64], incy: i32) {
    unsafe {
        cblas_dcopy(n, x.as_ptr(), incx, y.as_mut_ptr(), incy);
    }
}

/// Scales a vector by a constant.
///
/// ```text
/// x := alpha * x
/// ```
///
/// # Note
///
/// Uses unrolled loops for increment equal to 1.
///
/// # Reference
///
/// <http://www.netlib.org/lapack/explore-html/d4/dd0/dscal_8f.html>
///
#[inline]
pub fn dscal(n: i32, alpha: f64, x: &mut [f64], incx: i32) {
    unsafe {
        cblas_dscal(n, alpha, x.as_ptr(), incx);
    }
}

/// Computes constant times a vector plus a vector.
///
/// ```text
/// y := alpha*x + y
/// ```
///
/// # Reference
///
/// <http://www.netlib.org/lapack/explore-html/d9/dcd/daxpy_8f.html>
///
#[inline]
pub fn daxpy(n: i32, alpha: f64, x: &[f64], incx: i32, y: &mut [f64], incy: i32) {
    unsafe {
        cblas_daxpy(n, alpha, x.as_ptr(), incx, y.as_ptr(), incy);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use russell_chk::*;

    #[test]
    fn ddot_works() {
        const IGNORED: f64 = 100000.0;
        let x = [20.0, 10.0, 30.0, IGNORED, IGNORED];
        let y = [-15.0, -5.0, -24.0, IGNORED, IGNORED, IGNORED];
        let (n, incx, incy) = (3, 1, 1);
        assert_eq!(ddot(n, &x, incx, &y, incy), -1070.0);
    }

    #[test]
    fn dcopy_works() {
        const IGNORED: f64 = 100000.0;
        let x = [20.0, 10.0, -30.0, IGNORED, IGNORED];
        let mut y = [200.0, 100.0, -300.0, IGNORED, IGNORED];
        let (n, incx, incy) = (3, 1, 1);
        dcopy(n, &x, incx, &mut y, incy);
        assert_vec_approx_eq!(x, &[20.0, 10.0, -30.0, IGNORED, IGNORED], 1e-15);
    }

    #[test]
    fn dscal_works() {
        const IGNORED: f64 = 100000.0;
        let alpha = 0.5;
        let mut x = [20.0, 10.0, -30.0, IGNORED, IGNORED];
        let (n, incx) = (3, 1);
        dscal(n, alpha, &mut x, incx);
        assert_vec_approx_eq!(x, &[10.0, 5.0, -15.0, IGNORED, IGNORED], 1e-15);
    }

    #[test]
    fn daxpy_works() {
        const IGNORED: f64 = 100000.0;
        let alpha = 0.5;
        let x = [20.0, 10.0, 48.0, IGNORED, IGNORED];
        let mut y = [-15.0, -5.0, -24.0, IGNORED, IGNORED, IGNORED];
        let (n, incx, incy) = (3, 1, 1);
        daxpy(n, alpha, &x, incx, &mut y, incy);
        assert_vec_approx_eq!(x, &[20.0, 10.0, 48.0, IGNORED, IGNORED], 1e-15);
        assert_vec_approx_eq!(y, &[-5.0, 0.0, 0.0, IGNORED, IGNORED, IGNORED], 1e-15);
    }
}
