use num_complex::Complex64;

extern "C" {
    // from /usr/include/x86_64-linux-gnu/cblas.h
    fn cblas_ddot(n: i32, x: *const f64, incx: i32, y: *const f64, incy: i32) -> f64;
    fn cblas_dcopy(n: i32, x: *const f64, incx: i32, y: *mut f64, incy: i32);
    fn cblas_zcopy(n: i32, x: *const Complex64, incx: i32, y: *mut Complex64, incy: i32);
    fn cblas_dscal(n: i32, alpha: f64, x: *const f64, incx: i32);
    fn cblas_zscal(n: i32, alpha: *const Complex64, x: *const Complex64, incx: i32);
    fn cblas_daxpy(n: i32, alpha: f64, x: *const f64, incx: i32, y: *mut f64, incy: i32);
    fn cblas_zaxpy(n: i32, alpha: *const Complex64, x: *const Complex64, incx: i32, y: *mut Complex64, incy: i32);
    fn cblas_dnrm2(n: i32, x: *const f64, incx: i32) -> f64;
    fn cblas_dasum(n: i32, x: *const f64, incx: i32) -> f64;
    fn cblas_idamax(n: i32, x: *const f64, incx: i32) -> i32;
}

/// Calculates the dot product of two vectors
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

/// Copies a vector into another
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

/// Copies a vector into another (complex version)
///
/// ```text
/// y := x
/// ```
///
/// # Reference
///
/// <https://www.netlib.org/lapack/explore-html/d6/d53/zcopy_8f.html>
///
#[inline]
pub fn zcopy(n: i32, x: &[Complex64], incx: i32, y: &mut [Complex64], incy: i32) {
    unsafe {
        cblas_zcopy(n, x.as_ptr(), incx, y.as_mut_ptr(), incy);
    }
}

/// Scales a vector by a constant
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

/// Scales a vector by a constant (complex version)
///
/// ```text
/// x := alpha * x
/// ```
///
/// # Reference
///
/// <http://www.netlib.org/lapack/explore-html/d2/d74/zscal_8f.html>
///
#[inline]
pub fn zscal(n: i32, alpha: Complex64, x: &mut [Complex64], incx: i32) {
    unsafe {
        cblas_zscal(n, &alpha, x.as_ptr(), incx);
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
        cblas_daxpy(n, alpha, x.as_ptr(), incx, y.as_mut_ptr(), incy);
    }
}

/// Computes constant times a vector plus a vector (Complex version)
///
/// ```text
/// y := alpha*x + y
/// ```
///
/// # Reference
///
/// <http://www.netlib.org/lapack/explore-html/d7/db2/zaxpy_8f.html>
///
#[inline]
pub fn zaxpy(n: i32, alpha: Complex64, x: &[Complex64], incx: i32, y: &mut [Complex64], incy: i32) {
    unsafe {
        cblas_zaxpy(n, &alpha, x.as_ptr(), incx, y.as_mut_ptr(), incy);
    }
}

/// Computes the sum of the absolute values (1-norm or taxicab norm)
///
/// ```text
/// ‖x‖₁ := sum_i |xᵢ|
/// ```
///
/// # Reference
///
/// <http://www.netlib.org/lapack/explore-html/de/d05/dasum_8f.html>
///
#[inline]
pub fn dasum(n: i32, x: &[f64], incx: i32) -> f64 {
    unsafe { cblas_dasum(n, x.as_ptr(), incx) }
}

/// Computes the Euclidean norm
///
/// ```text
/// ‖x‖₂ := sqrt(xᵀ ⋅ x)
/// ```
///
/// # Reference
///
/// <http://www.netlib.org/lapack/explore-html/d6/de0/dnrm2_8f90.html>
///
#[inline]
pub fn dnrm2(n: i32, x: &[f64], incx: i32) -> f64 {
    unsafe { cblas_dnrm2(n, x.as_ptr(), incx) }
}

/// Finds the index of the maximum absolute value
///
/// # Note
///
/// The index is **random** if the array contains **NAN**
///
/// # Reference
///
/// <http://www.netlib.org/lapack/explore-html/dd/de0/idamax_8f.html>
///
#[inline]
pub fn idamax(n: i32, x: &[f64], incx: i32) -> i32 {
    unsafe { cblas_idamax(n, x.as_ptr(), incx) }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{dasum, daxpy, dcopy, ddot, dnrm2, dscal, idamax, zaxpy, zcopy, zscal};
    use crate::to_i32;
    use num_complex::Complex64;
    use russell_chk::{assert_approx_eq, assert_complex_vec_approx_eq, assert_vec_approx_eq};

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
    fn zcopy_works() {
        const IGNORED: Complex64 = Complex64::new(100000.0, 200000.0);
        let x = [
            Complex64::new(20.0, 21.0),
            Complex64::new(10.0, 11.0),
            Complex64::new(-30.0, -31.0),
            IGNORED,
            IGNORED,
        ];
        let mut y = [
            Complex64::new(200.0, 201.0),
            Complex64::new(100.0, 101.0),
            Complex64::new(-300.0, -301.0),
            IGNORED,
            IGNORED,
        ];
        let (n, incx, incy) = (3, 1, 1);
        zcopy(n, &x, incx, &mut y, incy);
        assert_complex_vec_approx_eq!(
            x,
            &[
                Complex64::new(20.0, 21.0),
                Complex64::new(10.0, 11.0),
                Complex64::new(-30.0, -31.0),
                IGNORED,
                IGNORED
            ],
            1e-15
        );
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
    fn zscal_works() {
        const IGNORED: Complex64 = Complex64::new(100000.0, 200000.0);
        let alpha = Complex64::new(0.5, 0.0);
        let mut x = [
            Complex64::new(20.0, 21.0),
            Complex64::new(10.0, 11.0),
            Complex64::new(-30.0, -31.0),
            IGNORED,
            IGNORED,
        ];
        let (n, incx) = (3, 1);
        zscal(n, alpha, &mut x, incx);
        assert_complex_vec_approx_eq!(
            x,
            &[
                Complex64::new(10.0, 10.5),
                Complex64::new(5.0, 5.5),
                Complex64::new(-15.0, -15.5),
                IGNORED,
                IGNORED
            ],
            1e-15
        );

        let alpha = Complex64::new(0.5, 1.0);
        let mut x = [
            Complex64::new(20.0, 21.0),
            Complex64::new(10.0, 11.0),
            Complex64::new(-30.0, -31.0),
            IGNORED,
            IGNORED,
        ];
        let (n, incx) = (3, 1);
        zscal(n, alpha, &mut x, incx);
        assert_complex_vec_approx_eq!(
            x,
            &[
                Complex64::new(-11.0, 30.5),
                Complex64::new(-6.0, 15.5),
                Complex64::new(16.0, -45.5),
                IGNORED,
                IGNORED
            ],
            1e-15
        );
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

    #[test]
    fn zaxpy_works() {
        let x = [
            Complex64::new(20.0, 1.0),   // 0
            Complex64::new(10.0, 2.0),   // 1
            Complex64::new(30.0, 1.5),   // 2
            Complex64::new(-123.0, 0.5), // 3
            Complex64::new(-123.0, 0.5), // 4
        ];
        let mut y = [
            Complex64::new(-15.0, 1.5),  // 0
            Complex64::new(-5.0, -2.0),  // 1
            Complex64::new(-24.0, 1.0),  // 2
            Complex64::new(666.0, -0.5), // 3
            Complex64::new(666.0, 5.0),  // 4
        ];
        let (n, incx, incy) = (x.len() as i32, 1, 1);
        zaxpy(n, Complex64::new(1.0, 0.0), &x, incx, &mut y, incy);
        let x_correct = &[
            Complex64::new(20.0, 1.0),   // 0
            Complex64::new(10.0, 2.0),   // 1
            Complex64::new(30.0, 1.5),   // 2
            Complex64::new(-123.0, 0.5), // 3
            Complex64::new(-123.0, 0.5), // 4
        ];
        assert_complex_vec_approx_eq!(x, x_correct, 1e-15);
        let y_correct = &[
            Complex64::new(5.0, 2.5),   // 0
            Complex64::new(5.0, 0.0),   // 1
            Complex64::new(6.0, 2.5),   // 2
            Complex64::new(543.0, 0.0), // 3
            Complex64::new(543.0, 5.5), // 4
        ];
        assert_complex_vec_approx_eq!(y, y_correct, 1e-15);

        zaxpy(n, Complex64::new(0.5, 1.0), &x, incx, &mut y, incy);
        let y_correct = &[
            Complex64::new(14.0, 23.0),     // 0
            Complex64::new(8.0, 11.0),      // 1
            Complex64::new(19.5, 33.25),    // 2
            Complex64::new(481.0, -122.75), // 3
            Complex64::new(481.0, -117.25), // 4
        ];
        assert_complex_vec_approx_eq!(y, y_correct, 1e-15);
    }

    #[test]
    fn dasum_works() {
        let x = [-1.0, 1.0, -1.0, 1.0, 2.0, -2.0];
        let (n, incx) = (to_i32(x.len()), 1_i32);
        assert_approx_eq!(dasum(n, &x, incx), 8.0, 1e-15);
    }

    #[test]
    fn dnrm2_works() {
        let x = [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0];
        let (n, incx) = (to_i32(x.len()), 1_i32);
        assert_approx_eq!(dnrm2(n, &x, incx), 5.0, 1e-15);
    }

    #[test]
    fn idamax_works() {
        let x = [1.0, 2.0, 7.0, -8.0, -5.0, -10.0, -9.0, 10.0, 6.0];
        let (n, incx) = (to_i32(x.len()), 1_i32);
        let idx = idamax(n, &x, incx);
        assert_eq!(idx, 5);
    }
}
