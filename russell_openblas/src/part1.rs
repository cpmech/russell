use super::*;

#[rustfmt::skip]
extern "C" {
    fn cblas_ddot(n: i32, x: *const f64, incx: i32, y: *const f64, incy: i32) -> f64;
    fn cblas_dscal(n: i32, alpha: f64, x: *const f64, incx: i32);
    fn cblas_daxpy(n: i32, alpha: f64, x: *const f64, incx: i32, y: *const f64, incy: i32);
    fn cblas_dger( order: i32, m: i32, n: i32, alpha: f64, x: *const f64, incx: i32, y: *const f64, incy: i32, a: *mut f64, lda: i32);
}

/// Calculates the dot product of two vectors.
///
/// returns x dot y
///
/// Uses unrolled loops for increments equal to one.
///
/// See: <http://www.netlib.org/lapack/explore-html/d5/df6/ddot_8f.html>
///
pub fn ddot(n: i32, x: &[f64], incx: i32, y: &[f64], incy: i32) -> f64 {
    unsafe { cblas_ddot(n, x.as_ptr(), incx, y.as_ptr(), incy) }
}

/// Scales a vector by a constant.
///
/// x := alpha * x
///
/// Uses unrolled loops for increment equal to 1.
///
/// See: <http://www.netlib.org/lapack/explore-html/d4/dd0/dscal_8f.html>
///
pub fn dscal(n: i32, alpha: f64, x: &mut [f64], incx: i32) {
    unsafe {
        cblas_dscal(n, alpha, x.as_ptr(), incx);
    }
}

/// Computes constant times a vector plus a vector.
///
/// y += alpha*x + y
///
/// See: <http://www.netlib.org/lapack/explore-html/d9/dcd/daxpy_8f.html>
///
pub fn daxpy(n: i32, alpha: f64, x: &[f64], incx: i32, y: &mut [f64], incy: i32) {
    unsafe {
        cblas_daxpy(n, alpha, x.as_ptr(), incx, y.as_ptr(), incy);
    }
}

/// Performs the rank 1 operation (tensor product)
///
/// A := alpha*x*y**T + A
///
/// where alpha is a scalar, x is an m element vector, y is an n element
/// vector and A is an m by n matrix.
///
///  See: <http://www.netlib.org/lapack/explore-html/dc/da8/dger_8f.html>
///
pub fn dger(m: i32, n: i32, alpha: f64, x: &[f64], incx: i32, y: &[f64], incy: i32, a: &mut [f64], lda: i32) {
    unsafe {
        cblas_dger(
            CBLAS_COL_MAJOR,
            m,
            n,
            alpha,
            x.as_ptr(),
            incx,
            y.as_ptr(),
            incy,
            a.as_mut_ptr(),
            lda,
        );
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

    #[test]
    fn dger_works() {
        #[rustfmt::skip]
        let mut a = slice_to_colmajor(&[
            &[100.0, 100.0, 100.0],
            &[100.0, 100.0, 100.0],
            &[100.0, 100.0, 100.0],
            &[100.0, 100.0, 100.0],
        ]);
        let u = &[1.0, 2.0, 3.0, 4.0];
        let v = &[4.0, 3.0, 2.0];
        let m = 4; // m = nrow(a) = len(u)
        let n = 3; // n = ncol(a) = len(v)
        let lda = 4;
        let alpha = 0.5;
        dger(m, n, alpha, u, 1, v, 1, &mut a, lda);
        // a = 100 + 0.5⋅u⋅vᵀ
        let correct = slice_to_colmajor(&[
            &[102.0, 101.5, 101.0],
            &[104.0, 103.0, 102.0],
            &[106.0, 104.5, 103.0],
            &[108.0, 106.0, 104.0],
        ]);
        assert_vec_approx_eq!(a, correct, 1e-15);
    }
}
