use super::Vector;
use crate::to_i32;

extern "C" {
    // Calculates the dot product of two vectors
    // <https://www.netlib.org/lapack/explore-html/d5/df6/ddot_8f.html>
    fn cblas_ddot(n: i32, x: *const f64, incx: i32, y: *const f64, incy: i32) -> f64;
}

/// (ddot) Performs the inner (dot) product between two vectors resulting in a scalar value
///
/// ```text
///  s := u dot v
/// ```
///
/// See also: <https://www.netlib.org/lapack/explore-html/d5/df6/ddot_8f.html>
///
/// # Note
///
/// The lengths of both vectors may be different; the smallest length will be selected.
///
/// # Examples
///
/// ```
/// use russell_lab::{vec_inner, Vector};
/// let u = Vector::from(&[1.0, 2.0, 3.0]);
/// let v = Vector::from(&[5.0, -2.0, 0.0, 1.0]);
/// let s = vec_inner(&u, &v);
/// assert_eq!(s, 1.0);
/// ```
pub fn vec_inner(u: &Vector, v: &Vector) -> f64 {
    let n = if u.dim() < v.dim() { u.dim() } else { v.dim() };
    let n_i32 = to_i32(n);
    unsafe { cblas_ddot(n_i32, u.as_data().as_ptr(), 1, v.as_data().as_ptr(), 1) }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{vec_inner, Vector};

    #[test]
    fn vec_inner_works() {
        const IGNORED: f64 = 100000.0;
        let x = Vector::from(&[20.0, 10.0, 30.0, IGNORED]);
        let y = Vector::from(&[-15.0, -5.0, -24.0]);
        assert_eq!(vec_inner(&x, &y), -1070.0);
    }

    #[test]
    fn vec_inner_alt_works() {
        const IGNORED: f64 = 100000.0;
        let x = Vector::from(&[-15.0, -5.0, -24.0]);
        let y = Vector::from(&[20.0, 10.0, 30.0, IGNORED]);
        assert_eq!(vec_inner(&x, &y), -1070.0);
    }
}
