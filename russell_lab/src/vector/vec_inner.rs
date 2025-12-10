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
    if n == 0 {
        0.0
    } else if n == 1 {
        u[0] * v[0]
    } else if n == 2 {
        u[0] * v[0] + u[1] * v[1]
    } else if n == 3 {
        u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
    } else if n == 4 {
        u[0] * v[0] + u[1] * v[1] + u[2] * v[2] + u[3] * v[3]
    } else if n == 5 {
        u[0] * v[0] + u[1] * v[1] + u[2] * v[2] + u[3] * v[3] + u[4] * v[4]
    } else if n == 6 {
        u[0] * v[0] + u[1] * v[1] + u[2] * v[2] + u[3] * v[3] + u[4] * v[4] + u[5] * v[5]
    } else if n == 7 {
        u[0] * v[0] + u[1] * v[1] + u[2] * v[2] + u[3] * v[3] + u[4] * v[4] + u[5] * v[5] + u[6] * v[6]
    } else if n == 8 {
        u[0] * v[0] + u[1] * v[1] + u[2] * v[2] + u[3] * v[3] + u[4] * v[4] + u[5] * v[5] + u[6] * v[6] + u[7] * v[7]
    } else if n == 9 {
        u[0] * v[0]
            + u[1] * v[1]
            + u[2] * v[2]
            + u[3] * v[3]
            + u[4] * v[4]
            + u[5] * v[5]
            + u[6] * v[6]
            + u[7] * v[7]
            + u[8] * v[8]
    } else {
        let n_i32 = to_i32(n);
        unsafe { cblas_ddot(n_i32, u.as_data().as_ptr(), 1, v.as_data().as_ptr(), 1) }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{vec_inner, Vector};

    #[test]
    fn vec_inner_works_0x1() {
        let u = Vector::new(0);
        let v = Vector::new(0);
        assert_eq!(vec_inner(&u, &v), 0.0);
    }

    #[test]
    fn vec_inner_works_1x1() {
        let u = Vector::from(&[-3.0]);
        let v = Vector::from(&[4.0]);
        assert_eq!(vec_inner(&u, &v), -12.0);
    }

    #[test]
    fn vec_inner_works_2x1() {
        let u = Vector::from(&[-3.0, 1.5]);
        let v = Vector::from(&[4.0, 2.0]);
        assert_eq!(vec_inner(&u, &v), -9.0);
    }

    #[test]
    fn vec_inner_works_3x1_larger_first() {
        const IGNORED: f64 = 100000.0;
        let u = Vector::from(&[20.0, 10.0, 30.0, IGNORED]);
        let v = Vector::from(&[-15.0, -5.0, -24.0]);
        assert_eq!(vec_inner(&u, &v), -1070.0);
    }

    #[test]
    fn vec_inner_works_3x1_larger_second() {
        const IGNORED: f64 = 100000.0;
        let u = Vector::from(&[-15.0, -5.0, -24.0]);
        let v = Vector::from(&[20.0, 10.0, 30.0, IGNORED]);
        assert_eq!(vec_inner(&u, &v), -1070.0);
    }

    #[test]
    fn vec_inner_works_9x1() {
        let u = Vector::from(&[4.0, -4.0, 0.0, 0.0, -6.0, 3.0, 0.0, 1.0, 5.0]);
        let v = Vector::from(&[2.0, -2.0, 2.0, -2.0, -3.0, 1.0, 0.0, 1.5, 1.0]);
        assert_eq!(vec_inner(&u, &v), 43.5);
    }

    #[test]
    fn vec_inner_works_1_to_20() {
        for n in 1..=20 {
            let mut u_data = vec![0.0; n];
            let mut v_data = vec![0.0; n];
            for i in 0..n {
                u_data[i] = (i + 1) as f64; // 1.0, 2.0, ..., n
                v_data[i] = ((i + 1) * 10) as f64; // 10.0, 20.0, ..., n*10
            }
            let u = Vector::from(&u_data);
            let v = Vector::from(&v_data);
            let expected: f64 = (1..=n).map(|i| (i as f64) * ((i * 10) as f64)).sum();
            let result = vec_inner(&u, &v);
            assert_eq!(result, expected);
        }
    }
}
