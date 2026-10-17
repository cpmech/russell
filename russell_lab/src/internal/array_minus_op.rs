use crate::StrError;
use std::ops::Sub;

/// Subtracts two arrays, element by element
///
/// ```text
/// w := u - v
/// ```
///
/// Generic over the element type: Sub is the only operation involved, so the real
/// and complex arrays share this one kernel. The unrolled head (n <= 8) and the
/// four-at-a-time body are the original kernel's; they monomorphize to the same
/// code for `f64` and `Complex64`.
#[inline]
pub(crate) fn array_minus_op<T: Copy + Sub<Output = T>>(w: &mut [T], u: &[T], v: &[T]) -> Result<(), StrError> {
    let n = w.len();
    if u.len() != n || v.len() != n {
        return Err("arrays are incompatible");
    }
    if n == 0 {
        return Ok(());
    }
    if n == 1 {
        w[0] = u[0] - v[0];
    } else if n == 2 {
        w[0] = u[0] - v[0];
        w[1] = u[1] - v[1];
    } else if n == 3 {
        w[0] = u[0] - v[0];
        w[1] = u[1] - v[1];
        w[2] = u[2] - v[2];
    } else if n == 4 {
        w[0] = u[0] - v[0];
        w[1] = u[1] - v[1];
        w[2] = u[2] - v[2];
        w[3] = u[3] - v[3];
    } else if n == 5 {
        w[0] = u[0] - v[0];
        w[1] = u[1] - v[1];
        w[2] = u[2] - v[2];
        w[3] = u[3] - v[3];
        w[4] = u[4] - v[4];
    } else if n == 6 {
        w[0] = u[0] - v[0];
        w[1] = u[1] - v[1];
        w[2] = u[2] - v[2];
        w[3] = u[3] - v[3];
        w[4] = u[4] - v[4];
        w[5] = u[5] - v[5];
    } else if n == 7 {
        w[0] = u[0] - v[0];
        w[1] = u[1] - v[1];
        w[2] = u[2] - v[2];
        w[3] = u[3] - v[3];
        w[4] = u[4] - v[4];
        w[5] = u[5] - v[5];
        w[6] = u[6] - v[6];
    } else if n == 8 {
        w[0] = u[0] - v[0];
        w[1] = u[1] - v[1];
        w[2] = u[2] - v[2];
        w[3] = u[3] - v[3];
        w[4] = u[4] - v[4];
        w[5] = u[5] - v[5];
        w[6] = u[6] - v[6];
        w[7] = u[7] - v[7];
    } else {
        let m = n % 4;
        for i in 0..m {
            w[i] = u[i] - v[i];
        }
        for i in (m..n).step_by(4) {
            w[i + 0] = u[i + 0] - v[i + 0];
            w[i + 1] = u[i + 1] - v[i + 1];
            w[i + 2] = u[i + 2] - v[i + 2];
            w[i + 3] = u[i + 3] - v[i + 3];
        }
    }
    Ok(())
}
