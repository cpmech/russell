use super::Vector;
use crate::StrError;

/// Finds the maximum absolute difference between the components of two vectors
///
/// Returns (i,max_abs_diff)
///
/// ```text
/// max_abs_diff := max_i ( |uᵢ - vᵢ| )
/// ```
///
/// # Warning
///
/// This function may be slow for large vectors.
///
/// # Example
///
/// ```
/// use russell_lab::*;
///
/// fn main() -> Result<(), StrError> {
///     let u = Vector::from(&[10.0, -20.0]);
///     let v = Vector::from(&[10.0, -20.01]);
///     let (i, max_abs_diff) = vec_max_abs_diff(&u, &v)?;
///     assert_eq!(i, 1);
///     approx_eq(max_abs_diff, 0.01, 1e-14);
///     Ok(())
/// }
/// ```
pub fn vec_max_abs_diff(u: &Vector, v: &Vector) -> Result<(usize, f64), StrError> {
    let m = u.dim();
    if v.dim() != m {
        return Err("vectors are incompatible");
    }
    let (mut i_found, mut max_abs_diff) = (0, 0.0);
    for i in 0..m {
        let abs_diff = f64::abs(u[i] - v[i]);
        if abs_diff > max_abs_diff {
            i_found = i;
            max_abs_diff = abs_diff;
        }
    }
    Ok((i_found, max_abs_diff))
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{vec_max_abs_diff, Vector};

    #[test]
    fn vec_max_abs_diff_fail_on_wrong_dims() {
        let u = Vector::new(2);
        let v = Vector::new(3);
        assert_eq!(vec_max_abs_diff(&u, &v), Err("vectors are incompatible"));
    }

    #[test]
    fn vec_max_abs_diff_works() {
        let u = Vector::from(&[1.0, 2.0, 3.0, 4.0]);
        let v = Vector::from(&[2.5, 1.0, 1.5, 2.0]);
        let (i, max_abs_diff) = vec_max_abs_diff(&u, &v).unwrap();
        assert_eq!(i, 3);
        assert_eq!(max_abs_diff, 2.0);
    }
}
