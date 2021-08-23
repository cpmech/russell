use super::*;
use russell_openblas::*;
use std::convert::TryInto;

/// Performs the scaling of a vector
///
/// ```text
/// u := alpha * u
/// ```
///
/// # Examples
///
/// ```
/// use russell_lab::*;
/// let mut u = Vector::from(&[1.0, 2.0, 3.0]);
/// scale_vector(&mut u, 0.5);
/// let correct = "┌     ┐\n\
///                │ 0.5 │\n\
///                │   1 │\n\
///                │ 1.5 │\n\
///                └     ┘";
/// assert_eq!(format!("{}", u), correct);
/// ```
///
pub fn scale_vector(u: &mut Vector, alpha: f64) {
    let n: i32 = u.data.len().try_into().unwrap();
    dscal(n, alpha, &mut u.data, 1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use russell_chk::*;

    #[test]
    fn scale_vector_works() {
        let mut u = Vector::from(&[6.0, 9.0, 12.0]);
        scale_vector(&mut u, 1.0 / 3.0);
        let correct = &[2.0, 3.0, 4.0];
        assert_vec_approx_eq!(u.data, correct, 1e-15);
    }
}
