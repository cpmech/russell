use super::*;
use russell_openblas::*;
use std::convert::TryInto;

/// Updates vector based on another vector (axpy)
///
/// ```text
/// v += alpha * u
/// ```
///
/// # Examples
///
/// ```
/// use russell_lab::*;
/// let u = Vector::from(&[10.0, 20.0, 30.0]);
/// let mut v = Vector::from(&[10.0, 20.0, 30.0]);
/// update_vector(&mut v, 0.1, &u);
/// let correct = "┌    ┐\n\
///                │ 11 │\n\
///                │ 22 │\n\
///                │ 33 │\n\
///                └    ┘";
/// assert_eq!(format!("{}", v), correct);
/// ```
///
pub fn update_vector(v: &mut Vector, alpha: f64, u: &Vector) {
    let n = v.data.len();
    if u.data.len() != n {
        panic!("the vectors must have the same dimension");
    }
    let n_i32: i32 = n.try_into().unwrap();
    daxpy(n_i32, alpha, &u.data, 1, &mut v.data, 1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use russell_chk::*;

    #[test]
    fn add_vectors_works() {
        let u = Vector::from(&[10.0, 20.0, 30.0]);
        let mut v = Vector::from(&[100.0, 200.0, 300.0]);
        update_vector(&mut v, 2.0, &u);
        let correct = &[120.0, 240.0, 360.0];
        assert_vec_approx_eq!(v.data, correct, 1e-15);
    }
}
