use super::*;
use russell_openblas::*;
use std::convert::TryInto;

/// Copy vector into another vector
///
/// ```text
/// v := u
/// ```
///
/// # Examples
///
/// ```
/// use russell_lab::*;
/// let u = Vector::from(&[1.0, 2.0, 3.0]);
/// let mut v = Vector::from(&[-1.0, -2.0, -3.0]);
/// copy_vector(&mut v, &u);
/// let correct = "┌   ┐\n\
///                │ 1 │\n\
///                │ 2 │\n\
///                │ 3 │\n\
///                └   ┘";
/// assert_eq!(format!("{}", v), correct);
/// ```
/// v := u
pub fn copy_vector(v: &mut Vector, u: &Vector) {
    let n = v.data.len();
    if u.data.len() != n {
        panic!("the vectors must have the same dimension");
    }
    let n_i32: i32 = n.try_into().unwrap();
    dcopy(n_i32, &u.data, 1, &mut v.data, 1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use russell_chk::*;

    #[test]
    fn copy_vectors_works() {
        let u = Vector::from(&[1.0, 2.0, 3.0]);
        let mut v = Vector::from(&[100.0, 200.0, 300.0]);
        copy_vector(&mut v, &u);
        let correct = &[1.0, 2.0, 3.0];
        assert_vec_approx_eq!(v.data, correct, 1e-15);
    }
}
