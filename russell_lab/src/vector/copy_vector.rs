use super::*;
use russell_openblas::*;
use std::convert::TryInto;

/// Copies vector
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
pub fn copy_vector(v: &mut Vector, u: &Vector) {
    let n = v.data.len();
    if u.data.len() != n {
        #[rustfmt::skip]
        panic!("dim of vector [u] (={}) must equal dim of vector [v] (={})", u.data.len(), n);
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
    fn copy_vector_works() {
        let u = Vector::from(&[1.0, 2.0, 3.0]);
        let mut v = Vector::from(&[100.0, 200.0, 300.0]);
        copy_vector(&mut v, &u);
        let correct = &[1.0, 2.0, 3.0];
        assert_vec_approx_eq!(v.data, correct, 1e-15);
    }

    #[test]
    #[should_panic(expected = "dim of vector [u] (=4) must equal dim of vector [v] (=3)")]
    fn copy_vector_panic_1() {
        let u = Vector::new(4);
        let mut v = Vector::new(3);
        copy_vector(&mut v, &u);
    }
}
