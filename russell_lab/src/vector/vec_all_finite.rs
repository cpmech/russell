use super::Vector;
use crate::StrError;

/// Returns true if all elements of the vector are finite; i.e, not infinite, not NaN
///
/// **Note:** (from Rust's internals) There is no need to handle NaN separately because,
/// if an element is NaN, the function is_finite() returns false, exactly as desired.
///
/// # Examples
///
/// ```
/// use russell_lab::{vec_all_finite, Vector, StrError};
///
/// fn main() -> Result<(), StrError> {
///     let u = Vector::from(&[1.0, 2.0, 3.0]);
///     assert_eq!(vec_all_finite(&u, false).err(), None);
///
///     let v = Vector::from(&[1.0, 2.0, f64::NAN]);
///     assert_eq!(
///         vec_all_finite(&v, false).err(),
///         Some("an element of the vector is either infinite or NaN")
///     );
///
///     let w = Vector::from(&[1.0, 2.0, f64::INFINITY]);
///     assert_eq!(
///         vec_all_finite(&w, false).err(),
///         Some("an element of the vector is either infinite or NaN")
///     );
///
///     Ok(())
/// }
/// ```
pub fn vec_all_finite(v: &Vector, debug: bool) -> Result<(), StrError> {
    for i in 0..v.dim() {
        if !v[i].is_finite() {
            if debug {
                println!("found an invalid vector element: v[{}] = {:?}", i, v[i]);
            }
            return Err("an element of the vector is either infinite or NaN");
        }
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::vec_all_finite;
    use crate::Vector;

    #[test]
    fn vec_all_finite_works() {
        assert_eq!(vec_all_finite(&Vector::from(&[1.0, 2.0, 3.0]), false).err(), None);
        assert_eq!(
            vec_all_finite(&Vector::from(&[1.0, f64::NAN, 3.0]), true).err(),
            Some("an element of the vector is either infinite or NaN")
        );
        assert_eq!(
            vec_all_finite(&Vector::from(&[1.0, 2.0, f64::INFINITY]), true).err(),
            Some("an element of the vector is either infinite or NaN")
        );
        assert_eq!(
            vec_all_finite(&Vector::from(&[1.0, 2.0, f64::NEG_INFINITY]), true).err(),
            Some("an element of the vector is either infinite or NaN")
        );
    }
}
