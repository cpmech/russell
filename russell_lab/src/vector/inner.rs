use super::*;
use russell_openblas::*;
use std::convert::TryInto;

/// Performs the inner (dot) product between two vectors resulting in a scalar value
///
/// ```text
///  s := u dot v
/// ```
///
/// # Note
///
/// The lengths of both vectors may be different; the smallest length will be selected.
///
/// # Examples
///
/// ```
/// use russell_lab::*;
/// let u = Vector::from(&[1.0, 2.0, 3.0]);
/// let v = Vector::from(&[5.0, -2.0, 0.0, 1.0]);
/// let s = inner(&u, &v);
/// assert_eq!(s, 1.0);
/// ```
///
pub fn inner(u: &Vector, v: &Vector) -> f64 {
    let n = if u.data.len() < v.data.len() {
        u.data.len()
    } else {
        v.data.len()
    };
    ddot(n.try_into().unwrap(), &u.data, 1, &v.data, 1)
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inner_works() {
        const IGNORED: f64 = 100000.0;
        let x = Vector::from(&[20.0, 10.0, 30.0, IGNORED]);
        let y = Vector::from(&[-15.0, -5.0, -24.0]);
        assert_eq!(inner(&x, &y), -1070.0);
    }

    #[test]
    fn inner_alt_works() {
        const IGNORED: f64 = 100000.0;
        let x = Vector::from(&[-15.0, -5.0, -24.0]);
        let y = Vector::from(&[20.0, 10.0, 30.0, IGNORED]);
        assert_eq!(inner(&x, &y), -1070.0);
    }
}
