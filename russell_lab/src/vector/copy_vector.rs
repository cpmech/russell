use super::*;
use russell_openblas::*;

/// Copies vector
///
/// ```text
/// v := u
/// ```
///
/// # Example
///
/// ```
/// # fn main() -> Result<(), &'static str> {
/// use russell_lab::*;
/// let u = Vector::from(&[1.0, 2.0, 3.0]);
/// let mut v = Vector::from(&[-1.0, -2.0, -3.0]);
/// copy_vector(&mut v, &u)?;
/// let correct = "┌   ┐\n\
///                │ 1 │\n\
///                │ 2 │\n\
///                │ 3 │\n\
///                └   ┘";
/// assert_eq!(format!("{}", v), correct);
/// # Ok(())
/// # }
/// ```
pub fn copy_vector(v: &mut Vector, u: &Vector) -> Result<(), &'static str> {
    let n = v.data.len();
    if u.data.len() != n {
        return Err("vectors have wrong dimensions");
    }
    let n_i32: i32 = to_i32(n);
    dcopy(n_i32, &u.data, 1, &mut v.data, 1);
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use russell_chk::*;

    #[test]
    fn copy_vector_works() -> Result<(), &'static str> {
        let u = Vector::from(&[1.0, 2.0, 3.0]);
        let mut v = Vector::from(&[100.0, 200.0, 300.0]);
        copy_vector(&mut v, &u)?;
        let correct = &[1.0, 2.0, 3.0];
        assert_vec_approx_eq!(v.data, correct, 1e-15);
        Ok(())
    }

    #[test]
    fn copy_vector_fails_on_wrong_dimensions() {
        let u = Vector::new(4);
        let mut v = Vector::new(3);
        assert_eq!(
            copy_vector(&mut v, &u),
            Err("vectors have wrong dimensions")
        );
    }
}
