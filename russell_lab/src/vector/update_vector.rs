use super::*;
use russell_openblas::*;

/// Updates vector based on another vector (axpy)
///
/// ```text
/// v += α⋅u
/// ```
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), &'static str> {
/// use russell_lab::*;
/// let u = Vector::from(&[10.0, 20.0, 30.0]);
/// let mut v = Vector::from(&[10.0, 20.0, 30.0]);
/// update_vector(&mut v, 0.1, &u)?;
/// let correct = "┌    ┐\n\
///                │ 11 │\n\
///                │ 22 │\n\
///                │ 33 │\n\
///                └    ┘";
/// assert_eq!(format!("{}", v), correct);
/// # Ok(())
/// # }
/// ```
pub fn update_vector(v: &mut Vector, alpha: f64, u: &Vector) -> Result<(), &'static str> {
    let n = v.data.len();
    if u.data.len() != n {
        return Err("vectors have wrong dimensions");
    }
    let n_i32: i32 = to_i32(n);
    daxpy(n_i32, alpha, &u.data, 1, &mut v.data, 1);
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use russell_chk::*;

    #[test]
    fn update_vector_works() -> Result<(), &'static str> {
        let u = Vector::from(&[10.0, 20.0, 30.0]);
        let mut v = Vector::from(&[100.0, 200.0, 300.0]);
        update_vector(&mut v, 2.0, &u)?;
        let correct = &[120.0, 240.0, 360.0];
        assert_vec_approx_eq!(v.data, correct, 1e-15);
        Ok(())
    }

    #[test]
    fn update_vector_fails_on_wrong_dimensions() {
        let u = Vector::new(4);
        let mut v = Vector::new(3);
        assert_eq!(
            update_vector(&mut v, 1.0, &u),
            Err("vectors have wrong dimensions")
        );
    }
}
