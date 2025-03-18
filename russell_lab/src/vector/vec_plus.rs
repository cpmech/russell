use super::Vector;
use crate::{array_plus_op, StrError};

/// Performs the addition of two vectors (without coefficients)
///
/// ```text
/// w := u + v
/// ```
///
/// # Examples
///
/// ```
/// use russell_lab::{vec_plus, Vector, StrError};
///
/// fn main() -> Result<(), StrError> {
///     let u = Vector::from(&[10.0, 20.0, 30.0, 40.0]);
///     let v = Vector::from(&[2.0, 1.5, 1.0, 0.5]);
///     let mut w = Vector::new(4);
///     vec_plus(&mut w, &u, &v)?;
///     let correct = "┌      ┐\n\
///                    │   12 │\n\
///                    │ 21.5 │\n\
///                    │   31 │\n\
///                    │ 40.5 │\n\
///                    └      ┘";
///     assert_eq!(format!("{}", w), correct);
///     Ok(())
/// }
/// ```
pub fn vec_plus(w: &mut Vector, u: &Vector, v: &Vector) -> Result<(), StrError> {
    array_plus_op(w.as_mut_data(), u.as_data(), v.as_data())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{vec_plus, Vector};
    use crate::vec_approx_eq;

    #[test]
    fn vec_plus_fail_on_wrong_dims() {
        let u_2 = Vector::new(2);
        let u_3 = Vector::new(3);
        let v_2 = Vector::new(2);
        let v_3 = Vector::new(3);
        let mut w_2 = Vector::new(2);
        assert_eq!(vec_plus(&mut w_2, &u_3, &v_2), Err("arrays are incompatible"));
        assert_eq!(vec_plus(&mut w_2, &u_2, &v_3), Err("arrays are incompatible"));
    }

    #[test]
    fn vec_plus_works() {
        const NOISE: f64 = 1234.567;
        #[rustfmt::skip]
        let u = Vector::from(&[
            1.0, 2.0,
            1.0, 2.0, 3.0, 4.0,
            1.0, 2.0, 3.0, 4.0,
            1.0, 2.0, 3.0, 4.0,
            1.0, 2.0, 3.0, 4.0,
        ]);
        #[rustfmt::skip]
        let v = Vector::from(&[
            0.5, 1.0,
            0.5, 1.0, 1.5, 2.0,
            0.5, 1.0, 1.5, 2.0,
            0.5, 1.0, 1.5, 2.0,
            0.5, 1.0, 1.5, 2.0,
        ]);
        let mut w = Vector::from(&vec![NOISE; u.dim()]);
        vec_plus(&mut w, &u, &v).unwrap();
        #[rustfmt::skip]
        let correct = &[
            1.5, 3.0,
            1.5, 3.0, 4.5, 6.0,
            1.5, 3.0, 4.5, 6.0,
            1.5, 3.0, 4.5, 6.0,
            1.5, 3.0, 4.5, 6.0,
        ];
        vec_approx_eq(&w, correct, 1e-15);
    }

    #[test]
    fn vec_plus_sizes_works() {
        const NOISE: f64 = 1234.567;
        for size in 0..13 {
            let mut u = Vector::new(size);
            let mut v = Vector::new(size);
            let mut w = Vector::from(&vec![NOISE; u.dim()]);
            let mut correct = vec![0.0; size];
            for i in 0..size {
                u[i] = i as f64;
                v[i] = i as f64;
                correct[i] = (2 * i) as f64;
            }
            vec_plus(&mut w, &u, &v).unwrap();
            vec_approx_eq(&w, &correct, 1e-15);
        }
    }
}
