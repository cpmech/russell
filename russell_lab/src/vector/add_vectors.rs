use super::Vector;
use crate::StrError;
use russell_openblas::{add_vectors_native, add_vectors_oblas};

const NATIVE_VERSUS_OPENBLAS_BOUNDARY: usize = 16;

/// Performs the addition of two vectors
///
/// ```text
/// w := α⋅u + β⋅v
/// ```
///
/// # Example
///
/// ```
/// use russell_lab::{add_vectors, Vector, StrError};
///
/// fn main() -> Result<(), StrError> {
///     let u = Vector::from(&[10.0, 20.0, 30.0, 40.0]);
///     let v = Vector::from(&[2.0, 1.5, 1.0, 0.5]);
///     let mut w = Vector::new(4);
///     add_vectors(&mut w, 0.1, &u, 2.0, &v)?;
///     let correct = "┌   ┐\n\
///                    │ 5 │\n\
///                    │ 5 │\n\
///                    │ 5 │\n\
///                    │ 5 │\n\
///                    └   ┘";
///     assert_eq!(format!("{}", w), correct);
///     Ok(())
/// }
/// ```
pub fn add_vectors(w: &mut Vector, alpha: f64, u: &Vector, beta: f64, v: &Vector) -> Result<(), StrError> {
    let n = w.dim();
    if u.dim() != n || v.dim() != n {
        return Err("vectors are incompatible");
    }
    if n == 0 {
        return Ok(());
    }
    if n > NATIVE_VERSUS_OPENBLAS_BOUNDARY {
        add_vectors_oblas(w.as_mut_data(), alpha, u.as_data(), beta, v.as_data());
    } else {
        add_vectors_native(w.as_mut_data(), alpha, u.as_data(), beta, v.as_data());
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{add_vectors, Vector, NATIVE_VERSUS_OPENBLAS_BOUNDARY};
    use crate::StrError;
    use russell_chk::assert_vec_approx_eq;

    #[test]
    fn add_vectors_fail_on_wrong_dims() {
        let u_2 = Vector::new(2);
        let u_3 = Vector::new(3);
        let v_2 = Vector::new(2);
        let v_3 = Vector::new(3);
        let mut w_2 = Vector::new(2);
        assert_eq!(
            add_vectors(&mut w_2, 1.0, &u_3, 1.0, &v_2),
            Err("vectors are incompatible")
        );
        assert_eq!(
            add_vectors(&mut w_2, 1.0, &u_2, 1.0, &v_3),
            Err("vectors are incompatible")
        );
    }

    #[test]
    fn add_vectors_works() -> Result<(), StrError> {
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
        add_vectors(&mut w, 1.0, &u, -4.0, &v)?;
        #[rustfmt::skip]
        let correct = [
            -1.0, -2.0,
            -1.0, -2.0, -3.0, -4.0,
            -1.0, -2.0, -3.0, -4.0,
            -1.0, -2.0, -3.0, -4.0,
            -1.0, -2.0, -3.0, -4.0,
        ];
        assert_vec_approx_eq!(w.as_data(), correct, 1e-15);
        Ok(())
    }

    #[test]
    fn add_vectors_sizes_works() -> Result<(), StrError> {
        const NOISE: f64 = 1234.567;
        for size in 0..(NATIVE_VERSUS_OPENBLAS_BOUNDARY + 3) {
            let mut u = Vector::new(size);
            let mut v = Vector::new(size);
            let mut w = Vector::from(&vec![NOISE; u.dim()]);
            let mut correct = vec![0.0; size];
            for i in 0..size {
                u[i] = i as f64;
                v[i] = i as f64;
                correct[i] = i as f64;
            }
            add_vectors(&mut w, 0.5, &u, 0.5, &v)?;
            assert_vec_approx_eq!(w.as_data(), correct, 1e-15);
        }
        Ok(())
    }
}
