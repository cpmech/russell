use super::ComplexVector;
use crate::{add_arrays_complex, StrError};
use num_complex::Complex64;

/// Performs the addition of two vectors
///
/// ```text
/// w := α⋅u + β⋅v
/// ```
///
/// # Example
///
/// ```
/// use russell_lab::{complex_vec_add, ComplexVector, StrError};
/// use num_complex::Complex64;
///
/// fn main() -> Result<(), StrError> {
///     let u = ComplexVector::from(&[10.0, 20.0, 30.0, 40.0]);
///     let v = ComplexVector::from(&[2.0, 1.5, 1.0, 0.5]);
///     let mut w = ComplexVector::new(4);
///     let alpha = Complex64::new(0.1, 0.0);
///     let beta = Complex64::new(2.0, 0.0);
///     complex_vec_add(&mut w, alpha, &u, beta, &v)?;
///     let correct = "┌      ┐\n\
///                    │ 5+0i │\n\
///                    │ 5+0i │\n\
///                    │ 5+0i │\n\
///                    │ 5+0i │\n\
///                    └      ┘";
///     assert_eq!(format!("{}", w), correct);
///     Ok(())
/// }
/// ```
pub fn complex_vec_add(
    w: &mut ComplexVector,
    alpha: Complex64,
    u: &ComplexVector,
    beta: Complex64,
    v: &ComplexVector,
) -> Result<(), StrError> {
    add_arrays_complex(w.as_mut_data(), alpha, u.as_data(), beta, v.as_data())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{complex_vec_add, ComplexVector};
    use crate::{complex_vec_approx_eq, MAX_DIM_FOR_NATIVE_BLAS};
    use num_complex::Complex64;

    #[test]
    fn complex_vec_add_fail_on_wrong_dims() {
        let u_2 = ComplexVector::new(2);
        let u_3 = ComplexVector::new(3);
        let v_2 = ComplexVector::new(2);
        let v_3 = ComplexVector::new(3);
        let mut w_2 = ComplexVector::new(2);
        let alpha = Complex64::new(1.0, 0.0);
        let beta = Complex64::new(1.0, 0.0);
        assert_eq!(
            complex_vec_add(&mut w_2, alpha, &u_3, beta, &v_2),
            Err("arrays are incompatible")
        );
        assert_eq!(
            complex_vec_add(&mut w_2, alpha, &u_2, beta, &v_3),
            Err("arrays are incompatible")
        );
    }

    #[test]
    fn complex_vec_add_works() {
        const NOISE: Complex64 = Complex64::new(1234.567, 3456.789);
        #[rustfmt::skip]
        let u = ComplexVector::from(&[
            1.0, 2.0,
            1.0, 2.0, 3.0, 4.0,
            1.0, 2.0, 3.0, 4.0,
            1.0, 2.0, 3.0, 4.0,
            1.0, 2.0, 3.0, 4.0,
        ]);
        #[rustfmt::skip]
        let v = ComplexVector::from(&[
            0.5, 1.0,
            0.5, 1.0, 1.5, 2.0,
            0.5, 1.0, 1.5, 2.0,
            0.5, 1.0, 1.5, 2.0,
            0.5, 1.0, 1.5, 2.0,
        ]);
        let mut w = ComplexVector::from(&vec![NOISE; u.dim()]);
        let alpha = Complex64::new(1.0, 0.0);
        let beta = Complex64::new(-4.0, 0.0);
        complex_vec_add(&mut w, alpha, &u, beta, &v).unwrap();
        #[rustfmt::skip]
        let correct = &[
            Complex64::new(-1.0,0.0), Complex64::new(-2.0,0.0),
            Complex64::new(-1.0,0.0), Complex64::new(-2.0,0.0), Complex64::new(-3.0,0.0), Complex64::new(-4.0,0.0),
            Complex64::new(-1.0,0.0), Complex64::new(-2.0,0.0), Complex64::new(-3.0,0.0), Complex64::new(-4.0,0.0),
            Complex64::new(-1.0,0.0), Complex64::new(-2.0,0.0), Complex64::new(-3.0,0.0), Complex64::new(-4.0,0.0),
            Complex64::new(-1.0,0.0), Complex64::new(-2.0,0.0), Complex64::new(-3.0,0.0), Complex64::new(-4.0,0.0),
        ];
        complex_vec_approx_eq(w.as_data(), correct, 1e-15);
    }

    #[test]
    fn complex_vec_add_sizes_works() {
        const NOISE: Complex64 = Complex64::new(1234.567, 3456.789);
        let alpha = Complex64::new(0.5, 0.0);
        let beta = Complex64::new(0.5, 0.0);
        for size in 0..(MAX_DIM_FOR_NATIVE_BLAS + 3) {
            let mut u = ComplexVector::new(size);
            let mut v = ComplexVector::new(size);
            let mut w = ComplexVector::from(&vec![NOISE; u.dim()]);
            let mut correct = vec![Complex64::new(0.0, 0.0); size];
            for i in 0..size {
                u[i] = Complex64::new(i as f64, i as f64);
                v[i] = Complex64::new(i as f64, i as f64);
                correct[i] = Complex64::new(i as f64, i as f64);
            }
            complex_vec_add(&mut w, alpha, &u, beta, &v).unwrap();
            complex_vec_approx_eq(w.as_data(), &correct, 1e-15);
        }
    }
}
