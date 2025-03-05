use super::ComplexVector;
use crate::{plus_arrays_complex, StrError};

/// Performs the addition of two vectors (without coefficients)
///
/// ```text
/// w := u + v
/// ```
///
/// # Examples
///
/// ```
/// use russell_lab::*;
///
/// fn main() -> Result<(), StrError> {
///     let u = ComplexVector::from(&[10.0, 20.0, 30.0, 40.0]);
///     let v = ComplexVector::from(&[2.0, 1.5, 1.0, 0.5]);
///     let mut w = ComplexVector::new(4);
///     complex_vec_plus(&mut w, &u, &v)?;
///     let correct = "┌         ┐\n\
///                    │   12+0i │\n\
///                    │ 21.5+0i │\n\
///                    │   31+0i │\n\
///                    │ 40.5+0i │\n\
///                    └         ┘";
///     assert_eq!(format!("{}", w), correct);
///     Ok(())
/// }
/// ```
pub fn complex_vec_plus(w: &mut ComplexVector, u: &ComplexVector, v: &ComplexVector) -> Result<(), StrError> {
    plus_arrays_complex(w.as_mut_data(), u.as_data(), v.as_data())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::complex_vec_plus;
    use crate::{complex_vec_approx_eq, cpx, Complex64, ComplexVector};

    #[test]
    fn complex_vec_plus_fail_on_wrong_dims() {
        let u_2 = ComplexVector::new(2);
        let u_3 = ComplexVector::new(3);
        let v_2 = ComplexVector::new(2);
        let v_3 = ComplexVector::new(3);
        let mut w_2 = ComplexVector::new(2);
        assert_eq!(complex_vec_plus(&mut w_2, &u_3, &v_2), Err("arrays are incompatible"));
        assert_eq!(complex_vec_plus(&mut w_2, &u_2, &v_3), Err("arrays are incompatible"));
    }

    #[test]
    fn complex_vec_plus_works() {
        const NOISE: Complex64 = cpx!(1234.567, 3456.789);
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
        complex_vec_plus(&mut w, &u, &v).unwrap();
        #[rustfmt::skip]
        let correct = &[
            cpx!(1.5,0.0), cpx!(3.0,0.0),
            cpx!(1.5,0.0), cpx!(3.0,0.0), cpx!(4.5,0.0), cpx!(6.0,0.0),
            cpx!(1.5,0.0), cpx!(3.0,0.0), cpx!(4.5,0.0), cpx!(6.0,0.0),
            cpx!(1.5,0.0), cpx!(3.0,0.0), cpx!(4.5,0.0), cpx!(6.0,0.0),
            cpx!(1.5,0.0), cpx!(3.0,0.0), cpx!(4.5,0.0), cpx!(6.0,0.0),
        ];
        complex_vec_approx_eq(&w, correct, 1e-15);
    }

    #[test]
    fn complex_vec_plus_sizes_works() {
        const NOISE: Complex64 = cpx!(1234.567, 3456.789);
        for size in 0..13 {
            let mut u = ComplexVector::new(size);
            let mut v = ComplexVector::new(size);
            let mut w = ComplexVector::from(&vec![NOISE; u.dim()]);
            let mut correct = vec![cpx!(0.0, 0.0); size];
            for i in 0..size {
                u[i] = cpx!(i as f64, i as f64);
                v[i] = cpx!(i as f64, i as f64);
                correct[i] = cpx!((2 * i) as f64, (2 * i) as f64);
            }
            complex_vec_plus(&mut w, &u, &v).unwrap();
            complex_vec_approx_eq(&w, &correct, 1e-15);
        }
    }
}
