use super::ComplexVector;
use crate::{to_i32, StrError, MAX_DIM_FOR_NATIVE_BLAS};
use num_complex::Complex64;

extern "C" {
    fn cblas_zaxpy(n: i32, alpha: *const Complex64, x: *const Complex64, incx: i32, y: *mut Complex64, incy: i32);
    fn cblas_zcopy(n: i32, x: *const Complex64, incx: i32, y: *mut Complex64, incy: i32);
    fn cblas_zscal(n: i32, alpha: *const Complex64, x: *mut Complex64, incx: i32);
}

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
    let n = w.dim();
    if u.dim() != n || v.dim() != n {
        return Err("vectors are incompatible");
    }
    if n == 0 {
        return Ok(());
    }
    if n > MAX_DIM_FOR_NATIVE_BLAS {
        let n_i32 = to_i32(n);
        unsafe {
            // w := v
            cblas_zcopy(n_i32, v.as_data().as_ptr(), 1, w.as_mut_data().as_mut_ptr(), 1);
            // w := beta * v
            cblas_zscal(n_i32, &beta, w.as_mut_data().as_mut_ptr(), 1);
            // w := alpha*u + w
            cblas_zaxpy(n_i32, &alpha, u.as_data().as_ptr(), 1, w.as_mut_data().as_mut_ptr(), 1);
        }
    } else {
        if n == 0 {
        } else if n == 1 {
            w[0] = alpha * u[0] + beta * v[0];
        } else if n == 2 {
            w[0] = alpha * u[0] + beta * v[0];
            w[1] = alpha * u[1] + beta * v[1];
        } else if n == 3 {
            w[0] = alpha * u[0] + beta * v[0];
            w[1] = alpha * u[1] + beta * v[1];
            w[2] = alpha * u[2] + beta * v[2];
        } else if n == 4 {
            w[0] = alpha * u[0] + beta * v[0];
            w[1] = alpha * u[1] + beta * v[1];
            w[2] = alpha * u[2] + beta * v[2];
            w[3] = alpha * u[3] + beta * v[3];
        } else if n == 5 {
            w[0] = alpha * u[0] + beta * v[0];
            w[1] = alpha * u[1] + beta * v[1];
            w[2] = alpha * u[2] + beta * v[2];
            w[3] = alpha * u[3] + beta * v[3];
            w[4] = alpha * u[4] + beta * v[4];
        } else if n == 6 {
            w[0] = alpha * u[0] + beta * v[0];
            w[1] = alpha * u[1] + beta * v[1];
            w[2] = alpha * u[2] + beta * v[2];
            w[3] = alpha * u[3] + beta * v[3];
            w[4] = alpha * u[4] + beta * v[4];
            w[5] = alpha * u[5] + beta * v[5];
        } else if n == 7 {
            w[0] = alpha * u[0] + beta * v[0];
            w[1] = alpha * u[1] + beta * v[1];
            w[2] = alpha * u[2] + beta * v[2];
            w[3] = alpha * u[3] + beta * v[3];
            w[4] = alpha * u[4] + beta * v[4];
            w[5] = alpha * u[5] + beta * v[5];
            w[6] = alpha * u[6] + beta * v[6];
        } else if n == 8 {
            w[0] = alpha * u[0] + beta * v[0];
            w[1] = alpha * u[1] + beta * v[1];
            w[2] = alpha * u[2] + beta * v[2];
            w[3] = alpha * u[3] + beta * v[3];
            w[4] = alpha * u[4] + beta * v[4];
            w[5] = alpha * u[5] + beta * v[5];
            w[6] = alpha * u[6] + beta * v[6];
            w[7] = alpha * u[7] + beta * v[7];
        } else {
            let m = n % 4;
            for i in 0..m {
                w[i] = alpha * u[i] + beta * v[i];
            }
            for i in (m..n).step_by(4) {
                w[i + 0] = alpha * u[i + 0] + beta * v[i + 0];
                w[i + 1] = alpha * u[i + 1] + beta * v[i + 1];
                w[i + 2] = alpha * u[i + 2] + beta * v[i + 2];
                w[i + 3] = alpha * u[i + 3] + beta * v[i + 3];
            }
        }
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{complex_vec_add, ComplexVector};
    use crate::MAX_DIM_FOR_NATIVE_BLAS;
    use num_complex::Complex64;
    use russell_chk::complex_vec_approx_eq;

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
            Err("vectors are incompatible")
        );
        assert_eq!(
            complex_vec_add(&mut w_2, alpha, &u_2, beta, &v_3),
            Err("vectors are incompatible")
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
