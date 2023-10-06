use super::{to_i32, MAX_DIM_FOR_NATIVE_BLAS};
use crate::StrError;
use num_complex::Complex64;

extern "C" {
    // real
    fn cblas_daxpy(n: i32, alpha: f64, x: *const f64, incx: i32, y: *mut f64, incy: i32);
    fn cblas_dcopy(n: i32, x: *const f64, incx: i32, y: *mut f64, incy: i32);
    fn cblas_dscal(n: i32, alpha: f64, x: *const f64, incx: i32);
    // complex
    fn cblas_zaxpy(n: i32, alpha: *const Complex64, x: *const Complex64, incx: i32, y: *mut Complex64, incy: i32);
    fn cblas_zcopy(n: i32, x: *const Complex64, incx: i32, y: *mut Complex64, incy: i32);
    fn cblas_zscal(n: i32, alpha: *const Complex64, x: *mut Complex64, incx: i32);
}

/// Adds two arrays
///
/// **Note:** This is an internal function used by `vec_add` and `mat_add`.
///
/// ```text
/// w := α⋅u + β⋅v
/// ```
#[inline]
pub(crate) fn add_arrays(w: &mut [f64], alpha: f64, u: &[f64], beta: f64, v: &[f64]) -> Result<(), StrError> {
    let n = w.len();
    if u.len() != n || v.len() != n {
        return Err("arrays are incompatible");
    }
    if n == 0 {
        return Ok(());
    }
    if n > MAX_DIM_FOR_NATIVE_BLAS {
        let n_i32 = to_i32(n);
        unsafe {
            // w := v
            cblas_dcopy(n_i32, v.as_ptr(), 1, w.as_mut_ptr(), 1);
            // w := beta * v
            cblas_dscal(n_i32, beta, w.as_mut_ptr(), 1);
            // w := alpha*u + w
            cblas_daxpy(n_i32, alpha, u.as_ptr(), 1, w.as_mut_ptr(), 1);
        }
    } else {
        if n == 1 {
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

/// Adds two arrays (complex version)
///
/// **Note:** This is an internal function used by `vec_add` and `mat_add`.
///
/// ```text
/// w := α⋅u + β⋅v
/// ```
#[inline]
pub fn add_arrays_complex(
    w: &mut [Complex64],
    alpha: Complex64,
    u: &[Complex64],
    beta: Complex64,
    v: &[Complex64],
) -> Result<(), StrError> {
    let n = w.len();
    if u.len() != n || v.len() != n {
        return Err("arrays are incompatible");
    }
    if n == 0 {
        return Ok(());
    }
    if n > MAX_DIM_FOR_NATIVE_BLAS {
        let n_i32 = to_i32(n);
        unsafe {
            // w := v
            cblas_zcopy(n_i32, v.as_ptr(), 1, w.as_mut_ptr(), 1);
            // w := beta * v
            cblas_zscal(n_i32, &beta, w.as_mut_ptr(), 1);
            // w := alpha*u + w
            cblas_zaxpy(n_i32, &alpha, u.as_ptr(), 1, w.as_mut_ptr(), 1);
        }
    } else {
        if n == 1 {
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
    use super::{add_arrays, add_arrays_complex};
    use crate::internal::MAX_DIM_FOR_NATIVE_BLAS;
    use num_complex::Complex64;
    use russell_chk::{complex_vec_approx_eq, vec_approx_eq};

    const NOISE: f64 = 1234.567;
    const NOISE_COMPLEX: Complex64 = Complex64::new(1234.567, 3456.789);

    #[test]
    fn add_arrays_works() {
        for size in 0..MAX_DIM_FOR_NATIVE_BLAS {
            let mut u = vec![0.0; size];
            let mut v = vec![0.0; size];
            let mut w = vec![NOISE; size];
            let mut correct = vec![0.0; size];
            for i in 0..size {
                u[i] = i as f64;
                v[i] = i as f64;
                correct[i] = i as f64;
            }
            add_arrays(&mut w, 0.5, &u, 0.5, &v).unwrap();
            vec_approx_eq(&w, &correct, 1e-15);
        }

        // more tests
        #[rustfmt::skip]
        let u = [
            1.0, 2.0,
            1.0, 2.0, 3.0, 4.0,
            1.0, 2.0, 3.0, 4.0,
            1.0, 2.0, 3.0, 4.0,
            1.0, 2.0, 3.0, 4.0,
        ];
        #[rustfmt::skip]
        let v = [
            0.5, 1.0,
            0.5, 1.0, 1.5, 2.0,
            0.5, 1.0, 1.5, 2.0,
            0.5, 1.0, 1.5, 2.0,
            0.5, 1.0, 1.5, 2.0,
        ];
        let mut w = vec![NOISE; u.len()];
        add_arrays(&mut w, 1.0, &u, -4.0, &v).unwrap();
        #[rustfmt::skip]
        let correct = &[
            -1.0, -2.0,
            -1.0, -2.0, -3.0, -4.0,
            -1.0, -2.0, -3.0, -4.0,
            -1.0, -2.0, -3.0, -4.0,
            -1.0, -2.0, -3.0, -4.0,
        ];
        vec_approx_eq(&w, correct, 1e-15);
    }

    #[test]
    fn add_arrays_complex_works() {
        for size in 0..MAX_DIM_FOR_NATIVE_BLAS {
            let mut u = vec![Complex64::new(0.0, 0.0); size];
            let mut v = vec![Complex64::new(0.0, 0.0); size];
            let mut w = vec![NOISE_COMPLEX; size];
            let mut correct = vec![Complex64::new(0.0, 0.0); size];
            for i in 0..size {
                u[i] = Complex64::new(i as f64, i as f64);
                v[i] = Complex64::new(i as f64, i as f64);
                correct[i] = Complex64::new(i as f64, i as f64);
            }
            add_arrays_complex(&mut w, Complex64::new(0.5, 0.0), &u, Complex64::new(0.5, 0.0), &v).unwrap();
            complex_vec_approx_eq(&w, &correct, 1e-15);
        }

        // more tests
        #[rustfmt::skip]
        let u = [
            Complex64::new(1.0,1.0), Complex64::new(2.0,2.0),
            Complex64::new(1.0,1.0), Complex64::new(2.0,2.0), Complex64::new(3.0,3.0), Complex64::new(4.0,4.0),
            Complex64::new(1.0,1.0), Complex64::new(2.0,2.0), Complex64::new(3.0,3.0), Complex64::new(4.0,4.0),
            Complex64::new(1.0,1.0), Complex64::new(2.0,2.0), Complex64::new(3.0,3.0), Complex64::new(4.0,4.0),
            Complex64::new(1.0,1.0), Complex64::new(2.0,2.0), Complex64::new(3.0,3.0), Complex64::new(4.0,4.0),
        ];
        #[rustfmt::skip]
        let v = [
            Complex64::new(0.5,0.5), Complex64::new(1.0,1.0),
            Complex64::new(0.5,0.5), Complex64::new(1.0,1.0), Complex64::new(1.5,1.5), Complex64::new(2.0,2.0),
            Complex64::new(0.5,0.5), Complex64::new(1.0,1.0), Complex64::new(1.5,1.5), Complex64::new(2.0,2.0),
            Complex64::new(0.5,0.5), Complex64::new(1.0,1.0), Complex64::new(1.5,1.5), Complex64::new(2.0,2.0),
            Complex64::new(0.5,0.5), Complex64::new(1.0,1.0), Complex64::new(1.5,1.5), Complex64::new(2.0,2.0),
        ];
        let mut w = vec![NOISE_COMPLEX; u.len()];
        add_arrays_complex(&mut w, Complex64::new(1.0, 0.0), &u, Complex64::new(-4.0, 0.0), &v).unwrap();
        #[rustfmt::skip]
        let correct = &[
            Complex64::new(-1.0,-1.0), Complex64::new(-2.0,-2.0),
            Complex64::new(-1.0,-1.0), Complex64::new(-2.0,-2.0), Complex64::new(-3.0,-3.0), Complex64::new(-4.0,-4.0),
            Complex64::new(-1.0,-1.0), Complex64::new(-2.0,-2.0), Complex64::new(-3.0,-3.0), Complex64::new(-4.0,-4.0),
            Complex64::new(-1.0,-1.0), Complex64::new(-2.0,-2.0), Complex64::new(-3.0,-3.0), Complex64::new(-4.0,-4.0),
            Complex64::new(-1.0,-1.0), Complex64::new(-2.0,-2.0), Complex64::new(-3.0,-3.0), Complex64::new(-4.0,-4.0),
        ];
        complex_vec_approx_eq(&w, correct, 1e-15);
    }
}
