use crate::{to_i32, zaxpy, zcopy, zscal};
use num_complex::Complex64;

/// Adds two vectors
///
/// ```text
/// w := alpha * u + beta * v
/// ```
///
/// # Note
///
/// IMPORTANT: the vectors must have the same size
///
/// This function does NOT check for the dimensions of the arguments
///
#[inline]
pub fn complex_add_vectors_oblas(
    w: &mut [Complex64],
    alpha: Complex64,
    u: &[Complex64],
    beta: Complex64,
    v: &[Complex64],
) {
    let n = w.len();
    let n_i32: i32 = to_i32(n);
    // w := v
    zcopy(n_i32, v, 1, w, 1);
    // w := beta * v
    zscal(n_i32, beta, w, 1);
    // w := alpha*u + w
    zaxpy(n_i32, alpha, u, 1, w, 1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::complex_add_vectors_oblas;
    use num_complex::Complex64;
    use russell_chk::assert_complex_vec_approx_eq;

    #[test]
    fn complex_add_vectors_oblas_sizes_works() {
        const NOISE: Complex64 = Complex64::new(1234.567, 3456.789);
        for size in 0..13 {
            let mut u = vec![Complex64::new(0.0, 0.0); size];
            let mut v = vec![Complex64::new(0.0, 0.0); size];
            let mut w = vec![NOISE; size];
            let mut correct = vec![Complex64::new(0.0, 0.0); size];
            for i in 0..size {
                u[i] = Complex64::new(i as f64, i as f64);
                v[i] = Complex64::new(i as f64, i as f64);
                correct[i] = Complex64::new(i as f64, i as f64);
            }
            complex_add_vectors_oblas(&mut w, Complex64::new(0.5, 0.0), &u, Complex64::new(0.5, 0.0), &v);
            assert_complex_vec_approx_eq!(w, correct, 1e-15);
        }
    }

    #[test]
    fn complex_add_vectors_oblas_works() {
        const NOISE: Complex64 = Complex64::new(1234.567, 3456.789);
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
        let mut w = vec![NOISE; u.len()];
        complex_add_vectors_oblas(&mut w, Complex64::new(1.0, 0.0), &u, Complex64::new(-4.0, 0.0), &v);
        #[rustfmt::skip]
        let correct = &[
            Complex64::new(-1.0,-1.0), Complex64::new(-2.0,-2.0),
            Complex64::new(-1.0,-1.0), Complex64::new(-2.0,-2.0), Complex64::new(-3.0,-3.0), Complex64::new(-4.0,-4.0),
            Complex64::new(-1.0,-1.0), Complex64::new(-2.0,-2.0), Complex64::new(-3.0,-3.0), Complex64::new(-4.0,-4.0),
            Complex64::new(-1.0,-1.0), Complex64::new(-2.0,-2.0), Complex64::new(-3.0,-3.0), Complex64::new(-4.0,-4.0),
            Complex64::new(-1.0,-1.0), Complex64::new(-2.0,-2.0), Complex64::new(-3.0,-3.0), Complex64::new(-4.0,-4.0),
        ];
        assert_complex_vec_approx_eq!(w, correct, 1e-15);
    }
}
