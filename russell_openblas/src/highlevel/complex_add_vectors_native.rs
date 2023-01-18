use num_complex::Complex64;

/// Adds two vectors (complex version)
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
pub fn complex_add_vectors_native(
    w: &mut [Complex64],
    alpha: Complex64,
    u: &[Complex64],
    beta: Complex64,
    v: &[Complex64],
) {
    let n = w.len();
    if n == 0 {
        return;
    }
    if n == 1 {
        w[0] = alpha * u[0] + beta * v[0];
        return;
    }
    if n == 2 {
        w[0] = alpha * u[0] + beta * v[0];
        w[1] = alpha * u[1] + beta * v[1];
        return;
    }
    if n == 3 {
        w[0] = alpha * u[0] + beta * v[0];
        w[1] = alpha * u[1] + beta * v[1];
        w[2] = alpha * u[2] + beta * v[2];
        return;
    }
    if n == 4 {
        w[0] = alpha * u[0] + beta * v[0];
        w[1] = alpha * u[1] + beta * v[1];
        w[2] = alpha * u[2] + beta * v[2];
        w[3] = alpha * u[3] + beta * v[3];
        return;
    }
    if n == 5 {
        w[0] = alpha * u[0] + beta * v[0];
        w[1] = alpha * u[1] + beta * v[1];
        w[2] = alpha * u[2] + beta * v[2];
        w[3] = alpha * u[3] + beta * v[3];
        w[4] = alpha * u[4] + beta * v[4];
        return;
    }
    if n == 6 {
        w[0] = alpha * u[0] + beta * v[0];
        w[1] = alpha * u[1] + beta * v[1];
        w[2] = alpha * u[2] + beta * v[2];
        w[3] = alpha * u[3] + beta * v[3];
        w[4] = alpha * u[4] + beta * v[4];
        w[5] = alpha * u[5] + beta * v[5];
        return;
    }
    if n == 7 {
        w[0] = alpha * u[0] + beta * v[0];
        w[1] = alpha * u[1] + beta * v[1];
        w[2] = alpha * u[2] + beta * v[2];
        w[3] = alpha * u[3] + beta * v[3];
        w[4] = alpha * u[4] + beta * v[4];
        w[5] = alpha * u[5] + beta * v[5];
        w[6] = alpha * u[6] + beta * v[6];
        return;
    }
    if n == 8 {
        w[0] = alpha * u[0] + beta * v[0];
        w[1] = alpha * u[1] + beta * v[1];
        w[2] = alpha * u[2] + beta * v[2];
        w[3] = alpha * u[3] + beta * v[3];
        w[4] = alpha * u[4] + beta * v[4];
        w[5] = alpha * u[5] + beta * v[5];
        w[6] = alpha * u[6] + beta * v[6];
        w[7] = alpha * u[7] + beta * v[7];
        return;
    }
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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::complex_add_vectors_native;
    use num_complex::Complex64;
    use russell_chk::complex_vec_approx_eq;

    #[test]
    fn complex_add_vectors_native_sizes_works() {
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
            complex_add_vectors_native(&mut w, Complex64::new(0.5, 0.0), &u, Complex64::new(0.5, 0.0), &v);
            complex_vec_approx_eq(&w, &correct, 1e-15);
        }
    }

    #[test]
    fn complex_add_vectors_native_works() {
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
        complex_add_vectors_native(&mut w, Complex64::new(1.0, 0.0), &u, Complex64::new(-4.0, 0.0), &v);
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
