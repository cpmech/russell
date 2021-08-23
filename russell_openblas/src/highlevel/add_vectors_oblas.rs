use crate::{daxpy, dcopy, dscal};
use std::convert::TryInto;

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
pub fn add_vectors_oblas(w: &mut [f64], alpha: f64, u: &[f64], beta: f64, v: &[f64]) {
    let n = w.len();
    let n_i32: i32 = n.try_into().unwrap();
    // w := v
    dcopy(n_i32, v, 1, w, 1);
    // w := beta * v
    dscal(n_i32, beta, w, 1);
    // w := alpha*u + w
    daxpy(n_i32, alpha, u, 1, w, 1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use russell_chk::*;

    #[test]
    fn add_vectors_oblas_sizes_works() {
        const NOISE: f64 = 1234.567;
        for size in 0..13 {
            let mut u = vec![0.0; size];
            let mut v = vec![0.0; size];
            let mut w = vec![NOISE; size];
            let mut correct = vec![0.0; size];
            for i in 0..size {
                u[i] = i as f64;
                v[i] = i as f64;
                correct[i] = i as f64;
            }
            add_vectors_oblas(&mut w, 0.5, &u, 0.5, &v);
            assert_vec_approx_eq!(w, correct, 1e-15);
        }
    }

    #[test]
    fn add_vectors_oblas_works() {
        const NOISE: f64 = 1234.567;
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
        add_vectors_oblas(&mut w, 1.0, &u, -4.0, &v);
        #[rustfmt::skip]
        let correct = &[
            -1.0, -2.0,
            -1.0, -2.0, -3.0, -4.0,
            -1.0, -2.0, -3.0, -4.0,
            -1.0, -2.0, -3.0, -4.0,
            -1.0, -2.0, -3.0, -4.0,
        ];
        assert_vec_approx_eq!(w, correct, 1e-15);
    }
}
