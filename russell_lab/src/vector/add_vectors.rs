use super::*;
use russell_openblas::*;

const NATIVE_VERSUS_OPENBLAS_BOUNDARY: usize = 16;

/// Performs the addition of two vectors
///
/// ```text
/// w := alpha * u + beta * v
/// ```
///
/// # Examples
///
/// ```
/// use russell_lab::*;
/// let u = Vector::from(&[10.0, 20.0, 30.0, 40.0]);
/// let v = Vector::from(&[2.0, 1.5, 1.0, 0.5]);
/// let mut w = Vector::new(4);
/// add_vectors(&mut w, 0.1, &u, 2.0, &v);
/// let correct = "┌   ┐\n\
///                │ 5 │\n\
///                │ 5 │\n\
///                │ 5 │\n\
///                │ 5 │\n\
///                └   ┘";
/// assert_eq!(format!("{}", w), correct);
/// ```
///
pub fn add_vectors(w: &mut Vector, alpha: f64, u: &Vector, beta: f64, v: &Vector) {
    let n = w.data.len();
    if u.data.len() != n {
        #[rustfmt::skip]
        panic!("the length of vector [u] (={}) must equal the length of vector [w] (={})", u.data.len(), n);
    }
    if v.data.len() != n {
        #[rustfmt::skip]
        panic!("the length of vector [v] (={}) must equal the length of vector [w] (={})", v.data.len(), n);
    }
    if n == 0 {
        return;
    }
    if n > NATIVE_VERSUS_OPENBLAS_BOUNDARY {
        add_vectors_oblas(&mut w.data, alpha, &u.data, beta, &v.data);
    } else {
        add_vectors_native(&mut w.data, alpha, &u.data, beta, &v.data);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use russell_chk::*;

    #[test]
    fn add_vectors_works() {
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
        add_vectors(&mut w, 1.0, &u, -4.0, &v);
        #[rustfmt::skip]
        let correct = &[
            -1.0, -2.0,
            -1.0, -2.0, -3.0, -4.0,
            -1.0, -2.0, -3.0, -4.0,
            -1.0, -2.0, -3.0, -4.0,
            -1.0, -2.0, -3.0, -4.0,
        ];
        assert_vec_approx_eq!(w.data, correct, 1e-15);
    }

    #[test]
    fn add_vectors_sizes_works() {
        const NOISE: f64 = 1234.567;
        for size in 0..(NATIVE_VERSUS_OPENBLAS_BOUNDARY + 3) {
            let mut u = Vector::new(size);
            let mut v = Vector::new(size);
            let mut w = Vector::from(&vec![NOISE; u.dim()]);
            let mut correct = vec![0.0; size];
            for i in 0..size {
                u.data[i] = i as f64;
                v.data[i] = i as f64;
                correct[i] = i as f64;
            }
            add_vectors(&mut w, 0.5, &u, 0.5, &v);
            assert_vec_approx_eq!(w.data, correct, 1e-15);
        }
    }
}
