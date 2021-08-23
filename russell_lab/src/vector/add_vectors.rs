use super::*;

const USE_ADD_VECTORS_NATIVE: bool = false;

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
    if USE_ADD_VECTORS_NATIVE {
        add_vectors_native(w, alpha, u, beta, v);
        return;
    }
    let n = w.data.len();
    if u.data.len() != n {
        #[rustfmt::skip]
        panic!("the length of vector [u] (={}) must equal the length of vector [w] (={})", u.data.len(), n);
    }
    if v.data.len() != n {
        #[rustfmt::skip]
        panic!("the length of vector [v] (={}) must equal the length of vector [w] (={})", v.data.len(), n);
    }
    const SIZE_LIMIT: usize = 99;
    let use_openblas = n > SIZE_LIMIT;
    if use_openblas {
        add_vectors_oblas(w, alpha, u, beta, v);
    } else {
        add_vectors_simd(w, alpha, u, beta, v);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use russell_chk::*;

    #[test]
    fn add_vectors_works() {
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
        let mut w = Vector::new(u.dim());
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
    fn add_vectors_openblas_works() {
        let n = 100;
        let mut u = Vector::new(n);
        let mut v = Vector::new(n);
        let mut correct = Vec::new();
        for i in 0..n {
            u.data[i] = i as f64;
            v.data[i] = i as f64;
            correct.push((2 * i) as f64);
        }
        let mut w = Vector::new(n);
        add_vectors(&mut w, 1.0, &u, 1.0, &v);
        assert_vec_approx_eq!(w.data, correct, 1e-15);
    }
}
