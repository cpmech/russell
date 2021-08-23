use super::*;
use russell_openblas::*;
use std::convert::TryInto;

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
    const SIZE_LIMIT: usize = 99;
    let use_openblas = n > SIZE_LIMIT;
    if use_openblas {
        let n_i32: i32 = n.try_into().unwrap();
        // w := v
        dcopy(n_i32, &v.data, 1, &mut w.data, 1);
        // w := beta * v
        dscal(n_i32, beta, &mut w.data, 1);
        // w := alpha*u + w
        daxpy(n_i32, alpha, &u.data, 1, &mut w.data, 1);
    } else {
        add_vectors_simd(w, alpha, u, beta, v);
        /*
        let m = n % 4;
        for i in 0..m {
            w.data[i] = alpha * u.data[i] + beta * v.data[i];
        }
        for i in (m..n).step_by(4) {
            w.data[i + 0] = alpha * u.data[i + 0] + beta * v.data[i + 0];
            w.data[i + 1] = alpha * u.data[i + 1] + beta * v.data[i + 1];
            w.data[i + 2] = alpha * u.data[i + 2] + beta * v.data[i + 2];
            w.data[i + 3] = alpha * u.data[i + 3] + beta * v.data[i + 3];
        }
        */
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
