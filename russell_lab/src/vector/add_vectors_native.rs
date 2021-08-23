use super::*;

#[inline]
pub(crate) fn add_vectors_native(w: &mut Vector, alpha: f64, u: &Vector, beta: f64, v: &Vector) {
    let n = w.data.len();
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
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use russell_chk::*;

    #[test]
    fn add_vectors_native_works() {
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
        add_vectors_native(&mut w, 1.0, &u, -4.0, &v);
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
}
