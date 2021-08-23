use super::*;
use russell_openblas::*;
use std::convert::TryInto;

#[inline]
pub(crate) fn add_vectors_oblas(w: &mut Vector, alpha: f64, u: &Vector, beta: f64, v: &Vector) {
    let n = w.data.len();
    let n_i32: i32 = n.try_into().unwrap();
    // w := v
    dcopy(n_i32, &v.data, 1, &mut w.data, 1);
    // w := beta * v
    dscal(n_i32, beta, &mut w.data, 1);
    // w := alpha*u + w
    daxpy(n_i32, alpha, &u.data, 1, &mut w.data, 1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use russell_chk::*;

    #[test]
    fn add_vectors_oblas_works() {
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
        add_vectors_oblas(&mut w, 1.0, &u, -4.0, &v);
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
