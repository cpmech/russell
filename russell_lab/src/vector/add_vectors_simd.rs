use super::*;
use core_simd::*;

#[inline]
pub(crate) fn add_vectors_simd(w: &mut Vector, alpha: f64, u: &Vector, beta: f64, v: &Vector) {
    let n = w.data.len();
    if u.data.len() != n {
        #[rustfmt::skip]
        panic!("the length of vector [u] (={}) must equal the length of vector [w] (={})", u.data.len(), n);
    }
    if v.data.len() != n {
        #[rustfmt::skip]
        panic!("the length of vector [v] (={}) must equal the length of vector [w] (={})", v.data.len(), n);
    }
    const LANES: usize = 4;
    let m = n % LANES;
    for i in 0..m {
        w.data[i] = alpha * u.data[i] + beta * v.data[i];
    }
    for i in (m..n).step_by(LANES) {
        #[rustfmt::skip]
        let r = 
              alpha * f64x4::from([u.data[i + 0], u.data[i + 1], u.data[i + 2], u.data[i + 3]])
            +  beta * f64x4::from([v.data[i + 0], v.data[i + 1], v.data[i + 2], v.data[i + 3]]);
        let s: [f64; LANES] = r.to_array();
        w.data[i + 0] = s[0];
        w.data[i + 1] = s[1];
        w.data[i + 2] = s[2];
        w.data[i + 3] = s[3];
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use russell_chk::*;

    #[test]
    fn simd_add_vectors_works() {
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
        add_vectors_simd(&mut w, 1.0, &u, -4.0, &v);
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
