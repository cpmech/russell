use super::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

union SimdToArray {
    array: [f64; 4],
    simd: __m256d,
}

#[inline]
#[cfg(target_arch = "x86_64")]
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
    unsafe {
        let a = _mm256_set1_pd(alpha);
        let b = _mm256_set1_pd(beta);
        for i in (m..n).step_by(LANES) {
            let x = _mm256_set_pd(u.data[i + 0], u.data[i + 1], u.data[i + 2], u.data[i + 3]);
            let y = _mm256_set_pd(v.data[i + 0], v.data[i + 1], v.data[i + 2], v.data[i + 3]);
            let ax = _mm256_mul_pd(a, x);
            let by = _mm256_mul_pd(b, y);
            let r = _mm256_add_pd(ax, by);
            // let lo = _mm256_extractf128_pd::<0>(r);
            // let hi = _mm256_extractf128_pd::<1>(r);
            // println!("{:?}, {:?}", lo, hi);
            let s = SimdToArray { simd: r };
            // println!("{:?}", s.array);
            w.data[i + 0] = s.array[3]; // must be reversed order
            w.data[i + 1] = s.array[2];
            w.data[i + 2] = s.array[1];
            w.data[i + 3] = s.array[0];
        }
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
