use num_complex::Complex;
use num_traits::{Num, NumCast};

/// Panics if two vectors are not approximately equal to each other
///
/// **Note:** Will also panic if NaN or Inf is found
///
/// Panics also if the vector dimensions differ
pub fn complex_vec_approx_eq<T>(u: &[Complex<T>], v: &[Complex<T>], tol: f64)
where
    T: Num + NumCast + Copy,
{
    let m = u.len();
    if m != v.len() {
        panic!("complex vector dimensions differ. {} != {}", m, v.len());
    }
    for i in 0..m {
        let diff_re = f64::abs(u[i].re.to_f64().unwrap() - v[i].re.to_f64().unwrap());
        if diff_re.is_nan() {
            panic!("complex_vec_approx_eq found NaN (real)");
        }
        if diff_re.is_infinite() {
            panic!("complex_vec_approx_eq found Inf (real)");
        }
        if diff_re > tol {
            panic!(
                "complex vectors are not approximately equal. @ {} diff_re = {:?}",
                i, diff_re
            );
        }
        let diff_im = f64::abs(u[i].im.to_f64().unwrap() - v[i].im.to_f64().unwrap());
        if diff_im.is_nan() {
            panic!("complex_vec_approx_eq found NaN (imag)");
        }
        if diff_im.is_infinite() {
            panic!("complex_vec_approx_eq found Inf (imag)");
        }
        if diff_im > tol {
            panic!(
                "complex vectors are not approximately equal. @ {} diff_im = {:?}",
                i, diff_im
            );
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::complex_vec_approx_eq;
    use num_complex::Complex64;

    #[test]
    #[should_panic(expected = "complex_vec_approx_eq found NaN (real)")]
    fn panics_on_nan_real() {
        complex_vec_approx_eq(&[Complex64::new(f64::NAN, 0.0)], &[Complex64::new(2.5, 0.0)], 1e-1);
    }

    #[test]
    #[should_panic(expected = "complex_vec_approx_eq found Inf (real)")]
    fn panics_on_inf_real() {
        complex_vec_approx_eq(&[Complex64::new(f64::INFINITY, 0.0)], &[Complex64::new(2.5, 0.0)], 1e-1);
    }

    #[test]
    #[should_panic(expected = "complex_vec_approx_eq found Inf (real)")]
    fn panics_on_neg_inf_real() {
        complex_vec_approx_eq(
            &[Complex64::new(f64::NEG_INFINITY, 0.0)],
            &[Complex64::new(2.5, 0.0)],
            1e-1,
        );
    }

    #[test]
    #[should_panic(expected = "complex_vec_approx_eq found NaN (imag)")]
    fn panics_on_nan_imag() {
        complex_vec_approx_eq(&[Complex64::new(2.5, f64::NAN)], &[Complex64::new(2.5, 0.0)], 1e-1);
    }

    #[test]
    #[should_panic(expected = "complex_vec_approx_eq found Inf (imag)")]
    fn panics_on_inf_imag() {
        complex_vec_approx_eq(&[Complex64::new(2.5, f64::INFINITY)], &[Complex64::new(2.5, 0.0)], 1e-1);
    }

    #[test]
    #[should_panic(expected = "complex_vec_approx_eq found Inf (imag)")]
    fn panics_on_neg_inf_imag() {
        complex_vec_approx_eq(
            &[Complex64::new(2.5, f64::NEG_INFINITY)],
            &[Complex64::new(2.5, 0.0)],
            1e-1,
        );
    }

    #[test]
    #[should_panic(expected = "complex vector dimensions differ. 2 != 3")]
    fn complex_vec_approx_eq_works_1() {
        let u = &[Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];
        let v = &[
            Complex64::new(2.5, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 2.0),
        ];
        complex_vec_approx_eq(u, v, 1e-15);
    }

    #[test]
    #[should_panic(expected = "complex vectors are not approximately equal. @ 0 diff_re = 1.5")]
    fn complex_vec_approx_eq_works_2() {
        let u = &[Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];
        let v = &[Complex64::new(2.5, 0.0), Complex64::new(1.0, 0.0)];
        complex_vec_approx_eq(u, v, 1e-15);
    }

    #[test]
    #[should_panic(expected = "complex vectors are not approximately equal. @ 1 diff_re =")]
    fn complex_vec_approx_eq_works_3() {
        let u = &[Complex64::new(0.0, 0.0), Complex64::new(1e-14, 0.0)];
        let v = &[Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)];
        complex_vec_approx_eq(u, v, 1e-15);
    }

    #[test]
    #[should_panic(expected = "complex vectors are not approximately equal. @ 0 diff_im = 1.5")]
    fn complex_vec_approx_eq_works_4() {
        let u = &[Complex64::new(0.0, 1.0), Complex64::new(0.0, 2.0)];
        let v = &[Complex64::new(0.0, 2.5), Complex64::new(0.0, 1.0)];
        complex_vec_approx_eq(u, v, 1e-15);
    }

    #[test]
    #[should_panic(expected = "complex vectors are not approximately equal. @ 1 diff_im =")]
    fn complex_vec_approx_eq_works_5() {
        let u = &[Complex64::new(0.0, 0.0), Complex64::new(0.0, 1e-14)];
        let v = &[Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)];
        complex_vec_approx_eq(u, v, 1e-15);
    }

    #[test]
    fn complex_vec_approx_eq_works_6() {
        let u = &[Complex64::new(0.0, 0.0)];
        let v = &[Complex64::new(1e-15, 0.0)];
        complex_vec_approx_eq(u, v, 1e-15);
    }

    #[test]
    fn complex_vec_approx_eq_works_7() {
        let u = &[Complex64::new(0.0, 0.0)];
        let v = &[Complex64::new(0.0, 1e-15)];
        complex_vec_approx_eq(u, v, 1e-15);
    }
}
