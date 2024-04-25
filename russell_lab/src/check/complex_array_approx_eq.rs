use num_complex::Complex;
use num_traits::{Num, NumCast};

/// Panics if two complex arrays (vectors) are not approximately equal to each other
///
/// # Panics
///
/// 1. Will panic if the dimensions are different
/// 2. Will panic if NAN, INFINITY, or NEG_INFINITY is found
/// 3. Will panic if the absolute difference of components is greater than the tolerance
///
/// # Examples
///
/// ## Accepts small error
///
/// ```
/// use russell_lab::*;
///
/// fn main() {
///     let a = vec![cpx!(3.0000001, 2.0000001), cpx!(1.0, 2.0)];
///     let b = vec![cpx!(3.0, 2.0),             cpx!(1.0, 2.0)];
///     complex_array_approx_eq(&a, &b, 1e-6);
/// }
/// ```
///
/// ## Panics on different values
///
/// ### Real part
///
/// ```should_panic
/// use russell_lab::*;
///
/// fn main() {
///     let a = vec![cpx!(1.0, 3.0), cpx!(1.0, 2.0)];
///     let b = vec![cpx!(2.0, 3.0), cpx!(1.0, 2.0)];
///     complex_array_approx_eq(&a, &b, 1e-6);
/// }
/// ```
///
/// ### Imaginary part
///
/// ```should_panic
/// use russell_lab::*;
///
/// fn main() {
///     let a = vec![cpx!(1.0, 3.0), cpx!(1.0, 2.0)];
///     let b = vec![cpx!(1.0, 4.0), cpx!(1.0, 2.0)];
///     complex_array_approx_eq(&a, &b, 1e-6);
/// }
/// ```
pub fn complex_array_approx_eq<T>(u: &[Complex<T>], v: &[Complex<T>], tol: f64)
where
    T: Num + NumCast + Copy,
{
    let m = u.len();
    if m != v.len() {
        panic!("vector dimensions differ. {} != {}", m, v.len());
    }
    for i in 0..m {
        let ui_re = u[i].re.to_f64().unwrap();
        let vi_re = v[i].re.to_f64().unwrap();

        if ui_re.is_nan() {
            panic!("NaN found in the first vector (real)");
        }
        if vi_re.is_nan() {
            panic!("NaN found in the second vector (real)");
        }
        if ui_re.is_infinite() {
            panic!("Inf found in the first vector (real)");
        }
        if vi_re.is_infinite() {
            panic!("Inf found in the second vector (real)");
        }

        let ui_im = u[i].im.to_f64().unwrap();
        let vi_im = v[i].im.to_f64().unwrap();

        if ui_im.is_nan() {
            panic!("NaN found in the first vector (imag)");
        }
        if vi_im.is_nan() {
            panic!("NaN found in the second vector (imag)");
        }
        if ui_im.is_infinite() {
            panic!("Inf found in the first vector (imag)");
        }
        if vi_im.is_infinite() {
            panic!("Inf found in the second vector (imag)");
        }

        let diff_re = f64::abs(ui_re - vi_re);
        if diff_re > tol {
            panic!("vectors are not approximately equal. diff_re[{}] = {:?}", i, diff_re);
        }

        let diff_im = f64::abs(ui_im - vi_im);
        if diff_im > tol {
            panic!("vectors are not approximately equal. diff_im[{}] = {:?}", i, diff_im);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::complex_array_approx_eq;
    use crate::Complex64;

    #[test]
    #[should_panic(expected = "NaN found in the first vector (real)")]
    fn panics_on_nan_real_1() {
        complex_array_approx_eq(&[Complex64::new(f64::NAN, 0.0)], &[Complex64::new(2.5, 0.0)], 1e-1);
    }

    #[test]
    #[should_panic(expected = "NaN found in the second vector (real)")]
    fn panics_on_nan_real_2() {
        complex_array_approx_eq(&[Complex64::new(2.5, 0.0)], &[Complex64::new(f64::NAN, 0.0)], 1e-1);
    }

    #[test]
    #[should_panic(expected = "Inf found in the first vector (real)")]
    fn panics_on_inf_real_1() {
        complex_array_approx_eq(&[Complex64::new(f64::INFINITY, 0.0)], &[Complex64::new(2.5, 0.0)], 1e-1);
    }

    #[test]
    #[should_panic(expected = "Inf found in the second vector (real)")]
    fn panics_on_inf_real_2() {
        complex_array_approx_eq(&[Complex64::new(2.5, 0.0)], &[Complex64::new(f64::INFINITY, 0.0)], 1e-1);
    }

    #[test]
    #[should_panic(expected = "Inf found in the first vector (real)")]
    fn panics_on_inf_real_3() {
        complex_array_approx_eq(
            &[Complex64::new(f64::NEG_INFINITY, 0.0)],
            &[Complex64::new(2.5, 0.0)],
            1e-1,
        );
    }

    #[test]
    #[should_panic(expected = "Inf found in the second vector (real)")]
    fn panics_on_inf_real_4() {
        complex_array_approx_eq(
            &[Complex64::new(2.5, 0.0)],
            &[Complex64::new(f64::NEG_INFINITY, 0.0)],
            1e-1,
        );
    }

    #[test]
    #[should_panic(expected = "NaN found in the first vector (imag)")]
    fn panics_on_nan_imag_1() {
        complex_array_approx_eq(&[Complex64::new(0.0, f64::NAN)], &[Complex64::new(2.5, 0.0)], 1e-1);
    }

    #[test]
    #[should_panic(expected = "NaN found in the second vector (imag)")]
    fn panics_on_nan_imag_2() {
        complex_array_approx_eq(&[Complex64::new(2.5, 0.0)], &[Complex64::new(0.0, f64::NAN)], 1e-1);
    }

    #[test]
    #[should_panic(expected = "Inf found in the first vector (imag)")]
    fn panics_on_inf_imag_1() {
        complex_array_approx_eq(&[Complex64::new(0.0, f64::INFINITY)], &[Complex64::new(2.5, 0.0)], 1e-1);
    }

    #[test]
    #[should_panic(expected = "Inf found in the second vector (imag)")]
    fn panics_on_inf_imag_2() {
        complex_array_approx_eq(&[Complex64::new(2.5, 0.0)], &[Complex64::new(0.0, f64::INFINITY)], 1e-1);
    }

    #[test]
    #[should_panic(expected = "Inf found in the first vector (imag)")]
    fn panics_on_inf_imag_3() {
        complex_array_approx_eq(
            &[Complex64::new(0.0, f64::NEG_INFINITY)],
            &[Complex64::new(2.5, 0.0)],
            1e-1,
        );
    }

    #[test]
    #[should_panic(expected = "Inf found in the second vector (imag)")]
    fn panics_on_inf_imag_4() {
        complex_array_approx_eq(
            &[Complex64::new(2.5, 0.0)],
            &[Complex64::new(0.0, f64::NEG_INFINITY)],
            1e-1,
        );
    }

    #[test]
    #[should_panic(expected = "vector dimensions differ. 2 != 3")]
    fn complex_array_approx_eq_works_1() {
        let u = &[Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];
        let v = &[
            Complex64::new(2.5, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 2.0),
        ];
        complex_array_approx_eq(u, v, 1e-15);
    }

    #[test]
    #[should_panic(expected = "vectors are not approximately equal. diff_re[0] = 1.5")]
    fn complex_array_approx_eq_works_2() {
        let u = &[Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];
        let v = &[Complex64::new(2.5, 0.0), Complex64::new(1.0, 0.0)];
        complex_array_approx_eq(u, v, 1e-15);
    }

    #[test]
    #[should_panic(expected = "vectors are not approximately equal. diff_re[1] =")]
    fn complex_array_approx_eq_works_3() {
        let u = &[Complex64::new(0.0, 0.0), Complex64::new(1e-14, 0.0)];
        let v = &[Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)];
        complex_array_approx_eq(u, v, 1e-15);
    }

    #[test]
    #[should_panic(expected = "vectors are not approximately equal. diff_im[0] = 1.5")]
    fn complex_array_approx_eq_works_4() {
        let u = &[Complex64::new(0.0, 1.0), Complex64::new(0.0, 2.0)];
        let v = &[Complex64::new(0.0, 2.5), Complex64::new(0.0, 1.0)];
        complex_array_approx_eq(u, v, 1e-15);
    }

    #[test]
    #[should_panic(expected = "vectors are not approximately equal. diff_im[1] =")]
    fn complex_array_approx_eq_works_5() {
        let u = &[Complex64::new(0.0, 0.0), Complex64::new(0.0, 1e-14)];
        let v = &[Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)];
        complex_array_approx_eq(u, v, 1e-15);
    }

    #[test]
    fn complex_array_approx_eq_works_6() {
        let u = &[Complex64::new(0.0, 0.0)];
        let v = &[Complex64::new(1e-15, 0.0)];
        complex_array_approx_eq(u, v, 1e-15);
    }

    #[test]
    fn complex_array_approx_eq_works_7() {
        let u = &[Complex64::new(0.0, 0.0)];
        let v = &[Complex64::new(0.0, 1e-15)];
        complex_array_approx_eq(u, v, 1e-15);
    }
}
