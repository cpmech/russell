use super::ComplexVector;
use crate::AsArray1D;
use num_complex::Complex64;

/// Panics if two vectors are not approximately equal to each other
///
/// Panics also if the vector dimensions differ
pub fn complex_vec_approx_eq<'a, T>(u: &ComplexVector, v: &'a T, tol: f64)
where
    T: AsArray1D<'a, Complex64>,
{
    let m = u.dim();
    if m != v.size() {
        panic!("complex vector dimensions differ. {} != {}", m, v.size());
    }
    for i in 0..m {
        let diff_re = f64::abs(u[i].re - v.at(i).re);
        if diff_re > tol {
            panic!(
                "complex vectors are not approximately equal. @ {} diff_re = {:?}",
                i, diff_re
            );
        }
        let diff_im = f64::abs(u[i].im - v.at(i).im);
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
    use super::{complex_vec_approx_eq, ComplexVector};
    use num_complex::Complex64;

    #[test]
    #[should_panic(expected = "complex vector dimensions differ. 2 != 3")]
    fn complex_vec_approx_eq_works_1() {
        let u = ComplexVector::new(2);
        let v = ComplexVector::new(3);
        complex_vec_approx_eq(&u, v.as_data(), 1e-15);
    }

    #[test]
    #[should_panic(expected = "complex vectors are not approximately equal. @ 0 diff_re = 1.5")]
    fn complex_vec_approx_eq_works_2() {
        let u = ComplexVector::from(&[1.0, 2.0, 3.0, 4.0]);
        let v = ComplexVector::from(&[2.5, 1.0, 1.5, 2.0]);
        complex_vec_approx_eq(&u, v.as_data(), 1e-15);
    }

    #[test]
    #[should_panic(expected = "complex vectors are not approximately equal. @ 2 diff_re =")]
    fn complex_vec_approx_eq_works_3() {
        let u = ComplexVector::new(3);
        let v = ComplexVector::from(&[0.0, 0.0, 1e-14]);
        complex_vec_approx_eq(&u, v.as_data(), 1e-15);
    }

    #[test]
    #[should_panic(expected = "complex vectors are not approximately equal. @ 0 diff_im = 1.5")]
    fn complex_vec_approx_eq_works_4() {
        let u = ComplexVector::from(&[Complex64::new(0.0, 1.0), Complex64::new(0.0, 2.0)]);
        let v = &[Complex64::new(0.0, 2.5), Complex64::new(0.0, 1.0)];
        complex_vec_approx_eq(&u, v, 1e-15);
    }

    #[test]
    #[should_panic(expected = "complex vectors are not approximately equal. @ 0 diff_im =")]
    fn complex_vec_approx_eq_works_5() {
        let u = ComplexVector::new(1);
        let v = &[Complex64::new(0.0, 1e-14)];
        complex_vec_approx_eq(&u, v, 1e-15);
    }

    #[test]
    fn complex_vec_approx_eq_works_6() {
        let u = ComplexVector::new(3);
        let v = ComplexVector::from(&[0.0, 0.0, 1e-15]);
        complex_vec_approx_eq(&u, v.as_data(), 1e-15);
    }

    #[test]
    fn complex_vec_approx_eq_works_7() {
        let u = ComplexVector::new(1);
        let v = &[Complex64::new(0.0, 1e-15)];
        complex_vec_approx_eq(&u, v, 1e-15);
    }
}
