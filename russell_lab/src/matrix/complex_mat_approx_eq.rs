use super::ComplexMatrix;
use crate::{AsArray2D, Complex64};

/// Panics if two matrices are not approximately equal to each other
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
///     let a = ComplexMatrix::from(&[
///         [cpx!(1.0, -1.0), cpx!(2.0, -2.0)],
///         [cpx!(3.0, -3.0), cpx!(4.0, -4.0)],
///     ]);
///     let b = &[
///         [cpx!(1.01, -1.01), cpx!(2.01, -2.01)],
///         [cpx!(3.01, -3.01), cpx!(4.01, -4.01)],
///     ];
///     complex_mat_approx_eq(&a, b, 0.011);
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
///     let a = ComplexMatrix::from(&[
///         [cpx!(1.0, 0.0), cpx!(2.0, 0.0)],
///         [cpx!(3.0, 0.0), cpx!(4.0, 0.0)],
///     ]);
///     let b = &[
///         [cpx!(2.5, 0.0), cpx!(1.0, 0.0)],
///         [cpx!(1.5, 0.0), cpx!(2.0, 0.0)],
///     ];
///     complex_mat_approx_eq(&a, b, 1e-15);
/// }
/// ```
///
/// ### Imaginary part
///
/// ```should_panic
/// use russell_lab::*;
///
/// fn main() {
///     let a = ComplexMatrix::from(&[
///         [cpx!(0.0, 1.0), cpx!(0.0, 2.0)],
///         [cpx!(0.0, 3.0), cpx!(0.0, 4.0)],
///     ]);
///     let b = &[
///         [cpx!(0.0, 2.5), cpx!(0.0, 1.0)],
///         [cpx!(0.0, 1.5), cpx!(0.0, 2.0)],
///     ];
///     complex_mat_approx_eq(&a, b, 1e-15);
/// }
/// ```
pub fn complex_mat_approx_eq<'a, T>(a: &ComplexMatrix, b: &'a T, tol: f64)
where
    T: AsArray2D<'a, Complex64>,
{
    let (m, n) = a.dims();
    let (mm, nn) = b.size();
    if m != mm {
        panic!("complex matrix dimensions differ. rows: {} != {}", m, mm);
    }
    if n != nn {
        panic!("complex matrix dimensions differ. columns: {} != {}", n, nn);
    }
    for i in 0..m {
        for j in 0..n {
            let diff_re = f64::abs(a.get(i, j).re - b.at(i, j).re);
            if diff_re.is_nan() {
                panic!("complex_mat_approx_eq found NaN (real)");
            }
            if diff_re.is_infinite() {
                panic!("complex_mat_approx_eq found Inf (real)");
            }
            if diff_re > tol {
                panic!(
                    "complex matrices are not approximately equal. @ ({},{}) diff_re = {:?}",
                    i, j, diff_re
                );
            }
            let diff_im = f64::abs(a.get(i, j).im - b.at(i, j).im);
            if diff_im.is_nan() {
                panic!("complex_mat_approx_eq found NaN (imag)");
            }
            if diff_im.is_infinite() {
                panic!("complex_mat_approx_eq found Inf (imag)");
            }
            if diff_im > tol {
                panic!(
                    "complex matrices are not approximately equal. @ ({},{}) diff_im = {:?}",
                    i, j, diff_im
                );
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{complex_mat_approx_eq, ComplexMatrix};
    use crate::{cpx, Complex64};

    #[test]
    #[should_panic(expected = "complex_mat_approx_eq found NaN (real)")]
    fn panics_on_nan_real() {
        complex_mat_approx_eq(
            &ComplexMatrix::from(&[[Complex64::new(f64::NAN, 0.0)]]),
            &ComplexMatrix::from(&[[Complex64::new(2.5, 0.0)]]),
            1e-1,
        );
    }

    #[test]
    #[should_panic(expected = "complex_mat_approx_eq found Inf (real)")]
    fn panics_on_inf_real() {
        complex_mat_approx_eq(
            &ComplexMatrix::from(&[[Complex64::new(f64::INFINITY, 0.0)]]),
            &ComplexMatrix::from(&[[Complex64::new(2.5, 0.0)]]),
            1e-1,
        );
    }

    #[test]
    #[should_panic(expected = "complex_mat_approx_eq found Inf (real)")]
    fn panics_on_neg_inf_real() {
        complex_mat_approx_eq(
            &ComplexMatrix::from(&[[Complex64::new(f64::NEG_INFINITY, 0.0)]]),
            &ComplexMatrix::from(&[[Complex64::new(2.5, 0.0)]]),
            1e-1,
        );
    }

    #[test]
    #[should_panic(expected = "complex_mat_approx_eq found NaN (imag)")]
    fn panics_on_nan_imag() {
        complex_mat_approx_eq(
            &ComplexMatrix::from(&[[Complex64::new(2.5, f64::NAN)]]),
            &ComplexMatrix::from(&[[Complex64::new(2.5, 0.0)]]),
            1e-1,
        );
    }

    #[test]
    #[should_panic(expected = "complex_mat_approx_eq found Inf (imag)")]
    fn panics_on_inf_imag() {
        complex_mat_approx_eq(
            &ComplexMatrix::from(&[[Complex64::new(2.5, f64::INFINITY)]]),
            &ComplexMatrix::from(&[[Complex64::new(2.5, 0.0)]]),
            1e-1,
        );
    }

    #[test]
    #[should_panic(expected = "complex_mat_approx_eq found Inf (imag)")]
    fn panics_on_neg_inf_imag() {
        complex_mat_approx_eq(
            &ComplexMatrix::from(&[[Complex64::new(2.5, f64::NEG_INFINITY)]]),
            &ComplexMatrix::from(&[[Complex64::new(2.5, 0.0)]]),
            1e-1,
        );
    }

    #[test]
    #[should_panic(expected = "complex matrix dimensions differ. rows: 2 != 1")]
    fn complex_mat_approx_eq_works_1() {
        let a = ComplexMatrix::new(2, 2);
        let b = ComplexMatrix::new(1, 2);
        complex_mat_approx_eq(&a, &b, 1e-15);
    }

    #[test]
    #[should_panic(expected = "complex matrix dimensions differ. columns: 2 != 3")]
    fn complex_mat_approx_eq_works_2() {
        let a = ComplexMatrix::new(2, 2);
        let b = ComplexMatrix::new(2, 3);
        complex_mat_approx_eq(&a, &b, 1e-15);
    }

    #[test]
    #[should_panic(expected = "complex matrices are not approximately equal. @ (0,0) diff_re = 1.5")]
    fn complex_mat_approx_eq_works_3() {
        let a = ComplexMatrix::from(&[
            [cpx!(1.0, 0.0), cpx!(2.0, 0.0)], //
            [cpx!(3.0, 0.0), cpx!(4.0, 0.0)], //
        ]);
        let b = &[
            [cpx!(2.5, 0.0), cpx!(1.0, 0.0)], //
            [cpx!(1.5, 0.0), cpx!(2.0, 0.0)], //
        ];
        complex_mat_approx_eq(&a, b, 1e-15);
    }

    #[test]
    #[should_panic(expected = "complex matrices are not approximately equal. @ (1,0) diff_re =")]
    fn complex_mat_approx_eq_works_4() {
        let a = ComplexMatrix::new(2, 1);
        let b = &[[cpx!(0.0, 0.0)], [cpx!(1e-14, 0.0)]];
        complex_mat_approx_eq(&a, b, 1e-15);
    }

    #[test]
    #[should_panic(expected = "complex matrices are not approximately equal. @ (0,0) diff_im = 1.5")]
    fn complex_mat_approx_eq_works_5() {
        let a = ComplexMatrix::from(&[
            [cpx!(0.0, 1.0), cpx!(0.0, 2.0)], //
            [cpx!(0.0, 3.0), cpx!(0.0, 4.0)], //
        ]);
        let b = &[
            [cpx!(0.0, 2.5), cpx!(0.0, 1.0)], //
            [cpx!(0.0, 1.5), cpx!(0.0, 2.0)], //
        ];
        complex_mat_approx_eq(&a, b, 1e-15);
    }

    #[test]
    #[should_panic(expected = "complex matrices are not approximately equal. @ (1,0) diff_im =")]
    fn complex_mat_approx_eq_works_6() {
        let a = ComplexMatrix::new(2, 1);
        let b = &[[cpx!(0.0, 0.0)], [cpx!(0.0, 1e-14)]];
        complex_mat_approx_eq(&a, b, 1e-15);
    }

    #[test]
    fn complex_mat_approx_eq_works_7() {
        let a = ComplexMatrix::new(2, 1);
        let b = &[[cpx!(0.0, 0.0)], [cpx!(1e-15, 0.0)]];
        complex_mat_approx_eq(&a, b, 1e-15);
    }

    #[test]
    fn complex_mat_approx_eq_works_8() {
        let a = ComplexMatrix::new(2, 1);
        let b = &[[cpx!(0.0, 0.0)], [cpx!(0.0, 1e-15)]];
        complex_mat_approx_eq(&a, b, 1e-15);
    }
}
