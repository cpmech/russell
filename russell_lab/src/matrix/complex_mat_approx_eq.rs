use super::ComplexMatrix;
use crate::AsArray2D;
use num_complex::Complex64;

/// Panics if two vectors are not approximately equal to each other
///
/// Panics also if the vector dimensions differ
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
            let diff_re = f64::abs(a[i][j].re - b.at(i, j).re);
            if diff_re > tol {
                panic!(
                    "complex matrices are not approximately equal. @ ({},{}) diff_re = {:?}",
                    i, j, diff_re
                );
            }
            let diff_im = f64::abs(a[i][j].im - b.at(i, j).im);
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
    use num_complex::Complex64;

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
        let a = ComplexMatrix::from(&[[1.0, 2.0], [3.0, 4.0]]);
        let b = &[
            [Complex64::new(2.5, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.5, 0.0), Complex64::new(2.0, 0.0)],
        ];
        complex_mat_approx_eq(&a, b, 1e-15);
    }

    #[test]
    #[should_panic(expected = "complex matrices are not approximately equal. @ (1,0) diff_re =")]
    fn complex_mat_approx_eq_works_4() {
        let a = ComplexMatrix::new(2, 1);
        let b = &[[Complex64::new(0.0, 0.0)], [Complex64::new(1e-14, 0.0)]];
        complex_mat_approx_eq(&a, b, 1e-15);
    }

    #[test]
    #[should_panic(expected = "complex matrices are not approximately equal. @ (0,0) diff_im = 1.5")]
    fn complex_mat_approx_eq_works_5() {
        let a = ComplexMatrix::from(&[
            [Complex64::new(0.0, 1.0), Complex64::new(0.0, 2.0)],
            [Complex64::new(0.0, 3.0), Complex64::new(0.0, 4.0)],
        ]);
        let b = &[
            [Complex64::new(0.0, 2.5), Complex64::new(0.0, 1.0)],
            [Complex64::new(0.0, 1.5), Complex64::new(0.0, 2.0)],
        ];
        complex_mat_approx_eq(&a, b, 1e-15);
    }

    #[test]
    #[should_panic(expected = "complex matrices are not approximately equal. @ (1,0) diff_im =")]
    fn complex_mat_approx_eq_works_6() {
        let a = ComplexMatrix::new(2, 1);
        let b = &[[Complex64::new(0.0, 0.0)], [Complex64::new(0.0, 1e-14)]];
        complex_mat_approx_eq(&a, b, 1e-15);
    }

    #[test]
    fn complex_mat_approx_eq_works_7() {
        let a = ComplexMatrix::new(2, 1);
        let b = &[[Complex64::new(0.0, 0.0)], [Complex64::new(1e-15, 0.0)]];
        complex_mat_approx_eq(&a, b, 1e-15);
    }

    #[test]
    fn complex_mat_approx_eq_works_8() {
        let a = ComplexMatrix::new(2, 1);
        let b = &[[Complex64::new(0.0, 0.0)], [Complex64::new(0.0, 1e-15)]];
        complex_mat_approx_eq(&a, b, 1e-15);
    }
}
