use super::Matrix;
use crate::AsArray2D;

/// Panics if two matrices are not approximately equal to each other
///
/// **Note:** Will also panic if NaN or Inf is found.
///
/// **Note:** Will also panic if the dimensions are different.
pub fn mat_approx_eq<'a, T>(a: &Matrix, b: &'a T, tol: f64)
where
    T: AsArray2D<'a, f64>,
{
    let (m, n) = a.dims();
    let (mm, nn) = b.size();
    if m != mm {
        panic!("matrix dimensions differ. rows: {} != {}", m, mm);
    }
    if n != nn {
        panic!("matrix dimensions differ. columns: {} != {}", n, nn);
    }
    for i in 0..m {
        for j in 0..n {
            let diff = f64::abs(a.get(i, j) - b.at(i, j));
            if diff.is_nan() {
                panic!("mat_approx_eq found NaN");
            }
            if diff.is_infinite() {
                panic!("mat_approx_eq found Inf");
            }
            if diff > tol {
                panic!(
                    "matrices are not approximately equal. @ ({},{}) diff = {:?}",
                    i, j, diff
                );
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{mat_approx_eq, Matrix};

    #[test]
    #[should_panic(expected = "mat_approx_eq found NaN")]
    fn panics_on_nan() {
        mat_approx_eq(&Matrix::from(&[[f64::NAN]]), &Matrix::from(&[[2.5]]), 1e-1);
    }

    #[test]
    #[should_panic(expected = "mat_approx_eq found Inf")]
    fn panics_on_inf() {
        mat_approx_eq(&Matrix::from(&[[f64::INFINITY]]), &Matrix::from(&[[2.5]]), 1e-1);
    }

    #[test]
    #[should_panic(expected = "mat_approx_eq found Inf")]
    fn panics_on_neg_inf() {
        mat_approx_eq(&Matrix::from(&[[f64::NEG_INFINITY]]), &Matrix::from(&[[2.5]]), 1e-1);
    }

    #[test]
    #[should_panic(expected = "matrix dimensions differ. rows: 2 != 3")]
    fn mat_approx_eq_works_1() {
        let a = Matrix::new(2, 2);
        let b = &[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]];
        mat_approx_eq(&a, b, 1e-15);
    }

    #[test]
    #[should_panic(expected = "matrix dimensions differ. columns: 2 != 3")]
    fn mat_approx_eq_works_2() {
        let a = Matrix::new(2, 2);
        let b = &[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        mat_approx_eq(&a, b, 1e-15);
    }

    #[test]
    #[should_panic(expected = "matrices are not approximately equal. @ (0,0) diff = 1.5")]
    fn mat_approx_eq_works_3() {
        let a = Matrix::from(&[[1.0, 2.0], [3.0, 4.0]]);
        let b = &[[2.5, 1.0], [1.5, 2.0]];
        mat_approx_eq(&a, b, 1e-15);
    }

    #[test]
    #[should_panic(expected = "matrices are not approximately equal. @ (1,0) diff =")]
    fn mat_approx_eq_works_4() {
        let a = Matrix::new(2, 1);
        let b = &[[0.0], [1e-14]];
        mat_approx_eq(&a, b, 1e-15);
    }

    #[test]
    fn mat_approx_eq_works_5() {
        let a = Matrix::new(2, 1);
        let b = &[[0.0], [1e-15]];
        mat_approx_eq(&a, b, 1e-15);
    }
}
