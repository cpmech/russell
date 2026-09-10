use crate::AsArray2D;

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
/// use russell_lab::{small_mat_approx_eq, Matrix};
///
/// fn main() {
///     let a = Matrix::from(&[
///         [1.0, 2.0],
///         [3.0, 4.0],
///     ]);
///     let b = Matrix::from(&[
///         [1.01, 2.01],
///         [3.01, 4.01],
///     ]);
///     small_mat_approx_eq(&a, &b, 0.011);
/// }
/// ```
///
/// ## Panics on different values
///
/// ```should_panic
/// use russell_lab::{small_mat_approx_eq, Matrix};
///
/// fn main() {
///     let a = Matrix::from(&[
///         [1.0, 2.0],
///         [3.0, 4.0],
///     ]);
///     let b = Matrix::from(&[
///         [1.01, 2.01],
///         [3.01, 4.01],
///     ]);
///     small_mat_approx_eq(&a, &b, 0.001);
/// }
/// ```
pub fn small_mat_approx_eq<'a, const M: usize, const N: usize, T>(a: &[[f64; N]; M], b: &'a T, tol: f64)
where
    T: AsArray2D<'a, f64>,
{
    let (mm, nn) = b.size();
    if M != mm {
        panic!("matrix dimensions differ. rows: {} != {}", M, mm);
    }
    if N != nn {
        panic!("matrix dimensions differ. columns: {} != {}", N, nn);
    }
    for i in 0..M {
        for j in 0..N {
            let diff = f64::abs(a[i][j] - b.at(i, j));
            if diff.is_nan() {
                panic!("small_mat_approx_eq found NaN");
            }
            if diff.is_infinite() {
                panic!("small_mat_approx_eq found Inf");
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
    use super::small_mat_approx_eq;

    #[test]
    #[should_panic(expected = "small_mat_approx_eq found NaN")]
    fn panics_on_nan() {
        small_mat_approx_eq(&[[f64::NAN]], &[[2.5]], 1e-1);
    }

    #[test]
    #[should_panic(expected = "small_mat_approx_eq found Inf")]
    fn panics_on_inf() {
        small_mat_approx_eq(&[[f64::INFINITY]], &[[2.5]], 1e-1);
    }

    #[test]
    #[should_panic(expected = "small_mat_approx_eq found Inf")]
    fn panics_on_neg_inf() {
        small_mat_approx_eq(&[[f64::NEG_INFINITY]], &[[2.5]], 1e-1);
    }

    #[test]
    #[should_panic(expected = "matrix dimensions differ. rows: 2 != 3")]
    fn small_mat_approx_eq_works_1() {
        let a = &[[0.0, 0.0], [0.0, 0.0]];
        let b = &[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]];
        small_mat_approx_eq(&a, b, 1e-15);
    }

    #[test]
    #[should_panic(expected = "matrix dimensions differ. columns: 2 != 3")]
    fn small_mat_approx_eq_works_2() {
        let a = &[[0.0, 0.0], [0.0, 0.0]];
        let b = &[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        small_mat_approx_eq(&a, b, 1e-15);
    }

    #[test]
    #[should_panic(expected = "matrices are not approximately equal. @ (0,0) diff = 1.5")]
    fn small_mat_approx_eq_works_3() {
        let a = &[[1.0, 2.0], [3.0, 4.0]];
        let b = &[[2.5, 1.0], [1.5, 2.0]];
        small_mat_approx_eq(&a, b, 1e-15);
    }

    #[test]
    #[should_panic(expected = "matrices are not approximately equal. @ (1,0) diff =")]
    fn small_mat_approx_eq_works_4() {
        let a = &[[0.0], [0.0]];
        let b = &[[0.0], [1e-14]];
        small_mat_approx_eq(&a, b, 1e-15);
    }

    #[test]
    fn small_mat_approx_eq_works_5() {
        let a = &[[0.0], [0.0]];
        let b = &[[0.0], [1e-15]];
        small_mat_approx_eq(&a, b, 1e-15);
    }
}
