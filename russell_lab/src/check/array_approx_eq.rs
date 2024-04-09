use num_traits::{Num, NumCast};

/// Panics if two arrays (vectors) are not approximately equal to each other
///
/// # Panics
///
/// 1. Will panic if the dimensions are different
/// 2. Will panic if NaN or Inf is found
/// 3. Will panic if the absolute difference of components is greater than the tolerance
pub fn array_approx_eq<T>(u: &[T], v: &[T], tol: f64)
where
    T: Num + NumCast + Copy,
{
    let m = u.len();
    if m != v.len() {
        panic!("vector dimensions differ. {} != {}", m, v.len());
    }
    for i in 0..m {
        let ui = u[i].to_f64().unwrap();
        let vi = v[i].to_f64().unwrap();
        if ui.is_nan() {
            panic!("NaN found in the first vector");
        }
        if vi.is_nan() {
            panic!("NaN found in the second vector");
        }
        if ui.is_infinite() {
            panic!("Inf found in the first vector");
        }
        if vi.is_infinite() {
            panic!("Inf found in the second vector");
        }
        let diff = f64::abs(u[i].to_f64().unwrap() - v[i].to_f64().unwrap());
        if diff > tol {
            panic!("vectors are not approximately equal. diff[{}] = {:?}", i, diff);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::array_approx_eq;

    #[test]
    #[should_panic(expected = "NaN found in the first vector")]
    fn panics_on_nan_1() {
        array_approx_eq(&[f64::NAN], &[2.5], 1e-1);
    }

    #[test]
    #[should_panic(expected = "NaN found in the second vector")]
    fn panics_on_nan_2() {
        array_approx_eq(&[2.5], &[f64::NAN], 1e-1);
    }

    #[test]
    #[should_panic(expected = "Inf found in the first vector")]
    fn panics_on_inf_1() {
        array_approx_eq(&[f64::INFINITY], &[2.5], 1e-1);
    }

    #[test]
    #[should_panic(expected = "Inf found in the second vector")]
    fn panics_on_inf_2() {
        array_approx_eq(&[2.5], &[f64::INFINITY], 1e-1);
    }

    #[test]
    #[should_panic(expected = "Inf found in the first vector")]
    fn panics_on_neg_inf_1() {
        array_approx_eq(&[f64::NEG_INFINITY], &[2.5], 1e-1);
    }

    #[test]
    #[should_panic(expected = "Inf found in the second vector")]
    fn panics_on_neg_inf_2() {
        array_approx_eq(&[2.5], &[f64::NEG_INFINITY], 1e-1);
    }

    #[test]
    #[should_panic(expected = "vector dimensions differ. 2 != 3")]
    fn array_approx_eq_works_1() {
        let u = &[0.0, 0.0];
        let v = &[0.0, 0.0, 0.0];
        array_approx_eq(u, v, 1e-15);
    }

    #[test]
    #[should_panic(expected = "vectors are not approximately equal. diff[0] = 1.5")]
    fn array_approx_eq_works_2() {
        let u = &[1.0, 2.0, 3.0, 4.0];
        let v = &[2.5, 1.0, 1.5, 2.0];
        array_approx_eq(u, v, 1e-15);
    }

    #[test]
    #[should_panic(expected = "vectors are not approximately equal. diff[2] =")]
    fn array_approx_eq_works_3() {
        let u = &[0.0, 0.0, 0.0];
        let v = &[0.0, 0.0, 1e-14];
        array_approx_eq(u, v, 1e-15);
    }

    #[test]
    fn array_approx_eq_works_4() {
        let u = &[0.0, 0.0, 0.0];
        let v = &[0.0, 0.0, 1e-15];
        array_approx_eq(u, v, 1e-15);
    }
}
