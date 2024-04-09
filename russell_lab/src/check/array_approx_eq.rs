use num_traits::{Num, NumCast};

/// Panics if two vectors are not approximately equal to each other
///
/// **Note:** Will also panic if NaN or Inf is found.
///
/// **Note:** Will also panic if the vector dimensions are different.
pub fn vec_approx_eq<T>(u: &[T], v: &[T], tol: f64)
where
    T: Num + NumCast + Copy,
{
    let m = u.len();
    if m != v.len() {
        panic!("vector dimensions differ. {} != {}", m, v.len());
    }
    for i in 0..m {
        let diff = f64::abs(u[i].to_f64().unwrap() - v[i].to_f64().unwrap());
        if diff.is_nan() {
            panic!("vec_approx_eq found NaN");
        }
        if diff.is_infinite() {
            panic!("vec_approx_eq found Inf");
        }
        if diff > tol {
            panic!("vectors are not approximately equal. @ {} diff = {:?}", i, diff);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::vec_approx_eq;

    #[test]
    #[should_panic(expected = "vec_approx_eq found NaN")]
    fn panics_on_nan() {
        vec_approx_eq(&[f64::NAN], &[2.5], 1e-1);
    }

    #[test]
    #[should_panic(expected = "vec_approx_eq found Inf")]
    fn panics_on_inf() {
        vec_approx_eq(&[f64::INFINITY], &[2.5], 1e-1);
    }

    #[test]
    #[should_panic(expected = "vec_approx_eq found Inf")]
    fn panics_on_neg_inf() {
        vec_approx_eq(&[f64::NEG_INFINITY], &[2.5], 1e-1);
    }

    #[test]
    #[should_panic(expected = "vector dimensions differ. 2 != 3")]
    fn vec_approx_eq_works_1() {
        let u = &[0.0, 0.0];
        let v = &[0.0, 0.0, 0.0];
        vec_approx_eq(u, v, 1e-15);
    }

    #[test]
    #[should_panic(expected = "vectors are not approximately equal. @ 0 diff = 1.5")]
    fn vec_approx_eq_works_2() {
        let u = &[1.0, 2.0, 3.0, 4.0];
        let v = &[2.5, 1.0, 1.5, 2.0];
        vec_approx_eq(u, v, 1e-15);
    }

    #[test]
    #[should_panic(expected = "vectors are not approximately equal. @ 2 diff =")]
    fn vec_approx_eq_works_3() {
        let u = &[0.0, 0.0, 0.0];
        let v = &[0.0, 0.0, 1e-14];
        vec_approx_eq(u, v, 1e-15);
    }

    #[test]
    fn vec_approx_eq_works_4() {
        let u = &[0.0, 0.0, 0.0];
        let v = &[0.0, 0.0, 1e-15];
        vec_approx_eq(u, v, 1e-15);
    }
}
