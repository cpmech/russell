use num_traits::{Num, NumCast};

/// Panics if two vectors are not approximately equal to each other
///
/// Panics also if the vector dimensions differ
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
