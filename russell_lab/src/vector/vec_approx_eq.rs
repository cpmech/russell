use super::Vector;
use crate::AsArray1D;

/// Panics if two vectors are not approximately equal to each other
///
/// Panics also if the vector dimensions differ
pub fn vec_approx_eq<'a, T>(u: &Vector, v: &'a T, tol: f64)
where
    T: AsArray1D<'a, f64>,
{
    let m = u.dim();
    if m != v.size() {
        panic!("vector dimensions differ. {} != {}", m, v.size());
    }
    for i in 0..m {
        let diff = f64::abs(u[i] - v.at(i));
        if diff > tol {
            panic!("vectors are not approximately equal. @ {} diff = {:?}", i, diff);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{vec_approx_eq, Vector};

    #[test]
    #[should_panic(expected = "vector dimensions differ. 2 != 3")]
    fn vec_approx_eq_works_1() {
        let u = Vector::new(2);
        let v = Vector::new(3);
        vec_approx_eq(&u, v.as_data(), 1e-15);
    }

    #[test]
    #[should_panic(expected = "vectors are not approximately equal. @ 0 diff = 1.5")]
    fn vec_approx_eq_works_2() {
        let u = Vector::from(&[1.0, 2.0, 3.0, 4.0]);
        let v = &[2.5, 1.0, 1.5, 2.0];
        vec_approx_eq(&u, v, 1e-15);
    }

    #[test]
    #[should_panic(expected = "vectors are not approximately equal. @ 2 diff =")]
    fn vec_approx_eq_works_3() {
        let u = Vector::new(3);
        let v = &[0.0, 0.0, 1e-14];
        vec_approx_eq(&u, v, 1e-15);
    }

    #[test]
    fn vec_approx_eq_works_4() {
        let u = Vector::new(3);
        let v = Vector::from(&[0.0, 0.0, 1e-15]);
        vec_approx_eq(&u, v.as_data(), 1e-15);
    }
}
