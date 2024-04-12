use num_traits::{Num, NumCast};

/// Panics if two numbers are not approximately equal to each other
///
/// # Panics
///
/// 1. Will panic if NAN, INFINITY, or NEG_INFINITY is found
/// 2. Will panic if the absolute difference is greater than the tolerance
///
/// # Input
///
/// `a` -- Left value
/// `b` -- Right value
/// `tol: f64` -- Error tolerance: panic occurs if `|a - b| > tol`
///
/// # Examples
///
/// ## Accepts small error
///
/// ```
/// use russell_lab::approx_eq;
///
/// fn main() {
///     let a = 3.0000001;
///     let b = 3.0;
///     approx_eq(a, b, 1e-6);
/// }
/// ```
///
/// ## Panics on different value
///
/// ```should_panic
/// use russell_lab::approx_eq;
///
/// fn main() {
///     let a = 1.0;
///     let b = 2.0;
///     approx_eq(a, b, 1e-6);
/// }
/// ```
pub fn approx_eq<T>(a: T, b: T, tol: f64)
where
    T: Num + NumCast + Copy,
{
    let aa = a.to_f64().unwrap();
    let bb = b.to_f64().unwrap();
    if aa.is_nan() {
        panic!("the first number is NaN");
    }
    if bb.is_nan() {
        panic!("the second number is NaN");
    }
    if aa.is_infinite() {
        panic!("the first number is Inf");
    }
    if bb.is_infinite() {
        panic!("the second number is Inf");
    }
    let diff = f64::abs(aa - bb);
    if diff > tol {
        panic!("numbers are not approximately equal. diff = {:?}", diff);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::approx_eq;

    #[test]
    #[should_panic(expected = "the first number is NaN")]
    fn panics_on_nan_1() {
        approx_eq(f64::NAN, 2.5, 1e-1);
    }

    #[test]
    #[should_panic(expected = "the second number is NaN")]
    fn panics_on_nan_2() {
        approx_eq(2.5, f64::NAN, 1e-1);
    }

    #[test]
    #[should_panic(expected = "the first number is Inf")]
    fn panics_on_inf_1() {
        approx_eq(f64::INFINITY, 2.5, 1e-1);
    }

    #[test]
    #[should_panic(expected = "the second number is Inf")]
    fn panics_on_inf_2() {
        approx_eq(2.5, f64::INFINITY, 1e-1);
    }

    #[test]
    #[should_panic(expected = "the first number is Inf")]
    fn panics_on_inf_3() {
        approx_eq(f64::NEG_INFINITY, 2.5, 1e-1);
    }

    #[test]
    #[should_panic(expected = "the second number is Inf")]
    fn panics_on_inf_4() {
        approx_eq(2.5, f64::NEG_INFINITY, 1e-1);
    }

    #[test]
    #[should_panic(expected = "numbers are not approximately equal. diff = 0.5")]
    fn panics_on_different_values() {
        approx_eq(2.0, 2.5, 1e-1);
    }

    #[test]
    #[should_panic(expected = "numbers are not approximately equal. diff = 0.5")]
    fn panics_on_different_values_f32() {
        approx_eq(2f32, 2.5f32, 1e-1);
    }

    #[test]
    fn accepts_approx_equal_values() {
        let a = 2.0;
        let b = 2.02;
        let tol = 0.03;
        approx_eq(a, b, tol);
    }

    #[test]
    fn accepts_approx_equal_values_f32() {
        approx_eq(2f32, 2.02f32, 0.03);
    }
}
