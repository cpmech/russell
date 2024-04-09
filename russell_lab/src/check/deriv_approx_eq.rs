use crate::deriv_central5;

/// Panics if derivative is not approximately equal to a numerical derivative
///
/// # Panics
///
/// 1. Will panic if NAN, INFINITY, or NEG_INFINITY is found
/// 2. Will panic if the absolute difference of derivative values is greater than the tolerance
///
/// **Note:** Will also panic if NaN or Inf is found
pub fn deriv_approx_eq<F, A>(dfdx: f64, at_x: f64, args: &mut A, tol: f64, f: F)
where
    F: FnMut(f64, &mut A) -> f64,
{
    if dfdx.is_nan() {
        panic!("the derivative is NaN");
    }
    if dfdx.is_infinite() {
        panic!("the derivative is Inf");
    }

    let dfdx_num = deriv_central5(at_x, args, f);

    if dfdx_num.is_nan() {
        panic!("the numerical derivative is NaN");
    }

    let diff = f64::abs(dfdx - dfdx_num);
    assert!(diff.is_finite());
    if diff > tol {
        panic!(
            "derivative is not approximately equal to numerical value. diff = {:?}",
            diff
        );
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::deriv_approx_eq;

    struct Arguments {}

    fn placeholder(_: f64, _: &mut Arguments) -> f64 {
        panic!("NOT HERE");
    }

    #[test]
    #[should_panic(expected = "NOT HERE")]
    fn placeholder_panics() {
        let args = &mut Arguments {};
        placeholder(0.0, args);
    }

    #[test]
    #[should_panic(expected = "the derivative is NaN")]
    fn panics_on_nan_1() {
        let args = &mut Arguments {};
        deriv_approx_eq(f64::NAN, 1.5, args, 1e-1, &placeholder);
    }

    #[test]
    #[should_panic(expected = "the derivative is Inf")]
    fn panics_on_inf_1() {
        let args = &mut Arguments {};
        deriv_approx_eq(f64::INFINITY, 1.5, args, 1e-1, &placeholder);
    }

    #[test]
    #[should_panic(expected = "the numerical derivative is NaN")]
    fn panics_on_nan_2() {
        let f = |_: f64, _: &mut Arguments| f64::NAN;
        let args = &mut Arguments {};
        deriv_approx_eq(0.0, 1.5, args, 1e-1, f);
    }

    #[test]
    #[should_panic(expected = "the numerical derivative is NaN")]
    fn panics_on_inf_2() {
        let f = |_: f64, _: &mut Arguments| f64::INFINITY; // yields NaN in central deriv because of Inf - Inf
        let args = &mut Arguments {};
        deriv_approx_eq(0.0, 1.5, args, 1e-1, f);
    }

    #[test]
    #[should_panic(expected = "derivative is not approximately equal to numerical value. diff = ")]
    fn panics_on_different_deriv() {
        let f = |x: f64, _: &mut Arguments| x * x / 2.0;
        let args = &mut Arguments {};
        let at_x = 1.5;
        let dfdx = 1.51;
        deriv_approx_eq(dfdx, at_x, args, 1e-2, f);
    }

    #[test]
    fn accepts_approx_equal_deriv() {
        let f = |x: f64, _: &mut Arguments| x * x / 2.0;
        let args = &mut Arguments {};
        let at_x = 1.5;
        let dfdx = 1.501;
        deriv_approx_eq(dfdx, at_x, args, 1e-2, f);
        deriv_approx_eq(dfdx, at_x, args, 1e-2, f);
    }
}
