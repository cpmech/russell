use super::deriv1_approx_eq;
use crate::StrError;

/// Panics if the first derivative is not approximately equal to a numerical derivative (central differences)
///
/// Verifies:
///
/// ```text
///  →
/// dv │   
/// —— │   
/// ds │s=at_s
/// ```
///
/// The numerical derivative is computed using a using central differences with 5 points
///
/// # Arguments
///
/// * `dvds` - The analytical derivative value at `at_s`
/// * `at_s` - The point at which the derivative is evaluated
/// * `args` - Additional arguments passed to the function `v(s, args)`
/// * `tol` - The tolerance for the approximate equality check
/// * `f` - The function to calculate `v(s)` for which the derivative is evaluated.
///    It must have the signature `f(v: &mut [f64], s: f64, args: &mut A) -> Result<(), StrError>`
///
/// # Panics
///
/// 1. Will panic if NAN, INFINITY, or NEG_INFINITY is found
/// 2. Will panic if the absolute difference of derivative values is greater than the tolerance
/// 3. Will panic if the function `f(v, s, args)` returns an error
///
/// # Notes
///
/// This function calls [deriv1_approx_eq()] internally for each component of `v`.
pub fn deriv1_vec_approx_eq<F, A>(dvds: &[f64], at_s: f64, args: &mut A, tol: f64, mut f: F)
where
    F: FnMut(&mut [f64], f64, &mut A) -> Result<(), StrError>,
{
    let mut v = vec![0.0; dvds.len()];
    for i in 0..dvds.len() {
        deriv1_approx_eq(dvds[i], at_s, args, tol, |s, a| {
            f(&mut v, s, a)?;
            Ok(v[i])
        });
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::deriv1_vec_approx_eq;
    use crate::StrError;

    struct Arguments {}

    fn placeholder(_: &mut [f64], _: f64, _: &mut Arguments) -> Result<(), StrError> {
        Err("NOT HERE")
    }

    #[test]
    fn placeholder_returns_error() {
        let args = &mut Arguments {};
        let mut v = vec![0.0; 1];
        assert_eq!(placeholder(&mut v, 0.0, args).err(), Some("NOT HERE"));
    }

    #[test]
    #[should_panic(expected = "the function returned an error: NOT HERE")]
    fn panics_on_function_error() {
        let args = &mut Arguments {};
        let dvds = vec![0.0; 1];
        deriv1_vec_approx_eq(&dvds, 1.5, args, 1e-1, &placeholder);
    }

    #[test]
    #[should_panic(expected = "the derivative is NaN")]
    fn panics_on_nan_1() {
        let args = &mut Arguments {};
        let dvds = vec![f64::NAN; 1];
        deriv1_vec_approx_eq(&dvds, 1.5, args, 1e-1, &placeholder);
    }

    #[test]
    #[should_panic(expected = "the derivative is Inf")]
    fn panics_on_inf_1() {
        let args = &mut Arguments {};
        let dvds = vec![f64::INFINITY; 1];
        deriv1_vec_approx_eq(&dvds, 1.5, args, 1e-1, &placeholder);
    }

    #[test]
    #[should_panic(expected = "the numerical derivative is NaN")]
    fn panics_on_nan_2() {
        let f = |v: &mut [f64], _: f64, _: &mut Arguments| {
            v[0] = f64::NAN;
            Ok(())
        };
        let args = &mut Arguments {};
        let dvds = vec![0.0; 1];
        deriv1_vec_approx_eq(&dvds, 1.5, args, 1e-1, f);
    }

    #[test]
    #[should_panic(expected = "the numerical derivative is NaN")]
    fn panics_on_inf_2() {
        let f = |v: &mut [f64], _: f64, _: &mut Arguments| {
            v[0] = f64::INFINITY; // yields NaN in central deriv because of Inf - Inf
            Ok(())
        };
        let args = &mut Arguments {};
        let dvds = vec![0.0; 1];
        deriv1_vec_approx_eq(&dvds, 1.5, args, 1e-1, f);
    }

    #[test]
    #[should_panic(expected = "derivative is not approximately equal to numerical value. diff = ")]
    fn panics_on_different_deriv_1() {
        let f = |v: &mut [f64], s: f64, _: &mut Arguments| {
            v[0] = s * s / 2.0;
            v[1] = s * s / 2.0;
            Ok(())
        };
        let args = &mut Arguments {};
        let at_s = 1.5;
        let dvds = &[1.51, 1.501];
        deriv1_vec_approx_eq(dvds, at_s, args, 1e-2, f);
    }

    #[test]
    #[should_panic(expected = "derivative is not approximately equal to numerical value. diff = ")]
    fn panics_on_different_deriv_2() {
        let f = |v: &mut [f64], s: f64, _: &mut Arguments| {
            v[0] = s * s / 2.0;
            v[1] = s * s / 2.0;
            Ok(())
        };
        let args = &mut Arguments {};
        let at_s = 1.5;
        let dvds = &[1.501, 1.51];
        deriv1_vec_approx_eq(dvds, at_s, args, 1e-2, f);
    }

    #[test]
    fn accepts_approx_equal_deriv() {
        let f = |v: &mut [f64], s: f64, _: &mut Arguments| {
            v[0] = s * s / 2.0;
            v[1] = s * s / 2.0;
            Ok(())
        };
        let args = &mut Arguments {};
        let at_s = 1.5;
        let dvds = &[1.501, 1.501];
        deriv1_vec_approx_eq(dvds, at_s, args, 1e-2, f);
    }
}
