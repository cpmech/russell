use super::Vector;
use crate::deriv2_approx_eq;
use crate::StrError;

/// Panics if the second derivative is not approximately equal to a numerical derivative (central differences)
///
/// Verifies:
///
/// ```text
///   →
/// d²v │   
/// ——— │   
/// ds² │s=at_s
/// ```
///
/// The numerical derivative is computed using a using central differences with 9 points
///
/// # Arguments
///
/// * `d2v_ds2` - The analytical derivative value at `at_s`
/// * `at_s` - The point at which the derivative is evaluated
/// * `args` - Additional arguments passed to the function
/// * `tol` - The tolerance for the approximate equality check
/// * `f` - The function to calculate `v(s)` for which the second derivative is evaluated.
///    It must have the signature `f(v: &mut Vector, s: f64, args: &mut A) -> Result<(), StrError>`
///
/// # Panics
///
/// 1. Will panic if NAN, INFINITY, or NEG_INFINITY is found
/// 2. Will panic if the absolute difference of derivative values is greater than the tolerance
/// 3. Will panic if the function `f(v, s, args)` returns an error
///
/// # Notes
///
/// This function calls [deriv2_approx_eq()] internally for each component of `v`.
pub fn vec_deriv2_approx_eq<F, A>(d2v_ds2: &Vector, at_s: f64, args: &mut A, tol: f64, mut f: F)
where
    F: FnMut(&mut Vector, f64, &mut A) -> Result<(), StrError>,
{
    let n = d2v_ds2.dim();
    let mut v = Vector::new(n);
    for i in 0..n {
        deriv2_approx_eq(d2v_ds2[i], at_s, args, tol, |s, a| {
            f(&mut v, s, a)?;
            Ok(v[i])
        });
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::vec_deriv2_approx_eq;
    use crate::StrError;
    use crate::Vector;

    struct Arguments {}

    fn placeholder(_: &mut Vector, _: f64, _: &mut Arguments) -> Result<(), StrError> {
        Err("NOT HERE")
    }

    #[test]
    fn placeholder_returns_error() {
        let args = &mut Arguments {};
        let mut v = Vector::new(1);
        assert_eq!(placeholder(&mut v, 0.0, args).err(), Some("NOT HERE"));
    }

    #[test]
    #[should_panic(expected = "the function returned an error: NOT HERE")]
    fn panics_on_function_error() {
        let args = &mut Arguments {};
        let dv_ds = Vector::new(1);
        vec_deriv2_approx_eq(&dv_ds, 1.5, args, 1e-1, &placeholder);
    }

    #[test]
    #[should_panic(expected = "the derivative is NaN")]
    fn panics_on_nan_1() {
        let args = &mut Arguments {};
        let dv_ds = Vector::from(&[f64::NAN; 1]);
        vec_deriv2_approx_eq(&dv_ds, 1.5, args, 1e-1, &placeholder);
    }

    #[test]
    #[should_panic(expected = "the derivative is Inf")]
    fn panics_on_inf_1() {
        let args = &mut Arguments {};
        let dv_ds = Vector::from(&[f64::INFINITY; 1]);
        vec_deriv2_approx_eq(&dv_ds, 1.5, args, 1e-1, &placeholder);
    }

    #[test]
    #[should_panic(expected = "the numerical derivative is NaN")]
    fn panics_on_nan_2() {
        let f = |v: &mut Vector, _: f64, _: &mut Arguments| {
            v[0] = f64::NAN;
            Ok(())
        };
        let args = &mut Arguments {};
        let dv_ds = Vector::new(1);
        vec_deriv2_approx_eq(&dv_ds, 1.5, args, 1e-1, f);
    }

    #[test]
    #[should_panic(expected = "the numerical derivative is NaN")]
    fn panics_on_inf_2() {
        let f = |v: &mut Vector, _: f64, _: &mut Arguments| {
            v[0] = f64::INFINITY; // yields NaN in central deriv because of Inf - Inf
            Ok(())
        };
        let args = &mut Arguments {};
        let dv_ds = Vector::new(1);
        vec_deriv2_approx_eq(&dv_ds, 1.5, args, 1e-1, f);
    }

    #[test]
    #[should_panic(expected = "derivative is not approximately equal to numerical value. diff = ")]
    fn panics_on_different_deriv_1() {
        let f = |v: &mut Vector, s: f64, _: &mut Arguments| {
            v[0] = s * s / 2.0;
            v[1] = s * s / 2.0;
            Ok(())
        };
        let args = &mut Arguments {};
        let at_s = 1.5;
        let dv_ds = Vector::from(&[1.1, 1.001]);
        vec_deriv2_approx_eq(&dv_ds, at_s, args, 1e-2, f);
    }

    #[test]
    #[should_panic(expected = "derivative is not approximately equal to numerical value. diff = ")]
    fn panics_on_different_deriv_2() {
        let f = |v: &mut Vector, s: f64, _: &mut Arguments| {
            v[0] = s * s / 2.0;
            v[1] = s * s / 2.0;
            Ok(())
        };
        let args = &mut Arguments {};
        let at_s = 1.5;
        let dv_ds = Vector::from(&[1.001, 1.1]);
        vec_deriv2_approx_eq(&dv_ds, at_s, args, 1e-2, f);
    }

    #[test]
    fn accepts_approx_equal_deriv() {
        let f = |v: &mut Vector, s: f64, _: &mut Arguments| {
            v[0] = s * s / 2.0;
            v[1] = s * s / 2.0;
            Ok(())
        };
        let args = &mut Arguments {};
        let at_s = 1.5;
        let dv_ds = Vector::from(&[1.001, 1.001]); // d2fdx2 = 1.0
        vec_deriv2_approx_eq(&dv_ds, at_s, args, 1e-2, f);
    }
}
