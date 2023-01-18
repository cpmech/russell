use crate::deriv_central5;

/// Panics if derivative is not approximately equal to numerical derivative
pub fn deriv_approx_eq<F, A>(dfdx: f64, at_x: f64, args: &mut A, tol: f64, f: F)
where
    F: FnMut(f64, &mut A) -> f64,
{
    let dfdx_num = deriv_central5(at_x, args, f);
    let diff = f64::abs(dfdx - dfdx_num);
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
