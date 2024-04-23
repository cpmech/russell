use crate::StrError;

/// Stepsize h for deriv1_backward6
const STEPSIZE_BACKWARD7: f64 = 1e-3;

const C0: f64 = 49.0 / 20.0;
const C1: f64 = -6.0;
const C2: f64 = 15.0 / 2.0;
const C3: f64 = -20.0 / 3.0;
const C4: f64 = 15.0 / 4.0;
const C5: f64 = -6.0 / 5.0;
const C6: f64 = 1.0 / 6.0;

/// Approximates the first derivative using backward difference with 7 points
///
/// Given `f(x)`, approximate:
///
/// ```text
/// df │   
/// —— │   
/// dx │x=at_x
/// ```
pub fn deriv1_backward7<F, A>(at_x: f64, args: &mut A, mut f: F) -> Result<f64, StrError>
where
    F: FnMut(f64, &mut A) -> Result<f64, StrError>,
{
    let h = STEPSIZE_BACKWARD7;
    let f0 = f(at_x, args)?;
    let f1 = f(at_x - h, args)?;
    let f2 = f(at_x - 2.0 * h, args)?;
    let f3 = f(at_x - 3.0 * h, args)?;
    let f4 = f(at_x - 4.0 * h, args)?;
    let f5 = f(at_x - 5.0 * h, args)?;
    let f6 = f(at_x - 6.0 * h, args)?;
    let approximation = (C0 * f0 + C1 * f1 + C2 * f2 + C3 * f3 + C4 * f4 + C5 * f5 + C6 * f6) / h;
    Ok(approximation)
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::deriv1_backward7;
    use crate::check::approx_eq;
    use crate::check::testing;

    #[test]
    fn deriv1_backward7_works() {
        let tests = testing::get_functions();
        println!(
            "{:>10}{:>15}{:>22}{:>11}",
            "function", "numerical", "analytical", "|num-ana|"
        );
        // for test in &[&tests[2]] {
        for test in &tests {
            let args = &mut 0;
            let num = deriv1_backward7(test.at_x, args, test.f).unwrap();
            let ana = (test.g)(test.at_x, args).unwrap();
            println!("{:>10}{:15.9}{:22}{:11.2e}", test.name, num, ana, f64::abs(num - ana),);
            approx_eq(num, ana, test.tol_g * 10.0); // this formula is not as precise as the Central5
        }
    }
}
