use crate::StrError;

/// Stepsize h for deriv2_forward8
const STEPSIZE_FORWARD8: f64 = 1e-3;

const C0: f64 = 469.0 / 90.0;
const C1: f64 = -223.0 / 10.0;
const C2: f64 = 879.0 / 20.0;
const C3: f64 = -949.0 / 18.0;
const C4: f64 = 41.0;
const C5: f64 = -201.0 / 10.0;
const C6: f64 = 1019.0 / 180.0;
const C7: f64 = -7.0 / 10.0;

/// Approximates the second derivative using forward difference with 8 points
///
/// Given `f(x)`, approximate:
///
/// ```text
/// d²f │   
/// ——— │   
/// dx² │x=at_x
/// ```
pub fn deriv2_forward8<F, A>(at_x: f64, args: &mut A, mut f: F) -> Result<f64, StrError>
where
    F: FnMut(f64, &mut A) -> Result<f64, StrError>,
{
    let h = STEPSIZE_FORWARD8;
    let hh = h * h;
    let f0 = f(at_x, args)?;
    let f1 = f(at_x + h, args)?;
    let f2 = f(at_x + 2.0 * h, args)?;
    let f3 = f(at_x + 3.0 * h, args)?;
    let f4 = f(at_x + 4.0 * h, args)?;
    let f5 = f(at_x + 5.0 * h, args)?;
    let f6 = f(at_x + 6.0 * h, args)?;
    let f7 = f(at_x + 7.0 * h, args)?;
    let approximation = (C0 * f0 + C1 * f1 + C2 * f2 + C3 * f3 + C4 * f4 + C5 * f5 + C6 * f6 + C7 * f7) / hh;
    Ok(approximation)
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::deriv2_forward8;
    use crate::check::approx_eq;
    use crate::check::testing;

    #[test]
    fn deriv2_forward9_works() {
        let tests = testing::get_functions();
        println!(
            "{:>10}{:>15}{:>22}{:>11}",
            "function", "numerical", "analytical", "|num-ana|"
        );
        for test in &tests {
            let args = &mut 0;
            let num = deriv2_forward8(test.at_x, args, test.f).unwrap();
            let ana = (test.h)(test.at_x, args).unwrap();
            println!("{:>10}{:15.9}{:22}{:11.2e}", test.name, num, ana, f64::abs(num - ana),);
            approx_eq(num, ana, test.tol_h_one_sided);
        }
    }
}
