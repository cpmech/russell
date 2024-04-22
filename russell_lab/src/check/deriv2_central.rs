use crate::StrError;

/// Stepsize h for deriv2_central8
const STEPSIZE_CENTRAL8: f64 = 1e-3;

const C4: f64 = -1.0 / 560.0;
const C3: f64 = 8.0 / 315.0;
const C2: f64 = -1.0 / 5.0;
const C1: f64 = 8.0 / 5.0;
const C0: f64 = -205.0 / 72.0;

/// Approximates the second derivative using central difference with 8 points
///
/// Given `f(x)`, approximate:
///
/// ```text
/// d²f │   
/// ——— │   
/// dx² │x=at_x
/// ```
pub fn deriv2_central8<F, A>(at_x: f64, args: &mut A, mut f: F) -> Result<f64, StrError>
where
    F: FnMut(f64, &mut A) -> Result<f64, StrError>,
{
    let h = STEPSIZE_CENTRAL8;
    let hh = h * h;
    let fm4 = f(at_x - 4.0 * h, args)?;
    let fm3 = f(at_x - 3.0 * h, args)?;
    let fm2 = f(at_x - 2.0 * h, args)?;
    let fm1 = f(at_x - 1.0 * h, args)?;
    let fzz = f(at_x, args)?;
    let fp1 = f(at_x + 1.0 * h, args)?;
    let fp2 = f(at_x + 2.0 * h, args)?;
    let fp3 = f(at_x + 3.0 * h, args)?;
    let fp4 = f(at_x + 4.0 * h, args)?;
    let approximation =
        (C4 * fm4 + C3 * fm3 + C2 * fm2 + C1 * fm1 + C0 * fzz + C1 * fp1 + C2 * fp2 + C3 * fp3 + C4 * fp4) / hh;
    Ok(approximation)
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::deriv2_central8;
    use crate::check::approx_eq;
    use crate::check::testing;

    #[test]
    fn deriv2_central8_works() {
        let tests = testing::get_functions();
        println!(
            "{:>10}{:>15}{:>22}{:>11}",
            "function", "numerical", "analytical", "|num-ana|"
        );
        // for test in &[&tests[2]] {
        for test in &tests {
            let args = &mut 0;
            let num = deriv2_central8(test.at_x, args, test.f).unwrap();
            let ana = (test.h)(test.at_x, args).unwrap();
            println!("{:>10}{:15.9}{:22}{:11.2e}", test.name, num, ana, f64::abs(num - ana),);
            approx_eq(num, ana, test.tol_h);
        }
    }
}
