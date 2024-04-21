use russell_lab::algo::NoArgs;
use russell_lab::check::{deriv1_approx_eq, deriv2_approx_eq};
use russell_lab::{StrError, Vector};

fn main() -> Result<(), StrError> {
    // f(x)
    let f = |x: f64, _: &mut NoArgs| Ok(1.0 / 2.0 - 1.0 / (1.0 + 16.0 * x * x));

    // g(x) = df/dx(x)
    let g = |x: f64, _: &mut NoArgs| Ok((32.0 * x) / f64::powi(1.0 + 16.0 * f64::powi(x, 2), 2));

    // h(x) = d²f/dx²(x)
    let h = |x: f64, _: &mut NoArgs| {
        Ok((-2048.0 * f64::powi(x, 2)) / f64::powi(1.0 + 16.0 * f64::powi(x, 2), 3)
            + 32.0 / f64::powi(1.0 + 16.0 * f64::powi(x, 2), 2))
    };

    let xx = Vector::linspace(-2.0, 2.0, 9)?;
    let args = &mut 0;
    println!("{:>4}{:>23}{:>23}", "x", "df/dx", "d²f/dx²");
    for x in xx {
        let dfdx = g(x, args)?;
        let d2dfx2 = h(x, args)?;
        println!("{:>4}{:>23}{:>23}", x, dfdx, d2dfx2);
        deriv1_approx_eq(dfdx, x, args, 1e-10, f);
        deriv2_approx_eq(d2dfx2, x, args, 1e-9, f);
    }
    Ok(())
}
