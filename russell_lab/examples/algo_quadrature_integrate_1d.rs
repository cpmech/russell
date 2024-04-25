use russell_lab::math::{elliptic_e, PI};
use russell_lab::Quadrature;
use russell_lab::{approx_eq, StrError};

fn main() -> Result<(), StrError> {
    //  Determine the perimeter P of an ellipse of length 2 and width 1
    //
    //      2π
    //     ⌠   ____________________
    // P = │ \╱ ¼ sin²(θ) + cos²(θ)  dθ
    //     ⌡
    //    0

    let mut quad = Quadrature::new();
    let args = &mut 0;
    let (perimeter, _) = quad.integrate(0.0, 2.0 * PI, args, |theta, _| {
        Ok(f64::sqrt(
            0.25 * f64::powi(f64::sin(theta), 2) + f64::powi(f64::cos(theta), 2),
        ))
    })?;
    println!("\nperimeter = {}", perimeter);

    // complete elliptic integral of the second kind E(0.75)
    let ee = elliptic_e(PI / 2.0, 0.75)?;

    // reference solution
    let ref_perimeter = 4.0 * ee;
    approx_eq(perimeter, ref_perimeter, 1e-14);
    Ok(())
}
