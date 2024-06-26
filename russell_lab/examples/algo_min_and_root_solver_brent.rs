use russell_lab::math::PI;
use russell_lab::*;

fn main() -> Result<(), StrError> {
    let args = &mut 0;

    // minimum
    let solver = MinSolver::new();
    let (xo, stats) = solver.brent(0.1, 0.3, args, |x, _| {
        Ok(1.0 / (1.0 - f64::exp(-2.0 * x) * f64::powi(f64::sin(5.0 * PI * x), 2)) - 1.5)
    })?;
    println!("\nx_optimal = {:?}", xo);
    println!("\n{}", stats);

    // root
    let solver = RootFinding::new();
    let (xo, stats) = solver.brent_find(0.3, 0.4, args, |x, _| {
        Ok(1.0 / (1.0 - f64::exp(-2.0 * x) * f64::powi(f64::sin(5.0 * PI * x), 2)) - 1.5)
    })?;
    println!("\nx_root = {:?}", xo);
    println!("\n{}", stats);
    Ok(())
}
