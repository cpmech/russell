use russell_lab::math::PI;
use russell_lab::*;

fn main() -> Result<(), StrError> {
    let args = &mut 0;

    // minimum
    let mut solver = MinSolver::new();
    solver.set_enable_stats(true);
    let xo = solver.brent(0.1, 0.3, args, |x, _| {
        Ok(1.0 / (1.0 - f64::exp(-2.0 * x) * f64::powi(f64::sin(5.0 * PI * x), 2)) - 1.5)
    })?;
    println!("\nx_optimal = {:?}", xo);
    println!("\n{}", solver.get_stats().unwrap());

    // root
    let mut solver = RootFinder::new();
    solver.set_enable_stats(true);
    let xo = solver.brent(0.3, 0.4, args, |x, _| {
        Ok(1.0 / (1.0 - f64::exp(-2.0 * x) * f64::powi(f64::sin(5.0 * PI * x), 2)) - 1.5)
    })?;
    println!("\nx_root = {:?}", xo);
    println!("\n{}", solver.get_stats().unwrap());
    Ok(())
}
