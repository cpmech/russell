use russell_lab::*;

fn main() -> Result<(), StrError> {
    // "4: f(x) = (x - 1)² + 5 sin(x)"
    let f = |x: f64, _: &mut NoArgs| Ok(f64::powi(x - 1.0, 2) + 5.0 * f64::sin(x));
    let args = &mut 0;

    // bracketing
    let mut bracketing = MinBracketing::new();
    bracketing.set_enable_stats(true);
    let bracket = bracketing.basic(-3.0, args, f)?;
    println!("\n(a, b) = ({}, {})", bracket.a, bracket.b);
    println!("\n{}", bracketing.get_stats().unwrap());

    // minimize
    let mut solver = MinSolver::new();
    solver.set_enable_stats(true);
    let xo = solver.brent(bracket.a, bracket.b, args, f)?;
    println!("\noptimal = {}", xo);
    println!("\n{}", bracketing.get_stats().unwrap());
    approx_eq(xo, -0.7790149303951403, 1e-8);
    Ok(())
}
