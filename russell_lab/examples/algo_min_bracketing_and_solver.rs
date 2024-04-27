use russell_lab::*;

fn main() -> Result<(), StrError> {
    // "4: f(x) = (x - 1)² + 5 sin(x)"
    let f = |x: f64, _: &mut NoArgs| Ok(f64::powi(x - 1.0, 2) + 5.0 * f64::sin(x));
    let args = &mut 0;

    // bracketing
    let bracketing = MinBracketing::new();
    let (bracket, stats) = bracketing.basic(-3.0, args, f)?;
    println!("\n(a, b) = ({}, {})", bracket.a, bracket.b);
    println!("\n{}", stats);

    // minimize
    let solver = MinSolver::new();
    let (xo, stats) = solver.brent(bracket.a, bracket.b, args, f)?;
    println!("\noptimal = {}", xo);
    println!("\n{}", stats);
    approx_eq(xo, -0.7790149303951403, 1e-8);
    Ok(())
}
