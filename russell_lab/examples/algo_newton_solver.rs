//! Demonstrates Newton solver for nonlinear systems
//!
//! This example shows how to use the Newton solver to solve systems of
//! nonlinear equations F(x) = 0.

use russell_lab::*;

fn main() -> Result<(), StrError> {
    println!("========================================");
    println!("Newton Solver Example: Linear System");
    println!("========================================\n");

    solve_linear_system()?;

    println!("\n========================================");
    println!("Newton Solver Example: Rosenbrock System");
    println!("========================================\n");

    solve_rosenbrock()?;

    println!("\n========================================");
    println!("Newton Solver Example: No Line Search");
    println!("========================================\n");

    solve_without_line_search()?;

    Ok(())
}

fn solve_linear_system() -> Result<(), StrError> {
    // Linear system: Ax = b, where A = [[2, 1], [1, 2]], b = [3, 3]
    // Solution: x = [1, 1]
    // F(x) = Ax - b, root at [1, 1]
    let mut x0 = Vector::from(&[0.0, 0.0]);
    let args = &mut ();

    let f = |x: &Vector, out: &mut Vector, _: &mut ()| {
        out[0] = 2.0 * x[0] + 1.0 * x[1] - 3.0;
        out[1] = 1.0 * x[0] + 2.0 * x[1] - 3.0;
        Ok(())
    };

    let jacobian = |j: &mut Matrix, _: &Vector, _: &mut ()| {
        j.set(0, 0, 2.0);
        j.set(0, 1, 1.0);
        j.set(1, 0, 1.0);
        j.set(1, 1, 2.0);
        Ok(())
    };

    let solver = NewtonSolver::new();
    let (x, stats) = solver.solve(&mut x0, args, f, jacobian)?;

    println!("Linear system: Ax = b");
    println!("A = [[2, 1], [1, 2]], b = [3, 3]");
    println!("Expected solution: x = [1, 1]");
    println!("Computed solution: x = [{:.8}, {:.8}]", x[0], x[1]);
    println!("Iterations: {}", stats.n_iterations);

    Ok(())
}

fn solve_rosenbrock() -> Result<(), StrError> {
    // Rosenbrock function F(x,y) = [1-x, 100(y-x²)]
    // Root at (1, 1)
    let mut x0 = Vector::from(&[0.0, 0.0]);
    let args = &mut ();

    let f = |x: &Vector, out: &mut Vector, _: &mut ()| {
        out[0] = 1.0 - x[0];
        out[1] = 100.0 * (x[1] - x[0] * x[0]);
        Ok(())
    };

    let jacobian = |j: &mut Matrix, x: &Vector, _: &mut ()| {
        j.set(0, 0, -1.0);
        j.set(0, 1, 0.0);
        j.set(1, 0, -200.0 * x[0]);
        j.set(1, 1, 100.0);
        Ok(())
    };

    let mut solver = NewtonSolver::new();
    solver.use_line_search = true;
    let (x, stats) = solver.solve(&mut x0, args, f, jacobian)?;

    println!("Rosenbrock function: F(x,y) = [1-x, 100(y-x²)]");
    println!("Expected solution: (x, y) = (1, 1)");
    println!("Computed solution: ({:.8}, {:.8})", x[0], x[1]);
    println!("Iterations: {}", stats.n_iterations);

    Ok(())
}

fn solve_without_line_search() -> Result<(), StrError> {
    // Same linear system, but without line search
    let mut x0 = Vector::from(&[0.0, 0.0]);
    let args = &mut ();

    let f = |x: &Vector, out: &mut Vector, _: &mut ()| {
        out[0] = 2.0 * x[0] + 1.0 * x[1] - 3.0;
        out[1] = 1.0 * x[0] + 2.0 * x[1] - 3.0;
        Ok(())
    };

    let jacobian = |j: &mut Matrix, _: &Vector, _: &mut ()| {
        j.set(0, 0, 2.0);
        j.set(0, 1, 1.0);
        j.set(1, 0, 1.0);
        j.set(1, 1, 2.0);
        Ok(())
    };

    let mut solver = NewtonSolver::new();
    solver.use_line_search = false;
    let (x, stats) = solver.solve(&mut x0, args, f, jacobian)?;

    println!("Linear system without line search");
    println!("Computed solution: x = [{:.8}, {:.8}]", x[0], x[1]);
    println!("Iterations: {}", stats.n_iterations);

    Ok(())
}