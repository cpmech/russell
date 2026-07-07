//! Demonstrates Newton solver for nonlinear systems
//!
//! This example shows how to use the Newton solver to solve systems of
//! nonlinear equations F(x) = 0.

use russell_lab::*;

fn main() -> Result<(), StrError> {
    println!("========================================");
    println!("Newton Solver Example");
    println!("========================================\n");
    solve_linear_system()?;

    println!("\n========================================");
    println!("Newton Solver Example: With Line Search");
    println!("========================================\n");
    solve_with_line_search()?;

    println!("\n========================================");
    println!("Newton Solver Example: Rosenbrock System");
    println!("========================================\n");
    solve_rosenbrock()?;

    Ok(())
}

fn solve_linear_system() -> Result<(), StrError> {
    // Linear system: Ax = b, where A = [[2, 1], [1, 2]], b = [3, 3]
    // Solution: x = [1, 1]
    // F(x) = Ax - b, root at [1, 1]

    let calc_f = |r: &mut Vector, x: &Vector, _: &mut ()| {
        r[0] = 2.0 * x[0] + 1.0 * x[1] - 3.0;
        r[1] = 1.0 * x[0] + 2.0 * x[1] - 3.0;
        Ok(())
    };

    let calc_jj = |jj: &mut Matrix, _: &Vector, _: &mut ()| {
        jj.set(0, 0, 2.0);
        jj.set(0, 1, 1.0);
        jj.set(1, 0, 1.0);
        jj.set(1, 1, 2.0);
        Ok(())
    };

    // Initial guess far from the root
    let mut x = Vector::from(&[0.0, 0.0]);
    let args = &mut ();

    let mut solver = NewtonSolver::new(2)?;
    solver.solve(&mut x, args, calc_f, calc_jj)?;

    println!("Linear system: Ax = b");
    println!("A = [[2, 1], [1, 2]], b = [3, 3]");
    println!("Expected solution: x = [1, 1]");
    println!("Computed solution: x = [{:.8}, {:.8}]", x[0], x[1]);
    println!("Iterations: {}", solver.stats.n_iterations);

    Ok(())
}

fn solve_with_line_search() -> Result<(), StrError> {
    // Same linear system, but without line search

    let calc_f = |r: &mut Vector, x: &Vector, _: &mut ()| {
        r[0] = 2.0 * x[0] + 1.0 * x[1] - 3.0;
        r[1] = 1.0 * x[0] + 2.0 * x[1] - 3.0;
        Ok(())
    };

    let calc_jj = |jj: &mut Matrix, _: &Vector, _: &mut ()| {
        jj.set(0, 0, 2.0);
        jj.set(0, 1, 1.0);
        jj.set(1, 0, 1.0);
        jj.set(1, 1, 2.0);
        Ok(())
    };

    // Initial guess far from the root
    let mut x = Vector::from(&[0.0, 0.0]);
    let args = &mut ();

    let mut solver = NewtonSolver::new(2)?;
    solver.set_use_line_search(true);
    solver.solve(&mut x, args, calc_f, calc_jj)?;

    println!("Linear system: Ax = b");
    println!("A = [[2, 1], [1, 2]], b = [3, 3]");
    println!("Expected solution: x = [1, 1]");
    println!("Computed solution (with line search): x = [{:.8}, {:.8}]", x[0], x[1]);
    println!("Iterations: {}", solver.stats.n_iterations);

    Ok(())
}

fn solve_rosenbrock() -> Result<(), StrError> {
    // Rosenbrock function F(x,y) = [1-x, 100(y-x²)]
    // Root at (1, 1)

    let calc_f = |r: &mut Vector, x: &Vector, _: &mut ()| {
        r[0] = 1.0 - x[0];
        r[1] = 100.0 * (x[1] - x[0] * x[0]);
        Ok(())
    };

    let calc_jj = |jj: &mut Matrix, x: &Vector, _: &mut ()| {
        jj.set(0, 0, -1.0);
        jj.set(0, 1, 0.0);
        jj.set(1, 0, -200.0 * x[0]);
        jj.set(1, 1, 100.0);
        Ok(())
    };

    // Initial guess far from the root
    let mut x = Vector::from(&[0.0, 0.0]);
    let args = &mut ();

    let mut solver = NewtonSolver::new(2)?;
    solver.set_use_line_search(false);
    solver.solve(&mut x, args, calc_f, calc_jj)?;

    println!("Rosenbrock function: F(x,y) = [1-x, 100(y-x²)]");
    println!("Expected solution: (x, y) = (1, 1)");
    println!("Computed solution: ({:.8}, {:.8})", x[0], x[1]);
    println!("Iterations: {}", solver.stats.n_iterations);

    Ok(())
}
