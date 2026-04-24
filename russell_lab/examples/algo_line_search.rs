//! Demonstrates line search for optimization
//!
//! This example shows how to use the backtracking Armijo line search algorithm.
//!
//! Line search is typically used in conjunction with gradient-based optimization
//! methods (e.g., gradient descent, Newton's method) to find an appropriate step size.

use russell_lab::*;

fn main() -> Result<(), StrError> {
    println!("========================================");
    println!("Line Search Example: Rosenbrock Function");
    println!("========================================\n");

    struct Args {}
    let args = &mut Args {};

    // Rosenbrock-like function: f(x) = (1-x)^4 + (1-x)^2
    // Minimum at x = 1, f(1) = 0
    let f = |x: f64, _: &mut Args| {
        let d = x - 1.0;
        Ok(d.powi(4) + d.powi(2))
    };

    // Starting point: x = 0
    // f(0) = (-1)^4 + (-1)^2 = 2
    // gradient = 4*(x-1)^3 + 2*(x-1) = 4*(-1) + 2*(-1) = -4 - 2 = -6
    // For steepest descent: direction = -gradient = 6 (toward positive x)
    let x = 0.0;
    let fx = 2.0;
    let direction = 1.0; // descent direction (positive)
    let slope = -6.0; // directional derivative: grad^T * direction

    println!("Starting point: x = {:.6}", x);
    println!("Function value: f(x) = {:.6}", fx);
    println!("Direction: {}", direction);
    println!("Slope (grad^T * direction): {:.6}", slope);
    println!();

    // Perform line search
    let alpha = line_search(x, direction, fx, slope, args, f)?;
    let x_new = x + alpha * direction;

    println!("Line search results:");
    println!("  Step size (alpha): {:.6}", alpha);
    println!("  New point: x = {:.6}", x_new);
    println!("  Expected minimum at x = 1.0");
    println!();

    // Verify the result
    let f_new = f(x_new, args)?;
    println!("  Function value at new point: {:.6}", f_new);
    println!("  Function value decreased: {:.6} -> {:.6}", fx, f_new);

    // Check Armijo condition
    let armijo_target = fx + 1e-4 * alpha * slope;
    let armijo_satisfied = f_new <= armijo_target;
    println!();
    println!("Armijo condition check:");
    println!("  Target: f(x) + c1 * alpha * slope = {:.6}", armijo_target);
    println!("  f(x_new) = {:.6}", f_new);
    println!("  Condition satisfied: {}", armijo_satisfied);

    println!("\n========================================\n");

    // Second example: with custom parameters
    println!("========================================");
    println!("Line Search Example: Custom Parameters");
    println!("========================================\n");

    let f = |x: f64, _: &mut Args| {
        let d = x - 3.0;
        Ok(d.powi(4) + d.powi(2))
    };

    let x = 0.0;
    let fx = 162.0;
    let direction = 1.0;
    let slope = -108.0;

    println!("Starting point: x = {:.6}", x);
    println!("Function value: f(x) = {:.6}", fx);
    println!("Slope: {:.6}", slope);
    println!();

    // Use LineSearcher with custom parameters
    let mut searcher = LineSearcher::new();
    searcher.c1 = 1e-3; // Less strict Armijo condition
    searcher.rho = 0.7; // Slower backtracking

    let (alpha, n_iter) = searcher.search(x, direction, fx, slope, args, f)?;
    let x_new = x + alpha * direction;

    println!("Line search results with custom parameters:");
    println!("  Step size (alpha): {:.6}", alpha);
    println!("  Number of iterations: {}", n_iter);
    println!("  New point: x = {:.6}", x_new);
    println!("  Function value: {:.6}", f(x_new, args)?);

    println!("\n========================================");
    Ok(())
}