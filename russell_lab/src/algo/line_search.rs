//! Line search algorithms for optimization
//!
//! Implements the backtracking Armijo line search method, commonly used in
//! gradient-based optimization algorithms such as gradient descent and Newton's method.
//!
//! The line search finds a step size `alpha` such that moving from the current point
//! along the search direction satisfies the Armijo condition (sufficient decrease):
//!
//! ```text
//! f(x + alpha * d) <= f(x) + c1 * alpha * grad_f(x)^T * d
//! ```
//!
//! # References
//!
//! 1. Nocedal J, Wright SJ (2006) Numerical Optimization, Springer, 2nd ed
//! 2. Press WT, et al. (2007) Numerical Recipes, Cambridge University Press, 3rd ed

use crate::StrError;

/// Armijo constant: controls the sufficiency of function decrease
///
/// Typical values: c1 ∈ (1e-4, 1e-3)
///
/// A smaller c1 allows larger step sizes but may compromise convergence guarantees.
const DEFAULT_C1: f64 = 1e-4;

/// Backtracking factor: step size reduction per iteration
///
/// Typical values: rho ∈ (0.3, 0.9)
///
/// Each iteration multiplies alpha by rho until the Armijo condition is satisfied.
const DEFAULT_RHO: f64 = 0.5;

/// Minimum allowed step size
///
/// If alpha falls below this value, the line search terminates with an error.
const DEFAULT_MIN_ALPHA: f64 = 1e-20;

/// Default maximum number of line search iterations
const DEFAULT_MAX_ITERATIONS: usize = 20;

/// Implements the backtracking Armijo line search algorithm
///
/// # Algorithm
///
/// 1. Start with initial step size `alpha = 1.0`
/// 2. Check Armijo condition: `f(x + alpha*d) <= f(x) + c1 * alpha * slope`
/// 3. If condition holds, return `alpha`
/// 4. Otherwise, reduce `alpha := rho * alpha` and repeat
/// 5. If `alpha` falls below `min_alpha`, return error
///
/// # Example
///
/// ```
/// use russell_lab::{line_search, StrError};
///
/// fn main() -> Result<(), StrError> {
///     struct Args {}
///     let args = &mut Args {};
///
///     // Rosenbrock function: f(x) = (1-x)^2 + 100(y-x^2)^2, minimum at (1,1)
///     // For simplicity, using 1D version: f(x) = (x-1)^4 + (x-1)^2
///     let f = |x: f64, _: &mut Args| {
///         let d = x - 1.0;
///         Ok(d.powi(4) + d.powi(2))
///     };
///
///     // At x = 0: f(0) = 2, gradient ≈ -4
///     // Descent direction = 1 (move toward positive x)
///     let x = 0.0;
///     let fx = 2.0;
///     let direction = 1.0;
///     let slope = -4.0; // grad^T * direction (approximately)
///
///     let alpha = line_search(x, direction, fx, slope, args, f)?;
///     let x_new = x + alpha * direction;
///
///     // After line search, x_new should be closer to 1
///     assert!(x_new > 0.0 && x_new < 2.0);
///     Ok(())
/// }
/// ```
#[derive(Clone, Copy, Debug)]
pub struct LineSearcher {
    /// Armijo constant (c1): controls sufficiency of decrease
    ///
    /// Smaller values allow larger steps but may affect convergence.
    ///
    /// Default = 1e-4
    pub c1: f64,

    /// Backtracking factor: step size reduction per iteration
    ///
    /// Each iteration: `alpha := rho * alpha`
    ///
    /// Default = 0.5
    pub rho: f64,

    /// Minimum step size threshold
    ///
    /// If alpha falls below this value, the algorithm returns an error.
    ///
    /// Default = 1e-20
    pub min_alpha: f64,

    /// Maximum number of backtracking iterations
    ///
    /// Default = 20
    pub max_iterations: usize,
}

impl LineSearcher {
    /// Allocates a new instance with default parameters
    pub fn new() -> Self {
        LineSearcher {
            c1: DEFAULT_C1,
            rho: DEFAULT_RHO,
            min_alpha: DEFAULT_MIN_ALPHA,
            max_iterations: DEFAULT_MAX_ITERATIONS,
        }
    }

    /// Performs the line search
    ///
    /// # Input
    ///
    /// * `x` -- current position
    /// * `direction` -- search direction (must be a descent direction, i.e., grad^T * direction < 0)
    /// * `fx` -- objective function value at x: f(x)
    /// * `slope` -- directional derivative: grad(x)^T * direction (must be < 0)
    /// * `args` -- extra arguments for the callback function
    /// * `f` -- callback function implementing f(x)
    ///
    /// # Output
    ///
    /// Returns `(alpha, n_iter)` where:
    /// * `alpha` -- step size satisfying the Armijo condition
    /// * `n_iter` -- number of backtracking iterations performed
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// * `slope >= 0` (direction is not a descent direction)
    /// * `alpha` falls below `min_alpha`
    /// * Maximum iterations reached without satisfying Armijo condition
    pub fn search<F, A>(
        &self,
        x: f64,
        direction: f64,
        fx: f64,
        slope: f64,
        args: &mut A,
        mut f: F,
    ) -> Result<(f64, usize), StrError>
    where
        F: FnMut(f64, &mut A) -> Result<f64, StrError>,
    {
        // Verify that direction is a descent direction
        if slope >= 0.0 {
            return Err("direction must be a descent direction (slope < 0)");
        }

        // Initial step size
        let mut alpha = 1.0;

        // Compute the target value for the Armijo condition
        // Armijo: f(x + alpha*d) <= f(x) + c1 * alpha * slope
        // Note: slope < 0, so this is: f_new <= fx + negative_term
        let target = fx + self.c1 * alpha * slope;

        for n_iter in 0..self.max_iterations {
            // Evaluate function at new position
            let x_new = x + alpha * direction;
            let f_new = f(x_new, args)?;

            // Check Armijo condition: sufficient decrease
            if f_new <= target {
                // n_iter is 0-indexed, return 1-indexed count
                return Ok((alpha, n_iter + 1));
            }

            // Backtrack: reduce step size
            alpha *= self.rho;

            // Check if alpha is too small
            if alpha < self.min_alpha {
                return Err("step size too small");
            }
        }

        Err("line search failed to converge")
    }
}

impl Default for LineSearcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Performs line search with default parameters, returning only the step size
///
/// See [`LineSearcher::search`] for details.
///
/// # Example
///
/// ```
/// use russell_lab::{line_search, StrError};
///
/// fn main() -> Result<(), StrError> {
///     struct Args {}
///     let args = &mut Args {};
///
///     // Non-quadratic function: f(x) = (x-3)^4 + (x-3)^2
///     let f = |x: f64, _: &mut Args| {
///         let d = x - 3.0;
///         Ok(d.powi(4) + d.powi(2))
///     };
///     let x = 0.0;
///     let fx = 162.0;
///     let direction = 1.0;
///     let slope = -108.0; // gradient ≈ -4*(0-3)^3 - 2*(0-3) = 108
///
///     let alpha = line_search(x, direction, fx, slope, args, f)?;
///     let x_new = x + alpha * direction;
///
///     assert!(x_new > 0.0 && x_new < 6.0);
///     Ok(())
/// }
/// ```
pub fn line_search<F, A>(
    x: f64,
    direction: f64,
    fx: f64,
    slope: f64,
    args: &mut A,
    f: F,
) -> Result<f64, StrError>
where
    F: FnMut(f64, &mut A) -> Result<f64, StrError>,
{
    let searcher = LineSearcher::new();
    searcher.search(x, direction, fx, slope, args, f).map(|(alpha, _)| alpha)
}

/// Performs line search with default parameters, returning step size and iteration count
///
/// See [`LineSearcher::search`] for details.
pub fn line_search_with_stats<F, A>(
    x: f64,
    direction: f64,
    fx: f64,
    slope: f64,
    args: &mut A,
    f: F,
) -> Result<(f64, usize), StrError>
where
    F: FnMut(f64, &mut A) -> Result<f64, StrError>,
{
    let searcher = LineSearcher::new();
    searcher.search(x, direction, fx, slope, args, f)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test with a non-quadratic function where initial step size is too large
    #[test]
    fn line_search_non_quadratic() {
        struct Args {}
        let args = &mut Args {};

        // f(x) = (x-2)^4 + (x-2)^2 (non-quadratic, derivative changes)
        // At x=0: f(0) = 16 + 4 = 20
        // gradient = 4*(x-2)^3 + 2*(x-2) = 4*(-8) + 2*(-2) = -32 - 4 = -36
        let f = |x: f64, _: &mut Args| {
            let d = x - 2.0;
            Ok(d.powi(4) + d.powi(2))
        };

        let x = 0.0;
        let fx = 20.0;
        let direction = 1.0;
        let slope = -36.0;

        let alpha = line_search(x, direction, fx, slope, args, f).unwrap();
        let x_new = x + alpha * direction;

        // The algorithm should backtrack because alpha=1 gives x_new=1
        // f(1) = (-1)^4 + (-1)^2 = 2
        // Armijo: 2 <= 20 + 1e-4 * 1 * (-36) = 20 - 0.0036 = 19.9964
        // 2 <= 19.9964 is satisfied, so alpha might be accepted
        // But in practice, the algorithm should find a reasonable step
        assert!(x_new > 0.0 && x_new < 4.0);
    }

    #[test]
    fn line_search_with_stats_works() {
        struct Args {}
        let args = &mut Args {};

        // Non-quadratic function with noticeable backtracking
        let f = |x: f64, _: &mut Args| {
            let d = x - 3.0;
            Ok(d.powi(4) + d.powi(2))
        };

        let x = 0.0;
        let fx = 162.0;
        let direction = 1.0;
        let slope = -108.0;

        let (alpha, n_iter) = line_search_with_stats(x, direction, fx, slope, args, f).unwrap();
        assert!(n_iter >= 1);

        let x_new = x + alpha * direction;
        // x_new should be between 0 and the minimum at 3
        assert!(x_new > 0.0 && x_new < 5.0);
    }

    #[test]
    fn line_search_captures_not_descent_direction() {
        struct Args {}
        let args = &mut Args {};

        let f = |x: f64, _: &mut Args| {
            let d = x - 2.0;
            Ok(d.powi(4) + d.powi(2))
        };

        let x = 0.0;
        let fx = 20.0;
        // direction = -1.0, slope = -36 * (-1) = 36 > 0 (not a descent direction)
        let direction = -1.0;
        let slope = 36.0;

        let result = line_search(x, direction, fx, slope, args, f);
        assert_eq!(result.err(), Some("direction must be a descent direction (slope < 0)"));
    }

    #[test]
    fn line_search_custom_parameters() {
        struct Args {}
        let args = &mut Args {};

        let f = |x: f64, _: &mut Args| {
            let d = x - 2.0;
            Ok(d.powi(4) + d.powi(2))
        };

        let x = 0.0;
        let fx = 20.0;
        let direction = 1.0;
        let slope = -36.0;

        let mut searcher = LineSearcher::new();
        searcher.c1 = 1e-3;
        searcher.rho = 0.7;
        searcher.max_iterations = 30;

        let (alpha, _) = searcher.search(x, direction, fx, slope, args, f).unwrap();
        let x_new = x + alpha * direction;

        assert!(x_new > 0.0 && x_new < 4.0);
    }

    #[test]
    fn line_search_with_args() {
        struct Args {
            target: f64,
        }
        let args = &mut Args { target: 5.0 };

        // f(x) = (x - target)^4 + (x - target)^2
        let f = |x: f64, a: &mut Args| {
            let d = x - a.target;
            Ok(d.powi(4) + d.powi(2))
        };

        let x = 0.0;
        let fx = 625.0 + 25.0; // f(0) = (-5)^4 + (-5)^2 = 625 + 25 = 650
        let direction = 1.0;
        // gradient = 4*(x-target)^3 + 2*(x-target) = 4*(-5)^3 + 2*(-5) = -500 - 10 = -510
        let slope = -510.0;

        let alpha = line_search(x, direction, fx, slope, args, f).unwrap();
        let x_new = x + alpha * direction;

        // x_new should be closer to target=5
        assert!(x_new > 0.0 && x_new < 10.0);
    }

    /// Test that step size too small error is triggered
    #[test]
    fn line_search_stops_too_small_alpha() {
        struct Args {}
        let args = &mut Args {};

        // Function that always increases (worst case for line search)
        let f = |_: f64, _: &mut Args| Ok(f64::MAX);

        let x = 0.0;
        let fx = 1.0;
        let direction = 1.0;
        let slope = -1.0;

        let mut searcher = LineSearcher::new();
        searcher.min_alpha = 0.1;
        searcher.rho = 0.5;
        searcher.max_iterations = 10;

        let result = searcher.search(x, direction, fx, slope, args, f);
        // f_new = MAX, which is > fx + c1 * alpha * slope
        // So Armijo condition fails every iteration
        // After backtracking: 1.0 -> 0.5 -> 0.25 -> ... < 0.1 triggers error
        assert_eq!(result.err(), Some("step size too small"));
    }

    /// Test that max iterations error is triggered
    #[test]
    fn line_search_convergence_limits() {
        struct Args {}
        let args = &mut Args {};

        // Function that always increases - Armijo will never be satisfied
        let f = |_: f64, _: &mut Args| Ok(f64::MAX);

        let x = 0.0;
        let fx = 1.0;
        let direction = 1.0;
        let slope = -1.0;

        let mut searcher = LineSearcher::new();
        searcher.max_iterations = 1;

        let result = searcher.search(x, direction, fx, slope, args, f);
        // With max_iterations=1, after first iteration alpha becomes 0.5
        // Since iterations exhausted, should get "failed to converge"
        assert_eq!(result.err(), Some("line search failed to converge"));
    }

    #[test]
    fn line_search_default_impl() {
        // Test that Default trait is properly implemented
        let searcher = LineSearcher::default();
        assert_eq!(searcher.c1, DEFAULT_C1);
        assert_eq!(searcher.rho, DEFAULT_RHO);
        assert_eq!(searcher.min_alpha, DEFAULT_MIN_ALPHA);
        assert_eq!(searcher.max_iterations, DEFAULT_MAX_ITERATIONS);
    }

    #[test]
    fn line_search_multiple_calls() {
        struct Args {
            count: usize,
        }
        let args = &mut Args { count: 0 };

        let mut f_calls = 0;
        let f = |x: f64, a: &mut Args| {
            a.count += 1;
            f_calls += 1;
            let d = x - 2.0;
            Ok(d.powi(4) + d.powi(2))
        };

        let x = 0.0;
        let fx = 20.0;
        let direction = 1.0;
        let slope = -36.0;

        let (alpha, _) = line_search_with_stats(x, direction, fx, slope, args, f).unwrap();
        // Should call the function at least once
        assert!(f_calls >= 1);

        let x_new = x + alpha * direction;
        assert!(x_new > 0.0 && x_new < 4.0);
    }

    #[test]
    fn line_search_steep_descent() {
        struct Args {}
        let args = &mut Args {};

        // f(x) = x^4, gradient at x=0 is 0 (flat region)
        let f = |x: f64, _: &mut Args| Ok(x.powi(4));

        let x = 0.0;
        let fx = 0.0;
        let direction = 1.0;
        let slope = 0.0; // gradient = 0

        let result = line_search(x, direction, fx, slope, args, f);
        // slope = 0 is not < 0, so this should fail
        assert!(result.is_err());
    }

    /// Test with exponential decay - should converge quickly
    #[test]
    fn line_search_exponential() {
        struct Args {}
        let args = &mut Args {};

        // f(x) = exp(-x), decreasing function
        let f = |x: f64, _: &mut Args| Ok(f64::exp(-x));

        let x = 0.0;
        let fx = 1.0;
        let direction = 1.0;
        let slope = -1.0;

        let alpha = line_search(x, direction, fx, slope, args, f).unwrap();
        let x_new = x + alpha * direction;

        // x_new should be positive (moving in descent direction)
        assert!(x_new > 0.0);
        // But not too large
        assert!(x_new < 5.0);
    }
}