//! Line search algorithms for optimization
//!
//! Implements the backtracking Armijo line search method, commonly used in
//! gradient-based optimization algorithms such as gradient descent and Newton's method.
//!
//! The line search finds a step size `alpha` such that moving from the current point
//! along the search direction satisfies the Armijo condition (sufficient decrease):
//!
//! ```text
//! f(x + alpha * p) <= f(x) + c1 * alpha * grad_f(x)^T * p
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
/// 2. Check Armijo condition: `f(x + alpha*p) <= f(x) + c1 * alpha * slope`
/// 3. If condition holds, return `alpha`
/// 4. Otherwise, reduce `alpha := rho * alpha` and repeat
/// 5. If `alpha` falls below `min_alpha`, return error
///
/// # Example
///
/// ```
/// use russell_lab::{LineSearcher, StrError};
///
/// fn main() -> Result<(), StrError> {
///     struct Args {}
///     let args = &mut Args {};
///
///     // f(x) = (x-1)^4 + (x-1)^2, minimum at x=1
///     let f = |x: f64, _: &mut Args| {
///         let d = x - 1.0;
///         Ok(d.powi(4) + d.powi(2))
///     };
///
///     // At x = 0: f(0) = 2, gradient = 4*(-1)^3 + 2*(-1) = -6
///     let x = 0.0;
///     let fx = 2.0;
///     let p = 1.0; // direction towards the minimum (positive direction)
///     let slope = -6.0; // grad_f(0) * p
///
///     let searcher = LineSearcher::new();
///     let (alpha, _) = searcher.search(x, p, fx, slope, args, f)?;
///     let x_new = x + alpha * p;
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

    /// Validates the parameters
    fn validate_params(&self) -> Result<(), StrError> {
        if self.c1 <= 0.0 || self.c1 >= 1.0 {
            return Err("c1 must satisfy 0 < c1 < 1");
        }
        if self.rho <= 0.0 || self.rho >= 1.0 {
            return Err("rho must satisfy 0 < rho < 1");
        }
        if self.min_alpha <= 0.0 {
            return Err("min_alpha must be > 0");
        }
        if self.min_alpha >= 1.0 {
            return Err("min_alpha must be < 1");
        }
        if self.max_iterations == 0 {
            return Err("max_iterations must be ≥ 1");
        }
        Ok(())
    }

    /// Performs the line search
    ///
    /// # Input
    ///
    /// * `x` -- current position
    /// * `p` -- search direction (must be a descent direction, i.e., grad^T * p < 0)
    /// * `fx` -- objective function value at x: f(x)
    /// * `slope` -- directional derivative: grad(x)^T * p (must be < 0)
    /// * `args` -- extra arguments for the callback function
    /// * `f` -- callback function implementing f(x)
    ///
    /// # Output
    ///
    /// Returns `(alpha, n_evals)` where:
    /// * `alpha` -- step size satisfying the Armijo condition
    /// * `n_evals` -- number of function evaluations performed (always >= 1)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// * `c1` is not in `(0, 1)`
    /// * `rho` is not in `(0, 1)`
    /// * `min_alpha` is not in `(0, 1)`
    /// * `max_iterations` is `0`
    /// * `slope >= 0` (direction is not a descent direction)
    /// * `alpha` falls below `min_alpha`
    /// * Maximum iterations reached without satisfying Armijo condition
    pub fn search<F, A>(
        &self,
        x: f64,
        p: f64,
        fx: f64,
        slope: f64,
        args: &mut A,
        mut f: F,
    ) -> Result<(f64, usize), StrError>
    where
        F: FnMut(f64, &mut A) -> Result<f64, StrError>,
    {
        self.validate_params()?;

        // Verify that direction is a descent direction
        if slope >= 0.0 {
            return Err("direction must be a descent direction (slope < 0)");
        }

        // Initial step size
        let mut alpha = 1.0;

        for n_iter in 0..self.max_iterations {
            // Armijo condition right-hand side: f(x) + c1 * alpha * slope
            // slope < 0, so this decreases proportionally to the current step size
            let target = fx + self.c1 * alpha * slope;

            // Evaluate function at new position
            let x_new = x + alpha * p;
            let f_new = f(x_new, args)?;

            // Check Armijo condition: sufficient decrease
            if f_new <= target {
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
///     let fx = 90.0; // f(0) = (0-3)^4 + (0-3)^2 = 81 + 9 = 90
///     let p = 1.0;
///     let slope = -114.0; // grad_f(0) = 4*(0-3)^3 + 2*(0-3) = -108 - 6 = -114
///
///     let alpha = line_search(x, p, fx, slope, args, f)?;
///     let x_new = x + alpha * p;
///
///     assert!(x_new > 0.0 && x_new < 6.0);
///     Ok(())
/// }
/// ```
pub fn line_search<F, A>(x: f64, p: f64, fx: f64, slope: f64, args: &mut A, f: F) -> Result<f64, StrError>
where
    F: FnMut(f64, &mut A) -> Result<f64, StrError>,
{
    let searcher = LineSearcher::new();
    searcher.search(x, p, fx, slope, args, f).map(|(alpha, _)| alpha)
}

/// Performs line search with default parameters, returning step size and iteration count
///
/// See [`LineSearcher::search`] for details.
pub fn line_search_with_stats<F, A>(
    x: f64,
    p: f64,
    fx: f64,
    slope: f64,
    args: &mut A,
    f: F,
) -> Result<(f64, usize), StrError>
where
    F: FnMut(f64, &mut A) -> Result<f64, StrError>,
{
    let searcher = LineSearcher::new();
    searcher.search(x, p, fx, slope, args, f)
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
        let p = 1.0;
        let slope = -36.0;

        let alpha = line_search(x, p, fx, slope, args, f).unwrap();
        let x_new = x + alpha * p;

        // At alpha=1: f(1) = (-1)^4 + (-1)^2 = 2
        // Armijo: 2 <= 20 + 1e-4 * 1 * (-36) = 19.9964 -- satisfied, alpha=1 accepted
        assert_eq!(alpha, 1.0);
        assert!(x_new > 0.0 && x_new < 4.0);
    }

    #[test]
    fn line_search_with_stats_works() {
        struct Args {}
        let args = &mut Args {};

        // Non-quadratic function; alpha=1 is accepted on the first evaluation
        let f = |x: f64, _: &mut Args| {
            let d = x - 3.0;
            Ok(d.powi(4) + d.powi(2))
        };

        // At x=0: f(0) = (0-3)^4 + (0-3)^2 = 81 + 9 = 90
        // grad_f(0) = 4*(0-3)^3 + 2*(0-3) = -108 - 6 = -114
        let x = 0.0;
        let fx = 90.0;
        let p = 1.0;
        let slope = -114.0;

        let (alpha, n_evals) = line_search_with_stats(x, p, fx, slope, args, f).unwrap();
        // f(1) = (1-3)^4 + (1-3)^2 = 20 <= 90 + 1e-4 * 1 * (-114) = 89.9886 -- accepted immediately
        assert_eq!(n_evals, 1);

        let x_new = x + alpha * p;
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
        // p = -1.0, slope = -36 * (-1) = 36 > 0 (not a descent direction)
        let p = -1.0;
        let slope = 36.0;

        let result = line_search(x, p, fx, slope, args, f);
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
        let p = 1.0;
        let slope = -36.0;

        let mut searcher = LineSearcher::new();
        searcher.c1 = 1e-3;
        searcher.rho = 0.7;
        searcher.max_iterations = 30;

        let (alpha, _) = searcher.search(x, p, fx, slope, args, f).unwrap();
        let x_new = x + alpha * p;

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
        let p = 1.0;
        // gradient = 4*(x-target)^3 + 2*(x-target) = 4*(-5)^3 + 2*(-5) = -500 - 10 = -510
        let slope = -510.0;

        let alpha = line_search(x, p, fx, slope, args, f).unwrap();
        let x_new = x + alpha * p;

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
        let p = 1.0;
        let slope = -1.0;

        let mut searcher = LineSearcher::new();
        searcher.min_alpha = 0.1;
        searcher.rho = 0.5;
        searcher.max_iterations = 10;

        let result = searcher.search(x, p, fx, slope, args, f);
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
        let p = 1.0;
        let slope = -1.0;

        let mut searcher = LineSearcher::new();
        searcher.max_iterations = 1;

        let result = searcher.search(x, p, fx, slope, args, f);
        // With max_iterations=1, after first iteration alpha becomes 0.5
        // Since iterations exhausted, should get "failed to converge"
        assert_eq!(result.err(), Some("line search failed to converge"));
    }

    #[test]
    fn line_search_multiple_calls() {
        struct Args {
            count: usize,
        }
        fn f(x: f64, a: &mut Args) -> Result<f64, StrError> {
            a.count += 1;
            let d = x - 2.0;
            Ok(d.powi(4) + d.powi(2))
        }

        let searcher = LineSearcher::new();
        let args = &mut Args { count: 0 };

        // First call from x=0, moving right: alpha=1 accepted on first evaluation
        // f(1) = 2 <= 20 + 1e-4 * 1 * (-36) = 19.9964
        let (alpha1, n_evals1) = searcher.search(0.0, 1.0, 20.0, -36.0, args, f).unwrap();
        assert_eq!(n_evals1, 1);
        assert_eq!(args.count, 1);
        assert!(alpha1 > 0.0 && alpha1 <= 1.0);

        // Second call from x=4, moving left: alpha=1 accepted on first evaluation
        // f(3) = 2 <= 20 + 1e-4 * 1 * (-36) = 19.9964
        let (alpha2, n_evals2) = searcher.search(4.0, -1.0, 20.0, -36.0, args, f).unwrap();
        assert_eq!(n_evals2, 1);
        assert_eq!(args.count, 2);
        assert!(alpha2 > 0.0 && alpha2 <= 1.0);
    }

    #[test]
    fn line_search_captures_zero_slope() {
        struct Args {}
        let args = &mut Args {};

        // f(x) = x^4, gradient is 0 at x=0 (flat region, not a descent direction)
        let f = |x: f64, _: &mut Args| Ok(x.powi(4));

        let x = 0.0;
        let fx = 0.0;
        let p = 1.0;
        let slope = 0.0; // gradient = 0, not a descent direction

        let result = line_search(x, p, fx, slope, args, f);
        assert_eq!(result.err(), Some("direction must be a descent direction (slope < 0)"));
    }

    #[test]
    fn line_search_validate_params_works() {
        struct Args {}
        let args = &mut Args {};
        let f = |x: f64, _: &mut Args| Ok(x);

        let mut searcher = LineSearcher::new();

        // invalid c1
        searcher.c1 = 0.0;
        assert_eq!(
            searcher.search(0.0, 1.0, 1.0, -1.0, args, f).err(),
            Some("c1 must satisfy 0 < c1 < 1")
        );
        searcher.c1 = 1.0;
        assert_eq!(
            searcher.search(0.0, 1.0, 1.0, -1.0, args, f).err(),
            Some("c1 must satisfy 0 < c1 < 1")
        );
        searcher.c1 = DEFAULT_C1;

        // invalid rho
        searcher.rho = 0.0;
        assert_eq!(
            searcher.search(0.0, 1.0, 1.0, -1.0, args, f).err(),
            Some("rho must satisfy 0 < rho < 1")
        );
        searcher.rho = 1.0;
        assert_eq!(
            searcher.search(0.0, 1.0, 1.0, -1.0, args, f).err(),
            Some("rho must satisfy 0 < rho < 1")
        );
        searcher.rho = DEFAULT_RHO;

        // invalid min_alpha
        searcher.min_alpha = 0.0;
        assert_eq!(
            searcher.search(0.0, 1.0, 1.0, -1.0, args, f).err(),
            Some("min_alpha must be > 0")
        );
        searcher.min_alpha = 1.0;
        assert_eq!(
            searcher.search(0.0, 1.0, 1.0, -1.0, args, f).err(),
            Some("min_alpha must be < 1")
        );
        searcher.min_alpha = DEFAULT_MIN_ALPHA;

        // invalid max_iterations
        searcher.max_iterations = 0;
        assert_eq!(
            searcher.search(0.0, 1.0, 1.0, -1.0, args, f).err(),
            Some("max_iterations must be ≥ 1")
        );
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
        let p = 1.0;
        let slope = -1.0;

        let alpha = line_search(x, p, fx, slope, args, f).unwrap();
        let x_new = x + alpha * p;

        // x_new should be positive (moving in descent direction)
        assert!(x_new > 0.0);
        // But not too large
        assert!(x_new < 5.0);
    }
}
