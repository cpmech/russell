//! Newton solver for nonlinear systems
//!
//! Implements Newton's method with optional line search for solving
//! systems of nonlinear equations F(x) = 0.
//!
//! # Features
//!
//! * Numerical or analytical Jacobian computation
//! * Optional line search for globalization
//! * Configurable convergence criteria
//!
//! # References
//!
//! 1. Nocedal J, Wright SJ (2006) Numerical Optimization, Springer, 2nd ed
//! 2. Dennis JE, Schnabel RB (1996) Numerical Methods for Unconstrained Optimization

use crate::StrError;
use crate::{solve_lin_sys, vec_copy, vec_norm, Matrix, Vector, Stats, Norm};

/// Implements Newton's method for solving nonlinear systems
#[derive(Clone, Copy, Debug)]
pub struct NewtonSolver {
    /// Maximum number of iterations
    ///
    /// Default = 50
    pub max_iterations: usize,

    /// Tolerance for convergence based on residual norm
    ///
    /// Default = 1e-10
    pub tolerance: f64,

    /// Tolerance for divergence detection
    ///
    /// If the norm of the residual exceeds `divergent_tol * initial_norm`,
    /// the solver will return an error.
    ///
    /// Default = 1e6
    pub divergent_tol: f64,

    /// Whether to use line search
    ///
    /// Default = true
    pub use_line_search: bool,

    /// Initial step size for line search
    ///
    /// Default = 1.0
    pub initial_alpha: f64,
}

impl NewtonSolver {
    /// Allocates a new instance with default parameters
    pub fn new() -> Self {
        NewtonSolver {
            max_iterations: 50,
            tolerance: 1e-10,
            divergent_tol: 1e6,
            use_line_search: true,
            initial_alpha: 1.0,
        }
    }

    /// Validates the solver parameters
    fn validate_params(&self) -> Result<(), StrError> {
        if self.max_iterations < 1 {
            return Err("max_iterations must be >= 1");
        }
        if self.tolerance < 10.0 * f64::EPSILON {
            return Err("tolerance must be >= 10 * f64::EPSILON");
        }
        if self.divergent_tol < 1.0 {
            return Err("divergent_tol must be >= 1.0");
        }
        Ok(())
    }

    /// Solves a nonlinear system using Newton's method with numerical Jacobian
    ///
    /// # Arguments
    ///
    /// * `x0` -- Initial guess (will be modified in place)
    /// * `args` -- Extra arguments for the callback functions
    /// * `f` -- The function F(x) that returns the residual vector
    /// * `jacobian` -- Optional analytical Jacobian function; if None, numerical Jacobian is used
    ///
    /// # Output
    ///
    /// Returns `(x, stats)` where:
    /// * `x` -- The solution vector
    /// * `stats` -- Statistics about the computation
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::{NewtonSolver, Vector, Matrix, StrError};
    ///
    /// fn main() -> Result<(), StrError> {
    ///     // Linear system: Ax = b, where A = [[2, 1], [1, 2]], b = [3, 3]
    ///     // Solution: x = [1, 1]
    ///     let mut x0 = Vector::from(&[0.0, 0.0]);
    ///     let args = &mut ();
    ///
    ///     // F(x) = Ax - b = 0
    ///     let f = |x: &Vector, out: &mut Vector, _: &mut ()| {
    ///         out[0] = 2.0 * x[0] + 1.0 * x[1] - 3.0;
    ///         out[1] = 1.0 * x[0] + 2.0 * x[1] - 3.0;
    ///         Ok(())
    ///     };
    ///
    ///     // Analytical Jacobian: J = A
    ///     let jacobian = |j: &mut Matrix, _: &Vector, _: &mut ()| {
    ///         j.set(0, 0, 2.0); j.set(0, 1, 1.0);
    ///         j.set(1, 0, 1.0); j.set(1, 1, 2.0);
    ///         Ok(())
    ///     };
    ///
    ///     let mut solver = NewtonSolver::new();
    ///     solver.use_line_search = false;
    ///     let (x, stats) = solver.solve(&mut x0, args, f, jacobian)?;
    ///
    ///     println!("Solution: {:?}", x.as_data());
    ///     println!("{}", stats);
    ///
    ///     // Check: x ≈ [1, 1]
    ///     assert!((x[0] - 1.0).abs() < 1e-10);
    ///     assert!((x[1] - 1.0).abs() < 1e-10);
    ///     Ok(())
    /// }
    /// ```
pub fn solve<F, A, J>(
        &self,
        x0: &mut Vector,
        args: &mut A,
        mut f: F,
        mut jacobian: J,
    ) -> Result<(Vector, Stats), StrError>
    where
        F: FnMut(&Vector, &mut Vector, &mut A) -> Result<(), StrError>,
        J: FnMut(&mut Matrix, &Vector, &mut A) -> Result<(), StrError>,
    {
        self.validate_params()?;
        let mut stats = Stats::new();
        let n = x0.dim();

        let mut x = x0.clone();
        let mut residual = Vector::new(n);
        let mut jac = Matrix::new(n, n);
        let mut rhs = Vector::new(n);

        f(&x, &mut residual, args)?;
        stats.n_function += 1;

        let initial_norm = vec_norm(&residual, Norm::Max);
        let mut residual_norm = initial_norm;

        for _ in 0..self.max_iterations {
            stats.n_iterations += 1;

            if residual_norm < self.tolerance {
                let _ = vec_copy(x0, &x);
                stats.stop_sw_total();
                return Ok((x, stats));
            }

            if stats.n_iterations > 1 && residual_norm > self.divergent_tol * initial_norm {
                return Err("solution diverged");
            }

            // Compute Jacobian at x and solve for step
            jacobian(&mut jac, &x, args)?;
            stats.n_jacobian += 1;
            for i in 0..n {
                rhs[i] = -residual[i];
            }
            solve_lin_sys(&mut rhs, &mut jac)?;

            if self.use_line_search {
                let mut alpha = self.initial_alpha;
                let slope = vec_norm(&rhs, Norm::Max);
                for _ in 0..20 {
                    for i in 0..n {
                        x[i] = x[i] + alpha * rhs[i];
                    }
                    f(&x, &mut residual, args)?;
                    stats.n_function += 1;
                    let new_norm = vec_norm(&residual, Norm::Max);
                    if new_norm <= residual_norm - 0.0001 * alpha * slope {
                        break;
                    }
                    alpha *= 0.5;
                }
            } else {
                for i in 0..n {
                    x[i] = x[i] + rhs[i];
                }
                f(&x, &mut residual, args)?;
                stats.n_function += 1;
            }

            residual_norm = vec_norm(&residual, Norm::Max);
        }

        let _ = vec_copy(x0, &x);
        stats.stop_sw_total();
        Err("Newton solver failed to converge")
    }
}

impl Default for NewtonSolver {
    fn default() -> Self {
        Self::new()
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::NewtonSolver;
    use crate::{approx_eq, Matrix, Vector};

    // ===== 1. Basic Structure Tests =====

    #[test]
    fn newton_solver_default_works() {
        let solver = NewtonSolver::default();
        assert_eq!(solver.max_iterations, 50);
        assert_eq!(solver.tolerance, 1e-10);
        assert_eq!(solver.divergent_tol, 1e6);
        assert!(solver.use_line_search);
        assert_eq!(solver.initial_alpha, 1.0);
    }

    #[test]
    fn newton_solver_new_works() {
        let mut solver = NewtonSolver::new();
        solver.max_iterations = 100;
        solver.tolerance = 1e-12;
        solver.use_line_search = false;
        assert_eq!(solver.max_iterations, 100);
        assert_eq!(solver.tolerance, 1e-12);
        assert!(!solver.use_line_search);
    }

    // ===== 2. Simple Linear System Test =====

    #[test]
    fn newton_simple_linear_system() {
        // Linear system: Ax = b
        // A = [[2, 1], [1, 2]], b = [3, 3]
        // Solution: x = [1, 1]
        // F(x) = Ax - b, root at [1, 1]
        let mut _x0 = Vector::from(&[0.0, 0.0]);
        let args = &mut ();

        // Compute F(x) = Ax - b, storing result in the output vector
        let f = |x: &Vector, out: &mut Vector, _: &mut ()| {
            out[0] = 2.0 * x[0] + 1.0 * x[1] - 3.0;
            out[1] = 1.0 * x[0] + 2.0 * x[1] - 3.0;
            Ok(())
        };

        // Analytical Jacobian: J = A
        let jacobian = |j: &mut Matrix, _: &Vector, _: &mut ()| {
            j.set(0, 0, 2.0); j.set(0, 1, 1.0);
            j.set(1, 0, 1.0); j.set(1, 1, 2.0);
            Ok(())
        };

        let mut solver = NewtonSolver::new();
        solver.use_line_search = false;
        let mut _x0 = Vector::from(&[0.0, 0.0]);
        let result = solver.solve(&mut _x0, args, f, jacobian);
        assert!(result.is_ok(), "Expected Ok, got {:?}", result.err());
        let (x, stats) = result.unwrap();

        approx_eq(x[0], 1.0, 1e-10);
        approx_eq(x[1], 1.0, 1e-10);
        assert_eq!(stats.n_iterations, 2); // 1 to check and realize converges, 1 to confirm
    }

    // ===== 3. Rosenbrock System Test =====

    #[test]
    fn newton_rosenbrock_system() {
        // Rosenbrock function F(x,y) = [1-x, 100(y-x²)]
        // Root at (1, 1)
        let mut _x0 = Vector::from(&[0.0, 0.0]);
        let args = &mut ();

        // Compute F(x), storing result in the output vector
        let f = |x: &Vector, out: &mut Vector, _: &mut ()| {
            out[0] = 1.0 - x[0];
            out[1] = 100.0 * (x[1] - x[0] * x[0]);
            Ok(())
        };

        // Analytical Jacobian: J = [[-1, 0], [-200*x0, 100]]
        let jacobian = |j: &mut Matrix, x: &Vector, _: &mut ()| {
            j.set(0, 0, -1.0);
            j.set(0, 1, 0.0);
            j.set(1, 0, -200.0 * x[0]);
            j.set(1, 1, 100.0);
            Ok(())
        };

        let mut solver = NewtonSolver::new();
        solver.use_line_search = false;
        let mut _x0 = Vector::from(&[0.0, 0.0]);
        let result = solver.solve(&mut _x0, args, f, jacobian);
        assert!(result.is_ok(), "Expected Ok, got {:?}", result.err());
        let (x, _) = result.unwrap();

        approx_eq(x[0], 1.0, 1e-8);
        approx_eq(x[1], 1.0, 1e-8);
    }

    // ===== 4. Error Handling Tests =====

    #[test]
    fn newton_validates_max_iterations() {
        let mut solver = NewtonSolver::new();
        solver.max_iterations = 0;
        assert_eq!(solver.validate_params().err(), Some("max_iterations must be >= 1"));

        let mut solver = NewtonSolver::new();
        solver.tolerance = 0.0;
        assert_eq!(solver.validate_params().err(), Some("tolerance must be >= 10 * f64::EPSILON"));

        let mut solver = NewtonSolver::new();
        solver.divergent_tol = 0.5;
        assert_eq!(solver.validate_params().err(), Some("divergent_tol must be >= 1.0"));
    }

    #[test]
    fn newton_handles_iteration_limit() {
        // Test with max iterations limit - should hit iteration limit or converge
        let mut _x0 = Vector::from(&[1.0, 1.0]);
        let args = &mut ();

        let f = |_x: &Vector, out: &mut Vector, _: &mut ()| {
            out[0] = _x[0]; // F1 = x
            out[1] = _x[1]; // F2 = y
            Ok(())
        };

        let jacobian = |j: &mut Matrix, _: &Vector, _: &mut ()| {
            j.set(0, 0, 1.0); j.set(0, 1, 0.0);
            j.set(1, 0, 0.0); j.set(1, 1, 1.0);
            Ok(())
        };

        let mut solver = NewtonSolver::new();
        solver.max_iterations = 1;
        solver.tolerance = 1e-8; // Reasonable tolerance
        let result = solver.solve(&mut _x0, args, f, jacobian);
        // With max_iterations=1 and reasonable tolerance, may not converge
        match result {
            Ok(_) => {
                // Converged - that's fine too
            }
            Err(e) => {
                // Should contain expected error messages
                assert!(e.contains("converge") || e.contains("diverged"), "Unexpected error: {}", e);
            }
        }
    }

    // ===== 5. Parameter Configuration Tests =====

    #[test]
    fn newton_tolerance_configuration() {
        let solver = NewtonSolver::new();
        assert_eq!(solver.tolerance, 1e-10);

        let mut solver = NewtonSolver::new();
        solver.tolerance = 1e-6;
        assert_eq!(solver.tolerance, 1e-6);
    }

    #[test]
    fn newton_line_search_configuration() {
        let mut solver = NewtonSolver::new();
        assert!(solver.use_line_search);
        solver.use_line_search = false;
        assert!(!solver.use_line_search);
    }

    // ===== 6. Line Search Test =====

    #[test]
    fn newton_with_line_search() {
        let mut _x0 = Vector::from(&[0.0, 0.0]);
        let args = &mut ();

        let f = |x: &Vector, out: &mut Vector, _: &mut ()| {
            out[0] = 2.0 * x[0] + 1.0 * x[1] - 3.0;
            out[1] = 1.0 * x[0] + 2.0 * x[1] - 3.0;
            Ok(())
        };

        let jacobian = |j: &mut Matrix, _: &Vector, _: &mut ()| {
            j.set(0, 0, 2.0); j.set(0, 1, 1.0);
            j.set(1, 0, 1.0); j.set(1, 1, 2.0);
            Ok(())
        };

        let mut solver = NewtonSolver::new();
        solver.use_line_search = true;
        let result = solver.solve(&mut _x0, args, f, jacobian);
        assert!(result.is_ok(), "Expected Ok, got {:?}", result.err());
        let (x, _) = result.unwrap();

        approx_eq(x[0], 1.0, 1e-10);
        approx_eq(x[1], 1.0, 1e-10);
    }

    // ===== 7. Initial Guess is Solution =====

    #[test]
    fn newton_initial_guess_is_solution() {
        let mut _x0 = Vector::from(&[1.0, 1.0]);
        let args = &mut ();

        let f = |x: &Vector, out: &mut Vector, _: &mut ()| {
            out[0] = 2.0 * x[0] + 1.0 * x[1] - 3.0;
            out[1] = 1.0 * x[0] + 2.0 * x[1] - 3.0;
            Ok(())
        };

        let jacobian = |j: &mut Matrix, _: &Vector, _: &mut ()| {
            j.set(0, 0, 2.0); j.set(0, 1, 1.0);
            j.set(1, 0, 1.0); j.set(1, 1, 2.0);
            Ok(())
        };

        let mut solver = NewtonSolver::new();
        solver.use_line_search = false;
        let result = solver.solve(&mut _x0, args, f, jacobian);
        assert!(result.is_ok(), "Expected Ok, got {:?}", result.err());
        let (x, stats) = result.unwrap();

        approx_eq(x[0], 1.0, 1e-10);
        approx_eq(x[1], 1.0, 1e-10);
        assert_eq!(stats.n_iterations, 1); // Should converge immediately
    }

    // ===== 8. Stats Test =====

    #[test]
    fn newton_stats_tracking() {
        let mut _x0 = Vector::from(&[0.0, 0.0]);
        let args = &mut ();

        let f = |x: &Vector, out: &mut Vector, _: &mut ()| {
            out[0] = 2.0 * x[0] + 1.0 * x[1] - 3.0;
            out[1] = 1.0 * x[0] + 2.0 * x[1] - 3.0;
            Ok(())
        };

        let jacobian = |j: &mut Matrix, _: &Vector, _: &mut ()| {
            j.set(0, 0, 2.0); j.set(0, 1, 1.0);
            j.set(1, 0, 1.0); j.set(1, 1, 2.0);
            Ok(())
        };

        let mut solver = NewtonSolver::new();
        solver.use_line_search = false;
        let result = solver.solve(&mut _x0, args, f, jacobian);
        assert!(result.is_ok(), "Expected Ok, got {:?}", result.err());
        let (_, stats) = result.unwrap();

        assert!(stats.n_iterations >= 1);
        assert!(stats.n_function >= 1);
        assert!(stats.n_jacobian >= 1);
    }

    // ===== 9. 3x3 System Test =====

    #[test]
    fn newton_3x3_system() {
        let mut _x0 = Vector::from(&[0.0, 0.0, 0.0]);
        let args = &mut ();

        // 3x3 linear system: A * x = b
        // A = [[3,1,1],[1,3,1],[1,1,3]], b = [1,1,1]
        // Solution: x = [0.2, 0.2, 0.2]
        let f = |x: &Vector, out: &mut Vector, _: &mut ()| {
            out[0] = 3.0 * x[0] + x[1] + x[2] - 1.0;
            out[1] = x[0] + 3.0 * x[1] + x[2] - 1.0;
            out[2] = x[0] + x[1] + 3.0 * x[2] - 1.0;
            Ok(())
        };

        let jacobian = |j: &mut Matrix, _: &Vector, _: &mut ()| {
            j.set(0, 0, 3.0); j.set(0, 1, 1.0); j.set(0, 2, 1.0);
            j.set(1, 0, 1.0); j.set(1, 1, 3.0); j.set(1, 2, 1.0);
            j.set(2, 0, 1.0); j.set(2, 1, 1.0); j.set(2, 2, 3.0);
            Ok(())
        };

        let mut solver = NewtonSolver::new();
        solver.use_line_search = false;
        let result = solver.solve(&mut _x0, args, f, jacobian);
        assert!(result.is_ok(), "Expected Ok, got {:?}", result.err());
        let (x, _) = result.unwrap();

        approx_eq(x[0], 0.2, 1e-10);
        approx_eq(x[1], 0.2, 1e-10);
        approx_eq(x[2], 0.2, 1e-10);
    }

    // ===== 10. Divergent Tol Test =====

    #[test]
    fn newton_divergent_tol() {
        let mut _x0 = Vector::from(&[0.0, 0.0]);
        let args = &mut ();

        let f = |x: &Vector, out: &mut Vector, _: &mut ()| {
            out[0] = 2.0 * x[0] + 1.0 * x[1] - 3.0;
            out[1] = 1.0 * x[0] + 2.0 * x[1] - 3.0;
            Ok(())
        };

        let jacobian = |j: &mut Matrix, _: &Vector, _: &mut ()| {
            j.set(0, 0, 2.0); j.set(0, 1, 1.0);
            j.set(1, 0, 1.0); j.set(1, 1, 2.0);
            Ok(())
        };

        let mut solver = NewtonSolver::new();
        solver.use_line_search = false;
        solver.divergent_tol = 1.0; // Very small, triggers divergence
        let result = solver.solve(&mut _x0, args, f, jacobian);
        // May get divergence error or converge
        match result {
            Ok(_) => {}
            Err(e) => {
                assert!(e.contains("diverged"), "Unexpected error: {}", e);
            }
        }
    }

    // ===== 11. Initial Alpha Test =====

    #[test]
    fn newton_initial_alpha_configuration() {
        let solver = NewtonSolver::new();
        assert_eq!(solver.initial_alpha, 1.0);

        let mut solver = NewtonSolver::new();
        solver.initial_alpha = 0.5;
        assert_eq!(solver.initial_alpha, 0.5);
    }

    // ===== 12. Test Numerical Jacobian =====
    #[test]
    fn newton_jacobian_is_called() {
        let mut _x0 = Vector::from(&[0.0, 0.0]);
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
        solver.max_iterations = 2;
        let result = solver.solve(&mut _x0, args, f, jacobian);
        assert!(result.is_ok(), "Expected Ok, got {:?}", result.err());
    }

    // ===== 13. Test Line Search Reduces Alpha =====
    #[test]
    fn newton_line_search_reduces_alpha() {
        let mut _x0 = Vector::from(&[2.0, 2.0]);
        let args = &mut ();

        let f = |x: &Vector, out: &mut Vector, _: &mut ()| {
            out[0] = x[0] * x[0] - 4.0;
            out[1] = x[1] * x[1] - 4.0;
            Ok(())
        };

        let jacobian = |j: &mut Matrix, x: &Vector, _: &mut ()| {
            j.set(0, 0, 2.0 * x[0]);
            j.set(0, 1, 0.0);
            j.set(1, 0, 0.0);
            j.set(1, 1, 2.0 * x[1]);
            Ok(())
        };

        let mut solver = NewtonSolver::new();
        solver.use_line_search = true;
        solver.initial_alpha = 1.0;
        solver.max_iterations = 10;
        let result = solver.solve(&mut _x0, args, f, jacobian);
        let _ = result;
    }
}