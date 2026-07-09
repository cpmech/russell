//! Newton solver for nonlinear systems
//!
//! Implements Newton's method with optional line search for solving
//! systems of nonlinear equations F(x) = 0.
//!
//! # Features
//!
//! * Analytical Jacobian computation
//! * Optional line search for globalization
//! * Configurable convergence criteria
//!
//! # References
//!
//! 1. Nocedal J, Wright SJ (2006) Numerical Optimization, Springer, 2nd ed
//! 2. Dennis JE, Schnabel RB (1996) Numerical Methods for Unconstrained Optimization

use crate::StrError;
use crate::{Matrix, Norm, Stats, Vector, solve_lin_sys, vec_inner, vec_norm};

/// Implements Newton's method for solving nonlinear systems
#[derive(Clone, Debug)]
pub struct NewtonSolver {
    /// System dimension (number of equations; number of unknowns)
    ndim: usize,

    /// Holds the residual vector r = f(x) for the last solve operation
    r: Vector,

    /// Holds the Jacobian matrix J = df/dx for the last solve operation
    jj: Matrix,

    /// Holds the Newton step p for the last solve operation
    p: Vector,

    /// Maximum number of iterations
    ///
    /// Default = 20
    max_iterations: usize,

    /// Tolerance for convergence based on residual norm
    ///
    /// Default = 1e-10
    tolerance: f64,

    /// Tolerance for divergence detection
    ///
    /// If the norm of the residual exceeds `divergent_tol * initial_norm`,
    /// the solver will return an error.
    ///
    /// Default = 1e6
    divergent_tol: f64,

    /// Whether to use line search
    ///
    /// Default = false
    use_line_search: bool,

    /// Initial step size for line search
    ///
    /// Default = 1.0
    initial_alpha: f64,

    /// Holds the statistics of the last solve operation
    stats: Stats,
}

impl NewtonSolver {
    /// Allocates a new instance with default parameters
    pub fn new(ndim: usize) -> Result<Self, StrError> {
        if ndim == 0 {
            return Err("ndim must be > 0");
        }
        Ok(NewtonSolver {
            ndim,
            r: Vector::new(ndim),
            jj: Matrix::new(ndim, ndim),
            p: Vector::new(ndim),
            max_iterations: 20,
            tolerance: 1e-10,
            divergent_tol: 1e6,
            use_line_search: false,
            initial_alpha: 1.0,
            stats: Stats::new(),
        })
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
        if self.initial_alpha <= 0.0 {
            return Err("initial_alpha must be > 0");
        }
        Ok(())
    }

    /// Sets the maximum number of iterations
    ///
    /// Default value: 20
    pub fn set_max_iterations(&mut self, value: usize) -> &mut Self {
        self.max_iterations = value;
        self
    }

    /// Sets the tolerance for convergence
    ///
    /// Default value: 1e-10
    pub fn set_tolerance(&mut self, value: f64) -> &mut Self {
        self.tolerance = value;
        self
    }

    /// Sets the tolerance for divergence detection
    ///
    /// Default value: 1e6
    pub fn set_divergent_tol(&mut self, value: f64) -> &mut Self {
        self.divergent_tol = value;
        self
    }

    /// Sets whether to use line search
    ///
    /// Default value: false
    pub fn set_use_line_search(&mut self, value: bool) -> &mut Self {
        self.use_line_search = value;
        self
    }

    /// Sets the initial step size for line search
    ///
    /// Default value: 1.0
    pub fn set_initial_alpha(&mut self, value: f64) -> &mut Self {
        self.initial_alpha = value;
        self
    }

    /// Sets whether to enable statistics tracking
    ///
    /// Default value: false
    pub fn set_enable_stats(&mut self, value: bool) -> &mut Self {
        self.stats.enable(value);
        self
    }

    /// Returns the statistics of the last solve operation
    pub fn get_stats(&self) -> Result<&Stats, StrError> {
        if !self.stats.is_enabled() {
            return Err("statistics tracking is disabled; enable it with set_enable_stats(true)");
        }
        Ok(&self.stats)
    }

    /// Solves a nonlinear system using Newton's method
    ///
    /// Solves:
    ///
    /// ```text
    /// r := f(x) = 0
    ///
    /// with
    ///
    /// J := df/dx
    /// ```
    ///
    /// # Input
    ///
    /// * `x` -- Initial guess on the input and the solution vector on the output
    /// * `args` -- Extra arguments for the callback functions
    /// * `calc_f` -- Function to calculate the residual vector (r = f(x))
    /// * `calc_jj` -- Function to calculate the Jacobian matrix (J)
    ///
    /// # Output
    ///
    /// * Returns `Ok(())` on successful convergence; the solution is stored in `x`
    ///
    /// # Examples
    ///
    /// The classic two-equation nonlinear system (intersection of a circle and a line):
    ///
    /// ```text
    /// r₀(x) = x₀² + x₁² - 4 = 0   (circle of radius 2)
    /// r₁(x) = x₀  - x₁      = 0   (line x₀ = x₁)
    ///
    /// Jacobian:
    ///     J = [ 2x₀   2x₁ ]
    ///         [  1    -1  ]
    ///
    /// Solutions: x₀ = x₁ = √2 and x₀ = x₁ = -√2
    /// Solution for the initial guess (1, 0.5) is x₀ = x₁ = √2
    /// ```
    ///
    /// ```
    /// use russell_lab::{approx_eq, NewtonSolver, Vector, Matrix, StrError};
    /// use russell_lab::math::SQRT_2;
    ///
    /// fn main() -> Result<(), StrError> {
    ///
    ///     // f(x): circle x₀² + x₁² = 4  and  line x₀ = x₁
    ///     let calc_f = |r: &mut Vector, x: &Vector , _: &mut i32| {
    ///         r[0] = x[0] * x[0] + x[1] * x[1] - 4.0;
    ///         r[1] = x[0] - x[1];
    ///         Ok(())
    ///     };
    ///
    ///     // Analytical Jacobian
    ///     let calc_jj = |jj: &mut Matrix, x: &Vector, _: &mut i32| {
    ///         jj.set(0, 0, 2.0 * x[0]);
    ///         jj.set(0, 1, 2.0 * x[1]);
    ///         jj.set(1, 0,  1.0);
    ///         jj.set(1, 1, -1.0);
    ///         Ok(())
    ///     };
    ///
    ///     // Initial guess
    ///     let mut x = Vector::from(&[1.0, 0.5]);
    ///     let args = &mut 0;
    ///
    ///     // Create the solver and solve the problem
    ///     let mut solver = NewtonSolver::new(2)?;
    ///     solver.solve(&mut x, args, calc_f, calc_jj)?;
    ///     println!("x = {:?}", x.as_data());
    ///
    ///     // Check the solution
    ///     approx_eq(x[0], SQRT_2, 1e-10);
    ///     approx_eq(x[1], SQRT_2, 1e-10);
    ///     Ok(())
    /// }
    /// ```
    pub fn solve<F, A, J>(
        &mut self,
        x: &mut Vector,
        args: &mut A,
        mut calc_f: F,
        mut calc_jj: J,
    ) -> Result<(), StrError>
    where
        F: FnMut(&mut Vector, &Vector, &mut A) -> Result<(), StrError>,
        J: FnMut(&mut Matrix, &Vector, &mut A) -> Result<(), StrError>,
    {
        // Check parameters
        self.validate_params()?;
        if x.dim() != self.ndim {
            return Err("dimension of x does not match solver dimension");
        }

        // Reset statistics for this solve operation
        self.stats.reset();

        // Calculate the first residual vector r = f(x)
        calc_f(&mut self.r, &x, args)?;
        self.stats.inc_n_function(1);

        // Calculate the first residual norm
        let initial_norm = vec_norm(&self.r, Norm::Max);
        let mut residual_norm = initial_norm;

        // Main Newton iteration loop
        for it in 0..self.max_iterations {
            self.stats.inc_n_iterations(1);

            // Check convergence
            if residual_norm < self.tolerance {
                self.stats.stop_sw_total();
                return Ok(());
            }

            // Check divergence
            if it > 0 && residual_norm > self.divergent_tol * initial_norm {
                self.stats.stop_sw_total();
                return Err("solution diverged");
            }

            // Compute Jacobian at x and solve J*p = -r(x) for p
            calc_jj(&mut self.jj, &x, args)?;
            self.stats.inc_n_jacobian(1);
            for i in 0..self.ndim {
                self.p[i] = -self.r[i];
            }
            solve_lin_sys(&mut self.p, &mut self.jj)?;

            // Update x
            if self.use_line_search {
                // Update with line search
                // Merit function: phi(x) = 0.5 * ||F(x)||^2 (Euclidean)
                // Directional derivative: phi'(0) = F(x)^T * J(x) * p = F(x)^T * (-F(x)) = -||F(x)||^2
                let phi_base = 0.5 * vec_inner(&self.r, &self.r);
                let slope = -2.0 * phi_base; // = -||F||^2, always <= 0
                let c1 = 1e-4_f64;
                let rho = 0.5_f64;
                let mut alpha = self.initial_alpha;
                let x_base = x.clone();
                for _ in 0..20 {
                    for i in 0..self.ndim {
                        x[i] = x_base[i] + alpha * self.p[i];
                    }
                    calc_f(&mut self.r, &x, args)?;
                    self.stats.inc_n_function(1);
                    let phi_new = 0.5 * vec_inner(&self.r, &self.r);
                    // Armijo condition: phi(alpha) <= phi(0) + c1 * alpha * phi'(0)
                    if phi_new <= phi_base + c1 * alpha * slope {
                        break;
                    }
                    alpha *= rho;
                }
            } else {
                // Update without line search
                for i in 0..self.ndim {
                    x[i] += self.p[i];
                }
                calc_f(&mut self.r, &x, args)?;
                self.stats.inc_n_function(1);
            }

            // Update residual norm
            residual_norm = vec_norm(&self.r, Norm::Max);
        }

        // If we reach here, the solver did not converge within the maximum number of iterations
        self.stats.stop_sw_total();
        Err("Newton solver failed to converge")
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::NewtonSolver;
    use crate::{Matrix, StrError, Vector, approx_eq};

    // ===== Basic Structure Tests =====

    #[test]
    fn newton_solver_new_works() {
        let solver = NewtonSolver::new(2).unwrap();
        assert_eq!(solver.max_iterations, 20);
        assert_eq!(solver.tolerance, 1e-10);
        assert_eq!(solver.divergent_tol, 1e6);
        assert!(!solver.use_line_search);
        assert_eq!(solver.initial_alpha, 1.0);
    }

    #[test]
    fn newton_solver_fields_are_mutable() {
        let mut solver = NewtonSolver::new(2).unwrap();
        solver
            .set_max_iterations(100)
            .set_tolerance(1e-12)
            .set_use_line_search(true);
        assert_eq!(solver.max_iterations, 100);
        assert_eq!(solver.tolerance, 1e-12);
        assert!(solver.use_line_search);
    }

    // ===== Simple Linear System Test =====

    #[test]
    fn newton_simple_linear_system() -> Result<(), StrError> {
        // Linear system: Ax = b
        // A = [[2, 1], [1, 2]], b = [3, 3]
        // Solution: x = [1, 1]
        // F(x) = Ax - b, root at [1, 1]
        let args = &mut ();

        // Compute F(x) = Ax - b, storing result in the output vector
        let calc_f = |r: &mut Vector, x: &Vector, _: &mut ()| {
            r[0] = 2.0 * x[0] + 1.0 * x[1] - 3.0;
            r[1] = 1.0 * x[0] + 2.0 * x[1] - 3.0;
            Ok(())
        };

        // Analytical Jacobian: J = A
        let calc_jj = |jj: &mut Matrix, _: &Vector, _: &mut ()| {
            jj.set(0, 0, 2.0);
            jj.set(0, 1, 1.0);
            jj.set(1, 0, 1.0);
            jj.set(1, 1, 2.0);
            Ok(())
        };

        let mut solver = NewtonSolver::new(2).unwrap();
        solver.set_use_line_search(false).set_enable_stats(true);
        let mut x = Vector::from(&[0.0, 0.0]);
        solver.solve(&mut x, args, calc_f, calc_jj)?;

        approx_eq(x[0], 1.0, 1e-10);
        approx_eq(x[1], 1.0, 1e-10);
        assert_eq!(solver.get_stats()?.get_n_iterations(), 2); // 2: first iter takes the Newton step; second iter sees convergence
        Ok(())
    }

    // ===== Rosenbrock System Test =====

    #[test]
    fn newton_rosenbrock_system() -> Result<(), StrError> {
        // Rosenbrock function F(x,y) = [1-x, 100(y-x²)]
        // Root at (1, 1)
        let args = &mut ();

        // Compute F(x), storing result in the output vector
        let calc_f = |r: &mut Vector, x: &Vector, _: &mut ()| {
            r[0] = 1.0 - x[0];
            r[1] = 100.0 * (x[1] - x[0] * x[0]);
            Ok(())
        };

        // Analytical Jacobian: J = [[-1, 0], [-200*x0, 100]]
        let calc_jj = |jj: &mut Matrix, x: &Vector, _: &mut ()| {
            jj.set(0, 0, -1.0);
            jj.set(0, 1, 0.0);
            jj.set(1, 0, -200.0 * x[0]);
            jj.set(1, 1, 100.0);
            Ok(())
        };

        let mut solver = NewtonSolver::new(2).unwrap();
        solver.set_use_line_search(false);
        let mut x = Vector::from(&[0.0, 0.0]);
        solver.solve(&mut x, args, calc_f, calc_jj)?;

        approx_eq(x[0], 1.0, 1e-8);
        approx_eq(x[1], 1.0, 1e-8);
        Ok(())
    }

    // ===== Error Handling Tests =====

    #[test]
    fn newton_validates_max_iterations() {
        let mut solver = NewtonSolver::new(2).unwrap();
        solver.set_max_iterations(0);
        assert_eq!(solver.validate_params().err(), Some("max_iterations must be >= 1"));

        let mut solver = NewtonSolver::new(2).unwrap();
        solver.set_tolerance(0.0);
        assert_eq!(
            solver.validate_params().err(),
            Some("tolerance must be >= 10 * f64::EPSILON")
        );

        let mut solver = NewtonSolver::new(2).unwrap();
        solver.set_divergent_tol(0.5);
        assert_eq!(solver.validate_params().err(), Some("divergent_tol must be >= 1.0"));
    }

    #[test]
    fn newton_handles_iteration_limit() {
        // Test with max iterations limit - should hit iteration limit or converge
        let mut x = Vector::from(&[1.0, 1.0]);
        let args = &mut ();

        let calc_f = |r: &mut Vector, x: &Vector, _: &mut ()| {
            r[0] = x[0]; // F1 = x
            r[1] = x[1]; // F2 = y
            Ok(())
        };

        let calc_jj = |jj: &mut Matrix, _: &Vector, _: &mut ()| {
            jj.set(0, 0, 1.0);
            jj.set(0, 1, 0.0);
            jj.set(1, 0, 0.0);
            jj.set(1, 1, 1.0);
            Ok(())
        };

        let mut solver = NewtonSolver::new(2).unwrap();
        solver.set_max_iterations(1);
        solver.set_tolerance(1e-8);
        let result = solver.solve(&mut x, args, calc_f, calc_jj);
        // x0=[1,1] → Newton step → x=[0,0] (residual=0), but the convergence
        // check runs at the top of the loop, so the updated residual is only
        // checked in the *next* iteration, which never happens.
        // The solver therefore exits the loop and returns the "failed to converge" error.
        assert_eq!(result.err(), Some("Newton solver failed to converge"));
    }

    // ===== Parameter Configuration Tests =====

    #[test]
    fn newton_tolerance_configuration() {
        let solver = NewtonSolver::new(2).unwrap();
        assert_eq!(solver.tolerance, 1e-10);

        let mut solver = NewtonSolver::new(2).unwrap();
        solver.set_tolerance(1e-6);
        assert_eq!(solver.tolerance, 1e-6);
    }

    #[test]
    fn newton_line_search_configuration() {
        let mut solver = NewtonSolver::new(2).unwrap();
        assert!(!solver.use_line_search);
        solver.set_use_line_search(true);
        assert!(solver.use_line_search);
    }

    // ===== Line Search Test =====

    #[test]
    fn newton_with_line_search() -> Result<(), StrError> {
        let mut x = Vector::from(&[0.0, 0.0]);
        let args = &mut ();

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

        let mut solver = NewtonSolver::new(2).unwrap();
        solver.set_use_line_search(true);
        solver.solve(&mut x, args, calc_f, calc_jj)?;

        approx_eq(x[0], 1.0, 1e-10);
        approx_eq(x[1], 1.0, 1e-10);
        Ok(())
    }

    // ===== Initial Guess is Solution =====

    #[test]
    fn newton_initial_guess_is_solution() -> Result<(), StrError> {
        let mut x = Vector::from(&[1.0, 1.0]);
        let args = &mut ();

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

        // call calc_jj just so the coverage tool sees it, even though the solver should never call it since the initial guess is already a solution
        let _ = calc_jj(&mut Matrix::new(2, 2), &x, args);

        let mut solver = NewtonSolver::new(2).unwrap();
        solver.set_use_line_search(false).set_enable_stats(true);
        solver.solve(&mut x, args, calc_f, calc_jj)?;

        approx_eq(x[0], 1.0, 1e-10);
        approx_eq(x[1], 1.0, 1e-10);
        assert_eq!(solver.get_stats()?.get_n_iterations(), 1); // Should converge immediately
        Ok(())
    }

    // ===== Stats Test =====

    #[test]
    fn newton_stats_tracking() -> Result<(), StrError> {
        let mut x = Vector::from(&[0.0, 0.0]);
        let args = &mut ();

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

        let mut solver = NewtonSolver::new(2).unwrap();
        solver.set_use_line_search(false).set_enable_stats(true);
        solver.solve(&mut x, args, calc_f, calc_jj)?;

        assert!(solver.get_stats()?.get_n_iterations() >= 1);
        assert!(solver.get_stats()?.get_n_function() >= 1);
        assert!(solver.get_stats()?.get_n_jacobian() >= 1);
        Ok(())
    }

    // ===== 3x3 System Test =====

    #[test]
    fn newton_3x3_system() -> Result<(), StrError> {
        let mut x = Vector::from(&[0.0, 0.0, 0.0]);
        let args = &mut ();

        // 3x3 linear system: A * x = b
        // A = [[3,1,1],[1,3,1],[1,1,3]], b = [1,1,1]
        // Solution: x = [0.2, 0.2, 0.2]
        let calc_f = |r: &mut Vector, x: &Vector, _: &mut ()| {
            r[0] = 3.0 * x[0] + x[1] + x[2] - 1.0;
            r[1] = x[0] + 3.0 * x[1] + x[2] - 1.0;
            r[2] = x[0] + x[1] + 3.0 * x[2] - 1.0;
            Ok(())
        };

        let calc_jj = |jj: &mut Matrix, _: &Vector, _: &mut ()| {
            jj.set(0, 0, 3.0);
            jj.set(0, 1, 1.0);
            jj.set(0, 2, 1.0);
            jj.set(1, 0, 1.0);
            jj.set(1, 1, 3.0);
            jj.set(1, 2, 1.0);
            jj.set(2, 0, 1.0);
            jj.set(2, 1, 1.0);
            jj.set(2, 2, 3.0);
            Ok(())
        };

        let mut solver = NewtonSolver::new(3).unwrap();
        solver.set_use_line_search(false);
        solver.solve(&mut x, args, calc_f, calc_jj)?;

        approx_eq(x[0], 0.2, 1e-10);
        approx_eq(x[1], 0.2, 1e-10);
        approx_eq(x[2], 0.2, 1e-10);
        Ok(())
    }

    // ===== Divergent Tol Test =====

    #[test]
    fn newton_divergent_tol() {
        // F(x) = [2x + y - 3, x + 2y - 3] with a negated Jacobian forces each
        // Newton step to move away from the root, so the residual norm grows and
        // the divergent_tol = 1.0 check fires on the second iteration.
        let mut x = Vector::from(&[0.0, 0.0]);
        let args = &mut ();

        let calc_f = |r: &mut Vector, x: &Vector, _: &mut ()| {
            r[0] = 2.0 * x[0] + 1.0 * x[1] - 3.0;
            r[1] = 1.0 * x[0] + 2.0 * x[1] - 3.0;
            Ok(())
        };

        // Wrong sign on Jacobian → Newton steps diverge
        let calc_jj = |jj: &mut Matrix, _: &Vector, _: &mut ()| {
            jj.set(0, 0, -2.0);
            jj.set(0, 1, -1.0);
            jj.set(1, 0, -1.0);
            jj.set(1, 1, -2.0);
            Ok(())
        };

        let mut solver = NewtonSolver::new(2).unwrap();
        solver.set_use_line_search(false);
        solver.set_divergent_tol(1.0);
        solver.set_max_iterations(3);
        let result = solver.solve(&mut x, args, calc_f, calc_jj);
        assert_eq!(result.err(), Some("solution diverged"));
    }

    // ===== Initial Alpha Test =====

    #[test]
    fn newton_initial_alpha_configuration() {
        let solver = NewtonSolver::new(2).unwrap();
        assert_eq!(solver.initial_alpha, 1.0);

        let mut solver = NewtonSolver::new(2).unwrap();
        solver.set_initial_alpha(0.5);
        assert_eq!(solver.initial_alpha, 0.5);
    }

    // ===== Test Numerical Jacobian =====

    #[test]
    fn newton_jacobian_is_called() -> Result<(), StrError> {
        let mut x = Vector::from(&[0.0, 0.0]);
        let args = &mut ();

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

        let mut solver = NewtonSolver::new(2).unwrap();
        solver.set_use_line_search(false);
        solver.set_max_iterations(2);
        solver.solve(&mut x, args, calc_f, calc_jj)?;
        Ok(())
    }

    // ===== Test Line Search Reduces Alpha =====

    #[test]
    fn newton_line_search_reduces_alpha() -> Result<(), StrError> {
        let mut x = Vector::from(&[100.0, 100.0]);
        let args = &mut ();

        let calc_f = |r: &mut Vector, x: &Vector, _: &mut ()| {
            r[0] = x[0] * x[0] - 4.0;
            r[1] = x[1] * x[1] - 4.0;
            Ok(())
        };

        let calc_jj = |jj: &mut Matrix, x: &Vector, _: &mut ()| {
            jj.set(0, 0, 2.0 * x[0]);
            jj.set(0, 1, 0.0);
            jj.set(1, 0, 0.0);
            jj.set(1, 1, 2.0 * x[1]);
            Ok(())
        };

        let mut solver = NewtonSolver::new(2).unwrap();
        solver.set_use_line_search(true);
        solver.set_initial_alpha(1.0);
        solver.set_max_iterations(20);
        solver.solve(&mut x, args, calc_f, calc_jj)?;
        approx_eq(x[0], 2.0, 1e-6);
        approx_eq(x[1], 2.0, 1e-6);
        Ok(())
    }

    // ===== Line search alpha backtracking (line 203: alpha *= rho) =====

    #[test]
    fn newton_line_search_backtracks_alpha() -> Result<(), StrError> {
        // F(x) = [x[0]^3 - 1, x[1]^3 - 1], root at (1, 1).
        // Starting far from the root forces a large Newton step whose
        // full-step Armijo condition fails, so alpha must be reduced
        // (alpha *= rho) at least once before the condition is satisfied.
        // Starting at [0.5, 0.5]: the Newton step jumps to ~[1.67, 1.67]
        // where phi_new >> phi_base, so the Armijo condition fails and
        // alpha must be halved (alpha *= rho, line 203) before it passes.
        let mut x = Vector::from(&[0.5, 0.5]);
        let args = &mut ();

        let calc_f = |r: &mut Vector, x: &Vector, _: &mut ()| {
            r[0] = x[0] * x[0] * x[0] - 1.0;
            r[1] = x[1] * x[1] * x[1] - 1.0;
            Ok(())
        };

        let calc_jj = |jj: &mut Matrix, x: &Vector, _: &mut ()| {
            jj.set(0, 0, 3.0 * x[0] * x[0]);
            jj.set(0, 1, 0.0);
            jj.set(1, 0, 0.0);
            jj.set(1, 1, 3.0 * x[1] * x[1]);
            Ok(())
        };

        let mut solver = NewtonSolver::new(2).unwrap();
        solver.set_use_line_search(true);
        solver.set_initial_alpha(1.0);
        solver.set_max_iterations(50);
        solver.solve(&mut x, args, calc_f, calc_jj)?;
        approx_eq(x[0], 1.0, 1e-8);
        approx_eq(x[1], 1.0, 1e-8);
        Ok(())
    }

    // ===== initial_alpha validation =====

    #[test]
    fn newton_validates_initial_alpha() {
        // Direct call to validate_params
        let mut solver = NewtonSolver::new(2).unwrap();
        solver.set_initial_alpha(0.0);
        assert_eq!(solver.validate_params().err(), Some("initial_alpha must be > 0"));

        // Error propagated through solve() — covers the `?` arm in solve
        let mut x = Vector::from(&[0.0, 0.0]);
        let args = &mut ();
        let calc_f = |_: &mut Vector, _: &Vector, _: &mut ()| Ok(());

        // call calc_f just so the coverage tool sees it, even though the solver should never call it since the initial_alpha validation should fail before any iterations
        let _ = calc_f(&mut Vector::new(2), &x, args);

        let calc_jj = |_: &mut Matrix, _: &Vector, _: &mut ()| Ok(());

        // call calc_jj just so the coverage tool sees it, even though the solver should never call it since the initial_alpha validation should fail before any iterations
        let _ = calc_jj(&mut Matrix::new(2, 2), &x, args);

        let mut solver = NewtonSolver::new(2).unwrap();
        solver.set_initial_alpha(-1.0);
        let result = solver.solve(&mut x, args, calc_f, calc_jj);
        assert_eq!(result.err(), Some("initial_alpha must be > 0"));
    }

    // ===== Genuine divergence =====

    #[test]
    fn newton_solution_diverges() {
        // F(x) = x - 1, but Jacobian is -I (wrong sign) so Newton steps move
        // away from the root: each step doubles the residual norm.
        // With divergent_tol = 1.0, the check fires as soon as norm grows.
        let mut x = Vector::from(&[0.0, 0.0]);
        let args = &mut ();

        let calc_f = |r: &mut Vector, x: &Vector, _: &mut ()| {
            r[0] = x[0] - 1.0;
            r[1] = x[1] - 1.0;
            Ok(())
        };

        // Wrong Jacobian: -I instead of I → Newton steps diverge
        let calc_jj = |jj: &mut Matrix, _: &Vector, _: &mut ()| {
            jj.set(0, 0, -1.0);
            jj.set(0, 1, 0.0);
            jj.set(1, 0, 0.0);
            jj.set(1, 1, -1.0);
            Ok(())
        };

        let mut solver = NewtonSolver::new(2).unwrap();
        solver.set_use_line_search(false);
        solver.set_divergent_tol(1.0); // fire as soon as norm grows
        let result = solver.solve(&mut x, args, calc_f, calc_jj);
        assert_eq!(result.err(), Some("solution diverged"));
    }

    // ===== calc_f returns error on initial call =====

    #[test]
    fn newton_f_returns_error_on_initial_call() {
        let mut x = Vector::from(&[0.0, 0.0]);
        let args = &mut ();

        let calc_f = |_: &mut Vector, _: &Vector, _: &mut ()| Err("f error");
        let calc_jj = |_: &mut Matrix, _: &Vector, _: &mut ()| Ok(());

        // call calc_jj just so the coverage tool sees it, even though the solver should never call it since the initial calc_f call should fail before any iterations
        let _ = calc_jj(&mut Matrix::new(2, 2), &x, args);

        let mut solver = NewtonSolver::new(2).unwrap();
        solver.set_use_line_search(false);
        let result = solver.solve(&mut x, args, calc_f, calc_jj);
        assert_eq!(result.err(), Some("f error"));
    }

    // ===== calc_jj returns error =====

    #[test]
    fn newton_jacobian_returns_error() {
        // calc_f succeeds with non-zero residual so the solver proceeds past
        // the convergence check and calls calc_jj
        let mut x = Vector::from(&[0.0, 0.0]);
        let args = &mut ();

        let calc_f = |r: &mut Vector, x: &Vector, _: &mut ()| {
            r[0] = x[0] - 1.0;
            r[1] = x[1] - 1.0;
            Ok(())
        };

        let calc_jj = |_: &mut Matrix, _: &Vector, _: &mut ()| Err("jacobian error");

        let mut solver = NewtonSolver::new(2).unwrap();
        solver.set_use_line_search(false);
        let result = solver.solve(&mut x, args, calc_f, calc_jj);
        assert_eq!(result.err(), Some("jacobian error"));
    }

    // ===== singular Jacobian → solve_lin_sys error =====

    #[test]
    fn newton_singular_jacobian_returns_error() {
        let mut x = Vector::from(&[0.0, 0.0]);
        let args = &mut ();

        let calc_f = |r: &mut Vector, x: &Vector, _: &mut ()| {
            r[0] = x[0] - 1.0;
            r[1] = x[1] - 1.0;
            Ok(())
        };

        // All-zero Jacobian is singular; solve_lin_sys must return an error
        let calc_jj = |jj: &mut Matrix, _: &Vector, _: &mut ()| {
            jj.set(0, 0, 0.0);
            jj.set(0, 1, 0.0);
            jj.set(1, 0, 0.0);
            jj.set(1, 1, 0.0);
            Ok(())
        };

        let mut solver = NewtonSolver::new(2).unwrap();
        solver.set_use_line_search(false);
        let result = solver.solve(&mut x, args, calc_f, calc_jj);
        assert!(result.is_err());
    }

    // ===== calc_f returns error inside the else branch (no line search) =====

    #[test]
    fn newton_f_returns_error_in_iteration() {
        // Initial calc_f call succeeds; the second call (inside the else branch
        // after the Newton step) fails.
        let mut call_count = 0usize;
        let mut x = Vector::from(&[0.0, 0.0]);
        let args = &mut ();

        let calc_f = |r: &mut Vector, x: &Vector, _: &mut ()| {
            call_count += 1;
            if call_count > 1 {
                return Err("f error on second call");
            }
            r[0] = x[0] - 1.0;
            r[1] = x[1] - 1.0;
            Ok(())
        };

        let calc_jj = |jj: &mut Matrix, _: &Vector, _: &mut ()| {
            jj.set(0, 0, 1.0);
            jj.set(0, 1, 0.0);
            jj.set(1, 0, 0.0);
            jj.set(1, 1, 1.0);
            Ok(())
        };

        let mut solver = NewtonSolver::new(2).unwrap();
        solver.set_use_line_search(false);
        let result = solver.solve(&mut x, args, calc_f, calc_jj);
        assert_eq!(result.err(), Some("f error on second call"));
    }

    // ===== calc_f returns error inside the line search loop =====

    #[test]
    fn newton_f_returns_error_in_line_search() {
        // Initial calc_f call succeeds; the second call (first trial inside the
        // line search loop) fails.
        let mut call_count = 0usize;
        let mut x = Vector::from(&[0.0, 0.0]);
        let args = &mut ();

        let calc_f = |r: &mut Vector, x: &Vector, _: &mut ()| {
            call_count += 1;
            if call_count > 1 {
                return Err("f error in line search");
            }
            r[0] = x[0] - 1.0;
            r[1] = x[1] - 1.0;
            Ok(())
        };

        let calc_jj = |jj: &mut Matrix, _: &Vector, _: &mut ()| {
            jj.set(0, 0, 1.0);
            jj.set(0, 1, 0.0);
            jj.set(1, 0, 0.0);
            jj.set(1, 1, 1.0);
            Ok(())
        };

        let mut solver = NewtonSolver::new(2).unwrap();
        solver.set_use_line_search(true);
        let result = solver.solve(&mut x, args, calc_f, calc_jj);
        assert_eq!(result.err(), Some("f error in line search"));
    }

    // ===== Test Converges in One Iteration =====

    #[test]
    fn newton_converges_immediately() -> Result<(), StrError> {
        let mut x = Vector::from(&[0.0, 0.0]);
        let args = &mut ();

        let calc_f = |r: &mut Vector, x: &Vector, _: &mut ()| {
            r[0] = x[0] - 1.0;
            r[1] = x[1] - 1.0;
            Ok(())
        };

        let calc_jj = |jj: &mut Matrix, _: &Vector, _: &mut ()| {
            jj.set(0, 0, 1.0);
            jj.set(0, 1, 0.0);
            jj.set(1, 0, 0.0);
            jj.set(1, 1, 1.0);
            Ok(())
        };

        let mut solver = NewtonSolver::new(2).unwrap();
        solver.set_use_line_search(false);
        solver.set_tolerance(1e-1);
        solver.solve(&mut x, args, calc_f, calc_jj)?;
        Ok(())
    }
}
