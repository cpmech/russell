use russell_lab::Stopwatch;

/// Holds benchmark information
pub struct BenchInfo {
    /// Number of calls to ODE system function
    pub n_function_eval: usize,

    /// Number of Jacobian matrix evaluations
    pub n_jacobian_eval: usize,

    /// Collects the number of steps, successful or not
    pub n_performed_steps: usize,

    /// Collects the number of accepted steps
    pub n_accepted_steps: usize,

    /// Collects the number of rejected steps
    pub n_rejected_steps: usize,

    /// Number of matrix assemblage/factorization
    pub n_matrix_factorization: usize,

    /// Last number of iterations
    pub n_iterations_last: usize,

    /// Max number of iterations among all steps
    pub n_iterations_max: usize,

    /// Optimal step size at the end
    pub h_optimal: f64,

    /// Max nanoseconds spent on steps
    pub nanos_step_max: u128,

    /// Max nanoseconds spent on Jacobian evaluation
    pub nanos_jacobian_max: u128,

    /// Max nanoseconds spent on the coefficient matrix factorization
    pub nanos_factor_max: u128,

    /// Max nanoseconds spent on the solution of the linear system
    pub nanos_lin_sol_max: u128,

    /// Total nanoseconds spent on the solution
    pub nanos_total: u128,

    pub(crate) sw_step: Stopwatch,
    pub(crate) sw_jacobian: Stopwatch,
    pub(crate) sw_factor: Stopwatch,
    pub(crate) sw_lin_sol: Stopwatch,
    pub(crate) sw_total: Stopwatch,
}

impl BenchInfo {
    /// Allocates a new instance
    pub fn new() -> Self {
        BenchInfo {
            n_function_eval: 0,
            n_jacobian_eval: 0,
            n_performed_steps: 0,
            n_accepted_steps: 0,
            n_rejected_steps: 0,
            n_matrix_factorization: 0,
            n_iterations_last: 0,
            n_iterations_max: 0,
            h_optimal: 0.0,
            nanos_step_max: 0,
            nanos_jacobian_max: 0,
            nanos_factor_max: 0,
            nanos_lin_sol_max: 0,
            nanos_total: 0,
            sw_step: Stopwatch::new(""),
            sw_jacobian: Stopwatch::new(""),
            sw_factor: Stopwatch::new(""),
            sw_lin_sol: Stopwatch::new(""),
            sw_total: Stopwatch::new(""),
        }
    }

    /// Resets the values
    pub(crate) fn reset(&mut self) {
        self.n_function_eval = 0;
        self.n_jacobian_eval = 0;
        self.n_performed_steps = 0;
        self.n_accepted_steps = 0;
        self.n_rejected_steps = 0;
        self.n_matrix_factorization = 0;
        self.n_iterations_last = 0;
        self.n_iterations_max = 0;
        self.h_optimal = 0.0;
        self.nanos_step_max = 0;
        self.nanos_jacobian_max = 0;
        self.nanos_factor_max = 0;
        self.nanos_lin_sol_max = 0;
        self.nanos_total = 0;
    }

    pub(crate) fn stop_sw_step(&mut self) {
        let nanos = self.sw_step.stop();
        if nanos > self.nanos_step_max {
            self.nanos_step_max = nanos;
        }
    }

    pub(crate) fn stop_sw_jacobian(&mut self) {
        let nanos = self.sw_jacobian.stop();
        if nanos > self.nanos_jacobian_max {
            self.nanos_jacobian_max = nanos;
        }
    }

    pub(crate) fn stop_sw_factor(&mut self) {
        let nanos = self.sw_factor.stop();
        if nanos > self.nanos_factor_max {
            self.nanos_factor_max = nanos;
        }
    }

    pub(crate) fn stop_sw_lin_sol(&mut self) {
        let nanos = self.sw_lin_sol.stop();
        if nanos > self.nanos_lin_sol_max {
            self.nanos_lin_sol_max = nanos;
        }
    }

    pub(crate) fn stop_sw_total(&mut self) {
        let nanos = self.sw_total.stop();
        self.nanos_total = nanos;
    }

    pub(crate) fn update_n_iterations_max(&mut self) {
        if self.n_iterations_last > self.n_iterations_max {
            self.n_iterations_max = self.n_iterations_last;
        }
    }
}
