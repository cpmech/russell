#![allow(unused)]

use crate::Method;
use crate::StrError;
use russell_sparse::{Genie, LinSolParams};

/// Defines the configuration parameters for the ODE solver
#[derive(Clone, Debug)]
pub struct OdeParams {
    /// Method
    pub(crate) method: Method,

    /// Linear solver kind
    pub(crate) genie: Genie,

    /// Configurations for sparse linear solver
    pub(crate) lin_sol_params: Option<LinSolParams>,

    /// Minimum stepsize allowed
    pub(crate) h_min: f64,

    /// Initial stepsize
    pub(crate) h_ini: f64,

    /// Max number of iterations
    pub(crate) n_iteration_max: usize,

    /// Max number of steps
    pub(crate) n_step_max: usize,

    /// Min step multiplier
    pub(crate) m_min: f64,

    /// Max step multiplier
    pub(crate) m_max: f64,

    /// Step multiplier factor
    pub(crate) m_factor: f64,

    /// Coefficient to multiply stepsize if first step is rejected [0 ⇒ use dx_new]
    pub(crate) m_first_rejection: f64,

    /// Use Gustafsson's predictive controller
    pub(crate) use_pred_control: bool,

    /// max theta to decide whether the Jacobian should be recomputed or not
    pub(crate) theta_max: f64,

    /// c1 of HW-VII p124 => min ratio to retain previous h
    pub(crate) coef_retain_prev_h_1: f64,

    /// c2 of HW-VII p124 => max ratio to retain previous h
    pub(crate) coef_retain_prev_h_2: f64,

    /// Strategy to select local error computation method
    pub(crate) strategy_local_error: usize,

    /// Solve R and C systems concurrently
    pub(crate) concurrent_r_and_c_systems: bool,

    /// Use modified Newton's method (constant Jacobian) in BwEuler
    pub use_modified_newton: bool,

    /// Use numerical Jacobian, even if analytical Jacobian is available
    pub use_numerical_jacobian: bool,

    /// Use RMS norm instead of Euclidean in BwEuler
    pub(crate) use_rms_norm: bool,

    /// Show messages, e.g. during iterations
    pub(crate) verbose: bool,

    /// Always start iterations with zero trial values (instead of collocation interpolation)
    pub(crate) use_zero_trial: bool,

    /// Lund stabilization coefficient β
    pub(crate) lund_beta: f64,

    /// Factor to multiply Lund stabilization coefficient β
    pub(crate) lund_beta_m: f64,

    /// Number of steps to check stiff situation. 0 ⇒ no check. [default = 1]
    pub(crate) stiffness_n_step: usize,

    /// Maximum value of ρs [default = 0.5]
    pub(crate) stiffness_max_ratio: f64,

    /// Number of "yes" stiff steps allowed [default = 15]
    pub(crate) stiffness_n_yes: usize,

    /// Number of "no" stiff steps to disregard stiffness [default = 6]
    pub(crate) stiffness_n_no: usize,

    /// Step size for dense output
    pub(crate) dense_h: f64,

    /// Activates dense output
    pub(crate) use_dense_output: bool,

    /// Number of dense steps
    pub(crate) dense_n_step: usize,

    /// absolute tolerance
    pub(crate) abs_tol: f64,

    /// relative tolerance
    pub(crate) rel_tol: f64,

    /// Newton's iterations tolerance
    pub(crate) tol_newton: f64,

    /// Min value of previous relative error
    pub(crate) rel_error_prev_min: f64,
}

impl OdeParams {
    /// Allocates a new instance with default values
    pub fn new(method: Method, lin_sol: Option<Genie>, lin_sol_params: Option<LinSolParams>) -> Self {
        let genie = match lin_sol {
            Some(g) => g,
            None => Genie::Umfpack,
        };
        let mut params = OdeParams {
            method,
            genie,
            lin_sol_params,
            h_min: 1.0e-10,
            h_ini: 1.0e-4,
            n_iteration_max: 7,
            n_step_max: 1000,
            m_min: 0.125,
            m_max: 5.0,
            m_factor: 0.9,
            m_first_rejection: 0.1,
            use_pred_control: true,
            theta_max: 1.0e-3,
            coef_retain_prev_h_1: 1.0,
            coef_retain_prev_h_2: 1.2,
            strategy_local_error: 3,
            concurrent_r_and_c_systems: true,
            use_modified_newton: false,
            use_numerical_jacobian: false,
            use_rms_norm: true,
            verbose: false,
            use_zero_trial: false,
            lund_beta: 0.0,
            lund_beta_m: 0.0,
            stiffness_n_step: 0,
            stiffness_max_ratio: 0.5,
            stiffness_n_yes: 15,
            stiffness_n_no: 6,
            dense_h: 0.0,
            use_dense_output: false,
            dense_n_step: 0,
            abs_tol: 0.0,
            rel_tol: 0.0,
            tol_newton: 0.0,
            rel_error_prev_min: 1.0e-4,
        };
        params.set_tolerances(1e-4, 1e-4).unwrap();
        if method == Method::Radau5 {
            params.rel_error_prev_min = 1.0e-2;
        }
        if method == Method::DoPri5 {
            params.lund_beta = 0.04;
            params.lund_beta_m = 0.75;
        }
        if method == Method::DoPri8 {
            params.lund_beta_m = 0.2;
        }
        params
    }

    /// Sets the tolerances
    ///
    /// # Input
    ///
    /// * `abs_tol` -- absolute tolerance
    /// * `rel_tol` -- relative tolerance
    pub fn set_tolerances(&mut self, abs_tol: f64, rel_tol: f64) -> Result<(), StrError> {
        // check
        if abs_tol <= 0.0 {
            return Err("absolute tolerance must be greater than zero");
        }
        if abs_tol <= 10.0 * f64::EPSILON {
            return Err("absolute tolerance must be grater than 10 * EPSILON");
        }
        if rel_tol <= 0.0 {
            return Err("relative tolerance must be greater than zero");
        }

        // set
        self.abs_tol = abs_tol;
        self.rel_tol = rel_tol;

        // change the tolerances (radau5 only)
        if self.method == Method::Radau5 {
            const BETA: f64 = 2.0 / 3.0;
            let quot = self.abs_tol / self.rel_tol;
            self.rel_tol = 0.1 * f64::powf(self.rel_tol, BETA);
            self.abs_tol = self.rel_tol * quot;
        }

        // tolerance for iterations
        self.tol_newton = f64::max(
            10.0 * f64::EPSILON / self.rel_tol,
            f64::min(0.03, f64::sqrt(self.rel_tol)),
        );
        Ok(())
    }

    /// TODO
    pub(crate) fn validate(&self) -> Result<(), StrError> {
        Ok(())
    }
}
