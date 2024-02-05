use crate::{Method, StrError};
use russell_sparse::{Genie, LinSolParams};

/// Holds parameters for the BwEuler method
#[derive(Clone, Copy, Debug)]
pub struct ParamsBwEuler {
    /// Absolute tolerance
    pub(crate) abs_tol: f64,

    /// Relative tolerance
    pub(crate) rel_tol: f64,

    /// Tolerance for Newton-Raphson method
    pub(crate) tol_newton: f64,

    /// Use modified Newton's method (constant Jacobian)
    pub use_modified_newton: bool,

    /// Use numerical Jacobian, even if the analytical Jacobian is available
    pub use_numerical_jacobian: bool,

    /// Use RMS norm instead of Euclidean norm to compare residuals
    pub use_rms_norm: bool,

    /// Max number of iterations (must be ≥ 1)
    pub n_iteration_max: usize,

    /// Linear solver kind
    pub lin_sol: Genie,

    /// Configurations for sparse linear solver
    pub lin_sol_params: Option<LinSolParams>,
}

/// Holds parameters for the Radau5 method
#[derive(Clone, Copy, Debug)]
pub struct ParamsRadau5 {}

/// Holds parameters for explicit Runge-Kutta methods
#[derive(Clone, Copy, Debug)]
pub struct ParamsERK {
    /// Absolute tolerance
    pub(crate) abs_tol: f64,

    /// Relative tolerance
    pub(crate) rel_tol: f64,

    /// Min step multiplier (must be ≥ 0.001 and < m_max)
    pub m_min: f64,

    /// Max step multiplier (must be ≥ 0.01 and > m_min)
    pub m_max: f64,

    /// Step multiplier factor (must be ≥ 0.1)
    pub m_factor: f64,

    /// Lund stabilization coefficient β (must be ≥ 0.0)
    pub lund_beta: f64,

    /// Factor to multiply Lund stabilization coefficient β (must be ≥ 0.0)
    pub lund_beta_m: f64,

    /// Activates dense output
    pub use_dense_output: bool,
}

/// Holds parameters for the ODE Solver
#[derive(Clone, Copy, Debug)]
pub struct Params {
    /// ODE solver method
    pub(crate) method: Method,

    /// Parameters for the BwEuler method
    pub bweuler: ParamsBwEuler,

    /// Parameters for the Radau5 method
    pub radau5: ParamsRadau5,

    /// Parameters for explicit Runge-Kutta methods
    pub erk: ParamsERK,

    /// Initial stepsize (must be ≥ 1e-8)
    pub h_ini: f64,

    /// Min value of previous relative error (must be ≥ 1e-8)
    pub rel_error_prev_min: f64,

    /// Max number of steps (must be ≥ 1)
    pub n_step_max: usize,

    /// Coefficient to multiply stepsize if first step is rejected (must be ≥ 0.0)
    ///
    /// If `m_first_rejection == 0.0`, the solver will use `h_new` on an rejected step.
    pub m_first_rejection: f64,
}

impl ParamsBwEuler {
    /// Allocates a new instance
    pub(crate) fn new() -> Self {
        let (abs_tol, rel_tol, tol_newton) = calc_tolerances(Method::BwEuler, 1e-4, 1e-4).unwrap();
        ParamsBwEuler {
            abs_tol,
            rel_tol,
            tol_newton,
            use_modified_newton: false,
            use_numerical_jacobian: false,
            use_rms_norm: true,
            n_iteration_max: 7,
            lin_sol: Genie::Umfpack,
            lin_sol_params: None,
        }
    }

    /// Validates all parameters
    pub(crate) fn validate(&self) -> Result<(), StrError> {
        if self.n_iteration_max < 1 {
            return Err("n_iteration_max must be ≥ 1");
        }
        Ok(())
    }
}

impl ParamsRadau5 {
    /// Allocates a new instance
    pub(crate) fn new() -> Self {
        // TODO
        ParamsRadau5 {}
    }

    /// Validates all parameters
    pub(crate) fn validate(&self) -> Result<(), StrError> {
        // TODO
        Ok(())
    }
}

impl ParamsERK {
    /// Allocates a new instance
    pub(crate) fn new(method: Method) -> Self {
        let (lund_beta, lund_beta_m) = match method {
            Method::DoPri5 => (0.04, 0.75),
            Method::DoPri8 => (0.0, 0.2),
            _ => (0.0, 0.0),
        };
        let (abs_tol, rel_tol, _) = calc_tolerances(Method::BwEuler, 1e-4, 1e-4).unwrap();
        ParamsERK {
            abs_tol,
            rel_tol,
            m_min: 0.125,
            m_max: 5.0,
            m_factor: 0.9,
            lund_beta,
            lund_beta_m,
            use_dense_output: false,
        }
    }

    /// Validates all parameters
    pub(crate) fn validate(&self) -> Result<(), StrError> {
        if self.m_min < 0.001 {
            return Err("m_min must be ≥ 0.001");
        }
        if self.m_min >= self.m_max {
            return Err("m_min must be < than m_max");
        }
        if self.m_max < 0.01 {
            return Err("m_max must be ≥ 0.01");
        }
        if self.m_max <= self.m_min {
            return Err("m_max must be > m_min");
        }
        if self.m_factor < 0.1 {
            return Err("m_factor must be ≥ 0.1");
        }
        if self.lund_beta < 0.0 {
            return Err("lund_beta must be ≥ 0.0");
        }
        if self.lund_beta_m < 0.0 {
            return Err("lund_beta_m must be ≥ 0.0");
        }
        Ok(())
    }
}

impl Params {
    /// Allocates a new instance
    pub fn new(method: Method) -> Self {
        Params {
            method,
            bweuler: ParamsBwEuler::new(),
            radau5: ParamsRadau5::new(),
            erk: ParamsERK::new(method),
            h_ini: 1e-4,
            rel_error_prev_min: 1e-4,
            n_step_max: 1000,
            m_first_rejection: 0.1,
        }
    }

    /// Sets the tolerances
    ///
    /// # Input
    ///
    /// * `abs_tol` -- absolute tolerance
    /// * `rel_tol` -- relative tolerance
    pub fn set_tolerances(&mut self, abs_tol: f64, rel_tol: f64) -> Result<(), StrError> {
        let (abs_tol, rel_tol, tol_newton) = calc_tolerances(self.method, abs_tol, rel_tol)?;
        self.bweuler.abs_tol = abs_tol;
        self.bweuler.rel_tol = rel_tol;
        self.bweuler.tol_newton = tol_newton;
        self.erk.abs_tol = abs_tol;
        self.erk.rel_tol = rel_tol;
        Ok(())
    }

    /// Validates all parameters
    pub(crate) fn validate(&self) -> Result<(), StrError> {
        self.bweuler.validate()?;
        self.radau5.validate()?;
        self.erk.validate()?;
        if self.h_ini < 1e-8 {
            return Err("h_ini must be ≥ 1e-8");
        }
        if self.rel_error_prev_min < 1e-8 {
            return Err("rel_error_prev_min must be ≥ 1e-8");
        }
        if self.n_step_max < 1 {
            return Err("n_step_max must be ≥ 1");
        }
        if self.m_first_rejection < 0.0 {
            return Err("m_first_rejection must be ≥ 0.0");
        }
        Ok(())
    }
}

/// Calculates tolerances
///
/// # Input
///
/// * `abs_tol` -- absolute tolerance
/// * `rel_tol` -- relative tolerance
///
/// # Output
///
/// Returns `(abs_tol, rel_tol, tol_newton)` where:
///
/// * `abs_tol` -- absolute tolerance
/// * `rel_tol` -- relative tolerance
/// * `tol_newton` -- tolerance for Newton's method
fn calc_tolerances(method: Method, abs_tol: f64, rel_tol: f64) -> Result<(f64, f64, f64), StrError> {
    // check
    if abs_tol <= 10.0 * f64::EPSILON {
        return Err("the absolute tolerance must be > 10 · EPSILON");
    }
    if rel_tol <= 10.0 * f64::EPSILON {
        return Err("the relative tolerance must be > 10 · EPSILON");
    }

    // set
    let mut abs_tol = abs_tol;
    let mut rel_tol = rel_tol;

    // change the tolerances (radau5 only)
    if method == Method::Radau5 {
        const BETA: f64 = 2.0 / 3.0;
        let quot = abs_tol / rel_tol;
        rel_tol = 0.1 * f64::powf(rel_tol, BETA);
        abs_tol = rel_tol * quot;
    }

    // tolerance for iterations
    let tol_newton = f64::max(10.0 * f64::EPSILON / rel_tol, f64::min(0.03, f64::sqrt(rel_tol)));
    Ok((abs_tol, rel_tol, tol_newton))
}
