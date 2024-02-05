use crate::{Method, StrError};
use russell_sparse::{Genie, LinSolParams};

/// Holds parameters for the BwEuler method
#[derive(Clone, Copy, Debug)]
pub struct ParamsBwEuler {
    /// absolute tolerance
    pub(crate) abs_tol: f64,

    /// relative tolerance
    pub(crate) rel_tol: f64,

    /// Newton's iterations tolerance
    pub(crate) tol_newton: f64,

    /// Use modified Newton's method (constant Jacobian)
    pub use_modified_newton: bool,

    /// Use numerical Jacobian, even if analytical Jacobian is available
    pub use_numerical_jacobian: bool,

    /// Use RMS norm instead of Euclidean
    pub use_rms_norm: bool,

    /// Max number of iterations
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
    /// absolute tolerance
    pub(crate) abs_tol: f64,

    /// relative tolerance
    pub(crate) rel_tol: f64,

    /// Min step multiplier
    pub(crate) m_min: f64,

    /// Max step multiplier
    pub(crate) m_max: f64,

    /// Step multiplier factor
    pub(crate) m_factor: f64,

    /// Lund stabilization coefficient β
    pub(crate) lund_beta: f64,

    /// Factor to multiply Lund stabilization coefficient β
    pub(crate) lund_beta_m: f64,

    /// Activates dense output
    pub(crate) use_dense_output: bool,
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

    /// Initial stepsize
    pub(crate) h_ini: f64,

    /// Min value of previous relative error
    pub(crate) rel_error_prev_min: f64,

    /// Max number of steps
    pub(crate) n_step_max: usize,

    /// Coefficient to multiply stepsize if first step is rejected [0 ⇒ use dx_new]
    pub(crate) m_first_rejection: f64,
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
}

impl ParamsRadau5 {
    pub(crate) fn new() -> Self {
        ParamsRadau5 {}
    }
}

impl ParamsERK {
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
}

impl Params {
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

    pub(crate) fn validate(&self) -> Result<(), StrError> {
        // TODO
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
