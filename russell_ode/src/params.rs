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
    pub genie: Genie,

    /// Configurations for sparse linear solver
    pub lin_sol_params: Option<LinSolParams>,
}

/// Holds parameters for the Radau5 method
#[derive(Clone, Copy, Debug)]
pub struct ParamsRadau5 {
    /// Absolute tolerance
    pub(crate) abs_tol: f64,

    /// Relative tolerance
    pub(crate) rel_tol: f64,

    /// Tolerance for Newton-Raphson method
    pub(crate) tol_newton: f64,

    /// Use numerical Jacobian, even if the analytical Jacobian is available
    pub use_numerical_jacobian: bool,

    /// Always start iterations with zero trial values (instead of collocation interpolation)
    pub zero_trial: bool,

    /// Max number of iterations (must be ≥ 1)
    pub n_iteration_max: usize,

    /// Linear solver kind
    pub genie: Genie,

    /// Configurations for sparse linear solver
    pub lin_sol_params: Option<LinSolParams>,

    // Max value to decide whether the Jacobian should be recomputed or not
    pub theta_max: f64,

    /// c1 of Hairer-Wanner (VII p124): min ratio to retain previous h
    pub c1h: f64,

    /// c2 of Hairer-Wanner (VII p124): max ratio to retain previous h
    pub c2h: f64,

    /// Min step multiplier (must be ≥ 0.001 and < m_max)
    pub m_min: f64,

    /// Max step multiplier (must be ≥ 0.01 and > m_min)
    pub m_max: f64,

    /// Step multiplier factor (must be ≥ 0.1)
    pub m_factor: f64,

    /// Enable concurrent factorization and solution of the two linear systems
    pub concurrent: bool,

    /// Gustafsson's predictive controller
    pub use_pred_control: bool,
}

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
        let (abs_tol, rel_tol, tol_newton) = calc_tolerances(false, 1e-4, 1e-4).unwrap();
        ParamsBwEuler {
            abs_tol,
            rel_tol,
            tol_newton,
            use_modified_newton: false,
            use_numerical_jacobian: false,
            use_rms_norm: true,
            n_iteration_max: 7, // line 436 of radau5.f
            genie: Genie::Umfpack,
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
        let (abs_tol, rel_tol, tol_newton) = calc_tolerances(true, 1e-4, 1e-4).unwrap();
        ParamsRadau5 {
            abs_tol,
            rel_tol,
            tol_newton,
            use_numerical_jacobian: false,
            zero_trial: false,
            n_iteration_max: 7, // line 436 of radau5.f
            genie: Genie::Umfpack,
            lin_sol_params: None,
            theta_max: 1e-3, // line 487 of radau5.f
            c1h: 1.0,        // line 508 of radau5.f
            c2h: 1.2,        // line 513 of radau5.f
            m_min: 0.125,    // line 534 of radau5.f
            m_max: 5.0,      // line 529 of radau5.f
            m_factor: 0.9,   // line 477 of radau5.f
            concurrent: true,
            use_pred_control: true,
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

impl ParamsERK {
    /// Allocates a new instance
    pub(crate) fn new(method: Method) -> Self {
        let (m_min, m_max, lund_beta, lund_beta_m) = match method {
            Method::DoPri5 => (0.2, 10.0, 0.04, 0.75), // lines (276, 281, 287, 381) of dopri5.f
            Method::DoPri8 => (0.333, 6.0, 0.0, 0.2),  // lines (276, 281, 287, 548) of dopri853.f
            _ => (0.2, 10.0, 0.0, 0.0),
        };
        let (abs_tol, rel_tol, _) = calc_tolerances(false, 1e-4, 1e-4).unwrap();
        ParamsERK {
            abs_tol,
            rel_tol,
            m_min,
            m_max,
            m_factor: 0.9, // line 265 of dopri5.f and dopri853.f
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
            return Err("m_min must be < m_max");
        }
        if self.m_max < 0.01 {
            return Err("m_max must be ≥ 0.01");
        }
        if self.m_factor < 0.1 {
            return Err("m_factor must be ≥ 0.1");
        }
        if self.m_factor > 1.0 {
            return Err("m_factor must be ≤ 1.0");
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
        let (h_ini, rel_error_prev_min) = match method {
            Method::Radau5 => (1e-6, 1e-2), // lines (746, 1018) of radau5.f
            _ => (1e-6, 1e-4),              // lines (no default value, 471 of dopri5.f and 661 of dopri853.f)
        };
        Params {
            method,
            bweuler: ParamsBwEuler::new(),
            radau5: ParamsRadau5::new(),
            erk: ParamsERK::new(method),
            h_ini,
            rel_error_prev_min,
            n_step_max: 100000, // lines 426 of radau5.f, 212 of dopri5.f, and 211 of dopri853.f
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
        match self.method {
            Method::BwEuler => {
                let (abs_tol, rel_tol, tol_newton) = calc_tolerances(false, abs_tol, rel_tol)?;
                self.bweuler.abs_tol = abs_tol;
                self.bweuler.rel_tol = rel_tol;
                self.bweuler.tol_newton = tol_newton;
            }
            Method::Radau5 => {
                let (abs_tol, rel_tol, tol_newton) = calc_tolerances(true, abs_tol, rel_tol)?;
                self.radau5.abs_tol = abs_tol;
                self.radau5.rel_tol = rel_tol;
                self.radau5.tol_newton = tol_newton;
            }
            _ => {
                let (abs_tol, rel_tol, _) = calc_tolerances(false, abs_tol, rel_tol)?;
                self.erk.abs_tol = abs_tol;
                self.erk.rel_tol = rel_tol;
            }
        }
        Ok(())
    }

    /// Validates all parameters
    pub(crate) fn validate(&self) -> Result<(), StrError> {
        match self.method {
            Method::BwEuler => self.bweuler.validate()?,
            Method::Radau5 => self.radau5.validate()?,
            _ => self.erk.validate()?,
        }
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
/// * `radau5` -- indicates that the tolerances must be altered for Radau5
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
fn calc_tolerances(radau5: bool, abs_tol: f64, rel_tol: f64) -> Result<(f64, f64, f64), StrError> {
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

    // change the tolerances (according to radau5.f)
    if radau5 {
        const BETA: f64 = 2.0 / 3.0; // line 402 of radau5.f
        let quot = abs_tol / rel_tol; // line 408 of radau5.f
        rel_tol = 0.1 * f64::powf(rel_tol, BETA); // line 409 of radau5.f
        abs_tol = rel_tol * quot; // line 410 of radau5.f
    }

    // tolerance for iterations (line 500 of radau5.f)
    let tol_newton = f64::max(10.0 * f64::EPSILON / rel_tol, f64::min(0.03, f64::sqrt(rel_tol)));
    Ok((abs_tol, rel_tol, tol_newton))
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Params;
    use crate::Method;
    use russell_lab::approx_eq;

    #[test]
    fn information_clone_copy_and_debug_work() {
        let params = Params::new(Method::BwEuler);
        let copy = params;
        let clone = params.clone();
        assert!(format!("{:?}", params).len() > 0);
        assert_eq!(copy.h_ini, params.h_ini);
        assert_eq!(clone.h_ini, params.h_ini);
        let c = params.bweuler.clone();
        assert_eq!(c.abs_tol, params.bweuler.abs_tol);

        let params = Params::new(Method::Radau5);
        let c = params.radau5.clone();
        assert!(format!("{:?}", c).len() > 0);

        let params = Params::new(Method::DoPri5);
        let c = params.erk.clone();
        assert_eq!(c.m_min, params.erk.m_min);
    }

    #[test]
    fn set_tolerances_works() {
        let mut params = Params::new(Method::BwEuler);
        assert_eq!(
            params.set_tolerances(0.0, 1e-4).err(),
            Some("the absolute tolerance must be > 10 · EPSILON")
        );
        assert_eq!(
            params.set_tolerances(1e-4, 0.0).err(),
            Some("the relative tolerance must be > 10 · EPSILON")
        );
        params.set_tolerances(0.01, 0.02).unwrap();
        assert_eq!(params.bweuler.abs_tol, 0.01);
        assert_eq!(params.bweuler.rel_tol, 0.02);
        assert_eq!(params.bweuler.tol_newton, 0.03);

        let mut params = Params::new(Method::DoPri5);
        params.set_tolerances(0.2, 0.3).unwrap();
        assert_eq!(params.erk.abs_tol, 0.2);
        assert_eq!(params.erk.rel_tol, 0.3);

        let mut params = Params::new(Method::Radau5);
        params.set_tolerances(0.1, 0.1).unwrap();
        approx_eq(params.radau5.abs_tol, 2.154434690031884E-02, 1e-17);
        approx_eq(params.radau5.rel_tol, 2.154434690031884E-02, 1e-17);
        assert_eq!(params.radau5.tol_newton, 0.03);
    }

    #[test]
    fn validate_works() {
        let mut params = Params::new(Method::BwEuler);
        params.validate().unwrap();
        params.bweuler.n_iteration_max = 0;
        assert_eq!(params.validate().err(), Some("n_iteration_max must be ≥ 1"));

        let mut params = Params::new(Method::DoPri5);
        params.erk.m_min = 0.0;
        assert_eq!(params.validate().err(), Some("m_min must be ≥ 0.001"));
        params.erk.m_min = 1.0;
        params.erk.m_max = 0.1;
        assert_eq!(params.validate().err(), Some("m_min must be < m_max"));
        params.erk.m_min = 0.002;
        params.erk.m_max = 0.005;
        assert_eq!(params.validate().err(), Some("m_max must be ≥ 0.01"));
        params.erk.m_min = 0.1;
        params.erk.m_max = 1.0;
        params.erk.m_factor = 0.0;
        assert_eq!(params.validate().err(), Some("m_factor must be ≥ 0.1"));
        params.erk.m_factor = 3.0;
        assert_eq!(params.validate().err(), Some("m_factor must be ≤ 1.0"));
        params.erk.m_factor = 0.9;
        params.erk.lund_beta = -1.0;
        assert_eq!(params.validate().err(), Some("lund_beta must be ≥ 0.0"));
        params.erk.lund_beta = 0.0;
        params.erk.lund_beta_m = -1.0;
        assert_eq!(params.validate().err(), Some("lund_beta_m must be ≥ 0.0"));

        params.erk.lund_beta_m = 0.0;
        params.h_ini = 0.0;
        assert_eq!(params.validate().err(), Some("h_ini must be ≥ 1e-8"));
        params.h_ini = 1e-8;
        params.rel_error_prev_min = 0.0;
        assert_eq!(params.validate().err(), Some("rel_error_prev_min must be ≥ 1e-8"));
        params.rel_error_prev_min = 1e-8;
        params.n_step_max = 0;
        assert_eq!(params.validate().err(), Some("n_step_max must be ≥ 1"));
        params.n_step_max = 1;
        params.m_first_rejection = -1.0;
        assert_eq!(params.validate().err(), Some("m_first_rejection must be ≥ 0.0"));

        let mut params = Params::new(Method::Radau5);
        params.validate().unwrap();
        params.radau5.n_iteration_max = 0;
        assert_eq!(params.validate().err(), Some("n_iteration_max must be ≥ 1"));
    }
}
