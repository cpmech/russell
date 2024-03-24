use crate::{Method, StrError};
use russell_sparse::{Genie, LinSolParams};

/// Holds the local error tolerances and the tolerance for the Newton's method
#[derive(Clone, Copy, Debug)]
pub(crate) struct ParamsTol {
    /// Absolute tolerance
    pub(crate) abs: f64,

    /// Relative tolerance
    pub(crate) rel: f64,

    /// Tolerance for the Newton-Raphson method
    pub(crate) newton: f64,
}

/// Holds parameters for the Newton-Raphson method
#[derive(Clone, Copy, Debug)]
pub struct ParamsNewton {
    /// Max number of iterations
    ///
    /// ```text
    /// n_iteration_max ≥ 1
    /// ```
    pub n_iteration_max: usize,

    /// Use numerical Jacobian, even if the analytical Jacobian is available
    pub use_numerical_jacobian: bool,

    /// Linear solver kind
    pub genie: Genie,

    /// Configurations for sparse linear solver
    pub lin_sol_params: Option<LinSolParams>,
}

/// Holds parameters to control the variable stepsize algorithm
#[derive(Clone, Copy, Debug)]
pub struct ParamsStep {
    /// Min step multiplier
    ///
    /// ```text
    /// 0.001 ≤ m_min < 0.5   and   m_min < m_max
    /// ```
    pub m_min: f64,

    /// Max step multiplier
    ///
    /// ```text
    /// 0.01 ≤ m_max ≤ 20   and   m_max > m_min
    /// ```
    pub m_max: f64,

    /// Step multiplier safety factor
    ///
    /// ```text
    /// 0.1 ≤ m ≤ 1
    /// ```
    pub m_safety: f64,

    /// Coefficient to multiply the stepsize if the first step is rejected
    ///
    /// ```text
    /// m_first_reject ≥ 0.0
    /// ```
    ///
    /// If `m_first_reject = 0`, the solver will use `h_new` on a rejected step.
    pub m_first_reject: f64,

    /// Initial stepsize
    ///
    /// ```text
    /// h_ini ≥ 1e-8
    /// ```
    pub h_ini: f64,

    /// Max number of steps
    ///
    /// ```text
    /// n_step_max ≥ 1
    /// ```
    pub n_step_max: usize,

    /// Min value of previous relative error
    ///
    /// ```text
    /// rel_error_prev_min ≥ 1e-8
    /// ```
    pub rel_error_prev_min: f64,
}

/// Holds parameters for the stiffness detection algorithm
#[derive(Clone, Copy, Debug)]
pub struct ParamsStiffness {
    /// Enables stiffness detection (for some methods such as DoPri5 and DoPri8)
    pub enabled: bool,

    /// Return an error if stiffness is detected
    ///
    /// **Note:** The default is `true`, i.e., the program will stop if stiffness is detected
    pub stop_with_error: bool,

    /// Save the results at the stations where stiffness has been detected
    ///
    /// **Note:** [crate::Output] must be provided to save the results.
    pub save_results: bool,

    /// Number of steps to ratify the stiffness, i.e., to make sure that stiffness is repeatedly been detected
    pub ratified_after_nstep: usize,

    /// Number of steps to ignore already detected stiffness stations
    pub ignored_after_nstep: usize,

    /// Number of initial accepted steps to skip before enabling the stiffness detection
    pub skip_first_n_accepted_step: usize,

    /// Holds the max h·ρ value approximating the boundary of the stability region and used to detected stiffness
    ///
    /// Note: ρ is the approximation of |λ|, where λ is the dominant eigenvalue of the Jacobian
    /// (see Hairer-Wanner Part II page 22)
    pub(crate) h_times_rho_max: f64,
}

/// Holds the parameters for the BwEuler method
#[derive(Clone, Copy, Debug)]
pub struct ParamsBwEuler {
    /// Use modified Newton's method (constant Jacobian)
    pub use_modified_newton: bool,
}

/// Holds the parameters for the Radau5 method
#[derive(Clone, Copy, Debug)]
pub struct ParamsRadau5 {
    /// Always start iterations with zero trial values (instead of collocation interpolation)
    pub zero_trial: bool,

    /// Max θ value to decide whether the Jacobian should be recomputed or not
    ///
    /// ```text
    /// theta_max ≥ 1e-7
    /// ```
    pub theta_max: f64,

    /// c1 of Hairer-Wanner (VII p124): min ratio to retain previous h (and thus the Jacobian)
    ///
    /// ```text
    /// 0.5 ≤ c1h ≤ 1.5   and   c1h < c2h
    /// ````
    pub c1h: f64,

    /// c2 of Hairer-Wanner (VII p124): max ratio to retain previous h (and thus the Jacobian)
    ///
    /// ```text
    /// 1 ≤ c2h ≤ 2   and   c2h > c1h
    /// ```
    pub c2h: f64,

    /// Enable concurrent factorization and solution of the two linear systems
    pub concurrent: bool,

    /// Gustafsson's predictive controller
    pub use_pred_control: bool,
}

/// Holds the parameters for explicit Runge-Kutta methods
#[derive(Clone, Copy, Debug)]
pub struct ParamsERK {
    /// Lund stabilization coefficient β
    ///
    /// ```text
    /// 0 ≤ lund_beta ≤ 0.1
    /// ```
    pub lund_beta: f64,

    /// Factor to multiply the Lund stabilization coefficient β
    ///
    /// ```text
    /// 0 ≤ lund_m ≤ 1
    /// ```
    pub lund_m: f64,
}

/// Holds all parameters for the ODE Solver
#[derive(Clone, Copy, Debug)]
pub struct Params {
    /// ODE solver method
    pub(crate) method: Method,

    /// Holds the local error tolerances and the tolerance for the Newton's method
    pub(crate) tol: ParamsTol,

    /// Holds parameters for the Newton-Raphson method
    pub newton: ParamsNewton,

    /// Holds parameters to control the variable stepsize algorithm
    pub step: ParamsStep,

    /// Holds parameters for the stiffness detection algorithm
    pub stiffness: ParamsStiffness,

    /// Parameters for the BwEuler method
    pub bweuler: ParamsBwEuler,

    /// Parameters for the Radau5 method
    pub radau5: ParamsRadau5,

    /// Parameters for explicit Runge-Kutta methods
    pub erk: ParamsERK,

    /// Enable debugging (print log messages)
    pub debug: bool,
}

// --- Implementations ------------------------------------------------------------------------------------

impl ParamsTol {
    /// Allocates a new instance
    pub(crate) fn new(method: Method) -> Self {
        let radau5 = method == Method::Radau5;
        let (abs, rel, newton) = calc_tolerances(radau5, 1e-4, 1e-4).unwrap();
        ParamsTol { abs, rel, newton }
    }
}

impl ParamsNewton {
    /// Allocates a new instance
    pub(crate) fn new() -> Self {
        ParamsNewton {
            n_iteration_max: 7, // line 436 of radau5.f
            use_numerical_jacobian: false,
            genie: Genie::Umfpack,
            lin_sol_params: None,
        }
    }

    /// Validates the parameters
    pub(crate) fn validate(&self) -> Result<(), StrError> {
        if self.n_iteration_max < 1 {
            return Err("parameter must satisfy: n_iteration_max ≥ 1");
        }
        Ok(())
    }
}

impl ParamsStep {
    /// Allocates a new instance
    pub(crate) fn new(method: Method) -> Self {
        let (m_min, m_max, m_safety, rel_error_prev_min) = match method {
            Method::Radau5 => (0.125, 5.0, 0.9, 1e-2), // lines (534, 529, 477, 1018) of radau5.f
            Method::DoPri5 => (0.2, 10.0, 0.9, 1e-4),  // lines (276, 281, 265, 471) of dopri5.f
            Method::DoPri8 => (0.333, 6.0, 0.9, 1e-4), // lines (276, 281, 265, 661) of dop853.f
            _ => (0.2, 10.0, 0.9, 1e-4),
        };
        ParamsStep {
            m_min,
            m_max,
            m_safety,
            m_first_reject: 0.1,
            h_ini: 1e-4,
            n_step_max: 100000, // lines (426, 212, 211) of (radau5.f, dopri5.f, dop853.f)
            rel_error_prev_min,
        }
    }

    /// Validates the parameters
    pub(crate) fn validate(&self) -> Result<(), StrError> {
        if self.m_min < 0.001 || self.m_min > 0.5 || self.m_min >= self.m_max {
            return Err("parameter must satisfy: 0.001 ≤ m_min < 0.5 and m_min < m_max");
        }
        if self.m_max < 0.01 || self.m_max > 20.0 {
            return Err("parameter must satisfy: 0.01 ≤ m_max ≤ 20 and m_max > m_min");
        }
        if self.m_safety < 0.1 || self.m_safety > 1.0 {
            return Err("parameter must satisfy: 0.1 ≤ m_safety ≤ 1");
        }
        if self.m_first_reject < 0.0 {
            return Err("parameter must satisfy: m_first_rejection ≥ 0");
        }
        if self.h_ini < 1e-8 {
            return Err("parameter must satisfy: h_ini ≥ 1e-8");
        }
        if self.n_step_max < 1 {
            return Err("parameter must satisfy: n_step_max ≥ 1");
        }
        if self.rel_error_prev_min < 1e-8 {
            return Err("parameter must satisfy: rel_error_prev_min ≥ 1e-8");
        }
        Ok(())
    }
}

impl ParamsStiffness {
    /// Allocates a new instance
    pub(crate) fn new(method: Method) -> Self {
        let h_times_lambda_max = match method {
            Method::DoPri5 => 3.25, // line 482 of dopri5.f
            Method::DoPri8 => 6.1,  // line 674 of dopri8.f
            _ => f64::MAX,          // undetermined
        };
        ParamsStiffness {
            enabled: false,
            stop_with_error: true,
            save_results: false,
            ratified_after_nstep: 15, // lines (485, 677) of (dopri5.f, dop853.f)
            ignored_after_nstep: 6,   // lines (492, 684) of (dopri5.f, dop853.f)
            skip_first_n_accepted_step: 10,
            h_times_rho_max: h_times_lambda_max,
        }
    }
}

impl ParamsBwEuler {
    /// Allocates a new instance
    pub(crate) fn new() -> Self {
        ParamsBwEuler {
            use_modified_newton: false,
        }
    }
}

impl ParamsRadau5 {
    /// Allocates a new instance
    pub(crate) fn new() -> Self {
        ParamsRadau5 {
            zero_trial: false,
            theta_max: 1e-3, // line 487 of radau5.f
            c1h: 1.0,        // line 508 of radau5.f
            c2h: 1.2,        // line 513 of radau5.f
            concurrent: true,
            use_pred_control: true,
        }
    }

    /// Validates the parameters
    pub(crate) fn validate(&self) -> Result<(), StrError> {
        if self.theta_max < 1e-7 {
            return Err("parameter must satisfy: theta_max ≥ 1e-7");
        }
        if self.c1h < 0.5 || self.c1h > 1.5 || self.c1h >= self.c2h {
            return Err("parameter must satisfy: 0.5 ≤ c1h ≤ 1.5 and c1h < c2h");
        }
        if self.c2h < 1.0 || self.c2h > 2.0 {
            return Err("parameter must satisfy: 1 ≤ c2h ≤ 2 and c2h > c1h");
        }
        Ok(())
    }
}

impl ParamsERK {
    /// Allocates a new instance
    pub(crate) fn new(method: Method) -> Self {
        let (lund_beta, lund_m) = match method {
            Method::DoPri5 => (0.04, 0.75), // lines (287, 381) of dopri5.f
            Method::DoPri8 => (0.0, 0.2),   // lines (287, 548) of dop853.f
            _ => (0.0, 0.0),
        };
        ParamsERK { lund_beta, lund_m }
    }

    /// Validates the parameters
    pub(crate) fn validate(&self) -> Result<(), StrError> {
        if self.lund_beta < 0.0 || self.lund_beta > 0.1 {
            return Err("parameter must satisfy: 0 ≤ lund_beta ≤ 0.1");
        }
        if self.lund_m < 0.0 || self.lund_m > 1.0 {
            return Err("parameter must satisfy: 0 ≤ lund_m ≤ 1");
        }
        Ok(())
    }
}

impl Params {
    /// Allocates a new instance
    pub fn new(method: Method) -> Self {
        Params {
            method,
            tol: ParamsTol::new(method),
            newton: ParamsNewton::new(),
            step: ParamsStep::new(method),
            stiffness: ParamsStiffness::new(method),
            bweuler: ParamsBwEuler::new(),
            radau5: ParamsRadau5::new(),
            erk: ParamsERK::new(method),
            debug: false,
        }
    }

    /// Sets the tolerances
    pub fn set_tolerances(&mut self, absolute: f64, relative: f64, newton: Option<f64>) -> Result<(), StrError> {
        let radau5 = self.method == Method::Radau5;
        let (abs, rel, newt) = calc_tolerances(radau5, absolute, relative)?;
        self.tol.abs = abs;
        self.tol.rel = rel;
        self.tol.newton = if let Some(n) = newton { n } else { newt };
        Ok(())
    }

    /// Validates all parameters
    pub(crate) fn validate(&self) -> Result<(), StrError> {
        self.newton.validate()?;
        self.step.validate()?;
        self.radau5.validate()?;
        self.erk.validate()?;
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
    use super::*;
    use russell_lab::approx_eq;

    #[test]
    fn derive_methods_work() {
        let tol = ParamsTol::new(Method::Radau5);
        let newton = ParamsNewton::new();
        let step = ParamsStep::new(Method::Radau5);
        let stiffness = ParamsStiffness::new(Method::Radau5);
        let bweuler = ParamsBwEuler::new();
        let radau5 = ParamsRadau5::new();
        let erk = ParamsERK::new(Method::DoPri5);
        let params = Params::new(Method::Radau5);
        let clone_tol = tol.clone();
        let clone_newton = newton.clone();
        let clone_step = step.clone();
        let clone_stiffness = stiffness.clone();
        let clone_bweuler = bweuler.clone();
        let clone_radau5 = radau5.clone();
        let clone_erk = erk.clone();
        let clone_params = params.clone();
        assert_eq!(format!("{:?}", tol), format!("{:?}", clone_tol));
        assert_eq!(format!("{:?}", newton), format!("{:?}", clone_newton));
        assert_eq!(format!("{:?}", step), format!("{:?}", clone_step));
        assert_eq!(format!("{:?}", stiffness), format!("{:?}", clone_stiffness));
        assert_eq!(format!("{:?}", bweuler), format!("{:?}", clone_bweuler));
        assert_eq!(format!("{:?}", radau5), format!("{:?}", clone_radau5));
        assert_eq!(format!("{:?}", erk), format!("{:?}", clone_erk));
        assert_eq!(format!("{:?}", params), format!("{:?}", clone_params));
    }

    #[test]
    fn set_tolerances_captures_errors() {
        for method in [Method::Radau5, Method::DoPri5] {
            let mut params = Params::new(method);
            assert_eq!(
                params.set_tolerances(0.0, 1e-4, None).err(),
                Some("the absolute tolerance must be > 10 · EPSILON")
            );
            assert_eq!(
                params.set_tolerances(1e-4, 0.0, None).err(),
                Some("the relative tolerance must be > 10 · EPSILON")
            );
        }
    }

    #[test]
    fn set_tolerances_works() {
        let mut params = Params::new(Method::Radau5);
        params.set_tolerances(0.1, 0.1, None).unwrap();
        approx_eq(params.tol.abs, 2.154434690031884E-02, 1e-17);
        approx_eq(params.tol.rel, 2.154434690031884E-02, 1e-17);
        assert_eq!(params.tol.newton, 0.03);

        params.set_tolerances(0.1, 0.1, Some(0.05)).unwrap();
        approx_eq(params.tol.abs, 2.154434690031884E-02, 1e-17);
        approx_eq(params.tol.rel, 2.154434690031884E-02, 1e-17);
        assert_eq!(params.tol.newton, 0.05);

        let mut params = Params::new(Method::DoPri5);
        params.set_tolerances(0.2, 0.3, None).unwrap();
        assert_eq!(params.tol.abs, 0.2);
        assert_eq!(params.tol.rel, 0.3);
    }

    #[test]
    fn params_newton_validate_works() {
        let mut params = ParamsNewton::new();
        params.n_iteration_max = 0;
        assert_eq!(
            params.validate().err(),
            Some("parameter must satisfy: n_iteration_max ≥ 1")
        );
        params.n_iteration_max = 10;
        assert_eq!(params.validate().is_err(), false);
    }

    #[test]
    fn params_step_validate_works() {
        let mut params = ParamsStep::new(Method::Radau5);
        params.m_min = 0.0;
        assert_eq!(
            params.validate().err(),
            Some("parameter must satisfy: 0.001 ≤ m_min < 0.5 and m_min < m_max")
        );
        params.m_min = 0.6;
        assert_eq!(
            params.validate().err(),
            Some("parameter must satisfy: 0.001 ≤ m_min < 0.5 and m_min < m_max")
        );
        params.m_min = 0.02;
        params.m_max = 0.01;
        assert_eq!(
            params.validate().err(),
            Some("parameter must satisfy: 0.001 ≤ m_min < 0.5 and m_min < m_max")
        );
        params.m_min = 0.001;
        params.m_max = 0.005;
        assert_eq!(
            params.validate().err(),
            Some("parameter must satisfy: 0.01 ≤ m_max ≤ 20 and m_max > m_min")
        );
        params.m_max = 30.0;
        assert_eq!(
            params.validate().err(),
            Some("parameter must satisfy: 0.01 ≤ m_max ≤ 20 and m_max > m_min")
        );
        params.m_max = 10.0;
        params.m_safety = 0.0;
        assert_eq!(
            params.validate().err(),
            Some("parameter must satisfy: 0.1 ≤ m_safety ≤ 1")
        );
        params.m_safety = 1.2;
        assert_eq!(
            params.validate().err(),
            Some("parameter must satisfy: 0.1 ≤ m_safety ≤ 1")
        );
        params.m_safety = 0.9;
        params.m_first_reject = -1.0;
        assert_eq!(
            params.validate().err(),
            Some("parameter must satisfy: m_first_rejection ≥ 0")
        );
        params.m_first_reject = 0.0;
        params.h_ini = 0.0;
        assert_eq!(params.validate().err(), Some("parameter must satisfy: h_ini ≥ 1e-8"));
        params.h_ini = 1e-4;
        params.n_step_max = 0;
        assert_eq!(params.validate().err(), Some("parameter must satisfy: n_step_max ≥ 1"));
        params.n_step_max = 10;
        params.rel_error_prev_min = 0.0;
        assert_eq!(
            params.validate().err(),
            Some("parameter must satisfy: rel_error_prev_min ≥ 1e-8")
        );
        params.rel_error_prev_min = 1e-6;
        assert_eq!(params.validate().is_err(), false);
    }

    #[test]
    fn params_radau5_validate_works() {
        let mut params = ParamsRadau5::new();
        params.theta_max = 0.0;
        assert_eq!(
            params.validate().err(),
            Some("parameter must satisfy: theta_max ≥ 1e-7")
        );
        params.theta_max = 1e-7;
        params.c1h = 0.0;
        assert_eq!(
            params.validate().err(),
            Some("parameter must satisfy: 0.5 ≤ c1h ≤ 1.5 and c1h < c2h")
        );
        params.c1h = 2.0;
        assert_eq!(
            params.validate().err(),
            Some("parameter must satisfy: 0.5 ≤ c1h ≤ 1.5 and c1h < c2h")
        );
        params.c1h = 1.3;
        params.c2h = 1.2;
        assert_eq!(
            params.validate().err(),
            Some("parameter must satisfy: 0.5 ≤ c1h ≤ 1.5 and c1h < c2h")
        );
        params.c1h = 1.0;
        params.c2h = 3.0;
        assert_eq!(
            params.validate().err(),
            Some("parameter must satisfy: 1 ≤ c2h ≤ 2 and c2h > c1h")
        );
        params.c2h = 1.2;
        assert_eq!(params.validate().is_err(), false);
    }

    #[test]
    fn params_erk_validate_works() {
        let mut params = ParamsERK::new(Method::DoPri5);
        params.lund_beta = -1.0;
        assert_eq!(
            params.validate().err(),
            Some("parameter must satisfy: 0 ≤ lund_beta ≤ 0.1")
        );
        params.lund_beta = 0.2;
        assert_eq!(
            params.validate().err(),
            Some("parameter must satisfy: 0 ≤ lund_beta ≤ 0.1")
        );
        params.lund_beta = 0.1;
        params.lund_m = -1.0;
        assert_eq!(params.validate().err(), Some("parameter must satisfy: 0 ≤ lund_m ≤ 1"));
        params.lund_m = 1.1;
        assert_eq!(params.validate().err(), Some("parameter must satisfy: 0 ≤ lund_m ≤ 1"));
        params.lund_m = 0.75;
        assert_eq!(params.validate().is_err(), false);
    }

    #[test]
    fn params_validate_works() {
        let mut params = Params::new(Method::Radau5);
        params.newton.n_iteration_max = 0;
        assert_eq!(
            params.validate().err(),
            Some("parameter must satisfy: n_iteration_max ≥ 1")
        );
        params.newton.n_iteration_max = 10;
        params.step.m_min = 0.0;
        assert_eq!(
            params.validate().err(),
            Some("parameter must satisfy: 0.001 ≤ m_min < 0.5 and m_min < m_max")
        );
        params.step.m_min = 0.001;
        params.radau5.theta_max = 0.0;
        assert_eq!(
            params.validate().err(),
            Some("parameter must satisfy: theta_max ≥ 1e-7")
        );
        params.radau5.theta_max = 1e-7;
        params.erk.lund_beta = -0.1;
        assert_eq!(
            params.validate().err(),
            Some("parameter must satisfy: 0 ≤ lund_beta ≤ 0.1")
        );
        params.erk.lund_beta = 0.1;
        assert_eq!(params.validate().is_err(), false);
    }
}
