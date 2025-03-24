use super::NlMethod;
use crate::StrError;
use russell_sparse::{Genie, LinSolParams};

/// Holds the local error tolerances and the tolerance for the Newton's method
#[derive(Clone, Copy, Debug)]
pub(crate) struct NlParamsTol {
    /// Absolute tolerance
    pub(crate) abs: f64,

    /// Relative tolerance
    pub(crate) rel: f64,

    /// Tolerance for the Newton-Raphson method
    pub(crate) newton: f64,
}

/// Holds parameters for the Newton-Raphson method
#[derive(Clone, Copy, Debug)]
pub struct NlParamsNewton {
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

    /// Writes the Gu = dG/du matrix and stop (with an error message)
    ///
    /// The file will be written `if n_accepted > nstep`
    ///
    /// Will write the following files:
    ///
    /// * `/tmp/russell_nonlin/ggu.{mtx,smat}` -- Gu matrix
    ///
    /// where `mtx` is the extension for the MatrixMarket files and `smat` is the extension
    /// for the vismatrix files (for visualization).
    ///
    /// # References
    ///
    /// * MatrixMarket: <https://math.nist.gov/MatrixMarket/formats.html>
    /// * Vismatrix: <https://github.com/cpmech/vismatrix>
    pub write_matrix_after_nstep_and_stop: Option<usize>,
}

/// Holds parameters to control the variable stepsize algorithm
#[derive(Clone, Copy, Debug)]
pub struct NlParamsStep {
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

/// Holds all parameters for the ODE Solver
#[derive(Clone, Copy, Debug)]
pub struct NlParams {
    /// ODE solver method
    pub(crate) method: NlMethod,

    /// Holds the local error tolerances and the tolerance for the Newton's method
    pub(crate) tol: NlParamsTol,

    /// Holds parameters for the Newton-Raphson method
    pub newton: NlParamsNewton,

    /// Holds parameters to control the variable stepsize algorithm
    pub step: NlParamsStep,

    /// Enable debugging (print log messages)
    pub debug: bool,
}

// --- Implementations ------------------------------------------------------------------------------------

impl NlParamsTol {
    /// Allocates a new instance
    pub(crate) fn new() -> Self {
        let (abs, rel, newton) = calc_tolerances(1e-4, 1e-4).unwrap();
        NlParamsTol { abs, rel, newton }
    }
}

impl NlParamsNewton {
    /// Allocates a new instance
    pub(crate) fn new() -> Self {
        NlParamsNewton {
            n_iteration_max: 10,
            use_numerical_jacobian: false,
            genie: Genie::Umfpack,
            lin_sol_params: None,
            write_matrix_after_nstep_and_stop: None,
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

impl NlParamsStep {
    /// Allocates a new instance
    pub(crate) fn new() -> Self {
        NlParamsStep {
            m_min: 0.001,
            m_max: 2.0,
            m_safety: 0.9,
            m_first_reject: 0.1,
            h_ini: 1e-4,
            n_step_max: 100_000,
            rel_error_prev_min: 1e-4,
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

impl NlParams {
    /// Allocates a new instance
    pub fn new(method: NlMethod) -> Self {
        NlParams {
            method,
            tol: NlParamsTol::new(),
            newton: NlParamsNewton::new(),
            step: NlParamsStep::new(),
            debug: false,
        }
    }

    /// Sets the tolerances
    pub fn set_tolerances(&mut self, absolute: f64, relative: f64, newton: Option<f64>) -> Result<(), StrError> {
        let (abs, rel, newt) = calc_tolerances(absolute, relative)?;
        self.tol.abs = abs;
        self.tol.rel = rel;
        self.tol.newton = if let Some(n) = newton { n } else { newt };
        Ok(())
    }

    /// Validates all parameters
    pub(crate) fn validate(&self) -> Result<(), StrError> {
        self.newton.validate()?;
        self.step.validate()?;
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
fn calc_tolerances(abs_tol: f64, rel_tol: f64) -> Result<(f64, f64, f64), StrError> {
    // check
    if abs_tol <= 10.0 * f64::EPSILON {
        return Err("the absolute tolerance must be > 10 · EPSILON");
    }
    if rel_tol <= 10.0 * f64::EPSILON {
        return Err("the relative tolerance must be > 10 · EPSILON");
    }

    // tolerance for iterations (line 500 of radau5.f)
    let tol_newton = f64::max(10.0 * f64::EPSILON / rel_tol, f64::min(0.03, f64::sqrt(rel_tol)));
    Ok((abs_tol, rel_tol, tol_newton))
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NlMethod;
    use russell_lab::approx_eq;

    #[test]
    fn derive_methods_work() {
        let tol = NlParamsTol::new();
        let newton = NlParamsNewton::new();
        let step = NlParamsStep::new();
        let params = NlParams::new(NlMethod::Arclength);
        let clone_tol = tol.clone();
        let clone_newton = newton.clone();
        let clone_step = step.clone();
        let clone_params = params.clone();
        assert_eq!(format!("{:?}", tol), format!("{:?}", clone_tol));
        assert_eq!(format!("{:?}", newton), format!("{:?}", clone_newton));
        assert_eq!(format!("{:?}", step), format!("{:?}", clone_step));
        assert_eq!(format!("{:?}", params), format!("{:?}", clone_params));
    }

    #[test]
    fn set_tolerances_captures_errors() {
        let mut params = NlParams::new(NlMethod::Arclength);
        assert_eq!(
            params.set_tolerances(0.0, 1e-4, None).err(),
            Some("the absolute tolerance must be > 10 · EPSILON")
        );
        assert_eq!(
            params.set_tolerances(1e-4, 0.0, None).err(),
            Some("the relative tolerance must be > 10 · EPSILON")
        );
    }

    #[test]
    fn set_tolerances_works() {
        let mut params = NlParams::new(NlMethod::Arclength);
        params.set_tolerances(0.1, 0.1, None).unwrap();
        approx_eq(params.tol.abs, 0.1, 1e-17);
        approx_eq(params.tol.rel, 0.1, 1e-17);
        assert_eq!(params.tol.newton, 0.03);

        params.set_tolerances(0.1, 0.1, Some(0.05)).unwrap();
        approx_eq(params.tol.abs, 0.1, 1e-17);
        approx_eq(params.tol.rel, 0.1, 1e-17);
        assert_eq!(params.tol.newton, 0.05);
    }

    #[test]
    fn params_newton_validate_works() {
        let mut params = NlParamsNewton::new();
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
        let mut params = NlParamsStep::new();
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
    fn params_validate_works() {
        let mut params = NlParams::new(NlMethod::Arclength);
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
        assert_eq!(params.validate().is_err(), false);
    }
}
