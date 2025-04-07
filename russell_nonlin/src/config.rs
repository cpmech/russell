use super::NlMethod;
use crate::StrError;
use russell_sparse::{Genie, LinSolParams};

/// Defines the smallest allowed h
pub const NL_CONFIG_H_MIN: f64 = 1e-7;

/// Defines the smallest allowed tolerance
pub const NL_CONFIG_TOL_MIN: f64 = 1e-12;

/// Holds configuration options for the nonlinear solver
#[derive(Clone, Copy, Debug)]
pub struct NlConfig {
    // basic options ----------------------------------------------------------------------
    //
    /// Nonlinear solver method
    pub(crate) method: NlMethod,

    /// Treat the problem as linear
    pub treat_as_linear: bool,

    /// Show stepping messages
    pub verbose: bool,

    /// Show iteration messages
    pub verbose_iterations: bool,

    /// Show legend
    pub verbose_legend: bool,

    // substepping ------------------------------------------------------------------------
    //
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
    /// h_ini ≥ NL_CONFIG_H_MIN
    /// ```
    ///
    /// See [NL_CONFIG_H_MIN]
    pub h_ini: f64,

    /// Allowed min stepsize
    pub h_min_allowed: f64,

    /// Max number of steps
    ///
    /// ```text
    /// n_step_max ≥ 1
    /// ```
    pub n_step_max: usize,

    /// Min value of previous relative error
    ///
    /// ```text
    /// rel_error_prev_min ≥ NL_CONFIG_H_MIN
    /// ```
    ///
    /// See [NL_CONFIG_H_MIN]
    pub rel_error_prev_min: f64,

    /// Max number of lambda increments
    pub n_lambda_max: usize,

    // linear solver ----------------------------------------------------------------------
    //
    /// Linear solver kind
    pub genie: Genie,

    /// Configurations for sparse linear solver
    pub lin_sol_config: Option<LinSolParams>,

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

    // iterations -------------------------------------------------------------------------
    //
    /// Tolerance on max(‖G‖∞,|H|)
    ///
    /// ```text
    /// tol_gh ≥ NL_CONFIG_TOL_MIN
    /// ```
    ///
    /// See [NL_CONFIG_TOL_MIN]
    pub tol_gh: f64,

    /// Tolerance on max(‖δu‖∞,|δλ|)
    ///
    /// ```text
    /// tol_ul ≥ NL_CONFIG_TOL_MIN
    /// ```
    ///
    /// See [NL_CONFIG_TOL_MIN]
    pub tol_ul: f64,

    /// Maximum max(‖δu‖∞,|δλ|) allowed
    pub max_ul_allowed: f64,

    /// Max number of iterations
    ///
    /// ```text
    /// n_iteration_max ≥ 1
    /// ```
    pub n_iteration_max: usize,

    /// Modified Newton's method with constant tangent matrix
    pub constant_tangent: bool,

    /// Use numerical Jacobian, even if the analytical Jacobian is available
    pub use_numerical_jacobian: bool,
}

impl NlConfig {
    /// Allocates a new instance
    pub fn new(method: NlMethod) -> Self {
        NlConfig {
            // basic options
            method,
            treat_as_linear: false,
            verbose: false,
            verbose_iterations: false,
            verbose_legend: false,
            // substepping
            m_min: 0.001,
            m_max: 2.0,
            m_safety: 0.9,
            m_first_reject: 0.1,
            h_ini: 1e-4,
            h_min_allowed: NL_CONFIG_H_MIN,
            n_step_max: 100_000,
            rel_error_prev_min: 1e-4,
            n_lambda_max: 10_000,
            // linear solver
            genie: Genie::Umfpack,
            lin_sol_config: None,
            write_matrix_after_nstep_and_stop: None,
            // iterations
            tol_gh: 1e-10,
            tol_ul: 1e-10,
            max_ul_allowed: 1e8,
            n_iteration_max: 10,
            constant_tangent: false,
            use_numerical_jacobian: false,
        }
    }

    /// Validates all data
    pub(crate) fn validate(&self) -> Result<(), StrError> {
        // substepping

        if self.m_min < 0.001 || self.m_min > 0.5 || self.m_min >= self.m_max {
            return Err("requirement: 0.001 ≤ m_min < 0.5 and m_min < m_max");
        }
        if self.m_max < 0.01 || self.m_max > 20.0 {
            return Err("requirement: 0.01 ≤ m_max ≤ 20 and m_max > m_min");
        }
        if self.m_safety < 0.1 || self.m_safety > 1.0 {
            return Err("requirement: 0.1 ≤ m_safety ≤ 1");
        }
        if self.m_first_reject < 0.0 {
            return Err("requirement: m_first_rejection ≥ 0");
        }
        if self.h_ini < NL_CONFIG_H_MIN {
            return Err("requirement: h_ini ≥ NL_CONFIG_H_MIN");
        }
        if self.n_step_max < 1 {
            return Err("requirement: n_step_max ≥ 1");
        }
        if self.rel_error_prev_min < NL_CONFIG_H_MIN {
            return Err("requirement: rel_error_prev_min ≥ NL_CONFIG_H_MIN");
        }

        // iterations

        if self.tol_gh < NL_CONFIG_TOL_MIN {
            return Err("requirement: tol_gh ≥ NL_CONFIG_TOL_MIN");
        }
        if self.tol_ul < NL_CONFIG_TOL_MIN {
            return Err("requirement: tol_ul ≥ NL_CONFIG_TOL_MIN");
        }
        if self.max_ul_allowed <= 0.0 {
            return Err("requirement: max_ul_allowed > 0");
        }
        if self.n_iteration_max < 1 {
            return Err("requirement: n_iteration_max ≥ 1");
        }
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::NlConfig;
    use crate::NlMethod;

    #[test]
    fn derive_methods_work() {
        let config = NlConfig::new(NlMethod::Arclength);
        let clone_config = config.clone();
        assert_eq!(format!("{:?}", config), format!("{:?}", clone_config));
    }

    #[test]
    fn config_validate_works() {
        let mut config = NlConfig::new(NlMethod::Arclength);

        // substepping

        config.m_min = 0.0;
        assert_eq!(
            config.validate().err(),
            Some("requirement: 0.001 ≤ m_min < 0.5 and m_min < m_max")
        );
        config.m_min = 0.6;
        assert_eq!(
            config.validate().err(),
            Some("requirement: 0.001 ≤ m_min < 0.5 and m_min < m_max")
        );
        config.m_min = 0.02;
        config.m_max = 0.01;
        assert_eq!(
            config.validate().err(),
            Some("requirement: 0.001 ≤ m_min < 0.5 and m_min < m_max")
        );
        config.m_min = 0.001;
        config.m_max = 0.005;
        assert_eq!(
            config.validate().err(),
            Some("requirement: 0.01 ≤ m_max ≤ 20 and m_max > m_min")
        );
        config.m_max = 30.0;
        assert_eq!(
            config.validate().err(),
            Some("requirement: 0.01 ≤ m_max ≤ 20 and m_max > m_min")
        );
        config.m_max = 10.0;
        config.m_safety = 0.0;
        assert_eq!(config.validate().err(), Some("requirement: 0.1 ≤ m_safety ≤ 1"));
        config.m_safety = 1.2;
        assert_eq!(config.validate().err(), Some("requirement: 0.1 ≤ m_safety ≤ 1"));
        config.m_safety = 0.9;
        config.m_first_reject = -1.0;
        assert_eq!(config.validate().err(), Some("requirement: m_first_rejection ≥ 0"));
        config.m_first_reject = 0.0;
        config.h_ini = 0.0;
        assert_eq!(config.validate().err(), Some("requirement: h_ini ≥ NL_CONFIG_H_MIN"));
        config.h_ini = 1e-4;
        config.n_step_max = 0;
        assert_eq!(config.validate().err(), Some("requirement: n_step_max ≥ 1"));
        config.n_step_max = 10;
        config.rel_error_prev_min = 0.0;
        assert_eq!(
            config.validate().err(),
            Some("requirement: rel_error_prev_min ≥ NL_CONFIG_H_MIN")
        );
        config.rel_error_prev_min = 1e-6;

        // iterations

        config.n_iteration_max = 0;
        assert_eq!(config.validate().err(), Some("requirement: n_iteration_max ≥ 1"));
        config.n_iteration_max = 10;
        config.tol_gh = 0.0;
        assert_eq!(config.validate().err(), Some("requirement: tol_gh ≥ NL_CONFIG_TOL_MIN"));
        config.tol_gh = 1e-10;
        config.tol_ul = 0.0;
        assert_eq!(config.validate().err(), Some("requirement: tol_ul ≥ NL_CONFIG_TOL_MIN"));
        config.tol_ul = 1e-10;
        config.max_ul_allowed = 0.0;
        assert_eq!(config.validate().err(), Some("requirement: max_ul_allowed > 0"));
        config.max_ul_allowed = 1e10;

        // all good
        assert_eq!(config.validate().is_err(), false);
    }
}
