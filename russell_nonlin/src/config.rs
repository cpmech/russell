use super::Method;
use crate::StrError;
use russell_sparse::{Genie, LinSolParams};

/// Defines the smallest allowed h
pub const CONFIG_H_MIN: f64 = 1e-10;

/// Defines the smallest allowed tolerance
pub const CONFIG_TOL_MIN: f64 = 1e-12;

/// Holds configuration options for the nonlinear solver
#[derive(Clone, Copy, Debug)]
pub struct Config {
    // basic options ----------------------------------------------------------------------
    //
    /// Nonlinear solver method
    pub(crate) method: Method,

    /// Show stepping messages
    pub(crate) verbose: bool,

    /// Show iteration messages
    pub(crate) verbose_iterations: bool,

    /// Show legend
    pub(crate) verbose_legend: bool,

    /// Show statistics
    pub(crate) verbose_stats: bool,

    /// Hide timings when displaying statistics
    pub(crate) hide_timings: bool,

    /// Indicates whether to record the stepsizes used (in statistics)
    pub(crate) record_stepsizes: bool,

    /// Indicates whether to record the iterations residuals or not (in statistics)
    pub(crate) record_iterations_residuals: bool,

    // substepping ------------------------------------------------------------------------
    //
    /// Min step multiplier
    pub(crate) m_min: f64,

    /// Max step multiplier
    pub(crate) m_max: f64,

    /// Step multiplier safety factor
    pub(crate) m_safety: f64,

    /// Coefficient to multiply the stepsize if the first step is rejected
    pub(crate) m_first_reject: f64,

    /// Coefficient to multiply the stepsize if the iterations are failing
    pub(crate) m_failure: f64,

    /// Initial stepsize
    pub(crate) h_ini: f64,

    /// Allowed min stepsize
    pub(crate) h_min_allowed: f64,

    /// Max number of steps
    pub(crate) n_step_max: usize,

    /// Min value of previous relative error
    pub(crate) rel_error_prev_min: f64,

    /// Number of continued rejections allowed
    pub(crate) n_cont_reject_allowed: usize,

    /// Number of continued (iteration) failures allowed
    pub(crate) n_cont_failure_allowed: usize,

    // linear solver ----------------------------------------------------------------------
    //
    /// Linear solver kind
    pub(crate) genie: Genie,

    /// Configurations for sparse linear solver
    pub(crate) lin_sol_config: Option<LinSolParams>,

    /// Writes the Gu = dG/du matrix and stop (with an error message)
    pub(crate) write_matrix_after_nstep_and_stop: Option<usize>,

    // iterations -------------------------------------------------------------------------
    //
    /// Absolute tolerance on ‖G,N‖∞
    pub(crate) tol_abs_residual: f64,

    /// Absolute tolerance on rms = Rel((δu,δλ))
    pub(crate) tol_abs_delta: f64,

    /// Relative tolerance on rms = Rel((δu,δλ))
    pub(crate) tol_rel_delta: f64,

    /// Allowed ‖δu,δλ‖∞ max
    pub(crate) allowed_delta_max: f64,

    /// Allowed number of iterations
    pub(crate) allowed_iterations: usize,

    /// Allowed number of continued divergence on ‖δu,δλ‖∞
    pub(crate) allowed_continued_divergence: usize,

    /// Modified Newton's method with constant tangent matrix
    pub(crate) constant_tangent: bool,

    /// Use numerical Jacobian, even if the analytical Jacobian is available
    pub(crate) use_numerical_jacobian: bool,

    // pseudo-arclength -------------------------------------------------------------------
    //
    /// Use the bordering algorithm throughout the entire simulation
    pub(crate) bordering: bool,

    /// Max angle between the tangent and the secant
    pub(crate) alpha_max: f64,

    /// Sets the maximum pseudo-arclength step size σ
    ///
    /// Note that `σ ≈ Δs` only if Δs is small, i.e., σ is the pseudo-arclength.
    pub(crate) sigma_max: f64,

    /// Records the predictor values for debugging
    pub(crate) debug_predictor: bool,
}

impl Config {
    /// Allocates a new instance
    pub fn new(method: Method) -> Self {
        Config {
            // basic options
            method,
            verbose: false,
            verbose_iterations: false,
            verbose_legend: false,
            verbose_stats: false,
            hide_timings: false,
            record_stepsizes: false,
            record_iterations_residuals: false,
            // substepping
            m_min: 0.001,
            m_max: 2.0,
            m_safety: 0.9,
            m_first_reject: 0.1,
            m_failure: 0.5,
            h_ini: 1e-4,
            h_min_allowed: CONFIG_H_MIN,
            n_step_max: 100_000,
            rel_error_prev_min: 1e-4,
            n_cont_reject_allowed: 10,
            n_cont_failure_allowed: 5,
            // linear solver
            genie: Genie::Umfpack,
            lin_sol_config: None,
            write_matrix_after_nstep_and_stop: None,
            // iterations
            tol_abs_residual: 1e-10,
            tol_abs_delta: 1e-10,
            tol_rel_delta: 1e-7,
            allowed_delta_max: 1e8,
            allowed_iterations: 12,
            allowed_continued_divergence: 1,
            constant_tangent: false,
            use_numerical_jacobian: false,
            // pseudo-arclength
            bordering: false,
            alpha_max: 5.0,
            sigma_max: 0.01,
            debug_predictor: false,
        }
    }

    // basic options ----------------------------------------------------------------------

    /// Sets the verbose flag
    pub fn set_verbose(&mut self, flag: bool, show_iterations: bool, show_stats: bool) -> &mut Self {
        self.verbose = flag;
        self.verbose_iterations = show_iterations;
        self.verbose_stats = show_stats;
        self
    }

    /// Shows the legend
    pub fn set_show_legend(&mut self, flag: bool) -> &mut Self {
        self.verbose_legend = flag;
        self
    }

    /// Hides timings when displaying statistics
    pub fn set_hide_timings(&mut self, flag: bool) -> &mut Self {
        self.hide_timings = flag;
        self
    }

    /// Indicates whether to record the stepsizes used or not (in statistics)
    pub fn set_record_stepsizes(&mut self, flag: bool) -> &mut Self {
        self.record_stepsizes = flag;
        self
    }

    /// Indicates whether to record the iterations residuals or not (in statistics)
    pub fn set_record_iterations_residuals(&mut self, flag: bool) -> &mut Self {
        self.record_iterations_residuals = flag;
        self
    }

    // substepping ------------------------------------------------------------------------

    /// Sets the min step multiplier
    ///
    /// ```text
    /// 0.001 ≤ m_min < 0.5   and   m_min < m_max
    /// ```
    pub fn set_m_min(&mut self, value: f64) -> &mut Self {
        self.m_min = value;
        self
    }

    /// Sets the max step multiplier
    ///
    /// ```text
    /// 0.01 ≤ m_max ≤ 20   and   m_max > m_min
    /// ```
    pub fn set_m_max(&mut self, value: f64) -> &mut Self {
        self.m_max = value;
        self
    }

    /// Sets the step multiplier safety factor
    ///
    /// ```text
    /// 0.1 ≤ m ≤ 1
    /// ```
    pub fn set_m_safety(&mut self, value: f64) -> &mut Self {
        self.m_safety = value;
        self
    }

    /// Sets the coefficient to multiply the stepsize if the first step is rejected
    ///
    /// ```text
    /// m_first_reject ≥ 0.0
    /// ```
    ///
    /// If `m_first_reject = 0`, the solver will use `h_new` on a rejected step.
    pub fn set_m_first_reject(&mut self, value: f64) -> &mut Self {
        self.m_first_reject = value;
        self
    }

    /// Sets the coefficient to multiply the stepsize if the iterations are failing
    ///
    /// ```text
    /// m_failure ≥ 0.001   (recommended = 0.5)
    /// ```
    pub fn set_m_failure(&mut self, value: f64) -> &mut Self {
        self.m_failure = value;
        self
    }

    /// Sets the initial stepsize
    ///
    /// ```text
    /// h_ini ≥ CONFIG_H_MIN
    /// ```
    ///
    /// See [CONFIG_H_MIN]
    pub fn set_h_ini(&mut self, value: f64) -> &mut Self {
        self.h_ini = value;
        self
    }

    /// Sets the allowed min stepsize
    pub fn set_h_min_allowed(&mut self, value: f64) -> &mut Self {
        self.h_min_allowed = value;
        self
    }

    /// Sets the max number of steps
    ///
    /// ```text
    /// n_step_max ≥ 1
    /// ```
    pub fn set_n_step_max(&mut self, value: usize) -> &mut Self {
        self.n_step_max = value;
        self
    }

    /// Sets the min value of previous relative error
    ///
    /// ```text
    /// rel_error_prev_min ≥ CONFIG_H_MIN
    /// ```
    ///
    /// See [CONFIG_H_MIN]
    pub fn set_rel_error_prev_min(&mut self, value: f64) -> &mut Self {
        self.rel_error_prev_min = value;
        self
    }

    /// Sets the number of continued rejections allowed
    pub fn set_n_cont_reject_allowed(&mut self, value: usize) -> &mut Self {
        self.n_cont_reject_allowed = value;
        self
    }

    /// Sets the number of continued (iteration) failures allowed
    pub fn set_n_cont_failure_allowed(&mut self, value: usize) -> &mut Self {
        self.n_cont_failure_allowed = value;
        self
    }

    // linear solver ----------------------------------------------------------------------

    /// Sets the linear solver kind
    pub fn set_genie(&mut self, genie: Genie) -> &mut Self {
        self.genie = genie;
        self
    }

    /// Sets configurations for sparse linear solver
    pub fn set_lin_sol_config(&mut self, config: Option<LinSolParams>) -> &mut Self {
        self.lin_sol_config = config;
        self
    }

    /// Sets the option to write matrix after n step and stop
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
    pub fn set_write_matrix_after_nstep_and_stop(&mut self, value: Option<usize>) -> &mut Self {
        self.write_matrix_after_nstep_and_stop = value;
        self
    }

    // iterations -------------------------------------------------------------------------

    /// Sets the absolute tolerance on ‖G,N‖∞
    ///
    /// ```text
    /// value ≥ CONFIG_TOL_MIN
    /// ```
    ///
    /// See [CONFIG_TOL_MIN]
    ///
    /// Default value: 1e-10
    pub fn set_tol_residual(&mut self, tol_abs: f64) -> &mut Self {
        self.tol_abs_residual = tol_abs;
        self
    }

    /// Sets the absolute and relative tolerance on rms = Rel((δu,δλ))
    ///
    /// ```text
    ///        ___________________________
    /// rms = √ (1/N) ∑ᵢ [(δu,δλ)ᵢ / scᵢ]²
    /// scᵢ = ϵₐ + ϵᵣ |(u,λ)⁰ᵢ|
    /// ϵₐ and ϵᵣ ≥ CONFIG_TOL_MIN
    /// ```
    ///
    /// See [CONFIG_TOL_MIN]
    ///
    /// Default values: tol_abs = 1e-10, tol_rel = 1e-7
    pub fn set_tol_delta(&mut self, tol_abs: f64, tol_rel: f64) -> &mut Self {
        self.tol_abs_delta = tol_abs;
        self.tol_rel_delta = tol_rel;
        self
    }

    /// Sets the allowed ‖δu,δλ‖∞ max
    ///
    /// Default value: 1e8
    pub fn set_allowed_delta_max(&mut self, value: f64) -> &mut Self {
        self.allowed_delta_max = value;
        self
    }

    /// Sets the allowed number of iterations
    ///
    /// ```text
    /// value ≥ 1
    /// ```
    ///
    /// Default value: 12
    pub fn set_allowed_iterations(&mut self, value: usize) -> &mut Self {
        self.allowed_iterations = value;
        self
    }

    /// Sets the allowed number of continued divergence on ‖δu,δλ‖∞
    ///
    /// Default value: 1
    pub fn set_allowed_continued_divergence(&mut self, value: usize) -> &mut Self {
        self.allowed_continued_divergence = value;
        self
    }

    /// Sets the constant tangent flag
    ///
    /// Modified Newton's method with constant tangent matrix
    ///
    /// Default value: false
    pub fn set_constant_tangent(&mut self, flag: bool) -> &mut Self {
        self.constant_tangent = flag;
        self
    }

    /// Sets the use numerical Jacobian flag
    ///
    /// Use numerical Jacobian, even if the analytical Jacobian is available
    ///
    /// Default value: false
    pub fn set_use_numerical_jacobian(&mut self, flag: bool) -> &mut Self {
        self.use_numerical_jacobian = flag;
        self
    }

    // pseudo-arclength -------------------------------------------------------------------

    /// Sets the bordering flag
    ///
    /// Use the bordering algorithm throughout the entire simulation
    ///
    /// Default value: false
    pub fn set_bordering(&mut self, flag: bool) -> &mut Self {
        self.bordering = flag;
        self
    }

    /// Sets the maximum angle between the tangent and the secant
    ///
    /// Default value: 5.0
    pub fn set_alpha_max(&mut self, value: f64) -> &mut Self {
        self.alpha_max = value;
        self
    }

    /// Sets the maximum pseudo-arclength step size σ
    ///
    /// Note that `σ ≈ Δs` only if Δs is small, i.e., σ is the pseudo-arclength.
    ///
    /// Default value: 0.01
    pub fn set_sigma_max(&mut self, value: f64) -> &mut Self {
        self.sigma_max = value;
        self
    }

    /// Records the predictor values for debugging
    pub fn set_debug_predictor(&mut self, flag: bool) -> &mut Self {
        self.debug_predictor = flag;
        self
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
        if self.m_failure < 0.001 {
            return Err("requirement: m_failure ≥ 0.001");
        }
        if self.h_ini < CONFIG_H_MIN {
            return Err("requirement: h_ini ≥ CONFIG_H_MIN");
        }
        if self.h_min_allowed < CONFIG_H_MIN {
            return Err("requirement: h_min_allowed ≥ CONFIG_H_MIN");
        }
        if self.n_step_max < 1 {
            return Err("requirement: n_step_max ≥ 1");
        }
        if self.rel_error_prev_min < CONFIG_H_MIN {
            return Err("requirement: rel_error_prev_min ≥ CONFIG_H_MIN");
        }

        // iterations

        if self.tol_abs_residual < CONFIG_TOL_MIN {
            return Err("requirement: tol_abs_residual ≥ CONFIG_TOL_MIN");
        }
        if self.tol_abs_delta < CONFIG_TOL_MIN {
            return Err("requirement: tol_abs_delta ≥ CONFIG_TOL_MIN");
        }
        if self.tol_rel_delta < CONFIG_TOL_MIN {
            return Err("requirement: tol_rel_delta ≥ CONFIG_TOL_MIN");
        }
        if self.allowed_delta_max <= 0.0 {
            return Err("requirement: allowed_delta_max > 0");
        }
        if self.allowed_iterations < 1 {
            return Err("requirement: allowed_iterations ≥ 1");
        }

        // pseudo-arclength
        if self.alpha_max <= 0.0 {
            return Err("requirement: alpha_max > 0");
        }
        if self.sigma_max <= 0.0 {
            return Err("requirement: sigma_max > 0");
        }
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Config;
    use crate::Method;

    #[test]
    fn derive_methods_work() {
        let config = Config::new(Method::Arclength);
        let clone_config = config.clone();
        assert_eq!(format!("{:?}", config), format!("{:?}", clone_config));
    }

    #[test]
    fn config_validate_works() {
        let mut config = Config::new(Method::Arclength);

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
        assert_eq!(config.validate().err(), Some("requirement: h_ini ≥ CONFIG_H_MIN"));
        config.h_ini = 1e-4;
        config.n_step_max = 0;
        assert_eq!(config.validate().err(), Some("requirement: n_step_max ≥ 1"));
        config.n_step_max = 10;
        config.rel_error_prev_min = 0.0;
        assert_eq!(
            config.validate().err(),
            Some("requirement: rel_error_prev_min ≥ CONFIG_H_MIN")
        );
        config.rel_error_prev_min = 1e-6;

        // iterations

        config.allowed_iterations = 0;
        assert_eq!(config.validate().err(), Some("requirement: allowed_iterations ≥ 1"));
        config.allowed_iterations = 10;
        config.tol_abs_residual = 0.0;
        assert_eq!(
            config.validate().err(),
            Some("requirement: tol_abs_residual ≥ CONFIG_TOL_MIN")
        );
        config.tol_abs_residual = 1e-10;
        config.tol_abs_delta = 0.0;
        assert_eq!(
            config.validate().err(),
            Some("requirement: tol_abs_delta ≥ CONFIG_TOL_MIN")
        );
        config.tol_abs_delta = 1e-10;
        config.tol_rel_delta = 0.0;
        assert_eq!(
            config.validate().err(),
            Some("requirement: tol_rel_delta ≥ CONFIG_TOL_MIN")
        );
        config.tol_rel_delta = 1e-10;
        config.allowed_delta_max = 0.0;
        assert_eq!(config.validate().err(), Some("requirement: allowed_delta_max > 0"));
        config.allowed_delta_max = 1e10;

        // all good
        assert_eq!(config.validate().is_err(), false);
    }
}
