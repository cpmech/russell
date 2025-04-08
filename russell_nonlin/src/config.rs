use super::Method;
use crate::StrError;
use russell_sparse::{Genie, LinSolParams};

/// Defines the smallest allowed h
pub const CONFIG_H_MIN: f64 = 1e-7;

/// Defines the smallest allowed tolerance
pub const CONFIG_TOL_MIN: f64 = 1e-12;

/// Holds configuration options for the nonlinear solver
#[derive(Clone, Copy, Debug)]
pub struct Config {
    // basic options ----------------------------------------------------------------------
    //
    /// Nonlinear solver method
    pub(crate) method: Method,

    /// Treat the problem as linear
    pub(crate) treat_as_linear: bool,

    /// Show stepping messages
    pub(crate) verbose: bool,

    /// Show iteration messages
    pub(crate) verbose_iterations: bool,

    /// Show legend
    pub(crate) verbose_legend: bool,

    /// Show statistics
    pub(crate) verbose_stats: bool,

    // substepping ------------------------------------------------------------------------
    //
    /// Min step multiplier
    ///
    /// ```text
    /// 0.001 ≤ m_min < 0.5   and   m_min < m_max
    /// ```
    pub(crate) m_min: f64,

    /// Max step multiplier
    ///
    /// ```text
    /// 0.01 ≤ m_max ≤ 20   and   m_max > m_min
    /// ```
    pub(crate) m_max: f64,

    /// Step multiplier safety factor
    ///
    /// ```text
    /// 0.1 ≤ m ≤ 1
    /// ```
    pub(crate) m_safety: f64,

    /// Coefficient to multiply the stepsize if the first step is rejected
    ///
    /// ```text
    /// m_first_reject ≥ 0.0
    /// ```
    ///
    /// If `m_first_reject = 0`, the solver will use `h_new` on a rejected step.
    pub(crate) m_first_reject: f64,

    /// Coefficient to multiply the stepsize if the iterations are failing
    ///
    /// ```text
    /// m_failure ≥ 0.001   (recommended = 0.5)
    /// ```
    pub(crate) m_failure: f64,

    /// Initial stepsize
    ///
    /// ```text
    /// h_ini ≥ CONFIG_H_MIN
    /// ```
    ///
    /// See [CONFIG_H_MIN]
    pub(crate) h_ini: f64,

    /// Allowed min stepsize
    pub(crate) h_min_allowed: f64,

    /// Max number of steps
    ///
    /// ```text
    /// n_step_max ≥ 1
    /// ```
    pub(crate) n_step_max: usize,

    /// Min value of previous relative error
    ///
    /// ```text
    /// rel_error_prev_min ≥ CONFIG_H_MIN
    /// ```
    ///
    /// See [CONFIG_H_MIN]
    pub(crate) rel_error_prev_min: f64,

    /// Number of continued rejections allowed
    pub(crate) n_cont_reject_allowed: usize,

    // linear solver ----------------------------------------------------------------------
    //
    /// Linear solver kind
    pub(crate) genie: Genie,

    /// Configurations for sparse linear solver
    pub(crate) lin_sol_config: Option<LinSolParams>,

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
    pub(crate) write_matrix_after_nstep_and_stop: Option<usize>,

    // iterations -------------------------------------------------------------------------
    //
    /// Tolerance on max(‖G‖∞,|H|)
    ///
    /// ```text
    /// tol_gh ≥ CONFIG_TOL_MIN
    /// ```
    ///
    /// See [CONFIG_TOL_MIN]
    pub(crate) tol_gh: f64,

    /// Tolerance on max(‖δu‖∞,|δλ|)
    ///
    /// ```text
    /// tol_ul ≥ CONFIG_TOL_MIN
    /// ```
    ///
    /// See [CONFIG_TOL_MIN]
    pub(crate) tol_ul: f64,

    /// Maximum max(‖δu‖∞,|δλ|) allowed
    pub(crate) max_ul_allowed: f64,

    /// Max number of iterations
    ///
    /// ```text
    /// n_iteration_max ≥ 1
    /// ```
    pub(crate) n_iteration_max: usize,

    /// Number of allowed continuing divergence on max(‖δu‖∞,|δλ|)
    pub(crate) n_allowed_cont_div_ul: usize,

    /// Modified Newton's method with constant tangent matrix
    pub(crate) constant_tangent: bool,

    /// Use numerical Jacobian, even if the analytical Jacobian is available
    pub(crate) use_numerical_jacobian: bool,
}

impl Config {
    /// Allocates a new instance
    pub fn new(method: Method) -> Self {
        Config {
            // basic options
            method,
            treat_as_linear: false,
            verbose: false,
            verbose_iterations: false,
            verbose_legend: false,
            verbose_stats: false,
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
            // linear solver
            genie: Genie::Umfpack,
            lin_sol_config: None,
            write_matrix_after_nstep_and_stop: None,
            // iterations
            tol_gh: 1e-10,
            tol_ul: 1e-10,
            max_ul_allowed: 1e8,
            n_iteration_max: 12,
            n_allowed_cont_div_ul: 1,
            constant_tangent: false,
            use_numerical_jacobian: false,
        }
    }

    /// Sets a flag to treat the problem as linear (and skip iterations)
    pub fn set_treat_as_linear(&mut self, flag: bool) -> &mut Self {
        self.treat_as_linear = flag;
        self
    }

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

    /// Sets the min step multiplier
    pub fn set_m_min(&mut self, value: f64) -> &mut Self {
        self.m_min = value;
        self
    }

    /// Sets the max step multiplier
    pub fn set_m_max(&mut self, value: f64) -> &mut Self {
        self.m_max = value;
        self
    }

    /// Sets the step multiplier safety factor
    pub fn set_m_safety(&mut self, value: f64) -> &mut Self {
        self.m_safety = value;
        self
    }

    /// Sets the coefficient to multiply the stepsize if the first step is rejected
    pub fn set_m_first_reject(&mut self, value: f64) -> &mut Self {
        self.m_first_reject = value;
        self
    }

    /// Sets the coefficient to multiply the stepsize if the iterations are failing
    pub fn set_m_failure(&mut self, value: f64) -> &mut Self {
        self.m_failure = value;
        self
    }

    /// Sets the initial stepsize
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
    pub fn set_n_step_max(&mut self, value: usize) -> &mut Self {
        self.n_step_max = value;
        self
    }

    /// Sets the min value of previous relative error
    pub fn set_rel_error_prev_min(&mut self, value: f64) -> &mut Self {
        self.rel_error_prev_min = value;
        self
    }

    /// Sets the number of continued rejections allowed
    pub fn set_n_cont_reject_allowed(&mut self, value: usize) -> &mut Self {
        self.n_cont_reject_allowed = value;
        self
    }

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
    pub fn set_write_matrix_after_nstep_and_stop(&mut self, value: Option<usize>) -> &mut Self {
        self.write_matrix_after_nstep_and_stop = value;
        self
    }

    /// Sets the tolerance on max(‖G‖∞,|H|)
    pub fn set_tol_gh(&mut self, value: f64) -> &mut Self {
        self.tol_gh = value;
        self
    }

    /// Sets the tolerance on max(‖δu‖∞,|δλ|)
    pub fn set_tol_ul(&mut self, value: f64) -> &mut Self {
        self.tol_ul = value;
        self
    }

    /// Sets the maximum max(‖δu‖∞,|δλ|) allowed
    pub fn set_max_ul_allowed(&mut self, value: f64) -> &mut Self {
        self.max_ul_allowed = value;
        self
    }

    /// Sets the max number of iterations
    pub fn set_n_iteration_max(&mut self, value: usize) -> &mut Self {
        self.n_iteration_max = value;
        self
    }

    /// Sets the number of allowed continuing divergence on max(‖δu‖∞,|δλ|)
    pub fn set_n_allowed_cont_div_ul(&mut self, value: usize) -> &mut Self {
        self.n_allowed_cont_div_ul = value;
        self
    }

    /// Sets the constant tangent flag
    pub fn set_constant_tangent(&mut self, flag: bool) -> &mut Self {
        self.constant_tangent = flag;
        self
    }

    /// Sets the use numerical Jacobian flag
    pub fn set_use_numerical_jacobian(&mut self, flag: bool) -> &mut Self {
        self.use_numerical_jacobian = flag;
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
        if self.m_failure < 0.1 {
            return Err("requirement: m_failure ≥ 0.001");
        }
        if self.h_ini < CONFIG_H_MIN {
            return Err("requirement: h_ini ≥ CONFIG_H_MIN");
        }
        if self.n_step_max < 1 {
            return Err("requirement: n_step_max ≥ 1");
        }
        if self.rel_error_prev_min < CONFIG_H_MIN {
            return Err("requirement: rel_error_prev_min ≥ CONFIG_H_MIN");
        }

        // iterations

        if self.tol_gh < CONFIG_TOL_MIN {
            return Err("requirement: tol_gh ≥ CONFIG_TOL_MIN");
        }
        if self.tol_ul < CONFIG_TOL_MIN {
            return Err("requirement: tol_ul ≥ CONFIG_TOL_MIN");
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

        config.n_iteration_max = 0;
        assert_eq!(config.validate().err(), Some("requirement: n_iteration_max ≥ 1"));
        config.n_iteration_max = 10;
        config.tol_gh = 0.0;
        assert_eq!(config.validate().err(), Some("requirement: tol_gh ≥ CONFIG_TOL_MIN"));
        config.tol_gh = 1e-10;
        config.tol_ul = 0.0;
        assert_eq!(config.validate().err(), Some("requirement: tol_ul ≥ CONFIG_TOL_MIN"));
        config.tol_ul = 1e-10;
        config.max_ul_allowed = 0.0;
        assert_eq!(config.validate().err(), Some("requirement: max_ul_allowed > 0"));
        config.max_ul_allowed = 1e10;

        // all good
        assert_eq!(config.validate().is_err(), false);
    }
}
