use super::{Method, SoderlindClass};
use crate::StrError;
use russell_sparse::{Genie, LinSolParams};

/// Defines the smallest allowed h
pub const CONFIG_H_MIN: f64 = 1e-10;

/// Defines the smallest allowed tolerance
pub const CONFIG_TOL_MIN: f64 = 1e-12;

/// Holds configuration options for the nonlinear solver
#[derive(Clone, Debug)]
pub struct Config {
    // basic options ----------------------------------------------------------------------
    //
    /// Nonlinear solver method
    pub(crate) method: Method,

    /// Save the output to a log file instead of stdout
    pub(crate) log_file: Option<String>,

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

    /// Indicates whether to record the iterations residuals or not (in statistics)
    pub(crate) record_iterations_residuals: bool,

    // automatic stepsize -----------------------------------------------------------------
    //
    /// Coefficient to multiply the stepsize if the iterations are failing
    pub(crate) m_failure: f64,

    /// Initial stepsize
    pub(crate) h_ini: f64,

    /// Max number of steps
    pub(crate) n_step_max: usize,

    /// Maximum allowed number of continued iteration failures
    pub(crate) n_cont_failure_max: usize,

    /// Maximum allowed number of continued rejections (due to large curvatures, etc.)
    pub(crate) n_cont_rejection_max: usize,

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

    /// Maximum allowed ‖δu,δλ‖∞ max
    pub(crate) delta_max_allowed: f64,

    /// Maximum allowed number of iterations
    pub(crate) n_iteration_max: usize,

    /// Maximum allowed number of continued divergence on ‖δu,δλ‖∞
    pub(crate) n_cont_divergence_max: usize,

    /// Modified Newton's method with constant tangent matrix
    pub(crate) constant_tangent: bool,

    /// Use numerical Jacobian, even if the analytical Jacobian is available
    pub(crate) use_numerical_jacobian: bool,

    // pseudo-arclength -------------------------------------------------------------------
    //
    /// Use the bordering algorithm throughout the entire simulation
    pub(crate) bordering: bool,

    /// Records the predictor values for debugging
    pub(crate) debug_predictor: bool,

    // stepsize control -----------------------------------------------------------------
    //
    /// Enables the Newton-Raphson stepsize control
    pub(crate) nr_control_enabled: bool,

    /// Enables the tangent vector stepsize control
    pub(crate) tg_control_enabled: bool,

    /// Uses the PID coefficients from Valli-Carey-Coutinho (VCC) for the tangent vector stepsize control
    ///
    /// The coefficients recommended in Ref 1 (page 212) are:
    ///
    /// ```text
    /// KP = 0.075
    /// KI = 0.175
    /// KD = 0.01
    /// ```
    ///
    /// See also Ref 2 and Ref 3.
    ///
    /// # References
    ///
    /// 1. Valli AMP, Carey GF, Coutinho ALGA (2005) Control strategies for timestep selection in nite element
    ///    simulation of incompressible flows and coupled reaction–convection–diffusion processes,
    ///    International Journal for Numerical Methods in Fluids, 47:201-231, <https://doi.org/10.1002/fld.805>
    /// 2. Barros GF, Cortes AMA, Coutinho ALGA (2021) Finite element solution of nonlocal Cahn–Hilliard
    ///    equations with feedback control time step size adaptivity, International Journal for Numerical Methods
    ///    in Engineering, 122:5028-5052, <https://doi.org/10.1002/nme.6755>
    /// 3. Kubatschek T, Forster A (2025) Investigation of existing and new approaches to step size control in a
    ///    continuation framework, Computers & Structures, 313:107747, <https://doi.org/10.1016/j.compstruc.2025.107747>
    pub(crate) tg_control_pid_vcc: bool,

    /// Optimal number of iterations for stepsize control using Newton-Raphson statistics
    pub(crate) nr_control_n_opt: usize,

    /// Beta coefficient used with the NR stepsize control
    pub(crate) nr_control_beta: f64,

    /// Absolute tolerance for the tangent vector stepsize control
    pub(crate) tg_control_atol: f64,

    /// Relative tolerance for the tangent vector stepsize control
    pub(crate) tg_control_rtol: f64,

    /// First exponent for the tangent vector stepsize control
    ///
    /// See Equation (18) on page 7 of Soderlind (2003)
    ///
    /// # References
    ///
    /// 1. Soderlind (2003) Digital filters in adaptive time-stepping,
    ///    ACM Transactions on Mathematical Software, 29(1), 1-26.
    /// 2. Soderlind and Wang (2006) Adaptive time-stepping and computational stability,
    ///    Journal of Computational and Applied Mathematics, 185, 225-243.
    pub(crate) tg_control_beta1: f64,

    /// Second exponent for the tangent vector stepsize control
    ///
    /// See Equation (18) on page 7 of Soderlind (2003)
    ///
    /// References:
    /// * Soderlind (2003) Digital filters in adaptive time-stepping,
    ///   ACM Transactions on Mathematical Software, 29(1), 1-26.
    /// * Soderlind and Wang (2006) Adaptive time-stepping and computational stability,
    ///   Journal of Computational and Applied Mathematics, 185, 225-243.
    pub(crate) tg_control_beta2: f64,

    /// Third exponent for the tangent vector stepsize control
    ///
    /// See Equation (18) on page 7 of Soderlind (2003)
    ///
    /// References:
    /// * Soderlind (2003) Digital filters in adaptive time-stepping,
    ///   ACM Transactions on Mathematical Software, 29(1), 1-26.
    /// * Soderlind and Wang (2006) Adaptive time-stepping and computational stability,
    ///   Journal of Computational and Applied Mathematics, 185, 225-243.
    pub(crate) tg_control_beta3: f64,

    /// Fourth exponent for the tangent vector stepsize control
    ///
    /// See Equation (18) on page 7 of Soderlind (2003)
    ///
    /// Reference:
    /// * Soderlind (2003) Digital filters in adaptive time-stepping,
    ///   ACM Transactions on Mathematical Software, 29(1), 1-26.
    /// * Soderlind and Wang (2006) Adaptive time-stepping and computational stability,
    ///   Journal of Computational and Applied Mathematics, 185, 225-243.
    pub(crate) tg_control_alpha2: f64,

    /// Fifth exponent for the tangent vector stepsize control
    ///
    /// See Equation (18) on page 7 of Soderlind (2003)
    ///
    /// Reference:
    /// * Soderlind (2003) Digital filters in adaptive time-stepping,
    ///   ACM Transactions on Mathematical Software, 29(1), 1-26.
    /// * Soderlind and Wang (2006) Adaptive time-stepping and computational stability,
    ///   Journal of Computational and Applied Mathematics, 185, 225-243.
    pub(crate) tg_control_alpha3: f64,
}

impl Config {
    /// Allocates a new instance
    ///
    /// The default method is [Method::Natural]; use [Config::set_method()] to change it.
    pub fn new() -> Self {
        let (b1, b2, b3, a2, a3) = SoderlindClass::H211PI.params();
        Config {
            // basic options
            method: Method::Natural,
            log_file: None,
            verbose: false,
            verbose_iterations: false,
            verbose_legend: false,
            verbose_stats: false,
            hide_timings: false,
            record_iterations_residuals: false,
            // automatic stepsize
            m_failure: 0.5,
            h_ini: 1e-4,
            n_step_max: 100_000,
            n_cont_failure_max: 5,
            n_cont_rejection_max: 5,
            // linear solver
            genie: Genie::Umfpack,
            lin_sol_config: None,
            write_matrix_after_nstep_and_stop: None,
            // iterations
            tol_abs_residual: 1e-10,
            tol_abs_delta: 1e-10,
            tol_rel_delta: 1e-7,
            delta_max_allowed: 1e8,
            n_iteration_max: 20,
            n_cont_divergence_max: 2,
            constant_tangent: false,
            use_numerical_jacobian: false,
            // pseudo-arclength
            bordering: false,
            debug_predictor: false,
            // stepsize control
            nr_control_enabled: true,
            tg_control_enabled: true,
            tg_control_pid_vcc: true,
            nr_control_n_opt: 3,
            nr_control_beta: 0.5,
            tg_control_atol: 1e-2,
            tg_control_rtol: 1e-2,
            tg_control_beta1: b1,
            tg_control_beta2: b2,
            tg_control_beta3: b3,
            tg_control_alpha2: a2,
            tg_control_alpha3: a3,
        }
    }

    // basic options ----------------------------------------------------------------------

    /// Sets the method
    pub fn set_method(&mut self, method: Method) -> &mut Self {
        self.method = method;
        self
    }

    /// Sets the log file path, to save the output instead of stdout
    pub fn set_log_file(&mut self, full_path: &str) -> &mut Self {
        self.log_file = Some(full_path.to_string());
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

    /// Hides timings when displaying statistics
    pub fn set_hide_timings(&mut self, flag: bool) -> &mut Self {
        self.hide_timings = flag;
        self
    }

    /// Indicates whether to record the iterations residuals or not (in statistics)
    pub fn set_record_iterations_residuals(&mut self, flag: bool) -> &mut Self {
        self.record_iterations_residuals = flag;
        self
    }

    // automatic stepsize -----------------------------------------------------------------

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
    /// h_ini > CONFIG_H_MIN
    /// ```
    ///
    /// See [CONFIG_H_MIN]
    pub fn set_h_ini(&mut self, value: f64) -> &mut Self {
        self.h_ini = value;
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

    /// Sets the maximum allowed number of continued iteration failures
    ///
    /// Default value: 5
    pub fn set_n_cont_failure_max(&mut self, value: usize) -> &mut Self {
        self.n_cont_failure_max = value;
        self
    }

    /// Sets the maximum allowed number of continued rejections (due to large curvatures, etc.)
    ///
    /// Default value: 5
    pub fn set_n_cont_rejection_max(&mut self, value: usize) -> &mut Self {
        self.n_cont_rejection_max = value;
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
    pub fn set_delta_max_allowed(&mut self, value: f64) -> &mut Self {
        self.delta_max_allowed = value;
        self
    }

    /// Sets the allowed number of iterations
    ///
    /// ```text
    /// value ≥ 1
    /// ```
    ///
    /// Default value: 20
    pub fn set_n_iteration_max(&mut self, value: usize) -> &mut Self {
        self.n_iteration_max = value;
        self
    }

    /// Sets the maximum allowed number of continued divergence on ‖δu,δλ‖∞
    ///
    /// Default value: 2
    pub fn set_n_cont_divergence_max(&mut self, value: usize) -> &mut Self {
        self.n_cont_divergence_max = value;
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

    /// Records the predictor values for debugging
    pub fn set_debug_predictor(&mut self, flag: bool) -> &mut Self {
        self.debug_predictor = flag;
        self
    }

    /// Sets the Newton-Raphson stepsize control flag
    ///
    /// Enables or disables the Newton-Raphson stepsize control
    ///
    /// Default value: true
    pub fn set_nr_control_enabled(&mut self, flag: bool) -> &mut Self {
        self.nr_control_enabled = flag;
        self
    }

    /// Sets the tangent vector stepsize control flag
    ///
    /// Enables or disables the tangent vector stepsize control
    ///
    /// Default value: true
    pub fn set_tg_control_enabled(&mut self, flag: bool) -> &mut Self {
        self.tg_control_enabled = flag;
        self
    }

    /// Sets the use of the PID coefficients from Valli-Carey-Coutinho (VCC) for the tangent vector stepsize control
    ///
    /// The coefficients recommended in Ref 1 (page 212) are:
    ///
    /// ```text
    /// KP = 0.075
    /// KI = 0.175
    /// KD = 0.01
    /// ```
    ///
    /// See also Ref 2 and Ref 3.
    ///
    /// # References
    ///
    /// 1. Valli AMP, Carey GF, Coutinho ALGA (2005) Control strategies for timestep selection in nite element
    ///    simulation of incompressible flows and coupled reaction–convection–diffusion processes,
    ///    International Journal for Numerical Methods in Fluids, 47:201-231, <https://doi.org/10.1002/fld.805>
    /// 2. Barros GF, Cortes AMA, Coutinho ALGA (2021) Finite element solution of nonlocal Cahn–Hilliard
    ///    equations with feedback control time step size adaptivity, International Journal for Numerical Methods
    ///    in Engineering, 122:5028-5052, <https://doi.org/10.1002/nme.6755>
    /// 3. Kubatschek T, Forster A (2025) Investigation of existing and new approaches to step size control in a
    ///    continuation framework, Computers & Structures, 313:107747, <https://doi.org/10.1016/j.compstruc.2025.107747>
    ///
    /// Default value: true
    pub fn set_tg_control_pid_vcc(&mut self, flag: bool) -> &mut Self {
        self.tg_control_pid_vcc = flag;
        self
    }

    /// Sets the optimal number of iterations for stepsize control using Newton-Raphson statistics
    ///
    /// Default value: 3
    pub fn set_nr_control_n_opt(&mut self, value: usize) -> &mut Self {
        self.nr_control_n_opt = value;
        self
    }

    /// Sets the beta coefficient used with the NR stepsize control
    ///
    /// Default value: 0.5
    pub fn set_nr_control_beta(&mut self, value: f64) -> &mut Self {
        self.nr_control_beta = value;
        self
    }

    /// Sets the absolute tolerance for the tangent vector stepsize control
    ///
    /// Default value: 1e-2
    pub fn set_tg_control_atol(&mut self, value: f64) -> &mut Self {
        self.tg_control_atol = value;
        self
    }

    /// Sets the relative tolerance for the tangent vector stepsize control
    ///
    /// Default value: 1e-2
    pub fn set_tg_control_rtol(&mut self, value: f64) -> &mut Self {
        self.tg_control_rtol = value;
        self
    }

    /// Sets the absolute and relative tolerance with the same value for the tangent vector stepsize control
    ///
    /// Default values: atol = 1e-2, rtol = 1e-2
    pub fn set_tg_control_atol_and_rtol(&mut self, tol: f64) -> &mut Self {
        self.tg_control_atol = tol;
        self.tg_control_rtol = tol;
        self
    }

    /// Sets the first exponent for the tangent vector stepsize control
    ///
    /// See Equation (18) on page 7 of Soderlind (2003)
    ///
    /// # References
    ///
    /// * Soderlind (2003) Digital filters in adaptive time-stepping,
    ///   ACM Transactions on Mathematical Software, 29(1), 1-26.
    /// * Soderlind and Wang (2006) Adaptive time-stepping and computational stability,
    ///   Journal of Computational and Applied Mathematics, 185, 225-243.
    pub fn set_tg_control_beta1(&mut self, value: f64) -> &mut Self {
        self.tg_control_beta1 = value;
        self
    }

    /// Sets the second exponent for the tangent vector stepsize control
    ///
    /// See Equation (18) on page 7 of Soderlind (2003)
    ///
    /// # References
    ///
    /// * Soderlind (2003) Digital filters in adaptive time-stepping,
    ///   ACM Transactions on Mathematical Software, 29(1), 1-26.
    /// * Soderlind and Wang (2006) Adaptive time-stepping and computational stability,
    ///   Journal of Computational and Applied Mathematics, 185, 225-243.
    pub fn set_tg_control_beta2(&mut self, value: f64) -> &mut Self {
        self.tg_control_beta2 = value;
        self
    }

    /// Sets the third exponent for the tangent vector stepsize control
    ///
    /// See Equation (18) on page 7 of Soderlind (2003)
    ///
    /// # References
    ///
    /// * Soderlind (2003) Digital filters in adaptive time-stepping,
    ///   ACM Transactions on Mathematical Software, 29(1), 1-26.
    /// * Soderlind and Wang (2006) Adaptive time-stepping and computational stability,
    ///   Journal of Computational and Applied Mathematics, 185, 225-243.
    pub fn set_tg_control_beta3(&mut self, value: f64) -> &mut Self {
        self.tg_control_beta3 = value;
        self
    }

    /// Sets the fourth exponent for the tangent vector stepsize control
    ///
    /// See Equation (18) on page 7 of Soderlind (2003)
    ///
    /// # References
    ///
    /// * Soderlind (2003) Digital filters in adaptive time-stepping,
    ///   ACM Transactions on Mathematical Software, 29(1), 1-26.
    /// * Soderlind and Wang (2006) Adaptive time-stepping and computational stability,
    ///   Journal of Computational and Applied Mathematics, 185, 225-243.
    pub fn set_tg_control_alpha2(&mut self, value: f64) -> &mut Self {
        self.tg_control_alpha2 = value;
        self
    }

    /// Sets the fifth exponent for the tangent vector stepsize control
    ///
    /// See Equation (18) on page 7 of Soderlind (2003)
    ///
    /// # References
    ///
    /// * Soderlind (2003) Digital filters in adaptive time-stepping,
    ///   ACM Transactions on Mathematical Software, 29(1), 1-26.
    /// * Soderlind and Wang (2006) Adaptive time-stepping and computational stability,
    ///   Journal of Computational and Applied Mathematics, 185, 225-243.
    pub fn set_tg_control_alpha3(&mut self, value: f64) -> &mut Self {
        self.tg_control_alpha3 = value;
        self
    }

    /// Sets the tangent vector stepsize control parameters using a problem class from Soderlind (2003)
    ///
    /// # References
    ///
    /// * Soderlind (2003) Digital filters in adaptive time-stepping,
    pub fn set_tg_control_soderlind(&mut self, class: SoderlindClass) -> &mut Self {
        let (b1, b2, b3, a2, a3) = class.params();
        self.tg_control_beta1 = b1;
        self.tg_control_beta2 = b2;
        self.tg_control_beta3 = b3;
        self.tg_control_alpha2 = a2;
        self.tg_control_alpha3 = a3;
        self
    }

    /// Validates all data
    pub(crate) fn validate(&self) -> Result<(), StrError> {
        // automatic stepsize

        if self.m_failure < 0.001 {
            return Err("requirement: m_failure ≥ 0.001");
        }
        if self.h_ini <= CONFIG_H_MIN {
            return Err("requirement: h_ini > 1e-10");
        }
        if self.n_step_max < 1 {
            return Err("requirement: n_step_max ≥ 1");
        }
        if self.n_cont_failure_max < 1 {
            return Err("requirement: n_cont_failure_max ≥ 1");
        }
        if self.n_cont_rejection_max < 1 {
            return Err("requirement: n_cont_rejection_max ≥ 1");
        }

        // iterations

        if self.tol_abs_residual < CONFIG_TOL_MIN {
            return Err("requirement: tol_abs_residual ≥ 1e-12");
        }
        if self.tol_abs_delta < CONFIG_TOL_MIN {
            return Err("requirement: tol_abs_delta ≥ 1e-12");
        }
        if self.tol_rel_delta < CONFIG_TOL_MIN {
            return Err("requirement: tol_rel_delta ≥ 1e-12");
        }
        if self.delta_max_allowed <= 0.0 {
            return Err("requirement: allowed_delta_max > 0.0");
        }
        if self.n_iteration_max < 1 {
            return Err("requirement: allowed_iterations ≥ 1");
        }
        if self.n_cont_divergence_max < 1 {
            return Err("requirement: n_cont_divergence_max ≥ 1");
        }

        // stepsize control

        if self.nr_control_n_opt < 1 {
            return Err("requirement: nr_control_n_opt ≥ 1");
        }
        if self.nr_control_beta <= 0.0 {
            return Err("requirement: nr_control_beta > 0.0");
        }
        if self.tg_control_atol < CONFIG_TOL_MIN {
            return Err("requirement: tg_control_atol ≥ 1e-12");
        }
        if self.tg_control_rtol < CONFIG_TOL_MIN {
            return Err("requirement: tg_control_rtol ≥ 1e-12");
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
        let mut config = Config::new();
        config.set_method(Method::Arclength);
        let clone_config = config.clone();
        assert_eq!(format!("{:?}", config), format!("{:?}", clone_config));
    }

    #[test]
    fn config_validate_works() {
        let mut config = Config::new();
        config.set_method(Method::Arclength);

        // automatic stepsize

        config.h_ini = 0.0;
        assert_eq!(config.validate().err(), Some("requirement: h_ini > 1e-10"));
        config.h_ini = 1e-4;
        config.n_step_max = 0;
        assert_eq!(config.validate().err(), Some("requirement: n_step_max ≥ 1"));
        config.n_step_max = 10;
        config.n_cont_failure_max = 0;
        assert_eq!(config.validate().err(), Some("requirement: n_cont_failure_max ≥ 1"));
        config.n_cont_failure_max = 5;
        config.n_cont_rejection_max = 0;
        assert_eq!(config.validate().err(), Some("requirement: n_cont_rejection_max ≥ 1"));
        config.n_cont_rejection_max = 5;

        // iterations

        config.n_iteration_max = 0;
        assert_eq!(config.validate().err(), Some("requirement: allowed_iterations ≥ 1"));
        config.n_iteration_max = 10;
        config.tol_abs_residual = 0.0;
        assert_eq!(config.validate().err(), Some("requirement: tol_abs_residual ≥ 1e-12"));
        config.tol_abs_residual = 1e-10;
        config.tol_abs_delta = 0.0;
        assert_eq!(config.validate().err(), Some("requirement: tol_abs_delta ≥ 1e-12"));
        config.tol_abs_delta = 1e-10;
        config.tol_rel_delta = 0.0;
        assert_eq!(config.validate().err(), Some("requirement: tol_rel_delta ≥ 1e-12"));
        config.tol_rel_delta = 1e-10;
        config.delta_max_allowed = 0.0;
        assert_eq!(config.validate().err(), Some("requirement: allowed_delta_max > 0.0"));
        config.delta_max_allowed = 1e10;
        config.n_cont_divergence_max = 0;
        assert_eq!(config.validate().err(), Some("requirement: n_cont_divergence_max ≥ 1"));
        config.n_cont_divergence_max = 2;

        // stepsize control

        config.nr_control_n_opt = 0;
        assert_eq!(config.validate().err(), Some("requirement: nr_control_n_opt ≥ 1"));
        config.nr_control_n_opt = 3;
        config.nr_control_beta = 0.0;
        assert_eq!(config.validate().err(), Some("requirement: nr_control_beta > 0.0"));
        config.nr_control_beta = 0.5;
        config.tg_control_atol = 0.0;
        assert_eq!(config.validate().err(), Some("requirement: tg_control_atol ≥ 1e-12"));
        config.tg_control_atol = 1e-2;
        config.tg_control_rtol = 0.0;
        assert_eq!(config.validate().err(), Some("requirement: tg_control_rtol ≥ 1e-12"));
        config.tg_control_rtol = 1e-2;

        // all good
        assert_eq!(config.validate().is_err(), false);
    }
}
