use super::{Method, RdiffType, SoderlindClass};
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

    /// Show header and footer
    pub(crate) verbose_header_footer: bool,

    /// Show statistics
    pub(crate) verbose_stats: bool,

    /// Hide timings when displaying statistics
    pub(crate) hide_timings: bool,

    /// Indicates whether to record the iterations residuals or not (in statistics)
    pub(crate) record_iterations_residuals: bool,

    /// Enables the precise stop using a component of the u vector (if any)
    pub(crate) enable_precise_stop_u_comp: bool,

    // automatic stepsize -----------------------------------------------------------------
    //
    /// Coefficient to multiply the stepsize if the iterations are failing
    pub(crate) m_failure: f64,

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

    /// Disables the relative delta analysis
    pub(crate) disable_rel_delta_analysis: bool,

    /// Maximum allowed number of iterations
    pub(crate) n_iteration_max: usize,

    /// Maximum allowed number of continued divergence on ‖G,N‖∞
    pub(crate) n_cont_residual_divergence_max: usize,

    /// Maximum allowed number of continued divergence on ‖δu,δλ‖∞
    pub(crate) n_cont_delta_divergence_max: usize,

    // natural parameter continuation only ------------------------------------------------
    //
    /// Use the Euler predictor in the Natural continuation method
    pub(crate) euler_predictor: bool,

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
    /// 1. Valli AMP, Carey GF, Coutinho ALGA (2005) Control strategies for timestep selection in finite element
    ///    simulation of incompressible flows and coupled reaction–convection–diffusion processes,
    ///    International Journal for Numerical Methods in Fluids, 47:201-231, <https://doi.org/10.1002/fld.805>
    /// 2. Barros GF, Cortes AMA, Coutinho ALGA (2021) Finite element solution of nonlocal Cahn–Hilliard
    ///    equations with feedback control time step size adaptivity, International Journal for Numerical Methods
    ///    in Engineering, 122:5028-5052, <https://doi.org/10.1002/nme.6755>
    /// 3. Kubatschek T, Forster A (2025) Investigation of existing and new approaches to step size control in a
    ///    continuation framework, Computers & Structures, 313:107747, <https://doi.org/10.1016/j.compstruc.2025.107747>
    pub(crate) tg_control_pid_vcc: bool,

    /// Smallest absolute value of the relative difference in the tangent vector stepsize control
    pub(crate) tg_control_rdiff_min: f64,

    /// Rho multiplier for when the absolute value of the relative difference is "tiny" in the tangent vector stepsize control
    pub(crate) tg_control_rho_for_tiny_rdiff: f64,

    /// Optimal number of iterations for stepsize control using Newton-Raphson statistics
    pub(crate) nr_control_n_opt: usize,

    /// Beta coefficient used with the NR stepsize control
    pub(crate) nr_control_beta: f64,

    /// Method for the tangent vector stepsize control
    pub(crate) tg_control_rdiff_type: RdiffType,

    /// Tolerance for the tangent vector stepsize control
    pub(crate) tg_control_tol: f64,

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
            verbose_header_footer: true,
            verbose_stats: false,
            hide_timings: false,
            record_iterations_residuals: false,
            enable_precise_stop_u_comp: false,
            // automatic stepsize
            m_failure: 0.5,
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
            disable_rel_delta_analysis: false,
            n_iteration_max: 20,
            n_cont_residual_divergence_max: 3,
            n_cont_delta_divergence_max: 5,
            // pseudo-arclength
            euler_predictor: true,
            bordering: true,
            debug_predictor: false,
            // stepsize control
            nr_control_enabled: false,
            tg_control_enabled: true,
            tg_control_pid_vcc: true,
            tg_control_rdiff_min: 1e-6,
            tg_control_rho_for_tiny_rdiff: 1.2,
            nr_control_n_opt: 3,
            nr_control_beta: 0.5,
            tg_control_rdiff_type: RdiffType::Ave,
            tg_control_tol: 0.5,
            tg_control_beta1: b1,
            tg_control_beta2: b2,
            tg_control_beta3: b3,
            tg_control_alpha2: a2,
            tg_control_alpha3: a3,
        }
    }

    // basic options ----------------------------------------------------------------------

    /// Returns the continuation type
    ///
    /// If arclength, indicates whether bordering or full is being used.
    pub fn get_continuation(&self) -> String {
        match self.method {
            Method::Natural => "Natural".to_string(),
            Method::Arclength => {
                let kind = if self.bordering {
                    "(bord)".to_string()
                } else {
                    "(full)".to_string()
                };
                "Arclength".to_string() + &kind
            }
        }
    }

    /// Returns the method
    ///
    /// Returns the continuation method configured for the solver:
    /// [`Method::Natural`] or [`Method::Arclength`].
    pub fn get_method(&self) -> Method {
        self.method
    }

    /// Sets the method
    ///
    /// Default value: [Method::Natural]
    pub fn set_method(&mut self, method: Method) -> &mut Self {
        self.method = method;
        self
    }

    /// Returns the log file path
    pub fn get_log_file(&self) -> Option<&str> {
        self.log_file.as_deref()
    }

    /// Sets the log file path, to save the output instead of stdout
    ///
    /// Default value: None
    pub fn set_log_file(&mut self, full_path: &str) -> &mut Self {
        self.log_file = Some(full_path.to_string());
        self
    }

    /// Returns the verbose flag
    pub fn get_verbose(&self) -> bool {
        self.verbose
    }

    /// Sets the verbose flag
    ///
    /// Default value: false
    pub fn set_verbose_only(&mut self, flag: bool) -> &mut Self {
        self.verbose = flag;
        self
    }

    /// Returns the verbose_iterations flag
    pub fn get_verbose_iterations(&self) -> bool {
        self.verbose_iterations
    }

    /// Sets the verbose_iterations flag
    ///
    /// Default value: false
    pub fn set_verbose_iterations(&mut self, flag: bool) -> &mut Self {
        self.verbose_iterations = flag;
        self
    }

    /// Returns the verbose_stats flag
    pub fn get_verbose_stats(&self) -> bool {
        self.verbose_stats
    }

    /// Sets the verbose_stats flag
    ///
    /// Default value: false
    pub fn set_verbose_stats(&mut self, flag: bool) -> &mut Self {
        self.verbose_stats = flag;
        self
    }

    /// Sets verbose, verbose_iterations, and verbose_stats flags at once
    ///
    /// Default values:
    ///
    /// * `verbose`: false
    /// * `verbose_iterations`: false
    /// * `verbose_stats`: false
    pub fn set_verbose(&mut self, verbose: bool, verbose_iterations: bool, verbose_stats: bool) -> &mut Self {
        self.verbose = verbose;
        self.verbose_iterations = verbose_iterations;
        self.verbose_stats = verbose_stats;
        self
    }

    /// Returns the verbose_legend flag
    pub fn get_verbose_legend(&self) -> bool {
        self.verbose_legend
    }

    /// Shows the legend
    ///
    /// Default value: false
    pub fn set_verbose_legend(&mut self, flag: bool) -> &mut Self {
        self.verbose_legend = flag;
        self
    }

    /// Returns the verbose_header_footer flag
    pub fn get_verbose_header_footer(&self) -> bool {
        self.verbose_header_footer
    }

    /// Shows the header and footer
    ///
    /// Default value: true
    pub fn set_verbose_header_footer(&mut self, flag: bool) -> &mut Self {
        self.verbose_header_footer = flag;
        self
    }

    /// Returns the hide_timings flag
    pub fn get_hide_timings(&self) -> bool {
        self.hide_timings
    }

    /// Hides timings when displaying statistics
    ///
    /// Default value: false
    pub fn set_hide_timings(&mut self, flag: bool) -> &mut Self {
        self.hide_timings = flag;
        self
    }

    /// Returns the record_iterations_residuals flag
    pub fn get_record_iterations_residuals(&self) -> bool {
        self.record_iterations_residuals
    }

    /// Indicates whether to record the iterations residuals or not (in statistics)
    ///
    /// Default value: false
    pub fn set_record_iterations_residuals(&mut self, flag: bool) -> &mut Self {
        self.record_iterations_residuals = flag;
        self
    }

    /// Returns the enable_precise_stop_u_comp flag
    pub fn get_enable_precise_stop_u_comp(&self) -> bool {
        self.enable_precise_stop_u_comp
    }

    /// Enables the precise stop using a component of the u vector (if any)
    ///
    /// Default value: false
    pub fn set_enable_precise_stop_u_comp(&mut self, flag: bool) -> &mut Self {
        self.enable_precise_stop_u_comp = flag;
        self
    }

    // automatic stepsize -----------------------------------------------------------------

    /// Returns the coefficient to multiply the stepsize if the iterations are failing
    pub fn get_m_failure(&self) -> f64 {
        self.m_failure
    }

    /// Sets the coefficient to multiply the stepsize if the iterations are failing
    ///
    /// ```text
    /// m_failure ≥ 0.001   (recommended = 0.5)
    /// ```
    ///
    /// Default value: 0.5
    pub fn set_m_failure(&mut self, value: f64) -> &mut Self {
        self.m_failure = value;
        self
    }

    /// Returns the max number of steps
    pub fn get_n_step_max(&self) -> usize {
        self.n_step_max
    }

    /// Sets the max number of steps
    ///
    /// ```text
    /// n_step_max ≥ 1
    /// ```
    ///
    /// Default value: 100_000
    pub fn set_n_step_max(&mut self, value: usize) -> &mut Self {
        self.n_step_max = value;
        self
    }

    /// Returns the maximum allowed number of continued iteration failures
    pub fn get_n_cont_failure_max(&self) -> usize {
        self.n_cont_failure_max
    }

    /// Sets the maximum allowed number of continued iteration failures
    ///
    /// Default value: 5
    pub fn set_n_cont_failure_max(&mut self, value: usize) -> &mut Self {
        self.n_cont_failure_max = value;
        self
    }

    /// Returns the maximum allowed number of continued rejections
    pub fn get_n_cont_rejection_max(&self) -> usize {
        self.n_cont_rejection_max
    }

    /// Sets the maximum allowed number of continued rejections (due to large curvatures, etc.)
    ///
    /// Default value: 5
    pub fn set_n_cont_rejection_max(&mut self, value: usize) -> &mut Self {
        self.n_cont_rejection_max = value;
        self
    }

    // linear solver ----------------------------------------------------------------------

    /// Returns the linear solver kind
    pub fn get_genie(&self) -> Genie {
        self.genie
    }

    /// Sets the linear solver kind
    ///
    /// Default value: Genie::Umfpack
    pub fn set_genie(&mut self, genie: Genie) -> &mut Self {
        self.genie = genie;
        self
    }

    /// Returns configurations for sparse linear solver
    pub fn get_lin_sol_config(&self) -> &Option<LinSolParams> {
        &self.lin_sol_config
    }

    /// Sets configurations for sparse linear solver
    ///
    /// Default value: None
    pub fn set_lin_sol_config(&mut self, config: Option<LinSolParams>) -> &mut Self {
        self.lin_sol_config = config;
        self
    }

    /// Returns the option to write matrix after n step and stop
    pub fn get_write_matrix_after_nstep_and_stop(&self) -> &Option<usize> {
        &self.write_matrix_after_nstep_and_stop
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
    ///
    /// Default value: None
    pub fn set_write_matrix_after_nstep_and_stop(&mut self, value: usize) -> &mut Self {
        self.write_matrix_after_nstep_and_stop = Some(value);
        self
    }

    // iterations -------------------------------------------------------------------------

    /// Returns the absolute tolerance on ‖G,N‖∞
    pub fn get_tol_abs_residual(&self) -> f64 {
        self.tol_abs_residual
    }

    /// Sets the absolute tolerance on ‖G,N‖∞
    ///
    /// ```text
    /// value ≥ CONFIG_TOL_MIN
    /// ```
    ///
    /// See [CONFIG_TOL_MIN]
    ///
    /// Default value: 1e-10
    pub fn set_tol_abs_residual(&mut self, tol_abs: f64) -> &mut Self {
        self.tol_abs_residual = tol_abs;
        self
    }

    /// Returns the absolute tolerance on rms = Rel((δu,δλ))
    pub fn get_tol_abs_delta(&self) -> f64 {
        self.tol_abs_delta
    }

    /// Sets the absolute tolerance on rms = Rel((δu,δλ))
    ///
    /// ```text
    /// value ≥ CONFIG_TOL_MIN
    /// ```
    ///
    /// See [CONFIG_TOL_MIN]
    ///
    /// Default value: 1e-10
    pub fn set_tol_abs_delta(&mut self, tol_abs: f64) -> &mut Self {
        self.tol_abs_delta = tol_abs;
        self
    }

    /// Returns the relative tolerance on rms = Rel((δu,δλ))
    pub fn get_tol_rel_delta(&self) -> f64 {
        self.tol_rel_delta
    }

    /// Sets the relative tolerance on rms = Rel((δu,δλ))
    ///
    /// ```text
    /// value ≥ CONFIG_TOL_MIN
    /// ```
    ///
    /// See [CONFIG_TOL_MIN]
    ///
    /// Default value: 1e-7
    pub fn set_tol_rel_delta(&mut self, tol_rel: f64) -> &mut Self {
        self.tol_rel_delta = tol_rel;
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

    /// Returns the allowed ‖δu,δλ‖∞ max
    pub fn get_delta_max_allowed(&self) -> f64 {
        self.delta_max_allowed
    }

    /// Sets the allowed ‖δu,δλ‖∞ max
    ///
    /// Default value: 1e8
    pub fn set_delta_max_allowed(&mut self, value: f64) -> &mut Self {
        self.delta_max_allowed = value;
        self
    }

    /// Returns the disable_rel_delta_analysis flag
    pub fn get_disable_rel_delta_analysis(&self) -> bool {
        self.disable_rel_delta_analysis
    }

    /// Disables the relative delta analysis
    ///
    /// Default value: false
    pub fn set_disable_rel_delta_analysis(&mut self, flag: bool) -> &mut Self {
        self.disable_rel_delta_analysis = flag;
        self
    }

    /// Returns the allowed number of iterations
    pub fn get_n_iteration_max(&self) -> usize {
        self.n_iteration_max
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

    /// Returns the maximum allowed number of continued divergence on ‖G,N‖∞
    pub fn get_n_cont_residual_divergence_max(&self) -> usize {
        self.n_cont_residual_divergence_max
    }

    /// Sets the maximum allowed number of continued divergence on ‖G,N‖∞
    ///
    /// Default value: 3
    pub fn set_n_cont_residual_divergence_max(&mut self, value: usize) -> &mut Self {
        self.n_cont_residual_divergence_max = value;
        self
    }

    /// Returns the maximum allowed number of continued divergence on ‖δu,δλ‖∞
    pub fn get_n_cont_delta_divergence_max(&self) -> usize {
        self.n_cont_delta_divergence_max
    }

    /// Sets the maximum allowed number of continued divergence on ‖δu,δλ‖∞
    ///
    /// Default value: 5
    pub fn set_n_cont_delta_divergence_max(&mut self, value: usize) -> &mut Self {
        self.n_cont_delta_divergence_max = value;
        self
    }

    // natural parameter continuation only ------------------------------------------------

    /// Returns the euler_predictor flag
    pub fn get_euler_predictor(&self) -> bool {
        self.euler_predictor
    }

    /// Uses the Euler predictor in the Natural continuation method
    ///
    /// Default value: true
    pub fn set_euler_predictor(&mut self, flag: bool) -> &mut Self {
        self.euler_predictor = flag;
        self
    }

    // pseudo-arclength -------------------------------------------------------------------

    /// Returns the bordering flag
    pub fn get_bordering(&self) -> bool {
        self.bordering
    }

    /// Sets the bordering flag
    ///
    /// Use the bordering algorithm throughout the entire simulation
    ///
    /// Default value: true
    pub fn set_bordering(&mut self, flag: bool) -> &mut Self {
        self.bordering = flag;
        self
    }

    /// Returns the debug_predictor flag
    pub fn get_debug_predictor(&self) -> bool {
        self.debug_predictor
    }

    /// Records the predictor values for debugging
    ///
    /// Default value: false
    pub fn set_debug_predictor(&mut self, flag: bool) -> &mut Self {
        self.debug_predictor = flag;
        self
    }

    /// Returns the Newton-Raphson stepsize control flag
    pub fn get_nr_control_enabled(&self) -> bool {
        self.nr_control_enabled
    }

    /// Sets the Newton-Raphson stepsize control flag
    ///
    /// Enables or disables the Newton-Raphson stepsize control
    ///
    /// Default value: false
    pub fn set_nr_control_enabled(&mut self, flag: bool) -> &mut Self {
        self.nr_control_enabled = flag;
        self
    }

    /// Returns the tangent vector stepsize control flag
    pub fn get_tg_control_enabled(&self) -> bool {
        self.tg_control_enabled
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

    /// Returns the PID VCC flag for tangent vector stepsize control
    pub fn get_tg_control_pid_vcc(&self) -> bool {
        self.tg_control_pid_vcc
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
    /// 1. Valli AMP, Carey GF, Coutinho ALGA (2005) Control strategies for timestep selection in finite element
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

    /// Returns the "tiny" absolute value of the relative difference in the tangent vector stepsize control
    pub fn get_tg_control_rdiff_min(&self) -> f64 {
        self.tg_control_rdiff_min
    }

    /// Sets the "tiny" absolute value of the relative difference in the tangent vector stepsize control
    ///
    /// Default value: 1e-6
    pub fn set_tg_control_rdiff_min(&mut self, value: f64) -> &mut Self {
        self.tg_control_rdiff_min = value;
        self
    }

    /// Returns the rho multiplier for when the absolute value of the relative difference is "tiny" in the tangent vector stepsize control
    pub fn get_tg_control_rho_for_tiny_rdiff(&self) -> f64 {
        self.tg_control_rho_for_tiny_rdiff
    }

    /// Sets the rho multiplier for when the absolute value of the relative difference is "tiny" in the tangent vector stepsize control
    ///
    /// Default value: 1.2
    pub fn set_tg_control_rho_for_tiny_rdiff(&mut self, value: f64) -> &mut Self {
        self.tg_control_rho_for_tiny_rdiff = value;
        self
    }

    /// Returns the optimal number of iterations for stepsize control using Newton-Raphson statistics
    pub fn get_nr_control_n_opt(&self) -> usize {
        self.nr_control_n_opt
    }

    /// Sets the optimal number of iterations for stepsize control using Newton-Raphson statistics
    ///
    /// Default value: 3
    pub fn set_nr_control_n_opt(&mut self, value: usize) -> &mut Self {
        self.nr_control_n_opt = value;
        self
    }

    /// Returns the beta coefficient used with the NR stepsize control
    pub fn get_nr_control_beta(&self) -> f64 {
        self.nr_control_beta
    }

    /// Sets the beta coefficient used with the NR stepsize control
    ///
    /// Default value: 0.5
    pub fn set_nr_control_beta(&mut self, value: f64) -> &mut Self {
        self.nr_control_beta = value;
        self
    }

    /// Returns the method for the tangent vector stepsize control
    pub fn get_tg_control_rdiff_type(&self) -> RdiffType {
        self.tg_control_rdiff_type
    }

    /// Sets the method for the tangent vector stepsize control
    ///
    /// Default value: [RdiffType::Ave]
    pub fn set_tg_control_rdiff_type(&mut self, value: RdiffType) -> &mut Self {
        self.tg_control_rdiff_type = value;
        self
    }

    /// Returns the tolerance for the tangent vector stepsize control
    pub fn get_tg_control_tol(&self) -> f64 {
        self.tg_control_tol
    }

    /// Sets the tolerance for the tangent vector stepsize control
    ///
    /// Default value: 0.5
    pub fn set_tg_control_tol(&mut self, value: f64) -> &mut Self {
        self.tg_control_tol = value;
        self
    }

    /// Returns the first exponent for the tangent vector stepsize control
    pub fn get_tg_control_beta1(&self) -> f64 {
        self.tg_control_beta1
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
    ///
    /// Default value: 1/6 (from SoderlindClass::H211PI)
    pub fn set_tg_control_beta1(&mut self, value: f64) -> &mut Self {
        self.tg_control_beta1 = value;
        self
    }

    /// Returns the second exponent for the tangent vector stepsize control
    pub fn get_tg_control_beta2(&self) -> f64 {
        self.tg_control_beta2
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
    ///
    /// Default value: 1/6 (from SoderlindClass::H211PI)
    pub fn set_tg_control_beta2(&mut self, value: f64) -> &mut Self {
        self.tg_control_beta2 = value;
        self
    }

    /// Returns the third exponent for the tangent vector stepsize control
    pub fn get_tg_control_beta3(&self) -> f64 {
        self.tg_control_beta3
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
    ///
    /// Default value: 0.0 (from SoderlindClass::H211PI)
    pub fn set_tg_control_beta3(&mut self, value: f64) -> &mut Self {
        self.tg_control_beta3 = value;
        self
    }

    /// Returns the fourth exponent for the tangent vector stepsize control
    pub fn get_tg_control_alpha2(&self) -> f64 {
        self.tg_control_alpha2
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
    ///
    /// Default value: 0.0 (from SoderlindClass::H211PI)
    pub fn set_tg_control_alpha2(&mut self, value: f64) -> &mut Self {
        self.tg_control_alpha2 = value;
        self
    }

    /// Returns the fifth exponent for the tangent vector stepsize control
    pub fn get_tg_control_alpha3(&self) -> f64 {
        self.tg_control_alpha3
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
    ///
    /// Default value: 0.0 (from SoderlindClass::H211PI)
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
        if self.n_cont_residual_divergence_max < 1 {
            return Err("requirement: n_cont_residual_div_max ≥ 1");
        }
        if self.n_cont_delta_divergence_max < 1 {
            return Err("requirement: n_cont_delta_div_max ≥ 1");
        }

        // stepsize control

        if self.nr_control_n_opt < 1 {
            return Err("requirement: nr_control_n_opt ≥ 1");
        }
        if self.nr_control_beta <= 0.0 {
            return Err("requirement: nr_control_beta > 0.0");
        }
        if self.tg_control_tol < CONFIG_TOL_MIN {
            return Err("requirement: tg_control_tol ≥ 1e-12");
        }

        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Config;
    use crate::{Method, RdiffType, SoderlindClass};
    use russell_sparse::{Genie, LinSolParams};

    #[test]
    fn derive_methods_work() {
        let mut config = Config::new();
        config.set_method(Method::Arclength);
        let clone_config = config.clone();
        assert_eq!(format!("{:?}", config), format!("{:?}", clone_config));
    }

    #[test]
    fn config_defaults_are_correct() {
        let (b1, b2, b3, a2, a3) = SoderlindClass::H211PI.params();
        let config = Config::new();

        // basic options
        assert_eq!(config.get_method(), Method::Natural);
        assert_eq!(config.get_log_file(), None);
        assert_eq!(config.get_verbose(), false);
        assert_eq!(config.get_verbose_iterations(), false);
        assert_eq!(config.get_verbose_legend(), false);
        assert_eq!(config.get_verbose_header_footer(), true);
        assert_eq!(config.get_verbose_stats(), false);
        assert_eq!(config.get_hide_timings(), false);
        assert_eq!(config.get_record_iterations_residuals(), false);
        assert_eq!(config.get_enable_precise_stop_u_comp(), false);

        // automatic stepsize
        assert_eq!(config.get_m_failure(), 0.5);
        assert_eq!(config.get_n_step_max(), 100_000);
        assert_eq!(config.get_n_cont_failure_max(), 5);
        assert_eq!(config.get_n_cont_rejection_max(), 5);

        // linear solver
        assert_eq!(config.get_genie(), Genie::Umfpack);
        assert_eq!(config.get_lin_sol_config().is_none(), true);
        assert_eq!(config.get_write_matrix_after_nstep_and_stop().is_none(), true);

        // iterations
        assert_eq!(config.get_tol_abs_residual(), 1e-10);
        assert_eq!(config.get_tol_abs_delta(), 1e-10);
        assert_eq!(config.get_tol_rel_delta(), 1e-7);
        assert_eq!(config.get_delta_max_allowed(), 1e8);
        assert_eq!(config.get_disable_rel_delta_analysis(), false);
        assert_eq!(config.get_n_iteration_max(), 20);
        assert_eq!(config.get_n_cont_residual_divergence_max(), 3);
        assert_eq!(config.get_n_cont_delta_divergence_max(), 5);

        // natural continuation
        assert_eq!(config.get_euler_predictor(), true);

        // pseudo-arclength
        assert_eq!(config.get_bordering(), true);
        assert_eq!(config.get_debug_predictor(), false);

        // stepsize control
        assert_eq!(config.get_nr_control_enabled(), false);
        assert_eq!(config.get_tg_control_enabled(), true);
        assert_eq!(config.get_tg_control_pid_vcc(), true);
        assert_eq!(config.get_tg_control_rdiff_min(), 1e-6);
        assert_eq!(config.get_tg_control_rho_for_tiny_rdiff(), 1.2);
        assert_eq!(config.get_nr_control_n_opt(), 3);
        assert_eq!(config.get_nr_control_beta(), 0.5);
        assert_eq!(config.get_tg_control_rdiff_type(), RdiffType::Ave);
        assert_eq!(config.get_tg_control_tol(), 0.5);
        assert_eq!(config.get_tg_control_beta1(), b1);
        assert_eq!(config.get_tg_control_beta2(), b2);
        assert_eq!(config.get_tg_control_beta3(), b3);
        assert_eq!(config.get_tg_control_alpha2(), a2);
        assert_eq!(config.get_tg_control_alpha3(), a3);
    }

    #[test]
    fn config_getters_and_setters_work() {
        let mut config = Config::new();

        // basic options
        config.set_method(Method::Arclength);
        assert_eq!(config.get_method(), Method::Arclength);

        config.set_log_file("/tmp/test.log");
        assert_eq!(config.get_log_file(), Some("/tmp/test.log"));

        config.set_verbose_only(false);
        assert_eq!(config.get_verbose(), false);
        config.set_verbose_only(true);
        assert_eq!(config.get_verbose(), true);

        config.set_verbose_iterations(true);
        assert_eq!(config.get_verbose_iterations(), true);
        config.set_verbose_iterations(false);
        assert_eq!(config.get_verbose_iterations(), false);

        config.set_verbose_legend(true);
        assert_eq!(config.get_verbose_legend(), true);
        config.set_verbose_legend(false);
        assert_eq!(config.get_verbose_legend(), false);

        config.set_verbose_header_footer(false);
        assert_eq!(config.get_verbose_header_footer(), false);
        config.set_verbose_header_footer(true);
        assert_eq!(config.get_verbose_header_footer(), true);

        config.set_verbose_stats(true);
        assert_eq!(config.get_verbose_stats(), true);
        config.set_verbose_stats(false);
        assert_eq!(config.get_verbose_stats(), false);

        config.set_hide_timings(true);
        assert_eq!(config.get_hide_timings(), true);
        config.set_hide_timings(false);
        assert_eq!(config.get_hide_timings(), false);

        config.set_record_iterations_residuals(true);
        assert_eq!(config.get_record_iterations_residuals(), true);
        config.set_record_iterations_residuals(false);
        assert_eq!(config.get_record_iterations_residuals(), false);

        config.set_enable_precise_stop_u_comp(true);
        assert_eq!(config.get_enable_precise_stop_u_comp(), true);
        config.set_enable_precise_stop_u_comp(false);
        assert_eq!(config.get_enable_precise_stop_u_comp(), false);

        // automatic stepsize
        config.set_m_failure(0.25);
        assert_eq!(config.get_m_failure(), 0.25);

        config.set_n_step_max(500);
        assert_eq!(config.get_n_step_max(), 500);

        config.set_n_cont_failure_max(10);
        assert_eq!(config.get_n_cont_failure_max(), 10);

        config.set_n_cont_rejection_max(10);
        assert_eq!(config.get_n_cont_rejection_max(), 10);

        // linear solver
        config.set_genie(Genie::Mumps);
        assert_eq!(config.get_genie(), Genie::Mumps);

        let params = LinSolParams::new();
        config.set_lin_sol_config(Some(params));
        assert!(config.get_lin_sol_config().is_some());
        config.set_lin_sol_config(None);
        assert!(config.get_lin_sol_config().is_none());

        config.set_write_matrix_after_nstep_and_stop(42);
        assert_eq!(config.get_write_matrix_after_nstep_and_stop(), &Some(42));

        // iterations
        config.set_tol_abs_residual(1e-8);
        assert_eq!(config.get_tol_abs_residual(), 1e-8);

        config.set_tol_abs_delta(1e-8);
        assert_eq!(config.get_tol_abs_delta(), 1e-8);

        config.set_tol_rel_delta(1e-5);
        assert_eq!(config.get_tol_rel_delta(), 1e-5);

        config.set_delta_max_allowed(1e6);
        assert_eq!(config.get_delta_max_allowed(), 1e6);

        config.set_disable_rel_delta_analysis(true);
        assert_eq!(config.get_disable_rel_delta_analysis(), true);
        config.set_disable_rel_delta_analysis(false);
        assert_eq!(config.get_disable_rel_delta_analysis(), false);

        config.set_n_iteration_max(50);
        assert_eq!(config.get_n_iteration_max(), 50);

        config.set_n_cont_residual_divergence_max(5);
        assert_eq!(config.get_n_cont_residual_divergence_max(), 5);

        config.set_n_cont_delta_divergence_max(10);
        assert_eq!(config.get_n_cont_delta_divergence_max(), 10);

        // natural continuation
        config.set_euler_predictor(false);
        assert_eq!(config.get_euler_predictor(), false);
        config.set_euler_predictor(true);
        assert_eq!(config.get_euler_predictor(), true);

        // pseudo-arclength
        config.set_bordering(false);
        assert_eq!(config.get_bordering(), false);
        config.set_bordering(true);
        assert_eq!(config.get_bordering(), true);

        config.set_debug_predictor(true);
        assert_eq!(config.get_debug_predictor(), true);
        config.set_debug_predictor(false);
        assert_eq!(config.get_debug_predictor(), false);

        // stepsize control
        config.set_nr_control_enabled(true);
        assert_eq!(config.get_nr_control_enabled(), true);
        config.set_nr_control_enabled(false);
        assert_eq!(config.get_nr_control_enabled(), false);

        config.set_tg_control_enabled(false);
        assert_eq!(config.get_tg_control_enabled(), false);
        config.set_tg_control_enabled(true);
        assert_eq!(config.get_tg_control_enabled(), true);

        config.set_tg_control_pid_vcc(false);
        assert_eq!(config.get_tg_control_pid_vcc(), false);
        config.set_tg_control_pid_vcc(true);
        assert_eq!(config.get_tg_control_pid_vcc(), true);

        config.set_tg_control_rdiff_min(1e-4);
        assert_eq!(config.get_tg_control_rdiff_min(), 1e-4);

        config.set_tg_control_rho_for_tiny_rdiff(1.5);
        assert_eq!(config.get_tg_control_rho_for_tiny_rdiff(), 1.5);

        config.set_nr_control_n_opt(10);
        assert_eq!(config.get_nr_control_n_opt(), 10);

        config.set_nr_control_beta(0.75);
        assert_eq!(config.get_nr_control_beta(), 0.75);

        config.set_tg_control_rdiff_type(RdiffType::Max);
        assert_eq!(config.get_tg_control_rdiff_type(), RdiffType::Max);

        config.set_tg_control_tol(0.25);
        assert_eq!(config.get_tg_control_tol(), 0.25);

        config.set_tg_control_beta1(0.1);
        assert_eq!(config.get_tg_control_beta1(), 0.1);

        config.set_tg_control_beta2(0.2);
        assert_eq!(config.get_tg_control_beta2(), 0.2);

        config.set_tg_control_beta3(0.3);
        assert_eq!(config.get_tg_control_beta3(), 0.3);

        config.set_tg_control_alpha2(0.4);
        assert_eq!(config.get_tg_control_alpha2(), 0.4);

        config.set_tg_control_alpha3(0.5);
        assert_eq!(config.get_tg_control_alpha3(), 0.5);
    }

    #[test]
    fn compound_setters_work() {
        let mut config = Config::new();

        // set_verbose sets all three verbose fields
        config.set_verbose(true, true, true);
        assert_eq!(config.get_verbose(), true);
        assert_eq!(config.get_verbose_iterations(), true);
        assert_eq!(config.get_verbose_stats(), true);

        config.set_verbose(false, false, false);
        assert_eq!(config.get_verbose(), false);
        assert_eq!(config.get_verbose_iterations(), false);
        assert_eq!(config.get_verbose_stats(), false);

        // set_tol_delta sets both delta tolerance fields
        config.set_tol_delta(1e-8, 1e-6);
        assert_eq!(config.get_tol_abs_delta(), 1e-8);
        assert_eq!(config.get_tol_rel_delta(), 1e-6);
    }

    #[test]
    fn setter_builder_pattern_works() {
        let mut config = Config::new();
        config
            .set_method(Method::Arclength)
            .set_m_failure(0.25)
            .set_n_step_max(2000)
            .set_tol_abs_residual(1e-11)
            .set_tg_control_beta1(0.5)
            .set_tg_control_tol(0.3);

        assert_eq!(config.get_method(), Method::Arclength);
        assert_eq!(config.get_m_failure(), 0.25);
        assert_eq!(config.get_n_step_max(), 2000);
        assert_eq!(config.get_tol_abs_residual(), 1e-11);
        assert_eq!(config.get_tg_control_beta1(), 0.5);
        assert_eq!(config.get_tg_control_tol(), 0.3);
    }

    #[test]
    fn set_tg_control_soderlind_works() {
        let mut config = Config::new();

        config.set_tg_control_soderlind(SoderlindClass::Ho312);
        let (b1, b2, b3, a2, a3) = SoderlindClass::Ho312.params();
        assert_eq!(config.get_tg_control_beta1(), b1);
        assert_eq!(config.get_tg_control_beta2(), b2);
        assert_eq!(config.get_tg_control_beta3(), b3);
        assert_eq!(config.get_tg_control_alpha2(), a2);
        assert_eq!(config.get_tg_control_alpha3(), a3);
    }

    #[test]
    fn get_continuation_works() {
        let mut config = Config::new();

        config.set_method(Method::Natural);
        assert_eq!(config.get_continuation(), "Natural");

        config.set_method(Method::Arclength);
        config.set_bordering(true);
        assert_eq!(config.get_continuation(), "Arclength(bord)");

        config.set_bordering(false);
        assert_eq!(config.get_continuation(), "Arclength(full)");
    }

    #[test]
    fn config_validate_works() {
        let mut config = Config::new();
        config.set_method(Method::Arclength);

        // automatic stepsize

        config.m_failure = 0.0005;
        assert_eq!(config.validate().err(), Some("requirement: m_failure ≥ 0.001"));
        config.m_failure = 0.5;
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
        config.n_cont_residual_divergence_max = 0;
        assert_eq!(
            config.validate().err(),
            Some("requirement: n_cont_residual_div_max ≥ 1")
        );
        config.n_cont_residual_divergence_max = 2;
        config.n_cont_delta_divergence_max = 0;
        assert_eq!(config.validate().err(), Some("requirement: n_cont_delta_div_max ≥ 1"));
        config.n_cont_delta_divergence_max = 2;

        // stepsize control

        config.nr_control_n_opt = 0;
        assert_eq!(config.validate().err(), Some("requirement: nr_control_n_opt ≥ 1"));
        config.nr_control_n_opt = 3;
        config.nr_control_beta = 0.0;
        assert_eq!(config.validate().err(), Some("requirement: nr_control_beta > 0.0"));
        config.nr_control_beta = 0.5;
        config.tg_control_tol = 0.0;
        assert_eq!(config.validate().err(), Some("requirement: tg_control_tol ≥ 1e-12"));
        config.tg_control_tol = 0.1;

        // all good
        assert_eq!(config.validate().is_err(), false);
    }
}
