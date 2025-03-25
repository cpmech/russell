use super::NlMethod;
use crate::StrError;
use russell_sparse::{Genie, LinSolParams};

/// Holds parameters to control the variable stepsize algorithm
#[derive(Clone, Copy, Debug)]
pub struct NlParams {
    // basic parameters -------------------------------------------------------------------
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

    // linear solver ----------------------------------------------------------------------
    //
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

    // iterations -------------------------------------------------------------------------
    //
    /// Tolerance on max(‖G‖∞,|H|)
    pub tol_gh: f64,

    /// Tolerance on max(‖δu‖∞,|δλ|)
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

impl NlParams {
    /// Allocates a new instance
    pub fn new(method: NlMethod) -> Self {
        NlParams {
            // basic parameters
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
            n_step_max: 100_000,
            rel_error_prev_min: 1e-4,
            // linear solver
            genie: Genie::Umfpack,
            lin_sol_params: None,
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

    /// Validates all parameters
    pub(crate) fn validate(&self) -> Result<(), StrError> {
        // substepping

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

        // iterations

        if self.n_iteration_max < 1 {
            return Err("parameter must satisfy: n_iteration_max ≥ 1");
        }
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::NlParams;
    use crate::NlMethod;

    #[test]
    fn derive_methods_work() {
        let params = NlParams::new(NlMethod::Arclength);
        let clone_params = params.clone();
        assert_eq!(format!("{:?}", params), format!("{:?}", clone_params));
    }

    #[test]
    fn params_validate_works() {
        let mut params = NlParams::new(NlMethod::Arclength);

        // substepping

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

        // iterations

        params.n_iteration_max = 0;
        assert_eq!(
            params.validate().err(),
            Some("parameter must satisfy: n_iteration_max ≥ 1")
        );
        params.n_iteration_max = 10;

        // all good
        assert_eq!(params.validate().is_err(), false);
    }
}
