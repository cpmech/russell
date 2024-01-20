#![allow(unused, non_snake_case)]

use crate::{DenseOutF, Method, StepOutF};
use russell_sparse::{Genie, LinSolParams};

/// Defines the configuration parameters for the ODE solver
#[derive(Clone, Debug)]
pub struct Configuration {
    /// Method
    pub(crate) method: Method,

    /// linear solver kind
    pub(crate) genie: Genie,

    /// configurations for sparse linear solver
    pub(crate) lin_sol_params: LinSolParams,

    /// minimum H allowed
    pub Hmin: f64,

    /// initial H
    pub IniH: f64,

    /// max num iterations (allowed)
    pub NmaxIt: usize,

    /// max num substeps
    pub NmaxSS: usize,

    /// min step multiplier
    pub Mmin: f64,

    /// max step multiplier
    pub Mmax: f64,

    /// step multiplier factor
    pub Mfac: f64,

    /// coefficient to multiply stepsize if first step is rejected [0 ⇒ use dx_new]
    pub MfirstRej: f64,

    /// use Gustafsson's predictive controller
    pub PredCtrl: bool,

    /// smallest number satisfying 1.0 + ϵ > 1.0
    pub Eps: f64,

    /// max theta to decide whether the Jacobian should be recomputed or not
    pub ThetaMax: f64,

    /// c1 of HW-VII p124 => min ratio to retain previous h
    pub C1h: f64,

    /// c2 of HW-VII p124 => max ratio to retain previous h
    pub C2h: f64,

    /// strategy to select local error computation method
    pub LerrStrat: usize,

    /// allow use of go channels (threaded); e.g. to solve R and C systems concurrently
    pub GoChan: bool,

    /// use constant tangent (Jacobian) in BwEuler
    pub CteTg: bool,

    /// use RMS norm instead of Euclidean in BwEuler
    pub UseRmsNorm: bool,

    /// show messages, e.g. during iterations
    pub Verbose: bool,

    /// always start iterations with zero trial values (instead of collocation interpolation)
    pub ZeroTrial: bool,

    /// Lund stabilization coefficient β
    pub StabBeta: f64,

    /// number of steps to check stiff situation. 0 ⇒ no check. [default = 1]
    pub StiffNstp: usize,

    /// maximum value of ρs [default = 0.5]
    pub StiffRsMax: f64,

    /// number of "yes" stiff steps allowed [default = 15]
    pub StiffNyes: usize,

    /// number of "not" stiff steps to disregard stiffness [default = 6]
    pub StiffNnot: usize,

    /// function to process step output (of accepted steps)
    pub stepF: Option<StepOutF>,

    /// function to process dense output
    pub denseF: Option<DenseOutF>,

    /// step size for dense output
    pub denseDx: f64,

    /// perform output of (variable) steps
    pub stepOut: bool,

    /// perform dense output is active
    pub denseOut: bool,

    /// number of dense steps
    pub denseNstp: usize,

    /// factor to multiply stabilization coefficient β
    pub stabBetaM: f64,

    /// absolute tolerance
    pub atol: f64,

    /// relative tolerance
    pub rtol: f64,

    /// Newton's iterations tolerance
    pub fnewt: f64,

    /// min value of rerrPrev
    pub rerrPrevMin: f64,

    /// use fixed steps
    pub fixed: bool,

    /// value of fixed stepsize
    pub fixedH: f64,

    /// number of fixed steps
    pub fixedNsteps: usize,
}

impl Configuration {
    /// Allocates a new instance with default values
    pub fn new(method: Method, lin_sol: Option<Genie>, lin_sol_params: Option<LinSolParams>) -> Self {
        let genie = match lin_sol {
            Some(g) => g,
            None => Genie::Umfpack,
        };
        let ls_params = match lin_sol_params {
            Some(p) => p,
            None => LinSolParams::new(),
        };
        Configuration {
            method,
            genie,
            lin_sol_params: ls_params,
            Hmin: 0.0,
            IniH: 0.0,
            NmaxIt: 0,
            NmaxSS: 0,
            Mmin: 0.0,
            Mmax: 0.0,
            Mfac: 0.0,
            MfirstRej: 0.0,
            PredCtrl: false,
            Eps: 0.0,
            ThetaMax: 0.0,
            C1h: 0.0,
            C2h: 0.0,
            LerrStrat: 0,
            GoChan: false,
            CteTg: false,
            UseRmsNorm: false,
            Verbose: false,
            ZeroTrial: false,
            StabBeta: 0.0,
            StiffNstp: 0,
            StiffRsMax: 0.0,
            StiffNyes: 0,
            StiffNnot: 0,
            stepF: None,
            denseF: None,
            denseDx: 0.0,
            stepOut: false,
            denseOut: false,
            denseNstp: 0,
            stabBetaM: 0.0,
            atol: 0.0,
            rtol: 0.0,
            fnewt: 0.0,
            rerrPrevMin: 0.0,
            fixed: false,
            fixedH: 0.0,
            fixedNsteps: 0,
        }
    }
}
