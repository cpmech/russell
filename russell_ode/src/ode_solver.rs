use crate::{DenseOutF, OdeMethod, StepOutF, StrError};
use russell_sparse::{Genie, LinSolParams};

/// Defines the configuration parameters for the ODE solver
#[derive(Clone, Debug)]
pub struct OdeSolParams {
    /// minimum H allowed
    Hmin: f64,

    /// initial H
    IniH: f64,

    /// max num iterations (allowed)
    NmaxIt: usize,

    /// max num substeps
    NmaxSS: usize,

    /// min step multiplier
    Mmin: f64,

    /// max step multiplier
    Mmax: f64,

    /// step multiplier factor
    Mfac: f64,

    /// coefficient to multiply stepsize if first step is rejected [0 ⇒ use dx_new]
    MfirstRej: f64,

    /// use Gustafsson's predictive controller
    PredCtrl: bool,

    /// smallest number satisfying 1.0 + ϵ > 1.0
    Eps: f64,

    /// max theta to decide whether the Jacobian should be recomputed or not
    ThetaMax: f64,

    /// c1 of HW-VII p124 => min ratio to retain previous h
    C1h: f64,

    /// c2 of HW-VII p124 => max ratio to retain previous h
    C2h: f64,

    /// strategy to select local error computation method
    LerrStrat: usize,

    /// allow use of go channels (threaded); e.g. to solve R and C systems concurrently
    GoChan: bool,

    /// use constant tangent (Jacobian) in BwEuler
    CteTg: bool,

    /// use RMS norm instead of Euclidean in BwEuler
    UseRmsNorm: bool,

    /// show messages, e.g. during iterations
    Verbose: bool,

    /// always start iterations with zero trial values (instead of collocation interpolation)
    ZeroTrial: bool,

    /// Lund stabilization coefficient β
    StabBeta: f64,

    /// number of steps to check stiff situation. 0 ⇒ no check. [default = 1]
    StiffNstp: usize,

    /// maximum value of ρs [default = 0.5]
    StiffRsMax: f64,

    /// number of "yes" stiff steps allowed [default = 15]
    StiffNyes: usize,

    /// number of "not" stiff steps to disregard stiffness [default = 6]
    StiffNnot: usize,

    /// linear solver kind
    lsKind: Genie,

    /// configurations for sparse linear solver
    LinSolConfig: LinSolParams,

    /// function to process step output (of accepted steps)
    stepF: Option<StepOutF>,

    /// function to process dense output
    denseF: Option<DenseOutF>,

    /// step size for dense output
    denseDx: f64,

    /// perform output of (variable) steps
    stepOut: bool,

    /// perform dense output is active
    denseOut: bool,

    /// number of dense steps
    denseNstp: usize,

    /// the ODE method
    method: OdeMethod,

    /// factor to multiply stabilization coefficient β
    stabBetaM: f64,

    /// absolute tolerance
    atol: f64,

    /// relative tolerance
    rtol: f64,

    /// Newton's iterations tolerance
    fnewt: f64,

    /// min value of rerrPrev
    rerrPrevMin: f64,

    /// use fixed steps
    fixed: bool,

    /// value of fixed stepsize
    fixedH: f64,

    /// number of fixed steps
    fixedNsteps: usize,
}

impl OdeSolParams {
    /// Allocates a new instance with default values
    pub fn new(ode_method: OdeMethod, lin_sol: Option<Genie>) -> Self {
        let genie = match lin_sol {
            Some(g) => g,
            None => Genie::Umfpack,
        };
        OdeSolParams {
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
            lsKind: genie,
            LinSolConfig: LinSolParams::new(),
            stepF: None,
            denseF: None,
            denseDx: 0.0,
            stepOut: false,
            denseOut: false,
            denseNstp: 0,
            method: ode_method,
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

/// Defines a unified interface for ODE solvers
pub trait OdeSolTrait {
    // Todo
}

/// Unifies the access to ODE solvers
pub struct OdeSolver<'a> {
    /// Holds the actual implementation
    pub actual: Box<dyn OdeSolTrait + 'a>,
}

impl<'a> OdeSolver<'a> {
    /// Allocates a new instance
    pub fn new(params: OdeSolParams) { //-> Result<Self, StrError> {
                                       // let actual: Box<dyn LinSolTrait> = match genie {
                                       //     Genie::Mumps => Box::new(SolverMUMPS::new()?),
                                       //     Genie::Umfpack => Box::new(SolverUMFPACK::new()?),
                                       //     Genie::IntelDss => Box::new(SolverIntelDSS::new()?),
                                       // };
                                       // Ok(OdeSolver { actual })
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn todo_works() {}
}
