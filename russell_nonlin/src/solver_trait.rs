use super::{IniDir, Status, Stop, Workspace};
use crate::StrError;
use russell_lab::Vector;

/// Defines the numerical solver
pub(crate) trait SolverTrait<A>: Send {
    /// Performs initialization
    ///
    /// 1. Calculates the initial stepsize
    /// 2. Determines the first tangent vector in pseudo-arclength
    fn initialize(
        &mut self,
        work: &mut Workspace,
        ddl_ini: f64,
        u: &Vector,
        l: f64,
        dir: IniDir,
        args: &mut A,
    ) -> Result<(), StrError>;

    /// Calculates u, λ and s such that G(u(s), λ(s)) = 0
    fn step(&mut self, work: &mut Workspace, u: &Vector, l: f64, stop: Stop, args: &mut A) -> Result<Status, StrError>;

    /// Handles the accept case by updating (u, l) and calculating a new stepsize
    ///
    /// Returns `rerr` the relative error used in stepsize adaptation
    fn accept(&mut self, work: &mut Workspace, u: &mut Vector, l: &mut f64, args: &mut A) -> Result<f64, StrError>;

    /// Handles the reject case by calculating a new stepsize
    fn reject(&mut self, work: &mut Workspace, args: &mut A);
}
