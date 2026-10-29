use super::{IniDir, Status, Stop, Workspace};
use crate::StrError;
use russell_lab::Vector;

/// Defines the numerical solver
pub(crate) trait SolverTrait<A>: Send {
    /// Performs initialization
    ///
    /// 1. Calculates the initial stepsize
    /// 2. Determines the first tangent vector in pseudo-arclength
    fn initialize(&mut self, work: &mut Workspace, ddl_ini: f64, u: &Vector, l: f64, dir: IniDir, args: &mut A) -> Result<(), StrError>;

    /// Performs one continuation step to advance the solution (u, λ)
    fn step(&mut self, work: &mut Workspace, u: &Vector, l: f64, stop: Stop, args: &mut A) -> Result<Status, StrError>;

    /// Performs a final step that lands exactly on a lambda target, when the method supports it
    ///
    /// The default implementation does nothing, because the natural method already clamps the
    /// step to the target. The arclength method uses it to close the drift left by a listed Δλ,
    /// which is converted to a step in s through the previous |dλ/ds|.
    ///
    /// Returns `None` when no step was taken, or the status of the extra step.
    fn polish(&mut self, _work: &mut Workspace, _u: &Vector, _l: f64, _stop: Stop, _args: &mut A) -> Result<Option<Status>, StrError> {
        Ok(None)
    }

    /// Handles the accept case by updating (u, l) and calculating a new stepsize
    ///
    /// Returns `rdiff` the relative difference used in stepsize adaptation
    fn accept(&mut self, work: &mut Workspace, u: &mut Vector, l: &mut f64, args: &mut A) -> Result<f64, StrError>;

    /// Handles the reject case by calculating a new stepsize
    fn reject(&mut self, work: &mut Workspace, args: &mut A);
}
