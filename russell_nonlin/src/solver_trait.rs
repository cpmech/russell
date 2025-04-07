use super::{State, Workspace};
use crate::StrError;

/// Defines the numerical solver
pub(crate) trait NlSolverTrait<A>: Send {
    /// Calculates u, λ and s such that G(u(s), λ(s)) = 0
    fn step(&mut self, work: &mut Workspace, state: &State, args: &mut A, auto: bool) -> Result<(), StrError>;

    /// Handles the accept case by updating the state and calculating a new stepsize
    fn accept(&mut self, work: &mut Workspace, state: &mut State, args: &mut A);

    /// Handles the reject case by calculating a new stepsize
    fn reject(&mut self, work: &mut Workspace, h: f64, args: &mut A, auto: bool);
}
