use super::{Direction, State, Workspace};
use crate::StrError;

/// Defines the numerical solver
pub(crate) trait SolverTrait<A>: Send {
    /// Perform initialization such as computing the first tangent vector in pseudo-arclength
    fn initialize(
        &mut self,
        work: &mut Workspace,
        state: &mut State,
        dir: Direction,
        args: &mut A,
    ) -> Result<(), StrError>;

    /// Calculates u, λ and s such that G(u(s), λ(s)) = 0
    fn step(&mut self, work: &mut Workspace, state: &State, args: &mut A) -> Result<(), StrError>;

    /// Handles the accept case by updating the state and calculating a new stepsize
    fn accept(&mut self, work: &mut Workspace, state: &mut State, args: &mut A) -> Result<(), StrError>;

    /// Handles the reject case by calculating a new stepsize
    fn reject(&mut self, work: &mut Workspace, h: f64, args: &mut A);
}
