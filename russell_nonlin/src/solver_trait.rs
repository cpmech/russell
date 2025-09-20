use super::{AutoStep, Direction, State, Stop, Workspace};
use crate::StrError;

/// Defines the numerical solver
pub(crate) trait SolverTrait<A>: Send {
    /// Performs initialization
    ///
    /// 1. Calculates the initial stepsize
    /// 2. Determines the first tangent vector in pseudo-arclength
    fn initialize(
        &mut self,
        work: &mut Workspace,
        state: &mut State,
        dir: Direction,
        stop: Stop,
        auto: AutoStep,
        args: &mut A,
    ) -> Result<(), StrError>;

    /// Calculates u, λ and s such that G(u(s), λ(s)) = 0
    fn step(&mut self, work: &mut Workspace, state: &State, args: &mut A) -> Result<(), StrError>;

    /// Handles the accept case by updating the state and calculating a new stepsize
    fn accept(&mut self, work: &mut Workspace, state: &mut State, args: &mut A) -> Result<(), StrError>;

    /// Handles the reject case by calculating a new stepsize
    fn reject(&mut self, work: &mut Workspace, args: &mut A);

    /// Calculates the stepsize that allows reaching the target lambda
    fn stepsize_to_reach_lambda(&mut self, work: &mut Workspace, state: &State, target_lambda: f64);
}
