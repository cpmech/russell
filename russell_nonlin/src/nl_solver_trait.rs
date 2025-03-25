use super::{State, Workspace};
use crate::StrError;

/// Defines the numerical solver
pub(crate) trait NlSolverTrait<A>: Send {
    /// Calculates u, λ and s such that G(u(s), λ(s)) = 0
    fn step(&mut self, work: &mut Workspace, state: &mut State, args: &mut A, auto: bool) -> Result<(), StrError>;
}
