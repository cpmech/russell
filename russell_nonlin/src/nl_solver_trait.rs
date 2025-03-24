use super::{NlParams, Workspace};
use crate::StrError;
use russell_lab::Vector;

/// Defines the numerical solver
pub(crate) trait NlSolverTrait<A>: Send {
    /// Calculates the quantities required to update u, λ and s
    fn step(&mut self, work: &mut Workspace, u: &Vector, l: f64, s: f64, h: f64, args: &mut A) -> Result<(), StrError>;

    /// Updates x and y and computes the next stepsize
    fn accept(
        &mut self,
        work: &mut Workspace,
        u: &Vector,
        l: &mut f64,
        s: &mut f64,
        h: f64,
        args: &mut A,
    ) -> Result<(), StrError>;

    /// Rejects the update
    fn reject(&mut self, work: &mut Workspace, h: f64);

    /// Update the parameters (e.g., for sensitive analyses)
    fn update_params(&mut self, params: NlParams);
}
