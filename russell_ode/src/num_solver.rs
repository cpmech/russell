use crate::StrError;
use crate::Workspace;
use russell_lab::Vector;

/// Defines the numerical solver
pub(crate) trait NumSolver<A> {
    /// Initializes the internal variables
    fn initialize(&mut self, x: f64, y: &Vector, args: &mut A) -> Result<(), StrError>;

    /// Calculates the quantities required to update x and y
    fn step(&mut self, work: &mut Workspace, x: f64, y: &Vector, h: f64, args: &mut A) -> Result<(), StrError>;

    /// Updates x and y and computes the next stepsize
    fn accept(
        &mut self,
        work: &mut Workspace,
        x: &mut f64,
        y: &mut Vector,
        h: f64,
        args: &mut A,
    ) -> Result<(), StrError>;

    /// Rejects the update
    fn reject(&mut self, work: &mut Workspace, h: f64);

    /// Computes the dense output
    fn dense_output(&self, y_out: &mut Vector, h: f64, x: f64, x_out: f64);
}
