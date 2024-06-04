#![allow(unused)]

use super::InterpLagrange;
use crate::StrError;

pub struct MultiRootSolverInterp {
    nn: usize,
}

impl MultiRootSolverInterp {
    /// Allocates a new instance
    pub fn new(nn: usize) -> Self {
        MultiRootSolverInterp { nn }
    }

    /// Finds multiple roots using the Lagrange interpolation method
    pub fn interpolation<F, A>(&self, xa: f64, xb: f64, args: &mut A, mut f: F) -> Result<(), StrError>
    where
        F: FnMut(f64, &mut A) -> Result<f64, StrError>,
    {
        Err("TODO")
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::MultiRootSolverInterp;
    use crate::algo::NoArgs;

    // #[test]
    fn multi_root_solver_works_simple() {
        // function
        let f = |x, _: &mut NoArgs| Ok(x * x - 1.0);
        let (xa, xb) = (-4.0, 4.0);

        // solver
        let solver = MultiRootSolverInterp::new(2);

        // find roots
        let args = &mut 0;
        solver.interpolation(xa, xb, args, f).unwrap();
    }
}
