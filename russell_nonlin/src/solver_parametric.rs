#![allow(unused)]

use super::{NlParams, NlSolverTrait, NlSystem, Workspace};
use crate::StrError;
use russell_lab::Vector;

pub struct SolverParametric<'a, A> {
    /// Holds the parameters
    params: NlParams,

    /// System
    system: NlSystem<'a, A>,
}

impl<'a, A> SolverParametric<'a, A> {
    /// Allocates a new instance
    pub fn new(params: NlParams, system: NlSystem<'a, A>) -> Self {
        SolverParametric { params, system }
    }
}

impl<'a, A> NlSolverTrait<A> for SolverParametric<'a, A> {
    /// Calculates the quantities required to update u, λ and s
    fn step(&mut self, work: &mut Workspace, u: &Vector, l: f64, s: f64, h: f64, args: &mut A) -> Result<(), StrError> {
        Err("TODO")
    }

    /// Updates x and y and computes the next stepsize
    fn accept(
        &mut self,
        work: &mut Workspace,
        u: &Vector,
        l: &mut f64,
        s: &mut f64,
        h: f64,
        args: &mut A,
    ) -> Result<(), StrError> {
        Err("TODO")
    }

    /// Rejects the update
    fn reject(&mut self, work: &mut Workspace, h: f64) {}

    /// Update the parameters (e.g., for sensitive analyses)
    fn update_params(&mut self, params: NlParams) {}
}
