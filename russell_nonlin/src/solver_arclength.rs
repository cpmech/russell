#![allow(unused)]

use super::{NlParams, NlSolverTrait, NlSystem, State, Workspace};
use crate::StrError;
use russell_lab::Vector;

pub struct SolverArclength<'a, A> {
    /// Holds the parameters
    params: NlParams,

    /// System
    system: NlSystem<'a, A>,
}

impl<'a, A> SolverArclength<'a, A> {
    /// Allocates a new instance
    pub fn new(params: NlParams, system: NlSystem<'a, A>) -> Self {
        SolverArclength { params, system }
    }
}

impl<'a, A> NlSolverTrait<A> for SolverArclength<'a, A> {
    /// Calculates u, λ and s such that G(u(s), λ(s)) = 0
    fn step(&mut self, work: &mut Workspace, state: &mut State, args: &mut A, auto: bool) -> Result<(), StrError> {
        Err("TODO: SolverArclength")
    }
}
