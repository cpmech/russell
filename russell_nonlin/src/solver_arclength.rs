#![allow(unused)]

use super::{NlConfig, NlSolverTrait, NlSystem, State, Workspace};
use crate::StrError;
use russell_lab::Vector;

pub struct SolverArclength<'a, A> {
    /// Configuration options
    config: NlConfig,

    /// System
    system: NlSystem<'a, A>,
}

impl<'a, A> SolverArclength<'a, A> {
    /// Allocates a new instance
    pub fn new(config: NlConfig, system: NlSystem<'a, A>) -> Self {
        SolverArclength { config, system }
    }
}

impl<'a, A> NlSolverTrait<A> for SolverArclength<'a, A> {
    /// Calculates u, λ and s such that G(u(s), λ(s)) = 0
    fn step(&mut self, work: &mut Workspace, state: &State, args: &mut A, auto: bool) -> Result<(), StrError> {
        Err("TODO: SolverArclength")
    }

    fn accept(&mut self, work: &mut Workspace, state: &mut State, args: &mut A) {
        panic!("TODO: SolverArclength")
    }

    /// Handles the reject case by calculating a new stepsize
    fn reject(&mut self, work: &mut Workspace, h: f64, args: &mut A, auto: bool) {
        panic!("TODO: SolverArclength")
    }
}
