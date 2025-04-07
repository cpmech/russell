#![allow(unused)]

use super::{Config, SolverTrait, StateRef, System, Workspace};
use crate::StrError;
use russell_lab::Vector;

pub struct SolverArclength<'a, A> {
    /// Configuration options
    config: Config,

    /// System
    system: System<'a, A>,
}

impl<'a, A> SolverArclength<'a, A> {
    /// Allocates a new instance
    pub fn new(config: Config, system: System<'a, A>) -> Self {
        SolverArclength { config, system }
    }
}

impl<'a, A> SolverTrait<A> for SolverArclength<'a, A> {
    /// Calculates u, λ and s such that G(u(s), λ(s)) = 0
    fn step(&mut self, work: &mut Workspace, state: &StateRef, args: &mut A, auto: bool) -> Result<(), StrError> {
        Err("TODO: SolverArclength")
    }

    fn accept(&mut self, work: &mut Workspace, state: &mut StateRef, args: &mut A) {
        panic!("TODO: SolverArclength")
    }

    /// Handles the reject case by calculating a new stepsize
    fn reject(&mut self, work: &mut Workspace, h: f64, args: &mut A, auto: bool) {
        panic!("TODO: SolverArclength")
    }
}
