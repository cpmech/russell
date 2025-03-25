#![allow(unused)]

use super::{NlParams, NlSolverTrait, NlState, NlSystem, Workspace};
use crate::StrError;
use russell_lab::{vec_add, vec_update, Vector};

pub struct SolverSimple<'a, A> {
    /// Holds the parameters
    params: NlParams,

    /// System
    system: NlSystem<'a, A>,
}

impl<'a, A> SolverSimple<'a, A> {
    /// Allocates a new instance
    pub fn new(params: NlParams, system: NlSystem<'a, A>) -> Self {
        let ndim = system.ndim;
        SolverSimple { params, system }
    }

    /// Performs a single iteration
    fn do_iteration(
        &mut self,
        iteration: usize,
        work: &mut Workspace,
        state: &mut NlState,
        args: &mut A,
        logging: bool,
    ) -> Result<(), StrError> {
        // calculate G(u, λ)
        (self.system.calc_gg)(&mut work.gg, &state.u, state.l, args)?;

        // check convergence on residual
        work.err.reset();
        work.err.analyze_gh(iteration, &work.gg, 0.0)?;
        if work.err.converged() {
            if logging {
                work.log.iteration(iteration, &work.err);
            }
            return Ok(());
        }

        // compute Jacobian matrix
        if iteration == 0 || !self.params.constant_tangent {
            // assemble Gu matrix
            if self.params.use_numerical_jacobian || self.system.calc_ggu.is_none() {
                // numerical Jacobian
                work.stats.n_function += self.system.ndim;
                panic!("TODO");
            } else {
                // analytical Jacobian
                (self.system.calc_ggu.as_ref().unwrap())(&mut work.ggu, &state.u, state.l, args)?;
            }

            // factorize K matrix
            work.ls.actual.factorize(&mut work.ggu, self.params.lin_sol_params)?;
        }

        // solve linear system
        work.ls.actual.solve(&mut work.mdu, &work.gg, false)?;

        // check convergence on δu
        work.err.analyze_ul(iteration, &work.mdu, 0.0)?;
        if logging {
            work.log.iteration(iteration, &work.err);
        }
        if work.err.converged() {
            return Ok(());
        }

        // avoid large norm(mdu)
        if work.err.large_du_dl() {
            // TODO: flag failed
            return Ok(());
        }

        // update
        vec_update(&mut state.u, -1.0, &work.mdu).unwrap();

        // update starred variables
        if let Some(f) = self.system.update_starred.as_ref() {
            (f)(&state.u, args)?;
        }

        // backup/restore secondary variables
        if let Some(f) = self.system.backup_restore_secondary.as_ref() {
            (f)(iteration == 0, args)?;
        }

        // update secondary variables
        if let Some(f) = self.system.update_secondary.as_ref() {
            (f)(&work.mdu, &state.u, args)?;
        }

        // exit if linear problem
        if self.params.treat_as_linear {
            work.err.set_converged_linear_problem();
            return Ok(());
        }
        Ok(())
    }
}

impl<'a, A> NlSolverTrait<A> for SolverSimple<'a, A> {
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
}
