#![allow(unused)]

use super::{NlParams, NlSolverTrait, NlSystem, State, Workspace};
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
    fn iterate(
        &mut self,
        iteration: usize,
        work: &mut Workspace,
        state: &mut State,
        args: &mut A,
        logging: bool,
    ) -> Result<(), StrError> {
        // calculate G(u, λ)
        (self.system.calc_gg)(&mut work.gg, &state.u, *state.l, args)?;

        // check convergence on G
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
                (self.system.calc_ggu.as_ref().unwrap())(&mut work.ggu, &state.u, *state.l, args)?;
            }

            // factorize Gu matrix
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
            return Ok(());
        }

        // update
        vec_update(&mut state.u, -1.0, &work.mdu).unwrap();

        // update starred variables
        if let Some(f) = self.system.update_starred.as_ref() {
            (f)(&state.u, args)?;
        }

        // backup/restore secondary variables
        if let Some(f) = self.system.prepare_to_update_secondary.as_ref() {
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
    /// Calculates u, λ and s such that G(u(s), λ(s)) = 0
    fn step(&mut self, work: &mut Workspace, state: &mut State, args: &mut A, auto: bool) -> Result<(), StrError> {
        /*
        // check for h too small
        if state.h < self.params.h_min_allowed {
            return Err("h is smaller than the allowed minimum");
        }

        // check for final step
        if *state.l + state.h >= 1.0 {
            if auto && *state.l + state.h != 1.0 {
                // only truncates if λ+Δλ is not exactly equal to 1.0
                state.h = f64::max(self.params.h_min_allowed, 1.0 - *state.l);
            }
            // self.last = true;
        }
        */

        // update λ
        *state.l += state.h;

        // prepare to iterate (e.g., reset algorithmic variables)
        if let Some(f) = self.system.prepare_to_iterate.as_ref() {
            (f)(args)?;
        }

        // iteration loop
        for iteration in 0..self.params.n_iteration_max {
            work.stats.n_iterations += 1;

            // run Newton-Raphson iteration
            self.iterate(iteration, work, state, args, true)?;

            // stop if converged
            if work.err.converged() {
                break;
            }

            // stop if norm(mdu) is too large
            if work.err.large_du_dl() {
                work.stats.n_large_du_dl += 1;
                if !auto {
                    work.log.error_large_ul(work.err.max_ul);
                }
                break;
            }
        }
        Ok(())
    }
}
