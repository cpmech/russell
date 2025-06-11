use super::{AutoStep, Config, Direction, SolverTrait, State, Stop, System, Workspace};
use crate::StrError;
use russell_lab::{vec_copy, vec_update};
use russell_sparse::numerical_jacobian;

/// Implements the natural parameter continuation method to solve G(u, λ) = 0
pub struct SolverNatural<'a, A> {
    /// Configuration options
    config: Config,

    /// System
    system: System<'a, A>,
}

impl<'a, A> SolverNatural<'a, A> {
    /// Allocates a new instance
    pub fn new(config: Config, system: System<'a, A>) -> Self {
        SolverNatural { config, system }
    }

    /// Performs a single iteration
    fn iterate(&mut self, work: &mut Workspace, args: &mut A, logging: bool) -> Result<(), StrError> {
        // calculate G(u, λ)
        work.stats.n_function += 1;
        (self.system.calc_gg)(&mut work.gg, work.l, &work.u, args)?;

        // check convergence on G
        work.err.analyze_residual(work.n_iteration, &work.gg, 0.0)?;
        if work.err.converged() {
            if logging {
                work.log.iteration(work.n_iteration, &work.err);
            }
            return Ok(());
        }

        // auxiliary flags
        let recompute_jacobian = work.n_iteration == 0 || !self.config.constant_tangent;
        let use_num_jacobian = self.config.use_numerical_jacobian || self.system.calc_ggu.is_none();

        // compute Jacobian matrix
        if recompute_jacobian {
            // assemble Gu matrix
            work.stats.sw_jacobian.reset();
            work.ggu.reset();
            if use_num_jacobian {
                // numerical Jacobian
                work.stats.n_function += self.system.ndim;
                numerical_jacobian(
                    &mut work.ggu,
                    1.0,
                    work.l,
                    &mut work.u,
                    &mut work.u_aux1,
                    &mut work.u_aux2,
                    args,
                    self.system.calc_gg.as_ref(),
                )?;
            } else {
                // analytical Jacobian
                work.stats.n_jacobian += 1;
                (self.system.calc_ggu.as_ref().unwrap())(&mut work.ggu, work.l, &work.u, args)?;
            }
            work.stats.stop_sw_jacobian();

            // factorize Gu matrix
            work.stats.sw_factor.reset();
            work.stats.n_factor += 1;
            work.ls.actual.factorize(&mut work.ggu, self.config.lin_sol_config)?;
            work.stats.stop_sw_factor();
        }

        // solve linear system
        work.stats.sw_lin_sol.reset();
        work.stats.n_lin_sol += 1;
        work.ls.actual.solve(&mut work.mdu, &work.gg, false)?;
        work.stats.stop_sw_lin_sol();

        // check convergence on δu
        work.err.analyze_delta(work.n_iteration, &work.mdu)?;
        if logging {
            work.log.iteration(work.n_iteration, &work.err);
        }
        if work.err.converged() {
            return Ok(());
        }

        // avoid large norm(mdu)
        if work.err.is_delta_large() {
            return Ok(()); // need to handle this case outside
        }

        // update: u ← u - mdu = u + δu
        vec_update(&mut work.u, -1.0, &work.mdu).unwrap();

        // external: update starred variables
        if let Some(f) = self.system.iteration_update_starred.as_ref() {
            f(&work.u, args);
        }

        // external: backup/restore secondary variables to prepare for the update
        if let Some(f) = self.system.iteration_prepare_to_update_secondary.as_ref() {
            f(work.n_iteration == 0, args);
        }

        // external: update secondary variables
        if let Some(f) = self.system.iteration_update_secondary.as_ref() {
            f(&work.mdu, &work.u, args)?;
        }
        Ok(())
    }
}

impl<'a, A> SolverTrait<A> for SolverNatural<'a, A> {
    /// Performs initialization
    ///
    /// 1. Calculates the initial stepsize
    /// 2. Determines the first tangent vector in pseudo-arclength
    fn initialize(
        &mut self,
        work: &mut Workspace,
        state: &mut State,
        _dir: Direction,
        stop: Stop,
        auto: AutoStep,
        _args: &mut A,
    ) -> Result<(), StrError> {
        work.h = match auto {
            AutoStep::Yes => match stop {
                Stop::Lambda(l1) => f64::min(self.config.h_ini, l1 - state.l),
                Stop::Steps(_) => self.config.h_ini,
            },
            AutoStep::No(h_eq) => {
                if h_eq < 10.0 * f64::EPSILON {
                    return Err("h must be ≥ 10.0 * f64::EPSILON for fixed stepsize");
                }
                match stop {
                    Stop::Lambda(l1) => {
                        let n = f64::ceil((l1 - state.l) / h_eq) as usize;
                        (l1 - state.l) / (n as f64)
                    }
                    Stop::Steps(_) => h_eq,
                }
            }
        };
        Ok(())
    }

    /// Calculates u such that G(u, λ) = 0
    ///
    /// * `auto` indicates that automatic stepsize control is used.
    ///   On auto mode, large (δu,δλ) is not an error; otherwise, it is an error
    fn step(&mut self, work: &mut Workspace, state: &State, args: &mut A) -> Result<(), StrError> {
        // set workspace with trial values
        vec_copy(&mut work.u, &state.u).unwrap(); // u_trial ← u0
        work.l = state.l + work.h; // λ_trial ← λ0 + h

        // external: create a copy of external state variables
        if work.auto {
            if let Some(f) = self.system.step_backup_state.as_ref() {
                f(args);
            }
        }

        // external: prepare to iterate (e.g., reset algorithmic variables)
        if let Some(f) = self.system.step_reset_algorithmic_variables.as_ref() {
            f(args);
        }

        // reset iteration error control
        work.err.reset(state);

        // iteration loop
        let logging = true;
        work.n_iteration = 0;
        for _ in 0..self.config.allowed_iterations {
            // stats
            work.stats.n_iteration_total += 1;
            work.stats.n_iteration_max = usize::max(work.stats.n_iteration_max, work.n_iteration + 1);

            // run Newton-Raphson iteration
            self.iterate(work, args, logging)?;

            // stop if converged
            if work.err.converged() {
                break;
            }

            // check for failures
            work.err.set_failures(work.n_iteration, &mut work.stats);
            if work.err.failed() {
                break;
            }
            work.n_iteration += 1;
        }

        // done
        work.acceptable = work.err.converged();
        Ok(())
    }

    /// Handles the accept case by updating the state and calculating a new stepsize
    fn accept(&mut self, work: &mut Workspace, state: &mut State, _args: &mut A) -> Result<(), StrError> {
        // update the state
        vec_copy(&mut state.u, &work.u).unwrap(); // u := u₁
        state.l = work.l; // λ := λ₁

        // calculate a new stepsize
        work.h_estimate = work.h; // TODO
        Ok(())
    }

    /// Handles the reject case by calculating a new stepsize
    fn reject(&mut self, work: &mut Workspace, args: &mut A) {
        // external: restore external state variables
        if work.auto {
            if let Some(f) = self.system.step_restore_state.as_ref() {
                f(args);
            }
        }

        // reduce the stepsize
        work.h_estimate = self.config.m_failure * work.h;
    }

    /// Calculates the stepsize that allows reaching the target lambda
    fn target_stepsize(&mut self, work: &mut Workspace, state: &State, lambda_target: f64) {
        assert!(lambda_target > state.l);
        work.h = lambda_target - state.l;
    }
}
