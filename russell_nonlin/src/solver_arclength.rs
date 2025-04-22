use super::{Config, SolverTrait, State, System, TgVec, Workspace};
use crate::StrError;
use russell_lab::{vec_copy, vec_update};
use russell_sparse::numerical_jacobian;

/// Implements the natural parameter continuation method to solve G(u, λ) = 0
pub struct SolverArclength<'a, A> {
    /// Configuration options
    config: Config,

    /// System
    system: System<'a, A>,

    /// Use the numerical Gu matrix
    use_num_ggu: bool,
}

impl<'a, A> SolverArclength<'a, A> {
    /// Allocates a new instance
    pub fn new(config: Config, system: System<'a, A>) -> Self {
        let use_num_ggu = config.use_numerical_jacobian || system.calc_ggu.is_none();
        SolverArclength {
            config,
            system,
            use_num_ggu,
        }
    }

    /// Calculates the Gu = ∂G/∂u matrix
    fn calc_ggu(&mut self, work: &mut Workspace, args: &mut A) -> Result<(), StrError> {
        // assemble Gu matrix
        work.stats.sw_jacobian.reset();
        work.ggu.reset();
        if self.use_num_ggu {
            // numerical
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
            // analytical
            work.stats.n_jacobian += 1;
            (self.system.calc_ggu.as_ref().unwrap())(&mut work.ggu, work.l, &work.u, args)?;
        }
        work.stats.stop_sw_jacobian();

        // factorize Gu matrix
        work.stats.sw_factor.reset();
        work.stats.n_factor += 1;
        work.ls.actual.factorize(&mut work.ggu, self.config.lin_sol_config)?;
        work.stats.stop_sw_factor();
        Ok(())
    }

    /// Calculates the Gλ = ∂G/∂λ matrix
    fn calc_ggl(&mut self, work: &mut Workspace, args: &mut A) -> Result<(), StrError> {
        match self.system.calc_ggl.as_ref() {
            Some(calc_ggl) => {
                (calc_ggl)(&mut work.ggl, work.l, &work.u, args)?;
            }
            None => return Err("calc_ggl is required for the Arclength method"),
        }
        Ok(())
    }

    /// Performs a single iteration
    fn iterate(&mut self, iteration: usize, work: &mut Workspace, args: &mut A, logging: bool) -> Result<(), StrError> {
        // calculate G(u, λ)
        work.stats.n_function += 1;
        (self.system.calc_gg)(&mut work.gg, work.l, &work.u, args)?;

        // check convergence on G
        work.err.analyze_residual(iteration, &work.gg, 0.0)?;
        if work.err.converged() {
            if logging {
                work.log.iteration(iteration, &work.err);
            }
            return Ok(());
        }

        // compute Gu matrix
        let recompute_ggu = iteration == 0 || !self.config.constant_tangent;
        if recompute_ggu {
            self.calc_ggu(work, args)?;
        }

        // solve linear system
        work.stats.sw_lin_sol.reset();
        work.stats.n_lin_sol += 1;
        work.ls.actual.solve(&mut work.mdu, &work.gg, false)?;
        work.stats.stop_sw_lin_sol();

        // check convergence on δu
        work.err.analyze_delta(iteration, &work.mdu, 0.0)?;
        if logging {
            work.log.iteration(iteration, &work.err);
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
            (f)(&work.u, args);
        }

        // external: backup/restore secondary variables to prepare for the update
        if let Some(f) = self.system.iteration_prepare_to_update_secondary.as_ref() {
            (f)(iteration == 0, args);
        }

        // external: update secondary variables
        if let Some(f) = self.system.iteration_update_secondary.as_ref() {
            (f)(&work.mdu, &work.u, args)?;
        }

        // exit if linear problem (done)
        if self.config.treat_as_linear {
            work.err.set_converged_linear_problem();
            return Ok(());
        }
        Ok(())
    }
}

impl<'a, A> SolverTrait<A> for SolverArclength<'a, A> {
    /// Perform initialization such as computing the first tangent vector in pseudo-arclength
    fn initialize(&mut self, work: &mut Workspace, state: &State, tg: TgVec, args: &mut A) -> Result<(), StrError> {
        // check if the tangent vector is available
        if state.duds.dim() != state.u.dim() {
            return Err("duds.ndim != to u.ndim; the tangent vector is required for the Arclength method");
        }

        // calculate Gu = ∂G/∂u
        self.calc_ggu(work, args)?;

        // calculate Gλ = ∂G/∂λ
        match self.system.calc_ggl.as_ref() {
            Some(calc_ggl) => {
                calc_ggl(&mut work.ggl, state.l, &state.u, args)?;
            }
            None => return Err("calc_ggl is required for the Arclength method"),
        }
        Ok(())
    }

    /// Calculates u such that G(u, λ) = 0
    ///
    /// * `auto` indicates that automatic stepsize control is used.
    ///   On auto mode, large (δu,δλ) is not an error; otherwise, it is an error
    fn step(&mut self, work: &mut Workspace, state: &State, args: &mut A) -> Result<(), StrError> {
        // set workspace with trial values
        vec_copy(&mut work.u, &state.u).unwrap(); // u_trial ← u0
        work.l = state.l + state.h; // λ_trial ← λ0 + h

        // external: create a copy of external state variables
        if work.auto {
            if let Some(f) = self.system.step_backup_state.as_ref() {
                (f)(args);
            }
        }

        // external: prepare to iterate (e.g., reset algorithmic variables)
        if let Some(f) = self.system.step_reset_algorithmic_variables.as_ref() {
            (f)(args);
        }

        // reset iteration error control
        work.err.reset(state);

        // iteration loop
        let logging = true;
        for iteration in 0..self.config.allowed_iterations {
            // stats
            work.stats.n_iterations_total += 1;
            work.stats.n_iterations_max = usize::max(work.stats.n_iterations_max, iteration + 1);

            // run Newton-Raphson iteration
            self.iterate(iteration, work, args, logging)?;

            // stop if converged
            if work.err.converged() {
                break;
            }

            // check for failures
            if work.err.failures(iteration, &mut work.stats) {
                work.iterations_failed = true;
                break;
            }
        }
        Ok(())
    }

    /// Handles the accept case by updating the state and calculating a new stepsize
    fn accept(&mut self, work: &mut Workspace, state: &mut State) {
        vec_copy(&mut state.u, &work.u).unwrap();
        state.l = work.l;
        work.h_new = state.h;
    }

    /// Handles the reject case by calculating a new stepsize
    fn reject(&mut self, work: &mut Workspace, h: f64, args: &mut A) {
        // external: restore external state variables
        if work.auto {
            if let Some(f) = self.system.step_restore_state.as_ref() {
                (f)(args);
            }
        }

        // estimate new stepsize
        let newt = work.stats.n_iterations_total;
        let num = self.config.m_safety * ((1 + 2 * self.config.allowed_iterations) as f64);
        let den = (newt + 2 * self.config.allowed_iterations) as f64;
        let fac = f64::min(self.config.m_safety, num / den);
        let div = f64::max(
            self.config.m_min,
            f64::min(self.config.m_max, f64::powf(work.rel_error, 0.25) / fac),
        );
        work.h_new = h / div;
    }
}
