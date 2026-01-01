use super::{AutoStep, Config, IniDir, Method, Status};
use super::{SolverTrait, Stop, System, Workspace};
use crate::StrError;
use russell_lab::{vec_copy, vec_update, Vector};
use russell_sparse::numerical_jacobian;

/// Implements the natural parameter continuation method to solve G(u, λ) = 0
pub struct SolverNatural<'a, A> {
    /// Configuration options
    config: Config,

    /// System
    system: System<'a, A>,

    /// Sign of the step size to maintain the direction of the continuation
    sign0: f64,

    /// Previous u variable for the curvature estimation
    u_prev: Vector,
}

impl<'a, A> SolverNatural<'a, A> {
    /// Allocates a new instance
    pub fn new(config: Config, system: System<'a, A>) -> Self {
        assert_eq!(config.method, Method::Natural);
        let ndim = system.ndim;
        SolverNatural {
            config,
            system,
            sign0: 1.0,
            u_prev: Vector::new(ndim),
        }
    }

    /// Performs a single iteration
    fn iterate(&mut self, work: &mut Workspace, u: &Vector, args: &mut A) -> Result<Status, StrError> {
        // calculate G(u, λ)
        work.stats.n_function += 1;
        (self.system.calc_gg)(&mut work.gg, work.l, &work.u, args)?;

        // check convergence on G
        let nan_or_inf = work.err.analyze_residual(work.n_iteration, &work.gg, 0.0);
        if nan_or_inf {
            return Ok(Status::NanOrInfResidual);
        }
        if work.err.converged() {
            work.log.iteration(work.n_iteration, &work.err);
            return Ok(Status::Success);
        }

        // auxiliary flags
        let recompute_jacobian = work.n_iteration == 0 || !self.config.constant_tangent;
        let use_num_jacobian = self.config.use_numerical_jacobian || self.system.calc_ggu.is_none();

        // compute Jacobian matrix
        if recompute_jacobian {
            // assemble Gu matrix
            let ndim = self.system.ndim;
            work.stats.sw_jacobian.reset();
            work.ggu.reset();
            if use_num_jacobian {
                // numerical Jacobian
                work.stats.num_jacobian = true;
                work.stats.n_function += self.system.ndim;
                numerical_jacobian(
                    &mut work.ggu,
                    ndim,
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
        let nan_or_inf = work.err.analyze_delta(work.n_iteration, &work.mdu);
        if nan_or_inf {
            return Ok(Status::NanOrInfDelta);
        }
        work.log.iteration(work.n_iteration, &work.err);
        if work.err.converged() {
            return Ok(Status::Success);
        }

        // capture failures
        let status = work.err.capture_failures(work.n_iteration);
        if status.failure() {
            return Ok(status);
        }

        // update: u ← u - mdu = u + δu
        vec_update(&mut work.u, -1.0, &work.mdu).unwrap();

        // external: update secondary variables
        if let Some(f) = self.system.update_secondary_state.as_ref() {
            let do_backup = false; // already done by the predictor
            let status = Status::from_sup(f(do_backup, &u, &work.u, args));
            if status.failure() {
                return Ok(status);
            }
        }

        // success
        Ok(Status::Success)
    }

    /// Calculates the normalized change between the secant vectors passing through the previous, current, and updated points
    ///
    /// Returns `gamma` where:
    ///
    /// * `gamma` -- is the ratio between the norm of the difference between the secant vectors
    ///    and the norm of the current secant vector.
    ///
    /// The secant vectors are:
    ///
    /// ```text
    /// previous: s₋₁ = x₀ - x₋₁
    /// current:  s₀  = x₁ - x₀
    /// ```
    ///
    /// Thus:
    ///
    /// ```text
    ///     ‖ s₀ - s₋₁ ‖   ‖ (x₁ - x₀) - (x₀ - x₋₁) ‖   ‖ x₁ - 2x₀ + x₋₁ ‖
    /// γ = ———————————— = —————————————————————————— = ——————————————————
    ///        ‖ s₀ ‖             ‖ x₁ - x₀ ‖              ‖ x₁ - x₀ ‖
    /// ```
    ///
    /// Note that `(u, l)` corresponds to the initial values `x₀ = (u₀, λ₀)`
    /// and `work` corresponds to the updated values `x₁ = (u₁, λ₁)`.
    fn calculate_rerr(&mut self, work: &mut Workspace, u: &Vector) -> f64 {
        if work.stats.n_accepted > 1 {
            let ndim = self.system.ndim;
            let mut sum = 0.0;
            for i in 0..ndim {
                let v = work.u[i] - 2.0 * u[i] + self.u_prev[i]; // u₁ - 2u₀ + u₋₁
                let r = work.u[i] - u[i]; // u₁ - u₀
                let den = self.config.tg_control_atol + self.config.tg_control_rtol * f64::abs(r);
                sum += v * v / (den * den);
            }
            f64::sqrt(sum / (ndim as f64))
        } else {
            0.0
        }
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
        _u: &Vector,
        l: f64,
        dir: IniDir,
        stop: Stop,
        auto: AutoStep,
        _args: &mut A,
    ) -> Result<(), StrError> {
        work.h = match auto {
            AutoStep::Yes => stop.h_ini(self.config.h_ini, l),
            AutoStep::No(h_eq) => stop.h_eq(h_eq, l),
        };
        self.sign0 = match dir {
            IniDir::Pos => 1.0,
            IniDir::Neg => -1.0,
        };
        Ok(())
    }

    /// Calculates u such that G(u, λ) = 0
    ///
    /// * `auto` indicates that automatic stepsize control is used.
    ///   On auto mode, large (δu,δλ) is not an error; otherwise, it is an error
    fn step(&mut self, work: &mut Workspace, u: &Vector, l: f64, stop: Stop, args: &mut A) -> Result<Status, StrError> {
        // external: create a copy of external state variables
        if work.auto {
            if let Some(f) = self.system.backup_secondary_state.as_ref() {
                f(args);
            }
        }

        // external: prepare to iterate (e.g., reset algorithmic variables)
        if let Some(f) = self.system.prepare_to_iterate.as_ref() {
            f(args);
        }

        // reset iteration error control
        work.err.reset(u, l);

        // start the recording of iteration errors
        work.stats.record_iterations_residuals_start();

        // predictor: set workspace with trial values
        vec_copy(&mut work.u, &u).unwrap(); // u_trial ← u0
        work.l = l + self.sign0 * work.h; // λ_trial ← λ0 + h

        // handle "targeting lambda" mode if needed
        if let Some((l1, is_min)) = stop.lambda() {
            if (work.l < l1 && is_min) || (work.l > l1 && !is_min) {
                work.h = (l1 - l) * self.sign0; // dir_mult will correct the difference
                work.l = l + self.sign0 * work.h; // λ_trial ← λ0 + h
            }
        }

        // predictor: update secondary variables (e.g., local state)
        if let Some(f) = self.system.update_secondary_state.as_ref() {
            let do_backup = true;
            let status = Status::from_sup(f(do_backup, &u, &work.u, args));
            if status.failure() {
                return Ok(status);
            }
        }

        // record the predictor for debugging
        if self.config.debug_predictor {
            if work.predictor_values_debug.is_none() {
                work.predictor_values_debug = Some((Vec::new(), Vec::new(), Vec::new()));
            }
            let predictor_values = work.predictor_values_debug.as_mut().unwrap();
            predictor_values.0.push(work.l);
            predictor_values.1.push(work.u[0]);
            if work.u.dim() > 1 {
                predictor_values.2.push(work.u[1]);
            }
        }

        // iteration loop
        let mut status = Status::Success;
        work.n_iteration = 0;
        for _ in 0..self.config.n_iteration_max {
            // stats
            work.stats.n_iteration_total += 1;

            // run Newton-Raphson iteration
            status = self.iterate(work, u, args)?;
            if status.failure() {
                break;
            }

            // append the iteration residuals to the current step
            work.stats.record_iterations_residuals_append(work.err.residual_max);

            // stop if converged
            if work.err.converged() {
                break;
            }

            // next iteration number
            work.n_iteration += 1;
        }

        // stop the recording of iteration errors
        work.stats.record_iterations_residuals_stop(work.err.converged());

        // log divergence
        if !work.err.converged() {
            work.log.did_not_converge();
        }

        // done
        Ok(status)
    }

    /// Handles the accept case by updating (u, l) and calculating a new stepsize
    ///
    /// Returns `rerr` the relative error used in stepsize adaptation
    fn accept(&mut self, work: &mut Workspace, u: &mut Vector, l: &mut f64, _args: &mut A) -> Result<f64, StrError> {
        // calculate the relative error
        let rerr = self.calculate_rerr(work, u);

        // save previous u
        vec_copy(&mut self.u_prev, &u).unwrap();

        // update the state
        vec_copy(u, &work.u).unwrap(); // u := u₁
        *l = work.l; // λ := λ₁

        // done
        Ok(rerr)
    }

    /// Handles the reject case by calculating a new stepsize
    fn reject(&mut self, work: &mut Workspace, args: &mut A) {
        // external: restore external state variables
        if work.auto {
            if let Some(f) = self.system.restore_secondary_state.as_ref() {
                f(args);
            }
        }

        // remove predictor values
        if self.config.debug_predictor {
            let predictor_values = work.predictor_values_debug.as_mut().unwrap();
            predictor_values.0.pop();
            predictor_values.1.pop();
            if work.u.dim() > 1 {
                predictor_values.2.pop();
            }
        }
    }
}
