use super::{AutoStep, Config, Direction, Status, CONFIG_H_MIN};
use super::{SolverTrait, State, Stop, System, Workspace};
use crate::StrError;
use russell_lab::math::PI;
use russell_lab::{vec_copy, vec_update, Vector};
use russell_sparse::numerical_jacobian;

/// Implements the natural parameter continuation method to solve G(u, λ) = 0
pub struct SolverNatural<'a, A> {
    /// Configuration options
    config: Config,

    /// System
    system: System<'a, A>,

    /// Direction multiplier (+1.0 or -1.0)
    dir_mult: f64,

    // variables for the curvature estimation
    ddu: Vector,
    prev_u: Vector,
    prev_l: f64,
}

impl<'a, A> SolverNatural<'a, A> {
    /// Allocates a new instance
    pub fn new(config: Config, system: System<'a, A>) -> Self {
        let ndim = system.ndim;
        SolverNatural {
            config,
            system,
            dir_mult: 1.0,
            ddu: Vector::new(ndim),
            prev_u: Vector::new(ndim),
            prev_l: 0.0,
        }
    }

    /// Performs a single iteration
    fn iterate(&mut self, work: &mut Workspace, state: &State, args: &mut A) -> Result<Status, StrError> {
        // calculate G(u, λ)
        work.stats.n_function += 1;
        (self.system.calc_gg)(&mut work.gg, work.l, &work.u, args)?;

        // check convergence on G
        work.err.analyze_residual(work.n_iteration, &work.gg, 0.0)?;
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
            let status = Status::from_sup(f(do_backup, &state.u, &work.u, args));
            if status.failure() {
                return Ok(status);
            }
        }

        // success
        Ok(Status::Success)
    }

    /// Calculates the angle between the secant vectors passing through the previous, current, and updated points
    ///
    /// Returns `alpha` where:
    ///
    /// * `alpha` -- is  the curvature angle in degrees, if available.
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
    ///         ⎛  s₋₁ · s₀  ⎞
    /// α = acos⎜ —————————— ⎟
    ///         ⎝ ‖s₋₁‖ ‖s₀‖ ⎠
    /// ```
    ///
    /// Note that `state` corresponds to the initial values `x₀ = (u₀, λ₀)`
    /// and `work` corresponds to the updated values `x₁ = (u₁, λ₁)`.
    fn calculate_alpha(&mut self, work: &mut Workspace, state: &State) -> Option<f64> {
        let ndim = self.system.ndim;
        let mut norm_prev = 0.0; // norm of the previous secant vector
        let mut norm_curr = 0.0; // norm of the current secant vector
        for i in 0..ndim {
            work.mdu[i] = state.u[i] - self.prev_u[i]; // s₋₁ = u₀ - u₋₁ (previous secant vector)
            self.ddu[i] = work.u[i] - state.u[i]; //      s₀  = u₁ - u₀  (current secant vector)
            norm_prev += work.mdu[i] * work.mdu[i];
            norm_curr += self.ddu[i] * self.ddu[i];
        }
        let mdl = state.l - self.prev_l; // s₋₁ = λ₀ - λ₋₁ (previous secant vector)
        let ddl = work.l - state.l; //      s₀  = λ₁ - λ₀  (current secant vector)
        norm_prev += mdl * mdl;
        norm_curr += ddl * ddl;
        norm_prev = f64::sqrt(norm_prev);
        norm_curr = f64::sqrt(norm_curr);
        if norm_prev > 0.0 && norm_curr > 0.0 {
            let mut cos_alpha = 0.0;
            for i in 0..ndim {
                cos_alpha += work.mdu[i] * self.ddu[i];
            }
            cos_alpha += mdl * ddl;
            cos_alpha /= norm_prev * norm_curr;
            cos_alpha = f64::clamp(cos_alpha, -1.0, 1.0);
            let alpha = f64::acos(cos_alpha) * 180.0 / PI;
            assert!(f64::is_finite(alpha)); // make sure alpha is finite
            Some(alpha)
        } else {
            None
        }
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
    /// Note that `state` corresponds to the initial values `x₀ = (u₀, λ₀)`
    /// and `work` corresponds to the updated values `x₁ = (u₁, λ₁)`.
    fn calculate_rerr(&mut self, work: &mut Workspace, state: &State) -> f64 {
        let ndim = self.system.ndim;
        let mut sum = 0.0;
        for i in 0..ndim {
            let v = work.u[i] - 2.0 * state.u[i] + self.prev_u[i]; // u₁ - 2u₀ + u₋₁
            let r = work.u[i] - state.u[i]; // u₁ - u₀
            let den = self.config.tg_control_atol + self.config.tg_control_rtol * f64::abs(r);
            sum += v * v / (den * den);
        }
        f64::sqrt(sum / (ndim as f64))
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
        dir: Direction,
        stop: Stop,
        auto: AutoStep,
        _args: &mut A,
    ) -> Result<(), StrError> {
        work.h = match auto {
            AutoStep::Yes => stop.h_ini(self.config.h_ini, state),
            AutoStep::No(h_eq) => stop.h_eq(h_eq, state),
        };
        self.dir_mult = match dir {
            Direction::Pos => 1.0,
            Direction::Neg => -1.0,
        };
        Ok(())
    }

    /// Calculates u such that G(u, λ) = 0
    ///
    /// * `auto` indicates that automatic stepsize control is used.
    ///   On auto mode, large (δu,δλ) is not an error; otherwise, it is an error
    fn step(&mut self, work: &mut Workspace, state: &State, stop: Stop, args: &mut A) -> Result<Status, StrError> {
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
        work.err.reset(state);

        // start the recording of iteration errors
        work.stats.record_iterations_residuals_start();

        // predictor: set workspace with trial values
        vec_copy(&mut work.u, &state.u).unwrap(); // u_trial ← u0
        work.l = state.l + self.dir_mult * work.h; // λ_trial ← λ0 + h

        // handle "targeting lambda" mode if needed
        if let Some((l1, is_min)) = stop.lambda() {
            if (work.l < l1 && is_min) || (work.l > l1 && !is_min) {
                work.h = (l1 - state.l) * self.dir_mult; // dir_mult will correct the difference
                assert!(work.h >= 0.0); // TODO: remove this
                if work.h <= CONFIG_H_MIN {
                    work.target_reached = true;
                    return Ok(Status::Success);
                }
                work.l = state.l + self.dir_mult * work.h; // λ_trial ← λ0 + h
            }
        }

        // predictor: update secondary variables (e.g., local state)
        if let Some(f) = self.system.update_secondary_state.as_ref() {
            let do_backup = true;
            let status = Status::from_sup(f(do_backup, &state.u, &work.u, args));
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
        for _ in 0..self.config.allowed_iterations {
            // stats
            work.stats.n_iteration_total += 1;

            // run Newton-Raphson iteration
            status = self.iterate(work, state, args)?;
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

        // exit on failure (may try again)
        if status.failure() {
            work.acceptable = false;
            return Ok(status);
        }

        // return if not auto mode (fixed stepsize; accept by default)
        if !work.auto {
            work.acceptable = true;
            return Ok(status);
        }

        //
        // adaptivity --- check if the angle (alpha) between previous secant and the current
        // secant vectors is below the tolerance
        //

        work.acceptable = true;
        if work.stats.n_accepted > 0 {
            if let Some(alpha) = self.calculate_alpha(work, state) {
                // check if alpha is acceptable
                work.acceptable = alpha >= 0.0 && alpha <= self.config.alpha_max;
                if !work.acceptable {
                    work.log.alpha_is_not_acceptable();
                    status = Status::LargeAlpha;
                }

                // check if alpha is way out of bounds
                if alpha > self.config.alpha_max_ultimate {
                    status = Status::ExtremelyLargeAlpha;
                }
            }
        }

        // done
        Ok(status)
    }

    /// Handles the accept case by updating the state and calculating a new stepsize
    ///
    /// Returns `rerr` the relative error used in stepsize adaptation
    fn accept(&mut self, work: &mut Workspace, state: &mut State, _args: &mut A) -> Result<f64, StrError> {
        // calculate the relative error
        // let mut rerr = 0.0;
        // if work.stats.n_accepted > 1 {
        //     if let Some(gamma) = self.calculate_gamma(work, state) {
        //         rerr = gamma / self.config.tg_control_rtol;
        //     }
        // }
        let rerr = if work.stats.n_accepted > 1 {
            self.calculate_rerr(work, state)
        } else {
            0.0
        };

        // save previous u and l
        vec_copy(&mut self.prev_u, &state.u).unwrap();
        self.prev_l = state.l;

        // update the state
        vec_copy(&mut state.u, &work.u).unwrap(); // u := u₁
        state.l = work.l; // λ := λ₁

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
