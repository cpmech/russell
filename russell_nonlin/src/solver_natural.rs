use super::{Config, IniDir, Method, Status, TrialPhase};
use super::{SolverTrait, Stop, System, Workspace};
use crate::{CONFIG_H_MIN, StrError};
use russell_lab::{Vector, vec_add, vec_copy, vec_update};
use russell_sparse::{CooMatrix, CscMatrix, LinSolver};

/// Implements the natural parameter continuation method
///
/// Solves `G(u, λ) = 0` by treating λ as a fixed parameter and using Newton-Raphson
/// to converge on u at each step. After convergence, λ is advanced by Δλ and the
/// process repeats until a stop criterion is met.
///
/// This is the simplest continuation method and works well for problems without
/// turning points (folds). For problems with folds, use `SolverArclength` instead.
pub struct SolverNatural<'a, A> {
    /// Configuration options
    config: &'a Config,

    /// System
    system: System<'a, A>,

    /// Sign of the step size to maintain the direction of the continuation
    sign0: f64,

    /// Indicates that the Jacobian matrix (Gu) has been computed at least once in the iteration
    ///
    /// This check is necessary because the predictor may be so good that the iteration
    /// stops without even computing the Jacobian matrix (e.g., in linear problems).
    iter_jac_computed: bool,

    /// Holds the Gλ = ∂G/∂λ vector (not used by this method)
    ggl: Vector,

    /// Holds the Gu = ∂G/∂u matrix for the bordering algorithm
    ggu: CooMatrix,

    /// Holds -δu (negative of iteration increment) for the bordering algorithm
    mdu: Vector,

    /// Linear solver
    ls: LinSolver<'a>,

    /// Previous u variable for secant direction estimation
    u_prev: Vector,

    /// Previous λ variable for secant direction estimation
    l_prev: f64,
}

impl<'a, A> SolverNatural<'a, A> {
    /// Allocates a new instance
    ///
    /// Initializes the natural parameter continuation solver. This method
    /// allocates internal workspace including the Jacobian matrix, linear
    /// solver, and auxiliary vectors.
    ///
    /// # Arguments
    ///
    /// * `config` -- solver configuration; must have `config.method == Method::Natural`
    /// * `system` -- the nonlinear system definition
    ///
    /// # Returns
    ///
    /// * `Ok(SolverNatural)` if allocation succeeds
    /// * `Err(StrError)` if the linear solver fails to initialize
    pub fn new(config: &'a Config, system: System<'a, A>) -> Result<Self, StrError> {
        assert_eq!(config.method, Method::Natural);
        let genie = config.genie;
        let ndim = system.ndim;
        let ggu = CooMatrix::new(ndim, ndim, system.nnz_ggu, system.sym_ggu).unwrap();
        Ok(SolverNatural {
            config,
            system,
            sign0: 1.0,
            iter_jac_computed: false,
            ggl: Vector::new(ndim),
            ggu,
            mdu: Vector::new(ndim),
            ls: LinSolver::new(genie)?,
            u_prev: Vector::new(ndim),
            l_prev: 0.0,
        })
    }

    /// Assembles and factorizes the Jacobian matrix
    ///
    /// Calculates Gu = ∂G/∂u and Gλ = ∂G/∂λ)
    fn assemble_and_factorize_jac(&mut self, work: &mut Workspace, args: &mut A) -> Result<(), StrError> {
        // assemble Gu and Gλ
        work.stats.sw_jacobian.reset();
        self.ggu.reset();
        work.stats.n_jacobian += 1;
        (self.system.calc_jac)(&mut self.ggu, &mut self.ggl, work.l, &work.u, args)?;
        work.stats.stop_sw_jacobian();

        // write Gu to a file
        if let Some(nstep) = self.config.write_matrix_after_nstep_and_stop {
            if nstep > work.stats.n_accepted {
                let csc = CscMatrix::from_coo(&self.ggu).unwrap();
                let key = format!("/tmp/russell_nonlin/ggu_natural-{:03}", work.stats.n_accepted).to_string();
                csc.write_matrix_market(&(key.clone() + ".smat"), true, 1e-14)?;
                csc.write_matrix_market(&(key + ".mtx"), true, 1e-14)?;
                return Err("MATRIX FILES GENERATED in /tmp/russell_nonlin/");
            }
        }

        // factorize Gu matrix
        work.stats.sw_factor.reset();
        work.stats.n_factor += 1;
        self.ls.actual.preprocess(&self.ggu, self.config.lin_sol_config)?;
        work.stats.stop_sw_factor();
        Ok(())
    }

    /// Performs a single iteration
    fn iterate(&mut self, work: &mut Workspace, u: &Vector, l: f64, args: &mut A) -> Result<Status, StrError> {
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

        // assemble and factorize the Jacobian matrix
        self.assemble_and_factorize_jac(work, args)?;

        // solve linear system
        work.stats.sw_lin_sol.reset();
        work.stats.n_lin_sol += 1;
        self.ls.actual.solve(&mut self.mdu, &work.gg, false)?;
        work.stats.stop_sw_lin_sol();

        // check convergence on δu
        let nan_or_inf = work.err.analyze_delta(work.n_iteration, &self.mdu);
        if nan_or_inf {
            return Ok(Status::NanOrInfDelta);
        }
        work.log.iteration(work.n_iteration, &work.err);

        // Check the absolute ceiling before accepting a converged delta. The arclength bounds
        // are inactive here because reset sets allowed_step_l2 to f64::INFINITY.
        if let Some(status) = work.err.check_large_delta() {
            return Ok(status);
        }
        if work.err.converged() {
            return Ok(Status::Success);
        }

        // capture failures
        let status = work.err.capture_failures(work.n_iteration);
        if status.failure() {
            return Ok(status);
        }

        // update: u ← u - mdu = u + δu
        vec_update(&mut work.u, -1.0, &self.mdu).unwrap();

        // external: update secondary variables
        if let Some(f) = self.system.update_secondary_state.as_ref() {
            let status = Status::from_sup(f(TrialPhase::Corrector, u, &work.u, l, work.l, args));
            if status.failure() {
                return Ok(status);
            }
        }

        // success
        Ok(Status::Success)
    }

    /// Calculates the normalized unit-secant chord change in `(u, λ)` space
    ///
    /// Computes the Euclidean norm of the difference between consecutive unit secant directions:
    ///
    /// ```text
    /// s₋₁ = (u₀ - u₋₁, λ₀ - λ₋₁),   s₀ = (u₁ - u₀, λ₁ - λ₀)
    ///
    /// ‖ s₀ / ‖s₀‖₂ - s₋₁ / ‖s₋₁‖₂ ‖₂ / tg_control_tol
    /// ```
    fn calculate_rerr(&self, work: &Workspace, u: &Vector, l: f64) -> f64 {
        if work.stats.n_accepted <= 1 {
            return 0.0;
        }
        let ndim = self.system.ndim;
        let mut n0_sq = (l - self.l_prev) * (l - self.l_prev);
        let mut n1_sq = (work.l - l) * (work.l - l);
        for i in 0..ndim {
            let d0 = u[i] - self.u_prev[i];
            let d1 = work.u[i] - u[i];
            n0_sq += d0 * d0;
            n1_sq += d1 * d1;
        }
        let n0 = f64::sqrt(n0_sq);
        let n1 = f64::sqrt(n1_sq);
        if n0 <= CONFIG_H_MIN || n1 <= CONFIG_H_MIN {
            return 0.0;
        }
        let dl0 = (l - self.l_prev) / n0;
        let dl1 = (work.l - l) / n1;
        let mut sum_sq = (dl1 - dl0) * (dl1 - dl0);
        for i in 0..ndim {
            let t0 = (u[i] - self.u_prev[i]) / n0;
            let t1 = (work.u[i] - u[i]) / n1;
            let diff = t1 - t0;
            sum_sq += diff * diff;
        }
        f64::sqrt(sum_sq) / self.config.tg_control_tol
    }
}

impl<'a, A> SolverTrait<A> for SolverNatural<'a, A> {
    /// Performs initialization
    ///
    /// 1. Calculates the initial stepsize
    /// 2. Determines the first tangent vector in pseudo-arclength
    fn initialize(&mut self, work: &mut Workspace, ddl_ini: f64, _u: &Vector, l: f64, dir: IniDir, _args: &mut A) -> Result<(), StrError> {
        // set flags
        self.iter_jac_computed = false;
        self.l_prev = l;

        // set the initial direction
        self.sign0 = match dir {
            IniDir::Pos => 1.0,
            IniDir::Neg => -1.0,
        };

        // initial stepsize: Δλ₀
        work.h = ddl_ini;
        Ok(())
    }

    /// Calculates u such that G(u, λ) = 0
    fn step(&mut self, work: &mut Workspace, u: &Vector, l: f64, stop: Stop, args: &mut A) -> Result<Status, StrError> {
        // external: create a copy of external state variables at the beginning of a fresh step
        //
        // A retried step skips this because the rejection already restored the snapshot.
        if work.backup_needed {
            if let Some(f) = self.system.backup_secondary_state.as_ref() {
                f(args);
            }
            work.backup_needed = false;
        }

        // external: prepare to iterate (e.g., reset algorithmic variables)
        if let Some(f) = self.system.prepare_to_iterate.as_ref() {
            f(args);
        }

        // from here on, any failure must restore the external state variables (see `Solver::solve`)
        work.step_in_flight = true;

        // reset iteration error control
        work.err.reset(u, l);

        // start the recording of iteration errors
        work.stats.record_iterations_residuals_start();

        // predictor: set workspace with trial values
        work.l = l + self.sign0 * work.h; // λ₁ = λ₀ + Δλ

        // handle "targeting lambda" mode if needed
        if let Some((l1, is_min)) = stop.lambda() {
            if (work.l <= l1 && is_min) || (work.l >= l1 && !is_min) {
                work.h = (l1 - l) * self.sign0; // dir_mult will correct the difference
                work.l = l + self.sign0 * work.h; // λ₁ = λ₀ + Δλ
            }
        }

        // predictor: calculate u_trial
        if self.config.euler_predictor {
            // Euler predictor: u₁ = u₀ + Δλ du/dλ
            if !self.iter_jac_computed {
                vec_copy(&mut work.u, u).unwrap();
                self.assemble_and_factorize_jac(work, args)?;
                self.iter_jac_computed = true;
            }
            // using the last factorized Gu: du/dλ = -Gu⁻¹ Gλ
            let ddl = work.l - l; // Δλ
            self.ls.actual.solve(&mut self.mdu, &self.ggl, false)?; // mdu := Gu⁻¹ Gλ
            vec_add(&mut work.u, 1.0, u, -ddl, &self.mdu).unwrap(); // u₁ = u₀ + Δλ (-Gu⁻¹ Gλ)
        } else {
            // Simple predictor: u₁ = u₀
            vec_copy(&mut work.u, u).unwrap();
        }

        // predictor: update secondary variables (e.g., local state)
        if let Some(f) = self.system.update_secondary_state.as_ref() {
            let status = Status::from_sup(f(TrialPhase::Predictor, u, &work.u, l, work.l, args));
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
            status = self.iterate(work, u, l, args)?;
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
    /// Returns `rdiff` the relative difference used in stepsize adaptation
    fn accept(&mut self, work: &mut Workspace, u: &mut Vector, l: &mut f64, _args: &mut A) -> Result<f64, StrError> {
        // the step is settled: the next step must take a fresh snapshot and a failure at the
        // caller's level must not restore this accepted step
        work.step_in_flight = false;
        work.backup_needed = true;

        // calculate the relative difference
        let rdiff = self.calculate_rerr(work, u, *l);

        // save previous (u, l)
        vec_copy(&mut self.u_prev, u).unwrap();
        self.l_prev = *l;

        // update the state
        vec_copy(u, &work.u).unwrap(); // u := u₁
        *l = work.l; // λ := λ₁

        // done
        Ok(rdiff)
    }

    /// Handles the reject case by restoring the state and computing a new stepsize
    ///
    /// Restores the external state variables and clears the debug predictor values. The retry
    /// starts again from the step-start snapshot, so it needs no fresh backup.
    fn reject(&mut self, work: &mut Workspace, args: &mut A) {
        // external: restore external state variables
        self.restore_secondary_state(args);
        work.step_in_flight = false;

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

    /// Restores the external state variables to the step-start snapshot
    fn restore_secondary_state(&mut self, args: &mut A) {
        if let Some(f) = self.system.restore_secondary_state.as_ref() {
            f(args);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use crate::{Config, DeltaLambda, IniDir, NoArgs, Samples, Solver, Status, Stop, SupStatus, System};
    use russell_lab::Vector;
    use russell_sparse::Sym;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn secondary_update_retry_rejects_the_step_and_retries() {
        let (mut system, mut u, _u_ref, mut args) = Samples::two_eq_ref();
        let calls = AtomicUsize::new(0);
        system.set_update_secondary_state(|_, _, _, _, _, _| {
            let n = calls.fetch_add(1, Ordering::SeqCst);
            if n == 0 { Ok(SupStatus::Retry) } else { Ok(SupStatus::Continue) }
        });

        let config = Config::new();
        let mut solver = Solver::new(&config, system).unwrap();
        let mut l = 0.0;
        let status = solver
            .solve(&mut args, &mut u, &mut l, IniDir::Pos, Stop::Steps(2), &DeltaLambda::auto(1e-4), None)
            .unwrap();

        assert_eq!(status, Status::Success);
        let stats = solver.get_stats();
        assert_eq!(stats.n_rejected, 1);
        assert_eq!(stats.n_accepted, 1);
        assert_eq!(stats.n_steps, 2);
    }

    #[test]
    fn natural_near_zero_dof_does_not_collapse_rdiff() {
        // u₀ = λ, u₁ = 1e-10 λ + 1e-4 λ² -> near-zero DOF u₁ must not inflate rdiff
        let system = System::new(
            2,
            Some(2),
            Sym::No,
            |gg, l, u, _args: &mut NoArgs| {
                gg[0] = u[0] - l;
                gg[1] = u[1] - (1e-10 * l + 1e-4 * l * l);
                Ok(())
            },
            |ggu, _ggl, _l, _u, _args: &mut NoArgs| {
                ggu.put(0, 0, 1.0).unwrap();
                ggu.put(1, 1, 1.0).unwrap();
                Ok(())
            },
        )
        .unwrap();

        let mut config = Config::new();
        config.set_record_step_summary(true);
        let mut solver = Solver::new(&config, system).unwrap();
        let mut u = Vector::from(&[0.0, 0.0]);
        let mut l = 0.0;
        let mut args = 0;
        let status = solver
            .solve(
                &mut args,
                &mut u,
                &mut l,
                IniDir::Pos,
                Stop::MaxLambda(1.0),
                &DeltaLambda::auto(0.1),
                None,
            )
            .unwrap();
        assert_eq!(status, Status::Success);
        let stats = solver.get_stats();
        assert!(stats.n_accepted < 10, "n_accepted = {}", stats.n_accepted);
        assert_eq!(stats.n_rejected, 0);
        let summaries = stats.step_summaries.as_ref().unwrap();
        // Exclude the final target-truncated step, whose truncated h alters |u1 - 2*u0 + u_{-1}| / |u1 - u0|.
        for s in &summaries[..summaries.len() - 1] {
            assert!(s.rdiff < 0.5, "step {} rdiff = {:e}", s.step_index, s.rdiff);
        }
    }
}
