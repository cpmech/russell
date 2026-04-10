use super::{Config, IniDir, Method, Status};
use super::{SolverTrait, Stop, System, Workspace};
use crate::StrError;
use russell_lab::{vec_add, vec_copy, vec_update, Vector};
use russell_sparse::{CooMatrix, CscMatrix, LinSolver};

/// Implements the natural parameter continuation method to solve G(u, λ) = 0
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

    /// Previous u variable for the curvature estimation
    u_prev: Vector,
}

impl<'a, A> SolverNatural<'a, A> {
    /// Allocates a new instance
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
        self.ls.actual.factorize(&mut self.ggu, self.config.lin_sol_config)?;
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
            let do_backup = false; // already done by the predictor
            let status = Status::from_sup(f(do_backup, &u, &work.u, l, work.l, args));
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
        ddl_ini: f64,
        _u: &Vector,
        _l: f64,
        dir: IniDir,
        _args: &mut A,
    ) -> Result<(), StrError> {
        // set flags
        self.iter_jac_computed = false;

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
                vec_copy(&mut work.u, &u).unwrap();
                self.assemble_and_factorize_jac(work, args)?;
                self.iter_jac_computed = true;
            }
            // using the last factorized Gu: du/dλ = -Gu⁻¹ Gλ
            let ddl = work.l - l; // Δλ
            self.ls.actual.solve(&mut self.mdu, &self.ggl, false)?; // mdu := Gu⁻¹ Gλ
            vec_add(&mut work.u, 1.0, &u, -ddl, &self.mdu).unwrap(); // u₁ = u₀ + Δλ (-Gu⁻¹ Gλ)
        } else {
            // Simple predictor: u₁ = u₀
            vec_copy(&mut work.u, &u).unwrap();
        }

        // predictor: update secondary variables (e.g., local state)
        if let Some(f) = self.system.update_secondary_state.as_ref() {
            let do_backup = true;
            let status = Status::from_sup(f(do_backup, &u, &work.u, l, work.l, args));
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
        // calculate the relative difference
        let rdiff = self.calculate_rerr(work, u);

        // save previous u
        vec_copy(&mut self.u_prev, &u).unwrap();

        // update the state
        vec_copy(u, &work.u).unwrap(); // u := u₁
        *l = work.l; // λ := λ₁

        // done
        Ok(rdiff)
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
