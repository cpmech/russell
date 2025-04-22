use super::{Config, SolverTrait, State, System, TgVec, Workspace};
use crate::StrError;
use russell_lab::{vec_copy, vec_copy_scaled, vec_inner, vec_update, Vector};
use russell_sparse::{numerical_jacobian, CooMatrix, LinSolver, Sym};

/// Implements the natural parameter continuation method to solve G(u, λ) = 0
///
/// The nonlinear problem is:
///
/// ```text
/// G(u(s), λ(s)) = 0  (1)
///
/// with Gu ≡ ∂G/∂u and Gλ ≡ ∂G/∂λ
/// ```
///
/// The pseudo-arclength normalization (constraint) is:
///
/// ```text
/// Nₒ = (u - u0)ᵀ duds0 + (λ - λ0)ᵀ dλds0 - Δs  (2)
/// ```
///
/// The augmented linear system solved at each Newton iteration is:
///
/// ```text
/// ┌              ┐ ┌    ┐   ┌     ┐
/// │  Gu      Gλ  │ │ δu │   │ -G  │
/// │              │ │    │ = │     │  (3)
/// │ duds0ᵀ dλds0 │ │ δλ │   │ -Nₒ │
/// └              ┘ └    ┘   └     ┘
///         A           x        b
/// ```
///
/// To calculate the initial tangent vector, the following applies:
///
/// ```text
/// dG/ds = ∂G/∂u du/ds + ∂G/∂λ dλ/ds ≡ 0  (4)
/// Gu du/ds + Gλ dλ/ds = 0                (5)
/// Gu du/ds = -Gλ dλ/ds                   (6)
/// du/ds = -(Gu⁻¹ Gλ) dλ/ds               (7)
///              z
/// z ≡ -Gu⁻¹ Gλ                           (8)
/// du/ds = dλ/ds z                        (9)
/// ```
///
/// Substituting (9) into (6) gives:
///
/// ```text
/// Gu (z dλ/ds) = -Gλ dλ/ds  (10)
/// Gu z = -Gλ                (11)
/// ```
///
/// Thus, `z` is the solution of the linear system `Gu z = -Gλ`.
///
/// With the norm of the tangent vector being 1, we have
/// (considering (9) again):
///
/// ```text
/// (du/ds)ᵀ du/ds + (dλ/ds)² = 1
/// (dλ/ds)² zᵀ z  + (dλ/ds)² = 1
/// dλ/ds = ±1 / √(1 + zᵀ z)
/// ```
///
/// Thus, at the initial point `(u0, λ0)`:
///
/// ```text
/// (dλ/ds)₀ = sign₀ / √(1 + z₀ᵀ z₀)
/// ```
///
/// Where `z₀` is the solution of `Gu z₀ = -Gλ₀`, which requires
/// that `Gu` be non-singular at the initial point.
///
/// The `sign₀` variable is determines the direction along the solution branch
/// and must be given by the user. An option to reuse the previous tangent
/// vector is also available.
pub struct SolverArclength<'a, A> {
    /// Configuration options
    config: Config,

    /// System
    system: System<'a, A>,

    /// Gλ = ∂G/∂λ vector (ndim)
    ggl: Vector,

    /// initial u0 (ndim)
    u0: Vector,

    /// initial λ0
    l0: f64,

    /// initial derivative du/ds @ (u0, λ0) (ndim)
    duds0: Vector,

    /// initial derivative dλ/ds @ (u0, λ0)
    dlds0: f64,

    /// Linear solver for the augmented system
    ls_aug: LinSolver<'a>,

    /// Augmented Jacobian matrix A
    aa: CooMatrix,

    /// Left-hand side vector of the linear problem A x = b
    x: Vector,

    /// Right-hand side vector of the linear problem A x = b
    b: Vector,

    /// Use the numerical Gu matrix
    use_num_ggu: bool,
}

impl<'a, A> SolverArclength<'a, A> {
    /// Allocates a new instance
    pub fn new(config: Config, system: System<'a, A>) -> Self {
        let use_num_ggu = config.use_numerical_jacobian || system.calc_ggu.is_none();
        let ndim = system.ndim;
        let nnz_jac = system.nnz_ggu + 2 * ndim + 1;
        SolverArclength {
            config,
            system,
            ggl: Vector::new(ndim),
            u0: Vector::new(ndim),
            l0: 0.0,
            duds0: Vector::new(ndim),
            dlds0: 0.0,
            ls_aug: LinSolver::new(config.genie).unwrap(),
            aa: CooMatrix::new(ndim + 1, ndim + 1, nnz_jac, Sym::No).unwrap(),
            x: Vector::new(ndim + 1),
            b: Vector::new(ndim + 1),
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

    /// Calculates the Gλ = ∂G/∂λ vector
    fn calc_ggl(&mut self, work: &mut Workspace, args: &mut A) -> Result<(), StrError> {
        match self.system.calc_ggl.as_ref() {
            Some(calc_ggl) => {
                (calc_ggl)(&mut self.ggl, work.l, &work.u, args)?;
            }
            None => return Err("calc_ggl is required for the Arclength method"),
        }
        Ok(())
    }

    /// Calculates the augmented Jacobian matrix
    ///
    /// ```text
    ///     ┌              ┐
    ///     │  Gu      Gλ  │
    /// A = │              │
    ///     │ duds0ᵀ dλds0 │
    ///     └              ┘
    /// ```
    fn calc_aa(&mut self, work: &mut Workspace, args: &mut A) -> Result<(), StrError> {
        // assemble Gu matrix into A
        work.stats.sw_jacobian.reset();
        self.aa.reset();
        if self.use_num_ggu {
            // numerical
            work.stats.n_function += self.system.ndim;
            numerical_jacobian(
                &mut self.aa,
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
            (self.system.calc_ggu.as_ref().unwrap())(&mut self.aa, work.l, &work.u, args)?;
        }

        // assemble Gλ, duds0, and dλds0 into A
        let ndim = self.system.ndim;
        self.calc_ggl(work, args)?;
        for i in 0..ndim {
            self.aa.put(i, ndim, self.ggl[i]).unwrap();
            self.aa.put(ndim, i, self.duds0[i]).unwrap();
        }
        self.aa.put(ndim, ndim, self.dlds0).unwrap();
        work.stats.stop_sw_jacobian();

        // factorize matrix A
        work.stats.sw_factor.reset();
        work.stats.n_factor += 1;
        self.ls_aug.actual.factorize(&mut self.aa, self.config.lin_sol_config)?;
        work.stats.stop_sw_factor();
        Ok(())
    }

    /// Performs a single iteration
    fn iterate(
        &mut self,
        iteration: usize,
        work: &mut Workspace,
        dds: f64,
        args: &mut A,
        logging: bool,
    ) -> Result<(), StrError> {
        // calculate G(u(s), λ(s))
        work.stats.n_function += 1;
        (self.system.calc_gg)(&mut work.gg, work.l, &work.u, args)?;

        // calculate Nₒ = (u - u0)ᵀ duds0 + (λ - λ0)ᵀ dλds0 - Δs
        let ndim = self.system.ndim;
        let mut nno = -dds;
        for i in 0..ndim {
            nno += (work.u[i] - self.u0[i]) * self.duds0[i];
        }
        nno += (work.l - self.l0) * self.dlds0;

        // check convergence on (G, Nₒ)
        work.err.analyze_residual(iteration, &work.gg, nno)?;
        if work.err.converged() {
            if logging {
                work.log.iteration(iteration, &work.err);
            }
            return Ok(());
        }

        // compute augmented Jacobian matrix
        let recompute_aa = iteration == 0 || !self.config.constant_tangent;
        if recompute_aa {
            self.calc_aa(work, args)?;
        }

        // set the right-hand side vector b = (-G, -Nₒ)
        for i in 0..ndim {
            self.b[i] = -work.gg[i];
        }
        self.b[ndim] = -nno;

        // solve linear system A x = b; thus x = (δu, δλ)
        work.stats.sw_lin_sol.reset();
        work.stats.n_lin_sol += 1;
        self.ls_aug.actual.solve(&mut self.x, &self.b, false)?;
        work.stats.stop_sw_lin_sol();

        // check convergence on x = (δu, δλ)
        work.err.analyze_delta(iteration, &self.x)?;
        if logging {
            work.log.iteration(iteration, &work.err);
        }
        if work.err.converged() {
            return Ok(());
        }

        // avoid large delta
        if work.err.is_delta_large() {
            return Ok(()); // need to handle this case outside
        }

        // update: u ← u + δu and λ ← λ + δλ
        for i in 0..ndim {
            work.u[i] += self.x[i];
        }
        work.l += self.x[ndim];

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
            for i in 0..ndim {
                work.mdu[i] = -self.x[i]; // mdu = - δu
            }
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

        // set initial values
        let ndim = self.system.ndim;
        for i in 0..ndim {
            self.u0[i] = state.u[i];
            work.u[i] = state.u[i];
        }
        self.l0 = state.l;
        work.l = state.l;

        // get sign of dlds or reuse previous tangent vector
        let sign0 = match tg {
            TgVec::Positive => 1.0,
            TgVec::Negative => -1.0,
            TgVec::Given => {
                vec_copy(&mut self.duds0, &state.duds).unwrap();
                self.dlds0 = state.dlds;
                return Ok(());
            }
        };

        // calculate Gu = ∂G/∂u and Gλ = ∂G/∂λ
        self.calc_ggu(work, args)?;
        self.calc_ggl(work, args)?;

        // (Gu must be non-singular) solve mdu := Gu⁻¹ · Gλ = -z0
        work.stats.sw_lin_sol.reset();
        work.stats.n_lin_sol += 1;
        work.ls.actual.solve(&mut work.mdu, &self.ggl, false)?; // mdu := -z0
        work.stats.stop_sw_lin_sol();

        // calculate tangent vector
        self.dlds0 = sign0 / f64::sqrt(1.0 + vec_inner(&work.mdu, &work.mdu));
        vec_copy_scaled(&mut self.duds0, -self.dlds0, &work.mdu).unwrap(); // "-1" because mdu = -z0
        Ok(())
    }

    /// Calculates (u,λ) such that G(u(s), λ(s)) = 0
    ///
    /// * `auto` indicates that automatic stepsize control is used.
    ///   On auto mode, large (δu,δλ) is not an error; otherwise, it is an error
    fn step(&mut self, work: &mut Workspace, state: &State, args: &mut A) -> Result<(), StrError> {
        // predictor
        let dds = state.h;
        vec_update(&mut work.u, dds, &self.duds0).unwrap(); // u1 = u0 + Δs · duds0
        work.l += dds * self.dlds0; // λ1 = λ0 + Δs · dlds0

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
            self.iterate(iteration, work, dds, args, logging)?;

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
        let ndim = self.system.ndim;
        for i in 0..ndim {
            state.u[i] = work.u[i];
            // TODO: need to calculate new tangent vector
            // state.duds[i] = self.duds0[i];
        }
        state.l = work.l;
        // TODO: need to calculate new tangent vector
        // state.dlds = self.dlds0;
        state.s += state.h;
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
