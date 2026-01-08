use super::{AutoStep, Config, IniDir, Method, Status, CONFIG_H_MIN};
use super::{SolverTrait, Stop, System, Workspace};
use crate::StrError;
use russell_lab::{approx_eq, vec_add, vec_copy, vec_copy_scaled, vec_inner, vec_norm};
use russell_lab::{Norm, Vector};
use russell_sparse::{numerical_jacobian, CooMatrix, LinSolver, Sym};

/// Implements the pseudo-arclength continuation method to solve G(u, λ) = 0
///
/// The nonlinear problem is:
///
/// ```text
/// G(u(s), λ(s)) = 0  (1)
///
/// with Gu ≡ ∂G/∂u  and  Gλ ≡ ∂G/∂λ
/// ```
///
/// The pseudo-arclength normalization (constraint) is:
///
/// ```text
/// N = θ (u - u₀)ᵀ du/ds|₀ + (2 - θ) (λ - λ₀) dλ/ds|₀ - σ  (2)
///
/// with Nu₀ ≡ ∂N/∂u|₀ = θ du/ds|₀
/// and  Nλ₀ ≡ ∂N/∂λ|₀ = (2 - θ) dλ/ds|₀
/// ```
///
/// The `θ` constant above is internally selected such that:
///
/// * `θ = 1`: normal operation
/// * `θ = 0`: targeting lambda
///
/// Note that `σ ≈ Δs` only if Δs is small, i.e., σ is not the arclength but the pseudo-arclength.
///
/// The augmented linear system solved at each Newton iteration is:
///
/// ```text
/// ┌           ┐ ┌    ┐   ┌    ┐
/// │ Gu    Gλ  │ │ δu │   │ -G │
/// │           │ │    │ = │    │  (3)
/// │ Nu₀ᵀ  Nλ₀ │ │ δλ │   │ -N │
/// └           ┘ └    ┘   └    ┘
///       A         x         b
/// ```
///
/// To calculate the initial tangent vector, the following applies:
///
/// ```text
/// dG/ds = ∂G/∂u du/ds + ∂G/∂λ dλ/ds ≡ 0  (4)
/// Gu du/ds + Gλ dλ/ds = 0                (5)
/// Gu du/ds = -Gλ dλ/ds                   (6)
/// du/ds = -(Gu⁻¹ Gλ) dλ/ds               (7)
///         ╰────┬────╯
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
/// (du/ds)ᵀ du/ds + (dλ/ds)² = 1  (12)
/// (dλ/ds)² zᵀ z  + (dλ/ds)² = 1  (13)
/// dλ/ds = ±1 / √(1 + zᵀ z)       (14)
/// ```
///
/// Thus, at the initial point `(u₀, λ₀)`:
///
/// ```text
/// dλ/ds|₀ = sign₀ / √(1 + z₀ᵀ z₀)  (15)
/// ```
///
/// Where `z₀` is the solution of `Gu₀ z₀ = -Gλ₀`, requiring
/// that `Gu₀` be non-singular at the initial point.
///
/// The `sign₀` variable is determines the direction along the solution branch
/// and must be given by the user. An option to reuse the previous tangent
/// vector is also available.
///
/// To determine the initial tangent vector, the non-augmented `Gu` Jacobian
/// matrix is required. Thus, having both the augmented `A` Jacobian and `Gu`
/// would need more than double the memory. To avoid this, the `Gu` matrix,
/// used just once to calculate the initial tangent vector, is considered via
/// the following modification of the augmented linear system:
///
/// ```text
/// ┌        ┐ ┌    ┐   ┌      ┐
/// │ Gu₀  0 │ │ z₀ │   │ -Gλ₀ │
/// │        │ │    │ = │      │  (16)
/// │  0   1 │ │ 0  │   │  0   │
/// └        ┘ └    ┘   └      ┘
/// ```
///
/// After Newton's iteration is completed (converged), the tangent vector needs to
/// be updated. To keep following the solution branch in the same direction, the
/// new tangent vector `(du/ds|₁, dλ/ds|₁)` must satisfy:
///
/// ```text
/// du/ds|₀ᵀ du/ds|₁ + dλ/ds|₀ dλ/ds|₁ = 1  (17)
/// ```
///
/// i.e., the inner product between the previous and new vectors is 1.
/// Also, from (5) we have:
///
/// ```text
/// Gu₁ du/ds|₁ + Gλ₁ dλ/ds|₁ = 0  (18)
/// ```
///
/// Thus, the new tangent vector can be calculated from the following linear system:
///
/// ```text
/// ┌           ┐ ┌         ┐   ┌   ┐
/// │ Gu₁   Gλ₁ │ │ du/ds|₁ │   │ 0 │
/// │           │ │         │ = │   │  (19)
/// │ Nu₀ᵀ  Nλ₀ │ │ dλ/ds|₁ │   │ 1 │
/// └           ┘ └         ┘   └   ┘
///       A₁           x₁         b₁
/// ```
///
/// Note that the augmented Jacobian matrix `A₁` is already factorized by the end of the
/// Newton iteration. Hence, it can be reused without adding a significant computation cost.
///
/// Finally, the new tangent vector must be rescaled such that:
///
/// ```text
/// du/ds|₁ᵀ du/ds|₁ + dλ/ds|₁² = 1  (20)
/// ```
///
/// # References
///
/// 1. Spence A, Graham IG (1999) Numerical Methods for Bifurcation Problems. In The Graduate Student’s Guide to
///    Numerical Analysis '98. Springer Series in Computational Mathematics. Ed. by Ainsworth M, Levesley J,
///    Marletta M. vol 26. Springer, Berlin, Heidelberg. <https://doi.org/10.1007/978-3-662-03972-4_5>
/// 2. Doedel EJ (2007) Lecture Notes on Numerical Analysis of Nonlinear Equations. In Numerical Continuation
///    Methods for Dynamical Systems: Path following and boundary value problems. Ed. by Krauskopf B, Osinga HM,
///    Galán-Vioque J. Springer Netherlands, <https://doi.org/10.1007/978-1-4020-6356-5>
/// 3. Mittelmann HD (1986) A Pseudo-Arclength Continuation Method for Nonlinear Eigenvalue Problems,
///    SIAM Journal on Numerical Analysis, 23:5, 1007-1016 <https://doi.org/10.1137/0723068>
pub struct SolverArclength<'a, A> {
    /// Configuration options
    config: &'a Config,

    /// System
    system: System<'a, A>,

    /// Gλ = ∂G/∂λ vector (ndim)
    ggl: Vector,

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

    /// Indicates that the Jacobian matrix (Gu or A) has been computed at least once in the iteration
    ///
    /// This check is necessary because the predictor may be so good that the iteration
    /// stops without even computing the Jacobian matrix.
    iter_jac_computed: bool,

    /// Theta variable to switch the operation mode via the normalization function
    ///
    /// ```text
    /// N = θ (u - u₀)ᵀ du/ds|₀ + (2 - θ) (λ - λ₀)ᵀ dλ/ds|₀ - σ
    /// ```
    ///
    /// * `θ = 1.0`: normal operation
    /// * `θ = 0.0`: targeting lambda
    theta: f64,

    // Previous du/ds vector for the stepsize control
    duds_prev: Vector,

    // Previous dλ/ds for the stepsize control
    dlds_prev: f64,
}

impl<'a, A> SolverArclength<'a, A> {
    /// Allocates a new instance
    pub fn new(config: &'a Config, system: System<'a, A>) -> Result<Self, StrError> {
        assert_eq!(config.method, Method::Arclength);
        let genie = config.genie;
        let use_num_ggu = config.use_numerical_jacobian || system.calc_ggu.is_none();
        if use_num_ggu && system.update_secondary_state.is_some() {
            return Err("The Arclength method cannot use numerical Jacobian with the secondary update function");
        }
        if !config.bordering && system.sym_ggu != Sym::No {
            return Err("The Arclength method requires sym_ggu = Sym::No when not using bordering, even if Gu is symmetric. This requirement is because the augmented matrix A is not symmetric in general.");
        }
        let ndim = system.ndim;
        let nnz_aa = if config.bordering {
            1 // this should be 0, but russell_sparse requires at least one non-zero
        } else {
            system.nnz_ggu + 2 * ndim + 1
        };
        Ok(SolverArclength {
            config,
            system,
            ggl: Vector::new(ndim),
            ls_aug: LinSolver::new(genie).unwrap(),
            aa: CooMatrix::new(ndim + 1, ndim + 1, nnz_aa, Sym::No).unwrap(),
            x: Vector::new(ndim + 1),
            b: Vector::new(ndim + 1),
            use_num_ggu,
            iter_jac_computed: false,
            theta: 1.0,
            duds_prev: Vector::new(ndim),
            dlds_prev: 0.0,
        })
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

    /// Calculates the Gu = ∂G/∂u matrix
    fn calc_ggu(&mut self, work: &mut Workspace, args: &mut A) -> Result<(), StrError> {
        assert_eq!(work.with_ggu, true);
        // assemble Gu matrix
        let ndim = self.system.ndim;
        work.stats.sw_jacobian.reset();
        work.ggu.reset();
        if self.use_num_ggu {
            // numerical
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

    /// Calculates the augmented Jacobian matrix
    ///
    /// The augmented Jacobian matrix is:
    ///
    /// ```text
    /// for_initial_tangent_vector == false:
    ///     ┌           ┐
    ///     │ Gu    Gλ  │
    /// A = │           │
    ///     │ Nu₀ᵀ  Nλ₀ │
    ///     └           ┘
    /// ```
    ///
    /// Nonetheless, when using A to calculate the initial tangent vector, it is:
    ///
    /// ```text
    /// for_initial_tangent_vector == true:
    ///     ┌        ┐
    ///     │ Gu₀  0 │
    /// A = │        │
    ///     │  0   1 │
    ///     └        ┘
    /// ```
    fn calc_aa(
        &mut self,
        work: &mut Workspace,
        args: &mut A,
        for_initial_tangent_vector: bool,
    ) -> Result<(), StrError> {
        assert!(!self.config.bordering);
        // assemble Gu matrix into A
        // IMPORTANT: note that we are passing A down to the calc_ggu callback function
        let ndim = self.system.ndim;
        work.stats.sw_jacobian.reset();
        self.aa.reset();
        if self.use_num_ggu {
            // numerical
            work.stats.num_jacobian = true;
            work.stats.n_function += self.system.ndim;
            numerical_jacobian(
                &mut self.aa,
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
            // analytical
            work.stats.n_jacobian += 1;
            (self.system.calc_ggu.as_ref().unwrap())(&mut self.aa, work.l, &work.u, args)?;
        }

        // set the last row and column of A
        if for_initial_tangent_vector {
            // put 0 on the last row and column and 1 on the diagonal
            // (putting zeros is only necessary because the sparse solver requires it for subsequent calls)
            for i in 0..ndim {
                self.aa.put(i, ndim, 0.0).unwrap();
                self.aa.put(ndim, i, 0.0).unwrap();
            }
            self.aa.put(ndim, ndim, 1.0).unwrap();
        } else {
            // put Gλ, Nu₀ᵀ=θdu/ds|₀, and Nλ₀=(2-θ)dλ/ds|₀ into A
            self.calc_ggl(work, args)?;
            for i in 0..ndim {
                self.aa.put(i, ndim, self.ggl[i]).unwrap();
                self.aa.put(ndim, i, self.theta * work.duds[i]).unwrap();
            }
            self.aa.put(ndim, ndim, (2.0 - self.theta) * work.dlds).unwrap();
        }
        work.stats.stop_sw_jacobian();

        // factorize matrix A
        work.stats.sw_factor.reset();
        work.stats.n_factor += 1;
        self.ls_aug.actual.factorize(&mut self.aa, self.config.lin_sol_config)?;
        work.stats.stop_sw_factor();
        Ok(())
    }

    /// Calculates the initial tangent vector (duds₀, dlds₀)
    ///
    /// Important: work.u and work.l must contain the initial state (u₀, λ₀)
    ///
    /// Steps:
    ///
    /// ```text
    /// Solve:     Gu₀ z = -Gλ₀
    /// Calculate: dλ/ds₀ = ±1 / √(1 + zᵀ z)
    /// Calculate: du/ds₀ = dλ/ds₀ z
    /// ```
    fn calc_initial_tangent(&mut self, work: &mut Workspace, sign0: f64, args: &mut A) -> Result<(), StrError> {
        if work.with_ggu {
            // use Gu directly

            // calculate Gu = ∂G/∂u
            self.calc_ggu(work, args)?;

            // solve mdu := Gu⁻¹ · Gλ = -z₀
            work.stats.sw_lin_sol.reset();
            work.stats.n_lin_sol += 1;
            work.ls.actual.solve(&mut work.mdu, &self.ggl, false)?; // mdu := -z₀
            work.stats.stop_sw_lin_sol();

            // calculate the tangent vector: du/ds|₀ = dλ/ds z₀
            work.dlds = sign0 / f64::sqrt(1.0 + vec_inner(&work.mdu, &work.mdu));
            vec_copy_scaled(&mut work.duds, -work.dlds, &work.mdu).unwrap(); // "-1" because mdu = -z₀
        } else {
            // use the augmented matrix A instead because Gu has not been allocated
            // ┌        ┐ ┌    ┐   ┌      ┐
            // │ Gu₀  0 │ │ z₀ │   │ -Gλ₀ │
            // │        │ │    │ = │      │
            // │  0   1 │ │ 0  │   │  0   │
            // └        ┘ └    ┘   └      ┘
            //      A        x         b

            // calculate the augmented matrix A := Gu = ∂G/∂u
            self.calc_aa(work, args, true)?;

            // set b = (-Gλ₀, 0)
            let ndim = self.system.ndim;
            for i in 0..ndim {
                self.b[i] = -self.ggl[i];
            }
            self.b[ndim] = 0.0;

            // solve x := A⁻¹ · b = (z₀, 0)
            work.stats.sw_lin_sol.reset();
            work.stats.n_lin_sol += 1;
            self.ls_aug.actual.solve(&mut self.x, &self.b, false)?;
            work.stats.stop_sw_lin_sol();

            // calculate the tangent vector: du/ds|₀ = dλ/ds z₀
            assert_eq!(self.x[ndim], 0.0);
            work.dlds = sign0 / f64::sqrt(1.0 + vec_inner(&self.x, &self.x));
            for i in 0..ndim {
                work.duds[i] = work.dlds * self.x[i];
            }
        }
        Ok(())
    }

    /// Performs a single iteration
    fn iterate(&mut self, work: &mut Workspace, u: &Vector, l: f64, args: &mut A) -> Result<Status, StrError> {
        // calculate G(u(s), λ(s))
        work.stats.n_function += 1;
        (self.system.calc_gg)(&mut work.gg, work.l, &work.u, args)?;

        // calculate N = θ (u - u₀)ᵀ du/ds|₀ + (2 - θ) (λ - λ₀) dλ/ds|₀ - σ
        let ndim = self.system.ndim;
        let mut du_part = 0.0; // (u - u₀)ᵀ du/ds|₀
        if self.theta > 0.0 {
            for i in 0..ndim {
                du_part += (work.u[i] - u[i]) * work.duds[i];
            }
        }
        let sigma = work.h;
        let nn = self.theta * du_part + (2.0 - self.theta) * (work.l - l) * work.dlds - sigma;

        // check convergence on (G, N)
        let nan_or_inf = work.err.analyze_residual(work.n_iteration, &work.gg, nn);
        if nan_or_inf {
            return Ok(Status::NanOrInfResidual);
        }
        if work.err.converged() {
            work.log.iteration(work.n_iteration, &work.err);
            return Ok(Status::Success);
        }

        // solve the linear system
        if self.config.bordering {
            // compute and factorize the Jacobian matrix
            let recompute_ggu = work.n_iteration == 0 || !self.config.constant_tangent;
            if recompute_ggu {
                self.calc_ggu(work, args)?;
                self.iter_jac_computed = true;
            }

            // calculate Gλ = ∂G/∂λ
            self.calc_ggl(work, args)?;

            // solve  δua := Gu⁻¹ · Gλ
            let dua = &mut work.mdu;
            work.stats.sw_lin_sol.reset();
            work.stats.n_lin_sol += 1;
            work.ls.actual.solve(dua, &self.ggl, false)?;
            work.stats.stop_sw_lin_sol();

            // solve δub := Gu⁻¹ · G
            let dub = &mut work.u_aux1;
            work.stats.sw_lin_sol.reset();
            work.stats.n_lin_sol += 1;
            work.ls.actual.solve(dub, &work.gg, false)?;
            work.stats.stop_sw_lin_sol();

            // calculate: den = Nu₀ᵀ δua - Nλ₀
            // where Nu₀ = θ du/ds|₀  and  Nλ₀ = (2 - θ) dλ/ds|₀
            let nnl = (2.0 - self.theta) * work.dlds;
            let den = self.theta * vec_inner(&work.duds, &dua) - nnl;
            if f64::abs(den) < CONFIG_H_MIN {
                return Ok(Status::BorderingSmallDenominator);
            }

            // calculate: δλ = (N - Nu₀ᵀ δub) / den
            let dl = (nn - self.theta * vec_inner(&work.duds, &dub)) / den;

            // calculate: δu = -δλ δua - δub  and set  x = (δu, δλ)
            for i in 0..ndim {
                self.x[i] = -dl * dua[i] - dub[i]; // δu
            }
            self.x[ndim] = dl;
        } else {
            // compute and factorize the augmented Jacobian matrix
            let recompute_aa = work.n_iteration == 0 || !self.config.constant_tangent;
            if recompute_aa {
                self.calc_aa(work, args, false)?;
                self.iter_jac_computed = true;
            }

            // set the right-hand side vector b = (-G, -N)
            for i in 0..ndim {
                self.b[i] = -work.gg[i];
            }
            self.b[ndim] = -nn;

            // solve linear system A x = b; thus x = (δu, δλ)
            work.stats.sw_lin_sol.reset();
            work.stats.n_lin_sol += 1;
            self.ls_aug.actual.solve(&mut self.x, &self.b, false)?;
            work.stats.stop_sw_lin_sol();
        }

        // check convergence on x = (δu, δλ)
        let nan_or_inf = work.err.analyze_delta(work.n_iteration, &self.x);
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

        // update: u ← u + δu and λ ← λ + δλ
        for i in 0..ndim {
            work.u[i] += self.x[i];
        }
        work.l += self.x[ndim];

        // external: update secondary variables (e.g., local state)
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
}

impl<'a, A> SolverTrait<A> for SolverArclength<'a, A> {
    /// Performs initialization
    ///
    /// 1. Calculates the initial stepsize
    /// 2. Determines the first tangent vector in pseudo-arclength
    ///
    /// **Note**: Gu₀ must be non-singular
    fn initialize(
        &mut self,
        work: &mut Workspace,
        u: &Vector,
        l: f64,
        dir: IniDir,
        stop: Stop,
        auto: AutoStep,
        args: &mut A,
    ) -> Result<(), StrError> {
        // initial stepsize (σ₀)
        work.h = match auto {
            AutoStep::Yes => stop.h_ini(self.system.get_h_ini(self.config.h_ini, args), l),
            AutoStep::No(h_eq) => stop.h_eq(h_eq, l),
        };

        // set initial values
        vec_copy(&mut work.u, &u).unwrap(); // u₀ = u
        work.l = l; // λ₀ = λ

        // calculate Gλ = ∂G/∂λ
        self.calc_ggl(work, args)?;

        // set the initial direction vector
        match dir {
            IniDir::Pos => self.calc_initial_tangent(work, 1.0, args)?,
            IniDir::Neg => self.calc_initial_tangent(work, -1.0, args)?,
        }
        Ok(())
    }

    /// Calculates (u,λ) such that G(u(s), λ(s)) = 0 and N = 0
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

        // predictor: λ₁ = λ₀ + (2 - θ) σ · dλds₀
        work.l = l + (2.0 - self.theta) * work.h * work.dlds;

        // handle "targeting lambda" mode if needed
        if let Some((l1, is_min)) = stop.lambda() {
            if (work.l < l1 && is_min) || (work.l > l1 && !is_min) {
                self.theta = 0.0; // set θ to targeting lambda mode
                work.h = 2.0 * (l1 - l) * work.dlds; // the sign of dlds will correct the difference
                work.l = l + 2.0 * work.h * work.dlds; // λ₁ = λ₀ + 2 σ · dλds₀
            }
        }

        // predictor: u₁ = u₀ + θ σ · du/ds₀
        if self.theta > 0.0 {
            // u₁ = u₀ + θ σ · duds₀
            vec_add(&mut work.u, 1.0, &u, self.theta * work.h, &work.duds).unwrap();
        } else {
            // u₁ = u₀
            vec_copy(&mut work.u, &u).unwrap();
        }

        // recalculate the predictor by truncating the stepsize if required and possible
        if let Some((i, u1, is_min)) = stop.u_comp() {
            if (work.u[i] < u1 && is_min) || (work.u[i] > u1 && !is_min) {
                if f64::abs(work.duds[i]) > CONFIG_H_MIN {
                    work.h = (u1 - u[i]) / work.duds[i];
                    work.l = l + (2.0 - self.theta) * work.h * work.dlds;
                    vec_add(&mut work.u, 1.0, &u, self.theta * work.h, &work.duds).unwrap();
                } else {
                    return Err("INTERNAL ERROR: duds[i] is too small");
                }
            }
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
    /// Note that:
    ///
    ///  * `work` -- contains the updated values (u₁, λ₁)
    ///  * `(u, l)` -- will be updated from (u₀, λ₀) to (u₁, λ₁)
    ///
    /// Returns `rerr` the relative error used in stepsize adaptation
    fn accept(&mut self, work: &mut Workspace, u: &mut Vector, l: &mut f64, args: &mut A) -> Result<f64, StrError> {
        // create a copy of the tangent vector at the initial point
        vec_copy(&mut self.duds_prev, &work.duds).unwrap();
        self.dlds_prev = work.dlds;

        // update the tangent vector
        let ndim = self.system.ndim;
        if self.config.bordering {
            // calculate Gu = ∂G/∂u at the updated state
            if !self.iter_jac_computed {
                // This is only needed if the iteration converged without computing the Jacobian
                // For example, when the predictor was good enough and no Jacobian was computed
                self.calc_ggu(work, args)?;
            }

            // Note that Gλ was calculated at the updated state during the iteration
            // solve mdu := Gu⁻¹ · Gλ = -z
            work.stats.sw_lin_sol.reset();
            work.stats.n_lin_sol += 1;
            work.ls.actual.solve(&mut work.mdu, &self.ggl, false)?; // mdu := -z
            work.stats.stop_sw_lin_sol();

            // calculate the tangent vector: du/ds|₀ = dλ/ds z₀
            work.dlds = 1.0 / f64::sqrt(1.0 + vec_inner(&work.mdu, &work.mdu));
            vec_copy_scaled(&mut work.duds, -work.dlds, &work.mdu).unwrap(); // "-1" because mdu = -z

            // fix the sign of the tangent vector to keep following in the same direction
            let dot = vec_inner(&work.duds, &self.duds_prev) + work.dlds * self.dlds_prev;
            if dot < 0.0 {
                for i in 0..ndim {
                    work.duds[i] = -work.duds[i];
                }
                work.dlds = -work.dlds;
            }
        } else {
            // compute Jacobian matrix at the updated state
            if !self.iter_jac_computed {
                // This is only needed if the iteration converged without computing the Jacobian
                // For example, when the predictor was good enough and no Jacobian was computed
                self.calc_aa(work, args, false)?;
            }

            // set b = (0, 1)
            self.b.fill(0.0);
            self.b[ndim] = 1.0;

            // solve x = A⁻¹ · b ≡ (du/ds|₁, dλ/ds|₁)
            work.stats.sw_lin_sol.reset();
            work.stats.n_lin_sol += 1;
            self.ls_aug.actual.solve(&mut self.x, &self.b, false)?;
            work.stats.stop_sw_lin_sol();

            // calculate the norm of x
            let norm = vec_norm(&self.x, Norm::Euc);

            // update the tangent vector
            for i in 0..ndim {
                work.duds[i] = self.x[i] / norm;
            }
            work.dlds = self.x[ndim] / norm;
        }

        // make sure the tangent vector is normalized (TODO: remove this check)
        approx_eq(vec_inner(&work.duds, &work.duds) + work.dlds * work.dlds, 1.0, 1e-14);

        // update the state
        vec_copy(u, &work.u).unwrap(); // u := u₁
        *l = work.l; // λ := λ₁

        //
        // stepsize control --- calculate the relative change in the tangent vector
        //

        // calculate the relative difference between dλ/du vectors (RMS of the error)
        let (atol, rtol) = (self.config.tg_control_atol, self.config.tg_control_rtol);
        let mut slope_prev; // previous (dλ/ds|₁) / (du/ds|₁) = dλ/du
        let mut slope; // (dλ/ds|₁) / (du/ds|₁) = dλ/du
        let mut delta;
        let mut den;
        let mut sum = 0.0;
        for i in 0..ndim {
            slope_prev = if f64::abs(self.duds_prev[i]) > CONFIG_H_MIN {
                self.dlds_prev / self.duds_prev[i]
            } else {
                1.0
            };
            slope = if f64::abs(work.duds[i]) > CONFIG_H_MIN {
                work.dlds / work.duds[i]
            } else {
                1.0
            };
            delta = slope - slope_prev;
            den = atol + rtol * f64::abs(slope_prev);
            sum += delta * delta / (den * den);
        }
        let rerr = f64::sqrt(sum / (ndim as f64));

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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::SolverArclength;
    use crate::{Config, Method, Samples};
    use russell_sparse::Sym;

    #[test]
    fn new_captures_errors() {
        let mut config = Config::new();
        config.set_method(Method::Arclength);
        config.set_use_numerical_jacobian(true);
        let (mut system, _, _, _) = Samples::simple_linear_problem(false, false, Sym::No);
        system.set_update_secondary_state(|_, _, _, _, _, _| Ok(false));
        assert_eq!(
            SolverArclength::new(&config, system).err(),
            Some("The Arclength method cannot use numerical Jacobian with the secondary update function")
        );
    }
}
