use crate::StrError;
use crate::{OdeSolverTrait, Params, System, Workspace};
use russell_lab::math::SQRT_6;
use russell_lab::{complex_vec_zip, cpx, format_fortran, vec_copy, Complex64, ComplexVector, Vector};
use russell_sparse::{numerical_jacobian, ComplexCscMatrix, CscMatrix};
use russell_sparse::{ComplexLinSolver, ComplexSparseMatrix, CooMatrix, Genie, LinSolver, SparseMatrix};
use std::thread;

/// Implements the Radau5 method (Radau IIA) (implicit, order 5, embedded) for ODEs and DAEs
///
/// **Note:** The implementation here follows closely the Fortran code named `radau5.f` explained
/// in the reference #2 with some differences. For instance, here, more memory is required than in
/// radau5.f because the variables here are slightly more clearly defined. Also, the coefficient
/// matrices used in the simplified Newton's method are stored in the sparse format (see `russell_sparse`)
/// and their respective linear systems may be solved concurrently. Despite the differences,
/// the Rust and Fortran codes yield similar results (check out the `tests` and `data` directories).
/// Note also that the Fortran code is *faster*.
///
/// # References
///
/// 1. E. Hairer, S. P. Nørsett, G. Wanner (2008) Solving Ordinary Differential Equations I.
///    Non-stiff Problems. Second Revised Edition. Corrected 3rd printing 2008. Springer Series
///    in Computational Mathematics, 528p
/// 2. E. Hairer, G. Wanner (2002) Solving Ordinary Differential Equations II.
///    Stiff and Differential-Algebraic Problems. Second Revised Edition.
///    Corrected 2nd printing 2002. Springer Series in Computational Mathematics, 614p
pub(crate) struct Radau5<'a, F, J, A>
where
    F: Fn(&mut Vector, f64, &Vector, &mut A) -> Result<(), StrError>,
    J: Fn(&mut CooMatrix, f64, f64, &Vector, &mut A) -> Result<(), StrError>,
{
    /// Holds the parameters
    params: Params,

    /// ODE system
    system: &'a System<F, J, A>,

    /// Holds the Jacobian matrix. J = df/dy
    jj: SparseMatrix,

    /// Coefficient matrix (for real system). K_real = γ M - J
    kk_real: SparseMatrix,

    /// Coefficient matrix (for real system). K_comp = (α + βi) M - J
    kk_comp: ComplexSparseMatrix,

    /// Linear solver (for real system)
    solver_real: LinSolver<'a>,

    /// Linear solver (for complex system)
    solver_comp: ComplexLinSolver<'a>,

    /// Indicates that the Jacobian can be reused (once)
    reuse_jacobian: bool,

    /// Indicates that the J, K_real, and K_comp matrices (and their factorizations) can be reused (once)
    reuse_jacobian_kk_and_fact: bool,

    /// Indicates that the Jacobian has been computed
    ///
    /// This flag assists in reusing the Jacobian if the step has been rejected.
    /// Make sure to set this flag to false in `accept`.
    jacobian_computed: bool,

    /// eta tolerance for stepsize control
    eta: f64,

    /// theta variable for stepsize control
    theta: f64,

    /// First function evaluation (for each accepted step)
    k_accepted: Vector,

    /// Scaling vector
    ///
    /// ```text
    /// scaling[i] = abs_tol + rel_tol ⋅ |y[i]|
    /// ```
    scaling: Vector,

    /// Vectors holding the updates. CONT1 of radau5.f
    ///
    /// ```text
    /// v[stg][dim] = ya[dim] + h*sum(a[stg][j]*f[j][dim], j, nstage)
    /// ```
    v0: Vector,
    v1: Vector,
    v2: Vector,
    v12: ComplexVector, // packed (v1, v2)

    /// Vectors holding the function evaluations. F{1,2,3} of radau5.f
    ///
    /// ```text
    /// k[stg][dim] = f(u[stg], v[stg][dim])
    /// ```
    k0: Vector,
    k1: Vector,
    k2: Vector,

    /// Normalized vectors, one for each of the 3 stages. Z{1,2,3} of radau5.f
    z0: Vector,
    z1: Vector,
    z2: Vector,

    /// Collocation values, one for each of the 3 stages. CONT{2,3,4} of radau5.f
    yc0: Vector,
    yc1: Vector,
    yc2: Vector,

    /// Workspace, one for each of the 3 stages
    w0: Vector,
    w1: Vector,
    w2: Vector,

    /// Incremental workspace, one for each of the 3 stages
    dw0: Vector,
    dw1: Vector,
    dw2: Vector,
    dw12: ComplexVector, // packed (dw1, dw2)
}

impl<'a, F, J, A> Radau5<'a, F, J, A>
where
    F: Fn(&mut Vector, f64, &Vector, &mut A) -> Result<(), StrError>,
    J: Fn(&mut CooMatrix, f64, f64, &Vector, &mut A) -> Result<(), StrError>,
{
    /// Allocates a new instance
    pub fn new(params: Params, system: &'a System<F, J, A>) -> Self {
        let ndim = system.ndim;
        let mass_nnz = match system.mass_matrix.as_ref() {
            Some(mass) => mass.get_info().2,
            None => ndim,
        };
        let jac_nnz = if params.newton.use_numerical_jacobian {
            ndim * ndim
        } else {
            system.jac_nnz
        };
        let nnz = mass_nnz + jac_nnz;
        let theta = params.radau5.theta_max;
        Radau5 {
            params,
            system,
            jj: SparseMatrix::new_coo(ndim, ndim, jac_nnz, system.jac_sym).unwrap(),
            kk_real: SparseMatrix::new_coo(ndim, ndim, nnz, system.jac_sym).unwrap(),
            kk_comp: ComplexSparseMatrix::new_coo(ndim, ndim, nnz, system.jac_sym).unwrap(),
            solver_real: LinSolver::new(params.newton.genie).unwrap(),
            solver_comp: ComplexLinSolver::new(params.newton.genie).unwrap(),
            reuse_jacobian: false,
            reuse_jacobian_kk_and_fact: false,
            jacobian_computed: false,
            eta: 1.0,
            theta,
            k_accepted: Vector::new(ndim),
            scaling: Vector::new(ndim),
            v0: Vector::new(ndim),
            v1: Vector::new(ndim),
            v2: Vector::new(ndim),
            v12: ComplexVector::new(ndim),
            k0: Vector::new(ndim),
            k1: Vector::new(ndim),
            k2: Vector::new(ndim),
            z0: Vector::new(ndim),
            z1: Vector::new(ndim),
            z2: Vector::new(ndim),
            yc0: Vector::new(ndim),
            yc1: Vector::new(ndim),
            yc2: Vector::new(ndim),
            w0: Vector::new(ndim),
            w1: Vector::new(ndim),
            w2: Vector::new(ndim),
            dw0: Vector::new(ndim),
            dw1: Vector::new(ndim),
            dw2: Vector::new(ndim),
            dw12: ComplexVector::new(ndim),
        }
    }

    /// Initializes the scaling and k_accepted vectors
    fn initialize(&mut self, work: &mut Workspace, x: f64, y: &Vector, args: &mut A) -> Result<(), StrError> {
        for i in 0..self.system.ndim {
            self.scaling[i] = self.params.tol.abs + self.params.tol.rel * f64::abs(y[i]);
        }
        work.stats.n_function += 1;
        (self.system.function)(&mut self.k_accepted, x, y, args)
    }

    /// Assembles the K_real and K_comp matrices
    fn assemble(&mut self, work: &mut Workspace, x: f64, y: &Vector, h: f64, args: &mut A) -> Result<(), StrError> {
        // auxiliary
        let jj = self.jj.get_coo_mut().unwrap(); // J = df/dy
        let kk_real = self.kk_real.get_coo_mut().unwrap(); // K_real = γ M - J
        let kk_comp = self.kk_comp.get_coo_mut().unwrap(); // K_comp = (α + βi) M - J

        // Jacobian matrix
        if self.reuse_jacobian {
            self.reuse_jacobian = false; // just once
        } else if !self.jacobian_computed {
            work.stats.sw_jacobian.reset();
            work.stats.n_jacobian += 1;
            if self.params.newton.use_numerical_jacobian || !self.system.jac_available {
                work.stats.n_function += self.system.ndim;
                let y_mut = &mut self.w0; // workspace (mutable y)
                let w1 = &mut self.dw0; // workspace
                let w2 = &mut self.dw1; // workspace
                vec_copy(y_mut, y).unwrap();
                numerical_jacobian(jj, 1.0, x, y_mut, w1, w2, args, &self.system.function)?;
            } else {
                (self.system.jacobian)(jj, 1.0, x, y, args)?;
            }
            self.jacobian_computed = true;
            work.stats.stop_sw_jacobian();
        }

        // coefficient matrices
        let alpha = ALPHA / h;
        let beta = BETA / h;
        let gamma = GAMMA / h;
        kk_real.assign(-1.0, jj).unwrap(); // K_real = -J
        kk_comp.assign_real(-1.0, 0.0, jj).unwrap(); // K_comp = -J
        match self.system.mass_matrix.as_ref() {
            Some(mass) => {
                kk_real.augment(gamma, mass).unwrap(); // K_real += γ M
                kk_comp.augment_real(alpha, beta, mass).unwrap(); // K_comp += (α + βi) M
            }
            None => {
                for m in 0..self.system.ndim {
                    kk_real.put(m, m, gamma).unwrap(); // K_real += γ I
                    kk_comp.put(m, m, cpx!(alpha, beta)).unwrap(); // K_comp += (α + βi) I
                }
            }
        }

        // write the matrices and stop
        if let Some(nstep) = self.params.newton.write_matrix_after_nstep_and_stop {
            if work.stats.n_accepted > nstep {
                let csc_jacobian = CscMatrix::from_coo(jj).unwrap();
                let csc_kk_real = CscMatrix::from_coo(&kk_real).unwrap();
                let csc_kk_comp = ComplexCscMatrix::from_coo(&kk_comp).unwrap();
                csc_jacobian.write_matrix_market("/tmp/russell_ode/jacobian.smat", true)?;
                csc_jacobian.write_matrix_market("/tmp/russell_ode/jacobian.mtx", false)?;
                csc_kk_real.write_matrix_market("/tmp/russell_ode/kk_real.smat", true)?;
                csc_kk_real.write_matrix_market("/tmp/russell_ode/kk_real.mtx", false)?;
                csc_kk_comp.write_matrix_market("/tmp/russell_ode/kk_comp.smat", true)?;
                csc_kk_comp.write_matrix_market("/tmp/russell_ode/kk_comp.mtx", false)?;
                return Err("MATRIX FILES GENERATED in /tmp/russell_ode/");
            }
        }
        Ok(())
    }

    /// Factorizes the real and complex systems in serial
    fn factorize(&mut self) -> Result<(), StrError> {
        self.solver_real
            .actual
            .factorize(&mut self.kk_real, self.params.newton.lin_sol_params)?;
        self.solver_comp
            .actual
            .factorize(&mut self.kk_comp, self.params.newton.lin_sol_params)
    }

    /// Factorizes the real and complex systems concurrently
    fn factorize_concurrently(&mut self) -> Result<(), StrError> {
        thread::scope(|scope| {
            let handle_real = scope.spawn(|| {
                self.solver_real
                    .actual
                    .factorize(&mut self.kk_real, self.params.newton.lin_sol_params)
                    .unwrap();
            });
            let handle_comp = scope.spawn(|| {
                self.solver_comp
                    .actual
                    .factorize(&mut self.kk_comp, self.params.newton.lin_sol_params)
                    .unwrap();
            });
            let err_real = handle_real.join();
            let err_comp = handle_comp.join();
            if err_real.is_err() && err_comp.is_err() {
                Err("real and complex factorizations failed")
            } else if err_real.is_err() {
                Err("real factorizations failed")
            } else if err_comp.is_err() {
                Err("complex factorizations failed")
            } else {
                Ok(())
            }
        })
    }

    /// Solves the real and complex linear systems
    fn solve_lin_sys(&mut self) -> Result<(), StrError> {
        self.solver_real
            .actual
            .solve(&mut self.dw0, &self.kk_real, &self.v0, false)?;
        self.solver_comp
            .actual
            .solve(&mut self.dw12, &self.kk_comp, &self.v12, false)?;
        Ok(())
    }

    /// Solves the real and complex linear systems concurrently
    fn solve_lin_sys_concurrently(&mut self) -> Result<(), StrError> {
        thread::scope(|scope| {
            let handle_real = scope.spawn(|| {
                self.solver_real
                    .actual
                    .solve(&mut self.dw0, &self.kk_real, &self.v0, false)
                    .unwrap();
            });
            let handle_comp = scope.spawn(|| {
                self.solver_comp
                    .actual
                    .solve(&mut self.dw12, &self.kk_comp, &self.v12, false)
                    .unwrap();
            });
            let err_real = handle_real.join();
            let err_comp = handle_comp.join();
            if err_real.is_err() && err_comp.is_err() {
                Err("real and complex solutions failed")
            } else if err_real.is_err() {
                Err("real solution failed")
            } else if err_comp.is_err() {
                Err("complex solution failed")
            } else {
                Ok(())
            }
        })
    }
}

impl<'a, F, J, A> OdeSolverTrait<A> for Radau5<'a, F, J, A>
where
    F: Fn(&mut Vector, f64, &Vector, &mut A) -> Result<(), StrError>,
    J: Fn(&mut CooMatrix, f64, f64, &Vector, &mut A) -> Result<(), StrError>,
{
    /// Enables dense output
    fn enable_dense_output(&mut self) -> Result<(), StrError> {
        Ok(())
    }

    /// Calculates the quantities required to update x and y
    fn step(&mut self, work: &mut Workspace, x: f64, y: &Vector, h: f64, args: &mut A) -> Result<(), StrError> {
        // Perform the initialization for the first time
        if work.stats.n_accepted == 0 {
            self.initialize(work, x, y, args)?;
        }

        // constants
        let concurrent = self.params.radau5.concurrent && self.params.newton.genie != Genie::Mumps;
        let ndim = self.system.ndim;

        // Jacobian, K_real, K_comp, and factorizations (for all iterations: simple Newton's method)
        if self.reuse_jacobian_kk_and_fact {
            self.reuse_jacobian_kk_and_fact = false; // just once
        } else {
            self.assemble(work, x, y, h, args)?;
            work.stats.sw_factor.reset();
            work.stats.n_factor += 1;
            if concurrent {
                self.factorize_concurrently()?;
            } else {
                self.factorize()?;
            }
            work.stats.stop_sw_factor();
        }

        // update u
        let u0 = x + C[0] * h;
        let u1 = x + C[1] * h;
        let u2 = x + C[2] * h;

        // starting values for newton iterations (first z and w)
        if work.stats.n_accepted == 0 || self.params.radau5.zero_trial {
            // zero trial
            for m in 0..ndim {
                self.z0[m] = 0.0;
                self.z1[m] = 0.0;
                self.z2[m] = 0.0;
                self.w0[m] = 0.0;
                self.w1[m] = 0.0;
                self.w2[m] = 0.0;
            }
        } else {
            // polynomial trial
            let c3q = h / work.h_prev;
            let c1q = MU1 * c3q;
            let c2q = MU2 * c3q;
            for m in 0..ndim {
                self.z0[m] = c1q * (self.yc0[m] + (c1q - MU4) * (self.yc1[m] + (c1q - MU3) * self.yc2[m]));
                self.z1[m] = c2q * (self.yc0[m] + (c2q - MU4) * (self.yc1[m] + (c2q - MU3) * self.yc2[m]));
                self.z2[m] = c3q * (self.yc0[m] + (c3q - MU4) * (self.yc1[m] + (c3q - MU3) * self.yc2[m]));
                self.w0[m] = TI[0][0] * self.z0[m] + TI[0][1] * self.z1[m] + TI[0][2] * self.z2[m];
                self.w1[m] = TI[1][0] * self.z0[m] + TI[1][1] * self.z1[m] + TI[1][2] * self.z2[m];
                self.w2[m] = TI[2][0] * self.z0[m] + TI[2][1] * self.z1[m] + TI[2][2] * self.z2[m];
            }
        }

        // auxiliary
        let dim = ndim as f64;
        let alpha = ALPHA / h;
        let beta = BETA / h;
        let gamma = GAMMA / h;
        self.eta = f64::powf(f64::max(self.eta, f64::EPSILON), 0.8); // FACCON on line 914 of radau5.f
        self.theta = self.params.radau5.theta_max;
        let mut ldw_old = 0.0;
        let mut thq_old = 0.0;

        // iterations
        let mut success = false;
        work.iterations_diverging = false;
        work.stats.n_iterations = 0; // line 931 of radau5.f
        for _ in 0..self.params.newton.n_iteration_max {
            // stats
            work.stats.n_iterations += 1;

            // evaluate f(x,y) at (u[i], v[i] = y+z[i])
            for m in 0..ndim {
                self.v0[m] = y[m] + self.z0[m];
                self.v1[m] = y[m] + self.z1[m];
                self.v2[m] = y[m] + self.z2[m];
            }
            work.stats.n_function += 3;
            (self.system.function)(&mut self.k0, u0, &self.v0, args)?;
            (self.system.function)(&mut self.k1, u1, &self.v1, args)?;
            (self.system.function)(&mut self.k2, u2, &self.v2, args)?;

            // compute the right-hand side vectors
            let (l0, l1, l2) = match self.system.mass_matrix.as_ref() {
                Some(mass) => {
                    mass.mat_vec_mul(&mut self.dw0, 1.0, &self.w0).unwrap(); // dw0 := M ⋅ w0
                    mass.mat_vec_mul(&mut self.dw1, 1.0, &self.w1).unwrap(); // dw1 := M ⋅ w1
                    mass.mat_vec_mul(&mut self.dw2, 1.0, &self.w2).unwrap(); // dw2 := M ⋅ w2
                    (&self.dw0, &self.dw1, &self.dw2)
                }
                None => (&self.w0, &self.w1, &self.w2),
            };
            {
                let (k0, k1, k2) = (&self.k0, &self.k1, &self.k2);
                for m in 0..ndim {
                    self.v0[m] = TI[0][0] * k0[m] + TI[0][1] * k1[m] + TI[0][2] * k2[m] - gamma * l0[m];
                    self.v1[m] = TI[1][0] * k0[m] + TI[1][1] * k1[m] + TI[1][2] * k2[m] - alpha * l1[m] + beta * l2[m];
                    self.v2[m] = TI[2][0] * k0[m] + TI[2][1] * k1[m] + TI[2][2] * k2[m] - beta * l1[m] - alpha * l2[m];
                }
            }

            // zip vectors
            complex_vec_zip(&mut self.v12, &self.v1, &self.v2).unwrap();

            // solve the linear systems
            work.stats.sw_lin_sol.reset();
            work.stats.n_lin_sol += 1;
            if concurrent {
                self.solve_lin_sys_concurrently()?;
            } else {
                self.solve_lin_sys()?;
            }
            work.stats.stop_sw_lin_sol();

            // update w and z
            for m in 0..ndim {
                self.w0[m] += self.dw0[m];
                self.w1[m] += self.dw12[m].re;
                self.w2[m] += self.dw12[m].im;
                self.z0[m] = T[0][0] * self.w0[m] + T[0][1] * self.w1[m] + T[0][2] * self.w2[m];
                self.z1[m] = T[1][0] * self.w0[m] + T[1][1] * self.w1[m] + T[1][2] * self.w2[m];
                self.z2[m] = T[2][0] * self.w0[m] + T[2][1] * self.w1[m] + T[2][2] * self.w2[m];
            }

            // rms norm of δw
            let mut ldw = 0.0;
            for m in 0..ndim {
                let ratio0 = self.dw0[m] / self.scaling[m];
                let ratio1 = self.dw12[m].re / self.scaling[m];
                let ratio2 = self.dw12[m].im / self.scaling[m];
                ldw += ratio0 * ratio0 + ratio1 * ratio1 + ratio2 * ratio2;
            }
            ldw = f64::sqrt(ldw / (3.0 * dim));

            // auxiliary
            let newt = work.stats.n_iterations;
            let nit = self.params.newton.n_iteration_max;

            // print debug messages
            if self.params.debug {
                println!(
                    "step = {:>5}, newt = {:>5}, ldw ={}, h ={}",
                    work.stats.n_steps,
                    newt,
                    format_fortran(ldw),
                    format_fortran(h),
                );
            }

            // check convergence
            if newt > 1 && newt < nit {
                let thq = ldw / ldw_old;
                if newt == 2 {
                    self.theta = thq;
                } else {
                    self.theta = f64::sqrt(thq * thq_old);
                }
                thq_old = thq;
                if self.theta < 0.99 {
                    self.eta = self.theta / (1.0 - self.theta); // FACCON on line 964 of radau5.f
                    let exp = (nit - 1 - newt) as f64; // line 967 of radau5.f
                    let rel_err = self.eta * ldw * f64::powf(self.theta, exp) / self.params.tol.newton;
                    if rel_err >= 1.0 {
                        // diverging
                        let q_newt = f64::max(1.0e-4, f64::min(20.0, rel_err));
                        let den = (4 + nit - 1 - newt) as f64;
                        work.h_multiplier_diverging = 0.8 * f64::powf(q_newt, -1.0 / den);
                        work.iterations_diverging = true;
                        return Ok(()); // will try again
                    }
                } else {
                    // diverging badly (unexpected step-rejection)
                    work.h_multiplier_diverging = 0.5;
                    work.iterations_diverging = true;
                    return Ok(()); // will try again
                }
            }

            // save old norm
            ldw_old = ldw;

            // success
            if self.eta * ldw < self.params.tol.newton {
                success = true;
                break;
            }
        }

        // check
        work.stats.update_n_iterations_max();
        if !success {
            return Err("Newton-Raphson method did not complete successfully");
        }

        // error estimate //////////////////////////////////////////////////////

        // auxiliary
        let ez = &mut self.w0; // e times z
        let mez = &mut self.w1; // γ M ez   or   γ ez
        let rhs = &mut self.w2; // right-hand side vector
        let err = &mut self.dw0; // error variable

        // compute ez, mez and rhs
        match self.system.mass_matrix.as_ref() {
            Some(mass) => {
                for m in 0..ndim {
                    ez[m] = E0 * self.z0[m] + E1 * self.z1[m] + E2 * self.z2[m];
                }
                mass.mat_vec_mul(mez, gamma, ez).unwrap();
                for m in 0..ndim {
                    rhs[m] = mez[m] + self.k_accepted[m]; // rhs = γ M ez + f0
                }
            }
            None => {
                for m in 0..ndim {
                    ez[m] = E0 * self.z0[m] + E1 * self.z1[m] + E2 * self.z2[m];
                    mez[m] = gamma * ez[m];
                    rhs[m] = mez[m] + self.k_accepted[m]; // rhs = γ ez + f0
                }
            }
        }

        // err := K_real⁻¹ rhs = (γ M - J)⁻¹ rhs   (HW-VII p123 Eq.(8.20))
        self.solver_real.actual.solve(err, &self.kk_real, rhs, false)?;
        work.rel_error = rms_norm(err, &self.scaling);

        // done with the error estimate
        if work.rel_error < 1.0 {
            return Ok(());
        }

        // handle particular case
        if work.stats.n_accepted == 0 || work.follows_reject_step {
            let ype = &mut self.dw1; // y plus err
            let fpe = &mut self.dw2; // f(x, y + err)
            for m in 0..ndim {
                ype[m] = y[m] + err[m];
            }
            work.stats.n_function += 1;
            (self.system.function)(fpe, x, &ype, args)?;
            for m in 0..ndim {
                rhs[m] = mez[m] + fpe[m];
            }
            self.solver_real.actual.solve(err, &self.kk_real, rhs, false)?;
            work.rel_error = rms_norm(err, &self.scaling);
        }
        Ok(())
    }

    /// Updates x and y and computes the next stepsize
    fn accept(
        &mut self,
        work: &mut Workspace,
        x: &mut f64,
        y: &mut Vector,
        h: f64,
        args: &mut A,
    ) -> Result<(), StrError> {
        // do not reuse current Jacobian and decomposition by default
        self.reuse_jacobian_kk_and_fact = false;
        self.reuse_jacobian = false;
        self.jacobian_computed = false;

        // update y and collocation points
        for m in 0..self.system.ndim {
            y[m] += self.z2[m];
            self.yc0[m] = (self.z1[m] - self.z2[m]) / MU4;
            self.yc1[m] = ((self.z0[m] - self.z1[m]) / MU5 - self.yc0[m]) / MU3;
            self.yc2[m] = self.yc1[m] - ((self.z0[m] - self.z1[m]) / MU5 - self.z0[m] / MU1) / MU2;
        }

        // estimate the new stepsize
        let newt = work.stats.n_iterations;
        let num = self.params.step.m_safety * ((1 + 2 * self.params.newton.n_iteration_max) as f64);
        let den = (newt + 2 * self.params.newton.n_iteration_max) as f64;
        let fac = f64::min(self.params.step.m_safety, num / den);
        let div = f64::max(
            self.params.step.m_min,
            f64::min(self.params.step.m_max, f64::powf(work.rel_error, 0.25) / fac),
        );
        let mut h_new = h / div;

        // predictive controller of Gustafsson
        if self.params.radau5.use_pred_control {
            if work.stats.n_accepted > 1 {
                let r2 = work.rel_error * work.rel_error;
                let rp = work.rel_error_prev;
                let fac = (work.h_prev / h) * f64::powf(r2 / rp, 0.25) / self.params.step.m_safety;
                let fac = f64::max(self.params.step.m_min, f64::min(self.params.step.m_max, fac));
                let div = f64::max(div, fac);
                h_new = h / div;
            }
        }

        // update h_new if not reusing factorizations
        let h_ratio = h_new / h;
        self.reuse_jacobian_kk_and_fact = self.theta <= self.params.radau5.theta_max
            && h_ratio >= self.params.radau5.c1h
            && h_ratio <= self.params.radau5.c2h;
        if !self.reuse_jacobian_kk_and_fact {
            work.h_new = h_new;
        }

        // check θ to decide if at least the Jacobian can be reused
        if !self.reuse_jacobian_kk_and_fact {
            self.reuse_jacobian = self.theta <= self.params.radau5.theta_max;
        }

        // update x
        *x += h;

        // re-initialize
        self.initialize(work, *x, y, args)
    }

    /// Rejects the update
    fn reject(&mut self, work: &mut Workspace, h: f64) {
        // estimate new stepsize
        let newt = work.stats.n_iterations;
        let num = self.params.step.m_safety * ((1 + 2 * self.params.newton.n_iteration_max) as f64);
        let den = (newt + 2 * self.params.newton.n_iteration_max) as f64;
        let fac = f64::min(self.params.step.m_safety, num / den);
        let div = f64::max(
            self.params.step.m_min,
            f64::min(self.params.step.m_max, f64::powf(work.rel_error, 0.25) / fac),
        );
        work.h_new = h / div;
    }

    /// Computes the dense output with x-h ≤ x_out ≤ x
    fn dense_output(&self, y_out: &mut Vector, x_out: f64, x: f64, y: &Vector, h: f64) {
        assert!(x_out >= x - h && x_out <= x);
        let s = (x_out - x) / h;
        for m in 0..self.system.ndim {
            y_out[m] = y[m] + s * (self.yc0[m] + (s - MU4) * (self.yc1[m] + (s - MU3) * self.yc2[m]));
        }
    }

    /// Update the parameters (e.g., for sensitive analyses)
    fn update_params(&mut self, params: Params) {
        self.params = params;
    }
}

/// Computes the scaled RMS norm
fn rms_norm(err: &Vector, scaling: &Vector) -> f64 {
    let ndim = err.dim();
    assert_eq!(scaling.dim(), ndim);
    let mut sum = 0.0;
    for m in 0..ndim {
        let ratio = err[m] / scaling[m];
        sum += ratio * ratio;
    }
    f64::max(1e-10, f64::sqrt(sum / (ndim as f64)))
}

// Radau5 constants ------------------------------------------------------------

const ALPHA: f64 = 2.6810828736277521338957907432111121010270319565630;
const BETA: f64 = 3.0504301992474105694263776247875679044407041991795;
const GAMMA: f64 = 3.6378342527444957322084185135777757979459360868739;
const E0: f64 = -2.7623054547485993983499285952820549558040707846130;
const E1: f64 = 0.37993559825272887786874736408712686858426119657697;
const E2: f64 = -0.091629609865225789249276201199804926431531138001387;
const MU1: f64 = 0.15505102572168219018027159252941086080340525193433;
const MU2: f64 = 0.64494897427831780981972840747058913919659474806567;
const MU3: f64 = -0.84494897427831780981972840747058913919659474806567;
const MU4: f64 = -0.35505102572168219018027159252941086080340525193433;
const MU5: f64 = -0.48989794855663561963945681494117827839318949613133;

const C: [f64; 3] = [(4.0 - SQRT_6) / 10.0, (4.0 + SQRT_6) / 10.0, 1.0];

const T: [[f64; 3]; 3] = [
    [
        9.1232394870892942792e-02,
        -0.14125529502095420843,
        -3.0029194105147424492e-02,
    ],
    [0.24171793270710701896, 0.20412935229379993199, 0.38294211275726193779],
    [0.96604818261509293619, 1.0, 0.0],
];

const TI: [[f64; 3]; 3] = [
    [4.3255798900631553510, 0.33919925181580986954, 0.54177053993587487119],
    [-4.1787185915519047273, -0.32768282076106238708, 0.47662355450055045196],
    [-0.50287263494578687595, 2.5719269498556054292, -0.59603920482822492497],
];

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Radau5;
    use crate::{HasJacobian, Method, OdeSolverTrait, Params, Samples, System, Workspace};
    use russell_lab::{format_fortran, format_scientific, Vector};
    use russell_sparse::Genie;
    use serial_test::serial;

    // IMPORTANT:
    // Since MUMPS is not thread-safe, we need to use serial_test::serial

    #[test]
    fn radau5_works() {
        // This test relates to Table 21.13 of Kreyszig's book, page 921

        // problem
        let (system, x0, y0, mut args, y_fn_x) = Samples::kreyszig_ex4_page920();
        let ndim = system.ndim;

        // allocate structs
        let params = Params::new(Method::Radau5);
        let mut solver = Radau5::new(params, &system);
        let mut work = Workspace::new(Method::Radau5);

        // message
        println!("{:>4}{:>23}{:>23}", "step", "err_y0", "err_y1");

        // numerical approximation
        let h = 0.4;
        let mut x = x0;
        let mut y = y0.clone();
        let mut y_ana = Vector::new(ndim);
        let mut n_fcn_correct = 0;
        for n in 0..2 {
            // call step
            solver.step(&mut work, x, &y, h, &mut args).unwrap();

            // update number of function evaluations
            let nit = work.stats.n_iterations;
            if n == 0 {
                assert_eq!(work.stats.n_iterations, 2);
                n_fcn_correct += 1 + 3 * nit + 1; // initialize + iterations + error-estimate
            } else {
                assert_eq!(work.stats.n_iterations, 1);
                n_fcn_correct += 3 * nit; // iterations
            }

            // important: update n_accepted (must precede `accept`)
            work.stats.n_accepted += 1;

            // call accept
            solver.accept(&mut work, &mut x, &mut y, h, &mut args).unwrap();

            // important: save previous stepsize and relative error (must succeed `accept`)
            work.h_prev = h;
            work.rel_error_prev = f64::max(params.step.rel_error_prev_min, work.rel_error);

            // update number of function evaluations
            n_fcn_correct += 1; // re-initialize

            // check the results
            y_fn_x(&mut y_ana, x, &mut args);
            let err_y0 = f64::abs(y[0] - y_ana[0]);
            let err_y1 = f64::abs(y[1] - y_ana[1]);
            println!("{:>4}{}{}", n, format_fortran(err_y0), format_fortran(err_y1));
            if n == 0 {
                assert!(err_y0 < 1.1e-2);
                assert!(err_y1 < 1.1e-1);
            } else {
                assert!(err_y0 < 5.15e-4);
                assert!(err_y1 < 5.15e-3);
            }
        }

        // check number of function evaluations
        assert_eq!(work.stats.n_function, n_fcn_correct);
        assert_eq!(work.stats.n_jacobian, 1); // simple Newton's method
    }

    #[test]
    fn radau5_works_num_jacobian() {
        // This test relates to Table 21.13 of Kreyszig's book, page 921

        // problem
        let (system, x0, y0, mut args, y_fn_x) = Samples::kreyszig_ex4_page920();
        let ndim = system.ndim;

        // allocate structs
        let mut params = Params::new(Method::Radau5);
        params.newton.use_numerical_jacobian = true;
        let mut solver = Radau5::new(params, &system);
        let mut work = Workspace::new(Method::Radau5);

        // message
        println!("{:>4}{:>23}{:>23}", "step", "err_y0", "err_y1");

        // numerical approximation
        let h = 0.4;
        let mut x = x0;
        let mut y = y0.clone();
        let mut y_ana = Vector::new(ndim);
        let mut n_fcn_correct = 0;
        for n in 0..2 {
            // call step
            solver.step(&mut work, x, &y, h, &mut args).unwrap();

            // update number of function evaluations
            let nit = work.stats.n_iterations;
            if n == 0 {
                assert_eq!(work.stats.n_iterations, 2);
                n_fcn_correct += 1 + 3 * nit + 1; // initialize + iterations + error-estimate
                n_fcn_correct += ndim; // to compute Jacobian (on the first step; simple Newton)
            } else {
                assert_eq!(work.stats.n_iterations, 2); // 1 iteration more than with analytical Jacobian
                n_fcn_correct += 3 * nit; // iterations
            }

            // important: update n_accepted (must precede `accept`)
            work.stats.n_accepted += 1;

            // call accept
            solver.accept(&mut work, &mut x, &mut y, h, &mut args).unwrap();

            // important: save previous stepsize and relative error (must succeed `accept`)
            work.h_prev = h;
            work.rel_error_prev = f64::max(params.step.rel_error_prev_min, work.rel_error);

            // update number of function evaluations
            n_fcn_correct += 1; // re-initialize

            // check the results
            y_fn_x(&mut y_ana, x, &mut args);
            let err_y0 = f64::abs(y[0] - y_ana[0]);
            let err_y1 = f64::abs(y[1] - y_ana[1]);
            println!("{:>4}{}{}", n, format_fortran(err_y0), format_fortran(err_y1));
            if n == 0 {
                assert!(err_y0 < 1.1e-2);
                assert!(err_y1 < 1.1e-1);
            } else {
                assert!(err_y0 < 5.15e-4);
                assert!(err_y1 < 5.15e-3);
            }
        }

        // check number of function evaluations
        assert_eq!(work.stats.n_function, n_fcn_correct);
    }

    #[test]
    #[serial]
    fn radau5_works_mass_matrix() {
        for symmetric in [true, false] {
            for genie in [Genie::Umfpack, Genie::Mumps] {
                // problem
                let (system, x0, y0, mut args, y_fn_x) = Samples::simple_system_with_mass_matrix(symmetric, genie);
                let ndim = system.ndim;

                // allocate structs
                let mut params = Params::new(Method::Radau5);
                params.newton.genie = genie;
                let mut solver = Radau5::new(params, &system);
                let mut work = Workspace::new(Method::Radau5);

                // message
                println!("\nsymmetric = {:?} --- {:?}", symmetric, genie);
                println!("{:>4}{:>10}{:>10}{:>10}", "step", "err_y0", "err_y1", "err_y2");

                // numerical approximation
                let h = 0.1;
                let mut x = x0;
                let mut y = y0.clone();
                let mut y_ana = Vector::new(ndim);
                for n in 0..4 {
                    // call step
                    solver.step(&mut work, x, &y, h, &mut args).unwrap();

                    // important: update n_accepted (must precede `accept`)
                    work.stats.n_accepted += 1;

                    // call accept
                    solver.accept(&mut work, &mut x, &mut y, h, &mut args).unwrap();

                    // important: save previous stepsize and relative error (must succeed `accept`)
                    work.h_prev = h;
                    work.rel_error_prev = f64::max(params.step.rel_error_prev_min, work.rel_error);

                    // check the results
                    y_fn_x(&mut y_ana, x, &mut args);
                    let err_y0 = f64::abs(y[0] - y_ana[0]);
                    let err_y1 = f64::abs(y[1] - y_ana[1]);
                    let err_y2 = f64::abs(y[2] - y_ana[2]);
                    println!(
                        "{:>4}{}{}{}",
                        n,
                        format_scientific(err_y0, 10, 2),
                        format_scientific(err_y1, 10, 2),
                        format_scientific(err_y2, 10, 2)
                    );
                    assert!(err_y0 < 1e-9);
                    assert!(err_y1 < 1e-9);
                    assert!(err_y2 < 1e-8);
                }
            }
        }
    }

    #[test]
    fn radau5_handles_errors() {
        struct Args {
            count_f: usize,
        }
        let system = System::new(
            1,
            |f, _, _, args: &mut Args| {
                f[0] = 1.0;
                args.count_f += 1;
                if args.count_f == 1 {
                    Err("f: stop (count = 1; initialize)")
                } else if args.count_f == 4 {
                    Err("f: stop (count = 4; num-jacobian)")
                } else {
                    Ok(())
                }
            },
            |jj, alpha, _x, _y, _args: &mut Args| {
                jj.reset();
                jj.put(0, 0, alpha * (0.0)).unwrap();
                Err("jj: stop")
            },
            HasJacobian::Yes,
            None,
            None,
        );
        let params = Params::new(Method::Radau5);
        let mut solver = Radau5::new(params, &system);
        let mut work = Workspace::new(Method::Radau5);
        let x = 0.0;
        let y = Vector::from(&[0.0]);
        let h = 0.1;
        let mut args = Args { count_f: 0 };
        assert_eq!(
            solver.step(&mut work, x, &y, h, &mut args).err(),
            Some("f: stop (count = 1; initialize)")
        );
        assert_eq!(solver.step(&mut work, x, &y, h, &mut args).err(), Some("jj: stop"));
        solver.params.newton.use_numerical_jacobian = true;
        assert_eq!(
            solver.step(&mut work, x, &y, h, &mut args).err(),
            Some("f: stop (count = 4; num-jacobian)")
        );

        solver.update_params(params);
    }
}
