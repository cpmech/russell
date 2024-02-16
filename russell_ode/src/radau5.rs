#![allow(unused)]

use crate::StrError;
use crate::{Method, NumSolver, ParamsRadau5, System, Workspace};
use num_complex::Complex64;
use russell_lab::math::{SQRT_3, SQRT_6};
use russell_lab::{complex_vec_unzip, complex_vec_zip, cpx, vec_copy, ComplexVector, Matrix, Vector};
use russell_sparse::LinSolTrait;
use russell_sparse::{ComplexLinSolver, ComplexSparseMatrix, CooMatrix, Genie, LinSolver, SolverUMFPACK, SparseMatrix};
use std::thread;

/// Implements the Radau5 method
pub(crate) struct Radau5<'a, F, J, A>
where
    F: Send + FnMut(&mut Vector, f64, &Vector, &mut A) -> Result<(), StrError>,
    J: Send + FnMut(&mut CooMatrix, f64, &Vector, f64, &mut A) -> Result<(), StrError>,
{
    /// Holds the parameters
    params: ParamsRadau5,

    /// ODE system
    system: System<'a, F, J, A>,

    /// Mass matrix (or diagonal)
    mass: CooMatrix,

    /// Indicates whether the mass matrix is provided or not
    with_mass: bool,

    /// Coefficient matrix (for real system)
    kk_real: SparseMatrix,

    /// Coefficient matrix (for real system)
    kk_comp: ComplexSparseMatrix,

    /// Linear solver (for real system)
    solver_real: LinSolver<'a>,

    /// Linear solver (for complex system)
    solver_comp: ComplexLinSolver<'a>,

    /// Scaling vector
    ///
    /// ```text
    /// scaling[i] = abs_tol + rel_tol ⋅ |y[i]|
    /// ```
    scaling: Vector,

    /// First function evaluation (for each accepted step)
    k_accepted: Vector,

    /// Vectors holding the updates
    ///
    /// ```text
    /// v[stg][dim] = ya[dim] + h*sum(a[stg][j]*f[j][dim], j, nstage)
    /// ```
    v0: Vector,
    v1: Vector,
    v2: Vector,
    v12: ComplexVector, // packed (v1, v2)

    /// Vectors holding the function evaluations
    ///
    /// ```text
    /// k[stg][dim] = f(u[stg], v[stg][dim])
    /// ```
    k0: Vector,
    k1: Vector,
    k2: Vector,

    /// Normalized vectors, one for each of the 3 stages
    z0: Vector,
    z1: Vector,
    z2: Vector,

    /// Collocation values, one for each of the 3 stages
    yc0: Vector,
    yc1: Vector,
    yc2: Vector,

    /// Workspace, one for each of the 3 stages
    w0: Vector,
    w1: Vector,
    w2: Vector,
    w12: ComplexVector, // packed (w1, w2)

    /// Incremental workspace, one for each of the 3 stages
    dw0: Vector,
    dw1: Vector,
    dw2: Vector,
    dw12: ComplexVector, // packed (dw1, dw2)

    /// Error estimate workspace
    ez: Vector,

    /// Error estimate workspace
    err: Vector,

    /// Error estimate workspace
    rhs: Vector,

    /// Indicates that the Jacobian can be reused (once)
    reuse_jacobian_once: bool,

    /// Indicates that the Jacobian and corresponding factorizations can be reused (once)
    reuse_jacobian_and_factors_once: bool,

    /// Indicates that the Jacobian is OK
    jacobian_is_ok: bool,

    // eta tolerance for stepsize control
    eta: f64,

    // theta variable for stepsize control
    theta: f64,
}

impl<'a, F, J, A> Radau5<'a, F, J, A>
where
    F: Send + FnMut(&mut Vector, f64, &Vector, &mut A) -> Result<(), StrError>,
    J: Send + FnMut(&mut CooMatrix, f64, &Vector, f64, &mut A) -> Result<(), StrError>,
{
    /// Allocates a new instance
    pub fn new(params: ParamsRadau5, system: System<'a, F, J, A>) -> Self {
        let ndim = system.ndim;
        let symmetry = Some(system.jac_symmetry);
        let one_based = params.genie == Genie::Mumps;
        let (mass, with_mass) = match system.mass_matrix {
            Some(mm) => (mm.clone(), true),
            None => (CooMatrix::new(ndim, ndim, ndim, symmetry, one_based).unwrap(), false),
        };
        let (_, _, mass_nnz, _) = mass.get_info();
        let nnz = mass_nnz + system.jac_nnz;
        let theta = params.theta_max;
        Radau5 {
            params,
            system,
            mass,
            with_mass,
            kk_real: SparseMatrix::new_coo(ndim, ndim, nnz, symmetry, one_based).unwrap(),
            kk_comp: ComplexSparseMatrix::new_coo(ndim, ndim, nnz, symmetry, one_based).unwrap(),
            solver_real: LinSolver::new(params.genie).unwrap(),
            solver_comp: ComplexLinSolver::new(params.genie).unwrap(),
            scaling: Vector::new(ndim),
            k_accepted: Vector::new(ndim),
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
            w12: ComplexVector::new(ndim),
            dw0: Vector::new(ndim),
            dw1: Vector::new(ndim),
            dw2: Vector::new(ndim),
            dw12: ComplexVector::new(ndim),
            ez: Vector::new(ndim),
            err: Vector::new(ndim),
            rhs: Vector::new(ndim),
            reuse_jacobian_once: false,
            reuse_jacobian_and_factors_once: false,
            jacobian_is_ok: false,
            eta: 1.0,
            theta,
        }
    }

    /// Assembles the K_real and K_comp matrices
    fn assemble(&mut self, work: &mut Workspace, x: f64, y: &Vector, h: f64, args: &mut A) -> Result<(), StrError> {
        // auxiliary
        let kk_real = self.kk_real.get_coo_mut()?;
        let kk_comp = self.kk_comp.get_coo_mut()?;

        // stat
        work.bench.sw_jacobian.reset();
        work.bench.n_jacobian += 1;

        // K_real := -J
        if self.params.use_numerical_jacobian || !self.system.jac_available {
            // numerical Jacobian
            work.bench.n_function += self.system.ndim;
            let y_mut = &mut self.w0; // using w[0] as a workspace
            vec_copy(y_mut, y).unwrap();
            self.system
                .numerical_jacobian(kk_real, x, y_mut, &self.k_accepted, -1.0, args)?;
        } else {
            // analytical Jacobian
            (self.system.jacobian)(kk_real, x, y, -1.0, args)?;
        }

        // factors
        let alpha = ALPHA / h;
        let beta = BETA / h;
        let gamma = GAMMA / h;

        // K_comp := -J  (must do this before augmenting K_real)
        kk_comp.assign_real(1.0, 0.0, kk_real).unwrap();

        // K_comp += (α + βi) M  thus  K_comp = (α + βi) M - J
        kk_comp.augment_real(alpha, beta, &self.mass).unwrap();

        // K_real += γ M  thus  K_real = γ M - J
        kk_real.augment(gamma, &self.mass).unwrap();

        // done
        self.jacobian_is_ok = true;
        work.bench.stop_sw_jacobian();
        Ok(())
    }

    /// Factorizes the real and complex systems in serial
    fn factorize(&mut self) -> Result<(), StrError> {
        self.solver_real
            .actual
            .factorize(&mut self.kk_real, self.params.lin_sol_params)?;
        self.solver_comp
            .actual
            .factorize(&mut self.kk_comp, self.params.lin_sol_params)
    }

    /// Factorizes the real and complex systems concurrently
    fn factorize_concurrently(&mut self) -> Result<(), StrError> {
        thread::scope(|scope| {
            let handle_real = scope.spawn(|| {
                self.solver_real
                    .actual
                    .factorize(&mut self.kk_real, self.params.lin_sol_params)
                    .unwrap();
            });
            let handle_comp = scope.spawn(|| {
                self.solver_comp
                    .actual
                    .factorize(&mut self.kk_comp, self.params.lin_sol_params)
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

    /// Computes the right-hand side of the linear systems
    // #[rustfmt::skip]
    fn right_hand_sides(&mut self, h: f64) {
        let alpha = ALPHA / h;
        let beta = BETA / h;
        let gamma = GAMMA / h;
        if self.with_mass {
            self.mass.mat_vec_mul(&mut self.dw0, 1.0, &self.w0).unwrap();
            self.mass.mat_vec_mul(&mut self.dw1, 1.0, &self.w1).unwrap();
            self.mass.mat_vec_mul(&mut self.dw2, 1.0, &self.w2).unwrap();
        } else {
            for m in 0..self.system.ndim {}
        }
    }
}

impl<'a, F, J, A> NumSolver<A> for Radau5<'a, F, J, A>
where
    F: Send + FnMut(&mut Vector, f64, &Vector, &mut A) -> Result<(), StrError>,
    J: Send + FnMut(&mut CooMatrix, f64, &Vector, f64, &mut A) -> Result<(), StrError>,
{
    /// Initializes the internal variables
    fn initialize(&mut self, x: f64, y: &Vector, args: &mut A) -> Result<(), StrError> {
        for i in 0..self.system.ndim {
            self.scaling[i] = self.params.abs_tol + self.params.rel_tol * f64::abs(y[i]);
        }
        (self.system.function)(&mut self.k_accepted, x, y, args)
    }

    /// Calculates the quantities required to update x and y
    fn step(&mut self, work: &mut Workspace, x: f64, y: &Vector, h: f64, args: &mut A) -> Result<(), StrError> {
        // constants
        let concurrent = self.params.concurrent && self.params.genie != Genie::Mumps;
        let ndim = self.system.ndim;
        let dim = ndim as f64;
        let mni = self.params.m_factor * ((1 + 2 * self.params.n_iteration_max) as f64);

        // access matrices
        let mass_values = self.mass.get_values();
        let kk_real = &mut self.kk_real;
        let kk_comp = &mut self.kk_comp;

        // Jacobian and factorizations (modified/simple Newton's method)
        if self.reuse_jacobian_and_factors_once {
            // if we can reuse the Jacobian and the factorizations, skip their calculations,
            // but set the flag to false to make the next call to compute them
            self.reuse_jacobian_and_factors_once = false;
        } else {
            // otherwise, perform the factorizations
            if self.reuse_jacobian_once {
                // if we can reuse the Jacobian, skip its calculation,
                // but set the flag to false to make the next call to compute it
                self.reuse_jacobian_once = false;
            } else if !self.jacobian_is_ok {
                // otherwise, if the Jacobian is not OK, calculate the Jacobian before the
                // iterations and use it in all iterations (modified/simple Newton's method)
                self.assemble(work, x, y, h, args)?;
            }
            // perform the factorizations
            work.bench.sw_factor.reset();
            work.bench.n_factor += 1;
            if concurrent {
                self.factorize_concurrently();
            } else {
                self.factorize();
            }
            work.bench.stop_sw_factor();
        }

        // update u
        let u0 = x + C[0] * h;
        let u1 = x + C[1] * h;
        let u2 = x + C[2] * h;

        // compute first z and w
        if work.first_step || self.params.zero_trial {
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
        let alpha = ALPHA / h;
        let beta = BETA / h;
        let gamma = GAMMA / h;
        self.eta = f64::powf(f64::max(self.eta, f64::EPSILON), 0.8); // FACCON on line 914 of radau5.f
        self.theta = self.params.theta_max;
        let mut ldw_old = 0.0;
        let mut thq_old = 0.0;

        // iterations
        let mut converged = false;
        work.iterations_diverging = false;
        work.bench.n_iterations = 0;
        for iteration in 0..self.params.n_iteration_max {
            // benchmark
            work.bench.n_iterations += 1;

            // evaluate f(x,y) at (u[i], v[i] = y+z[i])
            for m in 0..ndim {
                self.v0[m] = y[m] + self.z0[m];
                self.v1[m] = y[m] + self.z1[m];
                self.v2[m] = y[m] + self.z2[m];
            }
            work.bench.n_function += 3;
            (self.system.function)(&mut self.k0, u0, &self.v0, args)?;
            (self.system.function)(&mut self.k1, u1, &self.v1, args)?;
            (self.system.function)(&mut self.k2, u2, &self.v2, args)?;

            // compute the right-hand side vectors
            let (l0, l1, l2) = if self.with_mass {
                self.mass.mat_vec_mul(&mut self.dw0, 1.0, &self.w0).unwrap(); // dw0 := M ⋅ w0
                self.mass.mat_vec_mul(&mut self.dw1, 1.0, &self.w1).unwrap(); // dw1 := M ⋅ w1
                self.mass.mat_vec_mul(&mut self.dw2, 1.0, &self.w2).unwrap(); // dw2 := M ⋅ w2
                (&self.dw0, &self.dw1, &self.dw2)
            } else {
                (&self.w0, &self.w1, &self.w2)
            };
            {
                // TODO: use rustfmt::skip
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
            work.bench.sw_lin_sol.reset();
            work.bench.n_lin_sol += 1;
            if concurrent {
                self.solve_lin_sys_concurrently();
            } else {
                self.solve_lin_sys();
            }
            work.bench.stop_sw_lin_sol();

            // unzip vectors
            complex_vec_unzip(&mut self.dw1, &mut self.dw2, &self.dw12).unwrap();

            // update w and z
            for m in 0..ndim {
                self.w0[m] += self.dw0[m];
                self.w1[m] += self.dw1[m];
                self.w2[m] += self.dw2[m];
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

            // check convergence
            if iteration > 0 {
                let thq = ldw / ldw_old;
                if iteration == 1 {
                    self.theta = thq;
                } else {
                    self.theta = f64::sqrt(thq * thq_old);
                }
                thq_old = thq;
                if self.theta < 0.99 {
                    self.eta = self.theta / (1.0 - self.theta); // FACCON on line 964 of radau5.f
                    let newt = (iteration + 1) as f64;
                    let nit = self.params.n_iteration_max as f64;
                    let it_err = ldw * f64::powf(self.theta, nit - newt) / (1.0 - self.theta);
                    let it_rel_err = it_err / self.params.tol_newton;
                    if it_rel_err >= 1.0 {
                        // diverging
                        let q_newt = f64::max(1.0e-4, f64::min(20.0, it_rel_err));
                        work.h_multiplier_diverging = 0.8 * f64::powf(q_newt, -1.0 / (4.0 + nit - 1.0 - newt));
                        work.iterations_diverging = true;
                        break;
                    }
                } else {
                    // diverging badly (unexpected step-rejection)
                    work.h_multiplier_diverging = 0.5;
                    work.iterations_diverging = true;
                    return Ok(());
                }
            }

            // save old norm
            ldw_old = ldw;

            // converged
            if self.eta * ldw < self.params.tol_newton {
                converged = true;
                break;
            }
        }

        // did not converge
        if !converged {
            return Err("Newton-Raphson method did not converge");
        }

        // error estimate
        // TODO

        Ok(())
    }

    /// Updates x and y and computes the next stepsize
    fn accept(
        &mut self,
        _work: &mut Workspace,
        x: &mut f64,
        y: &mut Vector,
        h: f64,
        _args: &mut A,
    ) -> Result<(), StrError> {
        panic!("TODO");
        Ok(())
    }

    /// Rejects the update
    fn reject(&mut self, _work: &mut Workspace, _h: f64) {
        panic!("TODO");
    }

    /// Computes the dense output
    fn dense_output(&self, _y_out: &mut Vector, _h: f64, _x: f64, _x_out: f64) {}
}

// Radau5 constants ------------------------------------------------------------

const NSTAGE: usize = 3;

const ALPHA: f64 = 2.6810828736277521338957907432111121010270319565630;
const BETA: f64 = 3.0504301992474105694263776247875679044407041991795;
const GAMMA: f64 = 3.6378342527444957322084185135777757979459360868739;
const GAMMA0: f64 = 0.27488882959567736774782860359941477929459341400416;
const EE0: f64 = -2.7623054547485993983499285952820549558040707846130;
const EE1: f64 = 0.37993559825272887786874736408712686858426119657697;
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
    use crate::{Method, Params, Samples};

    #[test]
    fn new_works() {
        let (system, mut data, mut args) = Samples::hairer_wanner_eq1();
        let ndim = system.get_ndim();
        let params = Params::new(Method::Radau5);
        let mut solver = Radau5::new(params.radau5, system);
    }
}
