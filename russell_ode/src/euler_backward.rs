use crate::StrError;
use crate::{NumSolver, OdeParams, OdeSystem, Workspace};
use russell_lab::{vec_copy, vec_update, Vector};
use russell_sparse::{CooMatrix, Genie, LinSolver, SparseMatrix};
use std::marker::PhantomData;

/// Implements the backward Euler (implicit) solver
pub(crate) struct EulerBackward<'a, F, J, A>
where
    F: FnMut(&mut Vector, f64, &Vector, &mut A) -> Result<(), StrError>,
    J: FnMut(&mut CooMatrix, f64, &Vector, f64, &mut A) -> Result<(), StrError>,
{
    /// Holds the parameters
    params: &'a OdeParams,

    /// ODE system
    system: OdeSystem<'a, F, J, A>,

    /// Scaling vector
    ///
    /// ```text
    /// scaling[i] = abs_tol + rel_tol ⋅ |y[i]|
    /// ```
    scaling: Vector,

    /// Vector holding the function evaluation
    ///
    /// k := f(x_new, y_new)
    k: Vector,

    /// Auxiliary workspace (will contain y to be used in accept_update)
    w: Vector,

    /// Residual vector (right-hand side vector)
    r: Vector,

    /// Unknowns vector (the solution of the linear system)
    dy: Vector,

    /// Coefficient matrix K = h J - I
    kk: SparseMatrix,

    /// Linear solver
    solver: LinSolver<'a>,

    /// Handle generic argument
    phantom: PhantomData<A>,
}

impl<'a, F, J, A> EulerBackward<'a, F, J, A>
where
    F: FnMut(&mut Vector, f64, &Vector, &mut A) -> Result<(), StrError>,
    J: FnMut(&mut CooMatrix, f64, &Vector, f64, &mut A) -> Result<(), StrError>,
{
    /// Allocates a new instance
    pub fn new(params: &'a OdeParams, system: OdeSystem<'a, F, J, A>) -> Self {
        let ndim = system.ndim;
        let nnz = system.jac_nnz + ndim; // +ndim corresponds to the diagonal I matrix
        let symmetry = system.jac_symmetry;
        let one_based = if params.genie == Genie::Mumps { true } else { false };
        EulerBackward {
            params,
            system,
            scaling: Vector::new(ndim),
            k: Vector::new(ndim),
            w: Vector::new(ndim),
            r: Vector::new(ndim),
            dy: Vector::new(ndim),
            kk: SparseMatrix::new_coo(ndim, ndim, nnz, symmetry, one_based).unwrap(),
            solver: LinSolver::new(params.genie).unwrap(),
            phantom: PhantomData,
        }
    }
}

impl<'a, F, J, A> NumSolver<A> for EulerBackward<'a, F, J, A>
where
    F: FnMut(&mut Vector, f64, &Vector, &mut A) -> Result<(), StrError>,
    J: FnMut(&mut CooMatrix, f64, &Vector, f64, &mut A) -> Result<(), StrError>,
{
    /// Initializes the internal variables
    fn initialize(&mut self, _x: f64, y: &Vector) {
        // first scaling vector
        for i in 0..self.system.ndim {
            self.scaling[i] = self.params.abs_tol + self.params.rel_tol * f64::abs(y[i]);
        }
    }

    /// Calculates the quantities required to update x and y
    fn step(&mut self, work: &mut Workspace, x: f64, y: &Vector, h: f64, args: &mut A) -> Result<(), StrError> {
        // auxiliary
        let traditional_newton = !self.params.use_modified_newton;
        let ndim = self.system.ndim;
        let dim = ndim as f64;

        // trial update
        let x_new = x + h;
        let y_new = &mut self.w;
        vec_copy(y_new, &y).unwrap();

        // perform iterations
        let mut converged = false;
        work.bench.n_iterations_last = 0;
        for _ in 0..self.params.n_iteration_max {
            // benchmark
            work.bench.n_iterations_last += 1;

            // calculate k_new
            work.bench.n_function_eval += 1;
            (self.system.function)(&mut self.k, x_new, y_new, args)?; // k := f(x_new, y_new)

            // calculate the residual and its norm
            let mut r_norm = 0.0;
            for i in 0..ndim {
                self.r[i] = y_new[i] - y[i] - h * self.k[i];
                if self.params.use_rms_norm {
                    r_norm += f64::powf(self.r[i] / self.scaling[i], 2.0);
                } else {
                    r_norm += self.r[i] * self.r[i];
                }
            }
            if self.params.use_rms_norm {
                r_norm = f64::sqrt(r_norm / dim);
            } else {
                r_norm = f64::sqrt(r_norm);
            }

            // check convergence
            if r_norm < self.params.tol_newton {
                converged = true;
                break;
            }

            // compute K matrix (augmented Jacobian)
            if traditional_newton || work.first_step {
                // benchmark
                work.bench.sw_jacobian.reset();
                work.bench.n_jacobian_eval += 1;

                // calculate J_new := h J
                let kk = self.kk.get_coo_mut()?;
                if self.params.use_numerical_jacobian || !self.system.jac_available {
                    work.bench.n_function_eval += ndim;
                    self.system.numerical_jacobian(kk, x_new, y_new, &self.k, h, args)?;
                } else {
                    (self.system.jacobian)(kk, x_new, y_new, h, args)?;
                }

                // add diagonal entries => calculate K = h J_new - I
                for i in 0..self.system.ndim {
                    kk.put(i, i, -1.0).unwrap();
                }

                // benchmark
                work.bench.stop_sw_jacobian();

                // perform factorization
                work.bench.sw_factor.reset();
                self.solver.actual.factorize(&mut self.kk, self.params.lin_sol_params)?;
                work.bench.stop_sw_factor();
            }

            // solve the linear system
            work.bench.sw_lin_sol.reset();
            self.solver.actual.solve(&mut self.dy, &self.kk, &self.r, false)?;
            work.bench.stop_sw_lin_sol();

            // update y
            vec_update(y_new, 1.0, &self.dy).unwrap(); // y := y + δy
        }

        // check
        if !converged {
            return Err("Newton-Raphson method did not converge");
        }

        // done
        work.bench.update_n_iterations_max();
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
        *x += h;
        vec_copy(y, &self.w).unwrap();
        Ok(())
    }

    /// Rejects the update
    fn reject(&mut self, _work: &mut Workspace, _h: f64) {}

    /// Computes the dense output
    fn dense_output(&self, _y_out: &mut Vector, _h: f64, _x: f64, _x_out: f64) {}
}
