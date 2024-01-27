use crate::StrError;
use crate::{LinearSystem, NumSolver, OdeParams};
use russell_lab::{vec_copy, Vector};

pub(crate) struct EulerBackward<'a, F>
where
    F: FnMut(&mut Vector, f64, &Vector) -> Result<(), StrError>,
{
    /// Holds the parameters
    params: &'a OdeParams,

    /// Dimension of the ODE system
    ndim: usize,

    /// ODE system
    system: F,

    /// Vector holding the function evaluation
    ///
    /// k := f(x_new, y_new)
    k: Vector,

    /// Residual vector
    r: Vector,

    /// Auxiliary workspace (will contain y to be used in accept_update)
    w: Vector,

    /// Scaling vector
    ///
    /// ```text
    /// scaling[i] = abs_tol + rel_tol ⋅ |y[i]|
    /// ```
    scaling: Vector,

    /// Indicates that the first step is being computed
    first_step: bool,

    /// Number of calls to function
    n_function_eval: usize,

    /// Last number of iterations
    n_iterations_last: usize,

    /// Max number of iterations among all steps
    n_iterations_max: usize,

    /// Linear system
    lin_sys: LinearSystem<'a>,
}

impl<'a, F> EulerBackward<'a, F>
where
    F: FnMut(&mut Vector, f64, &Vector) -> Result<(), StrError>,
{
    pub fn new(params: &'a OdeParams, ndim: usize, system: F) -> Self {
        EulerBackward {
            params,
            ndim,
            system,
            k: Vector::new(ndim),
            r: Vector::new(ndim),
            w: Vector::new(ndim),
            scaling: Vector::new(ndim),
            first_step: true,
            n_function_eval: 0,
            n_iterations_last: 0,
            n_iterations_max: 0,
            lin_sys: LinearSystem::new(params, ndim),
        }
    }
}

impl<'a, F> NumSolver for EulerBackward<'a, F>
where
    F: FnMut(&mut Vector, f64, &Vector) -> Result<(), StrError>,
{
    /// Initializes the internal variables
    fn initialize(&mut self, _x: f64, y: &Vector) {
        // reset variables
        self.first_step = true;
        self.n_function_eval = 0;

        // first scaling vector
        for i in 0..self.ndim {
            self.scaling[i] = self.params.abs_tol + self.params.rel_tol * f64::abs(y[i]);
        }
    }

    /// Calculates the quantities required to update x and y
    ///
    /// Returns the (`relative_error`, `stiffness_ratio`)
    fn step(&mut self, x: f64, y: &Vector, h: f64) -> Result<(f64, f64), StrError> {
        // reset stat variables
        self.n_iterations_last = 0;
        let traditional_newton = !self.params.CteTg;

        // trial update
        let x_new = x + h;
        let y_new = &mut self.w;
        vec_copy(y_new, &y).unwrap();

        // perform iterations
        for _ in 0..self.params.NmaxIt {
            // update counter
            self.n_iterations_last += 1;

            // calculate k_new
            self.n_function_eval += 1;
            (self.system)(&mut self.k, x_new, y_new)?; // k := f(x_new, y_new)

            // calculate the residual and its norm
            let mut r_norm = 0.0;
            for i in 0..self.ndim {
                self.r[i] = y_new[i] - y[i] - h * self.k[i];
                if self.params.UseRmsNorm {
                    r_norm += f64::powf(self.r[i] / self.scaling[i], 2.0);
                } else {
                    r_norm += self.r[i] * self.r[i];
                }
            }
            if self.params.UseRmsNorm {
                r_norm = f64::sqrt(r_norm / (self.ndim as f64));
            } else {
                r_norm = f64::sqrt(r_norm);
            }

            // check convergence
            if r_norm < self.params.fnewt {
                break;
            }
        }

        // compute K matrix (augmented Jacobian)
        if traditional_newton || self.first_step {
            // TODO
        }

        // solve the linear system
        // TODO

        // compute stat variables
        if self.n_iterations_last > self.n_iterations_max {
            self.n_iterations_max = self.n_iterations_last;
        }

        // update variables
        self.first_step = false;
        Ok((0.0, 0.0))
    }

    fn accept(&mut self, y0: &mut Vector, _: f64, _: f64, _: f64, _: f64) -> Result<f64, StrError> {
        vec_copy(y0, &self.w).unwrap();
        Ok(0.0)
    }

    fn reject(&mut self, _: f64, _: f64) -> f64 {
        0.0
    }

    fn dense_output(&self, _: &mut Vector, _: f64, _: f64, _: f64) {}
}
