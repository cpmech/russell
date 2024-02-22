use crate::{OdeSolverTrait, StrError};
use russell_lab::{vec_max_abs_diff, Vector};
use std::collections::HashMap;

/// Holds the (x,y) results at accepted steps or interpolated within a "dense" sequence
pub struct Output<'a> {
    /// Indicates whether the accepted step output is to be performed or not
    save_step: bool,

    /// Holds the stepsize output during accepted steps
    pub step_h: Vec<f64>,

    /// Holds the x values output during accepted steps
    pub step_x: Vec<f64>,

    /// Holds the selected y components output during accepted steps
    pub step_y: HashMap<usize, Vec<f64>>,

    /// Holds the global error output during accepted steps (if the analytical solution is available)
    ///
    /// The global error is the maximum absolute difference between `y_numerical` and `y_analytical` (see [russell_lab::vec_max_abs_diff])
    pub step_global_error: Vec<f64>,

    /// Holds the stepsize to perform the dense output (None means disabled)
    dense_h: Option<f64>,

    /// Holds the indices of the accepted steps that were used to compute the dense output
    pub dense_step_index: Vec<usize>,

    /// Holds the x values requested by the dense output
    pub dense_x: Vec<f64>,

    /// Holds the selected y components requested by the dense output
    pub dense_y: HashMap<usize, Vec<f64>>,

    /// Holds an auxiliary y vector (e.g., to compute the analytical solution or the dense output)
    y_aux: Vector,

    /// Implements the analytical solution `y(x)` function
    y_analytical: Option<Box<dyn 'a + FnMut(&mut Vector, f64)>>,
}

impl<'a> Output<'a> {
    /// Allocates a new instance
    pub fn new() -> Self {
        const EMPTY: usize = 0;
        Output {
            save_step: false,
            step_h: Vec::new(),
            step_x: Vec::new(),
            step_y: HashMap::new(),
            step_global_error: Vec::new(),
            dense_h: None,
            dense_step_index: Vec::new(),
            dense_x: Vec::new(),
            dense_y: HashMap::new(),
            y_aux: Vector::new(EMPTY),
            y_analytical: None,
        }
    }

    /// Enables saving the results at accepted steps
    ///
    /// # Input
    ///
    /// * `selected_y_components` -- Specifies which components of the `y` vector are to be saved
    ///
    /// # Results
    ///
    /// * The results will be saved in the `step_h`, `step_x`, and `step_y` arrays
    /// * If the analytical solution is provided, the global error will be saved in the `step_global_error` array
    /// * The global error is the maximum absolute difference between `y_numerical` and `y_analytical` (see [russell_lab::vec_max_abs_diff])
    pub fn enable_step(&mut self, selected_y_components: &[usize]) -> &mut Self {
        self.save_step = true;
        for m in selected_y_components {
            self.step_y.insert(*m, Vec::new());
        }
        self
    }

    /// Disables saving the results at accepted steps
    pub fn disable_step(&mut self) -> &mut Self {
        self.save_step = false;
        self
    }

    /// Enables saving the results at a predefined "dense" sequence of steps
    ///
    /// # Input
    ///
    /// * `h_out` -- is the stepsize (possibly different than the actual `h` stepsize) for the equally spaced "dense" results
    /// * `selected_y_components` -- Specifies which components of the `y` vector are to be saved
    ///
    /// # Results
    ///
    /// * The results will be saved in the `dense_x` and `dense_y` arrays
    /// * The indices of the associated accepted step will be saved in the `dense_step_index` array
    pub fn enable_dense(&mut self, h_out: f64, selected_y_components: &[usize]) -> Result<&mut Self, StrError> {
        if h_out < 0.0 {
            return Err("h_out must be positive");
        }
        self.dense_h = Some(h_out);
        for m in selected_y_components {
            self.dense_y.insert(*m, Vec::new());
        }
        Ok(self)
    }

    /// Disables saving the results at a predefined "dense" sequence of steps
    pub fn disable_dense(&mut self) -> &mut Self {
        self.dense_h = None;
        self
    }

    /// Indicates whether dense output is enabled or not
    pub(crate) fn with_dense_output(&self) -> bool {
        self.dense_h.is_some()
    }

    /// Sets the analytical solution to compute the global error
    ///
    /// The results will be saved in the `step_global_error`
    ///
    /// # Input
    ///
    /// * `analytical` -- is a function(y, x) that computes y(x)
    pub fn set_analytical<F>(&mut self, y_analytical: F) -> &mut Self
    where
        F: 'a + FnMut(&mut Vector, f64),
    {
        self.y_analytical = Some(Box::new(y_analytical));
        self
    }

    /// Clears all resulting arrays
    pub fn clear(&mut self) {
        self.step_h.clear();
        self.step_x.clear();
        for ym in self.step_y.values_mut() {
            ym.clear();
        }
        self.step_global_error.clear();
        self.dense_step_index.clear();
        self.dense_x.clear();
        self.dense_y.clear();
    }

    /// Appends the results after an accepted step is computed
    pub(crate) fn push<A>(
        &mut self,
        accepted_step_index: usize,
        x: f64,
        y: &Vector,
        h: f64,
        solver: &Box<dyn OdeSolverTrait<A> + 'a>,
    ) -> Result<(), StrError> {
        // step output
        if self.save_step {
            self.step_h.push(h);
            self.step_x.push(x);
            for (m, ym) in self.step_y.iter_mut() {
                ym.push(y[*m]);
            }
            // global error
            if let Some(ana) = self.y_analytical.as_mut() {
                let ndim = y.dim();
                if self.y_aux.dim() != ndim {
                    self.y_aux = Vector::new(ndim); // first allocation
                }
                ana(&mut self.y_aux, x);
                let (_, err) = vec_max_abs_diff(y, &self.y_aux).unwrap();
                self.step_global_error.push(err);
            }
        }
        // dense output
        if let Some(h_out) = self.dense_h {
            if accepted_step_index == 0 {
                // first output
                self.dense_step_index.push(accepted_step_index);
                self.dense_x.push(x);
                for (m, ym) in self.dense_y.iter_mut() {
                    ym.push(y[*m]);
                }
            } else {
                // subsequent output
                let ndim = y.dim();
                if self.y_aux.dim() != ndim {
                    self.y_aux = Vector::new(ndim); // first allocation
                }
                let mut x_out = self.dense_x.last().unwrap() + h_out;
                while x_out < x {
                    self.dense_step_index.push(accepted_step_index);
                    self.dense_x.push(x_out);
                    solver.dense_output(&mut self.y_aux, x_out, x, y, h)?;
                    for (m, ym) in self.dense_y.iter_mut() {
                        ym.push(self.y_aux[*m]);
                    }
                    x_out += h_out;
                }
            }
        }
        Ok(())
    }
}
