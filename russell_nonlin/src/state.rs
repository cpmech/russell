use russell_lab::Vector;
use serde::{Deserialize, Serialize};

/// Holds the current solution of the nonlinear problem
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct State {
    /// Primary unknown values
    pub u: Vector,

    /// λ parameter
    pub l: f64,

    /// Arclength
    pub s: f64,

    /// Stepsize: either Δs (arclength) or Δλ (natural parameter)
    pub h: f64,

    /// Part of the tangent vector (duds,dλds) for the pseudo-arclength method
    ///
    /// **Note**: this vector is only allocated for the pseudo-arclength method
    ///
    /// (ndim)
    pub duds: Vector,

    /// Part of the tangent vector (duds,dλds) for the pseudo-arclength method
    pub dlds: f64,
}

impl State {
    /// Creates a new instance with zero values
    ///
    /// # Input
    ///
    /// * `ndim` -- number of dimensions (must match the system's ndim)
    /// * `with_tangent_vector` -- with tangent vector; this is required for the pseudo-arclength method
    pub fn new(ndim: usize, with_tangent_vector: bool) -> Self {
        let ndim_duds = if with_tangent_vector { ndim } else { 0 };
        State {
            u: Vector::new(ndim),
            l: 0.0,
            s: 0.0,
            h: 0.0,
            duds: Vector::new(ndim_duds),
            dlds: 0.0,
        }
    }
}
