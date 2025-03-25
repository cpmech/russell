use russell_lab::Vector;
use serde::{Deserialize, Serialize};

/// Holds the current solution of the nonlinear problem
#[derive(Clone, Debug, Deserialize)]
pub struct NlState {
    pub u: Vector,
    pub l: f64,
    pub s: f64,
    pub h: f64,
}

/// Holds the data generated at an accepted step (internal version)
///
/// This an internal version holding a reference to `u` to avoid temporary copies.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct NlStateAccess<'a> {
    pub u: &'a Vector,
    pub l: f64,
    pub s: f64,
    pub h: f64,
}
