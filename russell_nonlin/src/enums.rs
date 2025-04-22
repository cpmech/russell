use serde::{Deserialize, Serialize};

/// Specifies the stopping criterion for the continuation process.
#[derive(Clone, Copy, Debug)]
pub enum Stop {
    /// Stops when lambda reaches the specified value.
    Lambda(f64),

    /// Stops after a number of steps.
    Steps(usize),
}

/// Defines the initial tangent vector (duds0, dλds0) for the pseudo-arclength method.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub enum TgVec {
    /// Calculated using the positive sign of dλds0 (follows the positive direction on the branch).
    ///
    /// This requires the Jacobian matrix Gu0 = ∂G/∂u @ (u0,λ0) to be non-singular.
    Positive,

    /// Calculated using the negative sign of dλds0 (follows the negative direction on the branch).
    ///
    /// This requires the Jacobian matrix Gu0 = ∂G/∂u @ (u0,λ0) to be non-singular.
    Negative,

    /// Use a given (previous) tangent vector
    Given,
}

/// Specifies the method of continuation to be used.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Method {
    /// Pseudo-arclength continuation
    Arclength,

    /// Natural parameter continuation
    Natural,
}

impl Method {
    pub fn description(&self) -> &'static str {
        match self {
            Method::Arclength => "Pseudo-arclength continuation; solves G(u(s), λ(s)) = 0",
            Method::Natural => "Natural parameter continuation; solves G(u, λ) = 0",
        }
    }
}
