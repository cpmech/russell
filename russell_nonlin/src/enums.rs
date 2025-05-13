use serde::{Deserialize, Serialize};

/// Defines the initial direction (e.g., tangent vector) for the pseudo-arclength method.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub enum Direction {
    /// Use the positive sign of `dλ/ds₀` (follows the positive direction on the branch).
    ///
    /// This requires the Jacobian matrix `Gu₀ = ∂G/∂u @ (u₀,λ₀)` to be non-singular.
    Pos,

    /// Use the negative sign of dλds₀0 (follows the negative direction on the branch).
    ///
    /// This requires the Jacobian matrix `Gu₀ = ∂G/∂u @ (u₀,λ₀)` to be non-singular.
    Neg,

    /// Use a given (previous) tangent vector specified in the State object.
    Prev,
}

/// Specifies the stopping criterion for the continuation process.
#[derive(Clone, Copy, Debug)]
pub enum Stop {
    /// Stops when lambda reaches the specified value.
    Lambda(f64),

    /// Stops after a number of steps.
    Steps(usize),
}

/// Specifies the stepsize control method
#[derive(Clone, Copy, Debug)]
pub enum AutoStep {
    /// Automatic stepping with variable stepsizes
    Yes,

    /// Fixed stepsize (h is given)
    No(f64),
}

impl AutoStep {
    /// Indicates variable stepsize control.
    pub fn yes(&self) -> bool {
        match self {
            AutoStep::Yes => true,
            AutoStep::No(_) => false,
        }
    }

    /// Indicates fixed/equal stepsize.
    pub fn no(&self) -> bool {
        match self {
            AutoStep::Yes => false,
            AutoStep::No(_) => true,
        }
    }
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

/// Specifies the status of the continuation process.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Status {
    /// The continuation process was successful.
    Success,

    /// The continuation process failed.
    Failure,

    /// The continuation process was stopped by the user.
    Stopped,
}
