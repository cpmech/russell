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
}

/// Specifies the stopping criterion for the continuation process.
#[derive(Clone, Copy, Debug)]
pub enum Stop {
    /// Stops when lambda reaches the specified value.
    Lambda(f64),

    /// Stops after a number of steps.
    Steps(usize),

    /// Stops when a component of the `u` vector reaches a maximum value.
    ///
    /// Holds `(index, max_value)`.
    Component(usize, f64),
}

impl Stop {
    /// Returns the target lambda value, if specified
    pub fn lambda_target(&self) -> Option<f64> {
        match self {
            Stop::Lambda(l1) => Some(*l1),
            Stop::Steps(_) => None,
            Stop::Component(_, _) => None,
        }
    }
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

/// Specifies the problem classes in Soderlind (2003) that can be used to define the stepsize control parameters
///
/// Reference:
/// * Soderlind (2003) Digital filters in adaptive time-stepping,
///   ACM Transactions on Mathematical Software, 29(1), 1-26.
#[derive(Clone, Copy, Debug)]
pub enum SoderlindClass {
    /// Smooth to medium problem type
    Ho211,

    /// Medium to non-smooth problem type (holds parameter `b`)
    H211b(f64),

    /// Medium to non-smooth problem type
    H211PI,

    /// Medium
    Ho312,

    /// Non-smooth (holds parameter `b`)
    H312b(f64),

    /// Non-smooth
    H312PID,

    /// Smooth
    Ho321,

    /// Medium
    H321,
}

impl SoderlindClass {
    /// Returns the parameters (beta1, beta2, beta3, alpha2, alpha3) for the selected class
    ///
    /// From Table III on page 24 of Soderlind (2003).
    ///
    /// The parameter `b` is a user-defined parameter that can be adjusted and is used with the
    /// `H211b`, `H312b` classes.
    ///
    /// Reference:
    /// * Soderlind (2003) Digital filters in adaptive time-stepping,
    ///   ACM Transactions on Mathematical Software, 29(1), 1-26.
    pub fn params(&self) -> (f64, f64, f64, f64, f64) {
        match self {
            SoderlindClass::Ho211 => (1.0 / 2.0, 1.0 / 2.0, 0.0, 1.0 / 2.0, 0.0),
            SoderlindClass::H211b(b) => (1.0 / b, 1.0 / b, 0.0, 1.0 / b, 0.0),
            SoderlindClass::H211PI => (1.0 / 6.0, 1.0 / 6.0, 0.0, 0.0, 0.0),
            SoderlindClass::Ho312 => (1.0 / 4.0, 1.0 / 2.0, 1.0 / 4.0, 3.0 / 4.0, 1.0 / 4.0),
            SoderlindClass::H312b(b) => (1.0 / b, 2.0 / b, 1.0 / b, 3.0 / b, 1.0 / b),
            SoderlindClass::H312PID => (1.0 / 18.0, 1.0 / 9.0, 1.0 / 18.0, 0.0, 0.0),
            SoderlindClass::Ho321 => (5.0 / 4.0, 1.0 / 2.0, -3.0 / 4.0, -1.0 / 4.0, -3.0 / 4.0),
            SoderlindClass::H321 => (1.0 / 3.0, 1.0 / 18.0, -5.0 / 18.0, -5.0 / 6.0, -1.0 / 6.0),
        }
    }
}
