use super::{State, CONFIG_H_MIN};
use crate::StrError;
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
    /// Stops when a component of the `u` vector reaches a minimum value.
    ///
    /// Holds `(index, min_value)`.
    MinCompU(usize, f64),

    /// Stops when a component of the `u` vector reaches a maximum value.
    ///
    /// Holds `(index, max_value)`.
    MaxCompU(usize, f64),

    /// Stops when lambda reaches a minimum value.
    MinLambda(f64),

    /// Stops when lambda reaches a maximum value.
    ///
    /// Holds `(target, tolerance)`.
    MaxLambda(f64, f64),

    /// Stops after a number of steps.
    Steps(usize),
}

impl Stop {
    /// Validates the stopping criterion against the initial state.
    pub fn validate(&self, state: &State) -> Result<(), StrError> {
        match self {
            Stop::MinCompU(i, u1) => {
                if *i >= state.u.dim() {
                    return Err("Stop enum error: MinCompU index is out of bounds");
                }
                if *u1 >= state.u[*i] {
                    return Err("Stop enum error: MinCompU value must be less than the initial u value");
                }
            }
            Stop::MaxCompU(i, u1) => {
                if *i >= state.u.dim() {
                    return Err("Stop enum error: MaxCompU index is out of bounds");
                }
                if *u1 <= state.u[*i] {
                    return Err("Stop enum error: MaxCompU value must be greater than the initial u value");
                }
            }
            Stop::MinLambda(l1) => {
                if *l1 >= state.l {
                    return Err("Stop enum error: MinLambda value must be less than the initial lambda value");
                }
            }
            Stop::MaxLambda(l1, tol) => {
                if *l1 <= state.l {
                    return Err("Stop enum error: MaxLambda value must be greater than the initial lambda value");
                }
                if *tol <= CONFIG_H_MIN {
                    return Err("Stop enum error: MaxLambda tolerance must be greater than CONFIG_H_MIN");
                }
            }
            Stop::Steps(n) => {
                if *n < 1 {
                    return Err("Stop enum error: number of steps must be greater than 0");
                }
            }
        }
        Ok(())
    }

    /// Returns the target lambda value, if specified
    ///
    /// Returns `(lambda, is_min)` where `is_min` indicates if it is a minimum or maximum lambda target.
    pub fn lambda(&self) -> Option<(f64, bool)> {
        match self {
            Stop::MinCompU(_, _) => None,
            Stop::MaxCompU(_, _) => None,
            Stop::MinLambda(l1) => Some((*l1, true)),
            Stop::MaxLambda(l1, _) => Some((*l1, false)),
            Stop::Steps(_) => None,
        }
    }

    /// Indicates if the stopping criterion is met at the current step
    pub fn now(&self, step: usize, state: &State) -> bool {
        match self {
            Stop::MinCompU(i, u1) => state.u[*i] < *u1 || f64::abs(state.u[*i] - *u1) < CONFIG_H_MIN,
            Stop::MaxCompU(i, u1) => state.u[*i] > *u1 || f64::abs(*u1 - state.u[*i]) < CONFIG_H_MIN,
            Stop::MinLambda(l1) => state.l < *l1 || f64::abs(*l1 - state.l) < CONFIG_H_MIN,
            Stop::MaxLambda(l1, tol) => state.l > *l1 || f64::abs(state.l - *l1) < *tol,
            Stop::Steps(n) => (step + 1) == *n,
        }
    }

    /// Returns the initial stepsize `h_ini` based on the stopping criterion and the current state
    pub fn h_ini(&self, h_ini_default: f64, state: &State) -> f64 {
        match self {
            Stop::MinCompU(_, _) => h_ini_default,
            Stop::MaxCompU(_, _) => h_ini_default,
            Stop::MinLambda(l1) => f64::min(h_ini_default, f64::abs(state.l - *l1)),
            Stop::MaxLambda(l1, _) => f64::min(h_ini_default, f64::abs(*l1 - state.l)),
            Stop::Steps(_) => h_ini_default,
        }
    }

    /// Returns the equal/fixed stepsize `h_eq` based on the stopping criterion and the current state
    pub fn h_eq(&self, h_eq_default: f64, state: &State) -> f64 {
        match self {
            Stop::MinCompU(_, _) => h_eq_default,
            Stop::MaxCompU(_, _) => h_eq_default,
            Stop::MinLambda(l1) => {
                let n = f64::ceil(f64::abs(state.l - *l1) / h_eq_default) as usize;
                (state.l - *l1) / (n as f64)
            }
            Stop::MaxLambda(l1, _) => {
                let n = f64::ceil(f64::abs(*l1 - state.l) / h_eq_default) as usize;
                (*l1 - state.l) / (n as f64)
            }
            Stop::Steps(_) => h_eq_default,
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
    /// Validates the input data
    pub fn validate(&self) -> Result<(), StrError> {
        match self {
            AutoStep::Yes => Ok(()),
            AutoStep::No(h_eq) => {
                if *h_eq < 10.0 * f64::EPSILON {
                    return Err("AutoStep enum error: fixed stepsize h_eq must be ≥ 10.0 * f64::EPSILON");
                }
                Ok(())
            }
        }
    }

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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stop_validate_works() {
        let mut state = State::new(3);
        state.u[0] = 1.0;
        state.u[1] = 2.0;
        state.u[2] = 3.0;
        state.l = 5.0;

        // MinCompU - valid case
        let stop = Stop::MinCompU(1, 1.0);
        assert!(stop.validate(&state).is_ok());

        // MinCompU - index out of bounds
        let stop = Stop::MinCompU(3, 1.0);
        assert!(stop.validate(&state).is_err());

        // MinCompU - value not less than initial
        let stop = Stop::MinCompU(1, 2.5);
        assert!(stop.validate(&state).is_err());

        // MaxCompU - valid case
        let stop = Stop::MaxCompU(2, 4.0);
        assert!(stop.validate(&state).is_ok());

        // MaxCompU - index out of bounds
        let stop = Stop::MaxCompU(5, 4.0);
        assert!(stop.validate(&state).is_err());

        // MaxCompU - value not greater than initial
        let stop = Stop::MaxCompU(2, 2.0);
        assert!(stop.validate(&state).is_err());

        // MinLambda - valid case
        let stop = Stop::MinLambda(3.0);
        assert!(stop.validate(&state).is_ok());

        // MinLambda - value not less than initial
        let stop = Stop::MinLambda(6.0);
        assert!(stop.validate(&state).is_err());

        // MaxLambda - valid case
        let stop = Stop::MaxLambda(7.0, 1e-5);
        assert!(stop.validate(&state).is_ok());

        // MaxLambda - value not greater than initial
        let stop = Stop::MaxLambda(4.0, 1e-5);
        assert!(stop.validate(&state).is_err());

        // Steps - valid case
        let stop = Stop::Steps(10);
        assert!(stop.validate(&state).is_ok());

        // Steps - invalid case (zero steps)
        let stop = Stop::Steps(0);
        assert!(stop.validate(&state).is_err());
    }

    #[test]
    fn test_stop_lambda_works() {
        // MinCompU - returns None
        let stop = Stop::MinCompU(0, 1.0);
        assert_eq!(stop.lambda(), None);

        // MaxCompU - returns None
        let stop = Stop::MaxCompU(0, 1.0);
        assert_eq!(stop.lambda(), None);

        // MinLambda - returns Some with is_min=true
        let stop = Stop::MinLambda(2.5);
        assert_eq!(stop.lambda(), Some((2.5, true)));

        // MaxLambda - returns Some with is_min=false
        let stop = Stop::MaxLambda(7.5, 1e-5);
        assert_eq!(stop.lambda(), Some((7.5, false)));

        // Steps - returns None
        let stop = Stop::Steps(5);
        assert_eq!(stop.lambda(), None);
    }

    #[test]
    fn test_stop_now_works() {
        let mut state = State::new(2);
        state.u[0] = 1.5;
        state.u[1] = 3.5;
        state.l = 2.0;

        // MinCompU - criterion met
        let stop = Stop::MinCompU(0, 2.0);
        assert!(stop.now(0, &state));

        // MinCompU - criterion not met
        let stop = Stop::MinCompU(0, 1.0);
        assert!(!stop.now(0, &state));

        // MaxCompU - criterion met
        let stop = Stop::MaxCompU(1, 3.0);
        assert!(stop.now(0, &state));

        // MaxCompU - criterion not met
        let stop = Stop::MaxCompU(1, 4.0);
        assert!(!stop.now(0, &state));

        // MinLambda - criterion met
        let stop = Stop::MinLambda(3.0);
        assert!(stop.now(0, &state));

        // MinLambda - criterion not met
        let stop = Stop::MinLambda(1.0);
        assert!(!stop.now(0, &state));

        // MaxLambda - criterion met
        let stop = Stop::MaxLambda(1.5, 1e-5);
        assert!(stop.now(0, &state));

        // MaxLambda - criterion not met
        let stop = Stop::MaxLambda(3.0, 1e-5);
        assert!(!stop.now(0, &state));

        // Steps - criterion met (step+1 == n)
        let stop = Stop::Steps(5);
        assert!(stop.now(4, &state)); // step 4, so step+1 = 5

        // Steps - criterion not met
        let stop = Stop::Steps(5);
        assert!(!stop.now(3, &state)); // step 3, so step+1 = 4
    }

    #[test]
    fn test_stop_h_ini_works() {
        let mut state = State::new(1);
        state.u[0] = 5.0;
        state.l = 10.0;
        let h_default = 2.0;

        // MinCompU - returns default
        let stop = Stop::MinCompU(0, 1.0);
        assert_eq!(stop.h_ini(h_default, &state), h_default);

        // MaxCompU - returns default
        let stop = Stop::MaxCompU(0, 15.0);
        assert_eq!(stop.h_ini(h_default, &state), h_default);

        // MinLambda - returns min(default, abs(current - target))
        let stop = Stop::MinLambda(7.0);
        assert_eq!(stop.h_ini(h_default, &state), 2.0); // min(2.0, abs(10.0 - 7.0)) = min(2.0, 3.0) = 2.0

        let stop = Stop::MinLambda(9.5);
        assert_eq!(stop.h_ini(h_default, &state), 0.5); // min(2.0, abs(10.0 - 9.5)) = min(2.0, 0.5) = 0.5

        // MaxLambda - returns min(default, abs(target - current))
        let stop = Stop::MaxLambda(13.0, 1e-5);
        assert_eq!(stop.h_ini(h_default, &state), 2.0); // min(2.0, abs(13.0 - 10.0)) = min(2.0, 3.0) = 2.0

        let stop = Stop::MaxLambda(10.5, 1e-5);
        assert_eq!(stop.h_ini(h_default, &state), 0.5); // min(2.0, abs(10.5 - 10.0)) = min(2.0, 0.5) = 0.5

        // Steps - returns default
        let stop = Stop::Steps(10);
        assert_eq!(stop.h_ini(h_default, &state), h_default);
    }

    #[test]
    fn test_stop_h_eq_works() {
        let mut state = State::new(1);
        state.u[0] = 5.0;
        state.l = 10.0;
        let h_default = 1.5;

        // MinCompU - returns default
        let stop = Stop::MinCompU(0, 1.0);
        assert_eq!(stop.h_eq(h_default, &state), h_default);

        // MaxCompU - returns default
        let stop = Stop::MaxCompU(0, 15.0);
        assert_eq!(stop.h_eq(h_default, &state), h_default);

        // MinLambda - calculates equal stepsize
        let stop = Stop::MinLambda(7.0);
        // Distance: 10.0 - 7.0 = 3.0
        // n = ceil(3.0 / 1.5) = ceil(2.0) = 2
        // h_eq = (10.0 - 7.0) / 2 = 1.5
        assert_eq!(stop.h_eq(h_default, &state), 1.5);

        let stop = Stop::MinLambda(8.0);
        // Distance: 10.0 - 8.0 = 2.0
        // n = ceil(2.0 / 1.5) = ceil(1.33) = 2
        // h_eq = (10.0 - 8.0) / 2 = 1.0
        assert_eq!(stop.h_eq(h_default, &state), 1.0);

        // MaxLambda - calculates equal stepsize
        let stop = Stop::MaxLambda(13.0, 1e-5);
        // Distance: 13.0 - 10.0 = 3.0
        // n = ceil(3.0 / 1.5) = ceil(2.0) = 2
        // h_eq = (13.0 - 10.0) / 2 = 1.5
        assert_eq!(stop.h_eq(h_default, &state), 1.5);

        let stop = Stop::MaxLambda(12.0, 1e-5);
        // Distance: 12.0 - 10.0 = 2.0
        // n = ceil(2.0 / 1.5) = ceil(1.33) = 2
        // h_eq = (12.0 - 10.0) / 2 = 1.0
        assert_eq!(stop.h_eq(h_default, &state), 1.0);

        // Steps - returns default
        let stop = Stop::Steps(5);
        assert_eq!(stop.h_eq(h_default, &state), h_default);
    }
}
