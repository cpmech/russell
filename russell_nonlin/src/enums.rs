use super::CONFIG_H_MIN;
use crate::StrError;
use russell_lab::{Norm, Vector, vec_norm_chunk};
use serde::{Deserialize, Serialize};

/// Defines the type of the relative difference used in the stepsize control using the tangent vector
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RdiffType {
    /// Average relative difference
    ///
    /// ```text
    /// dx/ds = {du/ds..., dλ/ds}
    ///
    ///        | (dx/ds₁)[i] - (dx/ds₀)[i] |
    /// p[i] = —————————————————————————————
    ///              | (dx/ds₀)[i] |
    ///
    /// q = finite(p)
    ///
    /// rdiff = average(q)
    /// ```
    Ave,

    /// Maximum relative difference
    ///
    /// ```text
    /// dx/ds = {du/ds..., dλ/ds}
    ///
    ///        | (dx/ds₁)[i] - (dx/ds₀)[i] |
    /// p[i] = —————————————————————————————
    ///              | (dx/ds₀)[i] |
    ///
    /// q = finite(p)
    ///
    /// rdiff = maximum(q)
    /// ```
    Max,
}

/// Defines the initial direction of the tangent vector for the pseudo-arclength method
/// or the (constant) sign of Δλ for the Natural method.
///
/// For the pseudo-arclength method, the initial tangent vector `(du/ds₀, dλ/ds₀)` is
/// computed from the Jacobian at `(u₀, λ₀)`; the sign of `dλ/ds₀` is controlled by this enum.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub enum IniDir {
    /// Selects the positive value for dλ/ds₀ or Δλ
    ///
    /// For the Natural solver, this option follows an increase on λ. Otherwise,
    /// for the pseudo-arclength solver, it calculates the initial tangent vector by
    /// using a positive dλ/ds₀ as explained below.
    ///
    /// For the pseudo-arclength solver, the tangent is given by `(du/ds₀, dλ/ds₀)`
    /// and this option requires the initial Jacobian matrix `Gu₀ = ∂G/∂u @ (u₀,λ₀)`
    /// to be **non-singular**.
    ///
    /// Steps to calculate the initial tangent vector (pseudo-arclength only):
    ///
    /// ```text
    /// Solve:     Gu₀ z = -Gλ₀
    /// Calculate: dλ/ds₀ = 1 / √(1 + zᵀ z)
    /// Calculate: du/ds₀ = dλ/ds₀ z
    /// ```
    Pos,

    /// Selects the negative value for dλ/ds₀ or Δλ
    ///
    /// For the Natural solver, this option follows a decrease on λ. Otherwise,
    /// for the pseudo-arclength solver, it calculates the initial tangent vector by
    /// using a negative dλ/ds₀ as explained below.
    ///
    /// For the pseudo-arclength solver, the tangent is given by `(du/ds₀, dλ/ds₀)`
    /// and this option requires the initial Jacobian matrix `Gu₀ = ∂G/∂u @ (u₀,λ₀)`
    /// to be **non-singular**.
    ///
    /// Steps to calculate the initial tangent vector (pseudo-arclength only):
    ///
    /// ```text
    /// Solve:     Gu₀ z = -Gλ₀
    /// Calculate: dλ/ds₀ = -1 / √(1 + zᵀ z)
    /// Calculate: du/ds₀ = dλ/ds₀ z
    /// ```
    Neg,
}

/// Specifies the stopping criterion for the continuation process.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Stop {
    /// Stops when a component of the `u` vector reaches a minimum value.
    ///
    /// Holds `(index, min_value)`.
    MinCompU(usize, f64),

    /// Stops when a component of the `u` vector reaches a maximum value.
    ///
    /// Holds `(index, max_value)`.
    MaxCompU(usize, f64),

    /// Stops when the norm of the `u[begin..end]` values reaches a maximum value.
    ///
    /// Holds `(max_value, norm_type, begin, end)`.
    ///
    /// Note that `end` is exclusive, i.e., the slice goes up to `end - 1`.
    ///
    /// Requirements: `begin` must be < `end` and `end` must be ≤ `u.dim()`.
    MaxNormU(f64, Norm, usize, usize),

    /// Stops when lambda reaches a minimum value.
    MinLambda(f64),

    /// Stops when lambda reaches a maximum value.
    MaxLambda(f64),

    /// Stops after a number of steps.
    Steps(usize),
}

impl Stop {
    /// Validates the stopping criterion against the initial state.
    pub(crate) fn validate(&self, u: &Vector, l: f64) -> Result<(), StrError> {
        match self {
            Stop::MinCompU(i, u1) => {
                if *i >= u.dim() {
                    return Err("Stop enum error: MinCompU index is out of bounds");
                }
                if *u1 >= u[*i] {
                    return Err("Stop enum error: MinCompU value must be less than the initial u value");
                }
            }
            Stop::MaxCompU(i, u1) => {
                if *i >= u.dim() {
                    return Err("Stop enum error: MaxCompU index is out of bounds");
                }
                if *u1 <= u[*i] {
                    return Err("Stop enum error: MaxCompU value must be greater than the initial u value");
                }
            }
            Stop::MaxNormU(norm_u1, _, begin, end) => {
                if *begin >= *end {
                    return Err("Stop enum error: MaxNormU: begin must be < end");
                }
                if *end > u.dim() {
                    return Err("Stop enum error: MaxNormU: end must be ≤ u.dim");
                }
                if *norm_u1 <= 0.0 {
                    return Err("Stop enum error: MaxNormU value must be greater than 0.0");
                }
            }
            Stop::MinLambda(l1) => {
                if *l1 >= l {
                    return Err("Stop enum error: MinLambda value must be less than the initial lambda value");
                }
            }
            Stop::MaxLambda(l1) => {
                if *l1 <= l {
                    return Err("Stop enum error: MaxLambda value must be greater than the initial lambda value");
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
    pub(crate) fn lambda(&self) -> Option<(f64, bool)> {
        match self {
            Stop::MinCompU(_, _) => None,
            Stop::MaxCompU(_, _) => None,
            Stop::MaxNormU(_, _, _, _) => None,
            Stop::MinLambda(l1) => Some((*l1, true)),
            Stop::MaxLambda(l1) => Some((*l1, false)),
            Stop::Steps(_) => None,
        }
    }

    /// Returns the target u-component value, if specified
    ///
    /// Returns `(i, uᵢ, is_min)` where `is_min` indicates if it is a minimum or maximum uᵢ target.
    pub(crate) fn u_comp(&self) -> Option<(usize, f64, bool)> {
        match self {
            Stop::MinCompU(i, u1) => Some((*i, *u1, true)),
            Stop::MaxCompU(i, u1) => Some((*i, *u1, false)),
            Stop::MaxNormU(_, _, _, _) => None,
            Stop::MinLambda(_) => None,
            Stop::MaxLambda(_) => None,
            Stop::Steps(_) => None,
        }
    }

    /// Indicates if the stopping criterion is met at the current step
    pub(crate) fn now(&self, step: usize, u: &Vector, l: f64) -> bool {
        match self {
            Stop::MinCompU(i, u1) => u[*i] < *u1 || f64::abs(u[*i] - *u1) < CONFIG_H_MIN,
            Stop::MaxCompU(i, u1) => u[*i] > *u1 || f64::abs(*u1 - u[*i]) < CONFIG_H_MIN,
            Stop::MaxNormU(norm_u1, norm_type, begin, end) => {
                let norm_u = vec_norm_chunk(&u, *norm_type, *begin, *end);
                norm_u > *norm_u1 || f64::abs(norm_u - *norm_u1) < CONFIG_H_MIN
            }
            Stop::MinLambda(l1) => l < *l1 || f64::abs(*l1 - l) < CONFIG_H_MIN,
            Stop::MaxLambda(l1) => l > *l1 || f64::abs(l - *l1) < CONFIG_H_MIN,
            Stop::Steps(n) => (step + 1) == *n,
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
    /// Returns the name of the method
    pub fn name(&self) -> &'static str {
        match self {
            Method::Arclength => "Arclength",
            Method::Natural => "Natural",
        }
    }

    /// Returns a brief description of the method
    pub fn description(&self) -> &'static str {
        match self {
            Method::Arclength => "Pseudo-arclength continuation; solves G(u(s), λ(s)) = 0",
            Method::Natural => "Natural parameter continuation; solves G(u, λ) = 0",
        }
    }
}

/// Specifies the problem classes in Soderlind (2003) that can be used to define the stepsize control parameters
///
/// Reference:
/// * Soderlind (2003) Digital filters in adaptive time-stepping,
///   ACM Transactions on Mathematical Software, 29(1), 1-26.
#[derive(Clone, Copy, Debug, PartialEq)]
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
    ///
    /// # Returns
    ///
    /// Returns `(beta1, beta2, beta3, alpha2, alpha3)` — the PID controller
    /// parameters for stepsize adaptation.
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

/// Specifies Success or Failure
///
/// Holds the type of failure encountered during the continuation process
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Status {
    /// Failure: The denominator to calculate δλ in the bordering algorithm is too small
    BorderingSmallDenominator,

    /// Failure: Newton-Raphson iteration yielded a large (‖δu‖∞,|δλ|)
    ///
    /// (may try again)
    LargeDelta,

    /// Failure: Newton-Raphson iteration did not converge within the maximum number of iterations
    ///
    /// (may try again)
    ReachedMaxIterations,

    /// Failure: Newton-Raphson iteration detected continued residual divergence over one step
    ///
    /// (may try again)
    ContinuedResidualDivergence,

    /// Failure: Newton-Raphson iteration detected continued delta divergence over one step
    ///
    /// (may try again)
    ContinuedDeltaDivergence,

    /// Failure: The step is rejected
    ///
    /// (may try again)
    Rejection,

    /// Failure: The secondary update failed
    ///
    /// (may try again)
    SecondaryUpdateError(StrError),

    /// Failure: The stop criterion was not met
    ///
    /// (may try again)
    UnmetStopCriterion,

    /// Failure: The stepsize became too small.
    ///
    /// (must stop)
    SmallStepsize,

    /// The secondary update requested termination
    ///
    /// (must stop)
    SecondaryUpdateTerminate,

    /// Failure: Detected continued failure, after multiple tries
    ///
    /// (must stop)
    ContinuedFailure,

    /// Failure: Detected continued rejection over multiple tries
    ///
    /// (must stop)
    ContinuedRejection,

    /// Failure: Found NaN or Inf in the residual vector during iteration
    ///
    /// (must stop)
    NanOrInfResidual,

    /// Failure: Found NaN or Inf in the correction vector during iteration
    ///
    /// (must stop)
    NanOrInfDelta,

    /// No failure has occurred
    Success,
}

impl Status {
    /// Indicates success (no failure has occurred)
    pub fn success(&self) -> bool {
        *self == Status::Success
    }

    /// Indicates that a failure has occurred
    pub fn failure(&self) -> bool {
        *self != Status::Success
    }

    /// Indicates whether we can try again by reducing the stepsize
    pub(crate) fn try_again(&self) -> bool {
        match self {
            // may try again
            Status::BorderingSmallDenominator => true,
            Status::LargeDelta => true,
            Status::ReachedMaxIterations => true,
            Status::ContinuedResidualDivergence => true,
            Status::ContinuedDeltaDivergence => true,
            Status::Rejection => true,
            Status::SecondaryUpdateError(_) => true,
            Status::UnmetStopCriterion => true,
            // must stop
            Status::SmallStepsize => false,
            Status::SecondaryUpdateTerminate => false,
            Status::ContinuedFailure => false,
            Status::ContinuedRejection => false,
            Status::NanOrInfResidual => false,
            Status::NanOrInfDelta => false,
            // irrelevant
            Status::Success => false,
        }
    }

    /// Allocates a new instance from the result of a secondary update (SUP)
    pub(crate) fn from_sup(res: Result<bool, StrError>) -> Self {
        match res {
            Ok(terminate) => {
                if terminate {
                    Status::SecondaryUpdateTerminate
                } else {
                    Status::Success
                }
            }
            Err(e) => Status::SecondaryUpdateError(e),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use russell_lab::{Norm, Vector};

    #[test]
    fn test_stop_validate_works() {
        let u = Vector::from(&[1.0, 2.0, 3.0]);
        let l = 5.0;

        // MinCompU - valid case
        let stop = Stop::MinCompU(1, 1.0);
        assert!(stop.validate(&u, l).is_ok());

        // MinCompU - index out of bounds
        let stop = Stop::MinCompU(3, 1.0);
        assert!(stop.validate(&u, l).is_err());

        // MinCompU - value not less than initial
        let stop = Stop::MinCompU(1, 2.5);
        assert!(stop.validate(&u, l).is_err());

        // MaxCompU - valid case
        let stop = Stop::MaxCompU(2, 4.0);
        assert!(stop.validate(&u, l).is_ok());

        // MaxCompU - index out of bounds
        let stop = Stop::MaxCompU(5, 4.0);
        assert!(stop.validate(&u, l).is_err());

        // MaxCompU - value not greater than initial
        let stop = Stop::MaxCompU(2, 2.0);
        assert!(stop.validate(&u, l).is_err());

        // MaxNormU - valid case
        let stop = Stop::MaxNormU(4.0, Norm::Euc, 0, 3);
        assert!(stop.validate(&u, l).is_ok());

        // MaxNormU - begin >= end
        let stop = Stop::MaxNormU(4.0, Norm::Euc, 2, 2);
        assert!(stop.validate(&u, l).is_err());

        // MaxNormU - end > u.dim()
        let stop = Stop::MaxNormU(4.0, Norm::Euc, 0, 4);
        assert!(stop.validate(&u, l).is_err());

        // MaxNormU - value <= 0.0
        let stop = Stop::MaxNormU(0.0, Norm::Euc, 0, 3);
        assert!(stop.validate(&u, l).is_err());

        // MinLambda - valid case
        let stop = Stop::MinLambda(3.0);
        assert!(stop.validate(&u, l).is_ok());

        // MinLambda - value not less than initial
        let stop = Stop::MinLambda(6.0);
        assert!(stop.validate(&u, l).is_err());

        // MaxLambda - valid case
        let stop = Stop::MaxLambda(7.0);
        assert!(stop.validate(&u, l).is_ok());

        // MaxLambda - value not greater than initial
        let stop = Stop::MaxLambda(4.0);
        assert!(stop.validate(&u, l).is_err());

        // Steps - valid case
        let stop = Stop::Steps(10);
        assert!(stop.validate(&u, l).is_ok());

        // Steps - invalid case (zero steps)
        let stop = Stop::Steps(0);
        assert!(stop.validate(&u, l).is_err());
    }

    #[test]
    fn test_stop_lambda_works() {
        // MinCompU - returns None
        let stop = Stop::MinCompU(0, 1.0);
        assert_eq!(stop.lambda(), None);

        // MaxCompU - returns None
        let stop = Stop::MaxCompU(0, 1.0);
        assert_eq!(stop.lambda(), None);

        // MaxNormU - returns None
        let stop = Stop::MaxNormU(4.0, Norm::Euc, 0, 3);
        assert_eq!(stop.lambda(), None);

        // MinLambda - returns Some with is_min=true
        let stop = Stop::MinLambda(2.5);
        assert_eq!(stop.lambda(), Some((2.5, true)));

        // MaxLambda - returns Some with is_min=false
        let stop = Stop::MaxLambda(7.5);
        assert_eq!(stop.lambda(), Some((7.5, false)));

        // Steps - returns None
        let stop = Stop::Steps(5);
        assert_eq!(stop.lambda(), None);
    }

    #[test]
    fn test_stop_u_comp_works() {
        // MinCompU - returns Some with is_min=true
        let stop = Stop::MinCompU(2, 1.5);
        assert_eq!(stop.u_comp(), Some((2, 1.5, true)));

        // MaxCompU - returns Some with is_min=false
        let stop = Stop::MaxCompU(3, 10.0);
        assert_eq!(stop.u_comp(), Some((3, 10.0, false)));

        // MaxNormU - returns None
        let stop = Stop::MaxNormU(4.0, Norm::Euc, 0, 3);
        assert_eq!(stop.u_comp(), None);

        // MinLambda - returns None
        let stop = Stop::MinLambda(2.0);
        assert_eq!(stop.u_comp(), None);

        // MaxLambda - returns None
        let stop = Stop::MaxLambda(5.0);
        assert_eq!(stop.u_comp(), None);

        // Steps - returns None
        let stop = Stop::Steps(10);
        assert_eq!(stop.u_comp(), None);
    }

    #[test]
    fn test_stop_now_works() {
        let u = Vector::from(&[1.5, 3.5]);
        let l = 2.0;

        // MinCompU - criterion met
        let stop = Stop::MinCompU(0, 2.0);
        assert!(stop.now(0, &u, l));

        // MinCompU - criterion not met
        let stop = Stop::MinCompU(0, 1.0);
        assert!(!stop.now(0, &u, l));

        // MaxCompU - criterion met
        let stop = Stop::MaxCompU(1, 3.0);
        assert!(stop.now(0, &u, l));

        // MaxCompU - criterion not met
        let stop = Stop::MaxCompU(1, 4.0);
        assert!(!stop.now(0, &u, l));

        // MaxNormU - criterion met (norm > max)
        let stop = Stop::MaxNormU(2.0, Norm::Euc, 0, 2);
        assert!(stop.now(0, &u, l));

        // MaxNormU - criterion not met (norm < max)
        let stop = Stop::MaxNormU(5.0, Norm::Euc, 0, 2);
        assert!(!stop.now(0, &u, l));

        // MinLambda - criterion met
        let stop = Stop::MinLambda(3.0);
        assert!(stop.now(0, &u, l));

        // MinLambda - criterion not met
        let stop = Stop::MinLambda(1.0);
        assert!(!stop.now(0, &u, l));

        // MaxLambda - criterion met
        let stop = Stop::MaxLambda(1.5);
        assert!(stop.now(0, &u, l));

        // MaxLambda - criterion not met
        let stop = Stop::MaxLambda(3.0);
        assert!(!stop.now(0, &u, l));

        // Steps - criterion met (step+1 == n)
        let stop = Stop::Steps(5);
        assert!(stop.now(4, &u, l)); // step 4, so step+1 = 5

        // Steps - criterion not met
        let stop = Stop::Steps(5);
        assert!(!stop.now(3, &u, l)); // step 3, so step+1 = 4
    }

    #[test]
    fn test_status_methods_work() {
        // success
        assert!(Status::Success.success());
        assert!(!Status::SmallStepsize.success());
        assert!(!Status::LargeDelta.success());

        // failure
        assert!(!Status::Success.failure());
        assert!(Status::SmallStepsize.failure());
        assert!(Status::LargeDelta.failure());
        assert!(Status::SecondaryUpdateError("error").failure());
    }

    #[test]
    fn test_status_try_again_works() {
        // may try again
        assert!(Status::BorderingSmallDenominator.try_again());
        assert!(Status::LargeDelta.try_again());
        assert!(Status::ReachedMaxIterations.try_again());
        assert!(Status::ContinuedResidualDivergence.try_again());
        assert!(Status::ContinuedDeltaDivergence.try_again());
        assert!(Status::Rejection.try_again());
        assert!(Status::SecondaryUpdateError("error").try_again());
        assert!(Status::UnmetStopCriterion.try_again());

        // must stop
        assert!(!Status::SmallStepsize.try_again());
        assert!(!Status::SecondaryUpdateTerminate.try_again());
        assert!(!Status::ContinuedFailure.try_again());
        assert!(!Status::ContinuedRejection.try_again());
        assert!(!Status::NanOrInfResidual.try_again());
        assert!(!Status::NanOrInfDelta.try_again());

        // irrelevant (Success)
        assert!(!Status::Success.try_again());
    }

    #[test]
    fn test_status_from_sup_works() {
        // Ok(false) -> Success
        assert_eq!(Status::from_sup(Ok(false)), Status::Success);

        // Ok(true) -> SecondaryUpdateTerminate
        assert_eq!(Status::from_sup(Ok(true)), Status::SecondaryUpdateTerminate);

        // Err -> SecondaryUpdateError
        assert_eq!(
            Status::from_sup(Err("some error")),
            Status::SecondaryUpdateError("some error")
        );
    }

    #[test]
    fn test_method_name_and_description_work() {
        assert_eq!(Method::Natural.name(), "Natural");
        assert_eq!(Method::Arclength.name(), "Arclength");

        assert_eq!(
            Method::Natural.description(),
            "Natural parameter continuation; solves G(u, λ) = 0"
        );
        assert_eq!(
            Method::Arclength.description(),
            "Pseudo-arclength continuation; solves G(u(s), λ(s)) = 0"
        );
    }

    #[test]
    fn test_soderlind_class_params_work() {
        // Ho211
        let (b1, b2, b3, a2, a3) = SoderlindClass::Ho211.params();
        assert_eq!(b1, 0.5);
        assert_eq!(b2, 0.5);
        assert_eq!(b3, 0.0);
        assert_eq!(a2, 0.5);
        assert_eq!(a3, 0.0);

        // H211b
        let (b1, b2, b3, a2, a3) = SoderlindClass::H211b(2.0).params();
        assert_eq!(b1, 0.5);
        assert_eq!(b2, 0.5);
        assert_eq!(b3, 0.0);
        assert_eq!(a2, 0.5);
        assert_eq!(a3, 0.0);

        let (b1, b2, b3, a2, a3) = SoderlindClass::H211b(4.0).params();
        assert_eq!(b1, 0.25);
        assert_eq!(b2, 0.25);
        assert_eq!(b3, 0.0);
        assert_eq!(a2, 0.25);
        assert_eq!(a3, 0.0);

        // H211PI
        let (b1, b2, b3, a2, a3) = SoderlindClass::H211PI.params();
        assert_eq!(b1, 1.0 / 6.0);
        assert_eq!(b2, 1.0 / 6.0);
        assert_eq!(b3, 0.0);
        assert_eq!(a2, 0.0);
        assert_eq!(a3, 0.0);

        // Ho312
        let (b1, b2, b3, a2, a3) = SoderlindClass::Ho312.params();
        assert_eq!(b1, 0.25);
        assert_eq!(b2, 0.5);
        assert_eq!(b3, 0.25);
        assert_eq!(a2, 0.75);
        assert_eq!(a3, 0.25);

        // H312b
        let (b1, b2, b3, a2, a3) = SoderlindClass::H312b(2.0).params();
        assert_eq!(b1, 0.5);
        assert_eq!(b2, 1.0);
        assert_eq!(b3, 0.5);
        assert_eq!(a2, 1.5);
        assert_eq!(a3, 0.5);

        // H312PID
        let (b1, b2, b3, a2, a3) = SoderlindClass::H312PID.params();
        assert_eq!(b1, 1.0 / 18.0);
        assert_eq!(b2, 1.0 / 9.0);
        assert_eq!(b3, 1.0 / 18.0);
        assert_eq!(a2, 0.0);
        assert_eq!(a3, 0.0);

        // Ho321
        let (b1, b2, b3, a2, a3) = SoderlindClass::Ho321.params();
        assert_eq!(b1, 1.25);
        assert_eq!(b2, 0.5);
        assert_eq!(b3, -0.75);
        assert_eq!(a2, -0.25);
        assert_eq!(a3, -0.75);

        // H321
        let (b1, b2, b3, a2, a3) = SoderlindClass::H321.params();
        assert_eq!(b1, 1.0 / 3.0);
        assert_eq!(b2, 1.0 / 18.0);
        assert_eq!(b3, -5.0 / 18.0);
        assert_eq!(a2, -5.0 / 6.0);
        assert_eq!(a3, -1.0 / 6.0);
    }

    #[test]
    fn test_rdiff_type_derives_work() {
        let a = RdiffType::Ave;
        let b = RdiffType::Max;

        // Clone
        let c = a;
        assert_eq!(c, RdiffType::Ave);

        // Debug
        assert_eq!(format!("{:?}", b), "Max");

        // Copy
        let d = a;
        assert_eq!(a, d);
    }

    #[test]
    fn test_ini_dir_derives_work() {
        let a = IniDir::Pos;
        let b = IniDir::Neg;

        // Eq / PartialEq
        assert_eq!(a, IniDir::Pos);
        assert_ne!(a, b);

        // Clone
        let c = a;
        assert_eq!(c, IniDir::Pos);

        // Debug
        assert_eq!(format!("{:?}", b), "Neg");

        // Copy
        let d = a;
        assert_eq!(a, d);
    }
}
