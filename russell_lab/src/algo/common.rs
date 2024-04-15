/// Constant to indicate an uninitialized value
pub(crate) const UNINITIALIZED: f64 = f64::INFINITY;

/// Indicates that no arguments are needed
pub type NoArgs = u8;

/// Holds statistics for a bracket algorithm
#[derive(Clone, Copy, Debug)]
pub struct AlgoStats {
    /// Number of calls to f(x) (function evaluations)
    pub n_function: usize,

    /// Number of calls to the dy/dx function
    pub n_deriv: usize,

    /// Number of iterations
    pub n_iterations: usize,
}

impl AlgoStats {
    /// Allocates a new instance
    pub fn new() -> AlgoStats {
        AlgoStats {
            n_function: 0,
            n_deriv: 0,
            n_iterations: 0,
        }
    }
}

/// Holds the results of a root finding or minimum bracketing algorithm
///
/// The root yields `f(x_target) = 0.0`. The root is bracketed by a pair of points,
/// `a` and `b`, such that the function has opposite sign at those two points,
/// i.e., `f(a) f(b) < 0.0`.
///
/// The (local) minimum yields `f(x_target) = min{f(x)} in [a, b]`. The (local) minimum is
/// bracketed  by a triple of points `a`, `x_target`, and `c`, such that `f(x_target) < f(a)`
/// and `f(x_target) < f(b)`, with `a < x_target < b`.
#[derive(Clone, Copy, Debug)]
pub struct Bracket {
    /// Holds the lower bound
    pub a: f64,

    /// Holds the root or optimal coordinate
    pub x_target: f64,

    /// Holds the upper bound
    pub b: f64,

    /// Holds the function evaluated at the lower bound
    pub fa: f64,

    /// Holds the function evaluated at the root or optimal coordinate
    pub fx_target: f64,

    /// Holds the function evaluated at the upper bound
    pub fb: f64,
}
