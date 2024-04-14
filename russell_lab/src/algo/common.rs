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
