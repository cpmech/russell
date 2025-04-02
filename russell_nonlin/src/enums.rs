/// Specifies the stopping criterion for the continuation process.
#[derive(Clone, Copy, Debug)]
pub enum NlStop {
    /// Stops when lambda reaches the specified value.
    Lambda(f64),

    /// Stops after a number of steps.
    Steps(usize),
}

/// Specifies the method of continuation to be used.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NlMethod {
    /// Pseudo-arclength continuation
    Arclength,

    /// Natural parameter continuation
    Natural,
}

impl NlMethod {
    pub fn description(&self) -> &'static str {
        match self {
            NlMethod::Arclength => "Pseudo-arclength continuation; solves G(u(s), λ(s)) = 0",
            NlMethod::Natural => "Natural parameter continuation; solves G(u, λ) = 0",
        }
    }
}
