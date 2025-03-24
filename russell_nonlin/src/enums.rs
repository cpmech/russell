#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NlMethod {
    /// Pseudo-arclength continuation
    Arclength,

    /// Parameter continuation
    Parametric,

    /// Simple Newton-Raphson method without any parameter
    Simple,
}

impl NlMethod {
    pub fn description(&self) -> &'static str {
        match self {
            NlMethod::Arclength => "Pseudo-arclength continuation; solves G(u) = 0",
            NlMethod::Parametric => "Parameter continuation; solves G(u(s), λ(s)) = 0",
            NlMethod::Simple => "Simple Newton-Raphson method without any parameter; solves G(u, λ) = 0",
        }
    }
}
