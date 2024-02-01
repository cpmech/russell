use crate::Benchmark;

pub(crate) struct Workspace {
    /// Holds benchmark data
    pub(crate) bench: Benchmark,

    /// Indicates that this is the very first step
    pub(crate) first_step: bool,

    /// Indicates that the step has just been rejected
    pub(crate) reject_step: bool,

    /// Indicates that the iterations (in an implicit method) are diverging
    pub(crate) iterations_diverging: bool,

    /// Holds a multiplier to the stepsize when the iterations are diverging
    pub(crate) h_multiplier_diverging: f64,

    /// Holds the current relative error
    pub(crate) relative_error: f64,

    /// Holds the previous relative error
    pub(crate) previous_relative_error: f64,

    /// Holds the next stepsize estimate
    pub(crate) stepsize_new: f64,
}

impl Workspace {
    /// Allocates a new instance
    pub(crate) fn new() -> Self {
        Workspace {
            bench: Benchmark::new(),
            first_step: true,
            reject_step: false,
            iterations_diverging: false,
            h_multiplier_diverging: 1.0,
            relative_error: 0.0,
            previous_relative_error: 0.0,
            stepsize_new: 0.0,
        }
    }

    /// Resets all values
    pub(crate) fn reset(&mut self) {
        self.bench.reset();
        self.first_step = true;
        self.reject_step = false;
        self.iterations_diverging = false;
        self.h_multiplier_diverging = 1.0;
        self.relative_error = 0.0;
        self.previous_relative_error = 0.0;
        self.stepsize_new = 0.0;
    }
}
