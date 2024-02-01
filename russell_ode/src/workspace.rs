use crate::Benchmark;

pub(crate) struct Workspace {
    /// Holds benchmark data
    pub(crate) bench: Benchmark,

    /// Indicates that this is the very first step
    pub(crate) first_step: bool,

    /// Indicates that the step follows a reject
    pub(crate) follows_reject_step: bool,

    /// Indicates that the iterations (in an implicit method) are diverging
    pub(crate) iterations_diverging: bool,

    /// Holds a multiplier to the stepsize when the iterations are diverging
    pub(crate) h_multiplier_diverging: f64,

    /// Holds the current relative error
    pub(crate) rel_error: f64,

    /// Holds the previous relative error
    pub(crate) prev_rel_error: f64,

    /// Holds the next stepsize estimate
    pub(crate) h_new: f64,
}

impl Workspace {
    /// Allocates a new instance
    pub(crate) fn new() -> Self {
        Workspace {
            bench: Benchmark::new(),
            first_step: true,
            follows_reject_step: false,
            iterations_diverging: false,
            h_multiplier_diverging: 1.0,
            rel_error: 0.0,
            prev_rel_error: 0.0,
            h_new: 0.0,
        }
    }

    /// Resets all values
    pub(crate) fn reset(&mut self) {
        self.bench.reset();
        self.first_step = true;
        self.follows_reject_step = false;
        self.iterations_diverging = false;
        self.h_multiplier_diverging = 1.0;
        self.rel_error = 0.0;
        self.prev_rel_error = 0.0;
        self.h_new = 0.0;
    }
}
