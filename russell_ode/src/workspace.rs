use crate::{Benchmark, Method};

pub(crate) struct Workspace {
    /// Holds benchmark data
    pub(crate) bench: Benchmark,

    /// Indicates that the step follows a reject
    pub(crate) follows_reject_step: bool,

    /// Indicates that the iterations (in an implicit method) are diverging
    pub(crate) iterations_diverging: bool,

    /// Holds a multiplier to the stepsize when the iterations are diverging
    pub(crate) h_multiplier_diverging: f64,

    /// Holds the previous stepsize
    pub(crate) h_prev: f64,

    /// Holds the next stepsize estimate
    pub(crate) h_new: f64,

    /// Holds the previous relative error
    pub(crate) rel_error_prev: f64,

    /// Holds the current relative error
    pub(crate) rel_error: f64,
}

impl Workspace {
    /// Allocates a new instance
    pub(crate) fn new(method: Method) -> Self {
        Workspace {
            bench: Benchmark::new(method),
            follows_reject_step: false,
            iterations_diverging: false,
            h_multiplier_diverging: 1.0,
            h_prev: 0.0,
            h_new: 0.0,
            rel_error_prev: 0.0,
            rel_error: 0.0,
        }
    }

    /// Resets all values
    pub(crate) fn reset(&mut self, h: f64, rel_error_prev_min: f64) {
        self.bench.reset(h);
        self.follows_reject_step = false;
        self.iterations_diverging = false;
        self.h_multiplier_diverging = 1.0;
        self.h_prev = h;
        self.h_new = h;
        self.rel_error_prev = rel_error_prev_min;
        self.rel_error = 0.0;
    }
}
