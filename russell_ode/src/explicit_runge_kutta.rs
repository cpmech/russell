use crate::{Information, Method, RungeKuttaTrait, StrError};

pub struct ExplicitRungeKutta {
    method: Method,
}

impl ExplicitRungeKutta {
    pub fn new(method: Method) -> Result<Self, StrError> {
        let info = method.information();
        if !info.implicit && info.multiple_stages {
            Ok(ExplicitRungeKutta { method })
        } else {
            Err("The Runge-Kutta method must be explicit and multi-stage")
        }
    }
}

impl RungeKuttaTrait for ExplicitRungeKutta {
    fn information(&self) -> Information {
        self.method.information()
    }

    fn initialize(&mut self) {}

    fn step(&mut self) {}

    fn accept_update(&mut self) -> (f64, f64) {
        (0.0, 0.0)
    }

    fn reject_update(&mut self) -> f64 {
        0.0
    }

    fn dense_output(&self) {}
}
