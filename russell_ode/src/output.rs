use russell_lab::{vec_max_abs_diff, Vector};
use std::collections::HashMap;

pub struct Output {
    pub step_h: Vec<f64>,
    pub step_x: Vec<f64>,
    pub step_y: HashMap<usize, Vec<f64>>,
    pub step_global_error: Vec<f64>,
    analytical: Option<fn(&mut Vector, f64)>,
    y_ana: Vector,
}

impl Output {
    pub fn new(selected_y_components: &[usize], analytical: Option<fn(&mut Vector, f64)>) -> Self {
        let mut step_y = HashMap::new();
        for m in selected_y_components {
            step_y.insert(*m, Vec::new());
        }
        const EMPTY: usize = 0;
        Output {
            step_h: Vec::new(),
            step_x: Vec::new(),
            step_y,
            step_global_error: Vec::new(),
            analytical,
            y_ana: Vector::new(EMPTY),
        }
    }

    pub fn reset(&mut self) {
        self.step_h.clear();
        self.step_x.clear();
        for ym in self.step_y.values_mut() {
            ym.clear();
        }
        self.step_global_error.clear();
    }

    pub(crate) fn execute_step(&mut self, x: f64, y: &Vector, h: f64) {
        self.step_h.push(h);
        self.step_x.push(x);
        for (m, ym) in self.step_y.iter_mut() {
            ym.push(y[*m]);
        }
        if let Some(ana) = self.analytical {
            let ndim = y.dim();
            if self.y_ana.dim() != ndim {
                self.y_ana = Vector::new(ndim);
            }
            ana(&mut self.y_ana, x);
            let (_, err) = vec_max_abs_diff(y, &self.y_ana).unwrap();
            self.step_global_error.push(err);
        }
    }
}
