use ctm_demo::{Model, ModelType};
use plotpy::{Curve, Plot};
use russell_lab::{approx_eq, deriv1_forward7};
use russell_nonlin::StrError;
use russell_ode::Method;
use std::collections::HashMap;

/// Defines the strain (x) and stress (y) state
#[derive(Clone)]
struct StressStrainState {
    strain: f64,     // x
    stress: f64,     // y
    strain_bkp: f64, // backup of strain
    stress_bkp: f64, // backup of stress
}

impl StressStrainState {
    /// Allocates a new instance
    pub fn new() -> Self {
        StressStrainState {
            strain: 0.0,
            stress: 0.0,
            strain_bkp: 0.0,
            stress_bkp: 0.0,
        }
    }

    /// Creates a backup of the current state
    pub fn backup(&mut self) {
        self.strain_bkp = self.strain;
        self.stress_bkp = self.stress;
    }

    /// Restores the state from the backup
    pub fn restore(&mut self) {
        self.strain = self.strain_bkp;
        self.stress = self.stress_bkp;
    }
}

const SAVE_FIGURE: bool = true;

#[test]
fn test_hardening_softening_model() {
    // Allocate the model
    let method = Method::DoPri5;
    let mut model = Model::new(
        ModelType::HardeningSoftening,
        HashMap::from([("li", 10.0), ("lr", 3.0), ("y0r", 1.0), ("a", 3.0), ("b", 5.0)]),
        method,
    )
    .unwrap();

    let mut state = StressStrainState::new();

    let ddb0 = model.continuous_modulus(state.strain, state.stress); // first inconsistent modulus
    let dd0 = ddb0; // first consistent modulus == first inconsistent modulus

    let dx = 0.01; // local NR won't converge with large delta_strain
    let nd = 20;
    let mut xx = vec![state.strain; nd + 1];
    let mut yy = vec![state.stress; nd + 1];
    let mut ddb = vec![ddb0; nd + 1]; // inconsistent modulus
    let mut dd = vec![dd0; nd + 1]; // consistent modulus
    let mut dd_num = vec![dd0; nd + 1]; // numerical consistent modulus
    for i in 0..nd {
        dd_num[1 + i] = model
            .numerical_consistent_tangent_modulus(state.strain, state.stress, dx, false)
            .unwrap();
        model
            .backward_euler_update(&mut state.strain, &mut state.stress, dx)
            .unwrap();
        xx[1 + i] = state.strain;
        yy[1 + i] = state.stress;
        ddb[1 + i] = model.continuous_modulus(state.strain, state.stress);
        dd[1 + i] = model.consistent_tangent_modulus(state.strain, state.stress, dx);
    }

    if SAVE_FIGURE {
        let mut state_fine = StressStrainState::new();
        let delta_strain = 0.005;
        let np = 100;
        let mut xx_fine = vec![state_fine.strain; np + 1];
        let mut yy_fine = vec![state_fine.stress; np + 1];
        for i in 0..np {
            model
                .backward_euler_update(&mut state_fine.strain, &mut state_fine.stress, delta_strain)
                .unwrap();
            xx_fine[1 + i] = state_fine.strain;
            yy_fine[1 + i] = state_fine.stress;
        }
        let mut curve_fine = Curve::new();
        let mut curve = Curve::new();
        let mut curve_ddb = Curve::new();
        let mut curve_dd = Curve::new();
        let mut curve_dd_num = Curve::new();
        curve_fine.set_line_color("gray").draw(&xx_fine, &yy_fine);
        curve.set_marker_style("o").draw(&xx, &yy);
        curve_ddb.set_label("$\\bar{D}$ (inconsistent)").draw(&xx, &ddb);
        curve_dd.set_label("$D$ (consistent)").draw(&xx, &dd);
        curve_dd_num
            .set_label("$D$ (numerical, consistent)")
            .set_line_style("None")
            .set_marker_style("*")
            .draw(&xx, &dd_num);
        let mut plot = Plot::new();
        plot.set_subplot(2, 1, 1)
            .add(&curve_fine)
            .add(&curve)
            .grid_labels_legend("strain", "stress")
            .set_subplot(2, 1, 2)
            .add(&curve_ddb)
            .add(&curve_dd)
            .add(&curve_dd_num)
            .grid_labels_legend("strain", "moduli")
            .set_horiz_line(0.0, "black", "-", 1.0)
            .set_figure_size_points(600.0, 700.0)
            .save("/tmp/russell_nonlin/test_model_curve_and_modulus_1.svg")
            .unwrap();
    }
}
