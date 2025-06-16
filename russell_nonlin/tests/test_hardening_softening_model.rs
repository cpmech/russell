#![allow(unused)]

use plotpy::{Curve, Plot};
use russell_lab::approx_eq;
use russell_lab::Vector;
use russell_nonlin::StrError;
use russell_ode::Method as OdeMethod;
use russell_ode::OdeSolver;
use russell_ode::Params as OdeParams;
use russell_ode::System as OdeSystem;

/// Implements the y(x) model with hardening and softening
///
/// x is strain
/// y is stress
struct HardeningSofteningModel {
    lambda_i: f64,
    lambda_r: f64,
    alpha: f64,
    beta: f64,
    c1: f64,
    c2: f64,
    c3: f64,
}

/// Holds the strain (x) and stress (y) state
struct StressStrainState {
    strain: f64, // x
    stress: f64, // y
}

/// Holds the arguments for the ODE solver
struct OdeArgs {
    state: StressStrainState,
    model: HardeningSofteningModel,
}

/// Implements the stress-strain model
struct StressStrainModel<'a> {
    args: OdeArgs,
    ode: OdeSolver<'a, OdeArgs>,
    aux: Vector, // auxiliary: stress (single component in 1D)
}

impl HardeningSofteningModel {
    /// Allocates a new instance
    pub fn new(lambda_i: f64, lambda_r: f64, y_r: f64, alpha: f64, beta: f64) -> Self {
        // constants
        let c1 = beta * lambda_r;
        let c2 = 1.0;
        let c3 = f64::exp(beta * y_r) - c2;

        // new instance
        HardeningSofteningModel {
            lambda_i,
            lambda_r,
            alpha,
            beta,
            c1,
            c2,
            c3,
        }
    }

    /// Calculates the modulus (y is stress, x is strain)
    pub fn dydx(&self, x: f64, y: f64) -> f64 {
        // reference curve
        let c1x = self.c1 * x;
        let y_ref = if c1x >= 500.0 {
            0.0
        } else {
            -self.lambda_r * x + f64::ln(self.c3 + self.c2 * f64::exp(c1x)) / self.beta
        };
        let dydx_ref = if c1x >= 500.0 {
            0.0
        } else {
            let ec1x = f64::exp(c1x);
            -self.lambda_r + (self.c1 * self.c2 * ec1x) / (self.beta * (self.c3 + self.c2 * ec1x))
        };

        // slope (modulus)
        let dist = f64::max(0.0, y_ref - y);
        let lambda_f = dydx_ref;
        self.lambda_i + (lambda_f - self.lambda_i) * f64::exp(-self.alpha * dist)
    }
}

impl<'a> StressStrainModel<'a> {
    /// Allocates a new instance
    pub fn new(lambda_i: f64, lambda_r: f64, y_r: f64, alpha: f64, beta: f64) -> Result<Self, StrError> {
        // stress-strain stress state and model
        let state = StressStrainState {
            strain: 0.0,
            stress: 0.0,
        };
        let model = HardeningSofteningModel::new(lambda_i, lambda_r, y_r, alpha, beta);
        let args = OdeArgs { state, model };

        // ODE solver
        let params = OdeParams::new(OdeMethod::BwEuler);
        let system = OdeSystem::new(1, |f, x, y, args: &mut OdeArgs| {
            f[0] = args.model.dydx(x, y[0]);
            Ok(())
        });
        let ode = OdeSolver::new(params, system)?;

        // new instance
        Ok(StressStrainModel {
            args,
            ode,
            aux: Vector::new(1),
        })
    }

    /// Updates the stress-strain state given an increment of strain
    pub fn update(&mut self, state: &mut StressStrainState, delta_strain: f64) -> Result<(), StrError> {
        self.aux[0] = state.stress;
        let x0 = state.strain;
        let x1 = x0 + delta_strain;
        self.ode.solve(&mut self.aux, x0, x1, None, &mut self.args)?;
        state.strain = x1;
        state.stress = self.aux[0];
        Ok(())
    }
}

const SAVE_FIGURE: bool = true;

#[test]
fn test_hardening_softening_model() {
    let lambda_i = 10.0;
    let lambda_r = 3.0;
    let y_r = 1.0;
    let alpha = 3.0;
    let beta = 5.0;

    let mut model = StressStrainModel::new(lambda_i, lambda_r, y_r, alpha, beta).unwrap();
    let mut state = StressStrainState {
        strain: 0.0,
        stress: 0.0,
    };

    let delta_strain = 0.01;
    let np = 50;
    let mut xx = vec![state.strain; np + 1];
    let mut yy = vec![state.stress; np + 1];
    for i in 0..np {
        model.update(&mut state, delta_strain).unwrap();
        xx[1 + i] = state.strain;
        yy[1 + i] = state.stress;
    }

    let mut xx_ref = vec![
        0., 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18,
        0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37,
        0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5,
    ];

    let mut yy_ref = vec![
        0.,
        0.0921923529874959,
        0.180995936702942,
        0.265135546339778,
        0.34303769588157,
        0.412909639654633,
        0.472933902640526,
        0.521575383562834,
        0.557924601234833,
        0.581937761631112,
        0.594452730469534,
        0.596972402157104,
        0.591326450808571,
        0.579354305707458,
        0.562695563124671,
        0.542698423576125,
        0.520413194773086,
        0.496630461299565,
        0.471934147069921,
        0.446753034002368,
        0.421403915953387,
        0.39612495266096,
        0.371100159763253,
        0.346476741387926,
        0.322376942591868,
        0.298905838568251,
        0.276156158431712,
        0.254210961837679,
        0.23314478624825,
        0.213023733754134,
        0.193904874033716,
        0.175835268002485,
        0.158850897482815,
        0.142975694678168,
        0.128220878500363,
        0.114584707614984,
        0.102052740913371,
        0.0905985737010805,
        0.0801850545040245,
        0.070765847217885,
        0.062287241401217,
        0.0546900747507887,
        0.047911642057484,
        0.0418874859643884,
        0.0365529932436868,
        0.0318447432166188,
        0.0277015895758155,
        0.0240654765039555,
        0.0208820056359409,
        0.0181007846054456,
        0.0156755917049248,
    ];

    if SAVE_FIGURE {
        let mut curve = Curve::new();
        let mut curve_ref = Curve::new();
        curve.set_label("russell").set_line_style("None").set_marker_style("o");
        curve_ref.set_label("reference");
        curve.draw(&xx, &yy);
        curve_ref.draw(&xx_ref, &yy_ref);
        let mut plot = Plot::new();
        plot.add(&curve_ref)
            .add(&curve)
            .grid_labels_legend("strain", "stress")
            .set_figure_size_points(600.0, 350.0)
            .save("/tmp/russell_nonlin/test_hardening_softening_model.svg")
            .unwrap();
    }

    // check results
    for i in 0..np {
        approx_eq(yy[i], yy_ref[i], 0.003);
    }
}
