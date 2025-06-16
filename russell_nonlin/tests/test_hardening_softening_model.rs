#![allow(unused)]

use plotpy::{Curve, Plot};
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

    let delta_strain = 0.005;
    let np = 101;
    let mut xx = vec![state.strain; np + 1];
    let mut yy = vec![state.stress; np + 1];
    for i in 0..np {
        model.update(&mut state, delta_strain).unwrap();
        xx[1 + i] = state.strain;
        yy[1 + i] = state.stress;
    }

    if SAVE_FIGURE {
        let mut curve = Curve::new();
        curve.set_label("Stress-Strain Curve");
        curve.draw(&xx, &yy);
        let mut plot = Plot::new();
        plot.add(&curve)
            .grid_and_labels("strain", "stress")
            .set_figure_size_points(600.0, 350.0)
            .save("/tmp/russell_nonlin/test_hardening_softening_model.svg")
            .unwrap();
    }
}
