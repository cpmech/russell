use plotpy::{Curve, Plot};
use russell_lab::{approx_eq, deriv1_forward7};
use russell_nonlin::StrError;

const SAVE_FIGURE: bool = true;

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
#[derive(Clone)]
struct StressStrainState {
    strain: f64,     // x
    stress: f64,     // y
    strain_bkp: f64, // backup of strain
    stress_bkp: f64, // backup of stress
}

impl StressStrainState {
    pub fn new() -> Self {
        StressStrainState {
            strain: 0.0,
            stress: 0.0,
            strain_bkp: 0.0,
            stress_bkp: 0.0,
        }
    }
    pub fn backup(&mut self) {
        self.strain_bkp = self.strain;
        self.stress_bkp = self.stress;
    }
    pub fn restore(&mut self) {
        self.strain = self.strain_bkp;
        self.stress = self.stress_bkp;
    }
}

/// Implements the stress-strain model
struct StressStrainModel {
    model: HardeningSofteningModel,
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

    /// Calculates the reference curve ordinate, yr(x)
    fn yr(&self, x: f64) -> f64 {
        let c1x = self.c1 * x;
        if c1x >= 500.0 {
            0.0
        } else {
            -self.lambda_r * x + f64::ln(self.c3 + self.c2 * f64::exp(c1x)) / self.beta
        }
    }

    /// Calculates the slope of the reference curve dyr/dx
    fn dyr_dx(&self, x: f64) -> f64 {
        let c1x = self.c1 * x;
        if c1x >= 500.0 {
            0.0
        } else {
            let ec1x = f64::exp(c1x);
            let h = self.c3 + self.c2 * ec1x;
            -self.lambda_r + (self.c1 * self.c2 * ec1x) / (self.beta * h)
        }
    }

    /// Calculates the derivative of the slope of the reference curve w.r.t x
    ///
    /// Calculates `d(dyr/dx)/dx = d²yr/dx²`
    fn d2yr_dx2(&self, x: f64) -> f64 {
        let c1x = self.c1 * x;
        if c1x >= 500.0 {
            0.0
        } else {
            let ec1x = f64::exp(c1x);
            let h = self.c3 + self.c2 * ec1x;
            (self.c1 * self.c1 * self.c2 * self.c3 * ec1x) / (self.beta * h * h)
        }
    }

    /// Calculates the derivative of y with respect to x
    pub fn dy_dx(&self, x: f64, y: f64) -> f64 {
        let yr = self.yr(x);
        let del = f64::max(0.0, yr - y);
        let lambda_f = self.dyr_dx(x);
        self.lambda_i + (lambda_f - self.lambda_i) * f64::exp(-self.alpha * del)
    }

    /// Calculates the derivative of dy/dx with respect to x
    ///
    /// Calculates `d(dy/dx)/dx = d²y/dx²`
    pub fn d2y_dx2(&self, x: f64, y: f64) -> f64 {
        let yr = self.yr(x);
        let del = f64::max(0.0, yr - y);
        let lambda_f = self.dyr_dx(x);
        let dyr1 = self.dyr_dx(x);
        let dyr2 = self.d2yr_dx2(x);
        (dyr2 - dyr1 * self.alpha * (lambda_f - self.lambda_i)) * f64::exp(-self.alpha * del)
    }

    /// Calculates the derivative of dy/dx with respect to y
    ///
    /// Calculates `d(dy/dx)/dy = d²y/(dx dy)`
    pub fn d2y_dxdy(&self, x: f64, y: f64) -> f64 {
        let yr = self.yr(x);
        let del = f64::max(0.0, yr - y);
        let lambda_f = self.dyr_dx(x);
        self.alpha * (lambda_f - self.lambda_i) * f64::exp(-self.alpha * del)
    }
}

impl StressStrainModel {
    /// Allocates a new instance
    pub fn new(lambda_i: f64, lambda_r: f64, y_r: f64, alpha: f64, beta: f64) -> Result<Self, StrError> {
        Ok(StressStrainModel {
            model: HardeningSofteningModel::new(lambda_i, lambda_r, y_r, alpha, beta),
        })
    }

    /// Updates the stress-strain state given an increment of strain
    pub fn update(&mut self, state: &mut StressStrainState, delta_strain: f64) -> Result<(), StrError> {
        if f64::abs(delta_strain) < 1e-12 {
            return Ok(()); // no update needed
        }
        let x0 = state.strain;
        let dx = delta_strain;
        let x1 = x0 + dx;
        let y0 = state.stress;
        let f0 = self.model.dy_dx(x0, y0);
        let mut y1 = y0 + dx * f0;
        let mut converged = false;
        for _ in 0..21 {
            let f1 = self.model.dy_dx(x1, y1);
            let r = y1 - y0 - dx * f1;
            if f64::abs(r) < 1e-12 {
                converged = true;
                break;
            }
            let jj = self.model.d2y_dxdy(x1, y1);
            let dy = -r / (1.0 - dx * jj);
            if f64::abs(dy) < 1e-12 {
                converged = true;
                break;
            }
            y1 += dy;
        }
        if !converged {
            return Err("Newton-Raphson did not converge");
        }
        state.strain = x1;
        state.stress = y1;
        // self.aux[0] = state.stress;
        // self.ode.solve(&mut self.aux, x0, x1, None, &mut self.args)?;
        // state.stress = self.aux[0];
        Ok(())
    }

    /// Calculates the (consistent) modulus `D = dσ/dε`
    ///
    /// `Consistency` here means that derivative is taken from the Backward-Euler update
    pub fn modulus(&self, state: &StressStrainState, delta_strain: f64, inconsistent: bool) -> f64 {
        let ddb = self.model.dy_dx(state.strain, state.stress);
        if inconsistent {
            ddb
        } else {
            let ll = self.model.d2y_dx2(state.strain, state.stress);
            let jj = self.model.d2y_dxdy(state.strain, state.stress);
            (ddb + delta_strain * ll) / (1.0 - delta_strain * jj)
        }
    }

    pub fn numerical_modulus(&mut self, state: &StressStrainState, delta_strain: f64) -> Result<f64, StrError> {
        let mut args = state.clone();
        let dx_at = delta_strain;
        let calc_sigma = |dx, a: &mut StressStrainState| {
            a.backup();
            self.update(a, dx)?;
            let sigma = a.stress;
            a.restore();
            Ok(sigma)
        };
        // deriv1_central5(dx_at, &mut args, calc_sigma)
        deriv1_forward7(dx_at, &mut args, calc_sigma)
    }
}

#[test]
fn test_hardening_softening_model_1() {
    let lambda_i = 10.0;
    let lambda_r = 3.0;
    let y_r = 1.0;
    let alpha = 30.0;
    let beta = 30.0;

    let hs = HardeningSofteningModel::new(lambda_i, lambda_r, y_r, alpha, beta);

    let y_ref = hs.yr(0.0);
    let dy_dx_ref = hs.dyr_dx(0.0);
    assert_eq!(y_ref, y_r); // @ x=0.0, y_ref must equal y_r
    approx_eq(dy_dx_ref, -lambda_r, 1e-12); // @ x=0.0, dy/dx must equal -lambda_r if beta is large enough

    let dy_dx = hs.dy_dx(0.0, 0.0);
    approx_eq(dy_dx, lambda_i, 1e-11); // @ x=0.0, dy/dx must equal lambda_i if alpha is large enough

    let args = &mut 0;
    let x_at = 0.0;
    let y_at = 0.0;

    // check d²y/dx²
    let ana = hs.d2y_dx2(x_at, y_at);
    let num = deriv1_forward7(x_at, args, |x, _| Ok(hs.dy_dx(x, y_at))).unwrap();
    println!("d²y/dx²: analytical = {}, numerical = {}", ana, num);
    approx_eq(ana, num, 1e-10);

    // check d²y/(dx dy)
    let ana = hs.d2y_dxdy(0.0, 0.0);
    let num = deriv1_forward7(y_at, args, |y, _| Ok(hs.dy_dx(x_at, y))).unwrap();
    println!("d²y/(dx dy): analytical = {}, numerical = {}", ana, num);
    approx_eq(ana, num, 1e-11);
}

#[test]
fn test_hardening_softening_model_2() {
    let lambda_i = 10.0;
    let lambda_r = 3.0;
    let y_r = 1.0;
    let alpha = 3.0;
    let beta = 3.0;

    let hs = HardeningSofteningModel::new(lambda_i, lambda_r, y_r, alpha, beta);

    let args = &mut 0;
    let x_at = 0.0;
    let y_at = 0.0;

    // check d²y/dx²
    let ana = hs.d2y_dx2(x_at, y_at);
    let num = deriv1_forward7(x_at, args, |x, _| Ok(hs.dy_dx(x, y_at))).unwrap();
    println!("d²y/dx²: analytical = {}, numerical = {}", ana, num);
    approx_eq(ana, num, 1e-12);

    // check d²y/(dx dy)
    let ana = hs.d2y_dxdy(0.0, 0.0);
    let num = deriv1_forward7(y_at, args, |y, _| Ok(hs.dy_dx(x_at, y))).unwrap();
    println!("d²y/(dx dy): analytical = {}, numerical = {}", ana, num);
    approx_eq(ana, num, 1e-11);
}

#[test]
fn test_stress_strain_model_1() {
    let lambda_i = 10.0;
    let lambda_r = 3.0;
    let y_r = 1.0;
    let alpha = 3.0;
    let beta = 5.0;

    let mut model = StressStrainModel::new(lambda_i, lambda_r, y_r, alpha, beta).unwrap();
    let mut state = StressStrainState::new();

    let delta_strain = 0.01;
    let np = 50;
    let mut xx = vec![state.strain; np + 1];
    let mut yy = vec![state.stress; np + 1];
    for i in 0..np {
        model.update(&mut state, delta_strain).unwrap();
        xx[1 + i] = state.strain;
        yy[1 + i] = state.stress;
    }

    let xx_ref = vec![
        0., 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18,
        0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37,
        0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5,
    ];

    // from Mathematica
    let yy_ref = vec![
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
            .save("/tmp/russell_nonlin/test_stress_strain_model_1.svg")
            .unwrap();
    }

    // check results
    for i in 0..np {
        approx_eq(yy[i], yy_ref[i], 0.03);
    }
}

#[test]
fn test_stress_strain_model_2() {
    let lambda_i = 10.0;
    let lambda_r = 3.0;
    let y_r = 1.0;
    let alpha = 3.0;
    let beta = 5.0;

    let mut model = StressStrainModel::new(lambda_i, lambda_r, y_r, alpha, beta).unwrap();
    let mut state = StressStrainState::new();

    let ddb0 = model.modulus(&state, 0.0, true); // first inconsistent modulus
    let dd0 = ddb0; // first consistent modulus == first inconsistent modulus

    let delta_strain = 0.01; // local NR won't converge with large delta_strain
    let np = 20;
    let mut xx = vec![state.strain; np + 1];
    let mut yy = vec![state.stress; np + 1];
    let mut ddb = vec![ddb0; np + 1]; // inconsistent modulus
    let mut dd = vec![dd0; np + 1]; // consistent modulus
    let mut dd_num = vec![dd0; np + 1]; // numerical consistent modulus
    for i in 0..np {
        model.update(&mut state, delta_strain).unwrap();
        xx[1 + i] = state.strain;
        yy[1 + i] = state.stress;
        ddb[1 + i] = model.modulus(&state, delta_strain, true);
        dd[1 + i] = model.modulus(&state, delta_strain, false);
        dd_num[1 + i] = model.numerical_modulus(&state, delta_strain).unwrap();
    }

    if SAVE_FIGURE {
        let mut state_fine = StressStrainState::new();
        let delta_strain = 0.005;
        let np = 100;
        let mut xx_fine = vec![state_fine.strain; np + 1];
        let mut yy_fine = vec![state_fine.stress; np + 1];
        for i in 0..np {
            model.update(&mut state_fine, delta_strain).unwrap();
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
            .save("/tmp/russell_nonlin/test_stress_strain_model_2.svg")
            .unwrap();
    }

    // numerical consistent modulus
}
