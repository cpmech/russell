use plotpy::{Curve, Plot};
use russell_lab::{approx_eq, deriv1_forward7};
use russell_nonlin::StrError;

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

/// Implements the hardening and softening model
///
/// ```text
/// dy
/// ── = f(x, y)
/// dx
/// ```
///
/// where:
///
/// * x is strain
/// * y is stress
/// * f is the (continuous) modulus
struct HardeningSofteningModel {
    li: f64, // initial slope (λi)
    lr: f64, // reference slope (λr); second slope, after peak, going down
    a: f64,  // smoothing parameter (α); when going from λi to λr
    b: f64,  // smoothing parameter (β); when going from λr to 0
    c1: f64, // constant c1
    c2: f64, // constant c2
    c3: f64, // constant c3
}

impl HardeningSofteningModel {
    /// Allocates a new instance
    ///
    /// # Arguments
    ///
    /// * `li` - initial slope (λi)
    /// * `lr` - reference slope (λr); second slope, after peak, going down
    /// * `y0r` - reference ordinate (yr(0)); stress at zero strain (x=0). Note that this is not the same yr0 (which is 0 in this model)
    /// * `a` - smoothing parameter (α); when going from λi to λr
    /// * `b` - smoothing parameter (β); when going from λr to 0
    pub fn new(li: f64, lr: f64, y0r: f64, a: f64, b: f64) -> Self {
        let c1 = b * lr;
        let c2 = 1.0; // exp(β yr0) = exp(0) since yr0 = 0
        let c3 = f64::exp(b * y0r) - c2;
        HardeningSofteningModel {
            li,
            lr,
            a,
            b,
            c1,
            c2,
            c3,
        }
    }

    /// Calculates the reference curve ordinate, yr(x)
    ///
    /// (This is Model C0: Decay reaching an exactly horizontal line)
    ///
    /// ```text
    /// yr(x) = -λr x + ln(c3 + c2 * exp(c1 * x)) / β
    /// ```
    fn yr(&self, x: f64) -> f64 {
        let c1x = self.c1 * x;
        if c1x >= 500.0 {
            0.0
        } else {
            -self.lr * x + f64::ln(self.c3 + self.c2 * f64::exp(c1x)) / self.b
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
            -self.lr + (self.c1 * self.c2 * ec1x) / (self.b * h)
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
            (self.c1 * self.c1 * self.c2 * self.c3 * ec1x) / (self.b * h * h)
        }
    }

    /// Calculates the derivative of y with respect to x
    ///
    /// ```text
    /// dy                                              _
    /// ── = f(x,y)  where  f is the continuous modulus D
    /// dx
    /// ```
    pub fn f(&self, x: f64, y: f64) -> f64 {
        let yr = self.yr(x);
        let del = f64::max(0.0, yr - y);
        let lt = self.dyr_dx(x); // λt (target slope controlled by the reference curve)
        self.li + (lt - self.li) * f64::exp(-self.a * del)
    }

    /// Calculates the derivative of f with respect to x
    ///
    /// ```text
    /// df    d ⎛dy⎞   d²y   
    /// ── = ── ⎜──⎟ = ───
    /// dx   dx ⎝dx⎠   dx²   
    /// ```
    pub fn df_dx(&self, x: f64, y: f64) -> f64 {
        let yr = self.yr(x);
        let del = f64::max(0.0, yr - y);
        let lt = self.dyr_dx(x); // λt (target slope controlled by the reference curve)
        let d2 = self.d2yr_dx2(x);
        f64::exp(-self.a * del) * (d2 + self.a * self.li * lt - self.a * lt * lt)
    }

    /// Calculates the derivative of f with respect to y
    ///
    /// ```text
    /// df    d ⎛dy⎞    d²y   
    /// ── = ── ⎜──⎟ = ─────
    /// dy   dy ⎝dx⎠   dx dy
    /// ```
    pub fn df_dy(&self, x: f64, y: f64) -> f64 {
        let yr = self.yr(x);
        let del = f64::max(0.0, yr - y);
        let lt = self.dyr_dx(x); // λt (target slope controlled by the reference curve)
        f64::exp(-self.a * del) * self.a * (lt - self.li)
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
        let f0 = self.f(x0, y0);
        let mut y1 = y0 + dx * f0;
        let mut converged = false;
        for _ in 0..21 {
            let f1 = self.f(x1, y1);
            let r = y1 - y0 - dx * f1;
            if f64::abs(r) < 1e-12 {
                converged = true;
                break;
            }
            let jj = self.df_dy(x1, y1);
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
        Ok(())
    }

    /// Calculates the (consistent) modulus `D = dσ/dε`
    ///
    /// `Consistency` here means that derivative is taken from the Backward-Euler update
    pub fn modulus(&self, state: &StressStrainState, delta_strain: f64, inconsistent: bool) -> f64 {
        let ddb = self.f(state.strain, state.stress);
        if inconsistent {
            ddb
        } else {
            let ll = self.df_dx(state.strain, state.stress);
            let jj = self.df_dy(state.strain, state.stress);
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

const SAVE_FIGURE: bool = true;

#[test]
fn test_model_derivatives_1() {
    let li = 10.0;
    let lr = 3.0;
    let y0r = 1.0;
    let a = 30.0;
    let b = 30.0;

    let model = HardeningSofteningModel::new(li, lr, y0r, a, b);

    let yr = model.yr(0.0);
    let dyr_dx = model.dyr_dx(0.0);
    assert_eq!(yr, y0r); // @ x=0.0
    approx_eq(dyr_dx, -lr, 1e-12); // @ x=0.0, dy/dx must equal -lr if b is large enough

    let dy_dx = model.f(0.0, 0.0);
    approx_eq(dy_dx, li, 1e-11); // @ x=0.0, dy/dx must equal li if a is large enough

    let args = &mut 0;
    let x_at = 0.0;
    let y_at = 0.0;

    // check df/dx = d²y/dx²
    let ana = model.df_dx(x_at, y_at);
    let num = deriv1_forward7(x_at, args, |x, _| Ok(model.f(x, y_at))).unwrap();
    println!("df/dx = d²y/dx²:     ana = {}, num = {}", ana, num);
    approx_eq(ana, num, 1e-10);

    // check df/dy = d²y/(dx dy)
    let ana = model.df_dy(0.0, 0.0);
    let num = deriv1_forward7(y_at, args, |y, _| Ok(model.f(x_at, y))).unwrap();
    println!("df/dy = d²y/(dx dy): ana = {}, num = {}", ana, num);
    approx_eq(ana, num, 1e-11);
}

#[test]
fn test_model_derivatives_2() {
    let li = 10.0;
    let lr = 3.0;
    let y0r = 1.0;
    let a = 3.0;
    let b = 3.0;

    let model = HardeningSofteningModel::new(li, lr, y0r, a, b);

    let args = &mut 0;
    let x_at = 0.0;
    let y_at = 0.0;

    // check df/dx = d²y/dx²
    let ana = model.df_dx(x_at, y_at);
    let num = deriv1_forward7(x_at, args, |x, _| Ok(model.f(x, y_at))).unwrap();
    println!("df/dx = d²y/dx²:     ana = {}, num = {}", ana, num);
    approx_eq(ana, num, 1e-12);

    // check df/dy = d²y/(dx dy)
    let ana = model.df_dy(0.0, 0.0);
    let num = deriv1_forward7(y_at, args, |y, _| Ok(model.f(x_at, y))).unwrap();
    println!("df/dy = d²y/(dx dy): ana = {}, num = {}", ana, num);
    approx_eq(ana, num, 1e-11);
}

#[test]
fn test_model_curve_1() {
    let li = 10.0;
    let lr = 3.0;
    let y0r = 1.0;
    let a = 3.0;
    let b = 5.0;

    let mut model = HardeningSofteningModel::new(li, lr, y0r, a, b);
    let mut state = StressStrainState::new();

    let dx = 0.01;
    let np = 50;
    let mut xx = vec![state.strain; np + 1];
    let mut yy = vec![state.stress; np + 1];
    for i in 0..np {
        model.update(&mut state, dx).unwrap();
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
            .save("/tmp/russell_nonlin/test_model_curve_1.svg")
            .unwrap();
    }

    // check results
    for i in 0..np {
        approx_eq(yy[i], yy_ref[i], 0.022);
    }
}

#[test]
fn test_model_curve_and_modulus_1() {
    let li = 10.0;
    let lr = 3.0;
    let y0r = 1.0;
    let a = 3.0;
    let b = 5.0;

    let mut model = HardeningSofteningModel::new(li, lr, y0r, a, b);
    let mut state = StressStrainState::new();

    let ddb0 = model.modulus(&state, 0.0, true); // first inconsistent modulus
    let dd0 = ddb0; // first consistent modulus == first inconsistent modulus

    let dx = 0.01; // local NR won't converge with large delta_strain
    let np = 20;
    let mut xx = vec![state.strain; np + 1];
    let mut yy = vec![state.stress; np + 1];
    let mut ddb = vec![ddb0; np + 1]; // inconsistent modulus
    let mut dd = vec![dd0; np + 1]; // consistent modulus
    let mut dd_num = vec![dd0; np + 1]; // numerical consistent modulus
    for i in 0..np {
        model.update(&mut state, dx).unwrap();
        xx[1 + i] = state.strain;
        yy[1 + i] = state.stress;
        ddb[1 + i] = model.modulus(&state, dx, true);
        dd[1 + i] = model.modulus(&state, dx, false);
        dd_num[1 + i] = model.numerical_modulus(&state, dx).unwrap();
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
            .save("/tmp/russell_nonlin/test_model_curve_and_modulus_1.svg")
            .unwrap();
    }

    // numerical consistent modulus
}
