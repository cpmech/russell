#![allow(unused)]

use ctm_demo::{Model, ModelType};
use plotpy::{Canvas, Curve, Plot, RayEndpoint};
use russell_nonlin::{AutoStep, Config, Direction, Method, Output, Solver, State};
use russell_nonlin::{Stats, Status, Stop, StrError, System};
use russell_ode::Method as OdeMethod;
use russell_sparse::Sym;
use std::collections::HashMap;

/// Defines the stress(y)-strain(x) state for the Hardening-Softening model
#[derive(Clone)]
struct StressStrainState {
    strain: f64,          // x
    stress: f64,          // y
    strain_bkp_step: f64, // backup of strain (steps)
    stress_bkp_step: f64, // backup of stress (steps)
    strain_bkp_iter: f64, // backup of strain (iterations)
    stress_bkp_iter: f64, // backup of stress (iterations)
}

impl StressStrainState {
    /// Allocates a new instance
    pub fn new() -> Self {
        StressStrainState {
            strain: 0.0,
            stress: 0.0,
            strain_bkp_step: 0.0,
            stress_bkp_step: 0.0,
            strain_bkp_iter: 0.0,
            stress_bkp_iter: 0.0,
        }
    }

    /// Creates a backup of the current state (steps)
    pub fn step_backup(&mut self) {
        self.strain_bkp_step = self.strain;
        self.stress_bkp_step = self.stress;
    }

    /// Restores the state from the backup (steps)
    pub fn step_restore(&mut self) {
        self.strain = self.strain_bkp_step;
        self.stress = self.stress_bkp_step;
    }

    /// Creates a backup of the current state (iterations)
    pub fn iter_backup(&mut self) {
        self.strain_bkp_iter = self.strain;
        self.stress_bkp_iter = self.stress;
    }

    /// Restores the state from the backup (iterations)
    pub fn iter_restore(&mut self) {
        self.strain = self.strain_bkp_iter;
        self.stress = self.stress_bkp_iter;
    }
}

/// Defines the arguments for the nonlinear problem with the Hardening-Softening model
struct HSmodelArgs<'a> {
    model: Model<'a>,
    local_state: StressStrainState,
    ddx: f64,               // increment in strain
    xx: Vec<f64>,           // strain history (for plotting)
    yy: Vec<f64>,           // stress history (for plotting)
    xx_predictor: Vec<f64>, // strain predictor history (for plotting)
    yy_predictor: Vec<f64>, // stress predictor history (for plotting)
}

// Strain-displacement coefficient: strain = x = B * u
const B: f64 = 1.0;

/// Allocates a new system representing the Hardening-Softening nonlinear problem
///
/// Nonlinear problem:
///
/// ```text
/// G(u, λ) = q(u) - λ = 0
/// q(u) = y
/// ```
///
/// where y is stress and y(x(u)) is given by the hardening-softening model.
/// The strain (x) is related to the displacement via
///
/// ```text
///  x = B u
/// Here, B = 1
/// ```
fn new_hs_model_problem<'a>(
    use_continuous_modulus: bool,
) -> Result<(System<'a, HSmodelArgs<'a>>, HSmodelArgs<'a>), StrError> {
    // Define G(u, λ)
    let ndim = 1;
    let mut system = System::new(ndim, |gg, l, _u, args: &mut HSmodelArgs<'a>| {
        let y = args.local_state.stress;
        let q = y;
        gg[0] = q - l;
        Ok(())
    })?;

    // Function to compute Gu = ∂G/∂u
    // ∂G/∂u = ∂q/∂u = ∂q/∂y * ∂y/∂x * ∂x/∂u
    //       = 1 * ∂y/∂x * B
    // where ∂y/∂x is the consistent tangent modulus given by the hardening-softening model
    let nnz = Some(1);
    let sym = Sym::No;
    system.set_calc_ggu(nnz, sym, move |ggu, _l, _u, args: &mut HSmodelArgs<'a>| {
        let x = args.local_state.strain;
        let y = args.local_state.stress;
        let dy_dx = if use_continuous_modulus {
            args.model.continuous_modulus(x, y)
        } else {
            let ddx = args.ddx;
            args.model.consistent_tangent_modulus(x, y, ddx)
        };
        ggu.put(0, 0, 1.0 * dy_dx * B).unwrap();
        Ok(())
    })?;

    // Function to compute Gl = ∂G/∂λ
    // ∂G/∂λ = -1
    system.set_calc_ggl(|ggl, _l, _u, _args: &mut HSmodelArgs<'a>| {
        ggl[0] = -1.0;
        Ok(())
    });

    // Set the callback functions to manage the local state
    system
        .set_backup_secondary_state(|args| {
            args.local_state.step_backup();
        })
        .set_restore_secondary_state(|args| {
            args.local_state.step_restore();
        })
        .set_prepare_to_iterate(|args| {
            args.ddx = 0.0;
        })
        .set_update_secondary_state(|do_backup, u0, u1, args| {
            if do_backup {
                args.local_state.iter_backup();
            } else {
                args.local_state.iter_restore();
            }
            let x = &mut args.local_state.strain;
            let y = &mut args.local_state.stress;
            args.ddx = B * (u1[0] - u0[0]);
            args.model.backward_euler_update(x, y, args.ddx)?;
            if do_backup {
                args.xx_predictor.push(*x);
                args.yy_predictor.push(*y);
            }
            if *y < 0.0 {
                return Err("sup: negative stress is not allowed");
            }
            Ok(())
        });

    // Allocate the arguments
    let mut args = HSmodelArgs {
        model: Model::new(
            ModelType::HardeningSoftening,
            HashMap::from([("li", 10.0), ("lr", 3.0), ("y0r", 1.0), ("a", 3.0), ("b", 5.0)]),
            OdeMethod::DoPri5,
        )?,
        local_state: StressStrainState::new(),
        ddx: 0.0,
        xx: Vec::new(),
        yy: Vec::new(),
        xx_predictor: Vec::new(),
        yy_predictor: Vec::new(),
    };

    // done
    Ok((system, args))
}

// Mathematica reference values (u)
#[rustfmt::skip]
const MATHEMATICA_UU: [f64; 51] = [
    0., 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17,
    0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35,
    0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5,
];

// Mathematica reference values (lambda)
#[rustfmt::skip]
const MATHEMATICA_LL: [f64;51] = [
    0., 0.0921923529874959, 0.180995936702942, 0.265135546339778, 0.34303769588157, 0.412909639654633, 0.472933902640526, 0.521575383562834, 0.557924601234833, 0.581937761631112, 0.594452730469534, 0.596972402157104, 0.591326450808571, 0.579354305707458,
    0.562695563124671, 0.542698423576125, 0.520413194773086, 0.496630461299565, 0.471934147069921, 0.446753034002368, 0.421403915953387, 0.39612495266096, 0.371100159763253, 0.346476741387926, 0.322376942591868, 0.298905838568251, 0.276156158431712, 0.254210961837679,
    0.23314478624825, 0.213023733754134, 0.193904874033716, 0.175835268002485, 0.158850897482815, 0.142975694678168, 0.128220878500363, 0.114584707614984, 0.102052740913371, 0.0905985737010805, 0.0801850545040245, 0.070765847217885, 0.062287241401217, 0.0546900747507887,
    0.047911642057484, 0.0418874859643884, 0.0365529932436868, 0.0318447432166188, 0.0277015895758155, 0.0240654765039555, 0.0208820056359409, 0.0181007846054456, 0.0156755917049248,
];

/// Plot the results
fn do_plot(name: &str, uu: &Vec<f64>, ll: &Vec<f64>, fig_width: f64) -> Result<(), StrError> {
    let mut curve_ref = Curve::new();
    curve_ref
        .set_label("reference")
        .set_line_color("#40915d")
        .set_line_style("-")
        .draw(&MATHEMATICA_UU, &MATHEMATICA_LL);

    /*
    // draw tangent vectors
    let mut arrows = Canvas::new();
    arrows
        .set_arrow_scale(10.0)
        .set_edge_color("None")
        .set_face_color("#414141ff");
    for i in 0..(uu.len() - 1) {
        let sig = stepsizes[i];
        let xf = uu[i] + sig * duds[i];
        let yf = ll[i] + sig * dlds[i];
        arrows.draw_arrow(uu[i], ll[i], xf, yf);
    }

    // draw hyperplanes
    let mut hyperplanes = Curve::new();
    hyperplanes.set_line_style("--").set_line_color("#d0d0d0");
    for i in 0..uu.len() {
        let xa = uu[i] + sigma * duds[i];
        let ya = ll[i] + sigma * dlds[i];
        let phi = f64::atan2(dlds[i], duds[i]);
        let xb = xa - f64::sin(phi);
        let yb = ya + f64::cos(phi);
        let ep = RayEndpoint::Coords(xb, yb);
        hyperplanes.draw_ray(xa, ya, ep);
    }
    */

    // draw numerical solution: load-displacement curve
    let mut load_displacement_curve = Curve::new();
    load_displacement_curve
        .set_marker_style(".")
        .set_line_color("#d60943")
        .draw(uu, ll);

    /*
    // add predictor points to the load-displacement curve
    let mut predictor_curve1 = Curve::new();
    predictor_curve1
        .set_line_style("None")
        .set_marker_style("*")
        .draw(&u0_predictor, &l_predictor);

    // draw numerical solution: stress-strain curve
    let mut stress_strain_curve = Curve::new();
    stress_strain_curve.draw(&args.xx, &args.yy);

    // add predictor points to the stress-strain curve
    let mut predictor_curve2 = Curve::new();
    predictor_curve2
        .set_line_style("None")
        .set_marker_style("*")
        .draw(&args.xx_predictor, &args.yy_predictor);
    */

    // generate the plot
    let fig_height = fig_width * 0.618;
    let mut plot = Plot::new();
    plot
        // .add(&arrows)
        .add(&curve_ref)
        .add(&load_displacement_curve)
        // .add(&predictor_curve1)
        .grid_labels_legend("$u$", "$\\lambda$")
        // .set_range(0.088, 0.121, 0.57, 0.61)
        .set_figure_size_points(fig_width, fig_height)
        .save(&format!("/tmp/russell_nonlin/{}.svg", name))?;

    /*
    // generate the plot
    let mut plot = Plot::new();
    plot.set_subplot(1, 2, 1)
        // .add(&hyperplanes)
        .add(&arrows)
        .add(&curve_ref)
        .add(&load_displacement_curve)
        .add(&predictor_curve1)
        .grid_labels_legend("$u$", "$\\lambda$")
        // .set_range(-0.025, 0.525, -0.025, 0.625)
        // .set_range(0.07, 0.15, 0.54, 0.63)
        // .set_range(0.03, 0.2, 0.34, 0.63)
        .set_subplot(1, 2, 2)
        .add(&curve_ref)
        .add(&predictor_curve2)
        .add(&stress_strain_curve)
        .grid_labels_legend("$\\varepsilon$", "$\\sigma$")
        // .set_range(-0.025, 0.525, -0.025, 0.625)
        // .set_range(0.05, 0.2, 0.4, 0.6)
        .set_figure_size_points(800.0, 350.0)
        .save("/tmp/russell_nonlin/test_hardening_softening_model_1.svg")?;
    */
    Ok(())
}

// Whether to save the figure
const SAVE_FIGURE: bool = true;

// Function to run the Hardening-Softening model
fn run_hs_model(
    name: &str,
    settings: &HashMap<&str, f64>,
    use_continuous_modulus: bool,
    initial_u: f64,
    initial_l: f64,
    stop: Stop,
    auto: AutoStep,
    fig_width: f64,
) -> Result<Stats, StrError> {
    // Allocate the system and arguments
    let (system, mut args) = new_hs_model_problem(use_continuous_modulus)?;
    let ndim = system.get_ndim();

    // Prepare the configuration
    let mut config = Config::new(Method::Arclength);
    // let mut config = Config::new(Method::Natural);
    config
        .set_verbose(true, true, true)
        .set_h_ini(0.1)
        .set_h_min_allowed(1e-10)
        .set_alpha_max(5.0)
        .set_sigma_max(0.1)
        .set_debug_predictor(true)
        .set_record_stepsizes(true)
        .set_record_iterations_residuals(true);

    // Override the default settings
    for (&key, value) in settings.iter() {
        match key {
            "tg_control_atol" => config.set_tg_control_atol(*value),
            "tg_control_rtol" => config.set_tg_control_rtol(*value),
            "tg_control_atol_and_rtol" => config.set_tg_control_atol_and_rtol(*value),
            "tg_control_beta1" => config.set_tg_control_beta1(*value),
            "tg_control_beta2" => config.set_tg_control_beta2(*value),
            "tg_control_beta3" => config.set_tg_control_beta3(*value),
            "tg_control_alpha2" => config.set_tg_control_alpha2(*value),
            "tg_control_alpha3" => config.set_tg_control_alpha3(*value),
            _ => return Err("invalid setting"),
        };
    }

    // Define the nonlinear solver
    let mut solver = Solver::new(config, system)?;

    // Define the output data
    let out = &mut Output::new();
    out.set_callback(|_stats, _u, _l, _h, args: &mut HSmodelArgs| {
        let x = &mut args.local_state.strain;
        let y = &mut args.local_state.stress;
        args.xx.push(*x);
        args.yy.push(*y);
        Ok(false)
    });
    out.set_recording(true, &[0], &[0]);

    // Allocate the (global) state
    let mut state = State::new(ndim);

    // Initial values
    state.u[0] = initial_u;
    state.l = initial_l;
    args.local_state.strain = state.u[0] * B;
    args.local_state.stress = state.l;

    // Perform the numerical continuation
    let status = solver.solve(&mut args, &mut state, Direction::Pos, stop, auto, Some(out))?;
    assert_eq!(status, Status::Success);

    // results
    let uu = out.get_u_values(0);
    let ll = out.get_l_values();

    /*
    let duds = out.get_duds_values(0);
    let dlds = out.get_dlds_values();
    let (l_predictor, u0_predictor, _) = solver.get_debug_predictor_values();
    let stats = solver.get_stats();
    let stepsizes = stats.get_stepsizes();
    */

    // plot
    if SAVE_FIGURE {
        do_plot(name, uu, ll, fig_width)?;
    }

    // done
    Ok(solver.get_stats().clone())
}

#[test]
fn test_hardening_softening_model_full() -> Result<(), StrError> {
    // Stop::Steps(33),
    // AutoStep::No(sigma),
    // AutoStep::No(0.0191),
    // Stop::Steps(8),
    // AutoStep::No(sigma),
    // Stop::Steps(10),
    // AutoStep::No(0.05),
    // Stop::Steps(20),

    let settings = HashMap::from([
        ("tg_control_atol_and_rtol", 1e-2),
        ("tg_control_beta1", 1.0 / 3.0),
        ("tg_control_beta2", 1.0 / 18.0),
        ("tg_control_beta3", -5.0 / 18.0),
        ("tg_control_alpha2", -5.0 / 6.0),
        ("tg_control_alpha3", -1.0 / 6.0),
    ]);

    let use_continuous_modulus = false;

    let initial_u = 0.0;
    let initial_l = 0.0;

    let stop = Stop::Component(0, 0.5);
    let auto = AutoStep::Yes;

    let fig_width = 600.0;

    let stats = run_hs_model(
        "test_hs_model_full",
        &settings,
        use_continuous_modulus,
        initial_u,
        initial_l,
        stop,
        auto,
        fig_width,
    )?;

    Ok(())
}

#[test]
fn test_hardening_softening_model_from_peak() -> Result<(), StrError> {
    // Stop::Steps(33),
    // AutoStep::No(sigma),
    // AutoStep::No(0.0191),
    // Stop::Steps(8),
    // AutoStep::No(sigma),
    // Stop::Steps(10),
    // AutoStep::No(0.05),
    // Stop::Steps(20),

    let settings = HashMap::new();

    let use_continuous_modulus = false;

    let initial_u = MATHEMATICA_UU[10];
    let initial_l = MATHEMATICA_LL[10];

    let stop = Stop::Component(0, 0.5);
    let auto = AutoStep::Yes;

    let fig_width = 600.0;

    let stats = run_hs_model(
        "test_hs_model_from_peak",
        &settings,
        use_continuous_modulus,
        initial_u,
        initial_l,
        stop,
        auto,
        fig_width,
    )?;

    Ok(())
}
