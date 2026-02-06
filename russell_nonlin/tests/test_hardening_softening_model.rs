use ctm_demo::{Model, ModelType};
use plotpy::{linspace, Curve, Plot};
use russell_lab::{approx_eq, InterpChebyshev, Vector};
use russell_nonlin::{Config, DeltaLambda, IniDir, Output, SoderlindClass, Solver};
use russell_nonlin::{Stats, Status, Stop, StrError, System};
use russell_ode::Method as OdeMethod;
use russell_sparse::Sym;
use std::collections::HashMap;

/// Defines whether to generate the plot or not
const SAVE_FIGURE: bool = false;

#[test]
fn test_hardening_softening_model_full() -> Result<(), StrError> {
    // Settings
    let settings = HashMap::from([
        ("tg_control_atol_and_rtol", 1e-2),
        ("nr_control_n_opt", 3.0),
        ("nr_control_beta", 0.5),
    ]);

    // Input data
    let use_continuous_modulus = false;
    let initial_u = 0.0;
    let initial_l = 0.0;
    let stop = Stop::MaxCompU(0, 0.5);
    let ddl = DeltaLambda::auto(0.1);
    let fig_width = 600.0;

    // Simulation
    let (stats, max_err) = run_hs_model(
        "test_hardening_softening_model_full",
        &settings,
        Some(SoderlindClass::H211PI),
        use_continuous_modulus,
        initial_u,
        initial_l,
        IniDir::Pos,
        stop,
        ddl,
        Status::ContinuedFailure,
        fig_width,
    )?;

    // Check the solver statistics
    assert_eq!(stats.n_accepted, 10);
    assert_eq!(stats.n_rejected, 13);
    assert_eq!(stats.n_steps, 24);

    // Check the maximum error on lambda
    println!("\nMaximum error on lambda = {}\n", max_err);
    assert!(max_err < 0.044, "max_err = {} is greater than the tolerance", max_err);
    Ok(())
}

#[test]
fn test_hardening_softening_model_from_peak() -> Result<(), StrError> {
    // Settings
    let settings = HashMap::new();

    // Input data
    let use_continuous_modulus = false;
    let initial_u = MATHEMATICA_UU[10];
    let initial_l = MATHEMATICA_LL[10];
    let stop = Stop::MaxCompU(0, 0.5);
    let ddl = DeltaLambda::auto(0.1);
    let fig_width = 600.0;

    // Simulation
    let (stats, _) = run_hs_model(
        "test_hardening_softening_model_from_peak",
        &settings,
        None,
        use_continuous_modulus,
        initial_u,
        initial_l,
        IniDir::Pos,
        stop,
        ddl,
        Status::ContinuedFailure,
        fig_width,
    )?;

    // Check the solver statistics
    assert_eq!(stats.n_accepted, 0);
    assert_eq!(stats.n_rejected, 4);
    assert_eq!(stats.n_steps, 5);
    Ok(())
}

#[test]
fn test_hardening_softening_model_from_peak_backward() -> Result<(), StrError> {
    // Settings
    let settings = HashMap::from([("tg_control_atol_and_rtol", 1e-1)]);

    // Input data
    let use_continuous_modulus = false;
    let initial_u = MATHEMATICA_UU[10];
    let initial_l = MATHEMATICA_LL[10];
    let stop = Stop::MinCompU(0, 0.0);
    let ddl = DeltaLambda::auto(1e-3);
    let fig_width = 600.0;

    // Simulation
    let (stats, _) = run_hs_model(
        "test_hs_model_from_peak_backward",
        &settings,
        None,
        use_continuous_modulus,
        initial_u,
        initial_l,
        IniDir::Neg,
        stop,
        ddl,
        Status::Success,
        fig_width,
    )?;

    // Check the solver statistics
    assert_eq!(stats.n_accepted, 13);
    assert_eq!(stats.n_rejected, 0);
    assert_eq!(stats.n_steps, 13);
    Ok(())
}

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
    let nnz = Some(1);
    let sym = Sym::No;
    let mut system = System::new(
        ndim,
        nnz,
        sym,
        |gg, l, _u, args: &mut HSmodelArgs<'a>| {
            let y = args.local_state.stress;
            let q = y;
            gg[0] = q - l;
            Ok(())
        },
        // ∂G/∂u = ∂q/∂u = ∂q/∂y * ∂y/∂x * ∂x/∂u = 1 * ∂y/∂x * B
        // where ∂y/∂x is the consistent tangent modulus given by the hardening-softening model
        // ∂G/∂λ = -1
        move |ggu, ggl, _l, _u, args: &mut HSmodelArgs<'a>| {
            let x = args.local_state.strain;
            let y = args.local_state.stress;
            let dy_dx = if use_continuous_modulus {
                args.model.continuous_modulus(x, y)
            } else {
                let ddx = args.ddx;
                args.model.consistent_tangent_modulus(x, y, ddx)
            };
            ggu.put(0, 0, 1.0 * dy_dx * B).unwrap();
            ggl[0] = -1.0;
            Ok(())
        },
    )?;

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
        .set_update_secondary_state(|do_backup, u0, u1, _l0, _l1, args| {
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
            let stop_gracefully = false;
            Ok(stop_gracefully)
        });

    // Allocate the arguments
    let args = HSmodelArgs {
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

/// Defines the Mathematica reference values (u)
#[rustfmt::skip]
const MATHEMATICA_UU: [f64; 51] = [
    0., 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17,
    0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35,
    0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5,
];

/// Defines the Mathematica reference values (lambda)
#[rustfmt::skip]
const MATHEMATICA_LL: [f64;51] = [
    0., 0.0921923529874959, 0.180995936702942, 0.265135546339778, 0.34303769588157, 0.412909639654633, 0.472933902640526, 0.521575383562834, 0.557924601234833, 0.581937761631112, 0.594452730469534, 0.596972402157104, 0.591326450808571, 0.579354305707458,
    0.562695563124671, 0.542698423576125, 0.520413194773086, 0.496630461299565, 0.471934147069921, 0.446753034002368, 0.421403915953387, 0.39612495266096, 0.371100159763253, 0.346476741387926, 0.322376942591868, 0.298905838568251, 0.276156158431712, 0.254210961837679,
    0.23314478624825, 0.213023733754134, 0.193904874033716, 0.175835268002485, 0.158850897482815, 0.142975694678168, 0.128220878500363, 0.114584707614984, 0.102052740913371, 0.0905985737010805, 0.0801850545040245, 0.070765847217885, 0.062287241401217, 0.0546900747507887,
    0.047911642057484, 0.0418874859643884, 0.0365529932436868, 0.0318447432166188, 0.0277015895758155, 0.0240654765039555, 0.0208820056359409, 0.0181007846054456, 0.0156755917049248,
];

/// Defines the Mathematica reference values (u) (Chebyshev-Gauss-Lobatto spaced)
#[rustfmt::skip]
const _MATHEMATICA_UU_CHEBY: [f64; 51] = [
    0., 0.00049331789293211, 0.00197132467138053, 0.00442818731782782, 0.00785420971784223, 0.0122358709262116, 0.0175558785279371, 0.0237932368834951, 0.0309233299890341, 0.0389180186244962, 0.0477457514062631, 0.0573716893060527, 0.0677578431446471, 
    0.0788632235178278, 0.0906440025628276, 0.103053686926882, 0.116043301255251, 0.129561581474571, 0.143555177108732, 0.157968861828831, 0.172745751406263, 0.187827528208786, 0.203154671353569, 0.218666691608924, 0.234302370117672, 0.25, 0.265697629882328,
    0.281333308391076, 0.296845328646431, 0.312172471791214, 0.327254248593737, 0.34203113817117, 0.356444822891268, 0.370438418525429, 0.383956698744749, 0.396946313073118, 0.409355997437173, 0.421136776482172, 0.432242156855353, 0.442628310693947, 
    0.452254248593737, 0.461081981375504, 0.469076670010966, 0.476206763116505, 0.482444121472063, 0.487764129073788, 0.492145790282158, 0.495571812682172, 0.49802867532862, 0.499506682107068, 0.5,
];

/// Defines the Mathematica reference values (lambda) (Chebyshev-Gauss-Lobatto spaced)
#[rustfmt::skip]
const MATHEMATICA_LL_CHEBY: [f64; 51] = [
    0., 0.00461145774987275, 0.0183917943986726, 0.0411734321083657, 0.0726576679498955, 0.112382088901056, 0.15966784701565, 0.213540092031638, 0.272616001576795, 0.334963149055844, 0.397955651173981, 0.458206157740029, 0.511718529403169, 0.554421318733772,
    0.583076843361304, 0.596201018002729, 0.594416206769029, 0.579988368458721, 0.555908222381671, 0.525083109916001, 0.489919864817226, 0.452247353390525, 0.413412038517492, 0.374416443835349, 0.336037446855862, 0.298905810706376, 0.263548999392447,
    0.230406051609337, 0.199824939630881, 0.172052141978139, 0.147223261325688, 0.12536090930094, 0.10638296215564, 0.0901204945033013, 0.076342103793569, 0.0647800264378275, 0.0551539789866797, 0.0471900303273264, 0.0406337043155176, 0.0352573979245923,
    0.0308635706461138, 0.027284722692107, 0.0243814551821793, 0.0220395919122741, 0.0201669581120976, 0.018690170911755, 0.017551888191699, 0.0167083714526033, 0.0161275472959352, 0.0157875259071178, 0.0156755778523422,
];

/// Returns a Chebyshev interpolation of the Mathematica reference function
fn get_mathematica_ref_function() -> Result<InterpChebyshev, StrError> {
    let nn_max = 100;
    let tol = 1e-12;
    let mut interp = InterpChebyshev::new(nn_max, 0.0, 0.5)?;
    interp.adapt_data(tol, &MATHEMATICA_LL_CHEBY)?;
    Ok(interp)
}

/// Plot the results
fn do_plot(name: &str, uu: &Vec<f64>, ll: &Vec<f64>, fig_width: f64) -> Result<(), StrError> {
    // draw reference curve
    let mut curve_ref = Curve::new();
    let lambda_ref = get_mathematica_ref_function()?;
    let uu_ref = linspace(0.0, 0.5, 100);
    let ll_ref: Vec<_> = uu_ref.iter().map(|&u| lambda_ref.eval(u).unwrap()).collect();
    curve_ref
        .set_label("reference")
        .set_line_color("#40915d")
        .set_line_style("-")
        .draw(&uu_ref, &ll_ref);

    // draw numerical solution: load-displacement curve
    let mut load_displacement_curve = Curve::new();
    load_displacement_curve
        .set_marker_style(".")
        .set_line_color("#d60943")
        .draw(uu, ll);

    // generate the plot
    let fig_height = fig_width * 0.618;
    let mut plot = Plot::new();
    plot.add(&curve_ref)
        .add(&load_displacement_curve)
        .grid_labels_legend("$u$", "$\\lambda$")
        .set_figure_size_points(fig_width, fig_height)
        .save(&format!("/tmp/russell_nonlin/{}.svg", name))?;
    Ok(())
}

fn do_plot_stepsizes(name: &str, stepsizes: &[f64]) {
    let hh = &stepsizes[1..]; // the first one is duplicated
    let n = hh.len();
    let x = linspace(1.0, n as f64, n);

    let mut curve = Curve::new();
    curve.set_label("stepsize").set_line_style("-").set_marker_style(".");
    curve.draw(&x.as_slice(), &hh);

    let mut plot = Plot::new();
    plot.set_labels("step number", "stepsize $h$")
        .add(&curve)
        .save(&format!("/tmp/russell_nonlin/{}_stepsizes.svg", name))
        .unwrap();
}

/// Defines a function to run the Hardening-Softening model
///
/// Returns `(stats, max_err)` where `max_err` is the maximum error in lambda
fn run_hs_model(
    name: &str,
    settings: &HashMap<&str, f64>,
    so_class: Option<SoderlindClass>,
    use_continuous_modulus: bool,
    initial_u: f64,
    initial_l: f64,
    direction: IniDir,
    stop: Stop,
    ddl: DeltaLambda,
    expected_status: Status,
    fig_width: f64,
) -> Result<(Stats, f64), StrError> {
    // Allocate the system and arguments
    let (system, mut args) = new_hs_model_problem(use_continuous_modulus)?;

    // Prepare the configuration
    let mut config = Config::new();
    config
        .set_verbose(true, true, true)
        .set_debug_predictor(true)
        .set_record_iterations_residuals(true);

    // Override the default settings
    for (&key, value) in settings.iter() {
        match key {
            "nr_control_n_opt" => config.set_nr_control_n_opt(*value as usize),
            "nr_control_beta" => config.set_nr_control_beta(*value),
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

    // Set the Soderlind class, if given
    if let Some(so_class) = so_class {
        config.set_tg_control_soderlind(so_class);
    }

    // Define the nonlinear solver
    let mut solver = Solver::new(&config, system)?;

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
    let mut u = Vector::from(&[initial_u]);
    let mut l = initial_l;
    args.local_state.strain = u[0] * B;
    args.local_state.stress = l;

    // Perform the numerical continuation
    let status = solver.solve(&mut args, &mut u, &mut l, direction, stop, ddl, Some(out))?;
    assert_eq!(status, expected_status);

    // results
    let uu = out.get_u_values(0);
    let ll = out.get_l_values();

    // plot
    if SAVE_FIGURE {
        let hh = out.get_h_values();
        do_plot(name, uu, ll, fig_width)?;
        do_plot_stepsizes(name, &hh);
    }

    // check the results against reference values
    let lambda_ref = get_mathematica_ref_function()?;
    let mut max_err = 0.0;
    for (i, &u) in uu.iter().enumerate() {
        let l_ref = lambda_ref.eval(u)?;
        let l = ll[i];
        let err = f64::abs(l - l_ref);
        max_err = f64::max(max_err, err);
    }

    // check if u matches strain and λ matches stress
    for (i, &u) in uu.iter().enumerate() {
        approx_eq(u, args.xx[i], 1e-14);
        approx_eq(ll[i], args.yy[i], 1e-7);
    }

    // done
    Ok((solver.get_stats().clone(), max_err))
}
