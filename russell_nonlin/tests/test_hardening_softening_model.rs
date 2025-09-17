use ctm_demo::{Model, ModelType};
use plotpy::{Curve, Plot};
use russell_nonlin::{AutoStep, Config, Direction, Method, Output, Solver, State, Stop, System};
use russell_ode::Method as OdeMethod;
use russell_sparse::Sym;
use std::collections::HashMap;

/// Defines the strain (x) and stress (y) state
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

const USE_CONTINUOUS_MODULUS: bool = false;

const SAVE_FIGURE: bool = true;

#[test]
fn test_hardening_softening_model() {
    // Define the arguments for the nonlinear problem
    struct Args {
        model: Model<'static>,
        local_state: StressStrainState,
        ddx: f64,     // increment in strain
        xx: Vec<f64>, // strain history (for plotting)
        yy: Vec<f64>, // stress history (for plotting)
    }

    // Constants
    const B: f64 = 1.0; // x = strain = B * u

    // Nonlinear problem
    // G(u, λ) = q(u) - λ = 0
    // q(u) = y  where  y is stress
    // and y(x(u)) is given by the hardening-softening model,
    // where x is strain. Here, x = B u
    let ndim = 1;
    let mut system = System::new(ndim, |gg, l, _u, args: &mut Args| {
        let y = args.local_state.stress;
        let q = y;
        gg[0] = q - l;
        Ok(())
    })
    .unwrap();

    // Function to compute Gu = ∂G/∂u
    // ∂G/∂u = ∂q/∂u = ∂q/∂y * ∂y/∂x * ∂x/∂u
    //       = 1 * ∂y/∂x * B
    // where ∂y/∂x is the consistent tangent modulus given by the hardening-softening model
    let nnz = Some(1);
    let sym = Sym::No;
    system
        .set_calc_ggu(nnz, sym, |ggu, _l, _u, args: &mut Args| {
            let x = args.local_state.strain;
            let y = args.local_state.stress;
            let dy_dx = if USE_CONTINUOUS_MODULUS {
                args.model.continuous_modulus(x, y)
            } else {
                let ddx = args.ddx;
                args.model.consistent_tangent_modulus(x, y, ddx)
            };
            ggu.put(0, 0, 1.0 * dy_dx * B).unwrap();
            Ok(())
        })
        .unwrap();

    // Function to compute Gl = ∂G/∂λ
    // ∂G/∂λ = -1
    system.set_calc_ggl(|ggl, _l, _u, _args: &mut Args| {
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
        .set_update_secondary_state(|first_iteration, u0, u1, args| {
            if first_iteration {
                args.local_state.iter_backup();
            } else {
                args.local_state.iter_restore();
            }
            let x = &mut args.local_state.strain;
            let y = &mut args.local_state.stress;
            args.ddx = B * (u1[0] - u0[0]);
            args.model.backward_euler_update(x, y, args.ddx)?;
            // println!("u1 = {}, x = {}, y = {}", u1[0], x, y);
            Ok(())
        });

    // Allocate the arguments
    let mut args = Args {
        model: Model::new(
            ModelType::HardeningSoftening,
            HashMap::from([("li", 10.0), ("lr", 3.0), ("y0r", 1.0), ("a", 3.0), ("b", 5.0)]),
            OdeMethod::DoPri5,
        )
        .unwrap(),
        local_state: StressStrainState::new(),
        ddx: 0.0,
        xx: Vec::new(),
        yy: Vec::new(),
    };

    // Prepare the configuration
    let mut config = Config::new(Method::Arclength);
    // let mut config = Config::new(Method::Natural);
    config
        .set_verbose(true, true, true)
        .set_h_ini(0.1)
        // .set_alpha_max(3.0)
        // .set_sigma_max(0.1)
        .set_record_iterations_residuals(true);
    // .set_hide_timings(true)
    // .set_record_iterations_residuals(true)
    // .set_allowed_continued_divergence(1)
    // .set_sigma_max(0.3)
    // .set_h_ini(0.04);
    // .set_h_ini(0.4743);
    // .set_allowed_continued_divergence(3);

    // Define the nonlinear solver
    let mut solver = Solver::new(config, system).unwrap();

    // Define the output data
    let out = &mut Output::new();
    out.set_callback(|_stats, _u, _l, _h, args: &mut Args| {
        let x = &mut args.local_state.strain;
        let y = &mut args.local_state.stress;
        args.xx.push(*x);
        args.yy.push(*y);
        Ok(false)
    });
    out.set_recording(true, &[0], &[0]);

    // Allocate the (global) state
    let mut state = State::new(ndim);

    // Perform the numerical continuation
    let status = solver
        .solve(
            &mut args,
            &mut state,
            Direction::Pos,
            // Stop::Steps(200),
            // AutoStep::No(0.01),
            Stop::Steps(10),
            AutoStep::No(0.05),
            // Stop::Steps(200),
            // AutoStep::Yes,
            Some(out),
        )
        .unwrap();
    println!("status = {:?}", status);

    // results
    let u = out.get_u_values(0);
    let l = out.get_l_values();

    if SAVE_FIGURE {
        let mut load_displacement_curve = Curve::new();
        let mut stress_strain_curve = Curve::new();
        load_displacement_curve.draw(u, l);
        stress_strain_curve.draw(&args.xx, &args.yy);
        let mut plot = Plot::new();
        plot.set_subplot(1, 2, 1)
            .add(&load_displacement_curve)
            .grid_labels_legend("$u$", "$\\lambda$")
            .set_subplot(1, 2, 2)
            .add(&stress_strain_curve)
            .grid_labels_legend("$\\varepsilon$", "$\\sigma$")
            .set_figure_size_points(800.0, 350.0)
            .save("/tmp/russell_nonlin/test_hardening_softening_model_1.svg")
            .unwrap();
    }
}
