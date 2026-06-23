use super::StrError;
use super::{Dahlquist, HardeningSoftening, ModelTrait, ModelType};
use russell_lab::Vector;
use russell_ode::{Method, OdeSolver, Params, System};
use std::collections::HashMap;
use std::sync::Arc;

const N_ITERATIONS_MAX: usize = 20;
const BE_TOLERANCE: f64 = 1e-8;
const DELTA: f64 = 1e-5;

pub struct ArgsForODE {
    model: Arc<dyn ModelTrait>,
    x0: f64,
    ddx: f64,
}

/// Represents a stress-strain model with x being strain and y being stress
pub struct Model<'a> {
    actual: Arc<dyn ModelTrait>,
    ode_solver: OdeSolver<'a, ArgsForODE>,
}

impl<'a> Model<'a> {
    /// Allocates a new instance
    pub fn new(model_type: ModelType, params: HashMap<&str, f64>, ode_method: Method) -> Result<Self, StrError> {
        let actual: Arc<dyn ModelTrait> = match model_type {
            ModelType::Dahlquist => Arc::new(Dahlquist::new(params)?),
            ModelType::HardeningSoftening => Arc::new(HardeningSoftening::new(params)?),
        };
        let ode_params = Params::new(ode_method);
        let ode_system = System::new(1, |f, t, y, args: &mut ArgsForODE| {
            let x = args.x0 + t * args.ddx;
            f[0] = args.model.calc_f(x, y[0]) * args.ddx;
            Ok(())
        });
        let ode_solver = OdeSolver::new(ode_params, ode_system)?;
        Ok(Model { actual, ode_solver })
    }

    /// Performs a backward Euler update
    ///
    /// Calculates x_new and y_new from the total strain increment `Δx`
    pub fn backward_euler_update(&self, x: &mut f64, y: &mut f64, ddx: f64) -> Result<(), StrError> {
        let x0 = *x;
        let y0 = *y;
        let x1 = x0 + ddx;
        let f_trial = self.actual.calc_f(x1, y0);
        let y_trial = y0 + ddx * f_trial;
        *x = x1;
        *y = y_trial;
        let mut converged = false;
        for _ in 0..N_ITERATIONS_MAX {
            let f1 = self.actual.calc_f(*x, *y);
            let r1 = *y - y0 - ddx * f1;
            if f64::abs(r1) < BE_TOLERANCE {
                converged = true;
                break;
            }
            let jj1 = self.actual.calc_jj(*x, *y);
            let dy = -r1 / (1.0 - ddx * jj1);
            *y += dy;
        }
        if !converged {
            return Err("Backward Euler did not converge");
        }
        Ok(())
    }

    /// Performs an update using the ODE solver
    pub fn ode_update(&mut self, x: &mut f64, y: &mut f64, ddx: f64) -> Result<(), StrError> {
        let mut yy = Vector::from(&[*y]);
        let mut args = ArgsForODE {
            model: self.actual.clone(),
            x0: *x,
            ddx,
        };
        self.ode_solver.solve(&mut yy, 0.0, 1.0, None, &mut args, None)?;
        *x = args.x0 + ddx;
        *y = yy[0];
        Ok(())
    }

    /// Returns the continuous modulus f = dy/dx
    pub fn continuous_modulus(&self, x: f64, y: f64) -> f64 {
        self.actual.calc_f(x, y)
    }

    /// Calculates the consistent tangent modulus @ the update point (x1, y1)
    pub fn consistent_tangent_modulus(&self, x1: f64, y1: f64, ddx: f64) -> f64 {
        let f1 = self.actual.calc_f(x1, y1);
        let ll1 = self.actual.calc_ll(x1, y1);
        let jj1 = self.actual.calc_jj(x1, y1);
        (f1 + ddx * ll1) / (1.0 - ddx * jj1)
    }

    /// Approximates the consistent tangent modulus @ the update point (x1, y1), given the previous point (x0, y0)
    pub fn numerical_consistent_tangent_modulus(
        &mut self,
        x0: f64,
        y0: f64,
        ddx: f64,
        use_ode_solution: bool,
    ) -> Result<f64, StrError> {
        let mut xa = x0;
        let mut ya = y0;
        let mut xb = x0;
        let mut yb = y0;
        if use_ode_solution {
            self.ode_update(&mut xa, &mut ya, ddx)?;
            self.ode_update(&mut xb, &mut yb, ddx + DELTA)?;
        } else {
            self.backward_euler_update(&mut xa, &mut ya, ddx)?;
            self.backward_euler_update(&mut xb, &mut yb, ddx + DELTA)?;
        }
        Ok((yb - ya) / (xb - xa))
    }

    /// Performs a simulation of the model
    ///
    /// Returns `(xx, yy_be, yy_ode, com_list, ctm_list, num_ctm_list, num_ctm_ode_list)` where:
    ///
    /// - `xx` is the vector of x values (strain)
    /// - `yy_be` is the vector of y values (stress) calculated with backward Euler
    /// - `yy_ode` is the vector of y values (stress) calculated with the ODE solver
    /// - `com_list` is the list of continuous moduli
    /// - `ctm_list` is the list of consistent tangent moduli
    /// - `num_ctm_list` is the list of numerical consistent tangent moduli
    /// - `num_ctm_ode_list` is the list of numerical consistent tangent moduli calculated with the ODE solver
    pub fn simulate(
        &mut self,
        x_ini: f64,
        y_ini: f64,
        ddx: f64,
        nd: usize,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>), StrError> {
        // Initial values
        let mut x_be = x_ini;
        let mut x_ode = x_ini;
        let mut y_be = y_ini;
        let mut y_ode = y_ini;

        // Perform the backward Euler update
        let mut xx = vec![0.0; nd + 1];
        let mut yy_be = vec![0.0; nd + 1];
        let mut yy_ode = vec![0.0; nd + 1];
        let mut com_list = vec![0.0; nd + 1];
        let mut ctm_list = vec![0.0; nd + 1];
        let mut num_ctm_list = vec![0.0; nd + 1];
        let mut num_ctm_ode_list = vec![0.0; nd + 1];
        let com = self.continuous_modulus(x_be, y_be);
        xx[0] = x_be;
        yy_be[0] = y_be;
        yy_ode[0] = y_ode;
        com_list[0] = com;
        ctm_list[0] = com;
        num_ctm_list[0] = com;
        num_ctm_ode_list[0] = com;
        for k in 1..=nd {
            // x is x0 and y is y0
            let x0 = x_be;
            let y0 = y_be;
            // perform the backward Euler update
            self.backward_euler_update(&mut x_be, &mut y_be, ddx)?;
            // perform the ODE update
            self.ode_update(&mut x_ode, &mut y_ode, ddx)?;
            // x is now x1 and y is now y1
            let x1 = x_be;
            let y1 = y_be;
            // calculate the continuous modulus
            let com = self.continuous_modulus(x1, y1);
            // calculate the consistent tangent modulus
            let ctm = self.consistent_tangent_modulus(x1, y1, ddx);
            let num_ctm = self.numerical_consistent_tangent_modulus(x0, y0, ddx, false)?;
            let num_ctm_ode = self.numerical_consistent_tangent_modulus(x0, y0, ddx, true)?;
            // store the results
            xx[k] = x1;
            yy_be[k] = y1;
            yy_ode[k] = y_ode;
            com_list[k] = com;
            ctm_list[k] = ctm;
            num_ctm_list[k] = num_ctm;
            num_ctm_ode_list[k] = num_ctm_ode;
        }

        // Return the results
        Ok((xx, yy_be, yy_ode, com_list, ctm_list, num_ctm_list, num_ctm_ode_list))
    }
}

// tests /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{Dahlquist, Model, ModelType};
    use plotpy::{Curve, Plot, linspace};
    use russell_lab::approx_eq;
    use russell_ode::Method;
    use std::collections::HashMap;

    const SAVE_FIGURE: bool = false;

    #[test]
    fn test_dahlquist() {
        // Allocate the model
        let lambda = 5.0;
        let method = Method::DoPri5;
        let mut model = Model::new(ModelType::Dahlquist, HashMap::from([("lambda", lambda)]), method).unwrap();

        // Set initial conditions
        let x_ini = 0.0;
        let y_ini = Dahlquist::analytical_y(lambda, x_ini);

        // Define constants for the backward Euler update
        let ddx = 0.1;
        let nd = 5;

        // Perform the backward Euler update
        let (xx, yy, yy_ode, _, ctm_list, num_ctm_list, num_ctm_ode_list) =
            model.simulate(x_ini, y_ini, ddx, nd).unwrap();

        // Generate the plot
        if SAVE_FIGURE {
            // Fine y(x) curve
            let xx_fine = linspace(0.0, 0.5, 101);
            let yy_fine = xx_fine
                .iter()
                .map(|&x| Dahlquist::analytical_y(lambda, x))
                .collect::<Vec<_>>();
            let mut curve_fine = Curve::new();
            curve_fine.set_label("Analytical Solution").draw(&xx_fine, &yy_fine);

            // ODE solution
            let mut curve_ode = Curve::new();
            curve_ode
                .set_label(&format!("{:?} solution", method))
                .set_line_style("--")
                .draw(&xx, &yy_ode);

            // Continuous modulus curve
            let com_list = xx_fine
                .iter()
                .enumerate()
                .map(|(i, &x)| model.continuous_modulus(x, yy_fine[i]))
                .collect::<Vec<_>>();
            let mut curve_com = Curve::new();
            curve_com.set_label("Continuous Modulus").draw(&xx_fine, &com_list);

            // Backward Euler curve y(x)
            let mut curve = Curve::new();
            curve
                .set_label("Backward Euler")
                .set_line_style("None")
                .set_marker_style("o")
                .draw(&xx, &yy);

            // Consistent tangent modulus curve
            let mut curve_ctm = Curve::new();
            curve_ctm
                .set_label("Consistent Tangent Modulus")
                .set_line_style("None")
                .set_marker_style("o")
                .draw(&xx, &ctm_list);

            // Numerical consistent tangent modulus curve (BE version)
            let mut curve_ctm_num = Curve::new();
            curve_ctm_num
                .set_label("Numerical CTM (BE)")
                .set_line_style("None")
                .set_marker_void(true)
                .set_marker_size(5.0)
                .set_marker_style("o")
                .set_marker_line_color("black")
                .draw(&xx, &num_ctm_list);

            // Numerical consistent tangent modulus curve (ODE version)
            let mut curve_ctm_ode_num = Curve::new();
            curve_ctm_ode_num
                .set_label(&format!("Numerical CTM ({:?})", method))
                .set_line_style("None")
                .set_marker_style("*")
                .draw(&xx, &num_ctm_ode_list);

            // Generate the plot
            let mut plot = Plot::new();
            plot.set_subplot(1, 2, 1)
                .add(&curve_fine)
                .add(&curve_ode)
                .add(&curve)
                .grid_labels_legend("x", "y")
                .set_subplot(1, 2, 2)
                .add(&curve_com)
                .add(&curve_ctm)
                .add(&curve_ctm_num)
                .add(&curve_ctm_ode_num)
                .grid_labels_legend("x", "D")
                .set_figure_size_points(800.0, 300.0)
                .save("/tmp/consistent_tangent/test_dahlquist.svg")
                .unwrap();
        }

        // Check the results against Mathematica results
        let yy_ref = [
            1.0,
            0.6666666666666666,
            0.4444444444444444,
            0.2962962962962963,
            0.19753086419753085,
            0.1316872427983539,
        ];
        for i in 0..nd + 1 {
            approx_eq(yy[i], yy_ref[i], 1e-15);
        }

        // Check the results against Mathematica results
        let ctm_ref = [
            -5.0,
            -2.2222222222222223,
            -1.4814814814814812,
            -0.9876543209876543,
            -0.6584362139917694,
            -0.43895747599451296,
        ];
        for i in 0..nd + 1 {
            approx_eq(ctm_list[i], ctm_ref[i], 1e-15);
        }

        // Compare the consistent tangent moduli
        for i in 0..nd + 1 {
            approx_eq(ctm_list[i], num_ctm_list[i], 1e-4);
        }
    }

    // Mathematica reference values
    const MATHEMATICA_XX: [f64; 51] = [
        0., 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18,
        0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37,
        0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5,
    ];

    // Mathematica reference values
    #[rustfmt::skip]
    const MATHEMATICA_YY: [f64;51] = [
        0., 0.0921923529874959, 0.180995936702942, 0.265135546339778, 0.34303769588157, 0.412909639654633, 0.472933902640526, 0.521575383562834, 0.557924601234833, 0.581937761631112, 0.594452730469534, 0.596972402157104, 0.591326450808571, 0.579354305707458,
        0.562695563124671, 0.542698423576125, 0.520413194773086, 0.496630461299565, 0.471934147069921, 0.446753034002368, 0.421403915953387, 0.39612495266096, 0.371100159763253, 0.346476741387926, 0.322376942591868, 0.298905838568251, 0.276156158431712, 0.254210961837679,
        0.23314478624825, 0.213023733754134, 0.193904874033716, 0.175835268002485, 0.158850897482815, 0.142975694678168, 0.128220878500363, 0.114584707614984, 0.102052740913371, 0.0905985737010805, 0.0801850545040245, 0.070765847217885, 0.062287241401217, 0.0546900747507887,
        0.047911642057484, 0.0418874859643884, 0.0365529932436868, 0.0318447432166188, 0.0277015895758155, 0.0240654765039555, 0.0208820056359409, 0.0181007846054456, 0.0156755917049248,
    ];

    fn run_test(name: &str, first: usize, x_ini: f64, y_ini: f64, ddx: f64, nd: usize) {
        // Allocate the model
        let method = Method::DoPri5;
        let mut model = Model::new(
            ModelType::HardeningSoftening,
            HashMap::from([("li", 10.0), ("lr", 3.0), ("y0r", 1.0), ("a", 3.0), ("b", 5.0)]),
            method,
        )
        .unwrap();

        // Perform the backward Euler update
        let (xx, yy, yy_ode, com_list, ctm_list, num_ctm_list, num_ctm_ode_list) =
            model.simulate(x_ini, y_ini, ddx, nd).unwrap();

        // Generate the plot
        if SAVE_FIGURE {
            let mut curve_ref = Curve::new();
            curve_ref
                .set_label("Mathematica")
                .draw(&MATHEMATICA_XX, &MATHEMATICA_YY);

            // ODE solution
            let mut curve_ode = Curve::new();
            curve_ode
                .set_label(&format!("{:?} solution", method))
                .set_line_style("--")
                .draw(&xx, &yy_ode);

            // Backward Euler curve y(x)
            let mut curve = Curve::new();
            curve
                .set_label("Backward Euler")
                .set_line_style("None")
                .set_marker_style(".")
                .draw(&xx, &yy);

            // Continuous modulus curve
            let mut curve_com = Curve::new();
            curve_com.set_label("Continuous Modulus").draw(&xx, &com_list);

            // Consistent tangent modulus curve
            let mut curve_ctm = Curve::new();
            curve_ctm
                .set_label("Consistent Tangent Modulus")
                .set_line_style("None")
                .set_marker_style(".")
                .draw(&xx, &ctm_list);

            // Numerical consistent tangent modulus curve (BE version)
            let mut curve_ctm_num = Curve::new();
            curve_ctm_num
                .set_label("Numerical CTM (BE)")
                .set_line_style("None")
                .set_marker_void(true)
                .set_marker_size(5.0)
                .set_marker_style("o")
                .set_marker_line_color("black")
                .draw(&xx, &num_ctm_list);

            // Numerical consistent tangent modulus curve (ODE version)
            let mut curve_ctm_ode_num = Curve::new();
            curve_ctm_ode_num
                .set_label(&format!("Numerical CTM ({:?})", method))
                .set_line_style("None")
                .set_marker_style("*")
                .draw(&xx, &num_ctm_ode_list);

            // Generate the plot
            let mut plot = Plot::new();
            plot.set_subplot(1, 2, 1)
                .add(&curve_ref)
                .add(&curve_ode)
                .add(&curve)
                .grid_labels_legend("x", "y")
                .set_subplot(1, 2, 2)
                .add(&curve_com)
                .add(&curve_ctm)
                .add(&curve_ctm_num)
                .add(&curve_ctm_ode_num)
                .grid_labels_legend("x", "D")
                .set_figure_size_points(800.0, 300.0)
                .save(&format!("/tmp/ctm_demo/{}.svg", name))
                .unwrap();
        }

        // Check the results against Mathematica results
        if ddx > 0.0 {
            for i in 0..nd + 1 {
                approx_eq(yy[i], MATHEMATICA_YY[i], 0.022);
            }
        } else {
            for j in 0..nd + 1 {
                approx_eq(xx[j], MATHEMATICA_XX[first - j], 1e-15);
                approx_eq(yy[j], MATHEMATICA_YY[first - j], 0.2);
                // println!("x = {}, y = {} ({})", xx[j], yy[j], MATHEMATICA_YY[first - j]);
            }
        }

        // Compare the consistent tangent moduli
        for i in 0..nd + 1 {
            // println!("i = {}, x = {}, ctm = {}, num_ctm = {}", i, xx[i], ctm_list[i], num_ctm_list[i]);
            let tol = if ddx > 0.0 {
                if i < 26 { 0.001 } else { 0.03 }
            } else {
                0.002
            };
            approx_eq(ctm_list[i], num_ctm_list[i], tol);
        }
    }

    #[test]
    fn test_hardening_softening_curve() {
        let x_ini = 0.0;
        let y_ini = 0.0;
        let ddx = 0.01;
        let nd = 50;
        run_test("test_hardening_softening_forward", 0, x_ini, y_ini, ddx, nd);

        let first = 10;
        let x_ini = MATHEMATICA_XX[first];
        let y_ini = MATHEMATICA_YY[first];
        let ddx = -0.01;
        let nd = 9;
        run_test("test_hardening_softening_backward", first, x_ini, y_ini, ddx, nd);
    }

    #[test]
    fn test_hardening_softening_curve_coarse() {
        // Allocate the model
        let method = Method::DoPri5;
        let mut model = Model::new(
            ModelType::HardeningSoftening,
            HashMap::from([("li", 10.0), ("lr", 3.0), ("y0r", 1.0), ("a", 3.0), ("b", 5.0)]),
            method,
        )
        .unwrap();

        // Set initial conditions
        let x_ini = 0.0;
        let y_ini = 0.0;

        // Define constants for the backward Euler update
        let ddx = 0.05;
        let nd = 10;

        // Perform the backward Euler update
        let (xx, yy, yy_ode, com_list, ctm_list, num_ctm_list, num_ctm_ode_list) =
            model.simulate(x_ini, y_ini, ddx, nd).unwrap();

        // Generate the plot
        if SAVE_FIGURE {
            // ODE solution
            let mut curve_ode = Curve::new();
            curve_ode
                .set_label(&format!("{:?} solution", method))
                .set_line_style("--")
                .draw(&xx, &yy_ode);

            // Backward Euler curve y(x)
            let mut curve = Curve::new();
            curve.set_label("Backward Euler").set_marker_style("o").draw(&xx, &yy);

            // Continuous modulus curve
            let mut curve_com = Curve::new();
            curve_com.set_label("Continuous Modulus").draw(&xx, &com_list);

            // Consistent tangent modulus curve
            let mut curve_ctm = Curve::new();
            curve_ctm
                .set_label("Consistent Tangent Modulus")
                .set_line_style("None")
                .set_marker_style("o")
                .draw(&xx, &ctm_list);

            // Numerical consistent tangent modulus curve (BE version)
            let mut curve_ctm_num = Curve::new();
            curve_ctm_num
                .set_label("Numerical CTM (BE)")
                .set_line_style("None")
                .set_marker_void(true)
                .set_marker_size(5.0)
                .set_marker_style("o")
                .set_marker_line_color("black")
                .draw(&xx, &num_ctm_list);

            // Numerical consistent tangent modulus curve (ODE version)
            let mut curve_ctm_ode_num = Curve::new();
            curve_ctm_ode_num
                .set_label(&format!("Numerical CTM ({:?})", method))
                .set_line_style("None")
                .set_marker_style("*")
                .draw(&xx, &num_ctm_ode_list);

            // Generate the plot
            let mut plot = Plot::new();
            plot.set_subplot(1, 2, 1)
                .add(&curve_ode)
                .add(&curve)
                .grid_labels_legend("x", "y")
                .set_subplot(1, 2, 2)
                .add(&curve_com)
                .add(&curve_ctm)
                .add(&curve_ctm_num)
                .add(&curve_ctm_ode_num)
                .grid_labels_legend("x", "D")
                .set_figure_size_points(800.0, 300.0)
                .save("/tmp/ctm_demo/test_hardening_softening_coarse.svg")
                .unwrap();
        }

        // Compare the consistent tangent moduli
        for i in 0..nd + 1 {
            // println!("i = {}, x = {}, ctm = {}, num_ctm = {}", i, xx[i], ctm_list[i], num_ctm_list[i]);
            let tol = if i < 6 { 0.001 } else { 0.4 };
            approx_eq(ctm_list[i], num_ctm_list[i], tol);
        }
    }
}
