use plotpy::{Curve, Plot, SuperTitleParams};
use russell_lab::algo::{InterpGrid, InterpLagrange, InterpParams};
use russell_lab::math::PI;
use russell_lab::{mat_vec_mul, StrError, Vector};
use russell_ode::{no_jacobian, HasJacobian, Method, NoArgs, OdeSolver, Params, System};

const PATH_KEY: &str = "/tmp/russell_ode/pde_1d_heat_spectral_collocation";

struct Args {
    interp: InterpLagrange,
}

/// Runs the analysis with polynomial degree N (nn)
///
/// Approximates the solution of the diffusion equation
/// (see Equation 4.82 on page 115 of the Reference):
///
/// ```text
/// du   d²u
/// —— = ———
/// dt   dx²
///
/// x ∈ [-1, 1]
/// ```
///
/// with initial conditions:
///
/// ```text
/// u(x, 0) = sin((x + 1) π)
/// ```
///
/// and homogeneous boundary conditions, i.e.:
///
/// ```text
/// u(-1, t) = u(1, t) = 0
/// ```
///
/// Analytical solution:
///
/// ```text
/// u(x, t) = -exp(-π² t) sin(π x)
/// ```
///
/// Returns `(err_f, err_g, err_h)`
///
/// # Input
///
/// * `nn` -- polynomial degree `N`
/// * `grid_type` -- the type of grid
/// * `print_stats` -- prints the ODE solver
/// * `do_plot` -- generate plot
/// * `calc_errors` -- calculate the interpolation and derivative errors
///
/// # Output
///
/// * `err_f` -- the interpolation error
/// * `err_g` -- the first derivative error
/// * `err_h` -- the second derivative error
///
/// # Reference
///
/// * Kopriva DA (2009) Implementing Spectral Methods for Partial Differential Equations
///   Springer, 404p
fn run(
    nn: usize,
    grid_type: InterpGrid,
    print_stats: bool,
    do_plot: bool,
    calc_errors: bool,
) -> Result<(f64, f64, f64), StrError> {
    // ODE system
    let ndim = nn + 1;
    let system = System::new(
        ndim,
        |dudt: &mut Vector, _: f64, u: &Vector, args: &mut Args| {
            mat_vec_mul(dudt, 1.0, args.interp.get_dd2()?, u)?;
            dudt[0] = 0.0; // homogeneous boundary conditions
            dudt[nn] = 0.0; // homogeneous boundary conditions
            Ok(())
        }, //
        no_jacobian,
        HasJacobian::No,
        None,
        None,
    );

    // ODE solver
    let mut params = Params::new(Method::DoPri8);
    params.set_tolerances(1e-10, 1e-10, None)?;
    let mut ode = OdeSolver::new(params, &system)?;

    // interpolant
    let mut par = InterpParams::new();
    par.grid_type = grid_type;
    let mut args = Args {
        interp: InterpLagrange::new(nn, Some(par))?,
    };
    args.interp.calc_dd2_matrix();

    // initial conditions
    let (t0, t1) = (0.0, 0.1);
    let xx = args.interp.get_points().clone();
    let mut uu = xx.get_mapped(|x| f64::sin(PI * (x + 1.0)));

    // solve the problem
    ode.solve(&mut uu, t0, t1, None, None, &mut args)?;

    // print stats
    if print_stats {
        println!("\n================================================================");
        println!("N (polynomial degree)            = {}", nn);
        println!("Grid type                        = {:?}", grid_type);
        println!("{}", ode.stats());
    }

    // calc errors
    let f = |x, _: &mut NoArgs| Ok(-f64::exp(-PI * PI * t1) * f64::sin(PI * x));
    let g = |x, _: &mut NoArgs| Ok(-f64::exp(-PI * PI * t1) * f64::cos(PI * x) * PI);
    let h = |x, _: &mut NoArgs| Ok(f64::exp(-PI * PI * t1) * f64::sin(PI * x) * PI * PI);
    let aa = &mut 0;
    let (err_f, err_g, err_h) = if calc_errors {
        args.interp.estimate_max_error_all(true, aa, f, g, h)?
    } else {
        (0.0, 0.0, 0.0)
    };

    // plot the results @ t1
    if do_plot {
        let mut curve1 = Curve::new();
        let mut curve2 = Curve::new();
        let xx_ana = Vector::linspace(-1.0, 1.0, 201)?;
        let uu_ana = xx_ana.get_mapped(|x| f(x, aa).unwrap());
        curve1.draw(xx_ana.as_data(), uu_ana.as_data());
        curve2
            .set_line_style("None")
            .set_marker_style("o")
            .set_marker_void(true)
            .draw(xx.as_data(), uu.as_data());
        let mut plot = Plot::new();
        let gt = format!("{:?}", grid_type);
        let path = format!("{}_nn{}_{}.svg", PATH_KEY, nn, gt.to_lowercase(),);
        plot.add(&curve1)
            .add(&curve2)
            .grid_and_labels("$x$", "$u(x)$")
            .save(path.as_str())?;
    }
    Ok((err_f, err_g, err_h))
}

fn main() -> Result<(), StrError> {
    // grid types
    let grid_types = [
        InterpGrid::Uniform,
        InterpGrid::ChebyshevGauss,
        InterpGrid::ChebyshevGaussLobatto,
    ];

    // plot solutions
    for grid_type in &grid_types {
        run(12, *grid_type, true, true, false)?;
    }

    // error analysis
    println!("\n... error analysis ...");
    let mut curve_f = Curve::new();
    let mut curve_g = Curve::new();
    let mut curve_h = Curve::new();
    for grid_type in &grid_types {
        let gt = format!("{:?}", grid_type);
        curve_f.set_label(&gt);
        match grid_type {
            InterpGrid::Uniform => {
                curve_f.set_line_style(":");
                curve_g.set_line_style(":");
                curve_h.set_line_style(":");
            }
            InterpGrid::ChebyshevGauss => {
                curve_f.set_line_style("--");
                curve_g.set_line_style("--");
                curve_h.set_line_style("--");
            }
            InterpGrid::ChebyshevGaussLobatto => {
                curve_f.set_line_style("-");
                curve_g.set_line_style("-");
                curve_h.set_line_style("-");
            }
        };
        let mut nn_values = Vec::new();
        let mut ef_values = Vec::new();
        let mut eg_values = Vec::new();
        let mut eh_values = Vec::new();
        for nn in (4..40).step_by(2) {
            let (err_f, err_g, err_h) = run(nn, *grid_type, false, false, true)?;
            nn_values.push(nn as f64);
            ef_values.push(err_f);
            eg_values.push(err_g);
            eh_values.push(err_h);
        }
        curve_f.draw(&nn_values, &ef_values);
        curve_g.draw(&nn_values, &eg_values);
        curve_h.draw(&nn_values, &eh_values);
        let mut plot = Plot::new();
        let mut spp = SuperTitleParams::new();
        spp.set_y(0.95);
        plot.set_subplot(3, 1, 1)
            .set_log_y(true)
            .add(&curve_f)
            .legend()
            .grid_and_labels("$N$", "$err(f)$")
            .set_subplot(3, 1, 2)
            .set_log_y(true)
            .add(&curve_g)
            .legend()
            .grid_and_labels("$N$", "$err(g)$")
            .set_subplot(3, 1, 3)
            .set_log_y(true)
            .add(&curve_h)
            .legend()
            .grid_and_labels("$N$", "$err(h)$")
            .set_figure_size_points(400.0, 600.0)
            .save(format!("{}_errors.svg", PATH_KEY).as_str())?;
    }
    println!("... done ...");
    Ok(())
}
