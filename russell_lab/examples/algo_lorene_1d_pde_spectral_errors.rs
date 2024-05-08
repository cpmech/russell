use plotpy::{Curve, Plot};
use russell_lab::algo::{InterpGrid, InterpLagrange, InterpParams};
use russell_lab::math::NAPIER;
use russell_lab::{solve_lin_sys, Matrix, NoArgs, StrError, Vector};

const PATH_KEY: &str = "/tmp/russell_lab/pde_1d_lorene_spectral_errors";

/// Runs the simulation
///
/// This example corresponds to LORENE's example on page 25 of the Reference.
///
/// PDE:
///
/// ```text
/// d²u     du          x
/// ——— - 4 —— + 4 u = e  + C
/// dx²     dx
///
///     -4 e
/// C = ——————
///     1 + e²
///
/// x ∈ [-1, 1]
/// ```
///
/// Boundary conditions:
///
/// ```text
/// u(-1) = 0  and  u(1) = 0
/// ```
///
/// Solution:
///
/// ```text
///         x   sinh(1)  2x   C
/// u(x) = e  - ——————— e   + —
///             sinh(2)       4
/// ```
///
/// # Input
///
/// * `nn` -- polynomial degree `N`
/// * `grid_type` -- the type of grid
/// * `do_plot` -- generate plot
/// * `calc_error` -- calculate the interpolation error
///
/// # Output
///
/// * `err_f` -- the interpolation error
///
/// # Reference
///
/// * Gourgoulhon E (2005), An introduction to polynomial interpolation,
///   School on spectral methods: Application to General Relativity and Field Theory
///   Meudon, 14-18 November 2005
fn run(nn: usize, grid_type: InterpGrid, do_plot: bool, calc_error: bool) -> Result<f64, StrError> {
    // interpolant
    let mut params = InterpParams::new();
    params.grid_type = grid_type;
    let mut interp = InterpLagrange::new(nn, Some(params))?;

    // D1 and D2 matrices
    interp.calc_dd1_matrix();
    interp.calc_dd2_matrix();
    let dd1 = interp.get_dd1()?;
    let dd2 = interp.get_dd2()?;

    // source term (right-hand side)
    let xx = interp.get_points();
    let npoint = xx.dim();
    let mut b = Vector::initialized(npoint, |i| f64::exp(xx[i]) + CC);
    b[0] = 0.0;
    b[nn] = 0.0;

    // discrete differential operator
    let mut aa = Matrix::new(npoint, npoint);
    for i in 1..nn {
        for j in 1..nn {
            let o = if i == j { 1.0 } else { 0.0 };
            aa.set(i, j, dd2.get(i, j) - 4.0 * dd1.get(i, j) + 4.0 * o);
        }
    }
    aa.set(0, 0, 1.0);
    aa.set(nn, nn, 1.0);

    // linear system
    solve_lin_sys(&mut b, &mut aa)?;
    let uu = &b;

    // analytical solution
    const CC: f64 = -4.0 * NAPIER / (1.0 + NAPIER * NAPIER);
    let sh1 = f64::sinh(1.0);
    let sh2 = f64::sinh(2.0);
    let analytical = |x, _: &mut NoArgs| Ok(f64::exp(x) - f64::exp(2.0 * x) * sh1 / sh2 + CC / 4.0);

    // plot
    let aa = &mut 0;
    if do_plot {
        let mut curve_num1 = Curve::new();
        let mut curve_num2 = Curve::new();
        let mut curve_ana = Curve::new();
        let xx_plt = Vector::linspace(-1.0, 1.0, 201)?;
        let yy_num = xx_plt.get_mapped(|x| interp.eval(x, uu).unwrap());
        let yy_ana = xx_plt.get_mapped(|x| analytical(x, aa).unwrap());
        curve_ana
            .set_label("analytical")
            .draw(xx_plt.as_data(), yy_ana.as_data());
        curve_num1
            .set_label("numerical")
            .draw(xx_plt.as_data(), yy_num.as_data());
        curve_num2
            .set_line_style("None")
            .set_marker_style("o")
            .set_marker_void(true)
            .draw(xx.as_data(), uu.as_data());
        let mut plot = Plot::new();
        plot.add(&curve_ana)
            .add(&curve_num1)
            .add(&curve_num2)
            .legend()
            .grid_and_labels("$x$", "$u(x)$")
            .set_title(format!("N = {}", nn).as_str())
            .set_figure_size_points(500.0, 300.0)
            .save(format!("{}.svg", PATH_KEY).as_str())?;
    }

    // done
    let error = if calc_error {
        interp.estimate_max_error(aa, analytical)?
    } else {
        0.0
    };
    Ok(error)
}

fn main() -> Result<(), StrError> {
    // compare with LORENE
    run(4, InterpGrid::ChebyshevGaussLobatto, true, false)?;

    // grid types
    let grid_types = [
        InterpGrid::Uniform,
        InterpGrid::ChebyshevGauss,
        InterpGrid::ChebyshevGaussLobatto,
    ];

    // error analysis
    println!("\n... error analysis ...");
    let mut curve_f = Curve::new();
    for grid_type in &grid_types {
        let gt = format!("{:?}", grid_type);
        curve_f.set_label(&gt);
        match grid_type {
            InterpGrid::Uniform => {
                curve_f.set_line_style(":");
            }
            InterpGrid::ChebyshevGauss => {
                curve_f.set_line_style("--");
            }
            InterpGrid::ChebyshevGaussLobatto => {
                curve_f.set_line_style("-");
            }
        };
        let mut nn_values = Vec::new();
        let mut ef_values = Vec::new();
        for nn in (4..40).step_by(2) {
            let err_f = run(nn, *grid_type, false, true)?;
            nn_values.push(nn as f64);
            ef_values.push(err_f);
        }
        curve_f.draw(&nn_values, &ef_values);
        let mut plot = Plot::new();
        plot.set_log_y(true)
            .add(&curve_f)
            .legend()
            .grid_and_labels("$N$", "$error$")
            .set_figure_size_points(500.0, 300.0)
            .save(format!("{}_errors.svg", PATH_KEY).as_str())?;
    }
    println!("... done ...");
    Ok(())
}
