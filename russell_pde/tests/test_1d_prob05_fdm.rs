use plotpy::{linspace, Curve, Plot};
use russell_lab::approx_eq;
use russell_pde::{Fdm1d, Grid1d, ProblemSamples, StrError};

const SAVE_FIGURE: bool = false;

#[test]
fn test_1d_prob05_fdm() -> Result<(), StrError> {
    // problem setup
    let beta = f64::sqrt(87.4);
    let ll = 1.0;
    let g0 = 1.0;
    let phi_ll = 0.2;
    let (xmin, xmax, kx, ebcs, nbcs, source, analytical) = ProblemSamples::d1_problem_05(beta, ll, g0, phi_ll);

    // allocate the grid
    let nx = 15;
    let grid = Grid1d::new_uniform(xmin, xmax, nx)?;

    // allocate the solver
    let fdm = Fdm1d::new(grid, ebcs, nbcs, kx)?;

    // solve the problem
    let a = fdm.solve_sps(beta * beta, source)?;

    // analytical solution
    let mut max_err = 0.0;
    fdm.for_each_coord(|m, x| {
        let diff = f64::abs(a[m] - analytical(x));
        if diff > max_err {
            max_err = diff;
        }
        approx_eq(a[m], analytical(x), 0.0381);
    });
    println!("max_err = {:e}", max_err);

    // plot
    if SAVE_FIGURE {
        let mut curve_ana = Curve::new();
        let mut curve_num = Curve::new();
        curve_ana.set_label("Analytical");
        curve_num
            .set_label("Numerical")
            .set_marker_style("o")
            .set_line_style("None");
        let xx_ana = linspace(xmin, xmax, 101);
        let uu_ana = xx_ana.iter().map(|&x| analytical(x)).collect::<Vec<_>>();
        let mut xx_num = vec![0.0; nx];
        fdm.for_each_coord(|i, x| {
            xx_num[i] = x;
        });
        let uu_num = a.as_data();
        curve_ana.draw(&xx_ana, &uu_ana);
        curve_num.draw(&xx_num, &uu_num);
        let mut plot = Plot::new();
        plot.add(&curve_ana)
            .add(&curve_num)
            .grid_labels_legend("$x$", "$\\phi$")
            .save("/tmp/russell_pde/test_1d_prob05_fdm.svg")?;
    }
    Ok(())
}
