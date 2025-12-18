use plotpy::{linspace, Curve, Plot};
use russell_lab::approx_eq;
use russell_pde::{Fdm1d, Grid1d, ProblemSamples, StrError};

const SAVE_FIGURE: bool = false;

#[test]
fn test_1d_prob02_fdm_sps() -> Result<(), StrError> {
    // problem setup
    let (xmin, xmax, kx, beta, phi_inf, ebcs, nbcs, source, analytical) = ProblemSamples::d1_problem_02();
    let nx = 21;

    // allocate the grid
    let grid = Grid1d::new_uniform(xmin, xmax, nx)?;

    // allocate the solver
    let fdm = Fdm1d::new(grid, ebcs, nbcs, kx)?;

    // solve the problem
    let a = fdm.solve_ext(beta, phi_inf, source)?;

    // analytical solution
    fdm.for_each_coord(|m, x| {
        // println!("{}: ϕ = {} ({})", m, a[m], analytical(x));
        approx_eq(a[m], analytical(x), 0.0155);
    });

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
            .save("/tmp/russell_pde/test_1d_prob02_fdm_sps.svg")?;
    }
    Ok(())
}

#[test]
fn test_1d_prob02_fdm_lmm() -> Result<(), StrError> {
    // problem setup
    let (xmin, xmax, kx, beta, phi_inf, ebcs, nbcs, source, analytical) = ProblemSamples::d1_problem_02();
    let nx = 21;

    // allocate the grid
    let grid = Grid1d::new_uniform(xmin, xmax, nx)?;

    // allocate the solver
    let fdm = Fdm1d::new(grid, ebcs, nbcs, kx)?;

    // solve the problem
    let a = fdm.solve_ext_lmm(beta, phi_inf, source)?;

    // analytical solution
    fdm.for_each_coord(|m, x| {
        // println!("{}: ϕ = {} ({})", m, a[m], analytical(x));
        approx_eq(a[m], analytical(x), 0.0155);
    });
    Ok(())
}
