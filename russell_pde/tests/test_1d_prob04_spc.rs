use plotpy::{linspace, Curve, Plot};
use russell_lab::approx_eq;
use russell_pde::{ProblemSamples, Spc1d, StrError};

const SAVE_FIGURE: bool = false;

#[test]
fn test_1d_prob04_spc_sps() -> Result<(), StrError> {
    let (nn, tol) = (7, 2.26e-2);
    // let (nn, tol) = (16, 3.0965e-9); // Trefethen, page 138, Output 33

    // problem setup
    let (xmin, xmax, kx, ebcs, nbcs, source, analytical) = ProblemSamples::d1_problem_04();

    // allocate the solver
    let nx = nn + 1;
    let spc = Spc1d::new(xmin, xmax, nx, ebcs, nbcs, kx)?;

    // solve the problem
    let a = spc.solve_sps(0.0, source)?;

    // analytical solution
    let mut err_max = 0.0;
    spc.for_each_coord(|m, x| {
        let err = f64::abs(a[m] - analytical(x));
        if err > err_max {
            err_max = err;
        }
        approx_eq(a[m], analytical(x), tol);
    });
    println!("N = {}, max(err) = {:>10.5e}", nn, err_max);

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
        spc.for_each_coord(|i, x| {
            xx_num[i] = x;
        });
        let uu_num = a.as_data();
        curve_ana.draw(&xx_ana, &uu_ana);
        curve_num.draw(&xx_num, &uu_num);
        let mut plot = Plot::new();
        plot.add(&curve_ana)
            .add(&curve_num)
            .grid_labels_legend("$x$", "$\\phi$")
            .save("/tmp/russell_pde/test_1d_prob04_spc.svg")?;
    }
    Ok(())
}
