use plotpy::{linspace, Curve, Plot};
use russell_nonlin::{Config, DeltaLambda, IniDir, Method, Output, Samples, Solver, Stop};

/// Example: Arc-length continuation on a B-spline curve
///
/// ```bash
/// cargo run --example arc_bspline_custom
/// ```
fn main() {
    // -----------------------------------------------------------------------
    // 1. Define the nonlinear problem
    // -----------------------------------------------------------------------
    //
    // G(u, λ) = u - C(λ)
    //
    // where C(λ) is a point on a 2D B-spline curve parametrized by λ ∈ [0,1].
    // The `snap_back_delta` controls how much control point P3 snaps back,
    // creating a challenging fold/turning-point structure.
    let snap_back_delta = 1.5;
    let (system, mut u, mut lambda, mut args) = Samples::bspline_problem_1(snap_back_delta);
    println!("B-spline problem created with snap-back delta = {}", snap_back_delta);

    // -----------------------------------------------------------------------
    // 2. Configure the continuation solver
    // -----------------------------------------------------------------------
    //
    // The tangent (tg) control tolerance drives the adaptive step-size
    // mechanism — a small value (here 0.2) instructs the solver to request more
    // steps in highly-curved regions, which is needed when the solution branch
    // has sharp turns or snap-back behavior.
    let mut config = Config::new();
    config.set_method(Method::Arclength);
    config
        .set_verbose(true, true, true)
        .set_hide_timings(true)
        .set_bordering(false) // use the standard un-bordered formulation
        .set_tg_control_tol(0.2) // tighter control → more steps on curves
        .set_n_cont_delta_divergence_max(1);

    let mut solver = Solver::new(&config, system).expect("Failed to create solver — check config consistency");

    // -----------------------------------------------------------------------
    // 3. Record results at every accepted step
    // -----------------------------------------------------------------------
    let out = &mut Output::new();
    out.set_recording(
        true,
        &[0, 1], // record both components of u
        &[0, 1], // record the first two entries of the G vector
    );

    // -----------------------------------------------------------------------
    // 4. Trace the solution branch from λ=0 to λ=1
    // -----------------------------------------------------------------------
    let status = solver
        .solve(
            &mut args,
            &mut u,
            &mut lambda,
            IniDir::Pos,               // advance in the forward direction
            Stop::MaxLambda(1.0),      // stop when λ reaches 1.0
            &DeltaLambda::auto(0.007), // initial step size, adaptively adjusted
            Some(out),
        )
        .expect("Solver failed unexpectedly");

    // -----------------------------------------------------------------------
    // 5. Print diagnostics
    // -----------------------------------------------------------------------
    let stats = solver.get_stats();
    println!("\n{}", "─".repeat(50));
    println!(" Final status : {:?}", status);
    println!(" |_ λ          : {:.6}", lambda);
    println!(" |_ u          : [{:.6}, {:.6}]", u[0], u[1]);
    println!(" |_ Accepted   : {}", stats.n_accepted);
    println!(" |_ Rejected   : {}", stats.n_rejected);
    println!(" |_ Total steps: {}", stats.n_steps);
    println!("{}", "─".repeat(50));

    // (optional) Inspect the recorded data:
    //   out.get_l_values()   → λ at every accepted point
    //   out.get_u_values(0)  → u₀ at every accepted point
    //   out.get_u_values(1)  → u₁ at every accepted point

    // -----------------------------------------------------------------------
    // 6. Plot the B-spline curve and the numerical solution points
    // -----------------------------------------------------------------------
    // Get an access to the results
    let uu0 = out.get_u_values(0);
    let uu1 = out.get_u_values(1);

    // Draw B-spline curve
    let mut curve = Curve::new();
    let n_station = 201;
    let stations = linspace(0.0, 1.0, n_station);
    let mut xx = vec![0.0; stations.len()];
    let mut yy = vec![0.0; stations.len()];
    for i in 0..n_station {
        args.bspline.calc_point(&mut args.coords, stations[i], false).unwrap();
        xx[i] = args.coords[0];
        yy[i] = args.coords[1];
    }
    curve.draw(&xx, &yy);

    // Draw numerical solution points
    let mut curve_num = Curve::new();
    curve_num
        .set_label("numerical")
        .set_line_style("-")
        .set_line_color("green")
        .set_marker_style(".")
        .set_marker_color("red")
        .set_marker_line_color("red");
    curve_num.draw(uu0, uu1);

    // Create and save the plot
    let mut plot = Plot::new();
    plot.set_labels("$u_1$", "$u_2$")
        .add(&curve)
        .add(&curve_num)
        .set_range(-0.1, 2.7, -0.1, 1.2)
        .set_equal_axes(true)
        .set_figure_size_points(600.0, 600.0)
        .save("/tmp/russell_nonlin/doc_arclength_bspline.svg")
        .unwrap()
}
