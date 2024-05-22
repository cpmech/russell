use russell_lab::{approx_eq, format_fortran};
use russell_ode::{Method, OdeSolver, Params, Samples};

#[test]
fn test_dopri5_arenstorf_debug() {
    // get ODE system
    let (system, x0, mut y0, x1, mut args, _) = Samples::arenstorf();

    // set configuration parameters
    let mut params = Params::new(Method::DoPri5);
    params.step.h_ini = 1e-4;
    params.set_tolerances(1e-7, 1e-7, None).unwrap();
    params.debug = true;

    // allocate the solver
    let mut solver = OdeSolver::new(params, system).unwrap();

    // solve the ODE system
    let y = &mut y0;
    solver.solve(y, x0, x1, None, &mut args).unwrap();

    // get statistics
    let stat = solver.stats();

    // compare with dopri5.f
    approx_eq(y[0], 9.940021704030663E-01, 1e-11);
    approx_eq(y[1], 9.040891036151961E-06, 1e-11);
    approx_eq(y[2], 1.459758305600828E-03, 1e-9);
    approx_eq(y[3], -2.001245515834718E+00, 1e-9);
    approx_eq(stat.h_accepted, 5.258587607119909E-04, 1e-10);

    // print and check statistics
    println!("{}", stat.summary());
    println!(
        "y ={}{}{}{}",
        format_fortran(y[0]),
        format_fortran(y[1]),
        format_fortran(y[2]),
        format_fortran(y[3])
    );
    println!("h ={}", format_fortran(stat.h_accepted));
    assert_eq!(stat.n_function, 1429);
    assert_eq!(stat.n_steps, 238);
    assert_eq!(stat.n_accepted, 217);
    assert_eq!(stat.n_rejected, 21);
}
