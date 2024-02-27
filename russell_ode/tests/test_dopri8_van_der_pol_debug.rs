use russell_lab::{approx_eq, format_fortran, Vector};
use russell_ode::{Method, OdeSolver, Params, Samples};

#[test]
fn test_dopri8_van_der_pol_debug() {
    // get get ODE system
    const EPS: f64 = 1e-3;
    let (system, _, mut args) = Samples::van_der_pol(Some(EPS), false);

    // set configuration parameters
    let mut params = Params::new(Method::DoPri8);
    params.step.h_ini = 1e-6;
    params.set_tolerances(1e-9, 1e-9, None).unwrap();
    params.debug = true;

    // solve the ODE system
    let mut y0 = Vector::from(&[2.0, 0.0]);
    let x0 = 0.0;
    let x1 = 2.0;
    let mut solver = OdeSolver::new(params, system).unwrap();
    solver.solve(&mut y0, x0, x1, None, None, &mut args).unwrap();

    // get statistics
    let stat = solver.bench();

    // compare with dop853.f
    approx_eq(y0[0], 1.763234540172087E+00, 1e-13);
    approx_eq(y0[1], -8.356886819301910E-01, 1e-13);
    approx_eq(stat.h_accepted, 8.656983588595286E-04, 4.5e-7);

    // print and check statistics
    println!("{}", stat.summary());
    println!("y ={}{}", format_fortran(y0[0]), format_fortran(y0[1]));
    println!("h ={}", format_fortran(stat.h_accepted));
    assert_eq!(stat.n_function, 17509 - 2); // -2 when compared with dop853
    assert_eq!(stat.n_steps, 1469);
    assert_eq!(stat.n_accepted, 1348);
    assert_eq!(stat.n_rejected, 121);
}