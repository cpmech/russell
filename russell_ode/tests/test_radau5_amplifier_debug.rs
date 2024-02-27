use russell_lab::{approx_eq, format_fortran};
use russell_ode::{Method, OdeSolver, Params, Samples};

// NOTE: This test is a little more difficult to match FORTRAN's results
// with a high precision. The reason might be related to the use of different
// kind of sparse solvers and handling of banded matrices.

#[test]
fn test_radau5_amplifier() {
    // get get ODE system
    let (mut system, mut data, mut args, gen_mass) = Samples::amplifier();

    // set configuration parameters
    let mut params = Params::new(Method::Radau5);
    params.step.h_ini = 1e-6;
    params.set_tolerances(1e-11, 1e-5, None).unwrap();
    params.debug = true;

    // mass matrix
    let one_based = false; // change to true if using MUMPS
    let mass = gen_mass(one_based);
    system.set_mass_matrix(&mass).unwrap();

    // solve the ODE system
    let mut solver = OdeSolver::new(params, system).unwrap();
    solver
        .solve(&mut data.y0, data.x0, data.x1, None, None, &mut args)
        .unwrap();

    // get statistics
    let stat = solver.bench();

    // compare with radau5.f
    approx_eq(data.y0[0], 4.726947717958150E-01, 1e-6);
    approx_eq(data.y0[1], 5.535355787030912E+00, 1e-6);
    approx_eq(data.y0[2], 2.494964078383526E+00, 1e-7);
    approx_eq(data.y0[3], 2.434168328176757E+00, 1e-4);
    approx_eq(data.y0[4], 3.467272484822467E+00, 1e-4);
    approx_eq(data.y0[5], 2.849957445537722E+00, 1e-6);
    approx_eq(data.y0[6], 3.006521494823356E+00, 1e-7);
    approx_eq(data.y0[7], -5.561984048406549E-03, 1e-7);
    approx_eq(stat.h_accepted, 8.644857027106722E-04, 1e-4);

    // print and check statistics
    println!("{}", stat.summary());
    println!(
        "y1to4 ={}{}{}{}",
        format_fortran(data.y0[0]),
        format_fortran(data.y0[1]),
        format_fortran(data.y0[2]),
        format_fortran(data.y0[3])
    );
    println!(
        "y5to8 ={}{}{}{}",
        format_fortran(data.y0[4]),
        format_fortran(data.y0[5]),
        format_fortran(data.y0[6]),
        format_fortran(data.y0[7])
    );
    println!("h ={}", format_fortran(stat.h_accepted));
    assert_eq!(stat.n_function, 2552);
    assert_eq!(stat.n_jacobian, 211);
    assert_eq!(stat.n_factor, 267);
    assert_eq!(stat.n_lin_sol, 777);
    assert_eq!(stat.n_steps, 268);
    assert_eq!(stat.n_accepted, 216);
    assert_eq!(stat.n_rejected, 21);
    assert_eq!(stat.n_iterations_max, 6);
}
