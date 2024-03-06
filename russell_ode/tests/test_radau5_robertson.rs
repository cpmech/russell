use russell_lab::{approx_eq, format_fortran};
use russell_ode::{Method, OdeSolver, Output, Params, Samples};

#[test]
fn test_radau5_robertson() {
    // get get ODE system
    let (system, mut data, mut args) = Samples::robertson();

    // set configuration parameters
    let mut params = Params::new(Method::Radau5);
    params.step.h_ini = 1e-6;
    params.set_tolerances(1e-8, 1e-2, None).unwrap();

    // enable output of accepted steps
    let mut out = Output::new();
    out.enable_step(&[0, 1, 2]);

    // solve the ODE system
    let mut solver = OdeSolver::new(params, system).unwrap();
    solver
        .solve(&mut data.y0, data.x0, data.x1, None, Some(&mut out), &mut args)
        .unwrap();

    // get statistics
    let stat = solver.bench();

    // compare with radau5.f
    approx_eq(data.y0[0], 9.886740138499884E-01, 1e-15);
    approx_eq(data.y0[1], 3.447720471782070E-05, 1e-15);
    approx_eq(data.y0[2], 1.129150894529390E-02, 1e-15);
    approx_eq(stat.h_accepted, 8.160578540333708E-01, 1e-10);

    // print the results at accepted steps
    let n_step = out.step_x.len();
    for i in 0..n_step {
        println!(
            "step ={:>4}, x ={:5.2}, y ={}{}{}",
            i,
            out.step_x[i],
            format_fortran(out.step_y.get(&0).unwrap()[i]),
            format_fortran(out.step_y.get(&1).unwrap()[i]),
            format_fortran(out.step_y.get(&2).unwrap()[i]),
        );
    }

    // print and check statistics
    println!("{}", stat.summary());
    println!("y ={}{}", format_fortran(data.y0[0]), format_fortran(data.y0[1]));
    println!("h ={}", format_fortran(stat.h_accepted));
    assert_eq!(stat.n_function, 88);
    assert_eq!(stat.n_jacobian, 8);
    assert_eq!(stat.n_factor, 15);
    assert_eq!(stat.n_lin_sol, 24);
    assert_eq!(stat.n_steps, 17);
    assert_eq!(stat.n_accepted, 15);
    assert_eq!(stat.n_rejected, 1);
    assert_eq!(stat.n_iterations_max, 2);
}
