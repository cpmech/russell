use russell_lab::{approx_eq, format_fortran};
use russell_ode::{Method, OdeSolver, Output, Params, Samples};

#[test]
fn test_dopri5_arenstorf() {
    // get ODE system
    let (system, mut data, mut args) = Samples::arenstorf();

    // set configuration parameters
    let mut params = Params::new(Method::DoPri5);
    params.step.h_ini = 1e-4;
    params.set_tolerances(1e-7, 1e-7, None).unwrap();

    // enable dense output with 1.0 spacing
    let mut out = Output::new();
    out.enable_dense(1.0, &[0, 1, 2, 3]).unwrap();

    // solve the ODE system
    let mut solver = OdeSolver::new(params, system).unwrap();
    solver
        .solve(&mut data.y0, data.x0, data.x1, None, Some(&mut out), &mut args)
        .unwrap();

    // get statistics
    let stat = solver.bench();

    // compare with dopri5.f
    approx_eq(data.y0[0], 9.940021704030663E-01, 1e-11);
    approx_eq(data.y0[1], 9.040891036151961E-06, 1e-11);
    approx_eq(data.y0[2], 1.459758305600828E-03, 1e-9);
    approx_eq(data.y0[3], -2.001245515834718E+00, 1e-9);
    approx_eq(stat.h_accepted, 5.258587607119909E-04, 1e-10);

    // print dense output
    let n_dense = out.dense_step_index.len();
    for i in 0..n_dense {
        println!(
            "step ={:>4}, x ={:6.2}, y ={}{}{}{}",
            out.dense_step_index[i],
            out.dense_x[i],
            format_fortran(out.dense_y.get(&0).unwrap()[i]),
            format_fortran(out.dense_y.get(&1).unwrap()[i]),
            format_fortran(out.dense_y.get(&2).unwrap()[i]),
            format_fortran(out.dense_y.get(&3).unwrap()[i]),
        )
    }

    // print and check statistics
    println!("{}", stat.summary());
    println!(
        "y ={}{}{}{}",
        format_fortran(data.y0[0]),
        format_fortran(data.y0[1]),
        format_fortran(data.y0[2]),
        format_fortran(data.y0[3])
    );
    println!("h ={}", format_fortran(stat.h_accepted));
    assert_eq!(stat.n_function, 1429);
    assert_eq!(stat.n_steps, 238);
    assert_eq!(stat.n_accepted, 217);
    assert_eq!(stat.n_rejected, 21);
}
