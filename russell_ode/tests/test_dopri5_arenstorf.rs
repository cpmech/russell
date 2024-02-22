use russell_lab::format_fortran;
use russell_ode::{Method, OdeSolver, Output, Params, Samples};

#[test]
fn test_dopri5_arenstorf() {
    // get ODE system
    let (system, mut data, mut args) = Samples::arenstorf();

    // set configuration parameters
    let mut params = Params::new(Method::DoPri5);
    params.h_ini = 1e-4;
    params.set_tolerances(1e-7, 1e-7).unwrap();

    // enable dense output with 1.0 spacing
    let mut out = Output::new();
    out.enable_dense(1.0, &[0, 1]).unwrap();

    // solve the ODE system
    let mut solver = OdeSolver::new(params, system).unwrap();
    solver
        .solve(&mut data.y0, data.x0, data.x1, None, Some(&mut out), &mut args)
        .unwrap();

    // get statistics
    let stat = solver.bench();

    // compare with dopri5.f
    // let y_ref = [
    //     0.9940021704037415,
    //     9.040893396741956e-06,
    //     0.0014597586885445324,
    //     -2.0012455157289244,
    // ];
    // vec_approx_eq(data.y0.as_data(), &y_ref, 1e-9);

    // print dense output
    let n_dense = out.dense_step_index.len();
    for i in 0..n_dense {
        println!(
            "step ={:>4}, x ={:5.2}, y ={}{}",
            out.dense_step_index[i],
            out.dense_x[i],
            format_fortran(out.dense_y.get(&0).unwrap()[i]),
            format_fortran(out.dense_y.get(&1).unwrap()[i]),
        )
    }

    // print and check statistics
    println!("{}", stat);
    println!("y ={}{}", format_fortran(data.y0[0]), format_fortran(data.y0[1]));
    println!("h ={}", format_fortran(stat.h_optimal));
    // assert_eq!(stat.n_function, 1429);
    // assert_eq!(stat.n_jacobian, 0);
    // assert_eq!(stat.n_steps, 238);
    // assert_eq!(stat.n_accepted, 217);
    // assert_eq!(stat.n_rejected, 21);
    // assert_eq!(stat.n_iterations, 0);
    // assert_eq!(stat.n_iterations_max, 0);
}
