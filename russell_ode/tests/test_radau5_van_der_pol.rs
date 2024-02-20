use russell_lab::{approx_eq, format_fortran};
use russell_ode::{Method, OdeSolver, Params, Samples};

#[test]
fn test_radau5_van_der_pol() {
    const EPS: f64 = 1e-6;
    let (system, mut data, mut args) = Samples::van_der_pol(Some(EPS), false);
    let mut params = Params::new(Method::Radau5);
    params.h_ini = 1e-6;
    params.radau5.logging = true;
    let mut solver = OdeSolver::new(params, system).unwrap();
    solver
        .solve(&mut data.y0, data.x0, data.x1, None, None, &mut args)
        .unwrap();
    approx_eq(data.y0[0], 1.706163410178079E+00, 1e-14);
    approx_eq(data.y0[1], -8.927971289301175E-01, 1e-12);
    let b = solver.bench();
    println!("{}", b.summary());
    println!("y ={},{}", format_fortran(data.y0[0]), format_fortran(data.y0[1]));
    assert_eq!(b.n_function, 2248);
    assert_eq!(b.n_jacobian, 162);
    assert_eq!(b.n_factor, 253);
    assert_eq!(b.n_lin_sol, 668);
    assert_eq!(b.n_steps, 280);
    assert_eq!(b.n_accepted, 242);
    assert_eq!(b.n_rejected, 8);
    assert_eq!(b.n_iterations, 2);
    assert_eq!(b.n_iterations_max, 6);
}
