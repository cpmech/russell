use russell_lab::{approx_eq, Vector};
use russell_ode::{Method, OdeSolver, Params, System};
use std::thread;

#[test]
fn test_multithreaded() {
    // system
    let system = System::new(1, |f: &mut Vector, _x: f64, _y: &Vector, _args: &mut u8| {
        f[0] = 1.0;
        Ok(())
    });
    let system_clone = system.clone();

    // solve two systems concurrently
    thread::scope(|scope| {
        let first = scope.spawn(move || {
            let params = Params::new(Method::FwEuler);
            let mut solver = OdeSolver::new(params, system).unwrap();
            let x0 = 0.0;
            let x1 = 1.5;
            let mut y0 = Vector::from(&[x0]);
            let mut args = 0;
            solver.solve(&mut y0, x0, x1, None, &mut args).unwrap();
            approx_eq(y0[0], x1, 1e-15);
        });
        let second = scope.spawn(move || {
            let params = Params::new(Method::MdEuler);
            let mut solver = OdeSolver::new(params, system_clone).unwrap();
            let x0 = 0.0;
            let x1 = 1.5;
            let mut y0 = Vector::from(&[x0]);
            let mut args = 0;
            solver.solve(&mut y0, x0, x1, None, &mut args).unwrap();
            approx_eq(y0[0], x1, 1e-15);
        });
        let err1 = first.join();
        let err2 = second.join();
        if err1.is_err() && err2.is_err() {
            Err("first and second failed")
        } else if err1.is_err() {
            Err("first failed")
        } else if err2.is_err() {
            Err("second failed")
        } else {
            Ok(())
        }
    })
    .unwrap();
}
