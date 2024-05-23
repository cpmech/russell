use russell_lab::{approx_eq, Vector};
use russell_ode::{Method, NoArgs, OdeSolver, Params, System};
use std::thread;

struct Simulator<'a> {
    solver: OdeSolver<'a, NoArgs>,
    x0: f64,
    x1: f64,
    y: Vector,
    a: u8,
}

impl<'a> Simulator<'a> {
    fn new(method: Method) -> Self {
        let system = System::new(1, |f: &mut Vector, _x: f64, _y: &Vector, _args: &mut u8| {
            f[0] = 1.0;
            Ok(())
        });
        let params = Params::new(method);
        Simulator {
            solver: OdeSolver::new(params, system).unwrap(),
            x0: 0.0,
            x1: 1.5,
            y: Vector::from(&[0.0]),
            a: 0,
        }
    }
}

#[test]
fn test_multithreaded() {
    // run two simulations concurrently
    thread::scope(|scope| {
        let first = scope.spawn(move || {
            let mut sim = Simulator::new(Method::FwEuler);
            sim.solver.solve(&mut sim.y, sim.x0, sim.x1, None, &mut sim.a).unwrap();
            approx_eq(sim.y[0], sim.x1, 1e-15);
        });
        let second = scope.spawn(move || {
            let mut sim = Simulator::new(Method::MdEuler);
            sim.solver.solve(&mut sim.y, sim.x0, sim.x1, None, &mut sim.a).unwrap();
            approx_eq(sim.y[0], sim.x1, 1e-15);
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
