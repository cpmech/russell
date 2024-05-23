use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use russell_lab::{approx_eq, Vector};
use russell_ode::{Method, NoArgs, OdeSolver, Params, System};

struct SimData<'a> {
    solver: OdeSolver<'a, NoArgs>,
    x0: f64,
    x1: f64,
    y: Vector,
    a: u8,
}

impl<'a> SimData<'a> {
    fn new(method: Method) -> Self {
        let system = System::new(1, |f: &mut Vector, _x: f64, _y: &Vector, _args: &mut u8| {
            f[0] = 1.0;
            Ok(())
        });
        let params = Params::new(method);
        SimData {
            solver: OdeSolver::new(params, system).unwrap(),
            x0: 0.0,
            x1: 1.5,
            y: Vector::from(&[0.0]),
            a: 0,
        }
    }
}

struct Simulator<'a> {
    data: SimData<'a>,
}

impl<'a> Simulator<'a> {
    fn new(method: Method) -> Self {
        Simulator {
            data: SimData::new(method),
        }
    }
}

trait Runner: Send {
    fn run_and_check(&mut self);
}

impl<'a> Runner for Simulator<'a> {
    fn run_and_check(&mut self) {
        self.data
            .solver
            .solve(&mut self.data.y, self.data.x0, self.data.x1, None, &mut self.data.a)
            .unwrap();
        approx_eq(self.data.y[0], self.data.x1, 1e-15);
    }
}

#[test]
fn test_multithreaded() {
    // run simulations concurrently
    let mut runners: Vec<Box<dyn Runner>> = vec![
        Box::new(Simulator::new(Method::FwEuler)),
        Box::new(Simulator::new(Method::MdEuler)),
    ];
    runners.par_iter_mut().for_each(|r| r.run_and_check());
}
