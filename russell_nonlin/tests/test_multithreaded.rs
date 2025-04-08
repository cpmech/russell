use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use russell_lab::{approx_eq, Vector};
use russell_nonlin::{Config, Method, NoArgs, Solver, Stop, System};
use russell_sparse::{CooMatrix, Sym};

const LAMBDA_FINAL: f64 = 1.0;

struct SimData<'a> {
    solver: Solver<'a, NoArgs>,
    u: Vector,
    l: f64,
    stop: Stop,
    h: Option<f64>,
    args: NoArgs,
}

impl<'a> SimData<'a> {
    fn new(method: Method) -> Self {
        // define nonlinear system: G(u, λ) = u - λ
        let mut system = System::new(1, |gg: &mut Vector, l: f64, u: &Vector, _args: &mut u8| {
            gg[0] = u[0] - l;
            Ok(())
        })
        .unwrap();

        // set analytical Jacobian
        let nnz = Some(1);
        let sym = Sym::No;
        system
            .set_calc_ggu(
                nnz,
                sym,
                |ggu: &mut CooMatrix, _l: f64, _u: &Vector, _args: &mut NoArgs| {
                    ggu.reset();
                    // dG/du = 1
                    ggu.put(0, 0, 1.0).unwrap();
                    Ok(())
                },
            )
            .unwrap();

        // configuration
        let config = Config::new(method);

        // return data
        SimData {
            solver: Solver::new(config, system).unwrap(),
            u: Vector::from(&[0.0]),
            l: 0.0,
            stop: Stop::Lambda(LAMBDA_FINAL),
            h: Some(0.1),
            args: 0,
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
            .solve(
                &mut self.data.u,
                &mut self.data.l,
                self.data.stop,
                self.data.h,
                &mut self.data.args,
            )
            .unwrap();
        approx_eq(self.data.u[0], LAMBDA_FINAL, 1e-15);
        let stats = self.data.solver.stats();
        let nstep = 10;
        let niter = 10 * 2;
        assert_eq!(stats.n_function, niter);
        assert_eq!(stats.n_jacobian, nstep);
        assert_eq!(stats.n_iterations_max, 2);
        assert_eq!(stats.n_iterations_total, niter);
    }
}

#[test]
fn test_multithreaded() {
    // run simulations concurrently
    let mut runners: Vec<Box<dyn Runner>> = vec![
        Box::new(Simulator::new(Method::Natural)),
        Box::new(Simulator::new(Method::Natural)),
    ];
    runners.par_iter_mut().for_each(|r| r.run_and_check());
}
