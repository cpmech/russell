use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use russell_lab::{approx_eq, Vector};
use russell_nonlin::{AutoStep, Config, IniDir, NoArgs, Solver, Stop, System};
use russell_sparse::{CooMatrix, Sym};

const LAMBDA_FINAL: f64 = 1.0;

#[test]
fn test_multithreaded() {
    // configuration
    let config = Config::new();

    // run simulations concurrently
    let mut runners: Vec<Box<dyn Runner>> = vec![Box::new(Simulator::new(&config)), Box::new(Simulator::new(&config))];
    runners.par_iter_mut().for_each(|r| r.run_and_check());
}

struct SimData<'a> {
    solver: Solver<'a, NoArgs>,
    args: NoArgs,
    u: Vector,
    l: f64,
    dir: IniDir,
    stop: Stop,
    auto_step: AutoStep,
}

impl<'a> SimData<'a> {
    fn new(config: &'a Config) -> Self {
        // define nonlinear system: G(u, λ) = u - λ
        let ndim = 1;
        let nnz = Some(1);
        let sym = Sym::No;
        let system = System::new(
            ndim,
            nnz,
            sym,
            |gg: &mut Vector, l: f64, u: &Vector, _args: &mut u8| {
                gg[0] = u[0] - l;
                Ok(())
            },
            |ggu: &mut CooMatrix, ggl: &mut Vector, _l: f64, _u: &Vector, _args: &mut NoArgs| {
                ggu.reset();
                // dG/du = 1
                ggu.put(0, 0, 1.0).unwrap();
                // dG/dλ = -1
                ggl[0] = -1.0;
                Ok(())
            },
        )
        .unwrap();

        // return data
        SimData {
            solver: Solver::new(config, system).unwrap(),
            args: 0,
            u: Vector::new(ndim),
            l: 0.0,
            dir: IniDir::Pos,
            stop: Stop::MaxLambda(LAMBDA_FINAL),
            auto_step: AutoStep::No(0.1),
        }
    }
}

struct Simulator<'a> {
    data: SimData<'a>,
}

impl<'a> Simulator<'a> {
    fn new(config: &'a Config) -> Self {
        Simulator {
            data: SimData::new(config),
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
                &mut self.data.args,
                &mut self.data.u,
                &mut self.data.l,
                self.data.dir,
                self.data.stop,
                self.data.auto_step,
                None,
            )
            .unwrap();
        approx_eq(self.data.u[0], LAMBDA_FINAL, 1e-15);
        let stats = self.data.solver.get_stats();
        let nstep = 10;
        let niter = 10 * 2;
        assert_eq!(stats.n_function, niter);
        assert_eq!(stats.n_jacobian, nstep);
        assert_eq!(stats.n_iteration_total, niter);
    }
}
