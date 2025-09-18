use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use russell_lab::{approx_eq, Vector};
use russell_nonlin::{AutoStep, Config, Direction, Method, NoArgs, Solver, State, Stop, System};
use russell_sparse::{CooMatrix, Sym};

const LAMBDA_FINAL: f64 = 1.0;

struct SimData<'a> {
    solver: Solver<'a, NoArgs>,
    args: NoArgs,
    state: State,
    dir: Direction,
    stop: Stop,
    auto_step: AutoStep,
}

impl<'a> SimData<'a> {
    fn new(method: Method) -> Self {
        // define nonlinear system: G(u, λ) = u - λ
        let ndim = 1;
        let mut system = System::new(ndim, |gg: &mut Vector, l: f64, u: &Vector, _args: &mut u8| {
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
            args: 0,
            state: State::new(ndim),
            dir: Direction::Pos,
            stop: Stop::Lambda(LAMBDA_FINAL),
            auto_step: AutoStep::No(0.1),
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
                &mut self.data.args,
                &mut self.data.state,
                self.data.dir,
                self.data.stop,
                self.data.auto_step,
                None,
            )
            .unwrap();
        approx_eq(self.data.state.u[0], LAMBDA_FINAL, 1e-15);
        let stats = self.data.solver.get_stats();
        let nstep = 10;
        let niter = 10 * 2;
        assert_eq!(stats.n_function, niter);
        assert_eq!(stats.n_jacobian, nstep);
        assert_eq!(stats.n_iteration_max, 2);
        assert_eq!(stats.n_iteration_total, niter);
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
