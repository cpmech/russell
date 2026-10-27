use russell_lab::{ComplexVector, Stopwatch, StrError, Vector, cpx, set_num_threads, using_intel_mkl};
use russell_sparse::is_memory_error;
use russell_sparse::prelude::*;
use structopt::StructOpt;

// You may run this binary using cargo as follows:
// cargo run --release --bin solve_matrix_market -- -g umfpack ~/Downloads/matrix-market/dwg961a.mtx

/// Command line options
#[derive(StructOpt, Debug)]
#[structopt(name = "solve_matrix_market", about = "Solve a linear system with a Matrix-Market file.")]
struct Options {
    /// Matrix-market file
    matrix_market_file: String,

    /// Solver selection; accepted values: `direct`, `cg`, `bicgstab`, `gmres`, `mumps`, or `umfpack`
    #[structopt(short = "g", long, default_value = "Direct")]
    genie: String,

    /// Ordering strategy
    #[structopt(short = "o", long, default_value = "Auto")]
    ordering: String,

    /// Scaling strategy
    #[structopt(short = "s", long, default_value = "Auto")]
    scaling: String,

    /// Indicates that the coefficient matrix is positive-definite (only considered if the matrix is symmetric)
    #[structopt(short = "p", long)]
    positive_definite: bool,

    /// Number of threads for MUMPS
    #[structopt(short = "m", long, default_value = "0")]
    mumps_nt: u32,

    /// Number of threads
    #[structopt(short = "n", long, default_value = "0")]
    nt: u32,

    /// Overrides the prevention of number-of-threads issue with OpenBLAS (not recommended)
    #[structopt(long)]
    override_prevent_issue: bool,

    /// Activate verbose mode
    #[structopt(short = "v", long)]
    verbose: bool,

    /// Computes determinant
    #[structopt(short = "d", long)]
    determinant: bool,

    /// Computes error estimates (MUMPS only)
    #[structopt(short = "x", long)]
    error_estimates: bool,

    /// Computes condition numbers (MUMPS only; slow)
    #[structopt(short = "y", long)]
    condition_numbers: bool,

    /// Enforce unsymmetric strategy (not recommended) (UMFPACK only)
    #[structopt(short = "u", long)]
    enforce_unsymmetric_strategy: bool,

    /// Selects the supernodal LU in the Direct solver (experimental; Direct only)
    #[structopt(long)]
    direct_supernodal: bool,

    /// Iterative solver options (ignored by `direct`, `mumps`, and `umfpack`)
    #[structopt(flatten)]
    iterative: IterativeOptions,

    /// Writes vismatrix file
    #[structopt(long)]
    vismatrix: bool,

    /// Hide JSON output (useful to pipe MUMPS/UMFPACK logs to files)
    #[structopt(long)]
    hide_json: bool,

    /// Number of runs
    ///
    /// If more than one run:
    /// * The total time (initialize + preprocess + solve) field will hold the average value
    /// * The total time array will hold each individually result
    /// * The error fields will hold the maximum error among runs
    #[structopt(short = "r", long, default_value = "1")]
    nrun: u32,
}

// Iterative solver options (ignored by `direct`, `mumps`, and `umfpack`)
#[derive(StructOpt, Debug)]
struct IterativeOptions {
    /// Relative residual tolerance (iterative solvers; ignored by `direct`, `mumps`, and `umfpack`)
    #[structopt(long, default_value = "1e-10")]
    iter_tol: f64,

    /// Absolute residual tolerance (iterative solvers; ignored by `direct`, `mumps`, and `umfpack`)
    #[structopt(long, default_value = "0")]
    iter_abs_tol: f64,

    /// Maximum number of Krylov iterations (ignored by `direct`, `mumps`, and `umfpack`)
    #[structopt(long, default_value = "1000")]
    iter_max: usize,

    /// Preconditioner: `amg` or `identity` (ignored by `direct`, `mumps`, and `umfpack`)
    #[structopt(long, default_value = "amg")]
    iter_precond: String,

    /// GMRES restart dimension (GMRES only; ignored by `direct`, `mumps`, and `umfpack`)
    #[structopt(long, default_value = "30")]
    iter_restart: usize,

    /// Number of threads for the Krylov and AMG kernels (ignored by `direct`, `mumps`, and `umfpack`)
    #[structopt(long, default_value = "1")]
    iter_threads: usize,

    /// Minimum vector length before threads are used (ignored by `direct`, `mumps`, and `umfpack`)
    #[structopt(long, default_value = "1024")]
    iter_min_parallel_len: usize,

    /// AMG coarsening: `ruge-stuben`, `aggregation`, or `smoothed-aggregation` (ignored by `direct`, `mumps`, and `umfpack`)
    #[structopt(long, default_value = "ruge-stuben")]
    amg_coarsening: String,

    /// AMG smoother: `gauss-seidel`, `damped-jacobi`, `ilu0`, `chebyshev`, `spai0`, or `spai1` (ignored by `direct`, `mumps`, and `umfpack`)
    #[structopt(long, default_value = "gauss-seidel")]
    amg_smoother: String,

    /// AMG size at which the coarsest level is solved directly (ignored by `direct`, `mumps`, and `umfpack`)
    #[structopt(long, default_value = "64")]
    amg_coarse_enough: usize,

    /// Maximum number of AMG levels (ignored by `direct`, `mumps`, and `umfpack`)
    #[structopt(long, default_value = "32")]
    amg_max_levels: usize,
}

fn main() -> Result<(), StrError> {
    // parse options
    let opt = Options::from_args();

    // set the number of OpenMP threads
    if opt.nt > 0 {
        set_num_threads(opt.nt as usize);
    }

    // select linear solver
    let genie = match opt.genie.to_lowercase().as_str() {
        "direct" => Genie::Direct,
        "cg" => Genie::Iterative(IterativeMethod::Cg),
        "bicgstab" => Genie::Iterative(IterativeMethod::BiCgStab),
        "gmres" => Genie::Iterative(IterativeMethod::Gmres),
        "mumps" => Genie::Mumps,
        "umfpack" => Genie::Umfpack,
        _ => return Err("genie must be one of: direct, cg, bicgstab, gmres, mumps, umfpack"),
    };

    // select the symmetric handling option
    let handling = match genie {
        Genie::Mumps => MMsym::LeaveAsLower,
        Genie::Direct => MMsym::MakeItFull,
        Genie::Iterative(_) => MMsym::MakeItFull,
        Genie::Umfpack => MMsym::MakeItFull,
    };

    // configuration parameters
    let mut params = LinSolParams::new();
    params.ordering = Ordering::from(&opt.ordering);
    params.scaling = Scaling::from(&opt.scaling);
    params.positive_definite = opt.positive_definite;
    params.compute_determinant = opt.determinant;
    params.mumps_num_threads = opt.mumps_nt as usize;
    params.umfpack_enforce_unsymmetric_strategy = opt.enforce_unsymmetric_strategy;
    params.direct_supernodal = opt.direct_supernodal;
    params.compute_error_estimates = opt.error_estimates;
    params.compute_condition_numbers = opt.condition_numbers;
    params.mumps_override_prevent_nt_issue_with_openblas = opt.override_prevent_issue;
    params.verbose = opt.verbose;
    params.iterative.tolerance = opt.iterative.iter_tol;
    params.iterative.absolute_tolerance = opt.iterative.iter_abs_tol;
    params.iterative.max_iterations = opt.iterative.iter_max;
    params.iterative.preconditioner = parse_preconditioner(&opt.iterative.iter_precond)?;
    params.iterative.gmres_restart = opt.iterative.iter_restart;
    params.iterative.num_threads = opt.iterative.iter_threads;
    params.iterative.min_parallel_len = opt.iterative.iter_min_parallel_len;
    params.iterative.amg_coarsening = parse_amg_coarsening(&opt.iterative.amg_coarsening)?;
    params.iterative.amg_smoother = parse_amg_smoother(&opt.iterative.amg_smoother)?;
    params.iterative.amg_coarse_enough = opt.iterative.amg_coarse_enough;
    params.iterative.amg_max_levels = opt.iterative.amg_max_levels;
    if !using_intel_mkl() && opt.override_prevent_issue {
        println!("... WARNING: overriding the prevention of issue with OpenBLAS ...");
    }

    // allocate stats structure
    let mut stats = StatsLinSol::new();
    stats.main.solver = genie.to_string();
    stats.requests.ordering = format!("{:?}", params.ordering);
    stats.requests.scaling = format!("{:?}", params.scaling);
    stats.requests.mumps_num_threads = params.mumps_num_threads;
    stats.requests.positive_definite = params.positive_definite;

    // read the matrix
    let mut sw = Stopwatch::new();
    let (coo_real, coo_complex) = read_matrix_market(&opt.matrix_market_file, handling)?;
    stats.time_nanoseconds.read_matrix = sw.stop();

    // --- real ---------------------------------------------------------------------------------
    if let Some(coo) = coo_real {
        // write vismatrix file
        if opt.vismatrix {
            let csc = CscMatrix::from_coo(&coo)?;
            csc.write_matrix_market("/tmp/russell_sparse/solve_matrix_market_real.smat", true, 1e-14)?;
        }

        // save information about the matrix
        let (nrow, _, _, _) = coo.get_info();
        stats.set_matrix_name_from_path(&opt.matrix_market_file);
        stats.set_matrix_info_from_coo(&coo);

        // allocate vectors
        let mut x = Vector::new(nrow);
        let rhs = Vector::filled(nrow, 1.0);

        // run a number of times
        for i_run in 0..opt.nrun {
            // allocate and configure the solver
            let mut solver = LinSolver::new(genie)?;

            // call preprocess
            if let Err(e) = solver.actual.preprocess(&coo, Some(params)) {
                if is_memory_error(e) {
                    stats.main.out_of_memory = true;
                    if !opt.hide_json {
                        println!("{}", stats.get_json());
                    }
                    return Ok(());
                }
                return Err(e);
            }

            // solve linear system
            solver.actual.solve(&mut x, &rhs, opt.verbose)?;

            // update stats
            solver.actual.update_stats(&mut stats);

            // verify the solution
            sw.reset();
            let verify = VerifyLinSys::from(&coo, &x, &rhs)?;
            stats.time_nanoseconds.verify = sw.stop();

            // store largest errors
            if i_run == 0 {
                stats.verify = verify;
            } else {
                stats.verify = stats.verify.max_relative_error(&verify);
            }

            // check (debug)
            if stats.matrix.name == "bfwb62" {
                let tolerance = match genie {
                    Genie::Mumps => 1e-10,
                    Genie::Direct => 1e-10,
                    Genie::Iterative(_) => 1e-10,
                    Genie::Umfpack => 1e-10,
                };
                let correct_x = get_bfwb62_correct_x();
                for i in 0..nrow {
                    let diff = f64::abs(x.get(i) - correct_x.get(i));
                    if diff > tolerance {
                        println!("BFWB62 FAILED WITH NUMERICAL ERROR = {:.2e} @ {} COMPONENT", diff, i);
                    }
                }
            }
        }

    // --- complex ------------------------------------------------------------------------------
    } else {
        // write vismatrix file
        let coo = coo_complex.unwrap();
        if opt.vismatrix {
            let csc = ComplexCscMatrix::from_coo(&coo)?;
            csc.write_matrix_market("/tmp/russell_sparse/solve_matrix_market_complex.smat", true, 1e-14)?;
        }

        // save information about the matrix
        let (nrow, _, _, _) = coo.get_info();
        stats.set_matrix_name_from_path(&opt.matrix_market_file);
        stats.set_matrix_info_from_coo(&coo);

        // allocate vectors
        let mut x = ComplexVector::new(nrow);
        let rhs = ComplexVector::filled(nrow, cpx!(1.0, 1.0));

        // run a number of times
        for i_run in 0..opt.nrun {
            // allocate and configure the solver
            let mut solver = ComplexLinSolver::new(genie)?;

            // call preprocess
            if let Err(e) = solver.actual.preprocess(&coo, Some(params)) {
                if is_memory_error(e) {
                    stats.main.out_of_memory = true;
                    if !opt.hide_json {
                        println!("{}", stats.get_json());
                    }
                    return Ok(());
                }
                return Err(e);
            }

            // solve linear system
            solver.actual.solve(&mut x, &rhs, opt.verbose)?;

            // update stats
            solver.actual.update_stats(&mut stats);

            // verify the solution
            sw.reset();
            let verify = VerifyLinSys::from_complex(&coo, &x, &rhs)?;
            stats.time_nanoseconds.verify = sw.stop();

            // store largest errors
            if i_run == 0 {
                stats.verify = verify;
            } else {
                stats.verify = stats.verify.max_relative_error(&verify);
            }
        }
    }

    // print stats
    if !opt.hide_json {
        println!("{}", stats.get_json());
    }

    // done
    Ok(())
}

/// Parses the `--iter-precond` option
fn parse_preconditioner(value: &str) -> Result<Preconditioner, StrError> {
    match value.to_lowercase().as_str() {
        "amg" => Ok(Preconditioner::Amg),
        "identity" => Ok(Preconditioner::Identity),
        _ => Err("iter-precond must be one of: amg, identity"),
    }
}

/// Parses the `--amg-coarsening` option
fn parse_amg_coarsening(value: &str) -> Result<AmgCoarsening, StrError> {
    match value.to_lowercase().as_str() {
        "ruge-stuben" => Ok(AmgCoarsening::RugeStuben),
        "aggregation" => Ok(AmgCoarsening::Aggregation),
        "smoothed-aggregation" => Ok(AmgCoarsening::SmoothedAggregation),
        _ => Err("amg-coarsening must be one of: ruge-stuben, aggregation, smoothed-aggregation"),
    }
}

/// Parses the `--amg-smoother` option
fn parse_amg_smoother(value: &str) -> Result<AmgSmoother, StrError> {
    match value.to_lowercase().as_str() {
        "gauss-seidel" => Ok(AmgSmoother::GaussSeidel),
        "damped-jacobi" => Ok(AmgSmoother::DampedJacobi),
        "ilu0" => Ok(AmgSmoother::Ilu0),
        "chebyshev" => Ok(AmgSmoother::Chebyshev),
        "spai0" => Ok(AmgSmoother::Spai0),
        "spai1" => Ok(AmgSmoother::Spai1),
        _ => Err("amg-smoother must be one of: gauss-seidel, damped-jacobi, ilu0, chebyshev, spai0, spai1"),
    }
}

fn get_bfwb62_correct_x() -> Vector {
    Vector::from(&[
        -1.02570048377040759e+05,
        -1.08800418159713998e+05,
        -7.87848688672370918e+04,
        -6.12550631774225840e+04,
        -1.16611533352550643e+05,
        -8.91949258261042705e+04,
        -5.57584825429375196e+04,
        -3.37535346291137103e+04,
        -6.74159236038033268e+04,
        -5.61065283435406673e+04,
        -3.69561341372605821e+04,
        -2.67385128650871302e+04,
        -4.67349124343154253e+04,
        -4.18861901056076676e+04,
        -4.34393771636046149e+04,
        -1.11210692731083000e+04,
        -1.16010526640020762e+04,
        -4.31993854681577286e+04,
        -5.82924327463857844e+03,
        -2.42374319876188747e+04,
        -2.39432136682168457e+04,
        5.27355041927211232e+02,
        -1.24769422505944240e+04,
        -1.47005934749971748e+04,
        -4.95701604733381391e+04,
        -1.38451884223610182e+03,
        -1.57972501695015781e+04,
        -5.19172705598900066e+04,
        -4.99494464999615593e+04,
        -1.19678659380488571e+04,
        -1.56190973892000347e+04,
        -6.18809904102459404e+03,
        -1.05693761694190998e+04,
        -2.93013328593191145e+04,
        -9.15514607143451940e+03,
        -1.27058094439569140e+04,
        -1.93936053067287430e+04,
        -6.84836276779992295e+03,
        -1.07869319688850719e+04,
        -4.61926223513438963e+04,
        -1.99579363156562504e+04,
        -7.83564896339727693e+03,
        -6.37173129434054590e+03,
        -1.88075622025074267e+03,
        -8.71648101674354621e+03,
        -1.21683775603205122e+04,
        -1.91184585274694587e+03,
        -5.64233479410600103e+03,
        -6.47747230904305070e+03,
        -4.47783973932844674e+03,
        -9.82971659947420812e+03,
        -1.95594295004403466e+04,
        -2.09457080830507803e+04,
        -5.46686114796283709e+03,
        -5.28888244321673483e+03,
        -2.07962090362636227e+04,
        -9.33272319073228937e+03,
        1.96672299472196187e+02,
        -4.40813445835840230e+03,
        -4.87188111893421956e+03,
        -1.75640594405328884e+04,
        -1.77959327708208002e+04,
    ])
}
