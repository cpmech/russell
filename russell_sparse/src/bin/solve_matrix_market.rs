use russell_lab::{ComplexVector, Stopwatch, StrError, Vector, cpx, set_num_threads};
use russell_sparse::is_memory_error;
use russell_sparse::prelude::*;
use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::thread;
use std::time::{Duration, Instant};
use structopt::StructOpt;

// You may run this binary using cargo as follows:
// cargo run --release --bin solve_matrix_market -- -g umfpack ~/Downloads/matrix-market/dwg961a.mtx

/// Command line options
#[derive(StructOpt, Debug)]
#[structopt(name = "solve_matrix_market", about = "Solve a linear system with a Matrix-Market file.")]
struct Options {
    /// Matrix-market file
    matrix_market_file: String,

    /// Only get information about the matrix and exit
    #[structopt(short = "i", long)]
    information: bool,

    /// Solver selection; accepted values: `FAER`, `MUMPS`, or `UMFPACK`
    #[structopt(short = "g", long, default_value = "FAER")]
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

    /// LU factorization method for Faer: `auto`, `simplicial`, or `supernodal` (Faer only)
    #[structopt(long)]
    faer_method: Option<String>,

    /// Flop-ratio threshold for `--faer-method auto` (must be finite and >= 0; rejected unless `--faer-method auto` is active; Faer only)
    #[structopt(long)]
    faer_flop_ratio: Option<f64>,

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

    /// Abort each run when it exceeds this many seconds (0 disables the time guard)
    #[structopt(long, default_value = "200")]
    time_guard: u64,
}

fn main() -> Result<(), StrError> {
    // parse options
    let opt = Options::from_args();

    // set the number of OpenMP threads
    if opt.nt > 0 {
        set_num_threads(opt.nt as usize);
    }

    // select linear solver
    let genie = Genie::from(&opt.genie.to_uppercase())?;

    // select the symmetric handling option
    let handling = match genie {
        Genie::Mumps => MMsym::LeaveAsLower,
        Genie::Faer => MMsym::MakeItFull,
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
    params.faer_method = parse_faer_method(opt.faer_method.as_deref(), opt.faer_flop_ratio)?;
    params.compute_error_estimates = opt.error_estimates;
    params.compute_condition_numbers = opt.condition_numbers;
    params.verbose = opt.verbose;

    // allocate stats structure
    let mut stats = StatsLinSol::new();
    stats.main.solver = format!("{}", genie);
    stats.requests.ordering = format!("{:?}", params.ordering);
    stats.requests.scaling = format!("{:?}", params.scaling);
    stats.requests.mumps_num_threads = params.mumps_num_threads;
    stats.requests.positive_definite = params.positive_definite;

    // read the matrix
    let mut sw = Stopwatch::new();
    let (coo_real, coo_complex) = read_matrix_market(&opt.matrix_market_file, handling)?;
    stats.time_nanoseconds.read_matrix = sw.stop();

    // stop after reading the matrix when only its information is requested
    if opt.information {
        if let Some(coo) = &coo_real {
            stats.set_matrix_name_from_path(&opt.matrix_market_file);
            stats.set_matrix_info_from_coo(coo);
        } else if let Some(coo) = &coo_complex {
            stats.set_matrix_name_from_path(&opt.matrix_market_file);
            stats.set_matrix_info_from_coo(coo);
        }
        if !opt.hide_json {
            println!("{}", stats.get_json());
        }
        return Ok(());
    }

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
        for _i_run in 0..opt.nrun {
            // stop a long run
            let _time_guard = start_time_guard(opt.time_guard, &stats, opt.hide_json);

            // allocate and configure the solver
            let mut solver = LinSolver::new(genie)?;

            // call preprocess
            if let Err(e) = solver.actual.preprocess(&coo, Some(params)) {
                if is_memory_error(e) {
                    finish_on_memory_error(&mut stats, e, opt.hide_json);
                    return Ok(());
                }
                finish_on_solver_error(&mut stats, e, opt.hide_json);
                return Ok(());
            }

            // solve linear system
            if let Err(e) = solver.actual.solve(&mut x, &rhs, opt.verbose) {
                if is_memory_error(e) {
                    finish_on_memory_error(&mut stats, e, opt.hide_json);
                    return Ok(());
                }
                finish_on_solver_error(&mut stats, e, opt.hide_json);
                return Ok(());
            }

            // update stats
            solver.actual.update_stats(&mut stats);

            // verify the solution
            sw.reset();
            let verify = VerifyLinSys::from(&coo, &x, &rhs)?;
            stats.time_nanoseconds.verify = sw.stop();

            // store largest errors
            if stats.time_nanoseconds.initialize_array.len() == 1 {
                stats.verify = verify;
            } else {
                stats.verify = stats.verify.max_relative_error(&verify);
            }

            // check (debug)
            if stats.matrix.name == "bfwb62" {
                let tolerance = 1e-10;
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
            // stop a long run
            let _time_guard = start_time_guard(opt.time_guard, &stats, opt.hide_json);

            // allocate and configure the solver
            let mut solver = ComplexLinSolver::new(genie)?;

            // call preprocess
            if let Err(e) = solver.actual.preprocess(&coo, Some(params)) {
                if is_memory_error(e) {
                    finish_on_memory_error(&mut stats, e, opt.hide_json);
                    return Ok(());
                }
                finish_on_solver_error(&mut stats, e, opt.hide_json);
                return Ok(());
            }

            // solve linear system
            if let Err(e) = solver.actual.solve(&mut x, &rhs, opt.verbose) {
                if is_memory_error(e) {
                    finish_on_memory_error(&mut stats, e, opt.hide_json);
                    return Ok(());
                }
                finish_on_solver_error(&mut stats, e, opt.hide_json);
                return Ok(());
            }

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

    // the matrix was read, preprocessing succeeded, and every solve succeeded
    stats.main.solved = true;

    // print stats
    if !opt.hide_json {
        println!("{}", stats.get_json());
    }

    // done
    Ok(())
}

/// Cancels the watchdog when the guarded run finishes
struct TimeGuard {
    done: Option<Arc<AtomicBool>>,
}

impl Drop for TimeGuard {
    fn drop(&mut self) {
        if let Some(done) = &self.done {
            done.store(true, AtomicOrdering::Relaxed);
        }
    }
}

/// Starts a watchdog that emits a timeout result and exits when a single run exceeds `seconds`
///
/// A value of zero disables the guard. The guard is created inside the `nrun` loop,
/// so each repetition gets its own time budget.
fn start_time_guard(seconds: u64, stats: &StatsLinSol, hide_json: bool) -> TimeGuard {
    if seconds == 0 {
        return TimeGuard { done: None };
    }
    let done = Arc::new(AtomicBool::new(false));
    let watch = Arc::clone(&done);
    let mut snapshot = stats.clone();
    thread::spawn(move || {
        let start = Instant::now();
        loop {
            if watch.load(AtomicOrdering::Relaxed) {
                return;
            }
            if start.elapsed() >= Duration::from_secs(seconds) {
                mark_time_guard(&mut snapshot, seconds);
                if !hide_json {
                    println!("{}", snapshot.get_json());
                    let _ = std::io::stdout().flush();
                }
                exit_now(124);
            }
            thread::sleep(Duration::from_millis(50));
        }
    });
    TimeGuard { done: Some(done) }
}

/// Terminates immediately without running exit handlers
///
/// `std::process::exit` runs the C runtime and OpenMP cleanup handlers. When the
/// watchdog fires while the solver is inside MKL's `dgemm`, those handlers race
/// with the OpenMP runtime and can crash with SIGSEGV. `_exit` bypasses them.
#[cfg(unix)]
fn exit_now(status: i32) -> ! {
    unsafe extern "C" {
        fn _exit(status: i32) -> !;
    }
    unsafe { _exit(status) }
}

#[cfg(not(unix))]
fn exit_now(status: i32) -> ! {
    std::process::exit(status)
}

/// Marks the statistics as stopped by the time guard
fn mark_time_guard(stats: &mut StatsLinSol, seconds: u64) {
    stats.main.solved = false;
    stats.main.stopped_by_time_guard = true;
    stats.main.error_from_solver = Some(format!("time guard exceeded ({} s)", seconds));
}

/// Clears partial run results and emits the OOM JSON when requested
fn finish_on_memory_error(stats: &mut StatsLinSol, error: StrError, hide_json: bool) {
    stats.clear_run_results();
    stats.main.solved = false;
    stats.main.out_of_memory = true;
    stats.main.error_from_solver = Some(error.to_string());
    if !hide_json {
        println!("{}", stats.get_json());
    }
}

/// Records a solver error in the statistics and emits the JSON when requested
///
/// The caller stops the current matrix and exits successfully: a solver failure
/// is a recorded result, not a failure of the study.
fn finish_on_solver_error(stats: &mut StatsLinSol, error: StrError, hide_json: bool) {
    stats.main.solved = false;
    stats.main.error_from_solver = Some(error.to_string());
    if !hide_json {
        println!("{}", stats.get_json());
    }
}

/// Parses `--faer-method` and `--faer-flop-ratio`
fn parse_faer_method(method: Option<&str>, flop_ratio: Option<f64>) -> Result<FaerMethod, StrError> {
    if let Some(threshold) = flop_ratio {
        if !threshold.is_finite() || threshold < 0.0 {
            return Err("faer-flop-ratio must be finite and nonnegative");
        }
    }
    match method.map(|v| v.to_lowercase()).as_deref() {
        None => {
            if let Some(threshold) = flop_ratio {
                if matches!(FaerMethod::default(), FaerMethod::Auto { .. }) {
                    Ok(FaerMethod::Auto {
                        flop_ratio_threshold: threshold,
                    })
                } else {
                    Err("faer-flop-ratio requires --faer-method auto")
                }
            } else {
                Ok(FaerMethod::default())
            }
        }
        Some("auto") => Ok(FaerMethod::Auto {
            flop_ratio_threshold: flop_ratio.unwrap_or(FaerMethod::DEFAULT_FLOP_RATIO_THRESHOLD),
        }),
        Some("simplicial") => {
            if flop_ratio.is_some() {
                Err("faer-flop-ratio requires --faer-method auto")
            } else {
                Ok(FaerMethod::Simplicial)
            }
        }
        Some("supernodal") => {
            if flop_ratio.is_some() {
                Err("faer-flop-ratio requires --faer-method auto")
            } else {
                Ok(FaerMethod::Supernodal)
            }
        }
        Some(_) => Err("faer-method must be one of: auto, simplicial, supernodal"),
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

#[cfg(test)]
mod tests {
    use super::{finish_on_memory_error, finish_on_solver_error, mark_time_guard, parse_faer_method};
    use russell_sparse::prelude::*;

    #[test]
    fn parse_faer_method_validates_inputs() {
        assert_eq!(parse_faer_method(None, None).unwrap(), FaerMethod::Simplicial);
        assert_eq!(
            parse_faer_method(None, Some(1.0)).err(),
            Some("faer-flop-ratio requires --faer-method auto")
        );
        assert_eq!(
            parse_faer_method(Some("auto"), None).unwrap(),
            FaerMethod::Auto {
                flop_ratio_threshold: FaerMethod::DEFAULT_FLOP_RATIO_THRESHOLD,
            }
        );
        assert_eq!(
            parse_faer_method(Some("AUTO"), Some(2.5)).unwrap(),
            FaerMethod::Auto { flop_ratio_threshold: 2.5 }
        );
        assert_eq!(parse_faer_method(Some("simplicial"), None).unwrap(), FaerMethod::Simplicial);
        assert_eq!(parse_faer_method(Some("supernodal"), None).unwrap(), FaerMethod::Supernodal);
        assert_eq!(
            parse_faer_method(Some("simplicial"), Some(1.0)).err(),
            Some("faer-flop-ratio requires --faer-method auto")
        );
        assert_eq!(
            parse_faer_method(Some("supernodal"), Some(1.0)).err(),
            Some("faer-flop-ratio requires --faer-method auto")
        );
        assert_eq!(
            parse_faer_method(Some("auto"), Some(-0.1)).err(),
            Some("faer-flop-ratio must be finite and nonnegative")
        );
        assert_eq!(
            parse_faer_method(Some("auto"), Some(f64::NAN)).err(),
            Some("faer-flop-ratio must be finite and nonnegative")
        );
        assert_eq!(
            parse_faer_method(Some("auto"), Some(f64::INFINITY)).err(),
            Some("faer-flop-ratio must be finite and nonnegative")
        );
        assert_eq!(
            parse_faer_method(Some("other"), None).err(),
            Some("faer-method must be one of: auto, simplicial, supernodal")
        );
    }

    #[test]
    fn mark_time_guard_sets_flags() {
        let mut stats = StatsLinSol::new();
        stats.main.solved = true;
        mark_time_guard(&mut stats, 7);
        assert!(!stats.main.solved);
        assert!(stats.main.stopped_by_time_guard);
        assert_eq!(stats.main.error_from_solver.as_deref(), Some("time guard exceeded (7 s)"));
    }

    #[test]
    fn finish_on_solver_error_fills_stats() {
        let mut stats = StatsLinSol::new();
        finish_on_solver_error(&mut stats, "solver failed", true);
        assert!(!stats.main.solved);
        assert_eq!(stats.main.error_from_solver.as_deref(), Some("solver failed"));
        let json = stats.get_json();
        assert!(json.contains("\"error_from_solver\": \"solver failed\""));
    }

    #[test]
    fn finish_on_memory_error_clears_partial_runs() {
        let mut stats = StatsLinSol::new();
        stats.main.solver = format!("{}", Genie::Faer);
        stats.main.solved = true;
        stats.output.faer_effective_method = "Supernodal".to_string();
        stats.output.faer_flop_ratio = 2.0;
        stats.time_nanoseconds.initialize = 10;
        stats.time_nanoseconds.preprocess = 20;
        stats.time_nanoseconds.solve = 30;
        stats.time_nanoseconds.total_ifs = 60;
        stats.time_nanoseconds.initialize_array = vec![10];
        stats.time_nanoseconds.preprocess_array = vec![20];
        stats.time_nanoseconds.solve_array = vec![30];
        stats.time_nanoseconds.total_ifs_array = vec![60];
        stats.verify.relative_error = 1e-12;

        finish_on_memory_error(&mut stats, "not enough memory", true);

        assert!(!stats.main.solved);
        assert!(stats.main.out_of_memory);
        assert_eq!(stats.main.error_from_solver.as_deref(), Some("not enough memory"));
        assert_eq!(stats.main.solver, "FAER");
        assert!(stats.output.faer_effective_method.is_empty());
        assert_eq!(stats.output.faer_flop_ratio, 0.0);
        assert_eq!(stats.time_nanoseconds.total_ifs, 0);
        assert!(stats.time_nanoseconds.initialize_array.is_empty());
        assert!(stats.time_nanoseconds.preprocess_array.is_empty());
        assert!(stats.time_nanoseconds.solve_array.is_empty());
        assert!(stats.time_nanoseconds.total_ifs_array.is_empty());
        assert_eq!(stats.verify.relative_error, 0.0);
    }
}
