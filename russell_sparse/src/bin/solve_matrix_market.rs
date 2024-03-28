use num_complex::Complex64;
use russell_lab::{cpx, set_num_threads, using_intel_mkl, ComplexVector, Stopwatch, StrError, Vector};
use russell_sparse::prelude::*;
use structopt::StructOpt;

/// Command line options
#[derive(StructOpt, Debug)]
#[structopt(
    name = "solve_matrix_market",
    about = "Solve a linear system with a Matrix-Market file."
)]
struct Options {
    /// Matrix-market file
    matrix_market_file: String,

    /// Solver selection
    #[structopt(short = "g", long, default_value = "Umfpack")]
    genie: String,

    /// Ordering strategy
    #[structopt(short = "o", long, default_value = "Auto")]
    ordering: String,

    /// Scaling strategy
    #[structopt(short = "s", long, default_value = "Auto")]
    scaling: String,

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

    /// Writes vismatrix file
    #[structopt(long)]
    vismatrix: bool,

    /// Hide JSON output (useful to pipe MUMPS/UMFPACK logs to files)
    #[structopt(long)]
    hide_json: bool,
}

fn main() -> Result<(), StrError> {
    // parse options
    let opt = Options::from_args();

    // set the number of OpenMP threads
    if opt.nt > 0 {
        set_num_threads(opt.nt as usize);
    }

    // select linear solver
    let genie = Genie::from(&opt.genie);

    // select the symmetric handling option
    let handling = match genie {
        Genie::Klu => MMsym::MakeItFull,
        Genie::Mumps => MMsym::LeaveAsLower,
        Genie::Umfpack => MMsym::MakeItFull,
    };

    // configuration parameters
    let mut params = LinSolParams::new();
    params.ordering = Ordering::from(&opt.ordering);
    params.scaling = Scaling::from(&opt.scaling);
    params.compute_determinant = opt.determinant;
    params.mumps_num_threads = opt.mumps_nt as usize;
    params.umfpack_enforce_unsymmetric_strategy = opt.enforce_unsymmetric_strategy;
    params.compute_error_estimates = opt.error_estimates;
    params.compute_condition_numbers = opt.condition_numbers;
    params.mumps_override_prevent_nt_issue_with_openblas = opt.override_prevent_issue;
    params.verbose = opt.verbose;
    if !using_intel_mkl() && opt.override_prevent_issue {
        println!("... WARNING: overriding the prevention of issue with OpenBLAS ...");
    }

    // allocate stats structure
    let mut stats = StatsLinSol::new();
    stats.requests.ordering = format!("{:?}", params.ordering);
    stats.requests.scaling = format!("{:?}", params.scaling);
    stats.requests.mumps_num_threads = params.mumps_num_threads;

    // read the matrix
    let mut sw = Stopwatch::new();
    let (coo_real, coo_complex) = read_matrix_market(&opt.matrix_market_file, handling)?;
    stats.time_nanoseconds.read_matrix = sw.stop();

    // --- real ---------------------------------------------------------------------------------
    if let Some(coo) = coo_real {
        // write vismatrix file
        if opt.vismatrix {
            let csc = CscMatrix::from_coo(&coo)?;
            csc.write_matrix_market("/tmp/russell_sparse/solve_matrix_market_real.smat", true)?;
        }

        // save the COO matrix as a generic SparseMatrix
        let mut mat = SparseMatrix::from_coo(coo);

        // save information about the matrix
        let (nrow, ncol, nnz, sym) = mat.get_info();
        stats.set_matrix_name_from_path(&opt.matrix_market_file);
        stats.matrix.nrow = nrow;
        stats.matrix.ncol = ncol;
        stats.matrix.nnz = nnz;
        stats.matrix.complex = false;
        stats.matrix.symmetric = format!("{:?}", sym);

        // allocate and configure the solver
        let mut solver = LinSolver::new(genie)?;

        // call factorize
        solver.actual.factorize(&mut mat, Some(params))?;

        // allocate vectors
        let mut x = Vector::new(nrow);
        let rhs = Vector::filled(nrow, 1.0);

        // solve linear system
        solver.actual.solve(&mut x, &mat, &rhs, opt.verbose)?;

        // verify the solution
        sw.reset();
        stats.verify = VerifyLinSys::from(&mat, &x, &rhs)?;
        stats.time_nanoseconds.verify = sw.stop();

        // update stats
        solver.actual.update_stats(&mut stats);

    // check
    if stats.matrix.name == "bfwb62" {
        let tolerance = match genie {
            Genie::Klu => 1e-10,
            Genie::Mumps => 1e-10,
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

    // done
    Ok(())
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
