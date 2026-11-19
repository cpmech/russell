use super::{StatsLinSolMUMPS, VerifyLinSys};
use crate::{NumCooMatrix, StrError};
use num_traits::{Num, NumCast};
use russell_lab::{Complex64, format_nanoseconds, get_num_threads, using_intel_mkl};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::any::TypeId;
use std::ffi::OsStr;
use std::fs::{self, File};
use std::io::BufReader;
use std::ops::{AddAssign, MulAssign};
use std::path::Path;

/// Holds the main information such as platform
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StatsLinSolMain {
    /// Name of the platform that produced the statistics
    pub platform: String,

    /// BLAS library selected at build time (e.g., Native, OpenBLAS, or Intel MKL)
    pub blas_lib: String,

    /// Name of the solver that produced the statistics (e.g., `Faer`, `UMFPACK`, or `MUMPS`)
    pub solver: String,

    /// Indicates whether MUMPS support was compiled in
    pub has_mumps: bool,

    /// Indicates whether UMFPACK support was compiled in
    pub has_umfpack: bool,

    /// Indicates that the matrix was read, preprocessing succeeded, and the linear system was solved successfully
    ///
    /// This flag is `false` for information-only runs, which read the matrix but do not preprocess or solve it.
    pub solved: bool,

    /// Indicates that the run was stopped by the time guard before it could finish
    pub stopped_by_time_guard: bool,

    /// Indicates whether the solver ran out of memory
    pub out_of_memory: bool,

    /// Error message on exit
    pub error_from_solver: Option<String>,
}

/// Holds information about the sparse matrix
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StatsLinSolMatrix {
    /// Name of the matrix
    pub name: String,

    /// Number of rows
    pub nrow: usize,

    /// Number of columns
    pub ncol: usize,

    /// Number of stored non-zero values, counting duplicates
    pub nnz: usize,

    /// Number of non-zero values after summing duplicates (the pattern entries reported by the SuiteSparse Matrix Collection)
    pub nnz_actual: usize,

    /// Indicates whether the matrix holds complex values
    pub complex: bool,

    /// Symmetric storage mode
    pub symmetric: String,
}

/// Holds some requested parameters
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StatsLinSolRequests {
    /// Requested ordering algorithm
    pub ordering: String,

    /// Requested scaling strategy
    pub scaling: String,

    /// Requested number of MUMPS threads
    pub mumps_num_threads: usize,

    /// Indicates whether the matrix was declared positive definite
    pub positive_definite: bool,
}

/// Holds some output parameters
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StatsLinSolOutput {
    /// Ordering algorithm actually used by the solver
    pub effective_ordering: String,

    /// Scaling strategy actually used by the solver
    pub effective_scaling: String,

    /// Number of MUMPS threads actually used
    pub effective_mumps_num_threads: usize,

    /// Number of OpenMP threads used by the solver
    pub openmp_num_threads: usize,

    /// UMFPACK strategy applied during the factorization
    pub umfpack_strategy: String,

    /// UMFPACK reciprocal condition number estimate
    pub umfpack_rcond_estimate: f64,

    /// Effective factorization method used by the Faer solver (`"Simplicial"` or `"Supernodal"`), when applicable
    pub faer_effective_method: String,

    /// Symbolic flop-ratio proxy `Σ col_counts² / nnz(A)` computed by the Faer solver, when applicable
    pub faer_flop_ratio: f64,

    /// Numerical rank achieved by the BasicLU factorization, when applicable
    pub basiclu_rank: usize,

    /// Number of positive diagonal pivots reported by the QDLDL factorization (the positive inertia), when applicable
    pub qdldl_positive_inertia: usize,
}

/// Holds the determinant of the coefficient matrix (if requested)
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StatsLinSolDeterminant {
    /// Real part of the mantissa
    pub mantissa_real: f64,

    /// Imaginary part of the mantissa (complex systems only)
    pub mantissa_imag: f64,

    /// Base of the exponent
    pub base: f64,

    /// Exponent, such that `det = mantissa * base^exponent`
    pub exponent: f64,
}

/// Holds the average computer times in human readable format (post-processed)
///
/// **Note:** These are automatically converted from TimeNanoseconds when calling [StatsLinSol::get_json]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StatsLinSolTimeHuman {
    /// Time to read the matrix
    pub read_matrix: String,

    /// Raw initialization times, one per run (not averaged)
    pub initialize_array: Vec<String>,

    /// Average initialization time
    pub initialize: String,

    /// Raw preprocessing times, one per run (not averaged)
    pub preprocess_array: Vec<String>,

    /// Average preprocessing time
    pub preprocess: String,

    /// Raw solve times, one per run (not averaged)
    pub solve_array: Vec<String>,

    /// Average solve time
    pub solve: String,

    /// Raw totals of initialize + preprocess + solve, one per run (not averaged)
    pub total_ifs_array: Vec<String>,

    /// Average of initialize + preprocess + solve
    pub total_ifs: String,

    /// Time to verify the solution
    pub verify: String,
}

/// Holds the average computer times in nanoseconds
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StatsLinSolTimeNanoseconds {
    /// Time to read the matrix, in nanoseconds
    pub read_matrix: u128,

    /// Raw initialization times, one per run (not averaged), in nanoseconds
    pub initialize_array: Vec<u128>,

    /// Average initialization time, in nanoseconds
    pub initialize: u128,

    /// Raw preprocessing times, one per run (not averaged), in nanoseconds
    pub preprocess_array: Vec<u128>,

    /// Average preprocessing time, in nanoseconds
    pub preprocess: u128,

    /// Raw solve times, one per run (not averaged), in nanoseconds
    pub solve_array: Vec<u128>,

    /// Average solve time, in nanoseconds
    pub solve: u128,

    /// Raw totals of initialize + preprocess + solve, one per run (not averaged), in nanoseconds
    pub total_ifs_array: Vec<u128>,

    /// Average of initialize + preprocess + solve, in nanoseconds
    pub total_ifs: u128,

    /// Time to verify the solution, in nanoseconds
    pub verify: u128,
}

/// Holds the statistics of an iterative solve
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct StatsLinSolIterative {
    /// Krylov method that ran (`CG`, `BiCGStab`, `GMRES`)
    pub method: String,

    /// Preconditioner that ran (`AMG` or `Identity`)
    pub preconditioner: String,

    /// Relative residual tolerance requested
    pub tolerance: f64,

    /// Maximum number of iterations requested
    pub max_iterations: usize,

    /// GMRES restart dimension; 0 for methods that do not restart
    pub restart: usize,

    /// Number of threads requested for the Krylov and AMG kernels
    pub num_threads: usize,

    /// AMG coarsening that ran
    pub amg_coarsening: String,

    /// AMG smoother that ran
    pub amg_smoother: String,

    /// Number of levels in the hierarchy; 1 for the identity preconditioner
    pub amg_levels: usize,

    /// Iterations of each run, in order
    pub iterations_array: Vec<usize>,

    /// Achieved relative residual of each run, in order
    pub relative_residual_array: Vec<f64>,
}

/// Holds information about the solution of a linear system
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StatsLinSol {
    /// Main information, such as platform and solver
    pub main: StatsLinSolMain,

    /// Information about the matrix
    pub matrix: StatsLinSolMatrix,

    /// Parameters requested by the caller
    pub requests: StatsLinSolRequests,

    /// Parameters actually used by the solver
    pub output: StatsLinSolOutput,

    /// Determinant of the coefficient matrix, when requested
    pub determinant: StatsLinSolDeterminant,

    /// Verification data for the solution
    pub verify: VerifyLinSys,

    /// Times in human readable form (filled by [StatsLinSol::get_json])
    pub time_human: StatsLinSolTimeHuman,

    /// Times in nanoseconds
    pub time_nanoseconds: StatsLinSolTimeNanoseconds,

    /// MUMPS-specific statistics (only relevant if the solver is MUMPS)
    pub mumps_stats: StatsLinSolMUMPS,

    /// Iterative-solver statistics (only relevant if the solver is Iterative)
    pub iterative_stats: Option<StatsLinSolIterative>,
}

impl StatsLinSol {
    /// Allocates a blank structure
    pub fn new() -> Self {
        let unknown = "Unknown".to_string();
        StatsLinSol {
            main: StatsLinSolMain {
                platform: "Russell".to_string(),
                blas_lib: if using_intel_mkl() {
                    "Intel MKL".to_string()
                } else if cfg!(feature = "open_blas") {
                    "OpenBLAS".to_string()
                } else {
                    "Native".to_string()
                },
                solver: unknown.clone(),
                has_mumps: cfg!(feature = "mumps"),
                has_umfpack: cfg!(feature = "umfpack"),
                solved: false,
                stopped_by_time_guard: false,
                out_of_memory: false,
                error_from_solver: None,
            },
            matrix: StatsLinSolMatrix {
                name: unknown.clone(),
                nrow: 0,
                ncol: 0,
                nnz: 0,
                nnz_actual: 0,
                complex: false,
                symmetric: unknown.clone(),
            },
            requests: StatsLinSolRequests {
                ordering: unknown.clone(),
                scaling: unknown.clone(),
                mumps_num_threads: 0,
                positive_definite: false,
            },
            output: StatsLinSolOutput {
                effective_ordering: unknown.clone(),
                effective_scaling: unknown.clone(),
                effective_mumps_num_threads: 0,
                openmp_num_threads: 0,
                umfpack_strategy: unknown.clone(),
                umfpack_rcond_estimate: 0.0,
                faer_effective_method: String::new(),
                faer_flop_ratio: 0.0,
                basiclu_rank: 0,
                qdldl_positive_inertia: 0,
            },
            determinant: StatsLinSolDeterminant {
                mantissa_real: 0.0,
                mantissa_imag: 0.0,
                base: 0.0,
                exponent: 0.0,
            },
            verify: VerifyLinSys {
                max_abs_a: 0.0,
                max_abs_ax: 0.0,
                max_abs_diff: 0.0,
                relative_error: 0.0,
            },
            time_human: StatsLinSolTimeHuman {
                read_matrix: String::new(),
                initialize_array: Vec::new(),
                initialize: String::new(),
                preprocess_array: Vec::new(),
                preprocess: String::new(),
                solve_array: Vec::new(),
                solve: String::new(),
                total_ifs_array: Vec::new(),
                total_ifs: String::new(),
                verify: String::new(),
            },
            time_nanoseconds: StatsLinSolTimeNanoseconds {
                read_matrix: 0,
                initialize_array: Vec::new(),
                initialize: 0,
                preprocess_array: Vec::new(),
                preprocess: 0,
                solve_array: Vec::new(),
                solve: 0,
                total_ifs_array: Vec::new(),
                total_ifs: 0,
                verify: 0,
            },
            mumps_stats: StatsLinSolMUMPS {
                inf_norm_a: 0.0,
                inf_norm_x: 0.0,
                scaled_residual: 0.0,
                backward_error_omega1: 0.0,
                backward_error_omega2: 0.0,
                normalized_delta_x: 0.0,
                condition_number1: 0.0,
                condition_number2: 0.0,
            },
            iterative_stats: None,
        }
    }

    /// Sets the matrix name as the stem of a file path
    pub fn set_matrix_name_from_path(&mut self, filepath: &str) {
        self.matrix.name = match Path::new(filepath).file_stem() {
            Some(v) => match v.to_str() {
                Some(w) => w.to_string(),
                None => "Unknown".to_string(),
            },
            None => "Unknown".to_string(),
        };
    }

    /// Sets the matrix info from a reference to a COO matrix
    pub fn set_matrix_info_from_coo<T>(&mut self, coo: &NumCooMatrix<T>)
    where
        T: AddAssign + MulAssign + Num + NumCast + Copy + DeserializeOwned + Serialize + 'static,
    {
        let (nrow, ncol, nnz, sym) = coo.get_info();
        self.matrix.nrow = nrow;
        self.matrix.ncol = ncol;
        self.matrix.nnz = nnz;
        self.matrix.nnz_actual = coo.get_actual_nnz();
        self.matrix.complex = TypeId::of::<T>() == TypeId::of::<Complex64>();
        self.matrix.symmetric = format!("{:?}", sym);
    }

    /// Gets a JSON representation of the stats structure
    pub fn get_json(&mut self) -> String {
        self.compute_derived_values();
        serde_json::to_string_pretty(&self).unwrap()
    }

    /// Reads a JSON file containing a StatsLinSol data
    ///
    /// # Input
    ///
    /// * `full_path` -- may be a String, &str, or Path
    pub fn read_json<P>(full_path: &P) -> Result<Self, StrError>
    where
        P: AsRef<OsStr> + ?Sized,
    {
        let path = Path::new(full_path).to_path_buf();
        let input = File::open(path).map_err(|_| "cannot open file")?;
        let buffered = BufReader::new(input);
        let stat = serde_json::from_reader(buffered).map_err(|_| "cannot parse JSON file")?;
        Ok(stat)
    }

    /// Writes a JSON file with the results
    ///
    /// # Input
    ///
    /// * `full_path` -- may be a String, &str, or Path
    pub fn write_json<P>(&mut self, full_path: &P) -> Result<(), StrError>
    where
        P: AsRef<OsStr> + ?Sized,
    {
        self.compute_derived_values();
        let path = Path::new(full_path).to_path_buf();
        if let Some(p) = path.parent() {
            fs::create_dir_all(p).map_err(|_| "cannot create directory")?;
        }
        let mut file = File::create(&path).map_err(|_| "cannot create file")?;
        serde_json::to_writer_pretty(&mut file, &self).map_err(|_| "cannot write file")?;
        Ok(())
    }

    /// Computes derived values
    fn compute_derived_values(&mut self) {
        // number of threads
        self.output.openmp_num_threads = get_num_threads();

        // average of timings
        let nrun = self.time_nanoseconds.initialize_array.len();
        assert_eq!(self.time_nanoseconds.preprocess_array.len(), nrun);
        assert_eq!(self.time_nanoseconds.solve_array.len(), nrun);
        if nrun > 0 {
            self.time_nanoseconds.total_ifs_array = vec![0; nrun];
            self.time_human.initialize_array = vec![String::new(); nrun];
            self.time_human.preprocess_array = vec![String::new(); nrun];
            self.time_human.solve_array = vec![String::new(); nrun];
            self.time_human.total_ifs_array = vec![String::new(); nrun];
            let mut ave_init = 0.0;
            let mut ave_fact = 0.0;
            let mut ave_solve = 0.0;
            let mut ave_total = 0.0;
            for i in 0..nrun {
                let t_init = self.time_nanoseconds.initialize_array[i];
                let t_fact = self.time_nanoseconds.preprocess_array[i];
                let t_solve = self.time_nanoseconds.solve_array[i];
                let t_total = t_init + t_fact + t_solve;
                ave_init += t_init as f64;
                ave_fact += t_fact as f64;
                ave_solve += t_solve as f64;
                ave_total += t_total as f64;
                self.time_nanoseconds.total_ifs_array[i] = t_total;
                self.time_human.initialize_array[i] = format_nanoseconds(t_init);
                self.time_human.preprocess_array[i] = format_nanoseconds(t_fact);
                self.time_human.solve_array[i] = format_nanoseconds(t_solve);
                self.time_human.total_ifs_array[i] = format_nanoseconds(t_total);
            }
            let den = nrun as f64;
            ave_init /= den;
            ave_fact /= den;
            ave_solve /= den;
            ave_total /= den;
            self.time_nanoseconds.initialize = ave_init as u128;
            self.time_nanoseconds.preprocess = ave_fact as u128;
            self.time_nanoseconds.solve = ave_solve as u128;
            self.time_nanoseconds.total_ifs = ave_total as u128;
        }

        // human time strings
        self.time_human.read_matrix = format_nanoseconds(self.time_nanoseconds.read_matrix);
        self.time_human.initialize = format_nanoseconds(self.time_nanoseconds.initialize);
        self.time_human.preprocess = format_nanoseconds(self.time_nanoseconds.preprocess);
        self.time_human.solve = format_nanoseconds(self.time_nanoseconds.solve);
        self.time_human.total_ifs = format_nanoseconds(self.time_nanoseconds.total_ifs);
        self.time_human.verify = format_nanoseconds(self.time_nanoseconds.verify);
    }

    /// Clears partial run arrays, scalar timings, and verification fields after an out-of-memory error
    ///
    /// Keeps `main`, `matrix`, `requests`, and `time_nanoseconds.read_matrix` intact so the
    /// emitted OOM JSON contains no partial timings or errors from earlier runs in a multi-run loop.
    pub fn clear_run_results(&mut self) {
        let blank = StatsLinSol::new();
        self.main.solved = false;
        self.main.stopped_by_time_guard = false;
        self.output = blank.output;
        self.determinant = blank.determinant;
        self.verify = blank.verify;
        self.mumps_stats = blank.mumps_stats;
        self.iterative_stats = None;
        let read_matrix = self.time_nanoseconds.read_matrix;
        self.time_nanoseconds = blank.time_nanoseconds;
        self.time_nanoseconds.read_matrix = read_matrix;
        self.time_human = blank.time_human;
    }
}

/// Returns true if the error message indicates an out-of-memory condition
///
/// Covers UMFPACK, MUMPS, and the iterative solver.
pub fn is_memory_error(e: &str) -> bool {
    e.contains("MALLOC") || e.contains("Not enough memory") || e.contains("ALLOC_FAILED") || e.contains("memory is too small")
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::StatsLinSol;
    use crate::{CooMatrix, FaerMethod, IterativeMethod, LinSolParams, LinSolTrait, MMsym, Samples, SolverFaer, SolverIterative, read_matrix_market};
    use russell_lab::Vector;

    #[test]
    fn derive_works() {
        let stats = StatsLinSol::new();
        let clone = stats.clone();
        assert!(!format!("{:?}", stats).is_empty());
        assert_eq!(clone.main.platform, stats.main.platform);
        // serialize
        let json_out = serde_json::to_string(&stats).unwrap();
        // deserialize
        let json_in: StatsLinSol = serde_json::from_str(&json_out).unwrap();
        assert_eq!(json_in.main.platform, stats.main.platform);
    }

    #[test]
    fn set_matrix_name_from_path_works() {
        let mut stats = StatsLinSol::new();
        stats.set_matrix_name_from_path("/tmp/russell/just-testing.stats");
        assert_eq!(stats.matrix.name, "just-testing");

        stats.set_matrix_name_from_path("/tmp/russell/.stats");
        assert_eq!(stats.matrix.name, ".stats");

        stats.set_matrix_name_from_path("/tmp/russell/");
        assert_eq!(stats.matrix.name, "russell"); // << no really what we want

        stats.set_matrix_name_from_path("");
        assert_eq!(stats.matrix.name, "Unknown");

        stats.set_matrix_name_from_path("🐶🐶🐶.stats");
        assert_eq!(stats.matrix.name, "🐶🐶🐶");
    }

    #[test]
    fn set_matrix_info_from_coo_works() {
        let (_, coo) = read_matrix_market("./data/matrix_market/ok_complex_symmetric_small.mtx", MMsym::LeaveAsLower).unwrap();
        let mut stats = StatsLinSol::new();
        stats.set_matrix_info_from_coo(coo.as_ref().unwrap());
        assert_eq!(stats.matrix.nrow, 5);
        assert_eq!(stats.matrix.ncol, 5);
        assert_eq!(stats.matrix.nnz, 7);
        assert_eq!(stats.matrix.nnz_actual, 11);
        assert_eq!(stats.matrix.complex, true);
        assert_eq!(stats.matrix.symmetric, "YesLower");
    }

    const ONE_SEC: u128 = 1000000000;

    fn generate_data() -> StatsLinSol {
        let mut stats = StatsLinSol::new();
        stats.time_nanoseconds.read_matrix = ONE_SEC;
        stats.time_nanoseconds.initialize_array = vec![ONE_SEC, 2 * ONE_SEC, 6 * ONE_SEC]; // 1 + 2 + 6 = 9
        stats.time_nanoseconds.preprocess_array = vec![2 * ONE_SEC, 3 * ONE_SEC, 7 * ONE_SEC]; // 2 + 3 + 7 = 12
        stats.time_nanoseconds.solve_array = vec![3 * ONE_SEC, 7 * ONE_SEC, 14 * ONE_SEC]; // 3 + 7 + 14 = 24
        stats.time_nanoseconds.verify = ONE_SEC * 4;
        stats
    }

    #[test]
    fn get_json_works() {
        let mut stats = generate_data();

        let json = stats.get_json();
        assert!(!json.is_empty());
        assert!(stats.output.openmp_num_threads > 0);

        assert_eq!(stats.time_nanoseconds.initialize, 3 * ONE_SEC); // 9/3
        assert_eq!(stats.time_nanoseconds.preprocess, 4 * ONE_SEC); // 12/3
        assert_eq!(stats.time_nanoseconds.solve, 8 * ONE_SEC); // 24/3
        assert_eq!(
            stats.time_nanoseconds.total_ifs_array,
            &[6 * ONE_SEC, 12 * ONE_SEC, 27 * ONE_SEC] // 6 + 12 + 27 = 45
        );
        assert_eq!(stats.time_nanoseconds.total_ifs, 15 * ONE_SEC); // 45/3

        assert_eq!(stats.time_human.read_matrix, "1s");
        assert_eq!(stats.time_human.initialize, "3s");
        assert_eq!(stats.time_human.preprocess, "4s");
        assert_eq!(stats.time_human.solve, "8s");
        assert_eq!(stats.time_human.total_ifs, "15s");
        assert_eq!(stats.time_human.verify, "4s");

        assert_eq!(stats.time_human.initialize_array, &["1s", "2s", "6s"]);
        assert_eq!(stats.time_human.preprocess_array, &["2s", "3s", "7s"]);
        assert_eq!(stats.time_human.solve_array, &["3s", "7s", "14s"]);
        assert_eq!(stats.time_human.total_ifs_array, &["6s", "12s", "27s"]);
    }

    #[test]
    fn run_flags_round_trip() {
        let mut stats = StatsLinSol::new();
        assert!(!stats.main.solved);
        assert!(!stats.main.stopped_by_time_guard);
        stats.main.solved = true;
        stats.main.stopped_by_time_guard = true;
        let json = stats.get_json();
        assert!(json.contains("\"solved\": true"));
        assert!(json.contains("\"stopped_by_time_guard\": true"));
        let parsed: StatsLinSol = serde_json::from_str(&json).unwrap();
        assert!(parsed.main.solved);
        assert!(parsed.main.stopped_by_time_guard);
    }

    #[test]
    fn write_read_json_works() {
        let mut stats = generate_data();

        let path = "/tmp/russell/write_json_works.json";
        stats.write_json(path).unwrap();
        let res = StatsLinSol::read_json(path).unwrap();
        assert!(res.output.openmp_num_threads > 0);

        assert_eq!(res.time_nanoseconds.initialize, 3 * ONE_SEC); // 9/3
        assert_eq!(res.time_nanoseconds.preprocess, 4 * ONE_SEC); // 12/3
        assert_eq!(res.time_nanoseconds.solve, 8 * ONE_SEC); // 24/3
        assert_eq!(
            res.time_nanoseconds.total_ifs_array,
            &[6 * ONE_SEC, 12 * ONE_SEC, 27 * ONE_SEC] // 6 + 12 + 27 = 45
        );
        assert_eq!(res.time_nanoseconds.total_ifs, 15 * ONE_SEC); // 45/3

        assert_eq!(res.time_human.read_matrix, "1s");
        assert_eq!(res.time_human.initialize, "3s");
        assert_eq!(res.time_human.preprocess, "4s");
        assert_eq!(res.time_human.solve, "8s");
        assert_eq!(res.time_human.total_ifs, "15s");
        assert_eq!(res.time_human.verify, "4s");

        assert_eq!(res.time_human.initialize_array, &["1s", "2s", "6s"]);
        assert_eq!(res.time_human.preprocess_array, &["2s", "3s", "7s"]);
        assert_eq!(res.time_human.solve_array, &["3s", "7s", "14s"]);
        assert_eq!(res.time_human.total_ifs_array, &["6s", "12s", "27s"]);
    }

    /// Returns a small symmetric positive-definite system
    fn positive_definite_system() -> (CooMatrix, Vector) {
        let (coo, _, _, _) = Samples::positive_definite_3x3_full();
        (coo, Vector::filled(3, 1.0))
    }

    /// Runs one Faer solve and updates the statistics
    fn run_faer(params: Option<LinSolParams>) -> StatsLinSol {
        let (coo, rhs) = positive_definite_system();
        let mut solver = SolverFaer::new().unwrap();
        solver.preprocess(&coo, params).unwrap();
        let mut x = Vector::new(3);
        solver.solve(&mut x, &rhs, false).unwrap();
        let mut stats = StatsLinSol::new();
        solver.update_stats(&mut stats);
        stats
    }

    /// Runs one CG solve and updates the statistics
    fn run_iterative() -> StatsLinSol {
        let (coo, rhs) = positive_definite_system();
        let mut solver = SolverIterative::new(IterativeMethod::Cg).unwrap();
        solver.preprocess(&coo, None).unwrap();
        let mut x = Vector::new(3);
        solver.solve(&mut x, &rhs, false).unwrap();
        let mut stats = StatsLinSol::new();
        solver.update_stats(&mut stats);
        stats
    }

    #[test]
    fn iterative_json_has_the_block() {
        let mut stats = run_iterative();
        let json = stats.get_json();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        let block = value.get("iterative_stats").expect("the iterative block is missing");
        assert_eq!(block["method"], "CG");
        assert_eq!(block["preconditioner"], "AMG");
        assert!(block["amg_levels"].as_u64().unwrap() > 0);
        assert_eq!(block["iterations_array"].as_array().unwrap().len(), 1);
        assert_eq!(block["relative_residual_array"].as_array().unwrap().len(), 1);
        assert!(!json.contains("factorize"));
    }

    #[test]
    fn faer_json_has_no_iterative_block_and_records_faer_fields() {
        let mut stats = run_faer(None);
        let json = stats.get_json();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(value.get("iterative_stats").is_some_and(|v| v.is_null()));
        assert_eq!(stats.main.solver, "Faer");

        let mut simp_params = LinSolParams::new();
        simp_params.faer_method = FaerMethod::Simplicial;
        let mut simp_stats = run_faer(Some(simp_params));
        let simp_json = simp_stats.get_json();
        let simp_value: serde_json::Value = serde_json::from_str(&simp_json).unwrap();
        assert_eq!(simp_value["output"]["faer_effective_method"].as_str(), Some("Simplicial"));
        assert_eq!(simp_value["output"]["faer_flop_ratio"].as_f64(), Some(0.0));

        let mut sup_params = LinSolParams::new();
        sup_params.faer_method = FaerMethod::Supernodal;
        let mut sup_stats = run_faer(Some(sup_params));
        let sup_json = sup_stats.get_json();
        let sup_value: serde_json::Value = serde_json::from_str(&sup_json).unwrap();
        assert_eq!(sup_value["output"]["faer_effective_method"].as_str(), Some("Supernodal"));
        assert!(sup_value["output"]["faer_flop_ratio"].as_f64().unwrap() > 0.0);
    }

    #[test]
    fn json_without_required_faer_output_fields_is_rejected() {
        let mut stats = run_faer(Some(LinSolParams {
            faer_method: FaerMethod::Supernodal,
            ..LinSolParams::new()
        }));
        let json = stats.get_json();
        let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
        value["main"]["solver"] = serde_json::Value::String("Direct".to_string());
        let output = value["output"].as_object_mut().unwrap();
        output.remove("faer_effective_method");
        output.remove("faer_flop_ratio");
        let legacy_json = serde_json::to_string(&value).unwrap();
        let res = serde_json::from_str::<StatsLinSol>(&legacy_json);
        let err = res.unwrap_err().to_string();
        assert!(err.contains("missing field"), "unexpected error: {err}");
    }

    #[test]
    fn clear_run_results_resets_partial_runs_and_keeps_read_matrix() {
        let mut stats = generate_data();
        stats.main.solver = "CG".to_string();
        stats.matrix.name = "test_mat".to_string();
        stats.main.solved = true;
        stats.main.stopped_by_time_guard = true;
        stats.verify.max_abs_diff = 1e-12;
        stats.verify.relative_error = 1e-13;
        stats.iterative_stats = Some(super::StatsLinSolIterative {
            method: "CG".to_string(),
            iterations_array: vec![5],
            relative_residual_array: vec![1e-11],
            ..super::StatsLinSolIterative::default()
        });
        stats.clear_run_results();
        stats.main.out_of_memory = true;
        let json = stats.get_json();
        assert_eq!(stats.main.solver, "CG");
        assert_eq!(stats.matrix.name, "test_mat");
        assert!(!stats.main.solved);
        assert!(!stats.main.stopped_by_time_guard);
        assert!(stats.main.out_of_memory);
        assert_eq!(stats.time_nanoseconds.read_matrix, ONE_SEC);
        assert_eq!(stats.time_human.read_matrix, "1s");
        assert_eq!(stats.time_nanoseconds.initialize, 0);
        assert_eq!(stats.time_nanoseconds.preprocess, 0);
        assert_eq!(stats.time_nanoseconds.solve, 0);
        assert_eq!(stats.time_nanoseconds.total_ifs, 0);
        assert_eq!(stats.time_nanoseconds.verify, 0);
        assert!(stats.time_nanoseconds.initialize_array.is_empty());
        assert!(stats.time_nanoseconds.preprocess_array.is_empty());
        assert!(stats.time_nanoseconds.solve_array.is_empty());
        assert!(stats.time_nanoseconds.total_ifs_array.is_empty());
        assert!(stats.time_human.initialize_array.is_empty());
        assert_eq!(stats.verify.max_abs_diff, 0.0);
        assert_eq!(stats.verify.relative_error, 0.0);
        assert!(stats.iterative_stats.is_none());
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(value.get("iterative_stats").is_some_and(|v| v.is_null()));
    }

    #[test]
    fn json_without_the_iterative_block_still_deserializes() {
        let mut stats = run_iterative();
        let json = stats.get_json();
        let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
        value.as_object_mut().unwrap().remove("iterative_stats");
        let old = serde_json::to_string(&value).unwrap();
        let parsed: StatsLinSol = serde_json::from_str(&old).unwrap();
        assert!(parsed.iterative_stats.is_none());
    }

    #[test]
    fn old_factorize_time_keys_are_rejected() {
        let mut stats = run_faer(None);
        let json = stats.get_json();
        let old = json
            .replace("\"preprocess_array\"", "\"factorize_array\"")
            .replace("\"preprocess\"", "\"factorize\"");
        assert_ne!(old, json);
        assert!(serde_json::from_str::<StatsLinSol>(&old).is_err());
    }

    #[test]
    fn blas_lib_reports_the_build() {
        let stats = StatsLinSol::new();
        let expected = if russell_lab::using_intel_mkl() {
            "Intel MKL"
        } else if cfg!(feature = "open_blas") {
            "OpenBLAS"
        } else {
            "Native"
        };
        assert_eq!(stats.main.blas_lib, expected);
        #[cfg(not(any(feature = "open_blas", feature = "intel_mkl")))]
        assert_eq!(stats.main.blas_lib, "Native");
    }

    #[test]
    fn update_stats_pushes_one_entry_per_run() {
        let (coo, rhs) = positive_definite_system();
        let mut solver = SolverIterative::new(IterativeMethod::Cg).unwrap();
        let mut stats = StatsLinSol::new();

        for _ in 0..2 {
            solver.preprocess(&coo, None).unwrap();
            let mut x = Vector::new(3);
            solver.solve(&mut x, &rhs, false).unwrap();
            solver.update_stats(&mut stats);
        }
        let block = stats.iterative_stats.as_ref().unwrap();
        assert_eq!(block.iterations_array.len(), 2);
        assert_eq!(block.relative_residual_array.len(), 2);
        assert_eq!(stats.time_nanoseconds.solve_array.len(), 2);

        // No solve since the last preprocess: nothing is pushed.
        solver.preprocess(&coo, None).unwrap();
        solver.update_stats(&mut stats);
        let block = stats.iterative_stats.as_ref().unwrap();
        assert_eq!(block.iterations_array.len(), 2);
        assert_eq!(block.relative_residual_array.len(), 2);
        assert_eq!(stats.time_nanoseconds.solve_array.len(), 2);
    }

    #[test]
    fn derive_with_no_runs_works() {
        let mut stats = StatsLinSol::new();
        let json = stats.get_json();
        assert!(!json.is_empty());

        assert_eq!(stats.time_nanoseconds.initialize, 0);
        assert_eq!(stats.time_nanoseconds.preprocess, 0);
        assert_eq!(stats.time_nanoseconds.solve, 0);
        assert_eq!(stats.time_nanoseconds.total_ifs, 0);
        assert!(stats.time_nanoseconds.total_ifs_array.is_empty());

        assert_eq!(stats.time_human.read_matrix, "0ns");
        assert_eq!(stats.time_human.initialize, "0ns");
        assert_eq!(stats.time_human.preprocess, "0ns");
        assert_eq!(stats.time_human.solve, "0ns");
        assert_eq!(stats.time_human.total_ifs, "0ns");
        assert_eq!(stats.time_human.verify, "0ns");

        assert!(stats.time_human.initialize_array.is_empty());
        assert!(stats.time_human.preprocess_array.is_empty());
        assert!(stats.time_human.solve_array.is_empty());
        assert!(stats.time_human.total_ifs_array.is_empty());
    }

    #[test]
    fn is_memory_error_works() {
        use super::is_memory_error;

        assert!(is_memory_error("MALLOC failed"));
        assert!(is_memory_error("Not enough memory on device"));
        assert!(is_memory_error("ALLOC_FAILED in some routine"));
        assert!(is_memory_error("memory is too small for operation"));

        assert!(!is_memory_error(""));
        assert!(!is_memory_error("some other error"));
    }

    #[test]
    fn read_json_errors() {
        assert!(StatsLinSol::read_json("/tmp/nonexistent_file_xyz_stats.json").is_err());
        assert!(StatsLinSol::read_json("/tmp").is_err());
    }
}
