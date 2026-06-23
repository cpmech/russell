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
    pub platform: String,
    pub blas_lib: String,
    pub solver: String,
    pub local_sparse: bool,
    pub out_of_memory: bool,
}

/// Holds information about the sparse matrix
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StatsLinSolMatrix {
    pub name: String,
    pub nrow: usize,
    pub ncol: usize,
    pub nnz: usize,
    pub nnz_actual: usize, // corresponds to Pattern Entries reported by [SuiteSparse Matrix Collection](https://sparse.tamu.edu)
    pub complex: bool,
    pub symmetric: String,
}

/// Holds some requested parameters
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StatsLinSolRequests {
    pub ordering: String,
    pub scaling: String,
    pub matching: String,
    pub pivoting: String,
    pub mumps_num_threads: usize,
    pub positive_definite: bool,
    pub hybrid_memory_factor: Option<f64>,
}

/// Holds some output parameters
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StatsLinSolOutput {
    pub effective_ordering: String,
    pub effective_scaling: String,
    pub effective_matching: String,
    pub effective_pivoting: String,
    pub effective_mumps_num_threads: usize,
    pub openmp_num_threads: usize,
    pub umfpack_strategy: String,
    pub umfpack_rcond_estimate: f64, // reciprocal condition number estimate
}

/// Holds the determinant of the coefficient matrix (if requested)
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StatsLinSolDeterminant {
    // det = mantissa * pow(base, exponent)
    pub mantissa_real: f64, // the real part of the mantissa
    pub mantissa_imag: f64, // the imaginary part of the mantissa (if complex)
    pub base: f64,
    pub exponent: f64,
}

/// Holds the average computer times in human readable format (post-processed)
///
/// **Note:** These are automatically converted from TimeNanoseconds when calling [StatsLinSol::get_json]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StatsLinSolTimeHuman {
    pub read_matrix: String,
    pub initialize_array: Vec<String>, // raw data; not averaged
    pub initialize: String,            // average
    pub factorize_array: Vec<String>,  // raw data; not averaged
    pub factorize: String,             // average
    pub solve_array: Vec<String>,      // raw data; not averaged
    pub solve: String,                 // average
    pub total_ifs_array: Vec<String>,  // raw data; not averaged: initialize + factorize + solve
    pub total_ifs: String,             // average: initialize + factorize + solve
    pub verify: String,
}

/// Holds the average computer times in nanoseconds
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StatsLinSolTimeNanoseconds {
    pub read_matrix: u128,
    pub initialize_array: Vec<u128>, // raw data; not averaged
    pub initialize: u128,            // average
    pub factorize_array: Vec<u128>,  // raw data; not averaged
    pub factorize: u128,             // average
    pub solve_array: Vec<u128>,      // raw data; not averaged
    pub solve: u128,                 // average
    pub total_ifs_array: Vec<u128>,  // raw data; not averaged
    pub total_ifs: u128,             // average
    pub verify: u128,
}

/// Holds information about the solution of a linear system
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StatsLinSol {
    pub main: StatsLinSolMain,
    pub matrix: StatsLinSolMatrix,
    pub requests: StatsLinSolRequests,
    pub output: StatsLinSolOutput,
    pub determinant: StatsLinSolDeterminant,
    pub verify: VerifyLinSys,
    pub time_human: StatsLinSolTimeHuman,
    pub time_nanoseconds: StatsLinSolTimeNanoseconds,
    pub mumps_stats: StatsLinSolMUMPS,
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
                } else {
                    "OpenBLAS".to_string()
                },
                solver: unknown.clone(),
                local_sparse: if cfg!(feature = "local_sparse") { true } else { false },
                out_of_memory: false,
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
                matching: unknown.clone(),
                pivoting: unknown.clone(),
                mumps_num_threads: 0,
                positive_definite: false,
                hybrid_memory_factor: None,
            },
            output: StatsLinSolOutput {
                effective_ordering: unknown.clone(),
                effective_scaling: unknown.clone(),
                effective_matching: unknown.clone(),
                effective_pivoting: unknown.clone(),
                effective_mumps_num_threads: 0,
                openmp_num_threads: 0,
                umfpack_strategy: unknown.clone(),
                umfpack_rcond_estimate: 0.0,
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
                factorize_array: Vec::new(),
                factorize: String::new(),
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
                factorize_array: Vec::new(),
                factorize: 0,
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
        assert_eq!(self.time_nanoseconds.factorize_array.len(), nrun);
        assert_eq!(self.time_nanoseconds.solve_array.len(), nrun);
        if nrun > 0 {
            self.time_nanoseconds.total_ifs_array = vec![0; nrun];
            self.time_human.initialize_array = vec![String::new(); nrun];
            self.time_human.factorize_array = vec![String::new(); nrun];
            self.time_human.solve_array = vec![String::new(); nrun];
            self.time_human.total_ifs_array = vec![String::new(); nrun];
            let mut ave_init = 0.0;
            let mut ave_fact = 0.0;
            let mut ave_solve = 0.0;
            let mut ave_total = 0.0;
            for i in 0..nrun {
                let t_init = self.time_nanoseconds.initialize_array[i];
                let t_fact = self.time_nanoseconds.factorize_array[i];
                let t_solve = self.time_nanoseconds.solve_array[i];
                let t_total = t_init + t_fact + t_solve;
                ave_init += t_init as f64;
                ave_fact += t_fact as f64;
                ave_solve += t_solve as f64;
                ave_total += t_total as f64;
                self.time_nanoseconds.total_ifs_array[i] = t_total;
                self.time_human.initialize_array[i] = format_nanoseconds(t_init);
                self.time_human.factorize_array[i] = format_nanoseconds(t_fact);
                self.time_human.solve_array[i] = format_nanoseconds(t_solve);
                self.time_human.total_ifs_array[i] = format_nanoseconds(t_total);
            }
            let den = nrun as f64;
            ave_init /= den;
            ave_fact /= den;
            ave_solve /= den;
            ave_total /= den;
            self.time_nanoseconds.initialize = ave_init as u128;
            self.time_nanoseconds.factorize = ave_fact as u128;
            self.time_nanoseconds.solve = ave_solve as u128;
            self.time_nanoseconds.total_ifs = ave_total as u128;
        }

        // human time strings
        self.time_human.read_matrix = format_nanoseconds(self.time_nanoseconds.read_matrix);
        self.time_human.initialize = format_nanoseconds(self.time_nanoseconds.initialize);
        self.time_human.factorize = format_nanoseconds(self.time_nanoseconds.factorize);
        self.time_human.solve = format_nanoseconds(self.time_nanoseconds.solve);
        self.time_human.total_ifs = format_nanoseconds(self.time_nanoseconds.total_ifs);
        self.time_human.verify = format_nanoseconds(self.time_nanoseconds.verify);
    }
}

/// Returns true if the error message indicates an out-of-memory condition
///
/// Covers all solvers: UMFPACK, MUMPS, and cuDSS.
pub fn is_memory_error(e: &str) -> bool {
    e.contains("MALLOC")
        || e.contains("Not enough memory")
        || e.contains("ALLOC_FAILED")
        || e.contains("cudaMalloc")
        || e.contains("memory is too small")
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::StatsLinSol;
    use crate::{MMsym, read_matrix_market};

    #[test]
    fn derive_works() {
        let stats = StatsLinSol::new();
        let clone = stats.clone();
        assert!(format!("{:?}", stats).len() > 0);
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
        let (_, coo) = read_matrix_market(
            "./data/matrix_market/ok_complex_symmetric_small.mtx",
            MMsym::LeaveAsLower,
        )
        .unwrap();
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
        stats.time_nanoseconds.factorize_array = vec![2 * ONE_SEC, 3 * ONE_SEC, 7 * ONE_SEC]; // 2 + 3 + 7 = 12
        stats.time_nanoseconds.solve_array = vec![3 * ONE_SEC, 7 * ONE_SEC, 14 * ONE_SEC]; // 3 + 7 + 14 = 24
        stats.time_nanoseconds.verify = ONE_SEC * 4;
        stats
    }

    #[test]
    fn get_json_works() {
        let mut stats = generate_data();

        let json = stats.get_json();
        assert!(json.len() > 0);
        assert!(stats.output.openmp_num_threads > 0);

        assert_eq!(stats.time_nanoseconds.initialize, 3 * ONE_SEC); // 9/3
        assert_eq!(stats.time_nanoseconds.factorize, 4 * ONE_SEC); // 12/3
        assert_eq!(stats.time_nanoseconds.solve, 8 * ONE_SEC); // 24/3
        assert_eq!(
            stats.time_nanoseconds.total_ifs_array,
            &[6 * ONE_SEC, 12 * ONE_SEC, 27 * ONE_SEC] // 6 + 12 + 27 = 45
        );
        assert_eq!(stats.time_nanoseconds.total_ifs, 15 * ONE_SEC); // 45/3

        assert_eq!(stats.time_human.read_matrix, "1s");
        assert_eq!(stats.time_human.initialize, "3s");
        assert_eq!(stats.time_human.factorize, "4s");
        assert_eq!(stats.time_human.solve, "8s");
        assert_eq!(stats.time_human.total_ifs, "15s");
        assert_eq!(stats.time_human.verify, "4s");

        assert_eq!(stats.time_human.initialize_array, &["1s", "2s", "6s"]);
        assert_eq!(stats.time_human.factorize_array, &["2s", "3s", "7s"]);
        assert_eq!(stats.time_human.solve_array, &["3s", "7s", "14s"]);
        assert_eq!(stats.time_human.total_ifs_array, &["6s", "12s", "27s"]);
    }

    #[test]
    fn write_read_json_works() {
        let mut stats = generate_data();

        let path = "/tmp/russell/write_json_works.json";
        stats.write_json(path).unwrap();
        let res = StatsLinSol::read_json(path).unwrap();
        assert!(res.output.openmp_num_threads > 0);

        assert_eq!(res.time_nanoseconds.initialize, 3 * ONE_SEC); // 9/3
        assert_eq!(res.time_nanoseconds.factorize, 4 * ONE_SEC); // 12/3
        assert_eq!(res.time_nanoseconds.solve, 8 * ONE_SEC); // 24/3
        assert_eq!(
            res.time_nanoseconds.total_ifs_array,
            &[6 * ONE_SEC, 12 * ONE_SEC, 27 * ONE_SEC] // 6 + 12 + 27 = 45
        );
        assert_eq!(res.time_nanoseconds.total_ifs, 15 * ONE_SEC); // 45/3

        assert_eq!(res.time_human.read_matrix, "1s");
        assert_eq!(res.time_human.initialize, "3s");
        assert_eq!(res.time_human.factorize, "4s");
        assert_eq!(res.time_human.solve, "8s");
        assert_eq!(res.time_human.total_ifs, "15s");
        assert_eq!(res.time_human.verify, "4s");

        assert_eq!(res.time_human.initialize_array, &["1s", "2s", "6s"]);
        assert_eq!(res.time_human.factorize_array, &["2s", "3s", "7s"]);
        assert_eq!(res.time_human.solve_array, &["3s", "7s", "14s"]);
        assert_eq!(res.time_human.total_ifs_array, &["6s", "12s", "27s"]);
    }

    #[test]
    fn derive_with_no_runs_works() {
        let mut stats = StatsLinSol::new();
        let json = stats.get_json();
        assert!(json.len() > 0);

        assert_eq!(stats.time_nanoseconds.initialize, 0);
        assert_eq!(stats.time_nanoseconds.factorize, 0);
        assert_eq!(stats.time_nanoseconds.solve, 0);
        assert_eq!(stats.time_nanoseconds.total_ifs, 0);
        assert!(stats.time_nanoseconds.total_ifs_array.is_empty());

        assert_eq!(stats.time_human.read_matrix, "0ns");
        assert_eq!(stats.time_human.initialize, "0ns");
        assert_eq!(stats.time_human.factorize, "0ns");
        assert_eq!(stats.time_human.solve, "0ns");
        assert_eq!(stats.time_human.total_ifs, "0ns");
        assert_eq!(stats.time_human.verify, "0ns");

        assert!(stats.time_human.initialize_array.is_empty());
        assert!(stats.time_human.factorize_array.is_empty());
        assert!(stats.time_human.solve_array.is_empty());
        assert!(stats.time_human.total_ifs_array.is_empty());
    }

    #[test]
    fn is_memory_error_works() {
        use super::is_memory_error;

        assert!(is_memory_error("MALLOC failed"));
        assert!(is_memory_error("Not enough memory on device"));
        assert!(is_memory_error("ALLOC_FAILED in some routine"));
        assert!(is_memory_error("cudaMalloc returned error"));
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
