use super::{StatsLinSolMUMPS, VerifyLinSys};
use crate::StrError;
use russell_lab::{format_nanoseconds, get_num_threads, using_intel_mkl};
use serde::{Deserialize, Serialize};
use serde_json;
use std::ffi::OsStr;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

/// Holds the main information such as platform
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StatsLinSolMain {
    pub platform: String,
    pub blas_lib: String,
    pub solver: String,
}

/// Holds information about the sparse matrix
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StatsLinSolMatrix {
    pub name: String,
    pub nrow: usize,
    pub ncol: usize,
    pub nnz: usize,
    pub symmetry: String,
}

/// Holds some requested parameters
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StatsLinSolRequests {
    pub ordering: String,
    pub scaling: String,
    pub mumps_num_threads: usize,
}

/// Holds some output parameters
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StatsLinSolOutput {
    pub effective_ordering: String,
    pub effective_scaling: String,
    pub effective_mumps_num_threads: usize,
    pub openmp_num_threads: usize,
    pub umfpack_strategy: String,
    pub umfpack_rcond_estimate: f64, // reciprocal condition number estimate
}

/// Holds the determinant of the coefficient matrix (if requested)
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StatsLinSolDeterminant {
    // det = mantissa * pow(base, exponent)
    pub mantissa: f64,
    pub base: f64,
    pub exponent: f64,
}

/// Holds the computer times in human readable format (post-processed)
///
/// **Note:** These are automatically converted from TimeNanoseconds when calling [StatsLinSol::get_json]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StatsLinSolTimeHuman {
    pub read_matrix: String,
    pub initialize: String,
    pub factorize: String,
    pub solve: String,
    pub total_ifs: String, // initialize + factorize + solve
    pub verify: String,
}

/// Holds the computer times in nanoseconds
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StatsLinSolTimeNanoseconds {
    pub read_matrix: u128,
    pub initialize: u128,
    pub factorize: u128,
    pub solve: u128,
    pub total_ifs: u128, // initialize + factorize + solve
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
            },
            matrix: StatsLinSolMatrix {
                name: unknown.clone(),
                nrow: 0,
                ncol: 0,
                nnz: 0,
                symmetry: unknown.clone(),
            },
            requests: StatsLinSolRequests {
                ordering: unknown.clone(),
                scaling: unknown.clone(),
                mumps_num_threads: 0,
            },
            output: StatsLinSolOutput {
                effective_ordering: unknown.clone(),
                effective_scaling: unknown.clone(),
                effective_mumps_num_threads: 0,
                openmp_num_threads: 0,
                umfpack_strategy: unknown.clone(),
                umfpack_rcond_estimate: 0.0,
            },
            determinant: StatsLinSolDeterminant {
                mantissa: 0.0,
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
                initialize: String::new(),
                factorize: String::new(),
                solve: String::new(),
                total_ifs: String::new(),
                verify: String::new(),
            },
            time_nanoseconds: StatsLinSolTimeNanoseconds {
                read_matrix: 0,
                initialize: 0,
                factorize: 0,
                solve: 0,
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

    /// Gets a JSON representation of the stats structure
    pub fn get_json(&mut self) -> String {
        self.output.openmp_num_threads = get_num_threads();
        self.time_nanoseconds.total_ifs =
            self.time_nanoseconds.initialize + self.time_nanoseconds.factorize + self.time_nanoseconds.solve;
        self.time_human.read_matrix = format_nanoseconds(self.time_nanoseconds.read_matrix);
        self.time_human.initialize = format_nanoseconds(self.time_nanoseconds.initialize);
        self.time_human.factorize = format_nanoseconds(self.time_nanoseconds.factorize);
        self.time_human.solve = format_nanoseconds(self.time_nanoseconds.solve);
        self.time_human.total_ifs = format_nanoseconds(self.time_nanoseconds.total_ifs);
        self.time_human.verify = format_nanoseconds(self.time_nanoseconds.verify);
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
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::StatsLinSol;
    use serde_json;

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
    fn get_json_works() {
        let mut stats = StatsLinSol::new();
        const ONE_SECOND: u128 = 1000000000;
        stats.time_nanoseconds.read_matrix = ONE_SECOND;
        stats.time_nanoseconds.initialize = ONE_SECOND;
        stats.time_nanoseconds.factorize = ONE_SECOND * 2;
        stats.time_nanoseconds.solve = ONE_SECOND * 3;
        stats.time_nanoseconds.verify = ONE_SECOND * 4;
        let json = stats.get_json();
        assert!(stats.output.openmp_num_threads > 0);
        assert_eq!(stats.time_nanoseconds.total_ifs, ONE_SECOND * 6);
        assert_eq!(stats.time_human.read_matrix, "1s");
        assert_eq!(stats.time_human.initialize, "1s");
        assert_eq!(stats.time_human.factorize, "2s");
        assert_eq!(stats.time_human.solve, "3s");
        assert_eq!(stats.time_human.total_ifs, "6s");
        assert_eq!(stats.time_human.verify, "4s");
        assert!(json.len() > 0);
    }

    #[test]
    fn read_json_works() {
        let stats = StatsLinSol::read_json("data/mumps-pre2.json").unwrap();
        assert_eq!(stats.main.platform, "Russell");
        assert_eq!(stats.matrix.name, "pre2");
        assert_eq!(stats.matrix.symmetry, "None");
    }
}
