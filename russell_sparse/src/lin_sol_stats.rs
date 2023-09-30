use serde::{Deserialize, Serialize};
use serde_json;

/// Holds information about the solution of a linear system
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct LinSolStats {
    pub platform: String,
    pub version: String,
    pub date: String,
    pub blas_lib: String,
    pub solver_name: String,
    pub solver_version: String,
    pub matrix_name: String,
    pub symmetry: String,
    pub nrow: usize,
    pub ncol: usize,
    pub nnz: usize,
    pub time_read_matrix_market_nanosecond: u128,
    pub time_read_matrix_market_human: String,
    pub time_factorize_nanosecond: u128,
    pub time_factorize_human: String,
    pub time_solve_nanosecond: u128,
    pub time_solve_human: String,
    pub time_total_nanosecond: u128, // initialize + factorize + solve (not including read matrix)
    pub time_total_human: String,
    pub requested_ordering: String,
    pub requested_scaling: String,
    pub requested_mumps_openmp_num_threads: usize,
    pub effective_ordering: String,
    pub effective_scaling: String,
    pub effective_strategy: String,
    pub openmp_num_threads: usize,
    pub verify_max_abs_a: f64,
    pub verify_max_abs_a_times_x: f64,
    pub verify_relative_error: f64,
    pub verify_time_nanosecond: u128,
    pub verify_time_human: String,
    pub compute_determinant: bool,
    pub determinant_mantissa: f64, // det = mantissa * pow(base, exponent)
    pub determinant_base: f64,
    pub determinant_exponent: f64,
    pub compute_error_estimates: bool,
    pub error_estimate_omega1: f64,
    pub error_estimate_omega2: f64,
    pub compute_condition_number_estimate: bool,
    pub reciprocal_condition_number_estimate: f64,
}

impl LinSolStats {
    pub fn get_json(&self) -> String {
        serde_json::to_string_pretty(&self).unwrap()
    }
}
