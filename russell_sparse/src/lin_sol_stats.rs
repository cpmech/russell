use super::{LinSolStatsMUMPS, LinSolver};
use serde::{Deserialize, Serialize};
use serde_json;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct LinSolStatsMain {
    pub platform: String,
    pub blas_lib: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct LinSolStatsMatrix {
    pub name: String,
    pub nrow: usize,
    pub ncol: usize,
    pub nnz: usize,
    pub symmetry: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct LinSolStatsSolver {
    pub name: String,
    pub version: String,
    pub documentation: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct LinSolStatsRequests {
    pub ordering: String,
    pub scaling: String,
    pub mumps_openmp_num_threads: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct LinSolStatsOutput {
    pub effective_ordering: String,
    pub effective_scaling: String,
    pub openmp_num_threads: usize,
    pub umfpack_strategy: String,
    pub umfpack_rcond: f64, // reciprocal condition number
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct LinSolStatsDeterminant {
    // det = mantissa * pow(base, exponent)
    pub computed: bool,
    pub mantissa: f64,
    pub base: f64,
    pub exponent: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct LinSolStatsError {
    pub computed: bool,
    pub max_abs_a: f64,
    pub max_abs_a_times_x: f64,
    pub relative_error: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct LinSolStatsTimeHuman {
    pub read_matrix_market: String,
    pub factorize: String,
    pub solve: String,
    pub total: String,
    pub verify: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct LinSolStatsTimeNanoseconds {
    pub read_matrix_market: u128,
    pub factorize: u128,
    pub solve: u128,
    pub total: u128,
    pub verify: u128,
}

/// Holds information about the solution of a linear system
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct LinSolStats {
    pub main: LinSolStatsMain,
    pub matrix: LinSolStatsMatrix,
    pub solver: LinSolStatsSolver,
    pub requests: LinSolStatsRequests,
    pub output: LinSolStatsOutput,
    pub determinant: LinSolStatsDeterminant,
    pub error: LinSolStatsError,
    pub time_human: LinSolStatsTimeHuman,
    pub time_nanoseconds: LinSolStatsTimeNanoseconds,
    // pub mumps_stats: LinSolStatsMUMPS,
}

impl LinSolStatsMain {
    pub fn new() -> Self {
        LinSolStatsMain {
            platform: "Russell".to_string(),
            blas_lib: "OpenBLAS".to_string(),
        }
    }
}

impl LinSolStatsMatrix {
    pub fn new() -> Self {
        LinSolStatsMatrix {
            name: String::new(),
            nrow: 0,
            ncol: 0,
            nnz: 0,
            symmetry: String::new(),
        }
    }
}

impl LinSolStatsSolver {
    pub fn new() -> Self {
        LinSolStatsSolver {
            name: String::new(),
            version: String::new(),
            documentation: String::new(),
        }
    }
}

impl LinSolStatsRequests {
    pub fn new() -> Self {
        LinSolStatsRequests {
            ordering: String::new(),
            scaling: String::new(),
            mumps_openmp_num_threads: 0,
        }
    }
}

impl LinSolStatsOutput {
    pub fn new() -> Self {
        LinSolStatsOutput {
            effective_ordering: String::new(),
            effective_scaling: String::new(),
            openmp_num_threads: 0,
            umfpack_strategy: String::new(),
            umfpack_rcond: 0.0,
        }
    }
}

impl LinSolStatsDeterminant {
    pub fn new() -> Self {
        LinSolStatsDeterminant {
            computed: false,
            mantissa: 0.0,
            base: 0.0,
            exponent: 0.0,
        }
    }
}

impl LinSolStatsError {
    pub fn new() -> Self {
        LinSolStatsError {
            computed: false,
            max_abs_a: 0.0,
            max_abs_a_times_x: 0.0,
            relative_error: 0.0,
        }
    }
}

impl LinSolStatsTimeHuman {
    pub fn new() -> Self {
        LinSolStatsTimeHuman {
            read_matrix_market: String::new(),
            factorize: String::new(),
            solve: String::new(),
            total: String::new(),
            verify: String::new(),
        }
    }
}

impl LinSolStatsTimeNanoseconds {
    pub fn new() -> Self {
        LinSolStatsTimeNanoseconds {
            read_matrix_market: 0,
            factorize: 0,
            solve: 0,
            total: 0,
            verify: 0,
        }
    }
}

impl LinSolStats {
    pub fn new(solver: &LinSolver) -> Self {
        LinSolStats {
            main: LinSolStatsMain::new(),
            matrix: LinSolStatsMatrix::new(),
            solver: LinSolStatsSolver::new(),
            requests: LinSolStatsRequests::new(),
            output: LinSolStatsOutput::new(),
            determinant: LinSolStatsDeterminant::new(),
            error: LinSolStatsError::new(),
            time_human: LinSolStatsTimeHuman::new(),
            time_nanoseconds: LinSolStatsTimeNanoseconds::new(),
            // mumps_stats: LinSolStatsMUMPS::new(),
        }
    }

    pub fn get_json(&self) -> String {
        serde_json::to_string_pretty(&self).unwrap()
    }
}
