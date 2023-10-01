use super::{LinSolParams, LinSolver, SparseMatrix, StatsLinSolMUMPS};
use serde::{Deserialize, Serialize};
use serde_json;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StatsLinSolMain {
    pub platform: String,
    pub blas_lib: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StatsLinSolMatrix {
    pub name: String,
    pub nrow: usize,
    pub ncol: usize,
    pub nnz: usize,
    pub symmetry: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StatsLinSolSolver {
    pub name: String,
    pub version: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StatsLinSolRequests {
    pub ordering: String,
    pub scaling: String,
    pub mumps_openmp_num_threads: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StatsLinSolOutput {
    pub effective_ordering: String,
    pub effective_scaling: String,
    pub openmp_num_threads: usize,
    pub umfpack_strategy: String,
    pub umfpack_rcond: f64, // reciprocal condition number
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StatsLinSolDeterminant {
    // det = mantissa * pow(base, exponent)
    pub computed: bool,
    pub mantissa: f64,
    pub base: f64,
    pub exponent: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StatsLinSolError {
    pub computed: bool,
    pub max_abs_a: f64,
    pub max_abs_a_times_x: f64,
    pub relative_error: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StatsLinSolTimeHuman {
    pub read_matrix_market: String,
    pub factorize: String,
    pub solve: String,
    pub total: String,
    pub verify: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StatsLinSolTimeNanoseconds {
    pub read_matrix_market: u128,
    pub factorize: u128,
    pub solve: u128,
    pub total: u128,
    pub verify: u128,
}

/// Holds information about the solution of a linear system
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StatsLinSol {
    pub main: StatsLinSolMain,
    pub matrix: StatsLinSolMatrix,
    pub solver: StatsLinSolSolver,
    pub requests: StatsLinSolRequests,
    pub output: StatsLinSolOutput,
    pub determinant: StatsLinSolDeterminant,
    pub error: StatsLinSolError,
    pub time_human: StatsLinSolTimeHuman,
    pub time_nanoseconds: StatsLinSolTimeNanoseconds,
    pub mumps_stats: StatsLinSolMUMPS,
}

impl StatsLinSol {
    pub fn get_json(&self) -> String {
        serde_json::to_string_pretty(&self).unwrap()
    }

    pub fn new(matrix_name: String, matrix: &SparseMatrix, solver: &LinSolver, params: Option<LinSolParams>) -> Self {
        let (nrow, ncol, nnz, symmetry) = matrix.get_info();
        let par = if let Some(p) = params { p } else { LinSolParams::new() };
        let sol = &solver.actual;
        let umfpack_rcond = 0.0;
        let (mantissa, base, exponent) = sol.get_determinant();
        StatsLinSol {
            main: StatsLinSolMain {
                platform: "Russell".to_string(),
                blas_lib: "OpenBLAS".to_string(),
            },
            matrix: StatsLinSolMatrix {
                name: matrix_name,
                nrow,
                ncol,
                nnz,
                symmetry: format!("{:?}", symmetry),
            },
            solver: StatsLinSolSolver {
                name: sol.get_name(),
                version: String::new(), //solver.actual.get_version(),
            },
            requests: StatsLinSolRequests {
                ordering: format!("{:?}", par.ordering),
                scaling: format!("{:?}", par.scaling),
                mumps_openmp_num_threads: par.mumps_openmp_num_threads,
            },
            output: StatsLinSolOutput {
                effective_ordering: sol.get_effective_ordering(),
                effective_scaling: sol.get_effective_scaling(),
                openmp_num_threads: 0,
                umfpack_strategy: sol.get_effective_strategy(),
                umfpack_rcond,
            },
            determinant: StatsLinSolDeterminant {
                computed: par.compute_determinant,
                mantissa,
                base,
                exponent,
            },
            error: StatsLinSolError {
                computed: false,
                max_abs_a: 0.0,
                max_abs_a_times_x: 0.0,
                relative_error: 0.0,
            },
            time_human: StatsLinSolTimeHuman {
                read_matrix_market: String::new(),
                factorize: String::new(),
                solve: String::new(),
                total: String::new(),
                verify: String::new(),
            },
            time_nanoseconds: StatsLinSolTimeNanoseconds {
                read_matrix_market: 0,
                factorize: 0,
                solve: 0,
                total: 0,
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
}
