use serde::{Deserialize, Serialize};

/// Holds information about the solution of a linear system
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SolutionInfo {
    pub platform: String,
    pub blas_lib: String,
    pub matrix_name: String,
    pub symmetry: String,
    pub layout: String,
    pub nrow: usize,
    pub ncol: usize,
    pub nnz: usize,
    pub time_read_matrix_market_nanosecond: u128,
    pub time_read_matrix_market_human: String,
    pub time_factorize_nanosecond: u128,
    pub time_factorize_human: String,
    pub time_solve_nanosecond: u128,
    pub time_solve_human: String,
    pub requested_ordering: String,
    pub requested_scaling: String,
    pub requested_openmp_num_threads: usize,
    pub effective_ordering: String,
    pub effective_scaling: String,
    pub effective_openmp_num_threads: usize,
    pub verify_max_abs_a: f64,
    pub verify_max_abs_a_times_x: f64,
    pub verify_relative_error: f64,
    pub verify_time_nanosecond: u128,
    pub verify_time_human: String,
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::SolutionInfo;
    use serde_json;

    #[test]
    fn derive_works() {
        let info = SolutionInfo {
            platform: "Russell".to_string(),
            blas_lib: "OpenBLAS".to_string(),
            matrix_name: "Unknown".to_string(),
            symmetry: "General".to_string(),
            layout: "Full".to_string(),
            nrow: 3,
            ncol: 3,
            nnz: 6,
            time_read_matrix_market_nanosecond: 176349,
            time_read_matrix_market_human: "176.349µs".to_string(),
            time_factorize_nanosecond: 176349,
            time_factorize_human: "176.349µs".to_string(),
            time_solve_nanosecond: 176349,
            time_solve_human: "176.349µs".to_string(),
            requested_ordering: "Amd".to_string(),
            requested_scaling: "Auto".to_string(),
            requested_openmp_num_threads: 0,
            effective_ordering: "Amd".to_string(),
            effective_scaling: "Sum".to_string(),
            effective_openmp_num_threads: 1,
            verify_max_abs_a: 0.0001,
            verify_max_abs_a_times_x: 1.0000000000000004,
            verify_relative_error: 4.440892098500626e-16,
            verify_time_nanosecond: 1322,
            verify_time_human: "1.322µs".to_string(),
        };
        let info_clone = info.clone();
        let correct = "SolutionInfo { platform: \"Russell\", blas_lib: \"OpenBLAS\", matrix_name: \"Unknown\", symmetry: \"General\", layout: \"Full\", nrow: 3, ncol: 3, nnz: 6, time_read_matrix_market_nanosecond: 176349, time_read_matrix_market_human: \"176.349µs\", time_factorize_nanosecond: 176349, time_factorize_human: \"176.349µs\", time_solve_nanosecond: 176349, time_solve_human: \"176.349µs\", requested_ordering: \"Amd\", requested_scaling: \"Auto\", requested_openmp_num_threads: 0, effective_ordering: \"Amd\", effective_scaling: \"Sum\", effective_openmp_num_threads: 1, verify_max_abs_a: 0.0001, verify_max_abs_a_times_x: 1.0000000000000004, verify_relative_error: 4.440892098500626e-16, verify_time_nanosecond: 1322, verify_time_human: \"1.322µs\" }";
        assert_eq!(format!("{:?}", info), correct);
        assert_eq!(info_clone.nrow, info.nrow);
        assert_eq!(info_clone.nnz, info.nnz);
        // serialize
        let info_json = serde_json::to_string(&info).unwrap();
        // deserialize
        let info_read: SolutionInfo = serde_json::from_str(&info_json).unwrap();
        assert_eq!(format!("{:?}", info_read), correct);
    }
}
