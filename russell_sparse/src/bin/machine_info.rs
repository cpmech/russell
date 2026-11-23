//! Exports OS, hardware, and solver-library information.
//!
//! Usage: `cargo run --release --bin machine_info [--cpu-plan] [output_file]`
//!
//! The OS, GPU, CPU, topology, and memory sections come from `russell_lab`.
//! The solver-library versions come from this crate, so they follow the features
//! the report is built with. With `--cpu-plan`, the CPU topology and the benchmark
//! thread plan are printed as shell assignments instead of the report.

use std::fs;
use std::path::PathBuf;
use structopt::StructOpt;

/// Command line options
#[derive(StructOpt, Debug)]
#[structopt(name = "machine_info", about = "Export OS, hardware, and solver-library information.")]
struct Options {
    /// Print the CPU plan as shell assignments instead of the report
    #[structopt(long)]
    cpu_plan: bool,

    /// Output file; prints to stdout when omitted
    #[structopt(value_name = "FILE")]
    output: Option<PathBuf>,
}

fn main() -> Result<(), String> {
    let opt = Options::from_args();
    let report = build_report(opt.cpu_plan);
    match opt.output {
        Some(path) => {
            fs::write(&path, report).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
            eprintln!("machine information written to {}", path.display());
        }
        None => print!("{report}"),
    }
    Ok(())
}

/// Builds the complete report or the shell-friendly CPU plan
fn build_report(cpu_plan: bool) -> String {
    if cpu_plan {
        return russell_lab::get_cpu_plan_with_overrides().to_shell();
    }
    format!(
        "Information about the OS:\n{}\n\nInformation about the libraries:\n{}\n",
        russell_lab::get_os_info(),
        russell_sparse::get_library_versions()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_contains_os_and_library_sections() {
        let report = build_report(false);
        assert!(report.contains("Information about the OS:"));
        assert!(report.contains("Information about the libraries:"));
        assert!(report.contains("SuiteSparse:"));
    }

    #[test]
    fn cpu_plan_contains_the_benchmark_keys() {
        let plan = build_report(true);
        assert!(plan.contains("P_CORES="));
        assert!(plan.contains("E_CORES="));
        assert!(plan.contains("MUMPS_THREADS="));
        assert!(plan.contains("BLAS_THREADS="));
        assert!(plan.contains("CPU_AFFINITY="));
    }
}
