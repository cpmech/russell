//! Exports system and solver-library information.
//!
//! Usage: `cargo run --release --bin sys_info [output_file]`
//!
//! When `output_file` is omitted, the report is printed to stdout. The benchmark
//! runner uses this binary to record the machine and library versions alongside
//! the results.

use std::fs;
use std::path::PathBuf;
use structopt::StructOpt;

/// Command line options
#[derive(StructOpt, Debug)]
#[structopt(name = "sys_info", about = "Export system and solver-library information.")]
struct Options {
    /// Output file; prints to stdout when omitted
    #[structopt(value_name = "FILE")]
    output: Option<PathBuf>,
}

fn main() -> Result<(), String> {
    let opt = Options::from_args();
    let report = build_report();
    match opt.output {
        Some(path) => {
            fs::write(&path, report).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
            eprintln!("system information written to {}", path.display());
        }
        None => print!("{report}"),
    }
    Ok(())
}

/// Builds the complete report from the OS, GPU, CPU, memory, and library data
fn build_report() -> String {
    format!(
        "Information about the system:\n{}\n\nInformation about the libraries:\n{}\n",
        russell_sparse::get_system_info(),
        russell_sparse::get_library_versions()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_contains_system_and_library_sections() {
        let report = build_report();
        assert!(report.contains("Information about the system:"));
        assert!(report.contains("Information about the libraries:"));
        assert!(report.contains("SuiteSparse:"));
    }
}
