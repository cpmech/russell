//! Exports operating-system and hardware information.
//!
//! Usage: `cargo run --release --bin os_info [--cpu-plan] [output_file]`
//!
//! With `--cpu-plan`, the CPU topology and the benchmark thread plan are printed
//! as shell assignments instead of the report. When `output_file` is omitted, the
//! output is printed to stdout.

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() -> Result<(), String> {
    let mut cpu_plan = false;
    let mut output: Option<PathBuf> = None;
    for arg in env::args_os().skip(1) {
        match arg.to_str() {
            Some("--cpu-plan") => cpu_plan = true,
            Some("-h") | Some("--help") => {
                print_usage();
                return Ok(());
            }
            Some(text) if text.starts_with('-') => return Err(format!("unknown option: {text}")),
            _ => {
                if output.is_some() {
                    return Err("only one output file is accepted".to_string());
                }
                output = Some(PathBuf::from(arg));
            }
        }
    }

    let report = build_report(cpu_plan);
    match output {
        Some(path) => {
            fs::write(&path, report).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
            eprintln!("OS information written to {}", path.display());
        }
        None => print!("{report}"),
    }
    Ok(())
}

/// Builds either the full OS report or the shell-friendly CPU plan
fn build_report(cpu_plan: bool) -> String {
    if cpu_plan {
        russell_lab::get_cpu_plan_with_overrides().to_shell()
    } else {
        format!("{}\n", russell_lab::get_os_info())
    }
}

/// Prints the command-line usage
fn print_usage() {
    println!("Usage: os_info [--cpu-plan] [FILE]");
    println!();
    println!("  --cpu-plan  print the CPU topology and thread plan as shell assignments");
    println!("  FILE        write the output to FILE instead of stdout");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_os = "linux")]
    fn report_contains_the_os_sections() {
        let report = build_report(false);
        assert!(report.contains("--- OS ---"));
        assert!(report.contains("--- CPU topology ---"));
        assert!(report.ends_with('\n'));
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
