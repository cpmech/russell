//! Compares solver timings across CPU-labelled result directories.
//!
//! Usage: `cargo run --release --bin xcompare_matrix_market -- <path>`
//!
//! `<path>` may be:
//! - a base name without a CPU suffix, such as `.../arch`, in which case the
//!   sibling directories `arch-i7`, `arch-i9`, ... are discovered;
//! - one CPU directory, such as `.../arch-i7`; its siblings are discovered;
//! - a parent directory containing CPU-labelled result directories.
//!
//! The CPU label is the trailing `-i<number>` in a directory name. The lowest
//! numbered CPU is used as the baseline. Times are the median of
//! `time_nanoseconds.total_ifs_array`, falling back to `total_ifs`.

use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

/// Matrices at or above this many nonzeros are reported as a separate stratum
const BIG_NNZ: u64 = 1_000_000;

/// One timed solver/matrix result
#[derive(Debug, Clone, Copy)]
struct Run {
    time_ns: u64,
    nnz: u64,
    complex: bool,
}

/// Results loaded from one CPU-labelled directory
#[derive(Debug)]
struct Dataset {
    label: String,
    path: PathBuf,
    json_count: usize,
    runs: HashMap<(String, String), Run>,
}

/// Aggregated speedup statistics
#[derive(Debug, Clone, Copy)]
struct Stats {
    count: usize,
    geometric_mean: f64,
    median: f64,
    arithmetic_mean: f64,
}

/// One paired baseline/target result
#[derive(Debug, Clone)]
struct Pair {
    matrix: String,
    solver: String,
    nnz: u64,
    complex: bool,
    speedup: f64,
}

fn main() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        print_usage();
        return Ok(());
    }
    let input = args.get(1).ok_or_else(|| "missing results path; run with --help for usage".to_string())?;

    let paths = detect_dataset_paths(Path::new(input))?;
    let datasets: Vec<Dataset> = paths.iter().map(|(label, path)| load_dataset(label, path)).collect::<Result<_, _>>()?;

    print_report(&datasets);
    Ok(())
}

/// Prints the detected datasets and one comparison block per target CPU
fn print_report(datasets: &[Dataset]) {
    println!("Detected CPU datasets:");
    for dataset in datasets {
        println!(
            "  {:<4} {} ({} JSON files, {} timed cells)",
            dataset.label,
            dataset.path.display(),
            dataset.json_count,
            dataset.runs.len()
        );
    }

    let baseline = &datasets[0];
    println!();
    println!("Baseline: {}", baseline.label);
    for target in &datasets[1..] {
        let pairs = paired_runs(baseline, target);
        println!();
        println!("=== {} -> {} ===", baseline.label, target.label);
        if pairs.is_empty() {
            println!("no paired numeric times");
            continue;
        }
        println!(
            "Speedup = time_{} / time_{}; values above 1.00x mean {} is faster.",
            baseline.label, target.label, target.label
        );
        println!("Paired numeric cells: {}", pairs.len());
        println!();
        print_scope_table(&pairs);
        println!();
        print_solver_table(&pairs);
    }
}

/// Finds CPU-labelled result directories for the given path
fn detect_dataset_paths(input: &Path) -> Result<Vec<(String, PathBuf)>, String> {
    let exists = input.is_dir();
    let (parent, prefix) = if exists && looks_like_dataset(input) {
        let name = input
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| format!("cannot determine the name of {}", input.display()))?;
        let base = cpu_suffix(name).map(|(base, _)| base).unwrap_or(name);
        let parent = non_empty_parent(input);
        (parent, Some(base.to_string()))
    } else if exists {
        (input, None)
    } else {
        let raw_prefix = input
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| format!("cannot determine the name of {}", input.display()))?;
        let base = cpu_suffix(raw_prefix).map(|(base, _)| base).unwrap_or(raw_prefix);
        (non_empty_parent(input), Some(base.to_string()))
    };

    let mut candidates: Vec<(String, String, PathBuf)> = Vec::new();
    let entries = fs::read_dir(parent).map_err(|e| format!("cannot read {}: {e}", parent.display()))?;
    for entry in entries {
        let path = entry.map_err(|e| format!("cannot read {}: {e}", parent.display()))?.path();
        if !path.is_dir() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let Some((base, label)) = cpu_suffix(name) else {
            continue;
        };
        if let Some(prefix) = &prefix {
            if base != prefix {
                continue;
            }
        }
        if !looks_like_dataset(&path) {
            continue;
        }
        candidates.push((base.to_string(), label.to_string(), path));
    }

    let base = if let Some(prefix) = prefix {
        prefix
    } else {
        let bases: BTreeSet<String> = candidates.iter().map(|(base, _, _)| base.clone()).collect();
        match bases.len() {
            0 => {
                return Err(format!("no CPU-labelled result directories found in {}", parent.display()));
            }
            1 => bases.into_iter().next().unwrap(),
            _ => {
                let list: Vec<String> = bases.into_iter().collect();
                return Err(format!(
                    "multiple dataset families found in {}: {}; pass one base path",
                    parent.display(),
                    list.join(", ")
                ));
            }
        }
    };

    let mut datasets: Vec<(String, PathBuf)> = candidates
        .into_iter()
        .filter(|(candidate, _, _)| *candidate == base)
        .map(|(_, label, path)| (label, path))
        .collect();
    datasets.sort_by_key(|dataset| cpu_sort_key(&dataset.0));
    if datasets.len() < 2 {
        return Err(format!(
            "found {} CPU dataset(s) for `{}`; need at least two to compare",
            datasets.len(),
            base
        ));
    }
    Ok(datasets)
}

/// Returns the parent directory, using `.` for a bare relative name
fn non_empty_parent(path: &Path) -> &Path {
    path.parent().filter(|parent| !parent.as_os_str().is_empty()).unwrap_or(Path::new("."))
}

/// Splits `arch-i7` into `("arch", "i7")`; rejects non-CPU suffixes
fn cpu_suffix(name: &str) -> Option<(&str, &str)> {
    let pos = name.rfind("-i")?;
    let number = &name[pos + 2..];
    if number.is_empty() || !number.chars().all(|ch| ch.is_ascii_digit()) {
        return None;
    }
    Some((&name[..pos], &name[pos + 1..]))
}

/// Sort key for CPU labels, e.g. `i7` before `i9` before `i10`
fn cpu_sort_key(label: &str) -> (u32, String) {
    let number = label.strip_prefix('i').and_then(|s| s.parse().ok()).unwrap_or(u32::MAX);
    (number, label.to_string())
}

/// Returns true when a directory looks like a result set (`<solver>/*.json`)
fn looks_like_dataset(dir: &Path) -> bool {
    let Ok(entries) = fs::read_dir(dir) else {
        return false;
    };
    let mut subdirs = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() && is_json(&path) {
            return true;
        }
        if path.is_dir() {
            subdirs.push(path);
        }
    }
    subdirs.iter().any(|subdir| subdir_has_json(subdir))
}

/// Returns true when a directory directly contains JSON files
fn subdir_has_json(dir: &Path) -> bool {
    fs::read_dir(dir)
        .map(|entries| entries.flatten().any(|entry| entry.path().is_file() && is_json(&entry.path())))
        .unwrap_or(false)
}

/// Returns true for a `.json` file
fn is_json(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("json"))
}

/// Loads all solver results below one CPU directory
fn load_dataset(label: &str, path: &Path) -> Result<Dataset, String> {
    let mut files = Vec::new();
    collect_json_files(path, &mut files).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let json_count = files.len();

    let mut runs = HashMap::new();
    for file in files {
        let text = match fs::read_to_string(&file) {
            Ok(text) => text,
            Err(e) => {
                eprintln!("warning: skipping {}: {e}", file.display());
                continue;
            }
        };
        let value: Value = match serde_json::from_str(&text) {
            Ok(value) => value,
            Err(e) => {
                eprintln!("warning: skipping {}: {e}", file.display());
                continue;
            }
        };
        if let Some((key, run)) = parse_run(&value, path, &file) {
            runs.entry(key).or_insert(run);
        }
    }
    if runs.is_empty() {
        return Err(format!("no timed solver results found in {}", path.display()));
    }
    Ok(Dataset {
        label: label.to_string(),
        path: path.to_path_buf(),
        json_count,
        runs,
    })
}

/// Collects all JSON files below `dir`
fn collect_json_files(dir: &Path, files: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_json_files(&path, files)?;
        } else if is_json(&path) {
            files.push(path);
        }
    }
    Ok(())
}

/// Extracts one timed record; the solver key prefers the solver subdirectory
fn parse_run(value: &Value, root: &Path, file: &Path) -> Option<((String, String), Run)> {
    if value.pointer("/main/solved").and_then(Value::as_bool) == Some(false) {
        return None;
    }
    let matrix = value.pointer("/matrix/name")?.as_str()?.to_string();
    let solver_from_json = value.pointer("/main/solver")?.as_str()?.to_lowercase();
    let solver = if file.parent() != Some(root) {
        file.parent()
            .and_then(|parent| parent.file_name())
            .and_then(|name| name.to_str())
            .map(|name| name.to_lowercase())
            .unwrap_or(solver_from_json)
    } else {
        solver_from_json
    };
    let time_ns = median_time(value)?;
    if time_ns == 0 {
        return None;
    }
    let nnz = value
        .pointer("/matrix/nnz_actual")
        .and_then(Value::as_u64)
        .or_else(|| value.pointer("/matrix/nnz").and_then(Value::as_u64))
        .unwrap_or(0);
    let complex = value.pointer("/matrix/complex").and_then(Value::as_bool).unwrap_or(false);
    Some(((matrix, solver), Run { time_ns, nnz, complex }))
}

/// Returns the median of the per-run total times, falling back to the stored mean
fn median_time(value: &Value) -> Option<u64> {
    let mut values: Vec<u64> = value
        .pointer("/time_nanoseconds/total_ifs_array")
        .and_then(Value::as_array)
        .map(|array| array.iter().filter_map(Value::as_u64).filter(|time| *time > 0).collect())
        .unwrap_or_default();
    if values.is_empty() {
        return value
            .pointer("/time_nanoseconds/total_ifs")
            .and_then(Value::as_u64)
            .filter(|time| *time > 0);
    }
    values.sort_unstable();
    let count = values.len();
    Some(if count % 2 == 0 {
        ((values[count / 2 - 1] as u128 + values[count / 2] as u128) / 2) as u64
    } else {
        values[count / 2]
    })
}

/// Pairs the numeric times of the baseline and target datasets
fn paired_runs(baseline: &Dataset, target: &Dataset) -> Vec<Pair> {
    let mut pairs = Vec::new();
    for (key, base) in &baseline.runs {
        let Some(target) = target.runs.get(key) else {
            continue;
        };
        pairs.push(Pair {
            matrix: key.0.clone(),
            solver: key.1.clone(),
            nnz: base.nnz.max(target.nnz),
            complex: base.complex || target.complex,
            speedup: base.time_ns as f64 / target.time_ns as f64,
        });
    }
    pairs.sort_by(|aa, bb| aa.matrix.cmp(&bb.matrix).then_with(|| aa.solver.cmp(&bb.solver)));
    pairs
}

/// Computes the aggregate statistics over a set of speedup ratios
fn stats(values: &[f64]) -> Option<Stats> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|aa, bb| aa.partial_cmp(bb).unwrap());
    let count = sorted.len();
    let geometric_mean = (sorted.iter().map(|value| value.ln()).sum::<f64>() / count as f64).exp();
    let median = if count % 2 == 0 {
        (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0
    } else {
        sorted[count / 2]
    };
    let arithmetic_mean = sorted.iter().sum::<f64>() / count as f64;
    Some(Stats {
        count,
        geometric_mean,
        median,
        arithmetic_mean,
    })
}

/// Prints the by-scope comparison table
fn print_scope_table(pairs: &[Pair]) {
    let scopes = [
        ("Overall", pairs.iter().map(|pair| pair.speedup).collect::<Vec<_>>()),
        ("real", pairs.iter().filter(|pair| !pair.complex).map(|pair| pair.speedup).collect()),
        ("complex", pairs.iter().filter(|pair| pair.complex).map(|pair| pair.speedup).collect()),
        (
            "nnz >= 1M",
            pairs.iter().filter(|pair| pair.nnz >= BIG_NNZ).map(|pair| pair.speedup).collect(),
        ),
        (
            "nnz < 1M",
            pairs.iter().filter(|pair| pair.nnz < BIG_NNZ).map(|pair| pair.speedup).collect(),
        ),
    ];
    let rows: Vec<(String, Stats)> = scopes
        .into_iter()
        .filter_map(|(label, values)| stats(&values).map(|stats| (label.to_string(), stats)))
        .collect();
    print_stats_table("By scope", "scope", &rows);
}

/// Prints the by-solver comparison table
fn print_solver_table(pairs: &[Pair]) {
    let mut values: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for pair in pairs {
        values.entry(pair.solver.clone()).or_default().push(pair.speedup);
    }
    let mut rows: Vec<(String, Stats)> = values
        .into_iter()
        .filter_map(|(solver, values)| stats(&values).map(|stats| (solver, stats)))
        .collect();
    rows.sort_by(|aa, bb| {
        bb.1.geometric_mean
            .partial_cmp(&aa.1.geometric_mean)
            .unwrap()
            .then_with(|| aa.0.cmp(&bb.0))
    });
    print_stats_table("By solver", "solver", &rows);
}

/// Prints one statistics table
fn print_stats_table(title: &str, first_header: &str, rows: &[(String, Stats)]) {
    if rows.is_empty() {
        return;
    }
    let headers = [
        first_header.to_string(),
        "cells".to_string(),
        "geo mean".to_string(),
        "median".to_string(),
        "arithmetic".to_string(),
    ];
    let cells: Vec<Vec<String>> = rows
        .iter()
        .map(|(label, stats)| {
            vec![
                label.clone(),
                stats.count.to_string(),
                fmt_ratio(stats.geometric_mean),
                fmt_ratio(stats.median),
                fmt_ratio(stats.arithmetic_mean),
            ]
        })
        .collect();

    let mut widths: Vec<usize> = headers.iter().map(|header| header.chars().count()).collect();
    for line in &cells {
        for (i, cell) in line.iter().enumerate() {
            widths[i] = widths[i].max(cell.chars().count());
        }
    }
    let right_aligned = [false, true, true, true, true];

    println!("{title}");
    println!("{}", format_row(&headers, &widths, &right_aligned));
    let separator: Vec<String> = widths.iter().map(|width| "-".repeat(*width)).collect();
    println!("{}", separator.join("  "));
    for line in &cells {
        println!("{}", format_row(line, &widths, &right_aligned));
    }
}

/// Formats a speedup ratio
fn fmt_ratio(value: f64) -> String {
    format!("{value:.2}x")
}

/// Formats one table row with the given column widths and alignment
fn format_row(cells: &[String], widths: &[usize], right_aligned: &[bool]) -> String {
    cells
        .iter()
        .enumerate()
        .map(|(i, cell)| {
            if right_aligned[i] {
                format!("{:>width$}", cell, width = widths[i])
            } else {
                format!("{:<width$}", cell, width = widths[i])
            }
        })
        .collect::<Vec<_>>()
        .join("  ")
}

/// Prints the command-line usage
fn print_usage() {
    println!("Usage: xcompare_matrix_market <results_path>");
    println!();
    println!("Compares solver times across CPU-labelled result directories.");
    println!("The path may be a base name (e.g. .../arch), one CPU directory");
    println!("(e.g. .../arch-i7), or a parent containing CPU directories.");
    println!("CPU labels are detected from trailing `-i<number>` suffixes.");
    println!("The lowest-numbered CPU is the baseline; speedup = baseline / target.");
    println!("Times are medians of `time_nanoseconds.total_ifs_array`, falling back to `total_ifs`.");
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn cpu_suffix_works() {
        assert_eq!(cpu_suffix("arch-i7"), Some(("arch", "i7")));
        assert_eq!(cpu_suffix("arch-i9"), Some(("arch", "i9")));
        assert_eq!(cpu_suffix("arch-i10"), Some(("arch", "i10")));
        assert_eq!(cpu_suffix("arch-2026-11-19"), None);
        assert_eq!(cpu_suffix("arch"), None);
        assert_eq!(cpu_suffix("arch-i"), None);
    }

    #[test]
    fn cpu_sort_key_orders_numerically() {
        assert!(cpu_sort_key("i7") < cpu_sort_key("i9"));
        assert!(cpu_sort_key("i9") < cpu_sort_key("i10"));
    }

    #[test]
    fn stats_works() {
        let stats = stats(&[1.0, 2.0, 4.0]).unwrap();
        assert_eq!(stats.count, 3);
        assert_eq!(stats.median, 2.0);
        assert_eq!(stats.arithmetic_mean, 7.0 / 3.0);
        assert!((stats.geometric_mean - 2.0).abs() < 1e-12);
    }

    #[test]
    fn median_time_works() {
        let value = json!({
            "time_nanoseconds": {
                "total_ifs": 999,
                "total_ifs_array": [5, 1, 3, 9, 7],
            }
        });
        assert_eq!(median_time(&value), Some(5));
        let value = json!({
            "time_nanoseconds": {
                "total_ifs": 999,
                "total_ifs_array": [4, 1, 3, 2],
            }
        });
        assert_eq!(median_time(&value), Some(2));
        let value = json!({"time_nanoseconds": {"total_ifs": 1234}});
        assert_eq!(median_time(&value), Some(1234));
    }

    #[test]
    fn parse_run_reads_total_time() {
        let value = json!({
            "main": {"solver": "Faer", "solved": true},
            "matrix": {"name": "mat1", "nnz": 10, "nnz_actual": 12, "complex": false},
            "time_nanoseconds": {"total_ifs": 1234},
        });
        let root = Path::new("/tmp/results/arch-i9");
        let file = root.join("faer").join("mat1.json");
        let ((matrix, solver), run) = parse_run(&value, root, &file).unwrap();
        assert_eq!(matrix, "mat1");
        assert_eq!(solver, "faer");
        assert_eq!(run.time_ns, 1234);
        assert_eq!(run.nnz, 12);
        assert!(!run.complex);
    }

    #[test]
    fn parse_run_skips_failed_runs() {
        let value = json!({
            "main": {"solver": "Faer", "solved": false},
            "matrix": {"name": "mat1", "nnz": 10},
            "time_nanoseconds": {"total_ifs": 1234},
        });
        let root = Path::new("/tmp/results/arch-i9");
        let file = root.join("faer").join("mat1.json");
        assert!(parse_run(&value, root, &file).is_none());
    }
}
