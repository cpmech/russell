//! Extracts solver timing results from the JSON files written by `solve_matrix_market`.
//!
//! Usage: `cargo run --release --bin xtract_matrix_market [results_dir]`
//!
//! The default directory is the one used by `run_solve_matrix_market.bash`:
//! `$OUTPUT_DIR` when set, otherwise `/tmp/russell_sparse/results/$SYSTEM`.

use russell_lab::format_nanoseconds_with_digits as fmt_nano;
use serde_json::Value;
use std::collections::{BTreeSet, HashMap};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

/// Preferred solver column order; columns are discovered from the result files.
const SOLVER_ORDER: [&str; 5] = ["mumps", "umfpack", "faer", "qdldl", "basiclu"];

/// Descriptions of the failure keys used in the solver columns.
const KEY_DESCRIPTIONS: &[(&str, &str)] = &[
    ("stop-time", "stopped by the time guard"),
    ("no-mem", "ran out of memory"),
    ("only-sym", "solver requires a symmetric matrix"),
    ("only-spd", "solver requires a symmetric positive-definite matrix"),
    ("not-pd", "input was assumed positive-definite, but it is not"),
    ("no-conv", "iterative solver did not converge"),
    ("breakdown", "iterative method broke down"),
    ("singular", "matrix or pivot is singular"),
    ("bad-input", "invalid matrix or input (shape, indices, NaN/inf, empty)"),
    ("not-avail", "solver is not available in this build"),
    ("too-big", "matrix is too large for the solver"),
    ("error", "other error; read the logs"),
];

/// Holds one row of the comparison table.
#[derive(Debug, Default)]
struct MatrixData {
    name: String,
    nnz: u64,
    complex: bool,
    symmetric: bool,
    positive_definite: bool,
    times: HashMap<String, u64>,
    failures: HashMap<String, &'static str>,
}

fn main() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        print_usage();
        return Ok(());
    }

    let dir = args.get(1).cloned().unwrap_or_else(default_results_dir);
    let path = Path::new(&dir);
    if !path.is_dir() {
        return Err(format!("results directory not found: {}", path.display()));
    }

    let mut files = Vec::new();
    collect_json_files(path, &mut files).map_err(|e| format!("cannot read {}: {}", path.display(), e))?;
    files.sort();
    if files.is_empty() {
        return Err(format!("no JSON files found in {}", path.display()));
    }

    let mut matrices: HashMap<String, MatrixData> = HashMap::new();
    for file in &files {
        let text = match fs::read_to_string(file) {
            Ok(text) => text,
            Err(e) => {
                eprintln!("warning: skipping {}: {}", file.display(), e);
                continue;
            }
        };
        let value: Value = match serde_json::from_str(&text) {
            Ok(value) => value,
            Err(e) => {
                eprintln!("warning: skipping {}: {}", file.display(), e);
                continue;
            }
        };
        if let Some(data) = parse_record(&value) {
            merge_record(&mut matrices, data);
        }
    }

    if matrices.is_empty() {
        return Err("no valid solver results found".to_string());
    }

    let mut rows: Vec<MatrixData> = matrices.into_values().collect();
    rows.sort_by(|aa, bb| aa.nnz.cmp(&bb.nnz).then_with(|| aa.name.cmp(&bb.name)));

    let (complex_rows, real_rows): (Vec<MatrixData>, Vec<MatrixData>) = rows.into_iter().partition(|row| row.complex);
    let real_solvers = solver_columns(&real_rows);
    let complex_solvers = solver_columns(&complex_rows);
    let used_keys = used_failure_keys(&real_rows)
        .into_iter()
        .chain(used_failure_keys(&complex_rows))
        .collect();

    println!("All solver columns show total time (initialize + preprocess + solve).");
    println!();
    print_table("Real-valued matrices", &real_rows, &real_solvers);
    println!();
    print_table("Complex-valued matrices", &complex_rows, &complex_solvers);
    print_key_descriptions(&used_keys);
    Ok(())
}

/// Returns the results directory used by the bash script
fn default_results_dir() -> String {
    if let Ok(dir) = env::var("OUTPUT_DIR") {
        if !dir.is_empty() {
            return dir;
        }
    }
    let system = env::var("SYSTEM").unwrap_or_else(|_| "arch".to_string());
    format!("/tmp/russell_sparse/results/{}", system)
}

/// Collects all JSON files below `dir`
fn collect_json_files(dir: &Path, files: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_json_files(&path, files)?;
        } else if path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("json"))
        {
            files.push(path);
        }
    }
    Ok(())
}

/// Extracts one matrix/solver record from a JSON value
fn parse_record(value: &Value) -> Option<MatrixData> {
    let solver = value.pointer("/main/solver")?.as_str()?.to_lowercase();
    if solver.is_empty() {
        return None;
    }
    let name = value.pointer("/matrix/name")?.as_str()?.to_string();
    // Prefer the actual full-matrix count; `nnz` depends on the solver's
    // symmetric handling (a triangular storage counts only one triangle).
    let nnz = value
        .pointer("/matrix/nnz_actual")
        .and_then(Value::as_u64)
        .or_else(|| value.pointer("/matrix/nnz").and_then(Value::as_u64))
        .unwrap_or(0);
    let complex = value.pointer("/matrix/complex").and_then(Value::as_bool).unwrap_or(false);
    let symmetric = value
        .pointer("/matrix/symmetric")
        .and_then(Value::as_str)
        .map(|sym| !sym.eq_ignore_ascii_case("no"))
        .unwrap_or(false);
    let positive_definite = value.pointer("/requests/positive_definite").and_then(Value::as_bool).unwrap_or(false);
    let solved = value
        .pointer("/main/solved")
        .and_then(Value::as_bool)
        .unwrap_or_else(|| value.pointer("/main/error_from_solver").and_then(Value::as_str).is_none());

    let mut data = MatrixData {
        name,
        nnz,
        complex,
        symmetric,
        positive_definite,
        ..Default::default()
    };
    if solved {
        let total_ifs = value.pointer("/time_nanoseconds/total_ifs").and_then(Value::as_u64).unwrap_or(0);
        if total_ifs == 0 {
            return None;
        }
        data.times.insert(solver, total_ifs);
    } else {
        data.failures.insert(solver, failure_key(value)?);
    }
    Some(data)
}

/// Classifies a failed run into one of the keys shown in the table
///
/// Returns `None` for records without timing and without failure information,
/// such as the JSON produced by an information-only run.
fn failure_key(value: &Value) -> Option<&'static str> {
    if value.pointer("/main/stopped_by_time_guard").and_then(Value::as_bool).unwrap_or(false) {
        return Some("stop-time");
    }
    if value.pointer("/main/out_of_memory").and_then(Value::as_bool).unwrap_or(false) {
        return Some("no-mem");
    }
    let raw = value.pointer("/main/error_from_solver").and_then(Value::as_str).unwrap_or("");
    if raw.is_empty() {
        return None;
    }
    if russell_sparse::is_memory_error(raw) {
        return Some("no-mem");
    }
    Some(classify_message(&raw.to_lowercase()))
}

/// Classifies an error message into one of the failure keys
fn classify_message(message: &str) -> &'static str {
    if message.contains("time guard") {
        "stop-time"
    } else if message.contains("not available") || message.contains("not compiled") || message.contains("enable the `") {
        "not-avail"
    } else if message.contains("symmetric positive definite") {
        "only-spd"
    } else if message.contains("requires a symmetric") || message.contains("requires sym::") {
        "only-sym"
    } else if message.contains("not positive definite") {
        "not-pd"
    } else if message.contains("did not converge") {
        "no-conv"
    } else if message.contains("broke down") {
        "breakdown"
    } else if message.contains("singular") {
        "singular"
    } else if message.contains("too large") {
        "too-big"
    } else if message.contains("nan")
        || message.contains("non-finite")
        || message.contains("not square")
        || message.contains("must be square")
        || message.contains("invalid")
        || message.contains("shape")
        || message.contains("dimension")
        || message.contains("duplicate")
        || message.contains("out of range")
        || message.contains("not sorted")
        || message.contains("at least one non-zero")
        || message.contains("rejected")
    {
        "bad-input"
    } else {
        "error"
    }
}

/// Merges one solver record into the matrix map
fn merge_record(matrices: &mut HashMap<String, MatrixData>, data: MatrixData) {
    if let Some(row) = matrices.get_mut(&data.name) {
        row.complex |= data.complex;
        row.symmetric |= data.symmetric;
        row.positive_definite |= data.positive_definite;
        row.nnz = row.nnz.max(data.nnz);
        for (solver, total_ifs) in data.times {
            if !row.times.contains_key(&solver) {
                row.times.insert(solver.clone(), total_ifs);
                row.failures.remove(&solver);
            }
        }
        for (solver, key) in data.failures {
            if !row.times.contains_key(&solver) {
                row.failures.entry(solver).or_insert(key);
            }
        }
    } else {
        matrices.insert(data.name.clone(), data);
    }
}

/// Returns the solver columns, with the common solvers first
fn solver_columns(rows: &[MatrixData]) -> Vec<String> {
    let mut found = BTreeSet::new();
    for row in rows {
        for solver in row.times.keys().chain(row.failures.keys()) {
            found.insert(solver.clone());
        }
    }
    let mut solvers = Vec::new();
    for solver in SOLVER_ORDER {
        if found.remove(solver) {
            solvers.push(solver.to_string());
        }
    }
    solvers.extend(found);
    solvers
}

/// Returns the failure keys that appear in the table
fn used_failure_keys(rows: &[MatrixData]) -> BTreeSet<&'static str> {
    rows.iter().flat_map(|row| row.failures.values().copied()).collect()
}

/// Returns the display name used for a matrix in the table
fn display_name(name: &str) -> &str {
    match name {
        "pres-cylin-3d-tet10-coarse" => "pres-cylin-c",
        "pres-cylin-3d-tet10-fine" => "pres-cylin-f",
        _ => name,
    }
}

/// Returns the symmetry classification used in the table
fn sym_label(row: &MatrixData) -> &'static str {
    if !row.symmetric {
        "no"
    } else if row.positive_definite {
        // SPD = symmetric positive definite (PSD would mean positive semi-definite)
        "spd"
    } else {
        "yes"
    }
}

/// Prints one comparison table
fn print_table(title: &str, rows: &[MatrixData], solvers: &[String]) {
    println!("{title}");
    if rows.is_empty() {
        println!("(no matrices)");
        return;
    }

    let mut headers: Vec<String> = vec!["matrix".to_string(), "nnz".to_string(), "sym".to_string()];
    headers.extend(solvers.iter().cloned());

    let mut cells: Vec<Vec<String>> = Vec::with_capacity(rows.len());
    for row in rows {
        let mut line = vec![display_name(&row.name).to_string(), row.nnz.to_string(), sym_label(row).to_string()];
        for solver in solvers {
            let cell = if let Some(ns) = row.times.get(solver) {
                fmt_nano(*ns as u128, 2)
            } else if let Some(key) = row.failures.get(solver) {
                (*key).to_string()
            } else {
                String::new()
            };
            line.push(cell);
        }
        cells.push(line);
    }

    let mut widths: Vec<usize> = headers.iter().map(|header| header.chars().count()).collect();
    for line in &cells {
        for (i, cell) in line.iter().enumerate() {
            widths[i] = widths[i].max(cell.chars().count());
        }
    }
    let right_aligned: Vec<bool> = (0..headers.len()).map(|i| i == 1 || i >= 3).collect();

    println!("{}", format_row(&headers, &widths, &right_aligned));
    let separator: Vec<String> = widths.iter().map(|width| "-".repeat(*width)).collect();
    println!("{}", separator.join("  "));
    for line in &cells {
        println!("{}", format_row(line, &widths, &right_aligned));
    }
}

/// Prints the descriptions of the failure keys that were used
fn print_key_descriptions(keys: &BTreeSet<&'static str>) {
    if keys.is_empty() {
        return;
    }
    println!();
    println!("keys:");
    for &(key, description) in KEY_DESCRIPTIONS {
        if keys.contains(key) {
            println!("  {:<10} {}", key, description);
        }
    }
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
    println!("Usage: xtract_matrix_market [results_dir]");
    println!();
    println!("Reads the JSON files written by solve_matrix_market and prints a solver timing table.");
    println!("Every solver column shows its absolute total time; no MUMPS baseline is required.");
    println!("Failure keys are printed after the table; blank cells mean no result was found.");
    println!("Default directory: $OUTPUT_DIR when set, otherwise /tmp/russell_sparse/results/$SYSTEM.");
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Builds a minimal successful record
    fn sample(solver: &str, name: &str, nnz: u64, symmetric: &str, positive_definite: bool, total_ifs: u64) -> Value {
        json!({
            "main": {"solver": solver, "solved": true},
            "matrix": {"name": name, "nnz": nnz, "symmetric": symmetric},
            "requests": {"positive_definite": positive_definite},
            "time_nanoseconds": {"total_ifs": total_ifs},
        })
    }

    #[test]
    fn parse_record_works() {
        let data = parse_record(&sample("Faer", "mat1", 10, "YesLower", true, 1234)).unwrap();
        assert_eq!(data.name, "mat1");
        assert_eq!(data.nnz, 10);
        assert!(data.symmetric);
        assert!(data.positive_definite);
        assert_eq!(data.times["faer"], 1234);
        assert!(data.failures.is_empty());
    }

    #[test]
    fn parse_record_keeps_failures() {
        let mut value = sample("QDLDL", "mat1", 10, "No", false, 0);
        value["main"]["solved"] = json!(false);
        value["main"]["error_from_solver"] = json!("QDLDL requires a symmetric matrix");
        let data = parse_record(&value).unwrap();
        assert_eq!(data.failures["qdldl"], "only-sym");
        assert!(data.times.is_empty());
    }

    #[test]
    fn parse_record_reads_the_complex_flag() {
        let mut value = sample("Faer", "mat1", 10, "No", false, 1234);
        value["matrix"]["complex"] = json!(true);
        assert!(parse_record(&value).unwrap().complex);
    }

    #[test]
    fn parse_record_prefers_actual_nnz() {
        let mut value = sample("Mumps", "mat1", 10, "YesLower", false, 1234);
        value["matrix"]["nnz_actual"] = json!(19);
        let data = parse_record(&value).unwrap();
        assert_eq!(data.nnz, 19);
    }

    #[test]
    fn parse_record_skips_success_without_time() {
        assert!(parse_record(&sample("Faer", "mat1", 10, "No", false, 0)).is_none());
    }

    #[test]
    fn parse_record_skips_information_only_records() {
        let mut value = sample("FAER", "mat1", 10, "No", false, 0);
        value["main"]["solved"] = json!(false);
        assert!(parse_record(&value).is_none());
    }

    #[test]
    fn failure_keys_are_classified() {
        assert_eq!(
            failure_key(&json!({
                "main": {"stopped_by_time_guard": true, "out_of_memory": false, "solved": false}
            }))
            .unwrap(),
            "stop-time"
        );
        assert_eq!(
            failure_key(&json!({
                "main": {"stopped_by_time_guard": false, "out_of_memory": true, "solved": false}
            }))
            .unwrap(),
            "no-mem"
        );
        assert!(failure_key(&json!({"main": {"solved": false}})).is_none());
        for (message, key) in [
            ("QDLDL requires a symmetric matrix", "only-sym"),
            ("CG failed: the matrix may not be symmetric positive definite", "only-spd"),
            ("the matrix is not positive definite as assumed", "not-pd"),
            ("the iterative solver did not converge", "no-conv"),
            ("the iterative solver broke down", "breakdown"),
            ("the matrix is singular", "singular"),
            ("the matrix is too large for the Faer solver", "too-big"),
            ("the matrix must be square", "bad-input"),
            ("MUMPS is not available; enable the `mumps` feature", "not-avail"),
            ("something unexpected", "error"),
        ] {
            assert_eq!(classify_message(&message.to_lowercase()), key, "message: {message}");
        }
    }

    #[test]
    fn merge_record_prefers_a_successful_time() {
        let mut matrices = HashMap::new();
        let mut failed = parse_record(&sample("Faer", "mat1", 10, "No", false, 1)).unwrap();
        failed.times.clear();
        failed.failures.insert("faer".to_string(), "error");
        merge_record(&mut matrices, failed);
        let data = parse_record(&sample("Faer", "mat1", 10, "No", false, 100)).unwrap();
        merge_record(&mut matrices, data);

        let row = &matrices["mat1"];
        assert_eq!(row.times["faer"], 100);
        assert!(!row.failures.contains_key("faer"));
    }

    #[test]
    fn merge_record_or_updates_flags_and_keeps_first_time() {
        let mut matrices = HashMap::new();
        merge_record(&mut matrices, parse_record(&sample("Mumps", "mat1", 10, "YesFull", true, 100)).unwrap());
        merge_record(&mut matrices, parse_record(&sample("UMFPACK", "mat1", 10, "No", false, 200)).unwrap());
        merge_record(&mut matrices, parse_record(&sample("Mumps", "mat1", 10, "YesFull", true, 300)).unwrap());

        let row = &matrices["mat1"];
        assert!(row.symmetric);
        assert!(row.positive_definite);
        assert_eq!(row.times["mumps"], 100);
        assert_eq!(row.times["umfpack"], 200);
    }

    #[test]
    fn display_names_are_shortened() {
        assert_eq!(display_name("pres-cylin-3d-tet10-coarse"), "pres-cylin-c");
        assert_eq!(display_name("pres-cylin-3d-tet10-fine"), "pres-cylin-f");
        assert_eq!(display_name("bwm2000"), "bwm2000");
    }

    #[test]
    fn sym_label_works() {
        let mut row = MatrixData::default();
        assert_eq!(sym_label(&row), "no");
        row.symmetric = true;
        assert_eq!(sym_label(&row), "yes");
        row.positive_definite = true;
        assert_eq!(sym_label(&row), "spd");
    }

    #[test]
    fn solver_columns_include_failures_and_are_ordered() {
        let mut row = MatrixData::default();
        for solver in ["gmres", "basiclu", "faer", "mumps", "qdldl"] {
            row.times.insert(solver.to_string(), 1);
        }
        row.failures.insert("umfpack".to_string(), "only-sym");
        assert_eq!(solver_columns(&[row]), ["mumps", "umfpack", "faer", "qdldl", "basiclu", "gmres"]);
    }

    #[test]
    fn solver_columns_work_without_mumps() {
        let mut row = MatrixData::default();
        row.times.insert("faer".to_string(), 1);
        assert_eq!(solver_columns(&[row]), ["faer"]);
    }

    #[test]
    fn used_failure_keys_are_collected() {
        let mut row = MatrixData::default();
        row.failures.insert("faer".to_string(), "only-sym");
        row.failures.insert("umfpack".to_string(), "no-mem");
        let keys = used_failure_keys(&[row]);
        assert!(keys.contains("only-sym"));
        assert!(keys.contains("no-mem"));
    }

    #[test]
    fn format_row_aligns_columns() {
        let cells = vec!["a".to_string(), "12".to_string()];
        let widths = vec![3, 4];
        let right = vec![false, true];
        assert_eq!(format_row(&cells, &widths, &right), "a      12");
    }
}
