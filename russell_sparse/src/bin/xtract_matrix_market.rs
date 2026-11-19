//! Extracts solver timing results from the JSON files written by `solve_matrix_market`.
//!
//! Usage: `cargo run --release --bin xtract_matrix_market [results_dir]`
//!
//! The default directory is the one used by `run_solve_matrix_market.bash`:
//! `$OUTPUT_DIR` when set, otherwise `/tmp/russell_sparse/results/$SYSTEM`.

use russell_lab::format_nanoseconds;
use serde_json::Value;
use std::collections::{BTreeSet, HashMap};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

/// Solvers whose time is required for every matrix.
const REQUIRED: [&str; 1] = ["mumps"];

/// Optional solver columns with a preferred position in the table.
const NAMED_OPTIONAL: [&str; 4] = ["umfpack", "faer", "qdldl", "basiclu"];

/// Holds one row of the comparison table.
#[derive(Debug, Default)]
struct MatrixData {
    name: String,
    nnz: u64,
    symmetric: bool,
    positive_definite: bool,
    times: HashMap<String, u64>,
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
        if let Some((solver, data)) = parse_record(&value) {
            merge_record(&mut matrices, solver, data);
        }
    }

    if matrices.is_empty() {
        return Err("no valid solved runs found".to_string());
    }

    let mut rows: Vec<MatrixData> = matrices.into_values().collect();
    rows.sort_by(|aa, bb| aa.nnz.cmp(&bb.nnz).then_with(|| aa.name.cmp(&bb.name)));

    for row in &rows {
        for solver in REQUIRED {
            if !row.times.contains_key(solver) {
                return Err(format!("matrix {} has no valid {} time", row.name, solver));
            }
        }
    }

    let optional = optional_solvers(&rows);
    print_table(&rows, &optional);
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

/// Extracts the solver name and timing data from one JSON record
fn parse_record(value: &Value) -> Option<(String, MatrixData)> {
    let solver = value.pointer("/main/solver")?.as_str()?.to_lowercase();
    let solved = value.pointer("/main/solved").and_then(Value::as_bool).unwrap_or(true);
    if !solved {
        return None;
    }

    let name = value.pointer("/matrix/name")?.as_str()?.to_string();
    let nnz = value.pointer("/matrix/nnz").and_then(Value::as_u64).unwrap_or(0);
    let symmetric = value
        .pointer("/matrix/symmetric")
        .and_then(Value::as_str)
        .map(|sym| !sym.eq_ignore_ascii_case("no"))
        .unwrap_or(false);
    let positive_definite = value.pointer("/requests/positive_definite").and_then(Value::as_bool).unwrap_or(false);
    let total_ifs = value.pointer("/time_nanoseconds/total_ifs").and_then(Value::as_u64).unwrap_or(0);
    if total_ifs == 0 {
        return None;
    }

    Some((
        solver.clone(),
        MatrixData {
            name,
            nnz,
            symmetric,
            positive_definite,
            times: HashMap::from([(solver, total_ifs)]),
        },
    ))
}

/// Merges one solver record into the matrix map
fn merge_record(matrices: &mut HashMap<String, MatrixData>, solver: String, data: MatrixData) {
    if let Some(row) = matrices.get_mut(&data.name) {
        row.symmetric |= data.symmetric;
        row.positive_definite |= data.positive_definite;
        if row.nnz == 0 {
            row.nnz = data.nnz;
        }
        if let Some(total_ifs) = data.times.get(&solver) {
            row.times.entry(solver).or_insert(*total_ifs);
        }
    } else {
        matrices.insert(data.name.clone(), data);
    }
}

/// Returns the optional solver columns, with the named ones first
fn optional_solvers(rows: &[MatrixData]) -> Vec<String> {
    let mut found = BTreeSet::new();
    for row in rows {
        for solver in row.times.keys() {
            if !REQUIRED.contains(&solver.as_str()) {
                found.insert(solver.clone());
            }
        }
    }
    let mut optional = Vec::new();
    for solver in NAMED_OPTIONAL {
        if found.remove(solver) {
            optional.push(solver.to_string());
        }
    }
    optional.extend(found);
    optional
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

/// Prints the comparison table
fn print_table(rows: &[MatrixData], optional: &[String]) {
    let mut headers: Vec<String> = vec!["matrix".to_string(), "nnz".to_string(), "sym".to_string()];
    for solver in REQUIRED {
        headers.push(format!("{}_time", solver));
    }
    for solver in optional {
        headers.push(format!("{}_time", solver));
    }

    let mut cells: Vec<Vec<String>> = Vec::with_capacity(rows.len());
    for row in rows {
        let mut line = vec![row.name.clone(), row.nnz.to_string(), sym_label(row).to_string()];
        for solver in REQUIRED.iter().copied().chain(optional.iter().map(String::as_str)) {
            let time = row.times.get(solver).map(|ns| format_nanoseconds(*ns as u128)).unwrap_or_default();
            line.push(time);
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
    println!("MUMPS times are required for every matrix; the other solvers are optional.");
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
        let value = sample("Faer", "mat1", 10, "YesLower", true, 1234);
        let (solver, data) = parse_record(&value).unwrap();
        assert_eq!(solver, "faer");
        assert_eq!(data.name, "mat1");
        assert_eq!(data.nnz, 10);
        assert!(data.symmetric);
        assert!(data.positive_definite);
        assert_eq!(data.times["faer"], 1234);
    }

    #[test]
    fn parse_record_skips_unsolved_and_zero_times() {
        let mut value = sample("Faer", "mat1", 10, "No", false, 1234);
        value["main"]["solved"] = json!(false);
        assert!(parse_record(&value).is_none());
        let value = sample("Faer", "mat1", 10, "No", false, 0);
        assert!(parse_record(&value).is_none());
    }

    #[test]
    fn merge_record_or_updates_flags_and_keeps_first_time() {
        let mut matrices = HashMap::new();
        let (solver, data) = parse_record(&sample("Mumps", "mat1", 10, "YesFull", true, 100)).unwrap();
        merge_record(&mut matrices, solver, data);
        let (solver, data) = parse_record(&sample("UMFPACK", "mat1", 10, "No", false, 200)).unwrap();
        merge_record(&mut matrices, solver, data);
        let (solver, data) = parse_record(&sample("Mumps", "mat1", 10, "YesFull", true, 300)).unwrap();
        merge_record(&mut matrices, solver, data);

        let row = &matrices["mat1"];
        assert!(row.symmetric);
        assert!(row.positive_definite);
        assert_eq!(row.times["mumps"], 100);
        assert_eq!(row.times["umfpack"], 200);
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
    fn optional_solvers_are_ordered() {
        let mut row = MatrixData::default();
        for solver in ["gmres", "basiclu", "faer", "mumps", "umfpack", "qdldl"] {
            row.times.insert(solver.to_string(), 1);
        }
        assert_eq!(optional_solvers(&[row]), ["umfpack", "faer", "qdldl", "basiclu", "gmres"]);
    }

    #[test]
    fn format_row_aligns_columns() {
        let cells = vec!["a".to_string(), "12".to_string()];
        let widths = vec![3, 4];
        let right = vec![false, true];
        assert_eq!(format_row(&cells, &widths, &right), "a      12");
    }
}
