use crate::StrError;
use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::str::FromStr;

/// Reads a file containing tabled data
///
/// # Input
///
/// * `full_path` -- may be a String, &str, or Path
/// * `labels` -- if provided, it is used to check the header labels,
///   otherwise, the labels will be named "col0", "col1", "col2", ...
///
/// # Notes
///
/// * Comments start with the hash character '#'
/// * Lines starting with '#' or empty lines are ignored
/// * The end of the row (line) may contain comments too and will cause to stop reading data,
///   thus, the '#' marker in a row (line) must be at the end of the line.
pub fn read_table<T, P>(full_path: &P, labels: Option<&[&str]>) -> Result<HashMap<String, Vec<T>>, StrError>
where
    T: FromStr,
    P: AsRef<OsStr> + ?Sized,
{
    // read file
    let path = Path::new(full_path).to_path_buf();
    let input = File::open(path).map_err(|_| "cannot open file")?;
    let buffered = BufReader::new(input);
    let mut lines_iter = buffered.lines();

    // results
    let mut header_labels = Vec::new();
    let mut table: HashMap<String, Vec<T>> = HashMap::new();

    // parse rows, ignoring comments and empty lines
    let mut first_row = true;
    let mut number_of_columns = 0;
    loop {
        match lines_iter.next() {
            Some(v) => {
                // extract line
                let line = v.unwrap(); // must panic because no error expected here

                // ignore comments or empty lines
                let maybe_data = line.trim_start().trim_end_matches("\n");
                if maybe_data.starts_with("#") || maybe_data == "" {
                    continue; // nothing to parse
                }

                // remove whitespace
                let mut row_values = maybe_data.split_whitespace();

                // loop over columns
                let mut column_index = 0;
                loop {
                    match row_values.next() {
                        Some(s) => {
                            if s.starts_with("#") {
                                break; // ignore comments at the end of the row
                            }
                            if first_row {
                                // check header labels or create new labels
                                let label = match &labels {
                                    Some(ls) => {
                                        if column_index >= ls.len() {
                                            return Err("there are more columns than labels");
                                        }
                                        ls[column_index].to_string()
                                    }
                                    None => {
                                        format!("col{}", column_index)
                                    }
                                };
                                header_labels.push(label);
                            } else {
                                // parse value
                                if column_index >= header_labels.len() {
                                    return Err("there are more columns than labels");
                                }
                                let label = header_labels[column_index].clone();
                                let column = table.entry(label).or_insert(Vec::new());
                                let value = s.parse::<T>().map_err(|_| "cannot parse value")?;
                                column.push(value);
                            };
                            column_index += 1;
                        }
                        None => break,
                    }
                }

                // set or check the number of columns
                if first_row {
                    number_of_columns = column_index; // the first row determines the number of columns
                    first_row = false;
                } else {
                    if column_index != number_of_columns {
                        return Err("column data is missing");
                    }
                }
            }
            None => break,
        }
    }

    Ok(table)
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::read_table;
    use crate::StrError;
    use std::collections::HashMap;

    #[test]
    fn read_table_handles_problems() {
        let mut table: Result<HashMap<String, Vec<i32>>, StrError>;
        table = read_table("", None);
        assert_eq!(table.err(), Some("cannot open file"));

        table = read_table("not-found", None);
        assert_eq!(table.err(), Some("cannot open file"));

        table = read_table("./data/tables/ok1.txt", Some(&["column_0"]));
        assert_eq!(table.err(), Some("there are more columns than labels"));

        table = read_table("./data/tables/bad_more_columns_than_labels.txt", None);
        assert_eq!(table.err(), Some("there are more columns than labels"));

        table = read_table("./data/tables/bad_cannot_parse_value.txt", None);
        assert_eq!(table.err(), Some("cannot parse value"));

        table = read_table("./data/tables/bad_missing_data.txt", None);
        assert_eq!(table.err(), Some("column data is missing"));
    }

    const OK2_COL0: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    const OK2_COL1: [f64; 5] = [-6.0, 7.0, 8.0, 9.0, 10.0];
    const OK2_COL2: [f64; 5] = [0.1, 0.2, 0.2, 0.4, 0.5];

    #[test]
    fn read_table_works() {
        let full_path = "./data/tables/ok2.txt";
        let mut table: HashMap<String, Vec<f64>>;
        table = read_table(full_path, None).unwrap();
        let mut labels: Vec<_> = table.keys().collect();
        labels.sort();
        assert_eq!(labels, &["col0", "col1", "col2"]);
        assert_eq!(table.get("col0").unwrap(), &OK2_COL0);
        assert_eq!(table.get("col1").unwrap(), &OK2_COL1);
        assert_eq!(table.get("col2").unwrap(), &OK2_COL2);

        table = read_table(full_path, Some(&["sr", "ea", "er"])).unwrap();
        let mut labels: Vec<_> = table.keys().collect();
        labels.sort();
        assert_eq!(labels, &["ea", "er", "sr"]);
        assert_eq!(table.get("sr").unwrap(), &OK2_COL0);
        assert_eq!(table.get("ea").unwrap(), &OK2_COL1);
        assert_eq!(table.get("er").unwrap(), &OK2_COL2);
    }

    #[test]
    fn read_table_string_works() {
        let full_path = "./data/tables/ok3.txt";
        let table: HashMap<String, Vec<String>> = read_table(full_path, Some(&["names", "colors"])).unwrap();
        let mut labels: Vec<_> = table.keys().collect();
        labels.sort();
        assert_eq!(labels, &["colors", "names"]);
        assert_eq!(table.get("names").unwrap(), &["red", "green", "blue"]);
        assert_eq!(
            table.get("colors").unwrap(),
            &["\"#ff0000\"", "\"#00ff00\"", "\"#0000ff\""]
        );
    }
}
