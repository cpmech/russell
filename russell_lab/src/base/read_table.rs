use crate::StrError;
use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::str::FromStr;

/// Reads data in tabular format
///
/// Note: This function is a wrapper to [read_table()] with f64 numbers and String labels for the columns.
///
/// # Input
///
/// * `full_path` -- may be a String, &str, or Path
/// * `labels` -- the column names in the header of the file and works as keys for the resulting HashMap
///
/// # Examples
///
/// ```
/// use russell_lab::{read_data, StrError};
/// use std::env;
/// use std::path::PathBuf;
///
/// fn main() -> Result<(), StrError> {
///     // <my-data.txt>
///     //
///     // x y z
///     // 1 2 3
///     // 4 5 6
///     // 7 8 9
///
///     // get the asset's full path
///     let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
///     let full_path = root.join("data/tables/my-data.txt");
///
///     // read the file
///     let data = read_data(&full_path, &["x", "y", "z"])?;
///
///     // check the data
///     assert_eq!(data.get("x").unwrap(), &[1.0, 4.0, 7.0]);
///     assert_eq!(data.get("y").unwrap(), &[2.0, 5.0, 8.0]);
///     assert_eq!(data.get("z").unwrap(), &[3.0, 6.0, 9.0]);
///     Ok(())
/// }
/// ```
pub fn read_data<P>(full_path: &P, labels: &[&str]) -> Result<HashMap<String, Vec<f64>>, StrError>
where
    P: AsRef<OsStr> + ?Sized,
{
    read_table(full_path, Some(labels))
}

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
///
/// # Examples
///
/// The goal is to read the following file (`clay-data.txt`):
///
/// ```text
/// # Fujinomori clay test results
///
///      sr        ea        er   # header
/// 1.00000  -6.00000   0.10000   
/// 2.00000   7.00000   0.20000   
/// 3.00000   8.00000   0.20000   # << look at this line
///
/// # comments plus new lines are OK
///
/// 4.00000   9.00000   0.40000   
/// 5.00000  10.00000   0.50000   
///
/// # bye
/// ```
///
/// The code below illustrates how to do it.
///
/// Each column (`sr`, `ea`, `er`) is accessible via the `get` method of the [HashMap].
///
/// ```
/// use russell_lab::{read_table, StrError};
/// use std::collections::HashMap;
/// use std::env;
/// use std::path::PathBuf;
///
/// fn main() -> Result<(), StrError> {
///     // get the asset's full path
///     let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
///     let full_path = root.join("data/tables/clay-data.txt");
///
///     // read the file
///     let labels = &["sr", "ea", "er"];
///     let table: HashMap<String, Vec<f64>> = read_table(&full_path, Some(labels))?;
///
///     // check the columns
///     assert_eq!(table.get("sr").unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
///     assert_eq!(table.get("ea").unwrap(), &[-6.0, 7.0, 8.0, 9.0, 10.0]);
///     assert_eq!(table.get("er").unwrap(), &[0.1, 0.2, 0.2, 0.4, 0.5]);
///     Ok(())
/// }
/// ```
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
                                        if s != ls[column_index] {
                                            return Err("column data is missing");
                                        }
                                        ls[column_index].to_string()
                                    }
                                    None => {
                                        if header_labels.contains(&s.to_string()) {
                                            return Err("found duplicate column label");
                                        }
                                        s.to_string()
                                    }
                                };
                                header_labels.push(label);
                            } else {
                                // parse value
                                if column_index >= header_labels.len() {
                                    return Err("there are more columns than labels");
                                }
                                let value = s.parse::<T>().map_err(|_| "cannot parse value")?;
                                let label = &header_labels[column_index];
                                match table.get_mut(label) {
                                    Some(column) => column.push(value),
                                    None => {
                                        table.insert(label.clone(), vec![value]);
                                    }
                                }
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
    use super::{read_data, read_table};
    use crate::StrError;
    use std::collections::HashMap;

    #[test]
    fn read_table_handles_problems() {
        let mut table: Result<HashMap<String, Vec<f64>>, StrError>;
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

        let mut table: Result<HashMap<String, Vec<String>>, StrError>;
        table = read_table("", None);
        assert_eq!(table.err(), Some("cannot open file"));

        table = read_table("not-found", None);
        assert_eq!(table.err(), Some("cannot open file"));

        table = read_table("./data/tables/ok1.txt", Some(&["column_0"]));
        assert_eq!(table.err(), Some("there are more columns than labels"));

        table = read_table("./data/tables/bad_more_columns_than_labels.txt", None);
        assert_eq!(table.err(), Some("there are more columns than labels"));

        table = read_table("./data/tables/bad_missing_data.txt", None);
        assert_eq!(table.err(), Some("column data is missing"));

        table = read_table("./data/tables/ok1.txt", Some(&["column_0", "wrong"]));
        assert_eq!(table.err(), Some("column data is missing"));

        table = read_table("./data/tables/bad_duplicate_labels.txt", None);
        assert_eq!(table.err(), Some("found duplicate column label"));
    }

    const OK2_COL0: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    const OK2_COL1: [f64; 5] = [-6.0, 7.0, 8.0, 9.0, 10.0];
    const OK2_COL2: [f64; 5] = [0.1, 0.2, 0.2, 0.4, 0.5];

    #[test]
    fn read_table_works() {
        let full_path = "./data/tables/clay-data.txt";
        let mut table: HashMap<String, Vec<f64>>;

        // no labels
        table = read_table(full_path, None).unwrap();
        let mut labels: Vec<_> = table.keys().collect();
        labels.sort();
        assert_eq!(labels, &["ea", "er", "sr"]);
        assert_eq!(table.get("sr").unwrap(), &OK2_COL0);
        assert_eq!(table.get("ea").unwrap(), &OK2_COL1);
        assert_eq!(table.get("er").unwrap(), &OK2_COL2);

        // with labels
        table = read_table(full_path, Some(&["sr", "ea", "er"])).unwrap();
        let mut labels: Vec<_> = table.keys().collect();
        labels.sort();
        assert_eq!(labels, &["ea", "er", "sr"]);
        assert_eq!(table.get("sr").unwrap(), &OK2_COL0);
        assert_eq!(table.get("ea").unwrap(), &OK2_COL1);
        assert_eq!(table.get("er").unwrap(), &OK2_COL2);

        // wrong labels
        let table: Result<HashMap<String, Vec<f64>>, StrError> = read_table(full_path, Some(&["x", "y", "z"]));
        assert_eq!(table.err(), Some("column data is missing"));
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

    #[test]
    fn read_data_works() {
        let full_path = "./data/tables/ok1.txt";
        let data = read_data(full_path, &["column_0", "column_1"]).unwrap();
        assert_eq!(data.get("column_0").unwrap(), &[1.0]);
        assert_eq!(data.get("column_1").unwrap(), &[2.0]);
    }
}
