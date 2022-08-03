use crate::StrError;
use num_traits::Num;
use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

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
pub fn read_table<T, P>(full_path: &P, labels: Option<Vec<String>>) -> Result<HashMap<String, Vec<T>>, StrError>
where
    T: Num + Copy,
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
    let mut current_row_index = -1;
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
                            if current_row_index == -1 {
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
                                let label = header_labels[column_index].clone();
                                let column = table.entry(label).or_insert(Vec::new());
                                let value = T::from_str_radix(s, 10).map_err(|_| "cannot parse value")?;
                                column.push(value);
                            };
                            column_index += 1;
                        }
                        None => break,
                    }
                }

                // handle header line
                if current_row_index == -1 {
                    current_row_index += 1;
                    continue;
                }

                // set or check the number of columns
                if current_row_index == 0 {
                    number_of_columns = column_index; // the first row determines the number of columns
                } else {
                    if column_index != number_of_columns {
                        return Err("column data is missing");
                    }
                }
                current_row_index += 1;
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
    use russell_chk::assert_vec_approx_eq;
    use std::collections::HashMap;

    #[test]
    fn read_table_works() {
        let table: HashMap<String, Vec<f64>> = read_table("./data/tables/table1.txt", None).unwrap();
        let mut labels: Vec<_> = table.keys().collect();
        labels.sort();
        assert_eq!(
            labels,
            &["col0", "col1", "col2", "col3", "col4", "col5", "col6", "col7"]
        );
        let col0 = table.get("col0").unwrap();
        assert_vec_approx_eq!(col0, &[0.5, 0.64, 0.7, 0.78, 0.87, 0.96, 1.07, 1.19, 1.32, 1.46], 1e-15);
        let col2 = table.get("col2").unwrap();
        assert_vec_approx_eq!(
            col2,
            &[0.002, 0.083, 0.169, 0.332, 0.497, 0.658, 0.819, 0.972, 1.126, 1.287],
            1e-15
        );
        let col7 = table.get("col7").unwrap();
        assert_vec_approx_eq!(
            col7,
            &[-0.24, -0.25, -0.25, -0.26, -0.27, -0.28, -0.29, -0.30, -0.31, -0.32],
            1e-15
        );
    }
}
