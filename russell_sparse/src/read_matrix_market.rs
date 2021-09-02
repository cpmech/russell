use super::*;
use std::fs::File;
use std::io::{BufRead, BufReader};

/// Reads a MatrixMarket file into a SparseTriplet
///
/// # Input
///
/// * `filepath` -- The full file path with filename
///
pub fn read_matrix_market(filepath: &String) -> Result<SparseTriplet, &'static str> {
    let input = File::open(filepath).map_err(|_| "cannot open file")?;
    let buffered = BufReader::new(input);

    // sample headers
    // %%MatrixMarket matrix coordinate complex symmetric
    // %%MatrixMarket matrix coordinate real    general

    // let trip: *mut SparseTriplet;
    // let allocated = false;

    for maybe_line in buffered.lines() {
        let line = match maybe_line {
            Ok(v) => v,
            Err(_) => return Err("cannot read line"),
        };
        if line.starts_with("%%MatrixMarket") {
            // Todo
        }
    }

    Err("cannot read MatrixMarket file")
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_matrix_market_works() -> Result<(), &'static str> {
        let filepath = "./data/sparse-matrix/ok1.mtx".to_string();
        let trip = read_matrix_market(&filepath)?;
        assert_eq!(trip.dims(), (5, 5, 12));
        Ok(())
    }
}
