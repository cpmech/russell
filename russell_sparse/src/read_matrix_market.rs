use super::*;
use std::fs::File;
use std::io::{BufRead, BufReader};

struct MatrixMarketData {
    // header
    symmetric: bool,

    // dimensions
    m: i32,   // number of rows
    n: i32,   // number of columns
    nnz: i32, // number of non-zeros

    // current triple
    i: i32,   // current i-index
    j: i32,   // current j-index
    aij: f64, // current aij-value
    pos: i32, // current position in the list of triples
}

impl MatrixMarketData {
    fn new() -> Self {
        MatrixMarketData {
            symmetric: false,
            m: 0,
            n: 0,
            nnz: 0,
            i: 0,
            j: 0,
            aij: 0.0,
            pos: 0,
        }
    }

    #[inline]
    fn parse_header(&mut self, line: &String) -> Result<(), &'static str> {
        let mut data = line.trim_start().trim_end_matches("\n").split_whitespace();

        match data.next() {
            Some(v) => {
                if v != "%%MatrixMarket" {
                    return Err("the header (first line) must start with %%MatrixMarket");
                }
            }
            None => return Err("cannot find the keyword %%MatrixMarket on the first line"),
        }

        match data.next() {
            Some(v) => {
                if v != "matrix" {
                    return Err("after %%MatrixMarket, the first option must be \"matrix\"");
                }
            }
            None => return Err("cannot find the first option in the header line"),
        }

        match data.next() {
            Some(v) => {
                if v != "coordinate" {
                    return Err("after %%MatrixMarket, the second option must be \"coordinate\"");
                }
            }
            None => return Err("cannot find the second option in the header line"),
        }

        match data.next() {
            Some(v) => {
                if v != "real" {
                    return Err("after %%MatrixMarket, the third option must be \"real\"");
                }
            }
            None => return Err("cannot find the third option in the header line"),
        }

        match data.next() {
            Some(v) => match v {
                "general" => self.symmetric = false,
                "symmetric" => self.symmetric = true,
                _ => return Err("after %%MatrixMarket, the fourth option must be either \"general\" or \"symmetric\""),
            }
            None => return Err("cannot find the fourth option in the header line"),
        }

        Ok(())
    }

    #[inline]
    fn parse_dimensions(&mut self, line: &String) -> Result<bool, &'static str> {
        let maybe_data = line.trim_start().trim_end_matches("\n");
        if maybe_data.starts_with("%") || maybe_data == "" {
            return Ok(false); // ignore comments or empty lines; returns false == not parsed
        }

        let mut data = maybe_data.split_whitespace();

        match data.next() {
            Some(v) => self.m = v.parse().map_err(|_| "cannot parse number of rows")?,
            None => return Err("cannot read number of rows"),
        };

        match data.next() {
            Some(v) => self.n = v.parse().map_err(|_| "cannot parse number of columns")?,
            None => return Err("cannot read number of columns"),
        };

        match data.next() {
            Some(v) => self.nnz = v.parse().map_err(|_| "cannot parse number of non-zeros")?,
            None => return Err("cannot read number of non-zeros"),
        };

        if self.m < 1 || self.n < 1 || self.nnz < 1 {
            return Err("found invalid (zero or negative) dimensions");
        }

        Ok(true) // returns true == parsed
    }

    #[inline]
    fn parse_triple(&mut self, line: &String) -> Result<bool, &'static str> {
        let maybe_data = line.trim_start().trim_end_matches("\n");
        if maybe_data.starts_with("%") || maybe_data == "" {
            return Ok(false); // ignore comments or empty lines
        }

        if self.pos == self.nnz {
            return Err("there are more (i,j,aij) triples than specified");
        }

        let mut data = maybe_data.split_whitespace();

        match data.next() {
            Some(v) => self.i = v.parse().map_err(|_| "cannot parse index i")?,
            None => return Err("cannot read index i"),
        };

        match data.next() {
            Some(v) => self.j = v.parse().map_err(|_| "cannot parse index j")?,
            None => return Err("cannot read index j"),
        };

        match data.next() {
            Some(v) => self.aij = v.parse().map_err(|_| "cannot parse value aij")?,
            None => return Err("cannot read value aij"),
        };

        self.i -= 1; // MatrixMarket is one-based
        self.j -= 1;

        if self.i < 0 || self.i >= self.m || self.j < 0 || self.j >= self.n {
            return Err("found invalid indices");
        }

        self.pos += 1; // next position

        Ok(true) // returns true == parsed
    }
}

/// Reads a MatrixMarket file into a SparseTriplet
///
/// # Input
///
/// * `filepath` -- The full file path with filename
///
/// sample headers
/// %%MatrixMarket matrix coordinate complex symmetric
/// %%MatrixMarket matrix coordinate real    general
pub fn read_matrix_market(filepath: &String) -> Result<SparseTriplet, &'static str> {
    let input = File::open(filepath).map_err(|_| "cannot open file")?;
    let buffered = BufReader::new(input);
    let mut lines_iter = buffered.lines();

    // auxiliary data structure
    let mut data = MatrixMarketData::new();

    // read first line
    let header = match lines_iter.next() {
        Some(v) => match v {
            Ok(l) => l,
            Err(_) => return Err("cannot read the first line"),
        },
        None => return Err("file is empty"),
    };

    // parse header
    data.parse_header(&header)?;

    // read and parse dimensions
    loop {
        match lines_iter.next() {
            Some(v) => match v {
                Ok(line) => {
                    if data.parse_dimensions(&line)? {
                        break;
                    }
                }
                Err(_) => return Err("cannot find line with dimensions"),
            },
            None => break,
        }
    }

    // allocate triplet
    let mut trip = SparseTriplet::new(
        data.m as usize,
        data.n as usize,
        data.nnz as usize,
        data.symmetric,
    )?;

    // read and parse triples
    loop {
        match lines_iter.next() {
            Some(v) => match v {
                Ok(line) => {
                    if data.parse_triple(&line)? {
                        trip.put(data.i as usize, data.j as usize, data.aij);
                    }
                }
                Err(_) => return Err("cannot read line with (i,j,aij) triple"),
            },
            None => break,
        }
    }

    Ok(trip)
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_header_captures_errors() -> Result<(), &'static str> {
        let mut data = MatrixMarketData::new();

        assert_eq!(
            data.parse_header(&String::from("  \n")),
            Err("cannot find the keyword %%MatrixMarket on the first line")
        );
        assert_eq!(
            data.parse_header(&String::from("MatrixMarket  ")),
            Err("the header (first line) must start with %%MatrixMarket"),
        );

        assert_eq!(
            data.parse_header(&String::from("  %%MatrixMarket")),
            Err("cannot find the first option in the header line"),
        );
        assert_eq!(
            data.parse_header(&String::from("%%MatrixMarket   wrong")),
            Err("after %%MatrixMarket, the first option must be \"matrix\""),
        );

        assert_eq!(
            data.parse_header(&String::from("%%MatrixMarket matrix  ")),
            Err("cannot find the second option in the header line"),
        );
        assert_eq!(
            data.parse_header(&String::from("%%MatrixMarket   matrix wrong")),
            Err("after %%MatrixMarket, the second option must be \"coordinate\""),
        );

        assert_eq!(
            data.parse_header(&String::from("%%MatrixMarket matrix  coordinate")),
            Err("cannot find the third option in the header line"),
        );
        assert_eq!(
            data.parse_header(&String::from("%%MatrixMarket matrix    coordinate  wrong")),
            Err("after %%MatrixMarket, the third option must be \"real\""),
        );

        assert_eq!(
            data.parse_header(&String::from("%%MatrixMarket  matrix coordinate real")),
            Err("cannot find the fourth option in the header line"),
        );
        assert_eq!(
            data.parse_header(&String::from("  %%MatrixMarket matrix coordinate real wrong")),
               Err("after %%MatrixMarket, the fourth option must be either \"general\" or \"symmetric\""),
        );
        Ok(())
    }

    #[test]
    fn parse_dimensions_captures_errors() -> Result<(), &'static str> {
        let mut data = MatrixMarketData::new();

        assert_eq!(
            data.parse_dimensions(&String::from(" wrong \n")).err(),
            Some("cannot parse number of rows")
        );

        assert_eq!(
            data.parse_dimensions(&String::from(" 1 \n")).err(),
            Some("cannot read number of columns")
        );
        assert_eq!(
            data.parse_dimensions(&String::from(" 1 wrong")).err(),
            Some("cannot parse number of columns")
        );

        assert_eq!(
            data.parse_dimensions(&String::from(" 1 1   \n")).err(),
            Some("cannot read number of non-zeros")
        );
        assert_eq!(
            data.parse_dimensions(&String::from(" 1 1  wrong")).err(),
            Some("cannot parse number of non-zeros")
        );

        assert_eq!(
            data.parse_dimensions(&String::from(" 0 1  1")).err(),
            Some("found invalid (zero or negative) dimensions")
        );
        assert_eq!(
            data.parse_dimensions(&String::from(" 1 0  1")).err(),
            Some("found invalid (zero or negative) dimensions")
        );
        assert_eq!(
            data.parse_dimensions(&String::from(" 1 1  0")).err(),
            Some("found invalid (zero or negative) dimensions")
        );
        Ok(())
    }

    #[test]
    fn parse_triple_captures_errors() -> Result<(), &'static str> {
        let mut data = MatrixMarketData::new();
        data.m = 2;
        data.n = 2;
        data.nnz = 1;

        assert_eq!(
            data.parse_triple(&String::from(" wrong \n")).err(),
            Some("cannot parse index i")
        );

        assert_eq!(
            data.parse_triple(&String::from(" 1 \n")).err(),
            Some("cannot read index j")
        );
        assert_eq!(
            data.parse_triple(&String::from(" 1 wrong")).err(),
            Some("cannot parse index j")
        );

        assert_eq!(
            data.parse_triple(&String::from(" 1 1   \n")).err(),
            Some("cannot read value aij")
        );
        assert_eq!(
            data.parse_triple(&String::from(" 1 1  wrong")).err(),
            Some("cannot parse value aij")
        );

        assert_eq!(
            data.parse_triple(&String::from(" 0 1  1")).err(),
            Some("found invalid indices")
        );
        assert_eq!(
            data.parse_triple(&String::from(" 3 1  1")).err(),
            Some("found invalid indices")
        );
        assert_eq!(
            data.parse_triple(&String::from(" 1 0  1")).err(),
            Some("found invalid indices")
        );
        assert_eq!(
            data.parse_triple(&String::from(" 1 3  1")).err(),
            Some("found invalid indices")
        );
        Ok(())
    }

    #[test]
    fn read_matrix_market_works() -> Result<(), &'static str> {
        let filepath = "./data/sparse-matrix/ok1.mtx".to_string();
        let trip = read_matrix_market(&filepath)?;
        assert!(trip.symmetric == false);
        assert_eq!((trip.nrow, trip.ncol, trip.pos, trip.max), (5, 5, 12, 12));
        assert_eq!(trip.indices_i, &[0, 1, 0, 2, 4, 1, 2, 3, 4, 2, 1, 4]);
        assert_eq!(trip.indices_j, &[0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 4, 4]);
        assert_eq!(
            trip.values_a,
            &[2.0, 3.0, 3.0, -1.0, 4.0, 4.0, -3.0, 1.0, 2.0, 2.0, 6.0, 1.0]
        );
        Ok(())
    }
}
