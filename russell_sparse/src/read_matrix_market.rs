use super::{SparseTriplet, Symmetry};
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
            },
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

        self.m = data
            .next()
            .unwrap() // must panic because no error expected here
            .parse()
            .map_err(|_| "cannot parse number of rows")?;

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

        self.i = data
            .next()
            .unwrap() // must panic because no error expected here
            .parse()
            .map_err(|_| "cannot parse index i")?;

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
/// * `sym_mirror` -- Tells the reader to mirror the **off diagonal** entries,
///                   if the symmetric option is found in the header.
///
/// ## Remarks on sym_mirror
///
/// ```text
/// if i != j, read line and set a(i,j) = a(j,i) = line data
/// ```
///
/// If the matrix is symmetric, only entries in the **lower triangular** portion
/// are present in the MatrixMarket file (see reference). However, some solvers
/// (e.g., UMFPACK) require the complete sparse dataset (both off-diagonals),
/// even if the matrix is symmetric. Other solvers (e.g. Mu-M-P-S) must **not**
/// receive both off-diagonal sides when working with symmetric matrices.
/// Therefore, the user has to decide when to use the `sym_mirror` flag.
///
/// If `sym_mirror` is true, the reader will set `nnz` (number of non-zero values)
/// with twice the specified `nnz` value because we cannot know how many entries
/// are on the diagonal until the whole file is read. Nonetheless, the `SparseTriplet`
/// can be used normally by the user, since this information is internal to `SparseTriplet`.
///
/// # Output
///
/// * A SparseTriplet or an error message
///
/// # Panics
///
/// This function may panic but should not panic (please contact us if it panics).
///
/// # Example of MatrixMarket file
///
/// ```text
/// %%MatrixMarket matrix coordinate real general
/// %=================================================================================
/// %
/// % This ASCII file represents a sparse MxN matrix with L
/// % non-zeros in the following Matrix Market format:
/// %
/// % Reference: https://math.nist.gov/MatrixMarket/formats.html
/// %
/// % +----------------------------------------------+
/// % |%%MatrixMarket matrix coordinate real general | <--- header line
/// % |%                                             | <--+
/// % |% comments                                    |    |-- 0 or more comment lines
/// % |%                                             | <--+
/// % |    M  N  L                                   | <--- rows, columns, entries
/// % |    I1  J1  A(I1, J1)                         | <--+
/// % |    I2  J2  A(I2, J2)                         |    |
/// % |    I3  J3  A(I3, J3)                         |    |-- L lines
/// % |        . . .                                 |    |
/// % |    IL JL  A(IL, JL)                          | <--+
/// % +----------------------------------------------+
/// %
/// % Indices are 1-based, i.e. A(1,1) is the first element.
/// %
/// %=================================================================================
///   5  5  8
///     1     1   1.000e+00
///     2     2   1.050e+01
///     3     3   1.500e-02
///     1     4   6.000e+00
///     4     2   2.505e+02
///     4     4  -2.800e+02
///     4     5   3.332e+01
///     5     5   1.200e+01
/// ```
///
/// ## Remarks
///
/// * The first line is the **header line**
/// * The header must contain `%%MatrixMarket matrix coordinate real` followed by `general` or `symmetric` (separated by spaces)
/// * Thus, this function can only read the `coordinate` and `real` combination for now
/// * After the header line, the percentage character marks a comment line
/// * After the header line, a line with dimensions `m n nnz` must follow
/// * `m`, `n`, and `nnz` are the number of columns, rows, and non-zero values
/// * After the dimensions line, `nnz` data lines containing the triples (i,j,aij) must follow
/// * The indices start at one (1-based indices)
///
/// # Reference
///
/// <https://math.nist.gov/MatrixMarket/formats.html>
///
/// # Examples
///
/// ## Example 1 - General matrix
///
/// Given the following `simple_gen.mtx` file:
///
/// ```text
/// %%MatrixMarket matrix coordinate real general
/// 3 3  5
///  1 1  1.0
///  1 2  2.0
///  2 1  3.0
///  2 2  4.0
///  3 3  5.0
/// ```
///
/// Read the data:
///
/// ```
/// # fn main() -> Result<(), &'static str> {
/// use russell_lab::*;
/// use russell_sparse::*;
/// let filepath = "./data/matrix_market/simple_gen.mtx".to_string();
/// let trip = read_matrix_market(&filepath, false)?;
/// let (m, n) = trip.dims();
/// let mut a = Matrix::new(m, n);
/// trip.to_matrix(&mut a)?;
/// let correct = "┌       ┐\n\
///                │ 1 2 0 │\n\
///                │ 3 4 0 │\n\
///                │ 0 0 5 │\n\
///                └       ┘";
/// assert_eq!(format!("{}", a), correct);
/// # Ok(())
/// # }
/// ```
///
/// ## Example 2 - Symmetric matrix
///
/// Given the following `simple_sym.mtx` file:
///
/// ```text
/// %%MatrixMarket matrix coordinate real symmetric
/// 3 3  4
///  1 1  1.0
///  2 1  2.0
///  2 2  3.0
///  3 2  4.0
/// ```
///
/// Read the data:
///
/// ```
/// # fn main() -> Result<(), &'static str> {
/// use russell_lab::*;
/// use russell_sparse::*;
/// let filepath = "./data/matrix_market/simple_sym.mtx".to_string();
/// let trip = read_matrix_market(&filepath, true)?;
/// let (m, n) = trip.dims();
/// let mut a = Matrix::new(m, n);
/// trip.to_matrix(&mut a)?;
/// let correct = "┌       ┐\n\
///                │ 1 2 0 │\n\
///                │ 2 3 4 │\n\
///                │ 0 4 0 │\n\
///                └       ┘";
/// assert_eq!(format!("{}", a), correct);
/// # Ok(())
/// # }
/// ```
pub fn read_matrix_market(filepath: &String, sym_mirror: bool) -> Result<SparseTriplet, &'static str> {
    let input = File::open(filepath).map_err(|_| "cannot open file")?;
    let buffered = BufReader::new(input);
    let mut lines_iter = buffered.lines();

    // auxiliary data structure
    let mut data = MatrixMarketData::new();

    // read first line
    let header = match lines_iter.next() {
        Some(v) => v.unwrap(), // must panic because no error expected here
        None => return Err("file is empty"),
    };

    // parse header
    data.parse_header(&header)?;

    // read and parse dimensions
    loop {
        let line = lines_iter.next().unwrap().unwrap(); // must panic because no error expected here
        if data.parse_dimensions(&line)? {
            break;
        }
    }

    // set max number of entries
    let mut max = data.nnz;
    if data.symmetric && sym_mirror {
        max = 2 * data.nnz;
    }

    // symmetry option
    let sym = if data.symmetric {
        if sym_mirror {
            Symmetry::General
        } else {
            Symmetry::GeneralTriangular
        }
    } else {
        Symmetry::No
    };

    // allocate triplet
    let mut trip = SparseTriplet::new(data.m as usize, data.n as usize, max as usize, sym)?;

    // read and parse triples
    loop {
        match lines_iter.next() {
            Some(v) => {
                let line = v.unwrap(); // must panic because no error expected here
                if data.parse_triple(&line)? {
                    trip.put(data.i as usize, data.j as usize, data.aij);
                    if data.symmetric && sym_mirror && data.i != data.j {
                        trip.put(data.j as usize, data.i as usize, data.aij);
                    }
                }
            }
            None => break,
        }
    }

    // check data
    if data.pos != data.nnz {
        return Err("not all triples (i,j,aij) have been found");
    }

    Ok(trip)
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{read_matrix_market, MatrixMarketData};
    use crate::Symmetry;
    use russell_lab::*;

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
    fn read_matrix_market_handle_wrong_files() -> Result<(), &'static str> {
        assert_eq!(
            read_matrix_market(&String::from("__wrong__"), false).err(),
            Some("cannot open file")
        );
        assert_eq!(
            read_matrix_market(&String::from("./data/matrix_market/bad_empty_file.mtx"), false).err(),
            Some("file is empty")
        );
        assert_eq!(
            read_matrix_market(&String::from("./data/matrix_market/bad_missing_data.mtx"), false).err(),
            Some("not all triples (i,j,aij) have been found")
        );
        assert_eq!(
            read_matrix_market(&String::from("./data/matrix_market/bad_many_lines.mtx"), false).err(),
            Some("there are more (i,j,aij) triples than specified")
        );
        Ok(())
    }

    #[test]
    fn read_matrix_market_works() -> Result<(), &'static str> {
        let filepath = "./data/matrix_market/ok1.mtx".to_string();
        let trip = read_matrix_market(&filepath, false)?;
        assert!(matches!(trip.symmetry, Symmetry::No));
        assert_eq!((trip.nrow, trip.ncol, trip.pos, trip.max), (5, 5, 12, 12));
        assert_eq!(trip.indices_i, &[0, 1, 0, 2, 4, 1, 2, 3, 4, 2, 1, 4]);
        assert_eq!(trip.indices_j, &[0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 4, 4]);
        assert_eq!(
            trip.values_aij,
            &[2.0, 3.0, 3.0, -1.0, 4.0, 4.0, -3.0, 1.0, 2.0, 2.0, 6.0, 1.0]
        );
        Ok(())
    }

    #[test]
    fn read_matrix_market_sym_triangle_works() -> Result<(), &'static str> {
        let filepath = "./data/matrix_market/ok2.mtx".to_string();
        let trip = read_matrix_market(&filepath, false)?;
        assert!(matches!(trip.symmetry, Symmetry::GeneralTriangular));
        assert_eq!((trip.nrow, trip.ncol, trip.pos, trip.max), (5, 5, 15, 15));
        assert_eq!(trip.indices_i, &[0, 1, 2, 3, 4, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3]);
        assert_eq!(trip.indices_j, &[0, 1, 2, 3, 4, 1, 2, 3, 4, 2, 3, 4, 3, 4, 4]);
        assert_eq!(
            trip.values_aij,
            &[2.0, 2.0, 9.0, 7.0, 8.0, 1.0, 1.0, 3.0, 2.0, 2.0, 1.0, 1.0, 1.0, 5.0, 1.0],
        );
        Ok(())
    }

    #[test]
    fn read_matrix_market_sym_mirror_works() -> Result<(), &'static str> {
        let filepath = "./data/matrix_market/ok3.mtx".to_string();
        let trip = read_matrix_market(&filepath, true)?;
        assert!(matches!(trip.symmetry, Symmetry::General));
        assert_eq!((trip.nrow, trip.ncol, trip.pos, trip.max), (5, 5, 11, 14));
        assert_eq!(trip.indices_i, &[0, 1, 0, 2, 1, 3, 2, 3, 4, 1, 4, 0, 0, 0]);
        assert_eq!(trip.indices_j, &[0, 0, 1, 1, 2, 2, 3, 3, 1, 4, 4, 0, 0, 0]);
        assert_eq!(
            trip.values_aij,
            &[2.0, 3.0, 3.0, -1.0, -1.0, 2.0, 2.0, 3.0, 6.0, 6.0, 1.0, 0.0, 0.0, 0.0]
        );
        let mut a = Matrix::new(5, 5);
        trip.to_matrix(&mut a)?;
        let correct = "┌                ┐\n\
                            │  2  3  0  0  0 │\n\
                            │  3  0 -1  0  6 │\n\
                            │  0 -1  0  2  0 │\n\
                            │  0  0  2  3  0 │\n\
                            │  0  6  0  0  1 │\n\
                            └                ┘";
        assert_eq!(format!("{}", a), correct);
        Ok(())
    }
}
