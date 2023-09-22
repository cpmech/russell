use super::{CooMatrix, Layout, MMsymOption};
use crate::StrError;
use std::ffi::OsStr;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

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
    fn parse_header(&mut self, line: &str) -> Result<(), StrError> {
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
    fn parse_dimensions(&mut self, line: &str) -> Result<bool, StrError> {
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
    fn parse_triple(&mut self, line: &str) -> Result<bool, StrError> {
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

        self.i -= 1; // MatrixMarket is one-based, so make it zero-based here
        self.j -= 1;

        if self.i < 0 || self.i >= self.m || self.j < 0 || self.j >= self.n {
            return Err("found invalid indices");
        }

        self.pos += 1; // next position

        Ok(true) // returns true == parsed
    }
}

/// Reads a MatrixMarket file into a CooMatrix
///
/// # Input
///
/// * `full_path` -- may be a String, &str, or Path
/// * `symmetric_handling` -- Options to handle symmetric matrices
///
/// # Output
///
/// * `coo` -- the CooMatrix
/// * `symmetric` -- whether MatrixMarket is flagged as symmetric or not
///
/// ## Remarks on symmetric matrices
///
/// If the matrix is symmetric, only entries in the **lower triangular** portion
/// are present in the MatrixMarket file (see reference). Thus, the `symmetric_handling`
/// may be used to:
///
/// 1. Leave the data as it is, i.e., return a Lower Triangular CooMatrix (for MUMPS solver)
/// 2. Swap the lower triangle with the upper triangle, i.e., return an Upper Triangular CooMatrix (for Intel DSS solver)
/// 3. Duplicate the data to make a full matrix, i.e., return a Full CooMatrix (for UMFPACK solver)
///
/// # Panics
///
/// This function may panic but should not panic (please contact us if it panics :-).
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
/// Given the following `ok_simple_general.mtx` file:
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
/// use russell_lab::Matrix;
/// use russell_sparse::{read_matrix_market, Layout, MMsymOption, StrError};
///
/// fn main() -> Result<(), StrError> {
///     let filepath = "./data/matrix_market/ok_simple_general.mtx".to_string();
///     let (coo, sym) = read_matrix_market(&filepath, MMsymOption::LeaveAsLower)?;
///     let mut a = Matrix::new(coo.nrow, coo.ncol);
///     coo.to_matrix(&mut a)?;
///     let correct = "┌       ┐\n\
///                    │ 1 2 0 │\n\
///                    │ 3 4 0 │\n\
///                    │ 0 0 5 │\n\
///                    └       ┘";
///     assert!(!sym);
///     assert_eq!(coo.layout, Layout::Full);
///     assert_eq!(format!("{}", a), correct);
///     Ok(())
/// }
/// ```
///
/// ## Example 2 - Symmetric matrix
///
/// Given the following `ok_simple_symmetric.mtx` file:
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
/// use russell_lab::Matrix;
/// use russell_sparse::{read_matrix_market, Layout, MMsymOption, StrError};
///
/// fn main() -> Result<(), StrError> {
///     let filepath = "./data/matrix_market/ok_simple_symmetric.mtx".to_string();
///     let (coo, sym) = read_matrix_market(&filepath, MMsymOption::LeaveAsLower)?;
///     let mut a = Matrix::new(coo.nrow, coo.ncol);
///     coo.to_matrix(&mut a)?;
///     let correct = "┌       ┐\n\
///                    │ 1 2 0 │\n\
///                    │ 2 3 4 │\n\
///                    │ 0 4 0 │\n\
///                    └       ┘";
///     assert!(sym);
///     assert_eq!(coo.layout, Layout::Lower);
///     assert_eq!(format!("{}", a), correct);
///     Ok(())
/// }
/// ```
pub fn read_matrix_market<P>(full_path: &P, symmetric_handling: MMsymOption) -> Result<(CooMatrix, bool), StrError>
where
    P: AsRef<OsStr> + ?Sized,
{
    let path = Path::new(full_path).to_path_buf();
    let input = File::open(path).map_err(|_| "cannot open file")?;
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

    // specify the CooMatrix layout
    let layout = if data.symmetric {
        if data.m != data.n {
            return Err("MatrixMarket data is invalid: the number of rows must be equal the number of columns for symmetric matrices");
        }
        match symmetric_handling {
            MMsymOption::LeaveAsLower => Layout::Lower,
            MMsymOption::SwapToUpper => Layout::Upper,
            MMsymOption::MakeItFull => Layout::Full,
        }
    } else {
        Layout::Full
    };

    // set max number of entries
    let mut max = data.nnz;
    if data.symmetric && symmetric_handling == MMsymOption::MakeItFull {
        max = 2 * data.nnz;
    }

    // allocate triplet
    let mut coo = CooMatrix::new(layout, data.m as usize, data.n as usize, max as usize).unwrap();

    // read and parse triples
    loop {
        match lines_iter.next() {
            Some(v) => {
                let line = v.unwrap(); // must panic because no error expected here
                if data.parse_triple(&line)? {
                    if data.symmetric {
                        match symmetric_handling {
                            MMsymOption::LeaveAsLower => {
                                coo.put(data.i as usize, data.j as usize, data.aij).unwrap();
                            }
                            MMsymOption::SwapToUpper => {
                                coo.put(data.j as usize, data.i as usize, data.aij).unwrap();
                            }
                            MMsymOption::MakeItFull => {
                                coo.put(data.i as usize, data.j as usize, data.aij).unwrap();
                                if data.i != data.j {
                                    coo.put(data.j as usize, data.i as usize, data.aij).unwrap();
                                }
                            }
                        }
                    } else {
                        coo.put(data.i as usize, data.j as usize, data.aij).unwrap();
                    };
                }
            }
            None => break,
        }
    }

    // check data
    if data.pos != data.nnz {
        return Err("not all triples (i,j,aij) have been found");
    }

    Ok((coo, data.symmetric))
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{read_matrix_market, MatrixMarketData};
    use crate::{Layout, MMsymOption};
    use russell_lab::Matrix;

    #[test]
    fn parse_header_captures_errors() {
        let mut data = MatrixMarketData::new();

        assert_eq!(
            data.parse_header("  \n"),
            Err("cannot find the keyword %%MatrixMarket on the first line")
        );
        assert_eq!(
            data.parse_header("MatrixMarket  "),
            Err("the header (first line) must start with %%MatrixMarket"),
        );

        assert_eq!(
            data.parse_header("  %%MatrixMarket"),
            Err("cannot find the first option in the header line"),
        );
        assert_eq!(
            data.parse_header("%%MatrixMarket   wrong"),
            Err("after %%MatrixMarket, the first option must be \"matrix\""),
        );

        assert_eq!(
            data.parse_header("%%MatrixMarket matrix  "),
            Err("cannot find the second option in the header line"),
        );
        assert_eq!(
            data.parse_header("%%MatrixMarket   matrix wrong"),
            Err("after %%MatrixMarket, the second option must be \"coordinate\""),
        );

        assert_eq!(
            data.parse_header("%%MatrixMarket matrix  coordinate"),
            Err("cannot find the third option in the header line"),
        );
        assert_eq!(
            data.parse_header("%%MatrixMarket matrix    coordinate  wrong"),
            Err("after %%MatrixMarket, the third option must be \"real\""),
        );

        assert_eq!(
            data.parse_header("%%MatrixMarket  matrix coordinate real"),
            Err("cannot find the fourth option in the header line"),
        );
        assert_eq!(
            data.parse_header("  %%MatrixMarket matrix coordinate real wrong"),
            Err("after %%MatrixMarket, the fourth option must be either \"general\" or \"symmetric\""),
        );
    }

    #[test]
    fn parse_dimensions_captures_errors() {
        let mut data = MatrixMarketData::new();

        assert_eq!(
            data.parse_dimensions(" wrong \n").err(),
            Some("cannot parse number of rows")
        );

        assert_eq!(
            data.parse_dimensions(" 1 \n").err(),
            Some("cannot read number of columns")
        );
        assert_eq!(
            data.parse_dimensions(" 1 wrong").err(),
            Some("cannot parse number of columns")
        );

        assert_eq!(
            data.parse_dimensions(" 1 1   \n").err(),
            Some("cannot read number of non-zeros")
        );
        assert_eq!(
            data.parse_dimensions(" 1 1  wrong").err(),
            Some("cannot parse number of non-zeros")
        );

        assert_eq!(
            data.parse_dimensions(" 0 1  1").err(),
            Some("found invalid (zero or negative) dimensions")
        );
        assert_eq!(
            data.parse_dimensions(" 1 0  1").err(),
            Some("found invalid (zero or negative) dimensions")
        );
        assert_eq!(
            data.parse_dimensions(" 1 1  0").err(),
            Some("found invalid (zero or negative) dimensions")
        );
    }

    #[test]
    fn parse_triple_captures_errors() {
        let mut data = MatrixMarketData::new();
        data.m = 2;
        data.n = 2;
        data.nnz = 1;

        assert_eq!(data.parse_triple(" wrong \n").err(), Some("cannot parse index i"));

        assert_eq!(data.parse_triple(" 1 \n").err(), Some("cannot read index j"));
        assert_eq!(data.parse_triple(" 1 wrong").err(), Some("cannot parse index j"));

        assert_eq!(data.parse_triple(" 1 1   \n").err(), Some("cannot read value aij"));
        assert_eq!(data.parse_triple(" 1 1  wrong").err(), Some("cannot parse value aij"));

        assert_eq!(data.parse_triple(" 0 1  1").err(), Some("found invalid indices"));
        assert_eq!(data.parse_triple(" 3 1  1").err(), Some("found invalid indices"));
        assert_eq!(data.parse_triple(" 1 0  1").err(), Some("found invalid indices"));
        assert_eq!(data.parse_triple(" 1 3  1").err(), Some("found invalid indices"));
    }

    #[test]
    fn read_matrix_market_handle_wrong_files() {
        let h = MMsymOption::LeaveAsLower;
        assert_eq!(read_matrix_market("__wrong__", h).err(), Some("cannot open file"));
        assert_eq!(
            read_matrix_market("./data/matrix_market/bad_empty_file.mtx", h).err(),
            Some("file is empty")
        );
        assert_eq!(
            read_matrix_market("./data/matrix_market/bad_wrong_header.mtx", h).err(),
            Some("after %%MatrixMarket, the first option must be \"matrix\"")
        );
        assert_eq!(
            read_matrix_market("./data/matrix_market/bad_wrong_dims.mtx", h).err(),
            Some("found invalid (zero or negative) dimensions")
        );
        assert_eq!(
            read_matrix_market("./data/matrix_market/bad_missing_data.mtx", h).err(),
            Some("not all triples (i,j,aij) have been found")
        );
        assert_eq!(
            read_matrix_market("./data/matrix_market/bad_many_lines.mtx", h).err(),
            Some("there are more (i,j,aij) triples than specified")
        );
        assert_eq!(
            read_matrix_market("./data/matrix_market/bad_symmetric_rectangular.mtx", h).err(),
            Some("MatrixMarket data is invalid: the number of rows must be equal the number of columns for symmetric matrices")
        );
    }

    #[test]
    fn read_matrix_market_works() {
        let h = MMsymOption::LeaveAsLower;
        let filepath = "./data/matrix_market/ok_general.mtx".to_string();
        let (coo, sym) = read_matrix_market(&filepath, h).unwrap();
        assert!(!sym);
        assert_eq!(coo.layout, Layout::Full);
        assert_eq!((coo.nrow, coo.ncol, coo.pos, coo.max), (5, 5, 12, 12));
        assert_eq!(coo.indices_i, &[0, 1, 0, 2, 4, 1, 2, 3, 4, 2, 1, 4]);
        assert_eq!(coo.indices_j, &[0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 4, 4]);
        assert_eq!(
            coo.values_aij,
            &[2.0, 3.0, 3.0, -1.0, 4.0, 4.0, -3.0, 1.0, 2.0, 2.0, 6.0, 1.0]
        );
    }

    #[test]
    fn read_matrix_market_symmetric_lower_works() {
        let h = MMsymOption::LeaveAsLower;
        let filepath = "./data/matrix_market/ok_symmetric.mtx".to_string();
        let (coo, sym) = read_matrix_market(&filepath, h).unwrap();
        assert!(sym);
        assert_eq!(coo.layout, Layout::Lower);
        assert_eq!((coo.nrow, coo.ncol, coo.pos, coo.max), (5, 5, 15, 15));
        assert_eq!(coo.indices_i, &[0, 1, 2, 3, 4, 1, 2, 3, 4, 2, 3, 4, 3, 4, 4]);
        assert_eq!(coo.indices_j, &[0, 1, 2, 3, 4, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3]);
        assert_eq!(
            coo.values_aij,
            &[2.0, 2.0, 9.0, 7.0, 8.0, 1.0, 1.0, 3.0, 2.0, 2.0, 1.0, 1.0, 1.0, 5.0, 1.0],
        );
    }

    #[test]
    fn read_matrix_market_symmetric_upper_works() {
        let h = MMsymOption::SwapToUpper;
        let filepath = "./data/matrix_market/ok_symmetric.mtx".to_string();
        let (coo, sym) = read_matrix_market(&filepath, h).unwrap();
        assert!(sym);
        assert_eq!(coo.layout, Layout::Upper);
        assert_eq!((coo.nrow, coo.ncol, coo.pos, coo.max), (5, 5, 15, 15));
        assert_eq!(coo.indices_i, &[0, 1, 2, 3, 4, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3]);
        assert_eq!(coo.indices_j, &[0, 1, 2, 3, 4, 1, 2, 3, 4, 2, 3, 4, 3, 4, 4]);
        assert_eq!(
            coo.values_aij,
            &[2.0, 2.0, 9.0, 7.0, 8.0, 1.0, 1.0, 3.0, 2.0, 2.0, 1.0, 1.0, 1.0, 5.0, 1.0],
        );
    }

    #[test]
    fn read_matrix_market_symmetric_to_full_works() {
        let h = MMsymOption::MakeItFull;
        let filepath = "./data/matrix_market/ok_symmetric_small.mtx".to_string();
        let (coo, sym) = read_matrix_market(&filepath, h).unwrap();
        assert!(sym);
        assert_eq!(coo.layout, Layout::Full);
        assert_eq!((coo.nrow, coo.ncol, coo.pos, coo.max), (5, 5, 11, 14));
        assert_eq!(coo.indices_i, &[0, 1, 0, 2, 1, 3, 2, 3, 4, 1, 4, 0, 0, 0]);
        assert_eq!(coo.indices_j, &[0, 0, 1, 1, 2, 2, 3, 3, 1, 4, 4, 0, 0, 0]);
        assert_eq!(
            coo.values_aij,
            &[2.0, 3.0, 3.0, -1.0, -1.0, 2.0, 2.0, 3.0, 6.0, 6.0, 1.0, 0.0, 0.0, 0.0]
        );
        let mut a = Matrix::new(5, 5);
        coo.to_matrix(&mut a).unwrap();
        let correct = "┌                ┐\n\
                       │  2  3  0  0  0 │\n\
                       │  3  0 -1  0  6 │\n\
                       │  0 -1  0  2  0 │\n\
                       │  0  0  2  3  0 │\n\
                       │  0  6  0  0  1 │\n\
                       └                ┘";
        assert_eq!(format!("{}", a), correct);
    }
}
