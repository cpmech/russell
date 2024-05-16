use super::{CooMatrix, MMsym, Sym};
use crate::{ComplexCooMatrix, StrError};
use russell_lab::{cpx, Complex64};
use std::ffi::OsStr;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

struct MatrixMarketData {
    // header
    complex: bool,
    symmetric: bool,

    // dimensions
    m: i32,   // number of rows
    n: i32,   // number of columns
    nnz: i32, // number of non-zeros

    // current triple
    i: i32,   // current i-index
    j: i32,   // current j-index
    aij: f64, // current aij-value (real part if complex)
    bij: f64, // current imaginary part of the aij-value (if complex)
    pos: i32, // current position in the list of triples
}

impl MatrixMarketData {
    fn new() -> Self {
        MatrixMarketData {
            complex: false,
            symmetric: false,
            m: 0,
            n: 0,
            nnz: 0,
            i: 0,
            j: 0,
            aij: 0.0,
            bij: 0.0,
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
            Some(v) => match v {
                "real" => self.complex = false,
                "complex" => self.complex = true,
                _ => return Err("after %%MatrixMarket, the third option must be \"real\" or \"complex\""),
            },
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
    fn parse_values(&mut self, line: &str) -> Result<bool, StrError> {
        let maybe_data = line.trim_start().trim_end_matches("\n");
        if maybe_data.starts_with("%") || maybe_data == "" {
            return Ok(false); // ignore comments or empty lines
        }

        if self.pos == self.nnz {
            return Err("there are more values than specified");
        }

        let mut data = maybe_data.split_whitespace();

        self.i = data
            .next()
            .unwrap() // must panic because no error expected here
            .parse()
            .map_err(|_| "cannot parse i")?;

        match data.next() {
            Some(v) => self.j = v.parse().map_err(|_| "cannot parse j")?,
            None => return Err("cannot read j"),
        };

        match data.next() {
            Some(v) => self.aij = v.parse().map_err(|_| "cannot parse aij")?,
            None => return Err("cannot read aij"),
        };

        if self.complex {
            match data.next() {
                Some(v) => self.bij = v.parse().map_err(|_| "cannot parse bij")?,
                None => return Err("cannot read bij"),
            };
        }

        self.i -= 1; // MatrixMarket is one-based, so make it zero-based here
        self.j -= 1;

        if self.i < 0 || self.i >= self.m || self.j < 0 || self.j >= self.n {
            return Err("found an invalid index");
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
/// Returns either a [CooMatrix] or a [ComplexCooMatrix]. One of each will be `Some` while
/// the other will be `None`.
///
/// ## Remarks on symmetric matrices
///
/// If the matrix is symmetric, only entries in the **lower triangular** portion
/// are present in the MatrixMarket file (see reference). Thus, the `symmetric_handling`
/// may be used to:
///
/// 1. Leave the data as it is, i.e., return a lower triangular matrix (e.g., for MUMPS solver)
/// 2. Swap the lower triangle with the upper triangle, i.e., return an upper triangular matrix
/// 3. Duplicate the data to make a full matrix, i.e., return a full matrix (e.g., for UMFPACK solver)
///
/// # Examples of MatrixMarket file
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
/// % +----------------------------------------------------------------------+
/// % |%%MatrixMarket matrix coordinate {real, complex} {general, symmetric} | <--- header line
/// % |%                                                                     | <--+
/// % |% comments                                                            |    |-- 0 or more comment lines
/// % |%                                                                     | <--+
/// % |    M  N  L                                                           | <--- rows, columns, entries
/// % |    I1  J1  A(I1, J1) {B(I1, J1)}                                     | <--+
/// % |    I2  J2  A(I2, J2) {B(I2, J2)}                                     |    |
/// % |    I3  J3  A(I3, J3) {B(I3, J3)}                                     |    |-- L lines
/// % |        . . .                                                         |    |
/// % |    IL  JL  A(IL, JL) {B(IL, JL)}                                     | <--+
/// % +----------------------------------------------------------------------+
/// %
/// % Indices are 1-based, i.e. A(1,1) is the first element. The values within
/// % curly braces are optional. B(I,J) is the imaginary part of the value and
/// % is required for the complex case.
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
/// * The first line is the **header line**, in the following format:
///     * `%%MatrixMarket matrix coordinate {real, complex} {general, symmetric}` where only
///       one option within curly braces is present
/// * After the header line, the percentage character begins a comment line
/// * After the header line, a line with dimensions `m n nnz` must follow
/// * `m`, `n`, and `nnz` are the number of columns, rows, and non-zero values
/// * After the dimensions line, `nnz` data lines containing either `i j aij` (real) or `i j aij bij` (complex)
///   must follow. For the complex case, `aij` is the real part and `bij` is the imaginary part
/// * The indices start at one (1-based indices)
///
/// # Reference
///
/// <https://math.nist.gov/MatrixMarket/formats.html>
///
/// # Examples
///
/// ## Examples 1 - General matrix
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
/// use russell_sparse::prelude::*;
/// use russell_sparse::StrError;
///
/// fn main() -> Result<(), StrError> {
///     let name = "./data/matrix_market/ok_simple_general.mtx";
///     let (coo_real, _) = read_matrix_market(name, MMsym::LeaveAsLower)?;
///     let coo = coo_real.unwrap();
///     let (nrow, ncol, nnz, sym) = coo.get_info();
///     assert_eq!(nrow, 3);
///     assert_eq!(ncol, 3);
///     assert_eq!(nnz, 5);
///     assert_eq!(sym, Sym::No);
///     let a = coo.as_dense();
///     let correct = "┌       ┐\n\
///                    │ 1 2 0 │\n\
///                    │ 3 4 0 │\n\
///                    │ 0 0 5 │\n\
///                    └       ┘";
///     assert_eq!(format!("{}", a), correct);
///     Ok(())
/// }
/// ```
///
/// ## Examples 2 - Symmetric matrix
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
/// use russell_sparse::prelude::*;
/// use russell_sparse::StrError;
///
/// fn main() -> Result<(), StrError> {
///     let name = "./data/matrix_market/ok_simple_symmetric.mtx";
///     let (coo_real, _) = read_matrix_market(name, MMsym::LeaveAsLower)?;
///     let coo = coo_real.unwrap();
///     let (nrow, ncol, nnz, sym) = coo.get_info();
///     assert_eq!(nrow, 3);
///     assert_eq!(ncol, 3);
///     assert_eq!(nnz, 4);
///     assert_eq!(sym, Sym::YesLower);
///     let a = coo.as_dense();
///     let correct = "┌       ┐\n\
///                    │ 1 2 0 │\n\
///                    │ 2 3 4 │\n\
///                    │ 0 4 0 │\n\
///                    └       ┘";
///     assert_eq!(format!("{}", a), correct);
///     Ok(())
/// }
/// ```
pub fn read_matrix_market<P>(
    full_path: &P,
    symmetric_handling: MMsym,
) -> Result<(Option<CooMatrix>, Option<ComplexCooMatrix>), StrError>
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
        None => return Err("the file is empty"),
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

    // symmetric type
    let sym = if data.symmetric {
        if data.m != data.n {
            return Err("MatrixMarket data is invalid: the number of rows must equal the number of columns for symmetric matrices");
        }
        match symmetric_handling {
            MMsym::LeaveAsLower => Sym::YesLower,
            MMsym::SwapToUpper => Sym::YesUpper,
            MMsym::MakeItFull => Sym::YesFull,
        }
    } else {
        Sym::No
    };

    // set max number of entries
    let mut max = data.nnz;
    if data.symmetric && symmetric_handling == MMsym::MakeItFull {
        max = 2 * data.nnz;
    }

    // read and parse values
    if data.complex {
        let mut coo = ComplexCooMatrix::new(data.m as usize, data.n as usize, max as usize, sym).unwrap();
        loop {
            match lines_iter.next() {
                Some(v) => {
                    let line = v.unwrap(); // must panic because no error expected here
                    if data.parse_values(&line)? {
                        if data.symmetric {
                            match symmetric_handling {
                                MMsym::LeaveAsLower => {
                                    coo.put(data.i as usize, data.j as usize, cpx!(data.aij, data.bij))
                                        .unwrap();
                                }
                                MMsym::SwapToUpper => {
                                    coo.put(data.j as usize, data.i as usize, cpx!(data.aij, data.bij))
                                        .unwrap();
                                }
                                MMsym::MakeItFull => {
                                    coo.put(data.i as usize, data.j as usize, cpx!(data.aij, data.bij))
                                        .unwrap();
                                    if data.i != data.j {
                                        coo.put(data.j as usize, data.i as usize, cpx!(data.aij, data.bij))
                                            .unwrap();
                                    }
                                }
                            }
                        } else {
                            coo.put(data.i as usize, data.j as usize, cpx!(data.aij, data.bij))
                                .unwrap();
                        };
                    }
                }
                None => break,
            }
        }
        if data.pos != data.nnz {
            return Err("not all values have been found");
        }
        Ok((None, Some(coo)))
    } else {
        let mut coo = CooMatrix::new(data.m as usize, data.n as usize, max as usize, sym).unwrap();
        loop {
            match lines_iter.next() {
                Some(v) => {
                    let line = v.unwrap(); // must panic because no error expected here
                    if data.parse_values(&line)? {
                        if data.symmetric {
                            match symmetric_handling {
                                MMsym::LeaveAsLower => {
                                    coo.put(data.i as usize, data.j as usize, data.aij).unwrap();
                                }
                                MMsym::SwapToUpper => {
                                    coo.put(data.j as usize, data.i as usize, data.aij).unwrap();
                                }
                                MMsym::MakeItFull => {
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
        if data.pos != data.nnz {
            return Err("not all values have been found");
        }
        Ok((Some(coo), None))
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{read_matrix_market, MatrixMarketData};
    use crate::{MMsym, Sym};
    use russell_lab::{cpx, Complex64, Matrix};

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
            Err("after %%MatrixMarket, the third option must be \"real\" or \"complex\""),
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
    fn parse_values_captures_errors() {
        let mut data = MatrixMarketData::new();
        data.m = 2;
        data.n = 2;
        data.nnz = 1;

        assert_eq!(data.parse_values(" wrong \n").err(), Some("cannot parse i"));

        assert_eq!(data.parse_values(" 1 \n").err(), Some("cannot read j"));
        assert_eq!(data.parse_values(" 1 wrong").err(), Some("cannot parse j"));

        assert_eq!(data.parse_values(" 1 1   \n").err(), Some("cannot read aij"));
        assert_eq!(data.parse_values(" 1 1  wrong").err(), Some("cannot parse aij"));

        assert_eq!(data.parse_values(" 0 1  1").err(), Some("found an invalid index"));
        assert_eq!(data.parse_values(" 3 1  1").err(), Some("found an invalid index"));
        assert_eq!(data.parse_values(" 1 0  1").err(), Some("found an invalid index"));
        assert_eq!(data.parse_values(" 1 3  1").err(), Some("found an invalid index"));

        let mut data = MatrixMarketData::new();
        data.complex = true;
        data.m = 2;
        data.n = 2;
        data.nnz = 1;
        assert_eq!(data.parse_values(" 1 1  1").err(), Some("cannot read bij"));
        assert_eq!(data.parse_values(" 1 1  1 wrong").err(), Some("cannot parse bij"));
    }

    #[test]
    fn read_matrix_market_handle_wrong_files() {
        let h = MMsym::LeaveAsLower;
        assert_eq!(read_matrix_market("__wrong__", h).err(), Some("cannot open file"));
        assert_eq!(
            read_matrix_market("./data/matrix_market/bad_empty_file.mtx", h).err(),
            Some("the file is empty")
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
            read_matrix_market("./data/matrix_market/bad_wrong_dims_complex.mtx", h).err(),
            Some("found invalid (zero or negative) dimensions")
        );
        assert_eq!(
            read_matrix_market("./data/matrix_market/bad_missing_data.mtx", h).err(),
            Some("not all values have been found")
        );
        assert_eq!(
            read_matrix_market("./data/matrix_market/bad_missing_data_complex.mtx", h).err(),
            Some("not all values have been found")
        );
        assert_eq!(
            read_matrix_market("./data/matrix_market/bad_many_lines.mtx", h).err(),
            Some("there are more values than specified")
        );
        assert_eq!(
            read_matrix_market("./data/matrix_market/bad_many_lines_complex.mtx", h).err(),
            Some("there are more values than specified")
        );
        assert_eq!(
            read_matrix_market("./data/matrix_market/bad_symmetric_rectangular.mtx", h).err(),
            Some("MatrixMarket data is invalid: the number of rows must equal the number of columns for symmetric matrices")
        );
        assert_eq!(
            read_matrix_market("./data/matrix_market/bad_symmetric_rectangular_complex.mtx", h).err(),
            Some("MatrixMarket data is invalid: the number of rows must equal the number of columns for symmetric matrices")
        );
    }

    #[test]
    fn read_matrix_market_works() {
        let h = MMsym::LeaveAsLower;
        let filepath = "./data/matrix_market/ok_general.mtx".to_string();
        let (coo_real, coo_cpx) = read_matrix_market(&filepath, h).unwrap();
        assert!(coo_cpx.is_none());
        let coo = coo_real.unwrap();
        assert_eq!(coo.symmetric, Sym::No);
        assert_eq!((coo.nrow, coo.ncol, coo.nnz, coo.max_nnz), (5, 5, 12, 12));
        assert_eq!(coo.indices_i, &[0, 1, 0, 2, 4, 1, 2, 3, 4, 2, 1, 4]);
        assert_eq!(coo.indices_j, &[0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 4, 4]);
        assert_eq!(
            coo.values,
            &[2.0, 3.0, 3.0, -1.0, 4.0, 4.0, -3.0, 1.0, 2.0, 2.0, 6.0, 1.0]
        );
    }

    #[test]
    fn read_matrix_market_complex_works() {
        let h = MMsym::LeaveAsLower;
        let filepath = "./data/matrix_market/ok_complex_general.mtx".to_string();
        let (coo_real, coo_cpx) = read_matrix_market(&filepath, h).unwrap();
        assert!(coo_real.is_none());
        let coo = coo_cpx.unwrap();
        assert_eq!(coo.symmetric, Sym::No);
        assert_eq!((coo.nrow, coo.ncol, coo.nnz, coo.max_nnz), (5, 5, 12, 12));
        assert_eq!(coo.indices_i, &[0, 1, 0, 2, 4, 1, 2, 3, 4, 2, 1, 4]);
        assert_eq!(coo.indices_j, &[0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 4, 4]);
        assert_eq!(
            coo.values,
            &[
                cpx!(2.0, -1.0),
                cpx!(3.0, -8.0),
                cpx!(3.0, 80.0),
                cpx!(-1.0, 30.0),
                cpx!(4.0, 33.0),
                cpx!(4.0, 60.0),
                cpx!(-3.0, 6.0),
                cpx!(1.0, 8.0),
                cpx!(2.0, 3.0),
                cpx!(2.0, 1.0),
                cpx!(6.0, 9.0),
                cpx!(1.0, -2.0)
            ]
        );
    }

    #[test]
    fn read_matrix_market_symmetric_lower_works() {
        let h = MMsym::LeaveAsLower;
        let filepath = "./data/matrix_market/ok_symmetric.mtx".to_string();
        let (coo_real, coo_cpx) = read_matrix_market(&filepath, h).unwrap();
        assert!(coo_cpx.is_none());
        let coo = coo_real.unwrap();
        assert_eq!(coo.symmetric, Sym::YesLower);
        assert_eq!((coo.nrow, coo.ncol, coo.nnz, coo.max_nnz), (5, 5, 15, 15));
        assert_eq!(coo.indices_i, &[0, 1, 2, 3, 4, 1, 2, 3, 4, 2, 3, 4, 3, 4, 4]);
        assert_eq!(coo.indices_j, &[0, 1, 2, 3, 4, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3]);
        assert_eq!(
            coo.values,
            &[2.0, 2.0, 9.0, 7.0, 8.0, 1.0, 1.0, 3.0, 2.0, 2.0, 1.0, 1.0, 1.0, 5.0, 1.0],
        );
    }

    #[test]
    fn read_matrix_market_complex_symmetric_lower_works() {
        let h = MMsym::LeaveAsLower;
        let filepath = "./data/matrix_market/ok_complex_symmetric_small.mtx".to_string();
        let (coo_real, coo_cpx) = read_matrix_market(&filepath, h).unwrap();
        assert!(coo_real.is_none());
        let coo = coo_cpx.unwrap();
        assert_eq!(coo.symmetric, Sym::YesLower);
        assert_eq!((coo.nrow, coo.ncol, coo.nnz, coo.max_nnz), (5, 5, 7, 7));
        assert_eq!(coo.indices_i, &[0, 1, 2, 3, 3, 4, 4]);
        assert_eq!(coo.indices_j, &[0, 0, 1, 2, 3, 1, 4]);
        assert_eq!(
            coo.values,
            &[
                cpx!(2.0, 1.0),
                cpx!(3.0, 2.0),
                cpx!(-1.0, 3.0),
                cpx!(2.0, 4.0),
                cpx!(3.0, 5.0),
                cpx!(6.0, 6.0),
                cpx!(1.0, 7.0),
            ]
        );
    }

    #[test]
    fn read_matrix_market_symmetric_upper_works() {
        let h = MMsym::SwapToUpper;
        let filepath = "./data/matrix_market/ok_symmetric.mtx".to_string();
        let (maybe_coo, _) = read_matrix_market(&filepath, h).unwrap();
        let coo = maybe_coo.unwrap();
        assert_eq!(coo.symmetric, Sym::YesUpper);
        assert_eq!((coo.nrow, coo.ncol, coo.nnz, coo.max_nnz), (5, 5, 15, 15));
        assert_eq!(coo.indices_i, &[0, 1, 2, 3, 4, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3]);
        assert_eq!(coo.indices_j, &[0, 1, 2, 3, 4, 1, 2, 3, 4, 2, 3, 4, 3, 4, 4]);
        assert_eq!(
            coo.values,
            &[2.0, 2.0, 9.0, 7.0, 8.0, 1.0, 1.0, 3.0, 2.0, 2.0, 1.0, 1.0, 1.0, 5.0, 1.0],
        );
    }

    #[test]
    fn read_matrix_market_complex_symmetric_upper_works() {
        let h = MMsym::SwapToUpper;
        let filepath = "./data/matrix_market/ok_complex_symmetric_small.mtx".to_string();
        let (coo_real, coo_cpx) = read_matrix_market(&filepath, h).unwrap();
        assert!(coo_real.is_none());
        let coo = coo_cpx.unwrap();
        assert_eq!(coo.symmetric, Sym::YesUpper);
        assert_eq!((coo.nrow, coo.ncol, coo.nnz, coo.max_nnz), (5, 5, 7, 7));
        assert_eq!(coo.indices_i, &[0, 0, 1, 2, 3, 1, 4]);
        assert_eq!(coo.indices_j, &[0, 1, 2, 3, 3, 4, 4]);
        assert_eq!(
            coo.values,
            &[
                cpx!(2.0, 1.0),
                cpx!(3.0, 2.0),
                cpx!(-1.0, 3.0),
                cpx!(2.0, 4.0),
                cpx!(3.0, 5.0),
                cpx!(6.0, 6.0),
                cpx!(1.0, 7.0),
            ]
        );
    }

    #[test]
    fn read_matrix_market_symmetric_to_full_works() {
        let h = MMsym::MakeItFull;
        let filepath = "./data/matrix_market/ok_symmetric_small.mtx".to_string();
        let (maybe_coo, _) = read_matrix_market(&filepath, h).unwrap();
        let coo = maybe_coo.unwrap();
        assert_eq!(coo.symmetric, Sym::YesFull);
        assert_eq!((coo.nrow, coo.ncol, coo.nnz, coo.max_nnz), (5, 5, 11, 14));
        assert_eq!(coo.indices_i, &[0, 1, 0, 2, 1, 3, 2, 3, 4, 1, 4, 0, 0, 0]);
        assert_eq!(coo.indices_j, &[0, 0, 1, 1, 2, 2, 3, 3, 1, 4, 4, 0, 0, 0]);
        assert_eq!(
            coo.values,
            &[2.0, 3.0, 3.0, -1.0, -1.0, 2.0, 2.0, 3.0, 6.0, 6.0, 1.0, 0.0, 0.0, 0.0]
        );
        let mut a = Matrix::new(5, 5);
        coo.to_dense(&mut a).unwrap();
        let correct = "┌                ┐\n\
                       │  2  3  0  0  0 │\n\
                       │  3  0 -1  0  6 │\n\
                       │  0 -1  0  2  0 │\n\
                       │  0  0  2  3  0 │\n\
                       │  0  6  0  0  1 │\n\
                       └                ┘";
        assert_eq!(format!("{}", a), correct);
    }

    #[test]
    fn read_matrix_market_complex_symmetric_to_full_works() {
        let h = MMsym::MakeItFull;
        let filepath = "./data/matrix_market/ok_complex_symmetric_small.mtx".to_string();
        let (coo_real, coo_cpx) = read_matrix_market(&filepath, h).unwrap();
        assert!(coo_real.is_none());
        let coo = coo_cpx.unwrap();
        assert_eq!(coo.symmetric, Sym::YesFull);
        assert_eq!((coo.nrow, coo.ncol, coo.nnz, coo.max_nnz), (5, 5, 11, 14));
        assert_eq!(coo.indices_i, &[0, 1, 0, 2, 1, 3, 2, 3, 4, 1, 4, 0, 0, 0]);
        assert_eq!(coo.indices_j, &[0, 0, 1, 1, 2, 2, 3, 3, 1, 4, 4, 0, 0, 0]);
        assert_eq!(
            coo.values,
            &[
                cpx!(2.0, 1.0),
                cpx!(3.0, 2.0),
                cpx!(3.0, 2.0),
                cpx!(-1.0, 3.0),
                cpx!(-1.0, 3.0),
                cpx!(2.0, 4.0),
                cpx!(2.0, 4.0),
                cpx!(3.0, 5.0),
                cpx!(6.0, 6.0),
                cpx!(6.0, 6.0),
                cpx!(1.0, 7.0),
                cpx!(0.0, 0.0),
                cpx!(0.0, 0.0),
                cpx!(0.0, 0.0),
            ]
        );
    }
}
