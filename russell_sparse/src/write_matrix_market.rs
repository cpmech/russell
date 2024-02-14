use super::{CscMatrix, CsrMatrix, Symmetry};
use crate::StrError;
use std::ffi::OsStr;
use std::fmt::Write;
use std::fs::{self, File};
use std::io::Write as IoWrite;
use std::path::Path;

/// Writes a MatrixMarket file from a CooMatrix
///
/// # Input
///
/// * `full_path` -- may be a String, &str, or Path
/// * `vismatrix` -- generate a SMAT file for Vismatrix instead of a MatrixMarket
///
/// **Note:** The vismatrix format is is similar to the MatrixMarket format
/// without the header, and the indices start at zero.
///
/// # References
///
/// * MatrixMarket: <https://math.nist.gov/MatrixMarket/formats.html>
/// * Vismatrix: <https://github.com/cpmech/vismatrix>
pub fn csc_write_matrix_market<P>(mat: &CscMatrix, full_path: &P, vismatrix: bool) -> Result<(), StrError>
where
    P: AsRef<OsStr> + ?Sized,
{
    // output buffer
    let mut buffer = String::new();

    // handle one-based indexing
    let d = if vismatrix { 0 } else { 1 };

    // info
    let (nrow, ncol, nnz, symmetry) = mat.get_info();

    // write header
    if !vismatrix {
        if symmetry == Symmetry::No {
            write!(&mut buffer, "%%MatrixMarket matrix coordinate real general\n").unwrap();
        } else {
            write!(&mut buffer, "%%MatrixMarket matrix coordinate real symmetric\n").unwrap();
        }
    }

    // write dimensions
    write!(&mut buffer, "{} {} {}\n", nrow, ncol, nnz).unwrap();

    // access data
    let col_pointers = mat.get_col_pointers();
    let row_indices = mat.get_row_indices();
    let values = mat.get_values();

    // write triplets
    for j in 0..ncol {
        for p in col_pointers[j]..col_pointers[j + 1] {
            let i = row_indices[p as usize] as usize;
            let aij = values[p as usize];
            write!(&mut buffer, "{} {} {:?}\n", i + d, j + d, aij).unwrap();
        }
    }

    // create directory
    let path = Path::new(full_path);
    if let Some(p) = path.parent() {
        fs::create_dir_all(p).map_err(|_| "cannot create directory")?;
    }

    // write file
    let mut file = File::create(path).map_err(|_| "cannot create file")?;
    file.write_all(buffer.as_bytes()).map_err(|_| "cannot write file")?;

    // force sync
    file.sync_all().map_err(|_| "cannot sync file")?;
    Ok(())
}

/// Writes a MatrixMarket file from a CooMatrix
///
/// # Input
///
/// * `full_path` -- may be a String, &str, or Path
/// * `vismatrix` -- generate a SMAT file for Vismatrix instead of a MatrixMarket
///
/// **Note:** The vismatrix format is is similar to the MatrixMarket format
/// without the header, and the indices start at zero.
///
/// # References
///
/// * MatrixMarket: <https://math.nist.gov/MatrixMarket/formats.html>
/// * Vismatrix: <https://github.com/cpmech/vismatrix>
pub fn csr_write_matrix_market<P>(mat: &CsrMatrix, full_path: &P, vismatrix: bool) -> Result<(), StrError>
where
    P: AsRef<OsStr> + ?Sized,
{
    // output buffer
    let mut buffer = String::new();

    // handle one-based indexing
    let d = if vismatrix { 0 } else { 1 };

    // info
    let (nrow, ncol, nnz, symmetry) = mat.get_info();

    // write header
    if !vismatrix {
        if symmetry == Symmetry::No {
            write!(&mut buffer, "%%MatrixMarket matrix coordinate real general\n").unwrap();
        } else {
            write!(&mut buffer, "%%MatrixMarket matrix coordinate real symmetric\n").unwrap();
        }
    }

    // write dimensions
    write!(&mut buffer, "{} {} {}\n", nrow, ncol, nnz).unwrap();

    // access data
    let row_pointers = mat.get_row_pointers();
    let col_indices = mat.get_col_indices();
    let values = mat.get_values();

    // write triplets
    for i in 0..nrow {
        for p in row_pointers[i]..row_pointers[i + 1] {
            let j = col_indices[p as usize] as usize;
            let aij = values[p as usize];
            println!("{} {} {:?}", i, j, aij);
            write!(&mut buffer, "{} {} {:?}\n", i + d, j + d, aij).unwrap();
        }
    }

    // create directory
    let path = Path::new(full_path);
    if let Some(p) = path.parent() {
        fs::create_dir_all(p).map_err(|_| "cannot create directory")?;
    }

    // write file
    let mut file = File::create(path).map_err(|_| "cannot create file")?;
    file.write_all(buffer.as_bytes()).map_err(|_| "cannot write file")?;

    // force sync
    file.sync_all().map_err(|_| "cannot sync file")?;
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Samples;
    use std::fs;

    #[test]
    fn csc_write_matrix_market_works() {
        //  2  3  .  .  .
        //  3  .  4  .  6
        //  . -1 -3  2  .
        //  .  .  1  .  .
        //  .  4  2  .  1
        let (_, csc, _, _) = Samples::umfpack_unsymmetric_5x5(false);
        let full_path = "/tmp/russell_sparse/test_write_matrix_market_csc.mtx";
        csc_write_matrix_market(&csc, full_path, false).unwrap();
        let contents = fs::read_to_string(full_path).map_err(|_| "cannot open file").unwrap();
        assert_eq!(
            contents,
            "%%MatrixMarket matrix coordinate real general\n\
             5 5 12\n\
             1 1 2.0\n\
             2 1 3.0\n\
             1 2 3.0\n\
             3 2 -1.0\n\
             5 2 4.0\n\
             2 3 4.0\n\
             3 3 -3.0\n\
             4 3 1.0\n\
             5 3 2.0\n\
             3 4 2.0\n\
             2 5 6.0\n\
             5 5 1.0\n"
        );
        //  2  -1              2     sym
        // -1   2  -1    =>   -1   2
        //     -1   2             -1   2
        let (_, csc, _, _) = Samples::positive_definite_3x3(false);
        let full_path = "/tmp/russell_sparse/test_write_matrix_market_csc.mtx";
        csc_write_matrix_market(&csc, full_path, false).unwrap();
        let contents = fs::read_to_string(full_path).map_err(|_| "cannot open file").unwrap();
        assert_eq!(
            contents,
            "%%MatrixMarket matrix coordinate real symmetric\n\
             3 3 5\n\
             1 1 2.0\n\
             2 1 -1.0\n\
             2 2 2.0\n\
             3 2 -1.0\n\
             3 3 2.0\n"
        );
    }

    #[test]
    fn csr_write_matrix_market_works() {
        //  2  3  .  .  .
        //  3  .  4  .  6
        //  . -1 -3  2  .
        //  .  .  1  .  .
        //  .  4  2  .  1
        let (_, _, csr, _) = Samples::umfpack_unsymmetric_5x5(false);
        let full_path = "/tmp/russell_sparse/test_write_matrix_market_csr.mtx";
        csr_write_matrix_market(&csr, full_path, false).unwrap();
        let contents = fs::read_to_string(full_path).map_err(|_| "cannot open file").unwrap();
        assert_eq!(
            contents,
            "%%MatrixMarket matrix coordinate real general\n\
             5 5 12\n\
             1 1 2.0\n\
             1 2 3.0\n\
             2 1 3.0\n\
             2 3 4.0\n\
             2 5 6.0\n\
             3 2 -1.0\n\
             3 3 -3.0\n\
             3 4 2.0\n\
             4 3 1.0\n\
             5 2 4.0\n\
             5 3 2.0\n\
             5 5 1.0\n"
        );
        //  2  -1              2     sym
        // -1   2  -1    =>   -1   2
        //     -1   2             -1   2
        let (_, _, csr, _) = Samples::positive_definite_3x3(false);
        let full_path = "/tmp/russell_sparse/test_write_matrix_market_csr.mtx";
        csr_write_matrix_market(&csr, full_path, false).unwrap();
        let contents = fs::read_to_string(full_path).map_err(|_| "cannot open file").unwrap();
        assert_eq!(
            contents,
            "%%MatrixMarket matrix coordinate real symmetric\n\
             3 3 5\n\
             1 1 2.0\n\
             2 1 -1.0\n\
             2 2 2.0\n\
             3 2 -1.0\n\
             3 3 2.0\n"
        );
    }
}
