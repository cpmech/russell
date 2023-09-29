use super::SparseMatrix;
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
/// **Note:** The vismatrix format is is like the MatrixMarket format
/// without the header, and the indices start at zero.
///
/// # References
///
/// * MatrixMarket: <https://math.nist.gov/MatrixMarket/formats.html>
/// * Vismatrix: <https://github.com/cpmech/vismatrix>
///
/// # Examples
///
/// ```
/// use russell_sparse::prelude::*;
/// use russell_sparse::StrError;
///
/// const SAVE_FILE: bool = true;
///
/// fn main() -> Result<(), StrError> {
///     // allocate a square matrix and store as CSR matrix
///     // ┌                ┐
///     // │  2  3  0  0  0 │
///     // │  3  0  4  0  6 │
///     // │  0 -1 -3  2  0 │
///     // │  0  0  1  0  0 │
///     // │  0  4  2  0  1 │
///     // └                ┘
///     let nrow = 5;
///     let ncol = 5;
///     let row_pointers = vec![0, 2, 5, 8, 9, 12];
///     let col_indices = vec![
///         //                         p
///         0, 1, //    i = 0, count = 0, 1
///         0, 2, 4, // i = 1, count = 2, 3, 4
///         1, 2, 3, // i = 2, count = 5, 6, 7
///         2, //       i = 3, count = 8
///         1, 2, 4, // i = 4, count = 9, 10, 11
///            //              count = 12
///     ];
///     let values = vec![
///         //                                 p
///         2.0, 3.0, //        i = 0, count = 0, 1
///         3.0, 4.0, 6.0, //   i = 1, count = 2, 3, 4
///         -1.0, -3.0, 2.0, // i = 2, count = 5, 6, 7
///         1.0, //             i = 3, count = 8
///         4.0, 2.0, 1.0, //   i = 4, count = 9, 10, 11
///              //                    count = 12
///     ];
///     let symmetry = None;
///     let csr = CsrMatrix::new(nrow, ncol,
///         row_pointers, col_indices, values, symmetry)?;
///     let mut mat = SparseMatrix::from_csr(csr);
///     if SAVE_FILE {
///         let full_path = "/tmp/russell_sparse/doc-example-vismatrix.smat";
///         write_matrix_market(&mut mat, full_path, true)?;
///     }
///     Ok(())
/// }
/// ```
///
/// By running `vismatrix doc-example-vismatrix.smat` you get the following screen:
///
/// ![doc-example-vismatrix](https://raw.githubusercontent.com/cpmech/russell/main/russell_sparse/data/figures/doc-example-vismatrix.png)
pub fn write_matrix_market<P>(mat: &mut SparseMatrix, full_path: &P, vismatrix: bool) -> Result<(), StrError>
where
    P: AsRef<OsStr> + ?Sized,
{
    // output buffer
    let mut buffer = String::new();

    // information
    let (nrow, ncol, nnz, symmetry) = mat.get_info()?;
    if nrow == 0 || ncol == 0 || nnz == 0 {
        return Err("nrow, ncol, and nnz must be greater than zero");
    }

    // handle one-based indexing
    let d = if vismatrix { 0 } else { 1 };

    // write header and dimensions
    if !vismatrix {
        match symmetry {
            Some(_) => write!(&mut buffer, "%%MatrixMarket matrix coordinate real symmetric\n").unwrap(),
            None => write!(&mut buffer, "%%MatrixMarket matrix coordinate real general\n").unwrap(),
        };
    }
    write!(&mut buffer, "{} {} {}\n", nrow, ncol, nnz).unwrap();

    // write data
    match mat.get_csc() {
        Ok(csc) => {
            for j in 0..csc.ncol {
                for p in csc.col_pointers[j]..csc.col_pointers[j + 1] {
                    let i = csc.row_indices[p as usize] as usize;
                    let aij = csc.values[p as usize];
                    write!(&mut buffer, "{} {} {:?}\n", i + d, j + d, aij).unwrap();
                }
            }
        }
        Err(_) => match mat.get_csr() {
            Ok(csr) => {
                for i in 0..csr.nrow {
                    for p in csr.row_pointers[i]..csr.row_pointers[i + 1] {
                        let j = csr.col_indices[p as usize] as usize;
                        let aij = csr.values[p as usize];
                        write!(&mut buffer, "{} {} {:?}\n", i + d, j + d, aij).unwrap();
                    }
                }
            }
            Err(_) => {
                let csc = mat.get_csc_or_from_coo()?;
                for j in 0..csc.ncol {
                    for p in csc.col_pointers[j]..csc.col_pointers[j + 1] {
                        let i = csc.row_indices[p as usize] as usize;
                        let aij = csc.values[p as usize];
                        write!(&mut buffer, "{} {} {:?}\n", i + d, j + d, aij).unwrap();
                    }
                }
            }
        },
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
    use super::write_matrix_market;
    use crate::{Samples, SparseMatrix};
    use std::fs;

    #[test]
    fn write_matrix_market_works() {
        // ┌                ┐
        // │  2  3  0  0  0 │
        // │  3  0  4  0  6 │
        // │  0 -1 -3  2  0 │
        // │  0  0  1  0  0 │
        // │  0  4  2  0  1 │
        // └                ┘
        let (_, _, csr, _) = Samples::umfpack_unsymmetric_5x5(false);
        let mut mat = SparseMatrix::from_csr(csr);

        let full_path = "/tmp/russell_sparse/test_write_matrix_market.mtx";
        write_matrix_market(&mut mat, full_path, false).unwrap();
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
    }

    #[test]
    fn write_matrix_market_vismatrix_works() {
        // ┌                ┐
        // │  2  3  0  0  0 │
        // │  3  0  4  0  6 │
        // │  0 -1 -3  2  0 │
        // │  0  0  1  0  0 │
        // │  0  4  2  0  1 │
        // └                ┘
        let (_, _, csr, _) = Samples::umfpack_unsymmetric_5x5(false);
        let mut mat = SparseMatrix::from_csr(csr);

        let full_path = "/tmp/russell_sparse/test_write_matrix_market_vismatrix.smat";
        write_matrix_market(&mut mat, full_path, true).unwrap();
        let contents = fs::read_to_string(full_path).map_err(|_| "cannot open file").unwrap();
        assert_eq!(
            contents,
            "5 5 12\n\
             0 0 2.0\n\
             0 1 3.0\n\
             1 0 3.0\n\
             1 2 4.0\n\
             1 4 6.0\n\
             2 1 -1.0\n\
             2 2 -3.0\n\
             2 3 2.0\n\
             3 2 1.0\n\
             4 1 4.0\n\
             4 2 2.0\n\
             4 4 1.0\n"
        );
    }
}
