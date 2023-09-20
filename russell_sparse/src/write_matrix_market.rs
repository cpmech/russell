use super::{CsrMatrix, Layout};
use crate::StrError;
use std::ffi::OsStr;
use std::fmt::Write;
use std::fs::{self, File};
use std::io::Write as IoWrite;
use std::path::Path;

impl CsrMatrix {
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
    pub fn write_matrix_market<P>(&self, full_path: &P, vismatrix: bool) -> Result<(), StrError>
    where
        P: AsRef<OsStr> + ?Sized,
    {
        // output buffer
        let mut buffer = String::new();

        // write to buffer
        let d = if vismatrix { 0 } else { 1 };
        let nnz = self.values.len();
        if !vismatrix {
            if self.layout == Layout::Full {
                write!(&mut buffer, "%%MatrixMarket matrix coordinate real general\n").unwrap()
            } else {
                write!(&mut buffer, "%%MatrixMarket matrix coordinate real symmetric\n").unwrap()
            }
        }
        write!(&mut buffer, "{} {} {}\n", self.nrow, self.ncol, nnz).unwrap();
        for i in 0..self.nrow {
            for p in self.row_pointers[i]..self.row_pointers[i + 1] {
                let j = self.col_indices[p as usize] as usize;
                let aij = self.values[p as usize];
                write!(&mut buffer, "{} {} {:?}\n", i + d, j + d, aij).unwrap()
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
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use crate::{CsrMatrix, Layout};
    use std::fs;

    #[test]
    fn write_matrix_market_works() {
        // allocate a square matrix and store as CSR matrix
        // ┌                ┐
        // │  2  3  0  0  0 │
        // │  3  0  4  0  6 │
        // │  0 -1 -3  2  0 │
        // │  0  0  1  0  0 │
        // │  0  4  2  0  1 │
        // └                ┘
        let csr = CsrMatrix {
            layout: Layout::Full,
            nrow: 5,
            ncol: 5,
            row_pointers: vec![0, 2, 5, 8, 9, 12],
            col_indices: vec![
                //                         p
                0, 1, //    i = 0, count = 0, 1
                0, 2, 4, // i = 1, count = 2, 3, 4
                1, 2, 3, // i = 2, count = 5, 6, 7
                2, //       i = 3, count = 8
                1, 2, 4, // i = 4, count = 9, 10, 11
                   //              count = 12
            ],
            values: vec![
                //                                 p
                2.0, 3.0, //        i = 0, count = 0, 1
                3.0, 4.0, 6.0, //   i = 1, count = 2, 3, 4
                -1.0, -3.0, 2.0, // i = 2, count = 5, 6, 7
                1.0, //             i = 3, count = 8
                4.0, 2.0, 1.0, //   i = 4, count = 9, 10, 11
                     //                    count = 12
            ],
        };

        let full_path = "/tmp/russell_sparse/test_write_matrix_market.mtx";
        csr.write_matrix_market(full_path, false).unwrap();
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
        // allocate a square matrix and store as CSR matrix
        // ┌                ┐
        // │  2  3  0  0  0 │
        // │  3  0  4  0  6 │
        // │  0 -1 -3  2  0 │
        // │  0  0  1  0  0 │
        // │  0  4  2  0  1 │
        // └                ┘
        let csr = CsrMatrix {
            layout: Layout::Full,
            nrow: 5,
            ncol: 5,
            row_pointers: vec![0, 2, 5, 8, 9, 12],
            col_indices: vec![
                //                         p
                0, 1, //    i = 0, count = 0, 1
                0, 2, 4, // i = 1, count = 2, 3, 4
                1, 2, 3, // i = 2, count = 5, 6, 7
                2, //       i = 3, count = 8
                1, 2, 4, // i = 4, count = 9, 10, 11
                   //              count = 12
            ],
            values: vec![
                //                                 p
                2.0, 3.0, //        i = 0, count = 0, 1
                3.0, 4.0, 6.0, //   i = 1, count = 2, 3, 4
                -1.0, -3.0, 2.0, // i = 2, count = 5, 6, 7
                1.0, //             i = 3, count = 8
                4.0, 2.0, 1.0, //   i = 4, count = 9, 10, 11
                     //                    count = 12
            ],
        };

        let full_path = "/tmp/russell_sparse/test_write_matrix_market_vismatrix.smat";
        csr.write_matrix_market(full_path, true).unwrap();
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
