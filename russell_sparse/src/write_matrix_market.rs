use super::{CscMatrix, Sym};
use crate::StrError;
use std::ffi::OsStr;
use std::fmt::Write;
use std::fs::{self, File};
use std::io::Write as IoWrite;
use std::path::Path;

impl CscMatrix {
    /// Writes a MatrixMarket file
    ///
    /// # Input
    ///
    /// * `full_path` -- may be a String, &str, or Path
    /// * `vismatrix` -- generate a SMAT file for Vismatrix instead of a MatrixMarket
    ///
    /// # Notes
    ///
    /// 1. The vismatrix format is is similar to the MatrixMarket format
    ///    without the header, and the indices start at zero.
    /// 2. If the matrix is symmetric, then:
    ///     * Only the lower triangle + diagonal will be written (standard MatrixMarket format)
    ///     * For vismatrix, the full matrix will be written
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

        // write header
        if !vismatrix {
            if self.symmetric == Sym::No {
                write!(&mut buffer, "%%MatrixMarket matrix coordinate real general\n").unwrap();
            } else {
                write!(&mut buffer, "%%MatrixMarket matrix coordinate real symmetric\n").unwrap();
            }
        }

        // compute the number of non-zeros to be written
        let nnz = if vismatrix && (self.symmetric == Sym::YesLower || self.symmetric == Sym::YesUpper) {
            // will mirror off-diagonal elements
            let mut count = 0;
            for j in 0..self.ncol {
                for p in self.col_pointers[j]..self.col_pointers[j + 1] {
                    let i = self.row_indices[p as usize] as usize;
                    count += 1;
                    if i != j {
                        count += 1;
                    }
                }
            }
            count
        } else if !vismatrix && self.symmetric == Sym::YesFull {
            // will consider the lower-triangle only
            let mut count = 0;
            for j in 0..self.ncol {
                for p in self.col_pointers[j]..self.col_pointers[j + 1] {
                    let i = self.row_indices[p as usize] as usize;
                    if i >= j {
                        count += 1;
                    }
                }
            }
            count
        } else {
            // will use the default number of non-zeros
            self.col_pointers[self.ncol] as usize
        };

        // write dimensions
        write!(&mut buffer, "{} {} {}\n", self.nrow, self.ncol, nnz).unwrap();

        // write data
        if vismatrix {
            for j in 0..self.ncol {
                for p in self.col_pointers[j]..self.col_pointers[j + 1] {
                    let i = self.row_indices[p as usize] as usize;
                    let aij = self.values[p as usize];
                    write!(&mut buffer, "{} {} {:?}\n", i, j, aij).unwrap();
                    if self.symmetric == Sym::YesLower || self.symmetric == Sym::YesUpper {
                        if i != j {
                            // mirror off-diagonal elements
                            write!(&mut buffer, "{} {} {:?}\n", j, i, aij).unwrap();
                        }
                    }
                }
            }
        } else {
            for j in 0..self.ncol {
                for p in self.col_pointers[j]..self.col_pointers[j + 1] {
                    let i = self.row_indices[p as usize] as usize;
                    let aij = self.values[p as usize];
                    match self.symmetric {
                        Sym::No => write!(&mut buffer, "{} {} {:?}\n", i + 1, j + 1, aij).unwrap(),
                        Sym::YesLower => write!(&mut buffer, "{} {} {:?}\n", i + 1, j + 1, aij).unwrap(),
                        Sym::YesUpper => write!(&mut buffer, "{} {} {:?}\n", j + 1, i + 1, aij).unwrap(),
                        Sym::YesFull => {
                            if i >= j {
                                // consider the lower-triangle only
                                write!(&mut buffer, "{} {} {:?}\n", i + 1, j + 1, aij).unwrap()
                            }
                        }
                    }
                }
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
    use super::*;
    use crate::Samples;

    #[test]
    fn csc_write_matrix_market_works() {
        //  2  3  .  .  .
        //  3  .  4  .  6
        //  . -1 -3  2  .
        //  .  .  1  .  .
        //  .  4  2  .  1
        let (_, csc, _, _) = Samples::umfpack_unsymmetric_5x5();
        let full_path = "/tmp/russell_sparse/test_write_matrix_market_csc.mtx";
        csc.write_matrix_market(full_path, false).unwrap();
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
    }

    #[test]
    fn csc_write_matrix_market_sym_works() {
        let correct = "%%MatrixMarket matrix coordinate real symmetric\n\
                       3 3 5\n\
                       1 1 2.0\n\
                       2 1 -1.0\n\
                       2 2 2.0\n\
                       3 2 -1.0\n\
                       3 3 2.0\n";
        //  2  -1              2     sym
        // -1   2  -1    =>   -1   2
        //     -1   2             -1   2
        let (_, csc, _, _) = Samples::positive_definite_3x3_lower();
        let full_path = "/tmp/russell_sparse/test_write_matrix_market_csc_sym_lower.mtx";
        csc.write_matrix_market(full_path, false).unwrap();
        let contents = fs::read_to_string(full_path).map_err(|_| "cannot open file").unwrap();
        assert_eq!(contents, correct);
        //  2  -1              2  -1
        // -1   2  -1    =>        2  -1
        //     -1   2          sym     2
        let (_, csc, _, _) = Samples::positive_definite_3x3_upper();
        let full_path = "/tmp/russell_sparse/test_write_matrix_market_csc_sym_upper.mtx";
        csc.write_matrix_market(full_path, false).unwrap();
        let contents = fs::read_to_string(full_path).map_err(|_| "cannot open file").unwrap();
        assert_eq!(contents, correct);
        //  2  -1
        // -1   2  -1
        //     -1   2
        let (_, csc, _, _) = Samples::positive_definite_3x3_full();
        let full_path = "/tmp/russell_sparse/test_write_matrix_market_csc_sym_full.mtx";
        csc.write_matrix_market(full_path, false).unwrap();
        let contents = fs::read_to_string(full_path).map_err(|_| "cannot open file").unwrap();
        assert_eq!(contents, correct);
    }

    #[test]
    fn csc_write_matrix_market_works_vismatrix() {
        //  2  3  .  .  .
        //  3  .  4  .  6
        //  . -1 -3  2  .
        //  .  .  1  .  .
        //  .  4  2  .  1
        let (_, csc, _, _) = Samples::umfpack_unsymmetric_5x5();
        let full_path = "/tmp/russell_sparse/test_write_matrix_market_csc.smat";
        csc.write_matrix_market(full_path, true).unwrap();
        let contents = fs::read_to_string(full_path).map_err(|_| "cannot open file").unwrap();
        assert_eq!(
            contents,
            "5 5 12\n\
             0 0 2.0\n\
             1 0 3.0\n\
             0 1 3.0\n\
             2 1 -1.0\n\
             4 1 4.0\n\
             1 2 4.0\n\
             2 2 -3.0\n\
             3 2 1.0\n\
             4 2 2.0\n\
             2 3 2.0\n\
             1 4 6.0\n\
             4 4 1.0\n"
        );
    }

    #[test]
    fn csc_write_matrix_market_sym_works_vismatrix() {
        let correct = "3 3 7\n\
                       0 0 2.0\n\
                       1 0 -1.0\n\
                       0 1 -1.0\n\
                       1 1 2.0\n\
                       2 1 -1.0\n\
                       1 2 -1.0\n\
                       2 2 2.0\n";
        let correct_upper = "3 3 7\n\
                             0 0 2.0\n\
                             0 1 -1.0\n\
                             1 0 -1.0\n\
                             1 1 2.0\n\
                             1 2 -1.0\n\
                             2 1 -1.0\n\
                             2 2 2.0\n";
        //  2  -1              2     sym
        // -1   2  -1    =>   -1   2
        //     -1   2             -1   2
        let (_, csc, _, _) = Samples::positive_definite_3x3_lower();
        let full_path = "/tmp/russell_sparse/test_write_matrix_market_csc_sym_lower.smat";
        csc.write_matrix_market(full_path, true).unwrap();
        let contents = fs::read_to_string(full_path).map_err(|_| "cannot open file").unwrap();
        assert_eq!(contents, correct);
        //  2  -1              2  -1
        // -1   2  -1    =>        2  -1
        //     -1   2          sym     2
        let (_, csc, _, _) = Samples::positive_definite_3x3_upper();
        let full_path = "/tmp/russell_sparse/test_write_matrix_market_csc_sym_upper.smat";
        csc.write_matrix_market(full_path, true).unwrap();
        let contents = fs::read_to_string(full_path).map_err(|_| "cannot open file").unwrap();
        assert_eq!(contents, correct_upper);
        //  2  -1
        // -1   2  -1
        //     -1   2
        let (_, csc, _, _) = Samples::positive_definite_3x3_full();
        let full_path = "/tmp/russell_sparse/test_write_matrix_market_csc_sym_full.smat";
        csc.write_matrix_market(full_path, true).unwrap();
        let contents = fs::read_to_string(full_path).map_err(|_| "cannot open file").unwrap();
        assert_eq!(contents, correct);
    }
}
