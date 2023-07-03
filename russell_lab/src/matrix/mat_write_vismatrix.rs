use super::Matrix;
use crate::StrError;
use std::ffi::OsStr;
use std::fmt::Write;
use std::fs::{self, File};
use std::io::Write as IoWrite;
use std::path::Path;

/// Writes a text file that can be visualized with VisMatrix
///
/// See [github.com/cpmech/vismatrix](https://github.com/cpmech/vismatrix)
///
/// # Input
///
/// * `full_path` -- may be a String, &str, or Path. Note: VisMatrix uses the `.smat` extension.
/// * `tol` is a small positive constant to ignore nearly zero numbers.
/// Only values satisfying the condition `f64::abs(value) > tol` are written.
///
/// # Examples
///
/// ```
/// use russell_lab::{mat_write_vismatrix, Matrix, StrError};
/// use std::fs;
///
/// fn main() -> Result<(), StrError> {
///     let a = Matrix::from(&[[1, 2], [3, 4]]);
///     let path = "/tmp/russell_lab/test_mat_write_vismatrix.smat";
///     mat_write_vismatrix(path, &a, 0.0)?;
///     if false {
///         let contents = fs::read_to_string(path).map_err(|_| "cannot open file")?;
///         assert_eq!(
///             contents,
///             "2 2 4\n\
///              0 0 1.0\n\
///              0 1 2.0\n\
///              1 0 3.0\n\
///              1 1 4.0\n"
///         );
///     }
///     Ok(())
/// }
/// ```
///
/// ![vismatrix](https://raw.githubusercontent.com/cpmech/russell_lab/main/data/figures/test_mat_write_vismatrix.png)
pub fn mat_write_vismatrix<P>(full_path: &P, a: &Matrix, tol: f64) -> Result<(), StrError>
where
    P: AsRef<OsStr> + ?Sized,
{
    // check tolerance
    if tol < 0.0 {
        return Err("tol must be ≥ 0");
    }

    // prepare content and compute number of non-zero values (nnz)
    let (nrow, ncol) = a.dims();
    let mut nnz = 0;
    let mut buffer = String::new();
    for i in 0..nrow {
        for j in 0..ncol {
            let value = a.get(i, j);
            if f64::abs(value) > tol {
                write!(&mut buffer, "{} {} {:?}\n", i, j, value).unwrap();
                nnz += 1;
            }
        }
    }

    // prepare header
    let mut header = String::new();
    write!(&mut header, "{} {} {}\n", nrow, ncol, nnz).unwrap();

    // create directory
    let path = Path::new(full_path);
    if let Some(p) = path.parent() {
        fs::create_dir_all(p).map_err(|_| "cannot create directory")?;
    }

    // write data to file
    let mut file = File::create(path).map_err(|_| "cannot create file")?;
    file.write_all(header.as_bytes()).map_err(|_| "cannot write file")?;
    file.write_all(buffer.as_bytes()).map_err(|_| "cannot write file")?;

    // force sync
    file.sync_all().map_err(|_| "cannot sync file")?;
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::mat_write_vismatrix;
    use crate::Matrix;
    use std::fs;

    #[test]
    fn mat_write_vismatrix_captures_errors() {
        let a = Matrix::from(&[[1, 2], [3, 4]]);
        assert_eq!(
            mat_write_vismatrix("/tmp/russell_lab/test_mat_write_vismatrix.smat", &a, -1.0).err(),
            Some("tol must be ≥ 0")
        );
    }

    #[test]
    fn mat_write_vismatrix_works() {
        let a = Matrix::from(&[[1, 2], [3, 4]]);
        let path = "/tmp/russell_lab/test_mat_write_vismatrix.smat";
        mat_write_vismatrix(path, &a, 0.0).unwrap();
        let contents = fs::read_to_string(path).unwrap();
        assert_eq!(
            contents,
            "2 2 4\n\
             0 0 1.0\n\
             0 1 2.0\n\
             1 0 3.0\n\
             1 1 4.0\n"
        );
    }
}
