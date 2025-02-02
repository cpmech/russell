use super::Matrix;
use std::fmt::Write;

/// Generates numpy code with a nested list representing a matrix
///
/// When saved to a file, we can execute the numpy script as:
///
/// ```text
/// math < /tmp/russell_lab/the_matrix.m
/// ```
///
/// # Examples
///
/// ```
/// use russell_lab::{mat_to_numpy, Matrix};
///
/// fn main() {
///     #[rustfmt::skip]
///     let a = Matrix::from(&[
///         [ 1.0,  2.0],
///         [-3.0,  4.0],
///         [ 5.0,  6.0e-10],
///     ]);
///
///     let res = mat_to_numpy("a_matrix", &a);
///
///     let correct = "a_matrix = np.array([[1,2],[-3,4],[5,0.0000000006]],dtype=float)";
///
///     assert_eq!(res, correct);
/// }
/// ```
pub fn mat_to_numpy(name: &str, a: &Matrix) -> String {
    let (nrow, ncol) = a.dims();
    let mut buf = String::new();
    write!(&mut buf, "{} = np.array([", name).unwrap();
    for i in 0..nrow {
        if i > 0 {
            write!(&mut buf, "],").unwrap();
        }
        for j in 0..ncol {
            if j == 0 {
                write!(&mut buf, "[").unwrap();
            } else {
                write!(&mut buf, ",").unwrap();
            }
            let val = a.get(i, j);
            write!(&mut buf, "{}", val).unwrap();
        }
    }
    write!(&mut buf, "]],dtype=float)").unwrap();
    buf
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{mat_to_numpy, Matrix};

    #[test]
    fn mat_to_numpy_works() {
        #[rustfmt::skip]
        let a = Matrix::from(&[
            [  1.0,  20.0],
            [  3.01,  4.0],
            [-55.0,   6.0],
            [  7.0,   8.0e-10],
        ]);
        let res = mat_to_numpy("a_matrix", &a);
        assert_eq!(
            res,
            "a_matrix = np.array([[1,20],[3.01,4],[-55,6],[7,0.0000000008]],dtype=float)"
        )
    }
}
