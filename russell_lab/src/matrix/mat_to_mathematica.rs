use super::Matrix;
use std::fmt::Write;

/// Generates Mathematica code with a nested list representing a matrix
///
/// When saved to a file, we can execute the Mathematica script as:
///
/// ```text
/// math < /tmp/russell_lab/the_matrix.m
/// ```
///
/// # Examples
///
/// ```
/// use russell_lab::{mat_to_mathematica, Matrix};
///
/// fn main() {
///     #[rustfmt::skip]
///     let a = Matrix::from(&[
///         [ 1.0,  2.0],
///         [-3.0,  4.0],
///         [ 5.0,  6.0e-10],
///     ]);
///
///     let res = mat_to_mathematica("aMatrix", &a);
///
///     let correct = "aMatrix = {{1,2},{-3,4},{5,0.0000000006}};\n";
///
///     assert_eq!(res, correct);
/// }
/// ```
pub fn mat_to_mathematica(name: &str, a: &Matrix) -> String {
    let (nrow, ncol) = a.dims();
    let mut buf = String::new();
    write!(&mut buf, "{} = {{", name).unwrap();
    for i in 0..nrow {
        if i > 0 {
            write!(&mut buf, "}},").unwrap();
        }
        for j in 0..ncol {
            if j == 0 {
                write!(&mut buf, "{{").unwrap();
            } else {
                write!(&mut buf, ",").unwrap();
            }
            let val = a.get(i, j);
            write!(&mut buf, "{}", val).unwrap();
        }
    }
    write!(&mut buf, "}}}};\n").unwrap();
    buf
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{mat_to_mathematica, Matrix};

    #[test]
    fn mat_to_mathematica_works() {
        #[rustfmt::skip]
        let a = Matrix::from(&[
            [  1.0,  20.0],
            [  3.01,  4.0],
            [-55.0,   6.0],
            [  7.0,   8.0e-10],
        ]);
        let res = mat_to_mathematica("aMatrix", &a);
        assert_eq!(res, "aMatrix = {{1,20},{3.01,4},{-55,6},{7,0.0000000008}};\n")
    }
}
