use super::Matrix;
use crate::format_scientific;
use std::fmt::Write;

/// Generates Rust code with a static array representation of a matrix
///
/// Note: The numbers are scientifically formatted with 15 digits.
///
/// # Examples
///
/// ```
/// use russell_lab::{mat_to_static_array, Matrix};
///
/// fn main() {
///     #[rustfmt::skip]
///     let a = Matrix::from(&[
///         [ 1.0,  2.0],
///         [-3.0,  4.0],
///         [ 5.0, -6.0],
///         [ 7.0,  8.0e-10],
///     ]);
///
///     let res = mat_to_static_array("A_MATRIX", &a);
///
///     let correct =
/// r#"// A_MATRIX: nrow = 4, ncol = 2
/// #[rustfmt::skip]
/// const A_MATRIX: [[f64; 2]; 4] = [
///     [  1.000000000000000E+00,  2.000000000000000E+00 ],
///     [ -3.000000000000000E+00,  4.000000000000000E+00 ],
///     [  5.000000000000000E+00, -6.000000000000000E+00 ],
///     [  7.000000000000000E+00,  8.000000000000000E-10 ],
/// ];
/// "#;
///
///     assert_eq!(res, correct);
/// }
/// ```
pub fn mat_to_static_array(name: &str, a: &Matrix) -> String {
    let (nrow, ncol) = a.dims();
    let mut buf = String::new();
    write!(&mut buf, "// {}: nrow = {}, ncol = {}\n", name, nrow, ncol).unwrap();
    write!(&mut buf, "#[rustfmt::skip]\n").unwrap();
    write!(&mut buf, "const {}: [[f64; {}]; {}] = [\n", name, ncol, nrow).unwrap();
    for i in 0..nrow {
        if i > 0 {
            write!(&mut buf, " ],\n").unwrap();
        }
        for j in 0..ncol {
            if j == 0 {
                write!(&mut buf, "\x20\x20\x20\x20[").unwrap();
            } else {
                write!(&mut buf, ",").unwrap();
            }
            let val = a.get(i, j);
            write!(&mut buf, "{}", format_scientific(val, 23, 15)).unwrap();
        }
    }
    write!(&mut buf, " ],\n").unwrap();
    write!(&mut buf, "];\n").unwrap();
    buf
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{mat_to_static_array, Matrix};
    use crate::mat_approx_eq;

    #[test]
    fn mat_to_static_array_works() {
        #[rustfmt::skip]
        let a = Matrix::from(&[
            [  1.0,  20.0],
            [  3.01,  4.0],
            [-55.0,   6.0],
            [  7.0,   8.0e-10],
        ]);
        let res = mat_to_static_array("A_MATRIX", &a);

        let correct = r#"// A_MATRIX: nrow = 4, ncol = 2
#[rustfmt::skip]
const A_MATRIX: [[f64; 2]; 4] = [
    [  1.000000000000000E+00,  2.000000000000000E+01 ],
    [  3.010000000000000E+00,  4.000000000000000E+00 ],
    [ -5.500000000000000E+01,  6.000000000000000E+00 ],
    [  7.000000000000000E+00,  8.000000000000000E-10 ],
];
"#;
        assert_eq!(res, correct);

        // A_MATRIX: nrow = 4, ncol = 2
        #[rustfmt::skip]
        const A_MATRIX: [[f64; 2]; 4] = [
            [  1.000000000000000E+00,  2.000000000000000E+01 ],
            [  3.010000000000000E+00,  4.000000000000000E+00 ],
            [ -5.500000000000000E+01,  6.000000000000000E+00 ],
            [  7.000000000000000E+00,  8.000000000000000E-10 ],
        ];
        let b = Matrix::from(&A_MATRIX);
        mat_approx_eq(&a, &b, 1e-20);
    }
}
