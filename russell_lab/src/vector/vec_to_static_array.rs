use super::Vector;
use crate::format_scientific;
use std::fmt::Write;

/// Generates Rust code with a static array representation of a vector
///
/// Note: The numbers are scientifically formatted with 15 digits.
///
/// # Examples
///
/// ```
/// use russell_lab::{vec_to_static_array, Vector};
///
/// fn main() {
///     let v = Vector::from(&[1.0, -2.01, -3.8e-5]);
///
///     let res = vec_to_static_array("A_VECTOR", &v);
///
///     let correct =
/// r#"// A_VECTOR: dim = 3
/// #[rustfmt::skip]
/// const A_VECTOR: [f64; 3] = [
///       1.000000000000000E+00, -2.010000000000000E+00, -3.800000000000000E-05,
/// ];
/// "#;
///
///     assert_eq!(res, correct);
/// }
/// ```
pub fn vec_to_static_array(name: &str, v: &Vector) -> String {
    let dim = v.dim();
    let mut buf = String::new();
    write!(&mut buf, "// {}: dim = {}\n", name, dim).unwrap();
    write!(&mut buf, "#[rustfmt::skip]\n").unwrap();
    write!(&mut buf, "const {}: [f64; {}] = [\n", name, dim).unwrap();
    write!(&mut buf, "\x20\x20\x20\x20").unwrap();
    for i in 0..dim {
        if i > 0 {
            write!(&mut buf, ",").unwrap();
        }
        write!(&mut buf, "{}", format_scientific(v[i], 23, 15)).unwrap();
    }
    write!(&mut buf, ",\n];\n").unwrap();
    buf
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{vec_to_static_array, Vector};
    use crate::vec_approx_eq;

    #[test]
    fn vec_to_static_array_works() {
        #[rustfmt::skip]
        let v = Vector::from(&[
            1.01, -25.0, 3.0e-10,
        ]);
        let res = vec_to_static_array("A_VECTOR", &v);

        println!("{}", res);

        let correct = r#"// A_VECTOR: dim = 3
#[rustfmt::skip]
const A_VECTOR: [f64; 3] = [
      1.010000000000000E+00, -2.500000000000000E+01,  3.000000000000000E-10,
];
"#;
        assert_eq!(res, correct);

        // A_VECTOR: dim = 3
        #[rustfmt::skip]
        const A_VECTOR: [f64; 3] = [
            1.010000000000000E+00, -2.500000000000000E+01,  3.000000000000000E-10,
        ];
        let u = Vector::from(&A_VECTOR);
        vec_approx_eq(&v, &u, 1e-20);
    }
}
