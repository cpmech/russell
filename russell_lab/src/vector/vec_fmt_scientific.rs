use super::Vector;
use crate::format_scientific;
use std::cmp;
use std::fmt::Write;

pub fn vec_fmt_scientific(u: &Vector, precision: usize) -> String {
    let mut result = String::new();
    let f = &mut result;
    // handle empty vector
    if u.dim() == 0 {
        write!(f, "[]").unwrap();
        return result;
    }
    // find largest width
    let mut width = 0;
    let mut buf = String::new();
    for i in 0..u.dim() {
        write!(&mut buf, "{:.1$e}", u[i], precision).unwrap();
        let _ = buf.split_off(buf.find('e').unwrap());
        width = cmp::max(buf.chars().count(), width);
        buf.clear();
    }
    width += 4;
    // draw vector
    width += 1;
    write!(f, "┌{:1$}┐\n", " ", width + 1).unwrap();
    for i in 0..u.dim() {
        if i > 0 {
            write!(f, " │\n").unwrap();
        }
        write!(f, "│").unwrap();
        write!(f, "{}", format_scientific(u[i], width, precision)).unwrap();
    }
    write!(f, " │\n").unwrap();
    write!(f, "└{:1$}┘", " ", width + 1).unwrap();
    result
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::vec_fmt_scientific;
    use crate::Vector;

    #[test]
    fn vec_fmt_scientific_works() {
        let u = Vector::from(&[1.012444, 200.034123, 1e-8]);
        println!("{}", vec_fmt_scientific(&u, 3));
        assert_eq!(
            vec_fmt_scientific(&u, 3),
            "┌           ┐\n\
             │ 1.012E+00 │\n\
             │ 2.000E+02 │\n\
             │ 1.000E-08 │\n\
             └           ┘"
        );

        let u = Vector::from(&[1.012444, 200.034123, 1e+8]);
        println!("{}", vec_fmt_scientific(&u, 4));
        assert_eq!(
            vec_fmt_scientific(&u, 4),
            "┌            ┐\n\
             │ 1.0124E+00 │\n\
             │ 2.0003E+02 │\n\
             │ 1.0000E+08 │\n\
             └            ┘"
        );
    }
}
