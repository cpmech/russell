use std::cmp;
use std::fmt::Write;

/// Returns a string representation of a 2D array
///
/// Symbols from <http://www.i2symbol.com/symbols/line>
///
/// # Example
///
/// ```
/// use russell_tensor::*;
/// let a = [[11.1, 22.2, 33.3], [44.4, 55.5, 66.6], [77.7, 88.8, 99.9]];
/// let b = vec![vec![1.1, 2.2, 3.3], vec![4.4, 5.5, 6.6], vec![7.7, 8.8, 9.9]];
/// let str_a = "┌                ┐\n\
///              │ 11.1 22.2 33.3 │\n\
///              │ 44.4 55.5 66.6 │\n\
///              │ 77.7 88.8 99.9 │\n\
///              └                ┘";
/// let str_b = "┌             ┐\n\
///              │ 1.1 2.2 3.3 │\n\
///              │ 4.4 5.5 6.6 │\n\
///              │ 7.7 8.8 9.9 │\n\
///              └             ┘";
/// assert_eq!(fmt_2d_array(&a).unwrap(), str_a);
/// assert_eq!(fmt_2d_array(&b).unwrap(), str_b);
/// ```
///
pub fn fmt_2d_array<'a, T, U>(mat: &'a T) -> Result<String, std::fmt::Error>
where
    &'a T: std::iter::IntoIterator<Item = U>,
    U: std::iter::IntoIterator<Item = &'a f64>,
{
    let mut width = 0;
    let mut ncol = 0;
    let mut buf = String::new();
    for row in mat.into_iter() {
        for (j, &val) in row.into_iter().enumerate() {
            write!(&mut buf, "{}", val)?;
            width = cmp::max(buf.chars().count(), width);
            ncol = cmp::max(j + 1, ncol);
            buf.clear();
        }
    }
    width += 1;
    write!(buf, "┌{:1$}┐\n", " ", width * ncol + 1)?;
    for (i, row) in mat.into_iter().enumerate() {
        if i > 0 {
            write!(buf, " │\n")?;
        }
        for (j, &val) in row.into_iter().enumerate() {
            if j == 0 {
                write!(buf, "│")?;
            }
            write!(buf, "{:>1$}", val, width)?;
        }
    }
    write!(buf, " │\n")?;
    write!(buf, "└{:1$}┘", " ", width * ncol + 1)?;
    Ok(buf)
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fmt_2d_array_works() -> Result<(), std::fmt::Error> {
        let mat = [[11.1, 22.2, 33.3], [44.4, 55.5, 66.6], [77.7, 88.8, 99.9]];
        let buf = fmt_2d_array(&mat)?;
        println!("{}", buf);
        let correct = "┌                ┐\n\
                            │ 11.1 22.2 33.3 │\n\
                            │ 44.4 55.5 66.6 │\n\
                            │ 77.7 88.8 99.9 │\n\
                            └                ┘";
        assert_eq!(buf, correct);
        Ok(())
    }

    #[test]
    fn fmt_2d_array_vec_works() -> Result<(), std::fmt::Error> {
        let mat = vec![vec![11.1, 22.2, 33.3], vec![44.4, 55.5, 66.6], vec![77.7, 88.8, 99.9]];
        let buf = fmt_2d_array(&mat)?;
        println!("{}", buf);
        let correct = "┌                ┐\n\
                            │ 11.1 22.2 33.3 │\n\
                            │ 44.4 55.5 66.6 │\n\
                            │ 77.7 88.8 99.9 │\n\
                            └                ┘";
        assert_eq!(buf, correct);
        Ok(())
    }
}
