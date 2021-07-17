use super::*;
use std::cmp;
use std::fmt::Write;

/// Returns a string representation of a matrix
pub fn format_matrix(mat: &Vec<Vec<f64>>, nrow: usize, ncol: usize) -> Result<String, Error> {
    let mut width = 0;
    let mut buf = String::new();
    for i in 0..nrow {
        for j in 0..ncol {
            write!(&mut buf, "{}", mat[i][j])?;
            width = cmp::max(buf.chars().count(), width);
            buf.clear();
        }
    }
    width += 1;
    write!(buf, "┌{:1$}┐\n", " ", width * ncol + 1)?;
    for i in 0..nrow {
        if i > 0 {
            write!(buf, " │\n")?;
        }
        for j in 0..ncol {
            if j == 0 {
                write!(buf, "│")?;
            }
            write!(buf, "{:>1$}", mat[i][j], width)?;
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
    fn format_matrix_works() -> Result<(), Error> {
        let mat = vec![vec![11.1, 22.2, 33.3], vec![44.4, 55.5, 66.6], vec![77.7, 88.8, 99.9]];
        let buf = format_matrix(&mat, 3, 3)?;
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
