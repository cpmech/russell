use std::cmp;
use std::fmt::Write;

const NS_PER_NANOSECOND: u128 = 1;
const NS_PER_MICROSECOND: u128 = 1000 * NS_PER_NANOSECOND;
const NS_PER_MILLISECOND: u128 = 1000 * NS_PER_MICROSECOND;
const NS_PER_SECOND: u128 = 1000 * NS_PER_MILLISECOND;
const NS_PER_MINUTE: u128 = 60 * NS_PER_SECOND;
const NS_PER_HOUR: u128 = 60 * NS_PER_MINUTE;

/// Formats the nanoseconds in 1 second; value < 1 second
fn format_nanoseconds_in_seconds(buf: &mut String, value: u128) {
    if value < NS_PER_MICROSECOND {
        write!(buf, "{}ns", value).unwrap();
    } else if value < NS_PER_MILLISECOND {
        write!(buf, "{}µs", (value as f64) / (NS_PER_MICROSECOND as f64)).unwrap();
    } else {
        write!(buf, "{}ms", (value as f64) / (NS_PER_MILLISECOND as f64)).unwrap();
    }
}

/// Returns a nice string representing the value in nanoseconds
///
/// # Panics
///
/// This function may panic if the write! macro fails (rarely)
///
/// # Example
///
/// ```
/// use russell_lab::format_nanoseconds;
/// let res = format_nanoseconds(3_723_000_000_000);
/// assert_eq!(res, "1h2m3s");
/// ```
pub fn format_nanoseconds(nanoseconds: u128) -> String {
    if nanoseconds == 0 {
        return "0ns".to_string();
    }

    let mut value = nanoseconds;
    let mut buf = String::new();
    if value < NS_PER_SECOND {
        // nanoseconds is smaller than a second => use small units such as 2.5ms
        format_nanoseconds_in_seconds(&mut buf, value);
    } else {
        // nanoseconds is greater than a second => use large units such as 3m2.5s
        if value >= NS_PER_HOUR {
            let hours = value / NS_PER_HOUR;
            value -= hours * NS_PER_HOUR;
            write!(&mut buf, "{}h", hours).unwrap();
        }
        if value >= NS_PER_MINUTE {
            let minutes = value / NS_PER_MINUTE;
            value -= minutes * NS_PER_MINUTE;
            write!(&mut buf, "{}m", minutes).unwrap();
        }
        if value > 0 {
            if value < NS_PER_SECOND {
                format_nanoseconds_in_seconds(&mut buf, value);
            } else {
                let seconds = (value as f64) / (NS_PER_SECOND as f64);
                write!(&mut buf, "{}s", &seconds).unwrap();
            }
        }
    }

    buf
}

/// Returns a string representation of a 2D array
///
/// Symbols from <http://www.i2symbol.com/symbols/line>
///
/// # Example
///
/// ```
/// use russell_lab::format_2d_array;
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
/// assert_eq!(format_2d_array(&a).unwrap(), str_a);
/// assert_eq!(format_2d_array(&b).unwrap(), str_b);
/// ```
///
pub fn format_2d_array<'a, T, U>(mat: &'a T) -> Result<String, std::fmt::Error>
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
    use super::format_2d_array;
    use super::format_nanoseconds;

    #[test]
    fn format_nanoseconds_works() {
        // 0 => 0ns
        let mut res = format_nanoseconds(0);
        assert_eq!(res, "0ns");

        // 250 => 250ns
        res = format_nanoseconds(250);
        assert_eq!(res, "250ns");

        // 2500 (2.5e3) => 2.5µs
        res = format_nanoseconds(2_500);
        assert_eq!(res, "2.5µs");

        // 25,000 (2.5e4) = 25µs
        res = format_nanoseconds(25_000);
        assert_eq!(res, "25µs");

        // 250,000 (2.5e5) = 250µs
        res = format_nanoseconds(250_000);
        assert_eq!(res, "250µs");

        // 2,500,000 (2.5e6) = 2.5ms
        res = format_nanoseconds(2_500_000);
        assert_eq!(res, "2.5ms");

        // 25,000,000 (2.5e7) = 25ms
        res = format_nanoseconds(25_000_000);
        assert_eq!(res, "25ms");

        // 250,000,000 (2.5e8) = 250ms
        res = format_nanoseconds(250_000_000);
        assert_eq!(res, "250ms");

        // 2,500,000,000 (2.5e9) = 2.5s
        res = format_nanoseconds(2_500_000_000);
        assert_eq!(res, "2.5s");

        // 25,000,000,000 (2.5e10) = 25s
        res = format_nanoseconds(25_000_000_000);
        assert_eq!(res, "25s");

        // 250,000,000,000 (2.5e11) = 250s = 4m10s
        res = format_nanoseconds(250_000_000_000);
        assert_eq!(res, "4m10s");

        // 2,500,000,000,000 (2.5e12) = 2,500s = 41m40s
        res = format_nanoseconds(2_500_000_000_000);
        assert_eq!(res, "41m40s");

        // 25,000,000,000,000 (2.5e13) = 25,000s = 416m40s = 6h56m40s
        res = format_nanoseconds(25_000_000_000_000);
        assert_eq!(res, "6h56m40s");

        // 250,000,000,000,000 (2.5e14) = 250,000s = 4166m40s = 69h26m40s
        res = format_nanoseconds(250_000_000_000_000);
        assert_eq!(res, "69h26m40s");

        // 60,000,000,000 (60e9, 6e10) = 1m
        res = format_nanoseconds(60_000_000_000);
        assert_eq!(res, "1m");

        // 120,000,000,000 (120e9, 1.2e11) = 2m
        res = format_nanoseconds(120_000_000_000);
        assert_eq!(res, "2m");

        // 3,600,000,000,000 (3.6e12) = 1h
        res = format_nanoseconds(3_600_000_000_000);
        assert_eq!(res, "1h");

        // 3,723,000,000,000 = 1h2m3s
        res = format_nanoseconds(3_723_000_000_000);
        assert_eq!(res, "1h2m3s");

        // 3,600,000,000,001 (3.6e12 + 1ns) = 1h1ns
        res = format_nanoseconds(3_600_000_000_001);
        assert_eq!(res, "1h1ns");

        // 3,600,000,001,000 (3.6e12 + 1,000ns) = 1h1µs
        res = format_nanoseconds(3_600_000_001_000);
        assert_eq!(res, "1h1µs");

        // 3,600,000,100,001 (3.6e12 + 1,000,100,001ns) = 1h100.001µs
        res = format_nanoseconds(3_600_000_100_001);
        assert_eq!(res, "1h100.001µs");

        // 3,600,001,000,000 (3.6e12 + 1,000,000ns) = 1h1ms
        res = format_nanoseconds(3_600_001_000_000);
        assert_eq!(res, "1h1ms");

        // 3,601,000,000,000 (3.6e12 + 1,000,000,000ns) = 1h1s
        res = format_nanoseconds(3_601_000_000_000);
        assert_eq!(res, "1h1s");

        // 3,601,100,000,001 (3.6e12 + 1.1s) = 1h1.1s
        res = format_nanoseconds(3_601_100_000_000);
        assert_eq!(res, "1h1.1s");
    }

    #[test]
    fn format_2d_array_works() -> Result<(), std::fmt::Error> {
        let mat = [[11.1, 22.2, 33.3], [44.4, 55.5, 66.6], [77.7, 88.8, 99.9]];
        let buf = format_2d_array(&mat)?;
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
    fn format_2d_array_vec_works() -> Result<(), std::fmt::Error> {
        let mat = vec![vec![11.1, 22.2, 33.3], vec![44.4, 55.5, 66.6], vec![77.7, 88.8, 99.9]];
        let buf = format_2d_array(&mat)?;
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
