use num_traits::cast::ToPrimitive;
use num_traits::Num;
use std::fmt::{self, Write};

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

/// Returns a pretty string representing the value in nanoseconds
///
/// # Panics
///
/// This function may panic if the write! macro fails (rarely)
///
/// # Examples
///
/// ```
/// use russell_lab::format_nanoseconds;
///
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

/// Formats a number using the scientific notation
///
/// # Examples
///
/// ```
/// use russell_lab::format_scientific;
///
/// let res = format_scientific(3_723_000.0, 10, 3);
/// assert_eq!(res, " 3.723E+06");
/// ```
pub fn format_scientific(num: f64, width: usize, precision: usize) -> String {
    // based on <https://stackoverflow.com/questions/65264069/alignment-of-floating-point-numbers-printed-in-scientific-notation>
    const EXP_PAD: usize = 2;
    let mut result = format!("{:.precision$e}", num, precision = precision);
    let exp = result.split_off(result.find('e').unwrap());
    let (sign, exp) = if exp.starts_with("e-") {
        ('-', &exp[2..])
    } else {
        ('+', &exp[1..])
    };
    result.push_str(&format!("E{}{:0>pad$}", sign, exp, pad = EXP_PAD));
    format!("{:>width$}", result, width = width)
}

/// Formats a number using the scientific notation as in Fortran with the ES23.15 format
///
/// This function is useful to compare results with those computed by a Fortran routine.
///
/// **Note:** This function calls [format_scientific()] with `width = 23` and `precision = 15`.
///
/// # Examples
///
/// ```
/// use russell_lab::format_fortran;
///
/// let res = format_fortran(3_723_000.0);
/// assert_eq!(res, "  3.723000000000000E+06");
/// ```
pub fn format_fortran(num: f64) -> String {
    format_scientific(num, 23, 15)
}

/// Writes a formatted number into a buffer, considering the number of digits and the precision
///
/// This function optionally formats the number in scientific notation.
///
/// This function is useful when implementing the Display trait for a struct that contains numbers.
///
/// # Arguments
///
/// * `buffer` - the buffer to write into
/// * `number` - the number to format
/// * `standard_formatter` - the standard formatter (used to get the precision)
/// * `scientific_precision` - the precision for scientific notation (if any)
/// * `width` - the width of the field (may be 0 to ignore the width formatting)
pub fn write_formatted_number<T>(
    buffer: &mut String,
    number: T,
    width: usize,
    precision: Option<usize>,
    scientific_precision: Option<usize>,
) where
    T: Num + Copy + fmt::Display + ToPrimitive,
{
    match scientific_precision {
        // has scientific precision
        Some(p) => match number.to_f64() {
            // if number can be converted to f64, use scientific notation
            Some(v) => write!(buffer, "{}", format_scientific(v, width, p)).unwrap(),
            // otherwise, use standard formatting
            None => write!(buffer, "{:>w$.p$}", number, p = p, w = width).unwrap(),
        },
        // use default precision
        None => match precision {
            // has standard precision
            Some(p) => write!(buffer, "{:>w$.p$}", number, p = p, w = width).unwrap(),
            // no precision
            None => write!(buffer, "{:>w$}", number, w = width).unwrap(),
        },
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{format_fortran, format_nanoseconds, format_scientific, write_formatted_number};

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
    fn format_scientific_works() {
        assert_eq!(format_scientific(0.1111, 9, 2), " 1.11E-01");
        assert_eq!(format_scientific(0.02222, 11, 4), " 2.2220E-02");
        assert_eq!(format_scientific(3333.0, 10, 3), " 3.333E+03");
        assert_eq!(format_scientific(-44444.0, 9, 1), " -4.4E+04");
        assert_eq!(format_scientific(0.0, 8, 1), " 0.0E+00");
        assert_eq!(format_scientific(1.0, 23, 15), "  1.000000000000000E+00");
        assert_eq!(format_scientific(42.0, 23, 15), "  4.200000000000000E+01");
        assert_eq!(format_scientific(9999999999.00, 8, 1), " 1.0E+10");
        assert_eq!(format_scientific(999999999999.00, 23, 15), "  9.999999999990000E+11");
        assert_eq!(format_scientific(123456789.1011, 11, 4), " 1.2346E+08");
    }

    #[test]
    fn format_fortran_works() {
        assert_eq!(format_fortran(0.1111), "  1.111000000000000E-01");
        assert_eq!(format_fortran(0.02222), "  2.222000000000000E-02");
        assert_eq!(format_fortran(3333.0), "  3.333000000000000E+03");
        assert_eq!(format_fortran(-44444.0), " -4.444400000000000E+04");
        assert_eq!(format_fortran(0.0), "  0.000000000000000E+00");
        assert_eq!(format_fortran(1.0), "  1.000000000000000E+00");
        assert_eq!(format_fortran(42.0), "  4.200000000000000E+01");
        assert_eq!(format_fortran(9999999999.00), "  9.999999999000000E+09");
        assert_eq!(format_fortran(999999999999.00), "  9.999999999990000E+11");
        assert_eq!(format_fortran(123456789.1011), "  1.234567891011000E+08");
    }

    #[test]
    fn write_formatted_number_works() {
        let mut buffer = String::new();

        // Test with no precision, no scientific, no width
        write_formatted_number(&mut buffer, 42.0, 0, None, None);
        assert_eq!(buffer, "42");
        buffer.clear();

        // Test with width only
        write_formatted_number(&mut buffer, 42, 5, None, None);
        assert_eq!(buffer, "   42");
        buffer.clear();

        // Test with standard precision only
        write_formatted_number(&mut buffer, 123.456, 0, Some(2), None);
        assert_eq!(buffer, "123.46");
        buffer.clear();

        // Test with standard precision and width
        write_formatted_number(&mut buffer, 123.456, 10, Some(3), None);
        assert_eq!(buffer, "   123.456");
        buffer.clear();

        // Test with scientific precision only
        write_formatted_number(&mut buffer, 123.456, 0, None, Some(2));
        assert_eq!(buffer, "1.23E+02");
        buffer.clear();

        // Test with scientific precision and width
        write_formatted_number(&mut buffer, 123.456, 12, None, Some(3));
        assert_eq!(buffer, "   1.235E+02");
        buffer.clear();

        // Test negative number with scientific notation
        write_formatted_number(&mut buffer, -456.789, 14, None, Some(4));
        assert_eq!(buffer, "   -4.5679E+02");
        buffer.clear();

        // Test zero with scientific notation
        write_formatted_number(&mut buffer, 0.0, 9, None, Some(1));
        assert_eq!(buffer, "  0.0E+00");
        buffer.clear();

        // Test integer type
        write_formatted_number(&mut buffer, 42i32, 0, None, None);
        assert_eq!(buffer, "42");
        buffer.clear();

        // Test integer with scientific (should convert to f64)
        write_formatted_number(&mut buffer, 1234i32, 9, None, Some(2));
        assert_eq!(buffer, " 1.23E+03");
        buffer.clear();

        // Test very small number with scientific notation
        write_formatted_number(&mut buffer, 0.000123, 9, None, Some(2));
        assert_eq!(buffer, " 1.23E-04");
        buffer.clear();

        // Test large number with scientific notation
        write_formatted_number(&mut buffer, 1234567.89, 12, None, Some(3));
        assert_eq!(buffer, "   1.235E+06");
        buffer.clear();

        // Test with both standard precision and scientific (scientific takes precedence)
        write_formatted_number(&mut buffer, 123.456, 9, Some(5), Some(2));
        assert_eq!(buffer, " 1.23E+02");
        buffer.clear();

        // Test integer with standard precision
        write_formatted_number(&mut buffer, 42i32, 8, Some(3), None);
        assert_eq!(buffer, "      42");
        buffer.clear();

        // Test floating point with zero width
        write_formatted_number(&mut buffer, 3.14159, 0, Some(2), None);
        assert_eq!(buffer, "3.14");
        buffer.clear();

        // Test very large width
        write_formatted_number(&mut buffer, 1.0, 20, None, Some(1));
        assert_eq!(buffer, "             1.0E+00");
        buffer.clear();

        // Test negative with standard formatting
        write_formatted_number(&mut buffer, -123.456, 10, Some(2), None);
        assert_eq!(buffer, "   -123.46");
        buffer.clear();

        let n = 9_223_372_036_854_775_807i64;
        write_formatted_number(&mut buffer, n, 0, None, Some(3));
        assert_eq!(buffer, "9.223E+18");
        buffer.clear();

        let n = 9_223_372_036_854_775_808u128;
        write_formatted_number(&mut buffer, n, 0, None, Some(3));
        assert_eq!(buffer, "9.223E+18");
        buffer.clear();
    }
}
