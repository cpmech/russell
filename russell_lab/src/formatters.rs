use std::fmt::Write;

const NS_PER_NANOSECOND: u128 = 1;
const NS_PER_MICROSECOND: u128 = 1000 * NS_PER_NANOSECOND;
const NS_PER_MILLISECOND: u128 = 1000 * NS_PER_MICROSECOND;
const NS_PER_SECOND: u128 = 1000 * NS_PER_MILLISECOND;
const NS_PER_MINUTE: u128 = 60 * NS_PER_SECOND;
const NS_PER_HOUR: u128 = 60 * NS_PER_MINUTE;

/// Returns a nice string representing the value in nanoseconds
///
/// # Panics
///
/// This function may panic if the write! macro fails (rarely)
///
/// # Example
///
/// ```
/// use russell_lab::*;
/// let res = format_nanoseconds(3_723_000_000_000);
/// assert_eq!(res, "1h2m3s");
/// ```
pub fn format_nanoseconds(nanoseconds: u128) -> String {
    if nanoseconds == 0 {
        return "0ns".to_string();
    }

    // nanoseconds is smaller than a second => use small units such as 2.5ms
    let mut value = nanoseconds;
    let mut buf = String::new();
    if value < NS_PER_SECOND {
        if value < NS_PER_MICROSECOND {
            write!(&mut buf, "{}ns", value).unwrap();
        } else if value < NS_PER_MILLISECOND {
            write!(&mut buf, "{}µs", (value as f64) / (NS_PER_MICROSECOND as f64)).unwrap();
        } else {
            write!(&mut buf, "{}ms", (value as f64) / (NS_PER_MILLISECOND as f64)).unwrap();
        }
    }
    // nanoseconds is greater than a second => use large units such as 3m2.5s
    else {
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
            let seconds = (value as f64) / (NS_PER_SECOND as f64);
            write!(&mut buf, "{}s", &seconds).unwrap();
        }
    }

    buf
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

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
    }
}
