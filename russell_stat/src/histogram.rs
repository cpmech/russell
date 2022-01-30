use crate::StrError;
use num_traits::Num;
use std::cmp;
use std::fmt::{self, Write};

/// Implements a Histogram to count frequencies
///
/// The stations/bins are set as follows:
///
/// ```text
///    [ bin[0] )[ bin[1] )[ bin[2] )[ bin[3] )[ bin[4] )
/// ---|---------|---------|---------|---------|---------|---  x
///  s[0]      s[1]      s[2]      s[3]      s[4]      s[5]
/// ```
///
/// bin_i corresponds to station_i <= x < station_(i+1)
///
/// # Example
/// ```
/// use russell_stat::{Histogram, StrError};
///
/// fn main() -> Result<(), StrError> {
///     let data = [
///         -1.0, // outside
///         10.0, 10.1, // outside
///         9.0, // count = 1
///         8.1, 8.2, // count = 2
///         7.1, 7.2, 7.2, // count = 3
///         6.0, 6.1, 6.1, 6.2, 6.99, // count = 5
///         5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, // count = 9
///         4.0, 4.1, 4.1, 4.2, 4.99, // count = 5
///         3.1, 3.2, 3.2, // count = 3
///         2.1, // count = 1
///     ];
///
///     let stations: [f64; 11] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
///
///     let mut hist = Histogram::new(&stations)?;
///
///     hist.count(&data);
///     assert_eq!(hist.get_counts(), &[0, 0, 1, 3, 5, 9, 5, 3, 2, 1]);
///
///     hist.set_bar_max_len(10);
///     assert_eq!(
///         format!("{}", hist),
///         "[ 0, 1) | 0 \n\
///          [ 1, 2) | 0 \n\
///          [ 2, 3) | 1 🟦\n\
///          [ 3, 4) | 3 🟦🟦🟦\n\
///          [ 4, 5) | 5 🟦🟦🟦🟦🟦\n\
///          [ 5, 6) | 9 🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦\n\
///          [ 6, 7) | 5 🟦🟦🟦🟦🟦\n\
///          [ 7, 8) | 3 🟦🟦🟦\n\
///          [ 8, 9) | 2 🟦🟦\n\
///          [ 9,10) | 1 🟦\n\
///          \x20\x20\x20sum = 29\n"
///     );
///     Ok(())
/// }
pub struct Histogram<T>
where
    T: Num + Copy,
{
    // data
    stations: Vec<T>,
    counts: Vec<usize>,

    // used in Display
    bar_char: char,     // character used in bars
    bar_max_len: usize, // maximum length of bar (max number of characters)
}

impl<T> Histogram<T>
where
    T: Num + Copy + PartialOrd,
{
    /// Creates a new Histogram
    pub fn new(stations: &[T]) -> Result<Self, StrError> {
        if stations.len() < 2 {
            return Err("histogram must have at least 2 stations");
        }
        let nbins = stations.len() - 1;
        Ok(Histogram {
            stations: Vec::from(stations),
            counts: vec![0; nbins],
            bar_char: '🟦',
            bar_max_len: 30,
        })
    }

    /// Counts how many items fall within each bin
    pub fn count(&mut self, data: &[T]) {
        for x in data {
            if let Some(i) = self.find_bin(*x) {
                self.counts[i] += 1;
            }
        }
    }

    /// Erase all counts
    pub fn reset(&mut self) {
        for i in 0..self.counts.len() {
            self.counts[i] = 0;
        }
    }

    /// Returns a read-only access to the counts (frequencies)
    pub fn get_counts(&self) -> &Vec<usize> {
        &self.counts
    }

    /// Sets the character used in histogram drawn by Display
    pub fn set_bar_char(&mut self, bar_char: char) -> &mut Self {
        self.bar_char = bar_char;
        self
    }

    /// Sets the maximum length (number of characters) used in histogram draw by Display
    pub fn set_bar_max_len(&mut self, bar_max_len: usize) -> &mut Self {
        self.bar_max_len = bar_max_len;
        self
    }

    /// Finds which bin contains x
    fn find_bin(&self, x: T) -> Option<usize> {
        // handle values outside range
        let nstation = self.stations.len();
        if x < self.stations[0] {
            return None;
        }
        if x >= self.stations[nstation - 1] {
            return None;
        }

        // perform binary search
        let mut upper = nstation;
        let mut lower = 0;
        let mut mid;
        while upper - lower > 1 {
            mid = (upper + lower) / 2;
            if x >= self.stations[mid] {
                lower = mid
            } else {
                upper = mid
            }
        }
        Some(lower)
    }
}

impl<T> fmt::Display for Histogram<T>
where
    T: Num + Copy + fmt::Display,
{
    /// Draws histogram
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // find limits and number of characters
        let nbins = self.counts.len();
        let mut c_max = 0; // count_max
        let mut l_c_max = 0; // max length of c_max string
        let mut buf = String::new();
        for i in 0..nbins {
            // find c_max and string l_c_max
            let c = self.counts[i];
            write!(&mut buf, "{}", c)?;
            c_max = cmp::max(c_max, c);
            l_c_max = cmp::max(l_c_max, buf.chars().count());
            buf.clear();
        }

        // check count_max
        if c_max < 1 {
            write!(f, "zero data\n").unwrap();
            return Ok(());
        }

        // find number of characters of station number
        let mut l_s_max = 0; // max length of station numbers
        for i in 0..self.stations.len() {
            let station = self.stations[i];
            match f.precision() {
                Some(digits) => write!(&mut buf, "{:.1$}", station, digits)?,
                None => write!(&mut buf, "{}", station)?,
            }
            l_s_max = cmp::max(l_s_max, buf.chars().count());
            buf.clear();
        }

        // draw histogram
        let scale = (self.bar_max_len as f64) / (c_max as f64);
        let mut total = 0;
        for i in 0..nbins {
            let count = self.counts[i];
            let (left, right) = (self.stations[i], self.stations[i + 1]);
            total += count;
            match f.precision() {
                Some(digits) => write!(
                    f,
                    "[{:>3$.4$},{:>3$.4$}) | {:>5$}",
                    left, right, count, l_s_max, digits, l_c_max
                )?,
                None => write!(f, "[{:>3$},{:>3$}) | {:>4$}", left, right, count, l_s_max, l_c_max)?,
            }
            let n = scale * (count as f64);
            let bar = std::iter::repeat(self.bar_char).take(n as usize).collect::<String>();
            write!(f, " {}\n", bar)?;
        }
        write!(f, "{:>1$}\n", format!("sum = {}", total), 2 * l_s_max + l_c_max + 6)?;
        Ok(())
    }
}

///////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Histogram;
    use crate::StrError;

    #[test]
    fn new_fails_on_wrong_input() -> Result<(), StrError> {
        assert_eq!(
            Histogram::<i32>::new(&[]).err(),
            Some("histogram must have at least 2 stations")
        );
        Ok(())
    }

    #[test]
    fn new_works() -> Result<(), StrError> {
        let stations: [i32; 6] = [0, 1, 2, 3, 4, 5];
        let hist = Histogram::new(&stations)?;
        assert_eq!(hist.stations.len(), 6);
        assert_eq!(hist.counts.len(), 5);
        Ok(())
    }

    #[test]
    fn find_bin_works() -> Result<(), StrError> {
        let stations: [f64; 6] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let hist = Histogram::new(&stations)?;

        let res = hist.find_bin(-3.3);
        assert_eq!(res, None);

        let res = hist.find_bin(7.0);
        assert_eq!(res, None);

        for i in 0..stations.len() {
            let res = hist.find_bin(stations[i]);
            if i < stations.len() - 1 {
                assert_eq!(res, Some(i));
            } else {
                assert_eq!(res, None);
            }
        }

        let res = hist.find_bin(0.5);
        assert_eq!(res, Some(0));

        let res = hist.find_bin(1.5);
        assert_eq!(res, Some(1));

        let res = hist.find_bin(2.5);
        assert_eq!(res, Some(2));

        let res = hist.find_bin(3.99999999999999);
        assert_eq!(res, Some(3));

        let res = hist.find_bin(4.999999);
        assert_eq!(res, Some(4));

        Ok(())
    }

    #[test]
    fn count_and_reset_work() -> Result<(), StrError> {
        #[rustfmt::skip]
        let data = [
            0.0, 0.1, 0.2, 0.3, 0.9, // 5
            1.0, 1.0, 1.0, 1.2, 1.3, 1.4, 1.5, 1.99, // 8
            2.0, 2.5, // 2
            3.0, 3.5, // 2
            4.1, 4.5, 4.9, // 3
            -3.0, -2.0, -1.0, // outside
            5.0, 6.0, 7.0, 8.0, // outside
        ];
        let stations: [f64; 6] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let mut hist = Histogram::new(&stations)?;
        hist.count(&data);
        assert_eq!(hist.get_counts(), &[5, 8, 2, 2, 3]);
        hist.reset();
        assert_eq!(hist.get_counts(), &[0, 0, 0, 0, 0]);

        #[rustfmt::skip]
        let data: [i32; 12]= [
            0, 0, 0, 0, // 4
            1, // 1
            2, 2, 2, // 3
            // 0
            // 0
            5, 5, // 2
            -1, 10, // outside
        ];
        let stations: [i32; 6] = [0, 1, 2, 3, 4, 5];
        let mut hist = Histogram::new(&stations)?;
        hist.count(&data);
        assert_eq!(hist.counts, &[4, 1, 3, 0, 0]);
        hist.reset();
        assert_eq!(hist.counts, &[0, 0, 0, 0, 0]);
        Ok(())
    }

    #[test]
    fn display_returns_errors() -> Result<(), StrError> {
        let hist = Histogram::new(&[1, 2])?;
        assert_eq!(format!("{:.3}", hist), "zero data\n");
        Ok(())
    }

    #[test]
    fn display_works() -> Result<(), StrError> {
        #[rustfmt::skip]
        let data = [
            0.0, 0.1, 0.2, 0.3, 0.9, // 5
            1.0, 1.0, 1.0, 1.0, 1.0, 1.2, 1.3, 1.4, 1.5, 1.99, // 10
            2.0, 2.5, // 2
            // 0
            4.1, 4.5, 4.9, // 3
        ];
        let stations: [f64; 11] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let mut hist = Histogram::new(&stations)?;
        hist.count(&data);
        hist.set_bar_char('🔶').set_bar_max_len(10);
        assert_eq!(
            format!("{:.3}", hist),
            "[ 0.000, 1.000) |  5 🔶🔶🔶🔶🔶\n\
             [ 1.000, 2.000) | 10 🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶\n\
             [ 2.000, 3.000) |  2 🔶🔶\n\
             [ 3.000, 4.000) |  0 \n\
             [ 4.000, 5.000) |  3 🔶🔶🔶\n\
             [ 5.000, 6.000) |  0 \n\
             [ 6.000, 7.000) |  0 \n\
             [ 7.000, 8.000) |  0 \n\
             [ 8.000, 9.000) |  0 \n\
             [ 9.000,10.000) |  0 \n\
             \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20sum = 20\n"
        );

        #[rustfmt::skip]
        let data = [
            0.0, 0.1, 0.2, 0.3, 0.9, // 5
            1.0, 1.0, 1.0, 1.2, 1.3, 1.4, 1.5, 1.99, // 8
            2.0, 2.5, // 2
            3.0, 3.5, // 2
            4.1, 4.5, 4.9, // 3
            -3.0, -2.0, -1.0, // outside
            50.0, 60.0, 70.0, 80.0, // outside
        ];
        let stations: [f64; 11] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let mut hist = Histogram::new(&stations)?;
        hist.count(&data);
        assert_eq!(hist.counts, &[5, 8, 2, 2, 3, 0, 0, 0, 0, 0]);
        assert_eq!(
            format!("{}", hist),
            "[ 0, 1) | 5 🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦\n\
             [ 1, 2) | 8 🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦\n\
             [ 2, 3) | 2 🟦🟦🟦🟦🟦🟦🟦\n\
             [ 3, 4) | 2 🟦🟦🟦🟦🟦🟦🟦\n\
             [ 4, 5) | 3 🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦\n\
             [ 5, 6) | 0 \n\
             [ 6, 7) | 0 \n\
             [ 7, 8) | 0 \n\
             [ 8, 9) | 0 \n\
             [ 9,10) | 0 \n\
             \x20\x20\x20sum = 20\n"
        );
        Ok(())
    }
}
