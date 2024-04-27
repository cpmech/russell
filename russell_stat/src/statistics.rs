use std::fmt;

/// Holds basic statistics of a dataset
///
/// **Note:** The [statistics()] function allocates a new [Statistics] structure
pub struct Statistics {
    /// Minimum value
    pub min: f64,

    /// Maximum value
    pub max: f64,

    /// Arithmetic mean
    pub mean: f64,

    /// (sample) Standard deviation (applying Bessel's correction)
    pub std_dev: f64,
}

/// Calculates basic statistics of a dataset
///
/// # Examples
///
/// ```
/// use russell_stat::statistics;
///
/// let res = statistics(&[2, 4, 4, 4, 5, 5, 7, 9]);
/// assert_eq!(res.min, 2.0);
/// assert_eq!(res.max, 9.0);
/// assert_eq!(res.mean, 5.0);
/// assert_eq!(res.std_dev, f64::sqrt(32.0/7.0));
///
/// let res = statistics(&[1.0, 1.0, 1.0]);
/// assert_eq!(
///     format!("{}", res),
///     "min = 1\n\
///      max = 1\n\
///      mean = 1\n\
///      std_dev = 0\n"
/// );
/// ```
pub fn statistics<T>(x: &[T]) -> Statistics
where
    T: Into<f64> + Copy,
{
    // handle small slices
    if x.len() == 0 {
        return Statistics {
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            std_dev: 0.0,
        };
    }
    if x.len() == 1 {
        return Statistics {
            min: x[0].into(),
            max: x[0].into(),
            mean: x[0].into(),
            std_dev: 0.0,
        };
    }

    // average
    let sum = x.iter().fold(0.0, |acc, &curr| acc + curr.into());
    let n = x.len() as f64;
    let mean = sum / n;

    // limits and auxiliary data
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut corrector = 0.0;
    let mut variance = 0.0;
    for &val in x {
        let x = val.into();
        if x < min {
            min = x;
        }
        if x > max {
            max = x;
        }
        let diff = x - mean; // diff ← xi - bar(x)
        corrector += diff; // corrector ← Σ diff
        variance += diff * diff; // variance ← Σ diff²
    }

    // (sample) standard deviation (applying Bessel's correction)
    variance = (variance - corrector * corrector / n) / (n - 1.0);
    let std_dev = variance.sqrt();

    // results
    Statistics {
        min,
        max,
        mean,
        std_dev,
    }
}

impl fmt::Display for Statistics {
    /// Prints statistics
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match f.precision() {
            Some(digits) => write!(
                f,
                "min = {:.4$}\nmax = {:.4$}\nmean = {:.4$}\nstd_dev = {:.4$}\n",
                self.min, self.max, self.mean, self.std_dev, digits
            )
            .unwrap(),
            None => write!(
                f,
                "min = {}\nmax = {}\nmean = {}\nstd_dev = {}\n",
                self.min, self.max, self.mean, self.std_dev
            )
            .unwrap(),
        }
        Ok(())
    }
}

///////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::statistics;
    use russell_lab::approx_eq;

    #[test]
    fn statistics_handle_small_slices() {
        let x: [i32; 0] = [];
        let res = statistics(&x);
        assert_eq!(res.min, 0.0);
        assert_eq!(res.max, 0.0);
        assert_eq!(res.mean, 0.0);
        assert_eq!(res.std_dev, 0.0);

        let x = [1.23];
        let res = statistics(&x);
        assert_eq!(res.min, 1.23);
        assert_eq!(res.max, 1.23);
        assert_eq!(res.mean, 1.23);
        assert_eq!(res.std_dev, 0.0);
    }

    #[test]
    fn statistics_works() {
        let x = [100, 100, 102, 98, 77, 99, 70, 105, 98];
        let res = statistics(&x);
        assert_eq!(res.min, 70.0);
        assert_eq!(res.max, 105.0);
        assert_eq!(res.mean, 849.0 / 9.0);
        approx_eq(res.std_dev, 12.134661099511597, 1e-17);

        let x = [9, 2, 5, 4, 12, 7, 8, 11, 9, 3, 7, 4, 12, 5, 4, 10, 9, 6, 9, 4];
        let res = statistics(&x);
        assert_eq!(res.min, 2.0);
        assert_eq!(res.max, 12.0);
        assert_eq!(res.mean, 7.0);
        approx_eq(res.std_dev, f64::sqrt(178.0 / 19.0), 1e-17);
    }

    #[test]
    fn display_works() {
        let x = [9, 2, 5, 4, 12, 7, 8, 11, 9, 3, 7, 4, 12, 5, 4, 10, 9, 6, 9, 4];
        let res = statistics(&x);
        assert_eq!(
            format!("{:.3}", res),
            "min = 2.000\n\
             max = 12.000\n\
             mean = 7.000\n\
             std_dev = 3.061\n"
        );

        let x = [1, 1, 1];
        let res = statistics(&x);
        assert_eq!(
            format!("{}", res),
            "min = 1\n\
             max = 1\n\
             mean = 1\n\
             std_dev = 0\n"
        );
    }
}
