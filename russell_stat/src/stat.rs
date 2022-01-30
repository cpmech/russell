use crate::StrError;

/// Returns the average of values in the a given slice
///
/// # Example
///
/// ```
/// use russell_stat;
/// let x = [2, 4, 4, 4, 5, 5, 7, 9];
/// let result = russell_stat::ave(&x);
/// assert_eq!(result.unwrap(), 5.0);
/// ```
pub fn ave<T>(x: &[T]) -> Result<f64, StrError>
where
    T: Into<f64> + Copy,
{
    if x.len() == 0 {
        return Err("cannot compute average of empty slice");
    }
    let sum = x.iter().fold(0.0, |acc, &curr| acc + curr.into());
    let n = x.len() as f64;
    Ok(sum / n)
}

/// Returns the average of values and their standard deviation
///
/// # Example
///
/// ```
/// use russell_stat;
/// let x = [2, 4, 4, 4, 5, 5, 7, 9];
/// let result = russell_stat::ave_dev(&x);
/// let (ave, dev) = result.unwrap();
/// assert_eq!(ave, 5.0);
/// assert_eq!(dev, ((32.0/7.0) as f64).sqrt());
/// ```
pub fn ave_dev<T>(x: &[T]) -> Result<(f64, f64), StrError>
where
    T: Into<f64> + Copy,
{
    if x.len() < 2 {
        return Err("at least two values are needed");
    }

    // average
    let sum = x.iter().fold(0.0, |acc, &curr| acc + curr.into());
    let n = x.len() as f64;
    let ave = sum / n;

    // variance
    let mut corrector = 0.0;
    let mut variance = 0.0;
    for &val in x {
        let diff = val.into() - ave; // diff ← xi - bar(x)
        corrector += diff; // corrector ← Σ diff
        variance += diff * diff; // variance ← Σ diff²
    }

    // standard deviation
    variance = (variance - corrector * corrector / n) / (n - 1.0);
    let dev = variance.sqrt();

    // results
    Ok((ave, dev))
}

///////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{ave, ave_dev};
    use crate::StrError;
    use russell_chk::assert_approx_eq;

    #[test]
    fn ave_fails_on_empty_slice() {
        let x: [i32; 0] = [];
        assert_eq!(ave(&x), Err("cannot compute average of empty slice"));
    }

    #[test]
    fn ave_works() -> Result<(), StrError> {
        let x = [100, 100, 102, 98, 77, 99, 70, 105, 98];
        assert_eq!(ave(&x)?, 849.0 / 9.0);
        assert_eq!(ave(&x)?, 849.0 / 9.0); // again, to check move vs borrow
        Ok(())
    }

    #[test]
    fn ave_dev_fails_on_wrong_input() {
        let x: [i32; 0] = [];
        assert_eq!(ave_dev(&x), Err("at least two values are needed"));
    }

    #[test]
    fn ave_dev_works() -> Result<(), StrError> {
        let x = [100, 100, 102, 98, 77, 99, 70, 105, 98];
        let (ave, dev) = ave_dev(&x)?;
        assert_eq!(ave, 849.0 / 9.0);
        assert_approx_eq!(dev, 12.134661099511597, 1e-17);
        ave_dev(&x)?; // again, to check move vs borrow
        Ok(())
    }
}
