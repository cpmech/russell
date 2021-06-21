use super::err;

/// Returns the average of values in the a given slice
///
/// # Examples
///
/// ```
/// use russell_stat;
/// let x = [2, 4, 4, 4, 5, 5, 7, 9];
/// let result = russell_stat::ave(&x);
/// assert_eq!(result.unwrap(), 5.0);
/// ```
pub fn ave<T>(x: &[T]) -> err::Result<f64>
where
    T: Into<f64> + Copy,
{
    if x.len() == 0 {
        bail!("cannot compute average of empty slice");
    }
    let sum = x.iter().fold(0.0, |acc, &curr| acc + curr.into());
    let n = x.len() as f64;
    Ok(sum / n)
}

/// Returns the average of values and their standard deviation
///
/// # Examples
///
/// ```
/// use russell_stat;
/// let x = [2, 4, 4, 4, 5, 5, 7, 9];
/// let result = russell_stat::ave_dev(&x);
/// let (ave, dev) = result.unwrap();
/// assert_eq!(ave, 5.0);
/// assert_eq!(dev, ((32.0/7.0) as f64).sqrt());
/// ```
pub fn ave_dev<T>(x: &[T]) -> err::Result<(f64, f64)>
where
    T: Into<f64> + Copy,
{
    if x.len() < 2 {
        bail!("at least two values are needed");
    }

    // average
    let sum = x.iter().fold(0.0, |acc, &curr| acc + curr.into());
    let n = x.len() as f64;
    let ave = sum / n;

    // variance
    let mut c = 0.0; // corrector
    let mut vari = 0.0; // variance
    for &val in x {
        let d = val.into() - ave; // d ← xi - bar(x)
        c += d; // c ← Σ d  (corrector)
        vari += d * d; // vari ← Σ d²
    }

    // standard deviation
    vari = (vari - c * c / n) / (n - 1.0);
    let dev = vari.sqrt();

    Ok((ave, dev))
}
