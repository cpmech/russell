use super::err;

/// Returns the average of values in the a given slice
///
/// # Examples
///
/// ```
/// use russell_stat;
/// let x = [1, 2, 3, 4];
/// let result = russell_stat::ave(&x);
/// assert_eq!(result.unwrap(), 2.5);
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
