/// ave computes the average of values in the x slice
///  Input:
///   x -- sample
///  Output:
///   the average
pub fn ave<T>(x: &[T]) -> f64
where
    T: Into<f64> + Copy,
{
    // if x.len() == 0 {
    //     return Err();
    // }
    let sum = x.iter().fold(0.0, |acc, &curr| acc + curr.into());
    let n = x.len() as f64;
    sum / n
}
