use num_traits::{Num, NumCast};

/// Calculates the first, second (median), and third quartiles.
///
/// # Arguments
///
/// * `data` - A mutable reference to a vector of data points. It must be a vector of a
///   type that can be ordered and supports arithmetic operations.
///
/// **Warning**: This function modifies the input vector by sorting it.
///
/// # Notes
///
/// * The function uses the "linear" method of NumPy's `quantile` function.
///
/// # Panics
///
/// This function will panic if the input vector is empty.
pub fn calculate_quartiles<T>(data: &mut [T]) -> (f64, f64, f64)
where
    T: Num + NumCast + Copy + PartialOrd,
{
    // Check for empty input
    let n = data.len();
    if n == 0 {
        panic!("Input data vector must not be empty");
    }

    // Sort the dataset
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Calculate quartiles using linear interpolation (NumPy's default method)
    let q1 = calculate_quantile(data, 0.25);
    let q2 = calculate_quantile(data, 0.50);
    let q3 = calculate_quantile(data, 0.75);

    // Return the quartiles
    (q1, q2, q3)
}

/// Calculates a quantile of a sorted slice using linear interpolation.
///
/// This matches NumPy's default `quantile` method (linear interpolation).
///
/// # Arguments
///
/// * `data` - A sorted slice of data points
/// * `q` - The quantile to calculate (e.g., 0.25 for first quartile, 0.5 for median)
pub fn calculate_quantile<T>(data: &[T], q: f64) -> f64
where
    T: Num + NumCast + Copy,
{
    let n = data.len();

    // Calculate the virtual index using linear interpolation formula
    // This matches NumPy: index = q * (n - 1)
    let index = q * ((n - 1) as f64);
    let lower_index = index.floor() as usize;
    let upper_index = index.ceil() as usize;
    let fraction = index - lower_index as f64;

    // Linear interpolation between the two nearest data points
    let lower_value = data[lower_index].to_f64().unwrap();
    let upper_value = data[upper_index].to_f64().unwrap();

    lower_value + fraction * (upper_value - lower_value)
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::calculate_quartiles;

    #[test]
    fn calculate_quartiles_works() {
        // Python:
        // In:  np.quantile([7, 1, 3, 9, 5],[0.25,0.5,0.75])
        // Out: array([3., 5., 7.])
        let mut data = vec![7, 1, 3, 9, 5];
        let (q1, q2, q3) = calculate_quartiles(&mut data);
        assert_eq!(q1, 3.0);
        assert_eq!(q2, 5.0);
        assert_eq!(q3, 7.0);

        // Example with an odd number of elements
        // Python:
        // In : np.quantile([17, 2, 5, 11, 14, 8, 20],[0.25,0.5,0.75])
        // Out: array([ 6.5, 11. , 15.5])
        let mut data_odd = vec![17, 2, 5, 11, 14, 8, 20];
        let (q1, q2, q3) = calculate_quartiles(&mut data_odd);
        assert_eq!(q1, 6.5);
        assert_eq!(q2, 11.0);
        assert_eq!(q3, 15.5);

        // Example with an even number of elements
        // Python:
        // In:  np.quantile([2, 14, 17, 20, 5, 8, 11, 23],[0.25,0.5,0.75])
        // Out: array([ 7.25, 12.5 , 17.75])
        let mut data_even = vec![2, 14, 17, 20, 5, 8, 11, 23];
        let (q1, q2, q3) = calculate_quartiles(&mut data_even);
        assert_eq!(q1, 7.25);
        assert_eq!(q2, 12.5);
        assert_eq!(q3, 17.75);
    }
}
