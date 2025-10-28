use num_traits::{Num, NumCast};

/// Calculates a quantile of a sorted slice using linear interpolation.
///
/// This matches NumPy's default `quantile` method (linear interpolation).
///
/// # Arguments
///
/// * `data` - A sorted slice of data points
/// * `q` - The quantile to calculate (e.g., 0.25 for first quartile, 0.5 for median)
///
/// # Panics
///
/// This function will panic if the input slice is empty.
pub fn calculate_quantile<T>(data: &[T], q: f64) -> f64
where
    T: Num + NumCast + Copy,
{
    let n = data.len();
    if n == 0 {
        panic!("Input data slice must not be empty");
    }

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
        panic!("Input data slice must not be empty");
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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{calculate_quantile, calculate_quartiles};

    #[test]
    #[should_panic(expected = "Input data slice must not be empty")]
    fn calculate_quantile_panics_on_empty_input() {
        let data: Vec<i32> = vec![];
        calculate_quantile(&data, 0.5);
    }

    #[test]
    fn calculate_quantile_works() {
        // Test with small sorted dataset
        let data = vec![1, 2, 3, 4, 5];

        // Min (0th percentile)
        let q0 = calculate_quantile(&data, 0.0);
        assert_eq!(q0, 1.0);

        // 25th percentile
        let q25 = calculate_quantile(&data, 0.25);
        assert_eq!(q25, 2.0); // 0.25 * 4 = 1.0 -> data[1] = 2

        // Median (50th percentile)
        let q50 = calculate_quantile(&data, 0.5);
        assert_eq!(q50, 3.0); // 0.5 * 4 = 2.0 -> data[2] = 3

        // 75th percentile
        let q75 = calculate_quantile(&data, 0.75);
        assert_eq!(q75, 4.0); // 0.75 * 4 = 3.0 -> data[3] = 4

        // Max (100th percentile)
        let q100 = calculate_quantile(&data, 1.0);
        assert_eq!(q100, 5.0);

        // Others
        assert_eq!(calculate_quantile(&data, 0.1), 1.4);
        assert!(f64::abs(calculate_quantile(&data, 0.33) - 2.3200000000000003) < 1e-14);
    }

    #[test]
    fn calculate_quantile_with_interpolation() {
        // Test with data requiring interpolation
        let data = vec![10, 20, 30, 40, 50, 60];

        // 25th percentile: index = 0.25 * 5 = 1.25
        // Interpolate between data[1]=20 and data[2]=30
        // Result: 20 + 0.25 * (30 - 20) = 22.5
        let q25 = calculate_quantile(&data, 0.25);
        assert_eq!(q25, 22.5);

        // 75th percentile: index = 0.75 * 5 = 3.75
        // Interpolate between data[3]=40 and data[4]=50
        // Result: 40 + 0.75 * (50 - 40) = 47.5
        let q75 = calculate_quantile(&data, 0.75);
        assert_eq!(q75, 47.5);
    }

    #[test]
    fn calculate_quantile_with_even_length() {
        // Test median calculation with even number of elements
        let data = vec![1, 2, 3, 4];

        // Median: index = 0.5 * 3 = 1.5
        // Interpolate between data[1]=2 and data[2]=3
        // Result: 2 + 0.5 * (3 - 2) = 2.5
        let median = calculate_quantile(&data, 0.5);
        assert_eq!(median, 2.5);
    }

    #[test]
    fn calculate_quantile_with_odd_length() {
        // Test median calculation with odd number of elements
        let data = vec![1, 2, 3, 4, 5];

        // Median: index = 0.5 * 4 = 2.0
        // data[2] = 3 (no interpolation needed)
        let median = calculate_quantile(&data, 0.5);
        assert_eq!(median, 3.0);
    }

    #[test]
    fn calculate_quantile_single_element() {
        // Test with single element
        let data = vec![42];

        // Any quantile should return the single element
        assert_eq!(calculate_quantile(&data, 0.0), 42.0);
        assert_eq!(calculate_quantile(&data, 0.25), 42.0);
        assert_eq!(calculate_quantile(&data, 0.5), 42.0);
        assert_eq!(calculate_quantile(&data, 0.75), 42.0);
        assert_eq!(calculate_quantile(&data, 1.0), 42.0);
    }

    #[test]
    fn calculate_quantile_two_elements() {
        // Test with two elements
        let data = vec![10, 20];

        // Min
        assert_eq!(calculate_quantile(&data, 0.0), 10.0);

        // 25th percentile: index = 0.25 * 1 = 0.25
        // Interpolate: 10 + 0.25 * (20 - 10) = 12.5
        assert_eq!(calculate_quantile(&data, 0.25), 12.5);

        // Median: index = 0.5 * 1 = 0.5
        // Interpolate: 10 + 0.5 * (20 - 10) = 15.0
        assert_eq!(calculate_quantile(&data, 0.5), 15.0);

        // 75th percentile: index = 0.75 * 1 = 0.75
        // Interpolate: 10 + 0.75 * (20 - 10) = 17.5
        assert_eq!(calculate_quantile(&data, 0.75), 17.5);

        // Max
        assert_eq!(calculate_quantile(&data, 1.0), 20.0);
    }

    #[test]
    fn calculate_quantile_matches_numpy_example() {
        // Verify against NumPy reference values
        // Python: np.quantile([2, 5, 8, 11, 14, 17, 20, 23], [0.25, 0.5, 0.75])
        // Output: [7.25, 12.5, 17.75]
        let data = vec![2, 5, 8, 11, 14, 17, 20, 23];

        assert_eq!(calculate_quantile(&data, 0.25), 7.25);
        assert_eq!(calculate_quantile(&data, 0.5), 12.5);
        assert_eq!(calculate_quantile(&data, 0.75), 17.75);
    }

    #[test]
    #[should_panic(expected = "Input data slice must not be empty")]
    fn calculate_quartiles_panics_on_empty_input() {
        let mut data: Vec<i32> = vec![];
        calculate_quartiles(&mut data);
    }

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
