use super::quantile;
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
///
/// # Warnings
///
/// This function does not check for NaNs or Infinities in the input data.
///
/// # Examples
///
/// ```
/// use russell_stat::quartiles;
///
/// // Python Numpy example:
/// // np.quantile([1,2,3,4,5,6,7,8,9,10], [0.25, 0.5, 0.75])
///
/// // Define the dataset
/// let mut data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
///
/// // Calculate the quartiles
/// let (q1, q2, q3) = quartiles(&mut data);
/// assert_eq!(q1, 3.25);
/// assert_eq!(q2, 5.5);
/// assert_eq!(q3, 7.75);
/// ```
pub fn quartiles<T>(data: &mut [T]) -> (f64, f64, f64)
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
    let q1 = quantile(data, 0.25);
    let q2 = quantile(data, 0.50);
    let q3 = quantile(data, 0.75);

    // Return the quartiles
    (q1, q2, q3)
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::quartiles;

    #[test]
    #[should_panic(expected = "Input data slice must not be empty")]
    fn calculate_quartiles_panics_on_empty_input() {
        let mut data: Vec<i32> = vec![];
        quartiles(&mut data);
    }

    #[test]
    fn calculate_quartiles_works() {
        // Python:
        // In:  np.quantile([7, 1, 3, 9, 5],[0.25,0.5,0.75])
        // Out: array([3., 5., 7.])
        let mut data = vec![7, 1, 3, 9, 5];
        let (q1, q2, q3) = quartiles(&mut data);
        assert_eq!(q1, 3.0);
        assert_eq!(q2, 5.0);
        assert_eq!(q3, 7.0);

        // Example with an odd number of elements
        // Python:
        // In : np.quantile([17, 2, 5, 11, 14, 8, 20],[0.25,0.5,0.75])
        // Out: array([ 6.5, 11. , 15.5])
        let mut data_odd = vec![17, 2, 5, 11, 14, 8, 20];
        let (q1, q2, q3) = quartiles(&mut data_odd);
        assert_eq!(q1, 6.5);
        assert_eq!(q2, 11.0);
        assert_eq!(q3, 15.5);

        // Example with an even number of elements
        // Python:
        // In:  np.quantile([2, 14, 17, 20, 5, 8, 11, 23],[0.25,0.5,0.75])
        // Out: array([ 7.25, 12.5 , 17.75])
        let mut data_even = vec![2, 14, 17, 20, 5, 8, 11, 23];
        let (q1, q2, q3) = quartiles(&mut data_even);
        assert_eq!(q1, 7.25);
        assert_eq!(q2, 12.5);
        assert_eq!(q3, 17.75);
    }

    #[test]
    fn calculate_quartiles_single_element() {
        // With only one element, all quartiles should be the same
        let mut data = vec![42];
        let (q1, q2, q3) = quartiles(&mut data);
        assert_eq!(q1, 42.0);
        assert_eq!(q2, 42.0);
        assert_eq!(q3, 42.0);
    }

    #[test]
    fn calculate_quartiles_two_elements() {
        // Python: np.quantile([10, 20], [0.25, 0.5, 0.75])
        // Output: [12.5, 15.0, 17.5]
        let mut data = vec![10, 20];
        let (q1, q2, q3) = quartiles(&mut data);
        assert_eq!(q1, 12.5);
        assert_eq!(q2, 15.0);
        assert_eq!(q3, 17.5);
    }

    #[test]
    fn calculate_quartiles_three_elements() {
        // Python: np.quantile([1, 2, 3], [0.25, 0.5, 0.75])
        // Output: [1.5, 2.0, 2.5]
        let mut data = vec![1, 2, 3];
        let (q1, q2, q3) = quartiles(&mut data);
        assert_eq!(q1, 1.5);
        assert_eq!(q2, 2.0);
        assert_eq!(q3, 2.5);
    }

    #[test]
    fn calculate_quartiles_four_elements() {
        // Python: np.quantile([1, 2, 3, 4], [0.25, 0.5, 0.75])
        // Output: [1.75, 2.5, 3.25]
        let mut data = vec![1, 2, 3, 4];
        let (q1, q2, q3) = quartiles(&mut data);
        assert_eq!(q1, 1.75);
        assert_eq!(q2, 2.5);
        assert_eq!(q3, 3.25);
    }

    #[test]
    fn calculate_quartiles_with_floats() {
        // Python: np.quantile([1.5, 2.7, 3.2, 4.8, 5.1], [0.25, 0.5, 0.75])
        // Output: [2.7, 3.2, 4.8]
        let mut data = vec![1.5, 2.7, 3.2, 4.8, 5.1];
        let (q1, q2, q3) = quartiles(&mut data);
        assert_eq!(q1, 2.7);
        assert_eq!(q2, 3.2);
        assert_eq!(q3, 4.8);
    }

    #[test]
    fn calculate_quartiles_with_negative_values() {
        // Python: np.quantile([-10, -5, 0, 5, 10], [0.25, 0.5, 0.75])
        // Output: [-5.0, 0.0, 5.0]
        let mut data = vec![-10, -5, 0, 5, 10];
        let (q1, q2, q3) = quartiles(&mut data);
        assert_eq!(q1, -5.0);
        assert_eq!(q2, 0.0);
        assert_eq!(q3, 5.0);
    }

    #[test]
    fn calculate_quartiles_with_duplicates() {
        // Python: np.quantile([1, 2, 2, 2, 3, 4, 5], [0.25, 0.5, 0.75])
        // Output: [2.0, 2.0, 3.5]
        let mut data = vec![1, 2, 2, 2, 3, 4, 5];
        let (q1, q2, q3) = quartiles(&mut data);
        assert_eq!(q1, 2.0);
        assert_eq!(q2, 2.0);
        assert_eq!(q3, 3.5);
    }

    #[test]
    fn calculate_quartiles_all_same_values() {
        // When all values are the same, all quartiles should be that value
        let mut data = vec![5, 5, 5, 5, 5];
        let (q1, q2, q3) = quartiles(&mut data);
        assert_eq!(q1, 5.0);
        assert_eq!(q2, 5.0);
        assert_eq!(q3, 5.0);
    }

    #[test]
    fn calculate_quartiles_unsorted_input() {
        // Test that the function correctly sorts unsorted data
        // Python: np.quantile([100, 1, 50, 25, 75], [0.25, 0.5, 0.75])
        // Output: [25.0, 50.0, 75.0]
        let mut data = vec![100, 1, 50, 25, 75];
        let (q1, q2, q3) = quartiles(&mut data);
        assert_eq!(q1, 25.0);
        assert_eq!(q2, 50.0);
        assert_eq!(q3, 75.0);
    }

    #[test]
    fn calculate_quartiles_large_dataset() {
        // Test with a larger dataset
        // Python: np.quantile(list(range(1, 101)), [0.25, 0.5, 0.75])
        // Output: [25.75, 50.5, 75.25]
        let mut data: Vec<i32> = (1..=100).collect();
        let (q1, q2, q3) = quartiles(&mut data);
        assert_eq!(q1, 25.75);
        assert_eq!(q2, 50.5);
        assert_eq!(q3, 75.25);
    }

    #[test]
    fn calculate_quartiles_verifies_sorting() {
        // Verify that the data is actually sorted after the function call
        let mut data = vec![5, 2, 8, 1, 9, 3, 7];
        let _ = quartiles(&mut data);
        assert_eq!(data, vec![1, 2, 3, 5, 7, 8, 9]);
    }
}
