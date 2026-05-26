use super::quartiles;
use num_traits::{Num, NumCast};

/// Identifies outliers in the data using the 1.5 * IQR rule.
///
/// An outlier is defined as any data point that falls below Q1 - 1.5*IQR or above Q3 + 1.5*IQR,
/// where Q1 is the first quartile, Q3 is the third quartile, and IQR is the inter-quartile range.
///
/// # Arguments
///
/// * `data` - A reference to a slice of data points. It must be a slice of a
///   type that can be ordered and supports arithmetic operations.
///
/// # Returns
///
/// A vector of tuples `(index, value)` containing the index (in the **original array**)
/// and value of each outlier found in the dataset.
///
/// # Panics
///
/// This function will panic if the input slice is empty.
///
/// # Examples
///
/// ```
/// use russell_stat::outliers;
///
/// // A dataset with clear outliers
/// let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 100];
/// let outlier_info = outliers(&data);
/// // Index 9 is the position of 100 in the original array
/// assert_eq!(outlier_info, vec![(9, 100)]);
///
/// // Unsorted data - indices refer to original positions
/// let data = vec![100, 2, 3, 4, 5, 6, 7, 8, 9, 1];
/// let outlier_info = outliers(&data);
/// // Index 0 is where 100 was in the original array
/// assert_eq!(outlier_info, vec![(0, 100)]);
///
/// // A dataset without outliers
/// let data = vec![1, 2, 3, 4, 5];
/// let outlier_info = outliers(&data);
/// assert_eq!(outlier_info.len(), 0);
/// ```
///
/// # Algorithm
///
/// 1. Calculate Q1 (first quartile) and Q3 (third quartile)
/// 2. Calculate IQR = Q3 - Q1
/// 3. Lower bound = Q1 - 1.5 * IQR
/// 4. Upper bound = Q3 + 1.5 * IQR
/// 5. Any value < lower bound or > upper bound is an outlier
pub fn outliers<T>(data: &[T]) -> Vec<(usize, T)>
where
    T: Num + NumCast + Copy + PartialOrd,
{
    // Create indexed pairs to track original positions
    let mut indexed_data: Vec<(usize, T)> = data.iter().copied().enumerate().collect();

    // Sort by value while keeping track of original indices
    indexed_data.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // Extract sorted values for quartile calculation
    let sorted_values: Vec<T> = indexed_data.iter().map(|(_, v)| *v).collect();

    // Get quartiles and IQR
    let (q1, _, q3) = quartiles(&mut sorted_values.clone());
    let iqr = q3 - q1;

    // Calculate outlier boundaries
    let lower_bound = q1 - 1.5 * iqr;
    let upper_bound = q3 + 1.5 * iqr;

    // Find outliers with their original indices
    let mut outlier_info = Vec::new();
    for &(original_index, value) in indexed_data.iter() {
        let val_f64 = value.to_f64().unwrap();
        if val_f64 < lower_bound || val_f64 > upper_bound {
            outlier_info.push((original_index, value));
        }
    }

    outlier_info
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::outliers;

    #[test]
    #[should_panic(expected = "Input data slice must not be empty")]
    fn outliers_panics_on_empty_input() {
        let data: Vec<i32> = vec![];
        outliers(&data);
    }

    #[test]
    fn outliers_with_one_outlier() {
        // Dataset with one clear outlier
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 100];
        let outlier_info = outliers(&data);
        assert_eq!(outlier_info, vec![(9, 100)]);
    }

    #[test]
    fn outliers_with_no_outliers() {
        // Dataset without outliers
        let data = vec![1, 2, 3, 4, 5];
        let outlier_info = outliers(&data);
        assert_eq!(outlier_info.len(), 0);
    }

    #[test]
    fn outliers_with_multiple_outliers() {
        // Dataset with multiple outliers
        let data = vec![-100, 1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 200];
        let outlier_info = outliers(&data);
        // Lower bound: Q1 - 1.5*IQR, Upper bound: Q3 + 1.5*IQR
        // Should identify -100, 100, and 200 as outliers
        assert_eq!(outlier_info.len(), 3);
        // -100 is at index 0 in original array
        assert!(outlier_info.contains(&(0, -100)));
        // 100 is at index 10 in original array
        assert!(outlier_info.contains(&(10, 100)));
        // 200 is at index 11 in original array
        assert!(outlier_info.contains(&(11, 200)));
    }

    #[test]
    fn outliers_with_uniform_data() {
        // All values are the same, no outliers
        let data = vec![5, 5, 5, 5, 5];
        let outlier_info = outliers(&data);
        assert_eq!(outlier_info.len(), 0);
    }

    #[test]
    fn outliers_with_floats() {
        // Dataset with floating-point values
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 50.0];
        let outlier_info = outliers(&data);
        assert_eq!(outlier_info, vec![(9, 50.0)]);
    }

    #[test]
    fn outliers_with_negative_values() {
        // Dataset with negative outlier
        let data = vec![-100, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let outlier_info = outliers(&data);
        assert_eq!(outlier_info, vec![(0, -100)]);
    }

    #[test]
    fn outliers_boundary_case() {
        // Test values exactly at the boundary
        // For [1,2,3,4,5,6,7,8,9,10]:
        // Q1=3.25, Q3=7.75, IQR=4.5
        // Lower: 3.25 - 1.5*4.5 = -3.5
        // Upper: 7.75 + 1.5*4.5 = 14.5
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let outlier_info = outliers(&data);
        assert_eq!(outlier_info.len(), 0); // All values within bounds
    }

    #[test]
    fn outliers_single_element() {
        // Single element should not be an outlier (IQR = 0)
        let data = vec![42];
        let outlier_info = outliers(&data);
        assert_eq!(outlier_info.len(), 0);
    }

    #[test]
    fn outliers_two_elements() {
        // Two elements, no outliers possible with standard IQR rule
        let data = vec![1, 2];
        let outlier_info = outliers(&data);
        assert_eq!(outlier_info.len(), 0);
    }

    #[test]
    fn outliers_does_not_modify_input() {
        // Verify that the input data is not modified
        let data = vec![9, 1, 5, 3, 7];
        let _ = outliers(&data);
        assert_eq!(data, vec![9, 1, 5, 3, 7]); // Data unchanged
    }

    #[test]
    fn outliers_with_duplicates() {
        // Dataset with duplicates and outliers
        let data = vec![1, 2, 2, 3, 3, 3, 4, 4, 5, 100];
        let outlier_info = outliers(&data);
        assert_eq!(outlier_info, vec![(9, 100)]);
    }

    #[test]
    fn outliers_realistic_example() {
        // More realistic example: test scores
        let scores = vec![65, 70, 72, 75, 78, 80, 82, 85, 88, 90, 15]; // 15 is clearly an outlier
        let outlier_info = outliers(&scores);
        // 15 was at index 10 in the original array
        assert_eq!(outlier_info, vec![(10, 15)]);
    }

    #[test]
    fn outliers_indices_are_from_original_array() {
        // Verify that indices correspond to original positions before sorting
        let data = vec![100, 2, 3, 4, 5, 6, 7, 8, 9, 1];
        let outlier_info = outliers(&data);
        // 100 was at index 0 in the original array
        assert_eq!(outlier_info, vec![(0, 100)]);
        // Verify data is NOT modified
        assert_eq!(data, vec![100, 2, 3, 4, 5, 6, 7, 8, 9, 1]);
    }

    #[test]
    fn outliers_multiple_with_original_indices() {
        // Test multiple outliers with original index tracking
        let data = vec![5, -100, 3, 200, 4, 6, 7];
        let outlier_info = outliers(&data);
        // -100 was at index 1, 200 was at index 3
        assert_eq!(outlier_info.len(), 2);
        assert!(outlier_info.contains(&(1, -100)));
        assert!(outlier_info.contains(&(3, 200)));
    }
}
