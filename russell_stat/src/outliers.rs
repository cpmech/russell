use super::quartiles;
use num_traits::{Num, NumCast};

/// Identifies outliers in the data using the 1.5 * IQR rule.
///
/// An outlier is defined as any data point that falls below Q1 - 1.5*IQR or above Q3 + 1.5*IQR,
/// where Q1 is the first quartile, Q3 is the third quartile, and IQR is the inter-quartile range.
///
/// # Arguments
///
/// * `data` - A mutable reference to a slice of data points. It must be a slice of a
///   type that can be ordered and supports arithmetic operations.
///
/// **Warning**: This function modifies the input slice by sorting it.
///
/// # Returns
///
/// A vector containing all outliers found in the dataset.
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
/// let mut data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 100];
/// let outlier_values = outliers(&mut data);
/// assert_eq!(outlier_values, vec![100]);
///
/// // A dataset without outliers
/// let mut data = vec![1, 2, 3, 4, 5];
/// let outlier_values = outliers(&mut data);
/// assert_eq!(outlier_values.len(), 0);
/// ```
///
/// # Algorithm
///
/// 1. Calculate Q1 (first quartile) and Q3 (third quartile)
/// 2. Calculate IQR = Q3 - Q1
/// 3. Lower bound = Q1 - 1.5 * IQR
/// 4. Upper bound = Q3 + 1.5 * IQR
/// 5. Any value < lower bound or > upper bound is an outlier
pub fn outliers<T>(data: &mut [T]) -> Vec<T>
where
    T: Num + NumCast + Copy + PartialOrd,
{
    // Get quartiles and IQR
    let (q1, _, q3) = quartiles(data);
    let iqr = q3 - q1;

    // Calculate outlier boundaries
    let lower_bound = q1 - 1.5 * iqr;
    let upper_bound = q3 + 1.5 * iqr;

    // Find outliers
    let mut outlier_values = Vec::new();
    for &value in data.iter() {
        let val_f64 = value.to_f64().unwrap();
        if val_f64 < lower_bound || val_f64 > upper_bound {
            outlier_values.push(value);
        }
    }

    outlier_values
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::outliers;

    #[test]
    #[should_panic(expected = "Input data slice must not be empty")]
    fn outliers_panics_on_empty_input() {
        let mut data: Vec<i32> = vec![];
        outliers(&mut data);
    }

    #[test]
    fn outliers_with_one_outlier() {
        // Dataset with one clear outlier
        let mut data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 100];
        let outlier_values = outliers(&mut data);
        assert_eq!(outlier_values, vec![100]);
    }

    #[test]
    fn outliers_with_no_outliers() {
        // Dataset without outliers
        let mut data = vec![1, 2, 3, 4, 5];
        let outlier_values = outliers(&mut data);
        assert_eq!(outlier_values.len(), 0);
    }

    #[test]
    fn outliers_with_multiple_outliers() {
        // Dataset with multiple outliers
        let mut data = vec![-100, 1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 200];
        let outlier_values = outliers(&mut data);
        // Lower bound: Q1 - 1.5*IQR, Upper bound: Q3 + 1.5*IQR
        // Should identify -100, 100, and 200 as outliers
        assert!(outlier_values.contains(&-100));
        assert!(outlier_values.contains(&100));
        assert!(outlier_values.contains(&200));
    }

    #[test]
    fn outliers_with_uniform_data() {
        // All values are the same, no outliers
        let mut data = vec![5, 5, 5, 5, 5];
        let outlier_values = outliers(&mut data);
        assert_eq!(outlier_values.len(), 0);
    }

    #[test]
    fn outliers_with_floats() {
        // Dataset with floating-point values
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 50.0];
        let outlier_values = outliers(&mut data);
        assert_eq!(outlier_values, vec![50.0]);
    }

    #[test]
    fn outliers_with_negative_values() {
        // Dataset with negative outlier
        let mut data = vec![-100, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let outlier_values = outliers(&mut data);
        assert_eq!(outlier_values, vec![-100]);
    }

    #[test]
    fn outliers_boundary_case() {
        // Test values exactly at the boundary
        // For [1,2,3,4,5,6,7,8,9,10]:
        // Q1=3.25, Q3=7.75, IQR=4.5
        // Lower: 3.25 - 1.5*4.5 = -3.5
        // Upper: 7.75 + 1.5*4.5 = 14.5
        let mut data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let outlier_values = outliers(&mut data);
        assert_eq!(outlier_values.len(), 0); // All values within bounds
    }

    #[test]
    fn outliers_single_element() {
        // Single element should not be an outlier (IQR = 0)
        let mut data = vec![42];
        let outlier_values = outliers(&mut data);
        assert_eq!(outlier_values.len(), 0);
    }

    #[test]
    fn outliers_two_elements() {
        // Two elements, no outliers possible with standard IQR rule
        let mut data = vec![1, 2];
        let outlier_values = outliers(&mut data);
        assert_eq!(outlier_values.len(), 0);
    }

    #[test]
    fn outliers_verifies_sorting() {
        // Verify that the data is sorted after the function call
        let mut data = vec![9, 1, 5, 3, 7];
        let _ = outliers(&mut data);
        assert_eq!(data, vec![1, 3, 5, 7, 9]);
    }

    #[test]
    fn outliers_with_duplicates() {
        // Dataset with duplicates and outliers
        let mut data = vec![1, 2, 2, 3, 3, 3, 4, 4, 5, 100];
        let outlier_values = outliers(&mut data);
        assert_eq!(outlier_values, vec![100]);
    }

    #[test]
    fn outliers_realistic_example() {
        // More realistic example: test scores
        let mut scores = vec![65, 70, 72, 75, 78, 80, 82, 85, 88, 90, 15]; // 15 is clearly an outlier
        let outlier_values = outliers(&mut scores);
        assert_eq!(outlier_values, vec![15]);
    }
}
