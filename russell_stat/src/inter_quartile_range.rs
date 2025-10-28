use super::quartiles;
use num_traits::{Num, NumCast};

/// Calculates the inter-quartile range (IQR) of a dataset.
pub fn inter_quartile_range<T>(data: &mut [T]) -> f64
where
    T: Num + NumCast + Copy + PartialOrd,
{
    let (q1, _, q3) = quartiles(data);
    q3 - q1
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::inter_quartile_range;
    use crate::quartiles;

    #[test]
    fn calculate_inter_quartile_range_works() {
        // Python:
        // scipy.stats.iqr([10.0, 7.0, 4.0])
        assert_eq!(inter_quartile_range(&mut [10.0, 7.0, 4.0]), 3.0);

        // Python:
        // scipy.stats.iqr([7, 1, 3, 9, 5])
        assert_eq!(inter_quartile_range(&mut [7, 1, 3, 9, 5]), 4.0);

        // Python:
        // scipy.stats.iqr([2, 4, 18, 26, 3, 3, 7, 5, 5, 12])
        let data = &mut [2, 4, 18, 26, 3, 3, 7, 5, 5, 12];
        println!("quartiles = {:?}", quartiles(data));
        assert_eq!(inter_quartile_range(data), 7.5);
        println!("data = {:?}", data);
    }
}
