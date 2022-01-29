/// Defines the Probability Distribution trait
pub trait Distribution {
    /// Implements the Probability Density Function (CDF)
    fn pdf(&self, x: f64) -> f64;

    /// Implements the Cumulative Density Function (CDF)
    fn cdf(&self, x: f64) -> f64;
}
