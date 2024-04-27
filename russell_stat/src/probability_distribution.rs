use rand::Rng;

/// Defines the Probability Distribution trait
pub trait ProbabilityDistribution {
    /// Evaluates the Probability Density Function (PDF)
    ///
    /// The probability density function `f(x)` is such that
    /// (see Eq 9 on page 1033 of the Reference):
    ///
    /// ```text
    ///                 b
    ///                ⌠
    /// P(a < X ≤ b) = │ f(v) dv = F(b) - F(a)
    ///                ⌡
    ///               a
    ///
    /// with b > a
    /// ```
    ///
    /// where `X` is the continuous random variable, `P(a < X ≤ b)` is the probability that `X` is in the
    /// semi-open interval `(a, b]`, and `F(x)` is the cumulative probability distribution (CDF).
    ///
    /// Note that, for continuous random variables, the following probabilities are all the same
    /// (page 1033 of the reference):
    ///
    /// ```text
    /// prob = P(a < X < b)
    ///      = P(a < X ≤ b)
    ///      = P(a ≤ X < b)
    ///      = P(a ≤ X ≤ b)
    /// ```
    ///
    /// # References
    ///
    /// * Kreyszig, E (2011) Advanced engineering mathematics; in collaboration with Kreyszig H,
    ///    Edward JN 10th ed 2011, Hoboken, New Jersey, Wiley
    fn pdf(&self, x: f64) -> f64;

    /// Evaluates the Cumulative Distribution Function (CDF)
    ///
    /// The cumulative distribution function (or simply *distribution*) `F(x)` is such that
    /// (see Eq 1 on page 1029 of the Reference):
    ///
    /// ```text
    ///             x
    ///            ⌠
    /// P(X ≤ x) = │ f(v) dv = F(x)
    ///            ⌡
    ///          -∞
    /// ```
    ///
    /// where `X` is the continuous random variable, `P(X ≤ x)` is the probability that `X`
    /// assumes values not exceeding `x`, and `f(x)` is the  probability density function (PDF).
    ///
    /// # References
    ///
    /// * Kreyszig, E (2011) Advanced engineering mathematics; in collaboration with Kreyszig H,
    ///    Edward JN 10th ed 2011, Hoboken, New Jersey, Wiley
    fn cdf(&self, x: f64) -> f64;

    /// Returns the Mean
    fn mean(&self) -> f64;

    /// Returns the Variance
    fn variance(&self) -> f64;

    /// Generates a pseudo-random number belonging to this probability distribution
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64;
}
