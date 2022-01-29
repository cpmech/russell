//use crate::distribution_frechet::DistributionFrechet;
// Defines the name of a Probability Distribution
// pub enum Pd {
//     Frechet,
//     Gumbel,
//     Lognormal,
//     Normal,
//     Uniform,
// }

// Defines an alias for f(x) functions
//type FnX = fn(f64) -> f64;

// pub struct ProbabilityDistribution {
//     fn_pdf: FnX,
//     fn_cdf: FnX,
// }

// impl ProbabilityDistribution {
//     pub fn new(distribution: Pd) -> Self {
//         let (fn_pdf, fn_cdf) = match distribution {
//             Pd::Frechet => (DistributionFrechet::pdf, DistributionFrechet::cdf),
//             Pd::Gumbel => (DistributionFrechet::pdf, DistributionFrechet::cdf),
//             Pd::Lognormal => (DistributionFrechet::pdf, DistributionFrechet::cdf),
//             Pd::Normal => (DistributionFrechet::pdf, DistributionFrechet::cdf),
//             Pd::Uniform => (DistributionFrechet::pdf, DistributionFrechet::cdf),
//         };
//         ProbabilityDistribution { fn_pdf, fn_cdf }
//     }

//     /// Implements the probability density function (PDF)
//     pub fn probability_density_function(&self, x: f64) -> f64 {
//         0.0
//     }

//     /// Implements the cumulative density function (CDF)
//     pub fn cumulative_density_function(&self, x: f64) -> f64 {
//         0.0
//     }
// }

/// Defines the Probability Distribution trait
pub trait Distribution {
    fn probability_density_function(&self, x: f64) -> f64;
    fn cumulative_density_function(&self, x: f64) -> f64;
}
