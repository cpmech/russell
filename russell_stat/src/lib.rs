//! Russell - Rust Scientific Library
//!
//! `russell_stat`: Statistics calculations and (engineering) probability distributions
//!
//! **Important:** This crate depends on external libraries (non-Rust). Thus, please check the [Installation Instructions on the GitHub Repository](https://github.com/cpmech/russell).
//!
//! This library assists in developing statistical computations and simulations aiming at engineering applications, such as reliability analyses. This library provides a light interface to [rand_distr](https://crates.io/crates/rand_distr) and implements extra functionality.
//!
//! Some essential distributions considered in this library are those classified as [Extreme Value Distribution](https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution).

/// Defines the error output as a static string
pub type StrError = &'static str;

mod distribution_frechet;
mod distribution_gumbel;
mod distribution_lognormal;
mod distribution_normal;
mod distribution_uniform;
mod histogram;
mod probability_distribution;
mod statistics;
pub use crate::distribution_frechet::*;
pub use crate::distribution_gumbel::*;
pub use crate::distribution_lognormal::*;
pub use crate::distribution_normal::*;
pub use crate::distribution_uniform::*;
pub use crate::histogram::*;
pub use crate::probability_distribution::*;
pub use crate::statistics::*;

// run code from README file
#[cfg(doctest)]
mod test_readme {
    macro_rules! external_doc_test {
        ($x:expr) => {
            #[doc = $x]
            extern "C" {}
        };
    }
    external_doc_test!(include_str!("../README.md"));
}
