//! Russell - Rust Scientific Library
//!
//! **stat**: Statistics calculations, probability distributions, and pseudo random numbers

/// Returns package description
pub fn desc() -> String {
    "Statistics calculations, probability distributions, and pseudo random numbers".to_string()
}

/// Defines a type alias for the error type as a static string
pub type StrError = &'static str;

mod constants;
mod distribution_frechet;
mod distribution_gumbel;
mod distribution_lognormal;
mod distribution_normal;
mod distribution_uniform;
mod histogram;
mod math;
mod probability_distribution;
mod statistics;
pub use crate::constants::*;
pub use crate::distribution_frechet::*;
pub use crate::distribution_gumbel::*;
pub use crate::distribution_lognormal::*;
pub use crate::distribution_normal::*;
pub use crate::distribution_uniform::*;
pub use crate::histogram::*;
pub use crate::math::*;
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
