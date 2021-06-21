//! Russell - Rust Scientific Library
//!
//! **stat**: Statistics, Probabilites, and Random Numbers

// auxiliary ////////////////////////////////////
pub use assert_approx_eq::*;

// errors ///////////////////////////////////////
#[macro_use]
extern crate error_chain;
mod err {
    error_chain! {}
}

// tests ////////////////////////////////////////
#[cfg(test)]
#[path = "./stat_test.rs"]
mod stat_test;

// modules //////////////////////////////////////
mod stat;
pub use crate::stat::*;
