// errors ///////////////////////////////////////
#[macro_use]
extern crate error_chain;
mod errors {
    error_chain! {}
}
pub use errors::*;

// tests ////////////////////////////////////////
#[cfg(test)]
#[path = "./stat_test.rs"]
mod stat_test;

// modules //////////////////////////////////////
pub mod stat;
