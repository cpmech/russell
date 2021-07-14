//! Russell - Rust Scientific Library
//!
//! **tensor**: tensor analysis tools

// errors ///////////////////////////////////////
// #[macro_use]
// extern crate error_chain;
// mod err {
//     error_chain! {}
// }

// modules //////////////////////////////////////
mod constants;
mod tensor2;
pub use crate::constants::*;
pub use crate::tensor2::*;
