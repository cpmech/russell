//! Russell - Rust Scientific Library
//!
//! **tensor**: tensor analysis tools for continuum mechanics

// errors ///////////////////////////////////////
// #[macro_use]
// extern crate error_chain;
// mod err {
//     error_chain! {}
// }

// modules //////////////////////////////////////
mod constants;
mod tensor2;
mod tensor4;
pub use crate::constants::*;
pub use crate::tensor2::*;
pub use crate::tensor4::*;
