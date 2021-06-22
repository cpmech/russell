//! Russell - Rust Scientific Library
//!
//! **lab**: matrix-vector "laboratory" tools

// errors ///////////////////////////////////////
#[macro_use]
extern crate error_chain;
mod err {
    error_chain! {}
}

// modules //////////////////////////////////////
mod linspace;
pub use crate::linspace::*;
