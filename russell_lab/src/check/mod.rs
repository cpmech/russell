//! This module contains functions to compare float numbers and arrays for unit testing
//!
//! # Examples
//!
//! ### Check float point numbers (real)
//!
//! ```rust
//! use russell_lab::*;
//!
//! fn main() {
//!     approx_eq(0.123456789, 0.12345678, 1e-8);
//!     approx_eq(0.123456789, 0.1234567, 1e-7);
//!     approx_eq(0.123456789, 0.123456, 1e-6);
//!     approx_eq(0.123456789, 0.12345, 1e-5);
//!     approx_eq(0.123456789, 0.1234, 1e-4);
//! }
//! ```
//!
//! ### Check float point numbers (complex)
//!
//! ```
//! use num_complex::Complex64;
//! use russell_lab::*;
//!
//! fn main() {
//!     // check float point number
//!     approx_eq(0.0000123, 0.000012, 1e-6);
//!
//!     // check vector of float point numbers
//!     array_approx_eq(&[0.01, 0.012], &[0.012, 0.01], 1e-2);
//!
//!     // check derivative using central differences
//!     struct Arguments {}
//!     let f = |x: f64, _: &mut Arguments| Ok(-x);
//!     let args = &mut Arguments {};
//!     let at_x = 8.0;
//!     let dfdx = -1.01;
//!     deriv_approx_eq(dfdx, at_x, args, 1e-2, f);
//!
//!     // check complex numbers
//!     complex_approx_eq(Complex64::new(1.0, 8.0), Complex64::new(1.001, 8.0), 1e-2);
//! }
//! ```
//!
//! ### Check vectors of float point numbers (real)
//!
//! ```rust
//! use russell_lab::*;
//!
//! fn main() {
//!     let a = [0.123456789, 0.123456789, 0.123456789];
//!     let b = [0.12345678,  0.1234567,   0.123456];
//!     array_approx_eq(&a, &b, 1e-6);
//! }
//! ```
//!
//! ### Check vectors of float point numbers (complex)
//!
//! ```rust
//! use num_complex::Complex64;
//! use russell_lab::*;
//!
//! fn main() {
//!     let a = &[
//!         Complex64::new(0.123456789, 5.01),
//!         Complex64::new(0.123456789, 5.01),
//!         Complex64::new(0.123456789, 5.01)];
//!     let b = &[
//!         Complex64::new(0.12345678, 5.01),
//!         Complex64::new(0.1234567, 5.01),
//!         Complex64::new(0.123456, 5.01)];
//!     complex_array_approx_eq(a, b, 1e-6);
//! }
//! ```
//!
//! ### Check derivatives using finite differences
//!
//! ```rust
//! use russell_lab::*;
//!
//! struct Arguments {}
//!
//! fn main() {
//!     let f = |x: f64, _: &mut Arguments| Ok(-x);
//!     let args = &mut Arguments {};
//!     let at_x = 8.0;
//!     let dfdx = -1.01;
//!     deriv_approx_eq(dfdx, at_x, args, 1e-2, f);
//! }
//! ```

mod approx_eq;
mod array_approx_eq;
mod assert_alike;
mod complex_approx_eq;
mod complex_array_approx_eq;
mod deriv_approx_eq;
mod num_deriv;
pub use crate::check::approx_eq::*;
pub use crate::check::array_approx_eq::*;
pub use crate::check::assert_alike::*;
pub use crate::check::complex_approx_eq::*;
pub use crate::check::complex_array_approx_eq::*;
pub use crate::check::deriv_approx_eq::*;
pub use crate::check::num_deriv::*;
