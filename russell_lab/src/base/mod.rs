//! This module implements some auxiliary functionality, not necessarily related to linear algebra

mod as_array;
mod auxiliary_blas;
mod enums;
mod formatters;
mod generators;
mod linear_fitting;
mod macros;
mod read_table;
mod sort;
mod stopwatch;
pub use crate::base::as_array::*;
pub use crate::base::auxiliary_blas::*;
pub use crate::base::enums::*;
pub use crate::base::formatters::*;
pub use crate::base::generators::*;
pub use crate::base::linear_fitting::*;
pub use crate::base::read_table::*;
pub use crate::base::sort::*;
pub use crate::base::stopwatch::*;
