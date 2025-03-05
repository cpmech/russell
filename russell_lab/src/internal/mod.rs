//! This module implements some functions for internal use

mod array_minus_op;
mod array_plus_op;
mod array_plus_opx;
mod constants;
mod dgeev_data;
mod to_i32;
pub(crate) use array_minus_op::*;
pub(crate) use array_plus_op::*;
pub(crate) use array_plus_opx::*;
pub(crate) use constants::*;
pub(crate) use dgeev_data::*;
pub(crate) use to_i32::*;
