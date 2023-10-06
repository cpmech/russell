//! This module implements some functions for internal use

mod add_arrays;
mod constants;
mod dgeev_data;
mod to_i32;
pub(crate) use crate::internal::add_arrays::*;
pub(crate) use crate::internal::constants::*;
pub(crate) use crate::internal::dgeev_data::*;
pub(crate) use crate::internal::to_i32::*;
