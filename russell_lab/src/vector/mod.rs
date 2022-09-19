//! This module contains functions for calculations with vectors

mod aliases;
mod complex_vec_add;
mod complex_vec_copy;
mod complex_vec_zip;
mod num_vector;
mod vec_add;
mod vec_copy;
mod vec_inner;
mod vec_max_abs_diff;
mod vec_norm;
mod vec_rms_scaled;
mod vec_scale;
mod vec_update;
pub use crate::vector::aliases::*;
pub use crate::vector::complex_vec_add::*;
pub use crate::vector::complex_vec_copy::*;
pub use crate::vector::complex_vec_zip::*;
pub use crate::vector::num_vector::*;
pub use crate::vector::vec_add::*;
pub use crate::vector::vec_copy::*;
pub use crate::vector::vec_inner::*;
pub use crate::vector::vec_max_abs_diff::*;
pub use crate::vector::vec_norm::*;
pub use crate::vector::vec_rms_scaled::*;
pub use crate::vector::vec_scale::*;
pub use crate::vector::vec_update::*;
