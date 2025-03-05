//! This module contains functions for calculations with vectors

mod aliases;
mod complex_vec_add;
mod complex_vec_approx_eq;
mod complex_vec_copy;
mod complex_vec_minus;
mod complex_vec_norm;
mod complex_vec_plus;
mod complex_vec_scale;
mod complex_vec_unzip;
mod complex_vec_update;
mod complex_vec_zip;
mod num_vector;
mod vec_add;
mod vec_all_finite;
mod vec_approx_eq;
mod vec_copy;
mod vec_copy_scaled;
mod vec_fmt_scientific;
mod vec_inner;
mod vec_max_abs_diff;
mod vec_max_scaled;
mod vec_max_scaled_diff;
mod vec_minus;
mod vec_norm;
mod vec_plus;
mod vec_rms_scaled;
mod vec_rms_scaled_diff;
mod vec_scale;
mod vec_to_static_array;
mod vec_update;

pub use aliases::*;
pub use complex_vec_add::*;
pub use complex_vec_approx_eq::*;
pub use complex_vec_copy::*;
pub use complex_vec_minus::*;
pub use complex_vec_norm::*;
pub use complex_vec_plus::*;
pub use complex_vec_scale::*;
pub use complex_vec_unzip::*;
pub use complex_vec_update::*;
pub use complex_vec_zip::*;
pub use num_vector::*;
pub use vec_add::*;
pub use vec_all_finite::*;
pub use vec_approx_eq::*;
pub use vec_copy::*;
pub use vec_copy_scaled::*;
pub use vec_fmt_scientific::*;
pub use vec_inner::*;
pub use vec_max_abs_diff::*;
pub use vec_max_scaled::*;
pub use vec_max_scaled_diff::*;
pub use vec_minus::*;
pub use vec_norm::*;
pub use vec_plus::*;
pub use vec_rms_scaled::*;
pub use vec_rms_scaled_diff::*;
pub use vec_scale::*;
pub use vec_to_static_array::*;
pub use vec_update::*;
