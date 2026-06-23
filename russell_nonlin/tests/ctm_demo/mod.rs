pub type StrError = &'static str;

mod dahlquist;
pub mod enums;
mod hardening_softening;
pub mod model;
mod model_trait;

pub use dahlquist::*;
pub use enums::*;
use hardening_softening::*;
pub use model::*;
use model_trait::*;
