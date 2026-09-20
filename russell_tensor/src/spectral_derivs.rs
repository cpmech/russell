#![allow(unused)]

use crate::StrError;
use crate::{EigenProjsT2, Tensor4};

/// Assists in calculating the derivatives of the eigenprojectors
pub struct EigenDerivsT2 {
    eig: EigenProjsT2,
}

impl EigenDerivsT2 {
    /// Allocates a new instance
    pub fn new() -> Self {
        EigenDerivsT2 {
            eig: EigenProjsT2::new(),
        }
    }

    pub fn calculate_mx(&mut self, dpp: &mut [Tensor4<6>; 3]) -> Result<(), StrError> {
        Ok(())
    }
}
