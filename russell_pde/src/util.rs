use crate::StrError;
use crate::{EssentialBcs1d, NaturalBcs1d};

/// Kronecker delta function
#[inline]
pub(crate) fn delta(i: usize, j: usize) -> f64 {
    if i == j {
        1.0
    } else {
        0.0
    }
}

/// Validates boundary conditions setup
pub(crate) fn validate_bcs_1d(ebcs: &EssentialBcs1d, nbcs: &NaturalBcs1d) -> Result<(), StrError> {
    if ebcs.sides[0] && nbcs.sides[0] {
        return Err("Xmin side has both EBC and NBC");
    }
    if ebcs.sides[1] && nbcs.sides[1] {
        return Err("Xmax side has both EBC and NBC");
    }
    if !ebcs.is_periodic_along_x() {
        if !ebcs.sides[0] && !nbcs.sides[0] {
            return Err("Xmin side is missing either EBC or NBC");
        }
        if !ebcs.sides[1] && !nbcs.sides[1] {
            return Err("Xmax side is missing either EBC or NBC");
        }
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {}
