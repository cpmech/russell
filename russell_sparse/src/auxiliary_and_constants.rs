use super::CooMatrix;
use crate::StrError;

// Make sure that these constants match the c-code constants
pub(crate) const SUCCESSFUL_EXIT: i32 = 0;
pub(crate) const NULL_POINTER_ERROR: i32 = 100000;
pub(crate) const MALLOC_ERROR: i32 = 200000;
pub(crate) const VERSION_ERROR: i32 = 300000;
pub(crate) const NOT_AVAILABLE: i32 = 400000;
pub(crate) const NEED_FACTORIZATION: i32 = 500000;

/// Represents the type of boolean flags interchanged with the C-code
pub(crate) type CcBool = i32;

/// Converts usize to i32
#[inline]
pub(crate) fn to_i32(num: usize) -> Result<i32, StrError> {
    i32::try_from(num).map_err(|_| "cannot downcast usize to i32")
}

/// Checks the dimension of the arrays in the COO matrix when it's ready for conversions/computations
///
/// The following conditions must be satisfied:
///
/// ```text
/// nrow ≥ 1
/// ncol ≥ 1
/// pos ≥ 1 (i.e., nnz ≥ 1)
/// pos ≤ max
/// indices_i.len() == max_nnz
/// indices_j.len() == max_nnz
/// values.len() == max_nnz
/// ```
pub(crate) fn coo_ready_for_conversion(coo: &CooMatrix) -> Result<(), StrError> {
    if coo.nrow < 1 {
        return Err("converting COO matrix: nrow must be ≥ 1");
    }
    if coo.ncol < 1 {
        return Err("converting COO matrix: ncol must be ≥ 1");
    }
    if coo.nnz < 1 {
        return Err("converting COO matrix: pos = nnz must be ≥ 1");
    }
    if coo.nnz > coo.max_nnz {
        return Err("converting COO matrix: pos = nnz must be ≤ max");
    }
    if coo.indices_i.len() != coo.max_nnz {
        return Err("converting COO matrix: indices_i.len() must be = max");
    }
    if coo.indices_j.len() != coo.max_nnz {
        return Err("converting COO matrix: indices_j.len() must be = max");
    }
    if coo.values.len() != coo.max_nnz {
        return Err("converting COO matrix: values.len() must be = max");
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::coo_ready_for_conversion;
    use crate::CooMatrix;

    #[test]
    fn coo_ready_for_conversion_works() {
        let mut coo = CooMatrix {
            symmetry: None,
            one_based: false,
            nrow: 0,
            ncol: 0,
            nnz: 0,
            max_nnz: 0,
            indices_i: Vec::new(),
            indices_j: Vec::new(),
            values: Vec::new(),
        };
        assert_eq!(
            coo_ready_for_conversion(&coo).err(),
            Some("converting COO matrix: nrow must be ≥ 1")
        );
        coo.nrow = 1;
        assert_eq!(
            coo_ready_for_conversion(&coo).err(),
            Some("converting COO matrix: ncol must be ≥ 1")
        );
        coo.ncol = 1;
        assert_eq!(
            coo_ready_for_conversion(&coo).err(),
            Some("converting COO matrix: pos = nnz must be ≥ 1")
        );
        coo.nnz = 1;
        assert_eq!(
            coo_ready_for_conversion(&coo).err(),
            Some("converting COO matrix: pos = nnz must be ≤ max")
        );
        coo.max_nnz = 1;
        assert_eq!(
            coo_ready_for_conversion(&coo).err(),
            Some("converting COO matrix: indices_i.len() must be = max")
        );
        coo.indices_i.resize(1, 0);
        assert_eq!(
            coo_ready_for_conversion(&coo).err(),
            Some("converting COO matrix: indices_j.len() must be = max")
        );
        coo.indices_j.resize(1, 0);
        assert_eq!(
            coo_ready_for_conversion(&coo).err(),
            Some("converting COO matrix: values.len() must be = max")
        );
        coo.values.resize(1, 0.0);
        assert_eq!(coo_ready_for_conversion(&coo).err(), None);
    }
}
