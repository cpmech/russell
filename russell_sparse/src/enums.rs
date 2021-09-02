/// Matrix symmetry options for SolverMMP
pub enum EnumMmpSymmetry {
    /// Unsymmetric matrix
    No,

    /// Positive-definite symmetric matrix
    PosDef,

    /// General symmetric matrix
    General,
}

/// Ordering options for SolverMMP
pub enum EnumMmpOrdering {
    /// Ordering using the approximate minimum degree
    Amd,

    /// Ordering using the approximate minimum fill-in ordering
    Amf,

    /// Automatic ordering method selection
    Auto,

    /// Ordering by Karpis & Kumar from the University of Minnesota
    Metis,

    /// Ordering by Schulze from the University of Paderborn
    Pord,

    /// Ordering using the automatic quasi-dense row detection
    Qamd,
}

/// Scaling options for SolverMMP
pub enum EnumMmpScaling {
    /// Automatic scaling method selection
    Auto,

    /// Column scaling
    Column,

    /// Diagonal scaling
    Diagonal,

    /// No scaling applied or computed
    No,

    /// Row and column scaling based on infinite row/column norms
    RowCol,

    /// Simultaneous row and column iterative scaling
    RowColIterative,

    /// Similar to RcIterative but more rigorous and expensive to compute
    RowColRigorous,
}

pub(crate) fn code_mmp_symmetry(selection: EnumMmpSymmetry) -> i32 {
    match selection {
        EnumMmpSymmetry::No => 0,
        EnumMmpSymmetry::PosDef => 1,
        EnumMmpSymmetry::General => 2,
    }
}

pub(crate) fn code_mmp_ordering(selection: EnumMmpOrdering) -> i32 {
    match selection {
        EnumMmpOrdering::Amd => 0,
        EnumMmpOrdering::Amf => 2,
        EnumMmpOrdering::Auto => 7,
        EnumMmpOrdering::Metis => 5,
        EnumMmpOrdering::Pord => 4,
        EnumMmpOrdering::Qamd => 6,
    }
}

pub(crate) fn code_mmp_scaling(selection: EnumMmpScaling) -> i32 {
    match selection {
        EnumMmpScaling::Auto => 77,
        EnumMmpScaling::Column => 3,
        EnumMmpScaling::Diagonal => 1,
        EnumMmpScaling::No => 0,
        EnumMmpScaling::RowCol => 4,
        EnumMmpScaling::RowColIterative => 7,
        EnumMmpScaling::RowColRigorous => 8,
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_mmp_symmetry_works() {
        assert_eq!(code_mmp_symmetry(EnumMmpSymmetry::No), 0);
        assert_eq!(code_mmp_symmetry(EnumMmpSymmetry::PosDef), 1);
        assert_eq!(code_mmp_symmetry(EnumMmpSymmetry::General), 2);
    }

    #[test]
    fn code_mmp_ordering_works() {
        assert_eq!(code_mmp_ordering(EnumMmpOrdering::Amd), 0);
        assert_eq!(code_mmp_ordering(EnumMmpOrdering::Amf), 2);
        assert_eq!(code_mmp_ordering(EnumMmpOrdering::Auto), 7);
        assert_eq!(code_mmp_ordering(EnumMmpOrdering::Metis), 5);
        assert_eq!(code_mmp_ordering(EnumMmpOrdering::Pord), 4);
        assert_eq!(code_mmp_ordering(EnumMmpOrdering::Qamd), 6);
    }

    #[test]
    fn code_mmp_scaling_works() {
        assert_eq!(code_mmp_scaling(EnumMmpScaling::Auto), 77);
        assert_eq!(code_mmp_scaling(EnumMmpScaling::Column), 3);
        assert_eq!(code_mmp_scaling(EnumMmpScaling::Diagonal), 1);
        assert_eq!(code_mmp_scaling(EnumMmpScaling::No), 0);
        assert_eq!(code_mmp_scaling(EnumMmpScaling::RowCol), 4);
        assert_eq!(code_mmp_scaling(EnumMmpScaling::RowColIterative), 7);
        assert_eq!(code_mmp_scaling(EnumMmpScaling::RowColRigorous), 8);
    }
}
