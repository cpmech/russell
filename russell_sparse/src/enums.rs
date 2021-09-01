/// Options for matrix symmetry using MUMPS solver
pub enum EnumSymmetry {
    /// Unsymmetric matrix
    No,

    /// Positive-definite symmetric matrix
    PosDef,

    /// General symmetric matrix
    General,
}

/// Options for ordering using MUMPS solver
pub enum EnumOrdering {
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

/// Options for scaling using MUMPS solver
pub enum EnumScaling {
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

pub(crate) fn code_symmetry(selection: EnumSymmetry) -> i32 {
    match selection {
        EnumSymmetry::No => 0,
        EnumSymmetry::PosDef => 1,
        EnumSymmetry::General => 2,
    }
}

pub(crate) fn code_ordering(selection: EnumOrdering) -> i32 {
    match selection {
        EnumOrdering::Amd => 0,
        EnumOrdering::Amf => 2,
        EnumOrdering::Auto => 7,
        EnumOrdering::Metis => 5,
        EnumOrdering::Pord => 4,
        EnumOrdering::Qamd => 6,
    }
}

pub(crate) fn code_scaling(selection: EnumScaling) -> i32 {
    match selection {
        EnumScaling::Auto => 77,
        EnumScaling::Column => 3,
        EnumScaling::Diagonal => 1,
        EnumScaling::No => 0,
        EnumScaling::RowCol => 4,
        EnumScaling::RowColIterative => 7,
        EnumScaling::RowColRigorous => 8,
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_symmetry_works() {
        assert_eq!(code_symmetry(EnumSymmetry::No), 0);
        assert_eq!(code_symmetry(EnumSymmetry::PosDef), 1);
        assert_eq!(code_symmetry(EnumSymmetry::General), 2);
    }

    #[test]
    fn code_ordering_works() {
        assert_eq!(code_ordering(EnumOrdering::Amd), 0);
        assert_eq!(code_ordering(EnumOrdering::Amf), 2);
        assert_eq!(code_ordering(EnumOrdering::Auto), 7);
        assert_eq!(code_ordering(EnumOrdering::Metis), 5);
        assert_eq!(code_ordering(EnumOrdering::Pord), 4);
        assert_eq!(code_ordering(EnumOrdering::Qamd), 6);
    }

    #[test]
    fn code_scaling_works() {
        assert_eq!(code_scaling(EnumScaling::Auto), 77);
        assert_eq!(code_scaling(EnumScaling::Column), 3);
        assert_eq!(code_scaling(EnumScaling::Diagonal), 1);
        assert_eq!(code_scaling(EnumScaling::No), 0);
        assert_eq!(code_scaling(EnumScaling::RowCol), 4);
        assert_eq!(code_scaling(EnumScaling::RowColIterative), 7);
        assert_eq!(code_scaling(EnumScaling::RowColRigorous), 8);
    }
}
