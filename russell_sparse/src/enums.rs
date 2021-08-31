/// Options for matrix symmetry using MUMPS solver
pub enum EnumMumpsSymmetry {
    /// Unsymmetric matrix
    No,

    /// Positive-definite symmetric matrix
    PosDef,

    /// General symmetric matrix
    General,
}

/// Options for ordering using MUMPS solver
pub enum EnumMumpsOrdering {
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
pub enum EnumMumpsScaling {
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

pub(crate) fn enum_mumps_symmetry(selection: EnumMumpsSymmetry) -> i32 {
    match selection {
        EnumMumpsSymmetry::No => 0,
        EnumMumpsSymmetry::PosDef => 1,
        EnumMumpsSymmetry::General => 2,
    }
}

pub(crate) fn enum_mumps_ordering(selection: EnumMumpsOrdering) -> i32 {
    match selection {
        EnumMumpsOrdering::Amd => 0,
        EnumMumpsOrdering::Amf => 2,
        EnumMumpsOrdering::Auto => 7,
        EnumMumpsOrdering::Metis => 5,
        EnumMumpsOrdering::Pord => 4,
        EnumMumpsOrdering::Qamd => 6,
    }
}

pub(crate) fn enum_mumps_scaling(selection: EnumMumpsScaling) -> i32 {
    match selection {
        EnumMumpsScaling::Auto => 77,
        EnumMumpsScaling::Column => 3,
        EnumMumpsScaling::Diagonal => 1,
        EnumMumpsScaling::No => 0,
        EnumMumpsScaling::RowCol => 4,
        EnumMumpsScaling::RowColIterative => 7,
        EnumMumpsScaling::RowColRigorous => 8,
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enum_mumps_symmetry_works() {
        assert_eq!(enum_mumps_symmetry(EnumMumpsSymmetry::No), 0);
        assert_eq!(enum_mumps_symmetry(EnumMumpsSymmetry::PosDef), 1);
        assert_eq!(enum_mumps_symmetry(EnumMumpsSymmetry::General), 2);
    }

    #[test]
    fn enum_mumps_ordering_works() {
        assert_eq!(enum_mumps_ordering(EnumMumpsOrdering::Amd), 0);
        assert_eq!(enum_mumps_ordering(EnumMumpsOrdering::Amf), 2);
        assert_eq!(enum_mumps_ordering(EnumMumpsOrdering::Auto), 7);
        assert_eq!(enum_mumps_ordering(EnumMumpsOrdering::Metis), 5);
        assert_eq!(enum_mumps_ordering(EnumMumpsOrdering::Pord), 4);
        assert_eq!(enum_mumps_ordering(EnumMumpsOrdering::Qamd), 6);
    }

    #[test]
    fn enum_mumps_scaling_works() {
        assert_eq!(enum_mumps_scaling(EnumMumpsScaling::Auto), 77);
        assert_eq!(enum_mumps_scaling(EnumMumpsScaling::Column), 3);
        assert_eq!(enum_mumps_scaling(EnumMumpsScaling::Diagonal), 1);
        assert_eq!(enum_mumps_scaling(EnumMumpsScaling::No), 0);
        assert_eq!(enum_mumps_scaling(EnumMumpsScaling::RowCol), 4);
        assert_eq!(enum_mumps_scaling(EnumMumpsScaling::RowColIterative), 7);
        assert_eq!(enum_mumps_scaling(EnumMumpsScaling::RowColRigorous), 8);
    }
}
