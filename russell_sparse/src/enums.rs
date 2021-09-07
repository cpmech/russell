/// Defines the solver kinds
pub enum EnumSolverKind {
    /// The NON-THREAD-SAFE (Mu-M-P) Solver
    Mmp = 0,

    /// Tim Davis' UMFPACK Solver (**recommended**)
    Umf = 1,
}

/// Matrix symmetry options
pub enum EnumSymmetry {
    Auto = 0, // Automatic detection (UMF-only, otherwise No)

    /// General symmetric matrix
    General = 1,

    /// Unsymmetric matrix
    No = 2,

    /// Positive-definite symmetric matrix (MMP-only, otherwise General)
    PosDef = 3,
}

/// Ordering options
pub enum EnumOrdering {
    /// Ordering using the approximate minimum degree
    Amd = 0,

    /// Ordering using the approximate minimum fill-in (MMP-only, otherwise Auto)
    Amf = 1,

    /// Automatic ordering method selection
    Auto = 2,

    /// Try three methods and take the best (UMF-only, otherwise Auto)
    Best = 3,

    /// Use Amd for symmetric, Colamd for unsymmetric, or Metis (UMF-only, otherwise Auto)
    Cholmod = 4,

    /// Ordering by Karpis & Kumar from the University of Minnesota
    Metis = 5,

    /// The matrix is factorized as-is (UMF-only, otherwise Auto)
    No = 6,

    /// Ordering by Schulze from the University of Paderborn (MMP-only, otherwise Auto)
    Pord = 7,

    /// Ordering using the automatic quasi-dense row detection (MMP-only, otherwise Auto)
    Qamd = 8,

    /// Ordering using the Scotch package (MMP-only, otherwise Auto)
    Scotch = 9,
}

/// Scaling options for SolverMMP
pub enum EnumScaling {
    /// Automatic scaling method selection
    Auto = 0,

    /// Column scaling (MMP-only, otherwise Auto)
    Column = 1,

    /// Diagonal scaling (MMP-only, otherwise Auto)
    Diagonal = 2,

    /// Use the max absolute value in the row (UMF-only, otherwise Auto)
    Max = 3,

    /// No scaling applied or computed
    No = 4,

    /// Row and column scaling based on infinite row/column norms (MMP-only, otherwise Auto)
    RowCol = 5,

    /// Simultaneous row and column iterative scaling (MMP-only, otherwise Auto)
    RowColIter = 6,

    /// Similar to RcIterative but more rigorous and expensive to compute (MMP-only, otherwise Auto)
    RowColRig = 7,

    /// Use the sum of the absolute value in the row (UMF-only, otherwise Auto)
    Sum = 8,
}

pub(crate) fn str_enum_symmetry(code: i32) -> &'static str {
    match code {
        0 => "Auto (UMF-only, otherwise No)",
        1 => "General",
        2 => "No",
        3 => "PosDef (MMP-only, otherwise General)",
        _ => panic!("<internal error: invalid code>"),
    }
}

pub(crate) fn str_enum_ordering(code: i32) -> &'static str {
    match code {
        0 => "Amd",
        1 => "Amf (MMP-only, otherwise Auto)",
        2 => "Auto",
        3 => "Best (UMF-only, otherwise Auto)",
        4 => "Cholmod (UMF-only, otherwise Auto)",
        5 => "Metis",
        6 => "No (UMF-only, otherwise Auto)",
        7 => "Pord (MMP-only, otherwise Auto)",
        8 => "Qamd (MMP-only, otherwise Auto)",
        9 => "Scotch (MMP-only, otherwise Auto)",
        _ => panic!("<internal error: invalid code>"),
    }
}

pub(crate) fn str_enum_scaling(code: i32) -> &'static str {
    match code {
        0 => "Auto",
        1 => "Column (MMP-only, otherwise Auto)",
        2 => "Diagonal (MMP-only, otherwise Auto)",
        3 => "Max (UMF-only, otherwise Auto)",
        4 => "No",
        5 => "RowCol (MMP-only, otherwise Auto)",
        6 => "RowColIter (MMP-only, otherwise Auto)",
        7 => "RowColRig (MMP-only, otherwise Auto)",
        8 => "Sum (UMF-only, otherwise Auto)",
        _ => panic!("<internal error: invalid code>"),
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "<internal error: invalid code>")]
    fn str_enum_symmetry_panics_on_wrong_code() {
        str_enum_symmetry(123);
    }

    #[test]
    #[should_panic(expected = "<internal error: invalid code>")]
    fn str_enum_ordering_panics_on_wrong_code() {
        str_enum_ordering(123);
    }

    #[test]
    #[should_panic(expected = "<internal error: invalid code>")]
    fn str_enum_scaling_panics_on_wrong_code() {
        str_enum_scaling(123);
    }

    #[test]
    fn str_enum_symmetry_works() {
        assert_eq!(str_enum_symmetry(0), "Auto (UMF-only, otherwise No)");
        assert_eq!(str_enum_symmetry(1), "General");
        assert_eq!(str_enum_symmetry(2), "No");
        assert_eq!(str_enum_symmetry(3), "PosDef (MMP-only, otherwise General)");
    }

    #[test]
    fn str_enum_ordering_works() {
        assert_eq!(str_enum_ordering(0), "Amd");
        assert_eq!(str_enum_ordering(1), "Amf (MMP-only, otherwise Auto)");
        assert_eq!(str_enum_ordering(2), "Auto");
        assert_eq!(str_enum_ordering(3), "Best (UMF-only, otherwise Auto)");
        assert_eq!(str_enum_ordering(4), "Cholmod (UMF-only, otherwise Auto)");
        assert_eq!(str_enum_ordering(5), "Metis");
        assert_eq!(str_enum_ordering(6), "No (UMF-only, otherwise Auto)");
        assert_eq!(str_enum_ordering(7), "Pord (MMP-only, otherwise Auto)");
        assert_eq!(str_enum_ordering(8), "Qamd (MMP-only, otherwise Auto)");
        assert_eq!(str_enum_ordering(9), "Scotch (MMP-only, otherwise Auto)");
    }

    #[test]
    fn str_enum_scaling_works() {
        assert_eq!(str_enum_scaling(0), "Auto");
        assert_eq!(str_enum_scaling(1), "Column (MMP-only, otherwise Auto)");
        assert_eq!(str_enum_scaling(2), "Diagonal (MMP-only, otherwise Auto)");
        assert_eq!(str_enum_scaling(3), "Max (UMF-only, otherwise Auto)");
        assert_eq!(str_enum_scaling(4), "No");
        assert_eq!(str_enum_scaling(5), "RowCol (MMP-only, otherwise Auto)");
        assert_eq!(str_enum_scaling(6), "RowColIter (MMP-only, otherwise Auto)");
        assert_eq!(str_enum_scaling(7), "RowColRig (MMP-only, otherwise Auto)");
        assert_eq!(str_enum_scaling(8), "Sum (UMF-only, otherwise Auto)");
    }
}
