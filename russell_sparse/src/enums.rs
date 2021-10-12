/// Matrix symmetry options
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Symmetry {
    /// Unsymmetric matrix
    No,

    /// General symmetric matrix
    General,

    /// The matrix is symmetric and only the triangular part is present (MMP only)
    GeneralTriangular,

    /// The matrix is positive-definite, symmetric, and only the triangular part is present (MMP only)
    PosDefTriangular,
}

pub(crate) fn code_symmetry_mmp(option: Symmetry) -> Result<i32, &'static str> {
    match option {
        Symmetry::No => Ok(0),
        Symmetry::General => Err("General symmetry option is not available for MMP"),
        Symmetry::GeneralTriangular => Ok(2),
        Symmetry::PosDefTriangular => Ok(1),
    }
}

pub(crate) fn code_symmetry_umf(option: Symmetry) -> Result<i32, &'static str> {
    match option {
        Symmetry::No => Ok(0),
        Symmetry::General => Ok(1),
        Symmetry::GeneralTriangular => Err("GeneralTriangular symmetry option is not available for UMF"),
        Symmetry::PosDefTriangular => Err("PosDefTriangular symmetry option is not available for UMF"),
    }
}

/// Defines the solver kinds
pub enum EnumSolverKind {
    /// The NON-THREAD-SAFE (Mu-M-P) Solver (use in single-thread apps / with huge matrices)
    Mmp = 0,

    /// Tim Davis' UMFPACK Solver (recommended, unless the matrix is huge)
    Umf = 1,
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

/// Returns the EnumOrdering given its name
pub fn enum_ordering(ordering: &str) -> EnumOrdering {
    match ordering {
        "Amd" => EnumOrdering::Amd,
        "Amf" => EnumOrdering::Amf,
        "Auto" => EnumOrdering::Auto,
        "Best" => EnumOrdering::Best,
        "Cholmod" => EnumOrdering::Cholmod,
        "Metis" => EnumOrdering::Metis,
        "No" => EnumOrdering::No,
        "Pord" => EnumOrdering::Pord,
        "Qamd" => EnumOrdering::Qamd,
        "Scotch" => EnumOrdering::Scotch,
        _ => EnumOrdering::Auto,
    }
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

/// Returns the EnumScaling given its name
pub fn enum_scaling(scaling: &str) -> EnumScaling {
    match scaling {
        "Auto" => EnumScaling::Auto,
        "Column" => EnumScaling::Column,
        "Diagonal" => EnumScaling::Diagonal,
        "Max" => EnumScaling::Max,
        "No" => EnumScaling::No,
        "RowCol" => EnumScaling::RowCol,
        "RowColIter" => EnumScaling::RowColIter,
        "RowColRig" => EnumScaling::RowColRig,
        "Sum" => EnumScaling::Sum,
        _ => EnumScaling::Auto,
    }
}

pub(crate) fn str_enum_ordering(index: i32) -> &'static str {
    match index {
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
        _ => panic!("<internal error: invalid index>"),
    }
}

pub(crate) fn str_enum_scaling(index: i32) -> &'static str {
    match index {
        0 => "Auto",
        1 => "Column (MMP-only, otherwise Auto)",
        2 => "Diagonal (MMP-only, otherwise Auto)",
        3 => "Max (UMF-only, otherwise Auto)",
        4 => "No",
        5 => "RowCol (MMP-only, otherwise Auto)",
        6 => "RowColIter (MMP-only, otherwise Auto)",
        7 => "RowColRig (MMP-only, otherwise Auto)",
        8 => "Sum (UMF-only, otherwise Auto)",
        _ => panic!("<internal error: invalid index>"),
    }
}

pub(crate) fn str_mmp_ordering(mmp_code: i32) -> &'static str {
    match mmp_code {
        0 => "Amd",
        1 => "UserProvided",
        2 => "Amf",
        3 => "Scotch",
        4 => "Pord",
        5 => "Metis",
        6 => "Qamd",
        7 => "Auto",
        _ => "Unknown",
    }
}

pub(crate) fn str_mmp_scaling(mmp_code: i32) -> &'static str {
    match mmp_code {
        -1 => "UserProvided",
        0 => "No",
        1 => "Diagonal",
        3 => "Column",
        4 => "RowCol",
        7 => "RowColIter",
        8 => "RowColRig",
        77 => "Auto",
        _ => "Unknown",
    }
}

pub(crate) fn str_umf_ordering(umf_code: i32) -> &'static str {
    match umf_code {
        0 => "Cholmod",
        1 => "Amd",
        2 => "UserProvided",
        3 => "Metis",
        4 => "Best",
        5 => "No",
        6 => "UserProvided",
        _ => "Unknown",
    }
}

pub(crate) fn str_umf_scaling(umf_code: i32) -> &'static str {
    match umf_code {
        0 => "No",
        1 => "Sum",
        2 => "Max",
        _ => "Unknown",
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{
        enum_ordering, enum_scaling, str_enum_ordering, str_enum_scaling, str_mmp_ordering, str_mmp_scaling,
        str_umf_ordering, str_umf_scaling, EnumOrdering, EnumScaling,
    };

    #[test]
    fn enum_ordering_works() {
        assert!(matches!(enum_ordering("Amd"), EnumOrdering::Amd));
        assert!(matches!(enum_ordering("Amf"), EnumOrdering::Amf));
        assert!(matches!(enum_ordering("Auto"), EnumOrdering::Auto));
        assert!(matches!(enum_ordering("Best"), EnumOrdering::Best));
        assert!(matches!(enum_ordering("Cholmod"), EnumOrdering::Cholmod));
        assert!(matches!(enum_ordering("Metis"), EnumOrdering::Metis));
        assert!(matches!(enum_ordering("No"), EnumOrdering::No));
        assert!(matches!(enum_ordering("Pord"), EnumOrdering::Pord));
        assert!(matches!(enum_ordering("Qamd"), EnumOrdering::Qamd));
        assert!(matches!(enum_ordering("Scotch"), EnumOrdering::Scotch));
        assert!(matches!(enum_ordering("Unknown"), EnumOrdering::Auto));
    }

    #[test]
    fn enum_scaling_works() {
        assert!(matches!(enum_scaling("Auto"), EnumScaling::Auto));
        assert!(matches!(enum_scaling("Column"), EnumScaling::Column));
        assert!(matches!(enum_scaling("Diagonal"), EnumScaling::Diagonal));
        assert!(matches!(enum_scaling("Max"), EnumScaling::Max));
        assert!(matches!(enum_scaling("No"), EnumScaling::No));
        assert!(matches!(enum_scaling("RowCol"), EnumScaling::RowCol));
        assert!(matches!(enum_scaling("RowColIter"), EnumScaling::RowColIter));
        assert!(matches!(enum_scaling("RowColRig"), EnumScaling::RowColRig));
        assert!(matches!(enum_scaling("Sum"), EnumScaling::Sum));
        assert!(matches!(enum_scaling("Unknown"), EnumScaling::Auto));
    }

    #[test]
    #[should_panic(expected = "<internal error: invalid index>")]
    fn str_enum_ordering_panics_on_wrong_code() {
        str_enum_ordering(123);
    }

    #[test]
    #[should_panic(expected = "<internal error: invalid index>")]
    fn str_enum_scaling_panics_on_wrong_code() {
        str_enum_scaling(123);
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

    #[test]
    fn str_mmp_ordering_works() {
        assert_eq!(str_mmp_ordering(0), "Amd");
        assert_eq!(str_mmp_ordering(1), "UserProvided");
        assert_eq!(str_mmp_ordering(2), "Amf");
        assert_eq!(str_mmp_ordering(3), "Scotch");
        assert_eq!(str_mmp_ordering(4), "Pord");
        assert_eq!(str_mmp_ordering(5), "Metis");
        assert_eq!(str_mmp_ordering(6), "Qamd");
        assert_eq!(str_mmp_ordering(7), "Auto");
        assert_eq!(str_mmp_ordering(123), "Unknown");
    }

    #[test]
    fn str_mmp_scaling_works() {
        assert_eq!(str_mmp_scaling(-1), "UserProvided");
        assert_eq!(str_mmp_scaling(0), "No");
        assert_eq!(str_mmp_scaling(1), "Diagonal");
        assert_eq!(str_mmp_scaling(3), "Column");
        assert_eq!(str_mmp_scaling(4), "RowCol");
        assert_eq!(str_mmp_scaling(7), "RowColIter");
        assert_eq!(str_mmp_scaling(8), "RowColRig");
        assert_eq!(str_mmp_scaling(77), "Auto");
        assert_eq!(str_mmp_scaling(123), "Unknown");
    }

    #[test]
    fn str_umf_ordering_works() {
        assert_eq!(str_umf_ordering(0), "Cholmod");
        assert_eq!(str_umf_ordering(1), "Amd");
        assert_eq!(str_umf_ordering(2), "UserProvided");
        assert_eq!(str_umf_ordering(3), "Metis");
        assert_eq!(str_umf_ordering(4), "Best");
        assert_eq!(str_umf_ordering(5), "No");
        assert_eq!(str_umf_ordering(6), "UserProvided");
        assert_eq!(str_umf_ordering(123), "Unknown");
    }

    #[test]
    fn str_umf_scaling_works() {
        assert_eq!(str_umf_scaling(0), "No");
        assert_eq!(str_umf_scaling(1), "Sum");
        assert_eq!(str_umf_scaling(2), "Max");
        assert_eq!(str_umf_scaling(123), "Unknown");
    }
}
