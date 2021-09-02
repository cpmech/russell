/// Matrix symmetry options for SolverMMP
pub enum EnumMmpSymmetry {
    /// Unsymmetric matrix
    No = 0,

    /// Positive-definite symmetric matrix
    PosDef = 1,

    /// General symmetric matrix
    General = 2,
}

/// Ordering options for SolverMMP
pub enum EnumMmpOrdering {
    /// Ordering using the approximate minimum degree
    Amd = 0,

    /// Ordering using the approximate minimum fill-in ordering
    Amf = 2,

    /// Automatic ordering method selection
    Auto = 7,

    /// Ordering by Karpis & Kumar from the University of Minnesota
    Metis = 5,

    /// Ordering by Schulze from the University of Paderborn
    Pord = 4,

    /// Ordering using the automatic quasi-dense row detection
    Qamd = 6,
}

/// Scaling options for SolverMMP
pub enum EnumMmpScaling {
    /// Automatic scaling method selection
    Auto = 77,

    /// Column scaling
    Column = 3,

    /// Diagonal scaling
    Diagonal = 1,

    /// No scaling applied or computed
    No = 0,

    /// Row and column scaling based on infinite row/column norms
    RowCol = 4,

    /// Simultaneous row and column iterative scaling
    RowColIter = 7,

    /// Similar to RcIterative but more rigorous and expensive to compute
    RowColRig = 8,
}

/// Ordering options for SolverUMF (page 17)
pub enum EnumUmfOrdering {
    /// Ordering using the approximate minimum degree
    Amd = 0,

    /// Try three methods and take the best
    Best = 1,

    /// Use Amd for symmetric, Colamd for unsymmetric, or Metis
    Cholmod = 2,

    /// Default ordering method == Amd
    Default = 3,

    /// Ordering by Karpis & Kumar from the University of Minnesota
    Metis = 4,

    /// The matrix is factorized as-is (singletons removed)
    No = 5,
}

/// Scaling options for SolverUMF (page 49)
pub enum EnumUmfScaling {
    /// Default scaling method
    Default = 0,

    /// Use the max absolute value in the row
    Max = 1,

    /// No scaling is performed
    No = 2,

    /// Use the sum of the absolute value in the row
    Sum = 3,
}

pub(crate) fn str_mmp_symmetry(code: i32) -> &'static str {
    match code {
        0 => "No",
        1 => "PosDef",
        2 => "General",
        _ => panic!("invalid code"),
    }
}

pub(crate) fn str_mmp_ordering(code: i32) -> &'static str {
    match code {
        0 => "Amd",
        2 => "Amf",
        7 => "Auto",
        5 => "Metis",
        4 => "Pord",
        6 => "Qamd",
        _ => panic!("invalid code"),
    }
}

pub(crate) fn str_mmp_scaling(code: i32) -> &'static str {
    match code {
        77 => "Auto",
        3 => "Column",
        1 => "Diagonal",
        0 => "No",
        4 => "RowCol",
        7 => "RowColIter",
        8 => "RowColRig",
        _ => panic!("invalid code"),
    }
}

pub(crate) fn str_umf_ordering(code: i32) -> &'static str {
    match code {
        0 => "Amd",
        1 => "Best",
        2 => "Cholmod",
        3 => "Default",
        4 => "Metis",
        5 => "No",
        _ => panic!("invalid code"),
    }
}

pub(crate) fn str_umf_scaling(code: i32) -> &'static str {
    match code {
        0 => "Default",
        1 => "Max",
        2 => "No",
        3 => "Sum",
        _ => panic!("invalid code"),
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn str_mmp_symmetry_works() {
        assert_eq!(str_mmp_symmetry(0), "No");
        assert_eq!(str_mmp_symmetry(1), "PosDef");
        assert_eq!(str_mmp_symmetry(2), "General");
    }

    #[test]
    fn str_mmp_ordering_works() {
        assert_eq!(str_mmp_ordering(0), "Amd");
        assert_eq!(str_mmp_ordering(2), "Amf");
        assert_eq!(str_mmp_ordering(7), "Auto");
        assert_eq!(str_mmp_ordering(5), "Metis");
        assert_eq!(str_mmp_ordering(4), "Pord");
        assert_eq!(str_mmp_ordering(6), "Qamd");
    }

    #[test]
    fn str_mmp_scaling_works() {
        assert_eq!(str_mmp_scaling(77), "Auto");
        assert_eq!(str_mmp_scaling(3), "Column");
        assert_eq!(str_mmp_scaling(1), "Diagonal");
        assert_eq!(str_mmp_scaling(0), "No");
        assert_eq!(str_mmp_scaling(4), "RowCol");
        assert_eq!(str_mmp_scaling(7), "RowColIter");
        assert_eq!(str_mmp_scaling(8), "RowColRig");
    }

    #[test]
    fn str_umf_ordering_works() {
        assert_eq!(str_umf_ordering(0), "Amd");
        assert_eq!(str_umf_ordering(1), "Best");
        assert_eq!(str_umf_ordering(2), "Cholmod");
        assert_eq!(str_umf_ordering(3), "Default");
        assert_eq!(str_umf_ordering(4), "Metis");
        assert_eq!(str_umf_ordering(5), "No");
    }

    #[test]
    fn str_umf_scaling_works() {
        assert_eq!(str_umf_scaling(0), "Default");
        assert_eq!(str_umf_scaling(1), "Max");
        assert_eq!(str_umf_scaling(2), "No");
        assert_eq!(str_umf_scaling(3), "Sum");
    }
}
