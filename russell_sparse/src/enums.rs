use crate::StrError;

/// Defines how the CooMatrix (triplet) represents a matrix
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Layout {
    /// Lower triangular (e.g., for MUMPS)
    Lower,

    /// Upper triangular (e.g., for Intel DSS)
    Upper,

    /// Full matrix (e.g., for UMFPACK)
    Full,
}

/// Holds options to handle a MatrixMarket when the matrix is specified as being symmetric
///
/// **Note:** This is ignored if not the matrix is not specified as symmetric.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SymmetricHandling {
    /// Leave the layout as lower triangular (if symmetric)
    ///
    /// **Note:** Lower triangular is the standard MatrixMarket format.
    /// Thus, this option will do nothing.
    ///
    /// This option is useful for the MUMPS solver.
    LeaveAsLower,

    /// Convert the layout to upper triangular (if symmetric)
    ///
    /// **Note:** Since lower triangular is standard in MatrixMarket,
    /// this option will swap the lower triangle to the upper triangle.
    ///
    /// This option is useful for the Intel DSS solver.
    SwapToUpper,

    /// Make the matrix full (if symmetric)
    ///
    /// **Note:: Mirror the lower triangle to the upper triangle (duplicate data).
    /// The number of non-zeros will be slightly larger than just duplicating the lower triangle.
    ///
    /// This option is useful for the UMFPACK solver.
    MakeItFull,
}

/// Matrix symmetry option
#[derive(Clone, Copy, Debug)]
pub enum Symmetry {
    /// General symmetric matrix
    ///
    /// **Note:** When using the MUMPS solver, make sure to provide a lower triangular matrix
    General,

    /// The matrix is positive-definite and symmetric
    ///
    /// **Note:** When using the MUMPS solver, make sure to provide a lower triangular matrix
    PosDef,
}

/// Linear solver kind
#[derive(Clone, Copy, Debug)]
pub enum LinSolKind {
    /// The NON-THREAD-SAFE MUMPS Solver (use in single-thread apps / with huge matrices)
    Mumps,

    /// Tim Davis' UMFPACK Solver (recommended, unless the matrix is huge)
    Umfpack,
}

/// Ordering option
#[derive(Clone, Copy, Debug)]
pub enum Ordering {
    /// Ordering using the approximate minimum degree
    Amd = 0,

    /// Ordering using the approximate minimum fill-in (MUMPS-only, otherwise Auto)
    Amf = 1,

    /// Automatic ordering method selection
    Auto = 2,

    /// Try three methods and take the best (UMFPACK-only, otherwise Auto)
    Best = 3,

    /// Use Amd for symmetric, Colamd for unsymmetric, or Metis (UMFPACK-only, otherwise Auto)
    Cholmod = 4,

    /// Ordering by Karpis & Kumar from the University of Minnesota
    Metis = 5,

    /// The matrix is factorized as-is (UMFPACK-only, otherwise Auto)
    No = 6,

    /// Ordering by Schulze from the University of Paderborn (MUMPS-only, otherwise Auto)
    Pord = 7,

    /// Ordering using the automatic quasi-dense row detection (MUMPS-only, otherwise Auto)
    Qamd = 8,

    /// Ordering using the Scotch package (MUMPS-only, otherwise Auto)
    Scotch = 9,
}

/// Scaling option
#[derive(Clone, Copy, Debug)]
pub enum Scaling {
    /// Automatic scaling method selection
    Auto = 0,

    /// Column scaling (MUMPS-only, otherwise Auto)
    Column = 1,

    /// Diagonal scaling (MUMPS-only, otherwise Auto)
    Diagonal = 2,

    /// Use the max absolute value in the row (UMFPACK-only, otherwise Auto)
    Max = 3,

    /// No scaling applied or computed
    No = 4,

    /// Row and column scaling based on infinite row/column norms (MUMPS-only, otherwise Auto)
    RowCol = 5,

    /// Simultaneous row and column iterative scaling (MUMPS-only, otherwise Auto)
    RowColIter = 6,

    /// Similar to RcIterative but more rigorous and expensive to compute (MUMPS-only, otherwise Auto)
    RowColRig = 7,

    /// Use the sum of the absolute value in the row (UMFPACK-only, otherwise Auto)
    Sum = 8,
}

/// Returns the Ordering by name
pub fn enum_ordering(ordering: &str) -> Ordering {
    match ordering {
        "Amd" => Ordering::Amd,
        "Amf" => Ordering::Amf,
        "Auto" => Ordering::Auto,
        "Best" => Ordering::Best,
        "Cholmod" => Ordering::Cholmod,
        "Metis" => Ordering::Metis,
        "No" => Ordering::No,
        "Pord" => Ordering::Pord,
        "Qamd" => Ordering::Qamd,
        "Scotch" => Ordering::Scotch,
        _ => Ordering::Auto,
    }
}

/// Returns the Scaling by name
pub fn enum_scaling(scaling: &str) -> Scaling {
    match scaling {
        "Auto" => Scaling::Auto,
        "Column" => Scaling::Column,
        "Diagonal" => Scaling::Diagonal,
        "Max" => Scaling::Max,
        "No" => Scaling::No,
        "RowCol" => Scaling::RowCol,
        "RowColIter" => Scaling::RowColIter,
        "RowColRig" => Scaling::RowColRig,
        "Sum" => Scaling::Sum,
        _ => Scaling::Auto,
    }
}

pub(crate) fn code_symmetry_mumps(option: Option<Symmetry>) -> Result<i32, StrError> {
    match option {
        None => Ok(0),
        Some(v) => match v {
            Symmetry::General => Ok(2),
            Symmetry::PosDef => Ok(1),
        },
    }
}

pub(crate) fn code_symmetry_umfpack(option: Option<Symmetry>) -> Result<i32, StrError> {
    match option {
        None => Ok(0),
        Some(v) => match v {
            Symmetry::General => Ok(1),
            Symmetry::PosDef => Ok(1),
        },
    }
}

pub(crate) fn str_enum_ordering(index: i32) -> &'static str {
    match index {
        0 => "Amd",
        1 => "Amf (MUMPS-only, otherwise Auto)",
        2 => "Auto",
        3 => "Best (UMFPACK-only, otherwise Auto)",
        4 => "Cholmod (UMFPACK-only, otherwise Auto)",
        5 => "Metis",
        6 => "No (UMFPACK-only, otherwise Auto)",
        7 => "Pord (MUMPS-only, otherwise Auto)",
        8 => "Qamd (MUMPS-only, otherwise Auto)",
        9 => "Scotch (MUMPS-only, otherwise Auto)",
        _ => panic!("<internal error: invalid index>"),
    }
}

pub(crate) fn str_enum_scaling(index: i32) -> &'static str {
    match index {
        0 => "Auto",
        1 => "Column (MUMPS-only, otherwise Auto)",
        2 => "Diagonal (MUMPS-only, otherwise Auto)",
        3 => "Max (UMFPACK-only, otherwise Auto)",
        4 => "No",
        5 => "RowCol (MUMPS-only, otherwise Auto)",
        6 => "RowColIter (MUMPS-only, otherwise Auto)",
        7 => "RowColRig (MUMPS-only, otherwise Auto)",
        8 => "Sum (UMFPACK-only, otherwise Auto)",
        _ => panic!("<internal error: invalid index>"),
    }
}

pub(crate) fn str_mumps_ordering(mumps_code: i32) -> &'static str {
    match mumps_code {
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

pub(crate) fn str_mumps_scaling(mumps_code: i32) -> &'static str {
    match mumps_code {
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

pub(crate) fn str_umfpack_ordering(umfpack_code: i32) -> &'static str {
    match umfpack_code {
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

pub(crate) fn str_umfpack_scaling(umfpack_code: i32) -> &'static str {
    match umfpack_code {
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
        code_symmetry_mumps, code_symmetry_umfpack, enum_ordering, enum_scaling, str_enum_ordering, str_enum_scaling,
        str_mumps_ordering, str_mumps_scaling, str_umfpack_ordering, str_umfpack_scaling, Layout, LinSolKind, Ordering, Scaling,
        SymmetricHandling, Symmetry,
    };

    #[test]
    fn clone_copy_and_debug_work() {
        let layout = Layout::Full;
        let copy = layout;
        let clone = layout.clone();
        assert_eq!(format!("{:?}", layout), "Full");
        assert_eq!(copy, Layout::Full);
        assert_eq!(clone, Layout::Full);

        let handling = SymmetricHandling::LeaveAsLower;
        let copy = handling;
        let clone = handling.clone();
        assert_eq!(format!("{:?}", handling), "LeaveAsLower");
        assert_eq!(copy, SymmetricHandling::LeaveAsLower);
        assert_eq!(clone, SymmetricHandling::LeaveAsLower);

        let symmetry = Symmetry::General;
        let copy = symmetry;
        let clone = symmetry.clone();
        assert_eq!(format!("{:?}", symmetry), "General");
        assert_eq!(format!("{:?}", copy), "General");
        assert_eq!(format!("{:?}", clone), "General");

        let lin_sol_kind = LinSolKind::Mumps;
        let copy = lin_sol_kind;
        let clone = lin_sol_kind.clone();
        assert_eq!(format!("{:?}", lin_sol_kind), "Mumps");
        assert_eq!(format!("{:?}", copy), "Mumps");
        assert_eq!(format!("{:?}", clone), "Mumps");

        let ordering = Ordering::Amd;
        let copy = ordering;
        let clone = ordering.clone();
        assert_eq!(format!("{:?}", ordering), "Amd");
        assert_eq!(format!("{:?}", copy), "Amd");
        assert_eq!(format!("{:?}", clone), "Amd");

        let scaling = Scaling::Column;
        let copy = scaling;
        let clone = scaling.clone();
        assert_eq!(format!("{:?}", scaling), "Column");
        assert_eq!(format!("{:?}", copy), "Column");
        assert_eq!(format!("{:?}", clone), "Column");
    }

    #[test]
    fn enum_ordering_works() {
        assert!(matches!(enum_ordering("Amd"), Ordering::Amd));
        assert!(matches!(enum_ordering("Amf"), Ordering::Amf));
        assert!(matches!(enum_ordering("Auto"), Ordering::Auto));
        assert!(matches!(enum_ordering("Best"), Ordering::Best));
        assert!(matches!(enum_ordering("Cholmod"), Ordering::Cholmod));
        assert!(matches!(enum_ordering("Metis"), Ordering::Metis));
        assert!(matches!(enum_ordering("No"), Ordering::No));
        assert!(matches!(enum_ordering("Pord"), Ordering::Pord));
        assert!(matches!(enum_ordering("Qamd"), Ordering::Qamd));
        assert!(matches!(enum_ordering("Scotch"), Ordering::Scotch));
        assert!(matches!(enum_ordering("Unknown"), Ordering::Auto));
    }

    #[test]
    fn enum_scaling_works() {
        assert!(matches!(enum_scaling("Auto"), Scaling::Auto));
        assert!(matches!(enum_scaling("Column"), Scaling::Column));
        assert!(matches!(enum_scaling("Diagonal"), Scaling::Diagonal));
        assert!(matches!(enum_scaling("Max"), Scaling::Max));
        assert!(matches!(enum_scaling("No"), Scaling::No));
        assert!(matches!(enum_scaling("RowCol"), Scaling::RowCol));
        assert!(matches!(enum_scaling("RowColIter"), Scaling::RowColIter));
        assert!(matches!(enum_scaling("RowColRig"), Scaling::RowColRig));
        assert!(matches!(enum_scaling("Sum"), Scaling::Sum));
        assert!(matches!(enum_scaling("Unknown"), Scaling::Auto));
    }

    #[test]
    fn code_symmetry_works() {
        // mumps
        assert_eq!(code_symmetry_mumps(None), Ok(0));
        assert_eq!(code_symmetry_mumps(Some(Symmetry::General)), Ok(2));
        assert_eq!(code_symmetry_mumps(Some(Symmetry::PosDef)), Ok(1));
        // umfpack
        assert_eq!(code_symmetry_umfpack(None), Ok(0));
        assert_eq!(code_symmetry_umfpack(Some(Symmetry::General)), Ok(1));
        assert_eq!(code_symmetry_umfpack(Some(Symmetry::PosDef)), Ok(1));
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
        assert_eq!(str_enum_ordering(1), "Amf (MUMPS-only, otherwise Auto)");
        assert_eq!(str_enum_ordering(2), "Auto");
        assert_eq!(str_enum_ordering(3), "Best (UMFPACK-only, otherwise Auto)");
        assert_eq!(str_enum_ordering(4), "Cholmod (UMFPACK-only, otherwise Auto)");
        assert_eq!(str_enum_ordering(5), "Metis");
        assert_eq!(str_enum_ordering(6), "No (UMFPACK-only, otherwise Auto)");
        assert_eq!(str_enum_ordering(7), "Pord (MUMPS-only, otherwise Auto)");
        assert_eq!(str_enum_ordering(8), "Qamd (MUMPS-only, otherwise Auto)");
        assert_eq!(str_enum_ordering(9), "Scotch (MUMPS-only, otherwise Auto)");
    }

    #[test]
    fn str_enum_scaling_works() {
        assert_eq!(str_enum_scaling(0), "Auto");
        assert_eq!(str_enum_scaling(1), "Column (MUMPS-only, otherwise Auto)");
        assert_eq!(str_enum_scaling(2), "Diagonal (MUMPS-only, otherwise Auto)");
        assert_eq!(str_enum_scaling(3), "Max (UMFPACK-only, otherwise Auto)");
        assert_eq!(str_enum_scaling(4), "No");
        assert_eq!(str_enum_scaling(5), "RowCol (MUMPS-only, otherwise Auto)");
        assert_eq!(str_enum_scaling(6), "RowColIter (MUMPS-only, otherwise Auto)");
        assert_eq!(str_enum_scaling(7), "RowColRig (MUMPS-only, otherwise Auto)");
        assert_eq!(str_enum_scaling(8), "Sum (UMFPACK-only, otherwise Auto)");
    }

    #[test]
    fn str_mumps_ordering_works() {
        assert_eq!(str_mumps_ordering(0), "Amd");
        assert_eq!(str_mumps_ordering(1), "UserProvided");
        assert_eq!(str_mumps_ordering(2), "Amf");
        assert_eq!(str_mumps_ordering(3), "Scotch");
        assert_eq!(str_mumps_ordering(4), "Pord");
        assert_eq!(str_mumps_ordering(5), "Metis");
        assert_eq!(str_mumps_ordering(6), "Qamd");
        assert_eq!(str_mumps_ordering(7), "Auto");
        assert_eq!(str_mumps_ordering(123), "Unknown");
    }

    #[test]
    fn str_mumps_scaling_works() {
        assert_eq!(str_mumps_scaling(-1), "UserProvided");
        assert_eq!(str_mumps_scaling(0), "No");
        assert_eq!(str_mumps_scaling(1), "Diagonal");
        assert_eq!(str_mumps_scaling(3), "Column");
        assert_eq!(str_mumps_scaling(4), "RowCol");
        assert_eq!(str_mumps_scaling(7), "RowColIter");
        assert_eq!(str_mumps_scaling(8), "RowColRig");
        assert_eq!(str_mumps_scaling(77), "Auto");
        assert_eq!(str_mumps_scaling(123), "Unknown");
    }

    #[test]
    fn str_umfpack_ordering_works() {
        assert_eq!(str_umfpack_ordering(0), "Cholmod");
        assert_eq!(str_umfpack_ordering(1), "Amd");
        assert_eq!(str_umfpack_ordering(2), "UserProvided");
        assert_eq!(str_umfpack_ordering(3), "Metis");
        assert_eq!(str_umfpack_ordering(4), "Best");
        assert_eq!(str_umfpack_ordering(5), "No");
        assert_eq!(str_umfpack_ordering(6), "UserProvided");
        assert_eq!(str_umfpack_ordering(123), "Unknown");
    }

    #[test]
    fn str_umfpack_scaling_works() {
        assert_eq!(str_umfpack_scaling(0), "No");
        assert_eq!(str_umfpack_scaling(1), "Sum");
        assert_eq!(str_umfpack_scaling(2), "Max");
        assert_eq!(str_umfpack_scaling(123), "Unknown");
    }
}
