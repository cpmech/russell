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
pub enum MMsymOption {
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

/// Ordering option
#[derive(Clone, Copy, Debug)]
pub enum Ordering {
    /// Ordering using the approximate minimum degree
    Amd,

    /// Ordering using the approximate minimum fill-in (MUMPS-only, otherwise Auto)
    Amf,

    /// Automatic ordering method selection
    Auto,

    /// Try three methods and take the best (UMFPACK-only, otherwise Auto)
    Best,

    /// Use Amd for symmetric, Colamd for unsymmetric, or Metis (UMFPACK-only, otherwise Auto)
    Cholmod,

    /// Ordering by Karpis & Kumar from the University of Minnesota
    Metis,

    /// The matrix is factorized as-is (UMFPACK-only, otherwise Auto)
    No,

    /// Ordering by Schulze from the University of Paderborn (MUMPS-only, otherwise Auto)
    Pord,

    /// Ordering using the automatic quasi-dense row detection (MUMPS-only, otherwise Auto)
    Qamd,

    /// Ordering using the Scotch package (MUMPS-only, otherwise Auto)
    Scotch,
}

/// Scaling option
#[derive(Clone, Copy, Debug)]
pub enum Scaling {
    /// Automatic scaling method selection
    Auto,

    /// Column scaling (MUMPS-only, otherwise Auto)
    Column,

    /// Diagonal scaling (MUMPS-only, otherwise Auto)
    Diagonal,

    /// Use the max absolute value in the row (UMFPACK-only, otherwise Auto)
    Max,

    /// No scaling applied or computed
    No,

    /// Row and column scaling based on infinite row/column norms (MUMPS-only, otherwise Auto)
    RowCol,

    /// Simultaneous row and column iterative scaling (MUMPS-only, otherwise Auto)
    RowColIter,

    /// Similar to RcIterative but more rigorous and expensive to compute (MUMPS-only, otherwise Auto)
    RowColRig,

    /// Use the sum of the absolute value in the row (UMFPACK-only, otherwise Auto)
    Sum,
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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{enum_ordering, enum_scaling, Layout, MMsymOption, Ordering, Scaling};

    #[test]
    fn clone_copy_and_debug_work() {
        let layout = Layout::Full;
        let copy = layout;
        let clone = layout.clone();
        assert_eq!(format!("{:?}", layout), "Full");
        assert_eq!(copy, Layout::Full);
        assert_eq!(clone, Layout::Full);

        let handling = MMsymOption::LeaveAsLower;
        let copy = handling;
        let clone = handling.clone();
        assert_eq!(format!("{:?}", handling), "LeaveAsLower");
        assert_eq!(copy, MMsymOption::LeaveAsLower);
        assert_eq!(clone, MMsymOption::LeaveAsLower);

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
}
