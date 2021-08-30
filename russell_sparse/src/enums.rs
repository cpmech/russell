pub enum EnumMumpsSymmetry {
    /// Unsymmetric matrix
    No,

    /// Positive-definite symmetric matrix
    PosDef,

    /// General symmetric matrix
    General,
}

pub fn enum_mumps_symmetry(selection: EnumMumpsSymmetry) -> i32 {
    match selection {
        EnumMumpsSymmetry::No => 0,
        EnumMumpsSymmetry::PosDef => 1,
        EnumMumpsSymmetry::General => 2,
    }
}
