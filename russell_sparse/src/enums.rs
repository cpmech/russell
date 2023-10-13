use crate::StrError;

/// Specifies the underlying library that does all the magic
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Genie {
    /// Selects MUMPS (multi-frontal massively parallel sparse direct) solver
    ///
    /// Reference: <https://mumps-solver.org/index.php>
    Mumps,

    /// Selects UMFPACK (unsymmetric multi-frontal) solver
    ///
    /// Reference: <https://github.com/DrTimothyAldenDavis/SuiteSparse>
    Umfpack,

    /// Selects Intel DSS (direct sparse solver)
    ///
    /// Reference: <https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-2/direct-sparse-solver-dss-interface-routines.html>
    IntelDss,
}

/// Specifies how the matrix components are stored
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Storage {
    /// Lower triangular storage for symmetric matrix (e.g., for MUMPS)
    Lower,

    /// Upper triangular storage for symmetric matrix (e.g., for Intel DSS)
    Upper,

    /// Full matrix storage for symmetric or unsymmetric matrix (e.g., for UMFPACK)
    Full,
}

/// Specifies the type of matrix symmetry
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Symmetry {
    /// General symmetric
    General(Storage),

    /// Symmetric and positive-definite
    PositiveDefinite(Storage),
}

/// Holds options to handle a MatrixMarket when the matrix is specified as being symmetric
///
/// **Note:** This is ignored if not the matrix is not specified as symmetric.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MMsymOption {
    /// Leave the storage as lower triangular (if symmetric)
    ///
    /// **Note:** Lower triangular is the standard MatrixMarket format.
    /// Thus, this option will do nothing.
    ///
    /// This option is useful for the MUMPS solver.
    LeaveAsLower,

    /// Convert the storage to upper triangular (if symmetric)
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
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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

impl Genie {
    /// Returns the Genie by name (default is umfpack)
    ///
    /// ```text
    /// "mumps"    => Genie::Mumps,
    /// "umfpack"  => Genie::Umfpack,
    /// "inteldss" => Genie::IntelDss,
    /// _          => Genie::Umfpack,
    /// ```
    pub fn from(genie: &str) -> Self {
        match genie.to_lowercase().as_str() {
            "mumps" => Genie::Mumps,
            "umfpack" => Genie::Umfpack,
            "inteldss" => Genie::IntelDss,
            _ => Genie::Umfpack,
        }
    }

    /// Returns the string representation
    /// ```text
    /// Genie::Mumps    => "mumps"
    /// Genie::Umfpack  => "umfpack"
    /// Genie::IntelDss => "inteldss"
    /// ```
    pub fn to_string(&self) -> String {
        match self {
            Genie::Mumps => "mumps".to_string(),
            Genie::Umfpack => "umfpack".to_string(),
            Genie::IntelDss => "inteldss".to_string(),
        }
    }

    /// Returns which storage is required by the solver
    ///
    /// ```text
    /// MUMPS     : Storage::Lower
    /// UMFPACK   : Storage::Full
    /// Intel DSS : Storage::Upper
    /// ````
    pub fn storage(&self) -> Storage {
        match self {
            Genie::Mumps => Storage::Lower,
            Genie::Umfpack => Storage::Full,
            Genie::IntelDss => Storage::Upper,
        }
    }

    /// Returns the solver's required symmetry/storage configuration
    pub fn symmetry(&self, symmetric: bool, positive_definite: bool) -> Option<Symmetry> {
        let storage = self.storage();
        if symmetric || positive_definite {
            if positive_definite {
                Some(Symmetry::PositiveDefinite(storage))
            } else {
                Some(Symmetry::General(storage))
            }
        } else {
            None
        }
    }

    /// Returns whether the sparse matrix indices must be one-based or not
    pub fn one_based(&self) -> bool {
        match self {
            Genie::Mumps => true,
            Genie::Umfpack => false,
            Genie::IntelDss => false,
        }
    }
}

impl Symmetry {
    /// Returns true if the storage is triangular (lower or upper)
    pub fn triangular(&self) -> bool {
        match self {
            Self::General(storage) => *storage != Storage::Full,
            Self::PositiveDefinite(storage) => *storage != Storage::Full,
        }
    }

    /// Returns true if the storage is lower triangular
    pub fn lower(&self) -> bool {
        match self {
            Self::General(storage) => *storage == Storage::Lower,
            Self::PositiveDefinite(storage) => *storage == Storage::Lower,
        }
    }

    /// Returns true if the storage is upper triangular
    pub fn upper(&self) -> bool {
        match self {
            Self::General(storage) => *storage == Storage::Upper,
            Self::PositiveDefinite(storage) => *storage == Storage::Upper,
        }
    }

    /// Returns status flags indicating the type of symmetry, if any
    ///
    /// Returns `(general_symmetric, positive_definite)`
    ///
    /// # Input
    ///
    /// * `must_be_lower` -- makes sure that the storage is Lower
    /// * `must_be_upper` -- makes sure that the storage is Upper
    ///
    /// # Output
    ///
    /// * `general_symmetric` -- 1 if true, 0 otherwise
    /// * `positive_definite` -- 1 if true, 0 otherwise
    pub fn status(&self, must_be_lower: bool, must_be_upper: bool) -> Result<(i32, i32), StrError> {
        match self {
            Self::General(storage) => {
                if must_be_lower && *storage != Storage::Lower {
                    return Err("if the matrix is general symmetric, the required storage is lower triangular");
                }
                if must_be_upper && *storage != Storage::Upper {
                    return Err("if the matrix is general symmetric, the required storage is upper triangular");
                }
                Ok((1, 0))
            }
            Self::PositiveDefinite(storage) => {
                if must_be_lower && *storage != Storage::Lower {
                    return Err("if the matrix is positive-definite, the required storage is lower triangular");
                }
                if must_be_upper && *storage != Storage::Upper {
                    return Err("if the matrix is positive-definite, the required storage is upper triangular");
                }
                Ok((0, 1))
            }
        }
    }
}

impl Ordering {
    /// Returns the Ordering by name (default is Auto)
    /// ```text
    /// "amd"     => Ordering::Amd,
    /// "amf"     => Ordering::Amf,
    /// "auto"    => Ordering::Auto,
    /// "best"    => Ordering::Best,
    /// "cholmod" => Ordering::Cholmod,
    /// "metis"   => Ordering::Metis,
    /// "no"      => Ordering::No,
    /// "pord"    => Ordering::Pord,
    /// "qamd"    => Ordering::Qamd,
    /// "scotch"  => Ordering::Scotch,
    /// _         => Ordering::Auto,
    /// ```
    pub fn from(ordering: &str) -> Self {
        match ordering.to_lowercase().as_str() {
            "amd" => Ordering::Amd,
            "amf" => Ordering::Amf,
            "auto" => Ordering::Auto,
            "best" => Ordering::Best,
            "cholmod" => Ordering::Cholmod,
            "metis" => Ordering::Metis,
            "no" => Ordering::No,
            "pord" => Ordering::Pord,
            "qamd" => Ordering::Qamd,
            "scotch" => Ordering::Scotch,
            _ => Ordering::Auto,
        }
    }
}

impl Scaling {
    /// Returns the Scaling by name (default is Auto)
    /// ```text
    /// "auto"       => Scaling::Auto,
    /// "column"     => Scaling::Column,
    /// "diagonal"   => Scaling::Diagonal,
    /// "max"        => Scaling::Max,
    /// "no"         => Scaling::No,
    /// "rowcol"     => Scaling::RowCol,
    /// "rowcoliter" => Scaling::RowColIter,
    /// "rowcolrig"  => Scaling::RowColRig,
    /// "sum"        => Scaling::Sum,
    /// _            => Scaling::Auto,
    /// ```
    pub fn from(scaling: &str) -> Self {
        match scaling.to_lowercase().as_str() {
            "auto" => Scaling::Auto,
            "column" => Scaling::Column,
            "diagonal" => Scaling::Diagonal,
            "max" => Scaling::Max,
            "no" => Scaling::No,
            "rowcol" => Scaling::RowCol,
            "rowcoliter" => Scaling::RowColIter,
            "rowcolrig" => Scaling::RowColRig,
            "sum" => Scaling::Sum,
            _ => Scaling::Auto,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clone_copy_and_debug_work() {
        let genie = Genie::Mumps;
        let copy = genie;
        let clone = genie.clone();
        assert_eq!(format!("{:?}", genie), "Mumps");
        assert_eq!(copy, Genie::Mumps);
        assert_eq!(clone, Genie::Mumps);

        let storage = Storage::Full;
        let copy = storage;
        let clone = storage.clone();
        assert_eq!(format!("{:?}", storage), "Full");
        assert_eq!(copy, Storage::Full);
        assert_eq!(clone, Storage::Full);

        let symmetry = Symmetry::PositiveDefinite(Storage::Lower);
        let copy = symmetry;
        let clone = symmetry.clone();
        assert_eq!(format!("{:?}", symmetry), "PositiveDefinite(Lower)");
        assert_eq!(copy, Symmetry::PositiveDefinite(Storage::Lower));
        assert_eq!(clone, Symmetry::PositiveDefinite(Storage::Lower));

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
    fn ordering_functions_work() {
        assert_eq!(Ordering::from("Amd"), Ordering::Amd);
        assert_eq!(Ordering::from("Amf"), Ordering::Amf);
        assert_eq!(Ordering::from("Auto"), Ordering::Auto);
        assert_eq!(Ordering::from("Best"), Ordering::Best);
        assert_eq!(Ordering::from("Cholmod"), Ordering::Cholmod);
        assert_eq!(Ordering::from("Metis"), Ordering::Metis);
        assert_eq!(Ordering::from("No"), Ordering::No);
        assert_eq!(Ordering::from("Pord"), Ordering::Pord);
        assert_eq!(Ordering::from("Qamd"), Ordering::Qamd);
        assert_eq!(Ordering::from("Scotch"), Ordering::Scotch);
        assert_eq!(Ordering::from("Unknown"), Ordering::Auto);

        assert_eq!(Ordering::from("amd"), Ordering::Amd);
        assert_eq!(Ordering::from("amf"), Ordering::Amf);
        assert_eq!(Ordering::from("auto"), Ordering::Auto);
        assert_eq!(Ordering::from("best"), Ordering::Best);
        assert_eq!(Ordering::from("cholmod"), Ordering::Cholmod);
        assert_eq!(Ordering::from("metis"), Ordering::Metis);
        assert_eq!(Ordering::from("no"), Ordering::No);
        assert_eq!(Ordering::from("pord"), Ordering::Pord);
        assert_eq!(Ordering::from("qamd"), Ordering::Qamd);
        assert_eq!(Ordering::from("scotch"), Ordering::Scotch);
        assert_eq!(Ordering::from("unknown"), Ordering::Auto);
    }

    #[test]
    fn scaling_functions_work() {
        assert_eq!(Scaling::from("Auto"), Scaling::Auto);
        assert_eq!(Scaling::from("Column"), Scaling::Column);
        assert_eq!(Scaling::from("Diagonal"), Scaling::Diagonal);
        assert_eq!(Scaling::from("Max"), Scaling::Max);
        assert_eq!(Scaling::from("No"), Scaling::No);
        assert_eq!(Scaling::from("RowCol"), Scaling::RowCol);
        assert_eq!(Scaling::from("RowColIter"), Scaling::RowColIter);
        assert_eq!(Scaling::from("RowColRig"), Scaling::RowColRig);
        assert_eq!(Scaling::from("Sum"), Scaling::Sum);
        assert_eq!(Scaling::from("Unknown"), Scaling::Auto);

        assert_eq!(Scaling::from("auto"), Scaling::Auto);
        assert_eq!(Scaling::from("column"), Scaling::Column);
        assert_eq!(Scaling::from("diagonal"), Scaling::Diagonal);
        assert_eq!(Scaling::from("max"), Scaling::Max);
        assert_eq!(Scaling::from("no"), Scaling::No);
        assert_eq!(Scaling::from("rowcol"), Scaling::RowCol);
        assert_eq!(Scaling::from("rowcoliter"), Scaling::RowColIter);
        assert_eq!(Scaling::from("rowcolrig"), Scaling::RowColRig);
        assert_eq!(Scaling::from("sum"), Scaling::Sum);
        assert_eq!(Scaling::from("unknown"), Scaling::Auto);
    }

    #[test]
    fn genie_functions_work() {
        assert_eq!(Genie::from("mumps"), Genie::Mumps);
        assert_eq!(Genie::from("umfpack"), Genie::Umfpack);
        assert_eq!(Genie::from("inteldss"), Genie::IntelDss);
        assert_eq!(Genie::from("blah-blah-blah"), Genie::Umfpack);

        assert_eq!(Genie::from("Mumps"), Genie::Mumps);
        assert_eq!(Genie::from("Umfpack"), Genie::Umfpack);
        assert_eq!(Genie::from("IntelDss"), Genie::IntelDss);

        let l = Storage::Lower;
        let u = Storage::Upper;
        let f = Storage::Full;

        let gl = Some(Symmetry::General(l));
        let gu = Some(Symmetry::General(u));
        let gf = Some(Symmetry::General(f));

        let pl = Some(Symmetry::PositiveDefinite(l));
        let pu = Some(Symmetry::PositiveDefinite(u));
        let pf = Some(Symmetry::PositiveDefinite(f));

        let genie = Genie::Mumps;
        assert_eq!(genie.to_string(), "mumps");
        assert_eq!(genie.storage(), l);
        assert_eq!(genie.symmetry(false, false), None);
        assert_eq!(genie.symmetry(true, false), gl);
        assert_eq!(genie.symmetry(false, true), pl);
        assert_eq!(genie.symmetry(true, true), pl);
        assert_eq!(genie.one_based(), true);

        let genie = Genie::Umfpack;
        assert_eq!(genie.to_string(), "umfpack");
        assert_eq!(genie.storage(), f);
        assert_eq!(genie.symmetry(false, false), None);
        assert_eq!(genie.symmetry(true, false), gf);
        assert_eq!(genie.symmetry(false, true), pf);
        assert_eq!(genie.symmetry(true, true), pf);
        assert_eq!(genie.one_based(), false);

        let genie = Genie::IntelDss;
        assert_eq!(genie.to_string(), "inteldss");
        assert_eq!(genie.storage(), u);
        assert_eq!(genie.symmetry(false, false), None);
        assert_eq!(genie.symmetry(true, false), gu);
        assert_eq!(genie.symmetry(false, true), pu);
        assert_eq!(genie.symmetry(true, true), pu);
        assert_eq!(genie.one_based(), false);
    }

    #[test]
    fn symmetry_functions_work() {
        let l = Storage::Lower;
        let u = Storage::Upper;
        let f = Storage::Full;

        let gl = Symmetry::General(l);
        let gu = Symmetry::General(u);
        let gf = Symmetry::General(f);

        let pl = Symmetry::PositiveDefinite(l);
        let pu = Symmetry::PositiveDefinite(u);
        let pf = Symmetry::PositiveDefinite(f);

        assert_eq!(gl.triangular(), true);
        assert_eq!(gu.triangular(), true);
        assert_eq!(gf.triangular(), false);
        assert_eq!(gl.lower(), true);
        assert_eq!(gu.lower(), false);
        assert_eq!(gf.lower(), false);
        assert_eq!(gl.upper(), false);
        assert_eq!(gu.upper(), true);
        assert_eq!(gf.upper(), false);

        assert_eq!(gl.status(true, false), Ok((1, 0)));
        assert_eq!(gl.status(false, false), Ok((1, 0)));
        assert_eq!(
            gl.status(false, true),
            Err("if the matrix is general symmetric, the required storage is upper triangular")
        );

        assert_eq!(gu.status(false, true), Ok((1, 0)));
        assert_eq!(gu.status(false, false), Ok((1, 0)));
        assert_eq!(
            gu.status(true, false),
            Err("if the matrix is general symmetric, the required storage is lower triangular")
        );

        assert_eq!(pl.triangular(), true);
        assert_eq!(pu.triangular(), true);
        assert_eq!(pf.triangular(), false);
        assert_eq!(pl.lower(), true);
        assert_eq!(pu.lower(), false);
        assert_eq!(pf.lower(), false);
        assert_eq!(pl.upper(), false);
        assert_eq!(pu.upper(), true);
        assert_eq!(pf.upper(), false);

        assert_eq!(pl.status(true, false), Ok((0, 1)));
        assert_eq!(pl.status(false, false), Ok((0, 1)));
        assert_eq!(
            pl.status(false, true),
            Err("if the matrix is positive-definite, the required storage is upper triangular")
        );

        assert_eq!(pu.status(false, true), Ok((0, 1)));
        assert_eq!(pu.status(false, false), Ok((0, 1)));
        assert_eq!(
            pu.status(true, false),
            Err("if the matrix is positive-definite, the required storage is lower triangular")
        );
    }
}
