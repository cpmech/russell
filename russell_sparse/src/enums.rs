use serde::{Deserialize, Serialize};

/// Specifies the underlying library that does all the magic
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub enum Genie {
    /// Selects KLU (LU factorization)
    ///
    /// "Clark Kent" LU factorization algorithm (what SuperLU was before it became Super)
    ///
    /// Reference: <https://github.com/DrTimothyAldenDavis/SuiteSparse>
    Klu,

    /// Selects MUMPS (multi-frontal massively parallel sparse direct) solver
    ///
    /// Reference: <https://mumps-solver.org/index.php>
    Mumps,

    /// Selects UMFPACK (unsymmetric multi-frontal) solver
    ///
    /// Reference: <https://github.com/DrTimothyAldenDavis/SuiteSparse>
    Umfpack,
}

/// Specifies the type of matrix symmetry
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub enum Sym {
    /// Unknown symmetry (possibly unsymmetric)
    No,

    /// Symmetric with full representation (i.e., not triangular)
    YesFull,

    /// Symmetric with lower-triangle representation
    YesLower,

    /// Symmetric with upper-triangle representation
    YesUpper,
}

/// Holds options to handle a MatrixMarket when the matrix is specified as being symmetric
///
/// **Note:** This is ignored if the matrix is not symmetric.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub enum MMsym {
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
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
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
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
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
    pub fn from(genie: &str) -> Self {
        match genie.to_lowercase().as_str() {
            "klu" => Genie::Klu,
            "mumps" => Genie::Mumps,
            "umfpack" => Genie::Umfpack,
            _ => Genie::Umfpack,
        }
    }

    /// Returns the string representation
    pub fn to_string(&self) -> String {
        match self {
            Genie::Klu => "klu".to_string(),
            Genie::Mumps => "mumps".to_string(),
            Genie::Umfpack => "umfpack".to_string(),
        }
    }

    /// Returns the solver's required symmetry-representation
    pub fn symmetry(&self, symmetric: bool) -> Sym {
        if symmetric {
            match self {
                Genie::Klu => Sym::YesFull,
                Genie::Mumps => Sym::YesLower,
                Genie::Umfpack => Sym::YesFull,
            }
        } else {
            Sym::No
        }
    }
}

impl Sym {
    /// Returns true if the representation is Lower or Upper
    pub fn triangular(&self) -> bool {
        match self {
            Sym::YesLower => true,
            Sym::YesUpper => true,
            _ => false,
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
    fn derive_methods_work() {
        let genie = Genie::Mumps;
        let copy = genie;
        let clone = genie.clone();
        assert_eq!(format!("{:?}", genie), "Mumps");
        assert_eq!(copy, Genie::Mumps);
        assert_eq!(clone, Genie::Mumps);
        let json = serde_json::to_string(&genie).unwrap();
        let from_json: Genie = serde_json::from_str(&json).unwrap();
        assert_eq!(from_json, genie);

        let symmetry = Sym::YesLower;
        let copy = symmetry;
        let clone = symmetry.clone();
        assert_eq!(format!("{:?}", symmetry), "YesLower");
        assert_eq!(copy, Sym::YesLower);
        assert_eq!(clone, Sym::YesLower);
        let json = serde_json::to_string(&symmetry).unwrap();
        let from_json: Sym = serde_json::from_str(&json).unwrap();
        assert_eq!(from_json, symmetry);

        let handling = MMsym::LeaveAsLower;
        let copy = handling;
        let clone = handling.clone();
        assert_eq!(format!("{:?}", handling), "LeaveAsLower");
        assert_eq!(copy, MMsym::LeaveAsLower);
        assert_eq!(clone, MMsym::LeaveAsLower);
        let json = serde_json::to_string(&handling).unwrap();
        let from_json: MMsym = serde_json::from_str(&json).unwrap();
        assert_eq!(from_json, handling);

        let ordering = Ordering::Amd;
        let copy = ordering;
        let clone = ordering.clone();
        assert_eq!(format!("{:?}", ordering), "Amd");
        assert_eq!(format!("{:?}", copy), "Amd");
        assert_eq!(format!("{:?}", clone), "Amd");
        let json = serde_json::to_string(&ordering).unwrap();
        let from_json: Ordering = serde_json::from_str(&json).unwrap();
        assert_eq!(from_json, ordering);

        let scaling = Scaling::Column;
        let copy = scaling;
        let clone = scaling.clone();
        assert_eq!(format!("{:?}", scaling), "Column");
        assert_eq!(format!("{:?}", copy), "Column");
        assert_eq!(format!("{:?}", clone), "Column");
        let json = serde_json::to_string(&scaling).unwrap();
        let from_json: Scaling = serde_json::from_str(&json).unwrap();
        assert_eq!(from_json, scaling);
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
        assert_eq!(Genie::from("blah-blah-blah"), Genie::Umfpack);

        assert_eq!(Genie::from("Mumps"), Genie::Mumps);
        assert_eq!(Genie::from("Umfpack"), Genie::Umfpack);

        let genie = Genie::Mumps;
        assert_eq!(genie.to_string(), "mumps");
        assert_eq!(genie.symmetry(false), Sym::No);
        assert_eq!(genie.symmetry(true), Sym::YesLower);

        let genie = Genie::Umfpack;
        assert_eq!(genie.to_string(), "umfpack");
        assert_eq!(genie.symmetry(false,), Sym::No);
        assert_eq!(genie.symmetry(true), Sym::YesFull);
    }

    #[test]
    fn symmetry_functions_work() {
        assert_eq!(Sym::No.triangular(), false);
        assert_eq!(Sym::YesFull.triangular(), false);
        assert_eq!(Sym::YesLower.triangular(), true);
        assert_eq!(Sym::YesUpper.triangular(), true);
    }
}
