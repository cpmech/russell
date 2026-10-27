use crate::StrError;
use serde::{Deserialize, Serialize};

/// Selects a Krylov method for the iterative solvers
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Deserialize, Serialize)]
pub enum IterativeMethod {
    /// Preconditioned conjugate gradients, for symmetric positive-definite systems
    Cg,

    /// Right-preconditioned BiCGStab, for general square systems
    BiCgStab,

    /// Right-preconditioned restarted GMRES, for general square systems
    Gmres,
}

/// Specifies the underlying library that does all the magic
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Deserialize, Serialize)]
pub enum Genie {
    /// Selects the pure-Rust direct solver
    ///
    /// Reference: <https://github.com/cpmech/algebra_linear>
    Direct,

    /// Selects a pure-Rust iterative solver with algebraic multigrid preconditioning
    ///
    /// Reference: <https://github.com/ddemidov/amgcl>
    Iterative(IterativeMethod),

    /// Selects MUMPS (multi-frontal massively parallel sparse direct) solver
    ///
    /// Reference: <https://mumps-solver.org/index.php>
    Mumps,

    /// Selects UMFPACK (unsymmetric multi-frontal) solver
    ///
    /// Requires the `umfpack` feature, which is not enabled by default.
    ///
    /// Reference: <https://github.com/DrTimothyAldenDavis/SuiteSparse>
    Umfpack,
}

/// Indicates whether the matrix is symmetric or not (includes the storage type)
///
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub enum Sym {
    /// Not symmetric or unknown
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

/// Specifies the ordering option
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub enum Ordering {
    /// Ordering using the approximate minimum degree
    ///
    /// * MUMPS: ✅ available
    /// * Direct: ❌ unavailable; defaults to automatic
    /// * UMFPACK: ✅ available
    Amd,

    /// Ordering using the approximate minimum fill-in
    ///
    /// * MUMPS: ✅ available
    /// * Direct: ❌ unavailable; defaults to automatic
    /// * UMFPACK: ❌ unavailable; defaults to automatic
    Amf,

    /// Automatic ordering method selection (default)
    ///
    /// * MUMPS: ✅ available
    /// * Direct: ✅ available (Colamd)
    /// * UMFPACK: ✅ available
    Auto,

    /// Try three methods and take the best
    ///
    /// * MUMPS: ❌ unavailable; defaults to automatic
    /// * Direct: ❌ unavailable; defaults to automatic
    /// * UMFPACK: ✅ available
    Best,

    /// Block triangular form (BTF) combined with COLAMD. Supports global pivoting
    ///
    /// * MUMPS: ❌ unavailable; defaults to automatic
    /// * Direct: ❌ unavailable; defaults to automatic
    /// * UMFPACK: ❌ unavailable; defaults to automatic
    BtfColamd,

    /// Use Amd for symmetric, Colamd for unsymmetric, or Metis
    ///
    /// * MUMPS: ❌ unavailable; defaults to automatic
    /// * Direct: ❌ unavailable; defaults to automatic
    /// * UMFPACK: ✅ available
    Cholmod,

    /// Use the column approximate minimum degree ordering algorithm
    ///
    /// * MUMPS: ❌ unavailable; defaults to automatic
    /// * Direct: ✅ available
    /// * UMFPACK: ❌ unavailable; defaults to automatic
    Colamd,

    /// Ordering by Karpis & Kumar from the University of Minnesota
    ///
    /// * MUMPS: ✅ available
    /// * Direct: ❌ unavailable; defaults to automatic
    /// * UMFPACK: ✅ available
    Metis,

    /// The matrix is factorized as-is
    ///
    /// * MUMPS: ❌ unavailable; defaults to automatic
    /// * Direct: ❌ unavailable; defaults to automatic
    /// * UMFPACK: ✅ available
    No,

    /// Ordering by Schulze from the University of Paderborn
    ///
    /// * MUMPS: ✅ available
    /// * Direct: ❌ unavailable; defaults to automatic
    /// * UMFPACK: ❌ unavailable; defaults to automatic
    Pord,

    /// Ordering using the automatic quasi-dense row detection
    ///
    /// * MUMPS: ✅ available
    /// * Direct: ❌ unavailable; defaults to automatic
    /// * UMFPACK: ❌ unavailable; defaults to automatic
    Qamd,

    /// Ordering using the Scotch package
    ///
    /// * MUMPS: ✅ available
    /// * Direct: ❌ unavailable; defaults to automatic
    /// * UMFPACK: ❌ unavailable; defaults to automatic
    Scotch,
}

/// Specifies the scaling option
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub enum Scaling {
    /// Automatic scaling method selection (default)
    ///
    /// * MUMPS: ✅ available
    /// * Direct: ✅ available (none)
    /// * UMFPACK: ✅ available (defaults to sum-of-row scaling)
    Auto,

    /// Column scaling
    ///
    /// * MUMPS: ✅ available
    /// * Direct: ❌ unavailable; defaults to automatic
    /// * UMFPACK: ❌ unavailable; defaults to automatic
    Column,

    /// Diagonal scaling
    ///
    /// * MUMPS: ✅ available
    /// * Direct: ❌ unavailable; defaults to automatic
    /// * UMFPACK: ❌ unavailable; defaults to automatic
    Diagonal,

    /// Divide each row by the max absolute value in the row
    ///
    /// * MUMPS: ❌ unavailable; defaults to automatic
    /// * Direct: ❌ unavailable; defaults to automatic
    /// * UMFPACK: ✅ available
    Max,

    /// No scaling applied or computed
    ///
    /// * MUMPS: ✅ available
    /// * Direct: ✅ available
    /// * UMFPACK: ✅ available
    No,

    /// Row and column scaling based on infinite row/column norms
    ///
    /// * MUMPS: ✅ available
    /// * Direct: ❌ unavailable; defaults to automatic
    /// * UMFPACK: ❌ unavailable; defaults to automatic
    RowCol,

    /// Simultaneous row and column iterative scaling
    ///
    /// * MUMPS: ✅ available
    /// * Direct: ❌ unavailable; defaults to automatic
    /// * UMFPACK: ❌ unavailable; defaults to automatic
    RowColIter,

    /// Similar to RowColIter but more rigorous and expensive to compute
    ///
    /// * MUMPS: ✅ available
    /// * Direct: ❌ unavailable; defaults to automatic
    /// * UMFPACK: ❌ unavailable; defaults to automatic
    RowColRig,

    /// Divide each row by the sum of the absolute values in the row
    ///
    /// * MUMPS: ❌ unavailable; defaults to automatic
    /// * Direct: ❌ unavailable; defaults to automatic
    /// * UMFPACK: ✅ available (also the default)
    Sum,
}
impl Genie {
    /// Returns the Genie by name
    ///
    /// The recognized names are `direct`, `cg`, `bicgstab`, `gmres`, `mumps`,
    /// and `umfpack`, compared case-insensitively. Any other name yields an
    /// error.
    ///
    /// # Output
    ///
    /// Returns the [`Genie`] matching `genie`, or
    /// `"genie must be one of: direct, cg, bicgstab, gmres, mumps, umfpack"`.
    pub fn from(genie: &str) -> Result<Self, StrError> {
        match genie.to_lowercase().as_str() {
            "direct" => Ok(Genie::Direct),
            "cg" => Ok(Genie::Iterative(IterativeMethod::Cg)),
            "bicgstab" => Ok(Genie::Iterative(IterativeMethod::BiCgStab)),
            "gmres" => Ok(Genie::Iterative(IterativeMethod::Gmres)),
            "mumps" => Ok(Genie::Mumps),
            "umfpack" => Ok(Genie::Umfpack),
            _ => Err("genie must be one of: direct, cg, bicgstab, gmres, mumps, umfpack"),
        }
    }

    /// Tells whether this solver is available in the current build
    ///
    /// The Direct and Iterative solvers are pure Rust and always available. The
    /// other two are compiled in only when the crate feature of the same name
    /// is enabled:
    ///
    /// * `Mumps` requires the `mumps` feature
    /// * `Umfpack` requires the `umfpack` feature
    ///
    /// The answer is fixed at compile time, so a solver reported as unavailable
    /// can never be constructed; trying would make [crate::LinSolver::new] return
    /// an error naming the feature to enable.
    pub fn is_available(&self) -> bool {
        match self {
            Genie::Direct => true,
            Genie::Iterative(_) => true,
            Genie::Mumps => cfg!(feature = "mumps"),
            Genie::Umfpack => cfg!(feature = "umfpack"),
        }
    }

    /// Returns the solver's required Sym type
    pub fn get_sym(&self, symmetric: bool) -> Sym {
        if symmetric {
            match self {
                Genie::Direct => Sym::YesFull,
                Genie::Iterative(_) => Sym::YesFull,
                Genie::Mumps => Sym::YesLower,
                Genie::Umfpack => Sym::YesFull,
            }
        } else {
            Sym::No
        }
    }

    /// Tells whether this Genie selects an iterative solver
    pub fn is_iterative(&self) -> bool {
        matches!(self, Genie::Iterative(_))
    }
}

impl std::fmt::Display for Genie {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Genie::Direct => write!(f, "direct"),
            Genie::Iterative(IterativeMethod::Cg) => write!(f, "cg"),
            Genie::Iterative(IterativeMethod::BiCgStab) => write!(f, "bicgstab"),
            Genie::Iterative(IterativeMethod::Gmres) => write!(f, "gmres"),
            Genie::Mumps => write!(f, "mumps"),
            Genie::Umfpack => write!(f, "umfpack"),
        }
    }
}

impl Sym {
    /// Returns true if the representation is Lower or Upper
    pub fn triangular(&self) -> bool {
        matches!(self, Sym::YesLower | Sym::YesUpper)
    }

    /// Returns true if symmetric
    pub fn yes(&self) -> bool {
        *self != Sym::No
    }

    /// Returns true if not symmetric
    pub fn no(&self) -> bool {
        *self == Sym::No
    }
}

impl Ordering {
    /// Returns the Ordering by name (default is Auto)
    pub fn from(ordering: &str) -> Self {
        match ordering.to_lowercase().as_str() {
            "amd" => Ordering::Amd,
            "amf" => Ordering::Amf,
            "auto" => Ordering::Auto,
            "best" => Ordering::Best,
            "cholmod" => Ordering::Cholmod,
            "colamd" => Ordering::Colamd,
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
    use std::collections::HashMap;

    #[test]
    #[allow(clippy::clone_on_copy)] // test the derived Clone implementation
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
        let _ = HashMap::from([(genie, 1u8)]);

        let symmetric = Sym::YesLower;
        let copy = symmetric;
        let clone = symmetric.clone();
        assert_eq!(format!("{:?}", symmetric), "YesLower");
        assert_eq!(copy, Sym::YesLower);
        assert_eq!(clone, Sym::YesLower);
        let json = serde_json::to_string(&symmetric).unwrap();
        let from_json: Sym = serde_json::from_str(&json).unwrap();
        assert_eq!(from_json, symmetric);

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
        assert_eq!(Ordering::from("Colamd"), Ordering::Colamd);
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
        assert_eq!(Ordering::from("colamd"), Ordering::Colamd);
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
        assert_eq!(Genie::from("direct"), Ok(Genie::Direct));
        assert_eq!(Genie::from("cg"), Ok(Genie::Iterative(IterativeMethod::Cg)));
        assert_eq!(Genie::from("bicgstab"), Ok(Genie::Iterative(IterativeMethod::BiCgStab)));
        assert_eq!(Genie::from("gmres"), Ok(Genie::Iterative(IterativeMethod::Gmres)));
        assert_eq!(Genie::from("mumps"), Ok(Genie::Mumps));
        assert_eq!(Genie::from("umfpack"), Ok(Genie::Umfpack));

        assert_eq!(Genie::from("Direct"), Ok(Genie::Direct));
        assert_eq!(Genie::from("BiCgStab"), Ok(Genie::Iterative(IterativeMethod::BiCgStab)));
        assert_eq!(Genie::from("GMRES"), Ok(Genie::Iterative(IterativeMethod::Gmres)));
        assert_eq!(Genie::from("Mumps"), Ok(Genie::Mumps));
        assert_eq!(Genie::from("Umfpack"), Ok(Genie::Umfpack));

        // unknown names are errors; there is no fallback and no aliases
        let message = "genie must be one of: direct, cg, bicgstab, gmres, mumps, umfpack";
        assert_eq!(Genie::from("native"), Err(message));
        assert_eq!(Genie::from("iterative"), Err(message));
        assert_eq!(Genie::from("blah-blah-blah"), Err(message));

        let genie = Genie::Direct;
        assert_eq!(genie.to_string(), "direct");
        assert_eq!(genie.get_sym(false), Sym::No);
        assert_eq!(genie.get_sym(true), Sym::YesFull);
        assert!(!genie.is_iterative());

        let genie = Genie::Iterative(IterativeMethod::Cg);
        assert_eq!(genie.to_string(), "cg");
        assert_eq!(genie.get_sym(false), Sym::No);
        assert_eq!(genie.get_sym(true), Sym::YesFull);
        assert!(genie.is_iterative());

        let genie = Genie::Iterative(IterativeMethod::BiCgStab);
        assert_eq!(genie.to_string(), "bicgstab");
        assert!(genie.is_iterative());

        let genie = Genie::Iterative(IterativeMethod::Gmres);
        assert_eq!(genie.to_string(), "gmres");
        assert!(genie.is_iterative());

        let genie = Genie::Mumps;
        assert_eq!(genie.to_string(), "mumps");
        assert_eq!(genie.get_sym(false), Sym::No);
        assert_eq!(genie.get_sym(true), Sym::YesLower);
        assert!(!genie.is_iterative());

        let genie = Genie::Umfpack;
        assert_eq!(genie.to_string(), "umfpack");
        assert_eq!(genie.get_sym(false,), Sym::No);
        assert_eq!(genie.get_sym(true), Sym::YesFull);
        assert!(!genie.is_iterative());
    }

    #[test]
    fn genie_iterative_serde_round_trips() {
        let genie = Genie::Iterative(IterativeMethod::Cg);
        let json = serde_json::to_string(&genie).unwrap();
        assert_eq!(json, "{\"Iterative\":\"Cg\"}");
        let from_json: Genie = serde_json::from_str(&json).unwrap();
        assert_eq!(from_json, genie);

        // the old name is not an alias for Direct
        assert!(serde_json::from_str::<Genie>("\"Native\"").is_err());
    }

    #[test]
    fn genie_availability_matches_the_features() {
        assert!(Genie::Direct.is_available());
        assert!(Genie::Iterative(IterativeMethod::Cg).is_available());
        assert!(Genie::Iterative(IterativeMethod::BiCgStab).is_available());
        assert!(Genie::Iterative(IterativeMethod::Gmres).is_available());
        assert_eq!(Genie::Mumps.is_available(), cfg!(feature = "mumps"));
        assert_eq!(Genie::Umfpack.is_available(), cfg!(feature = "umfpack"));
    }

    #[test]
    fn sym_functions_work() {
        assert_eq!(Sym::No.no(), true);
        assert_eq!(Sym::YesFull.no(), false);
        assert_eq!(Sym::YesLower.no(), false);
        assert_eq!(Sym::YesUpper.no(), false);

        assert_eq!(Sym::No.yes(), false);
        assert_eq!(Sym::YesFull.yes(), true);
        assert_eq!(Sym::YesLower.yes(), true);
        assert_eq!(Sym::YesUpper.yes(), true);

        assert_eq!(Sym::No.triangular(), false);
        assert_eq!(Sym::YesFull.triangular(), false);
        assert_eq!(Sym::YesLower.triangular(), true);
        assert_eq!(Sym::YesUpper.triangular(), true);
    }
}
