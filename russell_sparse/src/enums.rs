use serde::{Deserialize, Serialize};

/// Specifies the underlying library that does all the magic
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub enum Genie {
    /// Selects cuDSS (NVIDIA CUDA Direct Sparse Solver)
    ///
    /// Reference: <https://developer.nvidia.com/cudss>
    Cudss,

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

/// Indicates whether the matrix is symmetric or not (includes the storage type)
///
/// **Note:** For unsymmetric matrices, when using the [Genie::Cudss], it is recommended to
/// select [Matching::Auto] when the matrix contains some zero entries on its diagonal.
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
    /// * cuDSS: ✅ available. Approximate minimum degree (AMD) reordering.
    /// * KLU: ✅ available
    /// * MUMPS: ✅ available
    /// * UMFPACK: ✅ available
    Amd,

    /// Ordering using the approximate minimum fill-in
    ///
    /// * cuDSS: ❌ unavailable; defaults to automatic
    /// * KLU: ❌ unavailable; defaults to automatic
    /// * MUMPS: ✅ available
    /// * UMFPACK: ❌ unavailable; defaults to automatic
    Amf,

    /// Automatic ordering method selection (default)
    ///
    /// * cuDSS: ✅ available. The default algorithm for reordering (equivalent to CUDSS_REORDERING_ALG_NESTED_DISSECTION).
    /// * KLU: ✅ available
    /// * MUMPS: ✅ available
    /// * UMFPACK: ✅ available
    Auto,

    /// Try three methods and take the best
    ///
    /// * cuDSS: ❌ unavailable; defaults to automatic
    /// * KLU: ❌ unavailable; defaults to automatic
    /// * MUMPS: ❌ unavailable; defaults to automatic
    /// * UMFPACK: ✅ available
    Best,

    /// Block triangular form (BTF) combined with COLAMD. Supports global pivoting
    ///
    /// * cuDSS: ✅ available. Block triangular form (BTF) combined with COLAMD. Supports global pivoting
    /// * KLU: ❌ unavailable; defaults to automatic
    /// * MUMPS: ❌ unavailable; defaults to automatic
    /// * UMFPACK: ❌ unavailable; defaults to automatic
    BtfColamd,

    /// Use Amd for symmetric, Colamd for unsymmetric, or Metis
    ///
    /// * cuDSS: ❌ unavailable; defaults to automatic
    /// * KLU: ❌ unavailable; defaults to automatic
    /// * MUMPS: ❌ unavailable; defaults to automatic
    /// * UMFPACK: ✅ available
    Cholmod,

    /// Use the column approximate minimum degree ordering algorithm
    ///
    /// * cuDSS: ✅ available. COLAMD with trivial block structure. Supports global pivoting.
    /// * KLU: ✅ available
    /// * MUMPS: ❌ unavailable; defaults to automatic
    /// * UMFPACK: ❌ unavailable; defaults to automatic
    Colamd,

    /// Ordering by Karpis & Kumar from the University of Minnesota
    ///
    /// * cuDSS: ✅ available. Nested dissection algorithm based on METIS
    /// * KLU: ❌ unavailable; defaults to automatic
    /// * MUMPS: ✅ available
    /// * UMFPACK: ✅ available
    Metis,

    /// The matrix is factorized as-is
    ///
    /// * cuDSS: ✅ available. Uses natural (identity) order for the internal ordering when no user permutation is supplied.
    /// * KLU: ❌ unavailable; defaults to automatic
    /// * MUMPS: ❌ unavailable; defaults to automatic
    /// * UMFPACK: ✅ available
    No,

    /// Ordering by Schulze from the University of Paderborn
    ///
    /// * cuDSS: ❌ unavailable; defaults to automatic
    /// * KLU: ❌ unavailable; defaults to automatic
    /// * MUMPS: ✅ available
    /// * UMFPACK: ❌ unavailable; defaults to automatic
    Pord,

    /// Ordering using the automatic quasi-dense row detection
    ///
    /// * cuDSS: ❌ unavailable; defaults to automatic
    /// * KLU: ❌ unavailable; defaults to automatic
    /// * MUMPS: ✅ available
    /// * UMFPACK: ❌ unavailable; defaults to automatic
    Qamd,

    /// Ordering using the Scotch package
    ///
    /// * cuDSS: ❌ unavailable; defaults to automatic
    /// * KLU: ❌ unavailable; defaults to automatic
    /// * MUMPS: ✅ available
    /// * UMFPACK: ❌ unavailable; defaults to automatic
    Scotch,
}

/// Specifies the scaling option
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub enum Scaling {
    /// Automatic scaling method selection (default)
    ///
    /// * cuDSS: ❌ no scaling parameter available
    /// * KLU: ✅ available
    /// * MUMPS: ✅ available
    /// * UMFPACK: ✅ available (defaults to sum-of-row scaling)
    Auto,

    /// Column scaling
    ///
    /// * cuDSS: ❌ no scaling parameter available
    /// * KLU: ❌ unavailable; defaults to automatic
    /// * MUMPS: ✅ available
    /// * UMFPACK: ❌ unavailable; defaults to automatic
    Column,

    /// Diagonal scaling
    ///
    /// * cuDSS: ❌ no scaling parameter available
    /// * KLU: ❌ unavailable; defaults to automatic
    /// * MUMPS: ✅ available
    /// * UMFPACK: ❌ unavailable; defaults to automatic
    Diagonal,

    /// Divide each row by the max absolute value in the row
    ///
    /// * cuDSS: ❌ no scaling parameter available
    /// * KLU: ✅ available
    /// * MUMPS: ❌ unavailable; defaults to automatic
    /// * UMFPACK: ✅ available
    Max,

    /// No scaling applied or computed
    ///
    /// * cuDSS: ❌ no scaling parameter available
    /// * KLU: ✅ available
    /// * MUMPS: ✅ available
    /// * UMFPACK: ✅ available
    No,

    /// Row and column scaling based on infinite row/column norms
    ///
    /// * cuDSS: ❌ no scaling parameter available
    /// * KLU: ❌ unavailable; defaults to automatic
    /// * MUMPS: ✅ available
    /// * UMFPACK: ❌ unavailable; defaults to automatic
    RowCol,

    /// Simultaneous row and column iterative scaling
    ///
    /// * cuDSS: ❌ no scaling parameter available
    /// * KLU: ❌ unavailable; defaults to automatic
    /// * MUMPS: ✅ available
    /// * UMFPACK: ❌ unavailable; defaults to automatic
    RowColIter,

    /// Similar to RowColIter but more rigorous and expensive to compute
    ///
    /// * cuDSS: ❌ no scaling parameter available
    /// * KLU: ❌ unavailable; defaults to automatic
    /// * MUMPS: ✅ available
    /// * UMFPACK: ❌ unavailable; defaults to automatic
    RowColRig,

    /// Divide each row by the sum of the absolute values in the row
    ///
    /// * cuDSS: ❌ no scaling parameter available
    /// * KLU: ✅ available
    /// * MUMPS: ❌ unavailable; defaults to automatic
    /// * UMFPACK: ✅ available (also the default)
    Sum,
}

/// Specifies the matching algorithm (cuDSS only)
///
/// **Important:** Note that the default is [Matching::None].
///
/// **Note:** It is recommended to select [Matching::Auto] when dealing with
/// unsymmetric matrices containing some zero entries on their diagonal.
///
/// See: <https://docs.nvidia.com/cuda/cudss/types.html#cudssmatchingalg-t>
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub enum Matching {
    /// No matching algorithm (default)
    ///
    /// CUDSS_MATCHING_ALG_NONE
    None,

    /// Automatic selection
    ///
    /// Same as CUDSS_MATCHING_ALG_MAX_DIAG_PRODUCT (the most robust option).
    /// Matching with scaling; requires matrix values during analysis.
    ///
    /// CUDSS_MATCHING_ALG_AUTO
    Auto,

    /// Column permutation to maximize the number of diagonal entries (values arbitrary).
    /// MC64 JOB=1. Does not use matrix values; not recommended unless justified.
    ///
    /// CUDSS_MATCHING_ALG_MAX_DIAG_COUNT
    MaxDiagCount,

    /// Column permutation to maximize the smallest value on the diagonal. MC64 JOB=2.
    ///
    /// CUDSS_MATCHING_ALG_MAX_MIN_DIAG
    MaxMinDiag,

    /// Alternate algorithm to maximize the smallest value on the diagonal. MC64 JOB=3.
    /// May differ in performance from CUDSS_MATCHING_ALG_MAX_MIN_DIAG.
    ///
    /// CUDSS_MATCHING_ALG_MAX_MIN_DIAG_ALT
    MaxMinDiagAlt,

    /// Column permutation to maximize the sum of diagonal entries. MC64 JOB=4.
    ///
    /// CUDSS_MATCHING_ALG_MAX_DIAG_SUM
    MaxDiagSum,

    /// Column permutation to maximize the product of diagonal entries;
    /// also computes row/column scaling so that nonzero diagonal entries are 1 in
    /// absolute value and off-diagonal entries are ≤ 1. MC64 JOB=5. Most impactful for accuracy;
    /// requires matrix values during analysis.
    ///
    /// CUDSS_MATCHING_ALG_MAX_DIAG_PRODUCT
    MaxDiagProduct,
}

/// Specifies the pivoting strategy (cuDSS only)
///
/// See: <https://docs.nvidia.com/cuda/cudss/types.html#cudsspivottype-t>
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub enum Pivoting {
    /// Automatic pivoting strategy (default)
    ///
    /// Automatically selects the appropriate pivot type based on the reordering algorithm and matrix type:
    /// 1. For [Ordering::Auto] and [Ordering::Amd]:
    ///     * Symmetric/Hermitian indefinite matrices: Uses [Pivoting::Diagonal]
    ///     * General matrices: Uses [Pivoting::LocalBlock]
    /// 2. For [Ordering::BtfColamd] and [Ordering::Colamd]: Uses [Pivoting::GlobalCol]
    ///
    /// CUDSS_PIVOT_AUTO
    Auto,

    /// No pivoting
    ///
    /// Used with [Ordering::Auto] and [Ordering::Amd] (MetisND and AMD).
    ///
    /// CUDSS_PIVOT_NONE
    None,

    /// Global column pivoting
    ///
    /// Used with [Ordering::BtfColamd] and [Ordering::Colamd].
    ///
    /// CUDSS_PIVOT_GLOBAL_COL
    GlobalCol,

    /// Global row pivoting
    ///
    /// Used with [Ordering::BtfColamd] and [Ordering::Colamd].
    ///
    /// CUDSS_PIVOT_GLOBAL_ROW
    GlobalRow,

    /// Diagonal pivoting
    ///
    /// Used with [Ordering::Auto] and [Ordering::Amd] (MetisND and AMD).
    ///
    /// For symmetric/Hermitian indefinite matrices, searches for pivot elements only within the diagonal of the supernode.
    ///
    /// CUDSS_PIVOT_DIAGONAL
    Diagonal,

    /// Complete block pivoting
    ///
    /// Use with [Ordering::Auto] and [Ordering::Amd].
    ///
    /// For general matrices, searches for pivot elements within the entire diagonal block of the supernode.
    ///
    /// CUDSS_PIVOT_LOCAL_BLOCK
    LocalBlock,
}

impl Genie {
    /// Returns the Genie by name (default is umfpack)
    pub fn from(genie: &str) -> Self {
        match genie.to_lowercase().as_str() {
            "cudss" => Genie::Cudss,
            "klu" => Genie::Klu,
            "mumps" => Genie::Mumps,
            "umfpack" => Genie::Umfpack,
            _ => Genie::Umfpack,
        }
    }

    /// Returns the string representation
    pub fn to_string(&self) -> String {
        match self {
            Genie::Cudss => "cudss".to_string(),
            Genie::Klu => "klu".to_string(),
            Genie::Mumps => "mumps".to_string(),
            Genie::Umfpack => "umfpack".to_string(),
        }
    }

    /// Returns the solver's required Sym type
    pub fn get_sym(&self, symmetric: bool) -> Sym {
        if symmetric {
            match self {
                Genie::Cudss => Sym::YesLower,
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

impl Matching {
    /// Returns the Matching by name (default is None)
    pub fn from(matching: &str) -> Self {
        match matching.to_lowercase().as_str() {
            "none" => Matching::None,
            "auto" => Matching::Auto,
            "maxdiagcount" => Matching::MaxDiagCount,
            "maxmindiag" => Matching::MaxMinDiag,
            "maxmindiagalt" => Matching::MaxMinDiagAlt,
            "maxdiagsum" => Matching::MaxDiagSum,
            "maxdiagproduct" => Matching::MaxDiagProduct,
            _ => Matching::None, // <<< this is the default
        }
    }
}

impl Pivoting {
    /// Returns the Pivoting by name (default is Auto)
    pub fn from(pivoting: &str) -> Self {
        match pivoting.to_lowercase().as_str() {
            "auto" => Pivoting::Auto,
            "none" => Pivoting::None,
            "globalcol" => Pivoting::GlobalCol,
            "globalrow" => Pivoting::GlobalRow,
            "diagonal" => Pivoting::Diagonal,
            "localblock" => Pivoting::LocalBlock,
            _ => Pivoting::Auto,
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

        let matching = Matching::MaxDiagProduct;
        let copy = matching;
        let clone = matching.clone();
        assert_eq!(format!("{:?}", matching), "MaxDiagProduct");
        assert_eq!(format!("{:?}", copy), "MaxDiagProduct");
        assert_eq!(format!("{:?}", clone), "MaxDiagProduct");
        let json = serde_json::to_string(&matching).unwrap();
        let from_json: Matching = serde_json::from_str(&json).unwrap();
        assert_eq!(from_json, matching);
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
    fn matching_functions_work() {
        assert_eq!(Matching::from("None"), Matching::None);
        assert_eq!(Matching::from("Auto"), Matching::Auto);
        assert_eq!(Matching::from("MaxDiagCount"), Matching::MaxDiagCount);
        assert_eq!(Matching::from("MaxMinDiag"), Matching::MaxMinDiag);
        assert_eq!(Matching::from("MaxMinDiagAlt"), Matching::MaxMinDiagAlt);
        assert_eq!(Matching::from("MaxDiagSum"), Matching::MaxDiagSum);
        assert_eq!(Matching::from("MaxDiagProduct"), Matching::MaxDiagProduct);

        assert_eq!(Matching::from("none"), Matching::None);
        assert_eq!(Matching::from("auto"), Matching::Auto);
        assert_eq!(Matching::from("maxdiagcount"), Matching::MaxDiagCount);
        assert_eq!(Matching::from("maxmindiag"), Matching::MaxMinDiag);
        assert_eq!(Matching::from("maxmindiagalt"), Matching::MaxMinDiagAlt);
        assert_eq!(Matching::from("maxdiagsum"), Matching::MaxDiagSum);
        assert_eq!(Matching::from("maxdiagproduct"), Matching::MaxDiagProduct);

        assert_eq!(Matching::from("unknown"), Matching::None);
    }

    #[test]
    fn pivoting_functions_work() {
        assert_eq!(Pivoting::from("Auto"), Pivoting::Auto);
        assert_eq!(Pivoting::from("None"), Pivoting::None);
        assert_eq!(Pivoting::from("GlobalCol"), Pivoting::GlobalCol);
        assert_eq!(Pivoting::from("GlobalRow"), Pivoting::GlobalRow);
        assert_eq!(Pivoting::from("Diagonal"), Pivoting::Diagonal);
        assert_eq!(Pivoting::from("LocalBlock"), Pivoting::LocalBlock);

        assert_eq!(Pivoting::from("auto"), Pivoting::Auto);
        assert_eq!(Pivoting::from("none"), Pivoting::None);
        assert_eq!(Pivoting::from("globalcol"), Pivoting::GlobalCol);
        assert_eq!(Pivoting::from("globalrow"), Pivoting::GlobalRow);
        assert_eq!(Pivoting::from("diagonal"), Pivoting::Diagonal);
        assert_eq!(Pivoting::from("localblock"), Pivoting::LocalBlock);

        assert_eq!(Pivoting::from("unknown"), Pivoting::Auto);
    }

    #[test]
    fn genie_functions_work() {
        assert_eq!(Genie::from("cudss"), Genie::Cudss);
        assert_eq!(Genie::from("mumps"), Genie::Mumps);
        assert_eq!(Genie::from("umfpack"), Genie::Umfpack);
        assert_eq!(Genie::from("blah-blah-blah"), Genie::Umfpack);

        assert_eq!(Genie::from("cuDSS"), Genie::Cudss);
        assert_eq!(Genie::from("KLU"), Genie::Klu);
        assert_eq!(Genie::from("Mumps"), Genie::Mumps);
        assert_eq!(Genie::from("Umfpack"), Genie::Umfpack);

        let genie = Genie::Cudss;
        assert_eq!(genie.to_string(), "cudss");
        assert_eq!(genie.get_sym(false), Sym::No);
        assert_eq!(genie.get_sym(true), Sym::YesLower);

        let genie = Genie::Klu;
        assert_eq!(genie.to_string(), "klu");
        assert_eq!(genie.get_sym(false), Sym::No);
        assert_eq!(genie.get_sym(true), Sym::YesFull);

        let genie = Genie::Mumps;
        assert_eq!(genie.to_string(), "mumps");
        assert_eq!(genie.get_sym(false), Sym::No);
        assert_eq!(genie.get_sym(true), Sym::YesLower);

        let genie = Genie::Umfpack;
        assert_eq!(genie.to_string(), "umfpack");
        assert_eq!(genie.get_sym(false,), Sym::No);
        assert_eq!(genie.get_sym(true), Sym::YesFull);
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
