//! Canonical benchmark matrix metadata.
//!
//! The JSON file is a snapshot of the `matrix-market` repository metadata. It is
//! embedded at compile time so the benchmark tools do not depend on a developer
//! path or the current working directory.

use serde::Deserialize;
use std::collections::HashMap;
use std::sync::OnceLock;

/// Metadata for one benchmark matrix
#[derive(Debug, Deserialize)]
pub struct MatrixInfo {
    #[serde(rename = "Name")]
    pub name: String,
    #[serde(rename = "Symmetric")]
    symmetric: String,
    #[serde(rename = "Positive Definite")]
    positive_definite: String,
    #[serde(rename = "Type")]
    matrix_type: String,
}

impl MatrixInfo {
    /// Returns true for a complex matrix
    pub fn complex(&self) -> bool {
        self.matrix_type.eq_ignore_ascii_case("complex")
    }

    /// Returns true when symmetry information applies
    pub fn symmetric(&self) -> bool {
        self.symmetric.eq_ignore_ascii_case("yes")
    }

    /// Returns true when the matrix is symmetric positive definite
    pub fn spd(&self) -> bool {
        self.symmetric() && self.positive_definite.eq_ignore_ascii_case("yes")
    }
}

/// Embedded metadata from `matrix-market`
const MATRICES_JSON: &str = include_str!("../../../data/matrix_market/matrices.json");

/// Returns all known matrices indexed by name
pub fn all() -> &'static HashMap<String, MatrixInfo> {
    static MATRICES: OnceLock<HashMap<String, MatrixInfo>> = OnceLock::new();
    MATRICES.get_or_init(|| {
        let entries: Vec<MatrixInfo> = serde_json::from_str(MATRICES_JSON).expect("cannot parse embedded matrices.json");
        entries.into_iter().map(|entry| (entry.name.clone(), entry)).collect()
    })
}

/// Returns the metadata for a matrix name
pub fn get(name: &str) -> Option<&'static MatrixInfo> {
    all().get(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_metadata_has_the_benchmark_set() {
        assert_eq!(all().len(), 44);
        assert_eq!(all().values().filter(|info| info.complex()).count(), 10);
        assert_eq!(all().values().filter(|info| !info.complex()).count(), 34);
        assert_eq!(all().values().filter(|info| info.symmetric()).count(), 14);
        assert_eq!(all().values().filter(|info| info.spd()).count(), 8);
    }

    #[test]
    fn metadata_distinguishes_hermitian_and_spd() {
        let mhd = get("mhd1280b").unwrap();
        assert!(mhd.complex());
        assert!(!mhd.symmetric());
        assert!(!mhd.spd());

        let pmsim = get("pmsim-hex8-27").unwrap();
        assert!(!pmsim.complex());
        assert!(pmsim.symmetric());
        assert!(pmsim.spd());
    }

    #[test]
    fn legacy_matrix_is_not_in_metadata() {
        assert!(get("3Dspectralwave").is_none());
    }
}
