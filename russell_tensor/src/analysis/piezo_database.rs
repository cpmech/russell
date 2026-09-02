use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Represents the entire database deserialized from the JSON file.
///
/// The JSON structure contains a specific `_metadata` key with global metadata,
/// and dynamically namespaced keys containing tensor data mapped by their
/// Materials Project ID (e.g., `"mp-3731"`). The `#[serde(flatten)]` macro
/// cleanly separates the metadata from the dynamic material dictionary.
#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct PiezoDatabase {
    /// Global metadata describing tensor mappings and formats used in the DB.
    #[serde(rename = "_metadata")]
    pub metadata: Metadata,

    /// A map holding all material objects, keyed by their Materials Project ID.
    #[serde(flatten)]
    pub materials: HashMap<String, Material>,
}

/// Stores standard definitions and rules applied across the database.
#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct Metadata {
    /// The string mapping denoting the Voigt reduction rules.
    pub voigt_mapping: String,

    /// Detailed notes regarding the specific IEEE standard mapping used.
    pub voigt_mapping_notes: String,
}

/// Represents the physical properties and tensors for a single crystalline material.
///
/// Note: Depending on the API availability from Materials Project, certain materials
/// (e.g. Ag3SI) may lack calculated elasticity data. As such, the strain tensor `d`
/// as well as the elastic tensors `C` and `S` are strictly optional.
#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct Material {
    /// The chemical formula of the material (e.g. "LiNbO3").
    pub formula: String,

    /// The crystal system classifying the lattice (e.g. "Trigonal", "Triclinic").
    pub crystal_system: String,

    /// The crystallographic point group notation.
    pub point_group: String,

    /// The space group symbol (e.g. "R3c").
    pub space_group_symbol: String,

    /// The standard space group number (1 to 230).
    pub space_group_number: i32,

    /// The fully expanded Cartesian Piezoelectric Stress Tensor ($e_{ijk}$) (3x3x3 array).
    /// Represents polarization generated per unit strain.
    pub e_tensor: Vec<Vec<Vec<f64>>>,

    /// The Voigt-reduced Piezoelectric Stress Tensor ($e_{i\alpha}$) (3x6 matrix).
    pub e_voigt: Vec<Vec<f64>>,

    /// The fully expanded Cartesian Piezoelectric Strain Tensor ($d_{ijk}$) (3x3x3 array).
    /// Represents polarization generated per unit stress, fully scaled for shear terms.
    pub d_tensor: Vec<Vec<Vec<f64>>>,

    /// The Voigt-reduced Piezoelectric Strain Tensor ($d_{i\alpha}$) (3x6 matrix).
    pub d_voigt: Vec<Vec<f64>>,

    /// The fully expanded Cartesian Elastic Stiffness Tensor ($C_{ijkl}$) (3x3x3x3 array).
    /// Derived from the `elasticity` dataset in GPa.
    pub cc_tensor: Vec<Vec<Vec<Vec<f64>>>>,

    /// The Voigt-reduced Elastic Stiffness Tensor ($C_{\alpha\beta}$) (6x6 matrix).
    pub cc_voigt: Vec<Vec<f64>>,

    // Elastic compliance tensors
    /// The fully expanded Cartesian Elastic Compliance Tensor ($S_{ijkl}$) (3x3x3x3 array).
    /// Derived symmetrically via the inverse stiffness.
    pub ss_tensor: Vec<Vec<Vec<Vec<f64>>>>,

    /// The Voigt-reduced Elastic Compliance Tensor ($S_{\alpha\beta}$) (6x6 matrix).
    pub ss_voigt: Vec<Vec<f64>>,

    // Scalar Elastic Moduli
    /// Bulk modulus (Voigt average) in GPa.
    pub kk_v: f64,

    /// Shear modulus (Voigt average) in GPa.
    pub gg_v: f64,

    /// Bulk modulus (Reuss average) in GPa.
    pub kk_r: f64,

    /// Shear modulus (Reuss average) in GPa.
    pub gg_r: f64,

    /// Bulk modulus (Voigt-Reuss-Hill average) in GPa.
    pub kk_h: f64,

    /// Shear modulus (Voigt-Reuss-Hill average) in GPa.
    pub gg_h: f64,

    /// Universal anisotropy index.
    pub aa_u: f64,

    // Dielectric permittivity tensors ($\epsilon_{ij}$) (3x3 Cartesian array).
    /// Extracted directly from the `dielectric` dataset.
    pub epsilon_tensor: Vec<Vec<f64>>,
}

impl PiezoDatabase {
    /// Reads and parses the PiezoDatabase directly from a JSON file.
    ///
    /// # Arguments
    /// * `path` - A path reference to the JSON file to be loaded.
    ///
    /// # Errors
    /// Returns an error if the file cannot be opened/read or if the JSON format
    /// does not perfectly match the struct schema.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        let db: PiezoDatabase = serde_json::from_str(&content)?;
        Ok(db)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_piezo_data_json() {
        // Look for the JSON file in the parent directory
        let path = Path::new("data/piezo_data.json");
        assert!(path.exists(), "piezo_data.json not found in the parent directory");

        let db = PiezoDatabase::from_file(path).expect("Failed to parse JSON database");

        // Test metadata
        assert_eq!(db.metadata.voigt_mapping, "11, 22, 33, 23, 13, 12");

        // Test parsing a specific material (e.g. LiNbO3 -> mp-3731)
        assert!(db.materials.contains_key("mp-3731"), "Missing mp-3731");
        let linbo3 = &db.materials["mp-3731"];
        assert_eq!(linbo3.formula, "LiNbO3");
        assert_eq!(linbo3.crystal_system, "Trigonal");
        assert_eq!(linbo3.space_group_number, 161);

        // Check tensor structures
        let e_voigt = &linbo3.e_voigt;
        assert_eq!(e_voigt.len(), 3);
        assert_eq!(e_voigt[0].len(), 6);

        let e_tensor = &linbo3.e_tensor;
        assert_eq!(e_tensor.len(), 3);
        assert_eq!(e_tensor[0].len(), 3);
        assert_eq!(e_tensor[0][0].len(), 3);

        let cc_voigt = &linbo3.cc_voigt;
        assert_eq!(cc_voigt.len(), 6);
        assert_eq!(cc_voigt[0].len(), 6);

        assert!(linbo3.kk_v > 0.0, "kk_v should be parsed");
        assert!(linbo3.aa_u > 0.0, "aa_u should be parsed");
    }
}
