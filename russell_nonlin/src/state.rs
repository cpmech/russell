use crate::StrError;
use russell_lab::Vector;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::BufReader;
use std::path::Path;

/// Holds the current solution of the nonlinear problem
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct State {
    /// Primary unknown values
    pub u: Vector,

    /// λ parameter
    pub l: f64,

    /// Arclength
    pub s: f64,

    /// Stepsize: either Δs (arclength) or Δλ (natural parameter)
    pub h: f64,

    /// Part of the tangent vector (duds,dλds) for the pseudo-arclength method
    ///
    /// **Note**: this vector is only allocated for the pseudo-arclength method
    ///
    /// (ndim)
    pub duds: Vector,

    /// Part of the tangent vector (duds,dλds) for the pseudo-arclength method
    pub dlds: f64,
}

impl State {
    /// Creates a new instance with zero values
    ///
    /// # Input
    ///
    /// * `ndim` -- number of dimensions (must match the system's ndim)
    /// * `with_tangent_vector` -- with tangent vector; this is required for the pseudo-arclength method
    pub fn new(ndim: usize, with_tangent_vector: bool) -> Self {
        let ndim_duds = if with_tangent_vector { ndim } else { 0 };
        State {
            u: Vector::new(ndim),
            l: 0.0,
            s: 0.0,
            h: 0.0,
            duds: Vector::new(ndim_duds),
            dlds: 0.0,
        }
    }

    /// Reads a JSON file containing the results
    pub fn read_json(full_path: &str) -> Result<Self, StrError> {
        let path = Path::new(full_path).to_path_buf();
        let input = File::open(path).map_err(|_| "cannot open file")?;
        let buffered = BufReader::new(input);
        let stat = serde_json::from_reader(buffered).map_err(|_| "cannot parse JSON file")?;
        Ok(stat)
    }

    /// Writes a JSON file with the results
    pub fn write_json(&self, full_path: &str) -> Result<(), StrError> {
        let path = Path::new(full_path).to_path_buf();
        if let Some(p) = path.parent() {
            fs::create_dir_all(p).map_err(|_| "cannot create directory")?;
        }
        let mut file = File::create(&path).map_err(|_| "cannot create file")?;
        serde_json::to_writer(&mut file, &self).map_err(|_| "cannot write file")?;
        Ok(())
    }
}
