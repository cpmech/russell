use crate::{OdeSolverTrait, Workspace};
use crate::{Stats, StrError};
use russell_lab::{vec_max_abs_diff, Vector};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::BufReader;
use std::marker::PhantomData;
use std::path::Path;

/// Defines the callback function for accepted steps
///
/// The callback function is `(stats: &Stats, h: f64, x: f64, y: &Vector, args: &mut A)`
/// and may return `true` to stop the simulation
type OutputCallback<A> = fn(&Stats, f64, f64, &Vector, &mut A) -> Result<bool, StrError>;

/// Holds the data generated at an accepted step or during the dense output
#[derive(Clone, Debug, Deserialize)]
pub struct OutData {
    pub h: f64,
    pub x: f64,
    pub y: Vector,
}

/// Holds the data generated at an accepted step or during the dense output
///
/// This (internal) version holds a reference to `y` and avoids copying it.
#[derive(Clone, Debug, Serialize)]
struct OutDataRef<'a> {
    pub h: f64,
    pub x: f64,
    pub y: &'a Vector,
}

/// Holds the "summary" regarding written files
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct OutSummary {
    pub count: usize,
}

/// Holds the (x,y) results at accepted steps or interpolated within a "dense" sequence
///
/// # Generics
///
/// The generic arguments are:
///
/// * `A` -- generic argument to assist in the `F` and `J` functions. It may be simply [crate::NoArgs] indicating that no arguments are needed.
pub struct Output<A> {
    /// Holds the callback function for every accepted step
    step_callback: OutputCallback<A>,

    /// Indicates whether the accepted step output is to be performed or not
    save_step: bool,

    /// Holds the stepsize output during accepted steps
    pub step_h: Vec<f64>,

    /// Holds the x values output during accepted steps
    pub step_x: Vec<f64>,

    /// Holds the selected y components output during accepted steps
    pub step_y: HashMap<usize, Vec<f64>>,

    /// Holds the global error output during accepted steps (if the analytical solution is available)
    ///
    /// The global error is the maximum absolute difference between `y_numerical` and `y_analytical` (see [russell_lab::vec_max_abs_diff])
    pub step_global_error: Vec<f64>,

    /// Holds the stepsize to perform the dense output (None means disabled)
    dense_h_out: Option<f64>,

    /// Holds the indices of the accepted steps that were used to compute the dense output
    pub dense_step_index: Vec<usize>,

    /// Holds the x values requested by the dense output
    pub dense_x: Vec<f64>,

    /// Holds the selected y components requested by the dense output
    pub dense_y: HashMap<usize, Vec<f64>>,

    /// Saves the stations where stiffness has been detected
    pub(crate) save_stiff: bool,

    /// Holds the indices of the accepted steps where stiffness has been detected
    pub stiff_step_index: Vec<usize>,

    /// Holds the x stations where stiffness has been detected
    pub stiff_x: Vec<f64>,

    /// Holds the h·ρ values where stiffness has been (first) detected
    ///
    /// Note: ρ is the approximation of |λ|, where λ is the dominant eigenvalue of the Jacobian
    /// (see Hairer-Wanner Part II page 22)
    pub stiff_h_times_rho: Vec<f64>,

    /// Holds an auxiliary y vector (e.g., to compute the analytical solution or the dense output)
    y_aux: Vector,

    /// Holds a function to compute the analytical solution y(x)
    pub y_analytical: Option<fn(&mut Vector, f64)>,

    /// Save the results to a file (step)
    file_step_key: Option<String>,

    /// Counts the number of file saves (step)
    file_step_count: usize,

    /// Save the results to a file (dense)
    file_dense_key: Option<String>,

    /// Counts the number of file saves (dense)
    file_dense_count: usize,

    /// Holds the stepsize to save a file with the dense output
    file_dense_h_out: f64,

    /// Holds the current x at the dense output
    file_dense_x: f64,

    /// Handle generic argument
    phantom: PhantomData<fn() -> A>,
}

impl OutData {
    /// Reads a JSON file containing the results
    pub fn read_json(full_path: &str) -> Result<Self, StrError> {
        let path = Path::new(full_path).to_path_buf();
        let input = File::open(path).map_err(|_| "cannot open file")?;
        let buffered = BufReader::new(input);
        let stat = serde_json::from_reader(buffered).map_err(|_| "cannot parse JSON file")?;
        Ok(stat)
    }
}

impl<'a> OutDataRef<'a> {
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

impl OutSummary {
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

impl<A> Output<A> {
    /// Allocates a new instance
    pub fn new() -> Self {
        const EMPTY: usize = 0;
        Output {
            step_callback: no_callback,
            save_step: false,
            step_h: Vec::new(),
            step_x: Vec::new(),
            step_y: HashMap::new(),
            step_global_error: Vec::new(),
            dense_h_out: None,
            dense_step_index: Vec::new(),
            dense_x: Vec::new(),
            dense_y: HashMap::new(),
            save_stiff: false,
            stiff_step_index: Vec::new(),
            stiff_x: Vec::new(),
            stiff_h_times_rho: Vec::new(),
            y_aux: Vector::new(EMPTY),
            y_analytical: None,
            file_step_key: None,
            file_step_count: 0,
            file_dense_key: None,
            file_dense_count: 0,
            file_dense_h_out: 0.0,
            file_dense_x: 0.0,
            phantom: PhantomData,
        }
    }

    /// Sets the callback function for accepted steps
    ///
    /// The callback function is `|stats, h, x, y, args|`, i.e.,
    /// `(stats: &Stats, h: f64, x: f64, y: &Vector, args: &mut A)`
    /// and may return `true` to stop the simulation
    pub fn set_step_callback(&mut self, callback: OutputCallback<A>) -> &mut Self {
        self.step_callback = callback;
        self
    }

    /// Enables recording the results at accepted steps
    ///
    /// # Input
    ///
    /// * `selected_y_components` -- Specifies which components of the `y` vector are to be saved
    ///
    /// # Results
    ///
    /// * The results will be saved in the `step_h`, `step_x`, and `step_y` arrays
    /// * If the analytical solution is provided, the global error will be saved in the `step_global_error` array
    /// * The global error is the maximum absolute difference between `y_numerical` and `y_analytical` (see [russell_lab::vec_max_abs_diff])
    pub fn enable_step(&mut self, selected_y_components: &[usize]) -> &mut Self {
        self.save_step = true;
        for m in selected_y_components {
            self.step_y.insert(*m, Vec::new());
        }
        self
    }

    /// Disables recording the results at accepted steps
    pub fn disable_step(&mut self) -> &mut Self {
        self.save_step = false;
        self
    }

    /// Enables the generation of files with the results at accepted steps
    pub fn enable_file_step(&mut self, filepath_without_extension: &str) -> &mut Self {
        self.file_step_key = Some(filepath_without_extension.to_string());
        self.file_step_count = 0;
        self
    }

    /// Disables the generation of files with the results at accepted steps
    pub fn disable_file_step(&mut self) -> &mut Self {
        self.file_step_key = None;
        self
    }

    /// Enables recording the results at a predefined "dense" sequence of steps
    ///
    /// # Input
    ///
    /// * `h_out` -- is the stepsize (possibly different than the actual `h` stepsize) for the equally spaced "dense" results
    /// * `selected_y_components` -- Specifies which components of the `y` vector are to be saved
    ///
    /// # Results
    ///
    /// * The results will be saved in the `dense_x` and `dense_y` arrays
    /// * The indices of the associated accepted step will be saved in the `dense_step_index` array
    pub fn enable_dense(&mut self, h_out: f64, selected_y_components: &[usize]) -> Result<&mut Self, StrError> {
        if h_out <= 0.0 {
            return Err("h_out must be ≥ 0.0");
        }
        self.dense_h_out = Some(h_out);
        for m in selected_y_components {
            self.dense_y.insert(*m, Vec::new());
        }
        Ok(self)
    }

    /// Disables recording the results at a predefined "dense" sequence of steps
    pub fn disable_dense(&mut self) -> &mut Self {
        self.dense_h_out = None;
        self
    }

    /// Enables the generation of files with the results from the "dense" sequence of steps
    pub fn enable_file_dense(&mut self, h_out: f64, filepath_without_extension: &str) -> Result<&mut Self, StrError> {
        if h_out <= 0.0 {
            return Err("h_out must be ≥ 0.0");
        }
        self.file_dense_key = Some(filepath_without_extension.to_string());
        self.file_dense_count = 0;
        self.file_dense_h_out = h_out;
        Ok(self)
    }

    /// Disables the generation of files with the results from the "dense" sequence of steps
    pub fn disable_file_dense(&mut self) -> &mut Self {
        self.file_dense_key = None;
        self
    }

    /// Indicates whether dense output is enabled or not
    pub(crate) fn with_dense_output(&self) -> bool {
        self.dense_h_out.is_some() || self.file_dense_key.is_some()
    }

    /// Clears all resulting arrays
    pub fn clear(&mut self) {
        self.step_h.clear();
        self.step_x.clear();
        for ym in self.step_y.values_mut() {
            ym.clear();
        }
        self.step_global_error.clear();
        self.dense_step_index.clear();
        self.dense_x.clear();
        for ym in self.dense_y.values_mut() {
            ym.clear();
        }
        self.stiff_step_index.clear();
        self.stiff_x.clear();
        self.stiff_h_times_rho.clear();
    }

    /// Executes the output for an accepted step
    pub(crate) fn accept<'a>(
        &mut self,
        work: &Workspace,
        h: f64,
        x: f64,
        y: &Vector,
        solver: &Box<dyn OdeSolverTrait<A> + 'a>,
        args: &mut A,
    ) -> Result<bool, StrError> {
        // callback
        let stop = (self.step_callback)(&work.stats, h, x, y, args)?;
        if stop {
            return Ok(stop);
        }

        // step output
        if self.save_step {
            self.step_h.push(h);
            self.step_x.push(x);
            for (m, ym) in self.step_y.iter_mut() {
                ym.push(y[*m]);
            }
            // global error
            if let Some(ana) = self.y_analytical.as_mut() {
                let ndim = y.dim();
                if self.y_aux.dim() != ndim {
                    self.y_aux = Vector::new(ndim); // first allocation
                }
                ana(&mut self.y_aux, x);
                let (_, err) = vec_max_abs_diff(y, &self.y_aux).unwrap();
                self.step_global_error.push(err);
            }
        }

        // file: step output
        if let Some(fp) = &self.file_step_key {
            let full_path = format!("{}_{}.json", self.file_step_count, fp).to_string();
            let results = OutDataRef { h, x, y };
            results.write_json(&full_path)?;
            self.file_step_count += 1;
        }

        // dense output
        if let Some(h_out) = self.dense_h_out {
            if work.stats.n_accepted == 0 {
                // first output
                self.dense_step_index.push(work.stats.n_accepted);
                self.dense_x.push(x);
                for (m, ym) in self.dense_y.iter_mut() {
                    ym.push(y[*m]);
                }
            } else {
                // subsequent output
                let ndim = y.dim();
                if self.y_aux.dim() != ndim {
                    self.y_aux = Vector::new(ndim); // first allocation
                }
                let mut x_out = self.dense_x.last().unwrap() + h_out;
                while x_out < x {
                    self.dense_step_index.push(work.stats.n_accepted);
                    self.dense_x.push(x_out);
                    solver.dense_output(&mut self.y_aux, x_out, x, y, h);
                    for (m, ym) in self.dense_y.iter_mut() {
                        ym.push(self.y_aux[*m]);
                    }
                    x_out += h_out;
                }
            }
        }

        // file: dense output
        if let Some(fp) = &self.file_dense_key {
            let h_out = self.file_dense_h_out;
            if work.stats.n_accepted == 0 {
                // first output
                let results = OutDataRef { h: h_out, x, y };
                let full_path = format!("{}_{}.json", fp, self.file_dense_count).to_string();
                results.write_json(&full_path)?;
                self.file_dense_count += 1;
                self.file_dense_x = x;
            } else {
                // subsequent output
                let mut x_out = self.file_dense_x + h_out;
                while x_out < x {
                    self.file_dense_x = x_out;
                    let results = OutDataRef { h: h_out, x: x_out, y };
                    let full_path = format!("{}_{}.json", fp, self.file_dense_count).to_string();
                    results.write_json(&full_path)?;
                    self.file_dense_count += 1;
                    x_out += h_out;
                }
            }
        }

        // stiffness results
        if self.save_stiff {
            self.stiff_h_times_rho.push(work.stiff_h_times_rho);
            if work.stiff_detected {
                self.stiff_step_index.push(work.stats.n_accepted);
                self.stiff_x.push(work.stiff_x_first_detect);
            }
        }
        Ok(false) // do not stop
    }

    /// Saves the results at the end of the simulation (and generate summary files)
    pub(crate) fn last(&mut self, work: &Workspace, h: f64, x: f64, y: &Vector, args: &mut A) -> Result<(), StrError> {
        // callback
        (self.step_callback)(&work.stats, h, x, y, args)?;

        // file: step output
        if let Some(fp) = &self.file_step_key {
            let full_path = format!("{}_summary.json", fp).to_string();
            let summary = OutSummary {
                count: self.file_dense_count,
            };
            summary.write_json(&full_path)?;
        }

        // dense output
        if let Some(_) = self.dense_h_out {
            self.dense_step_index.push(work.stats.n_accepted);
            self.dense_x.push(x);
            for (m, ym) in self.dense_y.iter_mut() {
                ym.push(y[*m]);
            }
        }

        // file: dense output
        if let Some(fp) = &self.file_dense_key {
            // data
            let full_path = format!("{}_{}.json", fp, self.file_dense_count).to_string();
            let h_out = self.file_dense_h_out;
            let results = OutDataRef { h: h_out, x, y };
            results.write_json(&full_path)?;
            self.file_dense_count += 1;
            self.file_dense_x = x;
            // summary
            let full_path = format!("{}_summary.json", fp).to_string();
            let summary = OutSummary {
                count: self.file_dense_count,
            };
            summary.write_json(&full_path)?;
        }
        Ok(())
    }
}

/// Placeholder function for no 'callbacks' during step output
pub fn no_callback<A>(_b: &Stats, _h: f64, _x: f64, _y: &Vector, _args: &mut A) -> Result<bool, StrError> {
    Ok(false) // do not stop
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derive_methods_work() {
        // OutData
        let out_data = OutData {
            h: 0.1,
            x: 1.0,
            y: Vector::new(1),
        };
        let clone = out_data.clone();
        assert_eq!(
            format!("{:?}", clone),
            "OutData { h: 0.1, x: 1.0, y: NumVector { data: [0.0] } }"
        );
        let json = "{\"h\":0.2,\"x\":2.0,\"y\":{ \"data\":[3.0]}}";
        let from_json: OutData = serde_json::from_str(&json).unwrap();
        assert_eq!(from_json.h, 0.2);
        assert_eq!(from_json.x, 2.0);
        assert_eq!(from_json.y.as_data(), &[3.0]);

        // OutDataRef
        let y = Vector::from(&[10.0]);
        let out_data_ref = OutDataRef { h: 0.5, x: 5.0, y: &y };
        let clone = out_data_ref.clone();
        assert_eq!(
            format!("{:?}", clone),
            "OutDataRef { h: 0.5, x: 5.0, y: NumVector { data: [10.0] } }"
        );
        let json = serde_json::to_string(&out_data_ref).unwrap();
        assert_eq!(json, "{\"h\":0.5,\"x\":5.0,\"y\":{\"data\":[10.0]}}");

        // OutSummary
        let summary = OutSummary { count: 123 };
        let clone = summary.clone();
        assert_eq!(format!("{:?}", clone), "OutSummary { count: 123 }");
        let json = serde_json::to_string(&summary).unwrap();
        let from_json: OutSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(from_json.count, summary.count);
    }
}
