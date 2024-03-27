use crate::{OdeSolverTrait, Workspace};
use crate::{Stats, StrError};
use russell_lab::{vec_max_abs_diff, Vector};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::BufReader;
use std::marker::PhantomData;
use std::path::Path;

/// Defines a function to compute y(x) (e.g, the analytical solution)
///
/// Use `|y, x, args|` or `|y: &mut Vector, x: f64, args, &mut A|`
pub type YxFunction<A> = fn(&mut Vector, f64, &mut A);

/// Defines a callback function to be executed during the output of results (accepted step or dense output)
///
/// Use `|stats, h, x, y, args|` or `|stats: &Stats, h: f64, x: f64, y: &Vector, args: &mut A|`
///
/// The function may return `true` to stop the computations
type OutCallback<A> = fn(&Stats, f64, f64, &Vector, &mut A) -> Result<bool, StrError>;

/// Holds the data generated at an accepted step or during the dense output
#[derive(Clone, Debug, Deserialize)]
pub struct OutData {
    pub h: f64,
    pub x: f64,
    pub y: Vector,
}

/// Holds the data generated at an accepted step or during the dense output (internal version)
///
/// This an internal version holding a reference to `y` to avoid temporary copies.
#[derive(Clone, Debug, Serialize)]
struct OutDataRef<'a> {
    pub h: f64,
    pub x: f64,
    pub y: &'a Vector,
}

/// Holds a counter of how many output files have been written
///
/// This data structure is useful to read back all generated files
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct OutCount {
    pub n: usize,
}

/// Holds the results at accepted steps or interpolated in a "dense" sequence of steps (dense output)
///
/// # Generics
///
/// The generic arguments are:
///
/// * `A` -- Is auxiliary argument for the `F`, `J`, `YxFunction`, and `OutCallback` functions.
///   It may be simply [crate::NoArgs] indicating that no arguments are effectively used.
pub struct Output<A> {
    // --- step --------------------------------------------------------------------------------------------
    /// Holds a callback function called on an accepted step
    step_callback: Option<OutCallback<A>>,

    /// Save the results to a file (step)
    step_file_key: Option<String>,

    /// Counts the number of file saves (step)
    step_file_count: usize,

    /// Tells Output to record the results from accepted steps
    step_recording: bool,

    /// Holds the stepsize computed at accepted steps
    pub step_h: Vec<f64>,

    /// Holds the x values computed at accepted steps
    pub step_x: Vec<f64>,

    /// Holds the selected y components computed at accepted steps
    pub step_y: HashMap<usize, Vec<f64>>,

    /// Holds the global error computed at accepted steps (if the YxFunction is available)
    ///
    /// The global error is the maximum absolute difference between the numerical results and
    /// the ones computed by `YxFunction` (see [russell_lab::vec_max_abs_diff])
    pub step_global_error: Vec<f64>,

    // --- dense -------------------------------------------------------------------------------------------
    /// Holds the stepsize to perform the dense output
    ///
    /// **Note:** The same `h_out` is used for the callback, file, and "recording" options
    dense_h_out: f64,

    /// Holds the last x at the dense output
    ///
    /// This value is needed to update `x_out` in a subsequent dense output, e.g., `x_out = dense_last_x + dense_h_out`
    dense_last_x: f64,

    /// Holds a callback function for the dense output
    dense_callback: Option<OutCallback<A>>,

    /// Save the results to a file (dense)
    dense_file_key: Option<String>,

    /// Counts the number of file saves (dense)
    dense_file_count: usize,

    /// Tells Output to record the dense output
    dense_recording: bool,

    /// Holds the indices of the accepted steps at the time the dense output was computed
    pub dense_step_index: Vec<usize>,

    /// Holds the x values computed during the dense output
    pub dense_x: Vec<f64>,

    /// Holds the selected y components computed during the dense output
    pub dense_y: HashMap<usize, Vec<f64>>,

    // --- stiffness ---------------------------------------------------------------------------------------
    /// Records the stations where stiffness has been detected
    pub(crate) stiff_record: bool,

    /// Holds the indices of the accepted steps where stiffness has been detected
    pub stiff_step_index: Vec<usize>,

    /// Holds the x stations where stiffness has been detected
    pub stiff_x: Vec<f64>,

    /// Holds the h·ρ values where stiffness has been (firstly) detected
    ///
    /// Note: ρ is the approximation of |λ|, where λ is the dominant eigenvalue of the Jacobian
    /// (see Hairer-Wanner Part II page 22)
    pub stiff_h_times_rho: Vec<f64>,

    // --- auxiliary ---------------------------------------------------------------------------------------
    /// Holds an auxiliary y vector (e.g., to compute the analytical solution or the dense output)
    y_aux: Vector,

    /// Holds the y(x) function (e.g., to compute the correct/analytical solution)
    yx_function: Option<YxFunction<A>>,

    /// Handles the generic argument
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

impl OutCount {
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
            // step
            step_callback: None,
            step_file_key: None,
            step_file_count: 0,
            step_recording: false,
            step_h: Vec::new(),
            step_x: Vec::new(),
            step_y: HashMap::new(),
            step_global_error: Vec::new(),
            // dense
            dense_h_out: f64::MAX,
            dense_last_x: 0.0,
            dense_callback: None,
            dense_file_key: None,
            dense_file_count: 0,
            dense_recording: false,
            dense_step_index: Vec::new(),
            dense_x: Vec::new(),
            dense_y: HashMap::new(),
            // stiffness
            stiff_record: false,
            stiff_step_index: Vec::new(),
            stiff_x: Vec::new(),
            stiff_h_times_rho: Vec::new(),
            // auxiliary
            y_aux: Vector::new(EMPTY),
            yx_function: None,
            phantom: PhantomData,
        }
    }

    /// Sets a callback function called on an accepted step
    ///
    /// Use `|stats, h, x, y, args|` or `|stats: &Stats, h: f64, x: f64, y: &Vector, args: &mut A|`
    ///
    /// The function may return `true` to stop the computations
    ///
    /// # Input
    ///
    /// * `enable` -- Enable/disable the output
    /// * `callback` -- Function to be executed on an accepted step
    pub fn set_step_callback(&mut self, enable: bool, callback: OutCallback<A>) -> &mut Self {
        if enable {
            self.step_callback = Some(callback);
        } else {
            self.step_callback = None;
        }
        self
    }

    /// Sets the generation of files with the results at accepted steps
    ///
    /// # Input
    ///
    /// * `enable` -- Enable/disable the output
    /// * `filepath_without_extension` -- E.g. `/tmp/russell_ode/my_simulation`
    pub fn set_step_file_writing(&mut self, enable: bool, filepath_without_extension: &str) -> &mut Self {
        self.step_file_count = 0;
        if enable {
            self.step_file_key = Some(filepath_without_extension.to_string());
        } else {
            self.step_file_key = None;
        }
        self
    }

    /// Sets the recording of results at accepted steps
    ///
    /// # Input
    ///
    /// * `enable` -- Enable/disable the output
    /// * `selected_y_components` -- Specifies which elements of the `y` vector are to be saved
    ///
    /// # Results
    ///
    /// * The results will be recorded in the `step_h`, `step_x`, and `step_y` arrays
    /// * If `YxFunction` is provided, the global error will be recorded in the `step_global_error` array
    /// * The global error is the maximum absolute difference between the numerical and analytical solution
    pub fn set_step_recording(&mut self, enable: bool, selected_y_components: &[usize]) -> &mut Self {
        self.step_recording = enable;
        self.step_y.clear();
        if enable {
            for m in selected_y_components {
                self.step_y.insert(*m, Vec::new());
            }
        }
        self
    }

    /// Sets a callback function called on the dense output
    ///
    /// Use `|stats, h, x, y, args|` or `|stats: &Stats, h: f64, x: f64, y: &Vector, args: &mut A|`
    ///
    /// The function may return `true` to stop the computations
    ///
    /// # Input
    ///
    /// * `enable` -- Enable/disable the output
    /// * `h_out` -- is the stepsize (possibly different than the actual `h` stepsize) for the equally spaced "dense" results
    ///
    /// **Note:** The same `h_out` is used for the callback, file, and "recording" options
    pub fn set_dense_callback(
        &mut self,
        enable: bool,
        h_out: f64,
        callback: OutCallback<A>,
    ) -> Result<&mut Self, StrError> {
        if h_out <= f64::EPSILON {
            return Err("h_out must be > EPSILON");
        }
        self.dense_h_out = h_out;
        if enable {
            self.dense_callback = Some(callback);
        } else {
            self.dense_callback = None;
        }
        Ok(self)
    }

    /// Sets the generation of files with the results from the dense output
    ///
    /// # Input
    ///
    /// * `enable` -- Enable/disable the output
    /// * `h_out` -- is the stepsize (possibly different than the actual `h` stepsize) for the equally spaced "dense" results
    /// * `filepath_without_extension` -- E.g. `/tmp/russell_ode/my_simulation`
    ///
    /// **Note:** The same `h_out` is used for the callback, file, and "recording" options
    pub fn set_dense_file_writing(
        &mut self,
        enable: bool,
        h_out: f64,
        filepath_without_extension: &str,
    ) -> Result<&mut Self, StrError> {
        if h_out <= f64::EPSILON {
            return Err("h_out must be > EPSILON");
        }
        if filepath_without_extension.len() < 4 {
            return Err("the length of the filepath without extension must be at least 4");
        }
        self.dense_h_out = h_out;
        self.dense_file_count = 0;
        if enable {
            self.dense_file_key = Some(filepath_without_extension.to_string());
        } else {
            self.dense_file_key = None;
        }
        Ok(self)
    }

    /// Sets the recording of results at a predefined dense sequence of steps
    ///
    /// # Input
    ///
    /// * `enable` -- Enable/disable the output
    /// * `h_out` -- is the stepsize (possibly different than the actual `h` stepsize) for the equally spaced "dense" results
    /// * `selected_y_components` -- Specifies which components of the `y` vector are to be saved
    ///
    /// # Results
    ///
    /// * The results will be recorded in the `dense_x` and `dense_y` arrays
    /// * The indices of the associated accepted step will be recorded in the `dense_step_index` array
    ///
    /// **Note:** The same `h_out` is used for the callback, file, and "recording" options
    pub fn set_dense_recording(
        &mut self,
        enable: bool,
        h_out: f64,
        selected_y_components: &[usize],
    ) -> Result<&mut Self, StrError> {
        if h_out <= f64::EPSILON {
            return Err("h_out must be > EPSILON");
        }
        self.dense_recording = enable;
        self.dense_h_out = h_out;
        if enable {
            for m in selected_y_components {
                self.dense_y.insert(*m, Vec::new());
            }
        }
        Ok(self)
    }

    /// Sets the function to compute the correct/reference results y(x)
    pub fn set_yx_correct(&mut self, y_fn_x: fn(&mut Vector, f64, &mut A)) -> &mut Self {
        self.yx_function = Some(y_fn_x);
        self
    }

    /// Indicates whether dense output is enabled or not
    pub(crate) fn with_dense_output(&self) -> bool {
        self.dense_callback.is_some() || self.dense_file_key.is_some() || self.dense_recording
    }

    /// Clears the results
    pub fn clear(&mut self) {
        // step
        self.step_h.clear();
        self.step_x.clear();
        self.step_y.clear();
        self.step_global_error.clear();
        // dense
        self.dense_step_index.clear();
        self.dense_x.clear();
        self.dense_y.clear();
        // stiffness
        self.stiff_step_index.clear();
        self.stiff_x.clear();
        self.stiff_h_times_rho.clear();
    }

    /// Executes the output at an accepted step
    pub(crate) fn execute<'a>(
        &mut self,
        work: &Workspace,
        h: f64,
        x: f64,
        y: &Vector,
        solver: &Box<dyn OdeSolverTrait<A> + 'a>,
        args: &mut A,
    ) -> Result<bool, StrError> {
        // --- step --------------------------------------------------------------------------------------------
        //
        // step output: callback
        if let Some(cb) = self.step_callback {
            let stop = cb(&work.stats, h, x, y, args)?;
            if stop {
                return Ok(stop);
            }
        }

        // step output: write file
        if let Some(fp) = &self.step_file_key {
            let full_path = format!("{}_{}.json", fp, self.step_file_count).to_string();
            let results = OutDataRef { h, x, y };
            results.write_json(&full_path)?;
            self.step_file_count += 1;
        }

        // step output: record results
        if self.step_recording {
            self.step_h.push(h);
            self.step_x.push(x);
            for (m, ym) in self.step_y.iter_mut() {
                ym.push(y[*m]);
            }
            if let Some(y_fn_x) = self.yx_function.as_mut() {
                if self.y_aux.dim() != y.dim() {
                    self.y_aux = Vector::new(y.dim());
                }
                y_fn_x(&mut self.y_aux, x, args);
                let (_, err) = vec_max_abs_diff(y, &self.y_aux).unwrap();
                self.step_global_error.push(err);
            }
        }

        // --- dense -------------------------------------------------------------------------------------------
        //
        if self.with_dense_output() {
            if work.stats.n_accepted == 0 {
                // record last x for subsequent calculations
                self.dense_last_x = x;

                // first dense output: callback
                if let Some(cb) = self.dense_callback {
                    let stop = cb(&work.stats, h, x, y, args)?;
                    if stop {
                        return Ok(stop);
                    }
                }

                // first dense output: write file
                if let Some(fp) = &self.dense_file_key {
                    let results = OutDataRef { h, x, y };
                    let full_path = format!("{}_{}.json", fp, self.dense_file_count).to_string();
                    results.write_json(&full_path)?;
                    self.dense_file_count += 1;
                }

                // first dense output: record results
                if self.dense_recording {
                    self.dense_step_index.push(work.stats.n_accepted);
                    self.dense_x.push(x);
                    for (m, ym) in self.dense_y.iter_mut() {
                        ym.push(y[*m]);
                    }
                }
            } else {
                // maybe allocate y_aux
                if self.y_aux.dim() != y.dim() {
                    self.y_aux = Vector::new(y.dim());
                }
                let y_out = &mut self.y_aux;

                // loop over h_out increments
                let mut x_out = self.dense_last_x + self.dense_h_out;
                while x_out < x {
                    // record last x for subsequent calculations
                    self.dense_last_x = x_out;

                    // interpolate y_out
                    solver.dense_output(y_out, x_out, x, y, h);

                    // subsequent dense output: callback
                    if let Some(cb) = self.dense_callback {
                        let stop = cb(&work.stats, h, x_out, y_out, args)?;
                        if stop {
                            return Ok(stop);
                        }
                    }

                    // subsequent dense output: write file
                    if let Some(fp) = &self.dense_file_key {
                        let results = OutDataRef { h, x: x_out, y: y_out };
                        let full_path = format!("{}_{}.json", fp, self.dense_file_count).to_string();
                        results.write_json(&full_path)?;
                        self.dense_file_count += 1;
                    }

                    // subsequent dense output: record results
                    if self.dense_recording {
                        self.dense_step_index.push(work.stats.n_accepted);
                        self.dense_x.push(x_out);
                        for (m, ym) in self.dense_y.iter_mut() {
                            ym.push(y_out[*m]);
                        }
                    }

                    // next station
                    x_out += self.dense_h_out;
                }
            }
        }

        // stiffness results
        if self.stiff_record {
            self.stiff_h_times_rho.push(work.stiff_h_times_rho);
            if work.stiff_detected {
                self.stiff_step_index.push(work.stats.n_accepted);
                self.stiff_x.push(work.stiff_x_first_detect);
            }
        }

        // done
        Ok(false) // do not stop
    }

    /// Saves the results at the end of the simulation (and generates count files)
    pub(crate) fn last(&mut self, work: &Workspace, h: f64, x: f64, y: &Vector, args: &mut A) -> Result<(), StrError> {
        // "step output: callback" and "step output: write file"
        // There is no need to handle these cases because the `step` method
        // already handled these options at the last (accepted) step

        // step output: write file
        if let Some(fp) = &self.step_file_key {
            let full_path = format!("{}_count.json", fp).to_string();
            let count = OutCount {
                n: self.step_file_count,
            };
            count.write_json(&full_path)?;
        }

        // dense output: callback
        if let Some(cb) = self.dense_callback {
            cb(&work.stats, h, x, y, args)?;
        }

        // dense output: write file
        if let Some(fp) = &self.dense_file_key {
            // data
            let full_path = format!("{}_{}.json", fp, self.dense_file_count).to_string();
            let results = OutDataRef { h, x, y };
            results.write_json(&full_path)?;
            self.dense_file_count += 1;
            self.dense_last_x = x;
            // count
            let full_path = format!("{}_count.json", fp).to_string();
            let count = OutCount {
                n: self.dense_file_count,
            };
            count.write_json(&full_path)?;
        }

        // dense output: record results
        if self.dense_recording {
            self.dense_step_index.push(work.stats.n_accepted);
            self.dense_x.push(x);
            for (m, ym) in self.dense_y.iter_mut() {
                ym.push(y[*m]);
            }
        }
        Ok(())
    }
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
        let count = OutCount { n: 123 };
        let clone = count.clone();
        assert_eq!(format!("{:?}", clone), "OutCount { n: 123 }");
        let json = serde_json::to_string(&count).unwrap();
        let from_json: OutCount = serde_json::from_str(&json).unwrap();
        assert_eq!(from_json.n, count.n);
    }

    #[test]
    fn read_write_files_work() {
        // Write OutDataRef
        let y = Vector::from(&[6.6]);
        let data_out = OutDataRef { h: 4.4, x: 5.5, y: &y };
        let path = "/tmp/russell_ode/test_out_data.json";
        data_out.write_json(path).unwrap();

        // Read OutData
        let data_in = OutData::read_json(path).unwrap();
        assert_eq!(data_in.h, 4.4);
        assert_eq!(data_in.x, 5.5);
        assert_eq!(data_in.y.as_data(), &[6.6]);

        // Write OutCount
        let sum_out = OutCount { n: 456 };
        let path = "/tmp/russell_ode/test_out_count.json";
        sum_out.write_json(path).unwrap();
        let sum_in = OutCount::read_json(path).unwrap();
        assert_eq!(sum_in.n, 456);
    }

    #[test]
    fn set_methods_handle_errors() {
        struct Args {}
        let mut out: Output<Args> = Output::new();
        assert_eq!(
            out.set_dense_callback(true, 0.0, |_, _, _, _, _| Ok(false)).err(),
            Some("h_out must be > EPSILON")
        );
        let path_key = "/tmp/russell_ode/test_output_errors";
        assert_eq!(
            out.set_dense_file_writing(true, 0.0, path_key).err(),
            Some("h_out must be > EPSILON")
        );
        assert_eq!(
            out.set_dense_file_writing(true, 0.1, "no").err(),
            Some("the length of the filepath without extension must be at least 4")
        );
        assert_eq!(
            out.set_dense_recording(true, 0.0, &[]).err(),
            Some("h_out must be > EPSILON")
        );
    }
}
