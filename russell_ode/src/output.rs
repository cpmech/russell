use crate::{OdeSolverTrait, Workspace};
use crate::{Stats, StrError};
use russell_lab::{vec_max_abs_diff, Vector};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;

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
/// * `A` -- generic argument to assist in the f(x,y) and Jacobian functions.
///   It may be simply [crate::NoArgs] indicating that no arguments are needed.
pub struct Output<'a, A> {
    /// Indicates whether the solver called initialize or not
    initialized: bool,

    /// Holds the initial x given to the solve function (set by the initialize function)
    x0: f64,

    /// Holds the final x given to the solve function (set by the initialize function)
    x1: f64,

    // --- step --------------------------------------------------------------------------------------------
    /// Holds a callback function called on an accepted step
    step_callback: Option<Arc<dyn Fn(&Stats, f64, f64, &Vector, &mut A) -> Result<bool, StrError> + Send + Sync + 'a>>,

    /// Save the results to a file (step)
    step_file_key: Option<String>,

    /// Counts the number of file saves (step)
    step_file_count: usize,

    /// Tells Output to record the results from accepted steps
    step_recording: bool,

    /// Holds the stepsize computed at accepted steps
    pub(crate) step_h: Vec<f64>,

    /// Holds the x values computed at accepted steps
    pub(crate) step_x: Vec<f64>,

    /// Holds the selected y components computed at accepted steps
    pub(crate) step_y: HashMap<usize, Vec<f64>>,

    /// Holds the global error computed at accepted steps (if the YxFunction is available)
    ///
    /// The global error is the maximum absolute difference between the numerical results and
    /// the ones computed by `YxFunction` (see [russell_lab::vec_max_abs_diff])
    pub(crate) step_global_error: Vec<f64>,

    // --- dense -------------------------------------------------------------------------------------------
    /// Holds a callback function for the dense output
    dense_callback: Option<Arc<dyn Fn(&Stats, f64, f64, &Vector, &mut A) -> Result<bool, StrError> + Send + Sync + 'a>>,

    /// Save the results to a file (dense)
    dense_file_key: Option<String>,

    /// Counts the number of file saves (dense)
    dense_file_count: usize,

    /// Tells Output to record the dense output
    dense_recording: bool,

    /// Uniform stepsize for dense output
    dense_h_out: Option<f64>,

    /// Holds the current index of the dense output station
    dense_index: usize,

    /// Holds the x values (specified by the user)
    pub(crate) dense_x: Vec<f64>,

    /// Holds the selected y components computed during the dense output
    pub(crate) dense_y: HashMap<usize, Vec<f64>>,

    // --- stiffness ---------------------------------------------------------------------------------------
    /// Records the stations where stiffness has been detected
    stiff_recording: bool,

    /// Holds the indices of the accepted steps where stiffness has been detected
    pub(crate) stiff_step_index: Vec<usize>,

    /// Holds the x stations where stiffness has been detected
    pub(crate) stiff_x: Vec<f64>,

    /// Holds the h·ρ values where stiffness has been (firstly) detected
    ///
    /// Note: ρ is the approximation of |λ|, where λ is the dominant eigenvalue of the Jacobian
    /// (see Hairer-Wanner Part II page 22)
    pub(crate) stiff_h_times_rho: Vec<f64>,

    // --- auxiliary ---------------------------------------------------------------------------------------
    /// Holds an auxiliary y vector (e.g., to compute the analytical solution or the dense output)
    y_aux: Vector,

    /// Holds the y(x) function (e.g., to compute the correct/analytical solution)
    yx_function: Option<Arc<dyn Fn(&mut Vector, f64, &mut A) + Send + Sync + 'a>>,
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

impl<'a, A> Output<'a, A> {
    /// Allocates a new instance
    pub(crate) fn new() -> Self {
        const EMPTY: usize = 0;
        Output {
            initialized: false,
            x0: 0.0,
            x1: 0.0,
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
            dense_callback: None,
            dense_file_key: None,
            dense_file_count: 0,
            dense_recording: false,
            dense_h_out: None,
            dense_index: 0,
            dense_x: Vec::new(),
            dense_y: HashMap::new(),
            // stiffness
            stiff_recording: false,
            stiff_step_index: Vec::new(),
            stiff_x: Vec::new(),
            stiff_h_times_rho: Vec::new(),
            // auxiliary
            y_aux: Vector::new(EMPTY),
            yx_function: None,
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
    /// * `callback` -- function to be executed on an accepted step
    pub fn set_step_callback(
        &mut self,
        callback: impl Fn(&Stats, f64, f64, &Vector, &mut A) -> Result<bool, StrError> + Send + Sync + 'a,
    ) -> &mut Self {
        self.step_callback = Some(Arc::new(callback));
        self
    }

    /// Sets the generation of files with the results at accepted steps
    ///
    /// # Input
    ///
    /// * `filepath_without_extension` -- example: `/tmp/russell_ode/my_simulation`
    pub fn set_step_file_writing(&mut self, filepath_without_extension: &str) -> &mut Self {
        self.step_file_key = Some(filepath_without_extension.to_string());
        self
    }

    /// Sets the recording of results at accepted steps
    ///
    /// # Input
    ///
    /// * `selected_y_components` -- specifies which elements of the `y` vector are to be saved
    ///
    /// # Results
    ///
    /// * The results will be recorded in the `step_h`, `step_x`, and `step_y` arrays
    /// * If `YxFunction` is provided, the global error will be recorded in the `step_global_error` array
    /// * The global error is the maximum absolute difference between the numerical and analytical solution
    pub fn set_step_recording(&mut self, selected_y_components: &[usize]) -> &mut Self {
        self.step_recording = true;
        for m in selected_y_components {
            self.step_y.insert(*m, Vec::new());
        }
        self
    }

    /// Sets the stepsize for dense output
    ///
    /// # Input
    ///
    /// * `h_out` -- stepsize; it must be > 10.0 * f64::EPSILON
    pub fn set_dense_h_out(&mut self, h_out: f64) -> Result<&mut Self, StrError> {
        if h_out <= 10.0 * f64::EPSILON {
            return Err("h_out must be > 10.0 * EPSILON");
        }
        self.dense_h_out = Some(h_out);
        Ok(self)
    }

    /// Sets the x stations for dense output
    ///
    /// # Input
    ///
    /// * `interior_x_out` -- specifies the interior x-stations for output (excluding x0 and x1).
    ///   The stations must be in `(x0, x1)` and must be sorted in ascending order
    ///
    /// **Note:** The same `x_out` is used for the `callback`, `file`, and `recording` options
    pub fn set_dense_x_out(&mut self, interior_x_out: &[f64]) -> Result<&mut Self, StrError> {
        let n_int = interior_x_out.len();
        let n = n_int + 2;
        self.dense_x = vec![0.0; n];
        for k in 0..n_int {
            if k > 0 {
                if interior_x_out[k] < interior_x_out[k - 1] {
                    return Err("the dense output stations x must be sorted in ascending order in (x0, x1)");
                }
                if interior_x_out[k] - interior_x_out[k - 1] <= 10.0 * f64::EPSILON {
                    return Err("the x spacing must be > 10.0 * EPSILON");
                }
            }
            self.dense_x[1 + k] = interior_x_out[k];
        }
        self.dense_h_out = None;
        Ok(self)
    }

    /// Sets a callback function called on the dense output
    ///
    /// Use `|stats, h, x, y, args|` or `|stats: &Stats, h: f64, x: f64, y: &Vector, args: &mut A|`
    ///
    /// The function may return `true` to stop the computations
    ///
    /// **Note:** Make sure to call [Output::set_dense_h_out()] or [Output::set_dense_x_out()] to set the spacing.
    /// Otherwise, only the initial (x0) and final (x1) stations will be output.
    ///
    /// # Input
    ///
    /// * `callback` -- function to be executed on the selected output stations
    pub fn set_dense_callback(
        &mut self,
        callback: impl Fn(&Stats, f64, f64, &Vector, &mut A) -> Result<bool, StrError> + Send + Sync + 'a,
    ) -> &mut Self {
        self.dense_callback = Some(Arc::new(callback));
        self
    }

    /// Sets the generation of files with the results from the dense output
    ///
    /// **Note:** Make sure to call [Output::set_dense_h_out()] or [Output::set_dense_x_out()] to set the spacing.
    /// Otherwise, only the initial (x0) and final (x1) stations will be output.
    ///
    /// # Input
    ///
    /// * `filepath_without_extension` -- example: `/tmp/russell_ode/my_simulation`
    ///
    /// **Note:** The same `x_out` is used for the callback, file, and "recording" options
    pub fn set_dense_file_writing(&mut self, filepath_without_extension: &str) -> Result<&mut Self, StrError> {
        if filepath_without_extension.len() < 4 {
            return Err("the length of the filepath without extension must be at least 4");
        }
        self.dense_file_key = Some(filepath_without_extension.to_string());
        Ok(self)
    }

    /// Sets the recording of results at a predefined dense sequence of steps
    ///
    /// **Note:** Make sure to call [Output::set_dense_h_out()] or [Output::set_dense_x_out()] to set the spacing.
    /// Otherwise, only the initial (x0) and final (x1) stations will be output.
    ///
    /// # Input
    ///
    /// * `selected_y_components` -- Specifies which components of the `y` vector are to be saved
    ///
    /// # Results
    ///
    /// * The results will be recorded in the `dense_x` and `dense_y` arrays
    /// * The indices of the associated accepted step will be recorded in the `dense_step_index` array
    pub fn set_dense_recording(&mut self, selected_y_components: &[usize]) -> &mut Self {
        self.dense_recording = true;
        for m in selected_y_components {
            self.dense_y.insert(*m, Vec::new());
        }
        self
    }

    /// Sets the function to compute the correct/reference results y(x)
    ///
    /// Use `|y, x, args|` or `|y: &mut Vector, x: f64, args, &mut A|`
    pub fn set_yx_correct(&mut self, y_fn_x: impl Fn(&mut Vector, f64, &mut A) + Send + Sync + 'a) -> &mut Self {
        self.yx_function = Some(Arc::new(y_fn_x));
        self
    }

    /// Initializes the output structure with initial and final x values
    ///
    /// **Note:** This function also clears the previous results.
    pub(crate) fn initialize(&mut self, x0: f64, x1: f64, stiff_recording: bool) -> Result<(), StrError> {
        assert!(x1 > x0);
        self.stiff_recording = stiff_recording;
        // clear previous results
        if self.initialized {
            if self.step_recording {
                self.step_h.clear();
                self.step_x.clear();
                self.step_global_error.clear();
                for (_, ym) in self.step_y.iter_mut() {
                    ym.clear();
                }
            }
            if self.stiff_recording {
                self.stiff_step_index.clear();
                self.stiff_x.clear();
                self.stiff_h_times_rho.clear();
            }
        }
        // handle dense output stations
        if self.with_dense_output() {
            if let Some(h_out) = self.dense_h_out {
                // uniform spacing
                let n = ((x1 - x0) / h_out) as usize + 1;
                if self.dense_x.len() != n {
                    self.dense_x.resize(n, 0.0);
                }
                self.dense_x[0] = x0;
                self.dense_x[n - 1] = x1;
                for i in 1..(n - 1) {
                    self.dense_x[i] = self.dense_x[i - 1] + h_out;
                }
            } else {
                // user-defined spacing
                if self.dense_x.len() == 0 {
                    self.dense_x = vec![0.0; 2]; // just x0 and x1
                }
                let n = self.dense_x.len();
                self.dense_x[0] = x0;
                self.dense_x[n - 1] = x1;
                if n > 2 {
                    if self.dense_x[1] <= x0 {
                        return Err("the first interior x_out for dense output must be > x0");
                    }
                    if self.dense_x[n - 2] >= x1 {
                        return Err("the last interior x_out for dense output must be < x1");
                    }
                }
            }
            // allocate vectors in dense_y
            let n = self.dense_x.len();
            for (_, ym) in self.dense_y.iter_mut() {
                if ym.len() != n {
                    ym.resize(n, 0.0);
                }
            }
        }
        // set initialized
        self.x0 = x0;
        self.x1 = x1;
        self.initialized = true;
        Ok(())
    }

    /// Indicates whether dense output is enabled or not
    pub(crate) fn with_dense_output(&self) -> bool {
        self.dense_callback.is_some() || self.dense_file_key.is_some() || self.dense_recording
    }

    /// Executes the output at an accepted step
    pub(crate) fn execute(
        &mut self,
        work: &Workspace,
        h: f64,
        x: f64,
        y: &Vector,
        solver: &Box<dyn OdeSolverTrait<A> + 'a>,
        args: &mut A,
    ) -> Result<bool, StrError> {
        assert!(self.initialized);

        // --- step --------------------------------------------------------------------------------------------
        //
        // step output: callback
        if let Some(cb) = self.step_callback.as_ref() {
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
                // initial station
                self.dense_index = 0;

                // first dense output: callback
                if let Some(cb) = self.dense_callback.as_ref() {
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
                    for (m, ym) in self.dense_y.iter_mut() {
                        ym[self.dense_index] = y[*m];
                    }
                }

                // next station
                self.dense_index += 1;
            } else {
                // maybe allocate y_aux
                if self.y_aux.dim() != y.dim() {
                    self.y_aux = Vector::new(y.dim());
                }
                let y_out = &mut self.y_aux;

                // loop over stations
                let n_out = self.dense_x.len() - 1; // -1 because x1 is handled by last()
                while self.dense_index < n_out {
                    // check range
                    let x_out = self.dense_x[self.dense_index];

                    // exit if the requested station is > x
                    if x_out > x {
                        break; // not yet
                    }

                    // interpolate y_out
                    solver.dense_output(y_out, x_out, x, y, h);

                    // subsequent dense output: callback
                    if let Some(cb) = self.dense_callback.as_ref() {
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
                        for (m, ym) in self.dense_y.iter_mut() {
                            ym[self.dense_index] = y_out[*m];
                        }
                    }

                    // next station
                    self.dense_index += 1;
                }
            }
        }

        // stiffness results
        if self.stiff_recording {
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
        // --- step --------------------------------------------------------------------------------------------
        //
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

        // --- dense -------------------------------------------------------------------------------------------
        //
        if self.with_dense_output() {
            // check
            assert_eq!(self.dense_index, self.dense_x.len() - 1);

            // dense output: callback
            if let Some(cb) = self.dense_callback.as_ref() {
                cb(&work.stats, h, x, y, args)?;
            }

            // dense output: write file
            if let Some(fp) = &self.dense_file_key {
                // data
                let full_path = format!("{}_{}.json", fp, self.dense_file_count).to_string();
                let results = OutDataRef { h, x, y };
                results.write_json(&full_path)?;
                self.dense_file_count += 1;
                // count
                let full_path = format!("{}_count.json", fp).to_string();
                let count = OutCount {
                    n: self.dense_file_count,
                };
                count.write_json(&full_path)?;
            }

            // dense output: record results
            if self.dense_recording {
                for (m, ym) in self.dense_y.iter_mut() {
                    ym[self.dense_index] = y[*m];
                }
            }
        }
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NoArgs;
    use russell_lab::array_approx_eq;

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
    fn set_dense_h_out_captures_errors() {
        let mut out = Output::<'_, NoArgs>::new();
        assert_eq!(
            out.set_dense_h_out(f64::EPSILON).err(),
            Some("h_out must be > 10.0 * EPSILON")
        );
    }

    #[test]
    fn set_dense_h_out_works() {
        let mut out = Output::<'_, NoArgs>::new();
        assert!(out.dense_h_out.is_none());
        out.set_dense_h_out(0.1).unwrap();
        assert_eq!(out.dense_h_out, Some(0.1));
    }

    #[test]
    fn set_dense_x_out_captures_errors() {
        let mut out = Output::<'_, NoArgs>::new();
        assert_eq!(
            out.set_dense_x_out(&[2.0, 1.0]).err(),
            Some("the dense output stations x must be sorted in ascending order in (x0, x1)")
        );
        assert_eq!(
            out.set_dense_x_out(&[1.0, 1.0]).err(),
            Some("the x spacing must be > 10.0 * EPSILON")
        );
        assert_eq!(
            out.set_dense_x_out(&[1.0, 1.0 + f64::EPSILON]).err(),
            Some("the x spacing must be > 10.0 * EPSILON")
        );
    }

    #[test]
    fn set_dense_x_out_works() {
        let mut out = Output::<'_, NoArgs>::new();

        out.set_dense_x_out(&[1.0, 2.0]).unwrap();
        assert_eq!(&out.dense_x, &[0.0, 1.0, 2.0, 0.0]);

        out.set_dense_x_out(&[]).unwrap();
        assert_eq!(&out.dense_x, &[0.0, 0.0]);
    }

    #[test]
    fn set_dense_file_writing_captures_errors() {
        let mut out = Output::<'_, NoArgs>::new();
        assert_eq!(
            out.set_dense_file_writing("no").err(),
            Some("the length of the filepath without extension must be at least 4")
        );
    }

    #[test]
    fn initialize_captures_errors() {
        let mut out = Output::<'_, NoArgs>::new();

        out.set_dense_x_out(&[3.0, 4.0]).unwrap();
        out.set_dense_recording(&[0]);
        assert_eq!(&out.dense_x, &[0.0, 3.0, 4.0, 0.0]);
        assert_eq!(
            out.initialize(3.0, 4.0, false).err(),
            Some("the first interior x_out for dense output must be > x0")
        );

        out.set_dense_x_out(&[3.1, 4.0, 5.0]).unwrap();
        out.set_dense_recording(&[0]);
        assert_eq!(&out.dense_x, &[0.0, 3.1, 4.0, 5.0, 0.0]);
        assert_eq!(
            out.initialize(3.0, 4.0, false).err(),
            Some("the last interior x_out for dense output must be < x1")
        );
    }

    #[test]
    fn initialize_with_dense_output_works() {
        let mut out = Output::<'_, NoArgs>::new();

        // without h_out and x_out
        out.set_dense_recording(&[0]);
        out.initialize(3.0, 4.0, false).unwrap();
        assert_eq!(&out.dense_x, &[3.0, 4.0]);

        // with h_out
        out.set_dense_h_out(0.1).unwrap();
        out.initialize(3.0, 4.0, false).unwrap();
        array_approx_eq(
            &out.dense_x,
            &[3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0],
            1e-15,
        );

        // with empty x_out
        out.set_dense_x_out(&[]).unwrap();
        out.initialize(3.0, 4.0, false).unwrap();
        assert_eq!(&out.dense_x, &[3.0, 4.0]);
        let y0_out = out.dense_y.get(&0).unwrap();
        assert_eq!(y0_out.len(), 2);

        // with x_out
        out.set_dense_x_out(&[3.5, 3.8]).unwrap();
        out.initialize(3.0, 4.0, false).unwrap();
        assert_eq!(&out.dense_x, &[3.0, 3.5, 3.8, 4.0]);
        let y0_out = out.dense_y.get(&0).unwrap();
        assert_eq!(y0_out.len(), 4);
    }

    #[test]
    fn initialize_with_step_output_works() {
        let mut out = Output::<'_, NoArgs>::new();

        // first call
        out.set_step_recording(&[0]);
        assert_eq!(out.step_y.len(), 1);
        out.initialize(1.0, 2.0, false).unwrap();

        // write some values
        out.step_h.push(11.11);
        out.step_x.push(22.22);
        out.step_y.get_mut(&0).unwrap().push(33.33);
        out.step_global_error.push(44.44);

        // initialize again
        out.initialize(1.0, 2.0, false).unwrap();
        assert_eq!(out.step_y.len(), 1);

        // check empty arrays
        assert_eq!(out.step_h.len(), 0);
        assert_eq!(out.step_x.len(), 0);
        assert_eq!(out.step_y.get_mut(&0).unwrap().len(), 0);
        assert_eq!(out.step_global_error.len(), 0);
    }

    #[test]
    fn initialize_with_stiff_recording_works() {
        let mut out = Output::<'_, NoArgs>::new();

        // first call
        out.initialize(1.0, 2.0, true).unwrap();

        // write some values
        out.stiff_h_times_rho.push(11.11);
        out.stiff_step_index.push(22);
        out.stiff_x.push(33.33);

        // initialize again
        out.initialize(1.0, 2.0, true).unwrap();

        // check empty arrays
        assert_eq!(out.stiff_h_times_rho.len(), 0);
        assert_eq!(out.stiff_step_index.len(), 0);
        assert_eq!(out.stiff_x.len(), 0);
    }
}
