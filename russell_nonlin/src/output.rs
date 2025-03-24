use crate::Workspace;
use crate::{Stats, StrError};
use russell_lab::{vec_max_abs_diff, Vector};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;

/// Holds the data generated at an accepted step
#[derive(Clone, Debug, Deserialize)]
pub struct OutData {
    pub u: Vector,
    pub l: f64,
    pub s: f64,
    pub h: f64,
}

/// Holds the data generated at an accepted step (internal version)
///
/// This an internal version holding a reference to `u` to avoid temporary copies.
#[derive(Clone, Debug, Serialize)]
struct OutDataRef<'a> {
    pub u: &'a Vector,
    pub l: f64,
    pub s: f64,
    pub h: f64,
}

/// Holds a counter of how many output files have been written
///
/// This data structure is useful to read back all generated files
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct OutCount {
    pub n: usize,
}

/// Holds the results at accepted steps
pub struct Output<'a, A> {
    /// Indicates whether the solver called initialize or not
    initialized: bool,

    /// Holds a callback function called on an accepted step
    ///
    /// The function is `fn (stats, u, λ, s, h, args)`
    callback: Option<Arc<dyn Fn(&Stats, &Vector, f64, f64, f64, &mut A) -> Result<bool, StrError> + Send + Sync + 'a>>,

    /// Save the results to a file (step)
    file_key: Option<String>,

    /// Counts the number of file saves (step)
    file_count: usize,

    /// Tells Output to record the results from accepted steps
    recording: bool,

    /// Holds the selected u components computed at accepted steps
    pub(crate) u: HashMap<usize, Vec<f64>>,

    /// Holds the λ (parameter) values computed at accepted steps
    pub(crate) l: Vec<f64>,

    /// Holds the s (arclength) values computed at accepted steps
    pub(crate) s: Vec<f64>,

    /// Holds the stepsize computed at accepted steps
    pub(crate) h: Vec<f64>,

    /// Holds the error computed at accepted steps (if the u_reference is available)
    ///
    /// The error is the maximum absolute difference between the numerical results and
    /// the ones computed by `calc_u_ref` (see [russell_lab::vec_max_abs_diff])
    pub(crate) error: Vec<f64>,

    /// Holds an auxiliary u vector (e.g., to compute the analytical solution)
    u_aux: Vector,

    /// Holds a function to compute the correct/reference results G(u, λ) = 0
    ///
    /// The function is `fn (u_correct, λ, args)`
    calc_u_ref: Option<Arc<dyn Fn(&mut Vector, f64, &mut A) + Send + Sync + 'a>>,
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
            callback: None,
            file_key: None,
            file_count: 0,
            recording: false,
            u: HashMap::new(),
            l: Vec::new(),
            s: Vec::new(),
            h: Vec::new(),
            error: Vec::new(),
            // auxiliary
            u_aux: Vector::new(EMPTY),
            calc_u_ref: None,
        }
    }

    /// Sets a callback function called on an accepted step
    ///
    /// The function is `fn (stats, u, λ, s, h, args)`
    ///
    /// The function may return `true` to stop the computations
    ///
    /// # Input
    ///
    /// * `callback` -- function to be executed on an accepted step
    pub fn set_step_callback(
        &mut self,
        callback: impl Fn(&Stats, &Vector, f64, f64, f64, &mut A) -> Result<bool, StrError> + Send + Sync + 'a,
    ) -> &mut Self {
        self.callback = Some(Arc::new(callback));
        self
    }

    /// Sets the generation of files with the results at accepted steps
    ///
    /// # Input
    ///
    /// * `filepath_without_extension` -- example: `/tmp/russell_ode/my_simulation`
    pub fn set_step_file_writing(&mut self, filepath_without_extension: &str) -> &mut Self {
        self.file_key = Some(filepath_without_extension.to_string());
        self
    }

    /// Sets the recording of results at accepted steps
    ///
    /// # Input
    ///
    /// * `selected_u_components` -- specifies which elements of the `u` vector are to be saved
    pub fn set_step_recording(&mut self, selected_u_components: &[usize]) -> &mut Self {
        self.recording = selected_u_components.len() > 0;
        for m in selected_u_components {
            self.u.insert(*m, Vec::new());
        }
        self
    }

    /// Sets the function to compute the correct/reference results of G(u, λ) = 0
    ///
    /// The function is `fn (u_correct, λ, args)`
    pub fn set_u_correct(&mut self, calc_u: impl Fn(&mut Vector, f64, &mut A) + Send + Sync + 'a) -> &mut Self {
        self.calc_u_ref = Some(Arc::new(calc_u));
        self
    }

    /// Initializes the output structure with initial and final values
    ///
    /// **Note:** This function also clears the previous results.
    pub(crate) fn initialize(&mut self) {
        // clear previous results
        if self.initialized {
            if self.recording {
                self.h.clear();
                self.l.clear();
                self.error.clear();
                for (_, um) in self.u.iter_mut() {
                    um.clear();
                }
            }
        }
        // set initialized
        self.initialized = true;
    }

    /// Executes the output at an accepted step
    pub(crate) fn execute(
        &mut self,
        work: &Workspace,
        u: &Vector,
        l: f64,
        s: f64,
        h: f64,
        args: &mut A,
    ) -> Result<bool, StrError> {
        assert!(self.initialized);

        // callback
        if let Some(cb) = self.callback.as_ref() {
            let stop = cb(&work.stats, u, l, s, h, args)?;
            if stop {
                return Ok(stop);
            }
        }

        // write file
        if let Some(fp) = &self.file_key {
            let full_path = format!("{}_{}.json", fp, self.file_count).to_string();
            let results = OutDataRef { u, l, s, h };
            results.write_json(&full_path)?;
            self.file_count += 1;
        }

        // record results
        if self.recording {
            self.l.push(l);
            self.s.push(s);
            self.h.push(h);
            for (m, um) in self.u.iter_mut() {
                um.push(u[*m]);
            }
            if let Some(calc_u) = self.calc_u_ref.as_mut() {
                if self.u_aux.dim() != u.dim() {
                    self.u_aux = Vector::new(u.dim());
                }
                calc_u(&mut self.u_aux, l, args);
                let (_, err) = vec_max_abs_diff(u, &self.u_aux).unwrap();
                self.error.push(err);
            }
        }

        // done
        Ok(false) // do not stop
    }

    /// Saves the count file
    pub(crate) fn last(&mut self) -> Result<(), StrError> {
        if let Some(fp) = &self.file_key {
            let full_path = format!("{}_count.json", fp).to_string();
            let count = OutCount { n: self.file_count };
            count.write_json(&full_path)?;
        }
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{OutCount, OutData, OutDataRef, Output};
    use crate::NoArgs;
    use russell_lab::Vector;

    #[test]
    fn derive_methods_work() {
        // OutData
        let out_data = OutData {
            u: Vector::new(1),
            l: 0.5,
            s: 0.2,
            h: 0.1,
        };
        let clone = out_data.clone();
        assert_eq!(
            format!("{:?}", clone),
            "OutData { u: NumVector { data: [0.0] }, l: 0.5, s: 0.2, h: 0.1 }"
        );
        let json = "{\"u\":{\"data\":[3.0]},\"l\":0.2,\"s\":0.3,\"h\":0.4}";
        let from_json: OutData = serde_json::from_str(&json).unwrap();
        assert_eq!(from_json.u.as_data(), &[3.0]);
        assert_eq!(from_json.l, 0.2);
        assert_eq!(from_json.s, 0.3);
        assert_eq!(from_json.h, 0.4);

        // OutDataRef
        let u = Vector::from(&[10.0]);
        let out_data_ref = OutDataRef {
            u: &u,
            l: 0.5,
            s: 0.15,
            h: 0.3,
        };
        let clone = out_data_ref.clone();
        assert_eq!(
            format!("{:?}", clone),
            "OutDataRef { u: NumVector { data: [10.0] }, l: 0.5, s: 0.15, h: 0.3 }"
        );
        let json = serde_json::to_string(&out_data_ref).unwrap();
        assert_eq!(json, "{\"u\":{\"data\":[10.0]},\"l\":0.5,\"s\":0.15,\"h\":0.3}");

        // OutCount
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
        let u = Vector::from(&[6.6]);
        let data_out = OutDataRef {
            u: &u,
            l: 0.5,
            s: 0.15,
            h: 0.3,
        };
        let path = "/tmp/russell_nonlin/test_out_data.json";
        data_out.write_json(path).unwrap();

        // Read OutData
        let data_in = OutData::read_json(path).unwrap();
        assert_eq!(data_in.u.as_data(), &[6.6]);
        assert_eq!(data_in.l, 0.5);
        assert_eq!(data_in.s, 0.15);
        assert_eq!(data_in.h, 0.3);

        // Write OutCount
        let sum_out = OutCount { n: 456 };
        let path = "/tmp/russell_nonlin/test_out_count.json";
        sum_out.write_json(path).unwrap();
        let sum_in = OutCount::read_json(path).unwrap();
        assert_eq!(sum_in.n, 456);
    }

    #[test]
    fn initialize_with_step_output_works() {
        let mut out = Output::<'_, NoArgs>::new();

        // first call
        out.set_step_recording(&[0]);
        assert_eq!(out.u.len(), 1);
        out.initialize();

        // write some values
        out.h.push(11.11);
        out.l.push(22.22);
        out.u.get_mut(&0).unwrap().push(33.33);
        out.error.push(44.44);

        // initialize again
        out.initialize();
        assert_eq!(out.u.len(), 1);

        // check empty arrays
        assert_eq!(out.h.len(), 0);
        assert_eq!(out.l.len(), 0);
        assert_eq!(out.u.get_mut(&0).unwrap().len(), 0);
        assert_eq!(out.error.len(), 0);
    }
}
