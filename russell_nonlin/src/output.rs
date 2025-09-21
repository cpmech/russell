use super::{State, Stats, StrError, Workspace};
use russell_lab::Vector;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;

/// Holds a counter of how many output files have been written
///
/// This data structure is useful to read back all generated files
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct OutCount {
    pub n: usize,
}

/// Holds the results at accepted steps
pub struct Output<'a, A> {
    /// Enables the recording of results (u, l, s, h, duds, dlds)
    recording: bool,

    /// Holds a callback function called on an accepted step
    ///
    /// The function is `fn (stats, u, λ, h, args) -> stop_gracefully`
    callback: Option<Arc<dyn Fn(&Stats, &Vector, f64, f64, &mut A) -> Result<bool, StrError> + Send + Sync + 'a>>,

    /// Save the results to a file (step)
    file_key: Option<String>,

    /// Counts the number of file saves (step)
    file_count: usize,

    /// Holds the selected u components computed at accepted steps
    u: HashMap<usize, Vec<f64>>,

    /// Holds the λ (parameter) values computed at accepted steps
    l: Vec<f64>,

    /// Holds the stepsize computed at accepted steps
    h: Vec<f64>,

    /// Holds the selected du/ds components computed at accepted steps (pseudo-arclength)
    duds: HashMap<usize, Vec<f64>>,

    /// Holds the dλ/ds values computed at accepted steps (pseudo-arclength)
    dlds: Vec<f64>,
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
    pub fn new() -> Self {
        Output {
            recording: false,
            callback: None,
            file_key: None,
            file_count: 0,
            u: HashMap::new(),
            l: Vec::new(),
            h: Vec::new(),
            duds: HashMap::new(),
            dlds: Vec::new(),
        }
    }

    // setters ----------------------------------------------------------------------------------------------------------

    /// Sets a callback function called on an accepted step
    ///
    /// The function is `fn (stats, u, λ, h, args) -> stop_gracefully`
    ///
    /// The function may return `true` to stop the computations
    ///
    /// # Input
    ///
    /// * `callback` -- function to be executed on an accepted step
    pub fn set_callback(
        &mut self,
        callback: impl Fn(&Stats, &Vector, f64, f64, &mut A) -> Result<bool, StrError> + Send + Sync + 'a,
    ) -> &mut Self {
        self.callback = Some(Arc::new(callback));
        self
    }

    /// Sets the generation of files with the results at accepted steps
    ///
    /// # Input
    ///
    /// * `filepath_without_extension` -- example: `/tmp/russell_ode/my_simulation`
    pub fn set_file_writing(&mut self, filepath_without_extension: &str) -> &mut Self {
        self.file_key = Some(filepath_without_extension.to_string());
        self
    }

    /// Enables the recording of results (u, l, h, duds, dlds)
    ///
    /// Also specifies which components of the u and du/ds vectors are to be recorded
    pub fn set_recording(&mut self, recording: bool, u_components: &[usize], duds_components: &[usize]) -> &mut Self {
        self.recording = recording;
        for m in u_components {
            self.u.insert(*m, Vec::new());
        }
        for m in duds_components {
            self.duds.insert(*m, Vec::new());
        }
        self
    }

    // getters ----------------------------------------------------------------------------------------------------------

    /// Returns the selected u components computed at accepted steps
    pub fn get_u_values(&self, m: usize) -> &Vec<f64> {
        self.u.get(&m).unwrap()
    }

    /// Returns the λ values computed at accepted steps
    pub fn get_l_values(&self) -> &Vec<f64> {
        &self.l
    }

    /// Returns the h values computed at accepted steps
    pub fn get_h_values(&self) -> &Vec<f64> {
        &self.h
    }

    /// Returns the selected du/ds components computed at accepted steps
    pub fn get_duds_values(&self, m: usize) -> &Vec<f64> {
        self.duds.get(&m).unwrap()
    }

    /// Returns the dλ/ds values computed at accepted steps
    pub fn get_dlds_values(&self) -> &Vec<f64> {
        &self.dlds
    }

    // internal ---------------------------------------------------------------------------------------------------------

    /// Executes the output at an accepted step
    pub(crate) fn execute(&mut self, work: &Workspace, state: &State, args: &mut A) -> Result<bool, StrError> {
        // callback
        if let Some(cb) = self.callback.as_ref() {
            let stop_gracefully = cb(&work.stats, &state.u, state.l, work.h, args)?;
            if stop_gracefully {
                return Ok(stop_gracefully);
            }
        }

        // write file
        if let Some(fp) = &self.file_key {
            let full_path = format!("{}_{}.json", fp, self.file_count).to_string();
            state.write_json(&full_path)?;
            self.file_count += 1;
        }

        // record results
        if self.recording {
            for (m, um) in self.u.iter_mut() {
                um.push(state.u[*m]);
            }
            self.l.push(state.l);
            self.h.push(work.h);
            if work.duds.dim() == state.u.dim() {
                // only for pseudo-arclength with available du/ds and dλ/ds
                for (m, duds_m) in self.duds.iter_mut() {
                    duds_m.push(work.duds[*m]);
                }
                self.dlds.push(work.dlds);
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
    use super::{OutCount, State};
    use russell_lab::Vector;

    #[test]
    fn derive_methods_work() {
        // State: read from JSON
        let out_data = State {
            u: Vector::new(1),
            l: 0.5,
        };
        let clone = out_data.clone();
        assert_eq!(format!("{:?}", clone), "State { u: NumVector { data: [0.0] }, l: 0.5 }");
        let json = "{\"u\":{\"data\":[3.0]},\"l\":0.2}";
        let from_json: State = serde_json::from_str(&json).unwrap();
        assert_eq!(from_json.u.as_data(), &[3.0]);
        assert_eq!(from_json.l, 0.2);

        // State: write to JSON
        let state = State {
            u: Vector::from(&[10.0]),
            l: 0.5,
        };
        let json = serde_json::to_string(&state).unwrap();
        assert_eq!(json, "{\"u\":{\"data\":[10.0]},\"l\":0.5}");

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
        // Write State
        let data_out = State {
            u: Vector::from(&[6.6]),
            l: 0.5,
        };
        let path = "/tmp/russell_nonlin/test_out_data.json";
        data_out.write_json(path).unwrap();

        // Read State
        let data_in = State::read_json(path).unwrap();
        assert_eq!(data_in.u.as_data(), &[6.6]);
        assert_eq!(data_in.l, 0.5);

        // Write OutCount
        let sum_out = OutCount { n: 456 };
        let path = "/tmp/russell_nonlin/test_out_count.json";
        sum_out.write_json(path).unwrap();
        let sum_in = OutCount::read_json(path).unwrap();
        assert_eq!(sum_in.n, 456);
    }
}
