use super::{Stats, StrError, Workspace};
use russell_lab::{vec_norm_chunk, Norm, Vector};
use std::collections::HashMap;
use std::sync::Arc;

/// Holds the results at accepted steps
pub struct Output<'a, A> {
    /// Enables the recording of results (u, l, s, h, duds, dlds)
    recording: bool,

    /// Enables the recording of the norm of u
    ///
    /// Holds `(norm_type, start, stop)`
    record_norm_u: Option<(Norm, usize, usize)>,

    /// Holds a callback function called on an accepted step
    ///
    /// The function is `fn (stats, u, λ, h, args) -> stop_gracefully`
    callback: Option<Arc<dyn Fn(&Stats, &Vector, f64, f64, &mut A) -> Result<bool, StrError> + Send + Sync + 'a>>,

    /// Holds the Euclidean norm of u computed at accepted steps
    norm_u: Vec<f64>,

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

impl<'a, A> Output<'a, A> {
    /// Allocates a new instance
    pub fn new() -> Self {
        Output {
            recording: false,
            record_norm_u: None,
            callback: None,
            norm_u: Vec::new(),
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

    /// Enables the recording of the norm of u
    ///
    /// Uses the following slice of u to compute the norm:
    ///
    /// ```text
    /// let slice = &u[start..stop];
    /// ```
    ///
    /// Note that `stop` is exclusive, i.e., the slice goes up to `stop - 1`.
    ///
    /// Requirements: `start` must be < `stop` and `stop` must be ≤ `u.dim()`.
    pub fn set_record_norm_u(&mut self, recording: bool, norm_type: Norm, start: usize, stop: usize) -> &mut Self {
        self.recording = recording;
        self.record_norm_u = Some((norm_type, start, stop));
        self
    }

    // getters ----------------------------------------------------------------------------------------------------------

    /// Returns the Euclidean norm of u computed at accepted steps
    pub fn get_norm_u_values(&self) -> &Vec<f64> {
        &self.norm_u
    }

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
    pub(crate) fn execute(&mut self, work: &Workspace, u: &Vector, l: f64, args: &mut A) -> Result<bool, StrError> {
        // callback
        if let Some(cb) = self.callback.as_ref() {
            let stop_gracefully = cb(&work.stats, &u, l, work.h, args)?;
            if stop_gracefully {
                return Ok(stop_gracefully);
            }
        }

        // record results
        if self.recording {
            if let Some((norm_type, start, stop)) = self.record_norm_u {
                let norm = vec_norm_chunk(&u, norm_type, start, stop);
                self.norm_u.push(norm);
            }
            for (m, um) in self.u.iter_mut() {
                um.push(u[*m]);
            }
            self.l.push(l);
            self.h.push(work.h);
            if work.duds.dim() == u.dim() {
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
}
