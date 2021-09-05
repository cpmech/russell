use super::format_nanoseconds;
use std::fmt;
use std::time::Instant;

/// Stopwatch assists in measuring computation time
pub struct Stopwatch {
    label: &'static str,
    initial_time: Instant,
    final_time: Instant,
}

impl Stopwatch {
    /// Creates a new Stopwatch
    ///
    /// # Input
    ///
    /// `label` -- is used when displaying the elapsed time
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::*;
    /// let sw = Stopwatch::new("elapsed time = ");
    /// assert_eq!(format!("{}", sw), "elapsed time = 0ns");
    /// ```
    pub fn new(label: &'static str) -> Self {
        let now = Instant::now();
        Stopwatch {
            label,
            initial_time: now,
            final_time: now,
        }
    }

    /// Stops the stopwatch to measure a duration time
    ///
    /// # Example
    ///
    /// ```
    /// use russell_lab::*;
    /// use std::thread::sleep;
    /// use std::time::Duration;
    /// let mut sw = Stopwatch::new("");
    /// sleep(Duration::new(0, 1_000));
    /// sw.stop();
    /// // println!("{}", sw); will show something like:
    /// // 63.099µs
    /// ```
    pub fn stop(&mut self) {
        self.final_time = Instant::now();
    }

    /// Resets the stopwatch to zero elapsed time
    ///
    /// ```
    /// use russell_lab::*;
    /// use std::thread::sleep;
    /// use std::time::Duration;
    /// let mut sw = Stopwatch::new("delta_t = ");
    /// sleep(Duration::new(0, 1_000));
    /// sw.stop();
    /// sw.reset();
    /// assert_eq!(format!("{}", sw), "delta_t = 0ns");
    /// ```
    pub fn reset(&mut self) {
        let now = Instant::now();
        self.initial_time = now;
        self.final_time = now;
    }
}

impl fmt::Display for Stopwatch {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let delta = self.final_time.duration_since(self.initial_time);
        write!(f, "{}{}", self.label, format_nanoseconds(delta.as_nanos()))?;
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;
    use std::time::Duration;

    #[test]
    fn new_and_display_work() {
        let sw1 = Stopwatch::new("");
        let sw2 = Stopwatch::new("calculation time = ");
        assert_eq!(format!("{}", sw1), "0ns");
        assert_eq!(format!("{}", sw2), "calculation time = 0ns");
    }

    #[test]
    fn stop_works() {
        let mut sw = Stopwatch::new("");
        sleep(Duration::new(0, 1_000));
        sw.stop();
        let delta = sw.final_time.duration_since(sw.initial_time);
        assert!(delta.as_nanos() > 0);
    }

    #[test]
    fn reset_works() {
        let mut sw = Stopwatch::new("");
        sleep(Duration::new(0, 1_000));
        sw.stop();
        sw.reset();
        let delta = sw.final_time.duration_since(sw.initial_time);
        assert_eq!(delta.as_nanos(), 0);
    }
}
