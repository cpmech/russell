use super::format_nanoseconds;
use std::fmt;
use std::time::Instant;

/// Assists in measuring computation time
///
/// # Examples
///
/// ## Printing elapsed times
///
/// ```
/// use russell_lab::Stopwatch;
/// use std::thread::sleep;
/// use std::time::Duration;
///
/// fn expensive_calculation() {
///     sleep(Duration::new(0, 1_000));
/// }
///
/// let mut sw = Stopwatch::new();
/// expensive_calculation();
/// sw.stop();
/// println!("{}", sw);
///
/// sw.reset();
/// expensive_calculation();
/// sw.stop();
/// println!("{}", sw);
///
/// sw.reset();
/// expensive_calculation();
/// sw.stop();
/// println!("{}", sw);
/// ```
///
/// ## Recording elapsed times
///
/// ```
/// use russell_lab::Stopwatch;
/// use std::thread::sleep;
/// use std::time::Duration;
///
/// fn expensive_calculation() {
///     sleep(Duration::new(0, 1_000));
/// }
///
/// let mut elapsed_times = vec![0_u128; 3];
/// let mut sw = Stopwatch::new();
///
/// expensive_calculation();
/// elapsed_times[0] = sw.stop_and_reset();
///
/// expensive_calculation();
/// elapsed_times[1] = sw.stop_and_reset();
///
/// expensive_calculation();
/// elapsed_times[2] = sw.stop_and_reset();
///
/// // println!("{:?}", elapsed_times); // will show something like:
/// // [57148, 55991, 55299]
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Stopwatch {
    initial_time: Instant,
    final_time: Instant,
}

impl Stopwatch {
    /// Creates and starts a new Stopwatch
    ///
    /// # Input
    ///
    /// `label` -- used when displaying the elapsed time
    ///
    /// # Note
    ///
    /// The method `stop` (or `stop_and_reset`) must be called to measure the elapsed time. Until then, the displayed elapsed time is zero, even though the stopwatch has already started.
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::Stopwatch;
    /// let sw = Stopwatch::new();
    /// assert_eq!(format!("{}", sw), "0ns");
    /// ```
    pub fn new() -> Self {
        let now = Instant::now();
        Stopwatch {
            initial_time: now,
            final_time: now,
        }
    }

    /// Stops the stopwatch and returns the elapsed time in nanoseconds
    ///
    /// # Output
    ///
    /// Returns the elapsed time in nanoseconds
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::Stopwatch;
    /// use std::thread::sleep;
    /// use std::time::Duration;
    /// let mut sw = Stopwatch::new();
    /// sleep(Duration::new(0, 1_000));
    /// let elapsed = sw.stop();
    /// assert!(elapsed > 0);
    /// // println!("{}", sw); // will show something like:
    /// // 63.099µs
    /// ```
    pub fn stop(&mut self) -> u128 {
        self.final_time = Instant::now();
        self.final_time.duration_since(self.initial_time).as_nanos()
    }

    /// Resets the stopwatch to zero elapsed time
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::Stopwatch;
    /// use std::thread::sleep;
    /// use std::time::Duration;
    /// let mut sw = Stopwatch::new();
    /// sleep(Duration::new(0, 1_000));
    /// sw.stop();
    /// sw.reset();
    /// assert_eq!(format!("{}", sw), "0ns");
    /// ```
    pub fn reset(&mut self) {
        let now = Instant::now();
        self.initial_time = now;
        self.final_time = now;
    }

    /// Stops the stopwatch and resets it to zero elapsed time
    ///
    /// # Output
    ///
    /// Returns the elapsed time
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::Stopwatch;
    /// use std::thread::sleep;
    /// use std::time::Duration;
    /// let mut sw = Stopwatch::new();
    /// sleep(Duration::new(0, 1_000));
    /// let elapsed = sw.stop_and_reset();
    /// // println!("{}", format_nanoseconds(elapsed)); // will show something like
    /// // current = 63.099µs
    /// assert!(elapsed > 0);
    /// assert_eq!(format!("{}", sw), "0ns");
    /// ```
    pub fn stop_and_reset(&mut self) -> u128 {
        // calc elapsed
        self.final_time = Instant::now();
        let elapsed = self.final_time.duration_since(self.initial_time).as_nanos();

        // reset
        let now = Instant::now();
        self.initial_time = now;
        self.final_time = now;
        elapsed
    }
}

impl fmt::Display for Stopwatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let delta = self.final_time.duration_since(self.initial_time);
        write!(f, "{}", format_nanoseconds(delta.as_nanos())).unwrap();
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Stopwatch;
    use std::thread::sleep;
    use std::time::Duration;

    #[test]
    fn clone_copy_and_debug_work() {
        let sw = Stopwatch::new();
        let copy = sw;
        let clone = sw.clone();
        assert_eq!(copy.initial_time, sw.initial_time);
        assert_eq!(clone.initial_time, sw.initial_time);
        assert!(format!("{:?}", sw).len() > 0);
    }

    #[test]
    fn new_and_display_work() {
        let sw1 = Stopwatch::new();
        assert_eq!(format!("{}", sw1), "0ns");
    }

    #[test]
    fn stop_works() {
        let mut sw = Stopwatch::new();
        sleep(Duration::new(0, 1_000));
        let elapsed = sw.stop();
        assert!(elapsed > 0);
    }

    #[test]
    fn reset_works() {
        let mut sw = Stopwatch::new();

        sleep(Duration::new(0, 1_000));
        let mut elapsed = sw.stop();
        assert!(elapsed > 0);

        sw.reset();
        let delta = sw.final_time.duration_since(sw.initial_time);
        assert_eq!(delta.as_nanos(), 0);

        sleep(Duration::new(0, 1_000));
        elapsed = sw.stop();
        assert!(elapsed > 0);
    }

    #[test]
    fn stop_and_reset_works() {
        let mut sw = Stopwatch::new();

        sleep(Duration::new(0, 1_000));
        let elapsed = sw.stop_and_reset();
        assert!(elapsed > 0);

        let delta = sw.final_time.duration_since(sw.initial_time);
        assert_eq!(delta.as_nanos(), 0);
    }
}
