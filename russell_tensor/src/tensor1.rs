use russell_lab::AsArray1D;

#[cfg(feature = "heap")]
use russell_lab::Vector;

/// Defines a first-order tensor (vector) in R3
///
/// The "standard" components are recorded here where "standard" means with respect to an orthonormal Cartesian system.
pub struct Tensor1 {
    /// Holds the 3 standard components (heap)
    ///
    /// Heap version => dynamically allocated memory
    #[cfg(feature = "heap")]
    pub(crate) vec: Vector,

    /// Holds the 3 standard components (stack)
    ///
    /// Stack version => fixed size memory
    #[cfg(not(feature = "heap"))]
    pub(crate) vec: [f64; 3],
}

impl Tensor1 {
    /// Allocates a new instance
    pub fn new() -> Self {
        #[cfg(feature = "heap")]
        {
            Tensor1 { vec: Vector::new(3) }
        }
        #[cfg(not(feature = "heap"))]
        {
            Tensor1 { vec: [0.0, 0.0, 0.0] }
        }
    }

    /// Allocates a new instance from a standard (dense) array
    ///
    /// # Input
    ///
    /// * `inp` -- the standard components; a 1D array (fixed-size array, slice, or vector)
    ///   with exactly 3 components
    ///
    /// # Panics
    ///
    /// A panic will occur if `inp` does not have exactly 3 components
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_tensor::Tensor1;
    ///
    /// let u = Tensor1::from(&[1.0, 2.0, 3.0]);
    /// assert_eq!(u.get(0), 1.0);
    /// assert_eq!(u.get(1), 2.0);
    /// assert_eq!(u.get(2), 3.0);
    /// ```
    pub fn from<'a, S>(inp: &'a S) -> Self
    where
        S: AsArray1D<'a, f64>,
    {
        assert_eq!(inp.size(), 3, "the input array must have exactly 3 components");
        let mut tensor = Tensor1::new();
        tensor.vec[0] = inp.at(0);
        tensor.vec[1] = inp.at(1);
        tensor.vec[2] = inp.at(2);
        tensor
    }

    /// Sets the i-th standard component
    ///
    /// # Input
    ///
    /// * `i` -- The index must be 0, 1, or 3
    /// * `value` -- The standard component value
    ///
    /// # Panics
    ///
    /// A panic may occur if the index is out of range
    pub fn set(&mut self, i: usize, value: f64) {
        self.vec[i] = value;
    }

    /// Gets the i-th standard component
    ///
    /// # Input
    ///
    /// * `i` -- The index must be 0, 1, or 3
    ///
    /// # Panics
    ///
    /// A panic may occur if the index is out of range
    pub fn get(&self, i: usize) -> f64 {
        self.vec[i]
    }

    /// Performs the cross product between this tensor and another
    ///
    /// ```text
    /// result = this × other
    /// ```
    pub fn cross(&self, result: &mut Tensor1, other: &Tensor1) {
        result.vec[0] = self.vec[1] * other.vec[2] - self.vec[2] * other.vec[1];
        result.vec[1] = self.vec[2] * other.vec[0] - self.vec[0] * other.vec[2];
        result.vec[2] = self.vec[0] * other.vec[1] - self.vec[1] * other.vec[0];
    }

    /// Calculates the dot (inner) product between this tensor and another
    ///
    /// ```text
    /// result = this . other
    /// ```
    pub fn dot(&self, other: &Tensor1) -> f64 {
        self.vec[0] * other.vec[0] + self.vec[1] * other.vec[1] + self.vec[2] * other.vec[2]
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::Tensor1;

    #[test]
    fn new_set_get_work() {
        let mut u = Tensor1::new();
        u.set(0, 123.0);
        assert_eq!(u.get(0), 123.0);
    }

    #[test]
    fn cross_and_dot_work() {
        let u = Tensor1::from(&[1.0, -2.0, 3.0]);
        let v = Tensor1::from(&[-1.0, 0.0, 1.0]);
        let mut w = Tensor1::new();
        u.cross(&mut w, &v);
        assert_eq!(w.get(0), -2.0);
        assert_eq!(w.get(1), -4.0);
        assert_eq!(w.get(2), -2.0);
        assert_eq!(u.dot(&w), 0.0);
        assert_eq!(v.dot(&w), 0.0);
    }
}
