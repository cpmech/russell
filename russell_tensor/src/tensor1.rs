#[cfg(feature = "heap")]
use russell_lab::Vector;

/// Defines a first-order tensor (vector)
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
            Tensor2 { vec: [0.0, 0.0, 0.0] }
        }
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
}
