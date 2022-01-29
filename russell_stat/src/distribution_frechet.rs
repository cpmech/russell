use crate::Distribution;

const FRECHET_MIN_DELTA_X: f64 = 1e-15;

pub struct DistributionFrechet {
    location: f64,
    scale: f64,
    shape: f64,
}

impl DistributionFrechet {
    /// Creates a new Frechet distribution
    pub fn new(location: f64, scale: f64, shape: f64) -> Self {
        DistributionFrechet {
            location,
            scale,
            shape,
        }
    }
}

impl Distribution for DistributionFrechet {
    fn probability_density_function(&self, x: f64) -> f64 {
        if x - self.location < FRECHET_MIN_DELTA_X {
            return 0.0;
        }
        let z = (x - self.location) / self.scale;
        f64::exp(-f64::powf(z, -self.shape)) * f64::powf(z, -1.0 - self.shape) * self.shape
            / self.scale
    }

    fn cumulative_density_function(&self, x: f64) -> f64 {
        if x - self.location < FRECHET_MIN_DELTA_X {
            return 0.0;
        }
        let z = (x - self.location) / self.scale;
        f64::exp(-f64::powf(z, -self.shape))
    }
}
