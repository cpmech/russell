use crate::{Distribution, SQRT_2, SQRT_PI};

extern "C" {
    fn erf(x: f64) -> f64;
}

pub struct DistributionNormal {
    mean: f64,    // μ: mean
    std_dev: f64, // σ: standard deviation
    a: f64,       // 1 / (σ sqrt(2 π))
    b: f64,       // -1 / (2 σ²)
}

impl DistributionNormal {
    /// Creates a new Normal distribution
    pub fn new(mean: f64, std_dev: f64) -> Self {
        DistributionNormal {
            mean,
            std_dev,
            a: 1.0 / (std_dev * SQRT_2 * SQRT_PI),
            b: -1.0 / (2.0 * std_dev * std_dev),
        }
    }
}

impl Distribution for DistributionNormal {
    fn probability_density_function(&self, x: f64) -> f64 {
        self.a * f64::exp(self.b * f64::powf(x - self.mean, 2.0))
    }

    fn cumulative_density_function(&self, x: f64) -> f64 {
        unsafe { (1.0 + erf((x - self.mean) / (self.std_dev * SQRT_2))) / 2.0 }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {

    #[test]
    fn new_works() {
        // TODO
    }
}
