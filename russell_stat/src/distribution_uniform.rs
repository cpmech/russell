use crate::{ProbabilityDistribution, StrError};
use rand::Rng;
use rand_distr::{Distribution, Uniform};

/// Implements the continuous Uniform distribution
///
/// See: <https://en.wikipedia.org/wiki/Continuous_uniform_distribution>
///
/// ![Uniform](https://raw.githubusercontent.com/cpmech/russell/main/russell_stat/data/figures/plot_distribution_functions_uniform.svg)
pub struct DistributionUniform {
    xmin: f64, // min x value
    xmax: f64, // max x value

    sampler: Uniform<f64>, // sampler
}

impl DistributionUniform {
    /// Allocates a new instance
    ///
    /// # Input
    ///
    /// * `xmin` -- min x value
    /// * `xmax` -- max x value
    pub fn new(xmin: f64, xmax: f64) -> Result<Self, StrError> {
        if xmax < xmin {
            return Err("invalid parameters");
        }
        Ok(DistributionUniform {
            xmin,
            xmax,
            sampler: Uniform::new(xmin, xmax),
        })
    }
}

impl ProbabilityDistribution for DistributionUniform {
    /// Evaluates the Probability Density Function (CDF)
    fn pdf(&self, x: f64) -> f64 {
        if x < self.xmin {
            return 0.0;
        }
        if x > self.xmax {
            return 0.0;
        }
        1.0 / (self.xmax - self.xmin)
    }

    /// Evaluates the Cumulative Distribution Function (CDF)
    fn cdf(&self, x: f64) -> f64 {
        if x < self.xmin {
            return 0.0;
        }
        if x > self.xmax {
            return 1.0;
        }
        (x - self.xmin) / (self.xmax - self.xmin)
    }

    /// Returns the Mean
    fn mean(&self) -> f64 {
        (self.xmin + self.xmax) / 2.0
    }

    /// Returns the Variance
    fn variance(&self) -> f64 {
        (self.xmax - self.xmin) * (self.xmax - self.xmin) / 12.0
    }

    /// Generates a pseudo-random number belonging to this probability distribution
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        self.sampler.sample(rng)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use crate::{DistributionUniform, ProbabilityDistribution};
    use rand::prelude::StdRng;
    use rand::SeedableRng;
    use russell_lab::approx_eq;

    // Data from the following R-code (run with Rscript uniform.R):
    /*
    a <- 1.5 # xmin
    b <- 2.5 # xmax
    X <- seq(0.5, 3.0, 0.5)
    Y <- matrix(ncol=5)
    first <- TRUE
    pdf <- dunif(X, a, b)
    cdf <- punif(X, a, b)
    for (i in 1:length(X)) {
        if (first) {
            Y <- rbind(c(X[i], a, b, pdf[i], cdf[i]))
            first <- FALSE
        } else {
            Y <- rbind(Y, c(X[i], a, b, pdf[i], cdf[i]))
        }
    }
    write.table(format(Y, digits=15), "/tmp/uniform.dat", row.names=FALSE, col.names=c("x","xmin","xmax","pdf","cdf"), quote=FALSE)
    print("file </tmp/uniform.dat> written")
    */

    #[test]
    fn uniform_handles_errors() {
        assert_eq!(DistributionUniform::new(2.0, 1.0).err(), Some("invalid parameters"));
    }

    #[test]
    fn uniform_works() {
        #[rustfmt::skip]
        // x, xmin, xmax, pdf, cdf
        let data = [
            [0.5, 1.5, 2.5, 0.0, 0.0],
            [1.0, 1.5, 2.5, 0.0, 0.0],
            [1.5, 1.5, 2.5, 1.0, 0.0],
            [2.0, 1.5, 2.5, 1.0, 0.5],
            [2.5, 1.5, 2.5, 1.0, 1.0],
            [3.0, 1.5, 2.5, 0.0, 1.0],
        ];
        for row in data {
            let [x, xmin, xmax, pdf, cdf] = row;
            let d = DistributionUniform::new(xmin, xmax).unwrap();
            approx_eq(d.pdf(x), pdf, 1e-14);
            approx_eq(d.cdf(x), cdf, 1e-14);
        }
    }

    #[test]
    fn mean_and_variance_work() {
        let d = DistributionUniform::new(1.0, 3.0).unwrap();
        approx_eq(d.mean(), 2.0, 1e-14);
        approx_eq(d.variance(), 1.0 / 3.0, 1e-14);
    }

    #[test]
    fn sample_works() {
        let mut rng = StdRng::seed_from_u64(1234);
        let dist_x = DistributionUniform::new(0.0, 2.0).unwrap();
        let dist_y = DistributionUniform::new(0.0, 1.0).unwrap();
        let x = dist_x.sample(&mut rng);
        let y = dist_y.sample(&mut rng);
        approx_eq(x, 0.23691851694908816, 1e-15);
        approx_eq(y, 0.16964948689475423, 1e-15);
    }
}
