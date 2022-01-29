use crate::{Distribution, StrError};

/// Defines the Uniform / Type II Extreme Value Distribution (largest value)
pub struct DistributionUniform {
    xmin: f64,
    xmax: f64,
}

impl DistributionUniform {
    /// Creates a new Uniform distribution
    ///
    /// # Input
    ///
    /// * `xmin` -- min x value
    /// * `xmax` -- max x value
    pub fn new(xmin: f64, xmax: f64) -> Result<Self, StrError> {
        Ok(DistributionUniform { xmin, xmax })
    }
}

impl Distribution for DistributionUniform {
    /// Implements the Probability Density Function (CDF)
    fn pdf(&self, x: f64) -> f64 {
        if x < self.xmin {
            return 0.0;
        }
        if x > self.xmax {
            return 0.0;
        }
        1.0 / (self.xmax - self.xmin)
    }

    /// Implements the Cumulative Density Function (CDF)
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
    fn sample(&self) -> f64 {
        0.0
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use crate::{Distribution, DistributionUniform, StrError};
    use russell_chk::assert_approx_eq;

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
    fn uniform_works() -> Result<(), StrError> {
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
            let d = DistributionUniform::new(xmin, xmax)?;
            assert_approx_eq!(d.pdf(x), pdf, 1e-14);
            assert_approx_eq!(d.cdf(x), cdf, 1e-14);
        }
        Ok(())
    }

    #[test]
    fn mean_and_variance_work() -> Result<(), StrError> {
        let d = DistributionUniform::new(1.0, 3.0)?;
        assert_approx_eq!(d.mean(), 2.0, 1e-14);
        assert_approx_eq!(d.variance(), 1.0 / 3.0, 1e-14);
        Ok(())
    }
}
