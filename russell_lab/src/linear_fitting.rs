use crate::{StrError, Vector};

/// Calculates the parameters of a linear model using least squares fitting
///
/// # Input
///
/// `x` -- the X-data vector with dimension n
/// `y` -- the Y-data vector with dimension n
/// `pass_through_zero` -- compute the parameters such that the line passes through zero (c = 0)
///
/// # Output
///
/// * `(c, m)` -- the y(x=0)=c intersect and the slope m
///
/// NOTE: this function returns `(0.0, f64::INFINITY)` in two situations:
///
/// * If `pass_through_zero == True` and `sum(X) == 0`
/// * If `pass_through_zero == False` and the line is vertical (null denominator)
pub fn linear_fitting(x: &Vector, y: &Vector, pass_through_zero: bool) -> Result<(f64, f64), StrError> {
    // dimension
    let nn = x.dim();
    if y.dim() != nn {
        return Err("vectors must have the same dimension");
    }

    // sums
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_xx = 0.0;
    for i in 0..nn {
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_xx += x[i] * x[i];
    }

    // calculate parameters
    let c;
    let m;
    let n = nn as f64;
    if pass_through_zero {
        if sum_xx == 0.0 {
            return Ok((0.0, f64::INFINITY));
        }
        c = 0.0;
        m = sum_xy / sum_xx;
    } else {
        let den = sum_x * sum_x - n * sum_xx;
        println!("den = {}", den);
        if den == 0.0 {
            return Ok((0.0, f64::INFINITY));
        }
        c = (sum_x * sum_xy - sum_xx * sum_y) / den;
        m = (sum_x * sum_y - n * sum_xy) / den;
    }

    // results
    Ok((c, m))
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::linear_fitting;
    use crate::Vector;
    use russell_chk::approx_eq;

    #[test]
    fn linear_fitting_handles_errors() {
        let x = Vector::from(&[1.0, 2.0]);
        let y = Vector::from(&[6.0, 5.0, 7.0, 10.0]);
        assert_eq!(
            linear_fitting(&x, &y, false).err(),
            Some("vectors must have the same dimension")
        );
    }

    #[test]
    fn linear_fitting_works() {
        let x = Vector::from(&[1.0, 2.0, 3.0, 4.0]);
        let y = Vector::from(&[6.0, 5.0, 7.0, 10.0]);

        let (c, m) = linear_fitting(&x, &y, false).unwrap();
        assert_eq!(c, 3.5);
        assert_eq!(m, 1.4);

        let (c, m) = linear_fitting(&x, &y, true).unwrap();
        assert_eq!(c, 0.0);
        approx_eq(m, 2.566666666666667, 1e-16);
    }

    #[test]
    fn linear_fitting_handles_division_by_zero() {
        let x = Vector::from(&[1.0, 1.0, 1.0, 1.0]);
        let y = Vector::from(&[1.0, 2.0, 3.0, 4.0]);

        let (c, m) = linear_fitting(&x, &y, false).unwrap();
        assert_eq!(c, 0.0);
        assert_eq!(m, f64::INFINITY);

        let x = Vector::from(&[0.0, 0.0, 0.0, 0.0]);
        let y = Vector::from(&[1.0, 2.0, 3.0, 4.0]);
        let (c, m) = linear_fitting(&x, &y, true).unwrap();
        assert_eq!(c, 0.0);
        assert_eq!(m, f64::INFINITY);
    }
}
