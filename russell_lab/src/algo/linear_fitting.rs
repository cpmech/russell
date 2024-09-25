use crate::StrError;
use num_traits::{cast, Num, NumCast};
use std::ops::{AddAssign, Mul};

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
/// # Special cases
///
/// This function returns `(0.0, f64::INFINITY)` in two situations:
///
/// * If `pass_through_zero == True` and `sum(X) == 0`
/// * If `pass_through_zero == False` and the line is vertical (null denominator)
///
/// # Panics
///
/// This function may panic if the number type cannot be converted to `f64`.
///
/// # Examples
///
/// ![Linear fitting](https://raw.githubusercontent.com/cpmech/russell/main/russell_lab/data/figures/algo_linear_fitting_1.svg)
///
/// ```
/// use russell_lab::{approx_eq, linear_fitting, StrError};
///
/// fn main() -> Result<(), StrError> {
///     // model: c is the y value @ x = 0; m is the slope
///     let x = [0.0, 1.0, 3.0, 5.0];
///     let y = [1.0, 0.0, 2.0, 4.0];
///     let (c, m) = linear_fitting(&x, &y, false)?;
///     approx_eq(c, 0.1864406779661015, 1e-15);
///     approx_eq(m, 0.6949152542372882, 1e-15);
///     Ok(())
/// }
/// ```
pub fn linear_fitting<T>(x: &[T], y: &[T], pass_through_zero: bool) -> Result<(f64, f64), StrError>
where
    T: AddAssign + Copy + Mul + Num + NumCast,
{
    // dimension
    let nn = x.len();
    if y.len() != nn {
        return Err("arrays must have the same lengths");
    }

    // sums
    let mut t_sum_x = T::zero();
    let mut t_sum_y = T::zero();
    let mut t_sum_xy = T::zero();
    let mut t_sum_xx = T::zero();
    for i in 0..nn {
        t_sum_x += x[i];
        t_sum_y += y[i];
        t_sum_xy += x[i] * y[i];
        t_sum_xx += x[i] * x[i];
    }

    // cast sums to f64
    let sum_x: f64 = cast(t_sum_x).unwrap();
    let sum_y: f64 = cast(t_sum_y).unwrap();
    let sum_xy: f64 = cast(t_sum_xy).unwrap();
    let sum_xx: f64 = cast(t_sum_xx).unwrap();

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
    use crate::approx_eq;

    #[test]
    fn linear_fitting_handles_errors() {
        let x = [1.0, 2.0];
        let y = [6.0, 5.0, 7.0, 10.0];
        assert_eq!(
            linear_fitting(&x, &y, false).err(),
            Some("arrays must have the same lengths")
        );
    }

    #[test]
    fn linear_fitting_works() {
        // f64 (heap)

        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![6.0, 5.0, 7.0, 10.0];

        let (c, m) = linear_fitting(&x, &y, false).unwrap();
        assert_eq!(c, 3.5);
        assert_eq!(m, 1.4);

        let (c, m) = linear_fitting(&x, &y, true).unwrap();
        assert_eq!(c, 0.0);
        approx_eq(m, 2.566666666666667, 1e-16);

        // usize (stack)

        let x = [1, 2, 3, 4_usize];
        let y = [6, 5, 7, 10_usize];

        let (c, m) = linear_fitting(&x, &y, false).unwrap();
        assert_eq!(c, 3.5);
        assert_eq!(m, 1.4);

        let (c, m) = linear_fitting(&x, &y, true).unwrap();
        assert_eq!(c, 0.0);
        approx_eq(m, 2.566666666666667, 1e-16);

        // i32 (slice)

        let x = &[1, 2, 3, 4_i32];
        let y = &[6, 5, 7, 10_i32];

        let (c, m) = linear_fitting(x, y, false).unwrap();
        assert_eq!(c, 3.5);
        assert_eq!(m, 1.4);

        let (c, m) = linear_fitting(x, y, true).unwrap();
        assert_eq!(c, 0.0);
        approx_eq(m, 2.566666666666667, 1e-16);
    }

    #[test]
    fn linear_fitting_handles_division_by_zero() {
        let x = [1.0, 1.0, 1.0, 1.0];
        let y = [1.0, 2.0, 3.0, 4.0];

        let (c, m) = linear_fitting(&x, &y, false).unwrap();
        assert_eq!(c, 0.0);
        assert_eq!(m, f64::INFINITY);

        let x = [0.0, 0.0, 0.0, 0.0];
        let y = [1.0, 2.0, 3.0, 4.0];
        let (c, m) = linear_fitting(&x, &y, true).unwrap();
        assert_eq!(c, 0.0);
        assert_eq!(m, f64::INFINITY);
    }
}
