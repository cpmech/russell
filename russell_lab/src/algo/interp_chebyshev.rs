use crate::math::{chebyshev_tn, PI};
use crate::StrError;
use crate::Vector;

/// Defines the tolerance to make sure that the range [xa, xb] is not zero
const TOL_RANGE: f64 = 1.0e-8;

// The grid points are sorted from +1 to -1;
//
// See the note below from Reference # 1 (page 86):
//
// "Note that the Chebyshev quadrature points as just defined are ordered
// from right to left. This violates our general convention that quadrature points
// are ordered from left to right (see Sect. 2.2.3). Virtually all of the classical
// literature on Chebyshev spectral methods uses this reversed order. Therefore,
// in the special case of the Chebyshev quadrature points we shall adhere to the
// ordering convention that is widely used in the literature (and implemented
// in the available software). We realize that our resolution of this dilemma
// imposes upon the reader the task of mentally reversing the ordering of the
// Chebyshev nodes whenever they are used in general formulas for orthogonal
// polynomials." (Canuto, Hussaini, Quarteroni, Zang)

/// Implements the Chebyshev interpolant and associated functions
///
/// # Notes
///
/// 1. This structure is meant for interpolating data and finding (all) the roots of an equation.
///    On the other hand, [crate::InterpLagrange] is meant for implementing spectral methods
///    for solving partial differential equations. Therefore, [crate::InterpLagrange] implements
///    the derivative matrices, whereas this structure does not.
/// 2. The [crate::InterpLagrange] renders the same results when using Chebyshev-Gauss-Lobatto points.
/// 3. Only Chebyshev-Gauss-Lobatto points are considered here.
/// 4. The Chebyshev-Gauss-Lobatto coordinates are sorted from +1 to -1 (as in Reference # 1).
///
/// # References
///
/// 1. Canuto C, Hussaini MY, Quarteroni A, Zang TA (2006) Spectral Methods: Fundamentals in
///    Single Domains. Springer. 563p
pub struct InterpChebyshev {
    /// Holds the polynomial degree
    nn: usize,

    /// Holds the number of points (= N + 1)
    np: usize,

    /// Holds the lower bound
    xa: f64,

    /// Holds the upper bound
    xb: f64,

    /// Holds the difference xb - xa
    dx: f64,

    /// Holds the expansion coefficients (Chebyshev-Gauss-Lobatto)
    coef: Vec<f64>,

    /// Holds the function evaluation at the Chebyshev-Gauss-Lobatto points
    data: Vec<f64>,

    /// Holds the constant y=c value for a zeroth-order function
    constant_fx: f64,
}

impl InterpChebyshev {
    /// Allocates a new instance
    ///
    /// # Input
    ///
    /// * `nn` -- polynomial degree
    /// * `xa` -- lower bound
    /// * `xb` -- upper bound (> xa + ϵ)
    /// * `args` -- extra arguments for the f(x) function
    /// * `f` -- is the callback function implementing `f(x)` as `f(x, args)`; it returns `f @ x` or it may return an error.
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::*;
    ///
    /// fn main() -> Result<(), StrError> {
    ///     // function
    ///     let f = |x, _: &mut NoArgs| Ok(1.0 / (1.0 + 16.0 * x * x));
    ///     let (xa, xb) = (-1.0, 1.0);
    ///
    ///     // interpolant
    ///     let degree = 10;
    ///     let args = &mut 0;
    ///     let interp = InterpChebyshev::new(degree, xa, xb, args, f)?;
    ///
    ///     // check
    ///     approx_eq(interp.eval(0.0).unwrap(), 1.0, 1e-15);
    ///     Ok(())
    /// }
    /// ```
    pub fn new<F, A>(nn: usize, xa: f64, xb: f64, args: &mut A, f: F) -> Result<Self, StrError>
    where
        F: FnMut(f64, &mut A) -> Result<f64, StrError>,
    {
        if xb <= xa + TOL_RANGE {
            return Err("xb must be greater than xa + ϵ");
        }
        let np = nn + 1;
        let mut interp = InterpChebyshev {
            nn,
            np,
            xa,
            xb,
            dx: xb - xa,
            coef: vec![0.0; np],
            data: vec![0.0; np],
            constant_fx: 0.0,
        };
        interp.initialize(args, f)?;
        Ok(interp)
    }

    /// Allocates a new instance using adaptive interpolation
    ///
    /// # Input
    ///
    /// * `nn_max` -- maximum polynomial degree (≤ 2048)
    /// * `tol` -- tolerance to truncate the Chebyshev series (e.g., 1e-8)
    /// * `xa` -- lower bound
    /// * `xb` -- upper bound (> xa + ϵ)
    /// * `args` -- extra arguments for the f(x) function
    /// * `f` -- is the callback function implementing `f(x)` as `f(x, args)`; it returns `f @ x` or it may return an error.
    ///
    /// # Examples
    ///
    /// ```
    /// use russell_lab::*;
    ///
    /// fn main() -> Result<(), StrError> {
    ///     // function
    ///     let f = |x, _: &mut NoArgs| Ok(x * x - 1.0);
    ///     let (xa, xb) = (-1.0, 1.0);
    ///
    ///     // interpolant
    ///     let nn_max = 200;
    ///     let tol = 1e-8;
    ///     let args = &mut 0;
    ///     let interp = InterpChebyshev::new_adapt(nn_max, tol, xa, xb, args, f)?;
    ///
    ///     // check
    ///     assert_eq!(interp.get_degree(), 2);
    ///     Ok(())
    /// }
    /// ```
    pub fn new_adapt<F, A>(nn_max: usize, tol: f64, xa: f64, xb: f64, args: &mut A, mut f: F) -> Result<Self, StrError>
    where
        F: FnMut(f64, &mut A) -> Result<f64, StrError>,
    {
        if nn_max > 2048 {
            return Err("the maximum degree N must be ≤ 2048");
        }
        let np_max = nn_max + 1;
        let mut work_coef = vec![0.0; np_max];
        let mut work_data = vec![0.0; np_max];
        let mut an_prev = 0.0;
        for nn in 1..=nn_max {
            chebyshev_coefficients(&mut work_coef, &mut work_data, nn, xa, xb, args, &mut f)?;
            let an = work_coef[nn];
            if nn > 1 && f64::abs(an_prev) < tol && f64::abs(an) < tol {
                let nn_final = nn - 2; // -2 because the last two coefficients are zero
                return Ok(InterpChebyshev::new(nn_final, xa, xb, args, f)?);
            }
            an_prev = an;
        }
        Err("adaptive interpolation did not converge")
    }

    /// Evaluates the interpolated f(x) function
    pub fn eval(&self, x: f64) -> Result<f64, StrError> {
        if self.nn == 0 {
            return Ok(self.constant_fx);
        }
        let mut sum = 0.0;
        for k in 0..self.np {
            let y = f64::max(-1.0, f64::min(1.0, (2.0 * x - self.xb - self.xa) / self.dx));
            sum += self.coef[k] * chebyshev_tn(k, y);
        }
        Ok(sum)
    }

    /// Estimates the max error by comparing the interpolation against a reference solution
    ///
    /// **Note:** The error is calculated by the max abs difference between the interpolation
    /// and the reference (e.g., analytical) solution at `nstation` discrete points.
    ///
    /// # Input
    ///
    /// * `nstation` -- number of stations (e.g, `10_000`) to compute `max(|f(x) - I{f}(x)|)`
    /// * `args` -- extra arguments for the callback function
    /// * `f` -- is the callback function implementing `f(x)` as `f(x, args)`; it returns `f @ x` or it may return an error.
    ///
    /// # Output
    ///
    /// * `err_f` -- is the max interpolation error in `[xa, xb]`
    pub fn estimate_max_error<F, A>(&self, nstation: usize, args: &mut A, mut f: F) -> Result<f64, StrError>
    where
        F: FnMut(f64, &mut A) -> Result<f64, StrError>,
    {
        let mut err_f = 0.0;
        let stations = Vector::linspace(self.xa, self.xb, nstation).unwrap();
        for p in 0..nstation {
            let fi = self.eval(stations[p])?;
            err_f = f64::max(err_f, f64::abs(fi - f(stations[p], args)?));
        }
        Ok(err_f)
    }

    /// Returns the polynomial degree N
    pub fn get_degree(&self) -> usize {
        self.nn
    }

    /// Calculates the coefficient and data vectors
    fn initialize<F, A>(&mut self, args: &mut A, mut f: F) -> Result<(), StrError>
    where
        F: FnMut(f64, &mut A) -> Result<f64, StrError>,
    {
        if self.nn == 0 {
            self.constant_fx = f((self.xa + self.xb) / 2.0, args)?;
        } else {
            chebyshev_coefficients(&mut self.coef, &mut self.data, self.nn, self.xa, self.xb, args, &mut f)?;
        }
        Ok(())
    }
}

/// Computes the Chebyshev-Gauss-Lobatto coefficients
fn chebyshev_coefficients<F, A>(
    work_coef: &mut [f64],
    work_data: &mut [f64],
    nn: usize,
    xa: f64,
    xb: f64,
    args: &mut A,
    f: &mut F,
) -> Result<(), StrError>
where
    F: FnMut(f64, &mut A) -> Result<f64, StrError>,
{
    // check
    let np = nn + 1;
    assert!(nn > 0);
    assert!(xb > xa);
    assert!(work_coef.len() >= np);
    assert!(work_data.len() >= np);

    // data vector
    let nf = nn as f64;
    let data = &mut work_data[0..np];
    for k in 0..np {
        let kf = k as f64;
        let x = (xb + xa + (xb - xa) * f64::cos(PI * kf / nf)) / 2.0;
        data[k] = (*f)(x, args)?;
    }

    // coefficients
    let coef = &mut work_coef[0..np];
    for j in 0..np {
        let jf = j as f64;
        let qj = if j == 0 || j == nn { 2.0 } else { 1.0 };
        coef[j] = 0.0;
        for k in 0..np {
            let kf = k as f64;
            let qk = if k == 0 || k == nn { 2.0 } else { 1.0 };
            coef[j] += data[k] * 2.0 * f64::cos(PI * jf * kf / nf) / (qj * qk * nf);
        }
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::InterpChebyshev;
    use crate::math::PI;
    use crate::{NoArgs, Vector};
    use plotpy::{Curve, Legend, Plot};

    const SAVE_FIGURE: bool = false;

    #[test]
    fn interp_chebyshev_new_adapt_works() {
        let functions = [
            |_: f64, _: &mut NoArgs| Ok(2.0),
            |x: f64, _: &mut NoArgs| Ok(x - 0.5),
            |x: f64, _: &mut NoArgs| Ok(x * x - 1.0),
            |x: f64, _: &mut NoArgs| Ok(x * x * x - 0.5),
            |x: f64, _: &mut NoArgs| Ok(x * x * x * x - 0.5),
            |x: f64, _: &mut NoArgs| Ok(x * x * x * x * x - 0.5),
            |x: f64, _: &mut NoArgs| Ok(f64::cos(16.0 * (x + 0.2)) * (1.0 + x) * f64::exp(x * x) / (1.0 + 9.0 * x * x)),
            |x: f64, _: &mut NoArgs| Ok(0.092834 * f64::sin(77.0001 + 19.87 * x)),
            |x: f64, _: &mut NoArgs| Ok(f64::ln(2.0 * f64::cos(x / 2.0))),
        ];
        let ranges = [
            (-1.0, 1.0),
            (-1.0, 1.0),
            (-1.0, 1.0),
            (-1.0, 1.0),
            (-1.0, 1.0),
            (-1.0, 1.0),
            (-1.0, 1.0),
            (-2.34567, 12.34567),
            (-0.995 * PI, 0.995 * PI),
        ];
        let err_tols = [0.0, 0.0, 1e-15, 1e-15, 1e-15, 1e-15, 1e-6, 1e-6, 1e-6];
        let nn_max = 400;
        let tol = 1e-7;
        let args = &mut 0;
        for (index, f) in functions.into_iter().enumerate() {
            let (xa, xb) = ranges[index];
            let interp = InterpChebyshev::new_adapt(nn_max, tol, xa, xb, args, f).unwrap();
            let nn = interp.get_degree();
            let err = interp.estimate_max_error(1000, args, f).unwrap();
            println!("{:0>3} : N = {:>3} : err = {:.2e}", index, nn, err);
            assert!(err <= err_tols[index]);
            if SAVE_FIGURE {
                let xx = Vector::linspace(xa, xb, 201).unwrap();
                let yy_ana = xx.get_mapped(|x| f(x, args).unwrap());
                let yy_int = xx.get_mapped(|x| interp.eval(x).unwrap());
                let mut curve_ana = Curve::new();
                let mut curve_int = Curve::new();
                curve_ana.set_label("analytical");
                curve_int
                    .set_label(&format!("interpolated,N={}", nn))
                    .set_line_style(":")
                    .set_marker_style(".")
                    .set_marker_every(5);
                curve_ana.draw(xx.as_data(), yy_ana.as_data());
                curve_int.draw(xx.as_data(), yy_int.as_data());
                let mut plot = Plot::new();
                let mut legend = Legend::new();
                legend.set_num_col(4);
                legend.set_outside(true);
                legend.draw();
                plot.add(&curve_ana)
                    .add(&curve_int)
                    .add(&legend)
                    .set_cross(0.0, 0.0, "gray", "-", 1.5)
                    .grid_and_labels("x", "f(x)")
                    .save(&format!(
                        "/tmp/russell_lab/test_interp_chebyshev_new_adapt_{:0>3}.svg",
                        index
                    ))
                    .unwrap();
            }
        }
    }
}
